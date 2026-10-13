"""Read PyTorch checkpoints (.pth/.pt) into NumPy arrays without importing torch.

Supports both on-disk formats written by torch.save:
  * the zip format (PyTorch >= 1.6): archive/data.pkl + archive/data/<key>
  * the legacy format (PyTorch <= 1.5): magic, protocol, sys_info, pickled
    object, storage keys, then the raw storages back to back.

Only plain tensors / OrderedDicts / dicts / lists are reconstructed.  Any
other class referenced by the pickle is refused, so a checkpoint cannot run
code while it is being read (unlike torch.load(weights_only=False)).
"""

from __future__ import annotations

import collections
import io
import pickle
import struct
import zipfile

import numpy as np

_STORAGE_DTYPES = {
    "FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16,
    "BFloat16Storage": None, "LongStorage": np.int64, "IntStorage": np.int32,
    "ShortStorage": np.int16, "CharStorage": np.int8, "ByteStorage": np.uint8,
    "BoolStorage": np.bool_,
}
_DTYPE_NAMES = {  # torch.<dtype> objects used by the zip format's TypedStorage
    "float32": np.float32, "float": np.float32, "float64": np.float64, "double": np.float64,
    "float16": np.float16, "half": np.float16, "int64": np.int64, "long": np.int64,
    "int32": np.int32, "int": np.int32, "int16": np.int16, "int8": np.int8,
    "uint8": np.uint8, "bool": np.bool_, "bfloat16": None,
}
LEGACY_MAGIC = 0x1950A86A20F9469CFC6C


class _StorageType:
    def __init__(self, name, dtype):
        self.name, self.dtype = name, dtype


class _Storage:
    def __init__(self, dtype, numel):
        self.dtype, self.numel, self.array = dtype, numel, None


class _LazyTensor:
    def __init__(self, storage, offset, size, stride):
        self.storage, self.offset, self.size, self.stride = storage, offset, tuple(size), tuple(stride)

    def materialize(self) -> np.ndarray:
        arr = self.storage.array
        if arr is None:
            raise ValueError("tensor storage was never read")
        if self.storage.dtype is None:  # bfloat16 -> float32
            arr = (arr.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        item = arr.itemsize
        if not self.size:
            return np.array(arr[self.offset])
        return np.lib.stride_tricks.as_strided(
            arr[self.offset:], shape=self.size,
            strides=tuple(s * item for s in self.stride)).copy()


def _rebuild_tensor(storage, storage_offset, size, stride, *args, **kwargs):
    return _LazyTensor(storage, storage_offset, size, stride)


def _rebuild_parameter(data, requires_grad=False, backward_hooks=None, *args):
    return data


class _Unpickler(pickle.Unpickler):
    def __init__(self, f, storage_factory):
        super().__init__(f)
        self._storage_factory = storage_factory

    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return _rebuild_tensor
        if module == "torch._utils" and name == "_rebuild_parameter":
            return _rebuild_parameter
        if module == "torch" and name in _STORAGE_DTYPES:
            return _StorageType(name, _STORAGE_DTYPES[name])
        if module == "torch" and name in _DTYPE_NAMES:
            return _StorageType(name, _DTYPE_NAMES[name])
        if module == "torch" and name == "Size":
            return tuple
        if module == "torch.serialization" and name == "_get_layout":
            return lambda *_: None
        if module == "torch" and name == "strided":
            return None
        raise pickle.UnpicklingError(f"refusing to load {module}.{name} from a checkpoint")

    def persistent_load(self, pid):
        return self._storage_factory(pid)


def _dtype_of(storage_type):
    if isinstance(storage_type, _StorageType):
        return storage_type.dtype
    raise pickle.UnpicklingError(f"unknown storage type {storage_type!r}")


def _materialize(obj):
    if isinstance(obj, _LazyTensor):
        return obj.materialize()
    if isinstance(obj, dict):
        return type(obj)((k, _materialize(v)) for k, v in obj.items()) if isinstance(obj, collections.OrderedDict) \
            else {k: _materialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_materialize(v) for v in obj)
    return obj


def _np_dtype(dtype):
    return np.uint16 if dtype is None else dtype  # bfloat16 read as raw uint16


def _load_zip(data: bytes):
    zf = zipfile.ZipFile(io.BytesIO(data))
    names = zf.namelist()
    pkl = next(n for n in names if n.endswith("data.pkl"))
    prefix = pkl[: -len("data.pkl")]
    storages = {}

    def factory(pid):
        # ('storage', storage_type, key, location, numel)
        _, stype, key, _location, numel = pid
        if key not in storages:
            dtype = _dtype_of(stype)
            st = _Storage(dtype, numel)
            raw = zf.read(f"{prefix}data/{key}")
            st.array = np.frombuffer(raw, dtype=np.dtype(_np_dtype(dtype)).newbyteorder("<"))
            storages[key] = st
        return storages[key]

    obj = _Unpickler(io.BytesIO(zf.read(pkl)), factory).load()
    return _materialize(obj)


def _load_legacy(data: bytes):
    f = io.BytesIO(data)
    magic = pickle.load(f)
    if magic != LEGACY_MAGIC:
        raise ValueError("not a PyTorch checkpoint")
    pickle.load(f)  # protocol version
    pickle.load(f)  # sys info
    storages = {}

    def factory(pid):
        typename = pid[0]
        if typename != "storage":
            raise pickle.UnpicklingError(f"unsupported persistent id {typename!r}")
        stype, root_key, _location, numel, view_meta = pid[1:]
        if root_key not in storages:
            storages[root_key] = _Storage(_dtype_of(stype), numel)
        if view_meta is not None:
            raise pickle.UnpicklingError("storage views are not supported")
        return storages[root_key]

    obj = _Unpickler(f, factory).load()
    keys = pickle.load(f)
    for key in keys:
        st = storages[key]
        (n,) = struct.unpack("<q", f.read(8))
        itemsize = np.dtype(_np_dtype(st.dtype)).itemsize
        st.array = np.frombuffer(f.read(n * itemsize), dtype=np.dtype(_np_dtype(st.dtype)).newbyteorder("<"))
    return _materialize(obj)


def load_checkpoint(data: bytes):
    """bytes of a .pth file -> nested dict/list of numpy arrays."""
    if data[:2] == b"PK":
        return _load_zip(data)
    return _load_legacy(data)


def state_dict(data: bytes) -> dict:
    """Best-effort extraction of a flat {name: array} state dict."""
    obj = load_checkpoint(data)
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "params", "params_ema", "model", "net"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    out = {}
    for k, v in obj.items():
        if isinstance(v, np.ndarray):
            out[k[len("module."):] if k.startswith("module.") else k] = v
    return out
