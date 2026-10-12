"""Minimal, dependency-free reader for the parts of an ONNX file we need.

The darktable / RawNIND models are distributed as ONNX.  To run them on the
GPU through PyTorch (ComfyUI's own runtime) we only need the weight tensors
(graph initializers) and the node list, so instead of requiring the `onnx`
package we decode the protobuf wire format directly.

Only the handful of fields used below are decoded; everything else is skipped.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

# ONNX TensorProto.DataType -> numpy dtype
_ONNX_DTYPES = {
    1: np.float32,
    2: np.uint8,
    3: np.int8,
    5: np.int16,
    6: np.int32,
    7: np.int64,
    10: np.float16,
    11: np.float64,
}


def _read_varint(buf: memoryview, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7


def _iter_fields(buf: memoryview):
    """Yield (field_number, wire_type, value) for one protobuf message.

    value is an int for varint/fixed types and a memoryview for
    length-delimited fields.
    """
    pos = 0
    end = len(buf)
    while pos < end:
        key, pos = _read_varint(buf, pos)
        fnum, wtype = key >> 3, key & 7
        if wtype == 0:  # varint
            val, pos = _read_varint(buf, pos)
        elif wtype == 1:  # 64-bit
            val = struct.unpack_from("<q", buf, pos)[0]
            pos += 8
        elif wtype == 2:  # length-delimited
            ln, pos = _read_varint(buf, pos)
            val = buf[pos:pos + ln]
            pos += ln
        elif wtype == 5:  # 32-bit
            val = struct.unpack_from("<i", buf, pos)[0]
            pos += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wtype}")
        yield fnum, wtype, val


def _packed_varints(view: memoryview) -> list[int]:
    out, pos = [], 0
    while pos < len(view):
        v, pos = _read_varint(view, pos)
        out.append(v)
    return out


def _signed64(v: int) -> int:
    return v - (1 << 64) if v >= (1 << 63) else v


@dataclass
class OnnxNode:
    op_type: str
    name: str
    inputs: list[str]
    outputs: list[str]
    attrs: dict = field(default_factory=dict)


@dataclass
class OnnxGraph:
    nodes: list[OnnxNode]
    initializers: dict[str, np.ndarray]
    inputs: list[str]
    outputs: list[str]


def _parse_tensor(view: memoryview) -> tuple[str, np.ndarray]:
    dims: list[int] = []
    dtype_code = 1
    name = ""
    raw = None
    float_data: list[float] = []
    int64_data: list[int] = []
    for fnum, wtype, val in _iter_fields(view):
        if fnum == 1:  # dims
            if wtype == 2:
                dims.extend(_signed64(v) for v in _packed_varints(val))
            else:
                dims.append(_signed64(val))
        elif fnum == 2:
            dtype_code = val
        elif fnum == 8:
            name = bytes(val).decode("utf-8")
        elif fnum == 9:
            raw = bytes(val)
        elif fnum == 4:  # float_data (packed)
            if wtype == 2:
                float_data.extend(np.frombuffer(bytes(val), dtype="<f4").tolist())
            else:
                float_data.append(struct.unpack("<f", struct.pack("<i", val))[0])
        elif fnum == 7:  # int64_data
            if wtype == 2:
                int64_data.extend(_signed64(v) for v in _packed_varints(val))
            else:
                int64_data.append(_signed64(val))
    dtype = _ONNX_DTYPES.get(dtype_code)
    if dtype is None:
        raise ValueError(f"tensor {name!r}: unsupported ONNX dtype {dtype_code}")
    if raw is not None:
        arr = np.frombuffer(raw, dtype=np.dtype(dtype).newbyteorder("<")).astype(dtype)
    elif float_data:
        arr = np.asarray(float_data, dtype=dtype)
    elif int64_data:
        arr = np.asarray(int64_data, dtype=dtype)
    else:
        arr = np.zeros(0, dtype=dtype)
    return name, arr.reshape(dims) if dims else arr.reshape(())


def _parse_attribute(view: memoryview):
    name, ival, fval, ints, floats, sval = "", None, None, [], [], None
    for fnum, wtype, val in _iter_fields(view):
        if fnum == 1:
            name = bytes(val).decode("utf-8")
        elif fnum == 2:  # f (fixed32 float)
            fval = struct.unpack("<f", struct.pack("<i", val))[0]
        elif fnum == 3:  # i
            ival = _signed64(val)
        elif fnum == 4:  # s
            sval = bytes(val).decode("utf-8", "replace")
        elif fnum == 8:  # ints
            if wtype == 2:
                ints.extend(_signed64(v) for v in _packed_varints(val))
            else:
                ints.append(_signed64(val))
        elif fnum == 7:  # floats
            if wtype == 2:
                floats.extend(np.frombuffer(bytes(val), dtype="<f4").tolist())
            else:
                floats.append(struct.unpack("<f", struct.pack("<i", val))[0])
    if ints:
        value = ints
    elif floats:
        value = floats
    elif ival is not None:
        value = ival
    elif fval is not None:
        value = fval
    else:
        value = sval
    return name, value


def _parse_node(view: memoryview) -> OnnxNode:
    inputs, outputs, name, op, attrs = [], [], "", "", {}
    for fnum, _, val in _iter_fields(view):
        if fnum == 1:
            inputs.append(bytes(val).decode("utf-8"))
        elif fnum == 2:
            outputs.append(bytes(val).decode("utf-8"))
        elif fnum == 3:
            name = bytes(val).decode("utf-8")
        elif fnum == 4:
            op = bytes(val).decode("utf-8")
        elif fnum == 5:
            k, v = _parse_attribute(val)
            attrs[k] = v
    return OnnxNode(op, name, inputs, outputs, attrs)


def _value_info_name(view: memoryview) -> str:
    for fnum, _, val in _iter_fields(view):
        if fnum == 1:
            return bytes(val).decode("utf-8")
    return ""


def load_onnx(data: bytes) -> OnnxGraph:
    """Parse ONNX model bytes into nodes + initializer arrays."""
    model = memoryview(data)
    graph_view = None
    for fnum, _, val in _iter_fields(model):
        if fnum == 7:  # ModelProto.graph
            graph_view = val
    if graph_view is None:
        raise ValueError("not an ONNX model (no graph found)")
    nodes, inits, gin, gout = [], {}, [], []
    for fnum, _, val in _iter_fields(graph_view):
        if fnum == 1:
            nodes.append(_parse_node(val))
        elif fnum == 5:
            k, arr = _parse_tensor(val)
            inits[k] = arr
        elif fnum == 11:
            gin.append(_value_info_name(val))
        elif fnum == 12:
            gout.append(_value_info_name(val))
    gin = [n for n in gin if n not in inits]
    return OnnxGraph(nodes, inits, gin, gout)
