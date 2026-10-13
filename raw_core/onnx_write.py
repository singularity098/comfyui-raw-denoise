"""Minimal ONNX serializer (protobuf wire format) for OnnxGraph objects.

Lets graphs built in Python (e.g. FastDVDnet from a .pth checkpoint) run in
ONNX Runtime without needing torch or the onnx package.
"""

from __future__ import annotations

import struct

import numpy as np

from .onnx_lite import OnnxGraph

_DTYPE_CODES = {np.dtype(np.float32): 1, np.dtype(np.float16): 10, np.dtype(np.int64): 7,
                np.dtype(np.int32): 6, np.dtype(np.uint8): 2, np.dtype(np.float64): 11}


def _varint(v: int) -> bytes:
    if v < 0:
        v += 1 << 64
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _key(field: int, wire: int) -> bytes:
    return _varint((field << 3) | wire)


def _len_field(field: int, payload: bytes) -> bytes:
    return _key(field, 2) + _varint(len(payload)) + payload


def _str(field: int, s: str) -> bytes:
    return _len_field(field, s.encode("utf-8"))


def _int(field: int, v: int) -> bytes:
    return _key(field, 0) + _varint(int(v))


def _attribute(name: str, value) -> bytes:
    out = _str(1, name)
    if isinstance(value, (list, tuple)):
        if all(isinstance(v, (int, np.integer)) for v in value):
            out += b"".join(_int(8, v) for v in value) + _int(20, 7)  # INTS
        else:
            out += b"".join(_key(7, 5) + struct.pack("<f", float(v)) for v in value) + _int(20, 6)
    elif isinstance(value, (int, np.integer)):
        out += _int(3, value) + _int(20, 2)
    elif isinstance(value, float):
        out += _key(2, 5) + struct.pack("<f", value) + _int(20, 1)
    elif isinstance(value, str):
        out += _len_field(4, value.encode("utf-8")) + _int(20, 3)
    else:
        raise TypeError(f"unsupported attribute {name}={value!r}")
    return out


def _tensor(name: str, arr: np.ndarray) -> bytes:
    arr = np.ascontiguousarray(arr)
    out = b"".join(_int(1, d) for d in arr.shape)
    out += _int(2, _DTYPE_CODES[arr.dtype])
    out += _str(8, name)
    out += _len_field(9, arr.astype(arr.dtype.newbyteorder("<"), copy=False).tobytes())
    return out


def _value_info(name: str, rank: int, elem_type: int = 1, dims=None) -> bytes:
    dims = dims or [f"d{i}" for i in range(rank)]
    shape = b""
    for d in dims:
        dim = _int(1, d) if isinstance(d, int) else _str(2, d)
        shape += _len_field(1, dim)
    tensor_type = _int(1, elem_type) + _len_field(2, shape)
    type_proto = _len_field(1, tensor_type)
    return _str(1, name) + _len_field(2, type_proto)


def to_onnx_bytes(g: OnnxGraph, opset: int = 17, input_dims=None, output_dims=None) -> bytes:
    graph = b""
    for nd in g.nodes:
        node = b"".join(_str(1, i) for i in nd.inputs)
        node += b"".join(_str(2, o) for o in nd.outputs)
        node += _str(3, nd.name or nd.outputs[0]) + _str(4, nd.op_type)
        node += b"".join(_len_field(5, _attribute(k, v)) for k, v in nd.attrs.items())
        graph += _len_field(1, node)
    graph += _str(2, "graph")
    for k, v in g.initializers.items():
        graph += _len_field(5, _tensor(k, v))
    for name in g.inputs:
        graph += _len_field(11, _value_info(name, 4, dims=input_dims or ["N", "C", "H", "W"]))
    for name in g.outputs:
        graph += _len_field(12, _value_info(name, 4, dims=output_dims or ["N", "Co", "Ho", "Wo"]))
    model = _int(1, 8)  # ir_version
    model += _str(2, "comfyui-raw-denoise")
    model += _len_field(7, graph)
    model += _len_field(8, _str(1, "") + _int(2, opset))
    return model
