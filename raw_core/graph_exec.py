"""Tiny ONNX graph executor with NumPy and PyTorch backends.

Supports exactly the operator set used by the RawNIND UtNet2 denoisers
(Conv, LeakyRelu, MaxPool, ConvTranspose, Concat, DepthToSpace).  Running the
graph through PyTorch lets ComfyUI users use whatever GPU torch already has
(CUDA, ROCm, Apple MPS, Intel XPU) with no extra packages and with dynamic
tile sizes.  The NumPy backend is a slow reference used for self-tests.
"""

from __future__ import annotations

import numpy as np

from .onnx_lite import OnnxGraph

SUPPORTED_OPS = {"Conv", "LeakyRelu", "MaxPool", "ConvTranspose", "Concat", "DepthToSpace", "Relu", "Add"}


def last_uses(graph: OnnxGraph) -> dict:
    """node index -> names whose last consumer is that node (so they can be freed)."""
    last = {}
    for i, nd in enumerate(graph.nodes):
        for name in nd.inputs:
            if name and name not in graph.initializers:
                last[name] = i
    out = {}
    for name, i in last.items():
        if name not in graph.outputs:
            out.setdefault(i, []).append(name)
    return out


def check_supported(graph: OnnxGraph) -> None:
    bad = sorted({n.op_type for n in graph.nodes} - SUPPORTED_OPS)
    if bad:
        raise NotImplementedError(f"model uses unsupported ONNX ops: {bad}")


# ----------------------------------------------------------------------------
# NumPy reference ops (NCHW, float32)
# ----------------------------------------------------------------------------

def _np_conv(x, w, b, pads, strides, dilations, group):
    if any(d != 1 for d in dilations):
        raise NotImplementedError("numpy backend: dilated conv not supported")
    if group != 1:
        ci_g = x.shape[1] // group
        co_g = w.shape[0] // group
        outs = [_np_conv(x[:, g * ci_g:(g + 1) * ci_g], w[g * co_g:(g + 1) * co_g],
                         None if b is None else b[g * co_g:(g + 1) * co_g], pads, strides, dilations, 1)
                for g in range(group)]
        return np.concatenate(outs, axis=1)
    pt, pl, pb, pr = pads
    x = np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)))
    n, c, h, wd = x.shape
    co, ci, kh, kw = w.shape
    sh, sw = strides
    oh = (h - kh) // sh + 1
    ow = (wd - kw) // sw + 1
    out = np.zeros((n, co, oh, ow), dtype=np.float32)
    wmat = w.reshape(co, ci, kh * kw)
    # accumulate one kernel tap at a time: memory-light im2col
    for ky in range(kh):
        for kx in range(kw):
            patch = x[:, :, ky:ky + sh * oh:sh, kx:kx + sw * ow:sw]  # n,ci,oh,ow
            out += np.einsum("oc,nchw->nohw", wmat[:, :, ky * kw + kx], patch, optimize=True)
    if b is not None:
        out += b.reshape(1, -1, 1, 1)
    return out


def _np_conv_transpose(x, w, b, strides, pads, kernel):
    # only the non-overlapping case (kernel == stride, no padding) is needed
    if list(kernel) != list(strides) or any(pads):
        raise NotImplementedError("numpy backend: only kernel==stride ConvTranspose")
    n, ci, h, wd = x.shape
    _, co, kh, kw = w.shape
    y = np.einsum("nchw,cokl->nohkwl", x, w, optimize=True)
    y = y.reshape(n, co, h * kh, wd * kw)
    if b is not None:
        y += b.reshape(1, -1, 1, 1)
    return y.astype(np.float32)


def _np_maxpool(x, k, s):
    if list(k) != [2, 2] or list(s) != [2, 2]:
        raise NotImplementedError("numpy backend: only 2x2/2 max pool")
    n, c, h, w = x.shape
    x = x[:, :, : h // 2 * 2, : w // 2 * 2]
    return x.reshape(n, c, h // 2, 2, w // 2, 2).max(axis=(3, 5))


def _np_depth_to_space(x, bs, mode):
    n, c, h, w = x.shape
    if mode == "CRD":
        y = x.reshape(n, c // (bs * bs), bs, bs, h, w).transpose(0, 1, 4, 2, 5, 3)
    else:  # DCR
        y = x.reshape(n, bs, bs, c // (bs * bs), h, w).transpose(0, 3, 4, 1, 5, 2)
    return y.reshape(n, c // (bs * bs), h * bs, w * bs)


class NumpyGraph:
    def __init__(self, graph: OnnxGraph):
        check_supported(graph)
        self.g = graph
        self.free = last_uses(graph)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        env = dict(self.g.initializers)
        env[self.g.inputs[0]] = x.astype(np.float32)
        for idx, nd in enumerate(self.g.nodes):
            a = nd.attrs
            ins = [env[i] if i else None for i in nd.inputs]
            op = nd.op_type
            if op == "Conv":
                y = _np_conv(ins[0], ins[1], ins[2] if len(ins) > 2 else None,
                             a.get("pads", [0, 0, 0, 0]), a.get("strides", [1, 1]),
                             a.get("dilations", [1, 1]), a.get("group", 1))
            elif op == "LeakyRelu":
                alpha = a.get("alpha", 0.01)
                y = np.where(ins[0] >= 0, ins[0], ins[0] * alpha).astype(np.float32)
            elif op == "Relu":
                y = np.maximum(ins[0], 0)
            elif op == "Add":
                y = ins[0] + ins[1]
            elif op == "MaxPool":
                y = _np_maxpool(ins[0], a["kernel_shape"], a.get("strides", a["kernel_shape"]))
            elif op == "ConvTranspose":
                y = _np_conv_transpose(ins[0], ins[1], ins[2] if len(ins) > 2 else None,
                                       a.get("strides", [1, 1]), a.get("pads", [0, 0, 0, 0]),
                                       a["kernel_shape"])
            elif op == "Concat":
                y = np.concatenate(ins, axis=a["axis"])
            elif op == "DepthToSpace":
                y = _np_depth_to_space(ins[0], a["blocksize"], a.get("mode", "DCR"))
            else:  # pragma: no cover
                raise NotImplementedError(op)
            env[nd.outputs[0]] = y
            for name in self.free.get(idx, ()):
                env.pop(name, None)
        return env[self.g.outputs[0]]


# ----------------------------------------------------------------------------
# PyTorch backend
# ----------------------------------------------------------------------------

class TorchGraph:
    """Executes the graph with torch.nn.functional on any torch device."""

    def __init__(self, graph: OnnxGraph, device, dtype=None):
        import torch

        check_supported(graph)
        self.torch = torch
        self.g = graph
        self.device = device
        self.dtype = dtype or torch.float32
        self.params = {
            k: torch.from_numpy(np.ascontiguousarray(v)).to(device=device, dtype=self.dtype)
            for k, v in graph.initializers.items()
            if v.dtype.kind == "f"
        }
        self.free = last_uses(graph)

    def to(self, device):
        if device != self.device:
            self.params = {k: v.to(device) for k, v in self.params.items()}
            self.device = device
        return self

    def __call__(self, x):
        torch = self.torch
        F = torch.nn.functional
        env = dict(self.params)
        env[self.g.inputs[0]] = x.to(device=self.device, dtype=self.dtype)
        del x  # keep only the graph's reference, so the input can be freed early
        for idx, nd in enumerate(self.g.nodes):
            a = nd.attrs
            ins = [env[i] if i else None for i in nd.inputs]
            op = nd.op_type
            if op == "Conv":
                pt, pl, pb, pr = a.get("pads", [0, 0, 0, 0])
                xin = ins[0]
                if (pt, pl) != (pb, pr):
                    xin = F.pad(xin, (pl, pr, pt, pb))
                    pad = (0, 0)
                else:
                    pad = (pt, pl)
                y = F.conv2d(xin, ins[1], ins[2] if len(ins) > 2 else None,
                             stride=tuple(a.get("strides", [1, 1])), padding=pad,
                             dilation=tuple(a.get("dilations", [1, 1])), groups=a.get("group", 1))
            elif op == "LeakyRelu":
                y = F.leaky_relu(ins[0], negative_slope=float(a.get("alpha", 0.01)))
            elif op == "Relu":
                y = F.relu(ins[0])
            elif op == "Add":
                y = ins[0] + ins[1]
            elif op == "MaxPool":
                y = F.max_pool2d(ins[0], kernel_size=tuple(a["kernel_shape"]),
                                 stride=tuple(a.get("strides", a["kernel_shape"])),
                                 ceil_mode=bool(a.get("ceil_mode", 0)))
            elif op == "ConvTranspose":
                pads = a.get("pads", [0, 0, 0, 0])
                if (pads[0], pads[1]) != (pads[2], pads[3]):
                    raise NotImplementedError("asymmetric ConvTranspose padding")
                y = F.conv_transpose2d(ins[0], ins[1], ins[2] if len(ins) > 2 else None,
                                       stride=tuple(a.get("strides", [1, 1])),
                                       padding=(pads[0], pads[1]),
                                       output_padding=tuple(a.get("output_padding", [0, 0])),
                                       groups=a.get("group", 1),
                                       dilation=tuple(a.get("dilations", [1, 1])))
            elif op == "Concat":
                y = torch.cat(ins, dim=a["axis"])
            elif op == "DepthToSpace":
                bs = a["blocksize"]
                if a.get("mode", "DCR") == "CRD":
                    y = F.pixel_shuffle(ins[0], bs)
                else:
                    n, c, h, w = ins[0].shape
                    y = ins[0].reshape(n, bs, bs, c // (bs * bs), h, w)
                    y = y.permute(0, 3, 4, 1, 5, 2).reshape(n, c // (bs * bs), h * bs, w * bs)
            else:  # pragma: no cover
                raise NotImplementedError(op)
            env[nd.outputs[0]] = y
            del ins, y
            for name in self.free.get(idx, ()):  # free activations after their last use
                env.pop(name, None)
        return env[self.g.outputs[0]]
