"""FastDVDnet (Tassano, Delon & Veit, CVPR 2020) as backend-neutral graphs.

FastDVDnet denoises frame t from frames t-2..t+2 in two stages:
    stage 1: DenBlock "temp1" on the triplets (t-2,t-1,t), (t-1,t,t+1), (t,t+1,t+2)
    stage 2: DenBlock "temp2" on the three stage-1 results
Each DenBlock is a small U-Net that predicts the noise of its middle frame.

The checkpoint's BatchNorm layers are folded into the convolutions, and each
DenBlock becomes a graph of Conv / Relu / Add / DepthToSpace nodes that the
executors in graph_exec.py run on NumPy (reference), PyTorch (any GPU) or
ONNX Runtime.  The graph input is the 12-channel tensor
[f0, noise_map, f1, noise_map, f2, noise_map]; its output is the estimated
noise of f1 (the caller computes f1 - output).

Weights: MIT licence, https://github.com/m-tassano/fastdvdnet
"""

from __future__ import annotations

import numpy as np

from .onnx_lite import OnnxGraph, OnnxNode

BN_EPS = 1e-5


class _Builder:
    def __init__(self, sd: dict, prefix: str):
        self.sd, self.prefix = sd, prefix
        self.nodes: list[OnnxNode] = []
        self.inits: dict[str, np.ndarray] = {}
        self.n = 0

    def _name(self, base):
        self.n += 1
        return f"{base}_{self.n}"

    def conv(self, x, wkey, bnkey=None, stride=1, group=1):
        w = self.sd[f"{self.prefix}.{wkey}.weight"].astype(np.float64)
        b = np.zeros(w.shape[0])
        if bnkey is not None:
            g = self.sd[f"{self.prefix}.{bnkey}.weight"].astype(np.float64)
            beta = self.sd[f"{self.prefix}.{bnkey}.bias"].astype(np.float64)
            mean = self.sd[f"{self.prefix}.{bnkey}.running_mean"].astype(np.float64)
            var = self.sd[f"{self.prefix}.{bnkey}.running_var"].astype(np.float64)
            scale = g / np.sqrt(var + BN_EPS)
            w = w * scale[:, None, None, None]
            b = beta - mean * scale
        wn, bn = self._name("w"), self._name("b")
        self.inits[wn] = w.astype(np.float32)
        self.inits[bn] = b.astype(np.float32)
        out = self._name("conv")
        k = w.shape[-1]
        self.nodes.append(OnnxNode("Conv", out, [x, wn, bn], [out], {
            "kernel_shape": [k, k], "pads": [k // 2] * 4, "strides": [stride, stride],
            "dilations": [1, 1], "group": group}))
        return out

    def relu(self, x):
        out = self._name("relu")
        self.nodes.append(OnnxNode("Relu", out, [x], [out]))
        return out

    def add(self, a, b):
        out = self._name("add")
        self.nodes.append(OnnxNode("Add", out, [a, b], [out]))
        return out

    def pixel_shuffle(self, x, r=2):
        out = self._name("ps")
        self.nodes.append(OnnxNode("DepthToSpace", out, [x], [out], {"blocksize": r, "mode": "CRD"}))
        return out

    def cvblock(self, x, key):
        x = self.relu(self.conv(x, f"{key}.convblock.0", f"{key}.convblock.1"))
        return self.relu(self.conv(x, f"{key}.convblock.3", f"{key}.convblock.4"))


def denblock_graph(sd: dict, prefix: str) -> OnnxGraph:
    b = _Builder(sd, prefix)
    x = "input"
    # InputCvBlock: grouped conv (3 groups = 3 frames, each RGB + noise map)
    x0 = b.relu(b.conv(x, "inc.convblock.0", "inc.convblock.1", group=3))
    x0 = b.relu(b.conv(x0, "inc.convblock.3", "inc.convblock.4"))
    # DownBlocks
    x1 = b.relu(b.conv(x0, "downc0.convblock.0", "downc0.convblock.1", stride=2))
    x1 = b.cvblock(x1, "downc0.convblock.3")
    x2 = b.relu(b.conv(x1, "downc1.convblock.0", "downc1.convblock.1", stride=2))
    x2 = b.cvblock(x2, "downc1.convblock.3")
    # UpBlocks
    x2 = b.cvblock(x2, "upc2.convblock.0")
    x2 = b.pixel_shuffle(b.conv(x2, "upc2.convblock.1"))
    x1 = b.cvblock(b.add(x1, x2), "upc1.convblock.0")
    x1 = b.pixel_shuffle(b.conv(x1, "upc1.convblock.1"))
    # OutputCvBlock
    x = b.relu(b.conv(b.add(x0, x1), "outc.convblock.0", "outc.convblock.1"))
    x = b.conv(x, "outc.convblock.3")
    b.nodes[-1].outputs = ["output"]
    return OnnxGraph(b.nodes, b.inits, ["input"], ["output"])


def load_fastdvdnet(sd: dict) -> tuple[OnnxGraph, OnnxGraph]:
    """state dict (numpy) -> (temp1 graph, temp2 graph)."""
    need = "temp1.inc.convblock.0.weight"
    if need not in sd:
        raise ValueError("not a FastDVDnet checkpoint")
    return denblock_graph(sd, "temp1"), denblock_graph(sd, "temp2")


def pack_input(f0, f1, f2, noise_map, xp=np):
    """Build the 12-channel DenBlock input. fN: (N,3,H,W), noise_map: (N,1,H,W)."""
    if xp is np:
        return np.concatenate([f0, noise_map, f1, noise_map, f2, noise_map], axis=1)
    return xp.cat([f0, noise_map, f1, noise_map, f2, noise_map], dim=1)
