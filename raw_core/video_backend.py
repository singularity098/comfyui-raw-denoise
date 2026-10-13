"""FastDVDnet execution back-ends for the video nodes (PyTorch GPU / ONNX Runtime)."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass

import numpy as np

from .fastdvdnet import load_fastdvdnet
from .graph_exec import NumpyGraph, TorchGraph
from .pth_loader import state_dict
from .video_denoise import NumpyOps, TiledRunner, TorchOps

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(HERE, "weights", "fastdvdnet")
MODELS = {
    "FastDVDnet": "model.pth",
    "FastDVDnet (clipped-noise variant)": "model_clipped_noise.pth",
}
MODEL_SHA256 = {
    "model.pth": "9d9d8413c33e3d9d961d07c530237befa1197610b9d60602ff42fd77975d2a17",
    "model_clipped_noise.pth": "8118974ac7defaa5037f73caf87e0cb53efcfa49ae77d55c05ab187f59e55949",
}


def model_path(name: str) -> str:
    fn = MODELS.get(name, name)
    p = os.path.join(WEIGHTS_DIR, fn)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"FastDVDnet weights missing: {p} (re-download the node pack)")
    return p


@dataclass
class Backend:
    ops: object
    stage1: object
    stage2: object
    description: str
    precision: str
    device: object = None

    def runner(self, tile: int = 0, log=print) -> TiledRunner:
        budget = None
        if self.ops.name == "torch" and self.device is not None and self.device.type == "cuda":
            try:
                import torch
                free, _total = torch.cuda.mem_get_info(self.device)
                budget = int(free * 0.55)
            except Exception:
                budget = None
        elif self.ops.name == "torch" and self.device is not None and self.device.type == "mps":
            budget = 3 << 30
        return TiledRunner(self.ops, self.stage1, self.stage2, tile=tile, budget_bytes=budget,
                           precision=self.precision, log=log)


_CACHE: dict = {}
_LOCK = threading.Lock()


def _self_test(tg, g, dtype_name: str) -> float:
    import torch
    rng = np.random.default_rng(7)
    x = (rng.random((1, 12, 48, 64), dtype=np.float32) * 0.6 + 0.1)
    ref = NumpyGraph(g)(x)
    with torch.inference_mode():
        got = tg(torch.from_numpy(x)).float().cpu().numpy()
    return float(np.abs(got - ref).max() / (np.abs(ref).max() + 1e-9))


def make_backend(model: str, backend: str = "auto", device=None, precision: str = "auto", log=print) -> Backend:
    path = model_path(model)
    key = (path, backend, str(device), precision)
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
        with open(path, "rb") as f:
            g1, g2 = load_fastdvdnet(state_dict(f.read()))
        be = None
        want_torch = backend in ("auto", "torch", "torch (cpu)")
        if want_torch:
            try:
                be = _torch_backend(g1, g2, backend, device, precision, log)
            except ImportError:
                if backend != "auto":
                    raise
            except Exception as e:  # noqa: BLE001
                if backend != "auto":
                    raise
                log(f"[Video Denoise] PyTorch backend unavailable ({e}); trying onnxruntime")
        if be is None:
            be = _ort_backend(g1, g2, log)
        log(f"[Video Denoise] FastDVDnet on {be.description}")
        _CACHE.clear()
        _CACHE[key] = be
        return be


def _torch_backend(g1, g2, backend, device, precision, log) -> Backend:
    import torch
    dev = torch.device("cpu") if backend == "torch (cpu)" or device is None else torch.device(device)
    if precision == "auto":
        precision = "fp16" if dev.type in ("cuda", "mps", "xpu") else "fp32"
    order = [precision] + (["fp32"] if precision != "fp32" else [])
    for prec in order:
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[prec]
        t1, t2 = TorchGraph(g1, dev, dtype), TorchGraph(g2, dev, dtype)
        tol = 1e-4 if prec == "fp32" else 4e-2
        err = max(_self_test(t1, g1, prec), _self_test(t2, g2, prec))
        if np.isfinite(err) and err < tol:
            def wrap(tg):
                def run(x):
                    with torch.inference_mode():
                        return tg(x)
                return run
            return Backend(TorchOps(dev, dtype), wrap(t1), wrap(t2), f"torch/{dev} ({prec})", prec, dev)
        log(f"[Video Denoise] {prec} self-test failed on {dev} (rel. error {err:.1e}); trying fp32")
    raise RuntimeError(f"FastDVDnet self-test failed on {dev}; use backend 'torch (cpu)' or 'onnxruntime'")


def _ort_backend(g1, g2, log) -> Backend:
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError("neither a working PyTorch nor onnxruntime is available") from e
    from .onnx_write import to_onnx_bytes
    avail = ort.get_available_providers()
    prov = [p for p in ("CUDAExecutionProvider", "ROCMExecutionProvider", "DmlExecutionProvider") if p in avail]
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    s1 = ort.InferenceSession(to_onnx_bytes(g1), so, providers=prov + ["CPUExecutionProvider"])
    s2 = ort.InferenceSession(to_onnx_bytes(g2), so, providers=prov + ["CPUExecutionProvider"])

    def mk(s):
        return lambda x: s.run(None, {"input": np.ascontiguousarray(x, np.float32)})[0]
    name = s1.get_providers()[0].replace("ExecutionProvider", "")
    return Backend(NumpyOps(), mk(s1), mk(s2), f"onnxruntime/{name}", "fp32", None)
