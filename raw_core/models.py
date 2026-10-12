"""Model discovery, download and inference back-ends.

The RawNIND weights are the ONNX exports published by the darktable project
(darktable-org/darktable-ai, release-5.6.0, package `rawdenoise-nind`,
GPL-3.0).  A `.dtmodel` file is a zip holding `model_bayer.onnx` and
`model_linear.onnx`; loose `.onnx` files work too.
"""

from __future__ import annotations

import hashlib
import os
import threading
import zipfile
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .graph_exec import NumpyGraph, TorchGraph
from .onnx_lite import load_onnx

MODEL_URL = ("https://github.com/darktable-org/darktable-ai/releases/download/"
             "release-5.6.0/rawdenoise-nind.dtmodel")
MODEL_FILENAME = "rawdenoise-nind.dtmodel"
MODEL_SHA256 = "d71b5f1e727c85a359e6f74dca9e2016c9d8fc3e2f7ac3e9b347d80ceca969af"


def download_model(dest_dir: str, log=print) -> str:
    """Fetch the RawNIND package from GitHub and verify its checksum."""
    import urllib.request

    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, MODEL_FILENAME)
    if os.path.isfile(dest):
        return dest
    tmp = dest + ".part"
    log(f"[RAW Denoise] downloading RawNIND model (58 MB) from {MODEL_URL}")
    h = hashlib.sha256()
    with urllib.request.urlopen(MODEL_URL, timeout=60) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            f.write(chunk)
    if h.hexdigest() != MODEL_SHA256:
        os.remove(tmp)
        raise RuntimeError("downloaded model failed its SHA-256 check; please download it manually")
    os.replace(tmp, dest)
    log(f"[RAW Denoise] saved {dest}")
    return dest


def read_model_bytes(path: str) -> dict[str, bytes]:
    """Return {'bayer': bytes?, 'linear': bytes?} from a .dtmodel/.zip/.onnx/dir."""
    found: dict[str, bytes] = {}

    def classify(name: str) -> Optional[str]:
        n = os.path.basename(name).lower()
        if not n.endswith(".onnx"):
            return None
        if "bayer" in n:
            return "bayer"
        if "linear" in n or "prgb" in n:
            return "linear"
        return None

    if os.path.isdir(path):
        for f in os.listdir(path):
            k = classify(f)
            if k:
                with open(os.path.join(path, f), "rb") as fh:
                    found[k] = fh.read()
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                k = classify(n)
                if k:
                    found[k] = z.read(n)
    else:
        k = classify(path) or "bayer"
        with open(path, "rb") as fh:
            found[k] = fh.read()
    if not found:
        raise FileNotFoundError(f"no RawNIND ONNX models found in {path}")
    return found


# ----------------------------------------------------------------------------
# runners: callable(np.ndarray (1,C,h,w) float32) -> np.ndarray (1,3,h',w')
# ----------------------------------------------------------------------------

class OrtRunner:
    def __init__(self, data: bytes):
        import onnxruntime as ort

        avail = ort.get_available_providers()
        # CoreML is deliberately skipped: it runs in FP16 on the GPU/ANE and the
        # Bayer model's activations overflow FP16 (noted by the darktable devs).
        prefs = [p for p in ("CUDAExecutionProvider", "ROCMExecutionProvider",
                             "DmlExecutionProvider") if p in avail]
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(data, sess_options=so,
                                         providers=prefs + ["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        shp = inp.shape
        self.fixed_tile = shp[2] if isinstance(shp[2], int) else None
        self.in_channels = shp[1] if isinstance(shp[1], int) else None
        self.device_name = "onnxruntime/" + self.sess.get_providers()[0].replace("ExecutionProvider", "")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {self.input_name: np.ascontiguousarray(x, dtype=np.float32)})[0]


class TorchRunner:
    def __init__(self, data: bytes, device):
        import torch

        graph = load_onnx(data)
        self.torch = torch
        self.graph = graph
        self.device = device
        self.exec = TorchGraph(graph, device, torch.float32)
        w0 = graph.initializers[graph.nodes[0].inputs[1]]
        self.in_channels = int(w0.shape[1])
        self.fixed_tile = None
        self.device_name = f"torch/{device}"
        self._verified = False

    def self_test(self):
        """Compare against the NumPy reference implementation on a small input.

        Guards against op-semantics differences on unusual torch builds/devices.
        """
        if self._verified:
            return
        rng = np.random.default_rng(1234)
        x = (rng.random((1, self.in_channels, 48, 48), dtype=np.float32) * 0.4 + 0.05)
        ref = NumpyGraph(self.graph)(x)
        got = self(x)
        err = float(np.abs(got - ref).max() / (np.abs(ref).max() + 1e-12))
        if not np.isfinite(err) or err > 1e-3:
            raise RuntimeError(
                f"RAW Denoise: torch backend self-test failed on {self.device} (rel. error {err:.2e}). "
                "Switch the node's backend to 'onnxruntime' or 'torch (cpu)'.")
        self._verified = True

    def __call__(self, x: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.inference_mode():
            t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
            y = self.exec(t)
            return y.float().cpu().numpy()


@dataclass
class ModelSet:
    bayer: Optional[object]
    linear: Optional[object]
    source: str


_CACHE: dict = {}
_LOCK = threading.Lock()


def load_models(path: str, backend: str = "auto", device=None, log=print) -> ModelSet:
    """backend: 'auto' | 'torch' | 'torch (cpu)' | 'onnxruntime'."""
    key = (os.path.abspath(path), backend, str(device))
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
        blobs = read_model_bytes(path)
        use_torch = backend.startswith("torch")
        if backend == "auto":
            try:
                import torch  # noqa: F401
                use_torch = True
            except ImportError:
                use_torch = False
        runners = {}
        for k, data in blobs.items():
            r = None
            if use_torch:
                try:
                    import torch
                    dev = torch.device("cpu") if backend == "torch (cpu)" else (device or torch.device("cpu"))
                    r = TorchRunner(data, dev)
                    r.self_test()
                except Exception as e:
                    if backend != "auto":
                        raise
                    log(f"[RAW Denoise] torch backend unavailable ({e}); trying onnxruntime")
                    r = None
            if r is None:
                try:
                    r = OrtRunner(data)
                except ImportError as e:
                    raise ImportError(
                        "backend 'onnxruntime' needs the onnxruntime package "
                        "(pip install onnxruntime, or onnxruntime-gpu for NVIDIA)") from e
            runners[k] = r
        ms = ModelSet(runners.get("bayer"), runners.get("linear"), path)
        if ms.linear is None and ms.bayer is None:
            raise RuntimeError(f"{path}: no usable models")
        _CACHE.clear()  # keep one model set resident
        _CACHE[key] = ms
        log(f"[RAW Denoise] loaded {os.path.basename(path)} "
            f"({', '.join(sorted(runners))}) on {next(iter(runners.values())).device_name}")
        return ms
