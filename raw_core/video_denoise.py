"""Streaming FastDVDnet video denoiser.

Memory stays constant however long the video is: frames are decoded one at a
time, a window of 5 frames is kept, and each output frame is encoded as soon
as it is ready.  On the GPU only the window, two cached stage-1 results and
the network's activations for one tile are resident.

Per output frame t (FastDVDnet uses frames t-2..t+2):
    stage 1: s1[i] = DenBlock1(f[i-1], f[i], f[i+1])   - cached, 1 new per frame
    stage 2: out[t] = DenBlock2(s1[t-1], s1[t], s1[t+1])
Neighbours never cross a scene cut or the ends of the clip (they are mirrored
inside the shot instead), so cuts don't ghost.

The network takes a per-pixel noise map.  Real camera noise depends on
brightness (dark areas are noisier in video), so a noise profile
sigma(brightness) is measured from the footage itself and turned into a map
for every frame; "strength" scales it.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

MAX_SIGMA = 75.0 / 255.0
LUMA = (0.2126, 0.7152, 0.0722)


# ----------------------------------------------------------------------------
# backend-neutral tensor ops (NCHW)
# ----------------------------------------------------------------------------

class NumpyOps:
    name = "numpy"

    def __init__(self):
        self.dtype = np.float32

    def from_u16(self, frame_hwc: np.ndarray):
        return (frame_hwc.transpose(2, 0, 1)[None].astype(np.float32) * (1.0 / 65535.0))

    def to_u16(self, t) -> np.ndarray:
        return np.clip(np.rint(t[0].transpose(1, 2, 0) * 65535.0), 0, 65535).astype(np.uint16)

    def to_float_hwc(self, t) -> np.ndarray:
        return np.ascontiguousarray(t[0].transpose(1, 2, 0), dtype=np.float32)

    def planes(self, arr) -> np.ndarray:
        """(3, H, W) uint8/uint16 code values -> (1, 3, H, W) float32."""
        return arr[None].astype(np.float32)

    def to_codes(self, t, bits: int) -> np.ndarray:
        mx = 2 ** bits - 1
        v = np.clip(np.rint(t[0]), 0, mx)
        return v.astype(np.uint8 if bits <= 8 else np.uint16)

    def cat(self, ts, axis=1):
        return np.concatenate(ts, axis=axis)

    def pad_reflect(self, t, bottom, right):
        if not bottom and not right:
            return t
        return np.pad(t, ((0, 0), (0, 0), (0, bottom), (0, right)), mode="reflect")

    def clip(self, t, lo, hi):
        return np.clip(t, lo, hi)

    def sqrt(self, t):
        return np.sqrt(t)

    def zeros(self, shape):
        return np.zeros(shape, np.float32)

    def box_blur(self, t, r):  # (1,1,H,W)
        a = t[0, 0].astype(np.float64)
        for axis in (0, 1):
            p = np.pad(a, [(r + 1, r) if ax == axis else (0, 0) for ax in range(2)], mode="edge")
            c = np.cumsum(p, axis=axis)
            n = a.shape[axis]
            a = (np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
                 - np.take(c, np.arange(0, n), axis=axis)) / (2 * r + 1)
        return a.astype(np.float32)[None, None]

    def interp(self, x, xs, ys):
        return np.interp(x, xs, ys).astype(np.float32)

    def luma(self, t):
        return (t[:, 0:1] * LUMA[0] + t[:, 1:2] * LUMA[1] + t[:, 2:3] * LUMA[2])

    def ramp(self, n, ov):
        return _ramp_np(n, ov)

    def outer(self, a, b):
        return a[:, None] * b[None, :]

    def thumb(self, t, k):
        y = self.luma(t)[0, 0]
        h, w = y.shape[0] // k * k, y.shape[1] // k * k
        return y[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))

    def mean_abs_diff(self, a, b) -> float:
        return float(np.mean(np.abs(a - b)))

    def sync(self):
        pass

    def empty_cache(self):
        pass


class TorchOps(NumpyOps):
    name = "torch"

    def __init__(self, device, dtype):
        import torch
        self.torch = torch
        self.F = torch.nn.functional
        self.device = device
        self.dtype = dtype

    def from_u16(self, frame_hwc):
        torch = self.torch
        # uint16 -> int16 bit trick (torch has limited uint16 support)
        s16 = (frame_hwc ^ 0x8000).view(np.int16)
        t = torch.from_numpy(np.ascontiguousarray(s16)).to(self.device, non_blocking=True)
        t = (t.to(torch.float32) + 32768.0) * (1.0 / 65535.0)
        return t.permute(2, 0, 1).unsqueeze(0).contiguous().to(self.dtype)

    def to_u16(self, t):
        torch = self.torch
        v = torch.round(t[0].permute(1, 2, 0).float().clamp(0, 1) * 65535.0) - 32768.0
        arr = v.to(torch.int16).cpu().numpy()
        return arr.view(np.uint16) ^ 0x8000

    def to_float_hwc(self, t):
        return t[0].permute(1, 2, 0).float().cpu().numpy()

    def planes(self, arr):
        torch = self.torch
        if arr.dtype == np.uint8:
            t = torch.from_numpy(np.ascontiguousarray(arr)).to(self.device, non_blocking=True).to(torch.float32)
        else:
            s16 = (arr ^ 0x8000).view(np.int16)
            t = torch.from_numpy(np.ascontiguousarray(s16)).to(self.device, non_blocking=True)
            t = t.to(torch.float32) + 32768.0
        return t.unsqueeze(0)

    def to_codes(self, t, bits):
        torch = self.torch
        mx = float(2 ** bits - 1)
        v = torch.round(t[0].float()).clamp(0, mx)
        if bits <= 8:
            return v.to(torch.uint8).cpu().numpy()
        return (v - 32768.0).to(torch.int16).cpu().numpy().view(np.uint16) ^ 0x8000

    def cat(self, ts, axis=1):
        return self.torch.cat(ts, dim=axis)

    def pad_reflect(self, t, bottom, right):
        if not bottom and not right:
            return t
        return self.F.pad(t.float(), (0, right, 0, bottom), mode="reflect").to(t.dtype)

    def clip(self, t, lo, hi):
        return t.clamp(lo, hi)

    def sqrt(self, t):
        return self.torch.sqrt(t)

    def zeros(self, shape):
        return self.torch.zeros(shape, device=self.device, dtype=self.torch.float32)

    def box_blur(self, t, r):
        x = self.F.pad(t.float(), (r, r, r, r), mode="replicate")
        x = self.F.avg_pool2d(x, kernel_size=(2 * r + 1, 1), stride=1)
        x = self.F.avg_pool2d(x, kernel_size=(1, 2 * r + 1), stride=1)
        return x

    def interp(self, x, xs, ys):
        torch = self.torch
        xs_t = torch.as_tensor(np.asarray(xs, np.float32), device=self.device)
        ys_t = torch.as_tensor(np.asarray(ys, np.float32), device=self.device)
        xc = x.float().clamp(float(xs[0]), float(xs[-1]))
        idx = torch.bucketize(xc, xs_t).clamp(1, len(xs) - 1)
        x0, x1 = xs_t[idx - 1], xs_t[idx]
        y0, y1 = ys_t[idx - 1], ys_t[idx]
        w = (xc - x0) / (x1 - x0).clamp_min(1e-9)
        return y0 + w * (y1 - y0)

    def ramp(self, n, ov):
        return self.torch.from_numpy(_ramp_np(n, ov)).to(self.device)

    def outer(self, a, b):
        return a[:, None] * b[None, :]

    def thumb(self, t, k):
        y = self.luma(t.float())
        return self.F.avg_pool2d(y, k)[0, 0]

    def mean_abs_diff(self, a, b):
        return float((a - b).abs().mean())

    def sync(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def empty_cache(self):
        try:
            if self.device.type == "cuda":
                self.torch.cuda.empty_cache()
            elif self.device.type == "mps":
                self.torch.mps.empty_cache()
        except Exception:
            pass


def _thumb_u16(frame_u16: np.ndarray, k: int, planar: bool = False) -> np.ndarray:
    """Small luma thumbnail for cut detection, from a decimated frame (cheap on CPU)."""
    step = max(1, k // 4)
    if planar:  # (3, H, W) YUV codes: the Y plane is enough
        y = frame_u16[0, ::step, ::step].astype(np.float32) * (1.0 / float(np.iinfo(frame_u16.dtype).max))
    else:
        f = frame_u16[::step, ::step].astype(np.float32) * (1.0 / 65535.0)
        y = f @ np.array(LUMA, np.float32)
    kk = max(1, k // step)
    h, w = y.shape[0] // kk * kk, y.shape[1] // kk * kk
    return y[:h, :w].reshape(h // kk, kk, w // kk, kk).mean(axis=(1, 3))


def _ramp_np(n, ov):
    w = np.ones(n, np.float32)
    if ov > 0:
        r = (np.arange(ov, dtype=np.float32) + 0.5) / ov
        r = r * r * (3 - 2 * r)
        w[:ov] = np.minimum(w[:ov], r)
        w[-ov:] = np.minimum(w[-ov:], r[::-1])
    return np.maximum(w, 1e-4)


def _is_oom(e: BaseException) -> bool:
    msg = str(e).lower()
    return ("out of memory" in msg or "cuda error: out of memory" in msg or "cudnn_status_alloc" in msg
            or "mps backend out of memory" in msg or type(e).__name__ == "OutOfMemoryError")


# ----------------------------------------------------------------------------
# tiled DenBlock
# ----------------------------------------------------------------------------

class TiledRunner:
    """Runs a DenBlock over a (1,12,H,W) input in tiles that fit the GPU.

    The tile size starts from a VRAM estimate and halves automatically on an
    out-of-memory error, so processing never fails just because the frame
    is large; it only gets a little slower.
    """

    BYTES_PER_PX = {"fp16": 700, "bf16": 700, "fp32": 1400}
    OVERLAP = 32

    def __init__(self, ops, stage1, stage2, tile: int = 0, budget_bytes: Optional[int] = None,
                 precision: str = "fp32", log=print):
        self.ops, self.stages, self.log = ops, (stage1, stage2), log
        self.precision = precision
        if tile and tile > 0:
            self.tile = max(128, tile // 4 * 4)
        elif budget_bytes:
            px = budget_bytes / self.BYTES_PER_PX.get(precision, 1400)
            self.tile = max(256, int(math.sqrt(px)) // 4 * 4)
        else:
            self.tile = 1024
        self.calls = 0

    def __call__(self, stage: int, x):
        while True:
            try:
                return self._run(stage, x)
            except Exception as e:  # noqa: BLE001
                if not _is_oom(e) or self.tile <= 128:
                    raise
                self.ops.empty_cache()
                self.tile = max(128, (self.tile // 2) // 4 * 4)
                self.log(f"[Video Denoise] GPU memory full - continuing with {self.tile}px tiles")

    def _run(self, stage, x):
        net = self.stages[stage]
        _, _, H, W = x.shape
        T = self.tile
        if H <= T and W <= T:
            self.calls += 1
            return net(x)
        ov = self.OVERLAP
        th, tw = min(T, H), min(T, W)
        ys = _starts(H, th, th - 2 * ov)
        xs = _starts(W, tw, tw - 2 * ov)
        ops = self.ops
        out = ops.zeros((1, 3, H, W))
        wsum = ops.zeros((1, 1, H, W))
        wt = ops.outer(ops.ramp(th, ov), ops.ramp(tw, ov))
        for y in ys:
            for x0 in xs:
                tile = x[:, :, y:y + th, x0:x0 + tw]
                o = net(tile)
                self.calls += 1
                out[:, :, y:y + th, x0:x0 + tw] += o.astype(np.float32) * wt if ops.name == "numpy" \
                    else o.float() * wt
                wsum[:, :, y:y + th, x0:x0 + tw] += wt
                del o, tile
        return out / wsum


def _starts(length, tile, step):
    """Tile origins: multiples of 4 (keeps the stride-2 layers in phase), last one flush."""
    if length <= tile:
        return [0]
    step = max(4, step // 4 * 4)
    return sorted(set(list(range(0, length - tile, step)) + [length - tile]))


# ----------------------------------------------------------------------------
# noise profile
# ----------------------------------------------------------------------------

@dataclass
class NoiseProfile:
    xs: np.ndarray        # brightness bin centres (luma, 0..1)
    ys: np.ndarray        # network sigma per bin (0..1 units)
    blocks: int = 0
    spread: Optional[np.ndarray] = None   # measured noise spread per bin (0..1), for texture

    def describe(self) -> str:
        pts = ", ".join(f"{x:.2f}:{y * 255:.1f}" for x, y in zip(self.xs[::2], self.ys[::2]))
        return f"noise sigma (0-255 units) by brightness: {pts}"

    @property
    def mean_sigma255(self) -> float:
        return float(np.median(self.ys) * 255)


def _profile_from_residual(noisy: np.ndarray, den: np.ndarray, bins: int = 12):
    """Per-brightness robust std of (noisy - denoised); arrays (H, W, 3) float."""
    r = (noisy - den).astype(np.float32)
    y = den @ np.array(LUMA, np.float32)
    ok = (noisy.max(axis=2) < 0.985) & (noisy.min(axis=2) > 0.002)
    edges = np.linspace(0, 1, bins + 1)
    out = np.full(bins, np.nan)
    counts = np.zeros(bins, int)
    for i in range(bins):
        m = ok & (y >= edges[i]) & (y < edges[i + 1])
        counts[i] = int(m.sum())
        if counts[i] >= 400:
            rr = r[m]
            mad = np.median(np.abs(rr - np.median(rr, axis=0)), axis=0) * 1.4826
            out[i] = float(np.sqrt(np.mean(mad ** 2)))
    return out, counts


# Network sigma for a measured noise spread m (robust std of what the network
# removes at a fixed probe level), both in 0-255 units.  Camera video noise is
# coarse (demosaicing, in-camera NR, compression), so an AWGN-trained network
# needs a sigma well above the spread, increasingly so for noisier footage.
# Fitted on simulated low-light footage with ground truth (Poisson-Gaussian
# sensor noise, camera-like correlation, H.264/H.265 at camera bit rates).
LAW_A, LAW_E = 2.1, 0.30


def sigma_for_spread(m255):
    return LAW_A * (np.asarray(m255, np.float64) / 3.0) ** LAW_E * m255


def spread_for_sigma(s255):
    return 3.0 * (np.asarray(s255, np.float64) / (3.0 * LAW_A)) ** (1.0 / (1.0 + LAW_E))


def flat_profile(sigma255: float) -> NoiseProfile:
    xs = np.array([0.0, 1.0], np.float32)
    return NoiseProfile(xs, np.full(2, sigma255 / 255, np.float32),
                        spread=np.full(2, spread_for_sigma(max(sigma255, 0.1)) / 255, np.float32))


# ----------------------------------------------------------------------------
# the frame pipeline
# ----------------------------------------------------------------------------

@dataclass
class DenoiseSettings:
    strength: float = 1.0       # multiplies the noise profile
    grain: float = 0.0          # 0..100 % of the removed luminance noise put back
    detect_cuts: bool = True
    cut_threshold: float = 0.12
    texture: float = 1.5        # lower the noise level where local detail exceeds the noise
                                # (kappa: detail/noise ratio at which sigma drops by 1/sqrt2; 0 = off)


class VideoDenoiser:
    """Feed decoded frames in order with push(); finished frames come back in order.

    out frames are (H, W, 3) uint16 RGB (or float via return_float).
    """

    def __init__(self, ops, runner: TiledRunner, profile: NoiseProfile, settings: DenoiseSettings,
                 width: int, height: int, first_index: int = 0, emit_from: int = 0,
                 return_float: bool = False, codec=None):
        from .video_color import FrameCodec
        self.ops, self.runner, self.profile, self.s = ops, runner, profile, settings
        # how frames arrive / leave: rgb48 (default) or planar YUV 4:4:4 code values
        self.codec = codec or FrameCodec(layout="rgb")
        self.out_lo, self.out_hi = (0.0, 1.0) if self.codec.layout == "rgb" else (self.codec.lo, self.codec.hi)
        self.W, self.H = width, height
        self.ph, self.pw = (-height) % 4, (-width) % 4
        self.blur_r = max(2, int(round(min(width, height) / 135)))
        self.frames = {}      # idx -> padded (1,3,Hp,Wp)
        self.nmaps = {}       # idx -> (1,1,Hp,Wp)
        self.s1 = {}          # idx -> stage-1 output
        self.thumbs = {}
        self.cuts = set()     # i such that a cut lies between i-1 and i
        self.diffs = []
        self.next_in = first_index
        self.next_out = max(emit_from, first_index)
        self.emit_from = emit_from
        self.first = first_index
        self.total = None     # known at flush()
        self.return_float = return_float
        self.thumb_k = max(1, min(width, height) // 36)
        self.stats = {"frames": 0, "cuts": 0, "seconds": 0.0}

    # -- shot-aware neighbour mapping --------------------------------------
    def _shot(self, c):
        s = self.first
        for k in self.cuts:
            if s < k <= c:
                s = k
        e = None
        for k in self.cuts:
            if k > c and (e is None or k - 1 < e):
                e = k - 1
        if e is None:
            e = (self.total - 1) if self.total is not None else None
        return s, e

    def _nb(self, c, j):
        """Actual frame used for virtual index j when denoising frame c: mirrored
        at the ends of c's shot (same border rule as the FastDVDnet reference code)."""
        s, e = self._shot(c)
        if e is not None and e - s < 1:
            return c
        for _ in range(4):
            if j < s:
                j = 2 * s - j
            elif e is not None and j > e:
                j = 2 * e - j
            else:
                break
        return min(max(j, s), e) if e is not None else max(j, s)

    # -- per-frame helpers --------------------------------------------------
    def _noise_map(self, f):
        """Per-pixel sigma: the noise profile at the local brightness, lowered
        where the local luma variance shows real detail well above the noise
        (an AWGN-trained network otherwise smooths fine texture that is merely
        noise-like at the level asked for)."""
        ops, p = self.ops, self.profile
        y = ops.luma(f.float() if ops.name == "torch" else f)          # fp32 always
        yb = ops.box_blur(y, self.blur_r)
        sig = ops.interp(yb, p.xs, p.ys) * float(self.s.strength)
        kap = float(self.s.texture)
        if kap > 0 and p.spread is not None:
            m = ops.interp(yb, p.xs, np.maximum(np.asarray(p.spread, np.float32), 0.5 / 255))
            mu = ops.box_blur(y, 3)
            var = ops.box_blur(y * y, 3) - mu * mu
            tex2 = ops.clip(var - m * m, 0.0, None)
            sig = sig / ops.sqrt(1.0 + tex2 / ((kap * m) * (kap * m)))
        sig = ops.clip(sig, 0.0, MAX_SIGMA)
        return sig.to(f.dtype) if ops.name == "torch" else sig.astype(np.float32)

    def _stage1(self, t, v):
        """Stage-1 result for virtual centre v (t-1, t or t+1) while denoising t."""
        key = (self._nb(t, v - 1), self._nb(t, v), self._nb(t, v + 1))
        if key not in self.s1:
            ops = self.ops
            a, b, c = (self.frames[k] for k in key)
            nm = self.nmaps[key[1]]
            x = ops.cat([a, nm, b, nm, c, nm])
            est = self.runner(0, x)
            del x
            self.s1[key] = (b - est.to(b.dtype)) if ops.name == "torch" else (b - est)
        return self.s1[key]

    def _output(self, t):
        ops = self.ops
        a = self._stage1(t, t - 1)
        b = self._stage1(t, t)
        c = self._stage1(t, t + 1)
        nm = self.nmaps[t]
        x = ops.cat([a, nm, b, nm, c, nm])
        est = self.runner(1, x)
        del x
        out = (b.float() - est.float()) if ops.name == "torch" else (b - est)
        out = ops.clip(out, self.out_lo, self.out_hi)
        g = float(self.s.grain) / 100.0
        if g > 0:
            noisy = self.frames[t].float() if ops.name == "torch" else self.frames[t]
            r = ops.luma(noisy - out)
            out = ops.clip(out + g * r, self.out_lo, self.out_hi)
        out = out[:, :, :self.H, :self.W]
        return ops.to_float_hwc(out) if self.return_float else self.codec.encode(ops, out)

    def _evict(self, t):
        for d in (self.frames, self.nmaps):
            for k in [k for k in d if k < t - 2]:
                del d[k]
        for k in [k for k in self.s1 if min(k) < t - 2]:
            del self.s1[k]

    # -- public API -------------------------------------------------------------
    def push(self, frame_u16: np.ndarray) -> list:
        """Add the next decoded frame; returns the frames that are now finished."""
        i = self.next_in
        self.next_in += 1
        if self.s.detect_cuts:
            th = _thumb_u16(frame_u16, self.thumb_k, planar=self.codec.layout == "yuv")
            if i - 1 in self.thumbs:
                d = float(np.mean(np.abs(th - self.thumbs[i - 1])))
                med = float(np.median(self.diffs[-30:])) if self.diffs else 0.0
                if d > self.s.cut_threshold and d > 4 * med + 0.02:
                    self.cuts.add(i)
                    self.stats["cuts"] += 1
                self.diffs.append(d)
            self.thumbs[i] = th
            for k in [k for k in self.thumbs if k < i - 1]:
                del self.thumbs[k]
        if i < self.emit_from - 2:
            return []  # before the part we output (resume): only needed for cut detection
        ops = self.ops
        f = self.codec.decode(ops, frame_u16)
        if ops.name == "torch":
            f = f.to(ops.dtype)
        f = ops.pad_reflect(f, self.ph, self.pw)
        self.frames[i] = f
        self.nmaps[i] = self._noise_map(f)
        return self._drain(final=False)

    def _ready(self, t):
        # frames up to t+2 decoded (or end known)
        need = t + 2
        if self.total is not None:
            need = min(need, self.total - 1)
        return self.next_in - 1 >= need

    def _drain(self, final):
        out = []
        while self.next_out < self.next_in and (final or self._ready(self.next_out)):
            t = self.next_out
            t0 = time.time()
            out.append(self._output(t))
            self.stats["seconds"] += time.time() - t0
            self.stats["frames"] += 1
            self.next_out += 1
            self._evict(self.next_out)
        return out

    def flush(self) -> list:
        self.total = self.next_in
        return self._drain(final=True)
