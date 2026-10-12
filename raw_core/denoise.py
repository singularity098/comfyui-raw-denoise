"""RawNIND raw denoising pipeline (framework independent, NumPy in/out).

Bayer sensors  : the network sees the packed RGGB mosaic and does denoise +
                 demosaic jointly (Lightroom's Denoise works the same way).
Other sensors  : (Fuji X-Trans, ...) LibRaw demosaics first, then the linear
                 Rec.2020 variant of the network denoises the RGB image.

Model outputs come at an arbitrary learned gain (the network was trained with
gain matching), so each tile is re-scaled so that its re-mosaiced output has
the same mean as the input mosaic.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Optional

import numpy as np

from .color import apply_matrix, rgb_from_cam_matrix
from .rawio import RawImage

ProgressFn = Optional[Callable[[int, int], None]]


# ----------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------

def rggb_offset(pattern: np.ndarray, color_desc: str) -> tuple[int, int]:
    """(dy, dx) of the red photosite inside the 2x2 Bayer tile."""
    for i in range(2):
        for j in range(2):
            if color_desc[int(pattern[i, j])] == "R":
                return i, j
    raise ValueError(f"no red photosite in Bayer pattern {pattern.tolist()}")


def pack_rggb(cfa: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """(H, W) mosaic -> (4, h, w) [R, G1, G2, B] starting at the R site."""
    c = cfa[dy:, dx:]
    h, w = c.shape[0] // 2, c.shape[1] // 2
    c = c[: 2 * h, : 2 * w]
    return np.stack([c[0::2, 0::2], c[0::2, 1::2], c[1::2, 0::2], c[1::2, 1::2]], axis=0)


def remosaic_rggb(rgb: np.ndarray) -> np.ndarray:
    """(3, 2h, 2w) RGB -> (4, h, w) packed RGGB samples."""
    return np.stack([rgb[0, 0::2, 0::2], rgb[1, 0::2, 1::2], rgb[1, 1::2, 0::2], rgb[2, 1::2, 1::2]], axis=0)


def _ramp(n: int, ov: int) -> np.ndarray:
    """1-D blending weight: rises over `ov` samples at both ends."""
    w = np.ones(n, dtype=np.float32)
    if ov > 0:
        r = (np.arange(ov, dtype=np.float32) + 0.5) / ov
        r = r * r * (3 - 2 * r)  # smoothstep
        w[:ov] = np.minimum(w[:ov], r)
        w[-ov:] = np.minimum(w[-ov:], r[::-1])
    return np.maximum(w, 1e-4)


def _tile_starts(length: int, tile: int, step: int) -> list[int]:
    if length <= tile:
        return [0]
    n = math.ceil((length - tile) / step) + 1
    starts = [min(i * step, length - tile) for i in range(n)]
    return sorted(set(starts))


# ----------------------------------------------------------------------------
# tiled inference
# ----------------------------------------------------------------------------

def run_tiled(
    infer: Callable[[np.ndarray], np.ndarray],
    arr: np.ndarray,
    tile: int,
    overlap: int,
    scale: int,
    gain_ref: Callable[[np.ndarray, np.ndarray], float],
    fixed_tile: Optional[int] = None,
    progress: ProgressFn = None,
    cancel: Optional[Callable[[], None]] = None,
    per_tile_gain: bool = True,
) -> tuple[np.ndarray, list[float]]:
    """Feather-blended tiled inference.

    arr       : (C, H, W) float32 input
    infer     : (1, C, t, t) -> (1, 3, t*scale, t*scale)
    gain_ref  : f(tile_in, tile_out) -> gain making tile_out match tile_in
    fixed_tile: if the model has a static input size, every tile is exactly that.
    Returns ((3, H*scale, W*scale) output, per-tile gains).
    """
    c, H, W = arr.shape
    if fixed_tile:
        tile = fixed_tile
    tile = max(32, tile // 16 * 16)
    overlap = int(min(overlap, tile // 4))
    pad = overlap  # every real pixel is then covered by a non-faded tile region
    # reflect-pad; also pad up to at least one full tile
    ph = max(H + 2 * pad, tile)
    pw = max(W + 2 * pad, tile)
    padded = np.pad(arr, ((0, 0), (pad, ph - H - pad), (pad, pw - W - pad)), mode="symmetric")
    step = tile - overlap
    ys = _tile_starts(ph, tile, step)
    xs = _tile_starts(pw, tile, step)
    S = scale
    out = np.zeros((3, ph * S, pw * S), dtype=np.float32)
    wsum = np.zeros((ph * S, pw * S), dtype=np.float32)
    wy = _ramp(tile * S, overlap * S)
    wx = _ramp(tile * S, overlap * S)
    wtile = wy[:, None] * wx[None, :]
    gains = []
    total = len(ys) * len(xs)
    done = 0
    for y in ys:
        for x in xs:
            if cancel:
                cancel()
            t_in = np.ascontiguousarray(padded[None, :, y:y + tile, x:x + tile], dtype=np.float32)
            t_out = infer(t_in)[0].astype(np.float32, copy=False)
            g = gain_ref(t_in[0], t_out)
            gains.append(g)
            if per_tile_gain:
                if not np.isfinite(g):  # e.g. an all-black tile: borrow a neighbour's gain
                    finite = [v for v in gains if np.isfinite(v)]
                    g = finite[-1] if finite else 1.0
                t_out = t_out * g
            out[:, y * S:(y + tile) * S, x * S:(x + tile) * S] += t_out * wtile
            wsum[y * S:(y + tile) * S, x * S:(x + tile) * S] += wtile
            done += 1
            if progress:
                progress(done, total)
    out /= np.maximum(wsum, 1e-8)
    out = out[:, pad * S:(pad + H) * S, pad * S:(pad + W) * S]
    if not per_tile_gain:
        g = gain_ref(arr, out)
        out *= g if np.isfinite(g) else 1.0
    return out, gains


def _robust_gain(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) < 1e-12:
        return float("nan")
    return float(num / den)


def bayer_gain(t_in: np.ndarray, t_out: np.ndarray) -> float:
    """Match the re-mosaiced output to the input mosaic (same photosites).

    Comparing like with like matters: the packed input has two greens per
    2x2 block, so matching a plain RGB mean to it (as some implementations
    do) brightens the result by several percent.
    """
    m = remosaic_rggb(t_out)
    return _robust_gain(float(t_in.astype(np.float64).sum()), float(m.astype(np.float64).sum()))


def rgb_gain(t_in: np.ndarray, t_out: np.ndarray) -> float:
    return _robust_gain(float(t_in.astype(np.float64).sum()), float(t_out.astype(np.float64).sum()))


# ----------------------------------------------------------------------------
# public entry points
# ----------------------------------------------------------------------------

def denoise_bayer(img: RawImage, infer, tile: int = 512, overlap: int = 64,
                  fixed_tile: Optional[int] = None, progress: ProgressFn = None,
                  cancel=None) -> np.ndarray:
    """Joint demosaic + denoise of a Bayer mosaic -> (H, W, 3) camera RGB."""
    assert img.cfa is not None and img.sensor == "bayer"
    H, W = img.cfa.shape
    dy, dx = rggb_offset(img.cfa_pattern, img.color_desc)
    packed = pack_rggb(img.cfa, dy, dx)
    rgb, _ = run_tiled(infer, packed, tile, overlap, 2, bayer_gain, fixed_tile=fixed_tile,
                       progress=progress, cancel=cancel, per_tile_gain=False)
    out = np.empty((H, W, 3), dtype=np.float32)
    hh, ww = rgb.shape[1], rgb.shape[2]
    out[dy:dy + hh, dx:dx + ww] = rgb.transpose(1, 2, 0)
    # rows/columns cut off to start on an R site (at most one each side)
    if dy:
        out[:dy] = out[dy:dy + 1]
    if dx:
        out[:, :dx] = out[:, dx:dx + 1]
    if dy + hh < H:
        out[dy + hh:] = out[dy + hh - 1:dy + hh]
    if dx + ww < W:
        out[:, dx + ww:] = out[:, dx + ww - 1:dx + ww]
    return out


def denoise_linear(img: RawImage, infer, tile: int = 512, overlap: int = 64,
                   fixed_tile: Optional[int] = None, progress: ProgressFn = None,
                   cancel=None) -> np.ndarray:
    """Denoise an already demosaiced image with the linear Rec.2020 model."""
    to2020 = rgb_from_cam_matrix(img.cam_xyz, "rec2020")
    wb = img.wb.astype(np.float32)
    lin = apply_matrix(img.cam_rgb * wb, to2020)  # (H, W, 3)
    # Highlights above 1.0 are set aside and added back untouched. Negative
    # values (shadow noise) must stay in: clipping them and adding them back
    # afterwards leaves dark specks, clipping without adding back biases the
    # shadows upwards.
    arr = np.ascontiguousarray(np.minimum(lin, 1.0).transpose(2, 0, 1))
    lin -= np.minimum(lin, 1.0)          # lin now holds the excess above 1.0
    excess = lin
    rgb, _ = run_tiled(infer, arr, tile, overlap, 1, rgb_gain, fixed_tile=fixed_tile,
                       progress=progress, cancel=cancel, per_tile_gain=False)
    del arr
    excess += rgb.transpose(1, 2, 0)
    del rgb
    cam = apply_matrix(excess, np.linalg.inv(to2020))
    del excess
    cam /= wb
    return cam


def protect_highlights(noisy: np.ndarray, den: np.ndarray, clip_level: float,
                       lo: float = 0.90, hi: float = 0.98) -> np.ndarray:
    """Hand clipped / nearly clipped areas back to the original data.

    Raw editors detect blown highlights by values sitting exactly at the
    white level; a network output that wanders to 0.97 there turns into
    magenta highlights in Lightroom.  Noise is negligible that high up, so
    the original values are used and everything is clipped at clip_level.
    """
    peak = noisy.max(axis=2) / max(clip_level, 1e-6)
    t = np.clip((peak - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    if np.any(t > 0):
        # The network smears clipped photosites a few pixels outwards, which
        # shows up as a magenta rim after white balance: widen the mask so the
        # rim also keeps the original data, then feather it.
        t = _box_blur(_max_filter(t, 5), 2)
        t = t * t * (3 - 2 * t)
    if not den.flags.writeable:
        den = den.copy()
    if np.any(t > 0):
        diff = noisy - den
        diff *= t[..., None]
        den += diff  # den is modified in place (callers pass a fresh array)
        del diff
    np.minimum(den, np.float32(clip_level), out=den)
    return den


def _max_filter(a: np.ndarray, r: int) -> np.ndarray:
    """Separable square max filter (radius r) without SciPy."""
    out = a.copy()
    for axis in (0, 1):
        src = out.copy()
        n = src.shape[axis]
        for d in range(1, r + 1):
            sl_dst = [slice(None)] * 2
            sl_src = [slice(None)] * 2
            sl_dst[axis], sl_src[axis] = slice(d, n), slice(0, n - d)
            np.maximum(out[tuple(sl_dst)], src[tuple(sl_src)], out=out[tuple(sl_dst)])
            sl_dst[axis], sl_src[axis] = slice(0, n - d), slice(d, n)
            np.maximum(out[tuple(sl_dst)], src[tuple(sl_src)], out=out[tuple(sl_dst)])
    return out


def _box_blur(a: np.ndarray, r: int) -> np.ndarray:
    """Separable box blur (radius r) via cumulative sums, edge-replicated."""
    out = a.astype(np.float32)
    for axis in (0, 1):
        p = np.pad(out, [(r + 1, r) if ax == axis else (0, 0) for ax in range(2)], mode="edge")
        c = np.cumsum(p, axis=axis, dtype=np.float64)
        n = out.shape[axis]
        hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
        lo = np.take(c, np.arange(0, n), axis=axis)
        out = ((hi - lo) / (2 * r + 1)).astype(np.float32)
    return out


def amount_to_residual(amount: float) -> float:
    """Fraction of the removed noise (amplitude) that is mixed back in.

    Quadratic so the slider behaves perceptually: 0 -> original noise,
    50 -> 25 % of the noise amplitude, 100 -> full network output.
    """
    a = float(np.clip(amount, 0.0, 100.0)) / 100.0
    return (1.0 - a) ** 2


def blend_amount(noisy: np.ndarray, den: np.ndarray, wb: np.ndarray,
                 amount: float, color_amount: float) -> np.ndarray:
    """Mix part of the removed noise back in (the "Amount" slider).

    amount / color_amount are 0..100.  The residual is split into luminance
    and chrominance in white-balanced camera space, so grain can be brought
    back without bringing back colour blotches.
    """
    rl = amount_to_residual(amount)
    rc = amount_to_residual(color_amount)
    if rl <= 0 and rc <= 0:
        return den
    wbf = wb.astype(np.float32)
    res = noisy - den
    res *= wbf  # residual, white balanced
    luma = res @ np.array([0.25, 0.5, 0.25], dtype=np.float32)
    res *= np.float32(rc)
    luma *= np.float32(rl - rc)
    res += luma[..., None]
    del luma
    res /= wbf
    res += den
    return res


def run_network(img: RawImage, models, mode: str = "auto", tile: int = 512,
                progress: ProgressFn = None, cancel=None, log=print) -> tuple[np.ndarray, str]:
    """Full-strength network output (camera RGB) with highlight protection.

    models: object with .bayer / .linear runners (see models.py), each a
            callable on (1, C, h, w) float32 arrays with a .fixed_tile attribute.
    mode  : "auto" | "bayer (joint demosaic)" | "linear (after demosaic)"
    """
    use_bayer = img.sensor == "bayer" and img.cfa is not None and models.bayer is not None
    if mode.startswith("linear"):
        use_bayer = False
    elif mode.startswith("bayer") and not use_bayer:
        log(f"[RAW Denoise] {img.sensor} sensor: the Bayer model can't be used here, "
            "using the linear model instead")
    t0 = time.time()
    if use_bayer:
        runner = models.bayer
        den = denoise_bayer(img, runner, tile=tile, fixed_tile=runner.fixed_tile,
                            progress=progress, cancel=cancel)
        how = "RawNIND Bayer (joint demosaic + denoise)"
    else:
        runner = models.linear
        if runner is None:
            raise RuntimeError("the selected model file has no linear (non-Bayer) variant")
        den = denoise_linear(img, runner, tile=tile, fixed_tile=runner.fixed_tile,
                             progress=progress, cancel=cancel)
        how = "RawNIND linear Rec.2020 (after LibRaw demosaic)"
    log(f"[RAW Denoise] {how} on {runner.device_name}: {time.time() - t0:.1f}s")
    den = protect_highlights(img.cam_rgb, den, img.clip_level)
    return den, how


def apply_amount(img: RawImage, den: np.ndarray, how: str, amount: float,
                 color_amount: float) -> RawImage:
    out = blend_amount(img.cam_rgb, den, img.wb, amount, color_amount)
    out = np.minimum(out, img.clip_level)
    return img.with_rgb(out, f"denoised: {how}, amount {amount:g}, colour {color_amount:g}")


def denoise_raw(img: RawImage, models, mode: str = "auto", amount: float = 50.0,
                color_amount: float = 100.0, tile: int = 512, progress: ProgressFn = None,
                cancel=None, log=print) -> RawImage:
    den, how = run_network(img, models, mode, tile, progress, cancel, log)
    return apply_amount(img, den, how, amount, color_amount)
