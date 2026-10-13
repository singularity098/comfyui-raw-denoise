"""Automatic noise measurement for the video denoiser.

A few short windows spread over the video are denoised with a fixed probe
level; the robust spread of what the network removes, per brightness band,
is the measured noise.  The network sigma per band follows from it through
video_denoise.sigma_for_spread (calibrated on simulated low-light footage with
ground truth); the spread itself is kept for the texture-aware noise map.
"""

from __future__ import annotations

import numpy as np

from . import video_io as V
from .video_color import codec_for
from .video_denoise import (MAX_SIGMA, DenoiseSettings, NoiseProfile, VideoDenoiser,
                            _profile_from_residual, flat_profile, sigma_for_spread)

PROBE_SIGMA = 25.0       # 0-255 scale
MIN_SIGMA = 3.0
BINS = 12


def measure_noise(info: V.VideoInfo, backend, runner, windows: int = 0, log=print, cancel=None) -> NoiseProfile:
    fps = float(info.fps)
    if windows <= 0:
        windows = int(np.clip(info.duration / 3.0, 1, 6))
    span = max(info.duration - 8 / fps, 0.0)
    times = [span * (i + 0.5) / windows for i in range(windows)] if span > 0 else [0.0]
    probe = flat_profile(PROBE_SIGMA)
    edges = np.linspace(0, 1, BINS + 1)
    xs = 0.5 * (edges[:-1] + edges[1:])
    acc, wsum = np.zeros(BINS), np.zeros(BINS)
    codec = codec_for(info)
    for t in times:
        if cancel:
            cancel()
        rd = V.FrameReader(info, start_seconds=t, max_frames=7, prefetch=7, codec=codec)
        frames = list(rd)
        if len(frames) < 3:
            continue
        den = VideoDenoiser(backend.ops, runner, probe,
                            DenoiseSettings(strength=1.0, detect_cuts=False, texture=0.0),
                            info.width, info.height, return_float=True, codec=codec)
        outs = []
        for f in frames:
            outs += den.push(f)
        outs += den.flush()
        mid = len(frames) // 2
        noisy = codec.decode_display(frames[mid])
        sig, cnt = _profile_from_residual(noisy, np.clip(outs[mid], 0.0, 1.0), BINS)
        ok = ~np.isnan(sig)
        acc[ok] += sig[ok] ** 2 * cnt[ok]
        wsum[ok] += cnt[ok]
    good = wsum > 0
    if not good.any():
        log("[Video Denoise] could not measure noise (very dark or blown-out footage?) - using level 20")
        return flat_profile(20.0)
    est = np.sqrt(acc[good] / wsum[good])
    m = np.interp(xs, xs[good], est)
    m = np.convolve(np.pad(m, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")
    ys = np.clip(sigma_for_spread(m * 255) / 255, MIN_SIGMA / 255, MAX_SIGMA)
    return NoiseProfile(xs.astype(np.float32), ys.astype(np.float32), int(wsum.sum()),
                        spread=m.astype(np.float32))
