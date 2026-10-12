"""Colour-space helpers (all matrices act on column vectors, data is ...x3)."""

from __future__ import annotations

import numpy as np

# Linear RGB (D65 white) -> XYZ
SRGB_TO_XYZ = np.array(
    [[0.4124564, 0.3575761, 0.1804375],
     [0.2126729, 0.7151522, 0.0721750],
     [0.0193339, 0.1191920, 0.9503041]], dtype=np.float64)

REC2020_TO_XYZ = np.array(
    [[0.6369580, 0.1446169, 0.1688810],
     [0.2627002, 0.6779981, 0.0593017],
     [0.0000000, 0.0280727, 1.0609851]], dtype=np.float64)

PRIMARIES = {"srgb": SRGB_TO_XYZ, "rec2020": REC2020_TO_XYZ}


def rgb_from_cam_matrix(cam_xyz: np.ndarray, target: str = "srgb") -> np.ndarray:
    """Matrix taking white-balanced camera RGB to linear `target` RGB.

    Same construction LibRaw/dcraw use: cam<-rgb = cam_xyz @ xyz<-rgb, rows
    normalised so RGB white maps to camera neutral (1,1,1), then inverted.
    """
    cam_rgb = cam_xyz[:3, :3].astype(np.float64) @ PRIMARIES[target]
    cam_rgb = cam_rgb / cam_rgb.sum(axis=1, keepdims=True)
    return np.linalg.inv(cam_rgb)


def daylight_wb_from_matrix(cam_xyz: np.ndarray) -> np.ndarray:
    """D65 white balance multipliers implied by the colour matrix (G = 1)."""
    cam_rgb = cam_xyz[:3, :3].astype(np.float64) @ SRGB_TO_XYZ
    mul = 1.0 / cam_rgb.sum(axis=1)
    return mul / mul[1]


def apply_matrix(img: np.ndarray, m: np.ndarray) -> np.ndarray:
    """img (...,3) float32 -> (...,3) float32, out = m @ px."""
    shp = img.shape
    out = img.reshape(-1, 3) @ m.T.astype(np.float32)
    return out.reshape(shp).astype(np.float32, copy=False)


def srgb_oetf(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def contrast_curve(x: np.ndarray) -> np.ndarray:
    """Gentle S-curve on display-encoded values in [0,1] (similar in spirit
    to the default tone curve raw converters apply)."""
    x = np.clip(x, 0.0, 1.0)
    # smoothstep blended with identity -> mild contrast, endpoints fixed
    s = x * x * (3.0 - 2.0 * x)
    return (0.65 * x + 0.35 * s).astype(np.float32)


def highlight_rolloff(x: np.ndarray, knee: float = 0.8) -> np.ndarray:
    """Soft-clip linear values above `knee` towards 1.0 (keeps colour ratios
    approximately by operating on each channel)."""
    x = np.maximum(x, 0.0)
    over = x > knee
    if not np.any(over):
        return x.astype(np.float32)
    k = 1.0 - knee
    y = x.copy()
    t = (x[over] - knee) / k
    y[over] = knee + k * (t / (1.0 + t))
    return y.astype(np.float32)
