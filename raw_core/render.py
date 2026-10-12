"""Simple, predictable raw rendering for previews and TIFF export."""

from __future__ import annotations

import numpy as np

from .color import apply_matrix, contrast_curve, highlight_rolloff, rgb_from_cam_matrix, srgb_oetf
from .rawio import RawImage, apply_orientation


def develop(img: RawImage, exposure_ev: float = 0.0, auto_exposure: bool = True,
            white_balance: str = "as shot", tone: str = "contrast",
            orient: bool = True) -> np.ndarray:
    """RawImage -> display-referred sRGB float32 (H, W, 3) in [0, 1]."""
    wb = img.wb
    if white_balance == "daylight" and img.daylight_wb is not None:
        wb = img.daylight_wb
    gain = (img.preview_gain if auto_exposure else 1.0) * (2.0 ** exposure_ev)
    m = rgb_from_cam_matrix(img.cam_xyz, "srgb")
    bal = img.cam_rgb * wb.astype(np.float32)
    # clipped highlights -> neutral white (LibRaw "clip" highlight mode)
    np.minimum(bal, np.float32(img.clip_level * float(np.min(wb))), out=bal)
    lin = apply_matrix(bal, m * gain)
    del bal
    if tone == "linear (no curve)":
        lin = np.clip(lin, 0, 1)
        out = srgb_oetf(lin)
    else:
        lin = highlight_rolloff(lin, 0.85)
        out = srgb_oetf(lin)
        if tone == "contrast":
            out = contrast_curve(out)
    if orient:
        out = apply_orientation(out, img.flip)
    return np.ascontiguousarray(out, dtype=np.float32)
