"""16-bit TIFF writer with an embedded sRGB ICC profile."""

from __future__ import annotations

import os

import numpy as np

_SRGB_ICC = None


def srgb_icc() -> bytes | None:
    global _SRGB_ICC
    if _SRGB_ICC is None:
        try:
            from PIL import ImageCms
            _SRGB_ICC = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        except Exception:
            _SRGB_ICC = b""
    return _SRGB_ICC or None


def write_tiff16(rgb: np.ndarray, path: str, software: str = "ComfyUI RAW AI Denoise",
                 compress: bool = True) -> str:
    """rgb: (H, W, 3) float in [0, 1], sRGB-encoded."""
    import tifffile

    data = np.clip(np.rint(np.asarray(rgb, dtype=np.float32) * 65535.0), 0, 65535).astype(np.uint16)
    extratags = []
    icc = srgb_icc()
    if icc:
        extratags.append((34675, "B", len(icc), icc, True))  # InterColorProfile
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tifffile.imwrite(path, data, photometric="rgb", software=software, metadata=None,
                     description=None, extratags=extratags,
                     compression="adobe_deflate" if compress else None,
                     predictor=2 if compress else None,
                     resolution=(300, 300))
    return path
