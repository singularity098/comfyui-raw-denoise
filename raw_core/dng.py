"""Linear DNG writer (the same kind of file Lightroom's Denoise produces).

The image is stored as demosaiced, scene-linear camera RGB (PhotometricInter-
pretation = LinearRaw) together with the camera's colour matrix and the
as-shot white balance, so Lightroom / Camera Raw / darktable / RawTherapee /
Capture One treat it as a raw file: white balance, exposure and profiles stay
fully editable.
"""

from __future__ import annotations

import datetime as _dt
import os
from fractions import Fraction

import numpy as np

from .rawio import RawImage, apply_orientation

# A small black offset keeps negative (below-black) values, so shadows are
# not biased upwards by clipping when some noise is mixed back in.
BLACK_OFFSET = 256
WHITE = 65535


def _srational(values, denom=10000):
    out = []
    for v in values:
        out += [int(round(float(v) * denom)), denom]
    return out


def _rational(v, max_den=100000):
    f = Fraction(float(v)).limit_denominator(max_den)
    return [max(f.numerator, 0), max(f.denominator, 1)]


def _ascii(s: str) -> str:
    return s.encode("ascii", "replace").decode("ascii")


_MAKE_NAMES = {
    "NIKON CORPORATION": "Nikon", "NIKON": "Nikon", "SONY": "Sony", "FUJIFILM": "Fujifilm",
    "OLYMPUS IMAGING CORP.": "Olympus", "OLYMPUS CORPORATION": "Olympus", "OLYMPUS": "Olympus",
    "OM DIGITAL SOLUTIONS": "OM Digital Solutions", "PENTAX": "Pentax", "RICOH IMAGING COMPANY, LTD.": "Ricoh",
    "PANASONIC": "Panasonic", "LEICA CAMERA AG": "Leica", "LEICA": "Leica", "SAMSUNG": "Samsung",
    "HASSELBLAD": "Hasselblad", "SIGMA": "Sigma", "CANON": "Canon", "APPLE": "Apple",
}


def camera_name(make: str, model: str) -> str:
    """UniqueCameraModel in Adobe's "Make Model" style, e.g. "Nikon D850".

    Lightroom uses this name to pick its camera profiles (Adobe Color etc.);
    if it doesn't match, the embedded colour matrix is used instead.
    """
    mk = _MAKE_NAMES.get(make.strip().upper(), make.strip())
    md = model.strip()
    for prefix in (make.strip(), mk, make.strip().split()[0] if make.strip() else ""):
        if prefix and md.upper().startswith(prefix.upper() + " "):
            md = md[len(prefix) + 1:]
            break
    return f"{mk} {md}".strip()


def write_linear_dng(img: RawImage, path: str, software: str = "ComfyUI RAW AI Denoise",
                     compress: bool = False) -> str:
    import tifffile

    cam = apply_orientation(img.cam_rgb, img.flip)  # store upright, Orientation = 1
    scale = WHITE - BLACK_OFFSET
    data = np.clip(np.rint(cam * scale + BLACK_OFFSET), 0, WHITE).astype(np.uint16)
    h, w = data.shape[:2]
    white = int(min(WHITE, round(img.clip_level * scale + BLACK_OFFSET)))

    neutral = 1.0 / np.asarray(img.wb, dtype=np.float64)
    neutral /= neutral.max()

    make, model = _ascii(img.make), _ascii(img.model)
    unique = camera_name(make, model)

    tags = [
        (50706, "B", 4, (1, 4, 0, 0), True),                   # DNGVersion
        (50707, "B", 4, (1, 2, 0, 0), True),                   # DNGBackwardVersion
        (50708, "s", 0, _ascii(unique), True),                 # UniqueCameraModel
        (50721, "2i", 9, _srational(np.asarray(img.cam_xyz).ravel()), True),  # ColorMatrix1
        (50778, "H", 1, 21, True),                             # CalibrationIlluminant1 = D65
        (50728, "2I", 3, sum((_rational(v) for v in neutral), []), True),     # AsShotNeutral
        (50714, "H", 3, (BLACK_OFFSET,) * 3, True),            # BlackLevel (per sample)
        (50717, "H", 3, (white,) * 3, True),                   # WhiteLevel (per sample)
        (50730, "2i", 1, (0, 1), True),                        # BaselineExposure
        (50829, "I", 4, (0, 0, h, w), True),                   # ActiveArea
        (50718, "2I", 2, (1, 1, 1, 1), True),                  # DefaultScale
        (50719, "2I", 2, (0, 1, 0, 1), True),                  # DefaultCropOrigin
        (50720, "2I", 2, (w, 1, h, 1), True),                  # DefaultCropSize
        (50827, "B", 0, os.path.basename(img.source_path).encode("utf-8"), True),  # OriginalRawFileName
        (274, "H", 1, 1, True),                                # Orientation (pixels stored upright)
        (271, "s", 0, make, True),                             # Make
        (272, "s", 0, model, True),                            # Model
    ]
    # TIFF/EP shooting data in IFD0 (read by Lightroom/ACR/exiftool)
    ex = img.exif or {}
    if img.iso:
        tags.append((34855, "H", 1, int(min(img.iso, 65535)), True))      # ISOSpeedRatings
    if ex.get("exposure_time"):
        tags.append((33434, "2I", 1, _rational(ex["exposure_time"], 1000000), True))
    if ex.get("f_number"):
        tags.append((33437, "2I", 1, _rational(ex["f_number"], 100), True))
    if ex.get("focal_length"):
        tags.append((37386, "2I", 1, _rational(ex["focal_length"], 100), True))
    ts = ex.get("timestamp")
    if isinstance(ts, _dt.datetime) and ts.year > 1971:
        s = ts.strftime("%Y:%m:%d %H:%M:%S")
        tags.append((36867, "s", 0, s, True))                  # DateTimeOriginal
        tags.append((306, "s", 0, s, True))                    # DateTime
    if ex.get("artist"):
        tags.append((315, "s", 0, _ascii(ex["artist"]), True))

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tifffile.imwrite(
        path,
        data,
        photometric=34892,          # LinearRaw
        planarconfig="contig",
        subfiletype=0,
        compression="adobe_deflate" if compress else None,
        predictor=2 if compress else None,
        software=_ascii(software),
        metadata=None,
        description=None,
        resolution=(300, 300),
        extratags=tags,
        rowsperstrip=max(1, (1 << 20) // (w * 6)),
    )
    return path
