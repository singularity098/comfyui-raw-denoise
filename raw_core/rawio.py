"""RAW file loading via rawpy (LibRaw).

Everything is kept in *sensor orientation* and in *camera-native RGB without
white balance*, normalised so that 0 = black level and 1 = sensor white level.
That is the same space a linear DNG stores, so nothing is lost between the
denoiser and the DNG writer.  Orientation and white balance are applied only
when rendering a preview or a TIFF.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

RAW_EXTENSIONS = {
    ".3fr", ".ari", ".arw", ".bay", ".cr2", ".cr3", ".crw", ".dcr", ".dng", ".erf",
    ".fff", ".iiq", ".k25", ".kdc", ".mdc", ".mef", ".mos", ".mrw", ".nef", ".nrw",
    ".orf", ".pef", ".raf", ".raw", ".rw2", ".rwl", ".sr2", ".srf", ".srw", ".x3f",
}

DEMOSAIC_ALGOS = ["AHD", "DHT", "AAHD", "VNG", "PPG", "LINEAR", "DCB"]


def _rawpy():
    try:
        import rawpy  # noqa: F401
    except ImportError as e:  # pragma: no cover
        import sys
        raise ImportError(
            "The 'rawpy' package (LibRaw) is not installed in the Python that runs ComfyUI, "
            "so RAW files can't be read. Close ComfyUI, run this command, then start ComfyUI again:\n"
            f'    "{sys.executable}" -m pip install rawpy tifffile'
        ) from e
    import rawpy
    return rawpy


def _open_raw(rawpy, path: str):
    """rawpy.imread with messages that say what to do."""
    try:
        return rawpy.imread(path)
    except rawpy.LibRawFileUnsupportedError as e:
        lr = ".".join(str(v) for v in getattr(rawpy, "libraw_version", ()))
        raise ValueError(
            f"LibRaw {lr} (inside rawpy {getattr(rawpy, '__version__', '?')}) can't decode "
            f"{os.path.basename(path)}. If your camera is newer than this LibRaw, update it with "
            f"'pip install -U rawpy' in ComfyUI's Python. Sony note: 'lossless compressed' ARW needs "
            f"LibRaw 0.21+; JPEG/HEIF files are not raw.") from e
    except (rawpy.LibRawIOError, rawpy.LibRawDataError) as e:
        raise ValueError(f"{os.path.basename(path)} could not be read - the file looks damaged or "
                         f"incomplete (copy it from the card again).") from e


@dataclass
class RawImage:
    """Scene-linear camera RGB plus everything needed to write a DNG."""

    cam_rgb: np.ndarray                 # (H, W, 3) float32, camera native RGB, no WB, 1.0 = white level
    wb: np.ndarray                      # (3,) as-shot WB multipliers, G = 1
    cam_xyz: np.ndarray                 # (3, 3) XYZ(D65) -> camera (Adobe ColorMatrix, D65)
    flip: int                           # LibRaw orientation code (0, 3, 5, 6)
    make: str
    model: str
    source_path: str
    sensor: str                         # "bayer", "xtrans" or "other"
    cfa: Optional[np.ndarray] = None    # (H, W) float32 normalised mosaic, sensor orientation
    cfa_pattern: Optional[np.ndarray] = None   # 2x2 / 6x6 colour indices (0=R 1=G 2=B 3=G2)
    color_desc: str = "RGBG"
    preview_gain: float = 1.0           # linear gain for "auto" preview exposure
    daylight_wb: Optional[np.ndarray] = None
    history: list = field(default_factory=list)
    iso: Optional[float] = None
    clip_level: float = 1.0             # normalised value at which the sensor saturates
    exif: dict = field(default_factory=dict)   # shutter, aperture, focal length, timestamp

    def with_rgb(self, cam_rgb: np.ndarray, note: str) -> "RawImage":
        return replace(self, cam_rgb=cam_rgb, cfa=None, cfa_pattern=None,
                       history=self.history + [note])

    @property
    def shape(self):
        return self.cam_rgb.shape


def _normalised_cfa(raw) -> tuple[np.ndarray, np.ndarray]:
    """Black-subtracted, white-normalised visible mosaic (float32) + pattern."""
    cfa = raw.raw_image_visible.astype(np.float32)
    colors = raw.raw_colors_visible
    black = np.asarray(raw.black_level_per_channel, dtype=np.float32)
    white = float(raw.white_level)
    pat = np.asarray(raw.raw_pattern)
    ph, pw = pat.shape
    out = np.empty_like(cfa)
    for i in range(ph):
        for j in range(pw):
            ch = int(colors[i, j])
            bl = black[ch]
            out[i::ph, j::pw] = (cfa[i::ph, j::pw] - bl) / max(white - bl, 1.0)
    # pattern as seen from the visible origin
    return out, np.asarray(colors[:ph, :pw]).copy()


def _dng_color_matrix(path: str) -> Optional[np.ndarray]:
    """XYZ->camera matrix straight from a DNG's ColorMatrix tags (daylight one
    preferred). LibRaw only exposes its built-in per-camera table, which is
    empty for cameras it doesn't know (phones, many native-DNG cameras)."""
    if os.path.splitext(path)[1].lower() != ".dng":
        return None
    try:
        import tifffile
        with tifffile.TiffFile(path) as tf:
            p0 = tf.pages[0]
            mats = []
            for mtag, itag in ((50721, 50778), (50722, 50779)):
                t = p0.tags.get(mtag)
                if t is None:
                    continue
                v = np.asarray(t.value, dtype=np.float64).ravel()
                if v.size == 18:
                    v = v[0::2] / np.where(v[1::2] == 0, 1, v[1::2])
                if v.size < 9:
                    continue
                ill = p0.tags.get(itag)
                mats.append((int(ill.value) if ill is not None else 0, v[:9].reshape(3, 3)))
        if not mats:
            return None
        daylight = {21: 0, 20: 1, 23: 2, 1: 3, 9: 4, 4: 5, 10: 6, 22: 7}
        mats.sort(key=lambda m: daylight.get(m[0], 99))
        return mats[0][1]
    except Exception:
        return None


def _clip_level(cfa: np.ndarray) -> float:
    """Sensor saturation point (normalised). Some cameras clip a little below
    the nominal white level; a pile-up of photosites at the maximum reveals it."""
    step = 1 if cfa.size < 30e6 else 2
    sub = cfa[::step, ::step]
    mx = float(sub.max())
    if mx <= 0.5:
        return 1.0
    near = np.count_nonzero(sub >= mx - 0.002 * max(mx, 1e-6))
    if near >= max(16, sub.size * 1e-6):
        return float(min(mx, 1.0)) if mx > 0.5 else 1.0
    return 1.0


def _wb3(mul4) -> Optional[np.ndarray]:
    m = np.asarray(mul4, dtype=np.float64)[:4]
    if m[1] <= 0:
        return None
    g = m[1]
    out = np.array([m[0], m[1], m[2]], dtype=np.float64) / g
    if not np.all(np.isfinite(out)) or np.any(out <= 0):
        return None
    return out


def demosaic_libraw(raw, algo: str = "AHD") -> np.ndarray:
    """LibRaw demosaic -> (H, W, 3) camera RGB, no WB, 1.0 = white level.

    LibRaw is told to apply the as-shot WB *for the demosaic only* (most
    algorithms work best on balanced data) with highlight mode "ignore" so no
    channel is clipped; we then divide the multipliers back out.
    """
    rawpy = _rawpy()
    algo_enum = getattr(rawpy.DemosaicAlgorithm, algo, rawpy.DemosaicAlgorithm.AHD)
    try:
        if not algo_enum.isSupported:
            algo_enum = rawpy.DemosaicAlgorithm.AHD
    except Exception:
        pass
    mul = np.asarray(raw.camera_whitebalance, dtype=np.float64)[:4]
    if mul[1] <= 0 or not np.all(np.isfinite(mul)):
        mul = np.asarray(raw.daylight_whitebalance, dtype=np.float64)[:4]
    if mul[3] <= 0:
        mul[3] = mul[1]
    user_wb = [float(v) for v in mul]
    rgb = raw.postprocess(
        demosaic_algorithm=algo_enum,
        output_color=rawpy.ColorSpace.raw,
        output_bps=16,
        gamma=(1.0, 1.0),
        no_auto_bright=True,
        user_wb=user_wb,
        use_camera_wb=False,
        use_auto_wb=False,
        highlight_mode=rawpy.HighlightMode.Ignore,
        user_flip=0,
        adjust_maximum_thr=0.0,
        bright=1.0,
        fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode.Off,
        median_filter_passes=0,
    )
    # LibRaw scales channel c by (mul[c] / max(mul)) * 65535 / (white - black)
    norm = np.array([mul[0], mul[1], mul[2]]) / mul.max()
    out = rgb.astype(np.float32)
    del rgb
    out *= (1.0 / (65535.0 * norm)).astype(np.float32)
    return out


def load_raw(path: str, demosaic: str = "AHD", keep_cfa: bool = True) -> RawImage:
    rawpy = _rawpy()
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with _open_raw(rawpy, path) as raw:
        pat = raw.raw_pattern
        flat = raw.raw_type == rawpy.RawType.Flat and pat is not None
        pat = np.asarray(pat) if pat is not None else None
        if not flat:
            sensor = "other"  # e.g. linear (already demosaiced) DNGs
        elif pat.shape == (2, 2):
            sensor = "bayer"
        elif pat.shape == (6, 6):
            sensor = "xtrans"
        else:
            sensor = "other"
        color_desc = raw.color_desc.decode("ascii", "replace")
        if sensor == "bayer" and sorted(set(color_desc[i] for i in np.unique(pat))) != ["B", "G", "R"]:
            sensor = "other"  # e.g. CMYG sensors

        cam_xyz = _dng_color_matrix(path)
        if cam_xyz is None:
            cam_xyz = np.asarray(raw.rgb_xyz_matrix, dtype=np.float64)[:3, :3]
        if not np.any(cam_xyz):
            # last resort: undo LibRaw's camera->sRGB matrix (row scale is lost,
            # which only matters for the DNG's white-balance metadata)
            rgb_cam = np.asarray(raw.color_matrix, dtype=np.float64)[:3, :3]
            if not np.any(rgb_cam):
                raise ValueError("this RAW file has no colour matrix (camera not supported by LibRaw)")
            from .color import SRGB_TO_XYZ
            cam_xyz = np.linalg.inv(rgb_cam) @ np.linalg.inv(SRGB_TO_XYZ)
        wb = _wb3(raw.camera_whitebalance)
        dwb = _wb3(raw.daylight_whitebalance)
        if wb is None:
            wb = dwb if dwb is not None else np.ones(3)

        cam_rgb = demosaic_libraw(raw, demosaic)
        cfa, cfa_pat = (None, None)
        if flat:
            full_cfa, full_pat = _normalised_cfa(raw)
            clip_level = _clip_level(full_cfa)
        else:
            full_cfa = full_pat = None
            clip_level = _clip_level(cam_rgb.max(axis=2))
        if keep_cfa and sensor == "bayer":
            cfa, cfa_pat = full_cfa, full_pat
            if cfa.shape != cam_rgb.shape[:2]:
                # LibRaw output size differs from the visible mosaic (rare:
                # pixel-aspect or fuji_width cameras) -> no CFA path
                cfa, cfa_pat, sensor = None, None, "other"

        flip = int(raw.sizes.flip)
        iso = None
        exif = {}
        try:  # rawpy >= 0.22 exposes LibRaw's imgother
            o = raw.other
            iso = float(o.iso_speed) or None
            exif = {"exposure_time": float(o.shutter_speed), "f_number": float(o.aperture),
                    "focal_length": float(o.focal_length), "timestamp": o.timestamp,
                    "artist": o.artist}
        except Exception:
            pass

    make, model, exif_iso = _camera_info(path)
    img = RawImage(cam_rgb=cam_rgb, wb=wb, cam_xyz=cam_xyz, flip=flip, make=make, model=model,
                   source_path=path, sensor=sensor, cfa=cfa, cfa_pattern=cfa_pat,
                   color_desc=color_desc, daylight_wb=dwb, history=[f"loaded ({demosaic} demosaic)"],
                   clip_level=clip_level, exif=exif)
    img.preview_gain = auto_preview_gain(img)
    img.iso = iso or exif_iso
    return img


def auto_preview_gain(img: RawImage) -> float:
    """LibRaw-style auto-bright: put the 99th percentile of the balanced image at 1.0."""
    h, w, _ = img.cam_rgb.shape
    step = max(1, int(np.sqrt(h * w / 250_000)))
    sub = img.cam_rgb[::step, ::step] * img.wb.astype(np.float32)
    # median-filter-ish robustness: use the per-pixel max channel of a 2x2 mean
    sh, sw = sub.shape[0] // 2 * 2, sub.shape[1] // 2 * 2
    sub = sub[:sh, :sw].reshape(sh // 2, 2, sw // 2, 2, 3).mean(axis=(1, 3))
    p = float(np.percentile(sub.max(axis=2), 99.0))
    if not np.isfinite(p) or p <= 1e-6:
        return 1.0
    return float(np.clip(1.0 / p, 1.0, 64.0))


def _clean(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bytes):
        v = v.decode("utf-8", "replace")
    return str(v).replace("\x00", " ").strip()


def _tiff_ifd0(buf: bytes) -> dict:
    """Read Make(271)/Model(272) and the EXIF ISO from a TIFF blob."""
    import io
    out = {}
    try:
        import tifffile
        with tifffile.TiffFile(io.BytesIO(buf)) as tf:
            p0 = tf.pages[0]
            for code in (271, 272):
                t = p0.tags.get(code)
                if t is not None:
                    out[code] = _clean(t.value)
            if p0.tags.get(34855) is not None:  # TIFF/EP ISO in IFD0 (our DNGs)
                v = p0.tags[34855].value
                out["iso"] = v[0] if isinstance(v, (tuple, list)) else v
            ex = p0.tags.get(34665)
            if ex is not None and isinstance(ex.value, dict):
                iso = ex.value.get("ISOSpeedRatings") or ex.value.get("PhotographicSensitivity")
                if iso:
                    out["iso"] = iso[0] if isinstance(iso, (tuple, list)) else iso
    except Exception:
        pass
    return out


def _camera_info(path: str):
    """Best-effort Make / Model / ISO without extra dependencies.

    Handles TIFF-based raws (CR2, NEF, ARW, DNG, ORF, PEF, RW2...), Canon CR3
    (TIFF blob inside the CMT1/CMT2 boxes) and Fujifilm RAF (fixed header).
    """
    make = model = ""
    iso = None
    try:
        with open(path, "rb") as f:
            head = f.read(1 << 20)
        if head.startswith(b"FUJIFILMCCD-RAW"):
            make = "FUJIFILM"
            model = _clean(head[0x1C:0x1C + 32])
        elif head[4:8] == b"ftyp" and b"crx " in head[8:16]:
            for box in (b"CMT1", b"CMT2"):
                i = head.find(box)
                if i >= 4:
                    size = int.from_bytes(head[i - 4:i], "big")
                    tags = _tiff_ifd0(head[i + 4:i - 4 + size])
                    make = make or tags.get(271, "")
                    model = model or tags.get(272, "")
                    iso = iso or tags.get("iso")
        else:
            with open(path, "rb") as f:
                tags = _tiff_ifd0(f.read())
            make, model, iso = tags.get(271, ""), tags.get(272, ""), tags.get("iso")
    except Exception:
        pass
    # Many cameras repeat the make in the model string ("Canon EOS R6", "NIKON D850")
    try:
        iso = float(iso) if iso else None
    except Exception:
        iso = None
    return make or "Unknown", model or "Camera", iso


def list_raw_files(folder: str, skip_outputs: bool = False) -> list[str]:
    """RAW files below `folder`. skip_outputs drops our own (and Lightroom's)
    denoised DNGs so batch runs over a folder don't process them again."""
    out = []
    if not os.path.isdir(folder):
        return out
    for root, _, files in os.walk(folder):
        for f in files:
            if skip_outputs and "-enhanced-nr" in f.lower():
                continue
            if os.path.splitext(f)[1].lower() in RAW_EXTENSIONS:
                out.append(os.path.relpath(os.path.join(root, f), folder).replace(os.sep, "/"))
    return sorted(out)


def apply_orientation(img: np.ndarray, flip: int) -> np.ndarray:
    """Rotate an (H, W, C) sensor-orientation array to display orientation."""
    if flip == 3:
        return img[::-1, ::-1]
    if flip == 5:  # 90 deg CCW
        return np.rot90(img, 1)
    if flip == 6:  # 90 deg CW
        return np.rot90(img, -1)
    return img
