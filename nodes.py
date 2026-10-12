"""ComfyUI nodes: RAW loading, RawNIND AI denoise, linear DNG / 16-bit TIFF export."""

from __future__ import annotations

import hashlib
import logging
import os
import time

import numpy as np

import folder_paths

from .raw_core import denoise as rd
from .raw_core.dng import camera_name, write_linear_dng
from .raw_core.models import MODEL_FILENAME, download_model, load_models
from .raw_core.rawio import DEMOSAIC_ALGOS, RAW_EXTENSIONS, RawImage, list_raw_files, load_raw
from .raw_core.render import develop
from .raw_core.tiff16 import write_tiff16

log = logging.getLogger("RAW Denoise")
logging.getLogger("tifffile").setLevel(logging.ERROR)

CATEGORY = "image/RAW Denoise"
MODEL_FOLDER = "raw_denoise"
MODEL_EXTS = {".dtmodel", ".onnx", ".zip"}

# register models/raw_denoise
_default_model_dir = os.path.join(folder_paths.models_dir, MODEL_FOLDER)
if MODEL_FOLDER in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(MODEL_FOLDER, _default_model_dir)
else:
    folder_paths.folder_names_and_paths[MODEL_FOLDER] = ([_default_model_dir], MODEL_EXTS)


def _info(msg: str):
    print(msg)


def _to_image_tensor(rgb: np.ndarray):
    import torch
    return torch.from_numpy(np.ascontiguousarray(rgb, dtype=np.float32))[None, ...]


def _from_image_tensor(t, idx: int = 0) -> np.ndarray:
    return t[idx].detach().cpu().float().numpy()


def _model_choices() -> list[str]:
    try:
        files = folder_paths.get_filename_list(MODEL_FOLDER)
    except Exception:
        files = []
    files = [f for f in files if os.path.splitext(f)[1].lower() in MODEL_EXTS]
    if MODEL_FILENAME not in files:
        files = [MODEL_FILENAME] + files  # downloaded automatically on first use
    return files


def _resolve_model(name: str) -> str:
    path = folder_paths.get_full_path(MODEL_FOLDER, name)
    if path and os.path.exists(path):
        return path
    if os.path.basename(name) == MODEL_FILENAME:
        return download_model(folder_paths.get_folder_paths(MODEL_FOLDER)[0], log=_info)
    raise FileNotFoundError(f"model '{name}' not found in models/{MODEL_FOLDER}")


NO_FILE = "(click 'Choose RAW file(s)…' below)"


def _raw_choices() -> list[str]:
    files = list_raw_files(folder_paths.get_input_directory())
    return files or [NO_FILE]


def _resolve_raw_path(raw_file: str, path_override: str, folder_index: int) -> str:
    p = (path_override or "").strip().strip('"').strip("'")
    if p:
        p = os.path.expanduser(p)
        if os.path.isdir(p):
            files = list_raw_files(p, skip_outputs=True)
            if not files:
                raise FileNotFoundError(f"no RAW files in folder {p}")
            return os.path.join(p, files[folder_index % len(files)])
        if os.path.isfile(p):
            return p
        raise FileNotFoundError(p)
    if not raw_file or raw_file.startswith("("):
        raise FileNotFoundError(
            "No RAW file selected. Click 'Choose RAW file(s)…' on the Load RAW Image node "
            "(or drag the file onto it), then press Run.")
    path = os.path.join(folder_paths.get_input_directory(), raw_file)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"'{raw_file}' is not in ComfyUI's input folder any more. Pick it again with "
            "'Choose RAW file(s)…'.")
    return path


def _describe(img: RawImage) -> str:
    h, w, _ = img.cam_rgb.shape
    iso = f"ISO {img.iso:g}" if img.iso else "ISO ?"
    sensor = {"bayer": "Bayer", "xtrans": "X-Trans"}.get(img.sensor, img.sensor)
    return (f"{os.path.basename(img.source_path)} | {camera_name(img.make, img.model)} | {iso} | "
            f"{sensor} sensor | {w}x{h} | " + " -> ".join(img.history))


# ----------------------------------------------------------------------------


class LoadRAW:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "raw_file": (_raw_choices(), {"tooltip": "The RAW file to process (ARW, CR2, CR3, NEF, RAF, ORF, RW2, DNG, ...). Use the 'Choose RAW file(s)…' button below or drag files onto the node; they are copied into ComfyUI/input/raw."}),
                "demosaic": (DEMOSAIC_ALGOS, {"default": "AHD", "tooltip": "LibRaw demosaic used for the 'before' image and for sensors the Bayer model can't take (Fuji X-Trans). Bayer files denoised in 'auto' mode are demosaiced by the network itself."}),
            },
            "optional": {
                "path_override": ("STRING", {"default": "", "tooltip": "Optional absolute path to a RAW file, or to a folder of RAW files (then folder_index picks the file)."}),
                "folder_index": ("INT", {"default": 0, "min": 0, "max": 100000, "control_after_generate": True, "tooltip": "When path_override is a folder: which file to load. Set the control to 'increment' and queue N runs to batch-process a folder."}),
            },
        }

    RETURN_TYPES = ("RAW_IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("raw", "preview", "info")
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "Loads a camera RAW file (via LibRaw) as scene-linear camera data, plus a quick rendered preview of the original."

    @classmethod
    def IS_CHANGED(cls, raw_file, demosaic, path_override="", folder_index=0):
        try:
            p = _resolve_raw_path(raw_file, path_override, folder_index)
            st = os.stat(p)
            return f"{p}|{st.st_mtime_ns}|{st.st_size}|{demosaic}"
        except Exception:
            return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, raw_file, demosaic, path_override="", folder_index=0):
        return True  # file list may be stale; existence is checked at run time

    def load(self, raw_file, demosaic, path_override="", folder_index=0):
        path = _resolve_raw_path(raw_file, path_override, folder_index)
        t0 = time.time()
        img = load_raw(path, demosaic=demosaic)
        _info(f"[RAW Denoise] loaded {os.path.basename(path)} in {time.time() - t0:.1f}s")
        preview = develop(img)
        return (img, _to_image_tensor(preview), _describe(img))


class _Cache:
    key = None
    den = None
    how = None


class RawNINDDenoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "raw": ("RAW_IMAGE",),
                "amount": ("INT", {"default": 50, "min": 0, "max": 100, "step": 1, "display": "slider", "tooltip": "Like Lightroom's Denoise 'Amount': 0 keeps the original noise, 100 is the full network output. Re-running with a different Amount reuses the cached network result."}),
                "color_noise": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1, "display": "slider", "tooltip": "How much colour (chroma) noise to remove. Keeping this at 100 while lowering Amount brings back fine grain without colour blotches."}),
                "model": (_model_choices(), {"tooltip": "RawNIND model package in models/raw_denoise. rawdenoise-nind.dtmodel (58 MB, GPL-3.0, from the darktable project) is downloaded automatically on first use."}),
                "mode": (["auto", "bayer (joint demosaic)", "linear (after demosaic)"], {"default": "auto", "tooltip": "auto: Bayer sensors are denoised on the raw mosaic (denoise + demosaic in one pass, like Lightroom); X-Trans and other sensors are demosaiced first and denoised in linear Rec.2020."}),
                "backend": (["auto", "torch", "torch (cpu)", "onnxruntime"], {"default": "auto", "tooltip": "auto/torch run on ComfyUI's GPU (CUDA, ROCm, Apple MPS...). onnxruntime needs the onnxruntime / onnxruntime-gpu package."}),
                "tile_size": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64, "tooltip": "Network tile size (packed pixels for Bayer = 2x on the sensor). Larger is faster on big GPUs; lower it if you run out of VRAM. Ignored by onnxruntime (fixed 512)."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("RAW_IMAGE",)
    RETURN_NAMES = ("raw",)
    FUNCTION = "denoise"
    CATEGORY = CATEGORY
    DESCRIPTION = ("AI raw noise reduction with the RawNIND network (Brummer & De Vleeschouwer 2025): "
                   "joint denoise + demosaic on the Bayer mosaic, in the spirit of Lightroom's Denoise.")

    def denoise(self, raw: RawImage, amount, color_noise, model, mode, backend, tile_size, unique_id=None):
        import comfy.model_management as mm
        import comfy.utils

        device = mm.get_torch_device() if backend in ("auto", "torch") else None
        fingerprint = hashlib.sha1(np.ascontiguousarray(raw.cam_rgb[::61, ::53]).tobytes()).hexdigest()
        key = (id(raw), fingerprint, raw.source_path, raw.cam_rgb.shape, model, mode, backend,
               tile_size, str(device), tuple(raw.history))
        if _Cache.key == key and _Cache.den is not None:
            den, how = _Cache.den, _Cache.how
            _info("[RAW Denoise] re-using cached network output (only Amount / colour changed)")
        else:
            _Cache.key = _Cache.den = None
            path = _resolve_model(model)
            models = load_models(path, backend=backend, device=device, log=_info)
            try:
                pbar = comfy.utils.ProgressBar(100, node_id=unique_id)
            except TypeError:  # older ComfyUI
                pbar = comfy.utils.ProgressBar(100)

            def progress(done, total):
                pbar.update_absolute(int(100 * done / max(total, 1)), 100)

            try:
                den, how = rd.run_network(raw, models, mode=mode, tile=tile_size, progress=progress,
                                          cancel=mm.throw_exception_if_processing_interrupted, log=_info)
            finally:
                try:
                    mm.soft_empty_cache()  # give cached VRAM back to the rest of the workflow
                except Exception:
                    pass
            _Cache.key, _Cache.den, _Cache.how = key, den, how
        out = rd.apply_amount(raw, den, how, amount, color_noise)
        return (out,)


class DevelopRAW:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "raw": ("RAW_IMAGE",),
                "exposure": (["auto (same as original)", "manual"], {"default": "auto (same as original)", "tooltip": "auto: brightness is set from the ORIGINAL file, so before/after previews always match."}),
                "exposure_ev": ("FLOAT", {"default": 0.0, "min": -6.0, "max": 6.0, "step": 0.1}),
                "white_balance": (["as shot", "daylight"], {"default": "as shot"}),
                "tone": (["contrast", "neutral", "linear (no curve)"], {"default": "contrast"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Renders RAW data to an sRGB IMAGE (white balance, colour matrix, exposure, simple tone curve) for previews, comparisons or further ComfyUI processing."

    def run(self, raw, exposure, exposure_ev, white_balance, tone):
        rgb = develop(raw, exposure_ev=exposure_ev, auto_exposure=exposure.startswith("auto"),
                      white_balance=white_balance, tone=tone)
        return (_to_image_tensor(rgb),)


def _unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 1
    while os.path.exists(f"{base}_{i}{ext}"):
        i += 1
    return f"{base}_{i}{ext}"


class SaveLinearDNG:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "raw": ("RAW_IMAGE",),
                "filename_suffix": ("STRING", {"default": "-Enhanced-NR", "tooltip": "Appended to the original file name (Lightroom uses -Enhanced-NR)."}),
                "output_folder": ("STRING", {"default": "raw_denoise", "tooltip": "Folder inside ComfyUI/output, or an absolute path. Leave empty to save next to the original RAW file."}),
                "overwrite": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("dng_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = ("Saves a linear DNG (demosaiced camera RGB + colour matrix + as-shot white balance), like Lightroom's "
                   "Enhanced-NR DNG: open it in Lightroom/ACR/darktable/RawTherapee and edit it like a raw file.")

    def save(self, raw, filename_suffix, output_folder, overwrite):
        stem = os.path.splitext(os.path.basename(raw.source_path))[0]
        folder = (output_folder or "").strip()
        if not folder:
            folder = os.path.dirname(raw.source_path)
        elif not os.path.isabs(folder):
            out_root = folder_paths.get_output_directory()
            folder = os.path.abspath(os.path.join(out_root, folder))
            if os.path.commonpath([folder, os.path.abspath(out_root)]) != os.path.abspath(out_root):
                raise ValueError("output_folder must stay inside ComfyUI/output (or be an absolute path)")
        path = os.path.join(folder, f"{stem}{filename_suffix}.dng")
        if not overwrite:
            path = _unique_path(path)
        if os.path.abspath(path) == os.path.abspath(raw.source_path):
            raise ValueError("refusing to overwrite the original RAW file")
        t0 = time.time()
        write_linear_dng(raw, path)
        _info(f"[RAW Denoise] wrote {path} ({os.path.getsize(path) / 1e6:.0f} MB, {time.time() - t0:.1f}s)")
        return {"ui": {"text": [path]}, "result": (path,)}


class SaveImage16:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "raw_denoise/ComfyUI"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = "Saves IMAGEs as 16-bit sRGB TIFF (deflate-compressed, sRGB ICC profile embedded). ComfyUI keeps images in float, so nothing is lost to 8 bits."

    def save(self, images, filename_prefix):
        out_dir = folder_paths.get_output_directory()
        h, w = int(images.shape[1]), int(images.shape[2])
        full_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, out_dir, w, h)
        paths = []
        for i in range(images.shape[0]):
            name = filename.replace("%batch_num%", str(i))
            p = os.path.join(full_folder, f"{name}_{counter:05}_.tif")
            write_tiff16(_from_image_tensor(images, i), p)
            paths.append(p)
            counter += 1
        _info("[RAW Denoise] wrote " + ", ".join(paths))
        return {"ui": {"text": paths}}


class CompareCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "before": ("IMAGE",),
                "after": ("IMAGE",),
                "center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "center_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "crop_size": ("INT", {"default": 640, "min": 64, "max": 4096, "step": 32}),
                "zoom": (["100%", "200%", "300%", "400%"], {"default": "100%", "tooltip": "Pixel zoom (nearest neighbour)."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("comparison",)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Side-by-side 100 % crops of two images (ComfyUI previews are downscaled, which hides noise)."

    def run(self, before, after, center_x, center_y, crop_size, zoom):
        a = _from_image_tensor(before)
        b = _from_image_tensor(after)
        if a.shape != b.shape:
            raise ValueError(f"before {a.shape} and after {b.shape} differ in size")
        h, w, _ = a.shape
        cs = int(min(crop_size, h, w))
        y = int(np.clip(round(center_y * h - cs / 2), 0, h - cs))
        x = int(np.clip(round(center_x * w - cs / 2), 0, w - cs))
        tiles = []
        for label, img in (("BEFORE", a), ("AFTER", b)):
            c = img[y:y + cs, x:x + cs]
            z = int(str(zoom).rstrip("%")) // 100
            if z > 1:
                c = c.repeat(z, axis=0).repeat(z, axis=1)
            tiles.append(_label(c, label))
        gap = np.ones((tiles[0].shape[0], 8, 3), dtype=np.float32)
        return (_to_image_tensor(np.concatenate([tiles[0], gap, tiles[1]], axis=1)),)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw
        im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 8 + 7 * len(text), 18], fill=(0, 0, 0))
        d.text((5, 3), text, fill=(255, 255, 255))
        return np.asarray(im).astype(np.float32) / 255.0
    except Exception:
        return img


NODE_CLASS_MAPPINGS = {
    "RawDenoise_LoadRAW": LoadRAW,
    "RawDenoise_RawNIND": RawNINDDenoise,
    "RawDenoise_Develop": DevelopRAW,
    "RawDenoise_SaveDNG": SaveLinearDNG,
    "RawDenoise_SaveTIFF16": SaveImage16,
    "RawDenoise_CompareCrop": CompareCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RawDenoise_LoadRAW": "Load RAW Image",
    "RawDenoise_RawNIND": "RAW AI Denoise (RawNIND)",
    "RawDenoise_Develop": "Develop RAW to Image",
    "RawDenoise_SaveDNG": "Save Linear DNG",
    "RawDenoise_SaveTIFF16": "Save Image (16-bit TIFF)",
    "RawDenoise_CompareCrop": "Before/After 100% Crop",
}
