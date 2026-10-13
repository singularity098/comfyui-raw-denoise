"""ComfyUI nodes: streaming AI video denoising (FastDVDnet)."""

from __future__ import annotations

import os
import time

import numpy as np

import folder_paths

from .raw_core import video_io as V
from .raw_core.video_backend import MODELS, make_backend
from .raw_core.video_denoise import DenoiseSettings, flat_profile
from .raw_core.video_job import RenderJob, preview_clip
from .raw_core.video_profile import measure_noise

CATEGORY = "image/RAW Denoise/video"
MODES = ["preview (short section)", "render full video"]


def _info(msg):
    print(msg)


def _to_image(rgb):
    import torch
    return torch.from_numpy(np.ascontiguousarray(rgb, dtype=np.float32))[None, ...]


def resolve_video_path(p: str) -> str:
    p = (p or "").strip().strip('"').strip("'")
    if not p:
        raise FileNotFoundError("No video chosen. Click 'Browse this computer…' or 'Upload…' on the "
                                "Load Video node, then press Run.")
    p = os.path.expanduser(p)
    if not os.path.isabs(p):
        p = os.path.join(folder_paths.get_input_directory(), p)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"video not found: {p}")
    return p


class LoadVideoForDenoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("STRING", {"default": "", "tooltip": "The video file. Use 'Browse this computer…' (no copying - best for big camera files) or 'Upload…', or paste a full path."}),
            }
        }

    RETURN_TYPES = ("VIDEO_SOURCE", "IMAGE", "STRING")
    RETURN_NAMES = ("video", "frame", "info")
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Picks a video for AI denoising without loading its frames into memory (frames are "
                   "streamed later), so any length works. Outputs one frame for a quick look.")

    @classmethod
    def IS_CHANGED(cls, video):
        try:
            p = resolve_video_path(video)
            st = os.stat(p)
            return f"{p}|{st.st_mtime_ns}|{st.st_size}"
        except Exception:
            return float("nan")

    def load(self, video):
        path = resolve_video_path(video)
        info = V.probe(path)
        frame = V.grab_frame(info, min(info.duration * 0.5, max(info.duration - 0.2, 0)))
        _info(f"[Video Denoise] {info.describe()}")
        if info.interlaced:
            _info(f"[Video Denoise] warning: {os.path.basename(path)} is interlaced ({info.field_order}). It is "
                  "denoised as whole frames, which softens motion; deinterlace it first for best results.")
        return ({"path": path, "info": info}, _to_image(frame), info.describe())


class _ProfileCache:
    data: dict = {}


class VideoAIDenoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO_SOURCE",),
                "mode": (MODES, {"default": MODES[0], "tooltip": "preview: denoise a few seconds and show a before/after clip (fast, for tuning). render: denoise the whole video to a file."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.05, "tooltip": "1.0 = remove the noise level measured in this video. Raise for cleaner/smoother, lower to keep more texture."}),
                "noise_level": ("INT", {"default": 0, "min": 0, "max": 60, "tooltip": "0 = measure automatically, per brightness band (recommended). 1-60 = force one fixed level (0-255 scale) for all brightnesses."}),
                "grain": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "display": "slider", "tooltip": "Put back this % of the removed luminance noise as fine grain (colour noise stays removed). 0 = clean."}),
                "preview_at": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1e6, "step": 0.5, "tooltip": "Preview position in seconds (-1 = middle of the video)."}),
                "preview_seconds": ("FLOAT", {"default": 2.0, "min": 0.3, "max": 30.0, "step": 0.5}),
                "codec": (list(V.CODECS), {"default": list(V.CODECS)[0], "tooltip": "Output format for 'render'. H.265 10-bit: small, plays in editors and most players. ProRes: large, best for editing. H.264: plays everywhere (8-bit). FFV1: lossless."}),
                "quality": (["standard", "high", "very high"], {"default": "high"}),
                "output_folder": ("STRING", {"default": "video_denoise", "tooltip": "Folder inside ComfyUI/output, or an absolute path. Empty = next to the original video."}),
                "filename_suffix": ("STRING", {"default": "-denoised"}),
                "model": (list(MODELS), {"default": list(MODELS)[0], "tooltip": "FastDVDnet (CVPR 2020, MIT licence), bundled. The clipped-noise variant was trained on noise clipped at black/white; the standard model was better on camera footage in testing."}),
                "backend": (["auto", "torch", "torch (cpu)", "onnxruntime"], {"default": "auto"}),
                "precision": (["auto", "fp16", "fp32"], {"default": "auto", "tooltip": "auto = fp16 on GPUs (faster, half the memory), checked against a reference before use."}),
                "tile_size": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 64, "tooltip": "0 = automatic from free VRAM (and it shrinks by itself if memory runs out)."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("before", "after", "output_path")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = ("Temporal AI video denoising (FastDVDnet, 5-frame window) that streams the video: memory use "
                   "doesn't grow with length, VRAM use adapts to the GPU, and long renders resume after a stop.")

    def run(self, video, mode, strength, noise_level, grain, preview_at, preview_seconds, codec, quality,
            output_folder, filename_suffix, model, backend, precision, tile_size, unique_id=None):
        import comfy.model_management as mm
        import comfy.utils

        info: V.VideoInfo = video["info"]
        device = mm.get_torch_device() if backend in ("auto", "torch") else None
        if device is not None:
            try:  # make room: unload other models ComfyUI keeps on the GPU
                mm.free_memory(3 << 30, device)
                mm.soft_empty_cache()
            except Exception:
                pass
        be = make_backend(model, backend, device, precision, log=_info)
        runner = be.runner(tile=tile_size, log=_info)
        settings = DenoiseSettings(strength=float(strength), grain=float(grain))
        cancel = mm.throw_exception_if_processing_interrupted
        try:
            pbar = comfy.utils.ProgressBar(100, node_id=unique_id)
        except TypeError:
            pbar = comfy.utils.ProgressBar(100)

        # noise profile (measured once per video/model, then cached)
        if noise_level > 0:
            profile = flat_profile(float(noise_level))
        else:
            st = os.stat(info.path)
            key = (info.path, st.st_mtime_ns, st.st_size, model)
            profile = _ProfileCache.data.get(key)
            if profile is None:
                t0 = time.time()
                profile = measure_noise(info, be, runner, log=_info, cancel=cancel)
                _info(f"[Video Denoise] measured {profile.describe()} ({time.time() - t0:.0f}s)")
                _ProfileCache.data = {key: profile}
        try:
            if mode.startswith("preview"):
                at = info.duration / 2 if preview_at < 0 else min(preview_at, max(info.duration - preview_seconds, 0))
                tmp = folder_paths.get_temp_directory()
                sub = "video_denoise"
                os.makedirs(os.path.join(tmp, sub), exist_ok=True)
                name = f"preview_{int(time.time() * 1000) % 10 ** 9}.webm"
                clip = os.path.join(tmp, sub, name)
                t0 = time.time()
                before, after = preview_clip(info, be.ops, runner, profile, settings, at, preview_seconds,
                                             clip_path=clip, cancel=cancel)
                _info(f"[Video Denoise] preview of {preview_seconds:g}s at {at:.1f}s took {time.time() - t0:.1f}s "
                      "(left: original, right: denoised; 100% crop)")
                pbar.update_absolute(100, 100)
                return {"ui": {"images": [{"filename": name, "subfolder": sub, "type": "temp"}],
                               "animated": (True,)},
                        "result": (_to_image(before), _to_image(after), clip)}

            enc = V.EncodeSettings(codec=V.CODECS[codec], quality=quality)
            out_path = self._out_path(info, enc, output_folder, filename_suffix)
            job = RenderJob(info, out_path, settings, enc, profile, model_id=model, log=_info)

            def progress(done, total):
                pbar.update_absolute(int(100 * done / max(total, 1)), 100)

            t0 = time.time()
            path = job.run(be.ops, runner, progress=progress, cancel=cancel)
            _info(f"[Video Denoise] wrote {path} ({os.path.getsize(path) / 1e6:.0f} MB) in "
                  f"{(time.time() - t0) / 60:.1f} min")
            frame = V.grab_frame(V.probe(path), min(info.duration * 0.5, max(info.duration - 0.2, 0)))
            orig = V.grab_frame(info, min(info.duration * 0.5, max(info.duration - 0.2, 0)))
            return {"ui": {"text": [path]}, "result": (_to_image(orig), _to_image(frame), path)}
        finally:
            be.ops.empty_cache()

    @staticmethod
    def _out_path(info, enc, output_folder, suffix):
        stem = os.path.splitext(os.path.basename(info.path))[0]
        folder = (output_folder or "").strip()
        if not folder:
            folder = os.path.dirname(info.path)
        elif not os.path.isabs(folder):
            root = os.path.abspath(folder_paths.get_output_directory())
            folder = os.path.abspath(os.path.join(root, folder))
            if os.path.commonpath([folder, root]) != root:
                raise ValueError("output_folder must stay inside ComfyUI/output (or be an absolute path)")
        path = os.path.join(folder, f"{stem}{suffix}{V.container_for(info, enc)}")
        if os.path.abspath(path) == os.path.abspath(info.path):
            raise ValueError("refusing to overwrite the original video")
        return path


NODE_CLASS_MAPPINGS = {
    "RawDenoise_LoadVideo": LoadVideoForDenoise,
    "RawDenoise_VideoDenoise": VideoAIDenoise,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RawDenoise_LoadVideo": "Load Video (for AI Denoise)",
    "RawDenoise_VideoDenoise": "Video AI Denoise (FastDVDnet)",
}
