"""ComfyUI RAW AI Denoise - AI noise reduction for camera RAW photos and videos.

Photos: Load RAW Image -> RAW AI Denoise (RawNIND) -> Save Linear DNG
        (+ Develop RAW to Image, Save Image 16-bit TIFF, Before/After 100% Crop)
Video:  Load Video (for AI Denoise) -> Video AI Denoise (FastDVDnet, streaming)
"""

import sys

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .video_nodes import NODE_CLASS_MAPPINGS as _VIDEO_NODES
from .video_nodes import NODE_DISPLAY_NAME_MAPPINGS as _VIDEO_NAMES

NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **_VIDEO_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **_VIDEO_NAMES}

WEB_DIRECTORY = "./web"  # "Choose RAW file(s)..." button / drag & drop

try:
    from aiohttp import web as _web
    from server import PromptServer

    from . import server_routes

    server_routes.register(PromptServer.instance.routes, _web)
except Exception as e:  # pragma: no cover - only outside a running ComfyUI
    print(f"[RAW Denoise] file-picker upload route not registered: {e}")

try:
    import rawpy  # noqa: F401
except ImportError:
    print("\n[RAW Denoise] WARNING: the 'rawpy' package is missing, so RAW files can't be read.\n"
          "  Install it into THIS Python (the one running ComfyUI) and restart:\n"
          f'    "{sys.executable}" -m pip install rawpy tifffile imageio-ffmpeg\n')

try:
    from .raw_core.video_io import find_ffmpeg as _find_ffmpeg
    _find_ffmpeg()
except Exception:
    print("\n[RAW Denoise] WARNING: ffmpeg not found, so the video nodes can't read or write video.\n"
          "  Install it into THIS Python (the one running ComfyUI) and restart:\n"
          f'    "{sys.executable}" -m pip install imageio-ffmpeg\n')

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
