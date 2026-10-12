"""HTTP routes used by the "Choose RAW file(s)..." button (web/raw_denoise.js).

Browsers never reveal a picked file's path on disk, so the picked RAW files are
copied into ComfyUI/input/raw/.  ComfyUI's own /upload/image endpoint reads the
whole request into memory and is capped at 100 MB by default (big uncompressed
ARW/NEF files exceed that), so this route streams to disk instead.
"""

from __future__ import annotations

import hashlib
import os
import re

from .raw_core.rawio import RAW_EXTENSIONS, list_raw_files

UPLOAD_SUBFOLDER = "raw"
_CHUNK = 1 << 20


def safe_filename(name: str) -> str:
    name = os.path.basename((name or "").replace("\\", "/"))
    name = re.sub(r'[\x00-\x1f<>:"/\\|?*]', "_", name).strip(" .")
    return name


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def finalize_upload(tmp_path: str, digest: str, folder: str, filename: str) -> str:
    """Move a fully received temp file into place.

    Same name + same content  -> keep the existing file (no duplicates).
    Same name + other content -> "name (1).ARW", "name (2).ARW", ...
    Returns the final file name.
    """
    stem, ext = os.path.splitext(filename)
    candidate, i = filename, 1
    while True:
        target = os.path.join(folder, candidate)
        if not os.path.exists(target):
            os.replace(tmp_path, target)
            return candidate
        if os.path.getsize(target) == os.path.getsize(tmp_path) and _sha256(target) == digest:
            os.remove(tmp_path)
            return candidate
        candidate = f"{stem} ({i}){ext}"
        i += 1


def register(routes, web):
    """Attach the routes to ComfyUI's PromptServer route table."""

    import folder_paths

    @routes.post("/raw_denoise/upload")
    async def raw_upload(request):
        input_dir = os.path.abspath(folder_paths.get_input_directory())
        folder = os.path.join(input_dir, UPLOAD_SUBFOLDER)
        os.makedirs(folder, exist_ok=True)
        saved, rejected = [], []
        reader = await request.multipart()
        while True:
            part = await reader.next()
            if part is None:
                break
            if not getattr(part, "filename", None):
                await part.release()
                continue
            name = safe_filename(part.filename)
            if not name or os.path.splitext(name)[1].lower() not in RAW_EXTENSIONS:
                rejected.append(part.filename)
                await part.release()
                continue
            tmp = os.path.join(folder, f".{name}.{os.getpid()}.part")
            h = hashlib.sha256()
            try:
                with open(tmp, "wb") as f:
                    while True:
                        chunk = await part.read_chunk(_CHUNK)
                        if not chunk:
                            break
                        h.update(chunk)
                        f.write(chunk)
                final = finalize_upload(tmp, h.hexdigest(), folder, name)
            except Exception:
                if os.path.exists(tmp):
                    os.remove(tmp)
                raise
            saved.append(f"{UPLOAD_SUBFOLDER}/{final}")
        return web.json_response({"files": saved, "rejected": rejected})

    @routes.get("/raw_denoise/list")
    async def raw_list(request):
        return web.json_response({"files": list_raw_files(folder_paths.get_input_directory())})
