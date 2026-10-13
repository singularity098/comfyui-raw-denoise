"""HTTP routes used by the file buttons in web/raw_denoise.js.

/raw_denoise/upload  Browsers never reveal a picked file's path on disk, so
                     uploaded RAW/video files are copied into ComfyUI/input/raw
                     or input/video.  ComfyUI's own /upload/image endpoint reads
                     the whole request into memory and is capped at 100 MB by
                     default, so this route streams to disk instead.
/raw_denoise/browse  Lists folders on the machine running ComfyUI, for the
                     "Browse this computer..." dialog (files are used in place,
                     nothing is copied).  Only answered for requests from the
                     same machine, unless RAW_DENOISE_ALLOW_REMOTE_BROWSE=1.
"""

from __future__ import annotations

import hashlib
import os
import re

from .raw_core.rawio import RAW_EXTENSIONS, list_raw_files
from .raw_core.video_io import VIDEO_EXTENSIONS

UPLOAD_SUBFOLDER = "raw"
KINDS = {"raw": ("raw", RAW_EXTENSIONS), "video": ("video", VIDEO_EXTENSIONS)}
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
        sub, exts = KINDS.get(request.query.get("kind", "raw"), KINDS["raw"])
        input_dir = os.path.abspath(folder_paths.get_input_directory())
        folder = os.path.join(input_dir, sub)
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
            if not name or os.path.splitext(name)[1].lower() not in exts:
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
            saved.append(f"{sub}/{final}")
        return web.json_response({"files": saved, "rejected": rejected})

    @routes.get("/raw_denoise/list")
    async def raw_list(request):
        return web.json_response({"files": list_raw_files(folder_paths.get_input_directory())})

    @routes.get("/raw_denoise/browse")
    async def raw_browse(request):
        if not is_local_request(request):
            return web.json_response({"error": "remote", "message":
                                      "Browsing folders only works when ComfyUI is opened on the same "
                                      "computer it runs on. Use Upload instead."}, status=403)
        kind = request.query.get("kind", "video")
        exts = KINDS.get(kind, KINDS["video"])[1]
        extra = [folder_paths.get_input_directory(), folder_paths.get_output_directory()]
        try:
            return web.json_response(list_folder(request.query.get("path", ""), exts, extra))
        except (OSError, ValueError) as e:
            return web.json_response({"error": "unreadable", "message": str(e)}, status=400)


def is_local_request(request) -> bool:
    if os.environ.get("RAW_DENOISE_ALLOW_REMOTE_BROWSE") == "1":
        return True
    if request.headers.get("X-Forwarded-For") or request.headers.get("Forwarded"):
        return False  # behind a proxy: the client may be anywhere
    remote = (request.remote or "").lower()
    return remote in ("127.0.0.1", "::1", "localhost") or remote.startswith("::ffff:127.") \
        or remote.startswith("127.")


def quick_places(extra=()) -> list:
    home = os.path.expanduser("~")
    places = [("Home", home)]
    for sub in ("Videos", "Movies", "Desktop", "Downloads", "Pictures", "Documents"):
        places.append((sub, os.path.join(home, sub)))
    for p in extra:
        places.append((("ComfyUI " + os.path.basename(os.path.normpath(p))), p))
    if os.name == "nt":
        import string
        for d in string.ascii_uppercase:
            root = f"{d}:\\"
            if os.path.exists(root):
                places.append((f"{d}: drive", root))
    else:
        for base in ("/Volumes", "/media/" + os.environ.get("USER", ""), "/run/media/" + os.environ.get("USER", ""), "/mnt"):
            try:
                for e in sorted(os.scandir(base), key=lambda e: e.name.lower()):
                    if e.is_dir() and not e.name.startswith("."):
                        places.append((e.name, e.path))
            except OSError:
                pass
        places.append(("/ (root)", "/"))
    seen, out = set(), []
    for label, p in places:
        ap = os.path.abspath(p)
        if ap not in seen and os.path.isdir(ap):
            seen.add(ap)
            out.append({"label": label, "path": ap})
    return out


def list_folder(path: str, exts, extra=(), limit: int = 5000) -> dict:
    path = os.path.abspath(os.path.expanduser(path or "~"))
    if os.path.isfile(path):
        path = os.path.dirname(path)
    if not os.path.isdir(path):
        raise ValueError(f"not a folder: {path}")
    dirs, files = [], []
    with os.scandir(path) as it:
        for e in it:
            if e.name.startswith(".") or e.name.startswith("$"):
                continue
            try:
                if e.is_dir():
                    dirs.append(e.name)
                elif os.path.splitext(e.name)[1].lower() in exts:
                    st = e.stat()
                    files.append({"name": e.name, "size": st.st_size, "mtime": int(st.st_mtime)})
            except OSError:
                continue
            if len(dirs) + len(files) >= limit:
                break
    dirs.sort(key=str.lower)
    files.sort(key=lambda f: f["name"].lower())
    parent = os.path.dirname(path.rstrip("\\/")) if path.rstrip("\\/") else None
    if parent == path or not parent:
        parent = None
    return {"path": path, "parent": parent, "sep": os.sep, "dirs": dirs, "files": files,
            "places": quick_places(extra)}
