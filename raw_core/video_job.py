"""Render jobs: segmented, resumable, constant-memory video denoising.

The output is written as ~10 s segments in a work folder.  If ComfyUI is
stopped (cancel, crash, power cut), the next run with the same settings skips
the finished segments.  When all are done they are joined without
re-encoding and the source's audio, timecode and metadata are added.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict
from typing import Callable, Optional

import numpy as np

from . import video_io as V
from .video_color import codec_for
from .video_denoise import DenoiseSettings, NoiseProfile, VideoDenoiser

MANIFEST = "manifest.json"
SEEK_INSTEAD_OF_SKIP = 900   # resume by seeking when more frames than this are done


def _fmt_eta(s: float) -> str:
    s = int(max(0, s))
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def job_key(info: V.VideoInfo, settings: DenoiseSettings, enc: V.EncodeSettings, model_id: str,
            profile: NoiseProfile) -> str:
    st = os.stat(info.path)
    blob = json.dumps({
        "src": os.path.abspath(info.path), "size": st.st_size, "mtime": int(st.st_mtime),
        "settings": asdict(settings), "enc": asdict(enc), "model": model_id,
        "profile": [round(float(v) * 255, 2) for v in profile.ys],
        "spread": None if profile.spread is None else [round(float(v) * 255, 2) for v in profile.spread],
        "pipeline": 2,
    }, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


class RenderJob:
    def __init__(self, info: V.VideoInfo, out_path: str, settings: DenoiseSettings,
                 enc: V.EncodeSettings, profile: NoiseProfile, model_id: str,
                 segment_seconds: float = 10.0, work_root: Optional[str] = None, log=print):
        self.info, self.out_path, self.settings, self.enc = info, out_path, settings, enc
        self.profile, self.log = profile, log
        self.seg_frames = max(24, int(round(segment_seconds * float(info.fps))))
        key = job_key(info, settings, enc, model_id, profile)
        root = work_root or os.path.join(os.path.dirname(os.path.abspath(out_path)), ".video_denoise_work")
        stem = os.path.splitext(os.path.basename(info.path))[0][:40]
        self.workdir = os.path.join(root, f"{stem}-{key}")
        os.makedirs(self.workdir, exist_ok=True)
        self.manifest_path = os.path.join(self.workdir, MANIFEST)
        self.manifest = self._load_manifest()

    # -- manifest -------------------------------------------------------------
    def _load_manifest(self):
        try:
            with open(self.manifest_path, encoding="utf-8") as f:
                m = json.load(f)
            if m.get("seg_frames") == self.seg_frames:
                return m
        except Exception:
            pass
        return {"version": 1, "seg_frames": self.seg_frames, "done": [], "total_frames": None,
                "source": self.info.path, "output": self.out_path}

    def _save_manifest(self):
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=1)
        os.replace(tmp, self.manifest_path)

    def seg_path(self, i: int) -> str:
        return os.path.join(self.workdir, f"seg_{i:05d}.mkv")

    def finished_segments(self) -> int:
        n = 0
        done = set(self.manifest.get("done", []))
        while n in done and os.path.isfile(self.seg_path(n)) and os.path.getsize(self.seg_path(n)) > 0:
            n += 1
        return n

    # -- run --------------------------------------------------------------------
    def run(self, ops, runner, progress: Optional[Callable[[int, int], None]] = None,
            cancel: Optional[Callable[[], None]] = None) -> str:
        info, N = self.info, self.seg_frames
        done_segs = self.finished_segments()
        total_est = self.manifest.get("total_frames") or info.frames_estimate
        start = done_segs * N
        if self.manifest.get("total_frames") is not None and start >= self.manifest["total_frames"]:
            return self._finish(done_segs)
        if start:
            self.log(f"[Video Denoise] resuming: {done_segs} finished segment(s) found, "
                     f"continuing at frame {start} ({_fmt_eta(start / float(info.fps))})")
        first = 0
        seek = 0.0
        if start > SEEK_INSTEAD_OF_SKIP:
            first = start - 3
            seek = (first - 0.5) / float(info.fps)
        codec = codec_for(info, self.enc)
        reader = V.FrameReader(info, start_seconds=seek, codec=codec)
        den = VideoDenoiser(ops, runner, self.profile, self.settings, info.width, info.height,
                            first_index=first, emit_from=start, codec=codec)
        writer, cur_seg = None, None
        produced = start
        t0, last_log = time.time(), 0.0
        try:
            def emit(frames):
                nonlocal writer, cur_seg, produced, last_log
                for fr in frames:
                    seg = produced // N
                    if seg != cur_seg:
                        if writer is not None:
                            writer.close()
                            self._mark_done(cur_seg)
                        writer = V.FrameWriter(info, self.enc, self.seg_path(seg) + ".part.mkv",
                                               codec=codec)
                        cur_seg = seg
                    writer.write(fr)
                    produced += 1
                    if progress:
                        progress(produced, max(total_est, produced))
                    now = time.time()
                    if now - last_log > 15:
                        last_log = now
                        rate = (produced - start) / max(now - t0, 1e-6)
                        left = max(total_est - produced, 0) / max(rate, 1e-6)
                        self.log(f"[Video Denoise] frame {produced}/{total_est}  {rate:.2f} fps  "
                                 f"ETA {_fmt_eta(left)}")

            for frame in reader:
                if cancel:
                    cancel()
                emit(den.push(frame))
            emit(den.flush())
            total = den.next_in if first == 0 else max(den.next_in, produced)
            if writer is not None:
                writer.close()
                self._mark_done(cur_seg)
                writer = None
            self.manifest["total_frames"] = produced
            self._save_manifest()
        except BaseException:
            if writer is not None:
                writer.close(abort=True)
                try:
                    os.remove(self.seg_path(cur_seg) + ".part.mkv")
                except OSError:
                    pass
            reader.close()
            raise
        elapsed = time.time() - t0
        self.log(f"[Video Denoise] denoised {produced - start} frames in {_fmt_eta(elapsed)} "
                 f"({(produced - start) / max(elapsed, 1e-6):.2f} fps); scene cuts detected: {den.stats['cuts']}")
        n_segs = (produced + N - 1) // N
        return self._finish(n_segs)

    def _mark_done(self, seg: int):
        part = self.seg_path(seg) + ".part.mkv"
        if os.path.exists(part):
            os.replace(part, self.seg_path(seg))
        done = set(self.manifest.get("done", []))
        done.add(int(seg))
        self.manifest["done"] = sorted(done)
        self._save_manifest()

    def _finish(self, n_segs: int) -> str:
        segs = [self.seg_path(i) for i in range(n_segs)]
        missing = [s for s in segs if not os.path.isfile(s)]
        if missing:
            raise RuntimeError(f"internal error: missing segments {missing[:3]}")
        os.makedirs(os.path.dirname(os.path.abspath(self.out_path)), exist_ok=True)
        V.mux_final(self.info, self.enc, segs, self.out_path, self.workdir)
        shutil.rmtree(self.workdir, ignore_errors=True)
        parent = os.path.dirname(self.workdir)
        try:
            if not os.listdir(parent):
                os.rmdir(parent)
        except OSError:
            pass
        return self.out_path


def preview_clip(info: V.VideoInfo, ops, runner, profile: NoiseProfile, settings: DenoiseSettings,
                 at_seconds: float, seconds: float, clip_path: Optional[str] = None,
                 crop_center=(0.5, 0.5), cancel=None):
    """Denoise a short section. Returns (before, after) middle frames as float
    (H, W, 3) arrays in [0, 1] and, if clip_path is given, writes a side-by-side
    100% crop clip (original | denoised) there.  Frames are streamed: only the
    crops and the middle frame are kept, so 4K previews stay small in memory."""
    fps = float(info.fps)
    n = max(5, int(round(seconds * fps)))
    start = max(0.0, at_seconds - 2.0 / fps)
    codec = codec_for(info)
    reader = V.FrameReader(info, start_seconds=start, max_frames=n + 4, codec=codec)
    den = VideoDenoiser(ops, runner, profile, settings, info.width, info.height,
                        return_float=True, codec=codec)
    H, W = info.height, info.width
    cw, ch = min(W, 960) // 2 * 2, min(H, 540) // 2 * 2
    cx = int(np.clip(round(crop_center[0] * W - cw / 2), 0, W - cw))
    cy = int(np.clip(round(crop_center[1] * H - ch / 2), 0, H - ch))
    target = (n + 4) // 2
    w = None
    if clip_path:
        sbs_info = V.VideoInfo(path=info.path, width=cw * 2 + 4, height=ch, fps=info.fps, duration=0,
                               codec="rawvideo", pix_fmt="rgb48le", color_range="tv",
                               colorspace=info.colorspace, primaries=info.primaries, transfer=info.transfer)
        if sbs_info.colorspace == "unknown":
            sbs_info.colorspace = "bt709" if info.matrix_for_scale == "bt709" else "smpte170m"
        w = V.FrameWriter(sbs_info, V.EncodeSettings(codec="vp9"), clip_path)
    pending = []                       # noisy frames waiting for their output
    state = {"i": 0, "before": None, "after": None}

    def take(outs):
        for out in outs:
            noisy = pending.pop(0)
            i = state["i"]
            state["i"] += 1
            if i <= target:            # keep the frame nearest the middle
                state["before"], state["after"] = noisy, out
            if w is not None:
                a = codec.decode_display(codec.crop(noisy, cx, cy, cw, ch))
                b = np.clip(out[cy:cy + ch, cx:cx + cw], 0.0, 1.0)
                frame = np.full((ch, cw * 2 + 4, 3), 65535, np.uint16)
                frame[:, :cw] = np.rint(a * 65535)
                frame[:, cw + 4:] = np.rint(b * 65535)
                w.write(frame)

    try:
        for f in reader:
            if cancel:
                cancel()
            pending.append(f)
            take(den.push(f))
        take(den.flush())
    except BaseException:
        if w is not None:
            w.close(abort=True)
        raise
    finally:
        reader.close()
    if w is not None:
        w.close()
    if state["before"] is None:
        raise ValueError(f"no frames could be read at {at_seconds:.1f}s")
    before = codec.decode_display(state["before"])
    after = np.clip(state["after"], 0.0, 1.0).astype(np.float32)
    return before, after
