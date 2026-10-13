"""Streaming video I/O through ffmpeg pipes.

Frames are decoded one at a time to 16-bit RGB and encoded one at a time, so
memory use does not grow with the length of the video.  Colour matrix and
range are passed explicitly in both directions (the same ones), so colours
round-trip exactly, and the source's colour tags (BT.709, BT.2020, HLG, PQ,
S-Log3 inside BT.709/BT.2020 containers...) are written to the output.
"""

from __future__ import annotations

import os
import queue
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional

import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mxf", ".mkv", ".avi", ".m4v", ".mts", ".m2ts", ".webm",
                    ".mpg", ".mpeg", ".wmv", ".3gp", ".ts", ".hevc", ".h264", ".dv", ".flv"}

_NO_WINDOW = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW


# ----------------------------------------------------------------------------
# ffmpeg discovery
# ----------------------------------------------------------------------------

_FFMPEG: Optional[str] = None
_ENCODERS: Optional[set] = None


def find_ffmpeg() -> str:
    global _FFMPEG
    if _FFMPEG:
        return _FFMPEG
    candidates = []
    for env in ("RAW_DENOISE_FFMPEG", "VHS_FORCE_FFMPEG_PATH"):
        if os.environ.get(env):
            candidates.append(os.environ[env])
    try:
        import imageio_ffmpeg
        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    sys_ff = shutil.which("ffmpeg")
    if sys_ff:
        candidates.append(sys_ff)
    for c in candidates:
        if c and os.path.isfile(c):
            _FFMPEG = c
            return c
    raise RuntimeError(
        "ffmpeg was not found. Install it into ComfyUI's Python with\n"
        f'    "{sys.executable}" -m pip install imageio-ffmpeg\n'
        "(or install ffmpeg and put it on your PATH), then restart ComfyUI.")


def available_encoders() -> set:
    global _ENCODERS
    if _ENCODERS is None:
        out = subprocess.run([find_ffmpeg(), "-hide_banner", "-encoders"], capture_output=True,
                             text=True, creationflags=_NO_WINDOW).stdout
        _ENCODERS = {m.group(1) for m in re.finditer(r"^\s*[VAS][\w.]{5}\s+(\S+)", out, re.M)}
    return _ENCODERS


# ----------------------------------------------------------------------------
# probing (parses ffmpeg's stream summary; ffprobe is not always available)
# ----------------------------------------------------------------------------

_RANGE_WORDS = {"tv", "pc", "limited", "full", "mpeg", "jpeg"}
_COMMON_RATES = {23.976: Fraction(24000, 1001), 29.97: Fraction(30000, 1001),
                 59.94: Fraction(60000, 1001), 119.88: Fraction(120000, 1001),
                 47.952: Fraction(48000, 1001), 239.76: Fraction(240000, 1001)}


@dataclass
class VideoInfo:
    path: str
    width: int                 # displayed size (after rotation)
    height: int
    fps: Fraction
    duration: float            # seconds
    codec: str
    pix_fmt: str
    color_range: str = "tv"    # tv | pc
    colorspace: str = "unknown"   # matrix: bt709, bt2020nc, smpte170m, bt470bg, ...
    primaries: str = "unknown"
    transfer: str = "unknown"
    rotation: int = 0          # degrees (display matrix), already applied by ffmpeg
    audio_codecs: list = field(default_factory=list)
    timecode: Optional[str] = None
    bit_depth: int = 8
    chroma: str = "420"        # 420 | 422 | 444
    field_order: str = "progressive"

    @property
    def frames_estimate(self) -> int:
        return max(1, int(round(self.duration * float(self.fps))))

    @property
    def matrix_for_scale(self) -> str:
        cs = self.colorspace
        if cs.startswith("bt2020"):
            return "bt2020"
        if cs in ("smpte170m", "bt470bg", "bt601", "fcc"):
            return "bt601"
        if cs == "smpte240m":
            return "smpte240m"
        if cs == "bt709":
            return "bt709"
        # unknown: same guess players make
        return "bt709" if self.height >= 720 or self.width >= 1280 else "bt601"

    @property
    def interlaced(self) -> bool:
        return self.field_order not in ("progressive", "unknown", "")

    @property
    def is_hdr(self) -> bool:
        return self.transfer in ("smpte2084", "arib-std-b67")

    def describe(self) -> str:
        tags = "/".join(x for x in (self.colorspace, self.primaries, self.transfer) if x != "unknown") or "untagged"
        hdr = " HDR" if self.is_hdr else ""
        return (f"{os.path.basename(self.path)} | {self.width}x{self.height} | {float(self.fps):.5g} fps | "
                f"{_fmt_time(self.duration)} (~{self.frames_estimate} frames) | {self.codec} {self.bit_depth}-bit "
                f"4:{self.chroma[1]}:{self.chroma[2]} | {tags}{hdr} | audio: {', '.join(self.audio_codecs) or 'none'}")


def _fmt_time(s: float) -> str:
    s = max(0.0, s)
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def _rate(text: str) -> Optional[Fraction]:
    try:
        v = float(text.rstrip("k")) * (1000 if text.endswith("k") else 1)
    except ValueError:
        return None
    for k, fr in _COMMON_RATES.items():
        if abs(v - k) < 0.006:
            return fr
    return Fraction(v).limit_denominator(1001)


def parse_probe(stderr: str, path: str) -> VideoInfo:
    dur = 0.0
    m = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", stderr)
    if m:
        dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    vline = None
    lines = stderr.splitlines()
    vidx = None
    for i, ln in enumerate(lines):
        if re.search(r"Stream #\d+:\d+.*: Video: ", ln) and "attached pic" not in ln:
            vline, vidx = ln, i
            break
    if vline is None:
        raise ValueError(f"{os.path.basename(path)}: no video stream found")
    codec = re.search(r"Video: (\w+)", vline).group(1)
    after = vline.split("Video: ", 1)[1]
    # pix_fmt(range, colour, field), WxH
    pm = re.search(r",\s*([a-z0-9_]+)(?:\(([^)]*)\))?,\s*(\d{2,5})x(\d{2,5})", after)
    if not pm:
        raise ValueError(f"{os.path.basename(path)}: could not read the video format:\n{vline.strip()}")
    pix_fmt, extra, w, h = pm.group(1), pm.group(2) or "", int(pm.group(3)), int(pm.group(4))
    info = VideoInfo(path=path, width=w, height=h, fps=Fraction(30), duration=dur, codec=codec, pix_fmt=pix_fmt)
    for tok in [t.strip() for t in extra.split(",") if t.strip()]:
        if tok in _RANGE_WORDS:
            info.color_range = "pc" if tok in ("pc", "full", "jpeg") else "tv"
        elif tok in ("progressive", "tt", "bb", "tb", "bt") or "first" in tok:
            info.field_order = tok
        elif "/" in tok:
            parts = tok.split("/")
            if len(parts) == 3:
                info.colorspace, info.primaries, info.transfer = parts
        elif re.fullmatch(r"[a-z0-9\-]+", tok):
            info.colorspace = info.primaries = info.transfer = tok
    if pix_fmt.startswith("yuvj"):
        info.color_range = "pc"
    fm =re.search(r"(\d+(?:\.\d+)?k?) fps", after) or re.search(r"(\d+(?:\.\d+)?k?) tbr", after)
    if fm:
        info.fps = _rate(fm.group(1)) or Fraction(30)
    # bit depth / chroma from pix_fmt
    bd = re.search(r"p(\d{2})(?:le|be)$", pix_fmt)
    info.bit_depth = int(bd.group(1)) if bd else 8
    if "444" in pix_fmt or pix_fmt.startswith(("rgb", "bgr", "gbr")):
        info.chroma = "444"
    elif "422" in pix_fmt:
        info.chroma = "422"
    else:
        info.chroma = "420"
    # side data following the stream line
    for ln in lines[vidx + 1: vidx + 25]:
        if re.search(r"Stream #\d+:\d+", ln):
            break
        rm = re.search(r"rotation of (-?\d+(?:\.\d+)?) degrees", ln)
        if rm:
            info.rotation = int(round(float(rm.group(1)))) % 360
        rm = re.match(r"\s+rotate\s*:\s*(-?\d+)", ln)
        if rm:
            info.rotation = (-int(rm.group(1))) % 360
    if info.rotation in (90, 270):
        info.width, info.height = info.height, info.width
    for am in re.finditer(r"Stream #\d+:\d+.*: Audio: (\w+)", stderr):
        info.audio_codecs.append(am.group(1))
    tm = re.search(r"^\s+timecode\s*:\s*(\d\d:\d\d:\d\d[:;]\d\d)", stderr, re.M)
    if tm:
        info.timecode = tm.group(1)
    return info


def probe(path: str) -> VideoInfo:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    r = subprocess.run([find_ffmpeg(), "-hide_banner", "-nostdin", "-i", path], capture_output=True,
                       text=True, errors="replace", creationflags=_NO_WINDOW)
    if "Invalid data found" in r.stderr or "moov atom not found" in r.stderr:
        raise ValueError(f"{os.path.basename(path)} is not a readable video (damaged or incomplete file?)")
    return parse_probe(r.stderr, path)


# ----------------------------------------------------------------------------
# decoding
# ----------------------------------------------------------------------------

def _rng(full: bool) -> str:
    return "full" if full else "limited"


def _scale_in(info: VideoInfo, codec) -> tuple:
    """(-vf chain, output pix_fmt, bytes per sample) for decoding to the codec's layout."""
    from .video_color import yuv444_fmt
    if codec.layout == "rgb":
        if info.pix_fmt.startswith(("rgb", "bgr", "gbr", "gray")):
            return "format=rgb48le", "rgb48le", 2
        rng = _rng(info.color_range == "pc")
        return (f"scale=in_color_matrix={info.matrix_for_scale}:in_range={rng}:out_range=full:"
                f"flags=bicubic+accurate_rnd+full_chroma_int+full_chroma_inp,format=rgb48le",
                "rgb48le", 2)
    # YUV sources: only the chroma is resampled (to 4:4:4) at the native depth;
    # same range in and out, so luma code values pass through untouched
    fmt = yuv444_fmt(codec.bits_in)
    rng = _rng(codec.full_in)
    # (lanczos up / area down: the closest to lossless of swscale's interpolating
    # filters for an unprocessed round trip)
    return (f"scale=in_range={rng}:out_range={rng}:"
            f"flags=lanczos+accurate_rnd+full_chroma_int+full_chroma_inp,format={fmt}",
            fmt, 1 if codec.bits_in <= 8 else 2)


class FrameReader:
    """Iterates decoded frames, read ahead in a thread.

    Frames are in the layout of `codec` (a video_color.FrameCodec, by default
    the one for the source): planar (3, H, W) YCbCr 4:4:4 code values (uint8
    or uint16) for YUV sources, interleaved (H, W, 3) uint16 RGB otherwise.
    """

    def __init__(self, info: VideoInfo, start_seconds: float = 0.0, max_frames: Optional[int] = None,
                 prefetch: int = 6, scale_to: Optional[tuple] = None, codec=None):
        from .video_color import codec_for
        self.info = info
        self.codec = codec or codec_for(info)
        self.w, self.h = (scale_to or (info.width, info.height))
        vf, self.pix_fmt, self.itemsize = _scale_in(info, self.codec)
        if scale_to:
            vf += f",scale={self.w}:{self.h}:flags=area"
        cmd = [find_ffmpeg(), "-hide_banner", "-nostdin", "-loglevel", "error"]
        if start_seconds > 0:
            cmd += ["-ss", f"{start_seconds:.6f}"]
        # every decoded frame exactly once (no duplicates/drops from rate conversion);
        # the output is written at the average frame rate
        cmd += ["-i", info.path, "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", vf,
                "-fps_mode", "passthrough"]
        if max_frames:
            cmd += ["-frames:v", str(int(max_frames))]
        cmd += ["-f", "rawvideo", "-pix_fmt", self.pix_fmt, "pipe:1"]
        self.cmd = cmd
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     bufsize=0, creationflags=_NO_WINDOW)
        self.frame_bytes = self.w * self.h * 3 * self.itemsize
        self.planar = self.codec.layout == "yuv"
        self.q: queue.Queue = queue.Queue(maxsize=prefetch)
        self.err = b""
        self._stop = False
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()
        self.te = threading.Thread(target=self._drain_err, daemon=True)
        self.te.start()

    def _drain_err(self):
        self.err = self.proc.stderr.read() or b""

    def _read_exact(self, n):
        buf = bytearray(n)
        view = memoryview(buf)
        got = 0
        while got < n:
            k = self.proc.stdout.readinto(view[got:])
            if not k:
                break
            got += k
        return buf if got == n else None

    def _run(self):
        try:
            while not self._stop:
                buf = self._read_exact(self.frame_bytes)
                if buf is None:
                    break
                dt = "u1" if self.itemsize == 1 else "<u2"
                shape = (3, self.h, self.w) if self.planar else (self.h, self.w, 3)
                self.q.put(np.frombuffer(buf, dtype=dt).reshape(shape))
        finally:
            self.q.put(None)

    def __iter__(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            yield item
        self.proc.wait()
        self.te.join(timeout=5)
        if self.proc.returncode not in (0, None) and not self._stop:
            raise RuntimeError(f"ffmpeg could not decode {os.path.basename(self.info.path)}:\n"
                               + self.err.decode("utf-8", "replace")[-2000:])

    def close(self):
        self._stop = True
        try:
            self.proc.kill()
        except Exception:
            pass
        try:  # unblock the reader thread if it is waiting on a full queue
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass


# ----------------------------------------------------------------------------
# encoding
# ----------------------------------------------------------------------------

CODECS = {
    "H.265 10-bit (small, high quality)": "hevc",
    "ProRes 422 HQ (for editing)": "prores",
    "H.264 8-bit (most compatible)": "h264",
    "FFV1 lossless (archival, huge)": "ffv1",
}
QUALITY_CRF = {"high": (14, 12), "very high": (10, 8), "standard": (18, 16)}  # (x265 crf, x264 crf)


@dataclass
class EncodeSettings:
    codec: str = "hevc"          # hevc | prores | h264 | ffv1
    quality: str = "high"        # standard | high | very high
    preset: str = ""             # x264/x265 preset; "" = fast for H.265, medium for H.264


def output_range(info: VideoInfo, enc: EncodeSettings) -> str:
    """ProRes has no full-range signalling, so it is always written limited range."""
    if enc.codec == "prores":
        return "tv"
    return "pc" if info.color_range == "pc" else "tv"


def output_pix_fmt(info: VideoInfo, enc: EncodeSettings) -> str:
    if enc.codec == "prores":
        return "yuv422p10le"
    if enc.codec == "vp9":
        return "yuv420p"
    if enc.codec == "h264":
        return "yuv420p"  # 8-bit 4:2:0 plays everywhere
    depth = 10 if (info.bit_depth > 8 or enc.codec in ("hevc", "ffv1")) else 8
    if enc.codec == "ffv1" and info.bit_depth <= 8:
        depth = 8
    sub = {"444": "444", "422": "422"}.get(info.chroma, "420")
    return f"yuv{sub}p{'10le' if depth == 10 else ''}"


def container_for(info: VideoInfo, enc: EncodeSettings) -> str:
    if enc.codec == "ffv1":
        return ".mkv"
    if enc.codec == "prores":
        return ".mov"
    # PCM audio (Sony XAVC S records LPCM) can't go into .mp4 with stream copy
    if any(a.startswith("pcm_") for a in info.audio_codecs):
        return ".mov"
    return ".mp4"


def _color_args(info: VideoInfo, out_range: str = None) -> list:
    args = ["-color_range", out_range or ("pc" if info.color_range == "pc" else "tv")]
    if info.colorspace != "unknown":
        args += ["-colorspace", info.colorspace]
    if info.primaries != "unknown":
        args += ["-color_primaries", info.primaries]
    if info.transfer != "unknown":
        args += ["-color_trc", info.transfer]
    return args


def encoder_args(info: VideoInfo, enc: EncodeSettings) -> list:
    pix = output_pix_fmt(info, enc)
    have = available_encoders()
    if enc.codec == "hevc":
        if "libx265" not in have:
            raise RuntimeError("this ffmpeg has no H.265 encoder (libx265); choose ProRes or FFV1, "
                               "or install imageio-ffmpeg")
        crf = QUALITY_CRF.get(enc.quality, QUALITY_CRF["high"])[0]
        params = "log-level=error"
        if info.transfer == "smpte2084":
            params += ":hdr10=1:repeat-headers=1"
        return ["-c:v", "libx265", "-preset", enc.preset or "fast", "-crf", str(crf), "-pix_fmt", pix,
                "-x265-params", params]
    if enc.codec == "h264":
        if "libx264" not in have:
            raise RuntimeError("this ffmpeg has no H.264 encoder (libx264)")
        crf = QUALITY_CRF.get(enc.quality, QUALITY_CRF["high"])[1]
        return ["-c:v", "libx264", "-preset", enc.preset or "medium", "-crf", str(crf), "-pix_fmt", pix]
    if enc.codec == "prores":
        return ["-c:v", "prores_ks", "-profile:v", "3", "-vendor", "apl0", "-pix_fmt", pix]
    if enc.codec == "ffv1":
        return ["-c:v", "ffv1", "-level", "3", "-slicecrc", "1", "-g", "1", "-pix_fmt", pix]
    if enc.codec == "vp9":  # in-browser previews: plays in every current browser
        if "libvpx-vp9" in have:
            return ["-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "8", "-row-mt", "1",
                    "-crf", "24", "-b:v", "0", "-pix_fmt", pix]
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", pix]
    raise ValueError(enc.codec)


def _scale_out(info: VideoInfo, enc: EncodeSettings) -> str:
    pix = output_pix_fmt(info, enc)
    rng = "full" if output_range(info, enc) == "pc" else "limited"
    return (f"scale=in_range=full:out_color_matrix={info.matrix_for_scale}:out_range={rng}:"
            f"flags=bicubic+accurate_rnd+full_chroma_int,format={pix}")


def _yuv_out(info: VideoInfo, enc: EncodeSettings, codec) -> tuple:
    """(input pix_fmt, -vf chain) for writing planar 4:4:4 codes from `codec`:
    ffmpeg only subsamples the chroma; range and matrix are already final."""
    from .video_color import yuv444_fmt
    rng = _rng(codec.full_out)
    return yuv444_fmt(codec.bits_out), (f"scale=in_range={rng}:out_range={rng}:"
                                        f"flags=area+accurate_rnd+full_chroma_int,"
                                        f"format={output_pix_fmt(info, enc)}")


class FrameWriter:
    """Encodes frames to a file; writes happen in a thread.

    With a YUV `codec` (video_color.FrameCodec) frames are planar (3, H, W)
    4:4:4 code values from codec.encode(); otherwise (H, W, 3) uint16 RGB.
    """

    def __init__(self, info: VideoInfo, enc: EncodeSettings, path: str, size: Optional[tuple] = None,
                 queue_size: int = 4, codec=None):
        w, h = size or (info.width, info.height)
        if codec is not None and codec.layout == "yuv":
            in_fmt, vf = _yuv_out(info, enc, codec)
            self.dtype = "u1" if codec.bits_out <= 8 else "<u2"
        else:
            in_fmt, vf = "rgb48le", _scale_out(info, enc)
            self.dtype = "<u2"
        cmd = [find_ffmpeg(), "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", in_fmt, "-s", f"{w}x{h}",
               "-r", f"{info.fps.numerator}/{info.fps.denominator}", "-i", "pipe:0",
               "-vf", vf] + encoder_args(info, enc) + \
            _color_args(info, output_range(info, enc)) + [path]
        self.cmd, self.path = cmd, path
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE, creationflags=_NO_WINDOW)
        self.q: queue.Queue = queue.Queue(maxsize=queue_size)
        self.error: Optional[BaseException] = None
        self.err = b""
        self.frames = 0
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()
        self.te = threading.Thread(target=self._drain_err, daemon=True)
        self.te.start()

    def _drain_err(self):
        self.err = self.proc.stderr.read() or b""

    def _run(self):
        try:
            while True:
                item = self.q.get()
                if item is None:
                    break
                self.proc.stdin.write(item)
                self.frames += 1
        except BaseException as e:  # broken pipe = encoder died; reported in close()
            self.error = e
            while self.q.get() is not None:
                pass

    def write(self, frame: np.ndarray):
        if self.error is not None:
            self.close()
        self.q.put(np.ascontiguousarray(frame, dtype=self.dtype).tobytes())

    def close(self, abort: bool = False):
        if abort:
            try:
                self.proc.kill()
            except Exception:
                pass
            self.q.put(None)
            self.t.join(timeout=10)
            return
        self.q.put(None)
        self.t.join()
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        self.proc.wait()
        self.te.join(timeout=5)
        if self.proc.returncode != 0 or self.error is not None:
            raise RuntimeError("ffmpeg encoding failed:\n" + self.err.decode("utf-8", "replace")[-2000:])


def mux_final(info: VideoInfo, enc: EncodeSettings, segments: list, out_path: str, workdir: str,
              copy_audio: bool = True) -> str:
    """Concatenate encoded segments (stream copy) and add the source's audio."""
    lst = os.path.join(workdir, "segments.txt")
    with open(lst, "w", encoding="utf-8") as f:
        for s in segments:
            f.write("file '" + os.path.abspath(s).replace("\\", "/").replace("'", r"'\''") + "'\n")
    cmd = [find_ffmpeg(), "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
           "-f", "concat", "-safe", "0", "-i", lst]
    if copy_audio and info.audio_codecs:
        cmd += ["-i", info.path, "-map", "0:v:0", "-map", "1:a?", "-c:a", "copy",
                "-map_metadata", "1"]
    else:
        cmd += ["-map", "0:v:0"]
    cmd += ["-c:v", "copy"] + _color_args(info, output_range(info, enc))
    ext = os.path.splitext(out_path)[1].lower()
    if enc.codec == "hevc" and ext in (".mp4", ".mov"):
        cmd += ["-tag:v", "hvc1"]
    if ext in (".mp4", ".mov"):
        cmd += ["-movflags", "+faststart+write_colr"]
    if info.timecode and ext in (".mov", ".mp4"):
        cmd += ["-timecode", info.timecode]
    tmp = out_path + ".part" + ext
    cmd += [tmp]
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", creationflags=_NO_WINDOW)
    if r.returncode != 0 and copy_audio and info.audio_codecs:
        # audio that the container can't hold as-is: convert it
        i = cmd.index("-c:a")
        cmd[i + 1] = "aac" if ext == ".mp4" else "pcm_s24le"
        cmd.insert(i + 2, "-b:a" if ext == ".mp4" else "-ar")
        cmd.insert(i + 3, "320k" if ext == ".mp4" else "48000")
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", creationflags=_NO_WINDOW)
    if r.returncode != 0:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError("ffmpeg could not assemble the final video:\n" + r.stderr[-2000:])
    os.replace(tmp, out_path)
    return out_path


def grab_frame(info: VideoInfo, seconds: float) -> np.ndarray:
    """One decoded frame at `seconds` as float32 RGB (H, W, 3) in [0, 1]."""
    rd = FrameReader(info, start_seconds=max(0.0, seconds), max_frames=1, prefetch=1)
    frames = list(rd)
    if not frames:
        raise ValueError(f"no frame at {seconds:.2f}s in {os.path.basename(info.path)}")
    return rd.codec.decode_display(frames[0])
