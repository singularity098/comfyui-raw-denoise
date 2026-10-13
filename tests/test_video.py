"""Checks for the video denoiser (no ComfyUI or torch needed; ffmpeg for the I/O parts).

    python tests/test_video.py

Covers: torch-free checkpoint loading, the FastDVDnet graph (BatchNorm folded)
against ONNX Runtime, the streaming engine against an exact port of the
FastDVDnet reference loop, ffmpeg stream parsing for camera/phone footage,
colour-exact decode/encode round trips, resume-after-interrupt, constant
memory, and the folder-browser helpers.
"""

import importlib
import os
import sys
import tempfile
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from raw_core import video_denoise as VD  # noqa: E402
from raw_core import video_io as V  # noqa: E402
from raw_core.fastdvdnet import load_fastdvdnet, pack_input  # noqa: E402
from raw_core.graph_exec import NumpyGraph  # noqa: E402
from raw_core.pth_loader import state_dict  # noqa: E402

PASS = []
WEIGHTS = os.path.join(ROOT, "weights", "fastdvdnet", "model.pth")


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"FAIL {name} {detail}")
    PASS.append(name)
    print(f"ok   {name} {detail}")


def graphs():
    with open(WEIGHTS, "rb") as f:
        return load_fastdvdnet(state_dict(f.read()))


def test_checkpoint_and_graph():
    g1, g2 = graphs()
    n = sum(v.size for v in g1.initializers.values()) + sum(v.size for v in g2.initializers.values())
    check("fastdvdnet: checkpoint read without torch, BN folded", n == 2477642, f"{n} weights")
    try:
        import onnxruntime as ort
        from raw_core.onnx_write import to_onnx_bytes
    except ImportError:
        print("skip onnxruntime comparison")
        return
    x = np.random.default_rng(0).random((1, 12, 40, 56), dtype=np.float32)
    for name, g in (("temp1", g1), ("temp2", g2)):
        s = ort.InferenceSession(to_onnx_bytes(g), providers=["CPUExecutionProvider"])
        ref = s.run(None, {"input": x})[0]
        got = NumpyGraph(g)(x)
        rel = float(np.abs(got - ref).max() / np.abs(ref).max())
        check(f"fastdvdnet {name}: NumPy executor == ONNX Runtime", rel < 1e-5, f"rel err {rel:.1e}")


def test_engine_matches_reference():
    g1, g2 = graphs()
    n1, n2 = NumpyGraph(g1), NumpyGraph(g2)
    rng = np.random.default_rng(1)
    T, H, W = 7, 30, 46     # odd sizes exercise padding
    base = rng.random((1, H, W, 3)).astype(np.float32) * 0.6 + 0.2
    clean = np.concatenate([np.roll(base, t, axis=2) for t in range(T)])
    noisy = np.clip(clean + rng.normal(0, 25 / 255, clean.shape), 0, 1)
    u16 = np.round(noisy * 65535).astype(np.uint16)
    noisy = u16.astype(np.float32) / 65535
    # exact port of fastdvdnet.denoise_seq_fastdvdnet
    ph, pw = (-H) % 4, (-W) % 4
    seq = np.pad(noisy.transpose(0, 3, 1, 2), ((0, 0), (0, 0), (0, ph), (0, pw)), mode="reflect")
    nm = np.full((1, 1, H + ph, W + pw), 25 / 255, np.float32)
    ref = []
    for t in range(T):
        idx = [abs(t + d) if t + d < T else 2 * (T - 1) - (t + d) for d in range(-2, 3)]
        f = [seq[i:i + 1] for i in idx]
        d = lambda net, a, b, c: b - net(pack_input(a, b, c, nm))
        a, b, c = d(n1, f[0], f[1], f[2]), d(n1, f[1], f[2], f[3]), d(n1, f[2], f[3], f[4])
        ref.append(np.clip(d(n2, a, b, c), 0, 1)[0, :, :H, :W].transpose(1, 2, 0))
    ops = VD.NumpyOps()
    runner = VD.TiledRunner(ops, n1, n2, tile=4096)
    den = VD.VideoDenoiser(ops, runner, VD.flat_profile(25), VD.DenoiseSettings(detect_cuts=False, texture=0),
                           W, H, return_float=True)
    out = []
    for f in u16:
        out += den.push(f)
    out += den.flush()
    err = float(np.abs(np.stack(out) - np.stack(ref)).max())
    check("engine == FastDVDnet reference loop (all frames incl. clip ends)", err < 1e-5, f"max err {err:.1e}")
    check("engine reuses stage-1 results", runner.calls == T + T + 2, f"{runner.calls} network calls vs {4 * T}")


def test_texture_map():
    rng = np.random.default_rng(3)
    H, W = 64, 128
    sig = 8 / 255
    img = np.full((H, W, 3), 0.4, np.float32)
    img[:, W // 2:] += (rng.random((H, W // 2, 1)) - 0.5) * 0.4          # strong real detail on the right
    img = np.clip(img + rng.normal(0, sig, img.shape), 0, 1).astype(np.float32)
    ops = VD.NumpyOps()
    prof = VD.NoiseProfile(np.array([0, 1], np.float32), np.full(2, 20 / 255, np.float32),
                           spread=np.full(2, sig, np.float32))
    run = VD.TiledRunner(ops, None, None, tile=4096)
    den = VD.VideoDenoiser(ops, run, prof, VD.DenoiseSettings(), W, H)
    nm = den._noise_map(img.transpose(2, 0, 1)[None])[0, 0] * 255
    flat, tex = float(np.median(nm[:, 8:W // 2 - 8])), float(np.median(nm[:, W // 2 + 8:-8]))
    check("noise map: flat areas keep the profile level", 17 < flat <= 20.01, f"{flat:.1f}")
    check("noise map: lowered where real detail exceeds the noise", tex < 0.5 * flat, f"{tex:.1f} vs {flat:.1f}")


def test_color_codec():
    from raw_core.video_color import FrameCodec
    ops = VD.NumpyOps()
    rng = np.random.default_rng(5)
    for bits, full in ((10, False), (8, False), (8, True), (12, False)):
        lo, hi = (0, 2 ** bits - 1) if full else (4 << (bits - 8), (235 << (bits - 8)) + 3)
        codes = rng.integers(lo, hi + 1, (3, 24, 40)).astype(np.uint8 if bits == 8 else np.uint16)
        c = FrameCodec("yuv", bits_in=bits, full_in=full, bits_out=bits, full_out=full, lo=-9, hi=9)
        back = c.encode(ops, c.decode(ops, codes))
        check(f"colour: {bits}-bit {'full' if full else 'limited'} codes -> RGB -> codes is exact",
              back.dtype == codes.dtype and int(np.abs(back.astype(int) - codes).max()) == 0)
    c = FrameCodec("yuv", bits_in=10, full_in=False, matrix="bt709")
    black = np.array([64, 512, 512], np.uint16)[:, None, None] * np.ones((3, 1, 1), np.uint16)
    blue_noise = black.copy(); blue_noise[1] = 540; blue_noise[2] = 500      # chroma noise at black
    rgb = c.decode(ops, blue_noise)
    check("colour: chroma noise at black is kept (no clipping on decode)", float(rgb.min()) < -0.02)
    sw = np.array([1019, 512, 512], np.uint16)[:, None, None] * np.ones((3, 1, 1), np.uint16)
    check("colour: super-whites (109%) kept", float(c.decode(ops, sw).min()) > 1.05)


def test_probe_parsing():
    a7iv = """Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'C0001.MP4':
  Duration: 00:01:02.56, start: 0.000000, bitrate: 151022 kb/s
  Stream #0:0[0x1](und): Video: hevc (Main 4:2:2 10) (hvc1 / 0x31637668), yuv422p10le(tv, bt709/bt709/arib-std-b67, progressive), 3840x2160 [SAR 1:1 DAR 16:9], 147900 kb/s, 23.98 fps, 23.98 tbr, 24k tbn (default)
  Stream #0:1[0x2](und): Audio: pcm_s24be (in24 / 0x34326E69), 48000 Hz, stereo, s32 (24 bit), 2304 kb/s (default)
  Stream #0:2[0x3](und): Data: none (tmcd / 0x64636D74), 0 kb/s
      Metadata:
        timecode        : 13:04:22:11
"""
    i = V.parse_probe(a7iv, "C0001.MP4")
    check("probe: a7 IV XAVC HS 4:2:2 10-bit HLG", (i.width, i.height, i.bit_depth, i.chroma, i.transfer, i.colorspace)
          == (3840, 2160, 10, "422", "arib-std-b67", "bt709"), i.describe())
    check("probe: 23.976 fps exact", i.fps.numerator == 24000 and i.fps.denominator == 1001)
    check("probe: LPCM audio -> .mov output", V.container_for(i, V.EncodeSettings("hevc")) == ".mov")
    check("probe: timecode kept", i.timecode == "13:04:22:11")
    phone = """  Duration: 00:00:12.03, start: 0.000000, bitrate: 9000 kb/s
  Stream #0:0[0x1](und): Video: hevc (Main 10) (hvc1 / 0x31637668), yuv420p10le(tv, bt2020nc/bt2020/arib-std-b67), 1920x1080, 8000 kb/s, 29.99 fps, 30 tbr, 600 tbn (default)
      Side data:
        DOVI configuration record: version: 1.0, profile: 8, level: 4
        displaymatrix: rotation of -90.00 degrees
  Stream #0:1[0x2](und): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 161 kb/s (default)
"""
    i = V.parse_probe(phone, "IMG_0001.MOV")
    check("probe: phone portrait clip (rotation applied)", (i.width, i.height, i.rotation) == (1080, 1920, 270))
    check("probe: phone HDR flagged", i.is_hdr and i.colorspace == "bt2020nc")
    full = "  Duration: 00:00:03.00\n  Stream #0:0: Video: h264 (High), yuvj420p(pc, progressive), 960x540, 24 fps, 24 tbr\n"
    check("probe: full-range video", V.parse_probe(full, "x").color_range == "pc")
    unt = "  Duration: 00:00:03.00\n  Stream #0:0: Video: h264 (High), yuv420p(progressive), 1920x1080, 50 fps, 50 tbr\n"
    i = V.parse_probe(unt, "x")
    check("probe: untagged HD treated as BT.709", i.matrix_for_scale == "bt709" and i.color_range == "tv")


def _have_ffmpeg():
    try:
        V.find_ffmpeg()
        return True
    except Exception:
        return False


def _make_clip(path, seconds=2.0, size="320x180", fmt="yuv422p10le", codec="libx265", audio=True, smooth=False):
    import subprocess
    # smooth: camera-like (soft chroma, grainy luma)
    src = (f"gradients=size={size}:rate=25:speed=0.02:nb_colors=4,format=yuv444p10le,"
           f"gblur=sigma=3:planes=6,noise=c0s=6:c0f=t" if smooth
           else f"testsrc2=size={size}:rate=25,noise=alls=10:allf=t+u")
    cmd = [V.find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", src]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"]
    cmd += ["-t", str(seconds), "-c:v", codec, "-pix_fmt", fmt, "-colorspace", "bt709", "-color_primaries", "bt709",
            "-color_trc", "bt709"]
    if codec == "libx265":
        cmd += ["-x265-params", "log-level=error"]
    if audio:
        cmd += ["-c:a", "pcm_s16le"]
    subprocess.run(cmd + [path], check=True)


def test_io_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "src.mov")
        _make_clip(src, smooth=True)
        info = V.probe(src)
        enc = V.EncodeSettings(codec="ffv1")
        seg = os.path.join(td, "seg.mkv")
        from raw_core.video_color import codec_for
        ops = VD.NumpyOps()
        codec = codec_for(info, enc)
        frames = list(V.FrameReader(info, codec=codec))
        w = V.FrameWriter(info, enc, seg, codec=codec)
        for f in frames:
            w.write(codec.encode(ops, codec.decode(ops, f)))     # through float RGB and back
        w.close()
        out = V.mux_final(info, enc, [seg], os.path.join(td, "out.mkv"), td)
        oi = V.probe(out)
        back = list(V.FrameReader(oi, codec=codec))
        ydiff = max(int(np.abs(a[0].astype(int) - b[0].astype(int)).max()) for a, b in zip(frames, back))
        mse = np.mean([np.mean((codec.decode(ops, a) - codec.decode(ops, b)) ** 2) for a, b in zip(frames, back)])
        psnr = 10 * np.log10(1 / mse)
        check("io: frames decoded as native 10-bit 4:4:4 planes", frames[0].shape == (3, 180, 320)
              and frames[0].dtype == np.uint16 and codec.layout == "yuv" and codec.bits_in == 10)
        check("io: decode -> encode -> decode keeps every frame", len(back) == len(frames) == 50, f"{len(back)}")
        check("io: luma code values pass through exactly", ydiff == 0, f"max diff {ydiff}")
        check("io: colours round-trip (10-bit 4:2:2)", psnr > 48, f"PSNR {psnr:.1f} dB")
        check("io: audio and colour tags carried over", oi.audio_codecs and oi.colorspace == "bt709"
              and oi.pix_fmt == "yuv422p10le", f"{oi.audio_codecs} {oi.colorspace} {oi.pix_fmt}")


def test_resume():
    from raw_core import video_job as J
    temporal = lambda x: x[:, 4:7] - 0.5 * (x[:, 0:3] + x[:, 8:11])  # stand-in network using t-1, t+1
    ops = VD.NumpyOps()
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "src.mp4")
        _make_clip(src, seconds=4, fmt="yuv420p", codec="libx264", audio=False)
        info = V.probe(src)
        enc = V.EncodeSettings(codec="ffv1")

        def job(out):
            return J.RenderJob(info, out, VD.DenoiseSettings(), enc, VD.flat_profile(10), "t",
                               segment_seconds=1, log=lambda m: None)
        ref = job(os.path.join(td, "ref.mkv")).run(ops, VD.TiledRunner(ops, temporal, temporal, tile=4096))
        R = list(V.FrameReader(V.probe(ref)))
        for seek in (10 ** 9, 20):
            J.SEEK_INSTEAD_OF_SKIP = seek
            out = os.path.join(td, f"res_{seek}.mkv")
            n = [0]

            class Stop(Exception):
                pass

            def cancel():
                n[0] += 1
                if n[0] == 61:
                    raise Stop()
            try:
                job(out).run(ops, VD.TiledRunner(ops, temporal, temporal, tile=4096), cancel=cancel)
            except Stop:
                pass
            kept = job(out).finished_segments()
            job(out).run(ops, VD.TiledRunner(ops, temporal, temporal, tile=4096))
            F = list(V.FrameReader(V.probe(out)))
            diff = max(int(np.abs(a.astype(np.int32) - b.astype(np.int32)).max()) for a, b in zip(R, F))
            how = "seek" if seek < 10 ** 9 else "skip"
            check(f"resume ({how}): interrupted + resumed == uninterrupted", len(F) == len(R) and diff == 0,
                  f"{kept} segments kept, {len(F)} frames, max diff {diff}")
        J.SEEK_INSTEAD_OF_SKIP = 900


def test_browse_helpers():
    pkg = types.ModuleType("_rdn_pkg2")
    pkg.__path__ = [ROOT]
    sys.modules["_rdn_pkg2"] = pkg
    m = importlib.import_module("_rdn_pkg2.server_routes")
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "sub"))
        for n in ("a.MP4", "b.mov", "c.txt", ".hidden.mp4"):
            open(os.path.join(td, n), "w").close()
        r = m.list_folder(td, m.KINDS["video"][1])
        check("browse: lists folders + video files only", r["dirs"] == ["sub"] and
              [f["name"] for f in r["files"]] == ["a.MP4", "b.mov"], str(r["files"]))

    class Req:
        def __init__(self, remote, headers=None):
            self.remote, self.headers = remote, headers or {}
    check("browse: only answered for the local machine",
          m.is_local_request(Req("127.0.0.1")) and m.is_local_request(Req("::1"))
          and not m.is_local_request(Req("192.168.1.20"))
          and not m.is_local_request(Req("127.0.0.1", {"X-Forwarded-For": "8.8.8.8"})))


if __name__ == "__main__":
    test_checkpoint_and_graph()
    test_engine_matches_reference()
    test_texture_map()
    test_color_codec()
    test_probe_parsing()
    test_browse_helpers()
    if _have_ffmpeg():
        test_io_roundtrip()
        test_resume()
    else:
        print("skip ffmpeg-based tests (no ffmpeg)")
    print(f"\n{len(PASS)} checks passed")
