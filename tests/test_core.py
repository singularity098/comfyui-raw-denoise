"""Self-contained checks for the numerical core (no ComfyUI, no torch needed).

    python tests/test_core.py [path/to/rawdenoise-nind.dtmodel]

Covers: ONNX parsing + reference executor vs onnxruntime, Bayer geometry for
all four CFA phases (via synthetic CFA DNGs read back through LibRaw), gain
matching, tiling seams, Amount blending, highlight protection, and the
linear-DNG round trip.
"""

import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from raw_core import denoise as rd  # noqa: E402
from raw_core.color import SRGB_TO_XYZ  # noqa: E402

PASS = []


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"FAIL {name} {detail}")
    PASS.append(name)
    print(f"ok   {name} {detail}")


# --------------------------------------------------------------------------
def write_cfa_dng(path, cfa_u16, pattern, cam_xyz, wb, black=512, white=16383):
    """Minimal CFA DNG (for tests) - pattern like 'BGGR'."""
    import tifffile
    idx = {"R": 0, "G": 1, "B": 2}
    h, w = cfa_u16.shape
    neutral = 1.0 / np.asarray(wb)
    neutral /= neutral.max()
    tags = [
        (50706, "B", 4, (1, 4, 0, 0), True),
        (50708, "s", 0, "Test Camera", True),
        (33421, "H", 2, (2, 2), True),
        (33422, "B", 4, tuple(idx[c] for c in pattern), True),
        (50710, "B", 3, (0, 1, 2), True),
        (50714, "H", 1, (black,), True),
        (50717, "H", 1, (white,), True),
        (50721, "2i", 9, sum(([int(round(v * 10000)), 10000] for v in np.ravel(cam_xyz)), []), True),
        (50778, "H", 1, 21, True),
        (50728, "2I", 3, sum(([int(round(v * 10000)), 10000] for v in neutral), []), True),
        (271, "s", 0, "Test", True),
        (272, "s", 0, "Camera", True),
    ]
    tifffile.imwrite(path, cfa_u16, photometric=32803, metadata=None, extratags=tags, subfiletype=0)


def synthetic_scene(h, w, seed=0):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = 0.25 + 0.15 * np.sin(xx / 7.0) * np.cos(yy / 11.0)
    g = 0.30 + 0.12 * np.sin((xx + yy) / 9.0)
    b = 0.20 + 0.10 * np.cos(xx / 5.0 - yy / 13.0)
    return np.stack([r, g, b], axis=2)


class FakeBayerModel:
    """Puts each packed sample back at its own photosite (exact for RGGB
    geometry) and scales by an arbitrary gain, like the real network."""
    fixed_tile = None
    device_name = "fake"

    def __call__(self, x):
        _, _, h, w = x.shape
        out = np.zeros((1, 3, 2 * h, 2 * w), dtype=np.float32)
        r, g1, g2, b = x[0]
        up = lambda a: np.repeat(np.repeat(a, 2, 0), 2, 1)
        out[0, 0] = up(r)
        out[0, 1] = up((g1 + g2) / 2)
        out[0, 2] = up(b)
        out[0, 0, 0::2, 0::2] = r
        out[0, 1, 0::2, 1::2] = g1
        out[0, 1, 1::2, 0::2] = g2
        out[0, 2, 1::2, 1::2] = b
        return out * 1.7e6


def test_geometry_all_phases():
    from raw_core.rawio import load_raw
    cam_xyz = np.linalg.inv(SRGB_TO_XYZ)  # camera == sRGB, simplest valid matrix
    wb = np.array([2.0, 1.0, 1.5])
    h, w = 300, 412
    scene = synthetic_scene(h, w)
    for pat in ("RGGB", "BGGR", "GRBG", "GBRG"):
        idx = {"R": 0, "G": 1, "B": 2}
        cfa = np.empty((h, w), np.float32)
        for i in range(2):
            for j in range(2):
                cfa[i::2, j::2] = scene[i::2, j::2, idx[pat[2 * i + j]]]
        u16 = np.round(cfa * (16383 - 512) + 512).astype(np.uint16)
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, f"t_{pat}.dng")
            write_cfa_dng(p, u16, pat, cam_xyz, wb)
            img = load_raw(p)
        check(f"load {pat}: sensor", img.sensor == "bayer", img.sensor)
        check(f"load {pat}: libraw demosaic keeps photosite values",
              np.allclose(img.cam_rgb[0::2, 0::2, idx[pat[0]]], img.cfa[0::2, 0::2], atol=2e-4))
        den = rd.denoise_bayer(img, FakeBayerModel(), tile=128, overlap=32)
        dy, dx = rd.rggb_offset(img.cfa_pattern, img.color_desc)
        err = 0.0
        for i in range(2):
            for j in range(2):
                c = idx[pat[2 * i + j]]
                ys = slice(i + (2 if i < dy else 0), h - 2, 2)
                xs = slice(j + (2 if j < dx else 0), w - 2, 2)
                err = max(err, float(np.abs(den[ys, xs, c] - img.cfa[ys, xs]).max()))
        check(f"bayer {pat}: photosites land on the right pixels/colours (incl. gain + tiling)",
              err < 1e-4, f"max err {err:.2e}")


def test_blend_and_highlights():
    rng = np.random.default_rng(0)
    den = rng.random((64, 64, 3)).astype(np.float32) * 0.5
    noisy = den + rng.normal(0, 0.05, den.shape).astype(np.float32)
    wb = np.array([2.0, 1.0, 1.6])
    check("amount 100 -> network output", np.allclose(rd.blend_amount(noisy, den, wb, 100, 100), den))
    check("amount 0, colour 0 -> original", np.allclose(rd.blend_amount(noisy, den, wb, 0, 0), noisy, atol=1e-6))
    out = rd.blend_amount(noisy, den, wb, 0, 100)
    lw = np.array([0.25, 0.5, 0.25], np.float32)
    lum = lambda a: (a * wb.astype(np.float32)) @ lw
    chroma = lambda a: a * wb.astype(np.float32) - lum(a)[..., None]
    check("amount 0 / colour 100: luminance noise kept", np.allclose(lum(out), lum(noisy), atol=1e-5))
    check("amount 0 / colour 100: colour noise removed", np.allclose(chroma(out), chroma(den), atol=1e-5))
    check("amount 50 keeps 25% of the noise amplitude", abs(rd.amount_to_residual(50) - 0.25) < 1e-9)

    noisy = np.full((40, 40, 3), 0.3, np.float32)
    noisy[15:25, 15:25] = 1.0
    den = np.full_like(noisy, 0.3)
    den[15:25, 15:25] = 0.9
    den[14, 14:26] = 0.6  # magenta-ish rim next to the clipped block
    out = rd.protect_highlights(noisy, den.copy(), 1.0)
    check("clipped block restored to white level", np.allclose(out[16:24, 16:24], 1.0))
    check("rim next to clipped block taken from original", np.allclose(out[14, 16:24], 0.3, atol=1e-6))
    check("far away pixels untouched", np.allclose(out[0:5, 0:5], 0.3))


def test_dng_roundtrip():
    import tifffile
    from raw_core.dng import write_linear_dng
    from raw_core.rawio import RawImage
    rng = np.random.default_rng(1)
    cam = (rng.random((50, 70, 3)) * 0.9).astype(np.float32)
    cam[0, 0] = -0.002
    img = RawImage(cam_rgb=cam, wb=np.array([2.1, 1.0, 1.4]), cam_xyz=np.linalg.inv(SRGB_TO_XYZ),
                   flip=6, make="NIKON CORPORATION", model="NIKON D850", source_path="/x/DSC_0001.NEF",
                   sensor="bayer", iso=6400)
    with tempfile.TemporaryDirectory() as td:
        p = write_linear_dng(img, os.path.join(td, "t.dng"))
        with tifffile.TiffFile(p) as tf:
            pg = tf.pages[0]
            data = pg.asarray()
            tags = {t.code: t.value for t in pg.tags.values()}
    check("dng: rotated for flip=6", data.shape == (70, 50, 3))
    black, white = tags[50714][0], tags[50717][0]
    back = (np.rot90(data, 1).astype(np.float32) - black) / (white - black)
    check("dng: pixel values round-trip", np.abs(back - np.clip(cam, -black / (white - black), 1)).max() < 2e-5)
    check("dng: negative values preserved", back[0, 0, 0] < 0)
    check("dng: LinearRaw", tags[262] == 34892)
    check("dng: camera name", tags[50708] == "Nikon D850", tags[50708])
    n = np.array(tags[50728], float).reshape(3, 2)
    expect = (1 / np.array([2.1, 1.0, 1.4])) / (1 / np.array([2.1, 1.0, 1.4])).max()
    check("dng: AsShotNeutral = 1/WB", np.allclose(n[:, 0] / n[:, 1], expect, rtol=1e-4))


def test_onnx(model_path):
    from raw_core.graph_exec import NumpyGraph
    from raw_core.models import read_model_bytes
    from raw_core.onnx_lite import load_onnx
    try:
        import onnxruntime as ort
    except ImportError:
        print("skip onnxruntime comparison (not installed)")
        return
    for k, data in read_model_bytes(model_path).items():
        g = load_onnx(data)
        sess = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
        shp = sess.get_inputs()[0].shape
        x = np.random.default_rng(2).random((1, shp[1], shp[2], shp[3]), dtype=np.float32) * 0.3
        ref = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        got = NumpyGraph(g)(x)
        rel = float(np.abs(got - ref).max() / np.abs(ref).max())
        check(f"onnx {k}: parser + reference executor match onnxruntime", rel < 1e-5, f"rel err {rel:.1e}")


def test_linear_dng_reload():
    """Our own linear DNG must read back through LibRaw with the same data,
    colour matrix and white balance."""
    from raw_core.dng import write_linear_dng
    from raw_core.rawio import RawImage, load_raw
    rng = np.random.default_rng(3)
    cam = (synthetic_scene(120, 160) * np.array([0.6, 1.0, 0.7])).astype(np.float32)
    cam += rng.normal(0, 0.002, cam.shape).astype(np.float32)
    cam_xyz = np.array([[0.6722, -0.0635, -0.0963], [-0.4287, 1.2460, 0.2028], [-0.0908, 0.2162, 0.5668]])
    img = RawImage(cam_rgb=cam, wb=np.array([2.0, 1.0, 1.5]), cam_xyz=cam_xyz, flip=0,
                   make="Canon", model="Canon EOS R7", source_path="/x/a.CR3", sensor="bayer", iso=3200)
    with tempfile.TemporaryDirectory() as td:
        p = write_linear_dng(img, os.path.join(td, "a-Enhanced-NR.dng"))
        back = load_raw(p)
    check("linear dng reload: treated as already demosaiced", back.sensor == "other", back.sensor)
    check("linear dng reload: colour matrix", np.allclose(back.cam_xyz, cam_xyz, atol=1e-4))
    check("linear dng reload: white balance", np.allclose(back.wb, [2.0, 1.0, 1.5], rtol=1e-3))
    check("linear dng reload: ISO", back.iso == 3200, back.iso)
    m = (cam > 0.01).all(2)
    ratio = np.median(back.cam_rgb[m] / cam[m], axis=0)
    check("linear dng reload: pixel data (LibRaw)", np.allclose(ratio, 1.0, atol=0.006), np.round(ratio, 4))


def test_upload_helpers():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "rdn_routes", os.path.join(os.path.dirname(HERE), "server_routes.py"),
        submodule_search_locations=None)
    # server_routes uses a relative import; load it as part of a throwaway package
    import types
    pkgname = "_rdn_pkg"
    pkg = types.ModuleType(pkgname)
    pkg.__path__ = [os.path.dirname(HERE)]
    sys.modules[pkgname] = pkg
    mod = importlib.import_module(pkgname + ".server_routes")
    check("upload: path traversal stripped", mod.safe_filename("../../etc/DSC1.ARW") == "DSC1.ARW")
    check("upload: windows path stripped", mod.safe_filename("C:\\Photos\\DSC2.ARW") == "DSC2.ARW")
    import hashlib
    with tempfile.TemporaryDirectory() as td:
        def put(content, name):
            tmp = os.path.join(td, ".tmp")
            open(tmp, "wb").write(content)
            return mod.finalize_upload(tmp, hashlib.sha256(content).hexdigest(), td, name)
        a = put(b"A" * 1000, "DSC0001.ARW")
        b = put(b"A" * 1000, "DSC0001.ARW")
        c = put(b"B" * 1000, "DSC0001.ARW")
        check("upload: identical re-upload reuses the file", a == b == "DSC0001.ARW")
        check("upload: same name, different photo -> renamed", c == "DSC0001 (1).ARW", c)
        check("upload: no temp files left", sorted(os.listdir(td)) == ["DSC0001 (1).ARW", "DSC0001.ARW"])


if __name__ == "__main__":
    test_upload_helpers()
    test_blend_and_highlights()
    test_dng_roundtrip()
    try:
        import rawpy  # noqa: F401
        test_geometry_all_phases()
        test_linear_dng_reload()
    except ImportError:
        print("skip CFA geometry tests (rawpy not installed)")
    if len(sys.argv) > 1:
        test_onnx(sys.argv[1])
    print(f"\n{len(PASS)} checks passed")
