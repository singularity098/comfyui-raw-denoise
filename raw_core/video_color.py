"""Exact, clip-free conversion between video code values and RGB.

ffmpeg only upsamples/downsamples chroma (yuv4:2:x <-> yuv4:4:4 at the
source's own bit depth); the YCbCr <-> RGB matrix and range scaling happen
here in floating point.  Nothing is clipped on the way in, so

  * chroma noise around black (common in low light) keeps its true values
    instead of being clipped at 0, which would lift and tint the blacks;
  * super-whites (Sony picture profiles record up to 109 %) survive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MATRIX = {  # (Kr, Kb)
    "bt709": (0.2126, 0.0722),
    "bt601": (0.299, 0.114),
    "bt2020": (0.2627, 0.0593),
    "smpte240m": (0.212, 0.087),
}


def native_bits(bits: int) -> int:
    """Bit depth of the planar 4:4:4 format a source is decoded to."""
    return 8 if bits <= 8 else bits if bits in (10, 12) else 16


def yuv444_fmt(bits: int) -> str:
    """Planar 4:4:4 pixel format at native_bits(bits).  The range is passed to
    ffmpeg separately (never via the deprecated yuvj* formats)."""
    b = native_bits(bits)
    return "yuv444p" if b == 8 else f"yuv444p{b}le"


@dataclass
class FrameCodec:
    layout: str          # "yuv" (planar 4:4:4 codes) or "rgb" (interleaved rgb48)
    bits_in: int = 8
    full_in: bool = False
    matrix: str = "bt709"
    bits_out: int = 10
    full_out: bool = False
    # RGB range kept for re-encoding (yuv layout): wide enough for every code
    # value, incl. saturated colours in the head/footroom (B reaches ~2.15)
    lo: float = -1.25
    hi: float = 2.25

    # --- range helpers ------------------------------------------------------
    @staticmethod
    def _scale(bits, full):
        if full:
            m = float(2 ** bits - 1)
            return 0.0, m, 2 ** (bits - 1), m          # y0, y range, c0, c range
        s = float(2 ** (bits - 8))
        return 16 * s, 219 * s, 128 * s, 224 * s

    @property
    def in_dtype(self):
        return np.uint8 if (self.layout == "yuv" and self.bits_in <= 8) else np.uint16

    @property
    def out_dtype(self):
        return np.uint8 if (self.layout == "yuv" and self.bits_out <= 8) else np.uint16

    # --- decode: codes -> RGB tensor (1,3,H,W), unclipped -------------------
    def decode(self, ops, frame):
        if self.layout == "rgb":
            return ops.from_u16(frame)
        c = ops.planes(frame)                     # (1,3,H,W) float32 code values
        y0, yr, c0, cr = self._scale(self.bits_in, self.full_in)
        kr, kb = MATRIX.get(self.matrix, MATRIX["bt709"])
        kg = 1.0 - kr - kb
        Y = (c[:, 0:1] - y0) * (1.0 / yr)
        Cb = (c[:, 1:2] - c0) * (1.0 / cr)
        Cr = (c[:, 2:3] - c0) * (1.0 / cr)
        R = Y + (2 * (1 - kr)) * Cr
        B = Y + (2 * (1 - kb)) * Cb
        G = (Y - kr * R - kb * B) * (1.0 / kg)
        return ops.cat([R, G, B])

    # --- encode: RGB tensor -> codes (numpy), for the encoder pipe -----------
    def encode(self, ops, rgb):
        if self.layout == "rgb":
            return ops.to_u16(rgb)
        rgb = ops.clip(rgb, self.lo, self.hi)
        kr, kb = MATRIX.get(self.matrix, MATRIX["bt709"])
        kg = 1.0 - kr - kb
        R, G, B = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        Y = kr * R + kg * G + kb * B
        Cb = (B - Y) * (1.0 / (2 * (1 - kb)))
        Cr = (R - Y) * (1.0 / (2 * (1 - kr)))
        y0, yr, c0, cr = self._scale(self.bits_out, self.full_out)
        codes = ops.cat([Y * yr + y0, Cb * cr + c0, Cr * cr + c0])
        return ops.to_codes(codes, self.bits_out)

    def crop(self, frame, x, y, w, h):
        if self.layout == "yuv":
            return frame[:, y:y + h, x:x + w]
        return frame[y:y + h, x:x + w]

    def display(self, ops, rgb) -> np.ndarray:
        """RGB tensor -> (H, W, 3) float32 in [0, 1] for ComfyUI images."""
        return np.clip(ops.to_float_hwc(rgb), 0.0, 1.0)

    def decode_display(self, frame) -> np.ndarray:
        from .video_denoise import NumpyOps
        ops = NumpyOps()
        return self.display(ops, self.decode(ops, frame))


def codec_for(info, enc=None) -> FrameCodec:
    """FrameCodec for a VideoInfo (and the EncodeSettings of the output, if any)."""
    from . import video_io as V
    rgb_src = info.pix_fmt.startswith(("rgb", "bgr", "gbr"))
    bits_out, full_out = 16, False
    if enc is not None:
        pix = V.output_pix_fmt(info, enc)
        bits_out = 10 if "10le" in pix else 12 if "12le" in pix else 16 if "16le" in pix else 8
        if rgb_src:
            bits_out = 16
        full_out = V.output_range(info, enc) == "pc"
    return FrameCodec(layout="rgb" if rgb_src else "yuv",
                      bits_in=16 if rgb_src else native_bits(info.bit_depth),
                      full_in=info.color_range == "pc", matrix=info.matrix_for_scale,
                      bits_out=bits_out, full_out=full_out)
