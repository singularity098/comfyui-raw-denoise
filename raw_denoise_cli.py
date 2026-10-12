#!/usr/bin/env python3
"""Batch RAW denoising from the command line (same engine as the ComfyUI nodes).

Examples
  python raw_denoise_cli.py IMG_0001.CR3
  python raw_denoise_cli.py ~/Pictures/shoot/*.NEF --amount 40 --out ~/Pictures/denoised
  python raw_denoise_cli.py photo.ARW --tiff --backend onnxruntime

Needs: numpy, rawpy, tifffile and either torch or onnxruntime.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from raw_core import denoise as rd  # noqa: E402
from raw_core.dng import write_linear_dng  # noqa: E402
from raw_core.models import MODEL_FILENAME, download_model, load_models  # noqa: E402
from raw_core.rawio import RAW_EXTENSIONS, load_raw  # noqa: E402
from raw_core.render import develop  # noqa: E402
from raw_core.tiff16 import write_tiff16  # noqa: E402

logging.getLogger("tifffile").setLevel(logging.ERROR)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Lightroom-style AI denoise for RAW files (RawNIND).")
    ap.add_argument("inputs", nargs="+", help="RAW files, folders or glob patterns")
    ap.add_argument("--amount", type=float, default=50, help="0-100, like Lightroom's Amount (default 50)")
    ap.add_argument("--color", type=float, default=100, help="0-100 colour-noise reduction (default 100)")
    ap.add_argument("--mode", default="auto", choices=["auto", "bayer", "linear"])
    ap.add_argument("--backend", default="auto", choices=["auto", "torch", "torch-cpu", "onnxruntime"])
    ap.add_argument("--device", default=None, help="torch device, e.g. cuda, mps, cpu (default: best available)")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--model", default=None, help=f"path to {MODEL_FILENAME} (downloaded if missing)")
    ap.add_argument("--out", default=None, help="output folder (default: next to each RAW)")
    ap.add_argument("--suffix", default="-Enhanced-NR")
    ap.add_argument("--tiff", action="store_true", help="also write a 16-bit sRGB TIFF")
    ap.add_argument("--no-dng", action="store_true", help="skip the DNG")
    ap.add_argument("--demosaic", default="AHD")
    args = ap.parse_args(argv)

    files = []
    for item in args.inputs:
        if os.path.isdir(item):
            files += sorted(os.path.join(item, f) for f in os.listdir(item)
                            if os.path.splitext(f)[1].lower() in RAW_EXTENSIONS)
        else:
            files += sorted(glob.glob(item)) or [item]
    if not files:
        ap.error("no input files")

    model = args.model
    if not model:
        here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        model = os.path.join(here, MODEL_FILENAME)
        if not os.path.isfile(model):
            model = download_model(here)
    backend = {"torch-cpu": "torch (cpu)"}.get(args.backend, args.backend)
    device = None
    if backend in ("auto", "torch"):
        try:
            import torch
            if args.device:
                device = torch.device(args.device)
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        except ImportError:
            backend = "onnxruntime"
    models = load_models(model, backend=backend, device=device)
    mode = {"bayer": "bayer (joint demosaic)", "linear": "linear (after demosaic)"}.get(args.mode, "auto")

    for path in files:
        t0 = time.time()
        img = load_raw(path, demosaic=args.demosaic)

        def progress(done, total):
            print(f"\r  {os.path.basename(path)}: tile {done}/{total}", end="", flush=True)

        out = rd.denoise_raw(img, models, mode=mode, amount=args.amount, color_amount=args.color,
                             tile=args.tile, progress=progress, log=lambda m: print("\n" + m))
        folder = args.out or os.path.dirname(os.path.abspath(path))
        stem = os.path.splitext(os.path.basename(path))[0] + args.suffix
        if not args.no_dng:
            p = write_linear_dng(out, os.path.join(folder, stem + ".dng"))
            print(f"  -> {p}")
        if args.tiff:
            p = write_tiff16(develop(out), os.path.join(folder, stem + ".tif"))
            print(f"  -> {p}")
        print(f"  done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
