# ComfyUI RAW AI Denoise

Lightroom-style AI noise reduction for camera RAW files, inside ComfyUI.

```
Load RAW Image ──► RAW AI Denoise (RawNIND) ──► Save Linear DNG   (→ Lightroom / ACR / darktable)
      │                       └──► Develop RAW to Image ──► previews, 16-bit TIFF, rest of your workflow
      └── original preview ───────────────► Before/After 100% Crop
```

## How it compares to Lightroom's Denoise

Lightroom's *Denoise* is a neural network that runs on the **raw sensor mosaic, before
demosaicing**, does noise reduction and demosaicing in one pass, and writes a new
**linear DNG** (`…-Enhanced-NR.dng`) that you keep editing like a raw file. Generic
ComfyUI denoisers (SCUNet, NAFNet, …) only see an already-rendered 8-bit sRGB image,
which is a much weaker starting point.

This pack reproduces Lightroom's approach with open components:

| | Lightroom Denoise | this node pack |
|---|---|---|
| Works on | raw Bayer / X-Trans mosaic | raw Bayer mosaic (joint denoise + demosaic). X-Trans & others: LibRaw demosaic, then a linear-RGB model |
| Network | Adobe proprietary | **RawNIND UtNet2** (Brummer & De Vleeschouwer 2025, 7.8 M params), trained on real raw noise/clean pairs; the same model darktable 5.6 uses |
| Strength | Amount 0–100 (default 50) | Amount 0–100 (default 50) + separate colour-noise control |
| Output | linear DNG, still fully editable | linear DNG (camera RGB + colour matrix + as-shot WB + ISO/shutter/aperture), or 16-bit TIFF / ComfyUI IMAGE |

Honest expectations: Adobe's network is larger and trained on far more cameras, and it
handles X-Trans natively. At very high ISO, expect Lightroom to keep somewhat more fine
texture; for most Bayer files the result here is in the same league (see the sample
crops). I have not compared against Lightroom on the same files.

## Install

1. Put this folder in `ComfyUI/custom_nodes/` (so you have `ComfyUI/custom_nodes/comfyui-raw-denoise/__init__.py`).
2. Install the two dependencies **with ComfyUI's Python**:
   * regular install / venv: `python -m pip install -r ComfyUI/custom_nodes/comfyui-raw-denoise/requirements.txt`
   * Windows portable: `python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\comfyui-raw-denoise\requirements.txt`
3. Restart ComfyUI (fully - the node pack adds a small JavaScript file and an upload route),
   reload the browser tab and drag `workflows/RAW_AI_Denoise.json` onto the canvas.
4. The model (`rawdenoise-nind.dtmodel`, 58 MB) downloads automatically on the first run into
   `ComfyUI/models/raw_denoise/` (SHA-256 checked). For offline machines, download it from
   <https://github.com/darktable-org/darktable-ai/releases/tag/release-5.6.0> and drop it there.

Nothing else is needed: inference runs through ComfyUI's own PyTorch on whatever GPU it uses
(CUDA, ROCm, Apple MPS, Intel XPU) or CPU. `onnxruntime` is optional (backend = onnxruntime).

## Use

1. On **Load RAW Image** click **📂 Choose RAW file(s)…** and pick your files in the normal
   file dialog (ARW, CR2, CR3, NEF, RAF, ORF, RW2, DNG… - Ctrl/Shift-click for several).
   You can also drag RAW files from Explorer/Finder onto that node. Browsers don't reveal file
   paths, so the files are copied into `ComfyUI/input/raw/` (identical files aren't copied twice;
   there is no 100 MB limit, unlike ComfyUI's image upload).
2. Press **Run**. The first run downloads the model.
3. Picked several files? **Run** processes the one shown in `raw_file`;
   **▶ Queue all N selected files** queues every one of them.
4. Look at **Before/After (100% crop)**; move `center_x` / `center_y` to inspect other areas
   (ComfyUI's normal previews are downscaled and hide noise).
5. Tune **Amount** (and **color_noise**). Only the blend is recomputed, so this is instant.
6. The DNG is written to `ComfyUI/output/raw_denoise/<name>-Enhanced-NR.dng`
   (clear `output_folder` to write it next to the original). Import it into Lightroom/ACR/
   darktable/RawTherapee/Capture One and edit as usual.

Power users: `path_override` takes an absolute file path or a folder (no copying); with a
folder, `folder_index` + the `increment` control batch-processes it.

## Troubleshooting

* **"The 'rawpy' package … is not installed"** - the most common problem. It must be installed
  into the Python that runs ComfyUI, which is usually *not* your system Python. The error (and
  the console at start-up) prints the exact command for your install, e.g. for Windows portable
  `"C:\ComfyUI_windows_portable\python_embeded\python.exe" -m pip install rawpy tifffile`.
  Restart ComfyUI afterwards. (rawpy has no wheel for Intel Macs.)
* **No "Choose RAW file(s)…" button** - restart ComfyUI completely and reload the browser tab
  (Ctrl+F5) so the node's JavaScript is loaded; check the folder is
  `custom_nodes/comfyui-raw-denoise/` with `__init__.py` directly inside it.
* **"LibRaw … can't decode …"** - the file is from a camera newer than the bundled LibRaw, or
  it isn't a raw file. `pip install -U rawpy` (same Python as above).
* Any other error: copy the text from the error dialog or the ComfyUI console.

## Nodes

| Node | What it does |
|---|---|
| **Load RAW Image** | **Choose RAW file(s)…** button (native file dialog, multi-select), drag & drop, **Queue all selected**. Reads any LibRaw-supported raw (ARW, CR2, CR3, NEF, RAF, ORF, RW2, PEF, DNG, …) as scene-linear camera RGB and keeps the Bayer mosaic for the network. Outputs `RAW_IMAGE`, a preview `IMAGE` and an info string (camera, ISO, sensor). |
| **RAW AI Denoise (RawNIND)** | `amount` 0–100 (Lightroom-like, default 50; 0 = original, 50 keeps 25 % of the noise amplitude as fine grain, 100 = full network output), `color_noise` 0–100 (default 100 = remove all colour noise), `mode` auto / bayer / linear, `backend` auto / torch / torch (cpu) / onnxruntime, `tile_size` (larger = faster on big GPUs, lower if VRAM runs out). |
| **Develop RAW to Image** | White balance + camera colour matrix + exposure + gentle tone curve → sRGB `IMAGE` for previews or further ComfyUI work. `auto` exposure is computed from the original, so before/after match. |
| **Save Linear DNG** | The Lightroom-equivalent output. Pixels are stored upright, 16-bit. |
| **Save Image (16-bit TIFF)** | 16-bit sRGB TIFF with ICC profile (bypassed in the example workflow; Ctrl+B to enable). |
| **Before/After 100% Crop** | Side-by-side crops at 100–400 %. |

## Details that matter

* **Highlights.** Areas that were clipped in the raw keep the original values at exactly the
  white level (plus a 5-pixel feathered margin). Without this, the network smears clipped
  photosites outward and Lightroom shows magenta fringes around specular highlights.
* **Brightness.** The network's output scale is arbitrary by design; it is re-matched to
  the input by comparing the re-mosaiced output with the raw photosites (like for like,
  so the image isn't brightened by the 2-green imbalance).
* **Shadows.** Values below black are kept (DNG black level 256), so mixing grain back in
  doesn't lift the blacks.
* **Lightroom specifics.** The DNG carries the camera's colour matrix and an Adobe-style
  camera name, so Lightroom normally offers its camera profiles (Adobe Color etc.). Lens
  information is not copied, so pick the lens profile manually if you use one. Lightroom
  applies a per-camera baseline exposure to proprietary raws; if the DNG looks a touch darker
  than the original, that is why (adjust Exposure).
* **Memory.** Roughly 150 bytes per pixel of RAM at peak (≈ 4–5 GB for 32 MP).

## Command line

The same engine without ComfyUI (needs numpy, rawpy, tifffile and torch or onnxruntime):

```
python raw_denoise_cli.py IMG_1234.CR3 --amount 50 --out denoised/
python raw_denoise_cli.py "D:/shoot/*.NEF" --tiff
```

## Tests

`python tests/test_core.py [path/to/rawdenoise-nind.dtmodel]` checks the upload helpers
(name sanitising, duplicate detection), the ONNX reader against
onnxruntime, Bayer geometry for all four CFA layouts (RGGB/BGGR/GRBG/GBRG via synthetic DNGs
read by LibRaw), gain matching, tiling seams, Amount blending, highlight protection and the
DNG round trip. On first use, the PyTorch backend also verifies itself against a NumPy
reference implementation and falls back to onnxruntime (if installed) when `backend = auto`.

## Credits & licences

* RawNIND model: Benoit Brummer & Christophe De Vleeschouwer (UCLouvain), *Learning Joint
  Denoising, Demosaicing, and Compression from the Raw Natural Image Noise Dataset*,
  arXiv:2501.08924. Weights GPL-3.0 / CC-BY-4.0. ONNX export by the darktable project
  (darktable-org/darktable-ai, GPL-3.0).
* RAW decoding: LibRaw via rawpy.
* This node pack: GPL-3.0 (same as ComfyUI).
