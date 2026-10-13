# ComfyUI RAW AI Denoise

AI noise reduction inside ComfyUI for **camera RAW photos** (Lightroom-style, with linear
DNG output) and **video** (temporal, streamed, for long clips - see [Video](#video)).

```
Load RAW Image ──► RAW AI Denoise (RawNIND) ──► Save Linear DNG   (→ Lightroom / ACR / darktable)
      │                       └──► Develop RAW to Image ──► previews, 16-bit TIFF, rest of your workflow
      └── original preview ───────────────► Before/After 100% Crop

Load Video (for AI Denoise) ──► Video AI Denoise (FastDVDnet) ──► denoised video file (H.265 / ProRes / …)
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
2. Install the dependencies (rawpy and tifffile for photos, imageio-ffmpeg for video) **with ComfyUI's Python**:
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

**📁 Browse this computer (no copy)…** opens a folder browser that reads files where they are
(it fills in `path_override`; only available when the browser runs on the ComfyUI machine).
`path_override` also takes a folder: `folder_index` + the `increment` control batch-processes it.

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
* **Video: "ffmpeg not found"** - install `imageio-ffmpeg` with ComfyUI's Python (as above), or
  point the environment variable `RAW_DENOISE_FFMPEG` at your own ffmpeg.
* **Video: "Browse this computer" says it is not available** - the folder browser only answers
  requests from the ComfyUI machine itself. When you use ComfyUI from another computer, upload
  instead (or set `RAW_DENOISE_ALLOW_REMOTE_BROWSE=1` if you trust your network).
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
| **Load Video (for AI Denoise)** | **Browse this computer…** (no copy), **Upload video(s)…**, drag & drop, **Render all selected**. Probes the file (resolution, frame rate, bit depth, chroma, colour tags, rotation, audio, timecode) and outputs a `VIDEO_SOURCE` reference plus one preview frame. It never loads the whole video. |
| **Video AI Denoise (FastDVDnet)** | Preview (short before/after clip on the node) or full render to a file - see [Video](#video). |

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

## Video

Two nodes, **Load Video (for AI Denoise)** and **Video AI Denoise (FastDVDnet)**, in
`image/RAW Denoise/video`. Example workflow: `workflows/Video_AI_Denoise.json`.

### Why not the usual Load Video → denoise → Video Combine workflow

ComfyUI passes video between nodes as one `IMAGE` batch of float32 frames, all in memory at
once. One minute of 4K at 25 fps is 1,500 × 3840 × 2160 × 3 × 4 bytes ≈ **150 GB**, and image
denoisers then try to push that batch through the GPU. That is why long clips run out of RAM
or VRAM. These nodes do it differently:

* **Streaming.** ffmpeg decodes frame by frame, each frame is denoised as soon as its
  neighbours are in, and goes straight to the encoder. Only a handful of frames are ever in
  memory, so RAM and VRAM use don't depend on the length (tested: a 5-minute, 9,000-frame clip
  ran at a flat ~210 MB of RAM from the first frame to the last).
* **VRAM-aware tiling.** The tile size is picked from the free VRAM. If memory still runs
  out, the tile is halved and the frame retried automatically. GPUs run in fp16 (half the
  memory) after a self-check against a reference implementation. Other models ComfyUI keeps
  on the GPU are unloaded first.
* **Resumable renders.** The output is written in 10-second segments (in
  `.video_denoise_work/` next to the output). If a render stops (Cancel, crash, power cut),
  queue it again with the same settings and it continues from the last finished segment. At
  the end the segments are joined without re-encoding and the audio is added.

### The model

**FastDVDnet** (Tassano, Delon & Veit, CVPR 2020) looks at 5 consecutive frames. Noise
changes from frame to frame and detail doesn't, so a temporal network can remove much more
noise than a single-image denoiser without smearing detail or flickering. It needs no
optical flow. The weights (MIT licence, 10 MB) are bundled, so nothing is downloaded.

FastDVDnet was trained on synthetic Gaussian noise, and real camera noise looks different:
it varies with brightness and is coarser (demosaicing, in-camera NR, compression). So the
node first **measures the noise of your clip** in a few short sections, per brightness band.
It then sets the network's noise level from those measurements, using a mapping calibrated
on simulated low-light footage with known ground truth. Where the local detail is clearly
stronger than the noise, it lowers the level so texture isn't smoothed away.
On 8 simulated low-light clips that weren't used for the calibration, the automatic setting
averaged 3.1 dB PSNR better than the noisy input. It was also 0.5 dB better than the best
single fixed level, picked with the ground truth in hand.

Colour is handled in the video's own YCbCr values at full precision, with an exact
BT.709 / BT.601 / BT.2020 conversion and nothing clipped. Low-light chroma noise around black
doesn't lift or tint the shadows, and super-whites (Sony picture profiles record up to 109 %)
survive.

### Use

1. Drag `workflows/Video_AI_Denoise.json` onto the canvas.
2. On **Load Video** click **📁 Browse this computer…** to pick a file where it is (no copy,
   so any size works; only available when the browser runs on the ComfyUI machine). You can
   also use **⬆ Upload video(s) from this browser…** or drag a file onto the node, which copies
   it to `ComfyUI/input/video/`.
3. **Run** with `mode` = *preview (short section)*. The first run measures the noise (cached
   for the session). It then denoises `preview_seconds` at `preview_at` and plays an
   *original | denoised* 100 % crop clip on the node; the Before/After node shows the middle
   frame.
4. Tune `strength` (1.0 = the measured level; 1.3-1.6 for very noisy low-light footage, 0.7
   to keep more texture) and `grain` (puts back some fine luminance grain, which often looks
   more natural than perfectly clean video).
5. Set `mode` = *render full video* and **Run**. Progress is shown on the node, with frame
   rate and ETA in the console. If you selected several files, **▶ Render all N selected videos**
   queues them all.

| Setting | |
|---|---|
| `strength` | multiplies the measured noise level (default 1.0) |
| `noise_level` | 0 = measure automatically; 1-60 = one fixed level (0-255 scale) for all brightnesses |
| `grain` | 0-100 % of the removed luminance noise put back as grain (colour noise stays removed) |
| `codec` | **H.265 10-bit** (default; small; keeps 4:2:2 if the source has it), **ProRes 422 HQ** (editing), **H.264 8-bit** (plays everywhere), **FFV1** (lossless archive) |
| `quality` | standard / high / very high (CRF 18/14/10 for H.265) |
| `output_folder` | inside `ComfyUI/output` (default `video_denoise`), an absolute path, or empty = next to the original |
| `backend`, `precision`, `tile_size` | auto is right for almost everyone; lower `tile_size` only if you want to leave VRAM for other apps |

### Camera notes

* **Sony a7 IV.** XAVC S, XAVC HS and XAVC S-I (H.264/H.265, 4:2:0/4:2:2, 8/10-bit) all read.
  The default H.265 10-bit output keeps 4:2:2 10-bit. The LPCM audio is copied untouched;
  because MP4 can't hold LPCM, the output is a `.mov`. Timecode and colour tags (BT.709, HLG,
  BT.2020) are carried over. S-Log3 footage can be denoised as shot; do it before grading.
  Sony's metadata track (lens and gyro data for Catalyst Browse or Gyroflow) is not copied, so
  if you stabilise, load the motion data from the original clip.
* **Phones.** Rotation is applied, so portrait clips come out upright. HDR tags (HLG, HDR10)
  are kept. Dolby Vision dynamic metadata is dropped, so iPhone Dolby Vision clips play as
  plain HLG. Variable-frame-rate clips are written at their average frame rate; audio stays in
  sync.
* Interlaced footage isn't deinterlaced; deinterlace it first.

### Speed

Rough estimates; I could only time this on a 2-core machine without a GPU. The network costs
about 5 TFLOP per 4K frame. On a current NVIDIA GPU in fp16 that works out to very roughly
2-10 frames per second at 4K, and about 4× that at 1080p. At 4K the H.265 encoder can be the
bottleneck; ProRes encodes faster. CPU-only works but is only practical for short or
low-resolution clips.

### Honest expectations

FastDVDnet is a compact research model from 2020. Commercial tools such as Topaz Video AI,
Neat Video or DaVinci Resolve Studio's temporal NR use bigger, motion-compensated models
trained on real camera noise. On moderate low-light noise this pack gives clean, stable,
flicker-free results. On extreme noise, expect those tools to keep more fine texture. I have
not compared them on the same footage. Use the preview to tune before a long render.

## Command line

The same engine without ComfyUI (needs numpy, rawpy, tifffile and torch or onnxruntime):

```
python raw_denoise_cli.py IMG_1234.CR3 --amount 50 --out denoised/
python raw_denoise_cli.py "D:/shoot/*.NEF" --tiff
```

## Tests

`python tests/test_video.py` checks the bundled FastDVDnet weights load without PyTorch and
match onnxruntime, that the streaming engine reproduces the reference FastDVDnet loop exactly
(incl. clip ends), the colour conversion (exact code-value round trips, nothing clipped),
probing of a7 IV / phone / full-range / untagged files, a decode → encode → decode round trip,
and that an interrupted and resumed render is bit-identical to an uninterrupted one.

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
* FastDVDnet: Matias Tassano, Julie Delon & Thomas Veit, *FastDVDnet: Towards Real-Time Deep
  Video Denoising Without Flow Estimation*, CVPR 2020. Code and weights MIT licence
  (github.com/m-tassano/fastdvdnet); weights bundled in `weights/fastdvdnet/` with their licence.
* RAW decoding: LibRaw via rawpy. Video decoding/encoding: FFmpeg (via imageio-ffmpeg).
* This node pack: GPL-3.0 (same as ComfyUI).
