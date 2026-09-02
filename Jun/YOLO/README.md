# YOLO Cloud-Chamber Particle-Track Detector

A YOLO11-based object detector that locates and classifies particle tracks in
cloud-chamber footage into five classes:

| id | class | short |
|----|-------|-------|
| 0 | Alpha | Alpha |
| 1 | Proton | Proton |
| 2 | Electron/Positron/Muon | e/e+/mu |
| 3 | Low-Energy Electron | Low-E e- |
| 4 | Knock-On Electron | Knock-On e- |

This is an alternative to the `Skeletonisation + RandomForest` approach in the
sibling folder. Instead of hand-crafted features or skeleton pixels, it learns
to detect tracks directly with a convolutional detector.

## How the dataset is built

The source images in `../Images/` are annotated with **red rectangles** around
each labelled track. `prepare_dataset.py`:

1. Detects the red boxes (HSV colour filtering, sorted largest-area first).
2. Maps each box to a class using `config.IMAGE_LABEL_MAP` (same convention as
   the RandomForest project).
3. Writes normalised YOLO labels (`cls cx cy w h`).
4. **Inpaints the red rectangles away** so the model learns the track itself,
   not the annotation lines (which don't exist in the inference video).
5. Produces a reproducible train/val split and `dataset/data.yaml`.

## Files

| file | purpose |
|------|---------|
| `config.py` | classes, paths, red-box params, hyper-parameters, label map |
| `preprocessing.py` | shared morphology-based frame enhancement (dataset build + inference) |
| `prepare_dataset.py` | red-box images → YOLO dataset (inpaint + enhance) |
| `train.py` | fine-tune YOLO11n on the dataset |
| `inference.py` | detect on images / annotate a video (morphology + TTA + tiling) |
| `extract_frames.py` | sample frames from the videos to expand the dataset |
| `cloud_chamber_yolo.ipynb` | end-to-end notebook (prep → train → infer → improve) |
| `requirements.txt` | Python dependencies |

## Running

Use the `rocm_env` conda environment (has a ROCm torch build + ultralytics):

```powershell
& "C:\Users\PC\anaconda3\envs\rocm_env\python.exe" prepare_dataset.py
& "C:\Users\PC\anaconda3\envs\rocm_env\python.exe" train.py --device cpu
& "C:\Users\PC\anaconda3\envs\rocm_env\python.exe" inference.py
```

`train.py` accepts `--device`, `--epochs`, `--imgsz`, `--batch` and `--model`,
e.g. a quicker run: `train.py --device cpu --epochs 60 --imgsz 640`.

Or open `cloud_chamber_yolo.ipynb` and select the `rocm_env` kernel.

## Notes & caveats

- **Small dataset.** There are only ~32 annotated images. Heavy augmentation
  and a nano model (`yolo11n`) are used to reduce overfitting, but detection
  quality is fundamentally limited by the amount of labelled data. Treat the
  metrics as indicative, not production-grade.
- **Training runs on CPU on this machine.** The AMD Radeon RX 9060 XT (gfx1200 /
  RDNA4) is detected by the ROCm torch build and inference-style GPU ops work,
  but *training* crashes when MIOpen JIT-compiles the batch-norm kernel
  (`fatal error: 'type_traits' file not found` → `miopenStatusUnknownError`).
  This is a known limitation of the Windows ROCm 7.2 preview for RDNA4. `train.py`
  therefore supports `--device cpu` and also **auto-falls back to CPU** if it
  detects a ROCm/MIOpen kernel error. Once ROCm ships fixed MIOpen artefacts,
  `--device 0` should work again.
- Class ids match the RandomForest project so results are comparable.

## Improving detection / recall

Whole-frame, high-confidence inference misses most of the faint tracks on the
video. Two kinds of fix are provided:

**Inference-time (no retraining, all toggled in `config.py`):**

- **Morphology enhancement** (`preprocessing.py`) — median denoising, white
  top-hat background subtraction, thresholding, opening, closing, and connected
  component filtering are applied identically to training and inference frames.
- **Low confidence** (`CONF_THRESHOLD = 0.10`) — favours recall.
- **Test-time augmentation** (`USE_TTA`) — multi-scale + flips.
- **Tiled / SAHI-style inference** (`USE_TILING`, `TILE_ROWS/COLS/OVERLAP`) —
  slices each 1080×1920 frame into overlapping tiles, detects per tile, and
  merges with class-wise NMS. Biggest win for small tracks (slow on CPU).

**The real fix — more/denser data:**

The dataset is tiny (~32 images, ~7 boxes/class) and, crucially, most visible
tracks in each frame are *not* labelled — so the model is taught to treat faint
tracks as background. Use `extract_frames.py` to sample more frames, label
**every** visible track, then rebuild (`prepare_dataset.py`) and retrain. Because
the morphology enhancement is baked into the dataset build, retraining keeps the
same preprocessing between training and inference.

`config.py` training defaults were also raised for the retrain
(`EPOCHS = 300`, `PATIENCE = 100`).
