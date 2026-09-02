"""
config.py — Shared constants for the YOLO cloud-chamber particle-track detector.

This module owns everything the YOLO pipeline needs to be reproducible:
class definitions, filesystem layout, red-box detection parameters (used to
convert the hand-annotated images into YOLO labels), and training/inference
hyper-parameters.

The class list is kept identical to the Skeletonisation + RandomForest project
so both approaches classify the exact same particle types.
"""

import os

# =============================================================================
# Particle Track Classes  (index == YOLO class id)
# =============================================================================
CLASS_NAMES = [
    "Alpha",                    # 0
    "Proton",                   # 1
    "Electron_Positron_Muon",   # 2
    "Low_Energy_Electron",      # 3
    "Knock_On_Electron",        # 4
]
NUM_CLASSES = len(CLASS_NAMES)

# Human-friendly labels for plotting / overlays (index-aligned with CLASS_NAMES)
DISPLAY_NAMES = {
    0: "Alpha",
    1: "Proton",
    2: "e/e+/mu",
    3: "Low-E e-",
    4: "Knock-On e-",
}

# BGR colours for drawing boxes on frames (index-aligned with CLASS_NAMES)
CLASS_COLORS = {
    0: (0, 0, 255),      # Red    - Alpha
    1: (255, 0, 0),      # Blue   - Proton
    2: (0, 255, 0),      # Green  - Electron/Positron/Muon
    3: (0, 255, 255),    # Yellow - Low-Energy Electron
    4: (255, 0, 255),    # Magenta- Knock-On Electron
}

# =============================================================================
# Paths
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../Jun/YOLO
JUN_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))        # .../Jun
IMAGES_DIR = os.path.join(JUN_DIR, "Images")                   # annotated source images
VIDEOS_DIR = os.path.join(JUN_DIR, "Videos")

# Where the auto-generated YOLO dataset (images/labels + data.yaml) is written
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")

# Training run outputs
RUNS_DIR = os.path.join(BASE_DIR, "runs")
PROJECT_NAME = "cloud_chamber"          # runs/<PROJECT_NAME>/<RUN_NAME>
RUN_NAME = "yolo11n_tracks"

# Convenience path to the best trained weights (populated after training)
BEST_WEIGHTS = os.path.join(RUNS_DIR, PROJECT_NAME, RUN_NAME, "weights", "best.pt")

# Default video I/O for inference
VIDEO_INPUT = os.path.join(VIDEOS_DIR, "test_vid_cropped.mp4")
VIDEO_OUTPUT = os.path.join(BASE_DIR, "output_yolo.mp4")

# =============================================================================
# Red-Box Annotation Detection
# =============================================================================
# The source images carry red rectangles around each labelled track.
# Red wraps around 0/180 in HSV, so two ranges are combined.
RED_HSV_LOWER1 = (0, 80, 80)
RED_HSV_UPPER1 = (10, 255, 255)
RED_HSV_LOWER2 = (160, 80, 80)
RED_HSV_UPPER2 = (180, 255, 255)

# Minimum box side length (px) to reject stray red specks
MIN_BOX_SIDE = 30

# Pixels shaved off each side of a detected box so the red line itself is
# not baked into the training crop / label.
BOX_INSET = 6

# =============================================================================
# Frame Pre-processing  (shared by dataset build AND inference)
# =============================================================================
# Cloud-chamber tracks are faint, low-contrast wisps. CLAHE (Contrast Limited
# Adaptive Histogram Equalisation) on the luminance channel makes them pop.
# The SAME enhancement must be applied to the training images and to the
# inference frames, otherwise the model sees a different domain at test time.
USE_CLAHE = True
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID = (8, 8)

# =============================================================================
# Dataset Split
# =============================================================================
# Fraction of images held out for validation. With only ~32 images the split
# is small; a fixed seed keeps it reproducible.
VAL_FRACTION = 0.2
SPLIT_SEED = 42

# =============================================================================
# Training Hyper-parameters
# =============================================================================
BASE_MODEL = "yolo11n.pt"     # nano — best fit for a tiny dataset
EPOCHS = 300
IMG_SIZE = 960                # source frames are tall (1080x1920); keep detail
BATCH = 8
PATIENCE = 100                # early-stop patience (raised: tiny dataset is noisy)
DEVICE = 0                    # ROCm GPU (AMD Radeon RX 9060 XT); use "cpu" to force CPU
SEED = 42

# =============================================================================
# Inference
# =============================================================================
# Low confidence favours recall (catching faint tracks). Detections can be
# filtered higher afterwards; missed tracks can never be recovered.
CONF_THRESHOLD = 0.10
IOU_THRESHOLD = 0.45
VIDEO_CODEC = "mp4v"

# Test-time augmentation (multi-scale + flips) — boosts recall at some speed cost.
USE_TTA = True

# ---- Tiled (SAHI-style) inference --------------------------------------------
# Tracks are small relative to a 1080x1920 frame. Slicing the frame into
# overlapping tiles and detecting in each recovers many small/faint tracks that
# whole-frame inference misses. Detections are merged back with class-wise NMS.
USE_TILING = True
TILE_ROWS = 2                 # vertical slices
TILE_COLS = 2                 # horizontal slices
TILE_OVERLAP = 0.2            # fractional overlap between adjacent tiles
TILE_CONF_THRESHOLD = 0.10    # per-tile confidence (kept low for recall)
DUPLICATE_IOU_THRESHOLD = 0.70  # suppress near-identical boxes across classes

# =============================================================================
# Image-to-Label Mapping
# =============================================================================
# Maps a substring of each image filename to the class ids of its red boxes.
# Boxes are matched to this list AFTER sorting detected boxes by area
# (largest first) — identical ordering convention to the RandomForest project.
#
# 0=Alpha, 1=Proton, 2=Electron/Positron/Muon,
# 3=Low-Energy Electron, 4=Knock-On Electron
IMAGE_LABEL_MAP = {
    # === Video 1: VID_20220609_215401_A ===
    "215401_A.mp4_snapshot_00.14": [2],
    "215401_A.mp4_snapshot_00.35": [2],
    "215401_A.mp4_snapshot_01.20": [0, 2],
    "215401_A.mp4_snapshot_01.25": [1],
    "215401_A.mp4_snapshot_01.26": [2],
    "215401_A.mp4_snapshot_01.34": [3],
    "215401_A.mp4_snapshot_01.39": [3],
    "215401_A.mp4_snapshot_01.44": [4],

    # === Video 2: VID_20220609_221024_A ===
    "221024_A.mp4_snapshot_00.03_[2022.06.10_18.14.32]": [1, 0, 4],
    "221024_A.mp4_snapshot_00.09": [2],
    "221024_A.mp4_snapshot_00.11": [1],
    "221024_A.mp4_snapshot_00.23": [2],
    "221024_A.mp4_snapshot_00.24": [2],
    "221024_A.mp4_snapshot_00.29": [1],
    "221024_A.mp4_snapshot_00.30": [1],
    "221024_A.mp4_snapshot_00.33": [3],
    "221024_A.mp4_snapshot_00.34_[2022.06.10_18.21.33]": [2, 4],
    "221024_A.mp4_snapshot_00.35": [4],
    "221024_A.mp4_snapshot_00.38": [0, 0],
    "221024_A.mp4_snapshot_00.45_[2022.06.10_18.24.31]": [2],
    "221024_A.mp4_snapshot_00.45_[2022.06.10_18.25.06]": [1],
    "221024_A.mp4_snapshot_00.59_[2022.06.10_18.27.11]": [3],
    "221024_A.mp4_snapshot_00.59_[2022.06.10_18.27.52]": [3],
    "221024_A.mp4_snapshot_01.21_[2022.06.10_18.30.35]": [0, 3],
    "221024_A.mp4_snapshot_01.21_[2022.06.10_18.31.23]": [2],
    "221024_A.mp4_snapshot_01.23": [0],
    "221024_A.mp4_snapshot_01.30": [0],
    "221024_A.mp4_snapshot_02.13": [1],
    "221024_A.mp4_snapshot_02.17": [1],
    "221024_A.mp4_snapshot_02.19": [2],
    "221024_A.mp4_snapshot_02.25": [1],
    "221024_A.mp4_snapshot_02.26": [2],
}
