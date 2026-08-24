"""
config.py — Shared constants and tunable parameters for the cloud chamber
particle track classifier.
"""

import os

# =============================================================================
# Particle Track Classes
# =============================================================================
PARTICLE_LABELS = {
    0: "Alpha",
    1: "Proton",
    2: "Electron/Positron/Muon",
    3: "Low-Energy Electron",
    4: "Knock-On Electron",
}

# BGR colours for bounding boxes on the output video
PARTICLE_COLORS = {
    0: (0, 0, 255),      # Red for Alpha
    1: (255, 0, 0),      # Blue for Proton
    2: (0, 255, 0),      # Green for Electron/Positron/Muon
    3: (0, 255, 255),    # Yellow for Low-Energy Electron
    4: (255, 0, 255),    # Magenta for Knock-On Electron
}

# Short labels for display on video frames
PARTICLE_SHORT_LABELS = {
    0: "Alpha",
    1: "Proton",
    2: "e/e+/mu",
    3: "Low-E e-",
    4: "Knock-On e-",
}

# =============================================================================
# Paths
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "..", "Images")
MODEL_PATH = os.path.join(BASE_DIR, "trained_model.pkl")

# =============================================================================
# Preprocessing Parameters
# =============================================================================
# Gaussian blur kernel size (must be odd)
BLUR_KERNEL = (13, 13)

# CLAHE (Contrast Limited Adaptive Histogram Equalisation)
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID = (8, 8)

# Non-local means denoising
DENOISE_H = 30
DENOISE_TEMPLATE_WINDOW = 7
DENOISE_SEARCH_WINDOW = 21

# Global threshold parameter (used after heavy blur)
GLOBAL_THRESHOLD = 150

# Morphological kernel size for closing / opening
MORPH_KERNEL_SIZE = 3

# =============================================================================
# Track Detection Parameters
# =============================================================================
# Minimum contour area (pixels) to be considered a track (filters noise)
MIN_CONTOUR_AREA = 80

# Maximum contour area — filter out unreasonably large detections
MAX_CONTOUR_AREA = 50000

# Height of the text overlay region to mask out (top of frame)
TEXT_OVERLAY_HEIGHT = 90

# =============================================================================
# Red Box Detection (for extracting training annotations)
# =============================================================================
# HSV range for detecting the red annotation rectangles
# Red in HSV wraps around 0/180, so we use two ranges
RED_HSV_LOWER1 = (0, 80, 80)
RED_HSV_UPPER1 = (10, 255, 255)
RED_HSV_LOWER2 = (160, 80, 80)
RED_HSV_UPPER2 = (180, 255, 255)

# Minimum perimeter length for a detected red rectangle (filters small noise)
MIN_RED_BOX_PERIMETER = 200

# =============================================================================
# Feature Extraction
# =============================================================================
FEATURE_NAMES = [
    "skeleton_length",
    "bbox_aspect_ratio",
    "track_thickness",
    "straightness",
    "total_curvature",
    "max_curvature",
    "num_branch_points",
    "num_endpoints",
    "branch_ratio",
    "contour_area",
    "skeleton_area_ratio",
    "bbox_fill_ratio",
]

# =============================================================================
# Random Forest Classifier
# =============================================================================
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 12
RF_MIN_SAMPLES_SPLIT = 5
RF_MIN_SAMPLES_LEAF = 2
RF_RANDOM_STATE = 42

# =============================================================================
# Synthetic Data Augmentation
# =============================================================================
SYNTHETIC_SAMPLES_PER_CLASS = 200

# =============================================================================
# Video Processing
# =============================================================================
# Background subtractor history (number of frames)
BG_HISTORY = 120
# Background subtractor threshold
BG_THRESHOLD = 25
# Whether to detect shadows in background subtraction
BG_DETECT_SHADOWS = False
# Minimum number of consecutive frames a track must appear to be annotated
MIN_TRACK_PERSISTENCE = 1
# Output video codec
VIDEO_CODEC = "mp4v"

# =============================================================================
# Corrected Image-to-Label Mapping
# =============================================================================
# Each entry maps a substring of the image filename to a list of labels.
# When an image has multiple red boxes, boxes are ordered by area (largest
# first) or top-to-bottom, and each box gets the corresponding label from
# the list.
#
# Labels: 0=Alpha, 1=Proton, 2=Electron/Positron/Muon,
#         3=Low-Energy Electron, 4=Knock-On Electron

IMAGE_LABEL_MAP = {
    # === Video 1: VID_20220609_215401_A ===
    "215401_A.mp4_snapshot_00.14": [2],              # Electron/Positron/Muon
    "215401_A.mp4_snapshot_00.35": [2],              # Electron/Positron/Muon
    "215401_A.mp4_snapshot_01.20": [0, 2],           # Box1(large): Alpha tracks, Box2(small): e/e+/mu
    "215401_A.mp4_snapshot_01.25": [1],              # Proton
    "215401_A.mp4_snapshot_01.26": [2],              # Electron/Positron/Muon
    "215401_A.mp4_snapshot_01.34": [3],              # Low-energy electron
    "215401_A.mp4_snapshot_01.39": [3],              # Low-energy electron
    "215401_A.mp4_snapshot_01.44": [4],              # Knock-on electron

    # === Video 2: VID_20220609_221024_A ===
    "221024_A.mp4_snapshot_00.03_[2022.06.10_18.14.32]": [1, 0, 4],  # Proton, Alpha, Knock-on
    "221024_A.mp4_snapshot_00.09": [2],              # Electron/Positron/Muon
    "221024_A.mp4_snapshot_00.11": [1],              # Proton
    "221024_A.mp4_snapshot_00.23": [2],              # Electron/Positron/Muon
    "221024_A.mp4_snapshot_00.24": [2],              # Electron/Positron/Muon
    "221024_A.mp4_snapshot_00.29": [1],              # Proton
    "221024_A.mp4_snapshot_00.30": [1],              # Proton
    "221024_A.mp4_snapshot_00.33": [3],              # Low-energy electron
    "221024_A.mp4_snapshot_00.34_[2022.06.10_18.21.33]": [2, 4],     # e/e+/mu, Knock-on
    "221024_A.mp4_snapshot_00.35": [4],              # Knock-on (single forked track box)
    "221024_A.mp4_snapshot_00.38": [0, 0],           # Both Alpha
    "221024_A.mp4_snapshot_00.45_[2022.06.10_18.24.31]": [2],        # Electron/Positron/Muon
    "221024_A.mp4_snapshot_00.45_[2022.06.10_18.25.06]": [1],        # Proton
    "221024_A.mp4_snapshot_00.59_[2022.06.10_18.27.11]": [3],        # Low-energy electron
    "221024_A.mp4_snapshot_00.59_[2022.06.10_18.27.52]": [3],        # Low-energy electron
    "221024_A.mp4_snapshot_01.21_[2022.06.10_18.30.35]": [0, 3],     # Alpha + Low-energy electron
    "221024_A.mp4_snapshot_01.21_[2022.06.10_18.31.23]": [2],        # Electron/Positron/Muon
    "221024_A.mp4_snapshot_01.23": [0],              # Alpha
    "221024_A.mp4_snapshot_01.30": [0],              # Alpha (possibly)
    "221024_A.mp4_snapshot_02.13": [1],              # Proton
    "221024_A.mp4_snapshot_02.17": [1],              # Proton
    "221024_A.mp4_snapshot_02.19": [2],              # Electron/Positron/Muon
    "221024_A.mp4_snapshot_02.25": [1],              # Proton
    "221024_A.mp4_snapshot_02.26": [2],              # Electron/Positron/Muon
}
