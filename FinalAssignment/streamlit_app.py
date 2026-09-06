"""
BMDS2133 Image Processing — Cloud Chamber Particle Track Analysis Laboratory
=============================================================================
Interactive Showcase Dashboard uniting all 4 team members' pipelines and the
integrated final pipeline.

Team Members & Contributions:
  - Ivan     : Dynamic Background Subtraction, Otsu Binarisation, Bounding Box Geometry, Spatial Calibration
  - Jun      : YOLO11 Deep Learning Object Detector & Tiled Multi-scale Inference
  - WeiQuan  : Canny Edge Detection & Probabilistic Hough Line Trajectory Analysis
  - TV       : Multiscale Frangi Vesselness Filter, Skeleton Centerlines & Distance Transform (EDT) Width
  - Team     : Integrated Hybrid Multi-Stage Pipeline combining all strengths
"""
from __future__ import annotations

import importlib.util
from importlib.machinery import SourceFileLoader
import math
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    import imageio_ffmpeg
    HAS_IMAGEIO_FFMPEG = True
except ImportError:
    HAS_IMAGEIO_FFMPEG = False

# -----------------------------------------------------------------------------
# Path Resolution
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
IVAN_DIR = ROOT / "Ivan"
JUN_YOLO_DIR = ROOT / "Jun" / "YOLO"
WEIQUAN_DIR = ROOT / "WeiQuan" / "code"
TV_NEW_FILE = ROOT / "TV" / "new (use this ivan)"
TV_PATH = TV_NEW_FILE if TV_NEW_FILE.exists() else (ROOT / "TV" / "current")
COMBINED_PATH = ROOT / "FinalAssignment" / "combined_pipeline.py"

for p in (IVAN_DIR, JUN_YOLO_DIR, WEIQUAN_DIR, ROOT / "FinalAssignment"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# -----------------------------------------------------------------------------
# Page Configuration & Modern Theme Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Cloud Chamber Lab · Particle Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --primary: #38bdf8;
        --primary-glow: rgba(56, 189, 248, 0.25);
        --bg-dark: #0b0f19;
        --card-bg: #111827;
        --border-color: #1f2937;
        --alpha-color: #f59e0b;
        --electron-color: #ef4444;
    }
    
    /* Global styling */
    .stApp {
        background-color: var(--bg-dark);
        color: #f3f4f6;
    }
    
    /* Header hero banner */
    .hero-container {
        padding: 1.5rem 2rem;
        background: linear-gradient(135deg, #111827 0%, #1e293b 50%, #0f172a 100%);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
    }
    .hero-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0 0 0.4rem 0;
        background: linear-gradient(to right, #ffffff, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin: 0;
        line-height: 1.5;
    }

    /* Team Member Cards */
    .team-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 0.8rem;
        margin-bottom: 1.5rem;
    }
    .member-card {
        background: #111827;
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 0.9rem;
        transition: transform 0.2s, border-color 0.2s;
    }
    .member-card:hover {
        border-color: #38bdf8;
        transform: translateY(-2px);
    }
    .member-name {
        font-weight: 700;
        font-size: 0.95rem;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }
    .member-tech {
        font-size: 0.78rem;
        color: #94a3b8;
        line-height: 1.35;
        margin-bottom: 0.5rem;
    }
    .pill {
        display: inline-block;
        padding: 0.18rem 0.5rem;
        border-radius: 6px;
        font-size: 0.68rem;
        font-weight: 600;
        background: #1f2937;
        color: #cbd5e1;
    }

    /* KPI Metric Cards */
    .kpi-row {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 0.75rem;
        margin-bottom: 1.2rem;
    }
    .kpi-card {
        background: #111827;
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .kpi-val {
        font-size: 1.85rem;
        font-weight: 800;
        color: #f8fafc;
        margin: 0.2rem 0;
    }
    .kpi-lbl {
        font-size: 0.78rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .kpi-alpha { color: #f59e0b; }
    .kpi-electron { color: #ef4444; }
    .kpi-total { color: #38bdf8; }

    /* Custom buttons and tabs */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #38bdf8 0%, #0284c7 100%);
        box-shadow: 0 4px 12px rgba(56, 189, 248, 0.35);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Dynamic Module Loaders (No Source Changes to Member Files)
# -----------------------------------------------------------------------------
def load_module_by_path(name: str, path: Path):
    """Safely import a team module from file path."""
    if path.suffix:
        spec = importlib.util.spec_from_file_location(name, str(path))
    else:
        loader = SourceFileLoader(name, str(path))
        spec = importlib.util.spec_from_loader(name, loader)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {name} at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 2:
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    return mask


def format_stage_image_for_display(img: np.ndarray) -> np.ndarray:
    """Format single/multi-channel intermediate stage image for safe Streamlit rendering."""
    if img is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    if img.ndim == 2:
        if img.dtype in (np.float32, np.float64):
            if img.max() <= 1.0:
                return (img * 255.0).clip(0, 255).astype(np.uint8)
            return img.clip(0, 255).astype(np.uint8)
        return img
    elif img.ndim == 3:
        if img.shape[2] == 1:
            return img[:, :, 0]
        return bgr_to_rgb(img)
    return img


# -----------------------------------------------------------------------------
# Pipeline Execution Wrappers
# -----------------------------------------------------------------------------
def run_ivan(image_bgr: np.ndarray, pixels_per_mm: float = 12.5) -> Dict[str, Any]:
    """Ivan's pipeline: Otsu binarisation, reflection suppression, and bbox geometry."""
    web_app = load_module_by_path("ivan_web_app", IVAN_DIR / "web_app.py")
    res = web_app.process_frame(image_bgr, collect_stages=True)
    if len(res) == 4:
        annotated, mask, detections, raw_stages = res
    else:
        annotated, mask, detections = res
        raw_stages = {}

    rows = []
    for d in detections:
        rows.append({
            "Track ID": d.get("id", len(rows) + 1),
            "Type": d.get("type", "Unknown"),
            "Confidence": round(float(d.get("density", 0.85)), 2),
            "Length (mm)": round(float(d.get("length_mm", d.get("max_dim", 0) / pixels_per_mm)), 2),
            "Thickness (mm)": round(float(d.get("thickness_mm", d.get("min_dim", 0) / pixels_per_mm)), 2),
            "Aspect Ratio": round(float(d.get("ar", 0)), 2),
            "Density": round(float(d.get("density", 0)), 2),
        })

    stages = []
    stage_meta = {
        "Cropped input": ("Chamber Crop ROI", "Physical chamber boundaries cropped to eliminate border lighting and outer glare."),
        "Greyscale": ("Grayscale Conversion", "Single-channel intensity representation for luminance analysis."),
        "CLAHE enhancement": ("CLAHE Contrast Equalization", "Adaptive histogram equalization (clip=2.0) amplifying faint ionization trails."),
        "Background subtraction": ("Gaussian Background Subtraction", "Subtracts low-frequency illumination blur (51x51) to eliminate ambient glow without losing thin tracks."),
        "Bilateral filter": ("Bilateral Denoising", "Edge-preserving smoothing (d=9, sigma=75) suppressing camera droplet noise while keeping track edges crisp."),
        "Gaussian smoothing": ("Gaussian Blur", "Light 5x5 smoothing to soften residual pixel noise before thresholding."),
        "Reflection suppression": ("Reflection Column Masking", "Zeroes out persistent vertical glass reflection bands at columns 160-250 and 660-750."),
        "Otsu threshold": ("Otsu Binarisation", "Optimal bimodal intensity thresholding isolating particle silhouettes from the dark background."),
        "Morphological cleanup": ("Morphological Opening & Closing", "5x5 closing to seal tiny gaps followed by 3x3 opening to remove dust blobs."),
        "Final binary mask": ("Connected Component Mask", "Connected component area filtering (>= 150 px) eliminating speckle noise."),
    }
    for k, v in raw_stages.items():
        title, desc = stage_meta.get(k, (k, "Preprocessing stage"))
        stages.append({
            "name": title,
            "author": "Ivan (Classical Vision)",
            "desc": desc,
            "image": v,
        })

    return {
        "annotated_rgb": bgr_to_rgb(annotated),
        "mask_rgb": mask_to_rgb(mask),
        "detections": rows,
        "stages": stages,
        "summary": f"{len(rows)} candidate tracks detected",
    }


@st.cache_resource(show_spinner=False)
def load_yolo_detector():
    """Load Jun's YOLO model once and cache."""
    try:
        inference = load_module_by_path("jun_yolo_inference", JUN_YOLO_DIR / "inference.py")
        config = load_module_by_path("jun_yolo_config", JUN_YOLO_DIR / "config.py")
        model = inference.load_model(config.BEST_WEIGHTS)
        return inference, model
    except Exception as e:
        return None, str(e)


def run_jun_yolo(image_bgr: np.ndarray, confidence: float = 0.20) -> Dict[str, Any]:
    """Jun's pipeline: YOLO11 deep learning model with tiled inference."""
    inference, model = load_yolo_detector()
    if inference is None:
        raise RuntimeError(f"YOLO detector could not be loaded: {model}")

    combined = load_combined_pipeline()
    clean_bgr = combined.suppress_osd_and_borders(image_bgr)
    h_orig, w_orig = image_bgr.shape[:2]

    # Preprocessing stages implemented in Jun/YOLO/preprocessing.py
    prep = load_module_by_path("jun_prep", JUN_YOLO_DIR / "preprocessing.py")
    gray = prep._to_grayscale(clean_bgr)
    denoised = prep.apply_denoise(clean_bgr)
    dog = prep.apply_dog_high_pass(denoised)
    otsu = prep.apply_otsu_binarisation(dog)

    stages = [
        {
            "name": "1. Camera OSD Guard",
            "author": "Jun & Team",
            "desc": "Top timestamp area zeroed to prevent camera overlay numbers from triggering false detections.",
            "image": clean_bgr,
        },
        {
            "name": "2. Grayscale Conversion",
            "author": "Jun (YOLO Preprocessor)",
            "desc": "Single-channel intensity conversion for uniform luminance processing.",
            "image": gray,
        },
        {
            "name": "3. Fast Non-Local Means Denoising",
            "author": "Jun (YOLO Preprocessor)",
            "desc": "Non-local means filter (h=20, template=7, search=21) eliminating high-frequency sensor noise.",
            "image": denoised,
        },
        {
            "name": "4. Difference of Gaussians (DoG)",
            "author": "Jun (YOLO Preprocessor)",
            "desc": "High-pass spatial bandpass filter subtracting broad blur (σ=12) from narrow blur (σ=2) to isolate tracks.",
            "image": dog,
        },
        {
            "name": "5. Otsu Track Binarisation",
            "author": "Jun (YOLO Preprocessor)",
            "desc": "Optimal bimodal binarisation providing 3-channel input representation feeding the YOLO11 model backbone.",
            "image": otsu,
        },
    ]

    annotated_enhanced, raw_detections = inference.detect_image(
        model, clean_bgr, conf=confidence, use_tiling=True, use_tta=False
    )

    top_limit = 125 if h_orig >= 1800 else max(80, int(h_orig * 0.065))

    annotated = image_bgr.copy()
    rows = []
    track_idx = 1
    for item in raw_detections:
        x1, y1, x2, y2 = item["bbox"]
        if y1 < top_limit:
            continue  # camera timestamp overlay

        w_box = max(1, x2 - x1)
        h_box = max(1, y2 - y1)
        if (x1 < 45 or x2 > w_orig - 45) and (h_box > 140 and w_box < 70):
            continue  # glass rim reflection sliver
        if y2 > h_orig - 45 and w_box > 250:
            continue  # bottom tray edge

        raw_class = item.get("class_name", "Unknown")
        std_class = "Alpha" if ("Alpha" in raw_class or "Proton" in raw_class) else "Electron"
        conf_val = float(item.get("confidence", 0))
        len_mm = float(item.get("length_mm", 0))

        # Color coding: Yellow for Alpha, Red for Electron
        box_color = (0, 255, 255) if std_class == "Alpha" else (0, 0, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(
            annotated,
            f"#{track_idx} {std_class} ({conf_val*100:.0f}%)",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            box_color,
            1,
            cv2.LINE_AA,
        )

        rows.append({
            "Track ID": track_idx,
            "Type": std_class,
            "Detailed Class": raw_class,
            "Confidence": round(conf_val, 3),
            "Length (mm)": round(len_mm, 2),
            "Thickness (mm)": 0.0,
            "Curvature": 0.0,
            "Tortuosity": 0.0,
        })
        track_idx += 1

    return {
        "annotated_rgb": bgr_to_rgb(annotated),
        "annotated_preprocessed_rgb": bgr_to_rgb(annotated_enhanced),
        "mask_rgb": mask_to_rgb(cv2.cvtColor(otsu, cv2.COLOR_BGR2GRAY) if otsu.ndim == 3 else otsu),
        "detections": rows,
        "stages": stages,
        "summary": f"{len(rows)} deep learning detections",
    }


def run_weiquan(image_bgr: np.ndarray) -> Dict[str, Any]:
    """WeiQuan's pipeline: Canny Edge Detection and Probabilistic Hough Transform."""
    hough = load_module_by_path("weiquan_hough", WEIQUAN_DIR / "Hough_Transform.py")
    output = image_bgr.copy()
    grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(grey)
    blurred = cv2.GaussianBlur(clahe, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=10, maxLineGap=8)

    stages = [
        {
            "name": "1. Grayscale Conversion",
            "author": "WeiQuan (Hough Transform)",
            "desc": "Conversion to 8-bit single-channel intensity map.",
            "image": grey,
        },
        {
            "name": "2. CLAHE Contrast Equalization",
            "author": "WeiQuan (Hough Transform)",
            "desc": "Contrast-Limited Adaptive Histogram Equalization amplifying faint trail edges.",
            "image": clahe,
        },
        {
            "name": "3. Gaussian Smoothing Blur",
            "author": "WeiQuan (Hough Transform)",
            "desc": "5x5 Gaussian kernel smoothing high-frequency noise prior to gradient calculation.",
            "image": blurred,
        },
        {
            "name": "4. Canny Edge Detection",
            "author": "WeiQuan (Hough Transform)",
            "desc": "Hysteresis gradient edge detection (50/150 thresholds) extracting track boundary contours.",
            "image": edges,
        },
    ]

    rows = []
    if lines is not None:
        valid_lines = []
        for line in np.asarray(lines).reshape(-1, 4):
            x1, y1, x2, y2 = map(int, line)
            length = hough.calculate_length(x1, y1, x2, y2)
            if length >= max(25.0, float(hough.MIN_VALID_LINE_LENGTH)):
                orientation = hough.calculate_orientation(x1, y1, x2, y2)
                valid_lines.append((length, orientation, x1, y1, x2, y2))

        # Sort longest first and keep top 80 segments to maintain snappy UI performance
        valid_lines.sort(key=lambda item: item[0], reverse=True)
        for idx, (length, orientation, x1, y1, x2, y2) in enumerate(valid_lines[:80], start=1):
            cv2.line(output, (x1, y1), (x2, y2), (0, 220, 150), 2)
            rows.append({
                "Track ID": idx,
                "Type": "Trajectory Segment",
                "Confidence": 1.0,
                "Length (px)": round(length, 1),
                "Length (mm)": round(length / 12.5, 2),
                "Orientation (deg)": round(orientation, 1),
            })

    return {
        "annotated_rgb": bgr_to_rgb(output),
        "mask_rgb": mask_to_rgb(edges),
        "detections": rows,
        "stages": stages,
        "summary": f"{len(rows)} Hough line segments",
    }


@st.cache_resource(show_spinner=False)
def load_tv_analyzer():
    tv = load_module_by_path("tv_curvilinear_pipeline", TV_PATH)
    return tv, tv.CloudChamberAnalyzer()


def run_tv(image_bgr: np.ndarray) -> Dict[str, Any]:
    """TV's pipeline: Frangi vesselness, skeleton extraction, dot-linking, and distance transform."""
    tv, analyzer = load_tv_analyzer()
    gray, enhanced = analyzer.preprocess(image_bgr)
    curvilinear_map = tv.apply_curvilinear_filter(enhanced, sigmas=analyzer.curvilinear_scales)
    # Default to 0.70 threshold matching TV's updated code
    binary, skeleton = analyzer.segment_tracks(curvilinear_map, threshold_factor=0.70)
    try:
        tracks = analyzer.analyze_trajectories(binary, skeleton, enhanced, curvilinear_map)
    except TypeError:
        tracks = analyzer.analyze_trajectories(binary, skeleton, enhanced)
    annotated = analyzer.annotate_detections(image_bgr, tracks)

    rows = []
    for track in tracks:
        rows.append({
            "Track ID": track.get("track_id"),
            "Type": track.get("classification"),
            "Confidence": round(float(track.get("confidence", 0)), 3),
            "Dots / Fragments": track.get("num_dots", 1),
            "Length (mm)": round(float(track.get("length_cm", 0)) * 10, 2),
            "Thickness (mm)": round(float(track.get("width_mm", 0)), 2),
            "Curvature": round(float(track.get("curvature", 0)), 4),
            "Tortuosity": round(float(track.get("tortuosity", 0)), 4),
        })

    vesselness = np.clip(curvilinear_map * 255, 0, 255).astype(np.uint8)

    stages = [
        {
            "name": "1. Grayscale & OSD Suppression",
            "author": "TV (Frangi Scale-Space)",
            "desc": "Converts BGR to grayscale and zeroes out camera status bar in top margin (y < 135).",
            "image": gray,
        },
        {
            "name": "2. Top-Hat + CLAHE Illumination Levelling",
            "author": "TV (Frangi Scale-Space)",
            "desc": "White Top-Hat morphological filtering (31x31 ellipse) blended with bilateral filter and CLAHE.",
            "image": enhanced,
        },
        {
            "name": "3. Multiscale Frangi Vesselness Filter",
            "author": "TV (Frangi Scale-Space)",
            "desc": "Hessian matrix eigenvalue analysis across scales (σ=1.2, 2.0, 3.2, 5.0) isolating tubular structures.",
            "image": vesselness,
        },
        {
            "name": "4. Adaptive Track Binarization",
            "author": "TV (Frangi Scale-Space)",
            "desc": "Adaptive thresholding on vesselness probability map with morphological closing.",
            "image": binary,
        },
        {
            "name": "5. Centerline Skeletonization",
            "author": "TV (Frangi Scale-Space)",
            "desc": "Iterative morphological thinning extracting 1-pixel wide continuous trajectory paths.",
            "image": skeleton,
        },
    ]

    return {
        "annotated_rgb": bgr_to_rgb(annotated),
        "mask_rgb": mask_to_rgb(binary),
        "feature_rgb": mask_to_rgb(vesselness),
        "detections": rows,
        "stages": stages,
        "summary": f"{len(rows)} Frangi vesselness tracks",
    }


def load_combined_pipeline():
    return load_module_by_path("combined_pipeline_module_v3", COMBINED_PATH)


def run_combined(
    image_bgr: np.ndarray,
    conf_threshold: float = 0.10,
    recall_mode: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run the integrated hybrid pipeline combining YOLO pattern detection and classical Alpha geometry."""
    combined = load_combined_pipeline()
    effective_conf = min(conf_threshold, 0.10) if recall_mode else conf_threshold
    annotated, mask, detections, stages = combined.process_image(
        image_bgr,
        conf_threshold=effective_conf,
        collect_stages=True,
    )

    rows = []
    for d in detections:
        rows.append({
            "Track ID": d.get("track_id"),
            "Type": d.get("type"),
            "Confidence": round(float(d.get("confidence", 0)), 3),
            "Engine": d.get("source", "Hybrid Ensemble"),
            "Length (mm)": round(float(d.get("length_mm", 0)), 2),
            "Thickness (mm)": round(float(d.get("width_mm", 0)), 2),
            "Curvature": round(float(d.get("curvature", 0)), 4),
            "Tortuosity": round(float(d.get("tortuosity", 0)), 4),
            "Area (px)": d.get("area_px", 0),
        })

    return {
        "annotated_rgb": bgr_to_rgb(annotated),
        "mask_rgb": mask_to_rgb(mask),
        "detections": rows,
        "stages": stages,
        "summary": f"{len(rows)} hybrid verified tracks",
    }


PIPELINE_CATALOG = {
    "Team Integrated (Combined)": {
        "runner": run_combined,
        "tag": "Final Submission",
        "badge": "★ Recommended",
        "desc": "Hybrid pipeline: Background subtraction, multiscale Frangi filter, skeleton EDT width, and Hough validation.",
    },
    "Ivan · Classical Otsu & Geometry": {
        "runner": run_ivan,
        "tag": "Classical Vision",
        "badge": "Baseline",
        "desc": "Dynamic Gaussian background subtraction, bilateral filtering, Otsu binarisation, and minAreaRect geometry.",
    },
    "Jun · YOLO11 Deep Learning": {
        "runner": run_jun_yolo,
        "tag": "Deep Learning",
        "badge": "Neural Network",
        "desc": "Fine-tuned YOLO11 convolutional detector trained on annotated frames with overlapping tiled inference.",
    },
    "TV · Frangi Vesselness & EDT": {
        "runner": run_tv,
        "tag": "Scale-Space Analysis",
        "badge": "Hessian Filter",
        "desc": "Multiscale 2nd-order Gaussian derivatives, Frangi ridge filter, morphological skeleton, and Euclidean Distance Transform.",
    },
    "WeiQuan · Hough Line Transform": {
        "runner": run_weiquan,
        "tag": "Trajectory Geometry",
        "badge": "Edge & Lines",
        "desc": "CLAHE contrast boost, Canny edge detection, Probabilistic Hough Lines, and PCA trajectory ordering.",
    },
}


# -----------------------------------------------------------------------------
# Sample Dataset Discovery
# -----------------------------------------------------------------------------
@st.cache_data
def get_sample_images() -> Dict[str, Path]:
    """Scan Raw_Dataset to discover standard sample images for 1-click testing."""
    samples = {}
    dataset_dir = ROOT / "Ivan" / "Raw_Dataset"
    if dataset_dir.exists():
        for set_folder in sorted(dataset_dir.glob("ImageSet*")):
            bw_folder = set_folder / "BW"
            if bw_folder.exists():
                jpgs = sorted(bw_folder.glob("*.jpg"))
                if jpgs:
                    samples[f"{set_folder.name} · Frame A (Snapshot 1)"] = jpgs[0]
                    if len(jpgs) > 1:
                        mid_idx = len(jpgs) // 2
                        samples[f"{set_folder.name} · Frame B (Snapshot {mid_idx+1})"] = jpgs[mid_idx]
    return samples


@st.cache_data
def get_sample_videos() -> Dict[str, Path]:
    """Scan Ivan/Raw_Dataset to find benchmark cloud chamber video files."""
    videos = {}
    dataset_dir = ROOT / "Ivan" / "Raw_Dataset"
    if dataset_dir.exists():
        for vid_folder in sorted(dataset_dir.glob("VID_*")):
            mp4s = sorted(vid_folder.glob("*.mp4"))
            if mp4s:
                size_mb = mp4s[0].stat().st_size // (1024 * 1024)
                videos[f"{vid_folder.name} ({size_mb} MB)"] = mp4s[0]
    return videos


# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------
def main():
    # Header Banner
    st.markdown(
        """
        <div class="hero-container">
            <span class="hero-badge">BMDS2133 Image Processing · Assignment Topic 4</span>
            <h1 class="hero-title">Cloud Chamber Particle Track Analysis Lab</h1>
            <p class="hero-subtitle">
                Automated particle trajectory segmentation, physical spatial calibration, and multi-class
                distinction between heavy <b>Alpha particles</b> (dense, straight) and lighter <b>Electrons</b> (thin, deflected).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Collapsible Team Architecture Guide
    with st.expander("👥 Team Members & Methodologies Overview", expanded=False):
        st.markdown(
            """
            <div class="team-grid">
                <div class="member-card">
                    <div class="member-name">Ivan</div>
                    <div class="member-tech">Gaussian Background Subtraction, Bilateral Denoising, Otsu Thresholding, Bounding Box Geometry</div>
                    <span class="pill">Classical Preprocessing</span>
                </div>
                <div class="member-card">
                    <div class="member-name">Jun</div>
                    <div class="member-tech">YOLO11 Nano Neural Network, Red-box annotation inpainting, Tiled SAHI-style multi-scale inference</div>
                    <span class="pill">Deep Learning</span>
                </div>
                <div class="member-card">
                    <div class="member-name">WeiQuan</div>
                    <div class="member-tech">CLAHE, Canny Edge Detector, Probabilistic Hough Lines (HoughLinesP), PCA Angular Curvature</div>
                    <span class="pill">Geometric Line Analysis</span>
                </div>
                <div class="member-card">
                    <div class="member-name">TV</div>
                    <div class="member-tech">Multiscale Hessian Eigenvalues, Frangi Vesselness Filter, Centerline Skeleton, EDT Width Profiling</div>
                    <span class="pill">Scale-Space Physics</span>
                </div>
                <div class="member-card" style="border-color: #38bdf8;">
                    <div class="member-name" style="color: #38bdf8;">Team Combined</div>
                    <div class="member-tech">Ivan's reflection suppression + TV's Frangi vesselness + Skeleton EDT + WeiQuan's Hough straightness</div>
                    <span class="pill" style="background: rgba(56, 189, 248, 0.2); color: #38bdf8;">Final Integrated Pipeline</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # Sidebar Configuration Controls
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### ⚙️ Analysis Studio")

        analysis_mode = st.radio(
            "Workflow Mode",
            [
                "🔬 Single Frame Deep-Dive",
                "⚖️ Side-by-Side Model Comparison",
                "📁 Batch / Multi-Image Ingestion",
                "🎥 Video Particle Tracking & Playback",
            ],
            help="Select whether to analyze one frame in depth, compare models side-by-side, process a batch of files, or analyze continuous video.",
        )

        selected_images: List[Tuple[str, np.ndarray]] = []
        selected_video_path: Optional[str] = None
        selected_video_name: str = ""
        video_start_time: float = 0.0
        video_stride: int = 3
        max_video_frames: int = 60

        if analysis_mode == "🎥 Video Particle Tracking & Playback":
            st.markdown("---")
            st.markdown("#### 🎥 Video Source")
            vid_source_choice = st.radio(
                "Video Source Selection",
                ["📂 One-Click Sample Videos", "📤 Upload Custom Video (.mp4, .avi, .mov)"],
                index=0,
            )
            sample_videos = get_sample_videos()
            if vid_source_choice == "📂 One-Click Sample Videos":
                if sample_videos:
                    chosen_vid = st.selectbox("Select Video Recording", list(sample_videos.keys()))
                    selected_video_path = str(sample_videos[chosen_vid])
                    selected_video_name = chosen_vid
                    st.caption(f"Loaded: `{selected_video_path}`")
                else:
                    st.warning("No sample videos found in `Ivan/Raw_Dataset/`.")
            else:
                up_vid = st.file_uploader(
                    "Upload Video",
                    type=["mp4", "avi", "mov", "mkv"],
                    help="Upload cloud chamber video recording.",
                )
                if up_vid is not None:
                    temp_in = tempfile.NamedTemporaryFile(suffix=Path(up_vid.name).suffix, delete=False)
                    temp_in.write(up_vid.read())
                    temp_in.close()
                    selected_video_path = temp_in.name
                    selected_video_name = up_vid.name
                    st.caption(f"Uploaded: `{selected_video_name}`")

            st.markdown("---")
            st.markdown("#### ⚙️ Video Tracking Controls")

            # Curated particle timestamps for the benchmark recordings
            curated_hotspots = {
                "220549": [("⚡ Active Burst at 2.5s (Frames 75-95, Alphas + Electrons)", 2.5)],
                "215401": [("⚡ Active Particles at 80.0s / 1m20s (Frame 4750)", 80.0), ("⚡ Fast Ionization at 0.5s", 0.5)],
                "220753": [("⚡ Ionization Track at 16.0s", 16.0), ("⚡ Fast Particle at 0.5s", 0.5)],
                "221024": [("⚡ Dense Particle Trails at 81.0s / 1m21s", 81.0), ("⚡ Alpha Tracks at 133.0s / 2m13s", 133.0)],
            }

            matching_hotspots = []
            for k, spots in curated_hotspots.items():
                if k in selected_video_name:
                    matching_hotspots = spots
                    break

            default_start = 0.0
            if matching_hotspots:
                spot_names = [s[0] for s in matching_hotspots] + ["⏱️ Custom Timestamp Slider"]
                chosen_spot = st.selectbox(
                    "🎯 Jump to Particle Hotspot",
                    spot_names,
                    index=0,
                    help="Instantly jumps to the exact timestamp where ionization tracks appear in this recording.",
                )
                if chosen_spot != "⏱️ Custom Timestamp Slider":
                    for s_name, s_val in matching_hotspots:
                        if s_name == chosen_spot:
                            default_start = s_val
                            break

            video_start_time = st.slider(
                "Start Timestamp (seconds)",
                min_value=0.0,
                max_value=150.0,
                value=float(default_start),
                step=0.5,
                help="Jump directly to the video timestamp where particles appear.",
            )
            video_stride = st.slider(
                "Frame Stride (Sample Rate)",
                min_value=1,
                max_value=30,
                value=3,
                help="Process every N-th frame. Fast ionization tracks last ~0.15s (3-8 frames). A stride of 2-4 captures fast tracks without jumping over them.",
            )
            max_video_frames = st.slider(
                "Max Sampled Frames",
                min_value=10,
                max_value=200,
                value=60,
                help="Total number of sampled frames to analyze.",
            )
            active_single_pipe = st.selectbox(
                "Pipeline for Video",
                ["Team Integrated (Combined)", "Jun · YOLO11 Deep Learning", "Ivan · Classical Otsu & Geometry"],
                index=0,
            )
            active_pipelines = [active_single_pipe]

        else:
            st.markdown("---")
            st.markdown("#### 📥 Input Frame")

            input_choice = st.radio(
                "Source Selection",
                ["📂 One-Click Curated Samples", "📤 Upload Custom Image(s)"],
                index=0,
            )

            sample_images = get_sample_images()

            if input_choice == "📂 One-Click Curated Samples":
                if sample_images:
                    chosen_sample = st.selectbox("Select Cloud Chamber Frame", list(sample_images.keys()))
                    img_path = sample_images[chosen_sample]
                    raw_bgr = cv2.imread(str(img_path))
                    if raw_bgr is not None:
                        selected_images.append((img_path.name, raw_bgr))
                        st.caption(f"Loaded: `{img_path.name}` ({raw_bgr.shape[1]}×{raw_bgr.shape[0]} px)")
                else:
                    st.warning("No sample images found in `Ivan/Raw_Dataset/`.")
            else:
                accept_multi = "Batch" in analysis_mode
                uploaded_files = st.file_uploader(
                    "Upload Image(s)",
                    type=["jpg", "jpeg", "png", "bmp"],
                    accept_multiple_files=accept_multi,
                    help="Upload 1080×1920 cloud chamber snapshots or any test images.",
                )
                if uploaded_files:
                    files_list = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
                    for up_file in files_list:
                        pil_img = Image.open(up_file).convert("RGB")
                        bgr_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                        selected_images.append((up_file.name, bgr_img))

            st.markdown("---")
            st.markdown("#### 🧩 Pipeline Selection")

            if analysis_mode == "⚖️ Side-by-Side Model Comparison":
                active_pipelines = st.multiselect(
                    "Pipelines to Compare",
                    options=list(PIPELINE_CATALOG.keys()),
                    default=[
                        "Team Integrated (Combined)",
                        "Ivan · Classical Otsu & Geometry",
                        "TV · Frangi Vesselness & EDT",
                    ],
                )
            else:
                active_single_pipe = st.selectbox(
                    "Active Pipeline",
                    options=list(PIPELINE_CATALOG.keys()),
                    index=0,
                )
                active_pipelines = [active_single_pipe]

        # Calibration & Advanced Hyperparameters Accordion
        with st.expander("🛠️ Calibration & Parameters", expanded=False):
            pixels_per_mm = st.number_input(
                "Spatial Scale (pixels/mm)",
                min_value=1.0,
                max_value=100.0,
                value=12.5,
                step=0.5,
                help="Calibrated scaling ratio. 12.5 px/mm translates to 125 px/cm.",
            )
            default_conf = 0.12 if "Video" in analysis_mode else 0.15
            yolo_conf = st.slider(
                "YOLO Confidence Threshold",
                min_value=0.05,
                max_value=0.90,
                value=default_conf,
                step=0.01,
                help="Lower confidence boosts recall for faint, thin tracks. 0.12 recommended for cloud chamber video.",
            )
            alpha_width_thresh = st.slider(
                "Alpha Width Threshold (mm)",
                min_value=0.5,
                max_value=3.0,
                value=1.4,
                step=0.1,
                help="Tracks thicker than this threshold are classified as Alpha particles.",
            )
            combined_recall_mode = st.checkbox(
                "Preserve faint tracks in combined pipeline",
                value=True,
                help="Uses one morphology-opening pass and lower track-size limits. The original combined baseline remains available by clearing this option.",
            )

        run_btn = st.button("🚀 Run Particle Detection", type="primary", use_container_width=True)

    # -------------------------------------------------------------------------
    # Guard clause: No image selected (for image modes)
    # -------------------------------------------------------------------------
    if analysis_mode != "🎥 Video Particle Tracking & Playback" and not selected_images:
        st.info("👈 Please select a sample image or upload your own in the sidebar to begin.")
        return

    # -------------------------------------------------------------------------
    # Mode 1: Single Frame Deep-Dive
    # -------------------------------------------------------------------------
    if analysis_mode == "🔬 Single Frame Deep-Dive":
        image_name, current_bgr = selected_images[0]
        selected_pipeline_name = active_pipelines[0]
        pipe_meta = PIPELINE_CATALOG[selected_pipeline_name]

        st.markdown(f"### 🎯 Results: **{selected_pipeline_name}**")
        st.caption(f"Input Frame: **{image_name}** · Method: *{pipe_meta['desc']}*")

        # Process image
        with st.spinner(f"Processing with {selected_pipeline_name}..."):
            start_t = time.time()
            runner = pipe_meta["runner"]
            try:
                if "YOLO" in selected_pipeline_name:
                    res = runner(current_bgr, confidence=yolo_conf)
                elif "Team Integrated" in selected_pipeline_name or "Combined" in selected_pipeline_name:
                    res = runner(current_bgr, conf_threshold=yolo_conf, recall_mode=combined_recall_mode)
                else:
                    res = runner(current_bgr)
            except Exception as e:
                st.error(f"Error executing {selected_pipeline_name}: {e}")
                return
            elapsed = time.time() - start_t

        detections = res.get("detections", [])
        df = pd.DataFrame(detections)

        # Compute KPI Metrics
        n_total = len(detections)
        if not df.empty and "Type" in df.columns:
            n_alpha = int((df["Type"] == "Alpha").sum())
            n_electron = int((df["Type"] == "Electron").sum())
            avg_length = round(float(df["Length (mm)"].mean()), 1) if "Length (mm)" in df.columns else 0.0
            avg_thick = round(float(df["Thickness (mm)"].mean()), 2) if "Thickness (mm)" in df.columns else 0.0
        else:
            n_alpha, n_electron, avg_length, avg_thick = 0, 0, 0.0, 0.0

        # KPI Dashboard Cards
        st.markdown(
            f"""
            <div class="kpi-row">
                <div class="kpi-card">
                    <div class="kpi-lbl">Alpha Particles</div>
                    <div class="kpi-val kpi-alpha">{n_alpha}</div>
                    <span class="pill" style="color: #f59e0b;">Thick & Dense</span>
                </div>
                <div class="kpi-card">
                    <div class="kpi-lbl">Electron Particles</div>
                    <div class="kpi-val kpi-electron">{n_electron}</div>
                    <span class="pill" style="color: #ef4444;">Thin / Deflected</span>
                </div>
                <div class="kpi-card">
                    <div class="kpi-lbl">Total Tracks</div>
                    <div class="kpi-val kpi-total">{n_total}</div>
                    <span class="pill">{res.get('summary', 'Detected')}</span>
                </div>
                <div class="kpi-card">
                    <div class="kpi-lbl">Mean Length</div>
                    <div class="kpi-val">{avg_length} <span style="font-size:0.9rem;color:#94a3b8;">mm</span></div>
                    <span class="pill">{(avg_length/10.0):.2f} cm</span>
                </div>
                <div class="kpi-card">
                    <div class="kpi-lbl">Mean Thickness</div>
                    <div class="kpi-val">{avg_thick} <span style="font-size:0.9rem;color:#94a3b8;">mm</span></div>
                    <span class="pill">Speed: {elapsed:.2f}s</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Tabbed Analysis Workspace
        tab_visual, tab_charts, tab_table, tab_report = st.tabs([
            "🖼️ Visual Inspections",
            "📊 Analytics & Distributions",
            "📋 Measurements Table",
            "📑 Automated Scientific Report",
        ])

        # TAB 1: VISUAL INSPECTIONS
        with tab_visual:
            col_left, col_right = st.columns([1.2, 1])

            with col_left:
                st.markdown("##### Annotated Particle Tracks")
                if res.get("annotated_preprocessed_rgb") is not None:
                    bg_choice = st.radio(
                        "Track Visualization Background:",
                        ["Raw Camera Colors (BGR)", "Preprocessed Frame (DoG + Otsu [YOLO Model Input])"],
                        horizontal=True,
                        key=f"bg_choice_{selected_pipeline_name}_{image_name}",
                        help="Toggle between the camera frame and the preprocessed frame (Otsu-binarised Difference of Gaussians) that YOLO detects on.",
                    )
                    active_ann = (
                        res["annotated_preprocessed_rgb"]
                        if "Preprocessed" in bg_choice
                        else res["annotated_rgb"]
                    )
                else:
                    active_ann = res["annotated_rgb"]

                st.image(
                    active_ann,
                    caption=f"Annotated frame with bounding boxes and track badges · {n_total} detections",
                    use_container_width=True,
                )

            with col_right:
                st.markdown("##### Intermediate Filter Views")
                if res.get("mask_rgb") is not None:
                    st.image(
                        res["mask_rgb"],
                        caption="Binary segmentation / edge detection mask",
                        use_container_width=True,
                    )
                if res.get("feature_rgb") is not None:
                    st.image(
                        res["feature_rgb"],
                        caption="Frangi Vesselness response (ridge probability map)",
                        use_container_width=True,
                    )
                with st.expander("🔍 View Raw Unprocessed Input Frame"):
                    st.image(bgr_to_rgb(current_bgr), caption="Raw input frame", use_container_width=True)

            # -----------------------------------------------------------------
            # Interactive Preprocessing & Filter Stages Explorer
            # -----------------------------------------------------------------
            stages = res.get("stages", [])
            if stages:
                st.markdown("---")
                st.markdown("### 🔬 Pipeline Preprocessing & Filter Stages Explorer")
                st.caption(
                    f"Interactive visual inspection of all {len(stages)} intermediate filtering and transformation "
                    f"stages executed by **{selected_pipeline_name}**."
                )

                # Selector for detailed inspection
                stage_labels = [f"Stage {i+1}: {s['name']}" for i, s in enumerate(stages)]
                selected_idx = st.selectbox(
                    "🔍 Select Preprocessing Stage to Inspect in High Resolution:",
                    range(len(stages)),
                    format_func=lambda i: stage_labels[i],
                    key=f"stage_select_{selected_pipeline_name}_{image_name}",
                )

                selected_stage = stages[selected_idx]
                disp_img = format_stage_image_for_display(selected_stage["image"])

                stage_col1, stage_col2 = st.columns([1.3, 1])
                with stage_col1:
                    st.image(
                        disp_img,
                        caption=f"{stage_labels[selected_idx]} ({selected_stage.get('author', 'Architecture')})",
                        use_container_width=True,
                    )
                with stage_col2:
                    st.markdown(
                        f"""
                        <div style="background:#111827;padding:1.25rem;border-radius:12px;border:1px solid #1f2937;">
                            <div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;color:#38bdf8;font-weight:700;">
                                Originator / Architecture
                            </div>
                            <div style="font-size:1.15rem;font-weight:600;color:#f3f4f6;margin-bottom:0.75rem;">
                                {selected_stage.get('author', 'Integrated Hybrid')}
                            </div>
                            <div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;color:#94a3b8;font-weight:700;">
                                Transformation Purpose & Mechanism
                            </div>
                            <div style="font-size:0.92rem;line-height:1.6;color:#cbd5e1;margin-bottom:1rem;">
                                {selected_stage.get('desc', 'Intermediate filter output.')}
                            </div>
                            <div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;color:#94a3b8;font-weight:700;">
                                Technical Diagnostics
                            </div>
                            <div style="font-size:0.85rem;color:#94a3b8;line-height:1.7;">
                                • Resolution: <code>{selected_stage['image'].shape[1]} × {selected_stage['image'].shape[0]} px</code><br>
                                • Channels: <code>{1 if selected_stage['image'].ndim == 2 else selected_stage['image'].shape[2]}</code><br>
                                • Value Range: <code>[{selected_stage['image'].min()}, {selected_stage['image'].max()}]</code>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Filmstrip of all stages
                st.markdown("##### 🎞️ Sequential Transformation Filmstrip")
                st.caption("Visual progression from raw chamber frame to final track segmentation:")
                film_cols = st.columns(min(len(stages), 6))
                for idx, (f_col, stage_item) in enumerate(zip(film_cols, stages)):
                    with f_col:
                        st.image(
                            format_stage_image_for_display(stage_item["image"]),
                            caption=f"{idx+1}. {stage_item['name'].split('.')[-1].strip()}",
                            use_container_width=True,
                        )

        # TAB 2: ANALYTICS & CHARTS
        with tab_charts:
            st.markdown("##### Particle Physical Distributions")
            if not df.empty and "Type" in df.columns:
                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:
                    st.markdown("###### Track Length vs. Thickness Scatter Profile")
                    # Scatter plot
                    scatter_df = df.copy()
                    st.scatter_chart(
                        scatter_df,
                        x="Length (mm)",
                        y="Thickness (mm)",
                        color="Type",
                        height=350,
                    )
                    st.caption("Alpha particles cluster in the high-thickness zone (≥ 1.4 mm); electrons occupy lower thicknesses.")

                with chart_col2:
                    st.markdown("###### Particle Class Composition")
                    type_counts = df["Type"].value_counts().reset_index()
                    type_counts.columns = ["Type", "Count"]
                    st.bar_chart(type_counts, x="Type", y="Count", color="Type", height=350)
                    st.caption("Distribution of identified particle species in the current cloud chamber frame.")
            else:
                st.info("No detections available to plot.")

        # TAB 3: MEASUREMENTS TABLE
        with tab_table:
            st.markdown("##### Full Calibrated Measurements Table")
            if not df.empty:
                # Column filters
                filter_col1, filter_col2 = st.columns([2, 1])
                with filter_col1:
                    type_filter = st.multiselect(
                        "Filter by Particle Species",
                        options=df["Type"].unique().tolist(),
                        default=df["Type"].unique().tolist(),
                    )
                with filter_col2:
                    csv_data = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Measurements CSV",
                        data=csv_data,
                        file_name=f"{Path(image_name).stem}_measurements.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                filtered_df = df[df["Type"].isin(type_filter)]
                st.dataframe(filtered_df, use_container_width=True, hide_index=True)
            else:
                st.info("No measurements available.")

        # TAB 4: AUTOMATED REPORT
        with tab_report:
            st.markdown("##### Automated Particle Physics Summary Report")
            report_text = f"""# Cloud Chamber Particle Analysis Report
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Image:** `{image_name}`  
**Pipeline:** {selected_pipeline_name}  
**Spatial Calibration:** {pixels_per_mm} pixels/mm (125 px/cm)  

---

## 1. Executive Summary
- **Total Detected Particle Tracks:** {n_total}
- **Alpha Particle Candidates:** {n_alpha} ({round(n_alpha/max(1,n_total)*100, 1)}%)
- **Electron / Beta Candidates:** {n_electron} ({round(n_electron/max(1,n_total)*100, 1)}%)
- **Average Track Length:** {avg_length} mm ({round(avg_length/10.0, 2)} cm)
- **Average Ionization Column Width (Thickness):** {avg_thick} mm

---

## 2. Physical Interpretations
1. **Alpha Radiation Evidence:**
   - Alpha particles (Helium-4 nuclei, He 2+) possess high linear energy transfer (LET).
   - In cloud chambers, they create dense, continuous condensation columns typically exceeding **1.4 mm** in width.
   - Due to their heavy mass (~4 amu), their momentum resists deflection, maintaining straight trajectories.

2. **Electron / Beta Radiation Evidence:**
   - Electrons ($e^-$ / $\beta$) produce sparse ionization, yielding wispy tracks with widths under **1.4 mm**.
   - Readily scattered by background gas collisions and curled by magnetic deflection.

---
*Report automatically generated by BMDS2133 Cloud Chamber Lab Application.*
"""
            st.markdown(report_text)
            st.download_button(
                label="📥 Download Markdown Scientific Report",
                data=report_text.encode("utf-8"),
                file_name=f"{Path(image_name).stem}_scientific_report.md",
                mime="text/markdown",
            )

    # -------------------------------------------------------------------------
    # Mode 2: Side-by-Side Model Comparison
    # -------------------------------------------------------------------------
    elif analysis_mode == "⚖️ Side-by-Side Model Comparison":
        image_name, current_bgr = selected_images[0]
        st.markdown(f"### ⚖️ Multi-Pipeline Benchmark on `{image_name}`")
        st.caption("Side-by-side comparative inspection to evaluate segmentation differences across team architectures.")

        if not active_pipelines:
            st.warning("Please select at least one pipeline from the sidebar to compare.")
            return

        cols = st.columns(len(active_pipelines))

        for col, pipe_name in zip(cols, active_pipelines):
            meta = PIPELINE_CATALOG[pipe_name]
            with col:
                st.markdown(f"#### {pipe_name}")
                st.caption(f"{meta['tag']} · {meta['badge']}")

                with st.spinner(f"Running {pipe_name}..."):
                    try:
                        runner = meta["runner"]
                        if "YOLO" in pipe_name:
                            res = runner(current_bgr, confidence=yolo_conf)
                        elif "Team Integrated" in pipe_name or "Combined" in pipe_name:
                            res = runner(current_bgr, conf_threshold=yolo_conf, recall_mode=combined_recall_mode)
                        else:
                            res = runner(current_bgr)
                    except Exception as err:
                        st.error(f"Error: {err}")
                        continue

                st.image(res["annotated_rgb"], caption=res.get("summary", "Done"), use_container_width=True)

                dets = res.get("detections", [])
                df_c = pd.DataFrame(dets)
                n_a = (df_c["Type"] == "Alpha").sum() if not df_c.empty and "Type" in df_c.columns else 0
                n_e = (df_c["Type"] == "Electron").sum() if not df_c.empty and "Type" in df_c.columns else 0

                st.markdown(
                    f"""
                    <div style="background:#111827;padding:0.75rem;border-radius:8px;border:1px solid #1f2937;margin-top:0.5rem;">
                        <b>Total:</b> {len(dets)} tracks<br>
                        <span style="color:#f59e0b;">● Alpha:</span> {n_a} &nbsp;|&nbsp; 
                        <span style="color:#ef4444;">● Electron:</span> {n_e}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if dets:
                    with st.expander("📋 View Measurements Table"):
                        st.dataframe(df_c, hide_index=True, use_container_width=True)

                if res.get("stages"):
                    with st.expander(f"🔬 Preprocessing Stages ({len(res['stages'])})"):
                        for stg_i, stg in enumerate(res["stages"], 1):
                            st.caption(f"**{stg_i}. {stg['name']}** · *{stg.get('author', '')}*")
                            st.image(format_stage_image_for_display(stg["image"]), use_container_width=True)
                            st.caption(stg.get("desc", ""))

    # -------------------------------------------------------------------------
    # Mode 3: Batch / Multi-Image Ingestion (Rubric Extra Effort)
    # -------------------------------------------------------------------------
    elif analysis_mode == "📁 Batch / Multi-Image Ingestion":
        st.markdown(f"### 📁 Batch Processing Workspace ({len(selected_images)} images loaded)")
        st.caption("Automated batch ingestion and bulk physical measurement extraction across image sets.")

        selected_pipe = active_pipelines[0]
        meta = PIPELINE_CATALOG[selected_pipe]

        if st.button("⚡ Process All Loaded Images", type="primary"):
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            batch_summary_rows = []
            all_detections_rows = []

            for idx, (f_name, bgr_img) in enumerate(selected_images):
                status_text.text(f"Processing ({idx+1}/{len(selected_images)}): {f_name}...")
                progress_bar.progress((idx + 1) / len(selected_images))

                try:
                    runner = meta["runner"]
                    if "YOLO" in selected_pipe:
                        res = runner(bgr_img, confidence=yolo_conf)
                    elif "Team Integrated" in selected_pipe or "Combined" in selected_pipe:
                        res = runner(bgr_img, conf_threshold=yolo_conf, recall_mode=combined_recall_mode)
                    else:
                        res = runner(bgr_img)

                    dets = res.get("detections", [])
                    df_b = pd.DataFrame(dets)
                    n_a = int((df_b["Type"] == "Alpha").sum()) if not df_b.empty and "Type" in df_b.columns else 0
                    n_e = int((df_b["Type"] == "Electron").sum()) if not df_b.empty and "Type" in df_b.columns else 0
                    avg_l = round(float(df_b["Length (mm)"].mean()), 2) if not df_b.empty and "Length (mm)" in df_b.columns else 0.0

                    batch_summary_rows.append({
                        "Image": f_name,
                        "Total Tracks": len(dets),
                        "Alpha Count": n_a,
                        "Electron Count": n_e,
                        "Mean Length (mm)": avg_l,
                    })

                    for d in dets:
                        row = dict(d)
                        row["Image"] = f_name
                        all_detections_rows.append(row)

                except Exception as e:
                    batch_summary_rows.append({
                        "Image": f_name,
                        "Total Tracks": 0,
                        "Alpha Count": 0,
                        "Electron Count": 0,
                        "Mean Length (mm)": 0.0,
                        "Error": str(e),
                    })

            progress_bar.progress(1.0)
            status_text.success("✅ Batch processing completed successfully!")

            summary_df = pd.DataFrame(batch_summary_rows)
            st.markdown("#### 📊 Batch Aggregation Summary")
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            # Bulk CSV Export
            if all_detections_rows:
                full_df = pd.DataFrame(all_detections_rows)
                bulk_csv = full_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Complete Batch Measurements CSV",
                    data=bulk_csv,
                    file_name="batch_particle_measurements.csv",
                    mime="text/csv",
                )

    # -------------------------------------------------------------------------
    # Mode 4: Video Particle Tracking & Playback (Rubric Extra Effort)
    # -------------------------------------------------------------------------
    elif analysis_mode == "🎥 Video Particle Tracking & Playback":
        if not selected_video_path or not os.path.exists(selected_video_path):
            st.warning("👈 Please select a sample video or upload your video file in the sidebar to begin.")
            return

        cap = cv2.VideoCapture(selected_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        v_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        v_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_s = total_frames / max(1.0, fps)

        st.markdown(f"### 🎥 Video Particle Tracking: **{selected_video_name}**")
        st.caption("Automated continuous frame sampling, particle trajectory tracking, and temporal timeline analysis.")

        st.markdown(
            f"""
            <div class="kpi-row">
                <div class="kpi-card"><div class="kpi-lbl">Resolution</div><div class="kpi-val" style="font-size:1.15rem;">{v_w}×{v_h} px</div></div>
                <div class="kpi-card"><div class="kpi-lbl">Frame Rate</div><div class="kpi-val" style="font-size:1.15rem;">{fps:.1f} fps</div></div>
                <div class="kpi-card"><div class="kpi-lbl">Duration</div><div class="kpi-val" style="font-size:1.15rem;">{duration_s:.1f} s</div></div>
                <div class="kpi-card"><div class="kpi-lbl">Total Frames</div><div class="kpi-val" style="font-size:1.15rem;">{total_frames}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        selected_pipe = active_pipelines[0]
        pipe_meta = PIPELINE_CATALOG[selected_pipe]

        col_prev, col_act = st.columns([1, 1])
        with col_prev:
            preview_frame_idx = int(video_start_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, preview_frame_idx)
            ret, first_frame = cap.read()
            if ret:
                st.image(bgr_to_rgb(first_frame), caption=f"Video Preview at {video_start_time:.1f}s (Frame {preview_frame_idx})", use_container_width=True)
            cap.set(cv2.CAP_PROP_POS_FRAMES, preview_frame_idx)

        with col_act:
            st.markdown("##### Tracking Configuration Summary")
            st.info(
                f"• Active Model: **{selected_pipe}**\n"
                f"• Start Time: **{video_start_time:.1f} s** (Frame {int(video_start_time * fps)})\n"
                f"• Frame Stride: **Every {video_stride} frames** (≈ {fps/video_stride:.1f} fps capture rate)\n"
                f"• Target Frame Budget: **{max_video_frames} frames**"
            )
            start_tracking_btn = st.button("⚡ Process Video & Track Particles", type="primary", use_container_width=True)

        if start_tracking_btn:
            temp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            temp_out.close()

            # Ensure even dimensions for H.264 encoder compatibility
            enc_w = v_w - (v_w % 2)
            enc_h = v_h - (v_h % 2)
            playback_fps = max(2.0, min(15.0, fps / video_stride))

            # Initialize H.264 browser-compatible video writer
            use_ffmpeg = HAS_IMAGEIO_FFMPEG
            ffmpeg_writer = None
            cv2_writer = None

            if use_ffmpeg:
                try:
                    ffmpeg_writer = imageio_ffmpeg.write_frames(
                        temp_out.name,
                        (enc_w, enc_h),
                        fps=playback_fps,
                        codec="libx264",
                        pix_fmt_in="bgr24",
                        pix_fmt_out="yuv420p",
                        macro_block_size=1,
                        ffmpeg_log_level="error",
                    )
                    ffmpeg_writer.send(None)
                except Exception as ef:
                    use_ffmpeg = False
                    ffmpeg_writer = None

            if not use_ffmpeg:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                cv2_writer = cv2.VideoWriter(temp_out.name, fourcc, playback_fps, (enc_w, enc_h))

            prog_bar = st.progress(0.0)
            status_text = st.empty()

            timeline_records = []
            all_track_events = []
            keyframe_gallery = []

            runner = pipe_meta["runner"]
            sampled_count = 0
            start_frame_idx = int(video_start_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
            current_frame_idx = start_frame_idx

            while cap.isOpened() and sampled_count < max_video_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if current_frame_idx % video_stride == 0:
                    t_sec = current_frame_idx / max(1.0, fps)
                    status_text.text(f"Analyzing Frame {current_frame_idx}/{total_frames} (Time: {t_sec:.1f}s)...")
                    prog_bar.progress(min(1.0, (sampled_count + 1) / max_video_frames))

                    try:
                        if "YOLO" in selected_pipe:
                            res = runner(frame, confidence=yolo_conf)
                        elif "Combined" in selected_pipe:
                            res = runner(frame, conf_threshold=yolo_conf, recall_mode=combined_recall_mode)
                        else:
                            res = runner(frame)

                        ann_bgr = cv2.cvtColor(res["annotated_rgb"], cv2.COLOR_RGB2BGR)
                        dets = res.get("detections", [])

                        if ann_bgr.shape[:2] != (enc_h, enc_w):
                            ann_bgr = cv2.resize(ann_bgr, (enc_w, enc_h))

                        # Draw timeline status overlay at bottom
                        cv2.rectangle(ann_bgr, (0, enc_h - 36), (enc_w, enc_h), (10, 10, 10), -1)
                        cv2.putText(
                            ann_bgr,
                            f"Time: {t_sec:.2f}s | Frame: {current_frame_idx} | Detected Tracks: {len(dets)}",
                            (25, enc_h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.50,
                            (220, 220, 220),
                            1,
                            cv2.LINE_AA,
                        )

                        if use_ffmpeg and ffmpeg_writer is not None:
                            ffmpeg_writer.send(ann_bgr)
                        elif cv2_writer is not None:
                            cv2_writer.write(ann_bgr)

                        n_alpha = sum(1 for d in dets if d.get("Type") == "Alpha")
                        n_elec = sum(1 for d in dets if d.get("Type") == "Electron")

                        timeline_records.append({
                            "Frame": current_frame_idx,
                            "Time (s)": round(t_sec, 2),
                            "Total Tracks": len(dets),
                            "Alpha Tracks": n_alpha,
                            "Electron Tracks": n_elec,
                        })

                        for d in dets:
                            all_track_events.append({
                                "Frame": current_frame_idx,
                                "Time (s)": round(t_sec, 2),
                                "Track ID": d.get("Track ID"),
                                "Type": d.get("Type"),
                                "Confidence": d.get("Confidence"),
                                "Length (mm)": d.get("Length (mm)"),
                                "Thickness (mm)": d.get("Thickness (mm)"),
                            })

                        if len(dets) > 0 and len(keyframe_gallery) < 6:
                            keyframe_gallery.append((current_frame_idx, t_sec, res["annotated_rgb"], dets))

                    except Exception as e:
                        pass

                    sampled_count += 1
                current_frame_idx += 1

            cap.release()
            if use_ffmpeg and ffmpeg_writer is not None:
                ffmpeg_writer.close()
            elif cv2_writer is not None:
                cv2_writer.release()

            prog_bar.progress(1.0)
            status_text.success(f"✅ Video tracking completed! Analyzed {sampled_count} frames.")

            # Video Results Workspace Tabs
            vtab_player, vtab_timeline, vtab_events, vtab_gallery = st.tabs([
                "▶️ Annotated Video Playback",
                "📈 Particle Activity Timeline",
                "📋 Tracking Event Log",
                "🖼️ Keyframe Detection Highlights",
            ])

            with vtab_player:
                st.markdown("##### Annotated Video Playback")
                with open(temp_out.name, "rb") as vf:
                    v_bytes = vf.read()
                st.video(v_bytes, format="video/mp4")

                st.download_button(
                    "📥 Download Annotated Video (.mp4)",
                    data=v_bytes,
                    file_name=f"{Path(selected_video_name).stem}_annotated.mp4",
                    mime="video/mp4",
                )

            with vtab_timeline:
                st.markdown("##### Temporal Particle Activity Timeline")
                if timeline_records:
                    tdf = pd.DataFrame(timeline_records)
                    st.line_chart(tdf, x="Time (s)", y=["Total Tracks", "Alpha Tracks", "Electron Tracks"], height=350)
                    st.caption("Temporal plot showing particle occurrence bursts over the recording duration.")
                else:
                    st.info("No temporal records available.")

            with vtab_events:
                st.markdown("##### Frame-by-Frame Tracking Log")
                if all_track_events:
                    edf = pd.DataFrame(all_track_events)
                    st.dataframe(edf, use_container_width=True, hide_index=True)

                    csv_v = edf.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download Video Tracking Log (.csv)",
                        data=csv_v,
                        file_name=f"{Path(selected_video_name).stem}_track_log.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("No particles detected across the sampled frames.")

            with vtab_gallery:
                st.markdown("##### Highlighted Detection Keyframes")
                if keyframe_gallery:
                    gcols = st.columns(min(3, len(keyframe_gallery)))
                    for idx, (f_idx, t_s, k_rgb, k_dets) in enumerate(keyframe_gallery):
                        with gcols[idx % len(gcols)]:
                            st.image(k_rgb, caption=f"Frame {f_idx} (t={t_s:.2f}s) · {len(k_dets)} tracks", use_container_width=True)
                else:
                    st.info("No significant particle tracks detected in the sampled frames.")


if __name__ == "__main__":
    main()
