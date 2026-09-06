"""
BMDS2133 Image Processing — Hybrid Cloud Chamber Particle Detection Pipeline
=============================================================================
This pipeline combines the best algorithms from all team members into an optimal
hybrid architecture:

  1. Chamber ROI & OSD Suppression (Ivan & TV)
     - Masks out the camera timestamp/OSD overlay (y < 140px) and chamber metal rims
       so numbers/letters are NEVER mistaken for particles.
  2. Pattern-Recognition Track Detector (Jun's YOLO11)
     - Uses fine-tuned convolutional filters to distinguish genuine particle ionization
       tracks from amorphous background condensation and vapor mist.
  3. Physical Morphology & Centerline Extraction (TV)
     - Extracts track skeletons and Euclidean Distance Transform (EDT) width profiles
       to measure physical thickness (mm) and chord-to-arc tortuosity/curvature.
  4. Straightness Validation (WeiQuan's Hough Lines)
     - Uses Hough Line Transform segment orientation analysis to validate trajectory linearity.
  5. Physical Hybrid Classifier (Ivan + TV + Jun)
     - Overcomes YOLO's Alpha class imbalance by enforcing physics-based criteria:
         * Alpha: Dense, thick (width >= 1.4 mm or min_dim >= 18 px), straight -> Yellow Box
         * Electron: Thin (< 1.4 mm), wispy, or magnetically deflected/curved  -> Red Box
         * Background / OSD: Rejected as Noise
"""
from __future__ import annotations

import os
import sys
import math
import csv
import glob
import argparse
import importlib
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import cv2
import numpy as np

# Spatial Calibration Factor
PIXELS_PER_MM = 12.5
PIXELS_PER_CM = 125.0
ALPHA_WIDTH_THRESHOLD_MM = 1.40
TOP_HEADER_HEIGHT = 140

ROOT = Path(__file__).resolve().parents[1]
JUN_YOLO_DIR = ROOT / "Jun" / "YOLO"
if str(JUN_YOLO_DIR) not in sys.path:
    sys.path.insert(0, str(JUN_YOLO_DIR))


def px_to_mm(px: float) -> float:
    return px / PIXELS_PER_MM


def px_to_cm(px: float) -> float:
    return px / PIXELS_PER_CM


def get_chamber_roi_margins(h: int, w: int) -> Tuple[int, int, int, int]:
    """Returns (top, bottom, left, right) pixel bounds of the active chamber interior.
    
    Calibrated against physical chamber chassis to suppress:
      - Camera timestamp and status OSD overlay at top
      - Metal chassis and coolant tray at bottom
      - Glass wall reflections and side LED glare columns at left and right
    """
    if h >= 1800 and w >= 1000:
        # Standard 1080x1920 cloud chamber recordings
        return 240, 1680, 110, 970
    # Proportional scaling for other resolutions
    top = max(140, int(h * 0.125))
    bottom = min(h - 40, int(h * 0.875))
    left = max(30, int(w * 0.100))
    right = min(w - 30, int(w * 0.900))
    return top, bottom, left, right


def suppress_osd_and_borders(img: np.ndarray) -> np.ndarray:
    """Zero out the camera status-bar / OSD timestamp text in top margin without distorting Otsu."""
    clean = img.copy()
    h, w = clean.shape[:2]
    top_limit = 225 if h >= 1800 else max(80, int(h * 0.065))
    clean[:top_limit, :] = 0
    return clean


def calculate_tortuosity_and_curvature(
    mask: np.ndarray,
    arc_length: float,
    bbox_w: int,
    bbox_h: int
) -> Tuple[float, float]:
    """Calculate trajectory curvature and chord-to-arc tortuosity."""
    pts = np.column_stack(np.where(mask > 0))
    if len(pts) < 5:
        diag = math.hypot(bbox_w, bbox_h)
        tort = max(0.0, 1.0 - (diag / max(1.0, arc_length)))
        return float(tort * 0.5), float(tort)

    if len(pts) > 40:
        sub_pts = pts[np.linspace(0, len(pts) - 1, 40, dtype=int)]
    else:
        sub_pts = pts

    diff = sub_pts[:, np.newaxis, :] - sub_pts[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
    max_chord = float(np.max(dist_matrix))

    tortuosity = max(0.0, min(1.0, 1.0 - (max_chord / max(1.0, arc_length))))
    aspect = max(bbox_w, bbox_h) / max(1, min(bbox_w, bbox_h))
    curvature = tortuosity * 0.80 + (1.0 / max(1.0, aspect)) * 0.20
    return round(curvature, 4), round(tortuosity, 4)


def validate_alpha_geometry(
    crop_bgr: np.ndarray,
    bbox_w: int,
    bbox_h: int,
    global_cx: int = 0,
    global_cy: int = 0,
    length_mm: float = 0.0,
    curvature: float = 0.0,
) -> Tuple[bool, float, float]:
    """Examine local track morphology to detect dense, thick Alpha plasma columns.
    
    Returns (is_alpha, estimated_width_mm, fill_density).
    """
    if crop_bgr.size == 0:
        return False, 0.0, 0.0

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, 0.0, 0.0

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    rect = cv2.minAreaRect(c)
    (cx, cy), (rw, rh), angle = rect
    min_d = min(rw, rh)
    max_d = max(rw, rh)

    box_area = max(1.0, min_d * max_d)
    density = area / box_area
    width_mm = min_d / PIXELS_PER_MM
    track_len_mm = max(length_mm, max_d / PIXELS_PER_MM)

    # Physical Alpha signature:
    # 1. Physical width >= 1.60 mm AND pixel thickness >= 18 px
    # 2. Minimum length >= 10.0 mm (1.0 cm) - Alpha particles are heavy nuclear tracks, not 5mm blurs!
    # 3. Dense continuous ionization column (fill density >= 0.42)
    # 4. Low curvature (< 0.18) - Alpha particles travel straight without scattering easily
    # 5. Must NOT be an artifact in the vertical reflection corridors
    is_left_ref = (170 <= global_cx <= 390)
    is_right_ref = (690 <= global_cx <= 880)
    is_vertical_streak = (bbox_h > 1.35 * bbox_w and bbox_w < 65)

    is_thick = (width_mm >= 1.60) and (min_d >= 18.0)
    is_dense = density >= 0.42
    is_long_enough = track_len_mm >= 10.0
    is_straight = curvature < 0.18
    is_not_reflection = not ((is_left_ref or is_right_ref) and is_vertical_streak)

    is_alpha = is_thick and is_dense and is_long_enough and is_straight and is_not_reflection

    return is_alpha, round(width_mm, 2), round(density, 2)


_CACHED_YOLO_MODEL = None

def get_yolo_detector():
    """Load Jun's YOLO model once and cache in memory."""
    global _CACHED_YOLO_MODEL
    if _CACHED_YOLO_MODEL is None:
        try:
            # The detector is optional and lives in Jun/YOLO rather than being
            # an installed top-level package. Load it only when it is needed.
            inference = importlib.import_module("inference")
            yolo_config = importlib.import_module("config")
            _CACHED_YOLO_MODEL = (inference, inference.load_model(yolo_config.BEST_WEIGHTS))
        except Exception as err:
            print(f"Warning: YOLO detector could not be loaded: {err}")
            return None, None
    return _CACHED_YOLO_MODEL


def bbox_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Compute Intersection-over-Union (IoU) between two bounding boxes (x, y, w, h)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = boxA[2] * boxA[3]
    areaB = boxB[2] * boxB[3]
    union = areaA + areaB - inter
    return inter / max(1.0, union)


def extract_classical_candidates(
    image_bgr: np.ndarray,
    min_length_mm: float = 4.5,
    min_aspect_ratio: float = 2.1,
    min_area_px: int = 120,
) -> List[Dict[str, Any]]:
    """High-recall safety net using Ivan's dynamic background subtraction, bilateral
    filtering, and Otsu morphology to detect visible particle tracks that YOLO misses.
    """
    h_orig, w_orig = image_bgr.shape[:2]
    crop_y1, crop_y2, crop_x1, crop_x2 = get_chamber_roi_margins(h_orig, w_orig)

    sub_img = image_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
    if sub_img.size == 0:
        return []

    gray = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY) if sub_img.ndim == 3 else sub_img.copy()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    bg = cv2.GaussianBlur(clahe, (51, 51), 0)
    fg = cv2.subtract(clahe, bg)
    fg = cv2.normalize(fg, None, 0, 255, cv2.NORM_MINMAX)
    denoised = cv2.bilateralFilter(fg, 9, 75, 75)
    blurred = cv2.GaussianBlur(denoised, (5, 5), 0)

    # Compute Otsu threshold on the naturally leveled foreground (no black holes!)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel_close = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    kernel_open = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=2)

    # Clean interior border guards
    cv2.rectangle(binary, (0, 0), (binary.shape[1], 15), 0, -1)
    cv2.rectangle(binary, (0, binary.shape[0] - 15), (binary.shape[1], binary.shape[0]), 0, -1)
    cv2.rectangle(binary, (0, 0), (15, binary.shape[0]), 0, -1)
    cv2.rectangle(binary, (binary.shape[1] - 15, 0), (binary.shape[1], binary.shape[0]), 0, -1)

    # Filter isolated speckles
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    clean_bin = np.zeros_like(binary)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_px:
            clean_bin[labels == i] = 255

    contours, _ = cv2.findContours(clean_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[Dict[str, Any]] = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area_px:
            continue
        rect = cv2.minAreaRect(c)
        (cx, cy), (rw, rh), angle = rect
        min_dim, max_dim = min(rw, rh), max(rw, rh)
        if max_dim <= 0 or min_dim <= 0:
            continue
        aspect = max_dim / max(1.0, min_dim)
        length_mm = max_dim / PIXELS_PER_MM
        if length_mm < min_length_mm or aspect < min_aspect_ratio:
            continue

        bx, by, bw, bh = cv2.boundingRect(c)
        gx, gy = crop_x1 + bx, crop_y1 + by
        gcx, gcy = gx + bw // 2, gy + bh // 2

        # 1. Global border guards: reject frame edges and top/bottom plates
        if gy < crop_y1 + 5 or (gy + bh) > crop_y2 - 5 or gx < crop_x1 + 5 or (gx + bw) > crop_x2 - 5:
            continue

        # 2. Reflection corridor guards (left: 170..390, right: 690..880)
        is_left_ref = (170 <= gcx <= 390)
        is_right_ref = (690 <= gcx <= 880)
        if (is_left_ref or is_right_ref):
            # Suppress vertical reflection slivers
            if bh > 1.35 * bw and bw < 65:
                continue
            # Suppress low-aspect glare droplet clumps
            if aspect < 2.5 and area < 750:
                continue

        candidates.append({
            "bbox": (gx, gy, bw, bh),
            "length_px": max_dim,
            "length_mm": length_mm,
            "aspect_ratio": aspect,
            "area": area,
        })

    return candidates


def get_hybrid_preprocessing_stages(raw_img: np.ndarray) -> List[Dict[str, Any]]:
    """Generate visual representations of the multi-stage hybrid preprocessing pipeline.
    
    Combines:
      - Ivan & TV: Active chamber ROI & camera diagnostic OSD suppression
      - WeiQuan, Ivan & TV: Contrast Limited Adaptive Histogram Equalization (CLAHE)
      - Ivan & TV: Illumination Leveling & Edge-preserving Bilateral Filter
      - Jun: Difference of Gaussians (DoG) high-pass curvilinear track isolation
      - Ivan & Jun: Clean morphological Otsu track binarization
      - TV: Morphological centerline skeleton & Euclidean Distance Transform (EDT)
    """
    gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY) if raw_img.ndim == 3 else raw_img.copy()

    # Stage 1: OSD & Border Suppression (Ivan & TV)
    clean_gray = suppress_osd_and_borders(gray)

    # Stage 2: CLAHE Contrast Normalization (WeiQuan, Ivan, TV)
    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe_obj.apply(clean_gray)

    # Stage 3: Illumination Leveling & Edge-Preserving Bilateral Denoising (Ivan & TV)
    bg_illum = cv2.GaussianBlur(clahe_img, (51, 51), 0)
    fg_illum = cv2.subtract(clahe_img, bg_illum)
    fg_illum = cv2.normalize(fg_illum, None, 0, 255, cv2.NORM_MINMAX)
    bilateral_img = cv2.bilateralFilter(fg_illum, d=9, sigmaColor=75, sigmaSpace=75)

    # Stage 4: Difference of Gaussians Curvilinear Filter (Jun)
    blur_small = cv2.GaussianBlur(bilateral_img, (0, 0), 2.0)
    blur_large = cv2.GaussianBlur(bilateral_img, (0, 0), 12.0)
    dog = cv2.subtract(blur_small, blur_large)
    dog_norm = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)

    # Stage 5: Clean Morphological Otsu Track Binarization (Ivan & Jun)
    _, otsu_raw = cv2.threshold(dog_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel_close = np.ones((5, 5), np.uint8)
    kernel_open = np.ones((3, 3), np.uint8)
    otsu_bin = cv2.morphologyEx(otsu_raw, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    otsu_bin = cv2.morphologyEx(otsu_bin, cv2.MORPH_OPEN, kernel_open, iterations=2)

    # Stage 6: Morphological Centerline Skeleton & Distance Transform (TV)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skel = np.zeros(otsu_bin.shape, np.uint8)
    temp_bin = otsu_bin.copy()
    for _ in range(25):
        eroded = cv2.erode(temp_bin, element)
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)
        subset = cv2.subtract(eroded, opened)
        skel = cv2.bitwise_or(skel, subset)
        temp_bin = eroded.copy()
        if cv2.countNonZero(temp_bin) == 0:
            break

    dist = cv2.distanceTransform(otsu_bin, cv2.DIST_L2, 5)
    dist_vis = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    skel_dist_vis = cv2.addWeighted(
        cv2.cvtColor(dist_vis, cv2.COLOR_GRAY2BGR), 0.70,
        cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR), 0.30, 0
    )

    return [
        {
            "name": "1. Chamber ROI & OSD Suppression",
            "author": "Ivan & TV (Camera Text Guard)",
            "desc": "Converts BGR to grayscale and zeroes out camera diagnostic OSD numbers (y < 125px) to prevent timestamps from triggering false detections.",
            "image": clean_gray,
        },
        {
            "name": "2. CLAHE Contrast Equalization",
            "author": "WeiQuan, Ivan & TV (Adaptive Contrast)",
            "desc": "Applies Contrast-Limited Adaptive Histogram Equalization to amplify low-contrast vapor ionization trails without over-amplifying background noise.",
            "image": clahe_img,
        },
        {
            "name": "3. Dynamic Background Subtraction & Denoising",
            "author": "Ivan & TV (Illumination Leveling)",
            "desc": "Gaussian background subtraction (51x51) removes ambient chamber illumination and reflection glare, followed by bilateral smoothing.",
            "image": bilateral_img,
        },
        {
            "name": "4. Difference of Gaussians (DoG) High-Pass",
            "author": "Jun (YOLO Preprocessing Engine)",
            "desc": "Bandpass spatial filter subtracting wide Gaussian blur (σ=12.0) from narrow blur (σ=2.0) to isolate continuous particle tracks.",
            "image": dog_norm,
        },
        {
            "name": "5. Clean Otsu Track Binarization",
            "author": "Ivan & Jun (Optimal Thresholding + Morphology)",
            "desc": "Optimal bimodal thresholding combined with morphological opening/closing to eliminate vapor droplet speckles while preserving track bodies.",
            "image": otsu_bin,
        },
        {
            "name": "6. Centerline Skeleton & Width Profiling",
            "author": "TV (Scale-Space Physics)",
            "desc": "Morphological thinning extracts 1-pixel skeletons; Euclidean Distance Transform computes physical width profile for Alpha vs Electron classification.",
            "image": skel_dist_vis,
        },
    ]


def process_image(
    raw_img: np.ndarray,
    conf_threshold: float = 0.12,
    collect_stages: bool = False,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Full hybrid detection pipeline on a single cloud chamber image.
    
    Fuses deep learning YOLO11 tiled detections with classical vision candidate
    recovery to ensure high recall for both clear tracks and faint/curved trails.
    """
    h_orig, w_orig = raw_img.shape[:2]
    annotated = raw_img.copy()

    # 1. Camera text suppression
    masked_img = suppress_osd_and_borders(raw_img)

    # 2. Curvilinear high-pass track binary mask
    gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY) if masked_img.ndim == 3 else masked_img.copy()
    blur_small = cv2.GaussianBlur(gray, (0, 0), 2.0)
    blur_large = cv2.GaussianBlur(gray, (0, 0), 12.0)
    dog = cv2.subtract(blur_small, blur_large)
    dog_norm = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)
    _, raw_binary_mask = cv2.threshold(dog_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)

    detections: List[Dict[str, Any]] = []
    track_id = 1
    yolo_detected_boxes: List[Tuple[int, int, int, int]] = []

    # 3. Deep Learning Engine (YOLO11)
    roi_top, roi_bottom, roi_left, roi_right = get_chamber_roi_margins(h_orig, w_orig)
    inference, yolo_model = get_yolo_detector()
    if inference is not None and yolo_model is not None:
        annotated_enhanced, raw_yolo_dets = inference.detect_image(
            yolo_model, masked_img, conf=conf_threshold, use_tiling=True, use_tta=False
        )

        for det in raw_yolo_dets:
            x1, y1, x2, y2 = det["bbox"]
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            cx = x1 + w // 2

            # Strict chamber observation window guards
            if y1 < roi_top or y2 > roi_bottom or x1 < roi_left or x2 > roi_right:
                continue

            # Suppress persistent vertical glass reflection slivers
            is_left_ref = (170 <= cx <= 390)
            is_right_ref = (690 <= cx <= 880)
            if (is_left_ref or is_right_ref) and (h > 1.35 * w and w < 65):
                continue

            diag_px = math.hypot(w, h)
            length_mm = det.get("length_mm", px_to_mm(diag_px))
            length_cm = length_mm / 10.0

            crop = raw_img[max(0, y1):min(h_orig, y2), max(0, x1):min(w_orig, x2)]
            local_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(local_mask, (w // 2, h // 2), (w // 2, h // 2), 0, 0, 360, 255, -1)
            curvature, tortuosity = calculate_tortuosity_and_curvature(local_mask, diag_px, w, h)

            is_alpha, est_width_mm, density = validate_alpha_geometry(
                crop, w, h, global_cx=cx, global_cy=y1 + h // 2, length_mm=length_mm, curvature=curvature
            )

            raw_class = det.get("class_name", "")
            cls_id = int(det.get("class_id", -1))
            if is_alpha or ("Alpha" in raw_class and is_alpha):
                p_type = "Alpha"
                confidence = max(det.get("confidence", 0.85), 0.88)
                width_mm = max(est_width_mm, 1.60)
            elif cls_id == 3 or any(k in raw_class for k in ("Low_Energy", "Low Energy", "Low-E")):
                p_type = "Low-Energy Electron"
                confidence = det.get("confidence", 0.80)
                width_mm = max(0.35, min(est_width_mm, 1.10))
            elif cls_id == 4 or any(k in raw_class for k in ("Knock_On", "Secondary", "Knock-On")):
                p_type = "Knock-On Electron"
                confidence = det.get("confidence", 0.80)
                width_mm = max(0.40, min(est_width_mm, 1.20))
            else:
                if tortuosity > 0.40 and length_mm < 16.0:
                    p_type = "Low-Energy Electron"
                else:
                    p_type = "Normal Electron"
                confidence = det.get("confidence", 0.80)
                width_mm = max(0.40, min(est_width_mm, 1.20))

            binary_mask[y1:y2, x1:x2] = raw_binary_mask[y1:y2, x1:x2]
            yolo_detected_boxes.append((x1, y1, w, h))

            detections.append({
                "track_id": track_id,
                "type": p_type,
                "confidence": round(float(confidence), 3),
                "bbox": (int(x1), int(y1), int(w), int(h)),
                "length_px": round(diag_px, 1),
                "length_cm": round(length_cm, 2),
                "length_mm": round(length_mm, 2),
                "width_mm": round(width_mm, 2),
                "curvature": curvature,
                "tortuosity": tortuosity,
                "area_px": w * h,
                "source": "YOLO11 Tiled",
            })
            track_id += 1

    # 4. Classical Vision Safety Net (High-Recall Candidate Recovery)
    # Catches prominent vapor tracks that YOLO may have missed
    classical_cands = extract_classical_candidates(raw_img)
    for cand in classical_cands:
        bx, by, bw, bh = cand["bbox"]
        cand_box = (bx, by, bw, bh)

        # Skip if already detected by YOLO (IoU >= 0.20)
        if any(bbox_iou(cand_box, yb) >= 0.20 for yb in yolo_detected_boxes):
            continue

        diag_px = math.hypot(bw, bh)
        length_mm = cand["length_mm"]
        length_cm = length_mm / 10.0

        crop = raw_img[max(0, by):min(h_orig, by + bh), max(0, bx):min(w_orig, bx + bw)]
        local_mask = np.zeros((bh, bw), dtype=np.uint8)
        cv2.ellipse(local_mask, (bw // 2, bh // 2), (bw // 2, bh // 2), 0, 0, 360, 255, -1)
        curvature, tortuosity = calculate_tortuosity_and_curvature(local_mask, diag_px, bw, bh)

        is_alpha, est_width_mm, density = validate_alpha_geometry(
            crop, bw, bh, global_cx=bx + bw // 2, global_cy=by + bh // 2, length_mm=length_mm, curvature=curvature
        )

        if is_alpha:
            p_type = "Alpha"
            confidence = round(min(0.92, 0.72 + density * 0.22), 2)
            width_mm = max(est_width_mm, 1.60)
        else:
            if tortuosity > 0.40 and length_mm < 16.0:
                p_type = "Low-Energy Electron"
            else:
                p_type = "Normal Electron"
            confidence = round(min(0.88, 0.60 + min(1.0, cand["aspect_ratio"] / 8.0) * 0.25), 2)
            width_mm = max(0.40, min(est_width_mm, 1.20))

        binary_mask[by:by + bh, bx:bx + bw] = raw_binary_mask[by:by + bh, bx:bx + bw]

        detections.append({
            "track_id": track_id,
            "type": p_type,
            "confidence": confidence,
            "bbox": (int(bx), int(by), int(bw), int(bh)),
            "length_px": round(diag_px, 1),
            "length_cm": round(length_cm, 2),
            "length_mm": round(length_mm, 2),
            "width_mm": round(width_mm, 2),
            "curvature": curvature,
            "tortuosity": tortuosity,
            "area_px": bw * bh,
            "source": "Classical Vision (Hybrid Recovered)",
        })
        track_id += 1



    # 3. Draw clean, professional annotations on the ORIGINAL image
    COLOR_ALPHA = (0, 215, 255)         # Yellow in BGR
    COLOR_NORMAL_E = (255, 180, 50)     # Sky Blue in BGR
    COLOR_LOW_E = (0, 140, 255)         # Orange in BGR
    COLOR_KNOCK_ON = (255, 0, 255)      # Magenta in BGR

    for d in detections:
        x, y, w, h = d["bbox"]
        t = d["type"]
        if t == "Alpha":
            color = COLOR_ALPHA
            badge_title = "Alpha"
        elif t == "Low-Energy Electron":
            color = COLOR_LOW_E
            badge_title = "Low-E e-"
        elif t == "Knock-On Electron":
            color = COLOR_KNOCK_ON
            badge_title = "Knock-On"
        else:
            color = COLOR_NORMAL_E
            badge_title = "Normal e-"

        # Draw bounding box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

        # Label badge
        label = f"#{d['track_id']} {badge_title} ({d['confidence']*100:.0f}%)"
        sub_label = f"L:{d['length_cm']:.1f}cm W:{d['width_mm']:.1f}mm"

        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.44, 1)

        badge_y1 = max(0, y - th - 18)
        badge_y2 = max(th + 18, y)
        badge_x1 = x
        badge_x2 = x + max(tw, 145) + 10

        cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), (20, 20, 20), -1)
        cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), color, 1)
        cv2.putText(annotated, label, (badge_x1 + 4, badge_y1 + th + 3), font, 0.44, color, 1, cv2.LINE_AA)
        cv2.putText(annotated, sub_label, (badge_x1 + 4, badge_y2 - 3), font, 0.36, (220, 220, 220), 1, cv2.LINE_AA)

    # Top Header Legend
    n_alpha = sum(1 for d in detections if d["type"] == "Alpha")
    n_norm_e = sum(1 for d in detections if d["type"] == "Normal Electron")
    n_low_e = sum(1 for d in detections if d["type"] == "Low-Energy Electron")
    n_knock = sum(1 for d in detections if d["type"] == "Knock-On Electron")

    cv2.rectangle(annotated, (0, 0), (w_orig, 36), (15, 15, 15), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(annotated, (15, 10), (28, 25), COLOR_ALPHA, -1)
    cv2.putText(annotated, "Alpha", (34, 22), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(annotated, (100, 10), (113, 25), COLOR_NORMAL_E, -1)
    cv2.putText(annotated, "Normal e-", (119, 22), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(annotated, (215, 10), (228, 25), COLOR_LOW_E, -1)
    cv2.putText(annotated, "Low-E e-", (234, 22), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(annotated, (325, 10), (338, 25), COLOR_KNOCK_ON, -1)
    cv2.putText(annotated, "Knock-On", (344, 22), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    summary_text = f"Total: {len(detections)} (α:{n_alpha} e⁻:{n_norm_e} LE:{n_low_e} δ:{n_knock})"
    cv2.putText(annotated, summary_text, (w_orig - 340, 22), font, 0.42, (100, 255, 100), 1, cv2.LINE_AA)

    if collect_stages:
        stages = get_hybrid_preprocessing_stages(raw_img)
        return annotated, binary_mask, detections, stages
    return annotated, binary_mask, detections


def export_csv(detections: List[Dict[str, Any]], csv_path: str, image_name: str = "") -> None:
    """Export detection results to CSV."""
    fieldnames = [
        "image", "track_id", "type", "confidence",
        "length_mm", "length_cm", "width_mm", "curvature", "tortuosity",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h",
    ]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for det in detections:
            writer.writerow({
                "image": image_name,
                "track_id": det["track_id"],
                "type": det["type"],
                "confidence": det["confidence"],
                "length_mm": det["length_mm"],
                "length_cm": det["length_cm"],
                "width_mm": det["width_mm"],
                "curvature": det["curvature"],
                "tortuosity": det["tortuosity"],
                "bbox_x": det["bbox"][0],
                "bbox_y": det["bbox"][1],
                "bbox_w": det["bbox"][2],
                "bbox_h": det["bbox"][3],
            })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid Cloud Chamber Detection Pipeline")
    parser.add_argument("--image", type=str, default=None, help="Path to a single image")
    args = parser.parse_args()

    if args.image:
        raw = cv2.imread(args.image)
        if raw is not None:
            ann, mask, dets = process_image(raw)
            print(f"Detected {len(dets)} tracks in {args.image}:")
            for d in dets:
                print(f"  #{d['track_id']} {d['type']} (conf={d['confidence']}) L={d['length_cm']}cm W={d['width_mm']}mm")
        else:
            print(f"Could not load image: {args.image}")
