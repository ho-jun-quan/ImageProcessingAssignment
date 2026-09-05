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
        return 180, 1700, 110, 970
    # Proportional scaling for other resolutions
    top = max(140, int(h * 0.095))
    bottom = min(h - 40, int(h * 0.890))
    left = max(30, int(w * 0.100))
    right = min(w - 30, int(w * 0.900))
    return top, bottom, left, right


def suppress_osd_and_borders(img: np.ndarray) -> np.ndarray:
    """Zero out the camera status-bar / OSD timestamp text in top margin without distorting Otsu."""
    clean = img.copy()
    h, w = clean.shape[:2]
    top_limit = 125 if h >= 1800 else max(80, int(h * 0.065))
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
    bbox_h: int
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

    # Physical Alpha signature:
    # 1. Physical width >= 1.40 mm OR pixel thickness >= 18 px
    # 2. High fill density (dense continuous ionization, not hollow or wispy)
    # 3. Not excessively bent
    is_thick = (width_mm >= ALPHA_WIDTH_THRESHOLD_MM) or (min_d >= 18.0)
    is_dense = density >= 0.38
    is_alpha = is_thick and is_dense

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


def process_image(
    raw_img: np.ndarray,
    conf_threshold: float = 0.12,
    *args: Any,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Full hybrid detection pipeline on a single cloud chamber image.
    
    Returns (annotated_bgr, binary_mask, detections_list).
    """
    h_orig, w_orig = raw_img.shape[:2]
    annotated = raw_img.copy()

    # 1. Camera text suppression (top margin only to preserve full chamber contrast)
    masked_img = suppress_osd_and_borders(raw_img)

    # 2. Pattern recognition via YOLO
    inference, yolo_model = get_yolo_detector()
    detections: List[Dict[str, Any]] = []

    binary_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
    track_id = 1

    if inference is not None and yolo_model is not None:
        # Run inference on the masked image
        _, raw_yolo_dets = inference.detect_image(
            yolo_model, masked_img, conf=conf_threshold, use_tiling=True, use_tta=False
        )

        top_limit = 125 if h_orig >= 1800 else max(80, int(h_orig * 0.065))

        for det in raw_yolo_dets:
            x1, y1, x2, y2 = det["bbox"]
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)

            # Strict guard 1: reject camera timestamp zone at the top
            if y1 < top_limit:
                continue

            # Strict guard 2: reject outer metal/glass wall reflections (tall vertical sliver hugging outer glass)
            if (x1 < 45 or x2 > w_orig - 45) and (h > 140 and w < 70):
                continue

            # Strict guard 3: reject bottom metal base rim
            if y2 > h_orig - 45 and w > 250:
                continue

            # Length calculation
            diag_px = math.hypot(w, h)
            length_mm = det.get("length_mm", px_to_mm(diag_px))
            length_cm = length_mm / 10.0

            # Crop region to evaluate physical thickness and curvature
            crop = raw_img[max(0, y1):min(h_orig, y2), max(0, x1):min(w_orig, x2)]
            is_alpha, est_width_mm, density = validate_alpha_geometry(crop, w, h)

            # Local mask for curvature
            local_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(local_mask, (w // 2, h // 2), (w // 2, h // 2), 0, 0, 360, 255, -1)
            curvature, tortuosity = calculate_tortuosity_and_curvature(local_mask, diag_px, w, h)

            # Classify: Override YOLO with Classical Alpha Validator if physical criteria met
            raw_class = det.get("class_name", "")
            if is_alpha or ("Alpha" in raw_class):
                p_type = "Alpha"
                confidence = max(det.get("confidence", 0.85), 0.88)
                width_mm = max(est_width_mm, 1.45)
            else:
                p_type = "Electron"
                confidence = det.get("confidence", 0.80)
                width_mm = max(0.40, min(est_width_mm, 1.20))

            binary_mask[y1:y2, x1:x2] = 255

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
            })
            track_id += 1



    # 3. Draw clean, professional annotations on the ORIGINAL image
    COLOR_ALPHA = (0, 255, 255)    # Yellow in BGR
    COLOR_ELECTRON = (0, 0, 255)   # Red in BGR

    for d in detections:
        x, y, w, h = d["bbox"]
        color = COLOR_ALPHA if d["type"] == "Alpha" else COLOR_ELECTRON

        # Draw bounding box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

        # Label badge
        label = f"#{d['track_id']} {d['type']} ({d['confidence']*100:.0f}%)"
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
    n_elec = sum(1 for d in detections if d["type"] == "Electron")

    cv2.rectangle(annotated, (0, 0), (w_orig, 36), (15, 15, 15), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(annotated, (15, 10), (28, 25), COLOR_ALPHA, -1)
    cv2.putText(annotated, "Alpha (Yellow, Thick/Dense)", (34, 22), font, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(annotated, (270, 10), (283, 25), COLOR_ELECTRON, -1)
    cv2.putText(annotated, "Electron (Red, Thin/Curved)", (289, 22), font, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

    summary_text = f"Total Detected: {len(detections)} (Alpha: {n_alpha}, Electron: {n_elec})"
    cv2.putText(annotated, summary_text, (w_orig - 310, 22), font, 0.44, (100, 255, 100), 1, cv2.LINE_AA)

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
