"""Report-ready cloud-chamber trajectory pipeline based on HoughLinesP.

The output is geometric trajectory evidence only.  This module intentionally
does not label a track as an alpha particle or an electron.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "image"
RESULT_DIR = ROOT / "result"
ROI_BOX_FILE = ROOT / "roi_boxes.csv"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Preprocessing and Canny parameters.
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)
SMOOTHING_KERNEL_SIZE = 5
BACKGROUND_SIGMA = 18.0
CANNY_LOW_THRESHOLD = 30
CANNY_HIGH_THRESHOLD = 100
SIGNAL_PERCENTILE = 98.5
SIGNAL_MASK_DILATION = 3
EDGE_MORPHOLOGY_KERNEL_SIZE = 3
EDGE_CLOSE_ITERATIONS = 1
EDGE_OPEN_ITERATIONS = 1
MIN_EDGE_COMPONENT_PIXELS = 10

# Probabilistic Hough Transform parameters.
HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 28
MIN_LINE_LENGTH = 24
MAX_LINE_GAP = 12

# Post-Hough cleanup and grouping parameters.
DUPLICATE_DISTANCE_PX = 14.0
DUPLICATE_ANGLE_TOLERANCE_DEG = 8.0
TRAJECTORY_JOIN_DISTANCE_PX = 32.0
TRAJECTORY_JOIN_ANGLE_TOLERANCE_DEG = 28.0
TRAJECTORY_PARALLEL_ANGLE_TOLERANCE_DEG = 14.0
ISOLATED_ORIENTATION_DISTANCE_PX = 48.0
ISOLATED_ORIENTATION_TOLERANCE_DEG = 32.0
STRONG_ISOLATED_LENGTH_FACTOR = 1.6

# Evidence thresholds.  Curvature is median local direction change per pixel.
MIN_SEGMENTS_FOR_CURVE = 3
CURVATURE_THRESHOLD_RAD_PER_PX = 0.005
CURVATURE_ANGLE_SUPPORT_DEG = 10.0
STRAIGHTNESS_THRESHOLD = 0.90
CURVED_STRAIGHTNESS_MAX = 0.985
STRAIGHT_ANGLE_SUPPORT_DEG = 8.0

BOTTOM_BORDER_MARGIN_RATIO = 0.08
BOTTOM_CAPTION_MAX_ANGLE_DEG = 15.0
AUTO_ROI_MARGIN_RATIO = 0.05
PIXELS_PER_CM: Optional[float] = None

Measurement = Dict[str, object]
Segment = Tuple[int, int, int, int]
ROI = Tuple[int, int, int, int]

IMAGE_FEATURE_FIELDS = [
    "image_name", "roi_x", "roi_y", "roi_width", "roi_height",
    "raw_hough_segments", "filtered_segments", "trajectory_count",
    "primary_trajectory_id", "trajectory_length_px", "orientation_angle_deg",
    "straightness", "curvature_rad_per_px", "average_angle_change_deg",
    "trajectory_evidence",
]
TRAJECTORY_FEATURE_FIELDS = [
    "image_name", "trajectory_id", "segment_count", "trajectory_length_px",
    "orientation_angle_deg", "straightness", "curvature_rad_per_px",
    "average_angle_change_deg", "trajectory_evidence",
]


def calculate_length(x1: int, y1: int, x2: int, y2: int) -> float:
    return math.hypot(x2 - x1, y2 - y1)


def calculate_orientation(x1: int, y1: int, x2: int, y2: int) -> float:
    return math.degrees(math.atan2(-(y2 - y1), x2 - x1)) % 180.0


def angle_difference(angle1: float, angle2: float) -> float:
    difference = abs(float(angle1) - float(angle2)) % 180.0
    return min(difference, 180.0 - difference)


def _normalise_roi(roi: ROI, image_shape: Sequence[int]) -> ROI:
    if len(roi) != 4:
        raise ValueError("ROI must contain x, y, width, and height")
    image_height, image_width = int(image_shape[0]), int(image_shape[1])
    x, y, width, height = (int(value) for value in roi)
    if width <= 0 or height <= 0:
        raise ValueError("ROI width and height must be positive")
    if x >= image_width or y >= image_height or x + width <= 0 or y + height <= 0:
        raise ValueError("ROI does not overlap the image")
    x1 = max(0, min(x, image_width - 1))
    y1 = max(0, min(y, image_height - 1))
    x2 = min(image_width, max(x1 + 1, x + width))
    y2 = min(image_height, max(y1 + 1, y + height))
    return x1, y1, x2 - x1, y2 - y1


def _default_roi(image_shape: Sequence[int]) -> ROI:
    height, width = int(image_shape[0]), int(image_shape[1])
    x_margin = max(1, int(width * AUTO_ROI_MARGIN_RATIO))
    y_margin = max(1, int(height * AUTO_ROI_MARGIN_RATIO))
    return _normalise_roi(
        (x_margin, y_margin, width - 2 * x_margin, height - 2 * y_margin), image_shape
    )


def _parse_roi_text(text: str) -> ROI:
    values = [part.strip() for part in text.replace(";", ",").split(",")]
    if len(values) != 4:
        values = text.split()
    if len(values) != 4:
        raise ValueError("Enter four values: x,y,width,height")
    return tuple(int(float(value)) for value in values)  # type: ignore[return-value]


def _load_roi_boxes(roi_file: Path = ROI_BOX_FILE) -> Dict[str, ROI]:
    if not roi_file.exists():
        return {}
    boxes: Dict[str, ROI] = {}
    with roi_file.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row_number, row in enumerate(reader, start=2):
            image_name = row.get("image_name") or row.get("filename") or row.get("file")
            if not image_name:
                print(f"Warning: ROI row {row_number} has no image_name; skipping")
                continue
            try:
                if all(row.get(field) not in (None, "") for field in ("x", "y", "width", "height")):
                    roi = tuple(int(float(row[field])) for field in ("x", "y", "width", "height"))
                elif all(row.get(field) not in (None, "") for field in ("x1", "y1", "x2", "y2")):
                    x1, y1, x2, y2 = (int(float(row[field])) for field in ("x1", "y1", "x2", "y2"))
                    roi = (x1, y1, x2 - x1, y2 - y1)
                else:
                    raise ValueError("expected x,y,width,height or x1,y1,x2,y2")
            except (TypeError, ValueError) as error:
                print(f"Warning: invalid ROI row {row_number}: {error}; skipping")
                continue
            boxes[Path(image_name).name] = roi
            boxes[Path(image_name).stem] = roi
    return boxes


def _resolve_roi(
    image_path: Path,
    image: np.ndarray,
    roi: Optional[ROI] = None,
    roi_boxes: Optional[Dict[str, ROI]] = None,
    allow_prompt: bool = False,
) -> ROI:
    boxes = roi_boxes or {}
    candidate = roi
    if candidate is None:
        for key in (image_path.name, image_path.stem):
            candidate = boxes.get(key)
            if candidate is not None:
                break
    if candidate is not None:
        return _normalise_roi(candidate, image.shape)

    if allow_prompt:
        print(f"No ROI configured for {image_path.name}.")
        print("[M] Select ROI  [C] Enter x,y,width,height  [A] Use interior ROI")
        try:
            choice = input("ROI method (Enter=A): ").strip().lower() or "a"
        except EOFError:
            choice = "a"
        if choice == "m":
            try:
                selected = cv2.selectROI(f"Select ROI - {image_path.name}", image, True, False)
                cv2.destroyWindow(f"Select ROI - {image_path.name}")
                if selected[2] > 0 and selected[3] > 0:
                    return _normalise_roi(tuple(int(value) for value in selected), image.shape)  # type: ignore[arg-type]
            except cv2.error as error:
                print(f"Could not open ROI selector: {error}")
        elif choice == "c":
            try:
                return _normalise_roi(_parse_roi_text(input("ROI x,y,width,height: ")), image.shape)
            except (EOFError, ValueError) as error:
                print(f"Invalid ROI: {error}")

    resolved = _default_roi(image.shape)
    print(f"Using automatic interior ROI for {image_path.name}: {resolved}")
    return resolved


def _remove_small_components(binary: np.ndarray, min_pixels: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    cleaned = np.zeros_like(binary)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_pixels:
            cleaned[labels == label] = 255
    return cleaned


def preprocess_image(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return an enhanced grayscale stage and cleaned Canny edge image."""

    grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Gaussian smoothing preserves the faint, sparse cloud trails better than
    # a large median filter, while still reducing the dense camera grain in
    # the bright-background frames.
    denoised = cv2.GaussianBlur(grey, (SMOOTHING_KERNEL_SIZE, SMOOTHING_KERNEL_SIZE), 0)
    clahe = cv2.createCLAHE(CLAHE_CLIP_LIMIT, CLAHE_GRID_SIZE)
    local_contrast = clahe.apply(denoised)

    # Background subtraction suppresses broad illumination and reflection
    # patterns before Canny without replacing the required edge detector.
    background = cv2.GaussianBlur(denoised, (0, 0), BACKGROUND_SIGMA)
    high_frequency = cv2.subtract(denoised, background)
    corrected = cv2.normalize(high_frequency, None, 0, 255, cv2.NORM_MINMAX)
    enhanced = cv2.addWeighted(local_contrast, 0.65, corrected, 0.35, 0)
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)

    edges = cv2.Canny(enhanced, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (EDGE_MORPHOLOGY_KERNEL_SIZE, EDGE_MORPHOLOGY_KERNEL_SIZE)
    )
    # Gate only very dense edge maps by the strongest local response.  Sparse
    # AWAN frames already contain useful isolated trails; bright-background
    # frames need this extra suppression of widespread sensor grain.
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)
    if edge_density > 0.05:
        signal_threshold = float(np.percentile(high_frequency, SIGNAL_PERCENTILE))
        signal_mask = np.where(high_frequency >= signal_threshold, 255, 0).astype(np.uint8)
        signal_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (SIGNAL_MASK_DILATION, SIGNAL_MASK_DILATION)
        )
        signal_mask = cv2.dilate(signal_mask, signal_kernel, iterations=1)
        edges = cv2.bitwise_and(edges, signal_mask)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=EDGE_CLOSE_ITERATIONS)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=EDGE_OPEN_ITERATIONS)
    edges = _remove_small_components(edges, MIN_EDGE_COMPONENT_PIXELS)
    border = max(2, EDGE_MORPHOLOGY_KERNEL_SIZE)
    edges[:border, :] = 0
    edges[-border:, :] = 0
    edges[:, :border] = 0
    edges[:, -border:] = 0
    return enhanced, edges


def run_hough_lines_p(
    edges: np.ndarray,
    hough_threshold: int = HOUGH_THRESHOLD,
    min_line_length: int = MIN_LINE_LENGTH,
    max_line_gap: int = MAX_LINE_GAP,
) -> List[Segment]:
    """Run OpenCV's Probabilistic Hough Transform (HoughLinesP)."""

    lines = cv2.HoughLinesP(
        edges, HOUGH_RHO, HOUGH_THETA, hough_threshold,
        minLineLength=min_line_length, maxLineGap=max_line_gap,
    )
    return [] if lines is None else [tuple(map(int, line)) for line in np.asarray(lines).reshape(-1, 4)]


def _segment_midpoint(segment: Segment) -> Tuple[float, float]:
    x1, y1, x2, y2 = segment
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def _segment_endpoint_distance(first: Segment, second: Segment) -> float:
    endpoints1 = np.array([[first[0], first[1]], [first[2], first[3]]], dtype=float)
    endpoints2 = np.array([[second[0], second[1]], [second[2], second[3]]], dtype=float)
    return float(np.min(np.linalg.norm(endpoints1[:, None] - endpoints2[None, :], axis=2)))


def _segments_are_duplicate(first: Segment, second: Segment) -> bool:
    orientation_gap = angle_difference(calculate_orientation(*first), calculate_orientation(*second))
    if orientation_gap > DUPLICATE_ANGLE_TOLERANCE_DEG:
        return False
    midpoint_gap = np.linalg.norm(np.subtract(_segment_midpoint(first), _segment_midpoint(second)))
    return float(midpoint_gap) <= DUPLICATE_DISTANCE_PX and _segment_endpoint_distance(first, second) <= DUPLICATE_DISTANCE_PX


def _has_nearby_consistent_orientation(segment: Segment, candidates: Sequence[Segment]) -> bool:
    midpoint = np.array(_segment_midpoint(segment))
    orientation = calculate_orientation(*segment)
    for candidate in candidates:
        if candidate == segment:
            continue
        if np.linalg.norm(midpoint - np.array(_segment_midpoint(candidate))) <= ISOLATED_ORIENTATION_DISTANCE_PX:
            if angle_difference(orientation, calculate_orientation(*candidate)) <= ISOLATED_ORIENTATION_TOLERANCE_DEG:
                return True
    return False


def _looks_like_bottom_caption_artifact(segment: Segment, image_height: int) -> bool:
    x1, y1, x2, y2 = segment
    margin = max(20, int(image_height * BOTTOM_BORDER_MARGIN_RATIO))
    near_bottom = min(y1, y2) >= image_height - margin
    orientation = calculate_orientation(x1, y1, x2, y2)
    return near_bottom and min(orientation, 180 - orientation) <= BOTTOM_CAPTION_MAX_ANGLE_DEG


def filter_hough_segments(segments: Sequence[Segment], image_height: int, min_line_length: int = MIN_LINE_LENGTH) -> List[Segment]:
    """Remove short, border-like, isolated-noise, and duplicate segments."""

    candidates = [
        segment for segment in segments
        if calculate_length(*segment) >= min_line_length
        and not _looks_like_bottom_caption_artifact(segment, image_height)
    ]
    if len(candidates) >= 3:
        median_length = float(np.median([calculate_length(*segment) for segment in candidates]))
        candidates = [
            segment for segment in candidates
            if calculate_length(*segment) >= STRONG_ISOLATED_LENGTH_FACTOR * median_length
            or _has_nearby_consistent_orientation(segment, candidates)
        ]
    kept: List[Segment] = []
    for segment in sorted(candidates, key=lambda item: calculate_length(*item), reverse=True):
        if not any(_segments_are_duplicate(segment, existing) for existing in kept):
            kept.append(segment)
    return kept


def _build_measurements(segments: Sequence[Segment]) -> List[Measurement]:
    measurements: List[Measurement] = []
    for index, (x1, y1, x2, y2) in enumerate(segments, start=1):
        measurements.append({
            "segment": index, "trajectory_id": 0, "x1": x1, "y1": y1,
            "x2": x2, "y2": y2, "midpoint_x": 0.5 * (x1 + x2),
            "midpoint_y": 0.5 * (y1 + y2), "length_px": calculate_length(x1, y1, x2, y2),
            "orientation_deg": calculate_orientation(x1, y1, x2, y2),
        })
    return measurements


def _segments_should_be_grouped(first: Measurement, second: Measurement) -> bool:
    angle_gap = angle_difference(first["orientation_deg"], second["orientation_deg"])
    if angle_gap > TRAJECTORY_JOIN_ANGLE_TOLERANCE_DEG:
        return False
    segment1 = (first["x1"], first["y1"], first["x2"], first["y2"])
    segment2 = (second["x1"], second["y1"], second["x2"], second["y2"])
    endpoint_gap = _segment_endpoint_distance(segment1, segment2)
    midpoint_gap = math.hypot(
        float(first["midpoint_x"]) - float(second["midpoint_x"]),
        float(first["midpoint_y"]) - float(second["midpoint_y"]),
    )
    return endpoint_gap <= TRAJECTORY_JOIN_DISTANCE_PX or (
        midpoint_gap <= TRAJECTORY_JOIN_DISTANCE_PX
        and angle_gap <= TRAJECTORY_PARALLEL_ANGLE_TOLERANCE_DEG
    )


def group_trajectory_segments(results: Sequence[Measurement]) -> List[List[Measurement]]:
    """Group nearby Hough segments using a small union-find graph."""

    if not results:
        return []
    parent = list(range(len(results)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for first in range(len(results)):
        for second in range(first + 1, len(results)):
            if _segments_should_be_grouped(results[first], results[second]):
                union(first, second)
    groups: Dict[int, List[Measurement]] = {}
    for index, result in enumerate(results):
        groups.setdefault(find(index), []).append(result)
    return sorted(groups.values(), key=lambda group: min(int(item["segment"]) for item in group))


def _average_orientation(results: Sequence[Measurement]) -> float:
    if not results:
        return 0.0
    weights = np.array([float(item["length_px"]) for item in results])
    angles = np.radians([2.0 * float(item["orientation_deg"]) for item in results])
    return float((0.5 * math.degrees(math.atan2(np.sum(weights * np.sin(angles)), np.sum(weights * np.cos(angles))))) % 180.0)


def calculate_curvature(results: Sequence[Measurement]) -> Tuple[List[Measurement], float, float, float]:
    """Order one trajectory and return ordered segments, angle change, length, curvature."""

    if not results:
        return [], 0.0, 0.0, 0.0
    if len(results) == 1:
        return list(results), 0.0, float(results[0]["length_px"]), 0.0
    midpoints = np.array([[item["midpoint_x"], item["midpoint_y"]] for item in results], dtype=float)
    centred = midpoints - np.mean(midpoints, axis=0)
    _, _, vh = np.linalg.svd(centred, full_matrices=False)
    ordered = [results[int(index)] for index in np.argsort(centred @ vh[0])]
    total_angle_change = 0.0
    local_curvatures: List[float] = []
    for current, following in zip(ordered, ordered[1:]):
        angle_change = angle_difference(current["orientation_deg"], following["orientation_deg"])
        total_angle_change += angle_change
        midpoint_gap = math.hypot(
            float(following["midpoint_x"]) - float(current["midpoint_x"]),
            float(following["midpoint_y"]) - float(current["midpoint_y"]),
        )
        effective_distance = max(midpoint_gap, 0.5 * (float(current["length_px"]) + float(following["length_px"])))
        if effective_distance > 0:
            local_curvatures.append(math.radians(angle_change) / effective_distance)
    curvature = float(np.median(local_curvatures)) if local_curvatures else 0.0
    length = float(sum(float(item["length_px"]) for item in ordered))
    return ordered, total_angle_change, length, curvature


def _calculate_straightness(results: Sequence[Measurement]) -> float:
    if not results:
        return 0.0
    if len(results) == 1:
        return 1.0
    chord = math.hypot(
        float(results[-1]["midpoint_x"]) - float(results[0]["midpoint_x"]),
        float(results[-1]["midpoint_y"]) - float(results[0]["midpoint_y"]),
    )
    path = sum(
        math.hypot(
            float(next_item["midpoint_x"]) - float(current["midpoint_x"]),
            float(next_item["midpoint_y"]) - float(current["midpoint_y"]),
        )
        for current, next_item in zip(results, results[1:])
    )
    ratio = chord / path if path > 0 else 1.0
    if len(results) == 2:
        ratio = min(ratio, 1.0 - angle_difference(results[0]["orientation_deg"], results[1]["orientation_deg"]) / 90.0)
    return max(0.0, min(1.0, float(ratio)))


def suggest_trajectory_pattern(segment_count: int, straightness: float, average_angle_change_deg: float, curvature_rad_per_px: float) -> str:
    """Return one of the report labels: straight, curved, or uncertain."""

    if segment_count >= MIN_SEGMENTS_FOR_CURVE and average_angle_change_deg >= CURVATURE_ANGLE_SUPPORT_DEG and curvature_rad_per_px >= CURVATURE_THRESHOLD_RAD_PER_PX and straightness < CURVED_STRAIGHTNESS_MAX:
        return "Curved trajectory"
    if segment_count >= 2 and straightness >= STRAIGHTNESS_THRESHOLD and average_angle_change_deg <= STRAIGHT_ANGLE_SUPPORT_DEG and curvature_rad_per_px < CURVATURE_THRESHOLD_RAD_PER_PX:
        return "Straight trajectory"
    return "Uncertain"


def _apply_trajectory_metrics(results: Sequence[Measurement]) -> List[Measurement]:
    annotated: List[Measurement] = []
    for trajectory_id, trajectory in enumerate(group_trajectory_segments(results), start=1):
        ordered, angle_change, length, curvature = calculate_curvature(trajectory)
        average_change = angle_change / max(1, len(ordered) - 1)
        straightness = _calculate_straightness(ordered)
        evidence = suggest_trajectory_pattern(len(ordered), straightness, average_change, curvature)
        for item in ordered:
            item["trajectory_id"] = trajectory_id
            item["trajectory_length_px"] = length
            item["trajectory_orientation_deg"] = _average_orientation(ordered)
            item["trajectory_straightness"] = straightness
            item["trajectory_curvature_rad_per_px"] = curvature
            item["trajectory_average_angle_change_deg"] = average_change
            item["trajectory_evidence"] = evidence
            annotated.append(item)
    return annotated


def _trajectory_summaries(results: Sequence[Measurement]) -> List[Measurement]:
    by_id: Dict[int, List[Measurement]] = {}
    for result in results:
        by_id.setdefault(int(result["trajectory_id"]), []).append(result)
    summaries: List[Measurement] = []
    for trajectory_id, trajectory in sorted(by_id.items()):
        first = trajectory[0]
        summaries.append({
            "trajectory_id": trajectory_id,
            "segment_count": len(trajectory),
            "trajectory_length_px": float(first["trajectory_length_px"]),
            "orientation_angle_deg": float(first["trajectory_orientation_deg"]),
            "straightness": float(first["trajectory_straightness"]),
            "curvature_rad_per_px": float(first["trajectory_curvature_rad_per_px"]),
            "average_angle_change_deg": float(first["trajectory_average_angle_change_deg"]),
            "trajectory_evidence": str(first["trajectory_evidence"]),
        })
    return summaries


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_measurements(path: Path, results: Sequence[Measurement]) -> None:
    _write_csv(path, [
        "segment", "trajectory_id", "x1", "y1", "x2", "y2", "midpoint_x", "midpoint_y",
        "length_px", "orientation_deg", "trajectory_length_px", "trajectory_orientation_deg",
        "trajectory_straightness", "trajectory_curvature_rad_per_px",
        "trajectory_average_angle_change_deg", "trajectory_evidence",
    ], results)


def _write_trajectory_features(path: Path, image_name: str, summaries: Sequence[Measurement]) -> None:
    _write_csv(path, TRAJECTORY_FEATURE_FIELDS, [{"image_name": image_name, **summary} for summary in summaries])


def _upsert_image_features(result_dir: Path, image_name: str, roi: ROI, raw_count: int, filtered_count: int, summaries: Sequence[Measurement]) -> Path:
    path = result_dir / "hough_features.csv"
    rows: List[Dict[str, object]] = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as csv_file:
            rows = [row for row in csv.DictReader(csv_file) if row.get("image_name") != image_name]
    primary = max(summaries, key=lambda item: float(item["trajectory_length_px"]), default=None)
    rows.append({
        "image_name": image_name, "roi_x": roi[0], "roi_y": roi[1], "roi_width": roi[2], "roi_height": roi[3],
        "raw_hough_segments": raw_count, "filtered_segments": filtered_count, "trajectory_count": len(summaries),
        "primary_trajectory_id": primary["trajectory_id"] if primary else "",
        "trajectory_length_px": round(float(primary["trajectory_length_px"]), 4) if primary else 0.0,
        "orientation_angle_deg": round(float(primary["orientation_angle_deg"]), 4) if primary else 0.0,
        "straightness": round(float(primary["straightness"]), 4) if primary else 0.0,
        "curvature_rad_per_px": round(float(primary["curvature_rad_per_px"]), 8) if primary else 0.0,
        "average_angle_change_deg": round(float(primary["average_angle_change_deg"]), 4) if primary else 0.0,
        "trajectory_evidence": primary["trajectory_evidence"] if primary else "Uncertain",
    })
    rows.sort(key=lambda row: str(row.get("image_name", "")).lower())
    _write_csv(path, IMAGE_FEATURE_FIELDS, rows)
    return path


def _colour_for_trajectory(trajectory_id: int) -> Tuple[int, int, int]:
    palette = [(0, 255, 255), (0, 165, 255), (255, 255, 0), (255, 0, 255), (0, 255, 0), (255, 128, 0)]
    return palette[(trajectory_id - 1) % len(palette)]


def draw_trajectory_overlay(image: np.ndarray, results: Sequence[Measurement], summaries: Sequence[Measurement], roi: ROI) -> np.ndarray:
    output = image.copy()
    x, y, width, height = roi
    cv2.rectangle(output, (x, y), (x + width - 1, y + height - 1), (255, 80, 0), 3)
    by_id: Dict[int, List[Measurement]] = {}
    for result in results:
        by_id.setdefault(int(result["trajectory_id"]), []).append(result)
    for summary in summaries:
        trajectory_id = int(summary["trajectory_id"])
        colour = _colour_for_trajectory(trajectory_id)
        points: List[Tuple[int, int]] = []
        for result in by_id[trajectory_id]:
            cv2.line(output, (int(result["x1"]), int(result["y1"])), (int(result["x2"]), int(result["y2"])), colour, 3, cv2.LINE_AA)
            points.append((int(result["midpoint_x"]), int(result["midpoint_y"])))
        if len(points) >= 2:
            cv2.polylines(output, [np.asarray(points, dtype=np.int32)], False, colour, 1, cv2.LINE_AA)
        anchor = points[len(points) // 2] if points else (x + 8, y + 25)
        label = f"T{trajectory_id}: {summary['trajectory_evidence']} | L={float(summary['trajectory_length_px']):.0f}px | angle={float(summary['orientation_angle_deg']):.1f} deg | S={float(summary['straightness']):.2f}"
        origin = (max(5, anchor[0] + 8), max(18, anchor[1] - 8))
        cv2.putText(output, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(output, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.48, colour, 1, cv2.LINE_AA)
    legend = [
        "Probabilistic Hough trajectory evidence",
        "L=length (px), angle=orientation, S=straightness",
        "Blue box=ROI; colours identify grouped trajectories",
    ]
    for index, text in enumerate(legend):
        cv2.putText(output, text, (16, 29 + index * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def analyze_frame(image: np.ndarray, roi: Optional[ROI] = None) -> Tuple[np.ndarray, np.ndarray, List[Measurement], List[Measurement]]:
    """Analyze an in-memory frame for the web UI without writing files.

    Returns ``(overlay, full_frame_edges, detections, trajectory_summaries)``.
    The detection records contain only geometry and trajectory evidence.
    """

    resolved = _normalise_roi(roi, image.shape) if roi is not None else _default_roi(image.shape)
    x, y, width, height = resolved
    roi_image = image[y : y + height, x : x + width]
    _, edges = preprocess_image(roi_image)
    raw_segments = run_hough_lines_p(edges)
    filtered_segments = filter_hough_segments(raw_segments, roi_image.shape[0])
    local_results = _apply_trajectory_metrics(_build_measurements(filtered_segments))
    results: List[Measurement] = []
    for local in local_results:
        result = dict(local)
        result["x1"] = int(result["x1"]) + x
        result["x2"] = int(result["x2"]) + x
        result["y1"] = int(result["y1"]) + y
        result["y2"] = int(result["y2"]) + y
        result["midpoint_x"] = float(result["midpoint_x"]) + x
        result["midpoint_y"] = float(result["midpoint_y"]) + y
        results.append(result)
    summaries = _trajectory_summaries(results)
    overlay = draw_trajectory_overlay(image, results, summaries, resolved)
    full_frame_edges = np.zeros(image.shape[:2], dtype=np.uint8)
    full_frame_edges[y : y + height, x : x + width] = edges
    return overlay, full_frame_edges, summaries, [
        {
            "trajectory_id": summary["trajectory_id"],
            "length": round(float(summary["trajectory_length_px"]), 2),
            "orientation_angle": round(float(summary["orientation_angle_deg"]), 2),
            "straightness": round(float(summary["straightness"]), 4),
            "curvature": round(float(summary["curvature_rad_per_px"]), 7),
            "evidence": summary["trajectory_evidence"],
            "segment_count": summary["segment_count"],
        }
        for summary in summaries
    ]


def _fit_panel(image: np.ndarray, width: int = 520, height: int = 420) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    top = (height - resized.shape[0]) // 2
    left = (width - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas


def _make_report_montage(original: np.ndarray, enhanced: np.ndarray, edges: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    panels = [("1. Original ROI", original), ("2. Background-corrected", enhanced), ("3. Canny edges", edges), ("4. Hough trajectories", overlay)]
    labelled: List[np.ndarray] = []
    for title, image in panels:
        panel = _fit_panel(image)
        cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, panel.shape[0] - 1), (90, 90, 90), 1)
        cv2.putText(panel, title, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        labelled.append(panel)
    montage = cv2.vconcat([cv2.hconcat(labelled[:2]), cv2.hconcat(labelled[2:])])
    cv2.putText(montage, "Cloud chamber trajectory analysis | Canny + HoughLinesP", (18, montage.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return montage


def _find_supported_images(image_dir: Path = IMAGE_DIR) -> List[Path]:
    if not image_dir.exists():
        return []
    return sorted([path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS], key=lambda path: path.name.lower())


def process_image(image_path: Path, result_dir: Path = RESULT_DIR, roi: Optional[ROI] = None, roi_boxes: Optional[Dict[str, ROI]] = None, allow_roi_prompt: bool = False) -> bool:
    """Process one frame, save all report artifacts, and print its evidence."""

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Warning: cannot read image: {image_path}")
        return False
    resolved = _resolve_roi(image_path, image, roi, roi_boxes, allow_roi_prompt)
    x, y, width, height = resolved
    roi_image = image[y : y + height, x : x + width]
    enhanced, edges = preprocess_image(roi_image)
    raw_segments = run_hough_lines_p(edges)
    filtered_segments = filter_hough_segments(raw_segments, roi_image.shape[0])
    local_results = _apply_trajectory_metrics(_build_measurements(filtered_segments))

    results: List[Measurement] = []
    for local in local_results:
        result = dict(local)
        result["x1"] = int(result["x1"]) + x
        result["x2"] = int(result["x2"]) + x
        result["y1"] = int(result["y1"]) + y
        result["y2"] = int(result["y2"]) + y
        result["midpoint_x"] = float(result["midpoint_x"]) + x
        result["midpoint_y"] = float(result["midpoint_y"]) + y
        results.append(result)
    summaries = _trajectory_summaries(results)
    overlay = draw_trajectory_overlay(image, results, summaries, resolved)
    report = _make_report_montage(roi_image, enhanced, edges, overlay[y : y + height, x : x + width])

    result_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    edge_canvas = np.zeros(image.shape[:2], dtype=np.uint8)
    edge_canvas[y : y + height, x : x + width] = edges
    outputs = {
        result_dir / f"{stem}_edges.jpg": edge_canvas,
        result_dir / f"{stem}_hough_result.jpg": overlay,
        result_dir / f"{stem}_hough_report.jpg": report,
    }
    for path, content in outputs.items():
        if not cv2.imwrite(str(path), content):
            raise OSError(f"Could not save output: {path}")
    _write_measurements(result_dir / f"{stem}_measurements.csv", results)
    _write_trajectory_features(result_dir / f"{stem}_trajectory_features.csv", image_path.name, summaries)
    _upsert_image_features(result_dir, image_path.name, resolved, len(raw_segments), len(filtered_segments), summaries)

    primary = max(summaries, key=lambda item: float(item["trajectory_length_px"]), default=None)
    print(f"Image: {image_path.name}")
    print(f"ROI: x={x}, y={y}, width={width}, height={height}")
    print(f"HoughLinesP segments: {len(raw_segments)} raw -> {len(filtered_segments)} filtered")
    print(f"Grouped trajectories: {len(summaries)}")
    if primary:
        print(
            f"Primary evidence: {primary['trajectory_evidence']} | "
            f"length={float(primary['trajectory_length_px']):.2f}px | "
            f"orientation={float(primary['orientation_angle_deg']):.2f}deg | "
            f"straightness={float(primary['straightness']):.3f} | "
            f"curvature={float(primary['curvature_rad_per_px']):.6f} rad/px"
        )
    else:
        print("Primary evidence: Uncertain (no reliable trajectory detected)")
    print("No Alpha/Electron classification is produced.")
    return True


def test_hough_parameters(image_paths: Optional[Sequence[Path]] = None, result_dir: Path = RESULT_DIR, roi_boxes: Optional[Dict[str, ROI]] = None) -> Path:
    selected = list(image_paths) if image_paths is not None else _find_supported_images()
    prepared: List[Tuple[np.ndarray, np.ndarray]] = []
    for image_path in selected:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        x, y, width, height = _resolve_roi(image_path, image, roi_boxes=roi_boxes)
        roi_image = image[y : y + height, x : x + width]
        prepared.append((roi_image, preprocess_image(roi_image)[1]))
    rows: List[Dict[str, object]] = []
    for threshold in (20, 28, 36):
        for minimum in (16, 24, 32):
            for gap in (6, 12, 18):
                raw_count = filtered_count = 0
                lengths: List[float] = []
                for roi_image, edges in prepared:
                    raw = run_hough_lines_p(edges, threshold, minimum, gap)
                    filtered = filter_hough_segments(raw, roi_image.shape[0], minimum)
                    raw_count += len(raw)
                    filtered_count += len(filtered)
                    lengths.extend(calculate_length(*segment) for segment in filtered)
                rows.append({
                    "hough_threshold": threshold, "min_line_length": minimum, "max_line_gap": gap,
                    "images_processed": len(prepared), "raw_hough_segments": raw_count,
                    "filtered_segments": filtered_count,
                    "average_segment_length_px": round(float(np.mean(lengths)), 4) if lengths else 0.0,
                    "noise_reduction_percentage": round(100.0 * (raw_count - filtered_count) / raw_count, 4) if raw_count else 0.0,
                })
    path = result_dir / "hough_parameter_comparison.csv"
    _write_csv(path, list(rows[0].keys()) if rows else ["hough_threshold"], rows)
    print(f"Saved parameter comparison: {path}")
    return path


def main() -> None:
    image_paths = _find_supported_images()
    while True:
        print("\n========================================")
        print("CLOUD CHAMBER HOUGHLINEP ANALYSIS")
        print("========================================")
        for index, image_path in enumerate(image_paths, start=1):
            print(f"{index}. {image_path.name}")
        print("A. Process all | T. Test parameters | R. Refresh | Q. Quit")
        choice = input("> ").strip().lower()
        if choice == "q":
            return
        if choice == "r":
            image_paths = _find_supported_images()
            continue
        if choice == "a":
            for image_path in image_paths:
                try:
                    process_image(image_path, roi_boxes=_load_roi_boxes())
                except Exception as error:
                    print(f"Warning: {image_path.name}: {error}")
            continue
        if choice == "t":
            test_hough_parameters(image_paths, RESULT_DIR, _load_roi_boxes())
            continue
        try:
            index = int(choice) - 1
        except ValueError:
            print("Invalid choice.")
            continue
        if 0 <= index < len(image_paths):
            process_image(image_paths[index], roi_boxes=_load_roi_boxes())
        else:
            print("Invalid image number.")


if __name__ == "__main__":
    main()
