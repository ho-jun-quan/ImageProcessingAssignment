"""Hough Transform track analysis for the WeiQuan image set."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Paths are based on this file, not on the current working directory.
WEIQUAN_DIR = Path(__file__).resolve().parent.parent
IMAGE_DIR = WEIQUAN_DIR / "image"
RESULT_DIR = WEIQUAN_DIR / "result"
ROI_BOX_FILE = WEIQUAN_DIR / "roi_boxes.csv"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Image preprocessing parameters
GAUSSIAN_KERNEL_SIZE = 5
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150

# Hough Transform parameters
HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
# Higher Hough threshold reduces false detections but may miss weak tracks.
HOUGH_THRESHOLD = 30
# Higher minimum line length removes small noise.
MIN_LINE_LENGTH = 20
# Larger maximum gap connects broken particle tracks.
MAX_LINE_GAP = 15

# Morphology is intentionally configurable so it can be calibrated against
# different AWAN image conditions without changing the Hough algorithm.
ENABLE_MORPHOLOGICAL_OPENING = True
ENABLE_MORPHOLOGICAL_CLOSING = True
MORPHOLOGY_KERNEL_SIZE = 3
MORPHOLOGY_OPEN_ITERATIONS = 1
MORPHOLOGY_CLOSE_ITERATIONS = 1

# Additional filtering/grouping parameters. These remove duplicate/noise
# detections after HoughLinesP() and keep nearby segments in separate tracks.
DUPLICATE_DISTANCE_PX = 12.0
DUPLICATE_ANGLE_TOLERANCE_DEG = 8.0
TRAJECTORY_JOIN_DISTANCE_PX = 35.0
TRAJECTORY_JOIN_ANGLE_TOLERANCE_DEG = 30.0
ISOLATED_ORIENTATION_DISTANCE_PX = 45.0
ISOLATED_ORIENTATION_TOLERANCE_DEG = 35.0
STRONG_ISOLATED_LENGTH_FACTOR = 1.5

# Curvature evidence is evaluated per grouped trajectory. A trajectory needs
# enough local segments before curvature can be considered reliable.
MIN_TRAJECTORY_SEGMENTS_FOR_CURVE = 3
CURVATURE_THRESHOLD_RAD_PER_PX = 0.005
CURVATURE_ANGLE_SUPPORT_DEG = 10.0
ALPHA_LIKE_STRAIGHTNESS_MIN = 0.85
ELECTRON_LIKE_STRAIGHTNESS_MAX = 0.80
MIN_SEGMENTS_FOR_RULE_CLASSIFICATION = 2

BOTTOM_BORDER_MARGIN_RATIO = 0.08
BOTTOM_CAPTION_MAX_ANGLE_DEG = 15

# Leave as None until calibration is known.
# Example: if 50 pixels = 1 cm, set PIXELS_PER_CM = 50.
PIXELS_PER_CM = None

# ROI input configuration. Populate ROI_BOXES in code, or create
# WeiQuan/roi_boxes.csv with columns: image_name,x,y,width,height.
# Coordinates use the original image and are zero-based pixels.
ROI_BOXES: Dict[str, Tuple[int, int, int, int]] = {}
ROI_COORDINATE_ORDER = "x,y,width,height"

Measurement = Dict[str, object]
Segment = Tuple[int, int, int, int]
ROI = Tuple[int, int, int, int]
FEATURE_FIELDNAMES = [
    "image_name",
    "roi_x",
    "roi_y",
    "roi_width",
    "roi_height",
    "segment_count",
    "total_length_px",
    "average_orientation",
    "average_angle_change",
    "curvature_rad_per_px",
    "straightness",
    "trajectory_pattern",
    "rule_based_classification",
]


def calculate_length(x1: int, y1: int, x2: int, y2: int) -> float:
    """Calculate a line segment length using Euclidean distance."""

    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_orientation(x1: int, y1: int, x2: int, y2: int) -> float:
    """Calculate an undirected line orientation from 0 to 180 degrees."""

    dx = x2 - x1
    dy = -(y2 - y1)
    return math.degrees(math.atan2(dy, dx)) % 180


def angle_difference(angle1: float, angle2: float) -> float:
    """Return the smallest difference between two line orientations."""

    difference = abs(angle1 - angle2)
    return 180 - difference if difference > 90 else difference


def suggest_trajectory_pattern(
    trajectory_count: int,
    curved_trajectory_count: int,
    average_angle_change_deg: float,
    curvature_rad_per_px: float,
) -> str:
    """Return Hough evidence, without making a final particle classification."""

    # This is trajectory evidence only. Other particle features must be
    # combined before making a final particle-class decision.
    if trajectory_count == 0:
        return "Uncertain"
    if (
        curved_trajectory_count > 0
        and average_angle_change_deg >= CURVATURE_ANGLE_SUPPORT_DEG
        and curvature_rad_per_px >= CURVATURE_THRESHOLD_RAD_PER_PX
    ):
        return "Curved trajectory"
    if (
        average_angle_change_deg < CURVATURE_ANGLE_SUPPORT_DEG
        and curvature_rad_per_px < CURVATURE_THRESHOLD_RAD_PER_PX
    ):
        return "Straight trajectory"
    return "Uncertain"


def calculate_curvature(
    results: Sequence[Measurement],
) -> Tuple[List[Measurement], float, float, float]:
    """Order segments from one trajectory and calculate its curvature.

    This function intentionally accepts one grouped trajectory.  Calling it on
    all Hough detections would measure curvature across unrelated noise tracks.
    """

    if len(results) < 2:
        return list(results), 0.0, 0.0, 0.0

    midpoints = np.array(
        [[result["midpoint_x"], result["midpoint_y"]] for result in results]
    )
    centred = midpoints - np.mean(midpoints, axis=0)

    # PCA determines the main direction so segments can be ordered along the
    # track, matching the original script's curvature calculation.
    _, _, vh = np.linalg.svd(centred, full_matrices=False)
    main_axis = vh[0]
    projection = centred @ main_axis
    order = np.argsort(projection)
    ordered_results = [results[int(index)] for index in order]

    total_angle_change_deg = 0.0
    local_curvatures: List[float] = []

    for current, next_segment in zip(ordered_results, ordered_results[1:]):
        angle_change = angle_difference(
            current["orientation_deg"], next_segment["orientation_deg"]
        )
        total_angle_change_deg += angle_change
        dx = next_segment["midpoint_x"] - current["midpoint_x"]
        dy = next_segment["midpoint_y"] - current["midpoint_y"]
        midpoint_distance = math.sqrt(dx**2 + dy**2)
        # Hough can return two short pieces whose centres are very close even
        # though they cover a substantial part of the same physical track.
        # Prevent that small gap from creating an unrealistically large
        # curvature value.
        effective_distance = max(
            midpoint_distance,
            0.5
            * (
                float(current["length_px"])
                + float(next_segment["length_px"])
            ),
        )
        if effective_distance > 0:
            local_curvatures.append(math.radians(angle_change) / effective_distance)

    # Median local curvature is less sensitive to one accidental long-angle
    # jump than a single total-angle-change measurement. It is still computed
    # only within this one grouped trajectory.
    curvature_rad_per_px = float(np.median(local_curvatures)) if local_curvatures else 0.0
    total_path_distance_px = sum(float(item["length_px"]) for item in ordered_results)

    return (
        ordered_results,
        total_angle_change_deg,
        total_path_distance_px,
        curvature_rad_per_px,
    )


def _normalise_roi(roi: ROI, image_shape: Sequence[int]) -> ROI:
    """Validate and clip an x, y, width, height ROI to the image bounds."""

    if len(roi) != 4:
        raise ValueError("ROI must contain x, y, width, and height")
    image_height, image_width = int(image_shape[0]), int(image_shape[1])
    x, y, width, height = (int(value) for value in roi)
    if width <= 0 or height <= 0:
        raise ValueError("ROI width and height must be positive")
    if (
        x >= image_width
        or y >= image_height
        or x + width <= 0
        or y + height <= 0
    ):
        raise ValueError("ROI does not overlap the image")

    x1 = max(0, min(x, image_width - 1))
    y1 = max(0, min(y, image_height - 1))
    x2 = max(x1 + 1, min(x + width, image_width))
    y2 = max(y1 + 1, min(y + height, image_height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("ROI does not overlap the image")
    return x1, y1, x2 - x1, y2 - y1


def _parse_roi_text(text: str) -> ROI:
    """Parse interactive ROI input in x,y,width,height order."""

    values = [part.strip() for part in text.replace(";", ",").split(",")]
    if len(values) != 4:
        values = text.split()
    if len(values) != 4:
        raise ValueError("Enter four values: x,y,width,height")
    return tuple(int(float(value)) for value in values)  # type: ignore[return-value]


def _load_roi_boxes(roi_file: Path = ROI_BOX_FILE) -> Dict[str, ROI]:
    """Load optional per-image ROIs from CSV without guessing a full frame."""

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


def _lookup_roi(image_path: Path, roi_boxes: Optional[Dict[str, ROI]] = None) -> Optional[ROI]:
    """Find an explicitly configured ROI by image filename or stem."""

    boxes = roi_boxes if roi_boxes is not None else {}
    for key in (image_path.name, image_path.stem):
        if key in boxes:
            return boxes[key]
        if key in ROI_BOXES:
            return ROI_BOXES[key]
    return None


def _resolve_roi(
    image_path: Path,
    image: np.ndarray,
    roi: Optional[ROI] = None,
    roi_boxes: Optional[Dict[str, ROI]] = None,
    allow_prompt: bool = False,
) -> Optional[ROI]:
    """Resolve an ROI from an argument, configuration, CSV, or user input."""

    candidate = roi if roi is not None else _lookup_roi(image_path, roi_boxes)
    if candidate is not None:
        return _normalise_roi(candidate, image.shape)
    if not allow_prompt:
        return None

    print(f"No ROI configured for {image_path.name}.")
    print("[M] Select ROI with the mouse")
    print("[C] Enter ROI coordinates as x,y,width,height")
    print("[S] Skip this image")
    try:
        choice = input("ROI method: ").strip().lower()
    except EOFError:
        return None

    if choice == "m":
        try:
            selected = cv2.selectROI(
                f"Select ROI - {image_path.name}",
                image,
                showCrosshair=True,
                fromCenter=False,
            )
            cv2.destroyWindow(f"Select ROI - {image_path.name}")
            if selected[2] <= 0 or selected[3] <= 0:
                return None
            return _normalise_roi(tuple(int(value) for value in selected), image.shape)  # type: ignore[arg-type]
        except cv2.error as error:
            print(f"Could not open ROI selector: {error}")
            return None
    if choice == "c":
        try:
            entered = _parse_roi_text(input("ROI x,y,width,height: "))
            return _normalise_roi(entered, image.shape)
        except (EOFError, ValueError) as error:
            print(f"Invalid ROI: {error}")
    return None


def _preprocess_image(image: np.ndarray) -> np.ndarray:
    """Apply contrast enhancement, denoising, morphology, and Canny edges."""

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(grey)

    if GAUSSIAN_KERNEL_SIZE < 1 or GAUSSIAN_KERNEL_SIZE % 2 == 0:
        raise ValueError("GAUSSIAN_KERNEL_SIZE must be a positive odd number")
    blurred = cv2.GaussianBlur(
        enhanced, (GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE), 0
    )

    if MORPHOLOGY_KERNEL_SIZE < 1 or MORPHOLOGY_KERNEL_SIZE % 2 == 0:
        raise ValueError("MORPHOLOGY_KERNEL_SIZE must be a positive odd number")
    if MORPHOLOGY_OPEN_ITERATIONS < 0 or MORPHOLOGY_CLOSE_ITERATIONS < 0:
        raise ValueError("Morphology iteration counts cannot be negative")

    # Opening removes isolated bright/dark specks. Closing reconnects small
    # breaks in a particle track. Both stages can be disabled independently
    # while tuning the pipeline for a different image subset.
    morphology_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (MORPHOLOGY_KERNEL_SIZE, MORPHOLOGY_KERNEL_SIZE),
    )
    cleaned = blurred
    if ENABLE_MORPHOLOGICAL_OPENING and MORPHOLOGY_OPEN_ITERATIONS:
        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_OPEN,
            morphology_kernel,
            iterations=MORPHOLOGY_OPEN_ITERATIONS,
        )
    if ENABLE_MORPHOLOGICAL_CLOSING and MORPHOLOGY_CLOSE_ITERATIONS:
        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_CLOSE,
            morphology_kernel,
            iterations=MORPHOLOGY_CLOSE_ITERATIONS,
        )
    return cv2.Canny(
        cleaned,
        threshold1=CANNY_LOW_THRESHOLD,
        threshold2=CANNY_HIGH_THRESHOLD,
    )


def _run_hough(
    edges: np.ndarray,
    hough_threshold: int = HOUGH_THRESHOLD,
    min_line_length: int = MIN_LINE_LENGTH,
    max_line_gap: int = MAX_LINE_GAP,
) -> List[Segment]:
    """Run the unchanged Probabilistic Hough Transform with configurable values."""

    lines = cv2.HoughLinesP(
        edges,
        rho=HOUGH_RHO,
        theta=HOUGH_THETA,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if lines is None:
        return []
    return [tuple(map(int, line)) for line in np.asarray(lines).reshape(-1, 4)]


def _segment_midpoint(segment: Segment) -> Tuple[float, float]:
    x1, y1, x2, y2 = segment
    return (float(x1 + x2) / 2, float(y1 + y2) / 2)


def _segment_endpoint_distance(segment1: Segment, segment2: Segment) -> float:
    endpoints1 = np.array([[segment1[0], segment1[1]], [segment1[2], segment1[3]]], dtype=float)
    endpoints2 = np.array([[segment2[0], segment2[1]], [segment2[2], segment2[3]]], dtype=float)
    return float(np.min(np.linalg.norm(endpoints1[:, None] - endpoints2[None, :], axis=2)))


def _segments_are_duplicate(segment1: Segment, segment2: Segment) -> bool:
    """Identify overlapping, nearby parallel detections of the same track."""

    orientation1 = calculate_orientation(*segment1)
    orientation2 = calculate_orientation(*segment2)
    if angle_difference(orientation1, orientation2) > DUPLICATE_ANGLE_TOLERANCE_DEG:
        return False
    midpoint1 = np.array(_segment_midpoint(segment1))
    midpoint2 = np.array(_segment_midpoint(segment2))
    if float(np.linalg.norm(midpoint1 - midpoint2)) > DUPLICATE_DISTANCE_PX:
        return False
    return _segment_endpoint_distance(segment1, segment2) <= DUPLICATE_DISTANCE_PX


def _has_nearby_consistent_orientation(
    segment: Segment,
    candidates: Sequence[Segment],
) -> bool:
    """Keep isolated segments unless they are likely orientation noise."""

    midpoint = np.array(_segment_midpoint(segment))
    orientation = calculate_orientation(*segment)
    for candidate in candidates:
        if candidate == segment:
            continue
        candidate_midpoint = np.array(_segment_midpoint(candidate))
        if float(np.linalg.norm(midpoint - candidate_midpoint)) <= ISOLATED_ORIENTATION_DISTANCE_PX:
            if angle_difference(orientation, calculate_orientation(*candidate)) <= ISOLATED_ORIENTATION_TOLERANCE_DEG:
                return True
    return False


def filter_hough_segments(
    segments: Sequence[Segment],
    image_height: int,
    min_line_length: int = MIN_LINE_LENGTH,
) -> List[Segment]:
    """Remove short, caption-like/orientation-noise, and duplicate segments."""

    candidates: List[Segment] = []
    for segment in segments:
        x1, y1, x2, y2 = segment
        if calculate_length(x1, y1, x2, y2) < min_line_length:
            continue
        if _looks_like_bottom_caption_artifact(x1, y1, x2, y2, image_height):
            continue
        candidates.append(segment)

    # A globally fixed orientation cutoff would incorrectly remove valid tracks
    # at arbitrary angles. Reject isolated short detections only when enough
    # neighbouring segments exist to establish local evidence; a much longer
    # isolated segment is retained because it may be the only Hough piece of a
    # faint but real track.
    if len(candidates) >= 3:
        lengths = np.array([calculate_length(*segment) for segment in candidates])
        median_length = float(np.median(lengths))
        candidates = [
            segment
            for segment in candidates
            if calculate_length(*segment) >= STRONG_ISOLATED_LENGTH_FACTOR * median_length
            or _has_nearby_consistent_orientation(segment, candidates)
        ]

    # HoughLinesP can return both edges of the same bright track. Keep the
    # longer detection when two nearby segments are parallel and overlapping.
    kept: List[Segment] = []
    for segment in sorted(
        candidates,
        key=lambda item: calculate_length(*item),
        reverse=True,
    ):
        if any(_segments_are_duplicate(segment, existing) for existing in kept):
            continue
        kept.append(segment)
    return kept


def _segments_should_be_grouped(result1: Measurement, result2: Measurement) -> bool:
    """Return whether two filtered segments plausibly belong to one track."""

    angle = angle_difference(result1["orientation_deg"], result2["orientation_deg"])
    if angle > TRAJECTORY_JOIN_ANGLE_TOLERANCE_DEG:
        return False
    segment1 = (result1["x1"], result1["y1"], result1["x2"], result1["y2"])
    segment2 = (result2["x1"], result2["y1"], result2["x2"], result2["y2"])
    midpoint1 = np.array([result1["midpoint_x"], result1["midpoint_y"]])
    midpoint2 = np.array([result2["midpoint_x"], result2["midpoint_y"]])
    midpoint_distance = float(np.linalg.norm(midpoint1 - midpoint2))
    endpoint_distance = _segment_endpoint_distance(segment1, segment2)
    return endpoint_distance <= TRAJECTORY_JOIN_DISTANCE_PX or midpoint_distance <= TRAJECTORY_JOIN_DISTANCE_PX


def group_trajectory_segments(results: Sequence[Measurement]) -> List[List[Measurement]]:
    """Group neighbouring filtered Hough segments into candidate trajectories."""

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
    return sorted(groups.values(), key=lambda group: min(item["segment"] for item in group))


def _build_measurements(segments: Sequence[Segment]) -> List[Measurement]:
    results: List[Measurement] = []
    for index, (x1, y1, x2, y2) in enumerate(segments):
        results.append(
            {
                "segment": index + 1,
                "trajectory_id": 0,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "midpoint_x": (x1 + x2) / 2,
                "midpoint_y": (y1 + y2) / 2,
                "length_px": calculate_length(x1, y1, x2, y2),
                "orientation_deg": calculate_orientation(x1, y1, x2, y2),
                "curvature_rad_per_px": 0.0,
            }
        )
    return results


def _calculate_straightness(results: Sequence[Measurement]) -> float:
    """Return chord length divided by the ordered trajectory path length."""

    if not results:
        return 0.0
    if len(results) < 2:
        return 1.0

    if len(results) == 2:
        # A two-segment chord/path ratio is always one because there is only
        # one midpoint interval, so include their orientation consistency.
        angle_change = angle_difference(
            results[0]["orientation_deg"], results[1]["orientation_deg"]
        )
        return max(0.0, min(1.0, 1.0 - angle_change / 90.0))

    first = results[0]
    last = results[-1]
    chord_length = math.hypot(
        float(last["midpoint_x"]) - float(first["midpoint_x"]),
        float(last["midpoint_y"]) - float(first["midpoint_y"]),
    )
    path_length = sum(
        math.hypot(
            float(next_result["midpoint_x"]) - float(current["midpoint_x"]),
            float(next_result["midpoint_y"]) - float(current["midpoint_y"]),
        )
        for current, next_result in zip(results, results[1:])
    )
    if path_length <= 0:
        return 1.0
    return max(0.0, min(1.0, chord_length / path_length))


def rule_based_particle_classification(
    segment_count: int,
    straightness: float,
    curvature_rad_per_px: float,
) -> str:
    """Apply an explicit, preliminary rule to the geometric evidence.

    Hough Transform does not measure particle thickness or energy. Therefore
    this is an Alpha-like/Electron-like geometric suggestion, not a final
    particle classification.
    """

    if segment_count < MIN_SEGMENTS_FOR_RULE_CLASSIFICATION:
        return "Uncertain"
    if (
        straightness >= ALPHA_LIKE_STRAIGHTNESS_MIN
        and curvature_rad_per_px < CURVATURE_THRESHOLD_RAD_PER_PX
    ):
        return "Alpha-like"
    if (
        segment_count >= MIN_TRAJECTORY_SEGMENTS_FOR_CURVE
        and (
            straightness <= ELECTRON_LIKE_STRAIGHTNESS_MAX
            or curvature_rad_per_px >= CURVATURE_THRESHOLD_RAD_PER_PX
        )
    ):
        return "Electron-like"
    return "Uncertain"


def _apply_trajectory_metrics(results: Sequence[Measurement]) -> Tuple[List[Measurement], float, float, float, int]:
    """Group results and attach per-trajectory curvature metrics."""

    trajectories = group_trajectory_segments(results)
    ordered_results: List[Measurement] = []
    total_angle_change_deg = 0.0
    total_path_distance_px = 0.0
    curved_trajectory_count = 0

    for trajectory_id, trajectory in enumerate(trajectories, start=1):
        ordered, angle_change, path_distance, curvature = calculate_curvature(trajectory)
        comparisons = max(len(ordered) - 1, 0)
        average_change = angle_change / comparisons if comparisons else 0.0
        straightness = _calculate_straightness(ordered)
        has_curvature_evidence = (
            len(ordered) >= MIN_TRAJECTORY_SEGMENTS_FOR_CURVE
            and curvature >= CURVATURE_THRESHOLD_RAD_PER_PX
            and average_change >= CURVATURE_ANGLE_SUPPORT_DEG
        )
        if has_curvature_evidence:
            pattern = "Curved trajectory"
        elif len(ordered) < MIN_TRAJECTORY_SEGMENTS_FOR_CURVE and (
            curvature >= CURVATURE_THRESHOLD_RAD_PER_PX
            or average_change >= CURVATURE_ANGLE_SUPPORT_DEG
        ):
            # A short trajectory can show a direction change, but does not
            # provide enough local evidence for a reliable curve decision.
            pattern = "Uncertain"
        else:
            pattern = "Straight trajectory"
        if pattern == "Curved trajectory":
            curved_trajectory_count += 1
        trajectory_length = sum(float(item["length_px"]) for item in ordered)
        for item in ordered:
            item["trajectory_id"] = trajectory_id
            item["trajectory_length_px"] = trajectory_length
            item["trajectory_total_angle_change_deg"] = angle_change
            item["trajectory_average_angle_change_deg"] = average_change
            item["trajectory_average_orientation_deg"] = _average_orientation(ordered)
            item["trajectory_curvature_rad_per_px"] = curvature
            item["trajectory_straightness"] = straightness
            item["curvature_rad_per_px"] = curvature
            item["trajectory_pattern"] = pattern
            item["trajectory_rule_based_classification"] = rule_based_particle_classification(
                len(ordered), straightness, curvature
            )
            ordered_results.append(item)
        total_angle_change_deg += angle_change
        total_path_distance_px += path_distance

    # This legacy aggregate is the strongest per-trajectory curvature value,
    # not curvature calculated by ordering unrelated image-wide segments.
    trajectory_curvatures = [
        float(item["trajectory_curvature_rad_per_px"])
        for item in ordered_results
        if "trajectory_curvature_rad_per_px" in item
    ]
    overall_curvature = max(trajectory_curvatures, default=0.0)
    return (
        ordered_results,
        total_angle_change_deg,
        total_path_distance_px,
        overall_curvature,
        curved_trajectory_count,
    )


def _average_orientation(results: Sequence[Measurement]) -> float:
    """Calculate a length-weighted mean for undirected 0-180 degree angles."""

    if not results:
        return 0.0
    weights = np.array([float(item["length_px"]) for item in results])
    doubled_angles = np.radians([2 * float(item["orientation_deg"]) for item in results])
    angle = 0.5 * math.degrees(math.atan2(np.sum(weights * np.sin(doubled_angles)), np.sum(weights * np.cos(doubled_angles))))
    return angle % 180


def _measurement_fieldnames() -> List[str]:
    return [
        "segment",
        "trajectory_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "midpoint_x",
        "midpoint_y",
        "length_px",
        "orientation_deg",
        "total_angle_change_deg",
        "average_angle_change_deg",
        "curvature_rad_per_px",
        "trajectory_length_px",
        "trajectory_total_angle_change_deg",
        "trajectory_average_angle_change_deg",
        "trajectory_average_orientation_deg",
        "trajectory_curvature_rad_per_px",
        "trajectory_straightness",
        "trajectory_pattern",
        "trajectory_rule_based_classification",
        "overall_trajectory_pattern",
    ]


def _write_measurements(
    csv_path: Path,
    results: Sequence[Measurement],
    total_angle_change_deg: float,
    average_angle_change_deg: float,
    curvature_rad_per_px: float,
    trajectory_pattern: str,
) -> None:
    """Export line measurements to a CSV file."""

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=_measurement_fieldnames())
        writer.writeheader()
        for result in results:
            row = dict(result)
            row["total_angle_change_deg"] = total_angle_change_deg
            row["average_angle_change_deg"] = average_angle_change_deg
            row["curvature_rad_per_px"] = curvature_rad_per_px
            row["overall_trajectory_pattern"] = trajectory_pattern
            writer.writerow(row)

        # Keep the trajectory pattern available in the CSV even when no segments
        # were detected and therefore there are no per-segment measurement rows.
        if not results:
            writer.writerow(
                {
                    "total_angle_change_deg": total_angle_change_deg,
                    "average_angle_change_deg": average_angle_change_deg,
                    "curvature_rad_per_px": curvature_rad_per_px,
                    "trajectory_pattern": trajectory_pattern,
                    "overall_trajectory_pattern": trajectory_pattern,
                }
            )


def _trajectory_summaries(results: Sequence[Measurement]) -> List[Measurement]:
    """Return one feature record for each grouped trajectory."""

    summaries: List[Measurement] = []
    for trajectory in group_trajectory_segments(results):
        if not trajectory:
            continue
        first = trajectory[0]
        summaries.append(
            {
                "trajectory_id": first["trajectory_id"],
                "segment_count": len(trajectory),
                "total_length_px": float(first.get("trajectory_length_px", 0.0)),
                "average_orientation": float(
                    first.get("trajectory_average_orientation_deg", 0.0)
                ),
                "average_angle_change": float(
                    first.get("trajectory_average_angle_change_deg", 0.0)
                ),
                "curvature_rad_per_px": float(
                    first.get("trajectory_curvature_rad_per_px", 0.0)
                ),
                "straightness": float(first.get("trajectory_straightness", 0.0)),
                "trajectory_pattern": first.get(
                    "trajectory_pattern", "Uncertain"
                ),
                "rule_based_classification": first.get(
                    "trajectory_rule_based_classification", "Uncertain"
                ),
            }
        )
    return summaries


def _write_hough_feature_row(
    result_dir: Path,
    image_name: str,
    roi: ROI,
    segment_count: int,
    total_length_px: float,
    average_orientation: float,
    average_angle_change: float,
    curvature_rad_per_px: float,
    straightness: float,
    trajectory_pattern: str,
    rule_based_classification: str,
) -> Path:
    """Upsert one image-level feature row for future supervised learning."""

    feature_path = result_dir / "hough_features.csv"
    rows: List[Dict[str, object]] = []
    if feature_path.exists():
        with feature_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                if row.get("image_name") and row["image_name"] != image_name:
                    rows.append({field: row.get(field, "") for field in FEATURE_FIELDNAMES})

    rows.append(
        {
            "image_name": image_name,
            "roi_x": roi[0],
            "roi_y": roi[1],
            "roi_width": roi[2],
            "roi_height": roi[3],
            "segment_count": segment_count,
            "total_length_px": round(total_length_px, 6),
            "average_orientation": round(average_orientation, 6),
            "average_angle_change": round(average_angle_change, 6),
            "curvature_rad_per_px": round(curvature_rad_per_px, 9),
            "straightness": round(straightness, 6),
            "trajectory_pattern": trajectory_pattern,
            "rule_based_classification": rule_based_classification,
        }
    )
    rows.sort(key=lambda row: str(row["image_name"]).lower())

    result_dir.mkdir(parents=True, exist_ok=True)
    with feature_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FEATURE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return feature_path


def _find_supported_images(image_dir: Path) -> List[Path]:
    """Return supported image files in a stable, case-insensitive order."""

    if not image_dir.exists():
        return []
    return sorted(
        (
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=lambda path: path.name.lower(),
    )


def _looks_like_bottom_caption_artifact(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    image_height: int,
) -> bool:
    """Identify likely horizontal text/caption lines at the bottom border."""

    bottom_margin = max(20, int(image_height * BOTTOM_BORDER_MARGIN_RATIO))
    near_bottom = min(y1, y2) >= image_height - bottom_margin
    orientation = calculate_orientation(x1, y1, x2, y2)
    horizontal = min(orientation, 180 - orientation) <= BOTTOM_CAPTION_MAX_ANGLE_DEG
    return near_bottom and horizontal


def process_image(
    image_path: Path,
    result_dir: Path,
    roi: Optional[ROI] = None,
    roi_boxes: Optional[Dict[str, ROI]] = None,
    allow_roi_prompt: bool = False,
) -> bool:
    """Run Hough Transform only inside a validated particle ROI."""

    print(f"Image: {image_path.name}")
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Warning: cannot read image, skipping: {image_path}")
        return False

    resolved_roi = _resolve_roi(
        image_path,
        image,
        roi=roi,
        roi_boxes=roi_boxes,
        allow_prompt=allow_roi_prompt,
    )
    if resolved_roi is None:
        print(
            f"Warning: no ROI provided for {image_path.name}; "
            "the full image will not be processed."
        )
        return False

    roi_x, roi_y, roi_width, roi_height = resolved_roi
    roi_image = image[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]
    output = image.copy()

    # All preprocessing and Hough operations below receive only roi_image.
    edges = _preprocess_image(roi_image)
    raw_segments = _run_hough(edges)
    filtered_segments = filter_hough_segments(raw_segments, roi_image.shape[0])
    results = _build_measurements(filtered_segments)

    for x1, y1, x2, y2 in filtered_segments:
        # Hough coordinates are local to the ROI; draw them in image space.
        cv2.line(
            output,
            (x1 + roi_x, y1 + roi_y),
            (x2 + roi_x, y2 + roi_y),
            (0, 255, 0),
            2,
        )

    (
        ordered_results,
        total_angle_change_deg,
        total_path_distance_px,
        curvature_rad_per_px,
        curved_trajectory_count,
    ) = _apply_trajectory_metrics(results)

    trajectory_summaries = _trajectory_summaries(ordered_results)
    trajectory_count = len(trajectory_summaries)
    primary_trajectory = max(
        trajectory_summaries,
        key=lambda summary: (
            float(summary["total_length_px"]),
            int(summary["segment_count"]),
        ),
        default=None,
    )

    # Image-level features use the longest continuous trajectory. This avoids
    # allowing a short unrelated noise group to determine curvature or pattern.
    if primary_trajectory is not None:
        total_segment_length_px = float(primary_trajectory["total_length_px"])
        average_orientation = float(primary_trajectory["average_orientation"])
        average_angle_change_deg = float(primary_trajectory["average_angle_change"])
        curvature_rad_per_px = float(primary_trajectory["curvature_rad_per_px"])
        straightness = float(primary_trajectory["straightness"])
        trajectory_pattern = str(primary_trajectory["trajectory_pattern"])
        rule_based_classification = str(primary_trajectory["rule_based_classification"])
    else:
        total_segment_length_px = 0.0
        average_orientation = 0.0
        average_angle_change_deg = 0.0
        curvature_rad_per_px = 0.0
        straightness = 0.0
        trajectory_pattern = "Uncertain"
        rule_based_classification = "Uncertain"

    # Keep exported segment coordinates in the original image coordinate
    # system, while all detection and grouping above remained ROI-local.
    for result in ordered_results:
        result["x1"] = int(result["x1"]) + roi_x
        result["y1"] = int(result["y1"]) + roi_y
        result["x2"] = int(result["x2"]) + roi_x
        result["y2"] = int(result["y2"]) + roi_y
        result["midpoint_x"] = float(result["midpoint_x"]) + roi_x
        result["midpoint_y"] = float(result["midpoint_y"]) + roi_y

    total_length_cm: Optional[float] = None
    curvature_rad_per_cm: Optional[float] = None
    if PIXELS_PER_CM is not None:
        total_length_cm = total_segment_length_px / PIXELS_PER_CM
        curvature_rad_per_cm = curvature_rad_per_px * PIXELS_PER_CM

    print(f"Detected segments: {len(results)}")
    print(f"Total length: {total_segment_length_px:.2f} pixels")
    print(f"Average orientation: {average_orientation:.2f} degrees")
    print(f"Average angle change: {average_angle_change_deg:.2f} degrees")
    print(f"Curvature: {curvature_rad_per_px:.6f} rad/pixel")
    print(f"Straightness: {straightness:.4f}")
    print(f"Trajectory pattern: {trajectory_pattern}")
    print(f"Rule-based classification: {rule_based_classification}")
    print(f"Grouped trajectories: {trajectory_count}")
    for summary in trajectory_summaries:
        print(
            f"Trajectory {summary['trajectory_id']}: "
            f"Number of segments={summary['segment_count']}, "
            f"Total trajectory length={float(summary['total_length_px']):.2f} px, "
            f"Average orientation={float(summary['average_orientation']):.2f} degrees, "
            f"Average angle change={float(summary['average_angle_change']):.2f} degrees, "
            f"Curvature={float(summary['curvature_rad_per_px']):.6f} rad/pixel, "
            f"Straightness={float(summary['straightness']):.4f}, "
            f"Pattern={summary['trajectory_pattern']}, "
            f"Rule classification={summary['rule_based_classification']}"
        )
    # Hough Transform provides trajectory evidence. Final particle
    # classification should combine additional features such as track thickness.
    print("Hough output is trajectory evidence only; combine it with other features for final particle classification.")
    if total_length_cm is not None and curvature_rad_per_cm is not None:
        print(f"Track length: {total_length_cm:.2f} cm")
        print(f"Curvature: {curvature_rad_per_cm:.6f} rad/cm")

    result_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    measurements_path = result_dir / f"{stem}_measurements.csv"
    edges_path = result_dir / f"{stem}_edges.jpg"
    hough_result_path = result_dir / f"{stem}_hough_result.jpg"

    _write_measurements(
        measurements_path,
        ordered_results,
        total_angle_change_deg,
        average_angle_change_deg,
        curvature_rad_per_px,
        trajectory_pattern,
    )
    feature_path = _write_hough_feature_row(
        result_dir,
        image_path.name,
        resolved_roi,
        len(results),
        total_segment_length_px,
        average_orientation,
        average_angle_change_deg,
        curvature_rad_per_px,
        straightness,
        trajectory_pattern,
        rule_based_classification,
    )
    edge_canvas = np.zeros(image.shape[:2], dtype=np.uint8)
    edge_canvas[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width] = edges
    cv2.rectangle(
        output,
        (roi_x, roi_y),
        (roi_x + roi_width - 1, roi_y + roi_height - 1),
        (255, 0, 0),
        2,
    )
    if not cv2.imwrite(str(edges_path), edge_canvas):
        raise OSError(f"Could not save edge image: {edges_path}")
    if not cv2.imwrite(str(hough_result_path), output):
        raise OSError(f"Could not save Hough result image: {hough_result_path}")

    print(f"Saved: {hough_result_path.name}")
    print(f"Saved: {edges_path.name}")
    print(f"Saved: {measurements_path.name}")
    print(f"Saved: {feature_path.name}")
    return True


def test_hough_parameters(
    image_paths: Optional[Sequence[Path]] = None,
    result_dir: Path = RESULT_DIR,
    roi_boxes: Optional[Dict[str, ROI]] = None,
) -> Path:
    """Evaluate a small Probabilistic Hough parameter grid and save a CSV.

    Noise reduction is reported as the percentage of raw HoughLinesP()
    segments removed by the post-detection filters. Results are aggregated
    over the selected image set so combinations can be compared consistently.
    """

    selected_images = list(image_paths) if image_paths is not None else _find_supported_images(IMAGE_DIR)
    configured_boxes = roi_boxes if roi_boxes is not None else _load_roi_boxes()
    prepared_images: List[Tuple[Path, np.ndarray, np.ndarray]] = []
    for image_path in selected_images:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: cannot read image for parameter test, skipping: {image_path}")
            continue
        resolved_roi = _resolve_roi(image_path, image, roi_boxes=configured_boxes)
        if resolved_roi is None:
            print(f"Warning: no ROI for parameter test, skipping: {image_path.name}")
            continue
        x, y, width, height = resolved_roi
        roi_image = image[y : y + height, x : x + width]
        prepared_images.append((image_path, roi_image, _preprocess_image(roi_image)))

    comparison_rows: List[Dict[str, object]] = []
    for hough_threshold in (20, 30, 40):
        for min_line_length in (10, 20, 30):
            for max_line_gap in (5, 10, 15):
                raw_count = 0
                filtered_count = 0
                lengths: List[float] = []
                for _, image, edges in prepared_images:
                    raw_segments = _run_hough(
                        edges,
                        hough_threshold=hough_threshold,
                        min_line_length=min_line_length,
                        max_line_gap=max_line_gap,
                    )
                    filtered_segments = filter_hough_segments(
                        raw_segments,
                        image.shape[0],
                        min_line_length=min_line_length,
                    )
                    raw_count += len(raw_segments)
                    filtered_count += len(filtered_segments)
                    lengths.extend(calculate_length(*segment) for segment in filtered_segments)

                noise_reduction = (
                    100.0 * (raw_count - filtered_count) / raw_count
                    if raw_count
                    else 0.0
                )
                combination = (
                    f"HOUGH_THRESHOLD={hough_threshold}, "
                    f"MIN_LINE_LENGTH={min_line_length}, "
                    f"MAX_LINE_GAP={max_line_gap}"
                )
                comparison_rows.append(
                    {
                        "parameter_combination": combination,
                        "hough_threshold": hough_threshold,
                        "min_line_length": min_line_length,
                        "max_line_gap": max_line_gap,
                        "images_processed": len(prepared_images),
                        "number_of_detected_segments": filtered_count,
                        "average_segment_length_px": float(np.mean(lengths)) if lengths else 0.0,
                        "noise_reduction_percentage": noise_reduction,
                    }
                )

    result_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = result_dir / "hough_parameter_comparison.csv"
    fieldnames = [
        "parameter_combination",
        "hough_threshold",
        "min_line_length",
        "max_line_gap",
        "images_processed",
        "number_of_detected_segments",
        "average_segment_length_px",
        "noise_reduction_percentage",
    ]
    with comparison_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)

    print("\nHough parameter comparison")
    print("Parameter combination | Number of detected segments | Average segment length | Noise reduction")
    for row in comparison_rows:
        print(
            f"{row['parameter_combination']} | "
            f"{row['number_of_detected_segments']} | "
            f"{row['average_segment_length_px']:.2f} px | "
            f"{row['noise_reduction_percentage']:.2f}%"
        )
    print(f"Saved parameter comparison: {comparison_path}")
    return comparison_path


def _display_menu(image_paths: Sequence[Path]) -> None:
    """Display the available images and interactive menu choices."""

    print("\n================================")
    print("HOUGH TRANSFORM TRACK ANALYSIS")
    print("================================")
    print("ROI is required for every image; use roi_boxes.csv or select/enter it when prompted.")
    print("\nImages found in WeiQuan/image:\n")

    if image_paths:
        for index, image_path in enumerate(image_paths, start=1):
            print(f"[{index}] {image_path.name}")
        print("[A] Process all images")
        print("[T] Test Hough parameters")
        print("[R] Refresh image list")
        print("[Q] Quit")
        print("\nPlease select an image number, A, T, R, or Q:")
    else:
        print("No supported images found in WeiQuan/image.")
        print("Please add .jpg, .jpeg, .png, or .bmp files into the image folder.")
        print("[T] Test Hough parameters (when images are available)")
        print("\n[R] Refresh image list")
        print("[Q] Quit")
        print("\nPlease select T, R, or Q:")


def _process_images(image_paths: Sequence[Path]) -> None:
    """Process the selected images and print a batch summary."""

    successful = 0
    failed = 0
    roi_boxes = _load_roi_boxes()

    for image_path in image_paths:
        try:
            if process_image(
                image_path,
                RESULT_DIR,
                roi_boxes=roi_boxes,
                allow_roi_prompt=True,
            ):
                successful += 1
            else:
                failed += 1
        except Exception as error:
            failed += 1
            print(f"Warning: failed to process {image_path.name}: {error}")

    print("Processing completed.")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Results saved to: {RESULT_DIR.resolve()}")


def main() -> None:
    """Run the interactive image selection menu."""

    image_paths = _find_supported_images(IMAGE_DIR)

    while True:
        _display_menu(image_paths)
        choice = input("\n> ").strip()

        if choice.lower() == "q":
            print("Exiting program.")
            return

        if choice.lower() == "r":
            image_paths = _find_supported_images(IMAGE_DIR)
            print(f"\nImage list refreshed. Found {len(image_paths)} image(s).")
            continue

        if choice.lower() == "t":
            test_hough_parameters(image_paths, RESULT_DIR)
            continue

        if not image_paths:
            print("\nInvalid choice. Please enter R, T, or Q to quit.")
            continue

        if choice.lower() == "a":
            _process_images(image_paths)
            continue

        try:
            selected_index = int(choice) - 1
        except ValueError:
            print("\nInvalid choice. Please enter an image number, A, R, or Q.")
            continue

        if 0 <= selected_index < len(image_paths):
            _process_images([image_paths[selected_index]])
        else:
            print("\nInvalid image number. Please choose a number from the list.")


if __name__ == "__main__":
    main()
