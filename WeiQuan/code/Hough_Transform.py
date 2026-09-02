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
RESULT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MIN_VALID_LINE_LENGTH = 20
BOTTOM_BORDER_MARGIN_RATIO = 0.08
BOTTOM_CAPTION_MAX_ANGLE_DEG = 15

# Leave as None until calibration is known.
# Example: if 50 pixels = 1 cm, set PIXELS_PER_CM = 50.
PIXELS_PER_CM = None

Measurement = Dict[str, object]


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
    detected_segments: int,
    average_angle_change_deg: float,
    curvature_rad_per_px: float,
) -> str:
    """Return a preliminary Hough-based trajectory pattern suggestion."""

    # This is a Hough-based trajectory suggestion, not the final particle classifier.
    if detected_segments == 0:
        return "Uncertain - no clear Hough segment detected"
    if detected_segments < 3 and average_angle_change_deg < 15:
        return "Straight trajectory - possible alpha-like evidence"
    if average_angle_change_deg >= 20 or curvature_rad_per_px >= 0.005:
        return "Curved trajectory - possible electron-like evidence"
    return "Uncertain - requires contour or skeleton features"


def calculate_curvature(
    results: Sequence[Measurement],
) -> Tuple[List[Measurement], float, float, float]:
    """Order segments and calculate the approximate track curvature.

    The detected segment midpoints are ordered along their principal axis.
    Curvature is then approximated as total orientation change divided by the
    distance travelled between neighbouring segment centres.
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
    total_path_distance_px = 0.0

    for current, next_segment in zip(ordered_results, ordered_results[1:]):
        total_angle_change_deg += angle_difference(
            current["orientation_deg"], next_segment["orientation_deg"]
        )
        dx = next_segment["midpoint_x"] - current["midpoint_x"]
        dy = next_segment["midpoint_y"] - current["midpoint_y"]
        total_path_distance_px += math.sqrt(dx**2 + dy**2)

    if total_path_distance_px > 0:
        curvature_rad_per_px = math.radians(total_angle_change_deg) / total_path_distance_px
    else:
        curvature_rad_per_px = 0.0

    return (
        ordered_results,
        total_angle_change_deg,
        total_path_distance_px,
        curvature_rad_per_px,
    )


def _measurement_fieldnames() -> List[str]:
    return [
        "segment",
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
        "trajectory_pattern",
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
            row["trajectory_pattern"] = trajectory_pattern
            writer.writerow(row)

        # Keep the trajectory pattern available in the CSV even when no segments were
        # detected and therefore there are no per-segment measurement rows.
        if not results:
            writer.writerow(
                {
                    "total_angle_change_deg": total_angle_change_deg,
                    "average_angle_change_deg": average_angle_change_deg,
                    "curvature_rad_per_px": curvature_rad_per_px,
                    "trajectory_pattern": trajectory_pattern,
                }
            )


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


def process_image(image_path: Path, result_dir: Path) -> bool:
    """Run the complete Hough Transform pipeline for one image."""

    print(f"Processing: {image_path.name}")
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Warning: cannot read image, skipping: {image_path}")
        return False

    output = image.copy()

    # Greyscale conversion, CLAHE contrast enhancement, and Gaussian blur.
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(grey)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Canny Edge Detection.
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    # Probabilistic Hough Line Transform.
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=10,
        maxLineGap=8,
    )

    results: List[Measurement] = []
    if lines is not None:
        # OpenCV may return either (N, 1, 4) or (N, 4).
        lines = np.asarray(lines).reshape(-1, 4)

        valid_lines = []
        for x1, y1, x2, y2 in lines:
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            length = calculate_length(x1, y1, x2, y2)
            if length <= MIN_VALID_LINE_LENGTH:
                continue
            if _looks_like_bottom_caption_artifact(
                x1, y1, x2, y2, image.shape[0]
            ):
                continue
            valid_lines.append((x1, y1, x2, y2))

        for index, (x1, y1, x2, y2) in enumerate(valid_lines):
            length = calculate_length(x1, y1, x2, y2)
            orientation = calculate_orientation(x1, y1, x2, y2)

            results.append(
                {
                    "segment": index + 1,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "midpoint_x": (x1 + x2) / 2,
                    "midpoint_y": (y1 + y2) / 2,
                    "length_px": length,
                    "orientation_deg": orientation,
                    "curvature_rad_per_px": 0.0,
                }
            )

            # Draw detected Hough line segments.
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

    (
        ordered_results,
        total_angle_change_deg,
        total_path_distance_px,
        curvature_rad_per_px,
    ) = calculate_curvature(results)
    for result in ordered_results:
        result["curvature_rad_per_px"] = curvature_rad_per_px

    total_segment_length_px = sum(result["length_px"] for result in results)
    average_orientation = (
        float(np.mean([result["orientation_deg"] for result in results]))
        if results
        else 0.0
    )
    number_of_angle_comparisons = max(len(ordered_results) - 1, 0)
    average_angle_change_deg = (
        total_angle_change_deg / number_of_angle_comparisons
        if number_of_angle_comparisons > 0
        else 0.0
    )

    total_length_cm: Optional[float] = None
    curvature_rad_per_cm: Optional[float] = None
    if PIXELS_PER_CM is not None:
        total_length_cm = total_segment_length_px / PIXELS_PER_CM
        curvature_rad_per_cm = curvature_rad_per_px * PIXELS_PER_CM

    trajectory_pattern = suggest_trajectory_pattern(
        len(results),
        average_angle_change_deg,
        curvature_rad_per_px,
    )

    print(f"Detected segments: {len(results)}")
    print(f"Total detected segment length: {total_segment_length_px:.2f} pixels")
    print(f"Average orientation: {average_orientation:.2f} degrees")
    print(f"Total orientation change: {total_angle_change_deg:.2f} degrees")
    print(f"Average orientation change: {average_angle_change_deg:.2f} degrees")
    print(f"Approximate curvature: {curvature_rad_per_px:.6f} rad/pixel")
    print(f"Hough-based trajectory pattern: {trajectory_pattern}")
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
    if not cv2.imwrite(str(edges_path), edges):
        raise OSError(f"Could not save edge image: {edges_path}")
    if not cv2.imwrite(str(hough_result_path), output):
        raise OSError(f"Could not save Hough result image: {hough_result_path}")

    print(f"Saved: {hough_result_path.name}")
    print(f"Saved: {edges_path.name}")
    print(f"Saved: {measurements_path.name}")
    return True


def _display_menu(image_paths: Sequence[Path]) -> None:
    """Display the available images and interactive menu choices."""

    print("\n================================")
    print("HOUGH TRANSFORM TRACK ANALYSIS")
    print("================================")
    print("\nImages found in WeiQuan/image:\n")

    if image_paths:
        for index, image_path in enumerate(image_paths, start=1):
            print(f"[{index}] {image_path.name}")
        print("[A] Process all images")
        print("[R] Refresh image list")
        print("[Q] Quit")
        print("\nPlease select an image number, A, R, or Q:")
    else:
        print("No supported images found in WeiQuan/image.")
        print("Please add .jpg, .jpeg, .png, or .bmp files into the image folder.")
        print("\n[R] Refresh image list")
        print("[Q] Quit")
        print("\nPlease select R or Q:")


def _process_images(image_paths: Sequence[Path]) -> None:
    """Process the selected images and print a batch summary."""

    successful = 0
    failed = 0

    for image_path in image_paths:
        try:
            if process_image(image_path, RESULT_DIR):
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

        if not image_paths:
            print("\nInvalid choice. Please enter R to refresh or Q to quit.")
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
