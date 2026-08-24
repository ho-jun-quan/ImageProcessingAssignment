"""
track_detector.py — Image preprocessing, track detection, red-box annotation
extraction, and skeletonisation for cloud chamber particle tracks.
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize
from collections import namedtuple

import config

# A detected track candidate with all the data needed for classification
TrackCandidate = namedtuple("TrackCandidate", [
    "bbox",           # (x, y, w, h) bounding box in the original frame
    "binary_mask",    # Binary mask of the track within the bbox crop
    "skeleton",       # 1-pixel skeleton (boolean array, same size as binary_mask)
    "grayscale_roi",  # Grayscale crop of the original frame at the bbox
])


# =============================================================================
# Preprocessing
# =============================================================================

def preprocess_frame(frame):
    """
    Convert a BGR frame to an enhanced grayscale image suitable for
    binarisation.

    Pipeline: BGR → Grayscale → CLAHE → Non-local Means Denoising
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    # CLAHE for local contrast enhancement
    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_TILE_GRID,
    )
    enhanced = clahe.apply(gray)

    # Non-local means denoising to smooth noise while preserving edges
    denoised = cv2.fastNlMeansDenoising(
        enhanced,
        None,
        h=config.DENOISE_H,
        templateWindowSize=config.DENOISE_TEMPLATE_WINDOW,
        searchWindowSize=config.DENOISE_SEARCH_WINDOW,
    )

    return denoised


def binarise(enhanced_gray, mask_text_overlay=True):
    """
    Convert an enhanced grayscale image to a clean binary mask where
    track pixels are white (255) and background is black (0).

    Pipeline: Heavy Gaussian Blur → Global Threshold → Morphological Open/Close
    """
    # Heavy blur to smooth out the noisy background grain
    blurred = cv2.GaussianBlur(enhanced_gray, config.BLUR_KERNEL, 0)
    
    # Global threshold (tracks are brighter than background after CLAHE + Blur)
    _, binary = cv2.threshold(
        blurred, 
        config.GLOBAL_THRESHOLD, 
        255, 
        cv2.THRESH_BINARY
    )

    # Morphological kernel
    kernel = np.ones(
        (config.MORPH_KERNEL_SIZE, config.MORPH_KERNEL_SIZE), np.uint8
    )

    # Morphological opening to remove small noise specks
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Morphological closing to bridge small gaps in tracks
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    # Mask out the text overlay region at the top of the frame
    if mask_text_overlay:
        closed[: config.TEXT_OVERLAY_HEIGHT, :] = 0

    return closed


# =============================================================================
# Track Detection
# =============================================================================

def detect_tracks(binary_mask):
    """
    Find individual track contours in a binary mask.

    Returns a list of (contour, bounding_box) tuples, filtered by area.
    bounding_box is (x, y, w, h).
    """
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    tracks = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if config.MIN_CONTOUR_AREA <= area <= config.MAX_CONTOUR_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            tracks.append((cnt, (x, y, w, h)))

    return tracks


def skeletonise_track(binary_roi):
    """
    Reduce a binary ROI (single track) to a 1-pixel-wide skeleton.

    Parameters
    ----------
    binary_roi : np.ndarray
        Binary image (0 or 255) of a single track.

    Returns
    -------
    skeleton : np.ndarray (bool)
        Boolean skeleton array.
    """
    boolean_mask = binary_roi > 0
    skeleton = skeletonize(boolean_mask)
    return skeleton


# =============================================================================
# Full Frame Pipeline
# =============================================================================

def extract_tracks_from_frame(frame, bg_subtractor=None):
    """
    Full detection pipeline for a single video frame.

    Parameters
    ----------
    frame : np.ndarray
        BGR video frame.
    bg_subtractor : cv2.BackgroundSubtractor or None
        If provided, uses background subtraction to isolate new tracks.
        If None, processes the raw frame directly.

    Returns
    -------
    tracks : list of TrackCandidate
    """
    # Optional background subtraction
    if bg_subtractor is not None:
        fg_mask = bg_subtractor.apply(frame)
        # Threshold the foreground mask
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Convert frame to grayscale and apply the foreground mask
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        masked_gray = cv2.bitwise_and(gray, gray, mask=fg_mask)

        # Enhance the masked region
        clahe = cv2.createCLAHE(
            clipLimit=config.CLAHE_CLIP_LIMIT,
            tileGridSize=config.CLAHE_TILE_GRID,
        )
        enhanced = clahe.apply(masked_gray)
    else:
        enhanced = preprocess_frame(frame)

    # Binarise
    binary = binarise(enhanced)

    # Detect contours
    detected = detect_tracks(binary)

    # Build TrackCandidate for each detection
    if len(frame.shape) == 3:
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_full = frame

    candidates = []
    for cnt, (x, y, w, h) in detected:
        # Crop the binary mask and grayscale ROI
        binary_roi = binary[y : y + h, x : x + w].copy()
        gray_roi = gray_full[y : y + h, x : x + w].copy()

        # Create a contour-specific mask within the ROI
        mask_roi = np.zeros_like(binary_roi)
        shifted_cnt = cnt - np.array([x, y])
        cv2.drawContours(mask_roi, [shifted_cnt], -1, 255, thickness=cv2.FILLED)
        binary_roi = cv2.bitwise_and(binary_roi, mask_roi)

        # Skeletonise
        skeleton = skeletonise_track(binary_roi)

        # Only keep if skeleton has enough pixels
        if np.sum(skeleton) >= 5:
            candidates.append(TrackCandidate(
                bbox=(x, y, w, h),
                binary_mask=binary_roi,
                skeleton=skeleton,
                grayscale_roi=gray_roi,
            ))

    return candidates


# =============================================================================
# Red Box Detection (Training Annotations)
# =============================================================================

def detect_red_boxes(image):
    """
    Detect red rectangular bounding boxes drawn on annotated images.

    The annotations are drawn as red rectangles. We detect them by colour
    filtering in HSV space, then finding rectangular contours.

    Parameters
    ----------
    image : np.ndarray
        BGR image with red annotation rectangles.

    Returns
    -------
    boxes : list of (x, y, w, h)
        Detected red bounding boxes, sorted by area (largest first).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Red wraps around in HSV, so combine two ranges
    mask1 = cv2.inRange(
        hsv,
        np.array(config.RED_HSV_LOWER1),
        np.array(config.RED_HSV_UPPER1),
    )
    mask2 = cv2.inRange(
        hsv,
        np.array(config.RED_HSV_LOWER2),
        np.array(config.RED_HSV_UPPER2),
    )
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Dilate to connect thin red lines into solid rectangles
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.dilate(red_mask, kernel, iterations=2)

    # Find contours of red regions
    contours, _ = cv2.findContours(
        red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        if perimeter < config.MIN_RED_BOX_PERIMETER:
            continue

        # Approximate the contour to a polygon
        approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)

        # We expect roughly rectangular shapes (4 vertices),
        # but we'll accept anything and use bounding rect
        x, y, w, h = cv2.boundingRect(cnt)

        # Filter out unreasonably small boxes
        if w > 30 and h > 30:
            boxes.append((x, y, w, h))

    # Sort by area (largest first) to match label ordering in config
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)

    return boxes


def extract_training_data_from_image(image_path, labels):
    """
    Extract training track candidates from a single annotated image.

    Parameters
    ----------
    image_path : str
        Path to the annotated image with red bounding boxes.
    labels : list of int
        Class labels for each detected red box (ordered by area, largest first).

    Returns
    -------
    tracks : list of (TrackCandidate, int)
        List of (track_candidate, label) pairs.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not load {image_path}")
        return []

    # Detect red annotation boxes
    red_boxes = detect_red_boxes(image)

    if len(red_boxes) == 0:
        print(f"Warning: No red boxes found in {image_path}")
        return []

    # Match boxes to labels (use min of available boxes and labels)
    n = min(len(red_boxes), len(labels))
    if len(red_boxes) != len(labels):
        print(
            f"Note: {image_path} — found {len(red_boxes)} boxes "
            f"but {len(labels)} labels provided. Using first {n}."
        )

    results = []
    for i in range(n):
        x, y, w, h = red_boxes[i]
        label = labels[i]

        # Crop the region inside the red box
        # Inset slightly to exclude the red line itself
        inset = 8
        x1 = max(0, x + inset)
        y1 = max(0, y + inset)
        x2 = min(image.shape[1], x + w - inset)
        y2 = min(image.shape[0], y + h - inset)

        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        # Process the ROI through the detection pipeline
        enhanced = preprocess_frame(roi)
        binary = binarise(enhanced, mask_text_overlay=False)

        # Detect tracks within this ROI
        detected = detect_tracks(binary)

        if len(roi.shape) == 3:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_roi = roi

        for cnt, (tx, ty, tw, th) in detected:
            track_binary = binary[ty : ty + th, tx : tx + tw].copy()
            track_gray = gray_roi[ty : ty + th, tx : tx + tw].copy()

            # Create contour mask
            mask = np.zeros_like(track_binary)
            shifted_cnt = cnt - np.array([tx, ty])
            cv2.drawContours(mask, [shifted_cnt], -1, 255, thickness=cv2.FILLED)
            track_binary = cv2.bitwise_and(track_binary, mask)

            skeleton = skeletonise_track(track_binary)

            if np.sum(skeleton) >= 5:
                # Adjust bbox to global coordinates
                global_bbox = (x1 + tx, y1 + ty, tw, th)
                candidate = TrackCandidate(
                    bbox=global_bbox,
                    binary_mask=track_binary,
                    skeleton=skeleton,
                    grayscale_roi=track_gray,
                )
                results.append((candidate, label))

    return results
