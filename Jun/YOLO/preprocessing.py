"""
preprocessing.py — Shared frame enhancement for the YOLO cloud-chamber pipeline.

Cloud-chamber tracks are faint, low-contrast vapour trails. The exact same
enhancement is applied when building the training dataset AND when running
inference, so the model never sees a different image domain at test time
(a mismatch here is a common cause of "works on the val set, misses everything
on the video").

The single entry point is `enhance_frame()`; it returns a three-channel,
morphology-cleaned image ready for YOLO.
"""

import cv2
import numpy as np

import config


def _to_grayscale(image):
    """Convert to grayscale while preserving the original if already single-channel."""
    if image is None or image.size == 0:
        return image
    if len(image.shape) == 2:
        return image
    if image.shape[2] == 1:
        return image[:, :, 0]
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_morphology(image):
    """Remove isolated specks and retain bright, connected particle tracks."""
    if image is None or image.size == 0:
        return image
    gray = _to_grayscale(image)

    background_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, config.MORPH_BACKGROUND_KERNEL
    )
    clean_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, config.MORPH_CLEAN_KERNEL
    )

    denoised = cv2.medianBlur(gray, 3)
    bright_tracks = cv2.morphologyEx(
        denoised, cv2.MORPH_TOPHAT, background_kernel
    )
    _, binary = cv2.threshold(
        bright_tracks, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, clean_kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, clean_kernel)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    cleaned = np.zeros_like(binary)
    for component_id in range(1, component_count):
        area = stats[component_id, cv2.CC_STAT_AREA]
        if area >= config.MORPH_MIN_COMPONENT_AREA:
            cleaned[labels == component_id] = 255

    return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)


def apply_denoise(image):
    """Light grayscale denoising to suppress sensor noise while keeping track silhouettes."""
    if image is None or image.size == 0:
        return image
    gray = _to_grayscale(image)
    denoised = cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=7, searchWindowSize=21)
    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)


def enhance_frame(image):
    """
    Apply the shared enhancement pipeline to a BGR frame. Returns a new image
    (or the original untouched if enhancement is disabled in config).
    """
    if image is None:
        return image
    out = image.copy()

    if config.USE_DENOISING:
        out = apply_denoise(out)

    return apply_morphology(out)
