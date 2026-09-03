"""
preprocessing.py — Shared frame enhancement for the YOLO cloud-chamber pipeline.

Cloud-chamber tracks are faint, low-contrast vapour trails. The exact same
enhancement is applied when building the training dataset AND when running
inference, so the model never sees a different image domain at test time
(a mismatch here is a common cause of "works on the val set, misses everything
on the video").

The single entry point is `enhance_frame()`; it returns a three-channel,
Otsu-binarised image ready for YOLO.
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


def apply_dog_high_pass(image):
    """Apply a Difference of Gaussians high-pass filter to reveal local tracks."""
    if image is None or image.size == 0:
        return image
    gray = _to_grayscale(image)

    small_blur = cv2.GaussianBlur(
        gray, (0, 0), config.DOG_SIGMA_SMALL
    )
    large_blur = cv2.GaussianBlur(
        gray, (0, 0), config.DOG_SIGMA_LARGE
    )
    dog = cv2.subtract(small_blur, large_blur)
    return cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)


def apply_otsu_binarisation(image):
    """Convert the high-pass response into a binary track mask using Otsu's method."""
    if image is None or image.size == 0:
        return image
    gray = _to_grayscale(image)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


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

    out = apply_dog_high_pass(out)
    return apply_otsu_binarisation(out)


def preprocess_for_yolo(image):
    """Return the exact image representation used by both YOLO training and inference."""
    return enhance_frame(image)
