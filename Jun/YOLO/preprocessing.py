"""
preprocessing.py — Shared frame enhancement for the YOLO cloud-chamber pipeline.

Cloud-chamber tracks are faint, low-contrast vapour trails. The exact same
enhancement is applied when building the training dataset AND when running
inference, so the model never sees a different image domain at test time
(a mismatch here is a common cause of "works on the val set, misses everything
on the video").

The single entry point is `enhance_frame()`; it is a no-op unless
`config.USE_CLAHE` is set.
"""

import cv2

import config


def apply_clahe(image):
    """
    Contrast Limited Adaptive Histogram Equalisation on the luminance (L)
    channel of a BGR image. Boosts local contrast so faint tracks stand out
    without blowing out the already-bright regions.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=tuple(config.CLAHE_TILE_GRID),
    )
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def enhance_frame(image):
    """
    Apply the shared enhancement pipeline to a BGR frame. Returns a new image
    (or the original untouched if enhancement is disabled in config).
    """
    if image is None:
        return image
    out = image
    if config.USE_CLAHE:
        out = apply_clahe(out)
    return out
