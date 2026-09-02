"""
prepare_dataset.py — Convert the red-box annotated cloud-chamber images into a
YOLO-format detection dataset.

For every image listed in config.IMAGE_LABEL_MAP this script:
  1. Detects the red annotation rectangles (HSV colour filtering).
  2. Sorts them by area (largest first) to match the label ordering.
  3. Converts each box to a normalised YOLO label (class cx cy w h).
  4. Inpaints the red rectangle away so the training image shows the *track*
     only — the red lines are an annotation artefact that will not exist in the
     inference video, so the detector must never learn to rely on them.
  5. Writes a reproducible train/val split plus a data.yaml manifest.

Run directly:
    python prepare_dataset.py
"""

import os
import random
import shutil

import cv2
import numpy as np
import yaml

import config
import preprocessing


def detect_red_boxes(image):
    """
    Detect red annotation rectangles and return them as (x, y, w, h) tuples
    sorted by area (largest first).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array(config.RED_HSV_LOWER1), np.array(config.RED_HSV_UPPER1))
    mask2 = cv2.inRange(hsv, np.array(config.RED_HSV_LOWER2), np.array(config.RED_HSV_UPPER2))
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Dilate so thin/broken rectangle edges join into one contour
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.dilate(red_mask, kernel, iterations=2)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > config.MIN_BOX_SIDE and h > config.MIN_BOX_SIDE:
            boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return boxes


def build_red_line_mask(image):
    """
    Build a binary mask of the red annotation pixels for inpainting.
    Dilated slightly so the anti-aliased fringe around each line is covered.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array(config.RED_HSV_LOWER1), np.array(config.RED_HSV_UPPER1))
    mask2 = cv2.inRange(hsv, np.array(config.RED_HSV_LOWER2), np.array(config.RED_HSV_UPPER2))
    red_mask = cv2.bitwise_or(mask1, mask2)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(red_mask, kernel, iterations=2)


def remove_red_annotations(image):
    """
    Return a copy of the image with the red rectangles inpainted away, so the
    underlying track is preserved but the annotation lines are gone.
    """
    mask = build_red_line_mask(image)
    return cv2.inpaint(image, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)


def find_source_image(name_substring):
    """Locate the source image whose filename contains name_substring."""
    for fname in os.listdir(config.IMAGES_DIR):
        if name_substring in fname:
            return os.path.join(config.IMAGES_DIR, fname)
    return None


def boxes_to_yolo_lines(boxes, labels, img_w, img_h):
    """
    Convert (x, y, w, h) boxes + class ids into YOLO label lines
    ('cls cx cy w h', all normalised to [0, 1]). Boxes are inset slightly so
    the label frames the track, not the red line.
    """
    lines = []
    n = min(len(boxes), len(labels))
    for i in range(n):
        x, y, w, h = boxes[i]
        cls = labels[i]

        inset = config.BOX_INSET
        x1 = max(0, x + inset)
        y1 = max(0, y + inset)
        x2 = min(img_w, x + w - inset)
        y2 = min(img_h, y + h - inset)
        if x2 <= x1 or y2 <= y1:
            continue

        cx = ((x1 + x2) / 2.0) / img_w
        cy = ((y1 + y2) / 2.0) / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines, n


def reset_dataset_dirs():
    """Create a clean images/{train,val} and labels/{train,val} tree."""
    if os.path.isdir(config.DATASET_DIR):
        shutil.rmtree(config.DATASET_DIR)
    for split in ("train", "val"):
        os.makedirs(os.path.join(config.DATASET_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(config.DATASET_DIR, "labels", split), exist_ok=True)


def prepare(verbose=True):
    """
    Build the full YOLO dataset. Returns a summary dict with counts.
    """
    reset_dataset_dirs()

    # Deterministic train/val split over the annotated image keys
    keys = list(config.IMAGE_LABEL_MAP.keys())
    rng = random.Random(config.SPLIT_SEED)
    rng.shuffle(keys)
    n_val = max(1, int(round(len(keys) * config.VAL_FRACTION)))
    val_keys = set(keys[:n_val])

    class_counts = {i: 0 for i in range(config.NUM_CLASSES)}
    stats = {
        "train_images": 0,
        "val_images": 0,
        "total_boxes": 0,
        "missing": [],
        "no_boxes": [],
        "mismatch": [],
    }

    for key in keys:
        labels = config.IMAGE_LABEL_MAP[key]
        src = find_source_image(key)
        if src is None:
            stats["missing"].append(key)
            if verbose:
                print(f"  [skip] no source image for '{key}'")
            continue

        image = cv2.imread(src)
        if image is None:
            stats["missing"].append(key)
            continue

        h, w = image.shape[:2]
        boxes = detect_red_boxes(image)
        if not boxes:
            stats["no_boxes"].append(key)
            if verbose:
                print(f"  [warn] no red boxes detected in '{os.path.basename(src)}'")
            continue

        if len(boxes) != len(labels):
            stats["mismatch"].append((os.path.basename(src), len(boxes), len(labels)))

        lines, used = boxes_to_yolo_lines(boxes, labels, w, h)
        if not lines:
            continue

        for i in range(used):
            class_counts[labels[i]] += 1
        stats["total_boxes"] += len(lines)

        # Clean image (red rectangles inpainted away) is what the model trains
        # on. Enhance it with the SAME pipeline used at inference time so the
        # training and video domains match.
        clean = remove_red_annotations(image)
        clean = preprocessing.enhance_frame(clean)

        split = "val" if key in val_keys else "train"
        stem = os.path.splitext(os.path.basename(src))[0]
        img_out = os.path.join(config.DATASET_DIR, "images", split, stem + ".jpg")
        lbl_out = os.path.join(config.DATASET_DIR, "labels", split, stem + ".txt")

        cv2.imwrite(img_out, clean)
        with open(lbl_out, "w") as f:
            f.write("\n".join(lines) + "\n")

        stats[f"{split}_images"] += 1

    _write_data_yaml()
    stats["class_counts"] = class_counts

    if verbose:
        _print_summary(stats)
    return stats


def _write_data_yaml():
    """Write the YOLO data.yaml manifest pointing at the generated splits."""
    data = {
        "path": config.DATASET_DIR.replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "nc": config.NUM_CLASSES,
        "names": config.CLASS_NAMES,
    }
    os.makedirs(config.DATASET_DIR, exist_ok=True)
    with open(config.DATA_YAML, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _print_summary(stats):
    print("=" * 60)
    print("DATASET PREPARATION SUMMARY")
    print("=" * 60)
    print(f"  Train images: {stats['train_images']}")
    print(f"  Val images:   {stats['val_images']}")
    print(f"  Total boxes:  {stats['total_boxes']}")
    print("\n  Boxes per class:")
    for cls_id, count in stats["class_counts"].items():
        print(f"    {cls_id} {config.CLASS_NAMES[cls_id]:<24} {count}")
    if stats["missing"]:
        print(f"\n  Missing source images: {len(stats['missing'])}")
    if stats["no_boxes"]:
        print(f"  Images with no detected boxes: {len(stats['no_boxes'])}")
    if stats["mismatch"]:
        print("\n  Box/label count mismatches (name, boxes, labels):")
        for name, nb, nl in stats["mismatch"]:
            print(f"    {name}: {nb} boxes vs {nl} labels")
    print(f"\n  data.yaml -> {config.DATA_YAML}")
    print("=" * 60)


if __name__ == "__main__":
    prepare()
