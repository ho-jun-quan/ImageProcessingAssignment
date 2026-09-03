"""
inference.py — Run a trained YOLO particle-track detector on images and video.

Provides:
  * load_model()            — load trained weights (falls back to base model)
  * detect_image()          — detect + annotate a single image (returns array)
  * process_image()         — detect + write a single annotated image
  * process_images()        — detect + write all images in a folder
  * process_video()         — detect frame-by-frame, write an annotated video

Recall-oriented tricks (all toggled in config) help catch the faint tracks that
whole-frame, high-confidence inference misses:
        * fast denoising, Difference of Gaussians high-pass filtering, and Otsu
            binarisation (identical to the dataset-build step)
  * test-time augmentation (multi-scale + flips)
  * tiled / SAHI-style inference with class-wise NMS merging
  * a low default confidence threshold

Run directly to process an image folder or the default video in config:
    python inference.py --images-dir ../Images
    python inference.py --video ../Videos/test_vid_cropped.mp4
"""

import argparse
import os

import cv2
import numpy as np
from ultralytics import YOLO

import config
import preprocessing


def load_model(weights=None):
    """
    Load a YOLO model from the given weights path. If none is supplied and the
    trained best.pt is missing, fall back to the pretrained base model (so the
    module still imports/runs before training).
    """
    if weights is None:
        weights = config.BEST_WEIGHTS if os.path.isfile(config.BEST_WEIGHTS) else config.BASE_MODEL
    if not os.path.isfile(weights):
        print(f"Note: '{weights}' not found; using base model '{config.BASE_MODEL}'.")
        weights = config.BASE_MODEL
    return YOLO(weights)


def _draw_detections(frame, boxes_xyxy, class_ids, confs, lengths_mm=None):
    """Draw class-coloured boxes + labels onto a BGR frame (in place copy)."""
    out = frame.copy()
    lengths_mm = lengths_mm or [None] * len(boxes_xyxy)
    for (x1, y1, x2, y2), cls_id, conf, length_mm in zip(
        boxes_xyxy, class_ids, confs, lengths_mm
    ):
        cls_id = int(cls_id)
        color = config.CLASS_COLORS.get(cls_id, (255, 255, 255))
        label = f"{config.DISPLAY_NAMES.get(cls_id, cls_id)} {conf:.2f}"
        if length_mm is not None:
            label += f" {length_mm:.1f} mm"

        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ytop = max(0, int(y1) - th - 6)
        cv2.rectangle(out, (int(x1), ytop), (int(x1) + tw + 4, int(y1)), color, -1)
        cv2.putText(out, label, (int(x1) + 2, int(y1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return out


def _show_stage_images(original, processed, annotated, annotated_unprocessed, delay_ms=800):
    """Display the original, preprocessed, annotated, and unprocessed-annotated images."""
    stages = {
        "Original": original,
        "Processed": processed,
        "Annotated": annotated,
        "Annotated (unprocessed)": annotated_unprocessed,
    }
    try:
        for title, image in stages.items():
            cv2.imshow(title, image)
        cv2.waitKey(delay_ms)
        for title in stages:
            cv2.destroyWindow(title)
    except cv2.error:
        pass


def _list_image_files(input_path):
    """Return a list of supported image paths from a file or directory."""
    if input_path is None:
        return []
    if os.path.isfile(input_path):
        return [input_path]
    if os.path.isdir(input_path):
        files = []
        for filename in sorted(os.listdir(input_path)):
            ext = os.path.splitext(filename)[1].lower()
            if ext in config.IMAGE_EXTENSIONS:
                files.append(os.path.join(input_path, filename))
        return files
    return []


def _calibrate_detections(boxes_xyxy, class_ids, confs, image_shape):
    """Keep detections inside the calibrated chamber and estimate bbox lengths."""
    height, width = image_shape[:2]
    x_min = config.CHAMBER_ROI_X[0] * width
    x_max = config.CHAMBER_ROI_X[1] * width
    y_min = config.CHAMBER_ROI_Y[0] * height
    y_max = config.CHAMBER_ROI_Y[1] * height
    mm_per_pixel_x = config.CHAMBER_WIDTH_MM / (x_max - x_min)
    mm_per_pixel_y = config.CHAMBER_HEIGHT_MM / (y_max - y_min)

    filtered_boxes = []
    filtered_classes = []
    filtered_confs = []
    lengths_mm = []
    for box, class_id, confidence in zip(boxes_xyxy, class_ids, confs):
        x1, y1, x2, y2 = (float(value) for value in box)
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        if config.FILTER_DETECTIONS_TO_CHAMBER and not (
            x_min <= center_x <= x_max and y_min <= center_y <= y_max
        ):
            continue

        x1 = max(x_min, min(x1, x_max))
        y1 = max(y_min, min(y1, y_max))
        x2 = max(x_min, min(x2, x_max))
        y2 = max(y_min, min(y2, y_max))
        if x2 <= x1 or y2 <= y1:
            continue

        width_mm = (x2 - x1) * mm_per_pixel_x
        height_mm = (y2 - y1) * mm_per_pixel_y
        filtered_boxes.append([x1, y1, x2, y2])
        filtered_classes.append(class_id)
        filtered_confs.append(confidence)
        lengths_mm.append(float(np.hypot(width_mm, height_mm)))

    return filtered_boxes, filtered_classes, filtered_confs, lengths_mm


def detect_image(model, image, conf=None, iou=None, use_tiling=None, use_tta=None):
    """
    Run detection on a single BGR image (path or array).

    Applies the shared denoising, DoG high-pass, and Otsu enhancement first, then either whole-frame or
    tiled inference (config.USE_TILING). Returns (annotated_image, detections)
    where detections is a list of dicts:
    {class_id, class_name, confidence, bbox=(x1,y1,x2,y2)}.
    """
    conf = conf if conf is not None else config.CONF_THRESHOLD
    iou = iou if iou is not None else config.IOU_THRESHOLD
    use_tiling = config.USE_TILING if use_tiling is None else use_tiling
    use_tta = config.USE_TTA if use_tta is None else use_tta

    if isinstance(image, str):
        image = cv2.imread(image)
    if image is None:
        raise ValueError("Could not read the input image.")

    # Same enhancement as the training data so the domains match.
    enhanced = preprocessing.preprocess_for_yolo(image)

    if use_tiling:
        boxes_xyxy, class_ids, confs = _detect_tiled(
            model, enhanced, conf, iou, use_tta
        )
    else:
        boxes_xyxy, class_ids, confs = _predict(
            model, enhanced, conf, iou, use_tta
        )

    boxes_xyxy, class_ids, confs, lengths_mm = _calibrate_detections(
        boxes_xyxy, class_ids, confs, enhanced.shape
    )

    detections = []
    for xyxy, cls_id, c, length_mm in zip(
        boxes_xyxy, class_ids, confs, lengths_mm
    ):
        cls_id = int(cls_id)
        detections.append({
            "class_id": cls_id,
            "class_name": config.CLASS_NAMES[cls_id] if cls_id < config.NUM_CLASSES else str(cls_id),
            "confidence": float(c),
            "bbox": tuple(int(v) for v in xyxy),
            "length_mm": length_mm,
        })

    # Draw on the ENHANCED frame so the overlay matches what the model saw.
    annotated = _draw_detections(enhanced, boxes_xyxy, class_ids, confs, lengths_mm)
    return annotated, detections


def process_image(model, input_path, output_path=None, conf=None,
                  iou=None, use_tiling=None, use_tta=None, display=True):
    """Load a single image, detect particles, and write annotated results into stage folders."""
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Image not found: {input_path}")

    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Could not read image: {input_path}")

    enhanced = preprocessing.preprocess_for_yolo(image)
    if use_tiling is None:
        use_tiling = config.USE_TILING
    if use_tta is None:
        use_tta = config.USE_TTA

    if use_tiling:
        boxes_xyxy, class_ids, confs = _detect_tiled(
            model, enhanced, conf if conf is not None else config.CONF_THRESHOLD,
            iou if iou is not None else config.IOU_THRESHOLD,
            use_tta,
        )
    else:
        boxes_xyxy, class_ids, confs = _predict(
            model, enhanced,
            conf if conf is not None else config.CONF_THRESHOLD,
            iou if iou is not None else config.IOU_THRESHOLD,
            use_tta,
        )

    boxes_xyxy, class_ids, confs, lengths_mm = _calibrate_detections(
        boxes_xyxy, class_ids, confs, enhanced.shape
    )

    detections = []
    for xyxy, cls_id, c, length_mm in zip(
        boxes_xyxy, class_ids, confs, lengths_mm
    ):
        cls_id = int(cls_id)
        detections.append({
            "class_id": cls_id,
            "class_name": config.CLASS_NAMES[cls_id] if cls_id < config.NUM_CLASSES else str(cls_id),
            "confidence": float(c),
            "bbox": tuple(int(v) for v in xyxy),
            "length_mm": length_mm,
        })

    annotated = _draw_detections(enhanced, boxes_xyxy, class_ids, confs, lengths_mm)
    annotated_unprocessed = _draw_detections(
        image, boxes_xyxy, class_ids, confs, lengths_mm
    )

    if output_path is None:
        output_dir = config.IMAGE_OUTPUT_DIR
    else:
        output_dir = os.path.dirname(output_path) or config.IMAGE_OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    original_path = os.path.join(output_dir, f"{base_name}_original.png")
    processed_path = os.path.join(output_dir, f"{base_name}_processed.png")
    annotated_path = os.path.join(output_dir, f"{base_name}_annotated.png")
    annotated_unprocessed_path = os.path.join(output_dir, f"{base_name}_annotated_unprocessed.png")

    cv2.imwrite(original_path, image)
    cv2.imwrite(processed_path, enhanced)
    cv2.imwrite(annotated_path, annotated)
    cv2.imwrite(annotated_unprocessed_path, annotated_unprocessed)

    if display:
        _show_stage_images(image, enhanced, annotated, annotated_unprocessed)

    return {
        "input_path": input_path,
        "output_dir": output_dir,
        "original_path": original_path,
        "processed_path": processed_path,
        "annotated_path": annotated_path,
        "annotated_unprocessed_path": annotated_unprocessed_path,
        "detections": detections,
        "count": len(detections),
    }


def process_images(model, input_dir=None, output_dir=None, conf=None,
                  iou=None, use_tiling=None, use_tta=None, display=True):
    """Process every supported image in a directory and save annotated results."""
    input_dir = input_dir or config.IMAGE_INPUT_DIR
    output_dir = output_dir or config.IMAGE_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    files = _list_image_files(input_dir)
    if not files:
        print(f"No image files found in: {input_dir}")
        return []

    results = []
    for image_path in files:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        result = process_image(
            model, image_path, output_path=os.path.join(output_dir, f"{base_name}_annotated.png"),
            conf=conf, iou=iou,
            use_tiling=use_tiling, use_tta=use_tta, display=display,
        )
        results.append(result)
        print(f"Processed {image_path} -> {result['output_dir']} ({result['count']} detections)")

    return results


def _predict(model, image, conf, iou, use_tta):
    """Whole-image predict; returns (boxes_xyxy, class_ids, confs) as lists."""
    result = model.predict(image, conf=conf, iou=iou, augment=use_tta, verbose=False)[0]
    boxes_xyxy, class_ids, confs = [], [], []
    for box in result.boxes:
        boxes_xyxy.append(box.xyxy[0].tolist())
        class_ids.append(int(box.cls[0]))
        confs.append(float(box.conf[0]))
    return boxes_xyxy, class_ids, confs


def _detect_tiled(model, image, conf, iou, use_tta):
    """
    SAHI-style tiled inference: slice the frame into overlapping tiles, detect
    in each, shift the boxes back to full-frame coordinates, then merge with
    class-wise NMS. Recovers small/faint tracks that whole-frame inference
    misses on high-resolution footage.
    """
    h, w = image.shape[:2]
    rows, cols = config.TILE_ROWS, config.TILE_COLS
    tile_conf = min(conf, config.TILE_CONF_THRESHOLD)

    tile_h = h // rows
    tile_w = w // cols
    ov_y = int(tile_h * config.TILE_OVERLAP)
    ov_x = int(tile_w * config.TILE_OVERLAP)

    all_boxes, all_cls, all_conf = [], [], []
    for r in range(rows):
        for c in range(cols):
            y0 = max(0, r * tile_h - ov_y)
            x0 = max(0, c * tile_w - ov_x)
            y1 = min(h, (r + 1) * tile_h + ov_y)
            x1 = min(w, (c + 1) * tile_w + ov_x)
            tile = image[y0:y1, x0:x1]
            if tile.size == 0:
                continue

            b, ci, cf = _predict(model, tile, tile_conf, iou, use_tta)
            for (bx1, by1, bx2, by2), cls_id, score in zip(b, ci, cf):
                all_boxes.append([bx1 + x0, by1 + y0, bx2 + x0, by2 + y0])
                all_cls.append(cls_id)
                all_conf.append(score)

    # Also run the whole (downscaled) frame so large tracks spanning tiles are
    # not lost at tile seams.
    b, ci, cf = _predict(model, image, conf, iou, use_tta)
    all_boxes.extend([list(x) for x in b])
    all_cls.extend(ci)
    all_conf.extend(cf)

    boxes, classes, scores = _class_wise_nms(all_boxes, all_cls, all_conf, iou)
    return _suppress_cross_class_duplicates(boxes, classes, scores)


def _class_wise_nms(boxes, class_ids, confs, iou_thresh):
    """Merge overlapping detections per class using OpenCV NMS."""
    if not boxes:
        return [], [], []

    keep_boxes, keep_cls, keep_conf = [], [], []
    for cls_id in set(class_ids):
        idxs = [i for i, c in enumerate(class_ids) if c == cls_id]
        rects = [[boxes[i][0], boxes[i][1],
                  boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1]] for i in idxs]
        scores = [confs[i] for i in idxs]
        picked = cv2.dnn.NMSBoxes(rects, scores, config.TILE_CONF_THRESHOLD, iou_thresh)
        for p in np.array(picked).flatten():
            i = idxs[int(p)]
            keep_boxes.append(boxes[i])
            keep_cls.append(class_ids[i])
            keep_conf.append(confs[i])
    return keep_boxes, keep_cls, keep_conf


def _suppress_cross_class_duplicates(boxes, class_ids, confs):
    """Remove near-identical boxes predicted with competing class labels."""
    if len(boxes) < 2:
        return boxes, class_ids, confs

    order = sorted(range(len(boxes)), key=lambda i: confs[i], reverse=True)
    kept = []
    for index in order:
        candidate = boxes[index]
        candidate_area = max(0, candidate[2] - candidate[0]) * max(0, candidate[3] - candidate[1])
        is_duplicate = False
        for kept_index in kept:
            other = boxes[kept_index]
            ix1 = max(candidate[0], other[0])
            iy1 = max(candidate[1], other[1])
            ix2 = min(candidate[2], other[2])
            iy2 = min(candidate[3], other[3])
            intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            other_area = max(0, other[2] - other[0]) * max(0, other[3] - other[1])
            union = candidate_area + other_area - intersection
            if union and intersection / union >= config.DUPLICATE_IOU_THRESHOLD:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(index)

    return ([boxes[i] for i in kept], [class_ids[i] for i in kept],
            [confs[i] for i in kept])


def process_video(model, input_path=None, output_path=None, conf=None,
                  iou=None, max_frames=None, progress_every=100,
                  use_tiling=None, use_tta=None):
    """
    Detect particle tracks in every frame of a video and write an annotated
    output video. Returns a stats dict (frame/detection counts per class).

    Tiling + TTA greatly improve recall but are slow per frame (especially on
    CPU); pass use_tiling=False / use_tta=False for a fast preview.
    """
    input_path = input_path or config.VIDEO_INPUT
    output_path = output_path or config.VIDEO_OUTPUT
    conf = conf if conf is not None else config.CONF_THRESHOLD
    iou = iou if iou is not None else config.IOU_THRESHOLD

    if not os.path.isfile(input_path):
        print(f"Video not found: {input_path}")
        return None

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Could not open video: {input_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_CODEC)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Video: {input_path}")
    print(f"  Resolution: {width}x{height}  FPS: {fps:.1f}  Frames: {total}")

    class_counts = {i: 0 for i in range(config.NUM_CLASSES)}
    frame_idx = 0
    total_dets = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        annotated, detections = detect_image(
            model, frame, conf=conf, iou=iou,
            use_tiling=use_tiling, use_tta=use_tta,
        )
        for det in detections:
            class_counts[det["class_id"]] += 1
            total_dets += 1

        writer.write(annotated)

        if progress_every and frame_idx % progress_every == 0:
            pct = 100.0 * frame_idx / total if total else 0.0
            print(f"  Frame {frame_idx}/{total} ({pct:.1f}%) — {total_dets} detections so far")

        if max_frames and frame_idx >= max_frames:
            break

    cap.release()
    writer.release()

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Output saved to: {output_path}")
    print(f"Frames processed: {frame_idx}")
    print(f"Total detections: {total_dets}\n")
    print("Detections per class:")
    for cls_id, count in class_counts.items():
        print(f"  {config.CLASS_NAMES[cls_id]}: {count}")

    return {
        "frames": frame_idx,
        "total_detections": total_dets,
        "class_counts": class_counts,
        "output_path": output_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO detection on images or videos.")
    parser.add_argument("--image", type=str, default=None, help="Single input image path")
    parser.add_argument("--images-dir", type=str, default=None, help="Directory of input images")
    parser.add_argument("--video", type=str, default=None, help="Single input video path")
    parser.add_argument("--output", type=str, default=None, help="Output path for a single image/video")
    args = parser.parse_args()

    model = load_model()

    if args.image:
        process_image(model, args.image, output_path=args.output)
    elif args.images_dir:
        process_images(model, input_dir=args.images_dir, output_dir=args.output)
    else:
        process_video(model, input_path=args.video or config.VIDEO_INPUT,
                      output_path=args.output or config.VIDEO_OUTPUT)
