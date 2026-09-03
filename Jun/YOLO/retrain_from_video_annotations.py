"""Build a YOLO dataset directly from CVAT XML annotations on the video in ../Videos.

This script reads the single video and its matching annotations.xml file,
extracts a sample of frames, converts each annotated track into a YOLO label,
and writes a dataset ready for `train.py`.
"""

import argparse
import os
import random
import shutil
from xml.etree import ElementTree as ET

import cv2
import numpy as np
import yaml

import config
import preprocessing


def _polyline_to_box(points_str):
    """Convert a CVAT polyline string into a bounding box."""
    if not points_str:
        return None

    pts = []
    for pair in points_str.split(';'):
        if not pair or ',' not in pair:
            continue
        x_str, y_str = pair.split(',', 1)
        try:
            pts.append((float(x_str), float(y_str)))
        except ValueError:
            continue

    if not pts:
        return None

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def parse_cvat_annotations(xml_path):
    """Return a list of annotations: {frame, label, x1, y1, x2, y2}."""
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f"CVAT annotations file not found: {xml_path}")

    root = ET.parse(xml_path).getroot()
    annotations = []
    track_map = {}

    for track in root.findall('track'):
        label_name = track.get('label')
        if label_name not in config.CVAT_LABEL_MAP:
            continue
        track_id = track.get('id')
        track_map[track_id] = config.CVAT_LABEL_MAP[label_name]

    for track in root.findall('track'):
        track_id = track.get('id')
        if track_id not in track_map:
            continue
        label_id = track_map[track_id]

        for box in track.findall('box'):
            frame = int(box.get('frame'))
            outside = box.get('outside')
            if outside == '1':
                continue
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))
            annotations.append({
                'frame': frame,
                'label': label_id,
                'x1': xtl,
                'y1': ytl,
                'x2': xbr,
                'y2': ybr,
            })

        for polyline in track.findall('polyline'):
            frame = int(polyline.get('frame'))
            outside = polyline.get('outside')
            if outside == '1':
                continue
            points = polyline.get('points')
            poly_box = _polyline_to_box(points)
            if poly_box is None:
                continue
            x1, y1, x2, y2 = poly_box
            annotations.append({
                'frame': frame,
                'label': label_id,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
            })

    return annotations


def export_training_frames(video_path, annotations, output_dir, frames_per_second=None, skip_existing=True):
    """Write every selected annotated video frame with matching YOLO labels."""
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')
    for split in ('train', 'val'):
        os.makedirs(os.path.join(images_dir, split), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, split), exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video: {video_path}')

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frame_to_annotations = {}
    for ann in annotations:
        frame_to_annotations.setdefault(int(ann['frame']), []).append(ann)

    annotated_frames = sorted(frame_to_annotations.keys())
    if frames_per_second and frames_per_second > 0:
        sample_stride = max(1, int(round(fps / frames_per_second)))
        sampled_frames = [
            frame_idx for frame_idx in annotated_frames
            if frame_idx % sample_stride == 0
        ]
    else:
        sampled_frames = annotated_frames

    rng = random.Random(42)
    sampled_frames = sampled_frames.copy()
    rng.shuffle(sampled_frames)
    n_val = max(1, int(round(len(sampled_frames) * 0.2)))
    val_frames = set(sampled_frames[:n_val])

    saved = 0
    class_counts = {i: 0 for i in range(config.NUM_CLASSES)}
    for frame_idx in sampled_frames:
        # The frame list is shuffled for a reproducible split, so sequential
        # cap.read() would associate annotations with the wrong video frame.
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue

        split = 'val' if frame_idx in val_frames else 'train'
        stem = f'frame_{frame_idx:06d}'
        image_path = os.path.join(images_dir, split, f'{stem}.png')
        label_path = os.path.join(labels_dir, split, f'{stem}.txt')

        if skip_existing and os.path.isfile(image_path):
            continue

        lines = []
        for ann in frame_to_annotations[frame_idx]:
            x1 = max(0.0, float(ann['x1']))
            y1 = max(0.0, float(ann['y1']))
            x2 = min(float(frame.shape[1]), float(ann['x2']))
            y2 = min(float(frame.shape[0]), float(ann['y2']))
            if x2 <= x1 or y2 <= y1:
                continue
            cx = ((x1 + x2) / 2.0) / frame.shape[1]
            cy = ((y1 + y2) / 2.0) / frame.shape[0]
            bw = (x2 - x1) / frame.shape[1]
            bh = (y2 - y1) / frame.shape[0]
            lines.append(f"{int(ann['label'])} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            class_counts[int(ann['label'])] += 1

        if not lines:
            continue

        processed = preprocessing.preprocess_for_yolo(frame)
        cv2.imwrite(image_path, processed)
        with open(label_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        saved += 1

    cap.release()
    return saved, class_counts


def write_data_yaml(dataset_dir):
    """Write YOLO data.yaml."""
    data = {
        'path': dataset_dir.replace('\\', '/'),
        'train': 'images/train',
        'val': 'images/val',
        'nc': config.NUM_CLASSES,
        'names': config.CLASS_NAMES,
    }
    with open(os.path.join(dataset_dir, 'data.yaml'), 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False)


def prepare_video_dataset(video_path, xml_path, dataset_dir, frames_per_second=None):
    """Generate a YOLO dataset based on the video + CVAT XML annotations."""
    if os.path.isdir(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.makedirs(dataset_dir, exist_ok=True)

    annotations = parse_cvat_annotations(xml_path)
    if not annotations:
        raise ValueError(f'No valid track annotations found in {xml_path}')

    saved, class_counts = export_training_frames(video_path, annotations, dataset_dir, frames_per_second=frames_per_second)
    write_data_yaml(dataset_dir)

    print('=' * 70)
    print('VIDEO-BASED YOLO DATASET READY')
    print('=' * 70)
    print(f'Video: {video_path}')
    print(f'Annotations: {xml_path}')
    print(f'Frames written: {saved}')
    print('Class counts:')
    for idx, count in class_counts.items():
        print(f'  {idx}: {config.CLASS_NAMES[idx]} -> {count}')
    print(f'Dataset dir: {dataset_dir}')
    print(f'data.yaml: {os.path.join(dataset_dir, "data.yaml")}')
    return dataset_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build a YOLO dataset from a CVAT XML annotation file tied to a video.')
    parser.add_argument('--video', default=config.VIDEO_SOURCE, help='Video file to sample frames from.')
    parser.add_argument('--xml', default=config.VIDEO_ANNOTATIONS_XML, help='CVAT annotation XML file.')
    parser.add_argument('--output', default=os.path.join(config.DATASET_DIR, 'video_annotations_dataset'), help='Output dataset directory.')
    parser.add_argument('--fps', type=int, default=None, help='Optional sampling rate; default exports every annotated frame.')
    args = parser.parse_args()

    prepare_video_dataset(args.video, args.xml, args.output, frames_per_second=args.fps)
