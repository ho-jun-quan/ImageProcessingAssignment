"""
extract_frames.py — Pull frames from the source videos to expand the dataset.

The single biggest limitation of the current detector is the tiny, sparsely
labelled dataset (~32 images, ~7 boxes per class, and most tracks per frame are
left unlabelled). This helper samples frames from the videos in ../Videos so
you can annotate more of them (label EVERY visible track in each frame).

The extracted frames use the same denoising, DoG high-pass, and Otsu pipeline used everywhere
else, so what you annotate matches what the model trains and predicts on.

Usage:
    python extract_frames.py                     # sample every Nth frame from all videos
    python extract_frames.py --every 30          # one frame every 30 frames
    python extract_frames.py --video ../Videos/test_vid.mp4 --every 15
    python extract_frames.py --no-enhance        # keep raw frames

Annotate the results (e.g. in LabelImg / Roboflow / CVAT) into YOLO txt files,
or draw red boxes and extend config.IMAGE_LABEL_MAP as before.
"""

import argparse
import os

import cv2

import config
import preprocessing

# Where sampled frames are written for annotation.
FRAMES_OUT_DIR = os.path.join(config.BASE_DIR, "extracted_frames")


def extract_from_video(video_path, every=30, enhance=True, out_dir=FRAMES_OUT_DIR):
    """Sample one frame every `every` frames from a single video."""
    if not os.path.isfile(video_path):
        print(f"  [skip] not found: {video_path}")
        return 0

    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [skip] could not open: {video_path}")
        return 0

    stem = os.path.splitext(os.path.basename(video_path))[0]
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {os.path.basename(video_path)}  frames={total}  every={every}")

    saved = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % every == 0:
            if enhance:
                frame = preprocessing.preprocess_for_yolo(frame)
            out_path = os.path.join(out_dir, f"{stem}_f{idx:06d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1
        idx += 1

    cap.release()
    print(f"  saved {saved} frames -> {out_dir}")
    return saved


def extract_all(every=30, enhance=True, video=None, out_dir=FRAMES_OUT_DIR):
    """Extract from a single video (if given) or every video in ../Videos."""
    if video:
        videos = [video]
    else:
        exts = (".mp4", ".avi", ".mov", ".mkv")
        videos = [
            os.path.join(config.VIDEOS_DIR, f)
            for f in sorted(os.listdir(config.VIDEOS_DIR))
            if f.lower().endswith(exts)
        ]

    total_saved = sum(extract_from_video(v, every, enhance, out_dir) for v in videos)
    print(f"\nDone. {total_saved} frames written to {out_dir}")
    print("Next: label EVERY visible track in each frame, then rebuild the dataset.")
    return total_saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames for annotation.")
    parser.add_argument("--video", default=None, help="Single video path (default: all in ../Videos)")
    parser.add_argument("--every", type=int, default=30, help="Sample one frame every N frames")
    parser.add_argument("--no-enhance", action="store_true", help="Skip morphology enhancement")
    parser.add_argument("--out", default=FRAMES_OUT_DIR, help="Output directory")
    args = parser.parse_args()

    extract_all(every=args.every, enhance=not args.no_enhance,
                video=args.video, out_dir=args.out)
