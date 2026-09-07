"""Standalone viewer for the shared Canny + HoughLinesP pipeline."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "WeiQuan", "code"))
from hough_pipeline import analyze_frame


def process_frame(image):
    """Return annotated trajectory evidence, Canny edges, and feature records."""

    annotated, edges, summaries, detections = analyze_frame(image)
    return annotated, edges, detections


def process_cloud_chamber_image(image_path: str) -> None:
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image.")
        return
    annotated, edges, detections = process_frame(image)
    for detection in detections:
        print(detection)
    scale = 0.5
    size = (int(annotated.shape[1] * scale), int(annotated.shape[0] * scale))
    resized_edges = cv2.resize(edges, size, interpolation=cv2.INTER_AREA)
    resized_annotated = cv2.resize(annotated, size, interpolation=cv2.INTER_AREA)
    cv2.imwrite("result_edges.jpg", resized_edges)
    cv2.imwrite("result_hough_trajectories.jpg", resized_annotated)
    print("Saved: result_edges.jpg and result_hough_trajectories.jpg")
    cv2.imshow("Canny Edges", resized_edges)
    cv2.imshow("Hough Trajectory Evidence", resized_annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_cloud_chamber_video(video_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = max(1, int(1000 / fps))
    print("Press Q to quit, P to pause/unpause.")
    paused = False
    frame_number = 0
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            frame_number += 1
            annotated, _, detections = process_frame(frame)
            if detections:
                print(f"Frame {frame_number}: {len(detections)} trajectory record(s)")
            cv2.imshow("Hough Trajectory Evidence", cv2.resize(annotated, None, fx=0.5, fy=0.5))
        key = cv2.waitKey(delay) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        if key in (ord("p"), ord("P")):
            paused = not paused
    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected_file = filedialog.askopenfilename(
        title="Select a cloud-chamber image or video",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp"),
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    if not selected_file:
        print("No file was selected.")
        return
    extension = Path(selected_file).suffix.lower()
    if extension in {".mp4", ".avi", ".mov", ".mkv"}:
        process_cloud_chamber_video(selected_file)
    else:
        process_cloud_chamber_image(selected_file)


if __name__ == "__main__":
    main()
