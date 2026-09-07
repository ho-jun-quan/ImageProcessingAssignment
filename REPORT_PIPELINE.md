# Cloud Chamber HoughLinesP Pipeline

The report pipeline is centred on the Probabilistic Hough Transform
(`cv2.HoughLinesP`) for particle-trajectory detection.

## Processing stages

1. Restrict the image to a reproducible region of interest (ROI). A per-image
   ROI can be supplied in `WeiQuan/roi_boxes.csv`; otherwise the interior 90%
   of the frame is used.
2. Apply median denoising, CLAHE, and local background subtraction to reduce
   uneven illumination and isolated background specks.
3. Apply Canny edge detection, then remove tiny connected components and use
   morphology to reconnect broken track edges.
4. Run `HoughLinesP` to detect candidate line segments. Short, border-like,
   isolated, and duplicate detections are filtered.
5. Group nearby segments with endpoint distance, midpoint distance, and local
   orientation consistency.
6. Report trajectory length, orientation angle, straightness, curvature, and
   one of three evidence labels: `Straight trajectory`, `Curved trajectory`,
   or `Uncertain`.

The pipeline does not force Alpha/Electron classification. Those labels require
additional calibrated physical features such as track width, brightness, and
energy-loss evidence.

## Run

From the repository root:

```powershell
python WeiQuan/code/Hough_Transform.py
```

The maintained implementation is `WeiQuan/code/hough_pipeline.py`; the
existing `Hough_Transform.py` entry point routes to it for compatibility.

For each image, the result folder contains:

- `*_hough_report.jpg`: four-panel report figure showing the ROI, corrected
  image, Canny edges, and colour-coded grouped trajectories.
- `*_hough_result.jpg`: full-frame trajectory overlay with IDs and geometry.
- `*_edges.jpg`: Canny edge map inside the ROI.
- `*_trajectory_features.csv`: one row per grouped trajectory.
- `*_measurements.csv`: one row per retained Hough segment.
- `hough_features.csv`: one image-level summary suitable for tables/charts.
