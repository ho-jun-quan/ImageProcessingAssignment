import cv2
import numpy as np
import os
import uuid
import json
import time
import threading
from flask import Flask, request, jsonify, send_from_directory, Response, render_template

app = Flask(__name__)

# Directories for uploads and processed results
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Background task state for async video processing
tasks = {}


# =============================================================================
# CORE PIPELINE (unchanged — Otsu + bounding box geometry)
# =============================================================================

# Image Calibration (Spatial Scaling)
# Assuming 12.5 pixels corresponds to 1 mm in physical space (adjust based on actual chamber dimensions)
PIXELS_PER_MM = 12.5
CALIBRATED_WIDTH = 880
CALIBRATED_HEIGHT = 1520


def rectify_frame(img, points):
    """Perspective-correct a chamber using points ordered clockwise from top-left."""
    source = np.float32(points)
    destination = np.float32([
        [0, 0],
        [CALIBRATED_WIDTH - 1, 0],
        [CALIBRATED_WIDTH - 1, CALIBRATED_HEIGHT - 1],
        [0, CALIBRATED_HEIGHT - 1]
    ])
    transform = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(img, transform, (CALIBRATED_WIDTH, CALIBRATED_HEIGHT))

def process_frame(img, already_cropped=False, collect_stages=False):
    """Core pipeline: takes a raw frame, returns (annotated_output, binary_mask, detections)."""

    stages = {}

    def record_stage(name, image):
        if collect_stages:
            stages[name] = image.copy()

    # Crop out the physical borders of the chamber (Top:Bottom, Left:Right)
    # Deep crop to exclude bright chamber edge bars at top and bottom
    if not already_cropped:
        img = img[180:1700, 100:980]
    output_img = img.copy()
    record_stage('Cropped input', img)

    # Preprocessing & Noise Reduction
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    record_stage('Greyscale', gray)

    # CLAHE with moderate clip limit to enhance faint particle tracks
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    record_stage('CLAHE enhancement', enhanced)

    # Background subtraction: estimate slow-varying illumination (reflection columns,
    # ambient glow) with a large Gaussian blur, then subtract it out.
    # This automatically removes reflection bands without needing hardcoded coordinates.
    background = cv2.GaussianBlur(enhanced, (51, 51), 0)
    foreground = cv2.subtract(enhanced, background)
    foreground = cv2.normalize(foreground, None, 0, 255, cv2.NORM_MINMAX)
    record_stage('Background subtraction', foreground)

    # Bilateral filter: smooths flat regions (noise) while preserving sharp edges (tracks)
    denoised = cv2.bilateralFilter(foreground, 9, 75, 75)
    record_stage('Bilateral filter', denoised)

    # Light Gaussian blur to further smooth remaining salt-and-pepper noise
    blurred = cv2.GaussianBlur(denoised, (5, 5), 0)
    record_stage('Gaussian smoothing', blurred)

    # Suppress the two persistent internal reflection columns before thresholding.
    # Coordinates are relative to the cropped chamber image.
    cv2.rectangle(blurred, (160, 0), (250, blurred.shape[0]), 0, -1)
    cv2.rectangle(blurred, (660, 0), (750, blurred.shape[0]), 0, -1)
    record_stage('Reflection suppression', blurred)

    # Segmentation (Otsu's Thresholding)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    record_stage('Otsu threshold', binary)

    # Morphological operations — close small gaps then aggressively open to remove noise blobs
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    open_kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel, iterations=2)
    record_stage('Morphological cleanup', binary)

    # Minimal edge blackout (background subtraction handles reflection columns)
    cv2.rectangle(binary, (0, 0), (binary.shape[1], 30), 0, -1)
    cv2.rectangle(binary, (0, binary.shape[0] - 30), (binary.shape[1], binary.shape[0]), 0, -1)
    cv2.rectangle(binary, (0, 0), (20, binary.shape[0]), 0, -1)
    cv2.rectangle(binary, (binary.shape[1] - 20, 0), (binary.shape[1], binary.shape[0]), 0, -1)

    # Remove isolated foreground components before contour extraction so the
    # displayed mask represents candidate tracks rather than background grain.
    component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    cleaned_binary = np.zeros_like(binary)
    for component_index in range(1, component_count):
        if component_stats[component_index, cv2.CC_STAT_AREA] >= 150:
            cleaned_binary[component_labels == component_index] = 255
    binary = cleaned_binary
    record_stage('Final binary mask', binary)

    # Object Detection (Contours)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Feature Extraction, Filtering & Classification
    detections = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        (cx, cy), (w, h), angle = rect
        min_dim = min(w, h)
        max_dim = max(w, h)
        box_area = min_dim * max_dim
        actual_area = cv2.contourArea(contour)

        if box_area == 0:
            continue

        aspect_ratio = max_dim / max(min_dim, 1)
        density = actual_area / box_area

        # Noise filters
        if max_dim < 25:
            continue
        if max_dim < 70 and aspect_ratio < 4:
            continue
        # Reject tiny speckle noise that survived morphological opening
        if actual_area < 150:
            continue

        x, y, bbox_width, bbox_height = cv2.boundingRect(contour)

        # Spatial Calibration: Convert pixel dimensions to millimeters
        length_mm = max_dim / PIXELS_PER_MM
        thickness_mm = min_dim / PIXELS_PER_MM

        # Classification
        if (density > 0.40 and min_dim > 15 and aspect_ratio < 8) or (min_dim > 60 and aspect_ratio < 8):
            particle_type = "Alpha"
            box_color = (0, 0, 255)
        else:
            particle_type = "Electron"
            box_color = (0, 255, 255)

        # Draw on output
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(output_img, [box], 0, box_color, 2)
        label = f"{particle_type} ({length_mm:.1f}mm)"
        cv2.putText(output_img, label, (int(cx), int(cy) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        detections.append({
            'type': particle_type,
            'bbox': {
                'x': int(x),
                'y': int(y),
                'width': int(bbox_width),
                'height': int(bbox_height)
            },
            'length_px': round(max_dim, 1),
            'length_mm': round(length_mm, 2),
            'thickness_px': round(min_dim, 1),
            'thickness_mm': round(thickness_mm, 2),
            'density': round(density, 2),
            'ar': round(aspect_ratio, 1)
        })

    record_stage('Final annotated output', output_img)
    if collect_stages:
        return output_img, binary, detections, stages
    return output_img, binary, detections


# =============================================================================
# VIDEO PROCESSING (background thread)
# =============================================================================

def process_video_task(task_id, filepath, calibration_points=None):
    """Process a video frame-by-frame in a background thread."""
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['error'] = 'Could not open video file.'
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tasks[task_id]['total'] = total

    # Read first frame to get output dimensions after cropping
    ret, first_frame = cap.read()
    if not ret:
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['error'] = 'Could not read first frame.'
        cap.release()
        return

    if calibration_points:
        first_frame = rectify_frame(first_frame, calibration_points)
    first_output, _, first_dets = process_frame(first_frame, already_cropped=bool(calibration_points))
    h_out, w_out = first_output.shape[:2]

    # Setup video writer — try codecs in order of browser compatibility
    result_name = f"{task_id}_result.mp4"
    result_path = os.path.join(RESULTS_FOLDER, result_name)

    writer = None
    for codec in ['avc1', 'mp4v']:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(result_path, fourcc, fps, (w_out, h_out))
        if writer.isOpened():
            break
        writer.release()
        writer = None

    if writer is None:
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['error'] = 'No compatible video codec found. Install FFmpeg for H.264 support.'
        cap.release()
        return

    # Write first frame
    writer.write(first_output)
    summary = {'Alpha': 0, 'Electron': 0}
    for d in first_dets:
        summary[d['type']] += 1

    frame_num = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        if calibration_points:
            frame = rectify_frame(frame, calibration_points)
        output_img, _, detections = process_frame(frame, already_cropped=bool(calibration_points))
        writer.write(output_img)

        for d in detections:
            summary[d['type']] += 1

        # Update progress (throttled — only update every 10 frames)
        if frame_num % 10 == 0 or frame_num == total:
            tasks[task_id]['progress'] = frame_num

    cap.release()
    writer.release()

    tasks[task_id]['status'] = 'done'
    tasks[task_id]['progress'] = total
    tasks[task_id]['result_url'] = f'/results/{result_name}'
    tasks[task_id]['summary'] = summary


# =============================================================================
# FLASK ROUTES
# =============================================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file or file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    task_id = uuid.uuid4().hex[:12]
    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    safe_name = f"{task_id}.{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, safe_name)
    file.save(filepath)

    calibration_points = None
    calibration_json = request.form.get('calibration')
    if calibration_json:
        try:
            calibration_points = json.loads(calibration_json)
            if (not isinstance(calibration_points, list) or len(calibration_points) != 4 or
                    any(not isinstance(point, list) or len(point) != 2 for point in calibration_points)):
                raise ValueError
            calibration_points = [[float(value) for value in point] for point in calibration_points]
        except (TypeError, ValueError, json.JSONDecodeError):
            return jsonify({'error': 'Calibration must contain four valid corner points.'}), 400

    if ext in ('jpg', 'jpeg', 'png', 'bmp'):
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'error': 'Could not read image.'}), 400

        if calibration_points:
            img = rectify_frame(img, calibration_points)
        output_img, binary, detections, stages = process_frame(
            img,
            already_cropped=bool(calibration_points),
            collect_stages=True
        )

        result_name = f"{task_id}_result.jpg"
        cv2.imwrite(os.path.join(RESULTS_FOLDER, result_name), output_img)
        stage_results = []
        for stage_index, (stage_name, stage_image) in enumerate(stages.items(), start=1):
            stage_name_file = f"{task_id}_stage_{stage_index:02d}.jpg"
            cv2.imwrite(os.path.join(RESULTS_FOLDER, stage_name_file), stage_image)
            stage_results.append({
                'name': stage_name,
                'url': f'/results/{stage_name_file}'
            })

        alpha_count = sum(1 for d in detections if d['type'] == 'Alpha')
        electron_count = sum(1 for d in detections if d['type'] == 'Electron')

        return jsonify({
            'type': 'image',
            'result_url': f'/results/{result_name}',
            'detections': detections,
            'stages': stage_results,
            'summary': {'Alpha': alpha_count, 'Electron': electron_count}
        })

    elif ext in ('mp4', 'avi', 'mov', 'mkv'):
        tasks[task_id] = {
            'status': 'processing',
            'progress': 0,
            'total': 0
        }
        thread = threading.Thread(
            target=process_video_task,
            args=(task_id, filepath, calibration_points),
            daemon=True
        )
        thread.start()

        return jsonify({
            'type': 'video',
            'task_id': task_id
        })

    else:
        return jsonify({'error': f'Unsupported file type: .{ext}'}), 400


@app.route('/progress/<task_id>')
def progress(task_id):
    """SSE endpoint — streams video processing progress to the browser."""
    def generate():
        while True:
            task = tasks.get(task_id)
            if not task:
                yield f"data: {json.dumps({'error': 'Task not found'})}\n\n"
                break

            yield f"data: {json.dumps(task)}\n\n"

            if task.get('status') in ('done', 'error'):
                break

            time.sleep(0.5)

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/results/<path:filename>')
def serve_result(filename):
    return send_from_directory(RESULTS_FOLDER, filename)




# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("\n  Cloud Chamber Particle Detector")
    print("  ================================")
    print("  Open  http://localhost:5000  in your browser\n")
    app.run(debug=False, threaded=True, port=5000)

