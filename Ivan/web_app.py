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

def process_frame(img):
    """Core pipeline: takes a raw frame, returns (annotated_output, binary_mask, detections)."""

    # Crop out the physical borders of the chamber (Top:Bottom, Left:Right)
    img = img[100:1820, 100:980]
    output_img = img.copy()

    # Preprocessing & Noise Reduction
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.medianBlur(enhanced, 5)

    # Segmentation (Otsu's Thresholding)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    open_kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel, iterations=2)

    # Black out reflections and ceiling
    cv2.rectangle(binary, (160, 0), (250, 1920), 0, -1)
    cv2.rectangle(binary, (660, 0), (750, 1920), 0, -1)
    cv2.rectangle(binary, (0, 0), (2000, 75), 0, -1)

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
        label = f"{particle_type} (D:{density:.2f} T:{min_dim:.0f})"
        cv2.putText(output_img, label, (int(cx), int(cy) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        detections.append({
            'type': particle_type,
            'length': round(max_dim, 1),
            'thickness': round(min_dim, 1),
            'density': round(density, 2),
            'ar': round(aspect_ratio, 1)
        })

    return output_img, binary, detections


# =============================================================================
# VIDEO PROCESSING (background thread)
# =============================================================================

def process_video_task(task_id, filepath):
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

    first_output, _, first_dets = process_frame(first_frame)
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

        output_img, _, detections = process_frame(frame)
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

    if ext in ('jpg', 'jpeg', 'png', 'bmp'):
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'error': 'Could not read image.'}), 400

        output_img, binary, detections = process_frame(img)

        result_name = f"{task_id}_result.jpg"
        cv2.imwrite(os.path.join(RESULTS_FOLDER, result_name), output_img)

        alpha_count = sum(1 for d in detections if d['type'] == 'Alpha')
        electron_count = sum(1 for d in detections if d['type'] == 'Electron')

        return jsonify({
            'type': 'image',
            'result_url': f'/results/{result_name}',
            'detections': detections,
            'summary': {'Alpha': alpha_count, 'Electron': electron_count}
        })

    elif ext in ('mp4', 'avi', 'mov', 'mkv'):
        tasks[task_id] = {
            'status': 'processing',
            'progress': 0,
            'total': 0
        }
        thread = threading.Thread(target=process_video_task, args=(task_id, filepath), daemon=True)
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

