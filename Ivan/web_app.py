import cv2
import numpy as np
import os
import sys
import uuid
import json
import time
import threading
from flask import Flask, request, jsonify, send_from_directory, Response, render_template

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'WeiQuan', 'code'))
from hough_pipeline import analyze_frame

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
# CORE PIPELINE (Canny + Probabilistic Hough Transform)
# =============================================================================

def process_frame(img):
    """Run the shared HoughLinesP trajectory pipeline on an in-memory frame."""

    annotated, edges, summaries, detections = analyze_frame(img)
    return annotated, edges, detections


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
    summary = {'Straight trajectory': 0, 'Curved trajectory': 0, 'Uncertain': 0}
    for d in first_dets:
        summary[d['evidence']] += 1

    frame_num = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        output_img, _, detections = process_frame(frame)
        writer.write(output_img)

        for d in detections:
            summary[d['evidence']] += 1

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

        evidence_summary = {
            'Straight trajectory': sum(1 for d in detections if d['evidence'] == 'Straight trajectory'),
            'Curved trajectory': sum(1 for d in detections if d['evidence'] == 'Curved trajectory'),
            'Uncertain': sum(1 for d in detections if d['evidence'] == 'Uncertain'),
        }

        return jsonify({
            'type': 'image',
            'result_url': f'/results/{result_name}',
            'detections': detections,
            'summary': evidence_summary
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
