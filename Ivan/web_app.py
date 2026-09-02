import cv2
import numpy as np
import os
import uuid
import json
import time
import threading
from flask import Flask, request, jsonify, send_from_directory, Response, render_template_string

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
    return render_template_string(HTML_TEMPLATE)


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
# HTML TEMPLATE
# =============================================================================

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cloud Chamber Particle Detector</title>
    <style>
        :root {
            --bg:          #0d1117;
            --surface:     #161b22;
            --surface-2:   #1c2129;
            --border:      #30363d;
            --text:        #e6edf3;
            --text-dim:    #8b949e;
            --alpha-color: #f85149;
            --electron-color: #f0c000;
            --accent:      #58a6ff;
            --radius:      12px;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        header {
            width: 100%;
            padding: 24px 32px;
            background: var(--surface);
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            gap: 14px;
        }

        header .logo {
            font-size: 28px;
            line-height: 1;
        }

        header h1 {
            font-size: 20px;
            font-weight: 600;
        }

        header .subtitle {
            color: var(--text-dim);
            font-size: 13px;
            margin-left: auto;
        }

        .container {
            width: 100%;
            max-width: 900px;
            padding: 32px 24px;
            display: flex;
            flex-direction: column;
            gap: 24px;
        }

        /* Upload Zone */
        .upload-zone {
            border: 2px dashed var(--border);
            border-radius: var(--radius);
            padding: 48px 24px;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s ease;
            background: var(--surface);
        }

        .upload-zone:hover,
        .upload-zone.drag-over {
            border-color: var(--accent);
            background: rgba(88, 166, 255, 0.05);
        }

        .upload-zone .icon { font-size: 48px; margin-bottom: 12px; }
        .upload-zone .title { font-size: 16px; font-weight: 500; margin-bottom: 6px; }
        .upload-zone .hint { color: var(--text-dim); font-size: 13px; }

        .upload-zone input[type="file"] { display: none; }

        /* Stats Cards */
        .stats {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }

        .stat-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 20px 24px;
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .stat-card .dot {
            width: 14px;
            height: 14px;
            border-radius: 50%;
            flex-shrink: 0;
        }

        .stat-card .dot.alpha { background: var(--alpha-color); box-shadow: 0 0 10px var(--alpha-color); }
        .stat-card .dot.electron { background: var(--electron-color); box-shadow: 0 0 10px var(--electron-color); }

        .stat-card .stat-label {
            color: var(--text-dim);
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .stat-card .stat-value {
            font-size: 32px;
            font-weight: 700;
        }

        /* Result Area */
        .result-area {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            overflow: hidden;
            display: none;
        }

        .result-area.visible { display: block; }

        .result-area img,
        .result-area video {
            width: 100%;
            display: block;
        }

        /* Progress Bar */
        .progress-container {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 24px;
            display: none;
        }

        .progress-container.visible { display: block; }

        .progress-label {
            font-size: 14px;
            margin-bottom: 12px;
            display: flex;
            justify-content: space-between;
        }

        .progress-label .pct { color: var(--accent); font-weight: 600; }

        .progress-bar-track {
            height: 8px;
            background: var(--surface-2);
            border-radius: 4px;
            overflow: hidden;
        }

        .progress-bar-fill {
            height: 100%;
            width: 0%;
            background: linear-gradient(90deg, var(--accent), #79c0ff);
            border-radius: 4px;
            transition: width 0.4s ease;
        }

        /* Loading Spinner */
        .spinner {
            display: none;
            text-align: center;
            padding: 32px;
        }

        .spinner.visible { display: block; }

        .spinner .ring {
            width: 40px;
            height: 40px;
            border: 3px solid var(--border);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin: 0 auto 12px;
        }

        @keyframes spin { to { transform: rotate(360deg); } }

        /* Detection Table */
        .detections-panel {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            overflow: hidden;
            display: none;
        }

        .detections-panel.visible { display: block; }

        .detections-panel .panel-header {
            padding: 16px 20px;
            font-weight: 600;
            font-size: 14px;
            border-bottom: 1px solid var(--border);
            background: var(--surface-2);
        }

        .det-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }

        .det-table th {
            text-align: left;
            padding: 10px 16px;
            color: var(--text-dim);
            font-weight: 500;
            border-bottom: 1px solid var(--border);
            background: var(--surface-2);
        }

        .det-table td {
            padding: 10px 16px;
            border-bottom: 1px solid var(--border);
        }

        .det-table tr:last-child td { border-bottom: none; }

        .det-table .badge {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }

        .badge.alpha   { background: rgba(248,81,73,0.15); color: var(--alpha-color); }
        .badge.electron { background: rgba(240,192,0,0.15); color: var(--electron-color); }

        .det-table-scroll { max-height: 320px; overflow-y: auto; }

        /* Error toast */
        .error-msg {
            background: rgba(248,81,73,0.12);
            border: 1px solid var(--alpha-color);
            color: var(--alpha-color);
            padding: 12px 20px;
            border-radius: var(--radius);
            display: none;
            font-size: 14px;
        }

        .error-msg.visible { display: block; }
    </style>
</head>
<body>
    <header>
        <div class="logo">&#9883;</div>
        <h1>Cloud Chamber Particle Detector</h1>
        <div class="subtitle">Otsu Segmentation &bull; Bounding Box Geometry</div>
    </header>

    <div class="container">
        <!-- Upload Zone -->
        <div class="upload-zone" id="uploadZone">
            <div class="icon">&#128194;</div>
            <div class="title">Drag &amp; drop an image or video here</div>
            <div class="hint">or click to browse &mdash; supports JPG, PNG, MP4, AVI, MOV</div>
            <input type="file" id="fileInput" accept=".jpg,.jpeg,.png,.bmp,.mp4,.avi,.mov,.mkv">
        </div>

        <!-- Error Message -->
        <div class="error-msg" id="errorMsg"></div>

        <!-- Loading Spinner (for images) -->
        <div class="spinner" id="spinner">
            <div class="ring"></div>
            <div>Processing image&hellip;</div>
        </div>

        <!-- Progress Bar (for videos) -->
        <div class="progress-container" id="progressContainer">
            <div class="progress-label">
                <span id="progressText">Processing video&hellip;</span>
                <span class="pct" id="progressPct">0%</span>
            </div>
            <div class="progress-bar-track">
                <div class="progress-bar-fill" id="progressFill"></div>
            </div>
        </div>

        <!-- Stats Cards -->
        <div class="stats" id="statsArea" style="display:none;">
            <div class="stat-card">
                <div class="dot alpha"></div>
                <div>
                    <div class="stat-label">Alpha Particles</div>
                    <div class="stat-value" id="alphaCount">0</div>
                </div>
            </div>
            <div class="stat-card">
                <div class="dot electron"></div>
                <div>
                    <div class="stat-label">Electron Particles</div>
                    <div class="stat-value" id="electronCount">0</div>
                </div>
            </div>
        </div>

        <!-- Result Display -->
        <div class="result-area" id="resultArea"></div>

        <!-- Detections Table -->
        <div class="detections-panel" id="detectionsPanel">
            <div class="panel-header">Detections</div>
            <div class="det-table-scroll">
                <table class="det-table">
                    <thead>
                        <tr>
                            <th>Type</th>
                            <th>Length</th>
                            <th>Thickness</th>
                            <th>Density</th>
                            <th>Aspect Ratio</th>
                        </tr>
                    </thead>
                    <tbody id="detectionsBody"></tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        const uploadZone = document.getElementById('uploadZone');
        const fileInput  = document.getElementById('fileInput');
        const spinner    = document.getElementById('spinner');
        const progressContainer = document.getElementById('progressContainer');
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        const progressPct  = document.getElementById('progressPct');
        const statsArea    = document.getElementById('statsArea');
        const alphaCount   = document.getElementById('alphaCount');
        const electronCount = document.getElementById('electronCount');
        const resultArea   = document.getElementById('resultArea');
        const detectionsPanel = document.getElementById('detectionsPanel');
        const detectionsBody  = document.getElementById('detectionsBody');
        const errorMsg = document.getElementById('errorMsg');

        // --- Drag & Drop ---
        uploadZone.addEventListener('click', () => fileInput.click());

        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('drag-over');
        });

        uploadZone.addEventListener('dragleave', () => {
            uploadZone.classList.remove('drag-over');
        });

        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('drag-over');
            if (e.dataTransfer.files.length > 0) {
                handleFile(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                handleFile(fileInput.files[0]);
            }
        });

        // --- Reset UI ---
        function resetUI() {
            spinner.classList.remove('visible');
            progressContainer.classList.remove('visible');
            statsArea.style.display = 'none';
            resultArea.classList.remove('visible');
            resultArea.innerHTML = '';
            detectionsPanel.classList.remove('visible');
            detectionsBody.innerHTML = '';
            errorMsg.classList.remove('visible');
        }

        function showError(msg) {
            errorMsg.textContent = msg;
            errorMsg.classList.add('visible');
        }

        // --- Handle file upload ---
        async function handleFile(file) {
            resetUI();

            const ext = file.name.split('.').pop().toLowerCase();
            const isVideo = ['mp4', 'avi', 'mov', 'mkv'].includes(ext);

            if (isVideo) {
                spinner.classList.remove('visible');
                progressContainer.classList.add('visible');
                progressFill.style.width = '0%';
                progressPct.textContent = '0%';
                progressText.textContent = 'Uploading video\u2026';
            } else {
                spinner.classList.add('visible');
            }

            // Upload
            const formData = new FormData();
            formData.append('file', file);

            let resp;
            try {
                resp = await fetch('/upload', { method: 'POST', body: formData });
            } catch (err) {
                resetUI();
                showError('Upload failed: ' + err.message);
                return;
            }

            const data = await resp.json();

            if (data.error) {
                resetUI();
                showError(data.error);
                return;
            }

            if (data.type === 'image') {
                // Image — result is ready immediately
                spinner.classList.remove('visible');
                showImageResult(data);

            } else if (data.type === 'video') {
                // Video — connect to progress stream
                progressText.textContent = 'Processing video\u2026';
                pollProgress(data.task_id);
            }
        }

        // --- Show image result ---
        function showImageResult(data) {
            // Stats
            alphaCount.textContent = data.summary.Alpha;
            electronCount.textContent = data.summary.Electron;
            statsArea.style.display = 'grid';

            // Image
            const img = document.createElement('img');
            img.src = data.result_url;
            resultArea.innerHTML = '';
            resultArea.appendChild(img);
            resultArea.classList.add('visible');

            // Detections table
            if (data.detections && data.detections.length > 0) {
                showDetectionsTable(data.detections);
            }
        }

        // --- Show video result ---
        function showVideoResult(data) {
            // Stats
            alphaCount.textContent = data.summary.Alpha;
            electronCount.textContent = data.summary.Electron;
            statsArea.style.display = 'grid';

            // Video player
            const video = document.createElement('video');
            video.src = data.result_url;
            video.controls = true;
            video.autoplay = true;
            video.loop = true;
            video.style.width = '100%';
            resultArea.innerHTML = '';
            resultArea.appendChild(video);
            resultArea.classList.add('visible');
        }

        // --- Detections table ---
        function showDetectionsTable(detections) {
            detectionsBody.innerHTML = '';
            for (const d of detections) {
                const tr = document.createElement('tr');
                const cls = d.type === 'Alpha' ? 'alpha' : 'electron';
                tr.innerHTML = `
                    <td><span class="badge ${cls}">${d.type}</span></td>
                    <td>${d.length}</td>
                    <td>${d.thickness}</td>
                    <td>${d.density}</td>
                    <td>${d.ar}</td>
                `;
                detectionsBody.appendChild(tr);
            }
            detectionsPanel.classList.add('visible');
        }

        // --- Poll video progress via SSE ---
        function pollProgress(taskId) {
            const evtSource = new EventSource('/progress/' + taskId);

            evtSource.onmessage = (event) => {
                const data = JSON.parse(event.data);

                if (data.error) {
                    evtSource.close();
                    resetUI();
                    showError(data.error);
                    return;
                }

                if (data.status === 'processing') {
                    const total = data.total || 1;
                    const progress = data.progress || 0;
                    const pct = Math.min(100, Math.round((progress / total) * 100));
                    progressFill.style.width = pct + '%';
                    progressPct.textContent = pct + '%';
                    progressText.textContent = `Processing frame ${progress} / ${total}`;
                }

                if (data.status === 'done') {
                    evtSource.close();
                    progressContainer.classList.remove('visible');
                    showVideoResult(data);
                }
            };

            evtSource.onerror = () => {
                evtSource.close();
                resetUI();
                showError('Lost connection to server during processing.');
            };
        }
    </script>
</body>
</html>
"""


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("\n  Cloud Chamber Particle Detector")
    print("  ================================")
    print("  Open  http://localhost:5000  in your browser\n")
    app.run(debug=False, threaded=True, port=5000)
