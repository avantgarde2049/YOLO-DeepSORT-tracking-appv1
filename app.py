import os
import cv2
import numpy as np
import uuid
import threading
import time
from datetime import datetime

# Flask imports
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash
from werkzeug.utils import secure_filename

# ML/CV imports
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a random secret key

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Global variables for tracking processing status
processing_status = {}
processing_threads = {}


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_video(job_id, input_path, output_path):
    """
    Process video with YOLO + DeepSORT tracking
    """
    try:
        processing_status[job_id] = {
            'status': 'processing',
            'progress': 0,
            'message': 'Initializing...',
            'start_time': datetime.now()
        }

        # Initialize YOLOv8 model
        processing_status[job_id]['message'] = 'Loading YOLO model...'
        model = YOLO("yolov8n.pt")

        # Initialize DeepSORT tracker
        processing_status[job_id]['message'] = 'Initializing tracker...'
        tracker = DeepSort(max_age=30)

        # Video capture and writer
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        processing_status[job_id]['message'] = 'Processing frames...'

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLOv8 detection
            results = model(frame, conf=0.5)

            # Extract detections (xyxy format)
            detections = []
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                clss = result.boxes.cls.cpu().numpy()

                for box, conf, cls in zip(boxes, confs, clss):
                    x1, y1, x2, y2 = box
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, int(cls)))

            # Update DeepSORT tracker
            tracks = tracker.update_tracks(detections, frame=frame)

            # Draw bounding boxes and IDs
            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                ltrb = track.to_ltrb()  # left, top, right, bottom

                # Draw bounding box
                x1, y1, x2, y2 = map(int, ltrb)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw track ID
                cv2.putText(
                    frame,
                    f"ID {track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            # Write frame to output
            out.write(frame)

            # Update progress
            frame_count += 1
            progress = int((frame_count / total_frames) * 100)
            processing_status[job_id]['progress'] = progress
            processing_status[job_id]['message'] = f'Processing frame {frame_count}/{total_frames}'

        # Release resources
        cap.release()
        out.release()

        processing_status[job_id] = {
            'status': 'completed',
            'progress': 100,
            'message': 'Processing completed successfully!',
            'end_time': datetime.now(),
            'output_file': os.path.basename(output_path)
        }

    except Exception as e:
        processing_status[job_id] = {
            'status': 'error',
            'progress': 0,
            'message': f'Error: {str(e)}',
            'end_time': datetime.now()
        }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Generate unique filename
        job_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()
        input_filename = f"{job_id}_input.{file_extension}"
        output_filename = f"{job_id}_output.mp4"

        input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        file.save(input_path)

        # Start processing in background thread
        thread = threading.Thread(target=process_video, args=(job_id, input_path, output_path))
        thread.daemon = True
        thread.start()
        processing_threads[job_id] = thread

        return redirect(url_for('status', job_id=job_id))

    flash('Invalid file type. Please upload MP4, AVI, MOV, MKV, or WMV files.')
    return redirect(request.url)


@app.route('/status/<job_id>')
def status(job_id):
    return render_template('status.html', job_id=job_id)


@app.route('/api/status/<job_id>')
def api_status(job_id):
    if job_id in processing_status:
        return jsonify(processing_status[job_id])
    else:
        return jsonify({'status': 'not_found', 'message': 'Job not found'}), 404


@app.route('/download/<job_id>')
def download_file(job_id):
    if job_id in processing_status and processing_status[job_id]['status'] == 'completed':
        output_filename = processing_status[job_id]['output_file']
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        if os.path.exists(output_path):
            return send_file(output_path, as_attachment=True, download_name=f'tracked_{output_filename}')
        else:
            return jsonify({'error': 'File not found'}), 404
    else:
        return jsonify({'error': 'Processing not completed or job not found'}), 404


@app.route('/cleanup')
def cleanup():
    """Clean up old files (optional endpoint for maintenance)"""
    try:
        # Clean up files older than 24 hours
        current_time = time.time()
        for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path):
                    if current_time - os.path.getctime(file_path) > 24 * 3600:  # 24 hours
                        os.remove(file_path)

        return jsonify({'message': 'Cleanup completed'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def create_templates():
    """Create HTML templates if they don't exist"""

    # Index template
    index_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Object Detection & Tracking</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .upload-form { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; }
        .upload-form:hover { border-color: #999; }
        .btn { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        .btn:hover { background: #0056b3; }
        .file-input { margin: 20px 0; }
        .flash-messages { margin: 20px 0; }
        .flash { padding: 10px; margin: 5px 0; border-radius: 4px; }
        .flash-error { background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
    </style>
</head>
<body>
    <h1>Object Detection & Tracking</h1>
    <p>Upload a video file to process with YOLO object detection and DeepSORT tracking.</p>

    {% with messages = get_flashed_messages() %}
        {% if messages %}
            <div class="flash-messages">
                {% for message in messages %}
                    <div class="flash flash-error">{{ message }}</div>
                {% endfor %}
            </div>
        {% endif %}
    {% endwith %}

    <form method="post" action="/upload" enctype="multipart/form-data">
        <div class="upload-form">
            <h3>Select Video File</h3>
            <p>Supported formats: MP4, AVI, MOV, MKV, WMV</p>
            <p>Maximum file size: 100MB</p>
            <div class="file-input">
                <input type="file" name="file" accept=".mp4,.avi,.mov,.mkv,.wmv" required>
            </div>
            <button type="submit" class="btn">Upload & Process</button>
        </div>
    </form>

    <div>
        <h3>How it works:</h3>
        <ol>
            <li>Upload your video file</li>
            <li>YOLO detects objects in each frame</li>
            <li>DeepSORT tracks objects across frames</li>
            <li>Download the processed video with tracking IDs</li>
        </ol>
    </div>
</body>
</html>'''

    # Status template
    status_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Processing Status</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .status-container { text-align: center; margin: 40px 0; }
        .progress-bar { width: 100%; height: 20px; background: #f0f0f0; border-radius: 10px; overflow: hidden; margin: 20px 0; }
        .progress-fill { height: 100%; background: #007bff; transition: width 0.3s ease; }
        .status-message { margin: 20px 0; font-size: 18px; }
        .btn { background: #28a745; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; text-decoration: none; display: inline-block; }
        .btn:hover { background: #218838; }
        .btn:disabled { background: #6c757d; cursor: not-allowed; }
        .error { color: #dc3545; }
        .success { color: #28a745; }
    </style>
</head>
<body>
    <h1>Processing Status</h1>

    <div class="status-container">
        <div id="status-message" class="status-message">Loading...</div>
        <div class="progress-bar">
            <div id="progress-fill" class="progress-fill" style="width: 0%"></div>
        </div>
        <div id="progress-text">0%</div>

        <div id="download-section" style="display: none; margin-top: 30px;">
            <a href="/download/{{ job_id }}" class="btn">Download Processed Video</a>
            <br><br>
            <a href="/">Process Another Video</a>
        </div>
    </div>

    <script>
        const jobId = '{{ job_id }}';

        function updateStatus() {
            fetch(`/api/status/${jobId}`)
                .then(response => response.json())
                .then(data => {
                    const statusMessage = document.getElementById('status-message');
                    const progressFill = document.getElementById('progress-fill');
                    const progressText = document.getElementById('progress-text');
                    const downloadSection = document.getElementById('download-section');

                    if (data.status === 'processing') {
                        statusMessage.textContent = data.message;
                        statusMessage.className = 'status-message';
                        progressFill.style.width = data.progress + '%';
                        progressText.textContent = data.progress + '%';
                    } else if (data.status === 'completed') {
                        statusMessage.textContent = data.message;
                        statusMessage.className = 'status-message success';
                        progressFill.style.width = '100%';
                        progressText.textContent = '100%';
                        downloadSection.style.display = 'block';
                        clearInterval(statusInterval);
                    } else if (data.status === 'error') {
                        statusMessage.textContent = data.message;
                        statusMessage.className = 'status-message error';
                        progressFill.style.width = '0%';
                        progressText.textContent = 'Error';
                        clearInterval(statusInterval);
                    } else if (data.status === 'not_found') {
                        statusMessage.textContent = 'Job not found';
                        statusMessage.className = 'status-message error';
                        clearInterval(statusInterval);
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                });
        }

        // Update status every 2 seconds
        const statusInterval = setInterval(updateStatus, 2000);
        updateStatus(); // Initial call
    </script>
</body>
</html>'''

    # Write templates to files
    with open('templates/index.html', 'w') as f:
        f.write(index_template)

    with open('templates/status.html', 'w') as f:
        f.write(status_template)


if __name__ == '__main__':
    # Create HTML templates before starting the app
    create_templates()
    print("Starting Flask app...")
    print("Access the app at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)