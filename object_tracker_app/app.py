
import os
import random
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from collections import defaultdict
from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Load YOLOv8 model
model = YOLO("yolov8n.pt")
logging.info("YOLOv8 model loaded.")

def draw_dashed_rect(image, pt1, pt2, color, thickness=1, dash_length=10):
    """Draw a dashed rectangle on the image"""
    x1, y1 = pt1
    x2, y2 = pt2

    # Draw top line
    draw_dashed_line(image, (x1, y1), (x2, y1), color, thickness, dash_length)
    # Draw bottom line
    draw_dashed_line(image, (x1, y2), (x2, y2), color, thickness, dash_length)
    # Draw left line
    draw_dashed_line(image, (x1, y1), (x1, y2), color, thickness, dash_length)
    # Draw right line
    draw_dashed_line(image, (x2, y1), (x2, y2), color, thickness, dash_length)

def draw_dashed_line(image, pt1, pt2, color, thickness=1, dash_length=10):
    """Draw a dashed line on the image"""
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    dashes = int(dist / dash_length)

    for i in range(dashes):
        start = [pt1[0] + (pt2[0] - pt1[0]) * i / dashes,
                pt1[1] + (pt2[1] - pt1[1]) * i / dashes]
        end = [pt1[0] + (pt2[0] - pt1[0]) * (i + 0.5) / dashes,
              pt1[1] + (pt2[1] - pt1[1]) * (i + 0.5) / dashes]
        cv2.line(image, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])),
                color, thickness)


def track_objects_in_video(video_path, output_dir):
    """
    Processes a video file to track objects using YOLOv8 and saves the output video.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save the output video.

    Returns:
        str: Path to the output video file.
    """
    logging.info(f"Starting video processing for: {video_path}")
    video_name = os.path.basename(video_path)
    video_out_path = os.path.join(output_dir, f"tracked_{video_name}")
    logging.info(f"Output video path will be: {video_out_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error: Could not open video file: {video_path}")
        return None

    ret, frame = cap.read()

    if not ret:
        logging.error(f"Error: Could not read the first frame from the video: {video_path}")
        cap.release()
        return None # Indicate failure

    # Get video properties for VideoWriter
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec

    cap_out = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))

    if not cap_out.isOpened():
        logging.error(f"Error: Could not create video writer for: {video_out_path}")
        cap.release()
        return None

    track_history = defaultdict(list)
    track_info = {}
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(100)]
    occlusion_threshold = 0.6
    disappeared_frames = {}
    max_disappeared = 5
    detection_threshold = 0.5
    frame_count = 0

    try:
        while ret:
            frame_count += 1
            logging.debug(f"Processing frame {frame_count}")
            results = model.track(frame, persist=True, conf=detection_threshold)
            annotator = Annotator(frame)
            current_detections = {}

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                confidences = results[0].boxes.conf.cpu().tolist()

                for box, track_id, confidence in zip(boxes, track_ids, confidences):
                    x, y, w, h = box
                    current_detections[track_id] = (x, y, w, h, confidence)

                for track_id in list(disappeared_frames.keys()):
                    if track_id in current_detections:
                        disappeared_frames[track_id] = 0

                for track_id, (x, y, w, h, confidence) in current_detections.items():
                    color = colors[track_id % len(colors)]
                    occluded = False
                    for other_id, (ox, oy, ow, oh, oconf) in current_detections.items():
                        if track_id == other_id:
                            continue

                        x1, y1 = x - w/2, y - h/2
                        x2, y2 = x + w/2, y + h/2
                        ox1, oy1 = ox - ow/2, oy - oh/2
                        ox2, oy2 = ox + ow/2, oy + oh/2

                        xi1 = max(x1, ox1)
                        yi1 = max(y1, oy1)
                        xi2 = min(x2, ox2)
                        yi2 = min(y2, oy2)
                        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                        box_area = (x2 - x1) * (y2 - y1)
                        other_area = (ox2 - ox1) * (oy2 - oy1)
                        union_area = box_area + other_area - inter_area
                        iou = inter_area / union_area if union_area > 0 else 0

                        if iou > occlusion_threshold and confidence < oconf:
                            occluded = True
                            break

                    if occluded:
                        draw_dashed_rect(annotator.im, (int(x - w/2), int(y - h/2)),
                                       (int(x + w/2), int(y + h/2)), color, thickness=2, dash_length=10)
                        label = f"ID {track_id} (Occluded)"
                    else:
                        annotator.box_label([x - w/2, y - h/2, x + w/2, y + h/2],
                                           f"ID {track_id} {confidence:.2f}", color=color)

                    track = track_history.get(track_id, [])
                    track.append((float(x), float(y)))
                    if len(track) > 30:
                        track.pop(0)
                    track_history[track_id] = track

                    points = [p for p in track]
                    if len(points) > 1:
                        cv2.polylines(frame, [np.array(points, np.int32).reshape((-1, 1, 2))],
                                     False, color, thickness=2)

                    track_info[track_id] = {
                        'last_seen': frame_count,
                        'position': (x, y),
                        'size': (w, h),
                        'occluded': occluded
                    }

            for track_id in list(track_history.keys()):
                if track_id not in current_detections:
                    disappeared_frames[track_id] = disappeared_frames.get(track_id, 0) + 1

                    if track_id in track_info and len(track_history[track_id]) >= 2:
                        last_pos = track_info[track_id]['position']
                        prev_pos = track_history[track_id][-2]
                        dx = last_pos[0] - prev_pos[0]
                        dy = last_pos[1] - prev_pos[1]
                        predicted_x = last_pos[0] + dx * disappeared_frames[track_id]
                        predicted_y = last_pos[1] + dy * disappeared_frames[track_id]
                        w, h = track_info[track_id]['size']
                        color = colors[track_id % len(colors)]
                        cv2.rectangle(frame,
                                     (int(predicted_x - w/2), int(predicted_y - h/2)),
                                     (int(predicted_x + w/2), int(predicted_y + h/2)),
                                     color, 1, cv2.LINE_AA)
                        cv2.putText(frame, f"ID {track_id} (Predicted)",
                                   (int(predicted_x - w/2), int(predicted_y - h/2 - 5)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                        track_history[track_id].append((predicted_x, predicted_y))
                        if len(track_history[track_id]) > 30:
                            track_history[track_id].pop(0)

                    if disappeared_frames[track_id] > max_disappeared:
                        track_history.pop(track_id, None)
                        track_info.pop(track_id, None)
                        disappeared_frames.pop(track_id, None)

            cap_out.write(frame)
            ret, frame = cap.read()
    except Exception as e:
        logging.error(f"Error during video processing: {e}", exc_info=True)
        return None
    finally:
        cap.release()
        cap_out.release()
        cv2.destroyAllWindows()
        logging.info("Video processing finished and resources released.")


    logging.info(f"Finished video processing. Output saved to: {video_out_path}")
    return video_out_path

@app.route('/')
def index():
    logging.info("Index route accessed.")
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    logging.info("Upload route accessed.")
    if 'video' not in request.files:
        logging.warning("No video file in request.")
        return redirect(request.url)
    file = request.files['video']
    if file.filename == '':
        logging.warning("No selected file.")
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logging.info(f"Saving uploaded file to: {upload_path}")
        try:
            file.save(upload_path)
            logging.info("File saved. Starting tracking...")
            output_path = track_objects_in_video(upload_path, app.config['OUTPUT_FOLDER'])
            if output_path:
                logging.info(f"Tracking complete. Redirecting to: {url_for('uploaded_file', filename=os.path.basename(output_path))}")
                return redirect(url_for('uploaded_file', filename=os.path.basename(output_path)))
            else:
                logging.error("Error during video processing.")
                return "Error processing video", 400
        except Exception as e:
            logging.error(f"Error during file upload or tracking: {e}", exc_info=True)
            return "An error occurred during file processing.", 500

@app.route('/outputs/<filename>')
def uploaded_file(filename):
    logging.info(f"Accessing output file: {filename}")
    try:
        return send_from_directory(app.config['OUTPUT_FOLDER'], filename)
    except FileNotFoundError:
        logging.error(f"Output file not found: {filename}")
        return "File not found.", 404
    except Exception as e:
        logging.error(f"Error serving output file: {e}", exc_info=True)
        return "An error occurred while serving the file.", 500


if __name__ == '__main__':
    # This is for running locally. For Colab, you'll need to use ngrok or a similar service
    # to expose the Flask app to the internet.
    # app.run(debug=True)
    pass
