import os
import pickle

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Paths
video_path = os.path.join('.', 'data', 'video.mp4')
output_path = os.path.join('.', 'output_deepsort.mp4')

# Initialize YOLOv8 model
model = YOLO("yolov8n.pt")  # or yolov8s.pt, yolov8m.pt, etc.

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)  # Adjust max_age as needed

# Video capture and writer
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError("Could not open video file")

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection
    results = model(frame, conf=0.5)  # Adjust confidence threshold

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

    # Display (optional)
    cv2.imshow("DeepSORT Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

pickle.dump(model,open("model.pkl","wb"))
