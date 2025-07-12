import cv2
import json
import numpy as np
import os

from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.deep_sort.detection import Detection
from deep_sort.tools import generate_detections as gdet


def run_deepsort(video_path, json_path, output_path,
                 model_filename=r'C:\Users\HP\Documents\intern\resources\networks\mars-small128.pb'):

    # Load detections from JSON
    with open(json_path, 'r') as f:
        detections_data = json.load(f)

    # Create Deep SORT encoder (feature extractor)
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    # Initialize Tracker
    max_cosine_distance = 0.4
    nn_budget = None
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker_instance = Tracker(metric)  # Avoid overwriting class name

    # Load video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    # Prepare detection lookup by frame_id
    detection_map = {d["frame_id"]: d["detections"] for d in detections_data}
    all_frame_tracks = []

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detection_map.get(frame_id, [])
        bboxes = []
        confidences = []

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            bboxes.append([x1, y1, x2 - x1, y2 - y1])  # x, y, w, h
            confidences.append(det['confidence'])

        features = encoder(frame, bboxes)
        det_objects = [
            Detection(bbox, conf, feature)
            for bbox, conf, feature in zip(bboxes, confidences, features)
        ]

        # Update tracker
        tracker_instance.predict()
        tracker_instance.update(det_objects)

        # Save track IDs for this frame
        frame_tracks = []
        for track in tracker_instance.tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            x, y, w, h = track.to_tlwh()
            track_id = track.track_id
            frame_tracks.append({
                "track_id": int(track_id),
                "bbox": [int(x), int(y), int(x + w), int(y + h)]
            })

            # Draw track on video
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            cv2.putText(frame, f'ID {track_id}', (int(x), int(y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        all_frame_tracks.append({
            "frame_id": frame_id,
            "detections": frame_tracks
        })

        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()
    print(f"✅ Tracking complete! Video saved to: {output_path}")

    # Save detections with track_ids to JSON
    json_output = os.path.splitext(output_path)[0] + "_with_ids.json"
    with open(json_output, "w") as f:
        json.dump(all_frame_tracks, f, indent=2)
    print(f"✅ JSON with track IDs saved to: {json_output}")


# === Run for tacticam.mp4 ===
run_deepsort(
    video_path='broadcast.mp4',
    json_path='broadcast_detections.json',
    output_path='broadcast_tracked.mp4'
)
