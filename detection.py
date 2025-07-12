import cv2
from ultralytics import YOLO
import json
from tqdm import tqdm
from PIL import Image

# Load YOLOv8 model
model = YOLO("best.pt")  # your actual model path

def process_video(video_path, output_json):
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    detections_all = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame (BGR) to RGB and then to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        # Run detection on a single frame (as list with one image)
        results = model.predict(source=[pil_img], conf=0.3, verbose=False)

        boxes = results[0].boxes

        frame_detections = {
            "frame_id": frame_id,
            "detections": []
        }

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                frame_detections["detections"].append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf
                })

        detections_all.append(frame_detections)
        frame_id += 1

    cap.release()

    with open(output_json, "w") as f:
        json.dump(detections_all, f, indent=2)
    print(f"✅ Saved detections to {output_json}")

# Run detection
process_video("broadcast.mp4", "broadcast_detections.json")
process_video("tacticam.mp4", "tacticam_detections.json")
