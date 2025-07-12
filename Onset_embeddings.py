import os
import json
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torchreid
from collections import defaultdict

def extract_osnet_embeddings(video_path, tracked_json_path, output_feature_path):
    print("🔄 Initializing...")

    # === Load video ===
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # === Load tracked detections ===
    with open(tracked_json_path, 'r') as f:
        data = json.load(f)

    # === Init OSNet Model ===
    model = torchreid.models.build_model(
        name='osnet_ain_x1_0',
        num_classes=1000,
        pretrained=True
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    


    # === Preprocessing (ImageNet stats) ===
    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # === Track ID → List of embeddings (limit 3 crops per ID) ===
    track_features = defaultdict(list)

    frame_dict = {item["frame_id"]: item["detections"] for item in data}
    frame_id = 0

    print("🚀 Processing video frame by frame...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % 10 == 0:
            print(f"🔎 Processing frame {frame_id}/{total_frames}")

        detections = frame_dict.get(frame_id, [])
        for det in detections:
            track_id = str(det["track_id"])

            # Limit to 3 crops per track to save time
            if len(track_features[track_id]) >= 3:
                continue

            x1, y1, x2, y2 = det["bbox"]
            crop = frame[y1:y2, x1:x2]

            # Skip invalid crops
            if crop.size == 0:
                continue

            crop = cv2.resize(crop, (128, 256))
            img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                feat = model(img_tensor)
                feat = feat / feat.norm(p=2)

            track_features[track_id].append(feat.squeeze().cpu().numpy())

        frame_id += 1

    cap.release()

    print("📦 Averaging features per track...")
    averaged_features = {}
    for tid, feats in track_features.items():
        feats_stack = np.stack(feats, axis=0)
        avg_feat = np.mean(feats_stack, axis=0)
        averaged_features[tid] = avg_feat.tolist()

    with open(output_feature_path, 'w') as f:
        json.dump(averaged_features, f, indent=2)

    print(f"✅ Done! OSNet embeddings saved to: {output_feature_path}")


# === Example Run ===
extract_osnet_embeddings(
    video_path="broadcast.mp4",
    tracked_json_path="broadcast_tracked_with_ids.json",
    output_feature_path="broadcast_embeddings1.json"
)
