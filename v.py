import json
import cv2
import numpy as np

def load_matches(path):
    with open(path, 'r') as f:
        return json.load(f)  # Expecting { "broadcast_id": "tacticam_id" }

def load_tracking_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return {frame["frame_id"]: frame["detections"] for frame in data}

def visualize_matches_on_tracked_videos(
    broadcast_video='broadcast_tracked.mp4',
    tacticam_video='tacticam_tracked.mp4',
    broadcast_json='broadcast_tracked_with_ids.json',
    tacticam_json='tacticam_tracked_with_ids.json',
    match_json='player_matches.json',
    output_path='matched_side_by_side1.mp4',
    max_frames=500
):
    print("🔄 Loading data...")
    cap_b = cv2.VideoCapture(broadcast_video)
    cap_t = cv2.VideoCapture(tacticam_video)
    W = int(cap_b.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap_b.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_b.get(cv2.CAP_PROP_FPS)

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (2 * W, H))

    matches = load_matches(match_json)  # {"5": "16", "10": "11", ...}
    b_data = load_tracking_json(broadcast_json)
    t_data = load_tracking_json(tacticam_json)

    frame_id = 0
    print("🎞️ Rendering matched players side-by-side...")
    while True:
        ret_b, frame_b = cap_b.read()
        ret_t, frame_t = cap_t.read()
        if not ret_b or not ret_t or frame_id >= max_frames:
            break

        # Annotate broadcast
        for det in b_data.get(frame_id, []):
            b_id = str(det['track_id'])
            x1, y1, x2, y2 = det['bbox']
            if b_id in matches:
                text = f"B:{b_id}→T:{matches[b_id]}"
                color = (0, 255, 0)
            else:
                text = f"B:{b_id}"
                color = (180, 180, 180)
            cv2.rectangle(frame_b, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_b, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Annotate tacticam
        for det in t_data.get(frame_id, []):
            t_id = str(det['track_id'])
            x1, y1, x2, y2 = det['bbox']
            if t_id in matches.values():
                color = (255, 0, 0)
            else:
                color = (120, 120, 120)
            cv2.rectangle(frame_t, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_t, f"T:{t_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Combine and write
        combined = np.hstack((frame_b, frame_t))
        writer.write(combined)

        if frame_id % 50 == 0:
            print(f"⏳ Frame {frame_id} processed...")

        frame_id += 1

    cap_b.release()
    cap_t.release()
    writer.release()
    print(f"✅ Done! Visualization saved to: {output_path}")


# === Run the visualization ===
if __name__ == "__main__":
    visualize_matches_on_tracked_videos()
