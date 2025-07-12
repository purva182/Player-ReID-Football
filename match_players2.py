import json
import numpy as np
from scipy.spatial.distance import cosine
from scipy.optimize import linear_sum_assignment


def load_embeddings(path):
    with open(path, 'r') as f:
        return json.load(f)


def load_tracks(path):
    with open(path, 'r') as f:
        data = json.load(f)

    track_dict = {}
    for frame in data:
        frame_id = frame["frame_id"]
        for det in frame["detections"]:
            tid = str(det["track_id"])
            x1, y1, x2, y2 = det["bbox"]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            track_dict.setdefault(tid, []).append((frame_id, cx, cy))
    return track_dict


def spatial_similarity(track1, track2):
    coords1 = np.array([(f, x, y) for f, x, y in track1])
    coords2 = np.array([(f, x, y) for f, x, y in track2])

    # Find overlapping frames
    frames1 = set(coords1[:, 0])
    frames2 = set(coords2[:, 0])
    overlap = frames1.intersection(frames2)
    if not overlap:
        return 0  # No spatial similarity if no time overlap

    dists = []
    for f in overlap:
        p1 = coords1[coords1[:, 0] == f][0][1:]
        p2 = coords2[coords2[:, 0] == f][0][1:]
        d = np.linalg.norm(p1.astype(float) - p2.astype(float))
        dists.append(d)

    avg_dist = np.mean(dists)
    return 1 / (1 + avg_dist)  # Normalize: smaller distance → higher score


def match_players_fused(bcast_emb, tact_emb, bcast_tracks, tact_tracks, alpha=0.6, beta=0.3):
    ids1 = list(bcast_emb.keys())
    ids2 = list(tact_emb.keys())
    cost_matrix = np.zeros((len(ids1), len(ids2)))

    for i, id1 in enumerate(ids1):
        for j, id2 in enumerate(ids2):
            vec1 = np.array(bcast_emb[id1])
            vec2 = np.array(tact_emb[id2])

            # Appearance similarity (cosine)
            appearance_sim = 1 - cosine(vec1, vec2)

            # Spatial similarity
            spa_sim = spatial_similarity(bcast_tracks[id1], tact_tracks[id2])

            # Temporal overlap
            frames1 = set(f for f, _, _ in bcast_tracks[id1])
            frames2 = set(f for f, _, _ in tact_tracks[id2])
            temp_overlap = len(frames1 & frames2) / max(1, len(frames1 | frames2))  # [0, 1]

            # Final fused score
            fused_score = alpha * appearance_sim + beta * spa_sim + (1 - alpha - beta) * temp_overlap
            cost_matrix[i, j] = 1 - fused_score  # for Hungarian algorithm

    row_idx, col_idx = linear_sum_assignment(cost_matrix)
    matches = {ids1[i]: ids2[j] for i, j in zip(row_idx, col_idx)}
    return matches


# === Run Matching ===
broadcast_emb = load_embeddings("broadcast_embeddings1.json")
tacticam_emb = load_embeddings("tacticam_embeddings1.json")
broadcast_tracks = load_tracks("broadcast_tracked_with_ids.json")
tacticam_tracks = load_tracks("tacticam_tracked_with_ids.json")

matches = match_players_fused(broadcast_emb, tacticam_emb, broadcast_tracks, tacticam_tracks)

# Save matches
with open("player_matches.json", "w") as f:
    json.dump(matches, f, indent=2)

print("✅ Matching complete. Results saved to player_matches.json")
