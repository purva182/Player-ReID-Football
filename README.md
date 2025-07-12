
# ⚽ Player Re-Identification Across Cameras using DeepSORT + OSNet

This project aims to identify and track football players across two different video perspectives: a **broadcast view** and a **tacticam view**. It combines **multi-object tracking (MOT)** with **Re-Identification (Re-ID)** using visual, spatial, and temporal cues.

## 🚀 Overview

- **Task**: Match players across two unsynchronized camera feeds.
- **Approach**:
  - Detect players in both videos using a pre-trained object detector (`best.pt` from LIAT.ai).
  - Track players over time using **DeepSORT**.
  - Extract deep appearance features using **OSNet**.
  - Match players between cameras using **cosine similarity** on averaged embeddings.
  - Visualize matches in a side-by-side video.

## 🧠 Features Used

| Feature Type | Description                          | Usage                          |
|--------------|--------------------------------------|--------------------------------|
| 🎨 Visual    | Player appearance (jersey, color)    | Extracted via OSNet embeddings |
| 📍 Spatial   | Bounding box positions               | Used by DeepSORT for tracking  |
| 🕒 Temporal  | Track consistency over time          | Maintained by DeepSORT         |

## 🛠️ Dependencies

- Python 3.8+
- OpenCV
- PyTorch
- torchvision
- numpy
- PIL
- [torchreid](https://github.com/KaiyangZhou/deep-person-reid)
- gdown

Install dependencies:

```bash
pip install opencv-python torch torchvision numpy pillow gdown
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
 [nwojke/deep_sort](https://github.com/nwojke/deep_sort) (original DeepSORT algorithm)
```

## 📂 Folder Structure

```
├── broadcast.mp4
├── tacticam.mp4
├── broadcast_tracked_with_ids.json
├── tacticam_tracked_with_ids.json
├── broadcast_embeddings1.json
├── tacticam_embeddings1.json
├── player_matches.json
├── matched_side_by_side1.mp4
├── run_deepsort.py
├── osnet_embeddings.py
├── match_players.py
├── visualize_matches.py
└── README.md
```

## 🧪 How to Run

### 1. Run DeepSORT Tracking on Both Videos

```bash
python run_deepsort.py
```

### 2. Extract OSNet Embeddings

```bash
python osnet_embeddings.py
```

### 3. Match Players Using Cosine Similarity

```bash
python match_players.py
```

### 4. Visualize Matches

```bash
python visualize_matches.py
```

## 📈 Results

- ✅ Tracked players across both camera views
- ✅ Generated ID-based mapping file: `player_matches.json`
- ✅ Created side-by-side visualization video: `matched_side_by_side1.mp4`
- ✅ Used DeepSORT + OSNet pipeline effectively

## ⚠️ Limitations

- ❌ No ground truth available to compute numeric accuracy.
- 🚫 Embeddings are not fine-tuned — pre-trained OSNet used as-is.
- 🎥 Videos are not perfectly synchronized; minor mismatches may occur.
- 🧪 Visual similarity does not always guarantee true identity match.

## ✅ Evaluation Strategy (Without Ground Truth)

- **Cosine similarity** between player embeddings used as matching score.
- **Visual validation** via annotated side-by-side video.
- **Manual spot-checking** for consistency of IDs over time.

## 📌 Future Work

- Fine-tune OSNet on domain-specific football data.
- Incorporate pose/keypoint-based features for better disambiguation.
- Use a frame-synchronization module to improve match alignment.
- Add mAP/CMC-style evaluation if annotations become available.

## 🧑‍💻 Author

This project was completed as part of an internship assignment.  
If shortlisted, the system can be extended further into a real-time, multi-camera tracking system.
