
# ⚽ Player Re-Identification Across Cameras using DeepSORT + OSNet

This project identifies and re-identifies football players across two different video perspectives: a **broadcast view** and a **tacticam view**. It combines **multi-object tracking (MOT)** with **Re-Identification (Re-ID)** using visual, spatial, and temporal features.

---

## 🚀 Project Pipeline

- 🎯 **Task**: Match players across unsynchronized multi-camera football video feeds.
- 🔍 **Steps**:
  - Detect players in both videos using a pre-trained YOLOv5 model (`best.pt` from LIAT.ai).
  - Track players over time using **DeepSORT**.
  - Extract appearance features using **OSNet**.
  - Compute **cosine similarity** between embeddings to match players across views.
  - Visualize player matches in a **side-by-side annotated video**.

---

## 🎒 Features Used

| Type      | Description                          | Implementation                   |
|-----------|--------------------------------------|----------------------------------|
| 🎨 Visual  | Player appearance (jersey, etc.)    | Extracted via OSNet embeddings   |
| 📍 Spatial | Player positions (bounding boxes)   | From DeepSORT tracker            |
| 🕒 Temporal| Movement over time                   | Used for maintaining consistency |

---

## 🔗 Model Download

⚠️ `best.pt` is not included in this repo due to GitHub's 100 MB limit.

📥 **Download `best.pt`** from:  
👉 [Google Drive Link  🤗 (https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view)]

Place it in the root directory before running `run_deepsort.py`.

---

## 🛠️ Installation

### 🔧 Dependencies

```bash
pip install opencv-python torch torchvision numpy pillow gdown
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
```

Also make sure to include DeepSORT:

```bash
git submodule update --init --recursive
```

---

## 📁 Folder Structure

```
├── deep_sort/                         # Submodule: DeepSORT
├── model_info/                        # OSNet / model metadata
├── resources/                         # Configs, optional data
├── broadcast.mp4                      # Broadcast-view video
├── tacticam.mp4                       # Tacticam-view video
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
├── .gitignore
└── README.md
```

---

## 🧪 How to Run

1. **Track Players (DeepSORT)**  
   ```bash
   python run_deepsort.py
   ```

2. **Extract OSNet Embeddings**  
   ```bash
   python osnet_embeddings.py
   ```

3. **Match Players Across Views**  
   ```bash
   python match_players.py
   ```

4. **Visualize Matched Players**  
   ```bash
   python visualize_matches.py
   ```

---

## ✅ Output Highlights

- 👥 Player IDs tracked in both views
- 📊 `player_matches.json` generated
- 🎞️ `matched_side_by_side1.mp4` shows matched IDs side-by-side
- 🔁 Matching based on cosine similarity of appearance embeddings

---

## ⚠️ Known Limitations

- No ground truth → evaluation is manual/visual.
- Videos not perfectly synchronized.
- Uses pre-trained OSNet — not fine-tuned on football data.
- Limited robustness to occlusion/similar jersey colors.

---

## 📈 Evaluation Without Ground Truth

- Matching based on **cosine similarity threshold**
- **Visual inspection** via the final video
- Manual checking for **consistency over time**

---

## 🧩 Future Work

- Fine-tune ReID model on domain-specific datasets.
- Use pose/keypoints for fine-grained features.
- Introduce temporal synchronization techniques.
- Add formal evaluation metrics (e.g., CMC, mAP) if annotations become available.

---

## 🙋‍♀️ About

This project was submitted as part of an internship assignment.  

---
