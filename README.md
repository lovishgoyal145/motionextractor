# MotionExtractor

Production-grade whole-body (133 keypoints) human motion extraction pipeline from video for Blender, animation, and 2D/3D character rigs. Powered by DWPose (YOLOX-L + DWPose ONNX) with deterministic multi-person tracking and subject-centric coordinate normalization.

---

## ⚡ Global CLI Commands

Run directly from any directory:

### 1. Single-Character Motion Extraction (`moex1`)
Extracts motion for the primary character in the video, filtering out background pedestrians/passersby:

```bash
moex1 /path/to/video.mp4
```

### 2. Multi-Character Motion Extraction (`moex2`)
Detects and tracks **all characters** simultaneously throughout the video with persistent IDs (`person_001`, `person_002`, ...):

```bash
moex2 /path/to/video.mp4
```

### Standard Python Invocation
```bash
python run.py MOVE /path/to/video.mp4
```
*(Where `MOVE` is any 4-letter command, e.g. `MOVE`, `POSE`, `EXTR`)*

---

## 📦 Model Weights (Published via GitHub Releases)

The official DWPose ONNX weights (~335 MB total) are published and hosted directly on the repository's [GitHub Release v1.0.0](https://github.com/lovishgoyal145/motionextractor/releases/tag/v1.0.0):

| Model | Size | Purpose | Direct Download Link |
| :--- | :--- | :--- | :--- |
| **`dw-ll_ucoco_384.onnx`** | 128 MB | Whole-Body 133 Keypoint Estimator | [Download](https://github.com/lovishgoyal145/motionextractor/releases/download/v1.0.0/dw-ll_ucoco_384.onnx) |
| **`yolox_l.onnx`** | 206 MB | Multi-Person Detector | [Download](https://github.com/lovishgoyal145/motionextractor/releases/download/v1.0.0/yolox_l.onnx) |

### Automatic Download
When setting up on any new device or machine, you do **not** need to manually find or place weights:
1. `DWPoseDetector` automatically downloads the models on first execution if missing.
2. Or you can pre-download them anytime via:
   ```bash
   python scripts/download_models.py
   ```

---

## 📁 Output Structure

Every run creates a non-colliding sequential directory under `/home/lovish/motion-data/` (`0001/`, `0002/`, `0003/`, ...):

```text
motion-data/
└── 0001/
    ├── motion.json           # Global motion asset index & timeline index
    ├── metadata.json         # Complete telemetry, SHA-256 hash, FPS, dependency versions
    ├── preview.mp4           # Visual preview with color-coded persistent IDs & skeletons
    ├── persons/              # Modular per-person motion data files:
    │   ├── person_001.json
    │   ├── person_002.json
    │   └── ...
    ├── README.txt            # Human-readable documentation & joint mapping
    └── source/               # [Optional: only if --archive-source is passed]
        └── original.mp4
```

---

## 🎨 Keypoint & Coordinate Convention (Blender Ready)

Each person file (`persons/person_XXX.json`) contains:
- **`keypoints_normalized`**: $[x, y, score]$ normalized around the mid-hip pelvis root, scaled by torso length (mid-hip to mid-shoulder). **$Y$ is UP, $X$ is RIGHT**.
- **`keypoints_pixel`**: $[x, y, score]$ original pixel coordinates in the video frame.
- **`root_trajectory_norm`**: $[dx, dy]$ pelvis trajectory displacement relative to person's frame 0.
- **Pre-grouped Joint Slices**:
  - `body_keypoints`: 17 COCO joints (nose, eyes, shoulders, elbows, wrists, hips, knees, ankles)
  - `feet_keypoints`: 6 feet keypoints (big toe, small toe, heel)
  - `face_landmarks`: 68 facial landmark coordinates
  - `left_hand_keypoints`: 21 finger joints
  - `right_hand_keypoints`: 21 finger joints

---

## 🚀 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/lovishgoyal145/motionextractor.git
   cd motionextractor
   ```

2. **Create Python environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install onnxruntime opencv-python scipy numpy tqdm
   ```

3. **Download Model Weights**:
   ```bash
   python scripts/download_models.py
   ```

4. **Install Global CLI commands** (Optional):
   ```bash
   cp scripts/moex1 ~/.local/bin/moex1
   cp scripts/moex2 ~/.local/bin/moex2
   chmod +x ~/.local/bin/moex1 ~/.local/bin/moex2
   ```

---

## 🧪 Testing

Run the automated test suite:
```bash
python -m unittest tests/test_pipeline.py -v
```
