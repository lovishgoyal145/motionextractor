#!/usr/bin/env python3
"""
Kaggle Remote Motion Extraction Worker Script
Adapted from SwapeDev Kaggle mechanism for autonomous GPU execution.
Runs DWPose (133 whole-body keypoints: body, feet, hands, face) on input video,
normalizes motion around the subject, and outputs motion.json, metadata.json,
and preview.mp4 to /kaggle/working/motions/<motion_id>/.
"""

import os
import sys
import time
import json
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def download_file(url: str, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        log(f"Model already present: {dest_path}")
        return
    log(f"Downloading {dest_path.name} from {url}...")
    urllib.request.urlretrieve(url, str(dest_path))
    log(f"Downloaded {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.1f} MB)")


def main():
    log("=== Kaggle DWPose Motion Extraction Worker Starting ===")

    # Ensure pip dependencies
    os.system("pip install -q onnxruntime-gpu opencv-python-headless tqdm")

    models_dir = Path("/kaggle/tmp/models/dwpose")
    models_dir.mkdir(parents=True, exist_ok=True)

    yolox_path = models_dir / "yolox_l.onnx"
    dwpose_path = models_dir / "dw-ll_ucoco_384.onnx"

    # Check /kaggle/input first
    input_root = Path("/kaggle/input")
    if input_root.exists():
        for root, _, files in os.walk(str(input_root)):
            for f in files:
                if f == "yolox_l.onnx" and not yolox_path.exists():
                    os.symlink(os.path.join(root, f), str(yolox_path))
                elif f == "dw-ll_ucoco_384.onnx" and not dwpose_path.exists():
                    os.symlink(os.path.join(root, f), str(dwpose_path))

    if not yolox_path.exists():
        download_file("https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx", yolox_path)
    if not dwpose_path.exists():
        download_file("https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx", dwpose_path)

    # Locate input video
    input_video = None
    if input_root.exists():
        for root, _, files in os.walk(str(input_root)):
            for f in files:
                if f.lower().endswith((".mp4", ".mov", ".mkv", ".avi")):
                    input_video = Path(root) / f
                    break
            if input_video:
                break

    if not input_video or not input_video.exists():
        # Fallback sample check
        sample_url = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4"
        input_video = Path("/kaggle/tmp/sample.mp4")
        if not input_video.exists():
            download_file(sample_url, input_video)

    log(f"Input video confirmed: {input_video}")

    # Output motion directory
    motion_id = f"motion_kaggle_{int(time.time())}"
    output_dir = Path(f"/kaggle/working/motions/{motion_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Execute extraction
    # Dynamically import detector & runner
    sys.path.insert(0, "/kaggle/working")
    sys.path.insert(0, str(Path(__file__).parent))

    from src.dwpose_detector import DWPoseDetector
    from src.normalization import MotionNormalizer
    from src.visualizer import MotionVisualizer
    from src.motion_asset import build_motion_asset
    import cv2
    from tqdm import tqdm

    detector = DWPoseDetector(models_dir=models_dir, use_gpu=True)
    normalizer = MotionNormalizer()
    visualizer = MotionVisualizer()

    cap = cv2.VideoCapture(str(input_video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log(f"Processing video: {w}x{h} @ {fps:.1f} fps ({total_frames} frames)")

    frames_data = []
    annotated_frames = []
    last_bbox = None
    frame_idx = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = round(frame_idx / max(1.0, fps), 3)
        res = detector.process_frame(frame, last_bbox=last_bbox)
        bbox = res["bbox"]
        last_bbox = bbox
        kpts = res["keypoints"]
        scores = res["scores"]

        norm_data = normalizer.normalize_frame(kpts, scores, bbox, (h, w))
        norm_data["frame_idx"] = frame_idx
        norm_data["timestamp"] = timestamp
        norm_data["bbox"] = [round(float(v), 2) for v in bbox]
        frames_data.append(norm_data)

        annotated = visualizer.draw_skeleton_frame(frame, kpts, scores, bbox, frame_idx, fps)
        annotated_frames.append(annotated)
        frame_idx += 1

    cap.release()
    elapsed = time.time() - t_start
    log(f"Extracted {frame_idx} frames in {elapsed:.2f}s ({frame_idx/elapsed:.1f} fps)")

    preview_path = visualizer.render_preview_video(annotated_frames, output_dir / "preview.mp4", fps=fps)
    motion_json, meta_json = build_motion_asset(
        motion_id=motion_id,
        video_path=input_video,
        img_shape=(h, w),
        fps=fps,
        frames_data=frames_data,
        output_dir=output_dir,
        processing_time_sec=elapsed,
    )

    log(f"Output files successfully generated in {output_dir}:")
    log(f"  - {motion_json.name} ({motion_json.stat().st_size / 1024:.1f} KB)")
    log(f"  - {meta_json.name} ({meta_json.stat().st_size / 1024:.1f} KB)")
    log(f"  - {preview_path.name} ({preview_path.stat().st_size / 1024:.1f} KB)")
    log("=== Kaggle Motion Extraction Complete ===")


if __name__ == "__main__":
    main()
