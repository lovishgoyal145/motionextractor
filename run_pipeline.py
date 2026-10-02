#!/usr/bin/env python3
"""
Motion Extraction Pipeline CLI
Extracts 133 whole-body animation-relevant keypoints (body, feet, hands, face)
from human video, normalizes motion around the subject, and outputs
Blender/model-independent MotionAsset (motion.json, metadata.json, preview.mp4).
"""

import os
import sys
import time
import argparse
from typing import Optional
from pathlib import Path
import cv2
from tqdm import tqdm

# Add root directory to python path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.dwpose_detector import DWPoseDetector
from src.normalization import MotionNormalizer
from src.visualizer import MotionVisualizer
from src.motion_asset import build_motion_asset
from src.kaggle_client import KagglePipelineRunner


def run_local(
    input_video: Path,
    output_base: Path,
    models_dir: Path,
    motion_id: Optional[str] = None,
) -> Path:
    """Runs DWPose motion extraction locally."""
    print("==================================================")
    print("      DWPose Motion Extraction (Local Engine)     ")
    print("==================================================")

    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    if motion_id is None:
        video_stem = input_video.stem.replace(" ", "_")
        motion_id = f"motion_{video_stem}_{int(time.time())}"

    out_dir = output_base / motion_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Motion ID: {motion_id}")
    print(f"Input Video: {input_video}")
    print(f"Output Directory: {out_dir}")

    # Initialize components
    detector = DWPoseDetector(models_dir=models_dir, use_gpu=False)
    normalizer = MotionNormalizer()
    visualizer = MotionVisualizer()

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video Specs: {w}x{h} @ {fps:.1f} fps, {total_frames} total frames (~{total_frames/fps:.1f}s)")

    frames_data = []
    annotated_frames = []
    last_bbox = None
    frame_idx = 0
    t_start = time.time()

    with tqdm(total=total_frames, desc="Extracting Motion") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = round(frame_idx / max(1.0, fps), 3)

            # 1. Detect person & extract 133 whole-body keypoints
            res = detector.process_frame(frame, last_bbox=last_bbox)
            bbox = res["bbox"]
            last_bbox = bbox
            kpts = res["keypoints"]
            scores = res["scores"]

            # 2. Subject-centric normalization (pelvis root, torso length scale)
            norm_data = normalizer.normalize_frame(kpts, scores, bbox, (h, w))
            norm_data["frame_idx"] = frame_idx
            norm_data["timestamp"] = timestamp
            norm_data["bbox"] = [round(float(v), 2) for v in bbox]
            frames_data.append(norm_data)

            # 3. Draw skeleton overlay frame for preview video
            annotated = visualizer.draw_skeleton_frame(frame, kpts, scores, bbox, frame_idx, fps)
            annotated_frames.append(annotated)

            frame_idx += 1
            pbar.update(1)

    cap.release()
    elapsed = time.time() - t_start
    print(f"\nPose extraction completed in {elapsed:.2f}s ({frame_idx/elapsed:.1f} fps)")

    # 4. Render preview.mp4
    print("Rendering visualization preview video (preview.mp4)...")
    preview_path = visualizer.render_preview_video(annotated_frames, out_dir / "preview.mp4", fps=fps)

    # 5. Build and save motion.json and metadata.json
    print("Building model-independent MotionAsset (motion.json, metadata.json)...")
    motion_json, meta_json = build_motion_asset(
        motion_id=motion_id,
        video_path=input_video,
        img_shape=(h, w),
        fps=fps,
        frames_data=frames_data,
        output_dir=out_dir,
        processing_time_sec=elapsed,
    )

    print("\nExtraction Successful! Generated Artifacts:")
    print(f"  [MotionAsset] {motion_json} ({motion_json.stat().st_size / 1024:.1f} KB)")
    print(f"  [Metadata]    {meta_json} ({meta_json.stat().st_size / 1024:.1f} KB)")
    print(f"  [Preview MP4] {preview_path} ({preview_path.stat().st_size / 1024:.1f} KB)")

    return out_dir


def run_kaggle(input_video: Path, output_base: Path, motion_id: Optional[str] = None):
    """Executes motion extraction remotely on Kaggle GPU using SwapeDev mechanism."""
    print("==================================================")
    print("      DWPose Motion Extraction (Kaggle Remote)    ")
    print("==================================================")
    runner = KagglePipelineRunner()
    print(f"Authenticated as Kaggle user: {runner.username}")

    configs_dir = REPO_ROOT / "configs"
    meta_path = configs_dir / "kernel-metadata.json"

    # Push kernel
    print(f"Pushing Kaggle kernel targeting {runner.username}/motion-extraction...")
    push_res = runner.push_kernel(configs_dir)
    if push_res.returncode != 0:
        print(f"Kaggle push failed:\n{push_res.stderr}\n{push_res.stdout}")
        sys.exit(1)
    print("Kernel pushed successfully! Waiting for execution...")

    # Poll status
    kernel_slug = f"{runner.username}/motion-extraction"
    for _ in range(60):
        time.sleep(10)
        status = runner.get_status(kernel_slug)
        print(f"Kaggle Kernel Status: {status}")
        if "complete" in status.lower():
            break
        if "error" in status.lower() or "cancel" in status.lower():
            print(f"Kernel terminated with error status: {status}")
            break

    # Download output
    target_out = output_base / (motion_id or f"kaggle_{int(time.time())}")
    print(f"Downloading outputs to {target_out}...")
    runner.download_outputs(kernel_slug, target_out)
    runner.cleanup()
    print("Kaggle run completed.")


def main():
    parser = argparse.ArgumentParser(description="DWPose Human Motion Extraction Pipeline")
    parser.add_argument("--input", type=str, default="input/input_human_3s.mp4", help="Path to input 3-4s human video clip")
    parser.add_argument("--mode", choices=["local", "kaggle"], default="local", help="Execution mode (local or kaggle)")
    parser.add_argument("--models-dir", type=str, default="models/dwpose", help="Path to DWPose ONNX models")
    parser.add_argument("--output-dir", type=str, default="motions", help="Directory where motions/<motion_id> is stored")
    parser.add_argument("--motion-id", type=str, default=None, help="Custom identifier for motion output")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = REPO_ROOT / input_path

    models_path = Path(args.models_dir)
    if not models_path.is_absolute():
        models_path = REPO_ROOT / models_path

    output_base = Path(args.output_dir)
    if not output_base.is_absolute():
        output_base = REPO_ROOT / output_base

    if args.mode == "local":
        run_local(
            input_video=input_path,
            output_base=output_base,
            models_dir=models_path,
            motion_id=args.motion_id,
        )
    else:
        run_kaggle(
            input_video=input_path,
            output_base=output_base,
            motion_id=args.motion_id,
        )


if __name__ == "__main__":
    main()
