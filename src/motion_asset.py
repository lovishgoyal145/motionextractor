"""
Multi-Person MotionAsset Generator
Produces Blender-ready, modular motion data:
- motion.json (global index & timeline)
- metadata.json (session info, SHA-256 hash, tracking stats, system info)
- persons/person_XXX.json (independent per-person motion data)
- README.txt (human-readable instructions and keypoint map)
- source/original.mp4 (optional archival copy)
"""

import os
import sys
import json
import shutil
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2
import onnxruntime as ort
from .dwpose_detector import KEYPOINT_NAMES


def compute_file_sha256(file_path: Path) -> str:
    """Computes SHA-256 hash of file in 64KB chunks to maintain O(1) memory."""
    h = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        return f"error_computing_hash: {e}"


def build_multiperson_motion_asset(
    session_id: str,
    command: str,
    video_path: Path,
    img_shape: Tuple[int, int],  # (height, width)
    fps: float,
    total_video_frames: int,
    tracks: List[Any],  # List of PersonTrack
    output_dir: Path,
    processing_time_sec: float = 0.0,
    archive_source: bool = False,
    status: str = "completed",
) -> Dict[str, Path]:
    """
    Constructs and writes all output artifacts to output_dir.
    Returns dictionary of generated file paths.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = Path(video_path).resolve()
    h, w = img_shape
    duration = total_video_frames / max(1.0, fps)

    # 1. Handle optional source archival
    source_copy_path = None
    if archive_source and video_path.exists():
        source_dir = output_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        source_copy_path = source_dir / f"original{video_path.suffix}"
        if video_path != source_copy_path:
            shutil.copy2(video_path, source_copy_path)

    # 2. Compute SHA-256 hash of original video
    sha256_hash = compute_file_sha256(video_path) if video_path.exists() else "not_available"
    file_size_bytes = video_path.stat().st_size if video_path.exists() else 0

    # 3. Create persons/ directory and write each person's file
    persons_dir = output_dir / "persons"
    persons_dir.mkdir(parents=True, exist_ok=True)

    persons_index = []
    per_person_summary = {}
    all_subject_confs = []
    all_keypoint_confs = []
    total_uncertain_frames = 0
    timeline_index: Dict[int, List[str]] = {}

    for track in tracks:
        p_id = track.person_id
        t_id = track.track_id
        frames = track.frames_data

        if not frames:
            continue

        start_f = track.first_frame
        end_f = track.last_frame
        num_frames = len(frames)
        coverage_pct = round((num_frames / max(1, total_video_frames)) * 100.0, 1)

        # Compute per-person confidences
        p_bbox_confs = [f["bbox"][4] for f in frames if len(f.get("bbox", [])) > 4]
        p_kpt_confs = [
            pt[2]
            for f in frames
            for pt in f.get("keypoints_normalized", [])
            if len(pt) > 2
        ]
        p_uncertain_cnt = sum(1 for f in frames if f.get("is_uncertain", False))
        total_uncertain_frames += p_uncertain_cnt

        mean_bbox_conf = round(float(np.mean(p_bbox_confs)), 3) if p_bbox_confs else 0.0
        mean_kpt_conf = round(float(np.mean(p_kpt_confs)), 3) if p_kpt_confs else 0.0

        all_subject_confs.extend(p_bbox_confs)
        all_keypoint_confs.extend(p_kpt_confs)

        person_doc = {
            "$schema_version": "2.0.0",
            "asset_type": "PersonMotionAsset",
            "session_id": session_id,
            "person_id": p_id,
            "track_id": t_id,
            "timing": {
                "fps": round(float(fps), 2),
                "start_frame": start_f,
                "end_frame": end_f,
                "frames_tracked": num_frames,
                "total_video_frames": total_video_frames,
                "coverage_percentage": coverage_pct,
            },
            "normalization_spec": {
                "root_joint": "mid_hip (pelvis)",
                "root_indices": [11, 12],
                "scale_unit": "torso_length (mid_hip to mid_shoulder)",
                "coordinate_system": "root_relative_2d (X right positive, Y up positive)",
                "reference_frame_0_root": [round(float(v), 2) for v in track.normalizer.ref_root_0] if track.normalizer.ref_root_0 is not None else None,
                "reference_torso_length_px": round(float(track.normalizer.ref_torso_len_0), 2) if track.normalizer.ref_torso_len_0 is not None else None,
            },
            "keypoint_groups": {
                "body": {"range": [0, 16], "count": 17},
                "feet": {"range": [17, 22], "count": 6},
                "face": {"range": [23, 90], "count": 68},
                "left_hand": {"range": [91, 111], "count": 21},
                "right_hand": {"range": [112, 132], "count": 21},
            },
            "keypoint_names": KEYPOINT_NAMES,
            "frames": frames,
        }

        person_file = persons_dir / f"{p_id}.json"
        with open(person_file, "w", encoding="utf-8") as pf:
            json.dump(person_doc, pf, indent=2)

        # Update timeline index
        for f in frames:
            f_idx = f["frame_idx"]
            if f_idx not in timeline_index:
                timeline_index[f_idx] = []
            timeline_index[f_idx].append(p_id)

        # Record summary
        rel_person_file = f"persons/{p_id}.json"
        persons_index.append({
            "person_id": p_id,
            "track_id": t_id,
            "file": rel_person_file,
            "frame_range": [start_f, end_f],
            "frames_tracked": num_frames,
            "coverage_percentage": coverage_pct,
            "mean_confidence": mean_kpt_conf,
            "is_uncertain_count": p_uncertain_cnt,
        })

        per_person_summary[p_id] = {
            "track_id": t_id,
            "first_frame": start_f,
            "last_frame": end_f,
            "frames_tracked": num_frames,
            "coverage_percentage": coverage_pct,
            "mean_subject_confidence": mean_bbox_conf,
            "mean_keypoint_confidence": mean_kpt_conf,
            "uncertain_frames_count": p_uncertain_cnt,
            "file": rel_person_file,
        }

    # 4. Build global motion.json
    motion_doc = {
        "$schema_version": "2.0.0",
        "asset_type": "MultiPersonMotionAsset",
        "session_id": session_id,
        "timing": {
            "fps": round(float(fps), 2),
            "duration_seconds": round(float(duration), 3),
            "total_frames": total_video_frames,
        },
        "video_space": {
            "width": int(w),
            "height": int(h),
            "source_filename": video_path.name,
            "source_path": str(video_path),
            "sha256": sha256_hash,
        },
        "normalization_spec": {
            "root_joint": "mid_hip (pelvis)",
            "root_indices": [11, 12],
            "scale_unit": "torso_length (mid_hip to mid_shoulder)",
            "coordinate_system": "root_relative_2d (X right positive, Y up positive)",
            "note": "Per-person normalized coordinates with frame-0 reference anchor.",
        },
        "keypoint_groups": {
            "body": {"range": [0, 16], "count": 17},
            "feet": {"range": [17, 22], "count": 6},
            "face": {"range": [23, 90], "count": 68},
            "left_hand": {"range": [91, 111], "count": 21},
            "right_hand": {"range": [112, 132], "count": 21},
        },
        "keypoint_names": KEYPOINT_NAMES,
        "tracked_persons_count": len(persons_index),
        "persons": persons_index,
        "timeline_index": {str(k): v for k, v in sorted(timeline_index.items())},
    }

    motion_file = output_dir / "motion.json"
    with open(motion_file, "w", encoding="utf-8") as mf:
        json.dump(motion_doc, mf, indent=2)

    # 5. Build metadata.json
    overall_mean_subj = round(float(np.mean(all_subject_confs)), 3) if all_subject_confs else 0.0
    overall_mean_kpt = round(float(np.mean(all_keypoint_confs)), 3) if all_keypoint_confs else 0.0

    metadata_doc = {
        "session_id": session_id,
        "command": command,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "source_video": {
            "filename": video_path.name,
            "original_path": str(video_path),
            "sha256": sha256_hash,
            "file_size_bytes": file_size_bytes,
            "resolution": [int(w), int(h)],
            "fps": round(float(fps), 2),
            "duration_seconds": round(float(duration), 3),
            "frame_count": total_video_frames,
        },
        "extraction_engine": {
            "model_family": "DWPose (Whole-body 133 Keypoints)",
            "detector": "yolox_l.onnx",
            "pose_estimator": "dw-ll_ucoco_384.onnx",
            "tracker": "Deterministic IoU + Spatial Hungarian Tracker",
            "runtime": "ONNXRuntime",
            "tracked_keypoints_total": 133,
        },
        "tracking_statistics": {
            "unique_persons_detected": len(persons_index),
            "overall_mean_subject_confidence": overall_mean_subj,
            "overall_mean_keypoint_confidence": overall_mean_kpt,
            "uncertain_frames_total": total_uncertain_frames,
            "per_person_summary": per_person_summary,
        },
        "performance": {
            "processing_time_seconds": round(float(processing_time_sec), 2),
            "processing_fps": round(total_video_frames / max(0.001, processing_time_sec), 2),
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "onnxruntime_version": ort.__version__,
            "opencv_version": cv2.__version__,
            "numpy_version": np.__version__,
            "device": "cpu",
        },
        "compatible_consumers": [
            "Blender Armature & Shape Key Retargeting",
            "2D/2.5D Cutout Character Riggers",
            "ControlNet OpenPose / DWPose Video Models",
            "AnimateAnyone / Character-Swap Conditioning",
        ],
    }

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as metaf:
        json.dump(metadata_doc, metaf, indent=2)

    # 6. Build README.txt
    readme_content = f"""================================================================================
DWPose Multi-Person Motion Extraction Asset
Session: {session_id} | Command: {command} | Status: {status}
================================================================================

1. Source Video Information:
   - Filename:     {video_path.name}
   - Original Path:{video_path}
   - SHA-256:      {sha256_hash}
   - Resolution:   {w}x{h}
   - Frame Rate:   {fps:.2f} fps
   - Total Frames: {total_video_frames} ({duration:.2f}s)

2. Output Structure:
   ├── motion.json           Global motion asset index & timeline mapping
   ├── metadata.json         Complete pipeline telemetry, SHA-256, & dependencies
   ├── preview.mp4           Visualized overlay with persistent IDs & skeletons
   ├── persons/              Modular per-person motion data files:
"""
    for p in persons_index:
        readme_content += f"   │   ├── {p['person_id']}.json (tracked {p['frames_tracked']} frames, {p['coverage_percentage']}% coverage, mean conf: {p['mean_confidence']})\n"

    readme_content += f"""   └── README.txt            This documentation

3. Blender Pipeline Integration:
   - Each person's motion in persons/person_XXX.json can be loaded independently into Blender.
   - Coordinates:
     * keypoints_normalized: [x, y, score] normalized around mid-hip pelvis root,
       scaled by torso length (mid-hip to mid-shoulder). Y is UP, X is RIGHT.
     * keypoints_pixel: [x, y, score] in original video image coordinates.
     * root_trajectory_norm: [dx, dy] displacement relative to person's frame 0.
   - Keypoint mapping (133 total):
     * 0-16:   17 COCO body joints
     * 17-22:  6 Feet joints (toes and heels)
     * 23-90:  68 Face landmarks
     * 91-111: 21 Left hand keypoints
     * 112-132:21 Right hand keypoints

4. Performance:
   - Processed {total_video_frames} frames in {processing_time_sec:.2f}s ({total_video_frames / max(0.001, processing_time_sec):.2f} fps)
   - Unique people detected: {len(persons_index)}
   - Uncertain frames flagged: {total_uncertain_frames}
================================================================================
"""
    readme_file = output_dir / "README.txt"
    with open(readme_file, "w", encoding="utf-8") as rf:
        rf.write(readme_content)

    results = {
        "motion_json": motion_file,
        "metadata_json": metadata_file,
        "readme_txt": readme_file,
        "persons_dir": persons_dir,
    }
    if source_copy_path:
        results["source_copy"] = source_copy_path

    return results


def build_motion_asset(
    motion_id: str,
    video_path: Path,
    img_shape: tuple,
    fps: float,
    frames_data: List[Dict[str, Any]],
    output_dir: Path,
    processing_time_sec: float = 0.0,
) -> Tuple[Path, Path]:
    """Legacy single-person wrapper for backwards compatibility."""
    class DummyTrack:
        def __init__(self, frames):
            self.person_id = "person_001"
            self.track_id = 1
            self.first_frame = frames[0]["frame_idx"] if frames else 0
            self.last_frame = frames[-1]["frame_idx"] if frames else 0
            self.frames_data = frames
            class DummyNormalizer:
                ref_root_0 = None
                ref_torso_len_0 = None
            self.normalizer = DummyNormalizer()

    tracks = [DummyTrack(frames_data)] if frames_data else []
    res = build_multiperson_motion_asset(
        session_id=motion_id,
        command="MOVE",
        video_path=Path(video_path),
        img_shape=img_shape,
        fps=fps,
        total_video_frames=len(frames_data),
        tracks=tracks,
        output_dir=Path(output_dir),
        processing_time_sec=processing_time_sec,
    )
    return res["motion_json"], res["metadata_json"]
