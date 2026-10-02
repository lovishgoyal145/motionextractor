#!/usr/bin/env python3
"""
Production Multi-Person DWPose Motion Extraction CLI
Usage:
    python run.py XXXX /path/to/video.mp4

Where XXXX is any exactly 4-letter command (e.g. MOVE, POSE, EXTR, ANIM).
Allocates sequential output directories in /home/lovish/motion-data/ (0001, 0002, ...)
and outputs modular Blender-compatible motion assets.
"""

import os
import sys
import time
import signal
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any
import cv2

# Ensure repo root is in python path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.sequential_allocator import get_next_sequence_dir, DEFAULT_OUTPUT_BASE
from src.dwpose_detector import DWPoseDetector
from src.tracker import MultiPersonTracker, TrackState
from src.visualizer import MotionVisualizer, StreamingPreviewWriter
from src.motion_asset import build_multiperson_motion_asset
from src.kaggle_client import KagglePipelineRunner


class MotionExtractionPipeline:
    """Production Multi-Person Motion Extraction Pipeline."""

    def __init__(
        self,
        command: str,
        video_path: Path,
        output_base: Path = DEFAULT_OUTPUT_BASE,
        models_dir: Optional[Path] = None,
        conf_thresh: float = 0.30,
        max_frames: Optional[int] = None,
        archive_source: bool = False,
        single_person: bool = False,
    ):
        self.command = command.upper()
        self.video_path = Path(video_path).resolve()
        self.output_base = Path(output_base).resolve()
        self.models_dir = Path(models_dir) if models_dir else REPO_ROOT / "models" / "dwpose"
        self.conf_thresh = conf_thresh
        self.max_frames = max_frames
        self.archive_source = archive_source
        self.single_person = single_person

        # Allocate next sequential folder atomically
        self.output_dir, self.session_id = get_next_sequence_dir(self.output_base)

        # Global interrupt tracking
        self.interrupted = False
        self.writer: Optional[StreamingPreviewWriter] = None
        self.tracker = MultiPersonTracker(iou_thresh=0.25, max_lost=30, min_hits=2)
        self.visualizer = MotionVisualizer(conf_thresh=0.25)
        self.t_start = 0.0
        self.elapsed = 0.0
        self.total_frames_processed = 0
        self.video_w = 0
        self.video_h = 0
        self.video_fps = 30.0
        self.total_video_frames = 0

    def _register_signals(self):
        def _handle_interrupt(sig, frame):
            print("\n\n[WARNING] Process interrupted by user! Gracefully finalizing partial artifacts...")
            self.interrupted = True

        signal.signal(signal.SIGINT, _handle_interrupt)
        signal.signal(signal.SIGTERM, _handle_interrupt)

    def validate_inputs(self):
        """Strict validation of command, paths, and video readability."""
        if len(self.command) != 4 or not self.command.isalpha():
            raise ValueError(f"Command must be an exactly 4-letter alphabetic code (got '{self.command}'). Example: MOVE")

        if not self.video_path.exists():
            raise FileNotFoundError(f"Input video file does not exist: {self.video_path}")

        if not self.video_path.is_file():
            raise ValueError(f"Input video path is not a file: {self.video_path}")

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video file: {self.video_path}. Verify codec and container.")

        self.video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if self.video_w <= 0 or self.video_h <= 0:
            raise ValueError(f"Invalid video dimensions: {self.video_w}x{self.video_h}")

        if self.max_frames and self.max_frames > 0:
            self.total_video_frames = min(self.total_video_frames, self.max_frames)

    def run_local(self) -> Path:
        """Executes local incremental multi-person extraction."""
        self._register_signals()
        self.validate_inputs()

        print("=" * 60)
        print("    DWPose Production Motion Extraction (Multi-Person)")
        print("=" * 60)
        print(f"Session / Sequence ID:  {self.session_id}")
        print(f"Action Command:         {self.command}")
        print(f"Input Video:            {self.video_path}")
        print(f"Output Directory:       {self.output_dir}")
        print(f"Video Specs:            {self.video_w}x{self.video_h} @ {self.video_fps:.2f} fps ({self.total_video_frames} frames)")
        print(f"Archive Source Copy:    {'YES' if self.archive_source else 'NO (Stored original path + SHA-256)'}")
        print("-" * 60)

        # Initialize detector
        print("Initializing DWPose detector ONNX sessions (CPU)...")
        detector = DWPoseDetector(models_dir=self.models_dir, use_gpu=False)

        # Initialize Streaming Preview Writer (O(1) memory)
        preview_path = self.output_dir / "preview.mp4"
        self.writer = StreamingPreviewWriter(
            output_path=preview_path,
            fps=self.video_fps,
            width=self.video_w,
            height=self.video_h,
        )

        cap = cv2.VideoCapture(str(self.video_path))
        self.t_start = time.time()
        frame_idx = 0

        try:
            while not self.interrupted:
                if self.max_frames and frame_idx >= self.max_frames:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = round(frame_idx / max(1.0, self.video_fps), 3)

                # 1. Multi-Person Detection via YOLOX-L
                raw_bboxes = detector.detect_people(frame, conf_thresh=self.conf_thresh)

                # In single-person mode at frame 0, initialize only the primary (top) detection
                if self.single_person and frame_idx == 0 and len(raw_bboxes) > 0:
                    raw_bboxes = [raw_bboxes[0]]

                # 2. Association & Tracking (IoU + Spatial Hungarian matching)
                track_pairs = self.tracker.update(raw_bboxes, frame_idx=frame_idx)

                # In single-person mode, filter to only the primary track (track_id 1)
                if self.single_person and track_pairs:
                    track_pairs = [p for p in track_pairs if p[0].track_id == 1]

                # 3. Dynamic-Batch Whole-Body Pose Estimation for matched/coasting people
                active_bboxes = []
                active_tracks_in_frame = []
                for track, det_idx in track_pairs:
                    active_bboxes.append(track.bbox)
                    active_tracks_in_frame.append(track)

                if active_bboxes:
                    kpts_list, scores_list = detector.estimate_poses(frame, active_bboxes)
                else:
                    kpts_list, scores_list = [], []

                # 4. Subject-Centric Normalization per tracked person
                for track, kpts, sc in zip(active_tracks_in_frame, kpts_list, scores_list):
                    norm_data = track.normalizer.normalize_frame(
                        keypoints=kpts,
                        scores=sc,
                        bbox=track.bbox,
                        img_shape=(self.video_h, self.video_w),
                    )
                    norm_data["frame_idx"] = frame_idx
                    norm_data["timestamp"] = timestamp
                    norm_data["bbox"] = [round(float(v), 2) for v in track.bbox]
                    norm_data["is_uncertain"] = track.is_uncertain
                    norm_data["uncertainty_reason"] = track.uncertainty_reason

                    track.frames_data.append(norm_data)

                # 5. Draw visualization frame and stream directly to disk (O(1) memory)
                annotated = self.visualizer.draw_multi_person_frame(
                    frame=frame,
                    active_tracks=active_tracks_in_frame,
                    frame_idx=frame_idx,
                    fps=self.video_fps,
                    total_frames=self.total_video_frames,
                    total_unique=len(self.tracker.get_confirmed_tracks()),
                )
                self.writer.write_frame(annotated)

                frame_idx += 1
                self.total_frames_processed = frame_idx

                # Progress display
                pct = int((frame_idx / max(1, self.total_video_frames)) * 100)
                cur_elapsed = max(0.001, time.time() - self.t_start)
                inst_fps = frame_idx / cur_elapsed
                active_count = len(active_tracks_in_frame)
                sys.stdout.write(
                    f"\rProcessing {pct:3d}% | frame {frame_idx:04d}/{self.total_video_frames:04d} | "
                    f"people: {active_count:2d} | speed: {inst_fps:.1f} fps"
                )
                sys.stdout.flush()

        finally:
            cap.release()
            self.elapsed = time.time() - self.t_start
            print()  # Newline after carriage return

        status = "interrupted" if self.interrupted else "completed"
        print(f"\nFinalizing preview video ({preview_path.name})...")
        final_preview = self.writer.close()

        # Build and write final artifacts
        confirmed_tracks = self.tracker.get_confirmed_tracks()
        if self.single_person and confirmed_tracks:
            primary = max(confirmed_tracks, key=lambda t: len(t.frames_data))
            confirmed_tracks = [primary]
        print(f"Writing motion assets for {len(confirmed_tracks)} tracked people...")

        artifacts = build_multiperson_motion_asset(
            session_id=self.session_id,
            command=self.command,
            video_path=self.video_path,
            img_shape=(self.video_h, self.video_w),
            fps=self.video_fps,
            total_video_frames=self.total_frames_processed,
            tracks=confirmed_tracks,
            output_dir=self.output_dir,
            processing_time_sec=self.elapsed,
            archive_source=self.archive_source,
            status=status,
        )

        # Print summary
        print("\n" + "=" * 60)
        print(f"Extraction {status.upper()}! Artifacts saved to: {self.output_dir}")
        print("=" * 60)
        print(f"  [motion.json]    {artifacts['motion_json'].name} ({artifacts['motion_json'].stat().st_size / 1024:.1f} KB)")
        print(f"  [metadata.json]  {artifacts['metadata_json'].name} ({artifacts['metadata_json'].stat().st_size / 1024:.1f} KB)")
        print(f"  [preview.mp4]    {final_preview.name} ({final_preview.stat().st_size / 1024:.1f} KB)")
        print(f"  [README.txt]     {artifacts['readme_txt'].name}")
        for p in confirmed_tracks:
            pf = self.output_dir / "persons" / f"{p.person_id}.json"
            if pf.exists():
                print(f"  [Person Asset]   {p.person_id}: {len(p.frames_data)} frames ({pf.stat().st_size / 1024:.1f} KB)")
        print("-" * 60)
        print(f"Total processing time: {self.elapsed:.2f}s ({self.total_frames_processed / max(0.001, self.elapsed):.2f} fps)")
        print(f"Total frames processed: {self.total_frames_processed}/{self.total_video_frames}")
        print(f"Unique people detected: {len(confirmed_tracks)}")
        print("=" * 60)

        return self.output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Production Multi-Person DWPose Motion Extraction CLI",
        usage="python run.py XXXX /path/to/video.mp4 [options]",
    )
    parser.add_argument("command", type=str, help="Exactly 4-letter action command (e.g. MOVE, POSE)")
    parser.add_argument("video_path", type=str, help="Path to input video file")
    parser.add_argument("--mode", choices=["local", "kaggle"], default="local", help="Execution mode (default: local)")
    parser.add_argument("--output-base", type=str, default="/home/lovish/motion-data", help="Output base directory")
    parser.add_argument("--models-dir", type=str, default=None, help="Directory containing DWPose ONNX models")
    parser.add_argument("--conf-thresh", type=float, default=0.30, help="Detection confidence threshold")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional max frames to process")
    parser.add_argument("--archive-source", action="store_true", help="Opt-in to copy input video to source/original.mp4")
    parser.add_argument("--single", action="store_true", help="Extract motion for single primary character only")

    args = parser.parse_args()

    # Strict validation of exactly 4-letter command
    cmd = args.command.strip().upper()
    if len(cmd) != 4 or not cmd.isalpha():
        print(f"\n[CLI ERROR] '{args.command}' is not a valid 4-letter command.")
        print("Expected exactly 4 letters, e.g.:")
        print("    python run.py MOVE /path/to/video.mp4\n")
        sys.exit(1)

    video_input = Path(args.video_path)
    if not video_input.is_absolute():
        video_input = (Path.cwd() / video_input).resolve()

    if not video_input.exists():
        print(f"\n[CLI ERROR] Video file not found: {video_input}\n")
        sys.exit(1)

    if args.mode == "local":
        pipeline = MotionExtractionPipeline(
            command=cmd,
            video_path=video_input,
            output_base=Path(args.output_base),
            models_dir=Path(args.models_dir) if args.models_dir else None,
            conf_thresh=args.conf_thresh,
            max_frames=args.max_frames,
            archive_source=args.archive_source,
            single_person=args.single,
        )
        try:
            pipeline.run_local()
        except Exception as e:
            print(f"\n[PIPELINE FAILURE] Extraction failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        # Remote Kaggle execution mode preserved
        print(f"Kaggle execution requested for {video_input} with command {cmd}")
        runner = KagglePipelineRunner()
        push_res = runner.push_kernel(REPO_ROOT / "configs")
        if push_res.returncode != 0:
            print(f"Kaggle push failed: {push_res.stderr}")
            sys.exit(1)
        print("Kaggle kernel pushed successfully.")


if __name__ == "__main__":
    main()
