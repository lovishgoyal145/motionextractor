"""
Skeleton & Landmark Visualizer
Draws whole-body DWPose skeleton, feet, hands, facial landmarks, and persistent IDs
for multiple people simultaneously, and provides a low-RAM StreamingPreviewWriter.
"""

import math
import cv2
import numpy as np
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any


# Body limb connections (COCO 18 format with neck)
BODY_LIMBS = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # Head to ears
    (0, 17),                                  # Nose to neck
    (17, 5), (17, 6),                         # Neck to shoulders
    (5, 7), (7, 9),                           # Left arm
    (6, 8), (8, 10),                          # Right arm
    (17, 11), (17, 12),                       # Torso / spine to hips
    (11, 13), (13, 15),                       # Left leg
    (12, 14), (14, 16),                       # Right leg
]

BODY_COLORS = [
    (255, 0, 85), (255, 0, 0), (255, 85, 0), (255, 170, 0),
    (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
    (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255),
    (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170),
]

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # Index
    (0, 9), (9, 10), (10, 11), (11, 12),     # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # Pinky
]

FINGER_COLORS = [
    (0, 0, 255), (0, 128, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0)
]

# Distinct palette per tracked person ID (BGR format)
TRACK_PALETTES = [
    (255, 230, 0),   # Person 1: Vibrant Cyan
    (0, 165, 255),   # Person 2: Warm Amber / Orange
    (255, 0, 200),   # Person 3: Magenta / Violet
    (50, 255, 100),  # Person 4: Lime Green
    (100, 100, 255), # Person 5: Coral
    (0, 215, 255),   # Person 6: Gold
    (200, 200, 0),   # Person 7: Teal
    (180, 100, 255), # Person 8: Lavender
]


class StreamingPreviewWriter:
    """
    Streams preview frames directly to disk one-by-one to avoid holding
    thousands of frames in RAM. Remuxes to standard H.264 mp4 on completion.
    """

    def __init__(self, output_path: Path, fps: float, width: int, height: int):
        self.output_path = Path(output_path).resolve()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.temp_raw = self.output_path.with_name("temp_raw_" + self.output_path.name)
        self.fps = float(fps)
        self.width = int(width)
        self.height = int(height)
        self.frames_written = 0

        # Primary codec for writing raw stream
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(self.temp_raw), fourcc, self.fps, (self.width, self.height))
        if not self.writer.isOpened():
            # Fallback to MJPG if mp4v fails
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.writer = cv2.VideoWriter(str(self.temp_raw), fourcc, self.fps, (self.width, self.height))

    def write_frame(self, frame: np.ndarray):
        """Writes single frame to disk stream."""
        if self.writer is not None and self.writer.isOpened():
            self.writer.write(frame)
            self.frames_written += 1

    def close(self) -> Path:
        """Finalizes raw stream and remuxes with ffmpeg for web playback."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None

        if self.frames_written == 0:
            return self.output_path

        # Remux using ffmpeg for standard browser/H.264 playback
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(self.temp_raw),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "fast",
                "-crf", "22",
                str(self.output_path),
            ]
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res.returncode == 0 and self.output_path.exists():
                self.temp_raw.unlink(missing_ok=True)
                return self.output_path
        except Exception:
            pass

        # Fallback to raw file if ffmpeg failed or was unavailable
        if self.temp_raw.exists():
            if self.output_path.exists():
                self.output_path.unlink()
            self.temp_raw.rename(self.output_path)

        return self.output_path


class MotionVisualizer:
    """Renders multi-person skeleton, feet, hands, face, and persistent ID tags."""

    def __init__(self, conf_thresh: float = 0.25):
        self.conf_thresh = conf_thresh

    def draw_multi_person_frame(
        self,
        frame: np.ndarray,
        active_tracks: List[Any],  # List of PersonTrack or dicts
        frame_idx: int = 0,
        fps: float = 30.0,
        total_frames: int = 0,
        total_unique: int = 0,
    ) -> np.ndarray:
        """
        Draws skeletons, persistent IDs, bounding boxes, and HUD for all active people.
        """
        canvas = frame.copy()
        h, w = frame.shape[:2]
        all_confidences = []

        for track_item in active_tracks:
            # Handle either PersonTrack instance or dict
            if hasattr(track_item, "track_id"):
                track_id = track_item.track_id
                person_id = track_item.person_id
                bbox = track_item.bbox
                is_uncertain = track_item.is_uncertain
                reason = track_item.uncertainty_reason
                # Get latest keypoints and scores if available in history
                if track_item.frames_data:
                    last_record = track_item.frames_data[-1]
                    keypoints = np.array(last_record.get("keypoints_pixel", []), dtype=np.float32)
                    scores = keypoints[:, 2] if keypoints.ndim == 2 and keypoints.shape[1] >= 3 else np.zeros(133, dtype=np.float32)
                    keypoints = keypoints[:, :2] if keypoints.ndim == 2 else np.zeros((133, 2), dtype=np.float32)
                else:
                    keypoints = np.zeros((133, 2), dtype=np.float32)
                    scores = np.zeros(133, dtype=np.float32)
            else:
                track_id = track_item.get("track_id", 1)
                person_id = track_item.get("person_id", f"person_{track_id:03d}")
                bbox = track_item.get("bbox", [0, 0, 0, 0, 0.0])
                is_uncertain = track_item.get("is_uncertain", False)
                reason = track_item.get("uncertainty_reason")
                keypoints = np.array(track_item.get("keypoints", np.zeros((133, 2))), dtype=np.float32)
                scores = np.array(track_item.get("scores", np.zeros(133)), dtype=np.float32)

            color = TRACK_PALETTES[(track_id - 1) % len(TRACK_PALETTES)]

            # 1. Draw Bounding Box & Persistent ID Tag
            bx1, by1, bx2, by2 = bbox[:4]
            bscore = float(bbox[4]) if len(bbox) > 4 else 0.0
            if bscore > 0:
                all_confidences.append(bscore)

            box_thickness = 2
            cv2.rectangle(canvas, (int(bx1), int(by1)), (int(bx2), int(by2)), color, box_thickness)

            tag = f"ID: {person_id[-3:]} ({bscore:.2f})"
            if is_uncertain:
                tag = f"ID: {person_id[-3:]} [UNCERTAIN]"

            # Label banner
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            label_y = max(th + 6, int(by1) - 6)
            cv2.rectangle(
                canvas,
                (int(bx1), label_y - th - 4),
                (int(bx1) + tw + 6, label_y + 4),
                color,
                thickness=-1,
            )
            cv2.putText(
                canvas,
                tag,
                (int(bx1) + 3, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

            # Skip skeleton drawing if keypoints are empty/zero (e.g. lost/coasting track)
            if keypoints is None or len(keypoints) < 133 or np.all(keypoints == 0):
                continue

            # 2. Compute Neck joint (midpoint of shoulders)
            body_kpts = keypoints[:17].copy()
            body_scores = scores[:17].copy()
            if body_scores[5] > self.conf_thresh and body_scores[6] > self.conf_thresh:
                neck = (body_kpts[5] + body_kpts[6]) / 2.0
                neck_score = (body_scores[5] + body_scores[6]) / 2.0
            else:
                neck = body_kpts[0]
                neck_score = body_scores[0]

            all_body_kpts = np.vstack([body_kpts, neck[None]])
            all_body_scores = np.append(body_scores, neck_score)

            # 3. Draw Body Limbs
            for idx, (j1, j2) in enumerate(BODY_LIMBS):
                if all_body_scores[j1] > self.conf_thresh and all_body_scores[j2] > self.conf_thresh:
                    p1 = (int(all_body_kpts[j1, 0]), int(all_body_kpts[j1, 1]))
                    p2 = (int(all_body_kpts[j2, 0]), int(all_body_kpts[j2, 1]))
                    limb_col = BODY_COLORS[idx % len(BODY_COLORS)]
                    cv2.line(canvas, p1, p2, limb_col, thickness=3, lineType=cv2.LINE_AA)

            # Draw Body Joints
            for idx in range(18):
                if all_body_scores[idx] > self.conf_thresh:
                    pt = (int(all_body_kpts[idx, 0]), int(all_body_kpts[idx, 1]))
                    cv2.circle(canvas, pt, 4, color, thickness=-1, lineType=cv2.LINE_AA)
                    cv2.circle(canvas, pt, 5, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

            # 4. Draw Feet Keypoints & Connections
            feet_links = [
                (15, 19, (255, 128, 0)),
                (19, 17, (255, 128, 0)),
                (17, 18, (255, 128, 0)),
                (16, 22, (0, 128, 255)),
                (22, 20, (0, 128, 255)),
                (20, 21, (0, 128, 255)),
            ]
            for j1, j2, fl_col in feet_links:
                s1 = scores[j1] if j1 < 133 else 0.0
                s2 = scores[j2] if j2 < 133 else 0.0
                if s1 > self.conf_thresh and s2 > self.conf_thresh:
                    p1 = (int(keypoints[j1, 0]), int(keypoints[j1, 1]))
                    p2 = (int(keypoints[j2, 0]), int(keypoints[j2, 1]))
                    cv2.line(canvas, p1, p2, fl_col, thickness=2, lineType=cv2.LINE_AA)
                    cv2.circle(canvas, p2, 3, fl_col, thickness=-1, lineType=cv2.LINE_AA)

            # 5. Draw Hands (Left: 91-111, Right: 112-132)
            for hand_offset in [91, 112]:
                wrist_pt = (int(keypoints[hand_offset, 0]), int(keypoints[hand_offset, 1]))
                wrist_score = scores[hand_offset]
                if wrist_score > self.conf_thresh:
                    cv2.circle(canvas, wrist_pt, 4, (0, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

                for e_idx, (j1, j2) in enumerate(HAND_EDGES):
                    k1 = hand_offset + j1
                    k2 = hand_offset + j2
                    if scores[k1] > self.conf_thresh and scores[k2] > self.conf_thresh:
                        p1 = (int(keypoints[k1, 0]), int(keypoints[k1, 1]))
                        p2 = (int(keypoints[k2, 0]), int(keypoints[k2, 1]))
                        finger_idx = e_idx // 4
                        f_col = FINGER_COLORS[finger_idx % len(FINGER_COLORS)]
                        cv2.line(canvas, p1, p2, f_col, thickness=2, lineType=cv2.LINE_AA)
                        cv2.circle(canvas, p2, 3, f_col, thickness=-1, lineType=cv2.LINE_AA)

            # 6. Draw Face (23-90)
            face_kpts = keypoints[23:91]
            face_scores = scores[23:91]
            for pt, sc in zip(face_kpts, face_scores):
                if sc > self.conf_thresh:
                    cv2.circle(canvas, (int(pt[0]), int(pt[1])), 2, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

        # 7. Draw Multi-Person HUD Information
        hud_w = min(360, w - 20)
        hud_h = 100
        hud_bg = canvas[10:10 + hud_h, 10:10 + hud_w]
        if hud_bg.shape[0] > 0 and hud_bg.shape[1] > 0:
            black_rect = np.zeros_like(hud_bg)
            canvas[10:10 + hud_h, 10:10 + hud_w] = cv2.addWeighted(hud_bg, 0.35, black_rect, 0.65, 0)

        mean_conf = float(np.mean(all_confidences)) if all_confidences else 0.0
        timestamp = frame_idx / max(1.0, fps)
        tot_str = f"/{total_frames}" if total_frames > 0 else ""

        cv2.putText(canvas, "DWPose Multi-Person Motion", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Frame: {frame_idx:04d}{tot_str} | Time: {timestamp:.2f}s ({fps:.1f} fps)", (18, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Active People: {len(active_tracks)} | Total Unique: {total_unique or len(active_tracks)}", (18, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Mean Subject Conf: {mean_conf:.2f}", (18, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 255, 0) if mean_conf > 0.5 else (0, 165, 255), 1, cv2.LINE_AA)

        return canvas

    def draw_skeleton_frame(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray,
        bbox: np.ndarray,
        frame_idx: int = 0,
        fps: float = 30.0,
    ) -> np.ndarray:
        """Backwards compatibility method for single subject."""
        item = {
            "track_id": 1,
            "person_id": "person_001",
            "bbox": bbox,
            "keypoints": keypoints,
            "scores": scores,
            "is_uncertain": False,
        }
        return self.draw_multi_person_frame(frame, [item], frame_idx, fps)

    def render_preview_video(
        self,
        frames: List[np.ndarray],
        output_path: Path,
        fps: float = 30.0,
    ) -> Path:
        """Legacy helper to render preview from in-memory frame list."""
        if not frames:
            raise ValueError("No frames provided for video rendering.")
        h, w = frames[0].shape[:2]
        writer = StreamingPreviewWriter(output_path, fps, w, h)
        for f in frames:
            writer.write_frame(f)
        return writer.close()
