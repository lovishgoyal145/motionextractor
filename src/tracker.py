"""
Deterministic Multi-Person Tracker
Provides persistent, stable person IDs across video frames using IoU overlap,
linear velocity extrapolation, and optimal Hungarian matching (scipy linear_sum_assignment).
Includes explicit ambiguity detection and uncertainty reporting.
"""

from enum import Enum
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from scipy.optimize import linear_sum_assignment
from .normalization import MotionNormalizer


class TrackState(Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    DELETED = "deleted"


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Computes Intersection-over-Union between two boxes [x1, y1, x2, y2]."""
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union_area = area_a + area_b - inter_area

    if union_area <= 1e-6:
        return 0.0
    return inter_area / union_area


class PersonTrack:
    """Represents a single tracked person across video frames."""

    def __init__(self, track_id: int, bbox: np.ndarray, frame_idx: int, confirmed: bool = False):
        self.track_id = track_id
        self.person_id = f"person_{track_id:03d}"
        self.bbox = np.array(bbox, dtype=np.float32)  # [x1, y1, x2, y2, score]
        self.velocity = np.zeros(4, dtype=np.float32)  # [dx1, dy1, dx2, dy2]
        self.alpha_vel = 0.35  # Velocity smoothing factor

        self.first_frame = frame_idx
        self.last_frame = frame_idx
        self.hits = 1
        self.missed_frames = 0
        self.state = TrackState.CONFIRMED if confirmed else TrackState.TENTATIVE

        self.is_uncertain = False
        self.uncertainty_reason: Optional[str] = None

        # Each person maintains an independent MotionNormalizer (reference frame 0 root)
        self.normalizer = MotionNormalizer()

        # Motion frame records for this person
        self.frames_data: List[Dict[str, Any]] = []

    def predict(self) -> np.ndarray:
        """Predicts bounding box position in next frame using current velocity."""
        pred = self.bbox.copy()
        pred[:4] += self.velocity
        return pred

    def update(self, bbox: np.ndarray, frame_idx: int):
        """Updates track with a new detection match."""
        new_bbox = np.array(bbox, dtype=np.float32)
        measured_vel = new_bbox[:4] - self.bbox[:4]

        # Update smoothed velocity
        if self.hits == 1:
            self.velocity = measured_vel
        else:
            self.velocity = (1.0 - self.alpha_vel) * self.velocity + self.alpha_vel * measured_vel

        self.bbox = new_bbox
        self.last_frame = frame_idx
        self.hits += 1
        self.missed_frames = 0
        self.state = TrackState.CONFIRMED

        # Uncertainty check based on detection confidence
        if new_bbox[4] < 0.40:
            self.is_uncertain = True
            self.uncertainty_reason = "low_detection_confidence"
        else:
            self.is_uncertain = False
            self.uncertainty_reason = None

    def mark_missed(self, frame_idx: int, max_lost: int = 30):
        """Coasts track position when detection is momentarily missed."""
        self.bbox[:4] += self.velocity
        self.missed_frames += 1
        self.is_uncertain = True
        self.uncertainty_reason = "occlusion_coasting"

        if self.missed_frames > max_lost:
            self.state = TrackState.DELETED
        else:
            self.state = TrackState.LOST


class MultiPersonTracker:
    """
    Deterministic Multi-Person Tracker.
    Associates DWPose person detections across consecutive frames.
    """

    def __init__(self, iou_thresh: float = 0.25, max_lost: int = 30, min_hits: int = 2):
        self.iou_thresh = iou_thresh
        self.max_lost = max_lost
        self.min_hits = min_hits
        self.next_track_id = 1
        self.tracks: List[PersonTrack] = []

    def update(self, detections: List[np.ndarray], frame_idx: int) -> List[Tuple[PersonTrack, Optional[int]]]:
        """
        Updates tracks with detections for the current frame.
        detections: List of [x1, y1, x2, y2, score]
        Returns list of (track, detection_idx) for active tracks in this frame.
        """
        active_tracks = [t for t in self.tracks if t.state != TrackState.DELETED]

        # Special initialization for frame 0: directly confirm all detections
        if frame_idx == 0 or len(active_tracks) == 0:
            for det in detections:
                track = PersonTrack(
                    track_id=self.next_track_id,
                    bbox=det,
                    frame_idx=frame_idx,
                    confirmed=(frame_idx == 0 or det[4] >= 0.50),
                )
                self.next_track_id += 1
                self.tracks.append(track)
            
            # Return confirmed/active tracks
            return [(t, i) for i, t in enumerate(self.tracks) if t.state == TrackState.CONFIRMED]

        # Step 1: Predict positions for existing tracks
        predicted_boxes = [t.predict() for t in active_tracks]

        # Step 2: Compute pairwise IoU cost matrix
        num_tracks = len(active_tracks)
        num_dets = len(detections)

        matched_tracks = set()
        matched_dets = set()
        track_det_pairs = []

        if num_tracks > 0 and num_dets > 0:
            cost_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)
            for i, pbox in enumerate(predicted_boxes):
                for j, dbox in enumerate(detections):
                    iou = compute_iou(pbox, dbox)
                    cost_matrix[i, j] = 1.0 - iou

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for t_idx, d_idx in zip(row_ind, col_ind):
                iou = 1.0 - cost_matrix[t_idx, d_idx]
                if iou >= self.iou_thresh:
                    matched_tracks.add(t_idx)
                    matched_dets.add(d_idx)
                    track = active_tracks[t_idx]
                    det = detections[d_idx]

                    # Step 3: Ambiguity detection check
                    # If another detection or track has very close IoU, flag uncertainty
                    competing_ious = [
                        1.0 - cost_matrix[k, d_idx]
                        for k in range(num_tracks)
                        if k != t_idx
                    ]
                    if competing_ious and (iou - max(competing_ious)) < 0.12:
                        track.update(det, frame_idx)
                        track.is_uncertain = True
                        track.uncertainty_reason = "ambiguous_track_overlap"
                    else:
                        track.update(det, frame_idx)

                    track_det_pairs.append((track, d_idx))

        # Step 4: Handle unmatched detections -> New tracks
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                # Require reasonable confidence to spawn new track
                if det[4] >= 0.35:
                    new_track = PersonTrack(
                        track_id=self.next_track_id,
                        bbox=det,
                        frame_idx=frame_idx,
                        confirmed=False,
                    )
                    self.next_track_id += 1
                    self.tracks.append(new_track)
                    # If min_hits is 1, confirm immediately
                    if self.min_hits <= 1:
                        new_track.state = TrackState.CONFIRMED
                        track_det_pairs.append((new_track, d_idx))

        # Step 5: Handle unmatched active tracks -> Coast or mark lost
        for t_idx, track in enumerate(active_tracks):
            if t_idx not in matched_tracks:
                track.mark_missed(frame_idx, max_lost=self.max_lost)
                if track.state == TrackState.LOST:
                    # Provide coasted prediction for visual continuity
                    track_det_pairs.append((track, None))

        # Return tracks that are currently CONFIRMED or LOST (active)
        return [
            pair for pair in track_det_pairs
            if pair[0].state in (TrackState.CONFIRMED, TrackState.LOST)
        ]

    def get_confirmed_tracks(self) -> List[PersonTrack]:
        """Returns all tracks that have reached CONFIRMED status."""
        return [t for t in self.tracks if t.hits >= self.min_hits or t.state == TrackState.CONFIRMED]
