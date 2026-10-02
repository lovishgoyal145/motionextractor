"""
Subject-Centric Motion Normalization
Normalizes 2D whole-body motion around the tracked subject (pelvis/mid-hip root)
scaled by torso length. This decouples camera distance and subject size, enabling
clean retargeting to 2D/2.5D characters or 3D rigs with different proportions.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple


class MotionNormalizer:
    """Computes subject-centric root-relative normalized pose coordinates."""

    def __init__(self):
        self.ref_root_0: Optional[np.ndarray] = None
        self.ref_torso_len_0: Optional[float] = None

    def normalize_frame(
        self,
        keypoints: np.ndarray,  # [133, 2]
        scores: np.ndarray,     # [133]
        bbox: np.ndarray,       # [x1, y1, x2, y2, score]
        img_shape: Tuple[int, int]  # (height, width)
    ) -> Dict[str, Any]:
        """
        Normalizes a single frame of whole-body keypoints.
        Returns:
            - root_pixel: [x, y]
            - root_screen_norm: [x/w, y/h]
            - root_trajectory_norm: [dx, dy] relative to frame 0 in torso units
            - torso_length_px: float
            - keypoints_normalized: [133, 3] (x_norm, y_norm, score) where Y is up
            - keypoints_pixel: [133, 3] (x_px, y_px, score)
        """
        h, w = img_shape
        bx1, by1, bx2, by2, bscore = bbox
        bw = max(1.0, bx2 - bx1)
        bh = max(1.0, by2 - by1)

        # 1. Determine Root Joint (Mid-Hip)
        # 11: left_hip, 12: right_hip
        l_hip = keypoints[11]
        r_hip = keypoints[12]
        l_hip_score = scores[11]
        r_hip_score = scores[12]

        if l_hip_score > 0.2 and r_hip_score > 0.2:
            root = (l_hip + r_hip) / 2.0
            root_conf = (l_hip_score + r_hip_score) / 2.0
        elif l_hip_score > 0.2:
            root = l_hip.copy()
            root_conf = float(l_hip_score)
        elif r_hip_score > 0.2:
            root = r_hip.copy()
            root_conf = float(r_hip_score)
        else:
            # Fallback to lower middle of bounding box
            root = np.array([(bx1 + bx2) / 2.0, by1 + bh * 0.6], dtype=np.float32)
            root_conf = 0.3

        # 2. Determine Reference Torso Length
        # 5: left_shoulder, 6: right_shoulder
        l_sh = keypoints[5]
        r_sh = keypoints[6]
        l_sh_score = scores[5]
        r_sh_score = scores[6]

        if l_sh_score > 0.2 and r_sh_score > 0.2:
            mid_shoulder = (l_sh + r_sh) / 2.0
            torso_length = float(np.linalg.norm(mid_shoulder - root))
        elif l_sh_score > 0.2:
            torso_length = float(np.linalg.norm(l_sh - root))
        elif r_sh_score > 0.2:
            torso_length = float(np.linalg.norm(r_sh - root))
        else:
            torso_length = float(bh * 0.35)

        # Guard against zero or degenerate scale
        if torso_length < 10.0:
            torso_length = max(10.0, float(bh * 0.35))

        # Store frame 0 baseline for trajectory calculation
        if self.ref_root_0 is None:
            self.ref_root_0 = root.copy()
            self.ref_torso_len_0 = torso_length

        # 3. Compute Root-Relative Normalized Keypoints (Y-up convention)
        kpts_norm = np.zeros((133, 3), dtype=np.float32)
        # x_norm: right positive
        kpts_norm[:, 0] = (keypoints[:, 0] - root[0]) / torso_length
        # y_norm: up positive
        kpts_norm[:, 1] = -(keypoints[:, 1] - root[1]) / torso_length
        # confidence
        kpts_norm[:, 2] = scores

        # 4. Pixel keypoints
        kpts_px = np.zeros((133, 3), dtype=np.float32)
        kpts_px[:, 0] = keypoints[:, 0]
        kpts_px[:, 1] = keypoints[:, 1]
        kpts_px[:, 2] = scores

        # 5. Root Trajectory displacement
        root_dx = float((root[0] - self.ref_root_0[0]) / self.ref_torso_len_0)
        root_dy = float(-(root[1] - self.ref_root_0[1]) / self.ref_torso_len_0)

        norm_list = kpts_norm.tolist()
        return {
            "root_pixel": [round(float(root[0]), 2), round(float(root[1]), 2)],
            "root_confidence": round(float(root_conf), 3),
            "root_screen_norm": [round(float(root[0] / w), 4), round(float(root[1] / h), 4)],
            "root_trajectory_norm": [round(root_dx, 4), round(root_dy, 4)],
            "torso_length_px": round(float(torso_length), 2),
            "keypoints_normalized": norm_list,
            "keypoints_pixel": kpts_px.tolist(),
            "body_keypoints": norm_list[0:17],
            "feet_keypoints": norm_list[17:23],
            "face_landmarks": norm_list[23:91],
            "left_hand_keypoints": norm_list[91:112],
            "right_hand_keypoints": norm_list[112:133],
        }
