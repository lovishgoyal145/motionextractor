"""
DWPose Whole-Body Pose Detector
Pure ONNX implementation of YOLOX-L detector + DWPose whole-body (133 keypoints).
Extracts:
- 17 Body keypoints
- 6 Feet keypoints
- 68 Face landmarks
- 42 Hand keypoints (21 per hand)
- Person bounding box & per-joint confidence scores
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List


KEYPOINT_NAMES = [
    # 0-16: COCO 17 Body keypoints
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    # 17-22: 6 Feet keypoints
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
    # 23-90: 68 Face landmarks
    *[f"face_{i}" for i in range(68)],
    # 91-111: 21 Left Hand keypoints
    "left_hand_wrist",
    "left_thumb_cmc", "left_thumb_mcp", "left_thumb_ip", "left_thumb_tip",
    "left_index_mcp", "left_index_pip", "left_index_dip", "left_index_tip",
    "left_middle_mcp", "left_middle_pip", "left_middle_dip", "left_middle_tip",
    "left_ring_mcp", "left_ring_pip", "left_ring_dip", "left_ring_tip",
    "left_pinky_mcp", "left_pinky_pip", "left_pinky_dip", "left_pinky_tip",
    # 112-132: 21 Right Hand keypoints
    "right_hand_wrist",
    "right_thumb_cmc", "right_thumb_mcp", "right_thumb_ip", "right_thumb_tip",
    "right_index_mcp", "right_index_pip", "right_index_dip", "right_index_tip",
    "right_middle_mcp", "right_middle_pip", "right_middle_dip", "right_middle_tip",
    "right_ring_mcp", "right_ring_pip", "right_ring_dip", "right_ring_tip",
    "right_pinky_mcp", "right_pinky_pip", "right_pinky_dip", "right_pinky_tip",
]


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> List[int]:
    """Pure numpy Non-Maximum Suppression."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


class DWPoseDetector:
    """End-to-end Whole-Body DWPose detector using ONNXRuntime."""

    def __init__(self, models_dir: Optional[Path] = None, use_gpu: bool = False):
        if models_dir is None:
            models_dir = Path(__file__).resolve().parents[1] / "models" / "dwpose"
        self.models_dir = Path(models_dir)

        yolox_path = self.models_dir / "yolox_l.onnx"
        GITHUB_RELEASE_URL = "https://github.com/lovishgoyal145/motionextractor/releases/download/v1.0.0"
        HUGGINGFACE_FALLBACK = "https://huggingface.co/yzd-v/DWPose/resolve/main"

        self._ensure_model(yolox_path, f"{GITHUB_RELEASE_URL}/yolox_l.onnx", f"{HUGGINGFACE_FALLBACK}/yolox_l.onnx")
        self._ensure_model(dwpose_path, f"{GITHUB_RELEASE_URL}/dw-ll_ucoco_384.onnx", f"{HUGGINGFACE_FALLBACK}/dw-ll_ucoco_384.onnx")

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.det_sess = ort.InferenceSession(str(yolox_path), sess_options=opts, providers=providers)
        self.pose_sess = ort.InferenceSession(str(dwpose_path), sess_options=opts, providers=providers)

    @staticmethod
    def _ensure_model(model_path: Path, primary_url: str, fallback_url: str):
        """Ensures ONNX model exists locally, downloading automatically from GitHub Releases if missing."""
        model_path = Path(model_path)
        if model_path.exists() and model_path.stat().st_size > 1000:
            return

        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Model weight '{model_path.name}' not found locally. Auto-downloading...")
        print(f"Downloading from {primary_url}...")
        import urllib.request
        try:
            urllib.request.urlretrieve(primary_url, str(model_path))
        except Exception as e:
            print(f"Primary download failed ({e}), trying fallback: {fallback_url}...")
            urllib.request.urlretrieve(fallback_url, str(model_path))
        print(f"Successfully downloaded {model_path.name} ({model_path.stat().st_size / (1024*1024):.1f} MB)!")

    def _decode_yolox(self, outputs: np.ndarray, img_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
        """Decodes YOLOX anchor-free feature maps."""
        grids = []
        expanded_strides = []
        strides = [8, 16, 32]

        hsizes = [img_size[0] // stride for stride in strides]
        wsizes = [img_size[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
        return outputs

    def detect_people(self, image: np.ndarray, conf_thresh: float = 0.30, iou_thresh: float = 0.45) -> List[np.ndarray]:
        """
        Detects all people in image.
        Returns: List of [x1, y1, x2, y2, score] in original image coordinates sorted by descending score.
        """
        h, w = image.shape[:2]
        r = min(640 / h, 640 / w)
        rw, rh = int(w * r), int(h * r)
        resized = cv2.resize(image, (rw, rh))
        padded = np.full((640, 640, 3), 114, dtype=np.uint8)
        padded[:rh, :rw] = resized

        inp = padded.transpose((2, 0, 1))[None].astype(np.float32)
        out = self.det_sess.run(None, {"images": inp})[0]
        decoded = self._decode_yolox(out, (640, 640))

        boxes = decoded[0, :, :4]
        scores = decoded[0, :, 4:5] * decoded[0, :, 5:]  # objectness * class_probs
        person_scores = scores[:, 0]  # Class 0 = Person

        mask = person_scores > conf_thresh
        if not np.any(mask):
            return []

        v_boxes = boxes[mask]
        v_scores = person_scores[mask]

        # Convert [cx, cy, bw, bh] -> [x1, y1, x2, y2]
        x1 = (v_boxes[:, 0] - v_boxes[:, 2] / 2) / r
        y1 = (v_boxes[:, 1] - v_boxes[:, 3] / 2) / r
        x2 = (v_boxes[:, 0] + v_boxes[:, 2] / 2) / r
        y2 = (v_boxes[:, 1] + v_boxes[:, 3] / 2) / r

        # Clip to image boundaries
        x1 = np.clip(x1, 0, w)
        y1 = np.clip(y1, 0, h)
        x2 = np.clip(x2, 0, w)
        y2 = np.clip(y2, 0, h)

        xyxy = np.stack([x1, y1, x2, y2], axis=1)
        keep = nms(xyxy, v_scores, iou_threshold=iou_thresh)
        if not keep:
            return []

        return [
            np.array([xyxy[i, 0], xyxy[i, 1], xyxy[i, 2], xyxy[i, 3], v_scores[i]], dtype=np.float32)
            for i in keep
        ]

    def detect_person(self, image: np.ndarray, conf_thresh: float = 0.35) -> Optional[np.ndarray]:
        """
        Detects primary person in image (highest confidence).
        Returns: [x1, y1, x2, y2, score] in original image coordinates or None.
        """
        people = self.detect_people(image, conf_thresh=conf_thresh)
        return people[0] if people else None

    def estimate_poses(self, image: np.ndarray, bboxes: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Runs DWPose on multiple cropped person bounding boxes using dynamic-batch ONNX execution.
        Returns:
            keypoints_list: List of [133, 2] in original image pixel space.
            scores_list: List of [133] confidence scores.
        """
        if not bboxes:
            return [], []

        norm_crops = []
        inv_transforms = []
        aspect_ratio = 0.75  # width : height = 288 : 384

        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

        for bbox in bboxes:
            bx1, by1, bx2, by2 = bbox[:4]
            bw = max(1.0, bx2 - bx1)
            bh = max(1.0, by2 - by1)
            center_x = (bx1 + bx2) / 2.0
            center_y = (by1 + by2) / 2.0

            if bw / bh > aspect_ratio:
                crop_w = bw * 1.2
                crop_h = crop_w / aspect_ratio
            else:
                crop_h = bh * 1.2
                crop_w = crop_h * aspect_ratio

            src_pts = np.float32([
                [center_x, center_y],
                [center_x, center_y - crop_h / 2],
                [center_x + crop_w / 2, center_y]
            ])
            dst_pts = np.float32([
                [288 / 2, 384 / 2],
                [288 / 2, 0],
                [288, 384 / 2]
            ])
            trans = cv2.getAffineTransform(src_pts, dst_pts)
            inv_trans = cv2.getAffineTransform(dst_pts, src_pts)
            inv_transforms.append(inv_trans)

            cropped = cv2.warpAffine(image, trans, (288, 384), flags=cv2.INTER_LINEAR)
            rgb_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB).astype(np.float32)
            norm_crop = (rgb_cropped - mean) / std
            norm_crops.append(norm_crop.transpose((2, 0, 1)))

        batch_input = np.stack(norm_crops, axis=0)  # [N, 3, 384, 288]
        simcc_x_batch, simcc_y_batch = self.pose_sess.run(None, {"input": batch_input})

        keypoints_list = []
        scores_list = []

        for i in range(len(bboxes)):
            kpts_x = np.argmax(simcc_x_batch[i], axis=1) / 2.0
            kpts_y = np.argmax(simcc_y_batch[i], axis=1) / 2.0
            scores_x = np.max(simcc_x_batch[i], axis=1)
            scores_y = np.max(simcc_y_batch[i], axis=1)
            scores = (scores_x + scores_y) / 2.0

            pts = np.stack([kpts_x, kpts_y, np.ones_like(kpts_x)], axis=0)  # [3, 133]
            orig_pts = inv_transforms[i] @ pts  # [2, 133]
            keypoints = orig_pts.T.astype(np.float32)  # [133, 2]

            keypoints_list.append(keypoints)
            scores_list.append(scores.astype(np.float32))

        return keypoints_list, scores_list

    def estimate_pose(self, image: np.ndarray, bbox: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs DWPose on single cropped person bounding box.
        Returns:
            keypoints: [133, 2] in original image pixel space.
            scores: [133] confidence scores.
        """
        kpts_list, scores_list = self.estimate_poses(image, [bbox])
        return kpts_list[0], scores_list[0]

    def process_frame(self, image: np.ndarray, last_bbox: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Processes single video frame for single primary subject.
        Falls back to last_bbox with margin if detection misses momentarily.
        """
        bbox = self.detect_person(image)
        if bbox is None and last_bbox is not None:
            bbox = last_bbox

        if bbox is None:
            h, w = image.shape[:2]
            bbox = np.array([0, 0, w, h, 0.0], dtype=np.float32)

        keypoints, scores = self.estimate_pose(image, bbox)
        return {
            "bbox": bbox,
            "keypoints": keypoints,
            "scores": scores,
        }

    def process_frame_multi(self, image: np.ndarray, conf_thresh: float = 0.30) -> List[Dict[str, Any]]:
        """
        Processes single video frame for ALL detected people.
        Returns list of dicts with bbox, keypoints, and confidence scores.
        """
        bboxes = self.detect_people(image, conf_thresh=conf_thresh)
        if not bboxes:
            return []

        kpts_list, scores_list = self.estimate_poses(image, bboxes)
        results = []
        for bbox, kpts, sc in zip(bboxes, kpts_list, scores_list):
            results.append({
                "bbox": bbox,
                "keypoints": kpts,
                "scores": sc,
            })
        return results
