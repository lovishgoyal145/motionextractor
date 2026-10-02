"""
Comprehensive Automated Test Suite for Multi-Person Motion Extraction Pipeline
Uses Python's built-in unittest module (no external test runner dependencies).
"""

import sys
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.sequential_allocator import get_next_sequence_dir
from src.tracker import MultiPersonTracker, TrackState, compute_iou
from src.normalization import MotionNormalizer
from src.motion_asset import build_multiperson_motion_asset, compute_file_sha256


class TestSequentialAllocator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_allocator_"))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_monotonic_allocation(self):
        out_base = self.temp_dir / "motion-data"
        d1, s1 = get_next_sequence_dir(out_base)
        d2, s2 = get_next_sequence_dir(out_base)
        d3, s3 = get_next_sequence_dir(out_base)

        self.assertEqual(s1, "0001")
        self.assertEqual(s2, "0002")
        self.assertEqual(s3, "0003")
        self.assertTrue(d1.exists())
        self.assertTrue(d2.exists())
        self.assertTrue(d3.exists())

    def test_collision_avoidance(self):
        out_base = self.temp_dir / "motion-data"
        (out_base / "0001").mkdir(parents=True)
        (out_base / "0005").mkdir(parents=True)

        d_next, s_next = get_next_sequence_dir(out_base)
        self.assertEqual(s_next, "0006")
        self.assertTrue(d_next.exists())


class TestMultiPersonTracker(unittest.TestCase):
    def test_compute_iou(self):
        box1 = np.array([0, 0, 10, 10], dtype=np.float32)
        box2 = np.array([0, 0, 10, 10], dtype=np.float32)
        self.assertAlmostEqual(compute_iou(box1, box2), 1.0, places=4)

        box3 = np.array([10, 10, 20, 20], dtype=np.float32)
        self.assertEqual(compute_iou(box1, box3), 0.0)

        box4 = np.array([5, 0, 15, 10], dtype=np.float32)
        self.assertAlmostEqual(compute_iou(box1, box4), 50.0 / 150.0, places=4)

    def test_multi_person_persistence_and_coasting(self):
        tracker = MultiPersonTracker(iou_thresh=0.25, max_lost=3, min_hits=2)

        # Frame 0: Two people detected
        d_f0 = [
            np.array([100, 100, 200, 300, 0.95], dtype=np.float32),
            np.array([500, 100, 600, 300, 0.90], dtype=np.float32),
        ]
        pairs_0 = tracker.update(d_f0, frame_idx=0)
        self.assertEqual(len(pairs_0), 2)
        id_a = pairs_0[0][0].track_id
        id_b = pairs_0[1][0].track_id
        self.assertNotEqual(id_a, id_b)

        # Frame 1: Slight movement
        d_f1 = [
            np.array([103, 100, 203, 300, 0.94], dtype=np.float32),
            np.array([504, 102, 604, 302, 0.91], dtype=np.float32),
        ]
        pairs_1 = tracker.update(d_f1, frame_idx=1)
        self.assertEqual(len(pairs_1), 2)
        curr_ids = {p[0].track_id for p in pairs_1}
        self.assertEqual(curr_ids, {id_a, id_b})

        # Frame 2: Person B temporarily occluded (not in detections)
        d_f2 = [
            np.array([106, 100, 206, 300, 0.93], dtype=np.float32),
        ]
        pairs_2 = tracker.update(d_f2, frame_idx=2)
        # Person B should be coasting in LOST state
        p_b = [p[0] for p in pairs_2 if p[0].track_id == id_b][0]
        self.assertTrue(p_b.is_uncertain)
        self.assertEqual(p_b.uncertainty_reason, "occlusion_coasting")

        # Frame 3: Person B re-emerges near predicted location
        d_f3 = [
            np.array([109, 100, 209, 300, 0.92], dtype=np.float32),
            np.array([512, 105, 612, 305, 0.89], dtype=np.float32),
        ]
        pairs_3 = tracker.update(d_f3, frame_idx=3)
        self.assertEqual(len(pairs_3), 2)
        curr_ids_3 = {p[0].track_id for p in pairs_3}
        self.assertEqual(curr_ids_3, {id_a, id_b})
        recovered_b = [p[0] for p in pairs_3 if p[0].track_id == id_b][0]
        self.assertFalse(recovered_b.is_uncertain)


class TestMotionNormalizer(unittest.TestCase):
    def test_normalization_invariants(self):
        normalizer = MotionNormalizer()
        kpts = np.zeros((133, 2), dtype=np.float32)
        scores = np.ones(133, dtype=np.float32)

        # Set hips (11, 12) at y=300, shoulders (5, 6) at y=100
        kpts[11] = [180, 300]
        kpts[12] = [220, 300]
        kpts[5] = [170, 100]
        kpts[6] = [230, 100]

        bbox = np.array([150, 80, 250, 400, 0.9], dtype=np.float32)
        res = normalizer.normalize_frame(kpts, scores, bbox, (1080, 1920))

        # Root mid-hip is at (200, 300)
        self.assertEqual(res["root_pixel"], [200.0, 300.0])
        self.assertAlmostEqual(res["root_confidence"], 1.0, places=2)
        self.assertAlmostEqual(res["torso_length_px"], 200.0, places=1)

        # In normalized coords: shoulders are at dy = -(100 - 300)/200 = +1.0 (Y up)
        kpts_norm = res["keypoints_normalized"]
        self.assertAlmostEqual(kpts_norm[5][1], 1.0, places=2)
        self.assertAlmostEqual(kpts_norm[6][1], 1.0, places=2)

        # Slices present
        self.assertEqual(len(res["body_keypoints"]), 17)
        self.assertEqual(len(res["feet_keypoints"]), 6)
        self.assertEqual(len(res["face_landmarks"]), 68)
        self.assertEqual(len(res["left_hand_keypoints"]), 21)
        self.assertEqual(len(res["right_hand_keypoints"]), 21)


class TestAssetGenerationAndSchema(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_schema_"))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_build_multiperson_asset_schema(self):
        out_dir = self.temp_dir / "test_asset"
        video_dummy = self.temp_dir / "dummy.mp4"
        video_dummy.write_bytes(b"dummy video content 12345")

        class MockTrack:
            def __init__(self, t_id):
                self.track_id = t_id
                self.person_id = f"person_{t_id:03d}"
                self.first_frame = 0
                self.last_frame = 2
                class MockNorm:
                    ref_root_0 = np.array([100.0, 200.0])
                    ref_torso_len_0 = 150.0
                self.normalizer = MockNorm()
                self.frames_data = [
                    {
                        "frame_idx": i,
                        "timestamp": round(i / 30.0, 3),
                        "bbox": [10.0, 10.0, 50.0, 100.0, 0.95],
                        "is_uncertain": False,
                        "uncertainty_reason": None,
                        "root_pixel": [30.0, 60.0],
                        "root_confidence": 0.9,
                        "root_screen_norm": [0.03, 0.06],
                        "root_trajectory_norm": [0.0, 0.0],
                        "torso_length_px": 50.0,
                        "keypoints_pixel": [[0.0, 0.0, 0.9]] * 133,
                        "keypoints_normalized": [[0.0, 0.0, 0.9]] * 133,
                    }
                    for i in range(3)
                ]

        tracks = [MockTrack(1), MockTrack(2)]

        artifacts = build_multiperson_motion_asset(
            session_id="0001",
            command="MOVE",
            video_path=video_dummy,
            img_shape=(720, 1280),
            fps=30.0,
            total_video_frames=3,
            tracks=tracks,
            output_dir=out_dir,
            processing_time_sec=1.5,
            archive_source=False,
            status="completed",
        )

        self.assertTrue(artifacts["motion_json"].exists())
        self.assertTrue(artifacts["metadata_json"].exists())
        self.assertTrue(artifacts["readme_txt"].exists())

        p1_file = out_dir / "persons" / "person_001.json"
        p2_file = out_dir / "persons" / "person_002.json"
        self.assertTrue(p1_file.exists())
        self.assertTrue(p2_file.exists())

        import json
        with open(artifacts["metadata_json"], "r") as f:
            meta = json.load(f)
            self.assertEqual(meta["session_id"], "0001")
            self.assertEqual(meta["command"], "MOVE")
            self.assertEqual(meta["source_video"]["sha256"], compute_file_sha256(video_dummy))
            self.assertEqual(meta["tracking_statistics"]["unique_persons_detected"], 2)
            self.assertIn("person_001", meta["tracking_statistics"]["per_person_summary"])
            self.assertIn("person_002", meta["tracking_statistics"]["per_person_summary"])
            self.assertEqual(meta["status"], "completed")

        with open(artifacts["motion_json"], "r") as f:
            motion = json.load(f)
            self.assertEqual(motion["$schema_version"], "2.0.0")
            self.assertEqual(motion["tracked_persons_count"], 2)
            self.assertEqual(len(motion["persons"]), 2)
            self.assertIn("0", motion["timeline_index"])
            self.assertEqual(motion["timeline_index"]["0"], ["person_001", "person_002"])


if __name__ == "__main__":
    unittest.main()
