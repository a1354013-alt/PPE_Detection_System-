import os
import random
import sys
import time
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np


fake_ultralytics = types.ModuleType("ultralytics")


class FakeYOLO:
    def __init__(self, *args, **kwargs):
        self.names = {0: "person", 1: "hardhat"}

    def track(self, *args, **kwargs):
        return []


fake_ultralytics.YOLO = FakeYOLO
sys.modules["ultralytics"] = fake_ultralytics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helmet_detector import HelmetDetector, MODEL_NOT_LOADED_ERROR


def create_detector(demo_mode=False):
    with patch("ultralytics.YOLO") as mock_yolo:
        mock_model = Mock()
        mock_model.names = {0: "person", 1: "hardhat"}
        mock_yolo.return_value = mock_model
        return HelmetDetector(demo_mode=demo_mode)


class TestInitialization(unittest.TestCase):
    @patch("ultralytics.YOLO")
    def test_class_map_exists_after_init(self, mock_yolo):
        mock_model = Mock()
        mock_model.names = {0: "person", 1: "hardhat"}
        mock_yolo.return_value = mock_model

        detector = HelmetDetector()
        self.assertTrue(hasattr(detector, "class_map"))
        self.assertIn("hardhat", detector.class_map)

    @patch("ultralytics.YOLO", side_effect=RuntimeError("boom"))
    def test_model_load_failure_sets_status(self, _mock_yolo):
        detector = HelmetDetector()
        self.assertFalse(detector.model_loaded)
        self.assertIn("Failed to load model", detector.model_status_message)

    def test_demo_rng_does_not_pollute_global_random(self):
        random.seed(99)
        expected = random.random()

        random.seed(99)
        create_detector(demo_mode=True)
        actual = random.random()

        self.assertEqual(actual, expected)


class TestProcessingSummary(unittest.TestCase):
    def test_generate_processing_summary_contains_real_values(self):
        detector = create_detector(demo_mode=True)
        detector.video_name = "demo_video.mp4"
        detector.processing_start_time = time.time() - 5
        detector.processing_end_time = time.time()
        detector.total_frames_processed = 150
        detector.violation_log.append({"missing_list": ["helmet"], "missing_items": "helmet"})

        summary = detector.generate_processing_summary()
        self.assertIn("Source Name: demo_video.mp4", summary)
        self.assertIn("Total Violations: 1", summary)
        self.assertIn("Model Status:", summary)


class TestCooldownLogic(unittest.TestCase):
    def setUp(self):
        self.detector = create_detector(demo_mode=True)
        self.detector.violation_cooldown = 2.0

    def test_cooldown_key_matches_sorted_missing_items(self):
        current_time = time.time()
        should_report, keys = self.detector._should_report_event("track_1", ["vest", "helmet"], current_time)

        self.assertTrue(should_report)
        self.assertEqual(keys[0], ("track_1", ("helmet", "vest")))

    def test_same_track_same_violation_in_cooldown(self):
        current_time = time.time()
        event_key = ("track_1", ("helmet",))
        self.detector.counted_violations[event_key] = current_time

        should_report, _ = self.detector._should_report_event("track_1", ["helmet"], current_time + 0.5)
        self.assertFalse(should_report)


class TestDetectionBoundaries(unittest.TestCase):
    def test_real_mode_without_model_returns_error_and_skips_demo(self):
        detector = create_detector(demo_mode=False)
        detector.model = None
        detector.model_loaded = False
        detector._detect_demo = Mock(side_effect=AssertionError("demo should not run"))
        frame = np.zeros((64, 64, 3), dtype=np.uint8)

        annotated, info = detector.detect(frame, ["helmet"], source_name="source", frame_number=1)

        detector._detect_demo.assert_not_called()
        self.assertIs(annotated, frame)
        self.assertFalse(info["violation_detected"])
        self.assertEqual(info["new_events"], [])
        self.assertEqual(info["stable_violations"], [])
        self.assertEqual(info["error"], MODEL_NOT_LOADED_ERROR)

    def test_demo_mode_only_emits_selected_target_items(self):
        detector = create_detector(demo_mode=True)
        detector.temporal_frames = 1
        detector.violation_cooldown = 0
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        _, info = detector.detect(frame, ["helmet"], source_name="demo", frame_number=1)

        self.assertTrue(set(info["missing_items"]).issubset({"helmet"}))
        for _, missing_items, _ in info["stable_violations"]:
            self.assertTrue(set(missing_items).issubset({"helmet"}))
        for event in info["new_events"]:
            self.assertTrue(set(event["missing_list"]).issubset({"helmet"}))
            self.assertNotIn("vest", event["missing_items"])
            self.assertNotIn("mask", event["missing_items"])
            self.assertNotIn("goggles", event["missing_items"])


if __name__ == "__main__":
    unittest.main()
