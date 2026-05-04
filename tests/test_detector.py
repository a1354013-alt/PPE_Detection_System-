import os
import sys
import time
import types
import unittest
from unittest.mock import Mock, patch


fake_ultralytics = types.ModuleType("ultralytics")


class FakeYOLO:
    def __init__(self, *args, **kwargs):
        self.names = {0: "person", 1: "hardhat"}

    def track(self, *args, **kwargs):
        return []


fake_ultralytics.YOLO = FakeYOLO
sys.modules["ultralytics"] = fake_ultralytics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helmet_detector import HelmetDetector


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


if __name__ == "__main__":
    unittest.main()
