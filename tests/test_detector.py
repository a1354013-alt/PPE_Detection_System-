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

from helmet_detector import HelmetDetector, MODEL_NOT_LOADED_ERROR, UNSUPPORTED_MODEL_ERROR


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        value = self.values[index]
        if isinstance(value, list):
            return FakeTensor(value)
        return value

    def tolist(self):
        return list(self.values)


class FakeBox:
    def __init__(self, cls_id, coords, conf=0.9, track_id=None):
        self.cls = FakeTensor([cls_id])
        self.xyxy = FakeTensor([coords])
        self.conf = FakeTensor([conf])
        self.id = None if track_id is None else FakeTensor([track_id])


class FakeResult:
    def __init__(self, names, boxes):
        self.names = names
        self.boxes = boxes

    def plot(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)


def make_result(names, detections):
    boxes = [FakeBox(cls_id, coords, conf=conf, track_id=track_id) for cls_id, coords, conf, track_id in detections]
    return FakeResult(names, boxes)


def create_detector(names=None, demo_mode=False):
    names = names or {0: "person", 1: "hardhat"}
    with patch("ultralytics.YOLO") as mock_yolo:
        mock_model = Mock()
        mock_model.names = names
        mock_yolo.return_value = mock_model
        detector = HelmetDetector(demo_mode=demo_mode)
        detector.temporal_frames = 1
        detector.violation_cooldown = 0
        return detector


class TestInitialization(unittest.TestCase):
    @patch("ultralytics.YOLO")
    def test_class_map_exists_after_init(self, mock_yolo):
        mock_model = Mock()
        mock_model.names = {0: "person", 1: "hardhat"}
        mock_yolo.return_value = mock_model

        detector = HelmetDetector()
        self.assertTrue(hasattr(detector, "class_map"))
        self.assertEqual(detector.normalize_class_name("hard_hat"), "helmet")
        self.assertEqual(detector.normalize_class_name("no-hardhat"), "missing_helmet")

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
        self.assertIn("Contract Mode:", summary)


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

    def test_temporal_state_requires_configured_frames(self):
        detector = create_detector(demo_mode=True)
        detector.temporal_frames = 3

        self.assertEqual(detector.update_temporal_state("track-1", ["helmet"]), [])
        self.assertEqual(detector.update_temporal_state("track-1", ["helmet"]), [])
        self.assertEqual(detector.update_temporal_state("track-1", ["helmet"]), ["helmet"])


class TestRealModeContracts(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_presence_based_person_with_helmet_has_no_violation(self):
        detector = create_detector({0: "person", 1: "hardhat"})
        result = make_result(
            {0: "person", 1: "hardhat"},
            [
                (0, [100, 100, 220, 360], 0.95, 1),
                (1, [130, 110, 190, 160], 0.90, 1),
            ],
        )
        detector.run_detection_with_tracking = Mock(return_value=([result], True))

        _, info = detector.detect(self.frame, ["helmet"], source_name="cam", frame_number=1)

        self.assertFalse(info["violation_detected"])
        self.assertEqual(info["new_events"], [])

    def test_presence_based_person_without_helmet_creates_missing_helmet(self):
        detector = create_detector({0: "person", 1: "hardhat"})
        result = make_result(
            {0: "person", 1: "hardhat"},
            [(0, [100, 100, 220, 360], 0.95, 1)],
        )
        detector.run_detection_with_tracking = Mock(return_value=([result], True))

        _, info = detector.detect(self.frame, ["helmet"], source_name="cam", frame_number=2)

        self.assertTrue(info["violation_detected"])
        self.assertEqual(info["missing_items"], ["helmet"])
        self.assertEqual(info["new_events"][0]["missing_list"], ["helmet"])

    def test_presence_based_only_reports_missing_selected_item(self):
        detector = create_detector({0: "person", 1: "hardhat", 2: "safety vest"})
        result = make_result(
            {0: "person", 1: "hardhat", 2: "safety vest"},
            [
                (0, [100, 100, 220, 360], 0.95, 1),
                (2, [120, 210, 200, 320], 0.88, 1),
            ],
        )
        detector.run_detection_with_tracking = Mock(return_value=([result], True))

        _, info = detector.detect(self.frame, ["helmet", "vest"], source_name="cam", frame_number=3)

        self.assertEqual(info["missing_items"], ["helmet"])
        self.assertEqual(info["new_events"][0]["missing_list"], ["helmet"])

    def test_model_without_person_or_violation_classes_is_unsupported(self):
        detector = create_detector({0: "helmet", 1: "vest"})

        is_valid, message = detector.get_contract_validation(["helmet"])

        self.assertFalse(is_valid)
        self.assertEqual(detector.model_capabilities["contract_mode"], "unsupported")
        self.assertIn("Real Mode requires either", message)

    def test_violation_class_based_model_reports_missing_without_person(self):
        detector = create_detector({0: "no_helmet"})
        result = make_result(
            {0: "no_helmet"},
            [(0, [100, 100, 220, 360], 0.95, 7)],
        )
        detector.run_detection_with_tracking = Mock(return_value=([result], True))

        _, info = detector.detect(self.frame, ["helmet"], source_name="cam", frame_number=4)

        self.assertTrue(info["violation_detected"])
        self.assertEqual(info["missing_items"], ["helmet"])
        self.assertEqual(info["new_events"][0]["track_id"], "7")

    def test_mixed_mode_deduplicates_presence_and_direct_violation(self):
        detector = create_detector({0: "person", 1: "helmet", 2: "no_vest"})
        result = make_result(
            {0: "person", 1: "helmet", 2: "no_vest"},
            [
                (0, [100, 100, 220, 360], 0.95, 11),
                (1, [130, 110, 190, 160], 0.93, 11),
                (2, [120, 210, 200, 320], 0.91, 11),
            ],
        )
        detector.run_detection_with_tracking = Mock(return_value=([result], True))

        _, info = detector.detect(self.frame, ["helmet", "vest"], source_name="cam", frame_number=5)

        self.assertTrue(info["violation_detected"])
        self.assertEqual(len(info["new_events"]), 1)
        self.assertEqual(info["new_events"][0]["track_id"], "11")
        self.assertEqual(info["new_events"][0]["missing_list"], ["vest"])

    def test_unsupported_detect_returns_clear_error(self):
        detector = create_detector({0: "person"})
        frame = np.zeros((32, 32, 3), dtype=np.uint8)

        annotated, info = detector.detect(frame, ["helmet"], source_name="cam", frame_number=6)

        self.assertIs(annotated, frame)
        self.assertEqual(info["error"], UNSUPPORTED_MODEL_ERROR)


if __name__ == "__main__":
    unittest.main()
