import unittest

from crowd_monitor import (
    CrowdMonitor,
    CrowdRegion,
    box_center_in_region,
    count_people_in_region,
    normalize_box,
)


class TestCrowdMonitor(unittest.TestCase):
    def test_below_threshold_no_event(self):
        monitor = CrowdMonitor([CrowdRegion("Entrance", 0, 0, 1, 1, threshold=3)], temporal_frames=1)
        events = monitor.update([[0.1, 0.1, 0.2, 0.2], [0.3, 0.1, 0.4, 0.2]], 1, 100.0)
        self.assertEqual(events, [])

    def test_temporal_frames_must_be_met(self):
        monitor = CrowdMonitor([CrowdRegion("Entrance", 0, 0, 1, 1, threshold=2)], temporal_frames=2)
        boxes = [[0.1, 0.1, 0.2, 0.2], [0.3, 0.1, 0.4, 0.2]]
        self.assertEqual(monitor.update(boxes, 1, 100.0), [])

    def test_temporal_frames_met_creates_event(self):
        monitor = CrowdMonitor([CrowdRegion("Entrance", 0, 0, 1, 1, threshold=2)], temporal_frames=2)
        boxes = [[0.1, 0.1, 0.2, 0.2], [0.3, 0.1, 0.4, 0.2]]
        monitor.update(boxes, 1, 100.0)
        events = monitor.update(boxes, 2, 101.0)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "crowd_gathering")
        self.assertEqual(events[0]["person_count"], 2)

    def test_roi_outside_people_are_not_counted(self):
        region = CrowdRegion("Left", 0, 0, 0.5, 1, threshold=1)
        self.assertEqual(count_people_in_region([[0.6, 0.1, 0.7, 0.2]], region), 0)

    def test_bbox_center_must_be_inside_roi(self):
        region = CrowdRegion("Left", 0, 0, 0.5, 1, threshold=1)
        self.assertTrue(box_center_in_region([0.4, 0.2, 0.5, 0.4], region))
        self.assertFalse(box_center_in_region([0.4, 0.2, 0.8, 0.4], region))

    def test_cooldown_prevents_duplicate_event(self):
        monitor = CrowdMonitor([CrowdRegion("Entrance", 0, 0, 1, 1, threshold=1)], temporal_frames=1, cooldown_seconds=10)
        boxes = [[0.1, 0.1, 0.2, 0.2]]
        self.assertEqual(len(monitor.update(boxes, 1, 100.0)), 1)
        self.assertEqual(monitor.update(boxes, 2, 105.0), [])

    def test_high_severity_when_count_is_double_threshold(self):
        monitor = CrowdMonitor([CrowdRegion("Entrance", 0, 0, 1, 1, threshold=2)], temporal_frames=1)
        boxes = [
            [0.1, 0.1, 0.2, 0.2],
            [0.3, 0.1, 0.4, 0.2],
            [0.5, 0.1, 0.6, 0.2],
            [0.7, 0.1, 0.8, 0.2],
        ]
        events = monitor.update(boxes, 1, 100.0)
        self.assertEqual(events[0]["severity"], "high")

    def test_multiple_regions_trigger_independently(self):
        monitor = CrowdMonitor(
            [
                CrowdRegion("Left", 0, 0, 0.5, 1, threshold=1),
                CrowdRegion("Right", 0.5, 0, 1, 1, threshold=1),
            ],
            temporal_frames=1,
        )
        events = monitor.update([[0.1, 0.1, 0.2, 0.2], [0.7, 0.1, 0.8, 0.2]], 1, 100.0)
        self.assertEqual({event["region_name"] for event in events}, {"Left", "Right"})

    def test_absolute_box_normalization(self):
        self.assertEqual(normalize_box([0, 0, 50, 100], frame_shape=(100, 100)), (0.0, 0.0, 0.5, 1.0))


if __name__ == "__main__":
    unittest.main()
