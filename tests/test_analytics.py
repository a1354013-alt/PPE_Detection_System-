from datetime import datetime, timedelta
import unittest

from analytics import (
    build_dashboard_summary,
    get_crowd_person_stats,
    get_crowd_region_counts,
    get_event_type_counts,
    get_ppe_missing_counts,
    get_ppe_missing_ratio,
    get_severity_counts,
    get_violation_trend,
)
from event_logger import ViolationEvent

class TestAnalytics(unittest.TestCase):
    def setUp(self):
        self.now = datetime.now()
        self.sample_events = [
            ViolationEvent(
                timestamp=(self.now - timedelta(seconds=30)).strftime("%Y-%m-%d %H:%M:%S"),
                source="test",
                track_id="1",
                person_count=1,
                missing_items="helmet",
                screenshot_path="",
                confidence=0.9,
                bbox=""
            ),
            ViolationEvent(
                timestamp=self.now.strftime("%Y-%m-%d %H:%M:%S"),
                source="test",
                track_id="2",
                person_count=1,
                missing_items="helmet, vest",
                screenshot_path="",
                confidence=0.85,
                bbox=""
            )
        ]
        self.crowd_events = [
            ViolationEvent(
                timestamp=self.now.strftime("%Y-%m-%d %H:%M:%S"),
                source="test",
                track_id="crowd:Entrance",
                person_count=6,
                missing_items="",
                screenshot_path="",
                confidence=0.0,
                bbox="",
                event_type="crowd_gathering",
                category="crowd",
                severity="high",
                details="Entrance region crowd alert: 6 people detected, threshold is 3.",
                region_name="Entrance",
                threshold=3,
            ),
            ViolationEvent(
                timestamp=self.now.strftime("%Y-%m-%d %H:%M:%S"),
                source="test",
                track_id="crowd:Entrance",
                person_count=4,
                missing_items="",
                screenshot_path="",
                confidence=0.0,
                bbox="",
                event_type="crowd_gathering",
                category="crowd",
                severity="medium",
                details="Entrance region crowd alert: 4 people detected, threshold is 3.",
                region_name="Entrance",
                threshold=3,
            ),
        ]

    def test_empty_events(self):
        events = []
        counts = get_ppe_missing_counts(events)
        self.assertEqual(counts["helmet"], 0)
        
        trend = get_violation_trend(events)
        self.assertEqual(len(trend["labels"]), 0)
        
        summary = build_dashboard_summary(events)
        self.assertEqual(summary["total_violations"], 0)

    def test_missing_counts(self):
        counts = get_ppe_missing_counts(self.sample_events)
        self.assertEqual(counts["helmet"], 2)
        self.assertEqual(counts["vest"], 1)
        self.assertEqual(counts["mask"], 0)

    def test_missing_ratio(self):
        ratio = get_ppe_missing_ratio(self.sample_events)
        self.assertAlmostEqual(ratio["helmet"], 2/3)
        self.assertAlmostEqual(ratio["vest"], 1/3)

    def test_trend(self):
        trend = get_violation_trend(self.sample_events, interval="10s")
        self.assertTrue(len(trend["labels"]) > 0)
        self.assertEqual(sum(trend["counts"]), 2)

    def test_old_data_compatibility(self):
        # 測試缺少欄位或格式異常 (雖然 dataclass 限制了欄位，但模擬從外部傳入的情況)
        old_event = ViolationEvent(
            timestamp="20230101_120000", # 不同格式
            source="test",
            track_id="3",
            person_count=1,
            missing_items="goggles",
            screenshot_path="",
            confidence=0.7,
            bbox=""
        )
        counts = get_ppe_missing_counts([old_event])
        self.assertEqual(counts["goggles"], 1)

    def test_event_type_count_includes_crowd_gathering(self):
        counts = get_event_type_counts(self.sample_events + self.crowd_events)
        self.assertEqual(counts["ppe_violation"], 2)
        self.assertEqual(counts["crowd_gathering"], 2)

    def test_severity_counts(self):
        counts = get_severity_counts(self.sample_events + self.crowd_events)
        self.assertEqual(counts["medium"], 3)
        self.assertEqual(counts["high"], 1)

    def test_crowd_region_counts(self):
        counts = get_crowd_region_counts(self.sample_events + self.crowd_events)
        self.assertEqual(counts["Entrance"], 2)

    def test_crowd_person_stats(self):
        stats = get_crowd_person_stats(self.sample_events + self.crowd_events)
        self.assertEqual(stats["max_person_count"], 6)
        self.assertAlmostEqual(stats["average_person_count"], 5.0)

    def test_dashboard_summary_contains_crowd_metrics(self):
        summary = build_dashboard_summary(self.sample_events + self.crowd_events)
        self.assertEqual(summary["event_type_counts"]["crowd_gathering"], 2)
        self.assertEqual(summary["crowd_region_counts"]["Entrance"], 2)
        self.assertEqual(summary["crowd_person_stats"]["max_person_count"], 6)

if __name__ == "__main__":
    unittest.main()
