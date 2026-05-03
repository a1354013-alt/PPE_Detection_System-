import unittest
from datetime import datetime, timedelta
from analytics import get_violation_trend, get_ppe_missing_counts, get_ppe_missing_ratio, build_dashboard_summary
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

if __name__ == "__main__":
    unittest.main()
