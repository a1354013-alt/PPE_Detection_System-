import unittest
import os
from event_logger import EventLogger, ViolationEvent
from datetime import datetime

class TestEventLogging(unittest.TestCase):
    def setUp(self):
        self.logger = EventLogger()
        self.stats = {
            "missing_counts": {"helmet": 2, "vest": 1, "goggles": 0, "mask": 0}
        }
        # 新增測試事件
        self.event1 = ViolationEvent(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            source="test_video.mp4",
            track_id="1",
            person_count=2,
            missing_items="helmet, vest",
            screenshot_path="",
            confidence=0.85,
            bbox="x=100,y=100,w=50,h=100"
        )
        self.event2 = ViolationEvent(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            source="test_video.mp4",
            track_id="2",
            person_count=2,
            missing_items="helmet",
            screenshot_path="",
            confidence=0.92,
            bbox="x=200,y=100,w=50,h=100"
        )

    def test_add_and_clear_events(self):
        self.logger.add_event(self.event1)
        self.assertEqual(len(self.logger.events), 1)
        self.logger.clear_events()
        self.assertEqual(len(self.logger.events), 0)

    def test_export_csv(self):
        self.logger.add_event(self.event1)
        self.logger.add_event(self.event2)
        filename = "test_report.csv"
        self.logger.export_csv(filename)
        self.assertTrue(os.path.exists(filename))
        os.remove(filename)

    def test_export_excel(self):
        self.logger.add_event(self.event1)
        filename = "test_report.xlsx"
        self.logger.export_excel(filename, self.stats)
        self.assertTrue(os.path.exists(filename))
        os.remove(filename)

    def test_export_pdf(self):
        self.logger.add_event(self.event1)
        filename = "test_report.pdf"
        self.logger.export_pdf(filename, self.stats)
        self.assertTrue(os.path.exists(filename))
        os.remove(filename)

    def test_empty_export(self):
        """測試沒有事件時匯出不會 crash"""
        filename = "empty_test.csv"
        self.logger.export_csv(filename)
        self.assertTrue(os.path.exists(filename))
        os.remove(filename)

if __name__ == '__main__':
    unittest.main()
