import os
import unittest
from datetime import datetime

from event_logger import EventLogger, ViolationEvent


class TestEventLogger(unittest.TestCase):
    def setUp(self):
        self.logger = EventLogger()
        self.stats = {"missing_counts": {"helmet": 2, "vest": 1, "goggles": 0, "mask": 0}}
        self.processing_summary = {"source_name": "demo.mp4", "demo_mode": True}
        self.event = ViolationEvent(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            source="test_video.mp4",
            track_id="1",
            person_count=2,
            missing_items="helmet, vest",
            screenshot_path="",
            confidence=0.85,
            bbox="x=100,y=100,w=50,h=100",
        )
        self.logger.add_event(self.event)

    def tearDown(self):
        for filename in ["test_report.csv", "test_report.xlsx", "test_report.pdf"]:
            if os.path.exists(filename):
                os.remove(filename)

    def test_export_csv(self):
        self.logger.export_csv("test_report.csv")
        self.assertTrue(os.path.exists("test_report.csv"))

    def test_export_excel_with_processing_summary(self):
        self.logger.export_excel("test_report.xlsx", self.stats, processing_summary=self.processing_summary)
        self.assertTrue(os.path.exists("test_report.xlsx"))

    def test_export_pdf_with_processing_summary(self):
        self.logger.export_pdf("test_report.pdf", self.stats, processing_summary=self.processing_summary)
        self.assertTrue(os.path.exists("test_report.pdf"))


if __name__ == "__main__":
    unittest.main()
