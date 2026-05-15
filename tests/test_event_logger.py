import os
import shutil
import unittest
from datetime import datetime

import pandas as pd

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
        self.crowd_event = ViolationEvent(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            source="test_video.mp4",
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
            frame_index=10,
        )
        self.temp_root = os.path.join(os.path.dirname(__file__), "_tmp_event_logger")
        shutil.rmtree(self.temp_root, ignore_errors=True)
        os.makedirs(self.temp_root, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def test_format_event_time_handles_legacy_timestamp(self):
        self.assertEqual(self.logger._format_event_time("20230101_120000"), "20230101_120000")
        self.assertEqual(self.logger._format_event_time("2023-01-01 12:00:00"), "12:00:00")

    def test_export_csv_creates_missing_parent_directory(self):
        filepath = os.path.join(self.temp_root, "nested", "test_report.csv")
        self.logger.export_csv(filepath)
        self.assertTrue(os.path.exists(filepath))
        df = pd.read_csv(filepath)
        self.assertIn("event_type", df.columns)
        self.assertEqual(df.loc[0, "event_type"], "ppe_violation")

    def test_export_excel_with_processing_summary(self):
        filepath = os.path.join(self.temp_root, "nested", "test_report.xlsx")
        self.logger.export_excel(filepath, self.stats, processing_summary=self.processing_summary)
        self.assertTrue(os.path.exists(filepath))

    def test_export_pdf_with_processing_summary_and_legacy_timestamp(self):
        self.logger.events[0] = ViolationEvent(
            timestamp="20230101_120000",
            source="test_video.mp4",
            track_id="1",
            person_count=2,
            missing_items="helmet, vest",
            screenshot_path="",
            confidence=0.85,
            bbox="x=100,y=100,w=50,h=100",
        )
        filepath = os.path.join(self.temp_root, "nested", "test_report.pdf")
        self.logger.export_pdf(filepath, self.stats, processing_summary=self.processing_summary)
        self.assertTrue(os.path.exists(filepath))

    def test_crowd_event_exports_all_formats(self):
        self.logger.add_event(self.crowd_event)
        stats = {
            "missing_counts": {"helmet": 1},
            "crowd_region_counts": {"Entrance": 1},
        }
        csv_path = os.path.join(self.temp_root, "crowd.csv")
        xlsx_path = os.path.join(self.temp_root, "crowd.xlsx")
        pdf_path = os.path.join(self.temp_root, "crowd.pdf")

        self.logger.export_csv(csv_path)
        self.logger.export_excel(xlsx_path, stats, processing_summary=self.processing_summary)
        self.logger.export_pdf(pdf_path, stats, processing_summary=self.processing_summary)

        df = pd.read_csv(csv_path)
        self.assertIn("crowd_gathering", set(df["event_type"]))
        self.assertTrue(os.path.exists(xlsx_path))
        self.assertTrue(os.path.exists(pdf_path))
        self.assertEqual(self.logger._event_type_counts(self.logger.events)["crowd_gathering"], 1)


if __name__ == "__main__":
    unittest.main()
