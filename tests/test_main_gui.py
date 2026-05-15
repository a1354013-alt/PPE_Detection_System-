import os
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import main_gui
from event_logger import EventLogger, ViolationEvent


class BoolVarStub:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class StringVarStub(BoolVarStub):
    pass


class DummyWindow:
    def after(self, *_args, **_kwargs):
        return None


class TestCli(unittest.TestCase):
    def test_parse_args_supports_demo_and_model(self):
        args = main_gui.parse_args(["--demo", "--model", "custom.pt"])
        self.assertTrue(args.demo)
        self.assertEqual(args.model, "custom.pt")

    @patch("main_gui.HelmetDetectionApp")
    @patch("main_gui.tk.Tk")
    def test_main_passes_demo_mode_to_app(self, mock_tk, mock_app):
        root = Mock()
        mock_tk.return_value = root

        main_gui.main(["--demo", "--model", "demo.pt"])

        mock_app.assert_called_once_with(root, "PPE Detection System Pro", demo_mode=True, model_path="demo.pt")
        root.protocol.assert_called_once()
        root.mainloop.assert_called_once()


class TestFinalizeFlow(unittest.TestCase):
    def setUp(self):
        self.messagebox_patches = [
            patch("main_gui.messagebox.showinfo"),
            patch("main_gui.messagebox.showwarning"),
            patch("main_gui.messagebox.showerror"),
        ]
        self.mock_messageboxes = [patcher.start() for patcher in self.messagebox_patches]
        self.addCleanup(lambda: [patcher.stop() for patcher in self.messagebox_patches])

    def test_handle_stop_uses_finalize_detection(self):
        app = main_gui.HelmetDetectionApp.__new__(main_gui.HelmetDetectionApp)
        app.finalize_detection = Mock()
        app.worker = Mock()

        app.handle_stop({"reason": "natural_end", "auto_report": True, "notify": False})

        self.assertIsNone(app.worker)
        app.finalize_detection.assert_called_once_with(
            auto_report=True,
            reason="natural_end",
            error_message="",
            notify=False,
        )

    def test_stop_and_report_sets_stop_only_and_waits_for_queue_finalize(self):
        app = main_gui.HelmetDetectionApp.__new__(main_gui.HelmetDetectionApp)
        app.finalize_completed = False
        app.last_finalize_result = None
        app.set_status = Mock()
        app.stop_event = threading.Event()
        app.stop_requested = False
        app.window = DummyWindow()
        app.worker = Mock()

        result = app.stop_and_report()

        self.assertIsNone(result)
        self.assertTrue(app.stop_event.is_set())
        self.assertTrue(app.stop_requested)
        app.set_status.assert_called_once()
        self.assertFalse(app.finalize_completed)

    def test_finalize_detection_is_idempotent(self):
        app = main_gui.HelmetDetectionApp.__new__(main_gui.HelmetDetectionApp)
        app.finalize_completed = False
        app.stop_event = threading.Event()
        app.running = True
        app.detector = Mock()
        app.detector.processing_end_time = None
        app.detector.get_processing_summary_data.return_value = {"source_name": "demo", "total_violations": 0}
        app._collect_finalize_artifacts = Mock(return_value=(None, None))
        app.btn_upload = Mock()
        app.btn_camera = Mock()
        app.vid = None
        app.set_status = Mock()
        app.demo_mode = False
        app.current_run_dir = None
        app.last_finalize_result = None

        first = app.finalize_detection(auto_report=True, reason="manual_stop", notify=False)
        second = app.finalize_detection(auto_report=True, reason="manual_stop", notify=False)

        self.assertIs(first, second)
        app._collect_finalize_artifacts.assert_called_once()

    def test_finalize_detection_releases_video_after_worker_done(self):
        app = main_gui.HelmetDetectionApp.__new__(main_gui.HelmetDetectionApp)
        app.finalize_completed = False
        app.stop_event = threading.Event()
        app.running = True
        app.detector = Mock()
        app.detector.processing_end_time = None
        app.detector.get_processing_summary_data.return_value = {"source_name": "demo", "total_violations": 0}
        app._collect_finalize_artifacts = Mock(return_value=(None, None))
        app.btn_upload = Mock()
        app.btn_camera = Mock()
        vid = Mock()
        app.vid = vid
        app.set_status = Mock()
        app.demo_mode = False
        app.current_run_dir = None
        app.last_finalize_result = None

        app.handle_stop({"reason": "natural_end", "auto_report": True, "notify": False})

        vid.release.assert_called_once()

    def test_finalize_detection_releases_video_only_once(self):
        app = main_gui.HelmetDetectionApp.__new__(main_gui.HelmetDetectionApp)
        app.finalize_completed = False
        app.stop_event = threading.Event()
        app.running = True
        app.detector = Mock()
        app.detector.processing_end_time = None
        app.detector.get_processing_summary_data.return_value = {"source_name": "demo", "total_violations": 0}
        app._collect_finalize_artifacts = Mock(return_value=(None, None))
        app.btn_upload = Mock()
        app.btn_camera = Mock()
        vid = Mock()
        app.vid = vid
        app.set_status = Mock()
        app.demo_mode = False
        app.current_run_dir = None
        app.last_finalize_result = None

        app.finalize_detection(auto_report=True, reason="manual_stop", notify=False)
        app.finalize_detection(auto_report=True, reason="manual_stop", notify=False)

        vid.release.assert_called_once()
        app._collect_finalize_artifacts.assert_called_once()

    def test_handle_stop_after_finalize_does_not_export_again(self):
        app = main_gui.HelmetDetectionApp.__new__(main_gui.HelmetDetectionApp)
        app.finalize_completed = False
        app.stop_event = threading.Event()
        app.running = True
        app.detector = Mock()
        app.detector.processing_end_time = None
        app.detector.get_processing_summary_data.return_value = {"source_name": "demo", "total_violations": 0}
        app._collect_finalize_artifacts = Mock(return_value=(None, None))
        app.btn_upload = Mock()
        app.btn_camera = Mock()
        vid = Mock()
        app.vid = vid
        app.set_status = Mock()
        app.demo_mode = False
        app.current_run_dir = None
        app.last_finalize_result = None
        worker = Mock()
        app.worker = worker

        app.handle_stop({"reason": "natural_end", "auto_report": True, "notify": False})
        app.worker = worker
        app.handle_stop({"reason": "natural_end", "auto_report": True, "notify": False})

        self.assertIsNone(app.worker)
        app._collect_finalize_artifacts.assert_called_once()
        vid.release.assert_called_once()


class TestStartDetectionValidation(unittest.TestCase):
    @patch("main_gui.cv2.VideoCapture")
    def test_start_detection_blocks_when_validate_model_support_fails(self, mock_capture):
        app = main_gui.HelmetDetectionApp.__new__(main_gui.HelmetDetectionApp)
        app.running = False
        app.demo_mode = False
        app.validate_model_support = Mock(return_value=False)
        app.reset_detection_state = Mock()
        app.set_status = Mock()

        app.start_detection("demo.mp4")

        app.validate_model_support.assert_called_once_with(show_message=True)
        app.reset_detection_state.assert_not_called()
        mock_capture.assert_not_called()
        self.assertFalse(app.running)
        app.set_status.assert_called_once()

    @patch("main_gui.messagebox.showerror")
    def test_validate_model_support_returns_false_for_unsupported_contract(self, mock_error):
        app = main_gui.HelmetDetectionApp.__new__(main_gui.HelmetDetectionApp)
        app.demo_mode = False
        app.update_model_info = Mock()
        app.detector = Mock()
        app.detector.model_loaded = True
        app.detector.get_contract_validation.return_value = (False, "unsupported")
        app.check_vars = {
            "helmet": BoolVarStub(True),
            "vest": BoolVarStub(False),
            "goggles": BoolVarStub(False),
            "mask": BoolVarStub(False),
        }

        result = app.validate_model_support(show_message=True)

        self.assertFalse(result)
        mock_error.assert_called_once()

    def test_crowd_settings_are_read_without_display(self):
        app = main_gui.HelmetDetectionApp.__new__(main_gui.HelmetDetectionApp)
        app.enable_crowd_region_alert = BoolVarStub(True)
        app.crowd_region_name = StringVarStub("Entrance")
        app.crowd_x1 = StringVarStub("0.1")
        app.crowd_y1 = StringVarStub("0.2")
        app.crowd_x2 = StringVarStub("0.8")
        app.crowd_y2 = StringVarStub("0.9")
        app.crowd_threshold = StringVarStub("7")
        app.crowd_temporal_frames = StringVarStub("4")
        app.crowd_cooldown_seconds = StringVarStub("12.5")

        settings = app.get_crowd_settings()

        self.assertTrue(settings["enabled"])
        self.assertEqual(settings["region_name"], "Entrance")
        self.assertEqual(settings["x1"], 0.1)
        self.assertEqual(settings["threshold"], 7)
        self.assertEqual(settings["temporal_frames"], 4)
        self.assertEqual(settings["cooldown_seconds"], 12.5)

    def test_crowd_event_ui_uses_generic_columns(self):
        app = main_gui.HelmetDetectionApp.__new__(main_gui.HelmetDetectionApp)
        app.stats = {"total_violations": 0, "missing_counts": {"helmet": 0}, "event_type_counts": {}, "severity_counts": {}, "crowd_region_counts": {}}
        app.lbl_total_v = Mock()
        app.update_chart = Mock()
        app.tree = Mock()
        app.tree.get_children.return_value = []

        app.add_event_to_ui(
            {
                "timestamp": "2026-05-15 10:00:00",
                "event_type": "crowd_gathering",
                "severity": "high",
                "details": "Entrance region crowd alert: 6 people detected, threshold is 5.",
                "region_name": "Entrance",
                "screenshot_path": "",
                "missing_list": [],
            }
        )

        self.assertEqual(app.stats["event_type_counts"]["crowd_gathering"], 1)
        self.assertEqual(app.stats["crowd_region_counts"]["Entrance"], 1)
        inserted_values = app.tree.insert.call_args.kwargs["values"]
        self.assertEqual(inserted_values[1], "crowd_gathering")
        self.assertEqual(inserted_values[2], "high")


class TestRunOutputs(unittest.TestCase):
    def test_new_run_folder_preserves_existing_files(self):
        app = main_gui.HelmetDetectionApp.__new__(main_gui.HelmetDetectionApp)
        temp_root = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(temp_root, ignore_errors=True))
        app.output_root = temp_root
        app.current_run_dir = None
        app.current_screenshots_dir = None
        app.current_reports_dir = None
        app.current_violation_csv_path = None
        app.current_heatmap_path = None

        first_run = app.ensure_run_output_dir()
        first_file = os.path.join(first_run, "screenshots", "first.txt")
        with open(first_file, "w", encoding="utf-8") as file_obj:
            file_obj.write("keep me")

        app.current_run_dir = None
        second_run = app.ensure_run_output_dir()

        self.assertNotEqual(first_run, second_run)
        self.assertTrue(os.path.exists(first_file))
        self.assertTrue(os.path.isdir(os.path.join(second_run, "screenshots")))

    def test_event_logger_exports_reports_inside_current_run_folder(self):
        app = main_gui.HelmetDetectionApp.__new__(main_gui.HelmetDetectionApp)
        temp_root = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(temp_root, ignore_errors=True))
        app.output_root = temp_root
        app.current_run_dir = None
        app.current_screenshots_dir = None
        app.current_reports_dir = None
        app.current_violation_csv_path = None
        app.current_heatmap_path = None
        app.event_logger = EventLogger()
        app.event_logger.add_event(
            ViolationEvent(
                timestamp="2026-05-15 10:00:00",
                source="demo.mp4",
                track_id="1",
                person_count=1,
                missing_items="helmet",
                screenshot_path="",
                confidence=0.9,
                bbox="x=1,y=2,w=3,h=4",
            )
        )
        app.stats = {"missing_counts": {"helmet": 1, "vest": 0, "goggles": 0, "mask": 0}}

        csv_path = app._build_report_filepath("csv")
        xlsx_path = app._build_report_filepath("xlsx")
        pdf_path = app._build_report_filepath("pdf")

        self.assertTrue(csv_path.startswith(app.current_reports_dir))
        self.assertTrue(xlsx_path.startswith(app.current_reports_dir))
        self.assertTrue(pdf_path.startswith(app.current_reports_dir))
        self.assertTrue(os.path.isdir(app.current_screenshots_dir))

        app._export_report_file("csv", csv_path, {})
        app._export_report_file("xlsx", xlsx_path, {})
        app._export_report_file("pdf", pdf_path, {})

        self.assertTrue(os.path.exists(csv_path))
        self.assertTrue(os.path.exists(xlsx_path))
        self.assertTrue(os.path.exists(pdf_path))


if __name__ == "__main__":
    unittest.main()
