import threading
import unittest
from unittest.mock import Mock, patch

import main_gui


class BoolVarStub:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


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
    def test_handle_stop_uses_finalize_detection(self):
        app = main_gui.HelmetDetectionApp.__new__(main_gui.HelmetDetectionApp)
        app.finalize_detection = Mock()

        app.handle_stop({"reason": "natural_end", "auto_report": True})

        app.finalize_detection.assert_called_once_with(auto_report=True, reason="natural_end", error_message="")

    def test_stop_and_report_uses_finalize_detection(self):
        app = main_gui.HelmetDetectionApp.__new__(main_gui.HelmetDetectionApp)
        app.set_status = Mock()
        app.stop_event = threading.Event()
        worker = Mock()
        app.worker = worker
        app.finalize_detection = Mock()

        app.stop_and_report()

        self.assertTrue(app.stop_event.is_set())
        worker.join.assert_called_once_with(timeout=5.0)
        app.finalize_detection.assert_called_once_with(auto_report=True, reason="manual_stop")


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

    @patch("main_gui.messagebox.showwarning")
    def test_validate_model_support_returns_false_for_unsupported_items(self, mock_warning):
        app = main_gui.HelmetDetectionApp.__new__(main_gui.HelmetDetectionApp)
        app.demo_mode = False
        app.update_model_info = Mock()
        app.detector = Mock()
        app.detector.model_loaded = True
        app.detector.get_model_classes.return_value = ["person"]
        app.check_vars = {
            "helmet": BoolVarStub(True),
            "vest": BoolVarStub(False),
            "goggles": BoolVarStub(False),
            "mask": BoolVarStub(False),
        }

        result = app.validate_model_support(show_message=True)

        self.assertFalse(result)
        mock_warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()
