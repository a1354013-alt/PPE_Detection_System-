import threading
import unittest
from unittest.mock import Mock, patch

import main_gui


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

        app.finalize_detection.assert_called_once_with(auto_report=True, reason="natural_end")

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


if __name__ == "__main__":
    unittest.main()
