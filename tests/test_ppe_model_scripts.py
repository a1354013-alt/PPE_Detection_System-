import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

from scripts import check_ppe_model, download_ppe_models


class TestDownloadPPEModels(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self.temp_dir, ignore_errors=True))

    @patch("scripts.download_ppe_models.validate_model_contract", return_value=0)
    @patch("scripts.download_ppe_models.download_from_hugging_face")
    def test_download_hexmon_to_correct_path(self, mock_download, _mock_validate):
        source = os.path.join(self.temp_dir, "source.pt")
        with open(source, "wb") as file_obj:
            file_obj.write(b"model")
        mock_download.return_value = source

        result = download_ppe_models.download_profile("hexmon", out_dir=self.temp_dir)

        self.assertEqual(result, 0)
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "hexmon_vyra_yolo_ppe_best.pt")))

    @patch("scripts.download_ppe_models.validate_model_contract", return_value=0)
    @patch("scripts.download_ppe_models.download_from_hugging_face")
    def test_download_hansung_to_correct_path(self, mock_download, _mock_validate):
        source = os.path.join(self.temp_dir, "source.pt")
        with open(source, "wb") as file_obj:
            file_obj.write(b"model")
        mock_download.return_value = source

        result = download_ppe_models.download_profile("hansung", out_dir=self.temp_dir)

        self.assertEqual(result, 0)
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "hansung_yolov8_ppe_best.pt")))

    @patch("scripts.download_ppe_models.validate_model_contract", return_value=0)
    @patch("scripts.download_ppe_models.download_from_hugging_face")
    def test_existing_without_force_does_not_download(self, mock_download, _mock_validate):
        target = os.path.join(self.temp_dir, "hexmon_vyra_yolo_ppe_best.pt")
        with open(target, "wb") as file_obj:
            file_obj.write(b"existing")

        result = download_ppe_models.download_profile("hexmon", out_dir=self.temp_dir, force=False)

        self.assertEqual(result, 0)
        mock_download.assert_not_called()

    @patch("scripts.download_ppe_models.download_from_hugging_face", side_effect=RuntimeError("network down"))
    def test_download_failure_has_friendly_error(self, _mock_download):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            result = download_ppe_models.download_profile("hexmon", out_dir=self.temp_dir, force=True)

        self.assertEqual(result, 1)
        self.assertIn("Failed to download", stdout.getvalue())
        self.assertIn("network down", stdout.getvalue())


class TestCheckPPEModel(unittest.TestCase):
    def setUp(self):
        handle, self.model_path = tempfile.mkstemp(suffix=".pt")
        os.close(handle)
        self.addCleanup(lambda: os.path.exists(self.model_path) and os.remove(self.model_path))

    @patch("scripts.check_ppe_model.get_model_names")
    def test_missing_person_exits_nonzero(self, mock_names):
        mock_names.return_value = ["Hardhat", "No-Hardhat", "Safety Vest", "No-Safety Vest", "Mask", "No-Mask"]

        result = check_ppe_model.validate_model_contract(
            self.model_path,
            profile=__import__("ppe_model_registry").get_profile("hansung"),
            output=False,
        )

        self.assertNotEqual(result, 0)

    @patch("scripts.check_ppe_model.get_model_names")
    def test_complete_classes_exit_zero(self, mock_names):
        mock_names.return_value = ["Person", "Hardhat", "No-Hardhat", "Safety Vest", "No-Safety Vest", "Mask", "No-Mask"]

        result = check_ppe_model.validate_model_contract(
            self.model_path,
            profile=__import__("ppe_model_registry").get_profile("hansung"),
            output=False,
        )

        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
