import io
import os
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from scripts import verify_delivery


class TestDeliveryScript(unittest.TestCase):
    def setUp(self):
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.original_cwd = os.getcwd()
        os.chdir(self.root_dir)

    def tearDown(self):
        os.chdir(self.original_cwd)

    def _capture_check(self, func):
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            result = func()
        return result, stdout_buffer.getvalue(), stderr_buffer.getvalue()

    def test_build_python_command_uses_current_interpreter(self):
        command = verify_delivery.build_python_command("-m", "compileall", "-q", ".")
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1:], ["-m", "compileall", "-q", "."])

    def test_gitignore_contains_required_rules(self):
        with open(".gitignore", "r", encoding="utf-8") as file_obj:
            content = file_obj.read()

        required_rules = [
            "reports/",
            "violations/",
            "outputs/",
            "*.mp4",
            "*.avi",
            "*.mov",
            "*.mkv",
            "*.pt",
            "*.pth",
            "*.onnx",
            "*.engine",
            "*.weights",
        ]

        for rule in required_rules:
            self.assertIn(rule, content)

        self.assertNotIn("```", content)

    def test_check_ignore_rules_detects_missing_rule(self):
        with open(".gitignore", "r", encoding="utf-8") as file_obj:
            original_content = file_obj.read()

        try:
            with open(".gitignore", "w", encoding="utf-8") as file_obj:
                file_obj.write("__pycache__/\n")
            result, stdout_text, stderr_text = self._capture_check(verify_delivery.check_ignore_rules)
            self.assertFalse(result)
            self.assertIn("missing rules", stdout_text)
            self.assertEqual(stderr_text, "")
        finally:
            with open(".gitignore", "w", encoding="utf-8") as file_obj:
                file_obj.write(original_content)

    def test_requirements_checker_detects_missing_package(self):
        with open("requirements.txt", "r", encoding="utf-8") as file_obj:
            original_content = file_obj.read()

        try:
            with open("requirements.txt", "w", encoding="utf-8") as file_obj:
                file_obj.write("numpy\npandas\n")
            result, stdout_text, stderr_text = self._capture_check(verify_delivery.check_requirements)
            self.assertFalse(result)
            self.assertIn("missing", stdout_text)
            self.assertEqual(stderr_text, "")
        finally:
            with open("requirements.txt", "w", encoding="utf-8") as file_obj:
                file_obj.write(original_content)

    def test_forbidden_artifact_checker_detects_files(self):
        fake_walk = [
            (".", ["tests"], ["README.md", "temp_test_model.pt"]),
            (os.path.join(".", "tests"), [], []),
        ]

        with patch("scripts.verify_delivery.os.walk", return_value=fake_walk):
            result, stdout_text, _ = self._capture_check(verify_delivery.check_forbidden_artifacts)
            self.assertFalse(result)
            self.assertIn("Forbidden artifacts found", stdout_text)


if __name__ == "__main__":
    unittest.main()
