import os
import sys
import unittest

from scripts import verify_delivery


class TestDeliveryScript(unittest.TestCase):
    def setUp(self):
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.original_cwd = os.getcwd()
        os.chdir(self.root_dir)

    def tearDown(self):
        os.chdir(self.original_cwd)

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
            self.assertFalse(verify_delivery.check_ignore_rules())
        finally:
            with open(".gitignore", "w", encoding="utf-8") as file_obj:
                file_obj.write(original_content)

    def test_requirements_checker_detects_missing_package(self):
        with open("requirements.txt", "r", encoding="utf-8") as file_obj:
            original_content = file_obj.read()

        try:
            with open("requirements.txt", "w", encoding="utf-8") as file_obj:
                file_obj.write("numpy\npandas\n")
            self.assertFalse(verify_delivery.check_requirements())
        finally:
            with open("requirements.txt", "w", encoding="utf-8") as file_obj:
                file_obj.write(original_content)

    def test_forbidden_artifact_checker_detects_files(self):
        temp_model = "temp_test_model.pt"
        try:
            with open(temp_model, "w", encoding="utf-8") as file_obj:
                file_obj.write("dummy")
            self.assertFalse(verify_delivery.check_forbidden_artifacts())
        finally:
            if os.path.exists(temp_model):
                os.remove(temp_model)


if __name__ == "__main__":
    unittest.main()
