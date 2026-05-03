import unittest
import os
import sys
import shutil
from scripts.verify_delivery import check_ignore_rules, check_requirements, check_forbidden_artifacts

class TestDeliveryScript(unittest.TestCase):
    def setUp(self):
        # Change to project root if needed
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.chdir(self.root_dir)

    def test_import_verify_delivery(self):
        """驗證 verify_delivery.py 可以被 import"""
        try:
            import scripts.verify_delivery
            self.assertTrue(True)
        except ImportError:
            self.fail("Failed to import scripts.verify_delivery")

    def test_ignore_rule_checker_detects_missing(self):
        """驗證 ignore rule checker 可以偵測缺少規則"""
        # Backup gitignore
        with open(".gitignore", "r") as f:
            original_content = f.read()
        
        try:
            # Create a temporary gitignore without reports/
            with open(".gitignore", "w") as f:
                f.write("*.pyc\n")
            
            # Should return False because reports/ is missing
            self.assertFalse(check_ignore_rules())
        finally:
            # Restore gitignore
            with open(".gitignore", "w") as f:
                f.write(original_content)

    def test_requirements_checker_detects_missing(self):
        """驗證 requirements checker 可以偵測缺少必要套件"""
        # Backup requirements.txt
        with open("requirements.txt", "r") as f:
            original_content = f.read()
        
        try:
            # Create a temporary requirements.txt without reportlab
            with open("requirements.txt", "w") as f:
                f.write("numpy\npandas\n")
            
            # Should return False because reportlab is missing
            self.assertFalse(check_requirements())
        finally:
            # Restore requirements.txt
            with open("requirements.txt", "w") as f:
                f.write(original_content)

    def test_forbidden_artifact_checker_detects_files(self):
        """驗證 forbidden artifact checker 可以偵測大型模型檔"""
        test_file = "temp_test_model.pt"
        try:
            # Create a dummy .pt file
            with open(test_file, "w") as f:
                f.write("dummy")
            
            # Should return False because .pt file is found
            self.assertFalse(check_forbidden_artifacts())
        finally:
            # Cleanup
            if os.path.exists(test_file):
                os.remove(test_file)

if __name__ == '__main__':
    unittest.main()
