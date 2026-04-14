import pytest
import os
import sys
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helmet_detector import HelmetDetector, MAX_LOG_SIZE


class TestHelmetDetector:
    """Test cases for HelmetDetector class"""

    def test_model_load_failure(self):
        """Test that model loading fails gracefully with non-existent file"""
        detector = HelmetDetector.__new__(HelmetDetector)
        detector.model = None
        detector.model_path = 'non_existent_model.pt'
        detector.conf = 0.4
        detector.iou = 0.45
        detector.class_map = {
            'hardhat': 'helmet',
            'head_helmet': 'helmet',
            'safety_helmet': 'helmet',
            'safety_vest': 'vest',
            'reflective_vest': 'vest',
            'goggles': 'goggles',
            'eye_protection': 'goggles',
            'mask': 'mask',
            'face_mask': 'mask'
        }
        detector.violation_coords = []
        detector.violation_buffer = []
        detector.violation_log = []
        
        success, msg = detector.load_model('non_existent_model_xyz.pt')
        
        assert success is False
        assert detector.model is None
        assert "失敗" in msg or "failed" in msg.lower()

    def test_violation_log_limit(self):
        """Test that violation log respects MAX_LOG_SIZE limit"""
        detector = HelmetDetector.__new__(HelmetDetector)
        detector.violation_log = []
        
        # Add more than MAX_LOG_SIZE records
        for i in range(MAX_LOG_SIZE + 100):
            detector.add_violation({"id": i, "type": "test"})
        
        # Log should not exceed MAX_LOG_SIZE
        assert len(detector.violation_log) == MAX_LOG_SIZE
        
        # Oldest entries should be removed (FIFO)
        assert detector.violation_log[0]["id"] == 100
        assert detector.violation_log[-1]["id"] == MAX_LOG_SIZE + 99

    def test_add_violation_basic(self):
        """Test basic add_violation functionality"""
        detector = HelmetDetector.__new__(HelmetDetector)
        detector.violation_log = []
        
        record = {"type": "helmet_missing", "timestamp": "2024-01-01"}
        detector.add_violation(record)
        
        assert len(detector.violation_log) == 1
        assert detector.violation_log[0] == record


class TestConfigLoading:
    """Test cases for config loading functionality"""

    def test_config_load_default(self):
        """Test that default config is returned when file doesn't exist"""
        from main_gui import HelmetDetectionApp
        import tkinter as tk
        
        # Create a mock window (won't be shown)
        root = tk.Tk()
        root.withdraw()
        
        app = HelmetDetectionApp.__new__(HelmetDetectionApp)
        app.window = root
        
        default_config = app.load_config('non_existent_config_12345.json')
        
        assert default_config["confidence_threshold"] == 0.5
        assert default_config["helmet_class_id"] == 0
        assert default_config["person_class_id"] == 1
        assert default_config["violation_threshold"] == 5
        
        root.destroy()

    def test_config_load_with_valid_file(self, tmp_path):
        """Test config loading with a valid JSON file"""
        from main_gui import HelmetDetectionApp
        import tkinter as tk
        
        config_file = tmp_path / "test_config.json"
        config_content = {
            "confidence_threshold": 0.7,
            "custom_setting": "test_value"
        }
        config_file.write_text(json.dumps(config_content))
        
        root = tk.Tk()
        root.withdraw()
        
        app = HelmetDetectionApp.__new__(HelmetDetectionApp)
        app.window = root
        
        config = app.load_config(str(config_file))
        
        # Should merge with defaults
        assert config["confidence_threshold"] == 0.7
        assert config["helmet_class_id"] == 0  # default
        assert config["custom_setting"] == "test_value"
        
        root.destroy()

    def test_config_load_with_invalid_json(self, tmp_path):
        """Test config loading with corrupted JSON file"""
        from main_gui import HelmetDetectionApp
        import tkinter as tk
        
        config_file = tmp_path / "corrupt_config.json"
        config_file.write_text("{ invalid json content")
        
        root = tk.Tk()
        root.withdraw()
        
        app = HelmetDetectionApp.__new__(HelmetDetectionApp)
        app.window = root
        
        config = app.load_config(str(config_file))
        
        # Should fall back to defaults
        assert config["confidence_threshold"] == 0.5
        assert config["helmet_class_id"] == 0
        
        root.destroy()


class TestDetectorSafety:
    """Test safety features in detector"""

    def test_detect_without_model(self):
        """Test that detect handles None model gracefully"""
        detector = HelmetDetector.__new__(HelmetDetector)
        detector.model = None
        
        import numpy as np
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        result_frame, info = detector.detect(frame, ["helmet"])
        
        assert info.get('error') == '模型未載入'
        assert result_frame is not None

    def test_max_log_size_constant(self):
        """Test that MAX_LOG_SIZE is defined and positive"""
        assert MAX_LOG_SIZE > 0
        assert MAX_LOG_SIZE == 500
