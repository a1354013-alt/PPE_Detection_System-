"""
測試 HelmetDetector 類別
不依賴真實 YOLO 模型，使用 mock 進行測試
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from collections import deque
import numpy as np
import sys


# 在導入 helmet_detector 之前先 mock cv2
mock_cv2 = MagicMock()
mock_cv2.FONT_HERSHEY_SIMPLEX = 1
mock_cv2.COLORMAP_JET = 1
mock_cv2.GaussianBlur = lambda *args: None
mock_cv2.normalize = lambda *args, **kwargs: None
mock_cv2.applyColorMap = lambda *args: None
mock_cv2.circle = lambda *args: None
mock_cv2.putText = lambda *args: None
sys.modules['cv2'] = mock_cv2

# Mock numpy
sys.modules['numpy'] = MagicMock()


class TestHelmetDetectorLazyImport:
    """測試 lazy import 功能"""
    
    def test_get_yolo_import_error(self):
        """測試當 ultralytics 未安裝時的 ImportError"""
        # 清除已載入的模組
        for mod in list(sys.modules.keys()):
            if 'helmet' in mod or 'ultralytics' in mod:
                del sys.modules[mod]
        
        with patch.dict('sys.modules', {'ultralytics': None}):
            from helmet_detector import _get_yolo
            
            with pytest.raises(ImportError) as exc_info:
                _get_yolo()
            
            assert "ultralytics" in str(exc_info.value)
    
    def test_get_yolo_success(self):
        """測試當 ultralytics 已安裝時成功返回 YOLO"""
        mock_yolo = Mock()
        
        # 清除已載入的模組
        for mod in list(sys.modules.keys()):
            if 'helmet' in mod or 'ultralytics' in mod:
                del sys.modules[mod]
        
        ultralytics_mock = MagicMock()
        ultralytics_mock.YOLO = mock_yolo
        with patch.dict('sys.modules', {'ultralytics': ultralytics_mock}):
            from helmet_detector import _get_yolo
            result = _get_yolo()
            
            assert result == mock_yolo


class TestHelmetDetectorInitialization:
    """測試 HelmetDetector 初始化"""
    
    @patch('helmet_detector._get_yolo')
    def test_init_default_params(self, mock_get_yolo):
        """測試預設參數初始化"""
        mock_model = Mock()
        mock_get_yolo.return_value = Mock(return_value=mock_model)
        
        # 清除已載入的模組
        for mod in list(sys.modules.keys()):
            if 'helmet' in mod:
                del sys.modules[mod]
        
        from helmet_detector import HelmetDetector
        detector = HelmetDetector()
        
        assert detector.confidence_threshold == 0.4
        assert detector.iou_threshold == 0.45
        assert detector.violation_threshold == 3
        assert detector.helmet_class_id == 0
        assert detector.person_class_id == 0
        assert detector.violation_buffer.maxlen == 3
    
    @patch('helmet_detector._get_yolo')
    def test_init_custom_params(self, mock_get_yolo):
        """測試自訂參數初始化"""
        mock_model = Mock()
        mock_get_yolo.return_value = Mock(return_value=mock_model)
        
        # 清除已載入的模組
        for mod in list(sys.modules.keys()):
            if 'helmet' in mod:
                del sys.modules[mod]
        
        from helmet_detector import HelmetDetector
        detector = HelmetDetector(
            model_path='custom.pt',
            confidence_threshold=0.6,
            iou_threshold=0.5,
            violation_threshold=5,
            helmet_class_id=1,
            person_class_id=2
        )
        
        assert detector.model_path == 'custom.pt'
        assert detector.confidence_threshold == 0.6
        assert detector.iou_threshold == 0.5
        assert detector.violation_threshold == 5
        assert detector.helmet_class_id == 1
        assert detector.person_class_id == 2
        assert detector.violation_buffer.maxlen == 5


class TestHelmetDetectorLoadModel:
    """測試模型載入功能"""
    
    @patch('helmet_detector._get_yolo')
    def test_load_model_success(self, mock_get_yolo):
        """測試成功載入模型"""
        mock_model = Mock()
        mock_yolo_class = Mock(return_value=mock_model)
        mock_get_yolo.return_value = mock_yolo_class
        
        # 注意：不要清除 helmet_detector 模組，因為 patch 需要它已載入
        from helmet_detector import HelmetDetector
        detector = HelmetDetector.__new__(HelmetDetector)
        detector.model = None
        detector.model_path = None
        
        success, msg = detector.load_model('test.pt')
        
        assert success is True
        assert "成功載入模型" in msg
        assert detector.model == mock_model
    
    @patch('helmet_detector._get_yolo')
    def test_load_model_import_error(self, mock_get_yolo):
        """測試 ImportError 處理"""
        mock_get_yolo.side_effect = ImportError("ultralytics not installed")
        
        # 清除已載入的模組
        for mod in list(sys.modules.keys()):
            if 'helmet' in mod:
                del sys.modules[mod]
        
        from helmet_detector import HelmetDetector
        detector = HelmetDetector.__new__(HelmetDetector)
        detector.model = None
        detector.model_path = None
        
        success, msg = detector.load_model('test.pt')
        
        assert success is False
        assert "模型載入失敗" in msg
    
    @patch('helmet_detector._get_yolo')
    def test_load_model_exception(self, mock_get_yolo):
        """測試一般 Exception 處理"""
        mock_get_yolo.side_effect = Exception("File not found")
        
        # 清除已載入的模組
        for mod in list(sys.modules.keys()):
            if 'helmet' in mod:
                del sys.modules[mod]
        
        from helmet_detector import HelmetDetector
        detector = HelmetDetector.__new__(HelmetDetector)
        detector.model = None
        detector.model_path = None
        
        success, msg = detector.load_model('test.pt')
        
        assert success is False
        assert "模型載入失敗" in msg


class TestViolationThreshold:
    """測試 violation_threshold 設定"""
    
    @patch('helmet_detector._get_yolo')
    def test_violation_buffer_uses_config(self, mock_get_yolo):
        """測試 buffer 大小使用 violation_threshold 設定"""
        mock_model = Mock()
        mock_get_yolo.return_value = Mock(return_value=mock_model)
        
        # 清除已載入的模組
        for mod in list(sys.modules.keys()):
            if 'helmet' in mod:
                del sys.modules[mod]
        
        from helmet_detector import HelmetDetector
        
        # 測試不同 threshold 值
        for threshold in [2, 3, 5, 7]:
            detector = HelmetDetector(violation_threshold=threshold)
            assert detector.violation_buffer.maxlen == threshold


class TestIsOverlapping:
    """測試空間關聯判定"""
    
    @patch('helmet_detector._get_yolo')
    def test_overlapping_center_in_head_zone(self, mock_get_yolo):
        """測試裝備中心點在頭部區域"""
        mock_model = Mock()
        mock_get_yolo.return_value = Mock(return_value=mock_model)
        
        # 清除已載入的模組
        for mod in list(sys.modules.keys()):
            if 'helmet' in mod:
                del sys.modules[mod]
        
        from helmet_detector import HelmetDetector
        detector = HelmetDetector()
        
        # 人：[100, 100, 200, 400] (寬 100, 高 300)
        # 頭部區域：y = 100 ~ 190 (top 30%)
        person_box = [100, 100, 200, 400]
        
        # 安全帽中心點在人寬度內且在頭部區域
        helmet_box = [140, 110, 160, 140]  # 中心 (150, 125)
        
        result = detector.is_overlapping(person_box, helmet_box)
        assert result is True
    
    @patch('helmet_detector._get_yolo')
    def test_not_overlapping_outside_width(self, mock_get_yolo):
        """測試裝備中心點不在人寬度內"""
        mock_model = Mock()
        mock_get_yolo.return_value = Mock(return_value=mock_model)
        
        # 清除已載入的模組
        for mod in list(sys.modules.keys()):
            if 'helmet' in mod:
                del sys.modules[mod]
        
        from helmet_detector import HelmetDetector
        detector = HelmetDetector()
        
        person_box = [100, 100, 200, 400]
        # 裝備在人右側外面
        item_box = [250, 110, 280, 140]  # 中心 (265, 125)
        
        result = detector.is_overlapping(person_box, item_box)
        assert result is False
    
    @patch('helmet_detector._get_yolo')
    def test_not_overlapping_below_head_zone(self, mock_get_yolo):
        """測試裝備在人腰部以下"""
        mock_model = Mock()
        mock_get_yolo.return_value = Mock(return_value=mock_model)
        
        # 清除已載入的模組
        for mod in list(sys.modules.keys()):
            if 'helmet' in mod:
                del sys.modules[mod]
        
        from helmet_detector import HelmetDetector
        detector = HelmetDetector()
        
        person_box = [100, 100, 200, 400]
        # 裝備在人腰部
        item_box = [140, 250, 160, 280]  # 中心 (150, 265)
        
        result = detector.is_overlapping(person_box, item_box)
        assert result is False


class TestClassMap:
    """測試類別映射"""
    
    @patch('helmet_detector._get_yolo')
    def test_class_map_exists(self, mock_get_yolo):
        """測試 class_map 存在且包含預期映射"""
        mock_model = Mock()
        mock_get_yolo.return_value = Mock(return_value=mock_model)
        
        # 清除已載入的模組
        for mod in list(sys.modules.keys()):
            if 'helmet' in mod:
                del sys.modules[mod]
        
        from helmet_detector import HelmetDetector
        detector = HelmetDetector()
        
        assert hasattr(detector, 'class_map')
        assert isinstance(detector.class_map, dict)
        assert 'hardhat' in detector.class_map
        assert 'safety_vest' in detector.class_map
