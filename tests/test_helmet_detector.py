"""
Tests for HelmetDetector - 不依賴真實 YOLO 模型
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock


class TestHelmetDetectorWithoutYOLO:
    """測試 HelmetDetector 在沒有 ultralytics 環境下的行為"""
    
    def test_import_without_ultralytics(self):
        """確認模組可以在沒有 ultralytics 的情況下被匯入"""
        # 這測試確保 helmet_detector.py 不會在 import 時就硬依賴 ultralytics
        import helmet_detector
        assert hasattr(helmet_detector, 'HelmetDetector')
        assert hasattr(helmet_detector, '_get_yolo')
    
    def test_get_yolo_raises_import_error_when_not_installed(self):
        """當 ultralytics 未安裝時，_get_yolo 應拋出 ImportError"""
        from helmet_detector import _get_yolo
        
        # 模擬 ImportError
        with patch.dict('sys.modules', {'ultralytics': None}):
            # 重新載入模組以清除快取
            import sys
            if 'helmet_detector' in sys.modules:
                del sys.modules['helmet_detector']
            
            import helmet_detector
            # 重置 _YOLO 為 None 以強制重新匯入
            helmet_detector._YOLO = None
            
            with pytest.raises(ImportError) as exc_info:
                helmet_detector._get_yolo()
            
            assert "ultralytics is required" in str(exc_info.value)


class TestHelmetDetectorWithMock:
    """使用 mock 測試 HelmetDetector 的邏輯"""
    
    @pytest.fixture
    def mock_yolo(self):
        """建立 mock YOLO 模型"""
        mock_model = Mock()
        mock_model.names = {0: 'person', 1: 'hardhat', 2: 'safety_vest'}
        
        # 模擬检测结果
        mock_box = Mock()
        mock_box.cls = [0]  # person
        mock_box.xyxy = [[100, 100, 200, 300]]
        mock_box.conf = [0.9]
        
        mock_result = Mock()
        mock_result.boxes = [mock_box]
        mock_result.names = {0: 'person', 1: 'hardhat', 2: 'safety_vest'}
        mock_result.plot.return_value = np.zeros((500, 800, 3), dtype=np.uint8)
        
        mock_model.return_value = [mock_result]
        return mock_model
    
    @patch('helmet_detector._get_yolo')
    def test_detector_init_with_mock(self, mock_get_yolo, mock_yolo):
        """測試偵測器初始化"""
        mock_get_yolo.return_value = lambda path: mock_yolo
        
        from helmet_detector import HelmetDetector
        detector = HelmetDetector(model_path='test.pt', conf=0.5, iou=0.5)
        
        assert detector.model is not None
        assert detector.conf == 0.5
        assert detector.iou == 0.5
    
    @patch('helmet_detector._get_yolo')
    def test_config_parameters_are_used(self, mock_get_yolo, mock_yolo):
        """測試 config 參數是否被正確儲存和使用"""
        mock_get_yolo.return_value = lambda path: mock_yolo
        
        from helmet_detector import HelmetDetector
        detector = HelmetDetector(
            model_path='test.pt',
            confidence_threshold=0.6,
            violation_threshold=5,
            helmet_class_id=1,
            person_class_id=0
        )
        
        assert detector.confidence_threshold == 0.6
        assert detector.violation_threshold == 5
        assert detector.helmet_class_id == 1
        assert detector.person_class_id == 0
        # violation_buffer 的 maxlen 應該等於 violation_threshold (如果 > 5)
        assert detector.violation_buffer.maxlen == 5  # max(5, 5) = 5
    
    @patch('helmet_detector._get_yolo')
    def test_detect_returns_error_when_model_not_loaded(self, mock_get_yolo):
        """測試當模型未載入時的錯誤處理"""
        from helmet_detector import HelmetDetector
        
        # 建立不載入模型的偵測器
        detector = HelmetDetector(model_path=None)
        
        frame = np.zeros((500, 800, 3), dtype=np.uint8)
        result_frame, info = detector.detect(frame, ['helmet'])
        
        assert info == {'error': '模型未載入'}
        # 返回的 frame 應該是原始的
        assert np.array_equal(result_frame, frame)
    
    @patch('helmet_detector._get_yolo')
    def test_violation_threshold_affects_detection(self, mock_get_yolo, mock_yolo):
        """測試 violation_threshold 影響違規判定"""
        mock_get_yolo.return_value = lambda path: mock_yolo
        
        from helmet_detector import HelmetDetector
        detector = HelmetDetector(model_path='test.pt', violation_threshold=3)
        
        # violation_buffer 的 maxlen 應該根據 violation_threshold 設定
        assert detector.violation_buffer.maxlen == 5  # max(5, 3) = 5
        
        # 測試 threshold 為較大的值
        detector2 = HelmetDetector(model_path='test.pt', violation_threshold=7)
        assert detector2.violation_buffer.maxlen == 7  # max(5, 7) = 7
    
    def test_load_model_failure_returns_false(self):
        """測試模型載入失敗時的處理"""
        from helmet_detector import HelmetDetector
        
        detector = HelmetDetector(model_path=None)
        
        # 模擬 load_model 失敗（因為沒有 ultralytics）
        success, msg = detector.load_model('nonexistent.pt')
        
        assert success is False
        assert "模型載入失敗" in msg
    
    @patch('helmet_detector._get_yolo')
    def test_get_model_classes_returns_mapped_names(self, mock_get_yolo, mock_yolo):
        """測試 get_model_classes 回射映射後的名稱"""
        mock_get_yolo.return_value = lambda path: mock_yolo
        
        from helmet_detector import HelmetDetector
        detector = HelmetDetector(model_path='test.pt')
        
        classes = detector.get_model_classes()
        
        # 應該包含映射後的名稱
        assert 'helmet' in classes  # hardhat -> helmet
        assert 'vest' in classes   # safety_vest -> vest
        assert 'person' in classes


class TestIsOverlapping:
    """測試空間關聯判定邏輯"""
    
    @patch('helmet_detector._get_yolo')
    def test_overlapping_when_item_in_head_zone(self, mock_get_yolo):
        """測試物品在頭部區域時判定為重疊"""
        from helmet_detector import HelmetDetector
        detector = HelmetDetector(model_path=None)
        
        # Person box: [x1, y1, x2, y2]
        person_box = [100, 100, 200, 400]  # height = 300
        # Item box in head zone (top 30% of person)
        item_box = [120, 80, 180, 150]  # center y = 115, within head zone
        
        result = detector.is_overlapping(person_box, item_box)
        assert result is True
    
    @patch('helmet_detector._get_yolo')
    def test_not_overlapping_when_item_outside_width(self, mock_get_yolo):
        """測試物品寬度超出人體時判定為不重疊"""
        from helmet_detector import HelmetDetector
        detector = HelmetDetector(model_path=None)
        
        person_box = [100, 100, 200, 400]
        # Item completely outside person's width
        item_box = [300, 80, 400, 150]
        
        result = detector.is_overlapping(person_box, item_box)
        assert result is False
    
    @patch('helmet_detector._get_yolo')
    def test_not_overlapping_when_item_too_low(self, mock_get_yolo):
        """測試物品位置太低時判定為不重疊"""
        from helmet_detector import HelmetDetector
        detector = HelmetDetector(model_path=None)
        
        person_box = [100, 100, 200, 400]
        # Item at bottom of person
        item_box = [120, 350, 180, 390]
        
        result = detector.is_overlapping(person_box, item_box)
        assert result is False


class TestHeatmapGeneration:
    """測試熱力圖生成"""
    
    @patch('helmet_detector._get_yolo')
    def test_generate_heatmap(self, mock_get_yolo):
        """測試熱力圖生成"""
        from helmet_detector import HelmetDetector
        detector = HelmetDetector(model_path=None)
        
        # 添加一些違規座標
        detector.violation_coords = [(100, 100), (200, 200), (300, 300)]
        
        shape = (500, 800, 3)
        heatmap = detector.generate_heatmap(shape)
        
        assert heatmap.shape == (500, 800, 3)
        assert heatmap.dtype == np.uint8
