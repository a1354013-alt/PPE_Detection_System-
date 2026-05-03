"""
測試模組：PPE Detection System
包含基本邏輯驗證測試

注意：Unit test 不可以真的下載或載入 YOLO 模型
使用 mock / fake model 避免依賴外部模型
"""

import unittest
import sys
import os
import types
import time
from unittest.mock import Mock, patch, MagicMock

# Create fake ultralytics module to avoid ModuleNotFoundError
fake_ultralytics = types.ModuleType("ultralytics")

class FakeYOLO:
    def __init__(self, *args, **kwargs):
        self.names = {0: 'person', 1: 'hardhat'}

    def track(self, *args, **kwargs):
        return []

fake_ultralytics.YOLO = FakeYOLO
sys.modules["ultralytics"] = fake_ultralytics

# 添加父目錄到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_mock_detector():
    """建立一個 mock 的 HelmetDetector，不載入真實模型"""
    # 先 patch ultralytics.YOLO 避免載入真實模型
    with patch('ultralytics.YOLO') as mock_yolo:
        from helmet_detector import HelmetDetector
        
        mock_model = Mock()
        mock_model.names = {0: 'person', 1: 'hardhat'}
        mock_yolo.return_value = mock_model
        
        detector = HelmetDetector(demo_mode=True)
        detector.model = None  # 確保不使用真實模型
        return detector


class TestInitialization(unittest.TestCase):
    """測試初始化順序與屬性"""
    
    @patch('ultralytics.YOLO')
    def test_init_order_class_map_exists(self, mock_yolo):
        """測試 Bug 1: 確認 HelmetDetector 初始化時 class_map 已存在，且 _check_model_capability 不會失敗"""
        from helmet_detector import HelmetDetector
        
        # 模擬 YOLO 模型
        mock_model = Mock()
        mock_model.names = {0: 'person', 1: 'hardhat'}
        mock_yolo.return_value = mock_model
        
        # 建立實例，這會觸發 load_model -> _check_model_capability
        # 如果 class_map 尚未初始化，_check_model_capability 會拋出 AttributeError
        try:
            detector = HelmetDetector(demo_mode=False)
            self.assertTrue(hasattr(detector, 'class_map'), "class_map 應該在初始化後存在")
            self.assertIn('hardhat', detector.class_map, "class_map 應該包含預定義的映射")
        except AttributeError as e:
            self.fail(f"初始化失敗，可能是 class_map 尚未建立就呼叫了使用它的方法: {e}")


class TestOverlap(unittest.TestCase):
    """測試空間重疊判定"""
    
    def setUp(self):
        self.detector = create_mock_detector()
    
    def test_item_in_zone(self):
        """測試物品在區域內的情況"""
        zone = (0, 0, 100, 50)  # x1, y1, x2, y2
        item = (40, 10, 60, 30)  # 中心點 (50, 20) 應該在 zone 內
        result = self.detector.is_overlapping_with_zone(item, zone)
        self.assertTrue(result)
    
    def test_item_outside_zone(self):
        """測試物品在區域外的情況"""
        zone = (0, 0, 100, 50)
        item = (150, 100, 170, 120)  # 完全在外面
        result = self.detector.is_overlapping_with_zone(item, zone)
        self.assertFalse(result)
    
    def test_item_partial_overlap(self):
        """測試部分重疊的情況"""
        zone = (0, 0, 100, 50)
        item = (90, 20, 110, 40)  # 中心點 (100, 30) 在邊緣
        result = self.detector.is_overlapping_with_zone(item, zone)
        # 中心點在邊界上，應該算在內
        self.assertTrue(result)


class TestZoneGeneration(unittest.TestCase):
    """測試人體區域劃分"""
    
    def setUp(self):
        self.detector = create_mock_detector()
    
    def test_zone_boundaries(self):
        """測試區域邊界計算正確性"""
        person_box = (0, 0, 100, 200)  # 寬 100, 高 200
        zones = self.detector.get_person_zones(person_box)
        
        # head: 上 25% = 50px
        self.assertEqual(zones["head"], (0, 0, 100, 50))
        
        # face_upper: 上 40% = 80px
        self.assertEqual(zones["face_upper"], (0, 0, 100, 80))
        
        # face_lower: 40% ~ 60% = 80px ~ 120px
        self.assertEqual(zones["face_lower"], (0, 80, 100, 120))
        
        # torso: 30% ~ 80% = 60px ~ 160px
        self.assertEqual(zones["torso"], (0, 60, 100, 160))
    
    def test_zone_non_zero_origin(self):
        """測試非零起點的區域計算"""
        person_box = (50, 100, 150, 300)  # 起點 (50, 100), 寬 100, 高 200
        zones = self.detector.get_person_zones(person_box)
        
        # head: y 從 100 開始，高度 50
        self.assertEqual(zones["head"][0], 50)
        self.assertEqual(zones["head"][1], 100)
        self.assertEqual(zones["head"][3], 150)  # 100 + 50


class TestTemporalBuffer(unittest.TestCase):
    """測試時間平滑 buffer"""
    
    def setUp(self):
        self.detector = create_mock_detector()
        self.detector.temporal_frames = 3
    
    def test_buffer_initialization(self):
        """測試 buffer 初始化"""
        self.assertEqual(len(self.detector.person_buffers), 0)
    
    def test_buffer_clear_on_reset(self):
        """測試 reset 時 buffer 清空"""
        self.detector.person_buffers[0] = ["test"]
        self.detector.reset()
        self.assertEqual(len(self.detector.person_buffers), 0)


class TestModelLoading(unittest.TestCase):
    """測試模型載入（使用 mock）"""
    
    @patch('ultralytics.YOLO')
    def test_default_model_load(self, mock_yolo):
        """測試預設模型載入（mock）"""
        from helmet_detector import HelmetDetector
        
        mock_model = Mock()
        mock_model.names = {0: 'person'}
        mock_yolo.return_value = mock_model
        
        detector = HelmetDetector()
        self.assertIsNotNone(detector.model)
        self.assertEqual(detector.model_path, 'yolov8n.pt')
    
    @patch('ultralytics.YOLO')
    def test_model_classes(self, mock_yolo):
        """測試模型類別獲取（mock）"""
        from helmet_detector import HelmetDetector
        
        mock_model = Mock()
        mock_model.names = {0: 'person', 1: 'hardhat'}
        mock_yolo.return_value = mock_model
        
        detector = HelmetDetector()
        classes = detector.get_model_classes()
        # 應該包含 helmet (mapped from hardhat)
        self.assertIn('helmet', classes)
    
    def test_config_loading(self):
        """測試配置文件載入"""
        detector = create_mock_detector()
        self.assertIn('confidence_threshold', detector.config)
        self.assertIn('iou_threshold', detector.config)


class TestViolationLogging(unittest.TestCase):
    """測試違規記錄"""
    
    def setUp(self):
        self.detector = create_mock_detector()
    
    def test_violation_log_structure(self):
        """測試違規記錄結構"""
        self.detector.violation_log.append({
            'frame': 1,
            'track_id': 'demo_0',
            'missing_items': ['helmet'],
            'center_x': 50.0,
            'center_y': 50.0
        })
        self.assertEqual(len(self.detector.violation_log), 1)
        self.assertEqual(self.detector.violation_log[0]['frame'], 1)
    
    def test_reset_clears_log(self):
        """測試 reset 清空記錄"""
        self.detector.violation_log.append({'frame': 1})
        self.detector.reset()
        self.assertEqual(len(self.detector.violation_log), 0)


class TestDemoMode(unittest.TestCase):
    """測試 Demo Mode"""
    
    def setUp(self):
        self.detector = create_mock_detector()
        self.detector.demo_mode = True
    
    def test_demo_mode_creates_events(self):
        """測試 Demo Mode 能產生事件"""
        import numpy as np
        frame = np.zeros((500, 800, 3), dtype=np.uint8)
        target_items = ['helmet', 'vest']
        
        # 需要多幀來滿足 temporal smoothing
        for i in range(5):
            result_frame, info = self.detector.detect(
                frame, target_items, source_name="test", frame_number=i
            )
        
        # 應該有產生事件
        self.assertIn('new_events', info)
        self.assertIn('is_demo', info)
        self.assertTrue(info['is_demo'])


class TestCooldownLogic(unittest.TestCase):
    """測試冷卻機制"""
    
    def setUp(self):
        self.detector = create_mock_detector()
        self.detector.violation_cooldown = 2.0
    
    def test_independent_track_ids(self):
        """測試不同 track_id 獨立冷卻"""
        curr_time = time.time()
        
        # Track 1 剛統計過某種違規
        event_key_1 = ('track_1', ('helmet',))
        self.detector.counted_violations[event_key_1] = curr_time
        
        # Track 2 應該仍可統計
        should_report, keys = self.detector._should_report_event(
            'track_2', ['helmet'], curr_time
        )
        self.assertTrue(should_report)
        self.assertEqual(len(keys), 1)
        self.assertEqual(keys[0], ('track_2', ('helmet',)))
    
    def test_same_track_different_violations(self):
        """測試 Bug 4: 同一個人如果缺失組合改變，視為新事件"""
        curr_time = time.time()
        
        # Track 1 剛統計過 helmet 違規
        event_key_1 = ('track_1', ('helmet',))
        self.detector.counted_violations[event_key_1] = curr_time
        
        # 同一個人現在缺失 helmet + vest，應該視為新事件
        should_report, keys = self.detector._should_report_event(
            'track_1', ['helmet', 'vest'], curr_time
        )
        self.assertTrue(should_report, "缺失組合改變應視為新事件")
        self.assertEqual(keys[0], ('track_1', ('helmet', 'vest')))
    
    def test_same_track_same_violation_in_cooldown(self):
        """測試同一 track_id 在冷卻期內不會重複報告相同組合"""
        curr_time = time.time()
        
        # Track 1 剛統計過 helmet 違規
        event_key_1 = ('track_1', ('helmet',))
        self.detector.counted_violations[event_key_1] = curr_time
        
        # 0.5 秒後（仍在冷卻期內）
        should_report, keys = self.detector._should_report_event(
            'track_1', ['helmet'], curr_time + 0.5
        )
        self.assertFalse(should_report)
    
    def test_cooldown_expired(self):
        """測試冷卻期過後可以再次報告"""
        curr_time = time.time()
        
        # Track 1 剛統計過
        event_key_1 = ('track_1', ('helmet',))
        self.detector.counted_violations[event_key_1] = curr_time
        
        # 3 秒後（超過冷卻期）
        should_report, keys = self.detector._should_report_event(
            'track_1', ['helmet'], curr_time + 3.0
        )
        self.assertTrue(should_report)


class TestReportExport(unittest.TestCase):
    """測試 Bug 2: 報告輸出路徑"""
    
    def setUp(self):
        from event_logger import EventLogger, ViolationEvent
        self.logger = EventLogger()
        self.event = ViolationEvent(
            timestamp="2026-05-03 12:00:00",
            source="test",
            track_id="1",
            person_count=1,
            missing_items="helmet",
            screenshot_path="test.jpg",
            confidence=0.9,
            bbox="[0,0,100,100]"
        )
        self.logger.add_event(self.event)
        
        # 確保 reports 目錄存在
        if not os.path.exists("reports"):
            os.makedirs("reports")

    def test_csv_export_path(self):
        """測試 CSV 匯出到指定路徑"""
        filepath = os.path.join("reports", "test_report.csv")
        if os.path.exists(filepath):
            os.remove(filepath)
            
        path = self.logger.export_csv(filepath)
        self.assertTrue(os.path.exists(path))
        self.assertTrue(path.startswith("reports"))
        os.remove(path)


if __name__ == '__main__':
    unittest.main()
