"""
測試模組：PPE Detection System
包含基本邏輯驗證測試
"""

import unittest
import sys
import os

# 添加父目錄到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helmet_detector import HelmetDetector


class TestOverlap(unittest.TestCase):
    """測試空間重疊判定"""
    
    def setUp(self):
        self.detector = HelmetDetector()
    
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
        self.detector = HelmetDetector()
    
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
        self.detector = HelmetDetector()
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
    """測試模型載入"""
    
    def test_default_model_load(self):
        """測試預設模型載入"""
        detector = HelmetDetector()
        self.assertIsNotNone(detector.model)
        self.assertEqual(detector.model_path, 'yolov8n.pt')
    
    def test_model_classes(self):
        """測試模型類別獲取"""
        detector = HelmetDetector()
        classes = detector.get_model_classes()
        # yolov8n.pt 有 COCO 類別，包含 'person'
        self.assertIn('person', classes)
    
    def test_config_loading(self):
        """測試配置文件載入"""
        detector = HelmetDetector(config_path='config.json')
        self.assertIn('confidence_threshold', detector.config)
        self.assertIn('iou_threshold', detector.config)


class TestViolationLogging(unittest.TestCase):
    """測試違規記錄"""
    
    def setUp(self):
        self.detector = HelmetDetector()
    
    def test_violation_log_structure(self):
        """測試違規記錄結構"""
        self.detector.violation_log.append({
            'frame': 1,
            'person_id': 0,
            'missing_items': ['helmet'],
            'coords': (50.0, 50.0)
        })
        self.assertEqual(len(self.detector.violation_log), 1)
        self.assertEqual(self.detector.violation_log[0]['frame'], 1)
    
    def test_reset_clears_log(self):
        """測試 reset 清空記錄"""
        self.detector.violation_log.append({'frame': 1})
        self.detector.reset()
        self.assertEqual(len(self.detector.violation_log), 0)


if __name__ == '__main__':
    unittest.main()
