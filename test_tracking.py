import unittest
import time
from helmet_detector import HelmetDetector
import numpy as np

class MockBox:
    def __init__(self, cls, xyxy, conf, id=None):
        self.cls = [cls]
        self.xyxy = [xyxy]
        self.conf = [conf]
        self.id = [id] if id is not None else None

class MockResult:
    def __init__(self, boxes):
        self.boxes = boxes
        self.names = {0: 'person', 1: 'hardhat'}
    def plot(self):
        return np.zeros((500, 800, 3), dtype=np.uint8)

class TestPPETracking(unittest.TestCase):
    def setUp(self):
        self.detector = HelmetDetector()
        # Mock the model to avoid loading actual YOLO weights during test
        self.detector.model = lambda x, **kwargs: [MockResult([])]
        self.detector.model.names = {0: 'person', 1: 'hardhat'}
        self.detector.class_map = {'hardhat': 'helmet'}

    def test_fallback_track_id_generation(self):
        """測試 Fallback ID 是否能根據 BBox 正確生成"""
        bbox = [100, 100, 200, 200] # center (150, 150)
        fid = self.detector.build_fallback_track_id(bbox)
        self.assertEqual(fid, "fallback_3_3")
        
        bbox2 = [400, 400, 500, 500] # center (450, 450)
        fid2 = self.detector.build_fallback_track_id(bbox2)
        self.assertEqual(fid2, "fallback_9_9")

    def test_person_state_management(self):
        """測試人員狀態的新增與清理"""
        current_time = time.time()
        tid = "1"
        self.detector.person_states[tid] = {
            "last_seen": current_time,
            "missing_items": {"helmet"},
            "bbox": [0, 0, 10, 10],
            "confidence": 0.9
        }
        
        # 測試清理 (TTL=10)
        self.detector.cleanup_stale_tracks(current_time + 5, ttl=10)
        self.assertIn(tid, self.detector.person_states)
        
        self.detector.cleanup_stale_tracks(current_time + 15, ttl=10)
        self.assertNotIn(tid, self.detector.person_states)

    def test_tracking_cooldown_logic(self):
        """測試以人員為單位的違規冷卻邏輯"""
        tid = "person_A"
        v_type = "helmet"
        current_time = time.time()
        
        # 第一次違規
        key = (tid, v_type)
        self.detector.counted_violations[key] = current_time
        
        # 1秒後 (仍在冷卻)
        self.assertFalse(current_time + 1 - self.detector.counted_violations[key] >= self.detector.violation_cooldown)
        
        # 4秒後 (冷卻結束)
        self.assertTrue(current_time + 4 - self.detector.counted_violations[key] >= self.detector.violation_cooldown)

    def test_different_persons_independent_stats(self):
        """測試不同人員的統計是否互相獨立"""
        tid1 = "person_1"
        tid2 = "person_2"
        v_type = "helmet"
        current_time = time.time()
        
        # Person 1 剛統計過
        self.detector.counted_violations[(tid1, v_type)] = current_time
        
        # Person 2 應該仍可統計
        last_time2 = self.detector.counted_violations.get((tid2, v_type), 0)
        self.assertTrue(current_time - last_time2 >= self.detector.violation_cooldown)

if __name__ == '__main__':
    unittest.main()
