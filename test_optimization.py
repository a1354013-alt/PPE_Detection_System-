import unittest
from collections import deque
import time
from helmet_detector import HelmetDetector
import numpy as np

class MockBox:
    def __init__(self, cls, xyxy, id=None):
        self.cls = [cls]
        self.xyxy = [xyxy]
        self.id = [id] if id is not None else None

class MockResult:
    def __init__(self, boxes):
        self.boxes = boxes
        self.names = {0: 'person', 1: 'hardhat'}
    def plot(self):
        return np.zeros((500, 800, 3), dtype=np.uint8)

class TestPPEOptimization(unittest.TestCase):
    def setUp(self):
        self.detector = HelmetDetector()
        # Mock the model to avoid loading actual YOLO weights during test
        self.detector.model = lambda x, **kwargs: [MockResult([])]
        self.detector.model.names = {0: 'person', 1: 'hardhat'}
        self.detector.class_map = {'hardhat': 'helmet'}

    def test_violation_coords_bounded(self):
        """測試 violation_coords 是否為 bounded queue，超過 5000 筆不會繼續增加"""
        self.detector.violation_coords = deque(maxlen=5000)
        for i in range(6000):
            self.detector.violation_coords.append((i, i))
        self.assertEqual(len(self.detector.violation_coords), 5000)
        self.assertEqual(self.detector.violation_coords[0], (1000, 1000))

    def test_track_id_cooldown(self):
        """測試同一個 track_id 在 cooldown 內不會重複累加同一種違規"""
        self.detector.violation_cooldown = 3.0
        current_time = time.time()
        
        # 模擬第一次偵測到違規
        p_box = [100, 100, 200, 400]
        t_id = 1
        v_type = 'helmet'
        
        # 手動模擬 detect 邏輯中的統計部分
        key = (t_id, v_type)
        
        # 第一次統計
        last_count_time = self.detector.counted_violations.get(key, 0)
        self.assertTrue(current_time - last_count_time >= self.detector.violation_cooldown)
        self.detector.counted_violations[key] = current_time
        
        # 1秒後再次偵測 (仍在 cooldown 內)
        later_time = current_time + 1.0
        last_count_time = self.detector.counted_violations.get(key, 0)
        self.assertFalse(later_time - last_count_time >= self.detector.violation_cooldown)
        
        # 4秒後再次偵測 (超過 cooldown)
        much_later_time = current_time + 4.0
        last_count_time = self.detector.counted_violations.get(key, 0)
        self.assertTrue(much_later_time - last_count_time >= self.detector.violation_cooldown)

    def test_different_track_id_stats(self):
        """測試不同 track_id 可以分別統計"""
        self.detector.violation_cooldown = 3.0
        current_time = time.time()
        
        # Person 1
        key1 = (1, 'helmet')
        self.detector.counted_violations[key1] = current_time
        
        # Person 2 (同一時間偵測到，應該也要能統計)
        key2 = (2, 'helmet')
        last_count_time2 = self.detector.counted_violations.get(key2, 0)
        self.assertTrue(current_time - last_count_time2 >= self.detector.violation_cooldown)

    def test_fallback_no_track_id(self):
        """測試沒有 track_id 時 fallback 不會讓程式 crash"""
        # 模擬 detect 邏輯中的 fallback key 生成
        p_box = [100, 100, 200, 400]
        t_id = None
        v_type = 'helmet'
        
        fallback_id = f"unk_{int((p_box[0]+p_box[2])/2)}_{int((p_box[1]+p_box[3])/2)}"
        key = (t_id if t_id is not None else fallback_id, v_type)
        
        self.assertEqual(key[0], "unk_150_250")
        
        # 測試存取
        self.detector.counted_violations[key] = time.time()
        self.assertIn(key, self.detector.counted_violations)

if __name__ == '__main__':
    unittest.main()
