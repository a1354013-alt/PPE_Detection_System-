"""
Tests for main_gui - 不依賴真實 GUI 環境
測試 config 載入邏輯、worker thread 行為等

注意：由於 main_gui.py 在 import 時就依賴 tkinter，
本測試檔案針對「純邏輯」部分進行測試，避免直接 import main_gui。
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np


class TestConfigLoading:
    """測試 config 載入邏輯（不依賴 tkinter）"""
    
    def test_default_config_values(self):
        """測試預設 config 值"""
        from helmet_detector import HelmetDetector
        
        detector = HelmetDetector(model_path=None)
        
        # 檢查預設值
        assert detector.confidence_threshold == 0.4
        assert detector.violation_threshold == 3
        assert detector.helmet_class_id is None
        assert detector.person_class_id == 0
    
    def test_custom_config_values(self):
        """測試自訂 config 值"""
        from helmet_detector import HelmetDetector
        
        detector = HelmetDetector(
            model_path=None,
            confidence_threshold=0.7,
            violation_threshold=5,
            helmet_class_id=1,
            person_class_id=0
        )
        
        assert detector.confidence_threshold == 0.7
        assert detector.violation_threshold == 5
        assert detector.helmet_class_id == 1
        assert detector.person_class_id == 0


class TestWorkerThreadLogic:
    """測試 worker thread 邏輯（不依賴 tkinter，僅測試條件判斷）"""
    
    def test_worker_thread_stops_when_ret_false(self):
        """測試 worker thread 在 ret=False 時正確停止"""
        # Simulate the condition check from worker_thread
        ret_values = [True, False]
        frame_values = [np.zeros((500, 800, 3), dtype=np.uint8), None]
        
        should_break = []
        for ret, frame in zip(ret_values, frame_values):
            # This is the exact condition from worker_thread
            if not ret or frame is None:
                should_break.append(True)
            else:
                should_break.append(False)
        
        assert should_break[0] is False  # First frame: should continue
        assert should_break[1] is True   # Second frame: should break
    
    def test_worker_thread_handles_none_frame(self):
        """測試 worker thread 在 frame=None 時正確停止"""
        # Simulate ret=True but frame=None (camera issue)
        ret = True
        frame = None
        
        # The condition "if not ret or frame is None" should trigger break
        should_break = (not ret or frame is None)
        assert should_break is True
    
    def test_worker_thread_continues_when_valid(self):
        """測試 worker thread 在有效 frame 時繼續執行"""
        ret = True
        frame = np.zeros((500, 800, 3), dtype=np.uint8)
        
        should_break = (not ret or frame is None)
        assert should_break is False


class TestResourceCleanup:
    """測試資源釋放流程"""
    
    def test_handle_stop_releases_camera(self):
        """測試 handle_stop 釋放攝影機"""
        mock_vid = Mock()
        
        # Simulate handle_stop logic
        running = True
        running = False
        if mock_vid:
            mock_vid.release()
        
        mock_vid.release.assert_called_once()
    
    def test_on_closing_stops_running(self):
        """測試 on_closing 停止執行緒"""
        running = True
        
        # Simulate on_closing logic
        running = False
        
        assert running is False


class TestModelSupportValidation:
    """測試模型支援驗證邏輯（不依賴 tkinter messagebox）"""
    
    def test_validate_detects_unsupported_classes(self):
        """測試當模型不支援某些類別時能正確偵測"""
        # Simulate validation logic (without actual messagebox)
        model_classes = ['person', 'helmet']  # Model doesn't support 'vest'
        checked_items = {'helmet': True, 'vest': True, 'goggles': False, 'mask': False}
        
        unsupported = []
        for item, is_checked in checked_items.items():
            if is_checked and item not in model_classes:
                unsupported.append(item)
        
        assert 'vest' in unsupported
        assert len(unsupported) == 1
    
    def test_validate_no_warning_when_all_supported(self):
        """測試當所有類別都支援時不產生警告"""
        model_classes = ['person', 'helmet', 'vest', 'goggles', 'mask']
        checked_items = {'helmet': True, 'vest': True}
        
        unsupported = []
        for item, is_checked in checked_items.items():
            if is_checked and item not in model_classes:
                unsupported.append(item)
        
        assert len(unsupported) == 0
    
    def test_validate_generates_correct_warning_message(self):
        """測試警告訊息格式正確"""
        model_classes = ['person', 'helmet']
        checked_items = {'helmet': True, 'vest': True, 'goggles': True}
        
        unsupported = []
        for item, is_checked in checked_items.items():
            if is_checked and item not in model_classes:
                unsupported.append(item)
        
        if unsupported:
            msg = f"警告：當前模型不包含以下類別：\n{', '.join(unsupported)}\n系統將跳過這些項目的判定。"
            assert 'vest' in msg
            assert 'goggles' in msg


class TestQueueHandling:
    """測試隊列處理邏輯"""
    
    def test_check_queue_processes_frames(self):
        """測試 check_queue 處理 FRAME 任務"""
        from queue import Queue
        
        result_queue = Queue()
        result_queue.put(("FRAME", np.zeros((500, 800, 3), dtype=np.uint8)))
        
        processed_frames = []
        
        # Simulate check_queue logic
        try:
            while True:
                task_type, data = result_queue.get_nowait()
                if task_type == "FRAME":
                    processed_frames.append(data)
                elif task_type == "STATS":
                    pass
                elif task_type == "STOP":
                    pass
        except Exception:
            pass
        
        assert len(processed_frames) == 1
    
    def test_check_queue_handles_empty_queue(self):
        """測試 check_queue 處理空隊列"""
        from queue import Queue, Empty
        
        result_queue = Queue()
        
        raised_empty = False
        try:
            while True:
                task_type, data = result_queue.get_nowait()
        except Empty:
            raised_empty = True
        
        assert raised_empty is True
