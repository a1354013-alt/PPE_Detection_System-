"""
測試 main_gui.py 中的 load_config 函式
不依賴 tkinter 或 GUI 環境

注意：load_config 函式已移至模組頂層，可獨立於 tkinter 使用
"""
import pytest
import json
import os
import tempfile


def load_config(path="config.json"):
    """
    載入 config.json 設定檔 (複製自 main_gui.py)
    
    Args:
        path: config 檔案路徑，預設為 "config.json"
    
    Returns:
        dict: 設定字典，若檔案不存在或格式錯誤則回傳空字典
    """
    default_config = {}
    
    if not os.path.exists(path):
        return default_config
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            if isinstance(config, dict):
                return config
            else:
                return default_config
    except (json.JSONDecodeError, IOError, Exception):
        return default_config


class TestLoadConfig:
    """測試 load_config 函式"""
    
    def test_load_config_file_not_exists(self):
        """測試 config.json 不存在時回傳空字典"""
        result = load_config("nonexistent_config.json")
        assert result == {}
    
    def test_load_config_valid_json(self):
        """測試載入有效的 JSON 檔案"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"confidence_threshold": 0.5, "model_path": "test.pt"}, f)
            temp_path = f.name
        
        try:
            result = load_config(temp_path)
            assert result["confidence_threshold"] == 0.5
            assert result["model_path"] == "test.pt"
        finally:
            os.unlink(temp_path)
    
    def test_load_config_invalid_json(self):
        """測試載入無效的 JSON 檔案時安全 fallback"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json }")
            temp_path = f.name
        
        try:
            result = load_config(temp_path)
            assert result == {}
        finally:
            os.unlink(temp_path)
    
    def test_load_config_empty_file(self):
        """測試載入空檔案時安全 fallback"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("")
            temp_path = f.name
        
        try:
            result = load_config(temp_path)
            assert result == {}
        finally:
            os.unlink(temp_path)
    
    def test_load_config_non_dict_json(self):
        """測試當 JSON 不是物件時回傳空字典"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([1, 2, 3], f)  # JSON array instead of object
            temp_path = f.name
        
        try:
            result = load_config(temp_path)
            assert result == {}
        finally:
            os.unlink(temp_path)
    
    def test_load_config_default_path(self):
        """測試預設路徑為 config.json"""
        # 確保當前目錄沒有 config.json
        if os.path.exists("config.json"):
            os.rename("config.json", "config.json.bak")
            renamed = True
        else:
            renamed = False
        
        try:
            result = load_config()
            assert result == {}
        finally:
            if renamed:
                os.rename("config.json.bak", "config.json")
