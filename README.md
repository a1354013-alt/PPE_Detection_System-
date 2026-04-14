# PPE 智慧監控系統 Pro - 技術說明文件

本專案為工業環境設計的個人防護裝備 (PPE) 自動化偵測系統，基於 YOLOv8 與 Python Tkinter 開發。

## 🛠 Requirements (環境需求)

請確保已安裝以下套件：

```bash
pip install -r requirements.txt
```

### 依賴套件清單
- `ultralytics` - YOLOv8 模型推理
- `opencv-python` - 影像處理
- `numpy` - 數值計算
- `pytest` - 測試框架
- `pillow` - 影像顯示
- `matplotlib` - 圖表繪製

## 📦 Installation (安裝步驟)

1. 克隆或下載專案：
```bash
git clone <repository-url>
cd PPE_Detection_System
```

2. 安裝依賴：
```bash
pip install -r requirements.txt
```

3. 首次執行時會自動下載預設模型 `yolov8n.pt`

## 🚀 How to Run (如何執行)

### GUI 模式（推薦）
```bash
python main_gui.py
```

### 功能說明
- **上傳影片**：選擇本地影片檔案進行偵測
- **開啟攝影機**：使用網路攝影機進行即時偵測
- **更換模型**：載入自定義 YOLO `.pt` 模型權重
- **停止並生成報告**：結束偵測並生成違規熱點圖

## 🧪 How to Test (如何測試)

執行所有測試：
```bash
pytest
```

執行特定測試：
```bash
pytest tests/test_detector.py -v
```

### 測試項目
- `test_model_load_failure` - 測試模型載入失敗處理
- `test_config_load_default` - 測試預設配置載入
- `test_violation_log_limit` - 測試違規記錄記憶體保護

## ⚙️ Config Description (配置說明)

系統使用 `config.json` 進行配置（可選）。若檔案不存在或損壞，將使用預設值。

### 預設配置
```json
{
    "confidence_threshold": 0.5,
    "helmet_class_id": 0,
    "person_class_id": 1,
    "violation_threshold": 5
}
```

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `confidence_threshold` | 偵測信心度閾值 | 0.5 |
| `helmet_class_id` | 安全帽類別 ID | 0 |
| `person_class_id` | 人員類別 ID | 1 |
| `violation_threshold` | 違規判定閾值 | 5 |

## 📂 模型權重配置 (Model Configuration)

1. **預設模型**：系統啟動時會自動下載 `yolov8n.pt` (COCO 資料集)
2. **自定義模型**：建議使用針對 PPE 訓練的模型（如 `ppe.pt`）
3. **類別映射**：系統內建 `class_map`，會自動將模型標籤映射至系統邏輯
4. **更換模型**：可透過 GUI 右側的「更換模型」按鈕載入自定義 `.pt` 檔

## 🔒 穩定性特性 (Stability Features)

- **模型載入安全**：載入失敗時不會導致系統崩潰
- **配置容錯**：配置文件損壞時自動使用預設值
- **記憶體保護**：違規記錄限制最大 500 筆，防止記憶體洩漏
- **攝影機安全釋放**：程式結束時正確釋放資源
- **空幀處理**：偵測迴圈中處理空幀避免崩潰
- **FPS 顯示**：即時顯示影格率便於效能監控

## ❓ 常見問題 (FAQ)

**Q: 為什麼勾選了背心卻沒反應？**
- A: 請檢查模型是否支援 `vest` 類別。若模型不含此類別，系統會提示「類別不匹配」。

**Q: 畫面出現延遲？**
- A: 系統預設使用 CPU 推論。若有 NVIDIA GPU，請確保安裝了 CUDA 版本的相關套件。

**Q: 模型載入失敗？**
- A: 確認 `.pt` 檔案路徑正確，且檔案未損壞。系統會顯示錯誤訊息並阻止偵測開始。
