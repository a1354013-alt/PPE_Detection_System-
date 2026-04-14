# PPE 智慧監控系統 Pro - 技術說明文件

本專案為工業環境設計的個人防護裝備 (PPE) 自動化偵測系統，基於 YOLOv8 與 Python Tkinter 開發。

## 🛠 環境需求 (Requirements)

請確保已安裝以下套件：

```bash
pip install ultralytics opencv-python pillow matplotlib numpy
```

或使用 requirements.txt:

```bash
pip install -r requirements.txt
```

## 📁 專案結構

```
/workspace/
├── main_gui.py           # 主程式 (GUI 介面)
├── helmet_detector.py    # 偵測核心邏輯
├── config.example.json   # 設定檔範例
├── requirements.txt      # Python 依賴
├── README.md            # 本文件
└── tests/               # 測試目錄
    ├── __init__.py
    ├── test_helmet_detector.py
    └── test_main_gui.py
```

## ⚙️ 配置說明 (Configuration)

### 模型配置

1. **預設模型**：系統啟動時會自動下載 `yolov8n.pt` (COCO 資料集)。
2. **自定義模型**：建議使用針對 PPE 訓練的模型（如 `ppe.pt`）。
3. **更換模型**：可透過 GUI 右側的「更換模型」按鈕載入自定義 `.pt` 檔。

### 設定檔 (選配)

系統支援透過 `config.example.json` 範例檔了解可用參數。目前主要參數透過程式碼內建預設值：

- `confidence_threshold`: YOLO 偵測信心度閾值 (預設 0.4)
- `violation_threshold`: 連續幀判定閾值 (預設 3 幀)
- `helmet_class_id`: 安全帽類別 ID (選配，null 表示自動映射)
- `person_class_id`: 人員類別 ID (預設 0)

## 🚀 核心優化說明

- **Thread-Safe UI**：採用 `queue.Queue` 與 `window.after` 機制，確保背景偵測執行緒不會干擾 Tkinter 主執行緒，徹底解決畫面卡死問題。
- **空間關聯判定**：裝備必須位於人體上方 30% 區域（頭部區）才判定為合格，避免背景誤報。
- **穩定性過濾**：引入連續幀判定機制（可配置 N 幀中需有 threshold 幀違規），過濾掉單幀的偵測抖動。
- **模型相容性檢查**：若載入的模型不包含勾選的類別（如模型沒訓練過「口罩」），系統會主動彈窗警告，避免誤判。
- **延遲匯入 YOLO**：`ultralytics` 採用 lazy import，讓單元測試可以在沒有真實模型的環境下執行。

## 🧪 測試 (Testing)

執行所有測試：

```bash
pytest tests/
```

測試特點：
- 不依賴真實 YOLO 模型（使用 mock）
- 不依賴真實攝影機
- 不依賴 GUI 顯示環境（headless 友善）

## ❓ 常見問題 (FAQ)

- **Q: 為什麼勾選了背心卻沒反應？**
  - A: 請檢查模型是否支援 `vest` 類別。若模型不含此類別，系統會提示「類別不匹配」。

- **Q: 如何在 CI/無桌面環境測試？**
  - A: 測試已設計為 headless 友善，直接執行 `pytest tests/` 即可。

- **Q: 畫面出現延遲？**
  - A: 系統預設使用 CPU 推論。若有 NVIDIA GPU，請確保安裝了 `onnxruntime-gpu` 或 `torch` 的 CUDA 版本。

## 📝 版本資訊

- 最後更新：2024
- Python 版本建議：3.8+
