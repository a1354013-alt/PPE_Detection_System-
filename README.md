# PPE 智慧監控系統 Pro - 技術說明文件

本專案為工業環境設計的個人防護裝備 (PPE) 自動化偵測系統，基於 YOLOv8 與 Python Tkinter 開發。

## 🛠 環境需求 (Requirements)

請確保已安裝以下套件：

```bash
pip install ultralytics opencv-python pillow matplotlib
```

## 📂 模型權重配置 (Model Configuration)

1. **預設模型**：系統啟動時會自動下載 `yolov8n.pt` (COCO 資料集)。
   - ⚠️ **重要**：`yolov8n.pt` 是通用物件偵測模型，**不是**專門的 PPE 模型。
   - 它僅能識別 `person` 類別，無法識別安全帽、背心等 PPE 項目。

2. **自定義模型**：建議使用針對 PPE 訓練的模型（如 `ppe.pt`）。
   - 必須使用包含 PPE 類別（helmet, vest, goggles, mask）的自訂模型才能進行有效偵測。

3. **類別映射**：系統內建 `class_map`，會自動將模型標籤（如 `hardhat`, `safety_vest`）映射至系統邏輯。

4. **更換模型**：可透過 GUI 右側的「更換模型」按鈕載入自定義 `.pt` 檔。

## 🚀 核心優化說明

### Thread-Safe UI
- 採用 `queue.Queue` 與 `window.after` 機制，確保背景偵測執行緒不會干擾 Tkinter 主執行緒。
- 在 UI thread 收集 `BooleanVar` 狀態，worker thread 只讀取普通字典，避免 thread 安全問題。
- 使用 `threading.Event` 控制停止流程，移除 `time.sleep()` 同步機制。

### PPE 空間區域判定
- **不同 PPE 使用不同人體區域**，不再共用 head zone：
  - `helmet`: head zone (上 25%)
  - `goggles`: face_upper zone (上 40%)
  - `mask`: face_lower zone (40% ~ 60%)
  - `vest`: torso zone (30% ~ 80%)

### 穩定性過濾 (Temporal Smoothing)
- 使用 **person-based buffer**，每個人獨立追蹤。
- 簡化版 index tracking：以 person index 作為簡易 ID。
- 需連續 N 幀（預設 3 幀，允許 1 幀誤差）出現相同缺失才判定為違規。
- ⚠️ **注意**：由於 YOLOv8 預設不帶 Tracking，此為簡化版本，多人交叉移動時可能產生誤判。

### 畫面比例保護
- 使用等比例縮放，禁止強制變形導致影像失真。

### Heatmap 說明
- Heatmap 為違規位置聚合視覺化，使用高斯模糊呈現熱點分佈。

### 違規記錄 (Violation Log)
- 系統會自動生成 `violations/violations.csv`，記錄：
  - timestamp: 時間戳記
  - frame: 幀數
  - person_id: 人員編號（簡易 index）
  - missing_items: 缺失項目
  - center_x, center_y: 違規位置座標

## 📁 專案結構

```
/workspace/
├── main_gui.py          # 主程式入口
├── helmet_detector.py   # PPE 偵測核心邏輯
├── config.json          # 配置文件
├── requirements.txt     # 依賴套件
├── README.md           # 說明文件
├── .gitignore          # Git 忽略檔案
├── tests/
│   └── test_detector.py # 單元測試
└── violations/         # 違規記錄輸出目錄（自動生成）
```

## 🔧 配置文件 (config.json)

```json
{
  "confidence_threshold": 0.5,
  "iou_threshold": 0.3,
  "temporal_frames": 3,
  "cooldown_seconds": 2
}
```

- `confidence_threshold`: 偵測信心值閾值
- `iou_threshold`: NMS IoU 閾值
- `temporal_frames`: 時間平滑幀數
- `cooldown_seconds`: 違規截圖冷卻時間（秒）

## ❓ 常見問題 (FAQ)

- **Q: 為什麼勾選了背心卻沒反應？**
  - A: 請檢查模型是否支援 `vest` 類別。若模型不含此類別，系統會彈窗警告「類別不匹配」。
  - 請使用包含 PPE 類別的自訂模型。

- **Q: 畫面出現延遲？**
  - A: 系統預設使用 CPU 推論。若有 NVIDIA GPU，請確保安裝了 CUDA 版本的 ultralytics。

- **Q: 為什麼多人場景有時會誤判？**
  - A: 由於使用簡化版 index tracking，當人員快速移動或交叉時可能產生短暫誤判。這是已知限制。

## 🧪 執行測試

```bash
cd /workspace
python -m pytest tests/test_detector.py -v
# 或使用 unittest
python -m unittest tests.test_detector
```

## 🚀 啟動系統

```bash
python main_gui.py
```

## ⚠️ 重要提醒

1. **預設 yolov8n.pt 不是 PPE 模型**，必須使用自訂 PPE 模型才能進行有效偵測。
2. **Temporal smoothing 為簡化版本**，使用 person index 而非真實 tracking ID。
3. **Heatmap 為位置聚合**，僅供視覺化參考。
4. 請務必閱讀並理解配置文件參數後再進行調整。
