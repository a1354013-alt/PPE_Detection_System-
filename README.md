# PPE 智慧監控系統 - 專業版 (PPE Detection System Pro)

本系統是一款基於 YOLOv8 的個人防護裝備 (PPE) 自動化偵測系統，專為工業安全監控設計。系統能即時識別人員是否佩戴安全帽、反光背心、護目鏡及口罩，並提供完整的違規事件追蹤與報告匯出功能。

## 核心功能

- **即時 PPE 偵測**：支援安全帽、背心、護目鏡、口罩等多種裝備。
- **人員追蹤 (Tracking)**：整合 YOLOv8 官方 Tracking 機制 (`model.track(frame, persist=True)`)，並實作基於空間位置的 Fallback ID，確保以「人員」為單位進行精確統計。
- **違規事件列表**：即時顯示違規時間、人員 ID、缺失項目及信心分數。
- **自動截圖存證**：偵測到違規時自動儲存現場畫面。
- **多格式報告匯出**：支援匯出 CSV、Excel 及 PDF 格式的完整違規報告。
- **熱力圖分析**：統計違規高發區域，生成視覺化熱力圖。

## 🛠 環境需求 (Requirements)

請確保已安裝以下套件：

```bash
pip install -r requirements.txt
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
- 需連續 N 幀（預設 3 幀，允許 1 幀誤差）出現相同缺失才判定為違規。

### Tracking Mode
- **Tracking Implementation**: YOLOv8 built-in tracking with `model.track(frame, persist=True)`
- **Tracking ID Source**: Directly from `box.id` in tracking results
- **Fallback Mechanism**: Coordinate-based ID generation when tracking unavailable
- **Duplicate Prevention**: Track_id based cooldown prevents duplicate counting

### 畫面比例保護
- 使用等比例縮放，禁止強制變形導致影像失真。

### Heatmap 說明
- Heatmap 為違規位置聚合視覺化，使用高斯模糊呈現熱點分佈。

### 違規記錄 (Violation Log)
- 系統會自動生成 `violations/violations.csv`，記錄：
  - timestamp: 時間戳記
  - frame: 幀數
  - track_id: 人員追蹤編號（來自 YOLOv8 tracking）
  - missing_items: 缺失項目
  - center_x, center_y: 違規位置座標

## 📁 專案結構

```
/workspace/
├── main_gui.py          # 主程式入口
├── helmet_detector.py   # PPE 偵測核心邏輯
├── event_logger.py      # 事件記錄與報告匯出
├── config.json          # 配置文件
├── requirements.txt     # 依賴套件
├── README.md           # 說明文件
├── .gitignore          # Git 忽略檔案
├── tests/
│   └── test_detector.py # 單元測試
├── reports/            # 報告輸出目錄（自動生成）
│   ├── ppe_violation_report_YYYYMMDD_HHMMSS.csv
│   ├── ppe_violation_report_YYYYMMDD_HHMMSS.xlsx
│   ├── ppe_violation_report_YYYYMMDD_HHMMSS.pdf
│   └── violation_heatmap.jpg
└── violations/         # 違規截圖輸出目錄（自動生成）
    └── v_YYYYMMDD_HHMMSS_track_id.jpg
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

主要相依套件包括：`ultralytics`, `opencv-python`, `pillow`, `matplotlib`, `pandas`, `openpyxl`, `reportlab`。

## 📂 使用流程

1. **啟動系統**：執行 `python main_gui.py`。
2. **載入模型**：預設使用 `yolov8n.pt`，可點擊「更換模型」載入自定義權重。
3. **選擇偵測項目**：在下方勾選欲監控的 PPE 項目。
4. **開始偵測**：
   - 點擊「上傳影片」分析現有影片檔。
   - 點擊「開啟攝影機」進行現場即時監控。
5. **查看事件**：違規事件會即時出現在左下方的列表中，雙擊事件可開啟對應截圖。
6. **匯出報告**：偵測結束後，點擊右側的匯出按鈕生成 CSV、Excel 或 PDF 報告。

## 📊 報告說明

### 匯出格式
- **CSV**：採用 UTF-8 with BOM 編碼，確保 Excel 開啟不亂碼。
- **Excel**：包含「事件明細」與「統計摘要」兩個工作表。
- **PDF**：包含統計摘要圖表、事件明細表，以及最近 5 筆違規事件的截圖展示。

### 報告欄位
- **Timestamp**：違規發生時間。
- **Source**：影像來源名稱。
- **Track ID**：人員追蹤編號。
- **Missing Items**：缺失的防護裝備。
- **Confidence**：偵測信心分數。
- **Screenshot Path**：截圖檔案路徑。
- **BBox**：人員在畫面中的座標位置。

## ❓ 常見問題 (FAQ)

- **Q: 為什麼勾選了背心卻沒反應？**
  - A: 請檢查模型是否支援 `vest` 類別。若模型不含此類別，系統會彈窗警告「類別不匹配」。
  - 請使用包含 PPE 類別的自訂模型。

- **Q: 畫面出現延遲？**
  - A: 系統預設使用 CPU 推論。若有 NVIDIA GPU，請確保安裝了 CUDA 版本的 ultralytics。

- **Q: 為什麼多人場景有時會誤判？**
  - A: 當人員快速移動或交叉時可能產生短暫誤判。系統已實作 tracking-based cooldown 機制來減少重複計數。

## 🧪 執行測試

```bash
cd /workspace
# 使用 unittest 執行所有測試
python3.11 -m unittest discover -v
```

## 📦 交付檢查 (Delivery Verification)

在專案交付、展示或打包前，請執行以下指令進行一鍵檢查，確保專案符合交付規範：

```bash
python scripts/verify_delivery.py
```

該腳本會檢查以下項目：
- **測試驗證**：執行 `compileall` 與所有單元測試。
- **忽略規則**：確認 `.gitignore` 包含必要的過濾規則（如 `reports/`, `*.pt` 等）。
- **README 指令**：確認文件包含正確的安裝與執行指令。
- **依賴項完整性**：掃描 `requirements.txt` 是否包含所有必要套件。
- **違禁檔案掃描**：確保沒有大型模型檔 (`.pt`, `.onnx`) 或偵測輸出檔被包進專案。

## 🚀 啟動系統

```bash
python main_gui.py
```

## ⚠️ 重要提醒

1. **預設 yolov8n.pt 不是 PPE 模型**，必須使用自訂 PPE 模型才能進行有效偵測。
2. **Tracking 模式**：系統使用 YOLOv8 內建 tracking (`model.track(frame, persist=True)`)，並提供 fallback 機制。
3. **Heatmap 為位置聚合**，僅供視覺化參考。
4. 請務必閱讀並理解配置文件參數後再進行調整。
5. **PDF 報告中文字顯示異常？** 目前 PDF 報告優先支援英文內容，若需完整支援中文，需額外配置中文字型檔。
