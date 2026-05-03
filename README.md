# PPE 智慧監控系統 - 專業版 (PPE Detection System Pro)

本系統是一款基於 YOLOv8 的個人防護裝備 (PPE) 自動化偵測系統，專為工業安全監控設計。系統能即時識別人員是否佩戴安全帽、反光背心、護目鏡及口罩，並提供完整的違規事件追蹤、Dashboard 數據視覺化與報告匯出功能。

## 核心功能

- **即時 PPE 偵測**：支援安全帽、背心、護目鏡、口罩等多種裝備。
- **Dashboard 數據中心**：新增視覺化統計分頁，包含違規趨勢圖與 PPE 缺失比例分析。
- **人員追蹤 (Tracking)**：整合 YOLOv8 官方 Tracking 機制，並實作基於空間位置的 Fallback ID，確保以「人員」為單位進行精確統計。
- **違規事件列表**：即時顯示違規時間、人員 ID、缺失項目及信心分數。
- **自動截圖存證**：偵測到違規時自動儲存現場畫面。
- **多格式報告匯出**：支援匯出 CSV、Excel 及 PDF 格式，PDF 報告已整合視覺化圖表。
- **熱力圖分析**：統計違規高發區域，生成視覺化熱力圖。

## 📊 Dashboard Features

本專案不僅是一個偵測工具，更是一個完整的安全監控 Dashboard：

- **Real-time Violation Trend Chart**：依時間統計違規事件數，支援每分鐘或每 10 秒的高頻率統計，即時掌握風險波動。
- **PPE Missing Distribution Chart**：視覺化呈現 helmet、vest、mask、goggles 等各類裝備的缺失比例與次數，協助管理者精準強化安全教育。
- **Summary Statistics**：提供總違規數、各類裝備缺失次數等關鍵指標的 Summary Cards。
- **Export Dashboard Report**：支援將 Dashboard 上的視覺化圖表直接匯出至 PDF 報告中，方便進行定期安全稽核。
- **Report Output Path**：
  - 報告檔案：`reports/ppe_violation_report_YYYYMMDD_HHMMSS.pdf`
  - 統計圖表：`reports/charts/trend_YYYYMMDD_HHMMSS.png` 與 `ratio_YYYYMMDD_HHMMSS.png`

## 🛠 環境需求 (Requirements)

請確保已安裝以下套件：

```bash
pip install -r requirements.txt
```

## 📂 專案結構

```
/workspace/
├── main_gui.py          # 主程式入口 (整合 Dashboard Tab)
├── analytics.py         # 統計資料處理模組 (新增)
├── helmet_detector.py   # PPE 偵測核心邏輯
├── event_logger.py      # 事件記錄與報告匯出 (支援圖表整合)
├── config.json          # 配置文件
├── requirements.txt     # 依賴套件
├── README.md           # 說明文件
├── tests/
│   ├── test_detector.py # 偵測器單元測試
│   └── test_analytics.py # 統計模組單元測試 (新增)
├── reports/            # 報告輸出目錄
│   └── charts/         # 統計圖表輸出目錄
└── violations/         # 違規截圖輸出目錄
```

## 🚀 作品集敘事 (Portfolio Narrative)

> **This project is not only a PPE detection demo, but also a lightweight safety monitoring dashboard that records violation events, visualizes risk trends, and exports inspection-ready reports.**
>
> 透過將即時電腦視覺偵測與數據分析 Dashboard 結合，本專案展示了如何將 AI 模型轉化為具備商業價值的監控系統。系統不僅解決了「有沒有戴」的問題，更透過趨勢分析與比例統計，為工安管理者提供了決策支援的數據基礎。

## 📂 使用流程

1. **啟動系統**：執行 `python main_gui.py`。
2. **載入模型**：預設使用 `yolov8n.pt`，可點擊「更換模型」載入自定義權重。
3. **開始偵測**：上傳影片或開啟攝影機。
4. **查看 Dashboard**：切換至 「Dashboard 統計」分頁，點擊 「Refresh Dashboard」 更新最新數據。
5. **匯出報告**：點擊「Export Dashboard Report」生成包含統計圖表的 PDF 報告。

## 🧪 執行測試

```bash
# 執行所有測試
python -m unittest discover -v
```

## ⚠️ 重要提醒

1. **PDF 報告圖表**：匯出 PDF 前，系統會自動生成最新的 PNG 圖表並儲存於 `reports/charts/`。
2. **數據處理**：統計模組具備高度容錯性，可處理空資料、缺少欄位的舊格式或不同格式的時間戳記。
