# PPE 智慧監控系統 Pro - 技術說明文件

本專案為工業環境設計的個人防護裝備 (PPE) 自動化偵測系統，基於 YOLOv8 與 Python Tkinter 開發。

## 🛠 環境需求 (Requirements)
請確保已安裝以下套件：
```bash
pip install ultralytics opencv-python pillow matplotlib
```

## 📂 模型權重配置 (Model Configuration)
1. **預設模型**：系統啟動時會自動下載 `yolov8n.pt` (COCO 資料集)。
2. **自定義模型**：建議使用針對 PPE 訓練的模型（如 `ppe.pt`）。
3. **類別映射**：系統內建 `class_map`，會自動將模型標籤（如 `hardhat`, `safety_vest`）映射至系統邏輯。
4. **更換模型**：可透過 GUI 右側的「更換模型」按鈕載入自定義 `.pt` 檔。

## 🚀 核心優化說明
- **Thread-Safe UI**：採用 `queue.Queue` 與 `window.after` 機制，確保背景偵測執行緒不會干擾 Tkinter 主執行緒，徹底解決畫面卡死問題。
- **空間關聯判定**：裝備必須位於人體上方 30% 區域（頭部區）才判定為合格，避免背景誤報。
- **穩定性過濾**：引入連續幀判定機制（5 幀中需有 3 幀違規），過濾掉單幀的偵測抖動。
- **模型相容性檢查**：若載入的模型不包含勾選的類別（如模型沒訓練過「口罩」），系統會主動彈窗警告，避免誤判。

## ❓ 常見問題 (FAQ)
- **Q: 為什麼勾選了背心卻沒反應？**
  - A: 請檢查模型是否支援 `vest` 類別。若模型不含此類別，系統會提示「類別不匹配」。
- **Q: 畫面出現延遲？**
  - A: 系統預設使用 CPU 推論。若有 NVIDIA GPU，請確保安裝了 `onnxruntime-gpu` 或 `torch` 的 CUDA 版本。
