import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
from PIL import Image, ImageTk
import threading
import time
import os
import json
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from helmet_detector import HelmetDetector, MAX_LOG_SIZE
import queue

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class HelmetDetectionApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1300x900")
        self.window.configure(bg="#1e1e1e")

        # 初始化偵測器
        self.detector = HelmetDetector()
        self.running = False
        self.vid = None
        self.cap = None
        
        # 執行緒安全隊列
        self.result_queue = queue.Queue()
        
        # 統計數據
        self.stats = {"total_violations": 0, 
                      "missing_counts": {"helmet": 0, "vest": 0, "goggles": 0, "mask": 0}}
        
        if not os.path.exists("violations"): os.makedirs("violations")
        
        # Load config safely
        self.config = self.load_config("config.json")

        self.setup_ui()
        self.check_queue() # 開始輪詢隊列
        
        # Register cleanup on window close
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def load_config(self, config_path):
        default_config = {
            "confidence_threshold": 0.5,
            "helmet_class_id": 0,
            "person_class_id": 1,
            "violation_threshold": 5
        }
        
        try:
            if not os.path.exists(config_path):
                logging.warning("Config not found. Using default.")
                return default_config
            
            with open(config_path, "r") as f:
                user_config = json.load(f)
            
            merged = {**default_config, **user_config}
            logging.info("Config loaded")
            return merged
            
        except Exception as e:
            logging.error(f"Config load failed: {e}")
            logging.warning("Using default config")
            return default_config

    def setup_ui(self):
        # --- 頂部標題 ---
        self.top_frame = tk.Frame(self.window, bg="#1e1e1e")
        self.top_frame.pack(pady=10, fill=tk.X)
        tk.Label(self.top_frame, text="PPE 智慧監控系統 - 專業版", font=("Arial", 22, "bold"), 
                 bg="#1e1e1e", fg="#00adb5").pack()

        # --- 主內容區 ---
        self.main_frame = tk.Frame(self.window, bg="#1e1e1e")
        self.main_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        # 左側影像區
        self.canvas = tk.Canvas(self.main_frame, width=800, height=500, bg="#000000", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, padx=20)

        # 右側資訊欄
        self.right_frame = tk.Frame(self.main_frame, bg="#1e1e1e")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=20)

        # 1. 模型管理
        self.model_frame = ttk.LabelFrame(self.right_frame, text="模型管理")
        self.model_frame.pack(fill=tk.X, pady=5)
        self.lbl_model = tk.Label(self.model_frame, text=f"當前模型：{os.path.basename(self.detector.model_path)}", 
                                  bg="#1e1e1e", fg="white", font=("Arial", 9))
        self.lbl_model.pack(pady=2, anchor=tk.W)
        tk.Button(self.model_frame, text="更換模型 (.pt)", command=self.change_model, bg="#393e46", fg="white").pack(fill=tk.X)

        # 2. 數據圖表
        self.fig, self.ax = plt.subplots(figsize=(4, 3), dpi=100)
        self.fig.patch.set_facecolor('#1e1e1e')
        self.ax.set_facecolor('#1e1e1e')
        self.ax.tick_params(colors='white')
        self.chart_canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 3. 統計數值
        self.stats_frame = ttk.LabelFrame(self.right_frame, text="即時統計")
        self.stats_frame.pack(fill=tk.X, pady=10)
        self.lbl_total_v = tk.Label(self.stats_frame, text="累計違規：0", font=("Arial", 14, "bold"), bg="#1e1e1e", fg="#ff4b2b")
        self.lbl_total_v.pack(pady=5)

        # --- 底部控制區 ---
        self.bottom_frame = tk.Frame(self.window, bg="#1e1e1e")
        self.bottom_frame.pack(pady=20, fill=tk.X)

        # 偵測項目勾選
        self.check_frame = tk.Frame(self.bottom_frame, bg="#1e1e1e")
        self.check_frame.pack()
        self.check_vars = {item: tk.BooleanVar(value=(item=="helmet")) for item in ["helmet", "vest", "goggles", "mask"]}
        for item, var in self.check_vars.items():
            cb = tk.Checkbutton(self.check_frame, text=item.capitalize(), variable=var, bg="#1e1e1e", fg="white", 
                                selectcolor="#393e46", activebackground="#1e1e1e", command=self.validate_model_support)
            cb.pack(side=tk.LEFT, padx=15)

        # 操作按鈕
        self.btn_frame = tk.Frame(self.bottom_frame, bg="#1e1e1e")
        self.btn_frame.pack(pady=10)
        btn_style = {"font": ("Arial", 10, "bold"), "width": 15, "height": 2, "bd": 0, "cursor": "hand2"}
        self.btn_upload = tk.Button(self.btn_frame, text="上傳影片", command=self.open_file, bg="#393e46", fg="white", **btn_style)
        self.btn_upload.grid(row=0, column=0, padx=10)
        self.btn_camera = tk.Button(self.btn_frame, text="開啟攝影機", command=self.open_camera, bg="#00adb5", fg="white", **btn_style)
        self.btn_camera.grid(row=0, column=1, padx=10)
        self.btn_stop = tk.Button(self.btn_frame, text="停止並生成報告", command=self.stop_and_report, bg="#ff4b2b", fg="white", **btn_style)
        self.btn_stop.grid(row=0, column=2, padx=10)

        self.update_chart()

    def change_model(self):
        path = filedialog.askopenfilename(filetypes=[("YOLO Model", "*.pt")])
        if path:
            success, msg = self.detector.load_model(path)
            if success:
                self.lbl_model.config(text=f"當前模型：{os.path.basename(path)}")
                self.validate_model_support()
                messagebox.showinfo("成功", msg)
            else:
                messagebox.showerror("錯誤", msg)

    def validate_model_support(self):
        """ 檢查模型是否支援勾選的類別 """
        model_classes = self.detector.get_model_classes()
        unsupported = []
        for item, var in self.check_vars.items():
            if var.get() and item not in model_classes:
                unsupported.append(item)
        
        if unsupported:
            messagebox.set_warning = True # 標記警告
            msg = f"警告：當前模型不包含以下類別：\n{', '.join(unsupported)}\n系統將跳過這些項目的判定。"
            messagebox.showwarning("類別不匹配", msg)

    def update_chart(self):
        self.ax.clear()
        labels = [k.capitalize() for k in self.stats["missing_counts"].keys()]
        values = list(self.stats["missing_counts"].values())
        self.ax.bar(labels, values, color=['#00adb5', '#ff4b2b', '#f9ed69', '#b83b5e'])
        self.ax.set_title("PPE 缺失統計", color='white', fontsize=10)
        self.fig.tight_layout()
        self.chart_canvas.draw()

    def check_queue(self):
        """ 輪詢隊列以更新 UI (Thread-safe) """
        try:
            while True:
                task_type, data = self.result_queue.get_nowait()
                if task_type == "FRAME":
                    self.render_frame(data)
                elif task_type == "STATS":
                    self.update_stats_ui(data)
                elif task_type == "STOP":
                    self.handle_stop()
        except queue.Empty:
            pass
        finally:
            self.window.after(30, self.check_queue)

    def render_frame(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(img))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.canvas.imgtk = imgtk

    def update_stats_ui(self, info):
        self.stats['total_violations'] += 1
        for m in info['missing_items']:
            if m in self.stats['missing_counts']:
                self.stats['missing_counts'][m] += 1
        self.lbl_total_v.config(text=f"累計違規：{self.stats['total_violations']}")
        self.update_chart()

    def open_file(self):
        path = filedialog.askopenfilename()
        if path: self.start_detection(path)

    def open_camera(self): self.start_detection(0)

    def start_detection(self, source):
        if self.running: return
        
        # Check if model is loaded before starting detection
        if self.detector.model is None:
            messagebox.showerror("錯誤", "Detection cannot start because model failed to load.")
            return
        
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            messagebox.showerror("錯誤", "無法開啟影像來源")
            return
        self.vid = self.cap
        self.running = True
        self.btn_upload.config(state=tk.DISABLED)
        self.btn_camera.config(state=tk.DISABLED)
        threading.Thread(target=self.worker_thread, daemon=True).start()

    def worker_thread(self):
        last_cap_time = 0
        fps_prev_time = time.time()
        fps_value = 0.0
        
        while self.running:
            ret, frame = self.vid.read()
            
            # Detection loop safety check
            if frame is None:
                logging.warning("Empty frame detected")
                continue
            
            if not ret:
                break
            
            # Calculate FPS
            curr_time = time.time()
            fps_value = 1.0 / (curr_time - fps_prev_time) if (curr_time - fps_prev_time) > 0 else 0.0
            fps_prev_time = curr_time
            
            display_frame = cv2.resize(frame, (800, 500))
            targets = [k for k, v in self.check_vars.items() if v.get()]
            annotated, info = self.detector.detect(display_frame, targets)
            
            # Add FPS display
            fps_text = f"FPS: {fps_value:.2f}"
            cv2.putText(
                annotated,
                fps_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # 發送影像到 UI
            self.result_queue.put(("FRAME", annotated))
            
            # 處理違規邏輯
            if info.get('violation_detected'):
                curr_time = time.time()
                if curr_time - last_cap_time > 3: # 3 秒冷卻
                    # Add violation with memory protection
                    self.detector.add_violation(info)
                    self.result_queue.put(("STATS", info))
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"violations/v_{timestamp}.jpg", annotated)
                    last_cap_time = curr_time
            
            time.sleep(0.01) # 稍微讓出 CPU
        self.result_queue.put(("STOP", None))

    def handle_stop(self):
        self.running = False
        if self.vid:
            self.vid.release()
        self.cleanup()
        self.btn_upload.config(state=tk.NORMAL)
        self.btn_camera.config(state=tk.NORMAL)

    def cleanup(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        logging.info("Camera released")

    def stop_and_report(self):
        self.running = False
        time.sleep(0.5)
        if self.detector.violation_coords:
            heatmap = self.detector.generate_heatmap((500, 800))
            cv2.imwrite("violation_heatmap.jpg", heatmap)
            messagebox.showinfo("報告", "報告已生成：violation_heatmap.jpg")
        else:
            messagebox.showinfo("提示", "偵測結束，無違規記錄。")

    def on_closing(self):
        self.running = False
        self.cleanup()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = HelmetDetectionApp(root, "PPE 智慧監控系統 Pro")
    root.mainloop()
