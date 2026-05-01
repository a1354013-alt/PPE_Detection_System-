import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
from PIL import Image, ImageTk
import threading
import time
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from helmet_detector import HelmetDetector
from event_logger import EventLogger, ViolationEvent
import queue

class HelmetDetectionApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1400x950")
        self.window.configure(bg="#1e1e1e")

        # 初始化偵測器與事件記錄器
        self.detector = HelmetDetector()
        self.event_logger = EventLogger()
        self.running = False
        self.vid = None
        self.source_name = "unknown"
        
        # 執行緒安全隊列
        self.result_queue = queue.Queue()
        
        # 統計數據
        self.stats = {"total_violations": 0, 
                      "missing_counts": {"helmet": 0, "vest": 0, "goggles": 0, "mask": 0}}
        
        if not os.path.exists("violations"): os.makedirs("violations")

        self.setup_ui()
        self.check_queue() # 開始輪詢隊列

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
        self.left_panel = tk.Frame(self.main_frame, bg="#1e1e1e")
        self.left_panel.pack(side=tk.LEFT, padx=20, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.left_panel, width=800, height=500, bg="#000000", highlightthickness=0)
        self.canvas.pack(pady=5)

        # 違規事件列表 (Treeview)
        self.tree_frame = ttk.LabelFrame(self.left_panel, text="即時違規事件列表")
        self.tree_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        columns = ("time", "track_id", "missing", "conf", "screenshot")
        self.tree = ttk.Treeview(self.tree_frame, columns=columns, show="headings", height=8)
        self.tree.heading("time", text="時間")
        self.tree.heading("track_id", text="人員 ID")
        self.tree.heading("missing", text="缺少裝備")
        self.tree.heading("conf", text="信心分數")
        self.tree.heading("screenshot", text="截圖路徑")
        
        self.tree.column("time", width=150)
        self.tree.column("track_id", width=100)
        self.tree.column("missing", width=200)
        self.tree.column("conf", width=80)
        self.tree.column("screenshot", width=250)
        
        scrollbar = ttk.Scrollbar(self.tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.tree.bind("<Double-1>", self.on_tree_double_click)

        # 右側資訊欄
        self.right_frame = tk.Frame(self.main_frame, bg="#1e1e1e")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=20, width=400)

        # 1. 模型管理
        self.model_frame = ttk.LabelFrame(self.right_frame, text="模型管理")
        self.model_frame.pack(fill=tk.X, pady=5)
        self.lbl_model = tk.Label(self.model_frame, text=f"當前模型: {os.path.basename(self.detector.model_path)}", 
                                  bg="#1e1e1e", fg="white", font=("Arial", 9))
        self.lbl_model.pack(pady=2, anchor=tk.W)
        tk.Button(self.model_frame, text="更換模型 (.pt)", command=self.change_model, bg="#393e46", fg="white").pack(fill=tk.X)

        # 2. 數據圖表
        self.fig, self.ax = plt.subplots(figsize=(4, 3), dpi=100)
        self.fig.patch.set_facecolor('#1e1e1e')
        self.ax.set_facecolor('#1e1e1e')
        self.ax.tick_params(colors='white')
        self.chart_canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.chart_canvas.get_tk_widget().pack(fill=tk.X, pady=5)

        # 3. 統計數值
        self.stats_frame = ttk.LabelFrame(self.right_frame, text="即時統計")
        self.stats_frame.pack(fill=tk.X, pady=5)
        self.lbl_total_v = tk.Label(self.stats_frame, text="累計違規: 0", font=("Arial", 14, "bold"), bg="#1e1e1e", fg="#ff4b2b")
        self.lbl_total_v.pack(pady=5)

        # 4. 報告匯出
        self.export_frame = ttk.LabelFrame(self.right_frame, text="報告匯出")
        self.export_frame.pack(fill=tk.X, pady=10)
        tk.Button(self.export_frame, text="匯出 CSV 報告", command=lambda: self.export_report("csv"), bg="#2d4059", fg="white").pack(fill=tk.X, pady=2)
        tk.Button(self.export_frame, text="匯出 Excel 報告", command=lambda: self.export_report("xlsx"), bg="#2d4059", fg="white").pack(fill=tk.X, pady=2)
        tk.Button(self.export_frame, text="匯出 PDF 報告", command=lambda: self.export_report("pdf"), bg="#2d4059", fg="white").pack(fill=tk.X, pady=2)

        # --- 底部控制區 ---
        self.bottom_frame = tk.Frame(self.window, bg="#1e1e1e")
        self.bottom_frame.pack(pady=10, fill=tk.X)

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
        self.btn_stop = tk.Button(self.btn_frame, text="停止並生成熱力圖", command=self.stop_and_report, bg="#ff4b2b", fg="white", **btn_style)
        self.btn_stop.grid(row=0, column=2, padx=10)

        self.update_chart()

    def change_model(self):
        path = filedialog.askopenfilename(filetypes=[("YOLO Model", "*.pt")])
        if path:
            success, msg = self.detector.load_model(path)
            if success:
                self.detector.reset_tracking()
                self.lbl_model.config(text=f"當前模型: {os.path.basename(path)}")
                self.validate_model_support()
                messagebox.showinfo("成功", msg)
            else:
                messagebox.showerror("錯誤", msg)

    def validate_model_support(self):
        model_classes = self.detector.get_model_classes()
        unsupported = []
        for item, var in self.check_vars.items():
            if var.get() and item not in model_classes:
                unsupported.append(item)
        
        if unsupported:
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
        try:
            while True:
                task_type, data = self.result_queue.get_nowait()
                if task_type == "FRAME":
                    self.render_frame(data)
                elif task_type == "EVENT":
                    self.add_event_to_ui(data)
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

    def add_event_to_ui(self, event_dict):
        # 更新統計
        self.stats['total_violations'] += 1
        for m in event_dict['missing_list']:
            if m in self.stats['missing_counts']:
                self.stats['missing_counts'][m] += 1
        self.lbl_total_v.config(text=f"累計違規: {self.stats['total_violations']}")
        self.update_chart()
        
        # 更新列表 (插入到最上方)
        self.tree.insert("", 0, values=(
            event_dict['timestamp'],
            event_dict['track_id'],
            event_dict['missing_items'],
            f"{event_dict['confidence']:.2f}",
            event_dict['screenshot_path']
        ))
        
        # 限制列表顯示數量
        if len(self.tree.get_children()) > 1000:
            self.tree.delete(self.tree.get_children()[-1])

    def on_tree_double_click(self, event):
        item = self.tree.selection()[0]
        screenshot_path = self.tree.item(item, "values")[4]
        if screenshot_path and os.path.exists(screenshot_path):
            # 在 Linux 下使用 xdg-open 開啟圖片
            os.system(f"xdg-open {screenshot_path}")
        else:
            messagebox.showwarning("提示", "找不到截圖檔案")

    def export_report(self, fmt):
        if not self.event_logger.events:
            messagebox.showwarning("提示", "目前沒有違規事件可供匯出")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ppe_violation_report_{timestamp}.{fmt}"
        
        try:
            if fmt == "csv":
                path = self.event_logger.export_csv(filename)
            elif fmt == "xlsx":
                path = self.event_logger.export_excel(filename, self.stats)
            elif fmt == "pdf":
                path = self.event_logger.export_pdf(filename, self.stats)
            
            messagebox.showinfo("成功", f"報告已匯出至：{path}")
        except Exception as e:
            messagebox.showerror("錯誤", f"匯出失敗：{str(e)}")

    def open_file(self):
        path = filedialog.askopenfilename()
        if path: 
            self.source_name = os.path.basename(path)
            self.start_detection(path)

    def open_camera(self): 
        self.source_name = "camera_0"
        self.start_detection(0)

    def start_detection(self, source):
        if self.running: return
        self.vid = cv2.VideoCapture(source)
        if not self.vid.isOpened():
            messagebox.showerror("錯誤", "無法開啟影像來源")
            return
        
        # 重置狀態
        self.detector.reset_tracking()
        self.event_logger.clear_events()
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        self.running = True
        self.btn_upload.config(state=tk.DISABLED)
        self.btn_camera.config(state=tk.DISABLED)
        threading.Thread(target=self.worker_thread, daemon=True).start()

    def worker_thread(self):
        last_cap_time = 0
        while self.running:
            ret, frame = self.vid.read()
            if not ret: break
            
            display_frame = cv2.resize(frame, (800, 500))
            targets = [k for k, v in self.check_vars.items() if v.get()]
            annotated, info = self.detector.detect(display_frame, targets, self.source_name)
            
            self.result_queue.put(("FRAME", annotated))
            
            if info.get('violation_detected'):
                new_events = info.get('new_events', [])
                for event_data in new_events:
                    # 截圖邏輯 (全域冷卻 3 秒)
                    curr_time = time.time()
                    screenshot_path = ""
                    if curr_time - last_cap_time > 3:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        screenshot_path = f"violations/v_{ts}_{event_data['track_id']}.jpg"
                        cv2.imwrite(screenshot_path, annotated)
                        last_cap_time = curr_time
                    
                    # 建立事件物件並記錄
                    event = ViolationEvent(
                        timestamp=event_data['timestamp'],
                        source=event_data['source'],
                        track_id=event_data['track_id'],
                        person_count=event_data['person_count'],
                        missing_items=event_data['missing_items'],
                        screenshot_path=screenshot_path,
                        confidence=event_data['confidence'],
                        bbox=event_data['bbox']
                    )
                    self.event_logger.add_event(event)
                    
                    # 發送到 UI 更新 (包含 event_data 方便 UI 使用)
                    event_data['screenshot_path'] = screenshot_path
                    self.result_queue.put(("EVENT", event_data))
            
            time.sleep(0.01)
        self.result_queue.put(("STOP", None))

    def handle_stop(self):
        self.running = False
        if self.vid: self.vid.release()
        self.btn_upload.config(state=tk.NORMAL)
        self.btn_camera.config(state=tk.NORMAL)

    def stop_and_report(self):
        self.running = False
        time.sleep(0.5)
        if self.detector.violation_coords:
            heatmap = self.detector.generate_heatmap((500, 800))
            cv2.imwrite("violation_heatmap.jpg", heatmap)
            messagebox.showinfo("報告", "熱力圖已生成：violation_heatmap.jpg")
        else:
            messagebox.showinfo("提示", "偵測結束，無違規記錄。")

    def on_closing(self):
        self.running = False
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = HelmetDetectionApp(root, "PPE 智慧監控系統 Pro")
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
