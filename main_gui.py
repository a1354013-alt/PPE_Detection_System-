import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
from PIL import Image, ImageTk
import threading
import time
import os
import sys
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from helmet_detector import HelmetDetector
from event_logger import EventLogger, ViolationEvent
import queue
from analytics import build_dashboard_summary, get_violation_trend, get_ppe_missing_counts


class HelmetDetectionApp:
    def __init__(self, window, window_title, demo_mode=False):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1400x950")
        self.window.configure(bg="#1e1e1e")

        self.detector = HelmetDetector(demo_mode=demo_mode)
        self.event_logger = EventLogger()
        self.running = False
        self.demo_mode = demo_mode

        self.worker = None
        self.stop_event = threading.Event()

        self.vid = None
        self.source_name = "unknown"

        self.result_queue = queue.Queue()

        self.stats = {
            "total_violations": 0,
            "missing_counts": {
                "helmet": 0,
                "vest": 0,
                "goggles": 0,
                "mask": 0
            }
        }

        self.enabled_items = {}

        # 確保輸出目錄存在
        os.makedirs("reports", exist_ok=True)
        os.makedirs("reports/charts", exist_ok=True)
        os.makedirs("violations", exist_ok=True)

        self.setup_ui()
        self.check_queue()
        
        if self.demo_mode:
            self.window.after(500, self.show_demo_info)

    def setup_ui(self):
        # 頂部標題
        self.top_frame = tk.Frame(self.window, bg="#1e1e1e")
        self.top_frame.pack(pady=10, fill=tk.X)

        tk.Label(
            self.top_frame,
            text="PPE 智慧監控系統 - 專業版",
            font=("Arial", 22, "bold"),
            bg="#1e1e1e",
            fg="#00adb5"
        ).pack()

        # 使用 Notebook 建立分頁
        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(pady=10, fill=tk.BOTH, expand=True)

        # 分頁 1: 即時監控
        self.monitor_tab = tk.Frame(self.notebook, bg="#1e1e1e")
        self.notebook.add(self.monitor_tab, text=" 即時監控 ")

        # 分頁 2: Dashboard 統計
        self.dashboard_tab = tk.Frame(self.notebook, bg="#1e1e1e")
        self.notebook.add(self.dashboard_tab, text=" Dashboard 統計 ")

        self.setup_monitor_tab()
        self.setup_dashboard_tab()

        # 底部控制欄
        self.bottom_frame = tk.Frame(self.window, bg="#1e1e1e")
        self.bottom_frame.pack(pady=10, fill=tk.X)

        self.check_frame = tk.Frame(self.bottom_frame, bg="#1e1e1e")
        self.check_frame.pack()

        self.check_vars = {
            item: tk.BooleanVar(value=(item == "helmet"))
            for item in ["helmet", "vest", "goggles", "mask"]
        }

        for item, var in self.check_vars.items():
            cb = tk.Checkbutton(
                self.check_frame,
                text=item.capitalize(),
                variable=var,
                bg="#1e1e1e",
                fg="white",
                selectcolor="#393e46",
                activebackground="#1e1e1e",
                command=self.validate_model_support
            )
            cb.pack(side=tk.LEFT, padx=15)

        self.btn_frame = tk.Frame(self.bottom_frame, bg="#1e1e1e")
        self.btn_frame.pack(pady=10)

        btn_style = {
            "font": ("Arial", 10, "bold"),
            "width": 15,
            "height": 2,
            "bd": 0,
            "cursor": "hand2"
        }

        self.btn_upload = tk.Button(
            self.btn_frame,
            text="上傳影片",
            command=self.open_file,
            bg="#393e46",
            fg="white",
            **btn_style
        )
        self.btn_upload.grid(row=0, column=0, padx=10)

        self.btn_camera = tk.Button(
            self.btn_frame,
            text="開啟攝影機",
            command=self.open_camera,
            bg="#00adb5",
            fg="white",
            **btn_style
        )
        self.btn_camera.grid(row=0, column=1, padx=10)

        self.btn_stop = tk.Button(
            self.btn_frame,
            text="停止偵測",
            command=self.stop_and_report,
            bg="#ff4b2b",
            fg="white",
            **btn_style
        )
        self.btn_stop.grid(row=0, column=2, padx=10)

    def setup_monitor_tab(self):
        self.main_frame = tk.Frame(self.monitor_tab, bg="#1e1e1e")
        self.main_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        self.left_panel = tk.Frame(self.main_frame, bg="#1e1e1e")
        self.left_panel.pack(side=tk.LEFT, padx=20, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            self.left_panel,
            width=800,
            height=500,
            bg="#000000",
            highlightthickness=0
        )
        self.canvas.pack(pady=5)

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

        self.right_frame = tk.Frame(self.main_frame, bg="#1e1e1e", width=400)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=20)

        self.model_frame = ttk.LabelFrame(self.right_frame, text="模型管理")
        self.model_frame.pack(fill=tk.X, pady=5)

        self.lbl_model = tk.Label(
            self.model_frame,
            text=f"當前模型：{os.path.basename(self.detector.model_path)}",
            bg="#1e1e1e",
            fg="white",
            font=("Arial", 9)
        )
        self.lbl_model.pack(pady=2, anchor=tk.W)

        tk.Button(
            self.model_frame,
            text="更換模型 (.pt)",
            command=self.change_model,
            bg="#393e46",
            fg="white"
        ).pack(fill=tk.X)

        self.fig_mini, self.ax_mini = plt.subplots(figsize=(4, 3), dpi=80)
        self.fig_mini.patch.set_facecolor("#1e1e1e")
        self.ax_mini.set_facecolor("#1e1e1e")
        self.ax_mini.tick_params(colors="white")

        self.chart_canvas_mini = FigureCanvasTkAgg(self.fig_mini, master=self.right_frame)
        self.chart_canvas_mini.get_tk_widget().pack(fill=tk.X, pady=5)

        self.stats_frame = ttk.LabelFrame(self.right_frame, text="即時統計")
        self.stats_frame.pack(fill=tk.X, pady=10)

        self.lbl_total_v = tk.Label(
            self.stats_frame,
            text="累計違規：0",
            font=("Arial", 14, "bold"),
            bg="#1e1e1e",
            fg="#ff4b2b"
        )
        self.lbl_total_v.pack(pady=5)

        self.export_frame = ttk.LabelFrame(self.right_frame, text="報告匯出")
        self.export_frame.pack(fill=tk.X, pady=10)

        tk.Button(
            self.export_frame,
            text="匯出 CSV 報告",
            command=lambda: self.export_report("csv"),
            bg="#2d4059",
            fg="white"
        ).pack(fill=tk.X, pady=2)

        tk.Button(
            self.export_frame,
            text="匯出 Excel 報告",
            command=lambda: self.export_report("xlsx"),
            bg="#2d4059",
            fg="white"
        ).pack(fill=tk.X, pady=2)

        tk.Button(
            self.export_frame,
            text="匯出 PDF 報告",
            command=lambda: self.export_report("pdf"),
            bg="#2d4059",
            fg="white"
        ).pack(fill=tk.X, pady=2)

    def setup_dashboard_tab(self):
        # 摘要區域
        self.summary_frame = tk.Frame(self.dashboard_tab, bg="#1e1e1e")
        self.summary_frame.pack(pady=20, fill=tk.X)

        self.summary_cards = {}
        card_items = [
            ("Total Violations", "total_v", "#ff4b2b"),
            ("Missing Helmet", "m_helmet", "#00adb5"),
            ("Missing Vest", "m_vest", "#f9ed69"),
            ("Missing Mask", "m_mask", "#b83b5e"),
            ("Missing Goggles", "m_goggles", "#eeeeee")
        ]

        for i, (label, key, color) in enumerate(card_items):
            card = tk.Frame(self.summary_frame, bg="#2d2d2d", bd=1, relief=tk.RAISED)
            card.pack(side=tk.LEFT, padx=15, fill=tk.BOTH, expand=True)
            
            tk.Label(card, text=label, bg="#2d2d2d", fg="white", font=("Arial", 10)).pack(pady=5)
            lbl_val = tk.Label(card, text="0", bg="#2d2d2d", fg=color, font=("Arial", 18, "bold"))
            lbl_val.pack(pady=5)
            self.summary_cards[key] = lbl_val

        # 圖表區域
        self.charts_frame = tk.Frame(self.dashboard_tab, bg="#1e1e1e")
        self.charts_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        # 趨勢圖
        self.fig_trend, self.ax_trend = plt.subplots(figsize=(6, 4), dpi=100)
        self.fig_trend.patch.set_facecolor("#1e1e1e")
        self.ax_trend.set_facecolor("#1e1e1e")
        self.ax_trend.tick_params(colors="white")
        
        self.trend_canvas = FigureCanvasTkAgg(self.fig_trend, master=self.charts_frame)
        self.trend_canvas.get_tk_widget().pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)

        # 比例圖
        self.fig_ratio, self.ax_ratio = plt.subplots(figsize=(5, 4), dpi=100)
        self.fig_ratio.patch.set_facecolor("#1e1e1e")
        self.ax_ratio.set_facecolor("#1e1e1e")
        self.ax_ratio.tick_params(colors="white")
        
        self.ratio_canvas = FigureCanvasTkAgg(self.fig_ratio, master=self.charts_frame)
        self.ratio_canvas.get_tk_widget().pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)

        # 按鈕區域
        self.dash_btn_frame = tk.Frame(self.dashboard_tab, bg="#1e1e1e")
        self.dash_btn_frame.pack(pady=20)

        tk.Button(
            self.dash_btn_frame,
            text="Refresh Dashboard",
            command=self.refresh_dashboard,
            bg="#00adb5",
            fg="white",
            font=("Arial", 10, "bold"),
            width=20
        ).pack(side=tk.LEFT, padx=10)

        tk.Button(
            self.dash_btn_frame,
            text="Export Dashboard Report",
            command=lambda: self.export_report("pdf"),
            bg="#2d4059",
            fg="white",
            font=("Arial", 10, "bold"),
            width=20
        ).pack(side=tk.LEFT, padx=10)

    def refresh_dashboard(self):
        summary = build_dashboard_summary(self.event_logger.events)
        
        # 更新 Cards
        self.summary_cards["total_v"].config(text=str(summary["total_violations"]))
        self.summary_cards["m_helmet"].config(text=str(summary["missing_counts"].get("helmet", 0)))
        self.summary_cards["m_vest"].config(text=str(summary["missing_counts"].get("vest", 0)))
        self.summary_cards["m_mask"].config(text=str(summary["missing_counts"].get("mask", 0)))
        self.summary_cards["m_goggles"].config(text=str(summary["missing_counts"].get("goggles", 0)))

        # 更新趨勢圖
        self.ax_trend.clear()
        trend_data = summary["trend"]
        if trend_data["labels"]:
            self.ax_trend.plot(trend_data["labels"], trend_data["counts"], marker='o', color="#00adb5")
            self.ax_trend.set_title("Violation Trend", color="white")
            plt.setp(self.ax_trend.get_xticklabels(), rotation=45, ha="right")
        else:
            self.ax_trend.text(0.5, 0.5, "No violation events yet", color="white", ha="center")
        self.fig_trend.tight_layout()
        self.trend_canvas.draw()

        # 更新比例圖
        self.ax_ratio.clear()
        counts = summary["missing_counts"]
        labels = [k.capitalize() for k in counts.keys() if counts[k] > 0]
        values = [counts[k.lower()] for k in labels]
        
        if values:
            self.ax_ratio.bar(labels, values, color=["#00adb5", "#ff4b2b", "#f9ed69", "#b83b5e"])
            self.ax_ratio.set_title("PPE Missing Distribution", color="white")
        else:
            self.ax_ratio.text(0.5, 0.5, "No PPE missing data", color="white", ha="center")
        self.fig_ratio.tight_layout()
        self.ratio_canvas.draw()

    def update_chart(self):
        # 更新監控分頁的小圖表
        self.ax_mini.clear()
        labels = [k.capitalize() for k in self.stats["missing_counts"].keys()]
        values = list(self.stats["missing_counts"].values())

        self.ax_mini.bar(labels, values, color=["#00adb5", "#ff4b2b", "#f9ed69", "#b83b5e"])
        self.ax_mini.set_title("PPE 缺失統計", color="white", fontsize=10)
        self.ax_mini.tick_params(colors="white")

        self.fig_mini.tight_layout()
        self.chart_canvas_mini.draw()

    def change_model(self):
        path = filedialog.askopenfilename(filetypes=[("YOLO Model", "*.pt")])
        if path:
            success, msg = self.detector.load_model(path)
            if success:
                self.detector.reset_tracking()
                self.lbl_model.config(text=f"當前模型：{os.path.basename(path)}")
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
        image = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.canvas.imgtk = imgtk

    def add_event_to_ui(self, event_dict):
        self.stats["total_violations"] += 1
        for missing_item in event_dict.get("missing_list", []):
            if missing_item in self.stats["missing_counts"]:
                self.stats["missing_counts"][missing_item] += 1

        self.lbl_total_v.config(text=f"累計違規：{self.stats['total_violations']}")
        self.update_chart()

        self.tree.insert("", 0, values=(
            event_dict.get("timestamp", ""),
            event_dict.get("track_id", ""),
            event_dict.get("missing_items", ""),
            f"{event_dict.get('confidence', 0):.2f}",
            event_dict.get("screenshot_path", "")
        ))

        if len(self.tree.get_children()) > 1000:
            self.tree.delete(self.tree.get_children()[-1])

    def on_tree_double_click(self, event):
        selected = self.tree.selection()
        if not selected: return
        item = selected[0]
        screenshot_path = self.tree.item(item, "values")[4]
        if screenshot_path and os.path.exists(screenshot_path):
            if os.name == "nt": os.startfile(screenshot_path)
            else: os.system(f'xdg-open "{screenshot_path}"')
        else:
            messagebox.showwarning("提示", "找不到截圖檔案")

    def export_report(self, fmt):
        if not self.event_logger.events:
            messagebox.showwarning("提示", "目前沒有違規事件可供匯出")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reports/ppe_violation_report_{timestamp}.{fmt}"

        try:
            # 匯出前先生成 Dashboard 圖表
            self.save_dashboard_charts(timestamp)
            
            if fmt == "csv":
                path = self.event_logger.export_csv(filename)
            elif fmt == "xlsx":
                path = self.event_logger.export_excel(filename, self.stats)
            elif fmt == "pdf":
                path = self.event_logger.export_pdf(filename, self.stats)
            else:
                messagebox.showerror("錯誤", "不支援的匯出格式")
                return

            messagebox.showinfo("成功", f"報告已匯出至：{path}")
        except Exception as e:
            messagebox.showerror("錯誤", f"匯出失敗：{str(e)}")

    def save_dashboard_charts(self, timestamp):
        summary = build_dashboard_summary(self.event_logger.events)
        
        # 儲存趨勢圖
        fig_t, ax_t = plt.subplots(figsize=(8, 4))
        trend_data = summary["trend"]
        if trend_data["labels"]:
            ax_t.plot(trend_data["labels"], trend_data["counts"], marker='o')
            ax_t.set_title("Violation Trend")
            plt.setp(ax_t.get_xticklabels(), rotation=45, ha="right")
        else:
            ax_t.text(0.5, 0.5, "No Data", ha="center")
        fig_t.tight_layout()
        fig_t.savefig(f"reports/charts/trend_{timestamp}.png")
        plt.close(fig_t)

        # 儲存比例圖
        fig_r, ax_r = plt.subplots(figsize=(6, 4))
        counts = summary["missing_counts"]
        labels = [k.capitalize() for k in counts.keys() if counts[k] > 0]
        values = [counts[k.lower()] for k in labels]
        if values:
            ax_r.bar(labels, values)
            ax_r.set_title("PPE Missing Distribution")
        else:
            ax_r.text(0.5, 0.5, "No Data", ha="center")
        fig_r.tight_layout()
        fig_r.savefig(f"reports/charts/ratio_{timestamp}.png")
        plt.close(fig_r)

    def reset_detection_state(self):
        self.stats = {
            "total_violations": 0,
            "missing_counts": {"helmet": 0, "vest": 0, "goggles": 0, "mask": 0}
        }
        self.detector.reset_tracking()
        self.event_logger.clear_events()
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.lbl_total_v.config(text="累計違規：0")
        self.update_chart()

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
        self.reset_detection_state()
        self.enabled_items = {key: var.get() for key, var in self.check_vars.items()}
        self.vid = cv2.VideoCapture(source)
        if not self.vid.isOpened():
            messagebox.showerror("錯誤", "無法開啟影像來源")
            return
        self.running = True
        self.stop_event.clear()
        self.btn_upload.config(state=tk.DISABLED)
        self.btn_camera.config(state=tk.DISABLED)
        self.worker = threading.Thread(target=self.detection_worker, daemon=True)
        self.worker.start()

    def detection_worker(self):
        last_cap_time = 0
        frame_number = 0
        while not self.stop_event.is_set():
            ret, frame = self.vid.read()
            if not ret: break
            frame_height, frame_width = frame.shape[:2]
            scale = min(800 / frame_width, 500 / frame_height)
            new_w, new_h = int(frame_width * scale), int(frame_height * scale)
            display_frame = cv2.resize(frame, (new_w, new_h))
            canvas_frame = cv2.copyMakeBorder(display_frame, 0, 500 - new_h, 0, 800 - new_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            targets = [k for k, enabled in self.enabled_items.items() if enabled]
            annotated, info = self.detector.detect(canvas_frame, targets, source_name=self.source_name, frame_number=frame_number)
            self.result_queue.put(("FRAME", annotated))
            if info.get("violation_detected"):
                for event_data in info.get("new_events", []):
                    curr_time = time.time()
                    screenshot_path = ""
                    if curr_time - last_cap_time > self.detector.violation_cooldown:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        screenshot_path = f"violations/v_{ts}_{event_data.get('track_id', 'unknown')}.jpg"
                        cv2.imwrite(screenshot_path, annotated)
                        last_cap_time = curr_time
                    event = ViolationEvent(
                        timestamp=event_data.get("timestamp", ""),
                        source=event_data.get("source", self.source_name),
                        track_id=event_data.get("track_id", ""),
                        person_count=event_data.get("person_count", 0),
                        missing_items=event_data.get("missing_items", ""),
                        screenshot_path=screenshot_path,
                        confidence=event_data.get("confidence", 0),
                        bbox=event_data.get("bbox", "")
                    )
                    self.event_logger.add_event(event)
                    event_data["screenshot_path"] = screenshot_path
                    self.result_queue.put(("EVENT", event_data))
            frame_number += 1
            time.sleep(0.01)
        self.result_queue.put(("STOP", None))

    def handle_stop(self):
        self.running = False
        if self.vid: self.vid.release(); self.vid = None
        self.btn_upload.config(state=tk.NORMAL)
        self.btn_camera.config(state=tk.NORMAL)

    def stop_and_report(self):
        self.stop_event.set()
        if self.worker: self.worker.join(timeout=5.0); self.worker = None
        self.handle_stop()
        self.detector.save_violation_log()
        messagebox.showinfo("提示", "偵測已停止。您可以切換至 Dashboard 分頁查看統計結果。")

    def on_closing(self):
        self.stop_event.set()
        if self.worker: self.worker.join(timeout=5.0)
        if self.vid: self.vid.release()
        self.detector.save_violation_log()
        self.window.destroy()

    def show_demo_info(self):
        messagebox.showinfo("Demo Mode", "系統正以 Demo 模式運行，將模擬 PPE 缺失事件以供功能展示。")

if __name__ == "__main__":
    root = tk.Tk()
    app = HelmetDetectionApp(root, "PPE 智慧監控系統 Pro")
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
