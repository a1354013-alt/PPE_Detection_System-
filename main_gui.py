import argparse
import os
import queue
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

from event_logger import EventLogger, ViolationEvent
from helmet_detector import HelmetDetector


class HelmetDetectionApp:
    def __init__(self, window, window_title, demo_mode=False, model_path=None):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1400x950")
        self.window.configure(bg="#1e1e1e")

        self.detector = HelmetDetector(model_path=model_path or "yolov8n.pt", demo_mode=demo_mode)
        self.event_logger = EventLogger()
        self.demo_mode = demo_mode
        self.running = False
        self.worker = None
        self.stop_event = threading.Event()
        self.result_queue = queue.Queue()
        self.vid = None
        self.source_name = "unknown"
        self.enabled_items = {}
        self.finalize_completed = False
        self.last_finalize_result = None

        self.stats = {
            "total_violations": 0,
            "missing_counts": {
                "helmet": 0,
                "vest": 0,
                "goggles": 0,
                "mask": 0,
            },
        }

        os.makedirs("reports", exist_ok=True)
        os.makedirs("violations", exist_ok=True)

        self.setup_ui()
        self.update_model_info()
        self.check_queue()

        if self.demo_mode:
            self.window.after(500, self.show_demo_info)

    def setup_ui(self):
        self.top_frame = tk.Frame(self.window, bg="#1e1e1e")
        self.top_frame.pack(pady=10, fill=tk.X)

        tk.Label(
            self.top_frame,
            text="PPE Detection System Pro",
            font=("Arial", 22, "bold"),
            bg="#1e1e1e",
            fg="#00adb5",
        ).pack()

        self.main_frame = tk.Frame(self.window, bg="#1e1e1e")
        self.main_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        self.left_panel = tk.Frame(self.main_frame, bg="#1e1e1e")
        self.left_panel.pack(side=tk.LEFT, padx=20, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.left_panel, width=800, height=500, bg="#000000", highlightthickness=0)
        self.canvas.pack(pady=5)

        self.tree_frame = ttk.LabelFrame(self.left_panel, text="Violation Events")
        self.tree_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        columns = ("time", "track_id", "missing", "conf", "screenshot")
        self.tree = ttk.Treeview(self.tree_frame, columns=columns, show="headings", height=8)
        self.tree.heading("time", text="Time")
        self.tree.heading("track_id", text="Track ID")
        self.tree.heading("missing", text="Missing PPE")
        self.tree.heading("conf", text="Conf")
        self.tree.heading("screenshot", text="Screenshot")
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

        self.model_frame = ttk.LabelFrame(self.right_frame, text="Model Information")
        self.model_frame.pack(fill=tk.X, pady=5)

        self.lbl_model_name = tk.Label(self.model_frame, bg="#1e1e1e", fg="white", font=("Arial", 10, "bold"), justify=tk.LEFT)
        self.lbl_model_name.pack(pady=2, anchor=tk.W)

        self.lbl_model_status = tk.Label(self.model_frame, bg="#1e1e1e", fg="white", font=("Arial", 9), justify=tk.LEFT, wraplength=360)
        self.lbl_model_status.pack(pady=2, anchor=tk.W)

        self.lbl_model_caps = tk.Label(self.model_frame, bg="#1e1e1e", fg="white", font=("Arial", 9), justify=tk.LEFT, wraplength=360)
        self.lbl_model_caps.pack(pady=2, anchor=tk.W)

        self.lbl_model_warning = tk.Label(self.model_frame, bg="#1e1e1e", fg="#f9ed69", font=("Arial", 9), justify=tk.LEFT, wraplength=360)
        self.lbl_model_warning.pack(pady=2, anchor=tk.W)

        tk.Button(
            self.model_frame,
            text="Load Model (.pt)",
            command=self.change_model,
            bg="#393e46",
            fg="white",
        ).pack(fill=tk.X, pady=(4, 0))

        self.fig, self.ax = plt.subplots(figsize=(4, 3), dpi=100)
        self.fig.patch.set_facecolor("#1e1e1e")
        self.ax.set_facecolor("#1e1e1e")
        self.ax.tick_params(colors="white")
        self.chart_canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.chart_canvas.get_tk_widget().pack(fill=tk.X, pady=5)

        self.stats_frame = ttk.LabelFrame(self.right_frame, text="Statistics")
        self.stats_frame.pack(fill=tk.X, pady=10)

        self.lbl_total_v = tk.Label(
            self.stats_frame,
            text="Total violations: 0",
            font=("Arial", 14, "bold"),
            bg="#1e1e1e",
            fg="#ff4b2b",
        )
        self.lbl_total_v.pack(pady=5)

        self.export_frame = ttk.LabelFrame(self.right_frame, text="Export Report")
        self.export_frame.pack(fill=tk.X, pady=10)

        tk.Button(
            self.export_frame,
            text="Export CSV",
            command=lambda: self.export_report("csv"),
            bg="#2d4059",
            fg="white",
        ).pack(fill=tk.X, pady=2)
        tk.Button(
            self.export_frame,
            text="Export Excel",
            command=lambda: self.export_report("xlsx"),
            bg="#2d4059",
            fg="white",
        ).pack(fill=tk.X, pady=2)
        tk.Button(
            self.export_frame,
            text="Export PDF",
            command=lambda: self.export_report("pdf"),
            bg="#2d4059",
            fg="white",
        ).pack(fill=tk.X, pady=2)

        self.bottom_frame = tk.Frame(self.window, bg="#1e1e1e")
        self.bottom_frame.pack(pady=10, fill=tk.X)

        self.check_frame = tk.Frame(self.bottom_frame, bg="#1e1e1e")
        self.check_frame.pack()
        self.check_vars = {item: tk.BooleanVar(value=(item == "helmet")) for item in ["helmet", "vest", "goggles", "mask"]}

        for item, var in self.check_vars.items():
            checkbox = tk.Checkbutton(
                self.check_frame,
                text=item.capitalize(),
                variable=var,
                bg="#1e1e1e",
                fg="white",
                selectcolor="#393e46",
                activebackground="#1e1e1e",
                command=self.validate_model_support,
            )
            checkbox.pack(side=tk.LEFT, padx=15)

        self.btn_frame = tk.Frame(self.bottom_frame, bg="#1e1e1e")
        self.btn_frame.pack(pady=10)

        button_style = {"font": ("Arial", 10, "bold"), "width": 15, "height": 2, "bd": 0, "cursor": "hand2"}
        self.btn_upload = tk.Button(self.btn_frame, text="Open File", command=self.open_file, bg="#393e46", fg="white", **button_style)
        self.btn_upload.grid(row=0, column=0, padx=10)
        self.btn_camera = tk.Button(self.btn_frame, text="Open Camera", command=self.open_camera, bg="#00adb5", fg="white", **button_style)
        self.btn_camera.grid(row=0, column=1, padx=10)
        self.btn_stop = tk.Button(self.btn_frame, text="Stop && Finalize", command=self.stop_and_report, bg="#ff4b2b", fg="white", **button_style)
        self.btn_stop.grid(row=0, column=2, padx=10)

        self.status_var = tk.StringVar(value="Ready.")
        self.status_label = tk.Label(self.window, textvariable=self.status_var, bg="#1e1e1e", fg="#cfd8dc", anchor="w")
        self.status_label.pack(fill=tk.X, padx=20, pady=(0, 10))

        self.update_chart()

    def set_status(self, text):
        self.status_var.set(text)

    def show_demo_info(self):
        messagebox.showinfo(
            "Demo Mode",
            "Demo Mode 會模擬事件列表、報告匯出與熱力圖流程，\n不代表真實模型推論結果。",
        )

    def update_model_info(self):
        snapshot = self.detector.get_model_status_snapshot()
        caps = snapshot["capabilities"]
        capability_lines = [
            f"Loaded: {'Yes' if snapshot['loaded'] else 'No'}",
            f"Model Path: {snapshot['path'] or 'N/A'}",
            f"Supports person: {'Yes' if caps.get('person') else 'No'}",
            f"Supports helmet: {'Yes' if caps.get('helmet') else 'No'}",
            f"Supports vest: {'Yes' if caps.get('vest') else 'No'}",
            f"Supports mask: {'Yes' if caps.get('mask') else 'No'}",
            f"Supports goggles: {'Yes' if caps.get('goggles') else 'No'}",
            f"Demo Mode: {'Yes' if snapshot['demo_mode'] else 'No'}",
        ]

        self.lbl_model_name.config(text=f"Current Model: {snapshot['display_name']}")
        self.lbl_model_status.config(text=f"Status: {snapshot['status_message']}")
        self.lbl_model_caps.config(text="\n".join(capability_lines))
        self.lbl_model_warning.config(text=snapshot["warning"])

    def change_model(self):
        path = filedialog.askopenfilename(filetypes=[("YOLO Model", "*.pt")])
        if not path:
            return

        success, message = self.detector.load_model(path)
        if success:
            self.detector.reset_tracking()
            self.update_model_info()
            self.validate_model_support(show_message=False)
            messagebox.showinfo("Model", message)
        else:
            self.update_model_info()
            messagebox.showerror("Model", message)

    def validate_model_support(self, show_message=True):
        if self.demo_mode:
            self.update_model_info()
            return True

        model_classes = set(self.detector.get_model_classes())
        unsupported = [item for item, var in self.check_vars.items() if var.get() and item not in model_classes]

        self.update_model_info()
        if unsupported and show_message:
            messagebox.showwarning(
                "Model Capability",
                "目前模型不支援以下選取項目：\n"
                f"{', '.join(unsupported)}\n\n"
                "若目前使用的是 yolov8n.pt / COCO 模型，請改載入 PPE 模型或使用 Demo Mode。",
            )
        return not unsupported

    def update_chart(self):
        self.ax.clear()
        labels = [name.capitalize() for name in self.stats["missing_counts"].keys()]
        values = list(self.stats["missing_counts"].values())
        self.ax.bar(labels, values, color=["#00adb5", "#ff4b2b", "#f9ed69", "#b83b5e"])
        self.ax.set_title("Missing PPE Statistics", color="white", fontsize=10)
        self.ax.tick_params(colors="white")
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
                    self.handle_stop(data)
        except queue.Empty:
            pass
        finally:
            self.window.after(30, self.check_queue)

    def render_frame(self, frame):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.canvas.imgtk = imgtk

    def add_event_to_ui(self, event_dict):
        self.stats["total_violations"] += 1
        for missing_item in event_dict.get("missing_list", []):
            if missing_item in self.stats["missing_counts"]:
                self.stats["missing_counts"][missing_item] += 1

        self.lbl_total_v.config(text=f"Total violations: {self.stats['total_violations']}")
        self.update_chart()

        self.tree.insert(
            "",
            0,
            values=(
                event_dict.get("timestamp", ""),
                event_dict.get("track_id", ""),
                event_dict.get("missing_items", ""),
                f"{event_dict.get('confidence', 0):.2f}",
                event_dict.get("screenshot_path", ""),
            ),
        )

        if len(self.tree.get_children()) > 1000:
            self.tree.delete(self.tree.get_children()[-1])

    def on_tree_double_click(self, _event):
        selected = self.tree.selection()
        if not selected:
            return

        screenshot_path = self.tree.item(selected[0], "values")[4]
        if screenshot_path and os.path.exists(screenshot_path):
            if os.name == "nt":
                os.startfile(screenshot_path)
            else:
                os.system(f'xdg-open "{screenshot_path}"')
            return

        messagebox.showwarning("Screenshot", "Screenshot file not found.")

    def export_report(self, fmt):
        if not self.event_logger.events:
            messagebox.showwarning("Export", "No violation events available for export.")
            return

        os.makedirs("reports", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join("reports", f"ppe_violation_report_{timestamp}.{fmt}")
        processing_summary = self.detector.get_processing_summary_data()

        try:
            if fmt == "csv":
                path = self.event_logger.export_csv(filepath)
            elif fmt == "xlsx":
                path = self.event_logger.export_excel(filepath, self.stats, processing_summary=processing_summary)
            elif fmt == "pdf":
                path = self.event_logger.export_pdf(filepath, self.stats, processing_summary=processing_summary)
            else:
                messagebox.showerror("Export", "Unsupported export format.")
                return

            messagebox.showinfo("Export", f"Report saved successfully:\n{os.path.abspath(path)}")
        except Exception as exc:
            messagebox.showerror("Export", f"Failed to export report:\n{exc}")

    def reset_detection_state(self):
        self.stats = {
            "total_violations": 0,
            "missing_counts": {"helmet": 0, "vest": 0, "goggles": 0, "mask": 0},
        }
        self.finalize_completed = False
        self.last_finalize_result = None
        self.detector.reset_tracking()
        self.event_logger.clear_events()
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.clear_status_labels()
        self.clear_previous_outputs()
        self.update_model_info()

    def clear_status_labels(self):
        self.lbl_total_v.config(text="Total violations: 0")
        self.update_chart()

    def clear_previous_outputs(self):
        if not os.path.exists("violations"):
            return

        for filename in os.listdir("violations"):
            filepath = os.path.join("violations", filename)
            if os.path.isfile(filepath):
                try:
                    os.remove(filepath)
                except OSError:
                    pass

    def open_file(self):
        path = filedialog.askopenfilename()
        if not path:
            return

        self.source_name = os.path.basename(path)
        self.start_detection(path)

    def open_camera(self):
        self.source_name = "camera_0"
        self.start_detection(0)

    def start_detection(self, source):
        if self.running:
            return

        self.reset_detection_state()
        self.enabled_items = {key: var.get() for key, var in self.check_vars.items()}
        self.vid = cv2.VideoCapture(source)
        if not self.vid.isOpened():
            messagebox.showerror("Source", "Unable to open the selected source.")
            self.vid = None
            return

        self.detector.processing_start_time = time.time()
        self.detector.processing_end_time = None
        self.detector.video_name = self.source_name
        self.running = True
        self.stop_event.clear()
        self.btn_upload.config(state=tk.DISABLED)
        self.btn_camera.config(state=tk.DISABLED)
        self.set_status(f"Running detection on {self.source_name}...")

        self.worker = threading.Thread(target=self.detection_worker, daemon=True)
        self.worker.start()

    def detection_worker(self):
        last_capture_time_by_track = {}
        frame_number = 0

        while not self.stop_event.is_set():
            ret, frame = self.vid.read()
            if not ret:
                break

            frame_height, frame_width = frame.shape[:2]
            scale = min(800 / frame_width, 500 / frame_height)
            new_w = int(frame_width * scale)
            new_h = int(frame_height * scale)
            display_frame = cv2.resize(frame, (new_w, new_h))
            canvas_frame = cv2.copyMakeBorder(
                display_frame,
                0,
                500 - new_h,
                0,
                800 - new_w,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )

            targets = [key for key, enabled in self.enabled_items.items() if enabled]
            annotated, info = self.detector.detect(
                canvas_frame,
                targets,
                source_name=self.source_name,
                frame_number=frame_number,
            )
            self.result_queue.put(("FRAME", annotated))

            if info.get("violation_detected"):
                for event_data in info.get("new_events", []):
                    current_time = time.time()
                    screenshot_path = ""
                    track_id = event_data.get("track_id", "unknown")
                    last_capture_time = last_capture_time_by_track.get(track_id, 0)

                    if current_time - last_capture_time > self.detector.violation_cooldown:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        screenshot_path = f"violations/v_{timestamp}_{track_id}.jpg"
                        cv2.imwrite(screenshot_path, annotated)
                        last_capture_time_by_track[track_id] = current_time

                    event = ViolationEvent(
                        timestamp=event_data.get("timestamp", ""),
                        source=event_data.get("source", self.source_name),
                        track_id=event_data.get("track_id", ""),
                        person_count=event_data.get("person_count", 0),
                        missing_items=event_data.get("missing_items", ""),
                        screenshot_path=screenshot_path,
                        confidence=event_data.get("confidence", 0),
                        bbox=event_data.get("bbox", ""),
                    )
                    self.event_logger.add_event(event)
                    event_data["screenshot_path"] = screenshot_path
                    self.result_queue.put(("EVENT", event_data))

            frame_number += 1
            time.sleep(0.01)

        stop_reason = "manual_stop" if self.stop_event.is_set() else "natural_end"
        self.result_queue.put(("STOP", {"reason": stop_reason, "auto_report": True}))

    def handle_stop(self, stop_data=None):
        stop_data = stop_data or {"reason": "natural_end", "auto_report": True}
        self.finalize_detection(auto_report=stop_data.get("auto_report", True), reason=stop_data.get("reason", "natural_end"))

    def finalize_detection(self, auto_report=True, reason="manual_stop", notify=True):
        if self.finalize_completed:
            return self.last_finalize_result

        self.finalize_completed = True
        self.stop_event.set()
        self.running = False

        if not self.detector.processing_end_time:
            self.detector.processing_end_time = time.time()

        if self.vid:
            self.vid.release()
            self.vid = None

        self.btn_upload.config(state=tk.NORMAL)
        self.btn_camera.config(state=tk.NORMAL)

        violations_path = None
        heatmap_path = None
        if auto_report:
            if self.detector.save_violation_log():
                violations_path = os.path.abspath("violations/violations.csv")

            if self.detector.violation_coords:
                heatmap = self.detector.generate_heatmap((500, 800, 3))
                heatmap_path = os.path.abspath(os.path.join("reports", "violation_heatmap.jpg"))
                cv2.imwrite(heatmap_path, heatmap)

        summary = self.detector.get_processing_summary_data()
        status_message = "Detection stopped." if reason == "manual_stop" else "Detection completed."
        self.set_status(f"{status_message} Source: {summary['source_name']} | Violations: {summary['total_violations']}")

        message_lines = [status_message, f"Source: {summary['source_name']}", f"Total violations: {summary['total_violations']}"]
        if violations_path:
            message_lines.append(f"Violation log: {violations_path}")
        if heatmap_path:
            message_lines.append(f"Heatmap: {heatmap_path}")
        if self.demo_mode:
            message_lines.append("Demo Mode was enabled for this run.")
        if not violations_path and not heatmap_path:
            message_lines.append("No violation artifacts were generated.")

        self.last_finalize_result = {
            "reason": reason,
            "summary": summary,
            "violations_path": violations_path,
            "heatmap_path": heatmap_path,
        }

        if notify:
            messagebox.showinfo("Detection Finished", "\n".join(message_lines))

        return self.last_finalize_result

    def stop_and_report(self):
        self.set_status("Stopping detection and finalizing artifacts...")
        self.stop_event.set()

        if self.worker is not None:
            self.worker.join(timeout=5.0)
            self.worker = None

        self.finalize_detection(auto_report=True, reason="manual_stop")

    def on_closing(self):
        self.stop_event.set()
        if self.worker is not None:
            self.worker.join(timeout=5.0)
            self.worker = None
        self.finalize_detection(auto_report=True, reason="manual_stop", notify=False)
        self.window.destroy()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="PPE Detection System GUI")
    parser.add_argument("--demo", action="store_true", help="Run the GUI in Demo Mode.")
    parser.add_argument("--model", help="Path to a YOLO/PPE model file.", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = tk.Tk()
    app = HelmetDetectionApp(
        root,
        "PPE Detection System Pro",
        demo_mode=args.demo,
        model_path=args.model,
    )
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
