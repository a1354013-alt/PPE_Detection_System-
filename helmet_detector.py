import csv
import json
import os
import random
import time
from collections import Counter, deque
from datetime import datetime

import cv2
import numpy as np


DEMO_RANDOM_SEED = 42
random.seed(DEMO_RANDOM_SEED)


class HelmetDetector:
    """PPE detector with tracking, temporal smoothing, cooldown, and demo mode."""

    PPE_ITEMS = ("helmet", "vest", "mask", "goggles")

    def __init__(self, model_path="yolov8n.pt", conf=0.4, iou=0.45, config_path="config.json", demo_mode=False):
        self.model = None
        self.model_path = model_path
        self.config_path = config_path
        self.demo_mode = demo_mode
        self.model_loaded = False
        self.model_status_message = "Model not loaded."
        self.model_capabilities = self._empty_model_capabilities()
        self.config = self.load_config(config_path)

        self.conf = self.config.get("confidence_threshold", conf)
        self.iou = self.config.get("iou_threshold", iou)
        self.temporal_frames = self.config.get("temporal_frames", 3)
        self.cooldown_seconds = self.config.get("cooldown_seconds", 2)

        os.makedirs("reports", exist_ok=True)
        os.makedirs("violations", exist_ok=True)

        self.class_map = {
            "hardhat": "helmet",
            "head_helmet": "helmet",
            "safety_helmet": "helmet",
            "helmet": "helmet",
            "safety_vest": "vest",
            "reflective_vest": "vest",
            "vest": "vest",
            "goggles": "goggles",
            "eye_protection": "goggles",
            "mask": "mask",
            "face_mask": "mask",
        }

        if not demo_mode or (model_path and os.path.exists(model_path)):
            self.load_model(model_path)
        else:
            self.model_status_message = "Demo Mode active. No PPE model loaded."

        self.person_states = {}
        self.person_buffers = {}
        self.counted_violations = {}
        self.violation_cooldown = float(self.cooldown_seconds)
        self.violation_log = []
        self.violation_coords = deque(maxlen=5000)
        self.demo_missing_pattern = ["helmet", "vest"]
        self.frame_count = 0
        self.start_time = time.time()
        self.processing_start_time = None
        self.processing_end_time = None
        self.total_frames_processed = 0
        self.video_name = ""

    def _empty_model_capabilities(self):
        return {
            "person": False,
            "helmet": False,
            "vest": False,
            "mask": False,
            "goggles": False,
            "supported_items": [],
            "is_ppe_model": False,
        }

    def load_config(self, config_path):
        default_config = {
            "confidence_threshold": 0.5,
            "iou_threshold": 0.3,
            "temporal_frames": 3,
            "cooldown_seconds": 2,
        }

        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as file_obj:
                    return json.load(file_obj)
            except (json.JSONDecodeError, IOError):
                return default_config

        return default_config

    def load_model(self, path):
        self.model_path = path
        try:
            from ultralytics import YOLO

            loaded_model = YOLO(path)
            self.model = loaded_model
            self.model_loaded = True
            self._check_model_capability()
            self.model_status_message = f"Model loaded successfully: {path}"
            return True, self.model_status_message
        except ImportError:
            self.model = None
            self.model_loaded = False
            self.model_capabilities = self._empty_model_capabilities()
            if self.demo_mode:
                self.model_status_message = "Demo Mode active. ultralytics is not installed."
                return True, self.model_status_message

            self.model_status_message = "ultralytics is not installed. Run: pip install ultralytics"
            return False, self.model_status_message
        except Exception as exc:
            self.model = None
            self.model_loaded = False
            self.model_capabilities = self._empty_model_capabilities()
            self.model_status_message = f"Failed to load model: {exc}"
            return False, self.model_status_message

    def _check_model_capability(self):
        if not self.model:
            self.model_capabilities = self._empty_model_capabilities()
            return

        available_classes = set()
        for name in self.model.names.values():
            mapped = self.class_map.get(str(name).lower(), str(name).lower())
            available_classes.add(mapped)

        self.model_capabilities = {
            "person": "person" in available_classes,
            "helmet": "helmet" in available_classes,
            "vest": "vest" in available_classes,
            "mask": "mask" in available_classes,
            "goggles": "goggles" in available_classes,
            "supported_items": sorted(available_classes),
            "is_ppe_model": any(item in available_classes for item in self.PPE_ITEMS),
        }

        print("\n=== Model Capability ===")
        print(f"person: {'yes' if self.model_capabilities['person'] else 'no'}")
        for item in self.PPE_ITEMS:
            print(f"{item}: {'yes' if self.model_capabilities[item] else 'no'}")
        if self.demo_mode:
            print("running in demo mode")
        print("========================\n")

    def get_model_classes(self):
        return list(self.model_capabilities.get("supported_items", []))

    def get_model_display_name(self):
        return os.path.basename(self.model_path) if self.model_path else "N/A"

    def get_model_status_snapshot(self):
        warning = ""
        if self.model_loaded and self.model_capabilities.get("person") and not self.model_capabilities.get("is_ppe_model"):
            warning = (
                "目前模型不是 PPE 專用模型，僅能偵測 person；若要偵測 helmet / vest / mask / "
                "goggles，請載入自訂 PPE 模型或使用 Demo Mode。"
            )
        elif not self.model_loaded and self.demo_mode:
            warning = "目前未載入 PPE 模型，Demo Mode 只會模擬展示流程，不代表真實模型推論結果。"

        return {
            "loaded": self.model_loaded,
            "path": self.model_path,
            "display_name": self.get_model_display_name(),
            "status_message": self.model_status_message,
            "capabilities": dict(self.model_capabilities),
            "warning": warning,
            "demo_mode": self.demo_mode,
        }

    def get_person_zones(self, person_box):
        px1, py1, px2, py2 = person_box
        person_height = py2 - py1

        head_h = int(person_height * 0.25)
        face_upper_h = int(person_height * 0.40)
        face_lower_y1 = int(py1 + person_height * 0.40)
        face_lower_y2 = int(py1 + person_height * 0.60)
        torso_y1 = int(py1 + person_height * 0.30)
        torso_y2 = int(py1 + person_height * 0.80)

        return {
            "head": (px1, py1, px2, py1 + head_h),
            "face_upper": (px1, py1, px2, py1 + face_upper_h),
            "face_lower": (px1, face_lower_y1, px2, face_lower_y2),
            "torso": (px1, torso_y1, px2, torso_y2),
        }

    def is_overlapping_with_zone(self, box_item, zone_box):
        ix1, iy1, ix2, iy2 = box_item
        zx1, zy1, zx2, zy2 = zone_box

        center_x = (ix1 + ix2) / 2
        center_y = (iy1 + iy2) / 2
        return zx1 <= center_x <= zx2 and zy1 <= center_y <= zy2

    def run_detection_with_tracking(self, frame):
        if self.model is None:
            return [], False

        try:
            results = self.model.track(frame, conf=self.conf, iou=self.iou, persist=True, verbose=False)
            return results, True
        except Exception:
            results = self.model(frame, conf=self.conf, iou=self.iou, verbose=False)
            return results, False

    def build_fallback_track_id(self, bbox):
        x1, y1, x2, y2 = bbox
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        return f"fallback_{center_x // 50}_{center_y // 50}"

    def cleanup_stale_tracks(self, current_time, ttl=10):
        stale_ids = [
            track_id
            for track_id, state in self.person_states.items()
            if current_time - state.get("last_seen", 0) > ttl
        ]

        for track_id in stale_ids:
            del self.person_states[track_id]
            keys_to_delete = [key for key in list(self.counted_violations.keys()) if key[0] == track_id]
            for key in keys_to_delete:
                del self.counted_violations[key]
            self.person_buffers.pop(track_id, None)

    def _check_ppe_missing(self, p_box, items, target_items, supported_classes):
        zones = self.get_person_zones(p_box)
        person_missing = []

        for target in target_items:
            if target not in supported_classes:
                continue

            if target == "helmet":
                zone = zones["head"]
            elif target == "goggles":
                zone = zones["face_upper"]
            elif target == "mask":
                zone = zones["face_lower"]
            else:
                zone = zones["torso"]

            found = any(self.is_overlapping_with_zone(item_box, zone) for item_box in items.get(target, []))
            if not found:
                person_missing.append(target)

        return person_missing

    def _should_report_event(self, track_id, stable_missing, current_time):
        if not stable_missing:
            return False, []

        event_key = (track_id, tuple(sorted(stable_missing)))
        last_count_time = self.counted_violations.get(event_key, 0)

        if current_time - last_count_time >= self.violation_cooldown:
            return True, [event_key]
        return False, []

    def detect(self, frame, target_items, source_name="unknown", frame_number=0):
        if self.demo_mode or self.model is None:
            return self._detect_demo(frame, target_items, source_name, frame_number)

        current_time = time.time()
        self.cleanup_stale_tracks(current_time)

        results, is_tracking = self.run_detection_with_tracking(frame)
        if not results:
            return frame, {
                "person_count": 0,
                "violation_detected": False,
                "missing_items": [],
                "stable_violations": [],
                "new_events": [],
                "fps": 0,
            }

        result = results[0]
        names = result.names

        persons = []
        items = {target: [] for target in target_items}

        for box in result.boxes:
            cls_id = int(box.cls[0])
            raw_label = str(names[cls_id]).lower()
            mapped_label = self.class_map.get(raw_label, raw_label)

            coords = box.xyxy[0].tolist()
            conf = float(box.conf[0])

            if is_tracking and box.id is not None:
                track_id = str(int(box.id[0]))
            else:
                track_id = self.build_fallback_track_id(coords)

            if mapped_label == "person":
                persons.append((coords, track_id, conf))
            elif mapped_label in target_items:
                items[mapped_label].append(coords)

        annotated_frame = result.plot()
        supported_classes = {self.class_map.get(str(name).lower(), str(name).lower()) for name in names.values()}

        frame_violations = []
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for p_box, track_id, person_conf in persons:
            person_missing = self._check_ppe_missing(p_box, items, target_items, supported_classes)

            self.person_states[track_id] = {
                "last_seen": current_time,
                "missing_items": set(person_missing),
                "bbox": p_box,
                "confidence": person_conf,
            }

            if track_id not in self.person_buffers:
                self.person_buffers[track_id] = deque(maxlen=self.temporal_frames)
            self.person_buffers[track_id].append(frozenset(person_missing))

            if len(self.person_buffers[track_id]) >= self.temporal_frames:
                recent_states = list(self.person_buffers[track_id])
                state_counts = Counter(recent_states)
                most_common_state, count = state_counts.most_common(1)[0]

                if most_common_state and count >= self.temporal_frames - 1:
                    stable_missing = list(most_common_state)
                    frame_violations.append((track_id, stable_missing, p_box, person_conf))

        new_event_data = []
        stable_violations_output = []

        for track_id, stable_missing, p_box, person_conf in frame_violations:
            should_report, cooldown_keys = self._should_report_event(track_id, stable_missing, current_time)
            if not should_report:
                continue

            for key in cooldown_keys:
                self.counted_violations[key] = current_time

            bbox_str = (
                f"x={int(p_box[0])},"
                f"y={int(p_box[1])},"
                f"w={int(p_box[2] - p_box[0])},"
                f"h={int(p_box[3] - p_box[1])}"
            )

            center_x = (p_box[0] + p_box[2]) / 2
            center_y = (p_box[1] + p_box[3]) / 2
            event = {
                "timestamp": timestamp_str,
                "frame": frame_number,
                "source": source_name,
                "track_id": track_id,
                "person_count": len(persons),
                "missing_items": ", ".join(stable_missing),
                "confidence": person_conf,
                "bbox": bbox_str,
                "center_x": center_x,
                "center_y": center_y,
                "missing_list": stable_missing,
                "bbox_raw": p_box,
                "is_demo": False,
            }

            new_event_data.append(event)
            self.violation_log.append(event)
            stable_violations_output.append((track_id, stable_missing, p_box))
            self.violation_coords.append((center_x, center_y))

            cv2.putText(
                annotated_frame,
                f"ID:{track_id} MISSING: {', '.join(stable_missing)}",
                (int(p_box[0]), int(p_box[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        all_missing = [item for _, missing_items, _ in stable_violations_output for item in missing_items]
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        self.total_frames_processed += 1

        return annotated_frame, {
            "person_count": len(persons),
            "violation_detected": len(frame_violations) > 0,
            "missing_items": list(set(all_missing)),
            "stable_violations": stable_violations_output,
            "new_events": new_event_data,
            "fps": fps,
        }

    def generate_heatmap(self, shape):
        heatmap = np.zeros(shape[:2], dtype=np.float32)
        for x_coord, y_coord in list(self.violation_coords):
            cv2.circle(heatmap, (int(x_coord), int(y_coord)), 30, 1, -1)

        heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)

    def save_violation_log(self, filepath="violations/violations.csv"):
        if not self.violation_log:
            return False

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", newline="", encoding="utf-8") as file_obj:
            writer = csv.writer(file_obj)
            writer.writerow([
                "timestamp",
                "frame",
                "source",
                "track_id",
                "person_count",
                "missing_items",
                "confidence",
                "bbox",
                "center_x",
                "center_y",
            ])

            for entry in self.violation_log:
                writer.writerow([
                    entry.get("timestamp", ""),
                    entry.get("frame", ""),
                    entry.get("source", ""),
                    entry.get("track_id", ""),
                    entry.get("person_count", ""),
                    entry.get("missing_items", ""),
                    entry.get("confidence", ""),
                    entry.get("bbox", ""),
                    entry.get("center_x", ""),
                    entry.get("center_y", ""),
                ])

        return True

    def reset_tracking(self):
        self.counted_violations = {}
        self.person_states = {}
        self.violation_coords.clear()
        self.person_buffers.clear()
        self.violation_log.clear()
        self.frame_count = 0
        self.start_time = time.time()
        self.total_frames_processed = 0
        self.processing_start_time = None
        self.processing_end_time = None
        self.video_name = ""

    def get_processing_summary_data(self):
        processing_time = 0
        avg_fps = 0
        if self.processing_start_time and self.processing_end_time:
            processing_time = self.processing_end_time - self.processing_start_time
            avg_fps = self.total_frames_processed / processing_time if processing_time > 0 else 0

        missing_counter = {}
        for event in self.violation_log:
            for item in event.get("missing_list", []):
                missing_counter[item] = missing_counter.get(item, 0) + 1

        most_missing = max(missing_counter, key=missing_counter.get) if missing_counter else "N/A"
        snapshot = self.get_model_status_snapshot()

        return {
            "source_name": self.video_name or "N/A",
            "start_time": datetime.fromtimestamp(self.processing_start_time).strftime("%Y-%m-%d %H:%M:%S")
            if self.processing_start_time
            else "N/A",
            "end_time": datetime.fromtimestamp(self.processing_end_time).strftime("%Y-%m-%d %H:%M:%S")
            if self.processing_end_time
            else "N/A",
            "total_processing_time_seconds": f"{processing_time:.2f}",
            "total_frames": self.total_frames_processed,
            "average_fps": f"{avg_fps:.2f}",
            "total_violations": len(self.violation_log),
            "most_missing_item": most_missing,
            "model_name": snapshot["display_name"],
            "model_status": snapshot["status_message"],
            "model_loaded": snapshot["loaded"],
            "demo_mode": self.demo_mode,
        }

    def generate_processing_summary(self):
        summary = self.get_processing_summary_data()
        return (
            "Processing Summary\n"
            f"Source Name: {summary['source_name']}\n"
            f"Start Time: {summary['start_time']}\n"
            f"End Time: {summary['end_time']}\n"
            f"Total Processing Time (s): {summary['total_processing_time_seconds']}\n"
            f"Total Frames: {summary['total_frames']}\n"
            f"Average FPS: {summary['average_fps']}\n"
            f"Total Violations: {summary['total_violations']}\n"
            f"Most Missing Item: {summary['most_missing_item']}\n"
            f"Model Name: {summary['model_name']}\n"
            f"Model Status: {summary['model_status']}\n"
            f"Demo Mode: {summary['demo_mode']}"
        )

    def reset(self):
        self.reset_tracking()

    def _detect_demo(self, frame, target_items, source_name="unknown", frame_number=0):
        current_time = time.time()
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        height, width = frame.shape[:2]
        annotated_frame = np.zeros((height, width, 3), dtype=np.uint8)
        num_persons = random.randint(1, 3)
        persons = []

        for index in range(num_persons):
            x1 = random.randint(50, max(60, width - 200))
            y1 = random.randint(50, max(60, height - 300))
            box_width = random.randint(80, 150)
            box_height = random.randint(200, 350)
            x2, y2 = x1 + box_width, y1 + box_height

            p_box = [x1, y1, x2, y2]
            track_id = f"demo_{index}"
            person_conf = random.uniform(0.7, 0.95)
            persons.append((p_box, track_id, person_conf))

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated_frame,
                f"Person ID:{track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        new_event_data = []
        stable_violations_output = []
        demo_patterns = [["helmet"], ["vest"], ["helmet", "vest"], ["helmet", "mask"], ["vest", "goggles"], []]

        for p_box, track_id, person_conf in persons:
            person_missing = random.choice(demo_patterns[:-1]) if random.random() < 0.8 else []

            self.person_states[track_id] = {
                "last_seen": current_time,
                "missing_items": set(person_missing),
                "bbox": p_box,
                "confidence": person_conf,
            }

            if track_id not in self.person_buffers:
                self.person_buffers[track_id] = deque(maxlen=self.temporal_frames)
            self.person_buffers[track_id].append(frozenset(person_missing))

            if len(self.person_buffers[track_id]) < self.temporal_frames:
                continue

            stable_missing = person_missing
            should_report, cooldown_keys = self._should_report_event(track_id, stable_missing, current_time)
            if not should_report:
                continue

            for key in cooldown_keys:
                self.counted_violations[key] = current_time

            bbox_str = (
                f"x={int(p_box[0])},"
                f"y={int(p_box[1])},"
                f"w={int(p_box[2] - p_box[0])},"
                f"h={int(p_box[3] - p_box[1])}"
            )

            center_x = (p_box[0] + p_box[2]) / 2
            center_y = (p_box[1] + p_box[3]) / 2
            event = {
                "timestamp": timestamp_str,
                "frame": frame_number,
                "source": source_name,
                "track_id": track_id,
                "person_count": len(persons),
                "missing_items": ", ".join(stable_missing) if stable_missing else "None",
                "confidence": person_conf,
                "bbox": bbox_str,
                "center_x": center_x,
                "center_y": center_y,
                "missing_list": stable_missing,
                "bbox_raw": p_box,
                "is_demo": True,
            }

            new_event_data.append(event)
            self.violation_log.append(event)
            stable_violations_output.append((track_id, stable_missing, p_box))
            self.violation_coords.append((center_x, center_y))

            if stable_missing:
                cv2.putText(
                    annotated_frame,
                    f"MISSING: {', '.join(stable_missing)}",
                    (int(p_box[0]), int(p_box[3]) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

        all_missing = [item for _, missing_items, _ in stable_violations_output for item in missing_items]
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        self.total_frames_processed += 1

        return annotated_frame, {
            "person_count": len(persons),
            "violation_detected": len(stable_violations_output) > 0,
            "missing_items": list(set(all_missing)),
            "stable_violations": stable_violations_output,
            "new_events": new_event_data,
            "is_demo": True,
            "fps": fps,
        }
