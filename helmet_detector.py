import csv
import json
import os
import random
import re
import time
from collections import Counter, deque
from datetime import datetime

import cv2
import numpy as np


DEMO_RANDOM_SEED = 42
MODEL_NOT_LOADED_ERROR = "Model is not loaded. Please load a PPE model or enable Demo Mode."
UNSUPPORTED_MODEL_ERROR = (
    "Real Mode requires either:\n"
    "1. person + selected PPE classes, or\n"
    "2. direct violation classes such as no_helmet / no_vest.\n\n"
    "The current model does not match the supported PPE contract. "
    "Please load a PPE-specific model or use Demo Mode."
)


class HelmetDetector:
    """PPE detector with tracking, temporal smoothing, cooldown, and demo mode."""

    PPE_ITEMS = ("helmet", "vest", "mask", "goggles")
    VIOLATION_ITEMS = tuple(f"missing_{item}" for item in PPE_ITEMS)

    PRESENCE_CLASS_ALIASES = {
        "person": "person",
        "helmet": "helmet",
        "hardhat": "helmet",
        "hard hat": "helmet",
        "hardhat helmet": "helmet",
        "safety helmet": "helmet",
        "head helmet": "helmet",
        "vest": "vest",
        "safety vest": "vest",
        "safetyvest": "vest",
        "reflective vest": "vest",
        "reflectivevest": "vest",
        "goggles": "goggles",
        "eye protection": "goggles",
        "eyeprotection": "goggles",
        "mask": "mask",
        "face mask": "mask",
        "facemask": "mask",
    }
    VIOLATION_CLASS_ALIASES = {
        "no helmet": "missing_helmet",
        "no hardhat": "missing_helmet",
        "no hard hat": "missing_helmet",
        "bare head": "missing_helmet",
        "no vest": "missing_vest",
        "no safety vest": "missing_vest",
        "no safetyvest": "missing_vest",
        "no mask": "missing_mask",
        "no goggles": "missing_goggles",
        "no ppe": "missing_all",
    }

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
        self.demo_rng = random.Random(DEMO_RANDOM_SEED)

        self.class_map = self._build_class_map()

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
        self.frame_count = 0
        self.start_time = time.time()
        self.processing_start_time = None
        self.processing_end_time = None
        self.total_frames_processed = 0
        self.video_name = ""

    def _build_class_map(self):
        class_map = {}
        for alias_map in (self.PRESENCE_CLASS_ALIASES, self.VIOLATION_CLASS_ALIASES):
            for alias, canonical in alias_map.items():
                class_map[self._sanitize_label(alias)] = canonical
        return class_map

    def _empty_model_capabilities(self):
        return {
            "person": False,
            "helmet": False,
            "vest": False,
            "mask": False,
            "goggles": False,
            "missing_helmet": False,
            "missing_vest": False,
            "missing_mask": False,
            "missing_goggles": False,
            "supported_items": [],
            "presence_items": [],
            "violation_items": [],
            "contract_mode": "unsupported",
            "is_ppe_model": False,
            "unsupported_reason": "",
        }

    def _build_empty_detection_info(self, fps=0, error=""):
        return {
            "person_count": 0,
            "violation_detected": False,
            "missing_items": [],
            "stable_violations": [],
            "new_events": [],
            "fps": fps,
            "is_demo": False,
            "error": error,
        }

    @staticmethod
    def _sanitize_label(label):
        lowered = str(label).strip().lower()
        lowered = lowered.replace("-", " ").replace("_", " ")
        return re.sub(r"\s+", " ", lowered)

    def normalize_class_name(self, label):
        sanitized = self._sanitize_label(label)
        return self.class_map.get(sanitized, sanitized)

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

    def _get_model_name_values(self):
        if not self.model or not hasattr(self.model, "names"):
            return []

        names = self.model.names
        if isinstance(names, dict):
            return list(names.values())
        return list(names)

    def _check_model_capability(self):
        if not self.model:
            self.model_capabilities = self._empty_model_capabilities()
            return

        canonical_classes = {self.normalize_class_name(name) for name in self._get_model_name_values()}
        presence_items = sorted(item for item in self.PPE_ITEMS if item in canonical_classes)
        violation_items = sorted(item for item in self.VIOLATION_ITEMS if item in canonical_classes or "missing_all" in canonical_classes)

        has_person = "person" in canonical_classes
        has_presence_contract = has_person and bool(presence_items)
        has_violation_contract = bool(violation_items)

        if has_presence_contract and has_violation_contract:
            contract_mode = "mixed"
            unsupported_reason = ""
        elif has_presence_contract:
            contract_mode = "presence-based"
            unsupported_reason = ""
        elif has_violation_contract:
            contract_mode = "violation-class-based"
            unsupported_reason = ""
        else:
            contract_mode = "unsupported"
            unsupported_reason = UNSUPPORTED_MODEL_ERROR

        self.model_capabilities = {
            "person": has_person,
            "helmet": "helmet" in canonical_classes,
            "vest": "vest" in canonical_classes,
            "mask": "mask" in canonical_classes,
            "goggles": "goggles" in canonical_classes,
            "missing_helmet": "missing_helmet" in canonical_classes or "missing_all" in canonical_classes,
            "missing_vest": "missing_vest" in canonical_classes or "missing_all" in canonical_classes,
            "missing_mask": "missing_mask" in canonical_classes or "missing_all" in canonical_classes,
            "missing_goggles": "missing_goggles" in canonical_classes or "missing_all" in canonical_classes,
            "supported_items": sorted(canonical_classes),
            "presence_items": presence_items,
            "violation_items": violation_items,
            "contract_mode": contract_mode,
            "is_ppe_model": bool(presence_items or violation_items),
            "unsupported_reason": unsupported_reason,
        }

    def get_model_classes(self):
        return list(self.model_capabilities.get("supported_items", []))

    def get_model_display_name(self):
        return os.path.basename(self.model_path) if self.model_path else "N/A"

    def get_model_status_snapshot(self):
        warning = ""
        if self.model_loaded and self.model_capabilities.get("contract_mode") == "unsupported":
            warning = UNSUPPORTED_MODEL_ERROR
        elif self.model_loaded and self.model_capabilities.get("person") and not self.model_capabilities.get("is_ppe_model"):
            warning = (
                "The loaded model detects person but does not expose PPE classes. "
                "Load a PPE-specific model for Real Mode or use Demo Mode for UI demonstrations."
            )
        elif not self.model_loaded and self.demo_mode:
            warning = "Demo Mode is active. The app is simulating PPE events without a real PPE model."

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

    def is_box_center_inside(self, inner_box, outer_box):
        ix1, iy1, ix2, iy2 = inner_box
        ox1, oy1, ox2, oy2 = outer_box
        center_x = (ix1 + ix2) / 2
        center_y = (iy1 + iy2) / 2
        return ox1 <= center_x <= ox2 and oy1 <= center_y <= oy2

    def run_detection_with_tracking(self, frame):
        if self.model is None:
            return [], False

        try:
            results = self.model.track(frame, conf=self.conf, iou=self.iou, persist=True, verbose=False)
            return results, True
        except Exception:
            results = self.model(frame, conf=self.conf, iou=self.iou, verbose=False)
            return results, False

    def build_fallback_track_id(self, bbox, prefix="fallback"):
        x1, y1, x2, y2 = bbox
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        return f"{prefix}_{center_x // 50}_{center_y // 50}"

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

    def update_temporal_state(self, track_id, missing_items):
        buffer_state = self.person_buffers.setdefault(track_id, deque(maxlen=self.temporal_frames))
        normalized_missing = frozenset(sorted(item for item in missing_items if item in self.PPE_ITEMS))
        buffer_state.append(normalized_missing)

        if len(buffer_state) < self.temporal_frames:
            return []

        state_counts = Counter(buffer_state)
        most_common_state, count = state_counts.most_common(1)[0]
        threshold = max(1, self.temporal_frames - 1)
        if most_common_state and count >= threshold:
            return sorted(most_common_state)
        return []

    def _should_report_event(self, track_id, stable_missing, current_time):
        if not stable_missing:
            return False, []

        event_key = (track_id, tuple(sorted(stable_missing)))
        last_count_time = self.counted_violations.get(event_key, 0)

        if current_time - last_count_time >= self.violation_cooldown:
            return True, [event_key]
        return False, []

    def get_contract_validation(self, selected_items):
        selected = [item for item in selected_items if item in self.PPE_ITEMS]
        caps = self.model_capabilities

        if not self.model_loaded:
            return False, MODEL_NOT_LOADED_ERROR

        contract_mode = caps.get("contract_mode", "unsupported")
        if contract_mode == "unsupported":
            return False, caps.get("unsupported_reason") or UNSUPPORTED_MODEL_ERROR

        presence_supported = set(caps.get("presence_items", []))
        violation_supported = {
            item.replace("missing_", "")
            for item in caps.get("violation_items", [])
            if item.startswith("missing_")
        }
        unsupported_selected = [
            item for item in selected if item not in presence_supported and item not in violation_supported
        ]
        if unsupported_selected:
            return (
                False,
                "The loaded model does not support the selected PPE items:\n"
                f"{', '.join(unsupported_selected)}\n\n"
                "Real Mode requires either:\n"
                "1. person + selected PPE classes, or\n"
                "2. direct violation classes such as no_helmet / no_vest.",
            )

        if contract_mode == "presence-based" and not caps.get("person"):
            return False, UNSUPPORTED_MODEL_ERROR

        return True, ""

    def _extract_box_values(self, box, is_tracking):
        coords = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        track_id = None
        if is_tracking and getattr(box, "id", None) is not None:
            try:
                track_id = str(int(box.id[0]))
            except (TypeError, ValueError, IndexError):
                track_id = None
        return coords, conf, track_id

    def _collect_detections(self, result, is_tracking, target_items):
        names = result.names
        persons = []
        items = {target: [] for target in target_items}
        direct_violations = []

        for box in result.boxes:
            cls_id = int(box.cls[0])
            raw_label = names[cls_id]
            mapped_label = self.normalize_class_name(raw_label)
            coords, conf, track_id = self._extract_box_values(box, is_tracking)

            if mapped_label == "person":
                persons.append((coords, track_id or self.build_fallback_track_id(coords, prefix="person"), conf))
            elif mapped_label in target_items:
                items[mapped_label].append(coords)
            elif mapped_label == "missing_all":
                direct_violations.append(
                    (track_id or self.build_fallback_track_id(coords, prefix="violation"), list(target_items), coords, conf)
                )
            elif mapped_label.startswith("missing_"):
                item_name = mapped_label.replace("missing_", "", 1)
                if item_name in target_items:
                    direct_violations.append(
                        (track_id or self.build_fallback_track_id(coords, prefix="violation"), [item_name], coords, conf)
                    )

        return persons, items, direct_violations

    def _merge_missing_by_track(self, persons, presence_missing_by_track, direct_violations):
        merged = {}

        for p_box, track_id, person_conf in persons:
            merged[track_id] = {
                "bbox": p_box,
                "confidence": person_conf,
                "missing": set(presence_missing_by_track.get(track_id, [])),
            }

        for direct_track_id, missing_items, bbox, conf in direct_violations:
            matched_track_id = direct_track_id
            matched_bbox = bbox
            matched_conf = conf
            for p_box, person_track_id, person_conf in persons:
                if self.is_box_center_inside(bbox, p_box):
                    matched_track_id = person_track_id
                    matched_bbox = p_box
                    matched_conf = person_conf
                    break

            entry = merged.setdefault(
                matched_track_id,
                {"bbox": matched_bbox, "confidence": matched_conf, "missing": set()},
            )
            entry["missing"].update(missing_items)

        return merged

    def detect(self, frame, target_items, source_name="unknown", frame_number=0):
        if self.demo_mode:
            return self._detect_demo(frame, target_items, source_name, frame_number)

        selected_items = [item for item in target_items if item in self.PPE_ITEMS]
        is_valid_contract, contract_error = self.get_contract_validation(selected_items)
        if not is_valid_contract:
            return frame, self._build_empty_detection_info(error=contract_error)

        current_time = time.time()
        self.cleanup_stale_tracks(current_time)

        results, is_tracking = self.run_detection_with_tracking(frame)
        if not results:
            return frame, self._build_empty_detection_info()

        result = results[0]
        persons, items, direct_violations = self._collect_detections(result, is_tracking, selected_items)
        annotated_frame = result.plot()
        supported_classes = set(self.model_capabilities.get("presence_items", []))

        presence_missing_by_track = {}
        if self.model_capabilities.get("contract_mode") in {"presence-based", "mixed"}:
            for p_box, track_id, person_conf in persons:
                person_missing = self._check_ppe_missing(p_box, items, selected_items, supported_classes)
                self.person_states[track_id] = {
                    "last_seen": current_time,
                    "missing_items": set(person_missing),
                    "bbox": p_box,
                    "confidence": person_conf,
                }
                presence_missing_by_track[track_id] = person_missing

        merged_missing = self._merge_missing_by_track(persons, presence_missing_by_track, direct_violations)

        frame_violations = []
        for track_id, track_state in merged_missing.items():
            stable_missing = self.update_temporal_state(track_id, track_state["missing"])
            if stable_missing:
                frame_violations.append((track_id, stable_missing, track_state["bbox"], track_state["confidence"]))

        new_event_data = []
        stable_violations_output = []
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
            "violation_detected": bool(stable_violations_output),
            "missing_items": sorted(set(all_missing)),
            "stable_violations": stable_violations_output,
            "new_events": new_event_data,
            "fps": fps,
            "is_demo": False,
            "error": "",
        }

    def generate_heatmap(self, shape):
        heatmap = np.zeros(shape[:2], dtype=np.float32)
        for x_coord, y_coord in list(self.violation_coords):
            cv2.circle(heatmap, (int(x_coord), int(y_coord)), 30, 1, -1)

        heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)

    def save_violation_log(self, filepath=None):
        if not self.violation_log:
            return False

        filepath = filepath or os.path.join("violations", "violations.csv")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", newline="", encoding="utf-8") as file_obj:
            writer = csv.writer(file_obj)
            writer.writerow(
                [
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
                ]
            )

            for entry in self.violation_log:
                writer.writerow(
                    [
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
                    ]
                )

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
        self.demo_rng = random.Random(DEMO_RANDOM_SEED)

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
            "contract_mode": snapshot["capabilities"].get("contract_mode", "unsupported"),
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
            f"Contract Mode: {summary['contract_mode']}\n"
            f"Demo Mode: {summary['demo_mode']}"
        )

    def reset(self):
        self.reset_tracking()

    def _build_demo_patterns(self, target_items):
        target_set = {item for item in target_items if item in self.PPE_ITEMS}
        base_patterns = [
            ["helmet"],
            ["vest"],
            ["helmet", "vest"],
            ["helmet", "mask"],
            ["vest", "goggles"],
        ]

        filtered_patterns = []
        for pattern in base_patterns:
            filtered = [item for item in pattern if item in target_set]
            if filtered not in filtered_patterns:
                filtered_patterns.append(filtered)

        if [] not in filtered_patterns:
            filtered_patterns.append([])

        return filtered_patterns

    def _detect_demo(self, frame, target_items, source_name="unknown", frame_number=0):
        current_time = time.time()
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        height, width = frame.shape[:2]
        annotated_frame = np.zeros((height, width, 3), dtype=np.uint8)
        num_persons = self.demo_rng.randint(1, 3)
        persons = []

        for index in range(num_persons):
            x1 = self.demo_rng.randint(50, max(60, width - 200))
            y1 = self.demo_rng.randint(50, max(60, height - 300))
            box_width = self.demo_rng.randint(80, 150)
            box_height = self.demo_rng.randint(200, 350)
            x2, y2 = x1 + box_width, y1 + box_height

            p_box = [x1, y1, x2, y2]
            track_id = f"demo_{index}"
            person_conf = self.demo_rng.uniform(0.7, 0.95)
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
        demo_patterns = self._build_demo_patterns(target_items)

        for p_box, track_id, person_conf in persons:
            if self.demo_rng.random() < 0.8:
                person_missing = list(self.demo_rng.choice(demo_patterns[:-1] or [[]]))
            else:
                person_missing = []

            self.person_states[track_id] = {
                "last_seen": current_time,
                "missing_items": set(person_missing),
                "bbox": p_box,
                "confidence": person_conf,
            }

            stable_missing = self.update_temporal_state(track_id, person_missing)
            if not stable_missing:
                continue

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
            "violation_detected": bool(stable_violations_output),
            "missing_items": sorted(set(all_missing)),
            "stable_violations": stable_violations_output,
            "new_events": new_event_data,
            "is_demo": True,
            "fps": fps,
            "error": "",
        }
