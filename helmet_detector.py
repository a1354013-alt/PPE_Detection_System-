import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque, Counter
import json
import csv
import os
import time
from datetime import datetime


class HelmetDetector:
    def __init__(self, model_path='yolov8n.pt', conf=0.4, iou=0.45, config_path='config.json'):
        self.model = None
        self.model_path = model_path
        self.config_path = config_path
        self.config = self.load_config(config_path)

        self.conf = self.config.get('confidence_threshold', conf)
        self.iou = self.config.get('iou_threshold', iou)
        self.temporal_frames = self.config.get('temporal_frames', 3)
        self.cooldown_seconds = self.config.get('cooldown_seconds', 2)

        self.load_model(model_path)

        self.class_map = {
            'hardhat': 'helmet',
            'head_helmet': 'helmet',
            'safety_helmet': 'helmet',
            'safety_vest': 'vest',
            'reflective_vest': 'vest',
            'goggles': 'goggles',
            'eye_protection': 'goggles',
            'mask': 'mask',
            'face_mask': 'mask'
        }

        self.violation_coords = deque(maxlen=5000)
        self.violation_buffer = deque(maxlen=5)

        self.person_states = {}
        self.counted_violations = {}
        self.violation_cooldown = float(self.cooldown_seconds)

        self.person_buffers = {}
        self.violation_log = []

    def load_config(self, config_path):
        default_config = {
            "confidence_threshold": 0.5,
            "iou_threshold": 0.3,
            "temporal_frames": 3,
            "cooldown_seconds": 2
        }

        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return default_config

        return default_config

    def load_model(self, path):
        try:
            self.model = YOLO(path)
            self.model_path = path
            return True, f"成功載入模型：{path}"
        except Exception as e:
            return False, f"模型載入失敗：{str(e)}"

    def get_model_classes(self):
        if not self.model:
            return []

        raw_names = self.model.names.values()
        mapped_names = set()

        for name in raw_names:
            mapped_names.add(self.class_map.get(name.lower(), name.lower()))

        return list(mapped_names)

    def get_person_zones(self, person_box):
        px1, py1, px2, py2 = person_box
        person_height = py2 - py1

        head_h = int(person_height * 0.25)
        head_zone = (px1, py1, px2, py1 + head_h)

        face_upper_h = int(person_height * 0.40)
        face_upper_zone = (px1, py1, px2, py1 + face_upper_h)

        face_lower_y1 = int(py1 + person_height * 0.40)
        face_lower_y2 = int(py1 + person_height * 0.60)
        face_lower_zone = (px1, face_lower_y1, px2, face_lower_y2)

        torso_y1 = int(py1 + person_height * 0.30)
        torso_y2 = int(py1 + person_height * 0.80)
        torso_zone = (px1, torso_y1, px2, torso_y2)

        return {
            "head": head_zone,
            "face_upper": face_upper_zone,
            "face_lower": face_lower_zone,
            "torso": torso_zone
        }

    def is_overlapping_with_zone(self, box_item, zone_box):
        ix1, iy1, ix2, iy2 = box_item
        zx1, zy1, zx2, zy2 = zone_box

        icx = (ix1 + ix2) / 2
        icy = (iy1 + iy2) / 2

        in_width = zx1 <= icx <= zx2
        in_height = zy1 <= icy <= zy2

        return in_width and in_height

    def run_detection_with_tracking(self, frame):
        try:
            results = self.model.track(
                frame,
                conf=self.conf,
                iou=self.iou,
                persist=True,
                verbose=False
            )
            return results, True
        except Exception:
            results = self.model(
                frame,
                conf=self.conf,
                iou=self.iou,
                verbose=False
            )
            return results, False

    def build_fallback_track_id(self, bbox):
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        return f"fallback_{cx // 50}_{cy // 50}"

    def cleanup_stale_tracks(self, current_time, ttl=10):
        stale_ids = [
            tid for tid, state in self.person_states.items()
            if current_time - state.get("last_seen", 0) > ttl
        ]

        for tid in stale_ids:
            del self.person_states[tid]

            keys_to_del = [
                key for key in self.counted_violations.keys()
                if key[0] == tid
            ]

            for key in keys_to_del:
                del self.counted_violations[key]

            if tid in self.person_buffers:
                del self.person_buffers[tid]

    def detect(self, frame, target_items, source_name="unknown", frame_number=0):
        if self.model is None:
            return frame, {'error': '模型未載入'}

        current_time = time.time()
        self.cleanup_stale_tracks(current_time)

        results, is_tracking = self.run_detection_with_tracking(frame)
        result = results[0]
        names = result.names

        persons = []
        items = {target: [] for target in target_items}

        for box in result.boxes:
            cls_id = int(box.cls[0])
            raw_label = names[cls_id].lower()
            mapped_label = self.class_map.get(raw_label, raw_label)

            coords = box.xyxy[0].tolist()
            conf = float(box.conf[0])

            if is_tracking and box.id is not None:
                track_id = str(int(box.id[0]))
            else:
                track_id = self.build_fallback_track_id(coords)

            if mapped_label == 'person':
                persons.append((coords, track_id, conf))
            elif mapped_label in target_items:
                items[mapped_label].append(coords)

        annotated_frame = result.plot()

        frame_violations = []
        stable_violations = []
        new_event_data = []
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        supported_classes = {
            self.class_map.get(name.lower(), name.lower())
            for name in names.values()
        }

        for p_box, track_id, person_conf in persons:
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
                elif target == "vest":
                    zone = zones["torso"]
                else:
                    zone = zones["head"]

                found = any(
                    self.is_overlapping_with_zone(item_box, zone)
                    for item_box in items.get(target, [])
                )

                if not found:
                    person_missing.append(target)

            self.person_states[track_id] = {
                "last_seen": current_time,
                "missing_items": set(person_missing),
                "bbox": p_box,
                "confidence": person_conf
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
                    frame_violations.append((track_id, stable_missing, p_box))
                    stable_violations.append((track_id, stable_missing, p_box))

                    cv2.putText(
                        annotated_frame,
                        f"ID:{track_id} MISSING: {', '.join(stable_missing)}",
                        (int(p_box[0]), int(p_box[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

                    center_x = (p_box[0] + p_box[2]) / 2
                    center_y = (p_box[1] + p_box[3]) / 2
                    self.violation_coords.append((center_x, center_y))

                    is_new_for_stats = False

                    for violation_type in stable_missing:
                        key = (track_id, violation_type)
                        last_count_time = self.counted_violations.get(key, 0)

                        if current_time - last_count_time >= self.violation_cooldown:
                            is_new_for_stats = True
                            self.counted_violations[key] = current_time

                    if is_new_for_stats:
                        bbox_str = (
                            f"x={int(p_box[0])},"
                            f"y={int(p_box[1])},"
                            f"w={int(p_box[2] - p_box[0])},"
                            f"h={int(p_box[3] - p_box[1])}"
                        )

                        event = {
                            'timestamp': timestamp_str,
                            'frame': frame_number,
                            'source': source_name,
                            'track_id': track_id,
                            'person_count': len(persons),
                            'missing_items': ", ".join(stable_missing),
                            'confidence': person_conf,
                            'bbox': bbox_str,
                            'center_x': center_x,
                            'center_y': center_y,
                            'missing_list': stable_missing
                        }

                        new_event_data.append(event)
                        self.violation_log.append(event)

        self.violation_buffer.append(len(stable_violations) > 0)
        is_stable_violation = sum(self.violation_buffer) >= 3

        all_missing = [
            item
            for _, missing_items, _ in stable_violations
            for item in missing_items
        ]

        return annotated_frame, {
            'person_count': len(persons),
            'violation_detected': is_stable_violation,
            'missing_items': list(set(all_missing)) if is_stable_violation else [],
            'stable_violations': stable_violations if is_stable_violation else [],
            'new_events': new_event_data if is_stable_violation else []
        }

    def generate_heatmap(self, shape):
        heatmap = np.zeros(shape[:2], dtype=np.float32)
        coords = list(self.violation_coords)

        for x, y in coords:
            cv2.circle(heatmap, (int(x), int(y)), 30, 1, -1)

        heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)

        return cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)

    def save_violation_log(self, filepath='violations/violations.csv'):
        if not self.violation_log:
            return False

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            writer.writerow([
                'timestamp',
                'frame',
                'source',
                'track_id',
                'person_count',
                'missing_items',
                'confidence',
                'bbox',
                'center_x',
                'center_y'
            ])

            for entry in self.violation_log:
                writer.writerow([
                    entry.get('timestamp', ''),
                    entry.get('frame', ''),
                    entry.get('source', ''),
                    entry.get('track_id', ''),
                    entry.get('person_count', ''),
                    entry.get('missing_items', ''),
                    entry.get('confidence', ''),
                    entry.get('bbox', ''),
                    entry.get('center_x', ''),
                    entry.get('center_y', '')
                ])

        return True

    def reset_tracking(self):
        self.counted_violations = {}
        self.person_states = {}
        self.violation_coords.clear()
        self.violation_buffer.clear()
        self.person_buffers.clear()
        self.violation_log.clear()

    def reset(self):
        self.reset_tracking()