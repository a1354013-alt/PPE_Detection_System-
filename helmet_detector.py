import cv2
import numpy as np
from collections import deque, Counter
import json
import csv
import os
import time
from datetime import datetime


class HelmetDetector:
    """
    PPE Detection System - 專業版
    
    核心功能：
    1. 使用 YOLOv8 model.track(..., persist=True) 取得 track_id
    2. 若 tracking 不可用，fallback 到座標式 ID
    3. 每個 track_id / fallback_id 獨立判斷違規
    4. 每種 missing_items 組合獨立 cooldown
    5. temporal smoothing 只負責判斷「是否穩定違規」
    6. cooldown 只負責避免重複報告
    7. new_events 必須可預期、可測試、不可被全域 buffer 吃掉
    """
    
    def __init__(self, model_path='yolov8n.pt', conf=0.4, iou=0.45, config_path='config.json', demo_mode=False):
        self.model = None
        self.model_path = model_path
        self.config_path = config_path
        self.demo_mode = demo_mode
        self.config = self.load_config(config_path)

        self.conf = self.config.get('confidence_threshold', conf)
        self.iou = self.config.get('iou_threshold', iou)
        self.temporal_frames = self.config.get('temporal_frames', 3)
        self.cooldown_seconds = self.config.get('cooldown_seconds', 2)

        # 載入模型（如果是 demo mode 且無模型，會稍後處理）
        if not demo_mode or model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.model = None
        
        # 類別映射：將模型輸出的標籤映射到系統標準標籤
        self.class_map = {
            'hardhat': 'helmet',
            'head_helmet': 'helmet',
            'safety_helmet': 'helmet',
            'helmet': 'helmet',
            'safety_vest': 'vest',
            'reflective_vest': 'vest',
            'vest': 'vest',
            'goggles': 'goggles',
            'eye_protection': 'goggles',
            'mask': 'mask',
            'face_mask': 'mask'
        }

        # === 人員追蹤狀態（每個 track_id 獨立）===
        self.person_states = {}  # track_id -> {last_seen, missing_items, bbox, confidence}
        self.person_buffers = {}  # track_id -> deque of frozenset(missing_items)
        
        # === 冷卻機制（每個 (track_id, violation_type) 組合獨立）===
        self.counted_violations = {}  # (track_id, violation_type) -> last_count_time
        self.violation_cooldown = float(self.cooldown_seconds)
        
        # === 違規記錄 ===
        self.violation_log = []  # 所有事件記錄（用於 CSV 匯出）
        self.violation_coords = deque(maxlen=5000)  # 熱力圖坐標
        
        # === Demo Mode 設定 ===
        self.demo_missing_pattern = ['helmet', 'vest']  # Demo Mode 預設缺失

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
        """執行 YOLO 偵測，優先使用 tracking 模式"""
        if self.model is None:
            return [], False
            
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
        """根據座標建立 fallback track ID"""
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        return f"fallback_{cx // 50}_{cy // 50}"

    def cleanup_stale_tracks(self, current_time, ttl=10):
        """清理超過 TTL 的軌跡狀態"""
        stale_ids = [
            tid for tid, state in self.person_states.items()
            if current_time - state.get("last_seen", 0) > ttl
        ]

        for tid in stale_ids:
            del self.person_states[tid]

            keys_to_del = [
                key for key in list(self.counted_violations.keys())
                if key[0] == tid
            ]

            for key in keys_to_del:
                del self.counted_violations[key]

            if tid in self.person_buffers:
                del self.person_buffers[tid]

    def _check_ppe_missing(self, p_box, items, target_items, supported_classes):
        """檢查單個人員的 PPE 缺失情況"""
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

        return person_missing

    def _should_report_event(self, track_id, stable_missing, current_time):
        """
        判斷是否應該報告新事件。
        
        規則：
        1. 每個 (track_id, violation_type) 組合獨立 cooldown
        2. 只有真正回傳給 UI 的事件才寫入 cooldown
        
        回傳：(should_report, updated_cooldown_keys)
        """
        if not stable_missing:
            return False, []
        
        cooldown_keys = []
        should_report = False
        
        for violation_type in stable_missing:
            key = (track_id, violation_type)
            last_count_time = self.counted_violations.get(key, 0)
            
            if current_time - last_count_time >= self.violation_cooldown:
                cooldown_keys.append(key)
                should_report = True
        
        return should_report, cooldown_keys

    def detect(self, frame, target_items, source_name="unknown", frame_number=0):
        """
        核心偵測函數。
        
        事件流程：
        1. 對每個人員進行 temporal smoothing 判斷穩定違規
        2. 對穩定違規者檢查 cooldown
        3. 只有通過 cooldown 檢查的事件才加入 new_events
        4. new_events 直接回傳給 UI/EventLogger，不被全域 buffer 過濾
        """
        # Demo Mode 處理
        if self.demo_mode or self.model is None:
            return self._detect_demo(frame, target_items, source_name, frame_number)
        
        current_time = time.time()
        self.cleanup_stale_tracks(current_time)

        results, is_tracking = self.run_detection_with_tracking(frame)
        
        if not results:
            return frame, {
                'person_count': 0,
                'violation_detected': False,
                'missing_items': [],
                'stable_violations': [],
                'new_events': []
            }
        
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

        # 獲取模型支援的類別
        supported_classes = {
            self.class_map.get(name.lower(), name.lower())
            for name in names.values()
        }

        # === 第一階段：對每個人員進行 temporal smoothing ===
        frame_violations = []  # 本幀所有穩定違規
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for p_box, track_id, person_conf in persons:
            # 檢查 PPE 缺失
            person_missing = self._check_ppe_missing(
                p_box, items, target_items, supported_classes
            )

            # 更新人員狀態
            self.person_states[track_id] = {
                "last_seen": current_time,
                "missing_items": set(person_missing),
                "bbox": p_box,
                "confidence": person_conf
            }

            # 更新 temporal buffer
            if track_id not in self.person_buffers:
                self.person_buffers[track_id] = deque(maxlen=self.temporal_frames)
            self.person_buffers[track_id].append(frozenset(person_missing))

            # 判斷是否為穩定違規
            if len(self.person_buffers[track_id]) >= self.temporal_frames:
                recent_states = list(self.person_buffers[track_id])
                state_counts = Counter(recent_states)
                most_common_state, count = state_counts.most_common(1)[0]

                if most_common_state and count >= self.temporal_frames - 1:
                    stable_missing = list(most_common_state)
                    frame_violations.append((track_id, stable_missing, p_box, person_conf))

        # === 第二階段：對穩定違規者檢查 cooldown，產生 new_events ===
        new_event_data = []
        stable_violations_output = []

        for track_id, stable_missing, p_box, person_conf in frame_violations:
            # 檢查是否應該報告（cooldown 檢查）
            should_report, cooldown_keys = self._should_report_event(
                track_id, stable_missing, current_time
            )

            if should_report:
                # 寫入 cooldown（只有真正要報告的事件）
                for key in cooldown_keys:
                    self.counted_violations[key] = current_time

                # 產生事件資料
                bbox_str = (
                    f"x={int(p_box[0])},"
                    f"y={int(p_box[1])},"
                    f"w={int(p_box[2] - p_box[0])},"
                    f"h={int(p_box[3] - p_box[1])}"
                )

                center_x = (p_box[0] + p_box[2]) / 2
                center_y = (p_box[1] + p_box[3]) / 2

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
                    'missing_list': stable_missing,
                    'bbox_raw': p_box
                }

                new_event_data.append(event)
                self.violation_log.append(event)
                stable_violations_output.append((track_id, stable_missing, p_box))

                # 記錄違規坐標（用於熱力圖）
                self.violation_coords.append((center_x, center_y))

                # 在畫面上標示
                cv2.putText(
                    annotated_frame,
                    f"ID:{track_id} MISSING: {', '.join(stable_missing)}",
                    (int(p_box[0]), int(p_box[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

        # === 第三階段：回傳結果 ===
        # violation_detected: 只要有穩定違規就為 True（用於 UI 顯示紅框等）
        # new_events: 只有通過 cooldown 檢查的事件（用於 EventLogger）
        all_missing = [
            item
            for _, missing_items, _ in stable_violations_output
            for item in missing_items
        ]

        return annotated_frame, {
            'person_count': len(persons),
            'violation_detected': len(frame_violations) > 0,
            'missing_items': list(set(all_missing)),
            'stable_violations': stable_violations_output,
            'new_events': new_event_data  # 直接回傳，不被全域 buffer 過濾
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
        """重置所有追蹤狀態"""
        self.counted_violations = {}
        self.person_states = {}
        self.violation_coords.clear()
        if hasattr(self, 'violation_buffer'):
            self.violation_buffer.clear()
        self.person_buffers.clear()
        self.violation_log.clear()

    def reset(self):
        """完全重置（含配置）"""
        self.reset_tracking()

    def _detect_demo(self, frame, target_items, source_name="unknown", frame_number=0):
        """
        Demo Mode 偵測：使用模擬 PPE 缺失資料。
        
        當沒有真實 PPE 模型時，用 person bbox 模擬 PPE 缺失，
        用於展示完整流程（事件、截圖、報告等）。
        """
        import random
        
        current_time = time.time()
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 建立一個黑色畫布作為 annotated_frame
        h, w = frame.shape[:2]
        annotated_frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 模擬 1-3 個人員
        num_persons = random.randint(1, 3)
        persons = []
        
        for i in range(num_persons):
            # 隨機產生人員 bbox
            x1 = random.randint(50, w - 200)
            y1 = random.randint(50, h - 300)
            bw = random.randint(80, 150)
            bh = random.randint(200, 350)
            x2, y2 = x1 + bw, y1 + bh
            
            p_box = [x1, y1, x2, y2]
            track_id = f"demo_{i}"
            person_conf = random.uniform(0.7, 0.95)
            
            persons.append((p_box, track_id, person_conf))
            
            # 在畫面上繪製人員框
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated_frame,
                f"Person ID:{track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )
        
        # 對每個人員模擬 PPE 缺失
        new_event_data = []
        stable_violations_output = []
        
        for p_box, track_id, person_conf in persons:
            # Demo Mode: 模擬固定的缺失模式
            person_missing = list(self.demo_missing_pattern)
            
            # 更新人員狀態
            self.person_states[track_id] = {
                "last_seen": current_time,
                "missing_items": set(person_missing),
                "bbox": p_box,
                "confidence": person_conf
            }
            
            # 更新 temporal buffer
            if track_id not in self.person_buffers:
                self.person_buffers[track_id] = deque(maxlen=self.temporal_frames)
            self.person_buffers[track_id].append(frozenset(person_missing))
            
            # 判斷是否為穩定違規（Demo Mode 下總是穩定）
            if len(self.person_buffers[track_id]) >= self.temporal_frames:
                stable_missing = person_missing
                
                # 檢查 cooldown
                should_report, cooldown_keys = self._should_report_event(
                    track_id, stable_missing, current_time
                )
                
                if should_report:
                    # 寫入 cooldown
                    for key in cooldown_keys:
                        self.counted_violations[key] = current_time
                    
                    # 產生事件資料
                    bbox_str = (
                        f"x={int(p_box[0])},"
                        f"y={int(p_box[1])},"
                        f"w={int(p_box[2] - p_box[0])},"
                        f"h={int(p_box[3] - p_box[1])}"
                    )
                    
                    center_x = (p_box[0] + p_box[2]) / 2
                    center_y = (p_box[1] + p_box[3]) / 2
                    
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
                        'missing_list': stable_missing,
                        'bbox_raw': p_box,
                        'is_demo': True
                    }
                    
                    new_event_data.append(event)
                    self.violation_log.append(event)
                    stable_violations_output.append((track_id, stable_missing, p_box))
                    self.violation_coords.append((center_x, center_y))
                    
                    # 在畫面上標示違規
                    cv2.putText(
                        annotated_frame,
                        f"MISSING: {', '.join(stable_missing)}",
                        (int(p_box[0]), int(p_box[3]) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )
        
        all_missing = [
            item
            for _, missing_items, _ in stable_violations_output
            for item in missing_items
        ]
        
        return annotated_frame, {
            'person_count': len(persons),
            'violation_detected': len(stable_violations_output) > 0,
            'missing_items': list(set(all_missing)),
            'stable_violations': stable_violations_output,
            'new_events': new_event_data,
            'is_demo': True
        }