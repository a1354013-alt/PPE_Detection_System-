import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import time
from datetime import datetime

class HelmetDetector:
    def __init__(self, model_path='yolov8n.pt', conf=0.4, iou=0.45):
        self.model = None
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.load_model(model_path)
        
        # 類別映射表：將模型標籤映射到系統邏輯標籤
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
        
        # 違規座標記錄 (限制最大長度為 5000)
        self.violation_coords = deque(maxlen=5000)
        
        # 穩定性過濾：記錄最近幀的違規狀態
        self.violation_buffer = deque(maxlen=5) 
        
        # 人員狀態管理 (track_id -> state)
        self.person_states = {}
        
        # 追蹤違規統計 (track_id, violation_type) -> last_count_time
        self.counted_violations = {}
        self.violation_cooldown = 3.0 # 3秒冷卻

    def load_model(self, path):
        try:
            self.model = YOLO(path)
            return True, f"成功載入模型: {path}"
        except Exception as e:
            return False, f"模型載入失敗: {str(e)}"

    def get_model_classes(self):
        if not self.model: return []
        raw_names = self.model.names.values()
        mapped_names = set()
        for name in raw_names:
            mapped_names.add(self.class_map.get(name.lower(), name.lower()))
        return list(mapped_names)

    def is_overlapping(self, box_person, box_item, ratio=0.3):
        px1, py1, px2, py2 = box_person
        ix1, iy1, ix2, iy2 = box_item
        icx = (ix1 + ix2) / 2
        icy = (iy1 + iy2) / 2
        in_width = px1 <= icx <= px2
        person_height = py2 - py1
        in_head_zone = py1 - (person_height * 0.1) <= icy <= (py1 + person_height * ratio)
        return in_width and in_head_zone

    def run_detection_with_tracking(self, frame):
        """封裝 Tracking 推論流程，包含 Fallback 機制"""
        try:
            results = self.model.track(frame, conf=self.conf, iou=self.iou, persist=True, verbose=False)
            return results, True
        except Exception as err:
            # print(f"Tracking failed, fallback to detection: {err}")
            results = self.model(frame, conf=self.conf, iou=self.iou, verbose=False)
            return results, False

    def build_fallback_track_id(self, bbox):
        """基於 BBox 中心點建立簡易 Fallback ID"""
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        return f"fallback_{cx // 50}_{cy // 50}"

    def cleanup_stale_tracks(self, current_time, ttl=10):
        """清理長時間未出現的人員狀態"""
        stale_ids = [
            tid for tid, state in self.person_states.items()
            if current_time - state.get("last_seen", 0) > ttl
        ]
        for tid in stale_ids:
            del self.person_states[tid]
            # 同時清理該人員的違規計數紀錄，避免 track_id 重用時受到舊冷卻時間影響
            keys_to_del = [k for k in self.counted_violations.keys() if k[0] == tid]
            for k in keys_to_del:
                del self.counted_violations[k]

    def detect(self, frame, target_items, source_name="unknown"):
        if self.model is None:
            return frame, {'error': '模型未載入'}
        
        current_time = time.time()
        self.cleanup_stale_tracks(current_time)
        
        results, is_tracking = self.run_detection_with_tracking(frame)
        result = results[0]
        names = result.names
        
        persons = [] # List of (coords, track_id, confidence)
        items = {t: [] for t in target_items}
        
        # 1. 收集並映射物件
        for box in result.boxes:
            cls_id = int(box.cls[0])
            raw_label = names[cls_id].lower()
            mapped_label = self.class_map.get(raw_label, raw_label)
            coords = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            
            # 取得 track_id
            track_id = None
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
        new_event_data = []
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 2. 空間關聯判定與狀態更新
        for p_box, t_id, p_conf in persons:
            person_missing = []
            for target in target_items:
                if not any(self.class_map.get(n.lower(), n.lower()) == target for n in names.values()):
                    continue
                
                found = any(self.is_overlapping(p_box, i_box) for i_box in items[target])
                if not found:
                    person_missing.append(target)
            
            # 更新人員狀態
            self.person_states[t_id] = {
                "last_seen": current_time,
                "missing_items": set(person_missing),
                "bbox": p_box,
                "confidence": p_conf
            }
            
            if person_missing:
                frame_violations.append(person_missing)
                # 繪製警告
                cv2.putText(annotated_frame, f"ID:{t_id} MISSING: {', '.join(person_missing)}", 
                            (int(p_box[0]), int(p_box[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                # 記錄座標
                self.violation_coords.append(((p_box[0]+p_box[2])/2, (p_box[1]+p_box[3])/2))
                
                # 處理統計邏輯 (track_id + violation_type + cooldown)
                is_new_for_stats = False
                for v_type in person_missing:
                    key = (t_id, v_type)
                    last_count_time = self.counted_violations.get(key, 0)
                    if current_time - last_count_time >= self.violation_cooldown:
                        is_new_for_stats = True
                        self.counted_violations[key] = current_time
                
                if is_new_for_stats:
                    bbox_str = f"x={int(p_box[0])},y={int(p_box[1])},w={int(p_box[2]-p_box[0])},h={int(p_box[3]-p_box[1])}"
                    new_event_data.append({
                        'timestamp': timestamp_str,
                        'source': source_name,
                        'track_id': t_id,
                        'person_count': len(persons),
                        'missing_items': ", ".join(person_missing),
                        'confidence': p_conf,
                        'bbox': bbox_str,
                        'missing_list': person_missing
                    })

        # 3. 穩定性過濾
        self.violation_buffer.append(len(frame_violations) > 0)
        is_stable_violation = sum(self.violation_buffer) >= 3
        
        return annotated_frame, {
            'person_count': len(persons),
            'violation_detected': is_stable_violation,
            'new_events': new_event_data if is_stable_violation else []
        }

    def generate_heatmap(self, shape):
        heatmap = np.zeros(shape[:2], dtype=np.float32)
        coords = list(self.violation_coords)
        for (x, y) in coords:
            cv2.circle(heatmap, (int(x), int(y)), 30, 1, -1)
        heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)

    def reset_tracking(self):
        """重置追蹤狀態"""
        self.counted_violations = {}
        self.person_states = {}
        self.violation_coords.clear()
        self.violation_buffer.clear()
