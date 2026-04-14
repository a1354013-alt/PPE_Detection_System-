import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque, Counter
import json
import csv
import os
from datetime import datetime


class HelmetDetector:
    def __init__(self, model_path='yolov8n.pt', conf=0.4, iou=0.45, config_path='config.json'):
        self.model = None
        self.model_path = model_path
        self.config_path = config_path
        self.config = self.load_config(config_path)
        
        # 從 config 讀取閾值（優先級：config > 參數 > 預設值）
        self.conf = self.config.get('confidence_threshold', conf)
        self.iou = self.config.get('iou_threshold', iou)
        self.temporal_frames = self.config.get('temporal_frames', 3)
        self.cooldown_seconds = self.config.get('cooldown_seconds', 2)
        
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
        
        # 違規座標記錄（僅在穩定判定後記錄）
        self.violation_coords = []
        
        # Person-based temporal buffer: person_index -> deque of missing items
        # 使用簡化的 index-based tracking
        self.person_buffers = {}  # person_index -> deque
        
        # Violation log
        self.violation_log = []

    def load_config(self, config_path):
        """載入配置文件，若不存在則返回預設配置"""
        default_config = {
            "confidence_threshold": 0.5,
            "iou_threshold": 0.3,
            "temporal_frames": 3,
            "cooldown_seconds": 2
        }
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return default_config
        return default_config

    def load_model(self, path):
        try:
            self.model = YOLO(path)
            self.model_path = path  # 確保 model_path 同步
            return True, f"成功載入模型：{path}"
        except Exception as e:
            return False, f"模型載入失敗：{str(e)}"

    def get_model_classes(self):
        if not self.model:
            return []
        # 取得模型原始類別並經過映射轉換
        raw_names = self.model.names.values()
        mapped_names = set()
        for name in raw_names:
            mapped_names.add(self.class_map.get(name.lower(), name.lower()))
        return list(mapped_names)

    def get_person_zones(self, person_box):
        """
        將人體邊框劃分為多個區域，供不同 PPE 項目使用
        
        回傳格式：
        {
            "head": (x1, y1, x2, y2),        # 上 25% - helmet
            "face_upper": (x1, y1, x2, y2),  # 上 40% - goggles
            "face_lower": (x1, y1, x2, y2),  # 40% ~ 60% - mask
            "torso": (x1, y1, x2, y2)        # 30% ~ 80% - vest
        }
        """
        px1, py1, px2, py2 = person_box
        person_height = py2 - py1
        
        # head: 上 25%
        head_h = int(person_height * 0.25)
        head_zone = (px1, py1, px2, py1 + head_h)
        
        # face_upper: 上 40%
        face_upper_h = int(person_height * 0.40)
        face_upper_zone = (px1, py1, px2, py1 + face_upper_h)
        
        # face_lower: 40% ~ 60%
        face_lower_y1 = int(py1 + person_height * 0.40)
        face_lower_y2 = int(py1 + person_height * 0.60)
        face_lower_zone = (px1, face_lower_y1, px2, face_lower_y2)
        
        # torso: 30% ~ 80%
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
        """
        檢查物品是否在指定的 PPE 區域內
        
        zone_box: 該 PPE 對應的人體區域 (head/face_upper/face_lower/torso)
        """
        ix1, iy1, ix2, iy2 = box_item
        zx1, zy1, zx2, zy2 = zone_box
        
        icx = (ix1 + ix2) / 2
        icy = (iy1 + iy2) / 2
        
        # 中心點必須在區域的寬度內
        in_width = zx1 <= icx <= zx2
        
        # 中心點必須在區域的高度內
        in_height = zy1 <= icy <= zy2
        
        return in_width and in_height

    def detect(self, frame, target_items, frame_number=0):
        if self.model is None:
            return frame, {'error': '模型未載入'}

        results = self.model(frame, conf=self.conf, iou=self.iou, verbose=False)
        result = results[0]
        names = result.names
        
        persons = []
        items = {t: [] for t in target_items}
        
        # 1. 收集並映射物件
        for box in result.boxes:
            cls_id = int(box.cls[0])
            raw_label = names[cls_id].lower()
            mapped_label = self.class_map.get(raw_label, raw_label)
            coords = box.xyxy[0].tolist()
            
            if mapped_label == 'person':
                persons.append(coords)
            elif mapped_label in target_items:
                items[mapped_label].append(coords)
        
        annotated_frame = result.plot()
        frame_violations = []  # List of (person_index, missing_items, person_box)
        
        # 2. 空間關聯判定 - 使用正確的 PPE 區域
        for idx, p_box in enumerate(persons):
            zones = self.get_person_zones(p_box)
            person_missing = []
            
            for target in target_items:
                # 檢查該目標是否在模型支援範圍內
                if not any(self.class_map.get(n.lower(), n.lower()) == target for n in names.values()):
                    continue  # 模型不支援此類別，跳過判定
                
                # 根據 PPE 類型選擇正確的區域
                if target == "helmet":
                    zone = zones["head"]
                elif target == "goggles":
                    zone = zones["face_upper"]
                elif target == "mask":
                    zone = zones["face_lower"]
                elif target == "vest":
                    zone = zones["torso"]
                else:
                    zone = zones["head"]  # 預設 fallback
                
                # 檢查該 PPE 是否在正確區域內
                found = any(
                    self.is_overlapping_with_zone(i_box, zone) 
                    for i_box in items[target]
                )
                
                if not found:
                    person_missing.append(target)
            
            if person_missing:
                frame_violations.append((idx, person_missing, p_box))
                # 繪製警告
                cv2.putText(annotated_frame, f"MISSING: {', '.join(person_missing)}", 
                            (int(p_box[0]), int(p_box[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 3. Person-based Temporal Smoothing
        # 清理不再存在的 person index
        current_indices = set(range(len(persons)))
        stale_indices = set(self.person_buffers.keys()) - current_indices
        for stale_idx in stale_indices:
            del self.person_buffers[stale_idx]
        
        # 為每個人更新 buffer
        stable_violations = []  # (person_index, missing_items, person_box)
        
        for idx, missing_items, p_box in frame_violations:
            if idx not in self.person_buffers:
                self.person_buffers[idx] = deque(maxlen=self.temporal_frames)
            
            # 記錄缺失狀態（使用 frozenset 以便比較）
            missing_set = frozenset(missing_items)
            self.person_buffers[idx].append(missing_set)
            
            # 檢查是否穩定：最近 N 幀中有足夠次數出現相同的缺失
            if len(self.person_buffers[idx]) >= self.temporal_frames:
                recent_states = list(self.person_buffers[idx])
                # 計算最常見的缺失組合
                state_counts = Counter(recent_states)
                most_common_state, count = state_counts.most_common(1)[0]
                
                # 如果最常見狀態出現次數達到閾值，判定為穩定違規
                if count >= self.temporal_frames - 1:  # 允許 1 幀誤差
                    stable_violations.append((idx, list(most_common_state), p_box))
                    # 僅在穩定判定後才記錄座標
                    self.violation_coords.append(((p_box[0]+p_box[2])/2, (p_box[1]+p_box[3])/2))
                    
                    # 記錄 violation log
                    self.violation_log.append({
                        'frame': frame_number,
                        'person_id': idx,
                        'missing_items': list(most_common_state),
                        'coords': ((p_box[0]+p_box[2])/2, (p_box[1]+p_box[3])/2)
                    })

        # 整理缺失項目清單
        all_missing = [item for _, items, _ in stable_violations for item in items]
        unique_missing = list(set(all_missing))
        is_stable_violation = len(stable_violations) > 0

        return annotated_frame, {
            'person_count': len(persons),
            'violation_detected': is_stable_violation,
            'missing_items': unique_missing if is_stable_violation else [],
            'stable_violations': stable_violations
        }

    def generate_heatmap(self, shape):
        heatmap = np.zeros(shape[:2], dtype=np.float32)
        for (x, y) in self.violation_coords:
            cv2.circle(heatmap, (int(x), int(y)), 30, 1, -1)
        heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)

    def save_violation_log(self, filepath='violations/violations.csv'):
        """將違規記錄儲存為 CSV"""
        if not self.violation_log:
            return False
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'frame', 'person_id', 'missing_items', 'center_x', 'center_y'])
            
            for entry in self.violation_log:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow([
                    timestamp,
                    entry['frame'],
                    entry['person_id'],
                    ';'.join(entry['missing_items']),
                    entry['coords'][0],
                    entry['coords'][1]
                ])
        
        return True

    def reset(self):
        """重置偵測狀態"""
        self.violation_coords.clear()
        self.person_buffers.clear()
        self.violation_log.clear()
