import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

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
        
        # 違規座標記錄
        self.violation_coords = []
        
        # 穩定性過濾：記錄每個人的缺失狀態 (Person ID -> Deque of missing items)
        # 註：由於 YOLOv8 預設不帶 Tracking，這裡簡化為「全域連續幀判定」
        self.violation_buffer = deque(maxlen=5) # 記錄最近 5 幀的違規狀態

    def load_model(self, path):
        try:
            self.model = YOLO(path)
            return True, f"成功載入模型: {path}"
        except Exception as e:
            return False, f"模型載入失敗: {str(e)}"

    def get_model_classes(self):
        if not self.model: return []
        # 取得模型原始類別並經過映射轉換
        raw_names = self.model.names.values()
        mapped_names = set()
        for name in raw_names:
            mapped_names.add(self.class_map.get(name.lower(), name.lower()))
        return list(mapped_names)

    def is_overlapping(self, box_person, box_item, ratio=0.3):
        """
        優化後的空間關聯判定
        ratio: 判定裝備必須在人體上方多少比例的區間內 (預設 0.3 代表頭部區域)
        """
        px1, py1, px2, py2 = box_person
        ix1, iy1, ix2, iy2 = box_item
        
        icx = (ix1 + ix2) / 2
        icy = (iy1 + iy2) / 2
        
        # 1. 中心點必須在人的寬度內
        in_width = px1 <= icx <= px2
        
        # 2. 裝備中心點必須在人的頂部區域 (py1 ~ py1 + height * ratio)
        person_height = py2 - py1
        in_head_zone = py1 - (person_height * 0.1) <= icy <= (py1 + person_height * ratio)
        
        return in_width and in_head_zone

    def detect(self, frame, target_items):
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
        frame_violations = []
        
        # 2. 空間關聯判定
        for p_box in persons:
            person_missing = []
            for target in target_items:
                # 檢查該目標是否在模型支援範圍內
                if not any(self.class_map.get(n.lower(), n.lower()) == target for n in names.values()):
                    continue # 模型不支援此類別，跳過判定
                
                found = any(self.is_overlapping(p_box, i_box) for i_box in items[target])
                if not found:
                    person_missing.append(target)
            
            if person_missing:
                frame_violations.append(person_missing)
                # 繪製警告
                cv2.putText(annotated_frame, f"MISSING: {', '.join(person_missing)}", 
                            (int(p_box[0]), int(p_box[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                # 記錄座標
                self.violation_coords.append(((p_box[0]+p_box[2])/2, (p_box[1]+p_box[3])/2))

        # 3. 穩定性過濾 (Temporal Smoothing)
        # 只有當最近 5 幀中有 3 幀以上出現違規，才判定為真實違規
        self.violation_buffer.append(len(frame_violations) > 0)
        is_stable_violation = sum(self.violation_buffer) >= 3
        
        # 整理缺失項目清單
        all_missing = [item for sublist in frame_violations for item in sublist]
        unique_missing = list(set(all_missing))

        return annotated_frame, {
            'person_count': len(persons),
            'violation_detected': is_stable_violation,
            'missing_items': unique_missing if is_stable_violation else []
        }

    def generate_heatmap(self, shape):
        heatmap = np.zeros(shape[:2], dtype=np.float32)
        for (x, y) in self.violation_coords:
            cv2.circle(heatmap, (int(x), int(y)), 30, 1, -1)
        heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
