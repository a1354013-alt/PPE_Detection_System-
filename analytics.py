import pandas as pd
from datetime import datetime
import os
from typing import List, Dict, Any
from event_logger import ViolationEvent

def parse_timestamp(ts: Any) -> datetime:
    """
    解析時間戳記，支援字串與 datetime 物件。
    """
    if isinstance(ts, datetime):
        return ts
    if isinstance(ts, str):
        # 嘗試常見格式
        formats = ["%Y-%m-%d %H:%M:%S", "%Y%m%d_%H%M%S"]
        for fmt in formats:
            try:
                return datetime.strptime(ts, fmt)
            except ValueError:
                continue
    return datetime.now()

def get_violation_trend(events: List[ViolationEvent], interval: str = "minute") -> Dict[str, Any]:
    """
    依時間統計違規事件數。
    """
    if not events:
        return {"labels": [], "counts": []}

    df = pd.DataFrame([{"timestamp": parse_timestamp(e.timestamp)} for e in events])
    
    if interval == "minute":
        res = df.set_index("timestamp").resample("1min").size()
    elif interval == "10s":
        res = df.set_index("timestamp").resample("10s").size()
    else:
        res = df.set_index("timestamp").resample("1min").size()

    return {
        "labels": [ts.strftime("%H:%M:%S") for ts in res.index],
        "counts": res.values.tolist()
    }

def get_ppe_missing_counts(events: List[ViolationEvent]) -> Dict[str, int]:
    """
    統計各類 PPE 缺失次數。
    """
    counts = {
        "helmet": 0,
        "vest": 0,
        "mask": 0,
        "goggles": 0
    }
    
    for e in events:
        # 處理 missing_items 字串，例如 "helmet, vest"
        items = [i.strip().lower() for i in str(e.missing_items).split(",") if i.strip()]
        for item in items:
            if item in counts:
                counts[item] += 1
            else:
                # 支援其他類別
                counts[item] = counts.get(item, 0) + 1
    return counts

def get_ppe_missing_ratio(events: List[ViolationEvent]) -> Dict[str, float]:
    """
    統計 PPE 缺失比例。
    """
    counts = get_ppe_missing_counts(events)
    total = sum(counts.values())
    if total == 0:
        return {k: 0.0 for k in counts}
    return {k: v / total for k, v in counts.items()}

def build_dashboard_summary(events: List[ViolationEvent]) -> Dict[str, Any]:
    """
    建立 Dashboard 摘要數據。
    """
    missing_counts = get_ppe_missing_counts(events)
    trend = get_violation_trend(events, interval="minute" if len(events) > 10 else "10s")
    
    return {
        "total_violations": len(events),
        "missing_counts": missing_counts,
        "trend": trend,
        "recent_events": [
            {
                "time": e.timestamp,
                "id": e.track_id,
                "missing": e.missing_items,
                "conf": e.confidence
            } for e in events[-5:][::-1] # 最近 5 筆，由新到舊
        ]
    }
