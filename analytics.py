from collections import Counter
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

from event_logger import ViolationEvent


def parse_timestamp(ts: Any) -> datetime:
    if isinstance(ts, datetime):
        return ts
    if isinstance(ts, str):
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y%m%d_%H%M%S"):
            try:
                return datetime.strptime(ts, fmt)
            except ValueError:
                continue
    return datetime.now()


def get_violation_trend(events: List[ViolationEvent], interval: str = "minute") -> Dict[str, Any]:
    if not events:
        return {"labels": [], "counts": []}

    df = pd.DataFrame([{"timestamp": parse_timestamp(e.timestamp)} for e in events])
    resample_rule = "10s" if interval == "10s" else "1min"
    res = df.set_index("timestamp").resample(resample_rule).size()
    return {"labels": [ts.strftime("%H:%M:%S") for ts in res.index], "counts": res.values.tolist()}


def get_ppe_missing_counts(events: List[ViolationEvent]) -> Dict[str, int]:
    counts = {"helmet": 0, "vest": 0, "mask": 0, "goggles": 0}
    for event in events:
        if getattr(event, "event_type", "ppe_violation") != "ppe_violation":
            continue
        items = [item.strip().lower() for item in str(event.missing_items).split(",")]
        for item in items:
            if not item or item in {"-", "none"}:
                continue
            counts[item] = counts.get(item, 0) + 1
    return counts


def get_ppe_missing_ratio(events: List[ViolationEvent]) -> Dict[str, float]:
    counts = get_ppe_missing_counts(events)
    total = sum(counts.values())
    if total == 0:
        return {key: 0.0 for key in counts}
    return {key: value / total for key, value in counts.items()}


def get_event_type_counts(events: List[ViolationEvent]) -> Dict[str, int]:
    return dict(Counter(getattr(event, "event_type", "ppe_violation") or "ppe_violation" for event in events))


def get_severity_counts(events: List[ViolationEvent]) -> Dict[str, int]:
    return dict(Counter(getattr(event, "severity", "medium") or "medium" for event in events))


def get_crowd_region_counts(events: List[ViolationEvent]) -> Dict[str, int]:
    crowd_events = [event for event in events if getattr(event, "event_type", "") == "crowd_gathering"]
    return dict(Counter((getattr(event, "region_name", "") or "unknown") for event in crowd_events))


def get_crowd_person_stats(events: List[ViolationEvent]) -> Dict[str, float]:
    person_counts = [
        int(getattr(event, "person_count", 0) or 0)
        for event in events
        if getattr(event, "event_type", "") == "crowd_gathering"
    ]
    if not person_counts:
        return {"max_person_count": 0, "average_person_count": 0.0}
    return {
        "max_person_count": max(person_counts),
        "average_person_count": sum(person_counts) / len(person_counts),
    }


def build_dashboard_summary(events: List[ViolationEvent]) -> Dict[str, Any]:
    missing_counts = get_ppe_missing_counts(events)
    trend = get_violation_trend(events, interval="minute" if len(events) > 10 else "10s")

    return {
        "total_violations": len(events),
        "event_type_counts": get_event_type_counts(events),
        "severity_counts": get_severity_counts(events),
        "missing_counts": missing_counts,
        "crowd_region_counts": get_crowd_region_counts(events),
        "crowd_person_stats": get_crowd_person_stats(events),
        "trend": trend,
        "recent_events": [
            {
                "time": event.timestamp,
                "id": event.track_id,
                "event_type": getattr(event, "event_type", "ppe_violation"),
                "severity": getattr(event, "severity", "medium"),
                "missing": event.missing_items,
                "conf": event.confidence,
                "details": getattr(event, "details", "") or event.missing_items,
            }
            for event in events[-5:][::-1]
        ],
    }
