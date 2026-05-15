import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


Box = Sequence[float]


@dataclass
class CrowdRegion:
    name: str
    x1: float
    y1: float
    x2: float
    y2: float
    threshold: int = 5
    severity: str = "medium"

    def __post_init__(self):
        self.x1 = max(0.0, min(1.0, float(self.x1)))
        self.y1 = max(0.0, min(1.0, float(self.y1)))
        self.x2 = max(0.0, min(1.0, float(self.x2)))
        self.y2 = max(0.0, min(1.0, float(self.y2)))
        if self.x2 < self.x1:
            self.x1, self.x2 = self.x2, self.x1
        if self.y2 < self.y1:
            self.y1, self.y2 = self.y2, self.y1
        self.threshold = max(1, int(self.threshold))
        self.name = str(self.name or "Entrance")


def normalize_box(box: Box, frame_shape: Optional[Tuple[int, int]] = None) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(value) for value in box[:4]]
    if all(0.0 <= value <= 1.0 for value in (x1, y1, x2, y2)):
        return x1, y1, x2, y2

    if frame_shape is None:
        raise ValueError("frame_shape is required for absolute pixel boxes")

    height, width = frame_shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError("frame_shape must contain positive height and width")

    return (
        max(0.0, min(1.0, x1 / width)),
        max(0.0, min(1.0, y1 / height)),
        max(0.0, min(1.0, x2 / width)),
        max(0.0, min(1.0, y2 / height)),
    )


def box_center_in_region(box: Box, region: CrowdRegion, frame_shape: Optional[Tuple[int, int]] = None) -> bool:
    x1, y1, x2, y2 = normalize_box(box, frame_shape=frame_shape)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return region.x1 <= center_x <= region.x2 and region.y1 <= center_y <= region.y2


def count_people_in_region(
    person_boxes: Iterable[Box],
    region: CrowdRegion,
    frame_shape: Optional[Tuple[int, int]] = None,
) -> int:
    return sum(1 for box in person_boxes if box_center_in_region(box, region, frame_shape=frame_shape))


def update_temporal_state(state: Dict[str, int], region_name: str, is_candidate: bool, temporal_frames: int) -> int:
    if is_candidate:
        state[region_name] = state.get(region_name, 0) + 1
    else:
        state[region_name] = 0
    return state[region_name]


def is_in_cooldown(last_event_times: Dict[str, float], region_name: str, timestamp: float, cooldown_seconds: float) -> bool:
    last_time = last_event_times.get(region_name)
    return last_time is not None and timestamp - last_time < cooldown_seconds


class CrowdMonitor:
    def __init__(self, regions=None, temporal_frames=3, cooldown_seconds=10):
        self.regions: List[CrowdRegion] = list(regions or [])
        self.temporal_frames = max(1, int(temporal_frames))
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self.temporal_state: Dict[str, int] = {}
        self.last_event_times: Dict[str, float] = {}

    def reset(self):
        self.temporal_state.clear()
        self.last_event_times.clear()

    @staticmethod
    def _timestamp_to_seconds(timestamp):
        if isinstance(timestamp, (int, float)):
            return float(timestamp)
        return time.time()

    @staticmethod
    def _event_severity(region: CrowdRegion, person_count: int) -> str:
        if person_count >= region.threshold * 2:
            return "high"
        return region.severity or "medium"

    def update(self, person_boxes, frame_index, timestamp, frame_shape=None, source_name="unknown", is_demo=False):
        events = []
        timestamp_seconds = self._timestamp_to_seconds(timestamp)

        for region in self.regions:
            person_count = count_people_in_region(person_boxes, region, frame_shape=frame_shape)
            is_candidate = person_count >= region.threshold
            stable_count = update_temporal_state(
                self.temporal_state,
                region.name,
                is_candidate,
                self.temporal_frames,
            )

            if stable_count < self.temporal_frames:
                continue
            if is_in_cooldown(self.last_event_times, region.name, timestamp_seconds, self.cooldown_seconds):
                continue

            self.last_event_times[region.name] = timestamp_seconds
            severity = self._event_severity(region, person_count)
            demo_prefix = "Demo " if is_demo else ""
            details = (
                f"{demo_prefix}{region.name} region crowd alert: "
                f"{person_count} people detected, threshold is {region.threshold}."
            )
            events.append(
                {
                    "event_type": "crowd_gathering",
                    "category": "crowd",
                    "severity": severity,
                    "region_name": region.name,
                    "person_count": person_count,
                    "threshold": region.threshold,
                    "frame_index": frame_index,
                    "frame": frame_index,
                    "timestamp": timestamp,
                    "source": source_name,
                    "details": details,
                    "screenshot_path": "",
                    "missing_items": "",
                    "missing_list": [],
                    "confidence": 0.0,
                    "bbox": "",
                    "track_id": f"crowd:{region.name}",
                    "is_demo": bool(is_demo),
                }
            )

        return events
