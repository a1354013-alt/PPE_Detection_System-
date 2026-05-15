import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class PPEModelProfile:
    key: str
    display_name: str
    repo_id: str
    filename: str
    local_path: str
    expected_classes: List[str]


def normalize_label(label) -> str:
    lowered = str(label).strip().lower()
    lowered = lowered.replace("-", " ").replace("_", " ")
    return re.sub(r"\s+", " ", lowered)


CLASS_ALIASES: Dict[str, str] = {
    "person": "person",
    "worker": "person",
    "people": "person",
    "helmet": "helmet",
    "hardhat": "helmet",
    "hard hat": "helmet",
    "safety helmet": "helmet",
    "no helmet": "no_helmet",
    "no hardhat": "no_helmet",
    "no hard hat": "no_helmet",
    "without helmet": "no_helmet",
    "vest": "safety_vest",
    "safety vest": "safety_vest",
    "reflective vest": "safety_vest",
    "no vest": "no_safety_vest",
    "no safety vest": "no_safety_vest",
    "without vest": "no_safety_vest",
    "no reflective vest": "no_safety_vest",
    "mask": "mask",
    "face mask": "mask",
    "no mask": "no_mask",
    "without mask": "no_mask",
    "goggles": "goggles",
    "safety goggles": "goggles",
    "no goggles": "no_goggles",
    "without goggles": "no_goggles",
}


def normalize_class_name(label) -> str:
    sanitized = normalize_label(label)
    return CLASS_ALIASES.get(sanitized, sanitized.replace(" ", "_"))


PPE_MODEL_PROFILES: Dict[str, PPEModelProfile] = {
    "hexmon_vyra_yolo_ppe": PPEModelProfile(
        key="hexmon_vyra_yolo_ppe",
        display_name="Hexmon Vyra YOLO PPE",
        repo_id="Hexmon/vyra-yolo-ppe-detection",
        filename="best.pt",
        local_path=os.path.join("models", "hexmon_vyra_yolo_ppe_best.pt"),
        expected_classes=[
            "Person",
            "Hardhat",
            "NO-Hardhat",
            "Safety Vest",
            "NO-Safety Vest",
            "Mask",
            "NO-Mask",
        ],
    ),
    "hansung_yolov8_ppe": PPEModelProfile(
        key="hansung_yolov8_ppe",
        display_name="Hansung YOLOv8 PPE",
        repo_id="Hansung-Cho/yolov8-ppe-detection",
        filename="best.pt",
        local_path=os.path.join("models", "hansung_yolov8_ppe_best.pt"),
        expected_classes=[
            "Person",
            "Hardhat",
            "No-Hardhat",
            "Safety Vest",
            "No-Safety Vest",
            "Mask",
            "No-Mask",
        ],
    ),
}

PROFILE_ALIASES = {
    "hexmon": "hexmon_vyra_yolo_ppe",
    "vyra": "hexmon_vyra_yolo_ppe",
    "hexmon_vyra_yolo_ppe": "hexmon_vyra_yolo_ppe",
    "hansung": "hansung_yolov8_ppe",
    "hansung_yolov8_ppe": "hansung_yolov8_ppe",
}

CAPABILITY_KEYS = (
    "person",
    "helmet",
    "no_helmet",
    "safety_vest",
    "no_safety_vest",
    "mask",
    "no_mask",
    "goggles",
    "no_goggles",
)


def get_profile(name: str) -> PPEModelProfile:
    key = PROFILE_ALIASES.get(str(name).strip().lower())
    if not key:
        raise KeyError(f"Unknown PPE model profile: {name}")
    return PPE_MODEL_PROFILES[key]


def normalized_class_set(class_names: Iterable[str]) -> set:
    return {normalize_class_name(name) for name in class_names}


def expected_class_set(profile: PPEModelProfile) -> set:
    return normalized_class_set(profile.expected_classes)


def get_missing_required_classes(class_names: Iterable[str], profile: PPEModelProfile) -> List[str]:
    detected = normalized_class_set(class_names)
    missing = expected_class_set(profile) - detected
    return sorted(missing)


def build_capability_summary(class_names: Iterable[str]) -> Dict[str, bool]:
    detected = normalized_class_set(class_names)
    return {key: key in detected for key in CAPABILITY_KEYS}
