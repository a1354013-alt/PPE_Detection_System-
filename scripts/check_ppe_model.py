import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ppe_model_registry import (
    build_capability_summary,
    get_missing_required_classes,
    get_profile,
    normalize_class_name,
)


MODEL_NOT_FOUND_MESSAGE = "Model file not found. Run: python scripts/download_ppe_models.py --model all"


def get_model_names(model_path):
    from ultralytics import YOLO

    model = YOLO(model_path)
    names = getattr(model, "names", {})
    if isinstance(names, dict):
        return list(names.values())
    return list(names)


def validate_model_contract(model_path, profile=None, names=None, output=True):
    if not os.path.exists(model_path):
        if output:
            print(MODEL_NOT_FOUND_MESSAGE)
        return 2

    try:
        detected_names = list(names) if names is not None else get_model_names(model_path)
    except Exception as exc:
        if output:
            print(f"Failed to load model '{model_path}': {exc}")
        return 2

    capabilities = build_capability_summary(detected_names)
    missing = get_missing_required_classes(detected_names, profile) if profile else []

    if output:
        print(f"Model path: {model_path}")
        print(f"Detected class names: {', '.join(map(str, detected_names)) or 'None'}")
        if profile:
            normalized_detected = {normalize_class_name(name) for name in detected_names}
            matched_required = [
                required for required in profile.expected_classes
                if normalize_class_name(required) in normalized_detected
            ]
            print(f"Matched required classes: {', '.join(matched_required) or 'None'}")
            print(f"Missing required classes: {', '.join(missing) or 'None'}")
        print("Capability summary:")
        for key in ("person", "helmet", "no_helmet", "safety_vest", "no_safety_vest", "mask", "no_mask"):
            print(f"  {key}: {str(capabilities[key]).lower()}")

    if profile:
        return 0 if not missing else 1
    has_ppe_key_class = any(
        capabilities[key]
        for key in ("helmet", "no_helmet", "safety_vest", "no_safety_vest", "mask", "no_mask", "goggles", "no_goggles")
    )
    return 0 if capabilities["person"] and has_ppe_key_class else 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Check a PPE YOLO model contract.")
    parser.add_argument("--model-path", help="Path to a .pt model file.")
    parser.add_argument("--profile", choices=["hexmon", "hansung"], help="Known PPE model profile to validate.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    profile = get_profile(args.profile) if args.profile else None
    model_path = args.model_path or (profile.local_path if profile else None)
    if not model_path:
        print("Please provide --model-path or --profile.")
        return 2
    return validate_model_contract(model_path, profile=profile)


if __name__ == "__main__":
    sys.exit(main())
