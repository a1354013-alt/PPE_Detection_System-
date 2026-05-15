import argparse
import os
import shutil
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ppe_model_registry import get_profile
from scripts.check_ppe_model import validate_model_contract


MODEL_CHOICES = ("hexmon", "hansung", "all")


def download_from_hugging_face(repo_id, filename):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename)


def resolve_target_path(profile, out_dir):
    return os.path.join(out_dir, os.path.basename(profile.local_path))


def download_profile(profile_name, out_dir="models", force=False):
    profile = get_profile(profile_name)
    target_path = resolve_target_path(profile, out_dir)
    os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)

    if os.path.exists(target_path) and not force:
        print(f"Model already exists, skipping download: {target_path}")
    else:
        try:
            downloaded_path = download_from_hugging_face(profile.repo_id, profile.filename)
            if os.path.abspath(downloaded_path) != os.path.abspath(target_path):
                shutil.copy2(downloaded_path, target_path)
            print(f"Downloaded {profile.display_name}: {target_path}")
        except Exception as exc:
            print(f"Failed to download {profile.display_name} from {profile.repo_id}: {exc}")
            return 1

    print(f"Local path: {target_path}")
    check_code = validate_model_contract(target_path, profile=profile)
    if check_code != 0:
        print(f"Downloaded model did not satisfy the expected PPE contract: {profile.display_name}")
    return check_code


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Download recommended PPE models from Hugging Face.")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="all", help="Model profile to download.")
    parser.add_argument("--force", action="store_true", help="Re-download even when the local model file exists.")
    parser.add_argument("--out-dir", default="models", help="Directory for downloaded model files.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    selected = ["hexmon", "hansung"] if args.model == "all" else [args.model]
    exit_code = 0
    for profile_name in selected:
        result = download_profile(profile_name, out_dir=args.out_dir, force=args.force)
        exit_code = max(exit_code, result)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
