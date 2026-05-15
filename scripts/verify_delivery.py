import os
import subprocess
import sys


def print_status(ok):
    print("OK" if ok else "FAIL")


def build_python_command(*args):
    return [sys.executable, *args]


def run_command(command, description, capture_output=True):
    print(f"Running: {description}...", end=" ", flush=True)
    try:
        if isinstance(command, str):
            result = subprocess.run(command, shell=True, capture_output=capture_output, text=True)
        else:
            result = subprocess.run(command, shell=False, capture_output=capture_output, text=True)

        if result.returncode == 0:
            print_status(True)
            return True, result.stdout if capture_output else ""

        print_status(False)
        if capture_output:
            print(f"\nError in {description}:\n{result.stderr or result.stdout}")
            return False, result.stderr or result.stdout
        return False, ""
    except Exception as exc:
        print_status(False)
        print(f"\nException in {description}: {exc}")
        return False, str(exc)


def check_tests():
    success1, _ = run_command(build_python_command("-m", "compileall", "-q", "."), "compileall", capture_output=False)
    success2, _ = run_command(build_python_command("-m", "unittest", "discover", "-v"), "unittest", capture_output=False)
    success3, _ = run_command(build_python_command("-m", "pytest", "-q"), "pytest", capture_output=False)
    return success1 and success2 and success3


def check_ignore_rules():
    print("Checking ignore rules...", end=" ", flush=True)
    gitignore_path = ".gitignore"
    if not os.path.exists(gitignore_path):
        print_status(False)
        print("\nError: .gitignore is missing")
        return False

    with open(gitignore_path, "r", encoding="utf-8") as file_obj:
        content = file_obj.read()

    required_rules = [
        "__pycache__/",
        "*.py[cod]",
        "*.pyo",
        "venv/",
        ".venv/",
        "env/",
        ".coverage",
        "coverage/",
        "htmlcov/",
        ".pytest_cache/",
        ".vscode/",
        ".idea/",
        "*.log",
        "*.tmp",
        ".env",
        ".env.local",
        "*.env.*",
        "reports/",
        "violations/",
        "outputs/",
        "runs/",
        "models/*.pt",
        "models/*.onnx",
        "models/*.engine",
        "!models/.gitkeep",
        "*.mp4",
        "*.avi",
        "*.mov",
        "*.mkv",
        "*.pt",
        "*.pth",
        "*.onnx",
        "*.engine",
        "*.weights",
    ]

    missing = [rule for rule in required_rules if rule not in content]
    has_code_fence = "```" in content

    if missing or has_code_fence:
        print_status(False)
        if missing:
            print(f"\nError: .gitignore is missing rules for: {', '.join(missing)}")
        if has_code_fence:
            print("\nError: .gitignore contains Markdown code fences and is not a clean Git ignore file")
        return False

    print_status(True)
    return True


def check_ppe_model_delivery_files():
    print("Checking PPE model delivery files...", end=" ", flush=True)
    required_paths = [
        "scripts/download_ppe_models.py",
        "scripts/check_ppe_model.py",
        "ppe_model_registry.py",
        "models/.gitkeep",
    ]
    missing = [path for path in required_paths if not os.path.exists(path)]
    if missing:
        print_status(False)
        print(f"\nError: Missing PPE model delivery files: {', '.join(missing)}")
        return False

    print_status(True)
    return True


def check_readme_commands():
    print("Checking README commands...", end=" ", flush=True)
    readme_path = "README.md"
    if not os.path.exists(readme_path):
        print_status(False)
        print("\nError: README.md is missing")
        return False

    with open(readme_path, "r", encoding="utf-8") as file_obj:
        content = file_obj.read()

    required_commands = [
        "pip install -r requirements.txt",
        "python main_gui.py",
        "## PPE Model Setup",
        "python scripts/download_ppe_models.py --model all",
        "python scripts/check_ppe_model.py --profile hexmon",
        "python scripts/check_ppe_model.py --profile hansung",
        "python -m compileall -q .",
        "python -m unittest discover -v",
        "pytest -q",
        "python scripts/verify_delivery.py",
    ]

    forbidden_commands = ["python3.11 -m compileall", "python3.11 -m unittest discover"]

    missing = [cmd for cmd in required_commands if cmd not in content]
    present_forbidden = [cmd for cmd in forbidden_commands if cmd in content]

    if missing or present_forbidden:
        print_status(False)
        if missing:
            print(f"\nError: README.md is missing commands: {', '.join(missing)}")
        if present_forbidden:
            print(f"\nError: README.md still contains hard-coded Python commands: {', '.join(present_forbidden)}")
        return False

    print_status(True)
    return True


def check_requirements():
    print("Checking requirements...", end=" ", flush=True)
    req_path = "requirements.txt"
    if not os.path.exists(req_path):
        print_status(False)
        print("\nError: requirements.txt is missing")
        return False

    with open(req_path, "r", encoding="utf-8") as file_obj:
        req_content = file_obj.read().lower()

    required_pkgs = [
        "ultralytics",
        "opencv-python",
        "numpy",
        "pandas",
        "matplotlib",
        "pillow",
        "openpyxl",
        "reportlab",
        "pytest",
        "huggingface_hub>=0.23,<1",
    ]

    missing = [pkg for pkg in required_pkgs if pkg not in req_content]
    if missing:
        print_status(False)
        print(f"\nError: requirements.txt is missing: {', '.join(missing)}")
        return False

    print_status(True)
    return True


def check_forbidden_artifacts():
    print("Checking for tracked forbidden artifacts...", end=" ", flush=True)
    forbidden_extensions = (".pt", ".pth", ".onnx", ".engine", ".weights", ".mp4", ".avi", ".mov", ".mkv", ".zip")
    forbidden_dirs = ("reports/", "violations/", "outputs/", "runs/")
    try:
        result = subprocess.run(["git", "ls-files"], shell=False, capture_output=True, text=True)
        if result.returncode == 0:
            tracked_files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        else:
            tracked_files = []
    except Exception:
        tracked_files = []

    if not tracked_files:
        tracked_files = []
        for root, _dirs, files in os.walk("."):
            for filename in files:
                tracked_files.append(os.path.normpath(os.path.join(root, filename)).lstrip("." + os.sep))

    found = [
        path for path in tracked_files
        if path.endswith(forbidden_extensions) or any(path == directory.rstrip("/") or path.startswith(directory) for directory in forbidden_dirs)
    ]

    if found:
        print_status(False)
        print("\nError: Forbidden tracked artifacts found:\n" + "\n".join(found))
        return False

    print_status(True)
    return True


def main():
    print("=== PPE Detection System Delivery Verification ===\n")

    checks = [
        ("compileall, unittest, and pytest", check_tests),
        ("ignore rules", check_ignore_rules),
        ("PPE model delivery files", check_ppe_model_delivery_files),
        ("README commands", check_readme_commands),
        ("requirements", check_requirements),
        ("delivery artifacts clean", check_forbidden_artifacts),
    ]

    all_passed = True
    for _, func in checks:
        if not func():
            all_passed = False

    if all_passed:
        print("\nAll delivery checks passed. Ready for delivery!")
        sys.exit(0)

    print("\nDelivery verification failed. Please fix the errors above.")
    sys.exit(1)


if __name__ == "__main__":
    main()
