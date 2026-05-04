import os
import subprocess
import sys


def print_status(ok):
    print("OK" if ok else "FAIL")


def build_python_command(*args):
    return [sys.executable, *args]


def run_command(command, description):
    print(f"Running: {description}...", end=" ", flush=True)
    try:
        if isinstance(command, str):
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
        else:
            result = subprocess.run(command, shell=False, capture_output=True, text=True)

        if result.returncode == 0:
            print_status(True)
            return True, result.stdout

        print_status(False)
        print(f"\nError in {description}:\n{result.stderr or result.stdout}")
        return False, result.stderr or result.stdout
    except Exception as exc:
        print_status(False)
        print(f"\nException in {description}: {exc}")
        return False, str(exc)


def check_tests():
    success1, _ = run_command(build_python_command("-m", "compileall", "-q", "."), "compileall")
    success2, _ = run_command(build_python_command("-m", "unittest", "discover", "-v"), "tests")
    return success1 and success2


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
        "python -m unittest discover -v",
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
    ]

    missing = [pkg for pkg in required_pkgs if pkg not in req_content]
    if missing:
        print_status(False)
        print(f"\nError: requirements.txt is missing: {', '.join(missing)}")
        return False

    print_status(True)
    return True


def check_forbidden_artifacts():
    print("Checking for forbidden artifacts...", end=" ", flush=True)
    forbidden_extensions = (".pt", ".pth", ".onnx", ".engine", ".weights", ".mp4", ".avi", ".mov", ".mkv", ".zip")
    forbidden_dirs = ("reports", "violations")
    ignored_dirs = {"venv", ".venv", "__pycache__", ".pytest_cache", ".git"}

    found = []
    for root, dirs, files in os.walk("."):
        dirs[:] = [directory for directory in dirs if directory not in ignored_dirs and not directory.startswith(".")]

        rel_root = os.path.relpath(root, ".")
        in_forbidden_dir = any(rel_root == forbidden_dir or rel_root.startswith(forbidden_dir + os.sep) for forbidden_dir in forbidden_dirs)

        for filename in files:
            rel_path = os.path.normpath(os.path.join(root, filename))
            if filename.endswith(forbidden_extensions) or in_forbidden_dir:
                found.append(rel_path)

    if found:
        print_status(False)
        print("\nError: Forbidden artifacts found:\n" + "\n".join(found))
        return False

    print_status(True)
    return True


def main():
    print("=== PPE Detection System Delivery Verification ===\n")

    checks = [
        ("compileall & tests", check_tests),
        ("ignore rules", check_ignore_rules),
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
