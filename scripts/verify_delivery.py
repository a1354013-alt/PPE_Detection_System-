import os
import sys
import subprocess
import re
import ast

def run_command(command, description):
    print(f"Running: {description}...", end=" ", flush=True)
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅")
            return True, result.stdout
        else:
            print("❌")
            print(f"\nError in {description}:\n{result.stderr or result.stdout}")
            return False, result.stderr or result.stdout
    except Exception as e:
        print("❌")
        print(f"\nException in {description}: {str(e)}")
        return False, str(e)

def check_tests():
    success1, _ = run_command("python3.11 -m compileall -q .", "compileall")
    # Use unittest discover as pytest might not be available or configured
    success2, _ = run_command("python3.11 -m unittest discover -v", "tests")
    return success1 and success2

def check_ignore_rules():
    print("Checking ignore rules...", end=" ", flush=True)
    gitignore_path = ".gitignore"
    if not os.path.exists(gitignore_path):
        print("❌")
        print("\nError: .gitignore is missing")
        return False
    
    with open(gitignore_path, "r") as f:
        content = f.read()
    
    required_rules = [
        "reports/", "violations/", "*.mp4", "*.avi", "*.mov", "*.mkv",
        "*.pt", "*.onnx", "*.engine", "*.weights", "*.pth"
    ]
    
    missing = []
    for rule in required_rules:
        # Simple check for rule existence
        if rule not in content:
            missing.append(rule)
    
    if missing:
        print("❌")
        print(f"\nError: .gitignore is missing rules for: {', '.join(missing)}")
        return False
    
    print("✅")
    return True

def check_readme_commands():
    print("Checking README commands...", end=" ", flush=True)
    readme_path = "README.md"
    if not os.path.exists(readme_path):
        print("❌")
        print("\nError: README.md is missing")
        return False
    
    with open(readme_path, "r") as f:
        content = f.read()
    
    required_commands = [
        "pip install -r requirements.txt",
        "python main_gui.py",
        "python3.11 -m unittest discover", # Adjusted for our test runner
        "python scripts/verify_delivery.py"
    ]
    
    missing = []
    for cmd in required_commands:
        if cmd not in content:
            missing.append(cmd)
    
    if missing:
        print("❌")
        print(f"\nError: README.md is missing commands: {', '.join(missing)}")
        return False
    
    print("✅")
    return True

def check_requirements():
    print("Checking requirements...", end=" ", flush=True)
    req_path = "requirements.txt"
    if not os.path.exists(req_path):
        print("❌")
        print("\nError: requirements.txt is missing")
        return False
    
    with open(req_path, "r") as f:
        req_content = f.read().lower()
    
    required_pkgs = [
        "ultralytics", "opencv-python", "numpy", "pandas", 
        "matplotlib", "pillow", "openpyxl", "reportlab"
    ]
    
    missing = []
    for pkg in required_pkgs:
        if pkg not in req_content:
            missing.append(pkg)
            
    if missing:
        print("❌")
        print(f"\nError: requirements.txt is missing: {', '.join(missing)}")
        return False
    
    print("✅")
    return True

def check_forbidden_artifacts():
    print("Checking for forbidden artifacts...", end=" ", flush=True)
    forbidden_extensions = ('.pt', '.pth', '.onnx', '.engine', '.weights', '.mp4', '.avi', '.mov', '.mkv', '.zip')
    forbidden_dirs = ('reports', 'violations')
    
    found = []
    
    for root, dirs, files in os.walk("."):
        # Skip hidden dirs and venv
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('venv', '.venv', '__pycache__', '.pytest_cache')]
        
        # Check files in current root
        for file in files:
            if file.endswith(forbidden_extensions):
                found.append(os.path.join(root, file))
        
        # Check if we are inside a forbidden dir and it has files
        rel_root = os.path.relpath(root, ".")
        for f_dir in forbidden_dirs:
            if rel_root == f_dir or rel_root.startswith(f_dir + os.sep):
                for file in files:
                    found.append(os.path.join(root, file))

    if found:
        print("❌")
        print(f"\nError: Forbidden artifacts found:\n" + "\n".join(found))
        return False
    
    print("✅")
    return True

def main():
    print("=== PPE Detection System Delivery Verification ===\n")
    
    checks = [
        ("compileall & tests", check_tests),
        ("ignore rules", check_ignore_rules),
        ("README commands", check_readme_commands),
        ("requirements", check_requirements),
        ("delivery artifacts clean", check_forbidden_artifacts)
    ]
    
    all_passed = True
    for name, func in checks:
        if not func():
            all_passed = False
            # We continue to show all errors
            
    if all_passed:
        print("\n✨ All delivery checks passed. Ready for delivery!")
        sys.exit(0)
    else:
        print("\n❌ Delivery verification failed. Please fix the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
