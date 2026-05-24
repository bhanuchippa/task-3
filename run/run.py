import os
import sys
import subprocess

def print_banner():
    banner = """
==============================================================
    AI Image Classifier - Automated Setup and Runner
==============================================================
"""
    print(banner)

def get_project_root():
    # Resolve the root directory (parent of the 'run' folder)
    run_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(run_dir, ".."))

def main():
    print_banner()
    project_root = get_project_root()
    os.chdir(project_root)
    
    # Store the absolute path of this launcher script
    script_path = os.path.abspath(sys.argv[0])
    
    # Determine the virtual environment paths
    venv_dir = os.path.join(project_root, "venv")
    if os.name == 'nt':
        venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        venv_python = os.path.join(venv_dir, "bin", "python")
        
    # Check if we are running inside the virtual environment
    current_executable = os.path.abspath(sys.executable)
    expected_executable = os.path.abspath(venv_python)
    
    in_venv = (sys.prefix != sys.base_prefix) or (current_executable == expected_executable)
    
    if not in_venv:
        # Create virtual environment if it does not exist
        if not os.path.exists(expected_executable):
            print("[INFO] Python virtual environment (venv) not found.")
            print("[INFO] Creating virtual environment (venv)...")
            try:
                subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
                print("[INFO] Virtual environment created successfully.")
            except Exception as e:
                print(f"[ERROR] Failed to create virtual environment: {e}")
                input("Press Enter to exit...")
                sys.exit(1)
        
        # Relaunch the script inside the virtual environment
        print("[INFO] Activating virtual environment and relaunching...")
        try:
            sys.exit(subprocess.call([venv_python, script_path] + sys.argv[1:]))
        except Exception as e:
            print(f"[ERROR] Failed to relaunch inside virtual environment: {e}")
            input("Press Enter to exit...")
            sys.exit(1)

    # -------------------------------------------------------------
    # From here on, we are guaranteed to be running inside the venv
    # -------------------------------------------------------------
    print("[INFO] Running inside virtual environment.")
    
    # Install dependencies
    print("[INFO] Checking and installing dependencies (this may take a minute)...")
    try:
        # Upgrade pip silently
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Install packages in requirements.txt
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("[INFO] Dependencies are up to date.")
    except Exception as e:
        print(f"[ERROR] Failed to install dependencies: {e}")
        input("Press Enter to exit...")
        sys.exit(1)
        
    # Check for model file
    model_path = os.path.join("model", "mobilenet_model.h5")
    if not os.path.exists(model_path):
        print("\n--------------------------------------------------------------")
        print("[NOTICE] Local trained model file (model/mobilenet_model.h5) was not found.")
        print("")
        print("Options:")
        print("  [1] Start Web App immediately (uses high-accuracy ImageNet mapping)")
        print("  [2] Train local model first (takes 10-30 minutes, requires TensorFlow)")
        print("--------------------------------------------------------------\n")
        
        choice = input("Enter your choice (1 or 2, default is 1): ").strip()
        if choice == "2":
            print("[INFO] Starting training...")
            try:
                # Run the model training script
                train_script = os.path.join("model", "train_model.py")
                subprocess.run([sys.executable, train_script], check=True)
            except Exception as e:
                print(f"[ERROR] Training failed: {e}")
                input("Press Enter to exit...")
                sys.exit(1)
        else:
            print("[INFO] Skipping training. Running web app directly using ImageNet mapping...")

    # Start the Flask app
    print("\n[INFO] Starting Flask web application...")
    print("Open your browser and go to: http://localhost:5000\n")
    try:
        subprocess.run([sys.executable, "app.py"])
    except KeyboardInterrupt:
        print("\n[INFO] Stopping server...")
    except Exception as e:
        print(f"[ERROR] Application failed to run: {e}")
        input("Press Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    main()
