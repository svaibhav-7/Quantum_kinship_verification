# -*- coding: utf-8 -*-
"""
=================================================================================
  QUANTUM KINSHIP VERIFICATION -- COLAB DIRECT RUNNER
=================================================================================

Run directly in Google Colab:
    !python colab_run.py

This script:
  1. Detects Colab & GPU environment (T4/A100/V100).
  2. Auto-installs missing dependencies (facenet-pytorch, qiskit, qiskit-aer).
  3. Verifies project path in Google Drive.
  4. Runs GPU-accelerated retraining (50% FIW + KFW-II + TSKinFace).
  5. Saves retrained ensemble weights to weights/active_ensemble/.
"""

import os
import sys
import subprocess

print("=" * 72)
print("  QUANTUM KINSHIP -- COLAB GPU RUNNER")
print("=" * 72)

# 1. Check Python & PyTorch CUDA
import torch
print(f"[1/4] PyTorch Version: {torch.__version__}")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"  [GPU DETECTED] Using GPU: {gpu_name}")
    print(f"  [GPU Memory]   Allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
else:
    print("  [WARNING] CUDA GPU not detected! Training will fall back to CPU.")
    print("  To enable GPU in Colab: Runtime -> Change runtime type -> T4 GPU")

# 2. Check & Install Dependencies
print("\n[2/4] Checking required packages...")
required_pkgs = ["facenet_pytorch", "qiskit", "qiskit_aer", "sklearn", "matplotlib"]
missing = []
for pkg in required_pkgs:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)

if missing:
    print(f"  Installing missing packages: {missing}...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "facenet-pytorch", "qiskit", "qiskit-aer", "scikit-learn", "matplotlib", "scipy", "pillow"
    ])
    print("  Packages successfully installed!")
else:
    print("  All required packages are installed.")

# 3. Project Root Resolution
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Check if we are running in Colab and set working directory
in_colab = 'google.colab' in sys.modules
if in_colab:
    print("\n[3/4] Running inside Google Colab Environment")
    drive_path = "/content/drive/MyDrive/Quantum_kinship"
    if os.path.exists(drive_path):
        os.chdir(drive_path)
        project_root = drive_path
        print(f"  Switched working directory to Google Drive: {os.getcwd()}")
    else:
        print(f"  Current working directory: {os.getcwd()}")
else:
    print(f"\n[3/4] Local Environment Detected: {project_root}")

# 4. Execute Training Script
script_path = os.path.join(project_root, "scripts", "training", "train_with_fiw.py")
if not os.path.exists(script_path):
    # Search fallback locations
    alt_script = os.path.join(project_root, "train_with_fiw.py")
    if os.path.exists(alt_script):
        script_path = alt_script
    else:
        print(f"  [ERROR] Cannot find train_with_fiw.py at {script_path}")
        sys.exit(1)

print(f"\n[4/4] Launching Training Script: {script_path}")
print("=" * 72 + "\n")

cmd = [sys.executable, script_path, "--epochs", "100", "--batch-size", "64"]
res = subprocess.run(cmd)

if res.returncode == 0:
    print("\n" + "=" * 72)
    print("  COLAB TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 72)
    if in_colab:
        try:
            from google.colab import files
            weights_file = os.path.join(project_root, "weights", "active_ensemble", "ensemble_kinship_fiw.pt")
            if not os.path.exists(weights_file):
                weights_file = os.path.join(project_root, "weights", "ensemble_kinship_fiw.pt")
            if os.path.exists(weights_file):
                print(f"  Triggering download for: {os.path.basename(weights_file)}")
                files.download(weights_file)
        except Exception as e:
            print(f"  Download prompt info: {e}")
else:
    print(f"\n[ERROR] Training failed with return code {res.returncode}")
    sys.exit(res.returncode)
