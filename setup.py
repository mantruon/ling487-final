# setup.py
# One-shot environment setup for Mac and Windows.
# Detects your platform and installs the correct PyTorch build,
# then installs all other dependencies from requirements.txt.
#
# Usage:
#   Mac / Linux:  python setup.py
#   Windows:      python setup.py

import subprocess
import sys
import platform

SYSTEM = platform.system()   # "Darwin", "Windows", "Linux"


def run(cmd: list[str]):
    """Run a pip command and exit on failure."""
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n❌ Command failed: {' '.join(cmd)}")
        sys.exit(result.returncode)


def install_torch():
    """Install the correct PyTorch build for this platform."""
    pip = [sys.executable, "-m", "pip", "install"]

    if SYSTEM == "Darwin":
        # Mac — MPS support is included in the standard PyTorch build
        print("🍎 Detected macOS — installing PyTorch with MPS support…")
        run(pip + ["torch>=2.0.0", "torchaudio>=2.0.0"])

    elif SYSTEM == "Windows" or SYSTEM == "Linux":
        # Check for NVIDIA GPU
        try:
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True
            )
            has_gpu = result.returncode == 0
        except FileNotFoundError:
            has_gpu = False

        if has_gpu:
            print("🟢 NVIDIA GPU detected — installing PyTorch with CUDA 12.1…")
            run(pip + [
                "torch>=2.0.0",
                "torchaudio>=2.0.0",
                "--index-url", "https://download.pytorch.org/whl/cu121"
            ])
        else:
            print("💻 No GPU detected — installing CPU-only PyTorch…")
            run(pip + ["torch>=2.0.0", "torchaudio>=2.0.0"])

    else:
        print(f"⚠️  Unknown platform '{SYSTEM}' — installing default PyTorch…")
        run(pip + ["torch>=2.0.0", "torchaudio>=2.0.0"])


def install_requirements():
    """Install everything else from requirements.txt (excluding torch lines)."""
    print("\n📦 Installing remaining dependencies from requirements.txt…")
    run([
        sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
    ])


def verify():
    """Quick sanity check after install."""
    print("\n🔍 Verifying install…")
    verify_script = """
import torch, platform
system = platform.system()
if system == "Darwin" and torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = f"cuda ({torch.cuda.get_device_name(0)})"
else:
    device = "cpu"
print(f"  torch version : {torch.__version__}")
print(f"  active device : {device}")
print(f"  platform      : {system}")
"""
    subprocess.run([sys.executable, "-c", verify_script])


if __name__ == "__main__":
    print(f"── Whisper Probe Setup ({'macOS' if SYSTEM == 'Darwin' else SYSTEM}) ──")
    install_torch()
    install_requirements()
    verify()
    print("\n✅ Setup complete. Run scripts in order: 01 → 02 → 03 → 04 → 05 → 06")
