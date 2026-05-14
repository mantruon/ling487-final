# setup.py
# One-shot environment setup for Mac and Windows using uv.
# Detects your platform and installs the correct PyTorch build,
# then installs all other dependencies from requirements.txt.
#
# Install uv first if you don't have it:
#   Mac/Linux:  curl -LsSf https://astral.sh/uv/install.sh | sh
#   Windows:    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
#
# Usage:
#   Mac / Linux:  python setup.py
#   Windows:      python setup.py

import subprocess
import sys
import platform
import shutil

SYSTEM = platform.system()


def run(cmd: list[str]):
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n❌ Command failed: {' '.join(cmd)}")
        sys.exit(result.returncode)


def ensure_uv():
    if shutil.which("uv") is None:
        print("❌ uv is not installed. Install it first:\n")
        if SYSTEM in ("Darwin", "Linux"):
            print("  curl -LsSf https://astral.sh/uv/install.sh | sh")
        else:
            print("  powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\"")
        print("\nThen re-run: python setup.py")
        sys.exit(1)
    result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
    print(f"✅ uv found: {result.stdout.strip()}")


def create_venv():
    print("\n🐍 Ensuring Python 3.12 is available...")
    run(["uv", "python", "install", "3.12"])
    print("\n📁 Creating virtual environment with uv (Python 3.12)...")
    run(["uv", "venv", ".venv", "--python", "3.12"])


def install_torch():
    uv_pip = ["uv", "pip", "install", "--python",
              ".venv/bin/python" if SYSTEM != "Windows" else ".venv\\Scripts\\python.exe"]

    if SYSTEM == "Darwin":
        print("\n🍎 Detected macOS — installing PyTorch with MPS support...")
        run(uv_pip + ["torch>=2.0.0", "torchaudio>=2.0.0"])

    elif SYSTEM in ("Windows", "Linux"):
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            has_gpu = result.returncode == 0
        except FileNotFoundError:
            has_gpu = False

        if has_gpu:
            # cu128 required for RTX 50 series (Blackwell, sm_120)
            print("\n🟢 NVIDIA GPU detected — installing PyTorch with CUDA 12.8...")
            run(uv_pip + [
                "torch>=2.0.0", "torchaudio>=2.0.0",
                "--index-url", "https://download.pytorch.org/whl/cu128"
            ])
        else:
            print("\n💻 No GPU detected — installing CPU-only PyTorch...")
            run(uv_pip + ["torch>=2.0.0", "torchaudio>=2.0.0"])
    else:
        print(f"\n⚠️  Unknown platform '{SYSTEM}' — installing default PyTorch...")
        run(uv_pip + ["torch>=2.0.0", "torchaudio>=2.0.0"])


def install_requirements():
    print("\n📦 Installing dependencies from requirements.txt...")
    python = (".venv/bin/python" if SYSTEM != "Windows"
              else ".venv\\Scripts\\python.exe")
    run(["uv", "pip", "install", "--python", python, "-r", "requirements.txt"])


def verify():
    print("\n🔍 Verifying install...")
    python = (".venv/bin/python" if SYSTEM != "Windows"
              else ".venv\\Scripts\\python.exe")
    verify_script = (
        "import torch, platform; "
        "system = platform.system(); "
        "device = 'mps' if system == 'Darwin' and torch.backends.mps.is_available() "
        "else f'cuda ({torch.cuda.get_device_name(0)})' if torch.cuda.is_available() "
        "else 'cpu'; "
        "print(f'  torch version : {torch.__version__}'); "
        "print(f'  active device : {device}'); "
        "print(f'  platform      : {system}')"
    )
    subprocess.run([python, "-c", verify_script])


def print_activate_instructions():
    print("\n── Activate your environment before running scripts ──")
    if SYSTEM in ("Darwin", "Linux"):
        print("  source .venv/bin/activate")
    else:
        print("  .venv\\Scripts\\activate")


if __name__ == "__main__":
    print(f"── Whisper Probe Setup ({'macOS' if SYSTEM == 'Darwin' else SYSTEM}) ──")
    ensure_uv()
    create_venv()
    install_torch()
    install_requirements()
    verify()
    print_activate_instructions()
    print("\n✅ Setup complete. Run scripts in order: 01 → 02 → 03 → 04 → 05 → 06")
