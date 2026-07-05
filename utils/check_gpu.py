#!/usr/bin/env python3
"""
TriNetra-AMRF — GPU & CUDA Diagnostics Utility
================================================

Verifies that the system has a compatible NVIDIA GPU with CUDA support.
Prints detailed information about:
  - PyTorch CUDA availability and version
  - GPU device name, compute capability, and memory
  - NVIDIA driver version (via nvidia-smi)
  - GPU utilization (via GPUtil / pynvml)

Usage:
    python utils/check_gpu.py

This script is intended to be run during environment setup (Phase 1)
to confirm that the hardware stack is ready for training and inference.
"""

import sys
import subprocess
import shutil


def check_torch_cuda():
    """Check PyTorch's CUDA availability and print device details."""
    print("=" * 64)
    print("  TriNetra-AMRF — GPU Diagnostics Report")
    print("=" * 64)
    print()

    try:
        import torch
    except ImportError:
        print("[ERROR] PyTorch is not installed.")
        print("        Install via: pip install torch torchvision --index-url "
              "https://download.pytorch.org/whl/cu121")
        return False

    print(f"  PyTorch version    : {torch.__version__}")
    print(f"  CUDA available     : {torch.cuda.is_available()}")
    print(f"  CUDA version       : {torch.version.cuda or 'N/A'}")
    print(f"  cuDNN version      : {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
    print(f"  cuDNN enabled      : {torch.backends.cudnn.enabled}")
    print()

    if not torch.cuda.is_available():
        print("[WARNING] CUDA is NOT available. Training will fall back to CPU.")
        print("          Ensure NVIDIA drivers and CUDA toolkit are installed.")
        return False

    num_gpus = torch.cuda.device_count()
    print(f"  Number of GPUs     : {num_gpus}")
    print()

    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        mem_total = props.total_mem / (1024 ** 3)  # Convert to GB
        print(f"  ── GPU {i} ──────────────────────────────────────────────")
        print(f"     Name              : {props.name}")
        print(f"     Compute Capability: {props.major}.{props.minor}")
        print(f"     Total Memory      : {mem_total:.2f} GB")
        print(f"     Multi-Processors  : {props.multi_processor_count}")
        print()

    # Current device info
    current = torch.cuda.current_device()
    print(f"  Current device     : cuda:{current} ({torch.cuda.get_device_name(current)})")
    print()

    return True


def check_nvidia_smi():
    """Run nvidia-smi to get driver-level GPU information."""
    print("─" * 64)
    print("  NVIDIA Driver Info (nvidia-smi)")
    print("─" * 64)
    print()

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        print("[WARNING] nvidia-smi not found on PATH.")
        print("          NVIDIA drivers may not be installed.")
        print()
        return False

    try:
        result = subprocess.run(
            [nvidia_smi,
             "--query-gpu=name,driver_version,memory.total,memory.free,temperature.gpu,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    print(f"     GPU Name          : {parts[0]}")
                    print(f"     Driver Version    : {parts[1]}")
                    print(f"     Total Memory      : {parts[2]} MiB")
                    print(f"     Free Memory       : {parts[3]} MiB")
                    print(f"     Temperature       : {parts[4]} °C")
                    print(f"     GPU Utilization   : {parts[5]} %")
                    print()
        else:
            print(f"[ERROR] nvidia-smi returned exit code {result.returncode}")
            if result.stderr:
                print(f"        {result.stderr.strip()}")
            print()
            return False
    except subprocess.TimeoutExpired:
        print("[ERROR] nvidia-smi timed out.")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to run nvidia-smi: {e}")
        return False

    return True


def check_gputil():
    """Use GPUtil library for additional GPU monitoring info."""
    print("─" * 64)
    print("  GPUtil / pynvml Status")
    print("─" * 64)
    print()

    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if not gpus:
            print("  No GPUs detected by GPUtil.")
        for gpu in gpus:
            print(f"     GPU {gpu.id}: {gpu.name}")
            print(f"       Load       : {gpu.load * 100:.1f} %")
            print(f"       Memory Used: {gpu.memoryUsed:.0f} / {gpu.memoryTotal:.0f} MiB "
                  f"({gpu.memoryUtil * 100:.1f} %)")
            print(f"       Temperature: {gpu.temperature} °C")
            print()
    except ImportError:
        print("  [INFO] GPUtil not installed. Install via: pip install GPUtil")
        print()

    try:
        import pynvml
        pynvml.nvmlInit()
        driver_ver = pynvml.nvmlSystemGetDriverVersion()
        print(f"  pynvml driver version: {driver_ver}")
        device_count = pynvml.nvmlDeviceGetCount()
        print(f"  pynvml device count  : {device_count}")
        pynvml.nvmlShutdown()
        print()
    except ImportError:
        print("  [INFO] pynvml not installed. Install via: pip install pynvml")
        print()
    except Exception as e:
        print(f"  [WARNING] pynvml error: {e}")
        print()


def main():
    """Run all GPU diagnostic checks and print a summary."""
    cuda_ok = check_torch_cuda()
    smi_ok = check_nvidia_smi()
    check_gputil()

    # ── Summary ─────────────────────────────────────────────────────────────
    print("=" * 64)
    print("  Summary")
    print("=" * 64)
    if cuda_ok and smi_ok:
        print("  ✅ GPU stack is healthy. Ready for TriNetra training/inference.")
    elif cuda_ok:
        print("  ⚠️  PyTorch CUDA works, but nvidia-smi had issues.")
        print("      Training should still work, but driver checks failed.")
    else:
        print("  ❌ CUDA is NOT available. TriNetra requires a CUDA-capable GPU.")
        print("     Install NVIDIA drivers + CUDA toolkit, then reinstall PyTorch.")
    print()

    sys.exit(0 if cuda_ok else 1)


if __name__ == "__main__":
    main()
