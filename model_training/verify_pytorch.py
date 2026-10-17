"""Verify that PyTorch is installed and can use the NVIDIA GPU."""

from __future__ import annotations

import platform
import sys
from pathlib import Path


def main() -> int:
    print("=" * 60)
    print("PyTorch installation verification")
    print("=" * 60)
    print(f"Python version     : {platform.python_version()}")
    print(f"Python executable  : {sys.executable}")

    try:
        import torch
    except Exception as exc:
        print("\n[FAILED] PyTorch could not be imported.")
        print(f"Error type         : {type(exc).__name__}")
        print(f"Error message      : {exc}")
        print("\nMake sure the pytorch_gpu environment is activated before running this script.")
        return 1

    torch_file = Path(torch.__file__).resolve()
    torch_dir = torch_file.parent

    print(f"Torch version      : {torch.__version__}")
    print(f"Torch install dir  : {torch_dir}")
    print(f"Torch module file  : {torch_file}")
    print(f"Built-in CUDA      : {torch.version.cuda or 'None'}")
    print(f"cuDNN version      : {torch.backends.cudnn.version() or 'None'}")
    print(f"CUDA available     : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\n[PARTIAL] PyTorch imports successfully, but CUDA is unavailable.")
        print("This may indicate a CPU build, an incompatible driver, or a CUDA configuration issue.")
        return 2

    gpu_count = torch.cuda.device_count()
    print(f"GPU count          : {gpu_count}")
    for index in range(gpu_count):
        props = torch.cuda.get_device_properties(index)
        memory_gib = props.total_memory / 1024**3
        print(
            f"GPU {index}              : {props.name} "
            f"({memory_gib:.2f} GiB, compute capability {props.major}.{props.minor})"
        )

    try:
        left = torch.randn((1024, 1024), device="cuda")
        right = torch.randn((1024, 1024), device="cuda")
        result = left @ right
        torch.cuda.synchronize()
        print(f"GPU tensor device  : {result.device}")
        print(f"GPU result shape   : {tuple(result.shape)}")
        print(f"GPU result checksum: {result.sum().item():.6f}")
    except Exception as exc:
        print("\n[FAILED] PyTorch found CUDA, but the GPU calculation failed.")
        print(f"Error type         : {type(exc).__name__}")
        print(f"Error message      : {exc}")
        return 3

    print("\n[SUCCESS] PyTorch is installed correctly and GPU computation works.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
