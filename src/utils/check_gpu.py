from __future__ import annotations

import sys


def main() -> None:
    print("Python:", sys.version)
    try:
        import torch
    except ImportError:
        print("PyTorch: not installed")
        print("CUDA available: False")
        print("No CUDA GPU detected by PyTorch.")
        return

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("PyTorch CUDA version:", torch.version.cuda)

    if torch.cuda.is_available():
        print("GPU count:", torch.cuda.device_count())
        print("Current device:", torch.cuda.current_device())
        print("GPU name:", torch.cuda.get_device_name(0))
    else:
        print("GPU count:", torch.cuda.device_count())
        print("No CUDA GPU detected by PyTorch.")


if __name__ == "__main__":
    main()
