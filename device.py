"""Where the tensors go, and whether autocast is on.

Training and sampling were written on LUMI (AMD MI250X, ROCm), where torch reports
the GPU as "cuda".  The same call works on NVIDIA.  On a machine with no GPU the
code still runs -- slowly -- and autocast is disabled, because bfloat16 autocast on
CPU is either unsupported or much slower than plain float32.
"""
import contextlib


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp(dev):
    """bfloat16 autocast on GPU, nothing on CPU."""
    import torch
    if dev.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()
