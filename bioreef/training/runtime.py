"""Runtime helpers: device resolution and a quiet image reader.

resolve_device picks the compute device from --gpu / config / auto; safe_imread is
a cv2.imread that suppresses libjpeg/libpng stderr spam on corrupt frames. Used by
the single-GPU training loop, the dataset, and the visualization scripts.
"""

import os
import sys

import cv2
import torch


def _normalize_device_spec(val):
    """Turn a --gpu / config 'device' value into a torch device string, or None.
    A bare index ('0', 0, 1) becomes 'cuda:0' — argparse hands the flag over as a
    string, so '0' must be treated as an index, not a literal device string. Full
    specs ('cuda:1', 'cpu', 'mps') pass through unchanged."""
    if val is None:
        return None
    s = str(val).strip()
    if s == "":
        return None
    if s.isdigit():                       # '0' / '1' -> a CUDA index
        return f"cuda:{s}"
    return s                              # 'cuda:1', 'cpu', 'mps', ...


def resolve_device(cli_gpu=None, config_device=""):
    """Pick the compute device. Precedence: --gpu flag > config 'device' > auto.
        cli_gpu (int|str|None): e.g. 2 or "cuda:2" or "cpu" from --gpu.
        config_device (str): the benchmark config's 'device' field.
    Falls back to cpu (with a note) if CUDA is unavailable."""
    spec = _normalize_device_spec(cli_gpu)
    if spec is None and config_device:
        spec = _normalize_device_spec(config_device)

    if spec is None:
        spec = "cuda:0" if torch.cuda.is_available() else "cpu"

    if spec.startswith("cuda") and not torch.cuda.is_available():
        print(f"[device] CUDA unavailable; '{spec}' -> cpu")
        spec = "cpu"
    return torch.device(spec)


def safe_imread(path: str):
    """cv2.imread that silences libjpeg/libpng stderr spam on corrupt frames."""
    stderr_fd = sys.stderr.fileno()
    old_stderr = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stderr_fd)
    os.close(devnull)
    try:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    finally:
        os.dup2(old_stderr, stderr_fd)
        os.close(old_stderr)
    return img
