"""DDP + logging infra (torchrun-launched) and a quiet image reader.

setup_ddp/cleanup_ddp/report_memory are only used by the multi-GPU trainer;
safe_imread is used by the dataset on every backend.
"""

import logging
import os
import sys

import cv2
import torch
import torch.distributed as dist


def setup_ddp() -> int:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp() -> None:
    dist.destroy_process_group()


def get_logger(local_rank: int) -> logging.Logger:
    logger = logging.getLogger("train_ddp")
    logger.setLevel(logging.INFO if local_rank == 0 else logging.WARNING)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(ch)
    return logger


def report_memory(local_rank: int) -> str:
    allocated = torch.cuda.memory_allocated(local_rank) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(local_rank) / (1024 ** 3)
    return f"VRAM [GPU {local_rank}]: {allocated:.2f} GB / {reserved:.2f} GB"


def resolve_device(cli_gpu=None, config_device=""):
    """Pick the compute device. Precedence: --gpu flag > config 'device' > auto.
        cli_gpu (int|str|None): e.g. 2 or "cuda:2" or "cpu" from --gpu.
        config_device (str): the benchmark config's 'device' field.
    Falls back to cpu (with a note) if CUDA is unavailable."""
    spec = None
    if cli_gpu is not None:
        spec = cli_gpu if isinstance(cli_gpu, str) else f"cuda:{cli_gpu}"
    elif config_device:
        spec = config_device

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
