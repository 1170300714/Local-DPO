import gc
import torch
from typing import Union

from .logger import get_logger

logger = get_logger()


def reset_memory(device: Union[str, torch.device]) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.reset_accumulated_memory_stats(device)


def print_memory(device: Union[str, torch.device], scope=None) -> None:
    memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
    max_memory_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    max_memory_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    header = "[Profile] GPU Memory"
    if scope is not None:
        header += f"({scope})"
    header += f": Alloc={memory_allocated:.3f} GB, MaxAlloc={max_memory_allocated:.3f} GB, MaxRSVD={max_memory_reserved:.3f} GB"
    logger.info(header)
