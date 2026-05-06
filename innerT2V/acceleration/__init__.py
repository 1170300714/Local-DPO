import torch
from .attention import optimize_transformer
from .activation_checkpoint import optimize_activation_checkpoint

from utils.logger import get_logger

logger = get_logger()


__all__ = ['accelerate_all']








def accelerate_all(*args, **kwargs):
    """
    Optimize one or two transformers along with text_encoder and vae.
    Supports flexible input for single or dual transformer cases.

    Args:
        *args: Variable arguments. The first argument is always the transformer(s),
               followed by text_encoder and vae.
        **kwargs: Additional keyword arguments for optimization.

    Returns:
        Optimized transformers, text_encoder, and vae.
        - If dual transformers are provided, returns (transformer1, transformer2), text_encoder, vae.
        - If a single transformer is provided, returns transformer1, text_encoder, vae.
    """
    if len(args) < 3:
        raise ValueError("Insufficient arguments. Expected at least transformer(s), text_encoder, and vae.")
    
    if isinstance(args[0], tuple) and len(args[0]) == 2:
        transformer1, transformer2 = args[0]
        text_encoder, vae = args[1], args[2]
        dual_transformer = True
    else:
        transformer1 = args[0]
        transformer2 = None
        text_encoder, vae = args[1], args[2]
        dual_transformer = False

    if 'ppu' in torch.cuda.get_device_name().lower():
        logger.warning("[Perf] Disable custom acceleration on PPU cluster")
        return (transformer1, transformer2, text_encoder, vae) if dual_transformer else (transformer1, text_encoder, vae)

    logger.info("[Perf] Disable attention recomputation in activation checkpointing")
    optimize_activation_checkpoint()

    logger.info("[Perf] Speedup transformer with fa3")
    transformer1 = optimize_transformer(transformer1)
    if dual_transformer:
        logger.info("[Perf] Speedup second transformer with fa3")
        transformer2 = optimize_transformer(transformer2)

    return (transformer1, transformer2, text_encoder, vae) if dual_transformer else (transformer1, text_encoder, vae)

