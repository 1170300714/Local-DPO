import torch
from extensions.xfuser.core.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from .transformer import parallelize_transformer
from .vae import parallelize_vae

from utils.logger import get_logger

logger = get_logger()

def hybrid_parallelize(
    transformer,
    vae,
    data_parallel_degree: int = 1,
    cfg_degree: int = 1,
    ulysses_degree: int = 1,
    ring_degree: int = 1,
):
    if not torch.distributed.is_initialized():
        logger.warning("Distributed environment is not initialized, model will remain unparallelized ...")
        return transformer, vae

    init_distributed_environment()

    if not model_parallel_is_initialized():
        logger.warning("Model parallel is not initialized, initializing...")
        initialize_model_parallel(
            data_parallel_degree=data_parallel_degree,
            classifier_free_guidance_degree=cfg_degree,
            sequence_parallel_degree=ulysses_degree * ring_degree,
            ulysses_degree=ulysses_degree,
            ring_degree=ring_degree,
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
            vae_parallel_size=0,
        )

    parallelize_transformer(transformer)
    parallelize_vae(vae)

    return transformer, vae


def hybrid_parallelize_pipeline(
    pipe,
    data_parallel_degree: int = 1,
    cfg_degree: int = 1,
    ulysses_degree: int = 1,
    ring_degree: int = 1,
):
    hybrid_parallelize(
        pipe.transformer,
        pipe.vae,
        data_parallel_degree=data_parallel_degree,
        cfg_degree=cfg_degree,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
    )
    return pipe
