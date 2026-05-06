import torch


parallel_config_template = {
    1: dict(
        data_parallel_degree=1,
        cfg_degree=1,
        ulysses_degree=1,
        ring_degree=1,
    ),
    2: dict(
        data_parallel_degree=1,
        cfg_degree=1,
        ulysses_degree=1,
        ring_degree=2,
    ),
    4: dict(
        data_parallel_degree=1,
        cfg_degree=1,
        ulysses_degree=2,
        ring_degree=2,
    ),
    6: dict(
        data_parallel_degree=1,
        cfg_degree=1,
        ulysses_degree=2,
        ring_degree=3,
    ),
    8: dict(
        data_parallel_degree=1,
        cfg_degree=2,
        ulysses_degree=2,
        ring_degree=2,
    ),
}


def get_parallel_config(mode: str):
    world_size = torch.distributed.get_world_size()
    if world_size <= 8:
        assert world_size in parallel_config_template, f"parallel config for {world_size} is not supported"
        cfg = parallel_config_template[world_size]
        if mode == 'train':
            if cfg.pop('cfg_degree') > 1:
                cfg['ring_degree'] *= 2
        return cfg

    assert world_size % 8 == 0, f"parallel config for {world_size} is not supported"
    cfg = {}
    if mode == 'train':
        cfg['ulysses_degree'] = 2
        cfg['ring_degree'] = 4
    else:
        cfg['cfg_degree'] = 2
        cfg['ulysses_degree'] = 2
        cfg['ring_degree'] = 2
    cfg['data_parallel_degree'] = world_size // 8
    return cfg
