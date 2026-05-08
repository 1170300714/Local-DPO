
import torch
import random
import numpy as np



def get_deterministic_worker_init_fn(seed):

    def deterministic_worker_init_fn(worker_id):
        worker_seed = seed
        if seed is not None:
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)
            random.seed(worker_seed)

    return deterministic_worker_init_fn