from collections import defaultdict
from typing import Sequence, Dict, Optional

import torch
from torch.utils.data import Dataset, BatchSampler, RandomSampler, WeightedRandomSampler, SequentialSampler
from accelerate.logging import get_logger

logger = get_logger(__name__)


class BucketBatchSampler(BatchSampler):
    def __init__(
        self,
        data_source: Dataset,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        generator: torch.Generator = None,
        weights: Optional[Sequence[float]] = None,
        batch_size_scales: Dict[int, int] = {},
    ):
        self.data_source = data_source
        self.batch_size = None
        self._batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator
        self.buckets = defaultdict(list)
        self.batch_size_scales = batch_size_scales

        self._raised_warning_for_drop_last = False
        self._raised_warning_for_batch_size_scales = False

        if self.shuffle:
            if weights is not None:
                sampler = WeightedRandomSampler(weights, len(data_source), replacement=False, generator=self.generator)
            else:
                sampler = RandomSampler(data_source, generator=self.generator)
        else:
            sampler = SequentialSampler(data_source)
        self.sampler = sampler

    def __len__(self):
        if self.drop_last and not self._raised_warning_for_drop_last:
            self._raised_warning_for_drop_last = True
            logger.warning(
                "Calculating the length for bucket sampler is not possible when `drop_last` is set to True. This may cause problems when setting the number of epochs used for training."
            )
        if self.batch_size_scales and not self._raised_warning_for_batch_size_scales:
            self._raised_warning_for_batch_size_scales = True
            logger.warning(
                "Calculating the length for bucket sampler is not possible when `batch_size_scales` is set. This may cause problems when setting the number of epochs used for training."
            )
        return (len(self.data_source) + self._batch_size - 1) // self._batch_size

    def __iter__(self):
        for index in self.sampler:
            bucket_info = self.data_source._get_meta_info(index, seed=42 + index)
            f = bucket_info['num_frames']
            h = bucket_info['height']
            w = bucket_info['width']
            batch_scale = bucket_info.get('batch_scale', None)
            batch_scale = batch_scale or self.batch_size_scales.get(f, 1)

            bucket_id = f'{f}-{h}-{w}'
            self.buckets[bucket_id].append(f'{index}-{f}-{h}-{w}')
            target_batch_size = self._batch_size * batch_scale
            if len(self.buckets[bucket_id]) == target_batch_size:
                yield self.buckets[bucket_id]
                self.buckets[bucket_id] = []

        if not self.drop_last:
            for bucket_id, bucket in list(self.buckets.items()):
                if len(bucket) == 0:
                    continue
                yield bucket
                self.buckets[bucket_id] = []


class DistributedBucketBatchSampler(BatchSampler):

    def __init__(
        self,
        data_source: Dataset,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        generator: torch.Generator = None,
        weights: Optional[Sequence[float]] = None,
        batch_size_scales: Dict[int, int] = {},
    ):

        self._base_batch_sampler = BucketBatchSampler(
            data_source,
            batch_size,
            shuffle,
            drop_last,
            generator,
            weights,
            batch_size_scales,
        )
        assert drop_last, "DistributedBucketBatchSampler only supports drop_last=True"

        from extensions.xfuser.core.distributed.parallel_state import (
            get_dp_group, model_parallel_is_initialized,
        )

        if model_parallel_is_initialized():
            self.world_size = get_dp_group().world_size
            self.rank = get_dp_group().rank_in_group
        else:
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()

    def __len__(self):
        return (len(self._base_batch_sampler) + self.world_size - 1) // self.world_size

    def __iter__(self):
        batch_to_yield = []
        for idx, batch in enumerate(self._base_batch_sampler):

            if idx % self.world_size == self.rank:
                batch_to_yield = batch

            if idx % self.world_size == self.world_size - 1:
                yield batch_to_yield
