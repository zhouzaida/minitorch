from typing import Optional

from .dataset import Dataset
from .sampler import BatchSampler, Sampler, SequentialSampler


class DataLoader:

    def __init__(self, dataset: Dataset, batch_size: int = 1,
                 sampler: Optional[Sampler] = None,
                 batch_sampler: Optional[Sampler] = None,
                 drop_last: bool = False):
        if batch_sampler is not None:
            # auto_collation with custom batch_sampler
            if batch_size != 1 or sampler is not None or drop_last:
                raise ValueError("batch_sampler option is mutually exclusive "
                                 "with batch_size, shuffle, sampler, and "
                                 "drop_last")
            batch_size = None
            drop_last = False
        elif batch_size is None:
            if drop_last:
                raise ValueError("batch_size=None option disables auto-batching "
                                 "and is mutually exclusive with drop_last")
        if sampler is None:  # give default samplers
            sampler = SequentialSampler(dataset)

        if batch_size is not None and batch_sampler is None:
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = sampler
        self.batch_sampler = batch_sampler

    def __iter__(self):
        pass
