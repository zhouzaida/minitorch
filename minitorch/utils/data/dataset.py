from abc import ABCMeta, abstractmethod


class Dataset(metaclass=ABCMeta):
    """An abstract class representing a Dataset.

    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overwrite __getitem__, supporting fetching a
    data sample for a given key. Subclasses could also optionally overwrite
    __len__.
    """

    @abstractmethod
    def __getitem__(self, index):
        pass
