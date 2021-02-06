from abc import ABCMeta, abstractmethod


class Optimizer(metaclass=ABCMeta):
    """Base class for all optimizers.

    Args:
        params (iterable): an iterable of Tensor.
            Specifies what Tensors should be optimized.
    """

    def __init__(self, params):
        ...

    @abstractmethod
    def step(self):
        pass
