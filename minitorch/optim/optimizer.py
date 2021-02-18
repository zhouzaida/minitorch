from abc import ABCMeta, abstractmethod


class Optimizer(metaclass=ABCMeta):
    """Base class for all optimizers."""

    @abstractmethod
    def step(self):
        pass
