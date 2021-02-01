from abc import ABCMeta, abstractmethod
from collections import OrderedDict

from typing import Union

from ..parameter import Parameter
from minitorch import Tensor


class Module(metaclass=ABCMeta):
    r"""Base class for all neural network modules.

    Your models should also subclass this class.

    Modules can also contain other Modules, allowing to nest them in
    a tree structure. 
    """

    def __init__(self):
        self._parameters = OrderedDict()
        self._modules = OrderedDict()

    def __call__(self, *inputs):
        self.forward(*inputs)

    @abstractmethod
    def forward(self, *inputs):

    def __getattr__(self, name: str) -> Union[Tensor, 'Module']:
        if name in self._parameters:
            return self._parameters[name]
        elif name in self._modules:
            return self._modules[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __setattr__(self, name: str, value: Union[Tensor, 'Module']) -> None:
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        else:
            raise TypeError(f"value should be type of Tensor or Module, but get {type(value).__name__}.")

    def zero_grad(self):
        for value in self._modules.values():
            value.zero_grad()
        for value in self._parameters.values():
            value.zero_grad()
