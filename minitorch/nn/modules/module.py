from abc import ABCMeta, abstractmethod
from collections import OrderedDict

from typing import Iterator, Union

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
        return self.forward(*inputs)

    @abstractmethod
    def forward(self, *inputs):
        """subclass must implement the method."""

    def __getattr__(self, name: str) -> Union[Tensor, 'Module']:
        _parameters = self.__dict__['_parameters']
        if name in _parameters:
            return _parameters[name]
        _modules = self.__dict__['_modules']
        if name in _modules:
            return _modules[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __setattr__(self, name: str, value: Union[Tensor, 'Module']) -> None:
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        else:
            object.__setattr__(self, name, value)

    def parameters(self):
        ...

    def named_modules(self, prefix: str = '') -> Iterator['Module']:
        yield prefix, self
        for name, module in self._modules.items():
            submodule_prefix = prefix + ('.' if prefix else '') + name
            for m in module.named_modules(submodule_prefix):
                yield m

    def zero_grad(self) -> None:
        for value in self._modules.values():
            value.zero_grad()
        for value in self._parameters.values():
            value.zero_grad()
