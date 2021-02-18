from abc import ABCMeta, abstractmethod
from collections import OrderedDict

from typing import Iterator, Union, Tuple

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
        pass

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

    def _named_members(self, get_members_fn, prefix='', recurse=True):
        memo = set()
        modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                yield name, v

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def modules(self) -> Iterator['Module']:
        for name, module in self.named_modules():
            yield module

    def named_modules(self, prefix: str = '') -> Iterator[Tuple[str, 'Module']]:
        yield prefix, self
        for name, module in self._modules.items():
            submodule_prefix = prefix + ('.' if prefix else '') + name
            yield from module.named_modules(submodule_prefix)

    def zero_grad(self) -> None:
        for p in self.parameters():
            if p.grad is not None:
                p.grad = None
