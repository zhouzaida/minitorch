from typing import Union, List
import numpy as np


def ensure_ndarray(data):
    if isinstance(data, np.ndarray):
        return data
    else:
        return np.array(data)


def ensure_tensor(data):
    if isinstance(data, Tensor):
        return data
    else:
        return Tensor(data)


class Tensor:
    def __init__(self,
                 data: Union[np.ndarray, list, float],
                 requires_grad: bool = False,
                 grad_fn=None):
        self.data = ensure_ndarray(data)
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = grad_fn

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __add__(self, other) -> 'Tensor':
        from minitorch.autograd.functional import add
        return add(self, ensure_tensor(other))

    def __radd__(self, other) -> 'Tensor':
        from minitorch.autograd.functional import add
        return add(ensure_tensor(other), self)

    def __iadd__(self, other) -> 'Tensor':
        from minitorch.autograd.functional import add
        self.data += ensure_tensor(other).data
        return self

    def __neg__(self) -> 'Tensor':
        from minitorch.autograd.functional import neg
        return neg(self)

    def __sub__(self, other) -> 'Tensor':
        from minitorch.autograd.functional import sub
        return sub(self, ensure_tensor(other))

    def __rsub__(self, other) -> 'Tensor':
        from minitorch.autograd.functional import sub
        return sub(ensure_tensor(other), self)

    def __isub__(self, other) -> 'Tensor':
        from minitorch.autograd.functional import sub
        self.data -= ensure_tensor(other).data
        return self

    def __mul__(self, other) -> 'Tensor':
        from minitorch.autograd.functional import mul
        return mul(self, ensure_tensor(other))

    def __rmul__(self, other) -> 'Tensor':
        from minitorch.autograd.functional import mul
        return mul(ensure_tensor(other), self)

    def __matmul__(self, other) -> 'Tensor':
        from minitorch.autograd.functional import matmul
        return matmul(self, other)

    def sum(self) -> 'Tensor':
        from minitorch.autograd.functional import sum
        return sum(self)

    def backward(self, grad: 'Tensor' = None):
        assert self.requires_grad
        if grad is None and self.shape != ():
            raise RuntimeError("grad can be implicitly created only for scalar outputs")
        grad = grad if grad else Tensor(1.0)
        from minitorch.autograd.engine import Engine
        engine = Engine()
        engine.execute(self, grad)
