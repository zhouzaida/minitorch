from abc import ABCMeta, abstractmethod

import numpy as np
from typing import List

from minitorch import Tensor
from .edge import Edge


def collect_next_edges(*tensors) -> List[Edge]:
    next_edges = []
    for t in tensors:
        if not t.requires_grad:
            continue
        if t.grad_fn is None:
            t.grad_fn = AccumulateGrad(t)
            next_edges.append(Edge(t.grad_fn))
        else:
            next_edges.append(Edge(t.grad_fn))
    return next_edges


def unbroadcast(grad_input: Tensor, input_shape: tuple) -> Tensor:
    """When broadcast is applied to an operation, unbroadcast should also
       be executed when backpropagating.

    References:
        1. https://numpy.org/doc/stable/user/basics.broadcasting.html
        2. http://coldattic.info/post/116/
        3. https://github.com/joelgrus/autograd/blob/part06/autograd/tensor.py#L150
    """
    if grad_input.shape == input_shape:
        return grad_input
    data = grad_input.data
    ndims_added = len(grad_input.shape) - len(input_shape)
    for _ in range(ndims_added):
        data = data.sum(axis=0)
    for i, dim in enumerate(input_shape):
        if dim == 1:
            data = data.sum(axis=i, keepdims=True)

    return Tensor(data=data)


class Node(metaclass=ABCMeta):

    def __call__(self, *grad_outputs):
        return self.apply(*grad_outputs)

    def set_next_edges(self, next_edges: List[Edge] = None):
        self.next_edges = next_edges

    @abstractmethod
    def apply(self, *grad_outputs):
        """You must implement the abstract method for custome Node"""


class AccumulateGrad(Node):

    def __init__(self, leaf_tensor: Tensor):
        self.leaf_tensor = leaf_tensor

    def apply(self, grad_output: Tensor):
        if self.leaf_tensor.grad is None:
            self.leaf_tensor.grad = grad_output
        else:
            self.leaf_tensor.grad += grad_output
        return None


############## reduce operator ##################

class SumBackward(Node):

    def __init__(self):
        self.axis = None
        self.shape: tuple = None

    def apply(self, grad_output: Tensor) -> tuple:
        if isinstance(self.axis, int):
            self.axis = [self.axis]
        if self.axis is None:
            shape = [1] * len(self.shape)
        else:
            shape = [1 if i in self.axis else self.shape[i] for i in range(len(self.shape))]
        data = grad_output.data.reshape(shape) + np.zeros(self.shape)
        return Tensor(data=data),


class MeanBackward(Node):

    def __init__(self):
        self.axis = None
        self.shape: tuple = None

    def apply(self, grad_output: Tensor) -> tuple:
        if isinstance(self.axis, int):
            self.axis = [self.axis]
        if self.axis is None:
            shape = [1] * len(self.shape)
        else:
            shape = [1 if i in self.axis else self.shape[i] for i in range(len(self.shape))]
        scale = np.prod(grad_output.shape) / np.prod(self.shape)
        data = scale * grad_output.data.reshape(shape) + np.zeros(self.shape)
        return Tensor(data=data),


############## unary operator ##################

class NegBackward(Node):

    def apply(self, grad_output: Tensor) -> list:
        return -grad_output,


class TBackward(Node):
    """Transpose"""

    def apply(self, grad_output: Tensor) -> list:
        return Tensor(data=grad_output.data.T)

############## binary operator ##################

class AddBackward(Node):

    def __init__(self):
        self.t1_shape: tuple = None
        self.t2_shape: tuple = None

    def apply(self, grad_output: Tensor) -> list:
        grad_input = []
        if self.t1_shape is not None:
            grad_input.append(unbroadcast(grad_output, self.t1_shape))
        if self.t2_shape is not None:
            grad_input.append(unbroadcast(grad_output, self.t2_shape))
        return grad_input


class SubBackward(Node):

    def __init__(self):
        self.t1_shape: tuple = None
        self.t2_shape: tuple = None

    def apply(self, grad_output: Tensor) -> list:
        grad_input = []
        if self.t1_shape is not None:
            grad_input.append(unbroadcast(grad_output, self.t1_shape))
        if self.t2_shape is not None:
            grad_input.append(unbroadcast(-grad_output, self.t2_shape))
        return grad_input


class MulBackward(Node):

    def __init__(self):
        self.t1: Tensor = None
        self.t1_shape: tuple = None
        self.t2: Tensor = None
        self.t2_shape: tuple = None

    def apply(self, grad_output: Tensor) -> list:
        grad_input = []
        if self.t2 is not None:
            grad_input.append(unbroadcast(self.t2 * grad_output, self.t1_shape))
        if self.t1 is not None:
            grad_input.append(unbroadcast(self.t1 * grad_output, self.t2_shape))
        return grad_input


class DivBackward(Node):

    def __init__(self):
        self.t1: Tensor = None
        self.t1_shape: tuple = None
        self.t2: Tensor = None
        self.t2_shape: tuple = None

    def apply(self, grad_output: Tensor) -> list:
        grad_input = []
        if self.t2 is not None and self.t1_shape is not None:
            grad_input.append(unbroadcast(1 / self.t2 * grad_output, self.t1_shape))
        if self.t1 is not None:
            grad_input.append(unbroadcast(-self.t1 / (self.t2 ** 2) * grad_output, self.t2_shape))
        return grad_input


class MatMulBackward(Node):

    def __init__(self):
        self.t1: Tensor = None
        self.t2: Tensor = None

    def apply(self, grad_output: Tensor) -> list:
        grad_input = []
        if self.t2 is not None:
            grad_input.append(grad_output @ Tensor(self.t2.data.T))
        if self.t1 is not None:
            grad_input.append(Tensor(self.t1.data.T) @ grad_output)
        return grad_input


class PowBackward(Node):

    def __init__(self):
        self.t1: Tensor = None
        self.t2: float = None

    def apply(self, grad_output: Tensor) -> tuple:
        return grad_output * self.t2 * self.t1 ** (self.t2-1),
