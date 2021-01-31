from abc import ABCMeta, abstractmethod
from typing import List

import numpy as np

from minitorch import Tensor
from .edge import Edge


def collect_next_edges(*tensors):
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

    def apply(self, *grad_outputs):
        grad, = grad_outputs
        if self.leaf_tensor.grad is None:
            self.leaf_tensor.grad = grad
        else:
            self.leaf_tensor.grad += grad
        return None


############## unary operator ##################

class NegBackward(Node):

    def apply(self, *grad_outputs):
        grad, = grad_outputs
        return -grad,


class SumBackward(Node):

    def __init__(self):
        self.shape = None
    
    def apply(self, *grad_outputs):
        grad, = grad_outputs
        return grad * Tensor(np.ones(self.shape)),


############## binary operator ##################

class AddBackward(Node):

    def apply(self, *grad_outputs):
        grad, = grad_outputs
        if len(self.next_edges) == 1:
            return grad,
        else:
            return grad, grad


class SubBackward(Node):

    def __init__(self):
        self.sign = 1

    def apply(self, *grad_outputs):
        grad, = grad_outputs
        if len(self.next_edges) == 1:
            return self.sign * grad,
        else:
            return grad, -grad


class MulBackward(Node):

    def __init__(self):
        self.t1 = None
        self.t2 = None

    def apply(self, *grad_outputs):
        grad, = grad_outputs
        output = []
        if self.t2 is not None:
            output.append(self.t2 * grad)
        if self.t1 is not None:
            output.append(self.t1 * grad)
        return output


class MatmulBackward(Node):

    def __init__(self):
        self.t1 = None
        self.t2 = None

    def apply(self, *grad_outputs):
        grad, = grad_outputs
        output = []
        if self.t2 is not None:
            output.append(grad @ self.t2)
        if self.t1 is not None:
            output.append(self.t1 @ grad)
        return output
