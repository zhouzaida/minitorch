from abc import ABCMeta, abstractmethod
from typing import List

from minitorch import Tensor
from .edge import Edge


sequence_nr = float('INF')

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

    def __init__(self):
        global sequence_nr
        self.sequence_nr = sequence_nr
        sequence_nr -= 1

    def __call__(self, *grad_outputs):
        return self.apply(*grad_outputs)

    # def __lt__(self, other):
    #     return self.sequence_nr < other.sequence_nr

    def set_next_edges(self, next_edges: List[Edge] = None):
        self.next_edges = next_edges
    
    @abstractmethod
    def apply(self, *grad_outputs):
        """You must implement the abstract method for custome Node"""


class AccumulateGrad(Node):

    def __init__(self, leaf_tensor: Tensor):
        super().__init__()
        self.leaf_tensor = leaf_tensor

    def apply(self, *grad_outputs):
        grad, = grad_outputs
        if self.leaf_tensor.grad is None:
            self.leaf_tensor.grad = grad
        else:
            self.leaf_tensor.grad += grad
        return None


class AddBackward(Node):

    def apply(self, *grad_outputs):
        grad, = grad_outputs
        return grad, grad


class MulBackward(Node):

    def __init__(self, t1: Tensor = None, t2: Tensor = None):
        super().__init__()
        self.t1 = Tensor(t1.data)
        self.t2 = Tensor(t2.data)

    def apply(self, *grad_outputs):
        grad, = grad_outputs
        return self.t2 * grad, self.t1 * grad
