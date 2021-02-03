from typing import Union, Tuple

from minitorch import Tensor
from .node import collect_next_edges
from .node import *


############## reduce operator ##################

def sum(t: Tensor, axis: Union[int, Tuple[int]] = None) -> Tensor:
    data = t.data.sum(axis=axis)
    requires_grad = t.requires_grad
    if requires_grad:
        sum_bw = SumBackward()
        sum_bw.set_next_edges(collect_next_edges(t))
        sum_bw.axis = axis
        sum_bw.shape = t.shape
        return Tensor(data=data,
                      requires_grad=True,
                      grad_fn=sum_bw)
    else:
        return Tensor(data=data)

############## unary operator ##################

def neg(t: Tensor) -> Tensor:
    data = -t.data
    requires_grad = t.requires_grad
    if requires_grad:
        neg_bw = NegBackward()
        neg_bw.set_next_edges(collect_next_edges(t))
        return Tensor(data=data,
                      requires_grad=True,
                      grad_fn=neg_bw)
    else:
        return Tensor(data=data)


def t(t: Tensor) -> Tensor:
    # transpose
    data = t.data.T
    requires_grad = t.requires_grad
    if requires_grad:
        t_bw = TBackward()
        t_bw.set_next_edges(collect_next_edges(t))
        return Tensor(data=data,
                      requires_grad=True,
                      grad_fn=t_bw)
    else:
        return Tensor(data=data)

############## binary operator ##################
def add(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data + t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    if requires_grad:
        add_bw = AddBackward()
        add_bw.set_next_edges(collect_next_edges(t1, t2))
        if t1.requires_grad:
            add_bw.t1_shape = t1.shape
        if t2.requires_grad:
            add_bw.t2_shape = t2.shape
        return Tensor(data=data,
                      requires_grad=True,
                      grad_fn=add_bw)
    else:
        return Tensor(data=data)


def sub(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data - t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    if requires_grad:
        sub_bw = SubBackward()
        sub_bw.set_next_edges(collect_next_edges(t1, t2))
        if t1.requires_grad:
            sub_bw.t1_shape = t1.shape
        if t2.requires_grad:
            sub_bw.t2_shape = t2.shape
        return Tensor(data=data,
                      requires_grad=True,
                      grad_fn=sub_bw)
    else:
        return Tensor(data=data)


def mul(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data * t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    if requires_grad:
        mul_bw = MulBackward()
        mul_bw.set_next_edges(collect_next_edges(t1, t2))
        if t1.requires_grad:
            mul_bw.t2 = Tensor(data=t2.data)
            mul_bw.t1_shape = t1.shape
        if t2.requires_grad:
            mul_bw.t1 = Tensor(data=t1.data)
            mul_bw.t2_shape = t2.shape
        return Tensor(data=data,
                      requires_grad=True,
                      grad_fn=mul_bw)
    else:
        return Tensor(data=data)


def matmul(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data @ t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    if requires_grad:
        matmul_bw = MatMulBackward()
        matmul_bw.set_next_edges(collect_next_edges(t1, t2))
        if t1.requires_grad:
            matmul_bw.t2 = t2
        if t2.requires_grad:
            matmul_bw.t1 = t1
        return Tensor(data=data,
                      requires_grad=True,
                      grad_fn=matmul_bw)
    else:
        return Tensor(data=data)
