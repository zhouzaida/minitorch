from minitorch import Tensor
from .node import collect_next_edges
from .node import *


############## unary operator ##################

def neg(t: Tensor) -> Tensor:
    data = -t.data
    requires_grad = t.requires_grad
    if requires_grad:
        neg_bw = NegBackward()
        neg_bw.set_next_edges(collect_next_edges(t))
        return Tensor(data=data,
                      requires_grad=requires_grad,
                      grad_fn=neg_bw)
    else:
        return Tensor(data=data)


def sum(t: Tensor) -> Tensor:
    data = t.data.sum()
    requires_grad = t.requires_grad
    if requires_grad:
        sum_bw = SumBackward()
        sum_bw.set_next_edges(collect_next_edges(t))
        sum_bw.shape = t.shape
        return Tensor(data=data,
                      requires_grad=requires_grad,
                      grad_fn=sum_bw)
    else:
        return Tensor(data=data)

############## binary operator ##################
def add(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data + t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    if requires_grad:
        add_bw = AddBackward()
        add_bw.set_next_edges(collect_next_edges(t1, t2))
        return Tensor(data=data,
                      requires_grad=requires_grad,
                      grad_fn=add_bw)
    else:
        return Tensor(data=data)


def sub(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data - t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    if requires_grad:
        sub_bw = SubBackward()
        sub_bw.set_next_edges(collect_next_edges(t1, t2))
        if t2.requires_grad:
            sub_bw.sign = -1
        return Tensor(data=data,
                      requires_grad=requires_grad,
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
        if t2.requires_grad:
            mul_bw.t1 = Tensor(data=t1.data)
        return Tensor(data=data,
                      requires_grad=requires_grad,
                      grad_fn=mul_bw)
    else:
        return Tensor(data=data)


def matmul(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data @ t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    if requires_grad:
        matmul_bw = MatmulBackward()
        matmul_bw.set_next_edges(collect_next_edges(t1, t2))
        if t1.requires_grad:
            matmul_bw.t2 = Tensor(data=t2.data.T)
        if t2.requires_grad:
            matmul_bw.t1 = Tensor(data=t1.data.T)
        return Tensor(data=data,
                      requires_grad=requires_grad,
                      grad_fn=matmul_bw)
    else:
        return Tensor(data=data)
