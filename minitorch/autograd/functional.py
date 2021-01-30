from minitorch import Tensor
from .node import collect_next_edges, AddBackward, MulBackward


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


def mul(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data * t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    if requires_grad:
        mul_bw = MulBackward(t1, t2)
        mul_bw.set_next_edges(collect_next_edges(t1, t2))
        return Tensor(data=data,
                      requires_grad=requires_grad,
                      grad_fn=mul_bw)
    else:
        return Tensor(data=data)
