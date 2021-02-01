from minitorch import Tensor
from .module import Module
from ..parameter import Parameter


class Linear(Module):
    """Applies a linear transformation to the incoming data: y = xA^T + b"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter()

    def forward(self, input: Tensor) -> Tensor:
        y = input @ self.weight
        