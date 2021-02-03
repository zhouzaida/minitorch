from minitorch import Tensor
from .module import Module
from ..parameter import Parameter


class Linear(Module):
    """Applies a linear transformation to the incoming data: y = xA + b"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(out_features, in_features)
        if bias:
            self.bias = Parameter(out_features)
        else:
            self.bias = None

    def forward(self, input: Tensor) -> Tensor:
        output = input @ self.weight.t()
        if self.bias is not None:
            output = output + self.bias
        return output
