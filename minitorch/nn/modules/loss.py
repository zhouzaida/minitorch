from minitorch import Tensor
from .module import Module


class MSELoss(Module):
    """Creates a criterion that measures the mean squared error (squared L2 norm) between
    each element in the input x and target y.
    """

    def __init__(self, reduction: str = 'mean'):
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        ...
