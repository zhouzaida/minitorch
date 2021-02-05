from minitorch import Tensor
from .module import Module


class MSELoss(Module):
    """Creates a criterion that measures the mean squared error (squared L2 norm) between
    each element in the input x and target y.
    """

    def __init__(self, reduction: str = 'mean'):
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        result = (input - target) ** 2
        if self.reduction is None:
            return result
        elif self.reduction == 'mean':
            return result.mean()
        elif self.reduction == 'sum':
            return result.sum()
        else:
            raise ValueError("reduction should be one of the 'none,mean,sum', "
                             f"rather than {self.reduction}")
