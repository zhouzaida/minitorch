from .optimizer import Optimizer


class SGD(Optimizer):
    """Implements stochastic gradient descent"""

    def __init__(self, params, lr):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        self.params = list(params)
        self.lr = lr

    def step(self):
        for param in self.params:
            param.data = param.data - self.lr * param.grad.data
