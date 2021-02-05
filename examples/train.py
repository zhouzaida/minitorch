import numpy as np

from minitorch import Tensor
import minitorch.nn as nn


class Model(nn.Module):

    def __init__(self, in_features=3):
        super().__init__()
        self.linear1 = nn.Linear(in_features, 5, bias=True)
        self.linear2 = nn.Linear(5, 1, bias=True)

    def forward(self, input):
        output = self.linear1(input)
        output = self.linear2(output)
        return output


def train(model, x, y, epoch=10):  # TODO
    mse_loss = nn.MSELoss()
    for i in range(1, epoch + 1):
        output = model(input)
        loss = mse_loss(output, y)
        loss.backward()


def test(model, x, y):
    ...


def main():
    coef = Tensor(np.array([1, 3, 2]))
    x_train = Tensor(np.randmo.rand(100, 3))
    y_train = x_train @ coef + 5
    x_test = Tensor(np.randmo.rand(20, 3))
    y_test = x_test @ coef + 5
    model = Model()
    train(model, x_train, y_train)
    test(x_test, y_test)


if __name__ == '__main__':
    main()
