from unittest import TestCase

from minitorch import Tensor
import minitorch.nn as nn


class TestActivation(TestCase):

    def test_sigmoid(self):
        input = Tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        target = Tensor([[6, 7, 8, 9, 10], [1, 2, 3, 4, 5]])
        loss = nn.MSELoss()
        output = loss(input, target)
        self.assertEqual(output.data.tolist(), 25.)

    def test_relu(self):
        ...
