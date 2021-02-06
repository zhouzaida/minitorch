from unittest import TestCase

import numpy as np

from minitorch import Tensor


class TestExp(TestCase):

    def test_exp(self):
        # scalar exp
        t1 = Tensor(2.0)
        t2 = t1.exp()
        np.testing.assert_allclose(t2.data, np.exp(2))

        t1 = Tensor(2.0, requires_grad=True)
        t2 = t1.exp()
        t2.backward()
        np.testing.assert_allclose(t1.grad.data, np.exp(2))

        # vector exp
        t1 = Tensor([1.0, 2.0])
        t2 = t1.exp()
        np.testing.assert_allclose(t2.data, np.exp([1, 2]))

        t1 = Tensor([1.0, 2.0], requires_grad=True)
        t2 = t1.exp()
        t2.backward(Tensor([1.0, 1.0]))
        np.testing.assert_allclose(t1.grad.data, np.exp([1, 2]))
