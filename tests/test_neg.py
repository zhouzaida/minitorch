from unittest import TestCase

from minitorch import Tensor


class TestNeg(TestCase):

    def test_neg(self):
        # scalar neg
        t1 = Tensor(1.0)
        t2 = -t1
        self.assertEqual(t2.data.tolist(), -1.0)

        t1 = Tensor(2.0, requires_grad=True)
        t2 = -t1
        t2.backward()
        self.assertEqual(t1.grad.data.tolist(), -1.0)

        # vector neg
        t1 = Tensor([1.0, 2.0])
        t2 = -t1
        self.assertEqual(t2.data.tolist(), [-1.0, -2.0])

        t1 = Tensor([1.0, 2.0], requires_grad=True)
        t2 = -t1
        t2.backward(Tensor([1.0, 1.0]))
        self.assertEqual(t1.grad.data.tolist(), [-1.0, -1.0])
