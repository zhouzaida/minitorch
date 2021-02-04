from unittest import TestCase

from minitorch import Tensor


class TestPow(TestCase):

    def test_pow(self):
        # scalar pow
        t1 = Tensor(2.0)
        t2 = t1 ** 3
        self.assertEqual(t2.data.tolist(), 8.0)

        t1 = Tensor(2.0, requires_grad=True)
        t2 = t1 ** 3
        t2.backward()
        self.assertEqual(t1.grad.data.tolist(), 12.0)

        # vector pow
        t1 = Tensor([1.0, 2.0])
        t2 = t1 ** 3
        self.assertEqual(t2.data.tolist(), [1.0, 8.0])

        t1 = Tensor([1.0, 2.0], requires_grad=True)
        t2 = t1 ** 3
        t2.backward(Tensor([1.0, 1.0]))
        self.assertEqual(t1.grad.data.tolist(), [3.0, 12.0])
