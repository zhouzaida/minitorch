from unittest import TestCase

from minitorch import Tensor


class TestSum(TestCase):
    
    def test_sum(self):
        t1 = Tensor([1., 2., 3.])
        t2 = t1.sum()
        self.assertEqual(t2.data.tolist(), 6.)

        t1 = Tensor([1., 2., 3.], requires_grad=True)
        t2 = t1.sum()
        t2.backward()
        self.assertEqual(t1.grad.data.tolist(), [1., 1., 1.])

        t1 = Tensor([1., 2., 3.], requires_grad=True)
        t2 = t1.sum()
        t2.backward(Tensor(2.0))
        self.assertEqual(t1.grad.data.tolist(), [2., 2., 2.])
