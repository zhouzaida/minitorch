from unittest import TestCase

from minitorch import Tensor


class TestSum(TestCase):

    def test_sum(self):
        t1 = Tensor([1., 2., 3.])
        t2 = t1.sum()
        self.assertEqual(t2.data.tolist(), 6.)

        # (3,) -> ()
        t1 = Tensor([1., 2., 3.], requires_grad=True)
        t2 = t1.sum()
        t2.backward()
        self.assertEqual(t1.grad.data.tolist(), [1., 1., 1.])

        # (2, 3) -> (3, )
        t1 = Tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
        t2 = t1.sum(axis=0)
        t2.backward(Tensor([1., 1., 1.]))
        self.assertEqual(t1.grad.data.tolist(), [[1., 1., 1.], [1., 1., 1.]])

        # (2, 3) -> (2, )
        t1 = Tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
        t2 = t1.sum(axis=1)
        t2.backward(Tensor([1., 1.]))
        self.assertEqual(t1.grad.data.tolist(), [[1., 1., 1.], [1., 1., 1.]])

        # (2, 3) -> (,)
        t1 = Tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
        t2 = t1.sum()
        t2.backward(Tensor(1.0))
        self.assertEqual(t1.grad.data.tolist(), [[1., 1., 1.], [1., 1., 1.]])
