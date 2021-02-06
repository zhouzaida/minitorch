from unittest import TestCase

from minitorch import Tensor


class TestMean(TestCase):

    def test_mean(self):
        t1 = Tensor([1., 2., 3.])
        t2 = t1.mean()
        self.assertEqual(t2.data.tolist(), 2.)

        # (3,) -> ()
        t1 = Tensor([1., 2., 3., 4.], requires_grad=True)
        t2 = t1.mean()
        t2.backward()
        self.assertEqual(t1.grad.data.tolist(), [1/4., 1/4, 1/4, 1/4])

        # (2, 3) -> (3, )
        t1 = Tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
        t2 = t1.mean(axis=0)
        t2.backward(Tensor([1., 1., 1.]))
        self.assertEqual(t1.grad.data.tolist(), [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])

        # (2, 3) -> (2, )
        t1 = Tensor([[1., 2., 3., 4.], [4., 5., 6.,7.]], requires_grad=True)
        t2 = t1.mean(axis=1)
        t2.backward(Tensor([1., 1.]))
        self.assertEqual(t1.grad.data.tolist(), [[1/4., 1/4, 1/4, 1/4], [1/4., 1/4, 1/4, 1/4]])

        # (2, 3) -> (,)
        t1 = Tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
        t2 = t1.mean()
        t2.backward(Tensor(1.0))
        self.assertEqual(t1.grad.data.tolist(), [[1/6, 1/6, 1/6], [1/6, 1/6, 1/6]])
