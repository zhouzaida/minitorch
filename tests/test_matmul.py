from unittest import TestCase

from minitorch import Tensor


class TestMatmul(TestCase):

    def test_matmul(self):
        t1 = Tensor([1.0, 2.0])
        t2 = Tensor([2.0, 3.0])
        t3 = t1 @ t2
        self.assertEqual(t3.data.tolist(), 8.0)

        t1 = Tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])  # 3 * 2
        t2 = Tensor([[2.0, 3.0], [3.0, 4.0]])  # 2 * 2
        t3 = t1 @ t2  # 3 * 2
        self.assertEqual(t3.data.tolist(), [[8.0, 11.], [13., 18.0], [18.0, 25.0]])

        t1 = Tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], requires_grad=True)  # 3 * 2
        t2 = Tensor([[2.0, 3.0], [3.0, 4.0]])  # 2 * 2
        t3 = t1 @ t2
        t3.backward(Tensor([[1.0, 2.0], [3.0, 4.0], [3.0, 5.0]]))
        self.assertEqual(t1.grad.data.tolist(), [[8.0, 11.0], [18.0, 25.0], [21.0, 29.0]])

        t1 = Tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])  # 3 * 2
        t2 = Tensor([[2.0, 3.0], [3.0, 4.0]], requires_grad=True)  # 2 * 2
        t3 = t1 @ t2
        t3.backward(Tensor([[1.0, 2.0], [3.0, 4.0], [3.0, 5.0]]))
        self.assertEqual(t2.grad.data.tolist(), [[16.0, 25.0], [23.0, 36.0]])

        t1 = Tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], requires_grad=True)  # 3 * 2
        t2 = Tensor([[2.0, 3.0], [3.0, 4.0]], requires_grad=True)  # 2 * 2
        t3 = t1 @ t2
        t3.backward(Tensor([[1.0, 2.0], [3.0, 4.0], [3.0, 5.0]]))
        self.assertEqual(t1.grad.data.tolist(), [[8.0, 11.0], [18.0, 25.0], [21.0, 29.0]])
        self.assertEqual(t2.grad.data.tolist(), [[16.0, 25.0], [23.0, 36.0]])
