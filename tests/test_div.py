from unittest import TestCase

from minitorch import Tensor


class TestDiv(TestCase):

    def test_simple_div(self):
        # scalar div
        t1 = Tensor(1.0)
        t2 = Tensor(2.0)
        t3 = t1 / t2
        self.assertEqual(t3.data.tolist(), 0.5)

        t1 = Tensor(1.0, requires_grad=True)
        t2 = Tensor(2.0)
        t3 = t1 / t2
        t3.backward()
        self.assertEqual(t1.grad.data.tolist(), 0.5)

        t1 = Tensor(1.0)
        t2 = Tensor(2.0, requires_grad=True)
        t3 = t1 / t2
        t3.backward()
        self.assertEqual(t2.grad.data.tolist(), -0.25)

        t1 = Tensor(1.0, requires_grad=True)
        t2 = Tensor(2.0, requires_grad=True)
        t3 = t1 / t2
        t3.backward()
        self.assertEqual(t1.grad.data.tolist(), 0.5)
        self.assertEqual(t2.grad.data.tolist(), -0.25)

        # vector div
        t1 = Tensor([1.0, 2.0])
        t2 = Tensor([2.0, 4.0])
        t3 = t1 / t2
        self.assertEqual(t3.data.tolist(), [0.5, 0.5])

        t1 = Tensor([1.0, 2.0], requires_grad=True)
        t2 = Tensor([2.0, 4.0])
        t3 = t1 / t2
        t3.backward(Tensor([1.0, 1.0]))
        self.assertEqual(t1.grad.data.tolist(), [0.5, 0.25])

        t1 = Tensor([1.0, 2.0])
        t2 = Tensor([2.0, 4.0], requires_grad=True)
        t3 = t1 / t2
        t3.backward(Tensor([1.0, 1.0]))
        self.assertEqual(t2.grad.data.tolist(), [-0.25, -1/8])

        t1 = Tensor([1.0, 2.0], requires_grad=True)
        t2 = Tensor([2.0, 4.0], requires_grad=True)
        t3 = t1 / t2
        t3.backward(Tensor([1.0, 1.0]))
        self.assertEqual(t1.grad.data.tolist(), [0.5, 0.25])
        self.assertEqual(t2.grad.data.tolist(), [-0.25, -1/8])

    def test_broadcast_div(self):
        # (2,) / ()
        t1 = Tensor([1.0, 2.0], requires_grad=True)
        t2 = Tensor(2.0, requires_grad=True)
        t3 = t1 / t2
        t3.backward(Tensor([1.0, 1.0]))
        self.assertEqual(t1.grad.data.tolist(), [0.5, 0.5])
        self.assertEqual(t2.grad.data.tolist(), -0.75)

        # (2,) / (1,)
        t1 = Tensor([1.0, 2.0], requires_grad=True)
        t2 = Tensor([2.0], requires_grad=True)
        t3 = t1 / t2
        t3.backward(Tensor([1.0, 1.0]))
        self.assertEqual(t1.grad.data.tolist(), [0.5, 0.5])
        self.assertEqual(t2.grad.data.tolist(), [-0.75])

        # (2, 2) / ()
        t1 = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        t2 = Tensor(2.0, requires_grad=True)
        t3 = t1 / t2
        t3.backward(Tensor([[1.0, 1.0], [1.0, 1.0]]))
        self.assertEqual(t1.grad.data.tolist(), [[0.5, 0.5], [0.5, 0.5]])
        self.assertEqual(t2.grad.data.tolist(), -2.5)

        # (2, 2) / (1,)
        t1 = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        t2 = Tensor([2.0], requires_grad=True)
        t3 = t1 / t2
        t3.backward(Tensor([[1.0, 1.0], [1.0, 1.0]]))
        self.assertEqual(t1.grad.data.tolist(), [[0.5, 0.5], [0.5, 0.5]])
        self.assertEqual(t2.grad.data.tolist(), [-2.5])

        # (2, 2) / (2, )
        t1 = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        t2 = Tensor([2.0, 4.0], requires_grad=True)
        t3 = t1 / t2
        t3.backward(Tensor([[1.0, 1.0], [1.0, 1.0]]))
        self.assertEqual(t1.grad.data.tolist(), [[0.5, 0.25], [0.5, 0.25]])
        self.assertEqual(t2.grad.data.tolist(), [-1.0, -0.375])
