from unittest import TestCase

from minitorch import Tensor


class TestMul(TestCase):

    def test_simple_mul(self):
        # scalar mul
        t1 = Tensor(1.0)
        t2 = Tensor(2.0)
        t3 = t1 * t2
        self.assertEqual(t3.data.tolist(), 2.0)

        t1 = Tensor(1.0, requires_grad=True)
        t2 = Tensor(2.0)
        t3 = t1 * t2
        t3.backward()
        self.assertEqual(t1.grad.data.tolist(), 2.0)

        t1 = Tensor(1.0)
        t2 = Tensor(2.0, requires_grad=True)
        t3 = t1 * t2
        t3.backward()
        self.assertEqual(t2.grad.data.tolist(), 1.0)

        t1 = Tensor(1.0, requires_grad=True)
        t2 = Tensor(2.0, requires_grad=True)
        t3 = t1 * t2
        t3.backward()
        self.assertEqual(t1.grad.data.tolist(), 2.0)
        self.assertEqual(t2.grad.data.tolist(), 1.0)

        # vector mul
        t1 = Tensor([1.0, 2.0])
        t2 = Tensor([2.0, 3.0])
        t3 = t1 * t2
        self.assertEqual(t3.data.tolist(), [2.0, 6.0])

        t1 = Tensor([1.0, 2.0], requires_grad=True)
        t2 = Tensor([2.0, 3.0])
        t3 = t1 * t2
        t3.backward(Tensor([1.0, 1.0]))
        self.assertEqual(t1.grad.data.tolist(), [2.0, 3.0])

        t1 = Tensor([1.0, 2.0])
        t2 = Tensor([2.0, 3.0], requires_grad=True)
        t3 = t1 * t2
        t3.backward(Tensor([1.0, 1.0]))
        self.assertEqual(t2.grad.data.tolist(), [1.0, 2.0])

        t1 = Tensor([1.0, 2.0], requires_grad=True)
        t2 = Tensor([2.0, 3.0], requires_grad=True)
        t3 = t1 * t2
        t3.backward(Tensor([1.0, 1.0]))
        self.assertEqual(t1.grad.data.tolist(), [2.0, 3.0])
        self.assertEqual(t2.grad.data.tolist(), [1.0, 2.0])

    def test_broadcast_mul(self):
        # (2,) * ()
        t1 = Tensor([1.0, 2.0], requires_grad=True)
        t2 = Tensor(2.0, requires_grad=True)
        t3 = t1 * t2
        t3.backward(Tensor([1.0, 1.0]))
        self.assertEqual(t1.grad.data.tolist(), [2.0, 2.0])
        self.assertEqual(t2.grad.data.tolist(), 3.0)

        # (2,) * (1,)
        t1 = Tensor([1.0, 2.0], requires_grad=True)
        t2 = Tensor([2.0], requires_grad=True)
        t3 = t1 * t2
        t3.backward(Tensor([1.0, 1.0]))
        self.assertEqual(t1.grad.data.tolist(), [2.0, 2.0])
        self.assertEqual(t2.grad.data.tolist(), [3.0])

        # (2, 2) * ()
        t1 = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        t2 = Tensor(2.0, requires_grad=True)
        t3 = t1 * t2
        t3.backward(Tensor([[1.0, 1.0], [1.0, 1.0]]))
        self.assertEqual(t1.grad.data.tolist(), [[2.0, 2.0], [2.0, 2.0]])
        self.assertEqual(t2.grad.data.tolist(), 10.0)

        # (2, 2) * (1,)
        t1 = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        t2 = Tensor([2.0], requires_grad=True)
        t3 = t1 * t2
        t3.backward(Tensor([[1.0, 1.0], [1.0, 1.0]]))
        self.assertEqual(t1.grad.data.tolist(), [[2.0, 2.0], [2.0, 2.0]])
        self.assertEqual(t2.grad.data.tolist(), [10.0])

        # (2, 2) * (2, )
        t1 = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        t2 = Tensor([2.0, 3.0], requires_grad=True)
        t3 = t1 * t2
        t3.backward(Tensor([[1.0, 1.0], [1.0, 1.0]]))
        self.assertEqual(t1.grad.data.tolist(), [[2.0, 3.0], [2.0, 3.0]])
        self.assertEqual(t2.grad.data.tolist(), [4.0, 6.0])
