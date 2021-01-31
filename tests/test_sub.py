from unittest import TestCase

from minitorch import Tensor


class TestSub(TestCase):
    
    def test_sub(self):
        # scalar sub
        t1 = Tensor(1.0)
        t2 = Tensor(2.0)
        t3 = t1 - t2
        self.assertEqual(t3.data.tolist(), -1.0)

        t1 = Tensor(2.0, requires_grad=True)
        t2 = Tensor(3.0)
        t3 = t1 - t2
        t3.backward()
        self.assertEqual(t1.grad.data.tolist(), 1.0)

        t1 = Tensor(2.0)
        t2 = Tensor(3.0, requires_grad=True)
        t3 = t1 - t2
        t3.backward()
        self.assertEqual(t2.grad.data.tolist(), -1.0)

        t1 = Tensor(2.0, requires_grad=True)
        t2 = Tensor(3.0, requires_grad=True)
        t3 = t1 - t2
        t3.backward()
        self.assertEqual(t1.grad.data.tolist(), 1.0)
        self.assertEqual(t2.grad.data.tolist(), -1.0)

        # vector sub
        t1 = Tensor([1.0, 2.0])
        t2 = Tensor([2.0, 3.0])
        t3 = t1 - t2
        self.assertEqual(t3.data.tolist(), [-1.0, -1.0])

        t1 = Tensor([1.0, 2.0], requires_grad=True)
        t2 = Tensor([2.0, 3.0])
        t3 = t1 - t2
        t3.backward(Tensor([1.0, 1.0]))
        self.assertEqual(t1.grad.data.tolist(), [1.0, 1.0])

        t1 = Tensor([1.0, 2.0])
        t2 = Tensor([2.0, 3.0], requires_grad=True)
        t3 = t1 - t2
        t3.backward(Tensor([1.0, 1.0]))
        self.assertEqual(t2.grad.data.tolist(), [-1.0, -1.0])

        t1 = Tensor([1.0, 2.0], requires_grad=True)
        t2 = Tensor([2.0, 3.0], requires_grad=True)
        t3 = t1 - t2
        t3.backward(Tensor([1.0, 1.0]))
        self.assertEqual(t1.grad.data.tolist(), [1.0, 1.0])
        self.assertEqual(t2.grad.data.tolist(), [-1.0, -1.0])
