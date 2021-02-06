from unittest import TestCase

from minitorch import Tensor


class TestReLU(TestCase):

    def test_relu(self):
        # scalar relu
        t1 = Tensor(2.0)
        t2 = t1.relu()
        self.assertEqual(t2.data.tolist(), 2.0)

        t1 = Tensor(2.0, requires_grad=True)
        t2 = t1.relu()
        t2.backward()
        self.assertEqual(t1.grad.data.tolist(), 1.0)

        # vector relu
        t1 = Tensor([-1.0, 2.0])
        t2 = t1.relu()
        self.assertEqual(t2.data.tolist(), [0, 2.0])

        t1 = Tensor([-1.0, 2.0], requires_grad=True)
        t2 = t1.relu()
        t2.backward(Tensor([1.0, 1.0]))
        self.assertEqual(t1.grad.data.tolist(), [0, 1.0])
