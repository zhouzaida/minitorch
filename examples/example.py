from minitorch import Tensor


t1 = Tensor(4.0)
t2 = Tensor(3.0, requires_grad=True)
t3 = -t2
t4 = t1 * t3
t4.backward()
print(f"t1 grad: {t1.grad}")  # t1 grad: Tensor(7.0, requires_grad=False)
print(f"t2 grad: {t2.grad}")  # t2 grad: None
