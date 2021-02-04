# minitorch

Implement a minimal neural network libary from scratch.

# TODO
+ support CUDA

# Requirements

1. create virtual environment
    ```bash
    python3 -m venv minitorch-env
    ```

2. activate virtual environment
    ```bash 
    source minitorch-env/bin/activate
    ```

3. intall dependencies
    ```bash
    pip install -r requirements.txt
    ```

# Quick Start

1. clone the codebase
    ```bash
    git clone git@github.com:zhouzaida/minitorch.git
    ```

2. install or develop
    ```python
    python setup.py install
    # or
    python setup.py develop
    ```

# Examples

+ create Tensor

    ```python
    from minitorch import Tensor

    t1 = Tensor(2.0)
    t2 = Tensor(3.0)
    t3 = t1 + t2
    print(t3)  # Tensor(3.0, requires_grad=False)
    ```

+ autograd

    ```python
    from minitorch import Tensor

    t1 = Tensor(2.0, requires_grad=True)
    t2 = Tensor(3.0)
    t3 = t1 + t2
    t4 = t1 * t3
    t4.backward()
    print(f"t1 grad: {t1.grad}")  # t1 grad: Tensor(7.0, requires_grad=False)
    print(f"t2 grad: {t2.grad}")  # t2 grad: None
    ```

+ gradient for broadcast

    ```python
    from minitorch import Tensor

    t1 = Tensor([1.0, 2.0], requires_grad=True)
    t2 = Tensor(2.0, requires_grad=True)
    t3 = t1 + t2
    t3.backward(Tensor([1.0, 1.0]))
    print(f"t1 grad: {t1.grad}")  # t1 grad: Tensor([1., 1.], requires_grad=False)
    print(f"t2 grad: {t2.grad}")  # t2 grad: Tensor(2.0, requires_grad=False)
    ```

+ create neural network

    ```python
    import minitorch
    from minitorch.nn.modules.module import Module
    from minitorch.nn.modules.linear import Linear


    input = minitorch.rand(2, 3)
    linear = Linear(3, 5, bias=True)
    output = linear(input)
    print(f"output: {output}")


    class Model(Module):

        def __init__(self):
            super().__init__()
            self.linear_1 = Linear(3, 5, bias=True)
            self.linear_2 = Linear(5, 6)

        def forward(self, input):
            output = self.linear_1(input)
            output = self.linear_2(output)
            return output

    input = minitorch.rand(2, 3)
    model = Model()
    output = model(input)
    print(f"output: {output}")
    ```

# References
+ [PyTorch](https://github.com/pytorch/pytorch)
+ [autograd](https://github.com/joelgrus/autograd)
+ [tinygrad](https://github.com/geohot/tinygrad)

# Tools
+ [mypy is a static type checker for Python](https://mypy.readthedocs.io/)
+ [Flake8: Your Tool For Style Guide Enforcement](https://flake8.pycqa.org/en/latest/)
+ [unittest](https://docs.python.org/3/library/unittest.html)
