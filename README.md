# minitorch

Implement a minimal neural network libary from scratch.

# TODO

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

# References
+ [PyTorch](https://github.com/pytorch/pytorch)
+ [mypy is a static type checker for Python](https://mypy.readthedocs.io/)
+ [autograd](https://github.com/joelgrus/autograd)