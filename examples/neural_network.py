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

for name, module in model.named_modules(prefix='model'):
    print(f"{name}: {module}")
