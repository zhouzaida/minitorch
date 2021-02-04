"""Execute danamic computational graph

Topological Sorting
"""

from collections import defaultdict, deque

from minitorch import Tensor
from .node import Node


class NodeTask:
    def __init__(self, node: Node, grad_input: Tensor):
        self.node = node
        self.grad_input = grad_input

    def update_grad_input(self, grad_input: Tensor):
        self.grad_input += grad_input


class Engine:

    def execute(self, tensor, grad_input):
        dependencies = self._compute_dependencies(tensor.grad_fn)
        not_ready_dict = {}
        ready_queue = deque([NodeTask(tensor.grad_fn, grad_input)])
        while ready_queue:
            node_task = ready_queue.popleft()
            grad_outputs = node_task.node(node_task.grad_input)
            if grad_outputs is None:
                continue
            for grad_output, edge in zip(grad_outputs, node_task.node.next_edges):
                next_node = edge.node
                dependencies[next_node] -= 1
                if next_node not in not_ready_dict:
                    not_ready_dict[next_node] = NodeTask(next_node, grad_output)
                else:
                    not_ready_dict[next_node].update_grad_input(grad_output)
                if dependencies[next_node] == 0:
                    ready_queue.append(not_ready_dict[next_node])

    def _compute_dependencies(self, root: Node):
        dependencies = defaultdict(int)
        dependencies[root] = 0
        queue = deque([root])
        while queue:
            node = queue.pop()
            if hasattr(node, "next_edges"):
                for edge in node.next_edges:
                    next_node = edge.node
                    dependencies[next_node] += 1
                    queue.append(next_node)
        return dependencies
