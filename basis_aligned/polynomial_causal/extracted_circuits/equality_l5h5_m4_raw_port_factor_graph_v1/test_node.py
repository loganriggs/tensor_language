import torch

from . import node


def test_raw_port_graph_closes_without_child_score_input():
    torch.manual_seed(23)
    shape = (2, 7, 64)
    derived = [torch.randn(shape, dtype=torch.bfloat16) for _ in range(4)]
    native = [torch.randn(shape, dtype=torch.bfloat16) for _ in range(4)]
    cos = torch.randn(2, 7, 32, dtype=torch.bfloat16)
    sin = torch.randn(2, 7, 32, dtype=torch.bfloat16)
    graph = node.decompose(derived, native, cos, sin)
    transformed = [node._transform(value, cos, sin) for value in native]
    expected = node._causal(node._dot(transformed[0], transformed[1]) *
                            node._dot(transformed[2], transformed[3])).float()
    torch.testing.assert_close(node.compose(graph), expected)
    for name in node.NODE_NAMES:
        torch.testing.assert_close(node.compose(graph) - node.remove_node(graph, name),
                                   graph[name])
