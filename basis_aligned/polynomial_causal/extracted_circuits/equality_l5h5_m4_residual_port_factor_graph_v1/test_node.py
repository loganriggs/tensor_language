import torch

from . import node


def test_residual_port_graph_closes_from_weights():
    torch.manual_seed(31)
    residuals = [torch.randn(2, 6, 96, dtype=torch.bfloat16) for _ in range(4)]
    weights = [torch.randn(96, 96, dtype=torch.bfloat16) for _ in range(4)]
    cos = torch.randn(2, 6, 16, dtype=torch.bfloat16)
    sin = torch.randn(2, 6, 16, dtype=torch.bfloat16)
    graph = node.decompose(*residuals, weights, cos, sin, head_index=1,
                           head_width=32)
    composed = node.compose(graph)
    child_raw = node._raw_ports(residuals[1], weights, 1, 32)
    transformed = [node.raw_node._transform(value, cos, sin) for value in child_raw]
    expected = node.raw_node._causal(
        node.raw_node._dot(transformed[0], transformed[1]) *
        node.raw_node._dot(transformed[2], transformed[3])).float()
    torch.testing.assert_close(composed, expected)
