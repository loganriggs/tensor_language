import torch

from . import node


def test_precision_factor_graph_closes_and_removes_named_nodes():
    torch.manual_seed(17)
    shape = (2, 6, 6)
    first_derived = torch.randn(shape)
    first_native = torch.randn(shape)
    second_derived = torch.randn(shape)
    second_native = torch.randn(shape)
    mask = torch.ones(shape[-2:], dtype=torch.bool).tril()
    native_child = (first_native * second_native).masked_fill(~mask, 0)
    native_child = native_child + torch.randn(shape).masked_fill(~mask, 0) * 1e-3
    graph = node.decompose(first_derived, first_native, second_derived,
                           second_native, native_child)
    torch.testing.assert_close(node.compose(graph), native_child)
    for name in node.NODE_NAMES:
        torch.testing.assert_close(node.compose(graph) - node.remove_node(graph, name),
                                   graph[name])
    rolled = node.replace_with_rolled_arithmetic(graph)
    expected = node.remove_node(graph, "arithmetic") + torch.roll(
        graph["arithmetic"], shifts=1, dims=1)
    torch.testing.assert_close(rolled, expected)
    torch.testing.assert_close(torch.roll(graph["arithmetic"], 1, 1).norm(),
                               graph["arithmetic"].norm())
