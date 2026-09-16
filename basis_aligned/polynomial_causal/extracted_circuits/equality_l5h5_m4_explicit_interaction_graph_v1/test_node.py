import torch

from . import node


def test_mobius_graph_closes_and_removes_only_interaction():
    torch.manual_seed(9)
    baseline = torch.randn(2, 5, 5)
    child_effect = torch.randn_like(baseline)
    remainder_effect = torch.randn_like(baseline)
    interaction = torch.randn_like(baseline)
    child = baseline + child_effect
    remainder = baseline + remainder_effect
    joint = (baseline + child_effect) + remainder_effect + interaction
    graph = node.decompose(baseline, child, remainder, joint)
    torch.testing.assert_close(node.compose(graph), joint, rtol=1e-6, atol=1e-6)
    expected_additive = (baseline + child_effect) + remainder_effect
    torch.testing.assert_close(node.remove_interaction(graph), expected_additive, rtol=1e-6, atol=1e-6)
    rolled = node.replace_with_rolled_interaction(graph)
    torch.testing.assert_close(rolled - expected_additive, torch.roll(graph["interaction"], 1, 1), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(torch.roll(graph["interaction"], 1, 1).norm(), graph["interaction"].norm())
