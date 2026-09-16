import torch

from . import node

def _synthetic_corners(batch=7):
    torch.manual_seed(29)
    baseline = torch.randn(batch)
    edges = {name: torch.randn(batch) for name in node.TERMS}
    corners = {}
    for corner in node.REQUIRED_CORNERS:
        value = baseline.clone()
        for mask, name in zip(node.MASKS, node.TERMS):
            if mask & corner == mask: value = value - edges[name]
        corners[corner] = value
    return corners, edges

def test_selected_mobius_edges_close():
    corners, expected = _synthetic_corners()
    actual = node.decompose(corners)
    for name in node.TERMS: torch.testing.assert_close(actual[name], expected[name])
    torch.testing.assert_close(node.compose(actual), sum(expected.values()))

def test_remove_edge_removes_only_named_contribution():
    corners, _ = _synthetic_corners(); edges = node.decompose(corners); name = "middle_writes_4_7*mlp_10"
    torch.testing.assert_close(node.compose(edges) - node.remove_edge(edges, name), edges[name])

def test_missing_corner_and_shape_mismatch_rejected():
    corners, _ = _synthetic_corners(); corners.pop(24)
    try: node.decompose(corners)
    except ValueError: pass
    else: raise AssertionError("missing corner accepted")
    corners, _ = _synthetic_corners(); corners[24] = torch.randn(2, 3)
    try: node.decompose(corners)
    except ValueError: pass
    else: raise AssertionError("shape mismatch accepted")
