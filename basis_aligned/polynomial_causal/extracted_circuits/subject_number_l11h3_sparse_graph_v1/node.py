"""Nine-edge Möbius graph for the extracted subject-number circuit."""
from __future__ import annotations

PORTS = ("embedding_recurrence", "early_writes_0_3", "middle_writes_4_7", "mlp_8", "mlp_10")
MASKS = (8, 1, 4, 2, 24, 12, 16, 9, 20)
TERMS = ("mlp_8", "embedding_recurrence", "middle_writes_4_7", "early_writes_0_3",
         "mlp_8*mlp_10", "middle_writes_4_7*mlp_8", "mlp_10",
         "embedding_recurrence*mlp_8", "middle_writes_4_7*mlp_10")
REQUIRED_CORNERS = (0, 1, 2, 4, 8, 9, 12, 16, 20, 24)

def _shape(value):
    return tuple(value.shape)

def decompose(corners):
    """Extract the nine signed damage edges from required corner margins."""
    missing = sorted(set(REQUIRED_CORNERS) - set(corners))
    if missing: raise ValueError(f"missing corner masks: {missing}")
    shapes = {_shape(corners[mask]) for mask in REQUIRED_CORNERS}
    if len(shapes) != 1: raise ValueError("all corner margins must have matching shapes")
    edges = {}
    for mask, name in zip(MASKS, TERMS):
        bits = [1 << i for i in range(len(PORTS)) if mask & (1 << i)]
        if len(bits) == 1:
            dividend = corners[mask] - corners[0]
        elif len(bits) == 2:
            dividend = corners[mask] - corners[bits[0]] - corners[bits[1]] + corners[0]
        else:
            raise ValueError("exported graph contains a higher-order edge")
        edges[name] = -dividend
    return edges

def compose(edges):
    """Predict joint intervention damage by summing frozen graph edges."""
    missing = [name for name in TERMS if name not in edges]
    if missing: raise ValueError(f"missing graph edges: {missing}")
    shapes = {_shape(edges[name]) for name in TERMS}
    if len(shapes) != 1: raise ValueError("all graph edges must have matching shapes")
    result = edges[TERMS[0]]
    for name in TERMS[1:]: result = result + edges[name]
    return result

def remove_edge(edges, name):
    """Return the graph prediction with exactly one named edge removed."""
    if name not in TERMS: raise KeyError(name)
    return compose({key: value for key, value in edges.items() if key != name} | {name: edges[name] * 0})
