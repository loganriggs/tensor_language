"""Deterministic sentinel covers for incremental family-separability screens.

The input graph is retrospective: target -> sibling controls that exceeded a
fixed cross-effect bar in a completed exhaustive receipt.  A cover therefore
detects known collision geometry.  It does not certify a projector retrained
against only those controls; independent validation and periodic exhaustive
calibration remain required.
"""

# BQGATE: LIBRARY
from __future__ import annotations


def collision_graph(members, *, arm="own", cross_bar=0.05):
    """Return target -> controls whose absolute measured effect exceeds the bar."""
    graph = {}
    for target, report in members.items():
        arms = report.get("arms", {})
        siblings = arms.get(arm, {}).get("siblings")
        if not isinstance(siblings, dict):
            continue
        graph[target] = {
            control for control, value in siblings.items()
            if abs(float(value)) > float(cross_bar)
        }
    return graph


def greedy_disjoint_covers(graph, *, maximum=3):
    """Greedily find deterministic, control-disjoint covers of nonempty targets.

    Each round removes its controls before constructing the next cover.  Ties
    choose the lexicographically largest control, making the receipt independent
    of set iteration order.
    """
    universe = {target for target, controls in graph.items() if controls}
    available = set().union(*(graph[target] for target in universe)) if universe else set()
    covers = []
    for _ in range(int(maximum)):
        remaining = set(universe)
        selected = []
        while remaining and available:
            score, control = max(
                (sum(control in graph[target] for target in remaining), control)
                for control in available
            )
            if score == 0:
                break
            selected.append(control)
            remaining -= {target for target in remaining if control in graph[target]}
            available.remove(control)
        covers.append({
            "controls": selected,
            "covered": sorted(universe - remaining),
            "uncovered": sorted(remaining),
        })
    return covers


def estimated_seconds(*, targets, controls, intercept=19.0, per_control=9.3):
    """Empirical v289 wall-time model for one incremental audit block."""
    return float(targets) * (float(intercept) + float(per_control) * float(controls))
