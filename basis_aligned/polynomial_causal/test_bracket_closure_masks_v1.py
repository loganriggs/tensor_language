from __future__ import annotations

import pytest
import torch

from bracket_closure_masks_v1 import (
    BracketDomain,
    ClosureCondition,
    DelimiterFamily,
    DelimiterRegistry,
    build_bracket_masks,
    stratified_cells,
    synthetic_canary_design,
)


REGISTRY = DelimiterRegistry(
    (
        DelimiterFamily("round", (10,), (11,)),
        DelimiterFamily("square", (20,), (21,)),
    ),
    quote_control_ids=(30,),
    punctuation_control_ids=(40,),
)


def test_prefix_parser_labels_compatible_incompatible_and_empty_stack_without_routing() -> None:
    rows = torch.tensor([
        [10, 90, 91, 11, 99],  # predict compatible ')' at p=2
        [20, 90, 91, 11, 99],  # predict incompatible ')' at p=2
        [90, 90, 91, 11, 99],  # predict ')' without opener
    ], dtype=torch.long).contiguous()
    masks = build_bracket_masks(
        rows, REGISTRY,
        (BracketDomain.PROSE, BracketDomain.CODE, BracketDomain.PROSE),
        first_prediction=0,
    )
    assert masks.compatible[0, 2]
    assert masks.incompatible[1, 2]
    assert masks.no_opener[2, 2]
    assert int(masks.depth[0, 2]) == 1
    assert int(masks.distance[0, 2]) == 3
    assert int(masks.family_index[0, 2]) == 0
    assert int(masks.domain_index[1, 2]) == 1
    masks.validate()


def test_nested_depth_distance_and_controls_are_disjoint() -> None:
    rows = torch.tensor([
        [10, 20, 90, 91, 21, 11, 30, 40],
    ], dtype=torch.long).contiguous()
    masks = build_bracket_masks(
        rows, REGISTRY, (BracketDomain.CODE,), first_prediction=0,
    )
    # Predict ']' from nested stack at p=3, then ')' after ']' has popped at p=4.
    assert masks.compatible[0, 3] and int(masks.depth[0, 3]) == 2
    assert masks.compatible[0, 4] and int(masks.depth[0, 4]) == 1
    assert masks.quote_control[0, 5]
    assert masks.punctuation_control[0, 6]
    cells = torch.stack(tuple(masks.named_cells().values())).to(torch.int8)
    assert int(cells.sum(0).max()) == 1
    strata = stratified_cells(masks)
    assert strata["code:family_1:compatible:depth_2:distance_1_8"][0, 3]


def test_first_prediction_and_registry_fail_closed() -> None:
    rows = torch.tensor([[10, 11, 90]], dtype=torch.long).contiguous()
    masks = build_bracket_masks(
        rows, REGISTRY, (BracketDomain.PROSE,), first_prediction=1,
    )
    assert not masks.compatible[0, 0]
    with pytest.raises(ValueError, match="disjoint"):
        DelimiterRegistry(
            (DelimiterFamily("a", (1,), (2,)), DelimiterFamily("b", (3,), (4,))),
            quote_control_ids=(2,), punctuation_control_ids=(5,),
        )
    with pytest.raises(ValueError, match="every document"):
        build_bracket_masks(rows, REGISTRY, (), first_prediction=0)


def test_synthetic_design_is_exact_balanced_and_typed() -> None:
    cells = synthetic_canary_design(2)
    # Per domain: 2 families * (2 conditions * 3 depths * 4 distances + no-opener)
    # plus quote and punctuation controls.
    assert len(cells) == 2 * (2 * (2 * 3 * 4 + 1) + 2)
    assert len(set(cells)) == len(cells)
    for domain in BracketDomain:
        selected = tuple(cell for cell in cells if cell.domain is domain)
        assert sum(cell.condition is ClosureCondition.COMPATIBLE for cell in selected) == 24
        assert sum(cell.condition is ClosureCondition.INCOMPATIBLE for cell in selected) == 24
        assert sum(cell.condition is ClosureCondition.NO_OPENER for cell in selected) == 2
        assert sum(cell.condition is ClosureCondition.QUOTE_CONTROL for cell in selected) == 1
        assert sum(cell.condition is ClosureCondition.PUNCTUATION_CONTROL for cell in selected) == 1
