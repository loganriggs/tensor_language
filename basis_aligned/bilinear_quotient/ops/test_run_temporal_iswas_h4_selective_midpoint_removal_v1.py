import ast
from pathlib import Path

import torch

import run_temporal_iswas_h4_selective_midpoint_removal_v1 as target


def test_midpoint_is_symmetric_and_exact():
    left = torch.tensor([1.0, 3.0]); right = torch.tensor([5.0, 7.0])
    expected = torch.tensor([3.0, 5.0])
    assert torch.equal(target.midpoint_tensor(left, right), expected)
    assert torch.equal(target.midpoint_tensor(right, left), expected)


def test_authority_hashes_and_populations():
    paths = {"prior": target.PRIOR, "original_builder": target.ORIGINAL_BUILDER,
        "ood_builder": target.OOD_BUILDER, "original_h4": target.ORIGINAL_H4,
        "ood_h4": target.OOD_H4, "parent_runner": target.PARENT_RUNNER,
        "joint": target.JOINT, "accounting": target.ACCOUNTING, "producer": target.PRODUCER}
    assert {name: target.sha(path) for name, path in paths.items()} == target.EXPECTED
    assert len(target.original.build_rows()) == len(target.ood.build_rows()) == 32


def test_union_and_price_are_literal():
    assert target.UNION == {9: (1, 4), 11: (3,), 15: (5,)}
    assert target.PRICE["model_forwards"] == 8
    assert target.PRICE["sequence_evaluations"] == 8 * 128
    assert target.PRICE["scored_token_positions"] == 8 * 128 * 2


def test_gate_schema_atomic_write_and_absent_result():
    source = Path(target.__file__).read_text(); tree = ast.parse(source)
    assert "# BQGATE: EXPERIMENT" in source and len(target.PREDICTION_KEYS) == 5
    assert any(isinstance(node, ast.Call) and getattr(node.func, "id", "") == "atomic_create_json"
               for node in ast.walk(tree))
    assert not target.OUT.exists()
