import ast
from pathlib import Path

import numpy as np
import torch

import run_temporal_iswas_h4_reader_factor_factorial_v1 as target


def test_patch_query_changes_only_selected_head_and_query():
    output = torch.zeros(2, 3, 18); donor = torch.arange(108.0).reshape(2, 3, 18)
    changed = target.patch_factor_tensor(output, donor, [1, 0], [2, 1], head=4, scope="query")
    assert torch.equal(changed[0, 2, 8:10], donor[1, 2, 8:10])
    assert torch.equal(changed[1, 1, 8:10], donor[0, 1, 8:10])
    mask = changed != 0; mask[0, 2, 8:10] = False; mask[1, 1, 8:10] = False
    assert not mask.any()


def test_patch_prefix_changes_selected_head_through_query():
    output = torch.zeros(2, 4, 18); donor = torch.arange(144.0).reshape(2, 4, 18)
    changed = target.patch_factor_tensor(output, donor, [1, 0], [2, 1], head=3, scope="prefix")
    assert torch.equal(changed[0, :3, 6:8], donor[1, :3, 6:8])
    assert torch.equal(changed[1, :2, 6:8], donor[0, :2, 6:8])
    assert not changed[0, 3].any() and not changed[1, 2:].any()


def test_mobius_additive_and_interacting_cases():
    q = np.array([1.0, 2.0]); q2 = np.array([2.0, 1.0])
    additive = target.mobius_report(q, q2, q + q2)
    assert additive["interaction_rms_over_both_rms"] == 0.0
    assert abs(additive["additive_cosine_to_both_effect"] - 1.0) < 1e-12
    interacting = target.mobius_report(q, q2, q + q2 + np.array([1.0, -1.0]))
    assert interacting["interaction_rms_over_both_rms"] > 0.0


def test_frozen_inventory_price_and_gate():
    assert len(target.ARMS) == 9 and len(target.arm_specs()) == 9
    assert target.PRICE["model_forwards"] == 1 + 2 * len(target.ARMS)
    assert target.PRICE["sequence_evaluations"] == target.PRICE["model_forwards"] * 128
    source = Path(target.__file__).read_text(); tree = ast.parse(source)
    assert "# BQGATE: EXPERIMENT" in source
    assert any(isinstance(node, ast.Call) and getattr(node.func, "id", "") == "atomic_create_json"
               for node in ast.walk(tree))
