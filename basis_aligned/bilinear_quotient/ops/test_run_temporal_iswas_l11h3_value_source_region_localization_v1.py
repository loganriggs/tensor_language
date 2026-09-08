import ast
from pathlib import Path

import torch

import run_temporal_iswas_l11h3_value_source_region_localization_v1 as target


def test_source_regions_are_disjoint_and_exhaustive():
    regions = target.source_regions([1, 2, 3, 4, 5, 6], [1, 9, 8, 4, 5, 6], 5)
    assert regions["precue"] == (0,) and regions["cue_span"] == (1, 2)
    assert regions["bridge"] == (3, 4) and regions["query"] == (5,)
    assert set(regions["cue_span"] + regions["bridge"] + regions["query"]) == set(regions["full_prefix"])


def test_value_patch_changes_only_selected_rows_positions_and_head():
    output = torch.zeros(2, 4, 18); donor = torch.arange(144.0).reshape(2, 4, 18)
    changed = target.patch_value_tensor(output, donor, [1, 0], [(1, 3), (2,)], head=3)
    assert torch.equal(changed[0, 1, 6:8], donor[1, 1, 6:8])
    assert torch.equal(changed[0, 3, 6:8], donor[1, 3, 6:8])
    assert torch.equal(changed[1, 2, 6:8], donor[0, 2, 6:8])
    changed[0, 1, 6:8] = 0; changed[0, 3, 6:8] = 0; changed[1, 2, 6:8] = 0
    assert not changed.any()


def test_vector_composition_exact_case():
    left, right, full = [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]
    import numpy as np
    result = target.vector_composition([np.asarray(left), np.asarray(right)], np.asarray(full))
    assert result["relative_l2_error"] == 0.0 and abs(result["cosine"] - 1.0) < 1e-12


def test_authority_price_gate_and_atomic_result():
    paths = {"prior": target.PRIOR, "original_builder": target.ORIGINAL_BUILDER,
        "ood_builder": target.OOD_BUILDER, "factor_result": target.FACTOR_RESULT,
        "removal_result": target.REMOVAL_RESULT, "accounting": target.ACCOUNTING,
        "producer": target.PRODUCER}
    assert {name: target.sha(path) for name, path in paths.items()} == target.EXPECTED
    assert target.PRICE["model_forwards"] == 2 * (1 + 2 * len(target.ARMS))
    source = Path(target.__file__).read_text(); tree = ast.parse(source)
    assert "# BQGATE: EXPERIMENT" in source
    assert any(isinstance(node, ast.Call) and getattr(node.func, "id", "") == "atomic_create_json"
               for node in ast.walk(tree))
    assert not target.OUT.exists()
