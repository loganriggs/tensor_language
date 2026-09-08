import numpy as np
import pytest
import torch

import run_temporal_iswas_l11h3_source_tensor_upstream_head_factorial_v1 as runner


def test_patch_heads_changes_only_selected_slice_and_positions():
    output = torch.zeros(2, 3, 18)
    donor = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    captures = {"L03H04": donor}
    got = runner.patch_heads(output, captures, ["L03H04"], [1, 0], [(0, 2), (1,)])
    expected = output.clone()
    expected[0, (0, 2), 8:10] = donor[1, (0, 2)]
    expected[1, 1, 8:10] = donor[0, 1]
    assert torch.equal(got, expected)


def test_vector_composition_is_exact_for_additive_parts():
    parts = [np.asarray([1.0, 0.0]), np.asarray([0.0, 2.0])]
    report = runner.vector_composition(parts, np.asarray([1.0, 2.0]))
    assert report["relative_l2_error"] == 0.0
    assert report["cosine"] == pytest.approx(1.0)


def test_candidate_union_and_top5_are_frozen():
    assert len(runner.LABELS) == 6
    assert set(runner.TOP5["temporal"]) | set(runner.TOP5["iswas"]) == set(runner.LABELS)
    assert set(runner.TOP5["temporal"]) & set(runner.TOP5["iswas"]) == set(runner.SHARED)
