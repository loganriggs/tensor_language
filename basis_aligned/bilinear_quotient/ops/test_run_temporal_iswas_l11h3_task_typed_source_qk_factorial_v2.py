import numpy as np
import torch

import run_temporal_iswas_l11h3_task_typed_source_qk_factorial_v2 as runner


def test_patch_head_rows_changes_only_declared_slice_and_positions():
    output = torch.zeros(2, 3, 18)
    donor = torch.arange(output.numel(), dtype=torch.float32).reshape_as(output)
    changed = runner.patch_head_rows(output, donor, [1, 0], [(0, 2), (1,)], head=3)
    expected = output.clone()
    expected[0, (0, 2), 6:8] = donor[1, (0, 2), 6:8]
    expected[1, 1, 6:8] = donor[0, 1, 6:8]
    assert torch.equal(changed, expected)


def test_factor_mask_uses_registered_q_k_q2_k2_order():
    assert runner.factor_mask(("q", "q2")) == 0b0101
    assert runner.factor_mask(runner.FACTORS) == 0b1111


def test_finite_rejects_nested_nonfinite_value():
    assert runner.finite({"x": [0.0, 1.0]})
    assert not runner.finite({"x": [np.inf]})
