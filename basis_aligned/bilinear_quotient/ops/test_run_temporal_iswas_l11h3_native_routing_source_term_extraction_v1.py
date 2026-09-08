import numpy as np
import torch

import run_temporal_iswas_l11h3_native_routing_source_term_extraction_v1 as runner


def test_add_head_delta_changes_only_head_three():
    output = torch.zeros(2, 3, 18)
    delta = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    changed = runner.add_head_delta(output, delta)
    assert torch.equal(changed[..., 6:8], delta)
    assert torch.equal(changed[..., :6], output[..., :6])
    assert torch.equal(changed[..., 8:], output[..., 8:])


def test_tensor_sha_is_deterministic_and_dtype_normalized():
    value = torch.tensor([[1.0, 2.0]])
    assert runner.tensor_sha(value) == runner.tensor_sha(value.double())


def test_finite_rejects_nested_nonfinite():
    assert runner.finite({"x": [0.0, 1.0]})
    assert not runner.finite({"x": [np.nan]})
