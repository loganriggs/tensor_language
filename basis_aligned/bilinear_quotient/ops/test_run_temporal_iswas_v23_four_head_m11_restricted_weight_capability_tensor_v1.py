import numpy as np

import run_temporal_iswas_v23_four_head_m11_restricted_weight_capability_tensor_v1 as experiment


def test_reader_basis_is_complete_four_cell_span():
    states = np.zeros((4, 16, 3, 6), dtype=np.float64)
    mask = np.ones((4, 16, 3), dtype=bool)
    for panel in (0, 1):
        for group in range(16):
            states[panel, group, :, 2*panel + group % 2] = 1.0
    basis, cells, singular = experiment.reader_basis_from_states(states, mask)
    assert cells.shape == (4, 6)
    assert basis.shape == (4, 6)
    assert np.all(singular[:4] > 0)
    assert np.max(np.abs(basis @ basis.T - np.eye(4))) < 1e-6


def test_weight_stage_price_has_no_activation_or_causal_execution():
    assert experiment.PRICE["checkpoint_loads"] == 1
    assert experiment.PRICE["model_forwards"] == 0
    assert experiment.PRICE["sequence_evaluations"] == 0
    assert experiment.PRICE["transformer_backwards"] == 0
