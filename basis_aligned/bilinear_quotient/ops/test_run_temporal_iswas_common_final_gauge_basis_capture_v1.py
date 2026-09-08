import torch

import run_temporal_iswas_common_final_gauge_basis_capture_v1 as runner


def test_scope_and_exact_price():
    assert runner.PRICE == {"checkpoint_loads": 1, "native_capture_forwards": 2,
                            "differentiable_transformer_forwards": 6,
                            "transformer_backward_forwards": 0, "model_updates": 0,
                            "example_evaluations": 320, "fit_parameters": 0}


def test_every_direct_authority_is_hash_bound():
    assert set(runner.FILES) == set(runner.EXPECTED)
    assert all(len(value) == 64 for value in runner.EXPECTED.values())


def test_column_major_storage_roundtrip_and_hash():
    value = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    stored = runner.stored_basis(torch, value)
    assert stored["shape"] == [6, 4]
    assert stored["roundtrip_max_abs_error"] == 0
    assert len(stored["sha256"]) == 64


def test_principal_cosines_ignore_basis_gauge():
    q = torch.linalg.qr(torch.randn(8, 3)).Q
    rotation = torch.linalg.qr(torch.randn(3, 3)).Q
    values = runner.principal(torch, q, q @ rotation)
    assert min(values) > .999999
