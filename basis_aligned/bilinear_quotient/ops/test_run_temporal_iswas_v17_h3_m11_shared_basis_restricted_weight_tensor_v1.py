from types import SimpleNamespace

import numpy as np
import torch

import run_temporal_iswas_v17_h3_m11_shared_basis_restricted_weight_tensor_v1 as run


def test_fit_basis_is_orthonormal_deterministic_and_fit_only():
    rows = [{"group_number": index} for index in range(16)]
    positions = tuple((0,) for _ in rows)
    absent = {label: torch.zeros(16, 1, 12) for label in ("A11", "M11")}
    present = {label: value.clone() for label, value in absent.items()}
    for label_index, label in enumerate(("A11", "M11")):
        for row in range(8):
            present[label][row, 0, (row + label_index) % 10] = row + 1
        present[label][8:, 0, 11] = 1000
    basis, _singular = run.fit_basis(absent, present, rows, positions, rank=8)
    assert torch.allclose(basis.T @ basis, torch.eye(8), atol=1e-5)
    assert torch.allclose(basis[11], torch.zeros(8))
    for column in range(8):
        pivot = torch.argmax(torch.abs(basis[:, column]))
        assert basis[pivot, column] >= 0


def test_project_reader_keeps_only_basis_delta():
    absent = torch.randn(2, 3, 5)
    present = absent.clone()
    present[..., 0] += 2
    present[..., 4] += 7
    basis = torch.eye(5)[:, :2]
    projected = run.project_reader(absent, present, basis)
    assert torch.allclose(projected[..., 0], present[..., 0])
    assert torch.allclose(projected[..., 1], present[..., 1])
    assert torch.allclose(projected[..., 2:], absent[..., 2:])


def test_tensor_record_has_literal_shape_hash_and_values():
    value = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    record = run.tensor_record(value)
    assert record["shape"] == [2, 3, 4]
    assert len(record["sha256"]) == 64
    assert record["values"] == value.reshape(-1).tolist()


def test_price_is_exact_and_dryrun_is_model_free(monkeypatch, capsys):
    assert run.PRICE["model_forwards"] == 3 + 6 == 9
    assert run.PRICE["sequence_evaluations"] == 9 * 16
    monkeypatch.setenv("BQLIB_DRYRUN", "1")
    monkeypatch.setattr(run.producer.Bilin18TorchBackend, "load",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("model")))
    run.main()
    payload = __import__("json").loads(capsys.readouterr().out)
    assert payload["authority_ok"] is True
    assert payload["model_loaded"] is False
