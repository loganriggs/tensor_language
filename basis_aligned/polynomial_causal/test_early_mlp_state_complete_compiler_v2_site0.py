from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

import early_mlp_state_complete_compiler_v2_site0 as site0


def test_site0_binds_every_prelabel_authority_and_closes_runner_test() -> None:
    for path, expected in site0.PINS.items():
        assert site0.file_sha256(path) == expected
    names = {path.name for path in site0.SOURCE_CLOSURE}
    assert "early_mlp_state_complete_compiler_v2_site0.py" in names
    assert "test_early_mlp_state_complete_compiler_v2_site0.py" in names
    protocol = json.loads(site0.SOLVER_PROTOCOL.read_text())
    assert protocol["status"] == "frozen_before_any_compiler_label_or_gradient_capture"
    interpretation = json.loads(site0.INTERPRETATION_RECEIPT.read_text())
    assert interpretation["resolution"]["site0_candidate"] == "QON"
    correction = json.loads(site0.SOLVER_CORRECTION_RECEIPT.read_text())
    assert len(correction["preserved_failures"]) == 3


def test_frozen_parameter_guard_restores_flags() -> None:
    module = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    list(module.parameters())[0].requires_grad_(False)
    before = [parameter.requires_grad for parameter in module.parameters()]
    with site0.FrozenParameters(module):
        assert not any(parameter.requires_grad for parameter in module.parameters())
    assert [parameter.requires_grad for parameter in module.parameters()] == before


def test_copy_and_valid_masks_match_registered_positions() -> None:
    idx = torch.arange(256).view(1, -1)
    targets = torch.arange(1, 257).view(1, -1)
    targets[0, 100] = idx[0, 90]
    valid = site0._valid_mask(targets)
    copy = site0._copy_mask(idx, targets)
    assert not bool(valid[:, :64].any())
    assert bool(valid[:, 64:].all())
    assert copy[0, 100]
    assert not bool(copy[:, :64].any())


def test_artifact_validator_requires_selected_program_and_site1_scope(
    monkeypatch, tmp_path,
) -> None:
    artifact = tmp_path / "program.pt"
    receipt = tmp_path / "receipt.json"
    torch.save({"status": "frozen_before_any_site1_capture",
                "selection": {"selected": "missing"}, "candidates": {}}, artifact)
    receipt.write_text(json.dumps({
        "artifact_sha256": site0.file_sha256(artifact),
        "authorized_for_training": True, "training_license_sites": [1],
    }))
    monkeypatch.setattr(site0, "ARTIFACT", artifact)
    monkeypatch.setattr(site0, "RECEIPT", receipt)
    with pytest.raises(RuntimeError, match="selected program"):
        site0.validate_artifact()


def test_artifact_validator_requires_all_registered_controls(
    monkeypatch, tmp_path,
) -> None:
    artifact = tmp_path / "program.pt"
    receipt = tmp_path / "receipt.json"
    torch.save({
        "status": "frozen_before_any_site1_capture",
        "selection": {"selected": "B8"},
        "candidates": {"B8": {"family": "B_state_complete_affine_euclidean"}},
        "controls": {},
    }, artifact)
    receipt.write_text(json.dumps({
        "artifact_sha256": site0.file_sha256(artifact),
        "authorized_for_training": True, "training_license_sites": [1],
    }))
    monkeypatch.setattr(site0, "ARTIFACT", artifact)
    monkeypatch.setattr(site0, "RECEIPT", receipt)
    with pytest.raises(RuntimeError, match="controls are incomplete"):
        site0.validate_artifact()


def test_site0_outputs_are_isolated_and_no_final_path_is_referenced() -> None:
    assert all("site0" in path.name for path in site0.OUTPUTS)
    source = site0.Path(site0.__file__).read_text()
    assert 'rows["compiler_final"]' not in source
    assert "compiler_final" not in [path.name for path in site0.OUTPUTS]


def test_site0_uses_role_restricted_loader() -> None:
    source = site0.Path(site0.__file__).read_text()
    assert "fresh_rows.load_roles_and_validate" in source
    assert '("compiler_fit", "compiler_validation")' in source


def test_literal_shuffle_moves_only_p_labels() -> None:
    captured = {
        "z": torch.arange(12, dtype=torch.float32).view(3, 4),
        "p": torch.arange(6, dtype=torch.float32).view(3, 2),
        "mo": torch.ones(3, 2),
        "c": torch.zeros(3, 2),
        "adjoint": torch.arange(6, dtype=torch.float32).view(3, 2) + 1,
    }
    permutation = torch.tensor([2, 0, 1])
    shuffled = site0.shuffled_fit_capture(captured, permutation)
    assert torch.equal(shuffled["p"], captured["p"][permutation])
    assert torch.equal(shuffled["c"], captured["p"][permutation] - captured["mo"])
    for key in ("z", "mo", "adjoint"):
        assert shuffled[key] is captured[key]
