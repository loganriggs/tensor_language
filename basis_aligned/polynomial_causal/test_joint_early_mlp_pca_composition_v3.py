import pytest

from joint_early_mlp_pca_composition_v3 import (
    ARM_STATES,
    arm_name,
    paired_document_cluster_lattice,
    score_registered_predictions,
)
import joint_early_mlp_pca_composition_authoritative_v3 as runner


def _additive_rows():
    output = {}
    for arm in ARM_STATES:
        gain = (1.0 if arm[0] == "P" else 2.0 if arm[0] == "E" else 0.0)
        gain += (1.0 if arm[1] == "P" else 2.0 if arm[1] == "E" else 0.0)
        gain += 1.0 if arm[2] == "E" else 0.0
        output[arm] = [5.0 - gain, 5.1 - gain, 4.9 - gain, 5.2 - gain]
    return output


def test_complete_additive_lattice_passes_every_registered_gate():
    rows = _additive_rows()
    discovery = paired_document_cluster_lattice(
        rows, ["a", "a", "b", "c"], draws=400, seed=1
    )
    heldout = paired_document_cluster_lattice(
        rows, ["d", "e", "e", "f"], draws=400, seed=2
    )
    scored = score_registered_predictions({"discovery": discovery, "heldout": heldout})
    assert len(discovery["arm_gain"]) == 18
    assert len(discovery["no_free_rider"]) == 12
    assert len(discovery["same_background_40pct_margin"]) == 12
    assert all(scored["registered_predictions"].values())
    assert discovery["same_background_40pct_margin"][
        "p0_given_p1_m2n"
    ]["point_estimate"] == pytest.approx(0.2)


def test_free_rider_fails_closed_even_when_package_has_positive_total_gain():
    rows = _additive_rows()
    for arm in (("P", "P", "N"), ("P", "P", "E")):
        background = ("N", "P", arm[2])
        rows[arm] = list(rows[background])
    analyses = {
        split: paired_document_cluster_lattice(
            rows, [f"{split}-a", f"{split}-a", f"{split}-b", f"{split}-c"],
            draws=300, seed=10 + index,
        )
        for index, split in enumerate(("discovery", "heldout"))
    }
    scored = score_registered_predictions(analyses)
    assert not scored["decisions"]["no_free_rider"]["p0_given_p1_m2n"]
    assert not scored["registered_predictions"][
        "pred_f_complete_modular_oracle_subspace_gate_passes"
    ]


def test_lattice_rejects_missing_arm_and_bad_documents():
    rows = _additive_rows()
    rows.pop(("P", "E", "N"))
    with pytest.raises(ValueError, match="complete 18-arm"):
        paired_document_cluster_lattice(rows, ["a", "a", "b", "c"], seed=1)
    rows = _additive_rows()
    with pytest.raises(ValueError, match="align"):
        paired_document_cluster_lattice(rows, ["a"], seed=1)


def test_arm_name_rejects_unregistered_state():
    assert arm_name(("P", "E", "N")) == "PEN"
    with pytest.raises(ValueError, match="unregistered"):
        arm_name(("P", "P", "P"))


def _pending_bindings():
    realization = "21ddc9ffdb7703aa570f88c5c7f4fa9fe007a988a1a7a3fd91058ee76a25ab8e"
    result = {
        "status": "scored_pending_integrity",
        "ship_realization_sha256": realization,
        "basis_artifact_sha256": "basis",
        "basis_receipt_sha256": "receipt",
    }
    manifest = {
        **result,
        "pending_result_sha256": "pending",
    }
    payload = {"ship_realization_sha256": realization}
    receipt = {
        "ship_realization_sha256": realization,
        "artifact_sha256": "basis",
        "preregistration_sha256": runner.PREREG_SHA256,
    }
    return result, manifest, payload, receipt


def test_pending_integrity_accepts_only_frozen_scientific_bindings():
    result, manifest, payload, receipt = _pending_bindings()
    runner.validate_pending_integrity(
        result, manifest, payload, receipt,
        pending_result_sha256="pending",
        current_basis_artifact_sha256="basis",
        current_basis_receipt_sha256="receipt",
        authority_exists=False,
    )


@pytest.mark.parametrize(
    "change,match",
    [
        ("pending", "scientific payload"),
        ("basis", "scored basis artifact"),
        ("receipt", "scored basis receipt"),
        ("authority", "overwrite"),
    ],
)
def test_pending_integrity_rejects_swaps_and_authority_overwrite(change, match):
    result, manifest, payload, receipt = _pending_bindings()
    kwargs = {
        "pending_result_sha256": "pending",
        "current_basis_artifact_sha256": "basis",
        "current_basis_receipt_sha256": "receipt",
        "authority_exists": False,
    }
    if change == "pending":
        kwargs["pending_result_sha256"] = "changed"
    elif change == "basis":
        kwargs["current_basis_artifact_sha256"] = "changed"
    elif change == "receipt":
        kwargs["current_basis_receipt_sha256"] = "changed"
    else:
        kwargs["authority_exists"] = True
    with pytest.raises(RuntimeError, match=match):
        runner.validate_pending_integrity(result, manifest, payload, receipt, **kwargs)


def test_v4_authority_binding_rejects_mismatch():
    exact = {
        "authorized_for_scored_experiments": True,
        "result_sha256": runner.PINNED_INPUTS[runner.EXACT_RESULT],
        "manifest_sha256": runner.PINNED_INPUTS[runner.EXACT_MANIFEST],
        "ship_realization_sha256": (
            "21ddc9ffdb7703aa570f88c5c7f4fa9fe007a988a1a7a3fd91058ee76a25ab8e"
        ),
    }
    runner.validate_exact_authority_binding(exact)
    exact["result_sha256"] = "changed"
    with pytest.raises(RuntimeError, match="authority binding"):
        runner.validate_exact_authority_binding(exact)


def test_failure_preserves_partial_payload_and_basis(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    result = tmp_path / "result.json"
    authority = tmp_path / "authority.json"
    basis = tmp_path / "basis.pt"
    basis.write_bytes(b"immutable-basis")
    result.write_text('{"sentinel": 7}\n')
    monkeypatch.setattr(runner, "MANIFEST", manifest)
    monkeypatch.setattr(runner, "RESULT", result)
    monkeypatch.setattr(runner, "AUTHORITY_RECEIPT", authority)
    before = basis.read_bytes()
    runner.mark_failed(ValueError("registered failure"), {"protected": "same"})
    failed = __import__("json").loads(result.read_text())
    assert failed["sentinel"] == 7
    assert failed["status"] == "failed_authoritative_mixed_composition"
    assert not authority.exists()
    assert basis.read_bytes() == before


def test_main_refuses_existing_namespace_before_claim(tmp_path, monkeypatch):
    existing = tmp_path / "existing.json"
    existing.write_text("do not overwrite")
    monkeypatch.setattr(runner, "OUTPUTS", (existing,))
    monkeypatch.setattr(runner, "LOCK", tmp_path / "lock")
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        runner.main()
    assert existing.read_text() == "do not overwrite"
    assert not (tmp_path / "lock").exists()
