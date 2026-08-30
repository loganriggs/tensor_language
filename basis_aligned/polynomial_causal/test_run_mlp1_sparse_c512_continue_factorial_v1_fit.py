import math

import pytest
import torch

import run_mlp1_sparse_c512_continue_factorial_v1_fit as subject


def record(seed, final, values=None):
    values = values or [final - 0.002, final - 0.001, final]
    curve = [
        {"step": step, "select_r2": value, "train_mse": 1.0, "learning_rate": 0.1}
        for step, value in zip(range(200, 2401, 200), [values[0]] * 9 + values)
    ]
    return {
        "seed": seed,
        "final_select_r2": final,
        "curve": curve,
        "convergence": subject.convergence_metrics(curve),
    }


def test_selection_uses_only_final_r2_and_breaks_tie_by_seed():
    rows = [record(0, 0.7), record(1, 0.8), record(2, 0.8)]
    assert subject.select_seed(rows)["seed"] == 1


def test_convergence_and_admission_are_separate():
    rows = [record(0, 0.79), record(1, 0.80), record(2, 0.81)]
    gates = subject.selection_gates(rows, 0.91)
    assert gates["selected_seed"] == 2
    assert gates["selected_curve_converged"] is True
    assert gates["seed_final_select_r2_std_le_0p02"] is True
    assert gates["admitted_to_final"] is True
    failed = subject.selection_gates(rows, 0.899)
    assert failed["admitted_to_final"] is False


def test_convergence_rejects_late_instability_and_wrong_cadence():
    unstable = record(0, 0.80, [0.77, 0.82, 0.80])
    assert unstable["convergence"]["converged"] is False
    with pytest.raises(RuntimeError, match="cadence"):
        subject.convergence_metrics(unstable["curve"][:-1])


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_selection_rejects_nonfinite_ce_recovery(value):
    rows = [record(0, 0.79), record(1, 0.80), record(2, 0.81)]
    with pytest.raises(RuntimeError, match="non-finite"):
        subject.selection_gates(rows, value)


def test_create_only_writer_runs_guard_adjacent_to_publication(tmp_path):
    target = tmp_path / "artifact.json"
    observed = []

    def guard():
        assert not target.exists()
        observed.append("guarded")

    subject.write_json_create_only(target, {"finite": 1.0}, pre_link_check=guard)
    assert observed == ["guarded"]
    assert target.is_file()
    with pytest.raises(FileExistsError):
        subject.write_json_create_only(target, {"finite": 2.0})


def test_bundle_replay_requires_exact_program_and_price():
    state = {
        "encoder": torch.zeros(512, 4608),
        "decoder": torch.zeros(1152, 512),
        "intercept": torch.zeros(1152),
    }
    state["encoder"][:, 0] = 1.0
    bundle = {
        "schema": "mlp1_sparse_c512_continue_factorial_v1_fit_bundle",
        "status": "selected_program_frozen_before_final",
        "authority_sha256": "a" * 64,
        "program": state,
        "selected_seed": 1,
        "price": subject.sparse.SparseDownProgram.price(),
        "final_opened": False,
    }
    subject.validate_bundle(bundle, state, "a" * 64, 1)
    bad = dict(bundle)
    bad["price"] = {"stored_float32_reals": 1}
    with pytest.raises(RuntimeError, match="bundle semantics"):
        subject.validate_bundle(bad, state, "a" * 64, 1)


def test_artifact_snapshot_binds_absence_and_later_presence(tmp_path):
    authority = tmp_path / "authority.json"
    bundle = tmp_path / "bundle.pt"
    before = subject.artifact_snapshot((authority, bundle))
    assert before == {authority.name: None, bundle.name: None}
    authority.write_text("{}\n")
    after = subject.artifact_snapshot((authority, bundle))
    assert after != before
    assert after[bundle.name] is None


def test_failure_input_observation_records_stability_and_explicit_drift(monkeypatch):
    expected = {"source_commit": "abc"}
    monkeypatch.setattr(subject, "protected_snapshot", lambda *args: dict(expected))
    assert subject.failure_input_observation(expected, "abc", {}, "a", "r") == {
        "status": "matches_initial", "snapshot": expected,
    }

    def drift(*args):
        raise RuntimeError("source drift")

    monkeypatch.setattr(subject, "protected_snapshot", drift)
    observed = subject.failure_input_observation(expected, "abc", {}, "a", "r")
    assert observed["status"] == "protected_input_validation_failed"
    assert observed["error_type"] == "RuntimeError"
    assert "source drift" in observed["error"]


def test_verify_protected_rejects_mismatch_between_claim_checks(monkeypatch):
    checks = []
    monkeypatch.setattr(
        subject.rows_life.base, "require_claim", lambda claim, lock: checks.append((claim, lock)),
    )
    monkeypatch.setattr(
        subject, "protected_snapshot", lambda *args: {"source_commit": "changed"},
    )
    with pytest.raises(RuntimeError, match="inputs changed"):
        subject.verify_protected(
            {"source_commit": "expected"}, "abc", {}, "a", "r", "claim",
        )
    assert checks == [("claim", subject.LOCK)]


def test_verify_protected_checks_claim_before_and_after_exact_replay(monkeypatch):
    expected = {"source_commit": "exact"}
    checks = []
    monkeypatch.setattr(
        subject.rows_life.base, "require_claim", lambda claim, lock: checks.append((claim, lock)),
    )
    monkeypatch.setattr(subject, "protected_snapshot", lambda *args: dict(expected))
    subject.verify_protected(expected, "abc", {}, "a", "r", "claim")
    assert checks == [("claim", subject.LOCK), ("claim", subject.LOCK)]


def test_success_guard_rejects_terminal_injected_during_protected_replay(
    monkeypatch, tmp_path,
):
    target = tmp_path / "authority.json"
    rival = tmp_path / "failure.json"
    expected = {"source_commit": "exact"}
    monkeypatch.setattr(subject.rows_life.base, "require_claim", lambda *args: None)

    def inject_rival(*args):
        rival.write_text("{}\n")
        return dict(expected)

    monkeypatch.setattr(subject, "protected_snapshot", inject_rival)

    def guard():
        subject.finish_publication_guard(
            expected, "abc", {}, "a", "r", "claim", (target, rival),
            "rival appeared",
        )

    with pytest.raises(RuntimeError, match="rival appeared"):
        subject.write_json_create_only(target, {"status": "success"}, pre_link_check=guard)
    assert rival.is_file()
    assert not target.exists()
