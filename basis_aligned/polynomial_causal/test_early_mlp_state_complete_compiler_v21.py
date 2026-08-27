from __future__ import annotations

import json

import pytest

import early_mlp_state_complete_compiler_v21 as runner


def _install_stage_paths(monkeypatch, tmp_path):
    paths = {
        "site0": (tmp_path / "site0.pt", tmp_path / "site0.json"),
        "site1": (tmp_path / "site1.pt", tmp_path / "site1.json"),
    }
    monkeypatch.setattr(runner, "STAGE_PATHS", paths)
    monkeypatch.setattr(runner, "DOWNSTREAM", {
        "site0": (*paths["site1"], tmp_path / "programs.pt", tmp_path / "unlock.json"),
        "site1": (tmp_path / "programs.pt", tmp_path / "unlock.json"),
    })
    row_receipt = tmp_path / "rows.json"
    row_receipt.write_text("{}")
    monkeypatch.setattr(runner, "ROWS_RECEIPT", row_receipt)
    monkeypatch.setattr(runner.authority, "protected_snapshot", lambda: {"pin": "ok"})
    monkeypatch.setattr(runner, "_validate_candidate_bank", lambda *_: None)
    monkeypatch.setattr(runner, "_validate_stage_semantics", lambda *_: None)
    monkeypatch.setattr(runner, "_validate_launch_state", lambda _: None)
    return paths


def _stage_inputs(stage):
    names = runner.STAGE_LEDGERS[stage]
    candidates = {name: {"candidate": {"state": {}, "metrics": {}}} for name in names}
    controls = ({
        "mean_site0": {}, "full_native_site0": {},
    } if stage == "site0" else {
        "mean_site1": {}, "full_native_site1_true_context": {},
        "full_native_site1_shuffle_context": {},
    })
    diagnostics = {
        "fit_permutation_sha256": "0" * 64,
        "capture_hashes": {},
        "contexts": {},
    }
    if stage == "site1":
        diagnostics["mean_control"] = {}
        diagnostics["mean_context"] = {}
        diagnostics["mean_score"] = {}
    else:
        diagnostics["mean_score"] = {}
    return candidates, controls, diagnostics


@pytest.mark.parametrize("stage", ["site0", "site1"])
def test_stage_ledger_is_written_reloaded_and_receipted_last(
    monkeypatch, tmp_path, stage,
) -> None:
    paths = _install_stage_paths(monkeypatch, tmp_path)
    inputs = _stage_inputs(stage)
    payload, receipt = runner.freeze_preselector_stage(
        stage, *inputs, launch_state=object(),
    )
    assert paths[stage][0].is_file() and paths[stage][1].is_file()
    assert payload["authorized_for_training"] is False
    assert payload["authorized_for_final_scoring"] is False
    assert receipt["artifact_sha256"] == runner.authority.file_sha256(paths[stage][0])
    assert receipt["artifact_bytes"] == paths[stage][0].stat().st_size
    assert json.loads(paths[stage][1].read_text()) == receipt


def test_stage_freeze_refuses_overwrite_and_downstream(monkeypatch, tmp_path) -> None:
    paths = _install_stage_paths(monkeypatch, tmp_path)
    inputs = _stage_inputs("site0")
    paths["site0"][0].write_bytes(b"existing")
    with pytest.raises(RuntimeError, match="already exists"):
        runner.freeze_preselector_stage(
            "site0", *inputs, launch_state=object(),
        )
    paths["site0"][0].unlink()
    paths["site1"][0].write_bytes(b"downstream")
    with pytest.raises(RuntimeError, match="downstream"):
        runner.freeze_preselector_stage(
            "site0", *inputs, launch_state=object(),
        )


def test_selector_runs_only_from_reloaded_external_bank(monkeypatch) -> None:
    payload = {
        "candidate_ledgers": {
            "true_site0": {"external_true": {}},
            "shuffle_site0": {"external_shuffle": {}},
        }
    }
    monkeypatch.setattr(runner, "load_frozen_stage", lambda _: (payload, {}))
    seen = []

    def true_selector(bank):
        seen.append(bank)
        return {"selected": next(iter(bank))}

    def shuffle_selector(bank):
        seen.append(bank)
        return {"selected": next(iter(bank))}

    monkeypatch.setattr(
        runner.authority.selection, "freeze_validation_selection", true_selector,
    )
    monkeypatch.setattr(runner.authority, "_total_shuffle_selection", shuffle_selector)
    selected = runner.select_frozen_stage("site0")
    assert [next(iter(bank)) for bank in seen] == ["external_true", "external_shuffle"]
    assert selected["true_site0"]["selected"] == "external_true"
    assert selected["shuffle_site0"]["selected"] == "external_shuffle"


def test_stage_receipt_tamper_fails_closed(monkeypatch, tmp_path) -> None:
    paths = _install_stage_paths(monkeypatch, tmp_path)
    inputs = _stage_inputs("site0")
    runner.freeze_preselector_stage(
        "site0", *inputs, launch_state=object(),
    )
    receipt = json.loads(paths["site0"][1].read_text())
    receipt["artifact_bytes"] += 1
    paths["site0"][1].write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match="receipt binding"):
        runner.load_frozen_stage("site0")


def test_program_source_closure_contains_runner_and_test() -> None:
    names = {path.name for path in runner.authority.PROGRAM_SOURCE_CLOSURE}
    assert "early_mlp_state_complete_compiler_v21.py" in names
    assert "test_early_mlp_state_complete_compiler_v21.py" in names


def test_launch_pin_gate_checks_every_inherited_pin(monkeypatch, tmp_path) -> None:
    parent = tmp_path / "parent.json"
    parent.write_text("changed")
    monkeypatch.setattr(runner.authority, "PINS", {parent: "0" * 64})
    monkeypatch.setattr(runner.authority, "ORIGINAL_ABSENT", ())
    with pytest.raises(RuntimeError, match="pinned input changed"):
        runner._validate_all_pins_and_historical_absences()


def test_exclusive_run_claim_is_create_only_and_owned(monkeypatch, tmp_path) -> None:
    lock = tmp_path / "compiler.lock"
    monkeypatch.setattr(runner, "RUN_LOCK", lock)
    with runner.exclusive_run_claim() as nonce:
        runner._require_run_claim(nonce)
        with pytest.raises(RuntimeError, match="already claimed"):
            with runner.exclusive_run_claim():
                pass
    assert not lock.exists()


def test_selector_metrics_recompute_from_raw_sufficient_statistics(monkeypatch) -> None:
    count = runner.authority.VALIDATION_TOKEN_COUNT
    copy_count = 17
    denominator = 2.0
    baseline = 3.0
    kl_sum = float(count)
    global_sum = 4.0 * count
    copy_sum = 3.25 * copy_count
    raw = {
        "candidate_teacher_kl_sum": kl_sum,
        "candidate_teacher_kl_count": count,
        "global_ce_sum": global_sum,
        "global_ce_count": count,
        "copy_ce_sum": copy_sum,
        "copy_ce_count": copy_count,
    }
    price = {"total_reals": 1}
    metrics = {
        "candidate_teacher_kl": 1.0,
        "oracle_denominator_kl": denominator,
        "remaining_kl_ratio": 0.5,
        "recovery": 0.5,
        "global_ce": 4.0,
        "copy_ce": 3.25,
        "copy_count": copy_count,
        "copy_worsening": 0.25,
        "price": price,
        "raw_sufficient_statistics": raw,
    }
    context = {
        "teacher_denominator": denominator,
        "copy_baseline": baseline,
        "copy_token_count": copy_count,
    }
    bank = {"candidate": {"state": {}, "metrics": metrics}}
    monkeypatch.setattr(
        runner.authority.selection, "state_price", lambda _: price,
    )
    runner.authority._validate_candidate_sufficient_statistics(
        bank, context, "true_site0",
    )
    metrics["recovery"] = 0.5001
    with pytest.raises(RuntimeError, match="does not recompute"):
        runner.authority._validate_candidate_sufficient_statistics(
            bank, context, "true_site0",
        )


@pytest.mark.parametrize("kwargs", [
    {
        "outer_model_returned": False,
        "component_tree_before": "same", "component_tree_after": "same",
    },
    {
        "outer_model_returned": True,
        "component_tree_before": "before", "component_tree_after": "after",
    },
])
def test_execution_closure_requires_outer_return_and_component_inertness(
    monkeypatch, kwargs,
) -> None:
    with pytest.raises(RuntimeError, match="lifecycle gate"):
        runner.close_execution(object(), **kwargs)


def test_execution_closure_measures_correction_inertness(monkeypatch) -> None:
    monkeypatch.setattr(
        runner, "_require_inert_correction_state",
        lambda _: (_ for _ in ()).throw(RuntimeError("not inert")),
    )
    with pytest.raises(RuntimeError, match="not inert"):
        runner.close_execution(
            object(), outer_model_returned=True,
            component_tree_before="same", component_tree_after="same",
        )


def test_stage_drift_after_artifact_write_never_writes_receipt(monkeypatch, tmp_path) -> None:
    paths = _install_stage_paths(monkeypatch, tmp_path)
    inputs = _stage_inputs("site0")
    calls = 0

    def validate(_):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("source drift")

    monkeypatch.setattr(runner, "_validate_launch_state", validate)
    with pytest.raises(RuntimeError, match="source drift"):
        runner.freeze_preselector_stage(
            "site0", *inputs, launch_state=object(),
        )
    assert paths["site0"][0].is_file()
    assert not paths["site0"][1].exists()


def test_launch_token_detects_source_row_and_protected_drift(monkeypatch, tmp_path) -> None:
    receipt = tmp_path / "rows.json"
    manifest = tmp_path / "manifest.json"
    receipt.write_text("rows")
    manifest.write_text("manifest")
    monkeypatch.setattr(runner, "ROWS_RECEIPT", receipt)
    monkeypatch.setattr(runner.authority, "MANIFEST", manifest)
    monkeypatch.setattr(runner, "_source_identity", lambda: ("commit", {"a": "hash"}))
    monkeypatch.setattr(runner.authority, "protected_snapshot", lambda: {"pin": "ok"})
    monkeypatch.setattr(
        runner.authority, "_validate_historical_row_authority", lambda _: None,
    )
    monkeypatch.setattr(runner, "_require_run_claim", lambda _: None)
    state = runner.LaunchState(
        protected=(("pin", "ok"),), source_commit="commit",
        source_hashes=(("a", "hash"),),
        rows_receipt_sha256=runner.authority.file_sha256(receipt),
        rows_manifest_sha256=runner.authority.file_sha256(manifest),
    )
    receipt.write_text("drift")
    with pytest.raises(RuntimeError, match="identity drifted"):
        runner._validate_launch_state(state)


def test_resume_after_site0_requires_same_committed_source_closure(
    monkeypatch, tmp_path,
) -> None:
    paths = _install_stage_paths(monkeypatch, tmp_path)
    for path in paths["site0"]:
        path.write_text("site0")
    training = tmp_path / "training.json"
    training.write_text("training")
    site0_manifest = tmp_path / "site0_manifest.json"
    site0_manifest.write_text("site0 manifest")
    monkeypatch.setattr(runner, "SITE0_MANIFEST", site0_manifest)
    monkeypatch.setattr(runner, "SITE1_MANIFEST", tmp_path / "site1_manifest.json")
    monkeypatch.setattr(
        runner, "FINAL_OUTPUTS", tuple(tmp_path / f"final{i}" for i in range(4)),
    )
    programs = tmp_path / "programs.pt"
    final_receipt = tmp_path / "final.json"
    manifest = tmp_path / "rows_manifest.json"
    manifest.write_text("manifest")
    monkeypatch.setattr(
        runner.authority, "ROWS_RECEIPT_SHA256",
        runner.authority.file_sha256(runner.ROWS_RECEIPT),
    )
    monkeypatch.setattr(runner, "SITE0_TRAINING_RECEIPT", training)
    monkeypatch.setattr(runner, "PROGRAMS_ARTIFACT", programs)
    monkeypatch.setattr(runner, "PROGRAMS_RECEIPT", final_receipt)
    monkeypatch.setattr(runner.authority, "MANIFEST", manifest)
    monkeypatch.setattr(runner, "_require_run_claim", lambda _: None)
    monkeypatch.setattr(runner, "_source_identity", lambda: ("commit", {"a": "hash"}))
    monkeypatch.setattr(runner.authority, "protected_snapshot", lambda: {"pin": "ok"})
    monkeypatch.setattr(
        runner.authority, "_validate_historical_row_authority", lambda _: None,
    )
    monkeypatch.setattr(
        runner, "load_site0_training_authorization",
        lambda: {"source_commit": "commit", "source_hashes": {"a": "hash"}},
    )
    state = runner.resume_after_site0(lock_nonce="nonce")
    assert state.source_commit == "commit"
    monkeypatch.setattr(
        runner, "load_site0_training_authorization",
        lambda: {"source_commit": "older", "source_hashes": {"a": "hash"}},
    )
    with pytest.raises(RuntimeError, match="differs at resume"):
        runner.resume_after_site0(lock_nonce="nonce")


@pytest.mark.parametrize("candidate_valid", [False, True])
def test_final_receipt_is_prevalidated_and_written_once_last(
    monkeypatch, tmp_path, candidate_valid,
) -> None:
    artifact = tmp_path / "programs.pt"
    receipt = tmp_path / "unlock.json"
    runner.authority.write_torch_atomic({}, artifact)
    monkeypatch.setattr(runner, "PROGRAMS_ARTIFACT", artifact)
    monkeypatch.setattr(runner, "PROGRAMS_RECEIPT", receipt)
    monkeypatch.setattr(runner, "ROWS_RECEIPT", tmp_path / "rows.json")
    runner.ROWS_RECEIPT.write_text("rows")
    site1_manifest = tmp_path / "site1_manifest.json"
    site1_manifest.write_text("site1 manifest")
    monkeypatch.setattr(runner, "SITE1_MANIFEST", site1_manifest)
    monkeypatch.setattr(runner, "_validate_launch_state", lambda _: None)
    monkeypatch.setattr(runner.authority, "_validate_program_bundle", lambda _: None)
    writes = []
    original_write = runner.authority.write_json_atomic

    def recording_write(value, path):
        writes.append(path)
        original_write(value, path)

    monkeypatch.setattr(runner.authority, "write_json_atomic", recording_write)
    if candidate_valid:
        monkeypatch.setattr(runner, "_validate_receipt_candidate", lambda *_: None)
    else:
        monkeypatch.setattr(
            runner, "_validate_receipt_candidate",
            lambda *_: (_ for _ in ()).throw(RuntimeError("bad candidate")),
        )
    launch = runner.LaunchState(
        protected=(), source_commit="c" * 40, source_hashes=(),
        rows_receipt_sha256="0" * 64, rows_manifest_sha256="1" * 64,
    )
    closure = runner.ExecutionClosure(True, True, "same", "same")
    if not candidate_valid:
        with pytest.raises(RuntimeError, match="bad candidate"):
            runner.write_final_unlock_after_outer_return(
                launch_state=launch, execution_closure=closure,
            )
        assert not receipt.exists() and not writes
    else:
        result = runner.write_final_unlock_after_outer_return(
            launch_state=launch, execution_closure=closure,
        )
        assert result["authorized_for_final_scoring"] is True
        assert writes == [receipt]
        assert receipt.is_file()


def test_main_is_inert_without_a_numerical_stage() -> None:
    source = runner.Path(runner.__file__).read_text()
    assert "if __name__ == \"__main__\"" in source
    assert "does not start a model forward by itself" in source
