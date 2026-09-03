"""Independent CPU-only final attacks for immutable R585 commit 1143aab7c."""

# BQLANE: cpu

from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve()
OPS = SCRIPT.parent
ROOT = OPS.parent
REPO = ROOT.parent.parent
PRODUCER = OPS / "induction_selector_payload_frozen_factor_rung585.py"
OWNER_TEST = OPS / "test_induction_selector_payload_frozen_factor_rung585.py"
DRYRUN = ROOT / "induction_selector_payload_frozen_factor_rung585_dryrun.json"
ADAPTER = OPS / "execute_induction_selector_payload_frozen_factor_rung585.py"
ADAPTER_TEST = OPS / "test_execute_induction_selector_payload_frozen_factor_rung585.py"

CANDIDATE_COMMIT = "1143aab7c444f32ff1f3fc59942c61ff652cb7d2"
CANDIDATE_HASHES = {
    PRODUCER: "a3987dc053ba9b18a92a950c526acb1127f2cec9ee97d1142158ca4ef6483ddd",
    OWNER_TEST: "5365d3d473f3385d3b052f7ff09af78f8e2209a0d3e6a75eca264beaf082c11f",
    DRYRUN: "de33550e530c35c1236095e2354d3724c7ff70de16242424f17d5ed7a81433a6",
    ADAPTER: "064b6bf1abdde4d196c43fbae0d589949778dea7ef0f167bebb00e26f5274e21",
    ADAPTER_TEST: "95d60bbca55f3dfd01fe9be92a26ad60e1b174a6006a68ab278cacb4f2a6e542",
}
FAILURE_EVIDENCE_BLOCKER = pytest.mark.xfail(
    strict=True,
    reason="1143aab7c does not reconstruct invalid-instrument failure clauses",
)
REGISTERED_PREDICATES = {
    "pred_a_saved_semantics_reconstruct":
        "endpoint direction factor intervention and primitive evidence reconstruct",
    "pred_b_scores_reconstruct":
        "FIT scales score reports bootstraps and scientific failures reconstruct",
    "pred_c_invalid_failures_reconstruct":
        "instrument and structural failure clauses reconstruct from raw evidence",
}


def _git_blob(path: Path) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{CANDIDATE_COMMIT}:{path.relative_to(REPO)}"], cwd=REPO
    )


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    producer_fd, producer_name = tempfile.mkstemp(
        prefix=".r585-final-review-producer-", suffix=".py", dir=OPS
    )
    os.close(producer_fd)
    producer_path = Path(producer_name)
    owner_fd, owner_name = tempfile.mkstemp(
        prefix=".r585-final-review-owner-", suffix=".py", dir=OPS
    )
    os.close(owner_fd)
    owner_path = Path(owner_name)
    producer_path.write_bytes(_git_blob(PRODUCER))
    owner_path.write_bytes(_git_blob(OWNER_TEST))
    module = _load(producer_path, "r585_final_review_producer")
    module.SCRIPT = producer_path
    module.TEST = owner_path
    try:
        yield module
    finally:
        producer_path.unlink(missing_ok=True)
        owner_path.unlink(missing_ok=True)


@pytest.fixture(scope="module")
def adapter():
    adapter_fd, adapter_name = tempfile.mkstemp(
        prefix=".r585-final-review-adapter-", suffix=".py", dir=OPS
    )
    os.close(adapter_fd)
    adapter_path = Path(adapter_name)
    adapter_path.write_bytes(_git_blob(ADAPTER))
    module = _load(adapter_path, "r585_final_review_adapter")
    try:
        yield module
    finally:
        adapter_path.unlink(missing_ok=True)


def test_exact_candidate_commit_and_hashes():
    assert subprocess.check_output(
        ["git", "rev-parse", CANDIDATE_COMMIT], cwd=REPO, text=True
    ).strip() == CANDIDATE_COMMIT
    for path, expected in CANDIDATE_HASHES.items():
        assert hashlib.sha256(_git_blob(path)).hexdigest() == expected


def test_phase_terminal_price_and_mutation_envelope(runner):
    fit = runner.phase_evidence_contract(["FIT"])
    full = runner.phase_evidence_contract(["FIT", "SELECT"])
    assert (fit["endpoint_count"], fit["directed_arm_record_count"],
            fit["factor_exactness_count"]) == (1_728, 11_232, 6_912)
    assert (full["endpoint_count"], full["directed_arm_record_count"],
            full["factor_exactness_count"]) == (2_592, 16_848, 10_368)
    assert runner.EXPECTED_OPERATION_COUNTS == {"FIT": 13_824, "SELECT": 6_912}
    assert runner.EXPECTED_PHASE_PRICE == {"FIT": 459, "SELECT": 231}
    assert runner.EXPECTED_TOTAL_PRICE == 690
    assert runner.FORBIDDEN_SPLITS == ("FINAL_TEST", "OOD")
    for terminal in runner.TERMINALS:
        fixture = runner.make_result_fixture(terminal)
        assert type(fixture["next_step"]) is str
        assert fixture["model_backwards"] == 0
        assert fixture["model_weights_updated"] is False
        with pytest.raises(ValueError, match="model-free fixture"):
            runner.validate_result(fixture)


def test_every_terminal_has_exact_scored_split_phase(runner, monkeypatch):
    monkeypatch.setattr(runner, "_validate_complete_evidence", lambda result, resolver: None)
    monkeypatch.setattr(runner, "build_execution_authority", lambda: {"manifests": {}})
    monkeypatch.setattr(
        runner, "validate_realized_bootstraps",
        lambda report, split, manifests: report["bootstrap_realization"],
    )
    for terminal in sorted(runner.TERMINALS):
        result = runner.make_result_fixture(terminal)
        result["raw_evidence"] = {}
        scored = runner.scored_splits_for_terminal(terminal, result["evaluated_splits"])
        result["split_scores"] = {
            split: {
                "bootstrap_realization": {
                    "count": 124, "cell_ids_sha256": split.lower() + "-test"
                }
            }
            for split in scored
        }
        runner.validate_result(result)
        wrong = copy.deepcopy(result)
        wrong["split_scores"]["unexpected"] = {
            "bootstrap_realization": {"count": 124, "cell_ids_sha256": "wrong"}
        }
        with pytest.raises(ValueError, match="split-score phase census"):
            runner.validate_result(wrong)


def test_endpoint_and_direction_authority_semantics_are_exact(runner):
    execution = runner.build_execution_authority()
    endpoint = copy.deepcopy(execution["endpoints"][0])
    runner._validate_endpoint_semantics([endpoint], {"endpoints": [endpoint]})
    for field in (
        "split", "endpoint_id", "token_ids", "length", "final_position",
        "source_positions", "payload_positions", "condition", "answer_id",
        "other_answer_id",
    ):
        changed = copy.deepcopy(endpoint)
        changed[field] = "changed" if field not in ("token_ids", "source_positions", "payload_positions") else [99]
        with pytest.raises(ValueError, match="endpoint semantic"):
            runner._validate_endpoint_semantics([changed], {"endpoints": [endpoint]})

    direction = copy.deepcopy(execution["directions"][0])
    rows = [{
        **direction, "arm": arm,
        "other_answer_id": direction["recipient_other_answer_id"],
    } for arm in runner.ARMS]
    authority = runner._validate_direction_semantics(
        rows, {"directions": [direction]}
    )
    assert authority[direction["directed_id"]] == direction
    for field in (
        "recipient_endpoint_id", "donor_endpoint_id", "row_id", "group_id",
        "family", "variant", "recipient_condition", "direction",
        "control_kind", "answer_changes", "recipient_answer_id",
        "donor_answer_id", "other_answer_id",
    ):
        changed = copy.deepcopy(rows)
        changed[0][field] = "changed"
        with pytest.raises(ValueError, match="recipient/donor|direction metadata"):
            runner._validate_direction_semantics(changed, {"directions": [direction]})

    endpoint_by_id = {row["endpoint_id"]: row for row in execution["endpoints"]}
    for row in execution["directions"]:
        recipient = endpoint_by_id[row["recipient_endpoint_id"]]
        donor = endpoint_by_id[row["donor_endpoint_id"]]
        assert row["donor_answer_id"] in {
            recipient["answer_id"], recipient["other_answer_id"]
        }
        assert row["recipient_answer_id"] in {
            donor["answer_id"], donor["other_answer_id"]
        }


def _factor_arrays():
    e = np.asarray([[
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        [[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0]],
    ]], dtype="<f4").reshape(2, 4, 2)
    u = np.arange(2 * 4 * 2 * 3, dtype=np.float32).reshape(2, 4, 2, 3) / 10 + 1
    canonical = np.sum(e[:, :, :, None] * u, axis=2).astype("<f4")
    remainder = np.ones_like(canonical)
    return {
        "native_e.npy": e,
        "native_u.npy": u.astype("<f4"),
        "canonical_term.npy": canonical,
        "native_head_output.npy": canonical + remainder,
        "non_equality_remainder.npy": remainder,
    }


def test_factor_rows_and_arm_specific_saved_vectors_reconstruct(runner):
    arrays = _factor_arrays()
    endpoint_order = ["recipient", "donor"]
    endpoints = {
        "recipient": {"split": "FIT", "endpoint_id": "recipient"},
        "donor": {"split": "FIT", "endpoint_id": "donor"},
    }
    factor_rows = [{
        "split": "FIT", "endpoint_id": endpoint, "site": site,
        "equality_factor_max_abs": 0.0,
        "equality_plus_independent_remainder_max_abs": 0.0,
    } for endpoint in endpoint_order for site in runner.TERM_NAMES]
    runner._validate_factor_row_semantics(factor_rows, endpoints)
    runner._validate_factor_exactness_rows(factor_rows, arrays, endpoint_order)
    bad_factor = copy.deepcopy(factor_rows)
    bad_factor[0]["equality_factor_max_abs"] = 0.25
    with pytest.raises(ValueError, match="factor exactness"):
        runner._validate_factor_exactness_rows(bad_factor, arrays, endpoint_order)

    direction = {
        "directed_id": "d", "recipient_endpoint_id": "recipient",
        "donor_endpoint_id": "donor",
    }
    rows = []
    lives = []
    deltas = []
    e, u = arrays["native_e.npy"], arrays["native_u.npy"]
    for arm in runner.ARMS:
        score = 1 if arm in ("score", "joint") else 0
        payload = 1 if arm in ("payload", "joint") else 0
        inserted = e[score, :, 0, None] * u[payload, :, 0] + e[
            score, :, 1, None
        ] * u[payload, :, 1]
        live = arrays["canonical_term.npy"][0]
        delta = (inserted - live).astype("<f4")
        norms = np.linalg.norm(delta, axis=1)
        rows.append({
            **direction, "arm": arm,
            "per_site_delta_norms": [float(value) for value in norms],
            "insertion_activity": float(np.median(norms)),
        })
        lives.append(live)
        deltas.append(delta)
    arrays["live_removed.npy"] = np.asarray(lives, dtype="<f4")
    arrays["hook_delta.npy"] = np.asarray(deltas, dtype="<f4")
    runner._validate_saved_factor_interventions(rows, arrays, endpoint_order)
    for mutation in ("delta", "norm", "activity"):
        changed_arrays = {key: value.copy() for key, value in arrays.items()}
        changed_rows = copy.deepcopy(rows)
        if mutation == "delta":
            changed_arrays["hook_delta.npy"][0, 0, 0] += 1
        elif mutation == "norm":
            changed_rows[0]["per_site_delta_norms"][0] += 1
        else:
            changed_rows[0]["insertion_activity"] += 1
        with pytest.raises(ValueError, match="live plus hook_delta|delta norm|insertion activity"):
            runner._validate_saved_factor_interventions(
                changed_rows, changed_arrays, endpoint_order
            )


def _measurement(answer, other, answer_logit, other_logit, log_normalizer):
    return {
        "answer_id": answer, "other_answer_id": other,
        "answer_logit": answer_logit, "other_logit": other_logit,
        "correct_margin": answer_logit - other_logit,
        "log_normalizer": log_normalizer,
        "correct_ce": log_normalizer - answer_logit,
    }


def _sufficient_statistics_fixture():
    semantic = {
        "split": "FIT", "token_ids": [1, 2, 3, 4, 5], "length": 5,
        "final_position": 4, "source_positions": [0, 2],
        "payload_positions": [1, 3], "condition": "s0p0",
        "answer_id": 3, "other_answer_id": 4,
    }
    replay = _measurement(3, 4, 2.0, 1.0, 3.0)
    recipient_native = _measurement(3, 4, 2.0, 1.0, 3.0)
    donor_native = _measurement(3, 4, 3.0, 1.0, 4.0)
    endpoints = [
        {**semantic, "endpoint_id": "recipient", "replay": replay, "native": recipient_native},
        {**semantic, "endpoint_id": "donor", "replay": donor_native, "native": donor_native},
    ]
    direction = {
        "split": "FIT", "directed_id": "direction", "row_id": "row",
        "group_id": "group", "family": "family", "variant": "variant",
        "recipient_condition": "s0p0", "direction": "base_to_donor",
        "control_kind": None, "answer_changes": False,
        "recipient_endpoint_id": "recipient", "donor_endpoint_id": "donor",
        "recipient_answer_id": 3, "donor_answer_id": 3,
        "recipient_other_answer_id": 4, "donor_other_answer_id": 4,
        "donor_coherence_sign": 1,
    }
    intervention = _measurement(3, 4, 2.5, 1.0, 3.5)
    record = {
        **{key: direction[key] for key in (
            "split", "directed_id", "row_id", "group_id", "family", "variant",
            "recipient_condition", "direction", "control_kind", "answer_changes",
            "recipient_endpoint_id", "donor_endpoint_id", "recipient_answer_id",
            "donor_answer_id",
        )},
        "arm": "score", "other_answer_id": 4,
        "replay_correct_margin": replay["correct_margin"],
        "correct_margin": intervention["correct_margin"],
        "replay_correct_ce": replay["correct_ce"],
        "correct_ce": intervention["correct_ce"],
        "n": 0.5, "d": 1.0, "q": 0.0,
        "insertion_activity": 1.0, "per_site_delta_norms": [1.0] * 4,
        "live_factor_max_error": 0.0, "hook_delta_sum_max_error": 0.0,
        "vocab_squared_difference_sum": 3.0, "vocab_size": 3,
        "vocab_rms": 1.0, "answer_logit": intervention["answer_logit"],
        "other_logit": intervention["other_logit"],
        "log_normalizer": intervention["log_normalizer"],
    }
    return endpoints, direction, record


def test_endpoint_measurements_primitives_and_n_d_q_reconstruct(runner):
    endpoints, direction, record = _sufficient_statistics_fixture()
    runner._validate_saved_primitive_logit_identities([record])
    runner._validate_saved_sufficient_statistics(
        endpoints, [record], {direction["directed_id"]: direction}
    )
    for field in ("correct_margin", "correct_ce", "vocab_rms"):
        changed = copy.deepcopy(record)
        changed[field] += 0.5
        with pytest.raises(ValueError, match="primitive logit"):
            runner._validate_saved_primitive_logit_identities([changed])
    for field in ("n", "d", "q"):
        changed = copy.deepcopy(record)
        changed[field] += 0.5
        with pytest.raises(ValueError, match=f"saved {field}"):
            runner._validate_saved_sufficient_statistics(
                endpoints, [changed], {direction["directed_id"]: direction}
            )
    changed_endpoints = copy.deepcopy(endpoints)
    changed_endpoints[0]["replay"]["correct_margin"] += 0.5
    with pytest.raises(ValueError, match="endpoint replay margin"):
        runner._validate_saved_sufficient_statistics(
            changed_endpoints, [record], {direction["directed_id"]: direction}
        )


def test_fit_scales_report_bootstraps_and_scientific_failures_reconstruct(
    runner, monkeypatch
):
    execution = runner.build_execution_authority()
    rows = runner.planted_intervention_records(execution, null=True)
    fit_rows = [row for row in rows if row["split"] == "FIT"]
    scales = runner.compute_fit_scales(fit_rows, execution["manifests"])
    report, failures = runner.score_split(
        fit_rows, "FIT", execution["manifests"], scales, replicates=16
    )
    monkeypatch.setattr(runner, "BOOTSTRAPS", 16)
    result = runner.make_result_fixture("factor_capacity_null")
    result["raw_evidence"] = {"fit_scales": scales}
    result["split_scores"] = {"FIT": report}
    for label, clauses in failures.items():
        result["failure_classes"][label] = clauses
    runner._validate_saved_score_reports(result, fit_rows, execution)

    wrong_scale = copy.deepcopy(result)
    first_scale = next(iter(wrong_scale["raw_evidence"]["fit_scales"]))
    wrong_scale["raw_evidence"]["fit_scales"][first_scale] = {"changed": True}
    with pytest.raises(ValueError, match="FIT scales"):
        runner._validate_saved_score_reports(wrong_scale, fit_rows, execution)

    wrong_report = copy.deepcopy(result)
    wrong_report["split_scores"]["FIT"]["bootstrap_realization"]["count"] -= 1
    with pytest.raises(ValueError, match="score report"):
        runner._validate_saved_score_reports(wrong_report, fit_rows, execution)

    wrong_failures = copy.deepcopy(result)
    wrong_failures["failure_classes"]["factor_capacity_null"] = ["correlated-summary-swap"]
    with pytest.raises(ValueError, match="failure clauses"):
        runner._validate_saved_score_reports(wrong_failures, fit_rows, execution)

    correlated_rows = copy.deepcopy(fit_rows)
    correlated_rows[0]["n"] += 0.25
    correlated_report, correlated_failures = runner.score_split(
        correlated_rows, "FIT", execution["manifests"], scales, replicates=16
    )
    correlated = copy.deepcopy(result)
    correlated["split_scores"] = {"FIT": correlated_report}
    for label, clauses in correlated_failures.items():
        correlated["failure_classes"][label] = clauses
    runner._validate_saved_score_reports(correlated, correlated_rows, execution)
    assert correlated["split_scores"] != result["split_scores"]


def test_managed_recovery_order_remains_conservative(runner, adapter, tmp_path, monkeypatch):
    source = inspect.getsource(adapter.preflight)
    assert source.index("recover_publication_preflight") < source.index(
        "require_unused_namespaces"
    )
    monkeypatch.setattr(adapter, "verify_frozen_bytes", lambda: {})
    out, receipt, evidence = (
        tmp_path / "result.json", tmp_path / "receipt.json", tmp_path / "evidence"
    )
    stage = runner.create_stage_root(tmp_path)
    partial = runner.make_result_fixture("invalid_instrument")
    out.write_text(json.dumps(partial, sort_keys=True, allow_nan=False))

    def recover():
        runner.recover_stale_publication(
            root=tmp_path, out=out, receipt=receipt, evidence=evidence
        )

    with pytest.raises(RuntimeError, match="recovered incomplete"):
        adapter.dispatch(
            {"BQLIB_DRYRUN": "1"}, recovery_function=recover,
            namespace_paths=(out, receipt, evidence),
        )
    assert not stage.exists() and not out.exists()
    assert len(list(tmp_path.glob(runner.RECOVERY_PREFIX + "*"))) == 1


def _write_jsonl(path, rows):
    path.write_text("".join(
        json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows
    ))


def _complete_invalid_result(runner, tmp_path, monkeypatch):
    endpoint_rows, direction, seed_record = _sufficient_statistics_fixture()
    endpoint_rows[0]["endpoint_id"] = "a-recipient"
    endpoint_rows[1]["endpoint_id"] = "b-donor"
    direction["recipient_endpoint_id"] = "a-recipient"
    direction["donor_endpoint_id"] = "b-donor"
    seed_record["recipient_endpoint_id"] = "a-recipient"
    seed_record["donor_endpoint_id"] = "b-donor"
    endpoint_authority = [{
        key: row[key] for key in (
            "split", "endpoint_id", "token_ids", "length", "final_position",
            "source_positions", "payload_positions", "condition", "answer_id",
            "other_answer_id",
        )
    } for row in endpoint_rows]
    execution = {
        "endpoints": endpoint_authority, "directions": [direction], "manifests": {},
    }
    operations = sorted([
        {
            "split": "FIT", "endpoint_id": endpoint["endpoint_id"],
            "site": site, "role": role,
        }
        for endpoint in endpoint_authority
        for site in runner.TERM_NAMES for role in runner.ROLES
    ], key=lambda row: (row["split"], row["endpoint_id"], row["site"], row["role"]))
    arrays = _factor_arrays()
    endpoint_order = [row["endpoint_id"] for row in endpoint_rows]
    e, u = arrays["native_e.npy"], arrays["native_u.npy"]
    live_rows, delta_rows, directed_rows = [], [], []
    for arm in sorted(runner.ARMS):
        score = 1 if arm in ("score", "joint") else 0
        payload = 1 if arm in ("payload", "joint") else 0
        inserted = e[score, :, 0, None] * u[payload, :, 0] + e[
            score, :, 1, None
        ] * u[payload, :, 1]
        live = arrays["canonical_term.npy"][0]
        delta = (inserted - live).astype("<f4")
        norms = np.linalg.norm(delta, axis=1)
        row = copy.deepcopy(seed_record)
        row["arm"] = arm
        row["per_site_delta_norms"] = [float(value) for value in norms]
        row["insertion_activity"] = float(np.median(norms))
        directed_rows.append(row)
        live_rows.append(live)
        delta_rows.append(delta)
    arrays["live_removed.npy"] = np.asarray(live_rows, dtype="<f4")
    arrays["hook_delta.npy"] = np.asarray(delta_rows, dtype="<f4")
    factor_rows = [{
        "split": "FIT", "endpoint_id": endpoint,
        "site": site, "equality_factor_max_abs": 0.0,
        "equality_plus_independent_remainder_max_abs": 0.0,
    } for endpoint in endpoint_order for site in sorted(runner.TERM_NAMES)]
    directed_order = [[direction["directed_id"], arm] for arm in sorted(runner.ARMS)]
    factor_order = [
        [endpoint, site] for endpoint in endpoint_order for site in sorted(runner.TERM_NAMES)
    ]
    contract = {
        "evaluated_splits": ["FIT"], "endpoint_count": 2,
        "direction_count": 1, "directed_arm_record_count": 3,
        "factor_exactness_count": 8,
        "array_shapes": {name: list(array.shape) for name, array in arrays.items()},
        "jsonl_counts": {
            "endpoint_measurements.jsonl": 2,
            "directed_arm_measurements.jsonl": 3,
            "factor_exactness.jsonl": 8,
        },
    }
    monkeypatch.setattr(runner, "phase_evidence_contract", lambda splits: contract)
    monkeypatch.setattr(runner, "build_execution_authority", lambda: execution)
    monkeypatch.setattr(
        runner, "build_endpoint_site_role_operations", lambda endpoints: operations
    )
    monkeypatch.setattr(runner, "EXPECTED_OPERATION_COUNTS", {"FIT": 16})

    descriptors = []
    for name, array in arrays.items():
        path = tmp_path / name
        np.save(path, array, allow_pickle=False)
        order = directed_order if name in ("live_removed.npy", "hook_delta.npy") else endpoint_order
        descriptors.append({
            "path": str((runner.EVIDENCE_DIR / name).relative_to(runner.ROOT.parent.parent)),
            "sha256": runner.sha256(path), "bytes": path.stat().st_size,
            "dtype": "<f4", "shape": list(array.shape),
            "row_order_sha256": runner.content_sha256(order),
        })
    for name, rows, order in (
        ("endpoint_measurements.jsonl", endpoint_rows, endpoint_order),
        ("directed_arm_measurements.jsonl", directed_rows, directed_order),
        ("factor_exactness.jsonl", factor_rows, factor_order),
    ):
        path = tmp_path / name
        _write_jsonl(path, rows)
        descriptors.append({
            "path": str((runner.EVIDENCE_DIR / name).relative_to(runner.ROOT.parent.parent)),
            "sha256": runner.sha256(path), "bytes": path.stat().st_size,
            "dtype": "jsonl", "shape": [len(rows)],
            "row_order_sha256": runner.content_sha256(order),
        })
    result = runner.make_result_fixture("invalid_instrument")
    result["failure_classes"]["invalid_instrument"] = [
        "invented-unrelated-integrity-failure"
    ]
    result["failed_clauses"] = ["invented-unrelated-integrity-failure"]
    result["split_scores"] = {}
    result["raw_evidence"] = {
        "schema": runner.EVIDENCE_SCHEMA,
        "endpoint_count": 2, "directed_arm_record_count": 3,
        "endpoint_site_role_operation_counts": {"FIT": 16},
        "endpoint_site_role_operation_sha256": runner.content_sha256(operations),
        "realized_endpoint_site_role_operations": {
            "FIT": {"count": 16, "sha256": runner.content_sha256(operations)}
        },
        "instrument_maxima": {
            "native_attention_reconstruction_max_abs": 0.0,
            "equality_factor_max_abs": 0.0,
            "equality_plus_independent_remainder_max_abs": 0.0,
            "replay_native_logit_max_abs": 0.0,
            "padding_tripwire_active_lengths": [],
        },
        "structural_identity_checks": [], "fit_scales": {},
    }
    result["evidence_files"] = descriptors
    return result, lambda logical: tmp_path / logical.name


@FAILURE_EVIDENCE_BLOCKER
def test_full_completed_invalid_result_rejects_unexplained_failure_clause(
    runner, tmp_path, monkeypatch
):
    result, resolver = _complete_invalid_result(runner, tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="instrument|failure|structural"):
        runner.validate_result(result, artifact_path_resolver=resolver)


@FAILURE_EVIDENCE_BLOCKER
@pytest.mark.parametrize("terminal", ["invalid_instrument", "select_invalid_instrument"])
def test_invalid_instrument_failure_clauses_are_reconstructed(
    runner, terminal, monkeypatch
):
    """Arbitrary integrity-failure text must not license an invalid terminal."""
    result = runner.make_result_fixture(terminal)
    result["raw_evidence"] = {"fit_scales": {}}
    result["failure_classes"][terminal] = ["invented-unrelated-integrity-failure"]
    result["failed_clauses"] = ["invented-unrelated-integrity-failure"]
    if terminal == "select_invalid_instrument":
        # Isolate SELECT's invalid list; FIT has already scored.  The producer
        # currently validates only the FIT report and never rebuilds this list.
        result["raw_evidence"]["fit_scales"] = {"frozen": True}
        result["split_scores"] = {"FIT": {"frozen": True}}

        def accept_fit(records, split, manifests, scales, replicates):
            return {"frozen": True}, {
                label: [] for label in (
                    "invalid_instrument", "native_denominator_or_scale_null",
                    "factor_capacity_null", "factorization_not_identified",
                    "insufficient_active_controls", "broad_contextual_equality_write",
                )
            }

        monkeypatch.setattr(
            runner, "compute_fit_scales", lambda records, manifests: {"frozen": True}
        )
        monkeypatch.setattr(runner, "score_split", accept_fit)
        with pytest.raises(ValueError, match="instrument|failure"):
            runner._validate_saved_score_reports(
                result, [{"split": "FIT"}], {"manifests": {}}
            )
    else:
        with pytest.raises(ValueError, match="instrument|failure"):
            runner._validate_saved_score_reports(result, [], {"manifests": {}})
