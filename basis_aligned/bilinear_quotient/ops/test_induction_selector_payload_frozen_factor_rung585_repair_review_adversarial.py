"""Independent CPU-only acceptance attacks for repaired R585 commit 27e4beaaf.

Strict xfails are prospective blockers found by this review.  The module never
loads the model, opens CUDA, or reads an R585 result namespace.
"""

# BQLANE: cpu

from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
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

CANDIDATE_COMMIT = "27e4beaaf94704031cc07fabbcdd8888aa990f46"
CANDIDATE_HASHES = {
    PRODUCER: "dcdb6470e481dcbc58e86997f4a4d0e3203607ae29a0b74b0e58f59abf62db58",
    OWNER_TEST: "cf4326ba6500814767e4b5ee17952753cbda39d6368e91c45a47a1ddce10cc63",
    DRYRUN: "a30d8206b11beb691e2b9dd2ce33a3a3c2df6752388643f13f0fc81442c69118",
    ADAPTER: "e96c72a83f199f84896ab17e3ce5e9aa9d01c8ec973e319cf4039a0866bcb301",
    ADAPTER_TEST: "8fc015ec973c7159c59231fd52f567ff6d11c15322433d4f5ff3e4bdf3dbaf60",
}
BLOCKED_REPAIR = pytest.mark.xfail(
    strict=True,
    reason="exact 27e4beaaf execution candidate does not fail closed",
)
REGISTERED_PREDICATES = {
    "pred_a_previous_six_blockers_closed":
        "the literal six frozen repair contracts hold on the exact candidate bytes",
    "pred_b_all_completed_terminals_are_auditable":
        "every completed scientific terminal requires authority-bound primitive evidence",
    "pred_c_managed_crash_recovery_is_reachable":
        "the managed no-argument path can quarantine partial final publication",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_blob(path: Path) -> bytes:
    relative = str(path.relative_to(REPO))
    return subprocess.check_output(
        ["git", "show", f"{CANDIDATE_COMMIT}:{relative}"], cwd=REPO
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
    descriptor, target_name = tempfile.mkstemp(
        prefix=".r585-review-producer-", suffix=".py", dir=OPS
    )
    os.close(descriptor)
    target = Path(target_name)
    owner_descriptor, owner_name = tempfile.mkstemp(
        prefix=".r585-review-owner-", suffix=".py", dir=OPS
    )
    os.close(owner_descriptor)
    owner = Path(owner_name)
    target.write_bytes(_git_blob(PRODUCER))
    owner.write_bytes(_git_blob(OWNER_TEST))
    module = _load(target, "r585_repair_review_producer")
    module.SCRIPT = target
    module.TEST = owner
    try:
        yield module
    finally:
        target.unlink(missing_ok=True)
        owner.unlink(missing_ok=True)


@pytest.fixture(scope="module")
def adapter():
    descriptor, target_name = tempfile.mkstemp(
        prefix=".r585-review-adapter-", suffix=".py", dir=OPS
    )
    os.close(descriptor)
    target = Path(target_name)
    target.write_bytes(_git_blob(ADAPTER))
    module = _load(target, "r585_repair_review_adapter")
    try:
        yield module
    finally:
        target.unlink(missing_ok=True)


@pytest.fixture(scope="module")
def execution(runner):
    return runner.build_execution_authority()


def test_exact_candidate_commit_and_bytes():
    assert subprocess.check_output(
        ["git", "rev-parse", CANDIDATE_COMMIT], cwd=REPO, text=True
    ).strip() == CANDIDATE_COMMIT
    assert {
        hashlib.sha256(_git_blob(path)).hexdigest() for path in CANDIDATE_HASHES
    } == set(CANDIDATE_HASHES.values())
    assert all(
        hashlib.sha256(_git_blob(path)).hexdigest() == digest
        for path, digest in CANDIDATE_HASHES.items()
    )


def test_literal_six_repairs_are_present(runner, execution):
    factor_source = inspect.getsource(runner.factorize_attention_event)
    assert "contract_without_induction_fetch" in factor_source
    assert "remainder = head_output - canonical_term" not in factor_source

    operations = execution["endpoint_site_role_operations"]
    assert len(operations) == 20_736
    assert runner.content_sha256(operations) == runner.EXPECTED_OPERATION_SHA256
    assert {split: sum(row["split"] == split for row in operations)
            for split in runner.SPLITS} == {"FIT": 13_824, "SELECT": 6_912}
    with pytest.raises(RuntimeError, match="operation census"):
        runner.validate_realized_operations(
            operations, [row for row in operations if row["split"] == "FIT"][:-1], "FIT"
        )

    planted = runner.planted_intervention_records(execution)
    scales = runner.compute_fit_scales(planted, execution["manifests"])
    report, _ = runner.score_split(
        planted, "FIT", execution["manifests"], scales, replicates=2
    )
    assert report["bootstrap_realization"] == {
        "count": 124,
        "cell_ids_sha256": runner.content_sha256([
            row["cell_id"] for row in execution["bootstrap_cells"]
            if row["cell_id"].startswith("FIT|")
        ]),
    }
    incomplete = copy.deepcopy(execution["manifests"])
    incomplete["target_cells"].remove(next(
        row for row in incomplete["target_cells"] if row["split"] == "FIT"
    ))
    with pytest.raises(RuntimeError, match="bootstrap.*census|census.*bootstrap"):
        runner.score_split(planted, "FIT", incomplete, scales, replicates=2)

    held = runner.make_result_fixture("held_operational_selector_payload_factorization")
    with pytest.raises(ValueError, match="raw evidence"):
        runner.validate_result(held)
    null = runner.make_result_fixture("factor_capacity_null")
    wrong_checkpoint = copy.deepcopy(null)
    wrong_checkpoint["checkpoint_weights_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checkpoint"):
        runner.validate_result(wrong_checkpoint)
    receipt = runner.make_receipt_fixture(null)
    receipt["result_path"] = "basis_aligned/bilinear_quotient/not-r585.json"
    with pytest.raises(ValueError, match="result_path"):
        runner.validate_receipt(receipt, null)

    primitive = {
        "directed_id": "planted", "arm": "score",
        "answer_logit": float("nan"), "other_logit": 1.0,
        "correct_margin": 0.0, "log_normalizer": 2.0, "correct_ce": 1.0,
        "vocab_squared_difference_sum": 1.0, "vocab_size": 1,
        "vocab_rms": 1.0, "live_factor_max_error": 0.0,
        "hook_delta_sum_max_error": 0.0,
    }
    assert runner.validate_primitive_logit_identities([primitive])[0].startswith(
        "nonfinite_primitive:"
    )


def test_price_split_and_mutation_envelope_is_exact(runner):
    dryrun = json.loads(_git_blob(DRYRUN))
    assert dryrun["price"] == {
        "FIT": 459, "SELECT": 231, "maximum": 690,
        "backwards": 0, "updates": 0,
    }
    assert dryrun["canonical_names"]["splits"] == ["FIT", "SELECT"]
    assert dryrun["canonical_names"]["forbidden_splits"] == ["FINAL_TEST", "OOD"]
    assert dryrun["model_loaded"] is dryrun["cuda_opened"] is False
    assert dryrun["outcomes_opened"] == []
    assert runner.EXPECTED_PHASE_PRICE == {"FIT": 459, "SELECT": 231}
    assert runner.EXPECTED_TOTAL_PRICE == 690


def test_direct_producer_recovery_quarantines_partial_publication(runner, tmp_path):
    out = tmp_path / "result.json"
    receipt = tmp_path / "receipt.json"
    evidence = tmp_path / "evidence"
    out.write_text("partial-result\n")
    stage = runner.create_stage_root(tmp_path)
    (stage / "evidence").mkdir()
    with pytest.raises(RuntimeError, match="recovered incomplete"):
        runner.recover_stale_publication(
            root=tmp_path, out=out, receipt=receipt, evidence=evidence
        )
    assert not out.exists() and not receipt.exists() and not evidence.exists()
    assert len(list(tmp_path.glob(runner.RECOVERY_PREFIX + "*"))) == 1
    assert runner.recover_stale_publication(
        root=tmp_path, out=out, receipt=receipt, evidence=evidence
    ) is None


@BLOCKED_REPAIR
def test_completed_scientific_null_cannot_drop_all_evidence(runner):
    result = runner.make_result_fixture("factor_capacity_null")
    assert result["model_forwards"] == 459
    assert result["evaluated_splits"] == ["FIT"]
    assert result["evidence_files"] == []
    with pytest.raises(ValueError, match="evidence"):
        runner.validate_result(result)


def _write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(
        json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows
    ))


def _descriptor(runner, path: Path, logical: Path, row_order, dtype, shape):
    return {
        "path": str(logical.relative_to(runner.ROOT.parent.parent)),
        "sha256": runner.sha256(path),
        "bytes": path.stat().st_size,
        "dtype": dtype,
        "shape": shape,
        "row_order_sha256": runner.content_sha256(row_order),
    }


@BLOCKED_REPAIR
def test_held_evidence_membership_must_equal_frozen_authority(
    runner, tmp_path, monkeypatch
):
    """Self-consistent invented IDs must not satisfy the held evidence audit."""
    array_shapes = {
        "native_e.npy": [1, 1, 2],
        "native_u.npy": [1, 1, 2, 3],
        "canonical_term.npy": [1, 1, 3],
        "native_head_output.npy": [1, 1, 3],
        "non_equality_remainder.npy": [1, 1, 3],
        "live_removed.npy": [3, 1, 3],
        "hook_delta.npy": [3, 1, 3],
    }
    json_counts = {
        "endpoint_measurements.jsonl": 1,
        "directed_arm_measurements.jsonl": 3,
        "factor_exactness.jsonl": 1,
    }
    expected_operations = [
        {"split": "FIT", "endpoint_id": "expected-endpoint", "site": "L5H5", "role": role}
        for role in ("A", "C")
    ]
    empty_hash = runner.content_sha256([])
    monkeypatch.setattr(runner, "HELD_ARRAY_SHAPES", array_shapes)
    monkeypatch.setattr(runner, "HELD_JSONL_COUNTS", json_counts)
    monkeypatch.setattr(runner, "EXPECTED_OPERATION_COUNTS", {"FIT": 2, "SELECT": 0})
    monkeypatch.setattr(
        runner, "EXPECTED_OPERATION_SHA256", runner.content_sha256(expected_operations)
    )
    monkeypatch.setattr(
        runner, "build_execution_authority",
        lambda: {"endpoints": [{"split": "FIT", "endpoint_id": "expected-endpoint"}]},
    )
    monkeypatch.setattr(
        runner, "build_endpoint_site_role_operations", lambda endpoints: expected_operations
    )
    original_max = np.max
    monkeypatch.setattr(
        runner.np, "max", lambda value: 0.0 if value.size == 0 else original_max(value)
    )

    invented_endpoint = "invented-endpoint"
    invented_direction = "invented-direction"
    endpoint_rows = [{"split": "FIT", "endpoint_id": invented_endpoint}]
    directed_rows = [
        {"split": "FIT", "directed_id": invented_direction, "arm": arm}
        for arm in ("joint", "payload", "score")
    ]
    factor_rows = [{
        "split": "FIT", "endpoint_id": invented_endpoint, "site": "L5H5",
        "equality_factor_max_abs": 0.0,
        "equality_plus_independent_remainder_max_abs": 0.0,
    }]
    e = np.asarray([[[1.0, 2.0]]], dtype="<f4")
    u = np.ones((1, 1, 2, 3), dtype="<f4")
    canonical = np.sum(e[:, :, :, None] * u, axis=2)
    remainder = np.ones_like(canonical)
    arrays = {
        "native_e.npy": e,
        "native_u.npy": u,
        "canonical_term.npy": canonical,
        "native_head_output.npy": canonical + remainder,
        "non_equality_remainder.npy": remainder,
        "live_removed.npy": np.zeros((3, 1, 3), dtype="<f4"),
        "hook_delta.npy": np.zeros((3, 1, 3), dtype="<f4"),
    }
    endpoint_order = [invented_endpoint]
    directed_order = [[invented_direction, arm] for arm in ("joint", "payload", "score")]
    factor_order = [[invented_endpoint, "L5H5"]]
    descriptors = []
    for name, array in arrays.items():
        path = tmp_path / name
        np.save(path, array, allow_pickle=False)
        order = directed_order if name in ("live_removed.npy", "hook_delta.npy") else endpoint_order
        descriptors.append(_descriptor(
            runner, path, runner.EVIDENCE_DIR / name, order, "<f4", list(array.shape)
        ))
    for name, rows, order in (
        ("endpoint_measurements.jsonl", endpoint_rows, endpoint_order),
        ("directed_arm_measurements.jsonl", directed_rows, directed_order),
        ("factor_exactness.jsonl", factor_rows, factor_order),
    ):
        path = tmp_path / name
        _write_jsonl(path, rows)
        descriptors.append(_descriptor(
            runner, path, runner.EVIDENCE_DIR / name, order, "jsonl", [len(rows)]
        ))
    result = {
        "raw_evidence": {
            "schema": runner.EVIDENCE_SCHEMA,
            "endpoint_count": 2_592,
            "directed_arm_record_count": 16_848,
            "endpoint_site_role_operation_counts": {"FIT": 2, "SELECT": 0},
            "endpoint_site_role_operation_sha256": runner.content_sha256(expected_operations),
            "realized_endpoint_site_role_operations": {
                "FIT": {"count": 2, "sha256": runner.content_sha256(expected_operations)},
                "SELECT": {"count": 0, "sha256": empty_hash},
            },
            "instrument_maxima": {
                "native_attention_reconstruction_max_abs": 0.0,
                "equality_factor_max_abs": 0.0,
                "equality_plus_independent_remainder_max_abs": 0.0,
                "replay_native_logit_max_abs": 0.0,
                "padding_tripwire_active_lengths": [19, 20, 21, 22, 27, 28, 29],
            },
        },
        "evidence_files": descriptors,
    }
    resolver = lambda logical: tmp_path / logical.name
    with pytest.raises(ValueError, match="membership|authority"):
        runner._validate_held_evidence(result, resolver)


@BLOCKED_REPAIR
def test_managed_adapter_reaches_partial_final_recovery_before_unused_guard(adapter):
    preflight_source = inspect.getsource(adapter.preflight)
    assert "recover_stale_publication" in preflight_source
    assert preflight_source.index("recover_stale_publication") < preflight_source.index(
        "require_unused_namespaces"
    )
