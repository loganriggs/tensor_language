"""Independent CPU-only attacks for the second repaired R585 candidate.

The reviewed producer and adapter are loaded from immutable commit a19c029fd,
never from the moving working-tree files.  Strict xfails are execution blockers
found prospectively, before an R585 outcome existed.
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

CANDIDATE_COMMIT = "a19c029fd178e716a94024d573a82308e78d32be"
CANDIDATE_HASHES = {
    PRODUCER: "8a4f20d06dd04cd81d6bb8c94377ee987b66bea4201395e61bbe23a1b5dd9a8c",
    OWNER_TEST: "57e52e8da53f3a6e7b194efb64f56d1ff9fb442c2c39547a6f1fed4263a10653",
    DRYRUN: "6fb41eb862c00f27673cfe694cf8670eae23f1d60a6a5dd85a35a5309b7e90f5",
    ADAPTER: "efaeb3ee746f1c18caa52ab4466d403a31c0b2fe509a2c118b39ebcdad10e2d9",
    ADAPTER_TEST: "67486003c4f14208179fd8b52099951b1212570d0830632ce37e85f3abcdddbe",
}
EVIDENCE_BINDING_BLOCKER = pytest.mark.xfail(
    strict=True,
    reason="a19c029fd checks evidence IDs/shapes but not saved computation semantics",
)
REGISTERED_PREDICATES = {
    "pred_a_prior_repairs_close":
        "the exact phase, membership, bootstrap, finite, and price contracts close",
    "pred_b_saved_computation_is_authority_bound":
        "saved directed metadata, primitive logits, and vectors rederive from authority",
    "pred_c_managed_recovery_is_conservative":
        "managed recovery quarantines recognized partial state and refuses other state",
}


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
    producer_descriptor, producer_name = tempfile.mkstemp(
        prefix=".r585-second-review-producer-", suffix=".py", dir=OPS
    )
    os.close(producer_descriptor)
    producer_path = Path(producer_name)
    owner_descriptor, owner_name = tempfile.mkstemp(
        prefix=".r585-second-review-owner-", suffix=".py", dir=OPS
    )
    os.close(owner_descriptor)
    owner_path = Path(owner_name)
    producer_path.write_bytes(_git_blob(PRODUCER))
    owner_path.write_bytes(_git_blob(OWNER_TEST))
    module = _load(producer_path, "r585_second_repair_review_producer")
    module.SCRIPT = producer_path
    module.TEST = owner_path
    try:
        yield module
    finally:
        producer_path.unlink(missing_ok=True)
        owner_path.unlink(missing_ok=True)


@pytest.fixture(scope="module")
def adapter():
    adapter_descriptor, adapter_name = tempfile.mkstemp(
        prefix=".r585-second-review-adapter-", suffix=".py", dir=OPS
    )
    os.close(adapter_descriptor)
    adapter_path = Path(adapter_name)
    adapter_path.write_bytes(_git_blob(ADAPTER))
    module = _load(adapter_path, "r585_second_repair_review_adapter")
    try:
        yield module
    finally:
        adapter_path.unlink(missing_ok=True)


def test_exact_candidate_commit_and_blobs():
    assert subprocess.check_output(
        ["git", "rev-parse", CANDIDATE_COMMIT], cwd=REPO, text=True
    ).strip() == CANDIDATE_COMMIT
    for path, expected in CANDIDATE_HASHES.items():
        assert hashlib.sha256(_git_blob(path)).hexdigest() == expected


def test_phase_shapes_prices_and_all_fixture_terminals_fail_closed(runner):
    fit = runner.phase_evidence_contract(["FIT"])
    assert fit == {
        "evaluated_splits": ["FIT"],
        "endpoint_count": 1_728,
        "direction_count": 3_744,
        "directed_arm_record_count": 11_232,
        "factor_exactness_count": 6_912,
        "array_shapes": {
            "native_e.npy": [1_728, 4, 2],
            "native_u.npy": [1_728, 4, 2, 1_152],
            "canonical_term.npy": [1_728, 4, 1_152],
            "native_head_output.npy": [1_728, 4, 1_152],
            "non_equality_remainder.npy": [1_728, 4, 1_152],
            "live_removed.npy": [11_232, 4, 1_152],
            "hook_delta.npy": [11_232, 4, 1_152],
        },
        "jsonl_counts": {
            "endpoint_measurements.jsonl": 1_728,
            "directed_arm_measurements.jsonl": 11_232,
            "factor_exactness.jsonl": 6_912,
        },
    }
    full = runner.phase_evidence_contract(["FIT", "SELECT"])
    assert full["endpoint_count"] == 2_592
    assert full["direction_count"] == 5_616
    assert full["directed_arm_record_count"] == 16_848
    assert full["factor_exactness_count"] == 10_368
    assert full["array_shapes"] == runner.HELD_ARRAY_SHAPES
    assert full["jsonl_counts"] == runner.HELD_JSONL_COUNTS
    assert runner.EXPECTED_PHASE_PRICE == {"FIT": 459, "SELECT": 231}
    assert runner.EXPECTED_TOTAL_PRICE == 690
    for terminal in sorted(runner.TERMINALS):
        fixture = runner.make_result_fixture(terminal)
        assert type(fixture["next_step"]) is str
        with pytest.raises(ValueError, match="model-free fixture"):
            runner.validate_result(fixture)


def test_terminal_bootstrap_phase_semantics_are_exact(runner, monkeypatch):
    monkeypatch.setattr(runner, "_validate_complete_evidence", lambda result, resolver: None)
    monkeypatch.setattr(runner, "build_execution_authority", lambda: {"manifests": {}})

    def realized(report, split, manifests):
        return report["bootstrap_realization"]

    monkeypatch.setattr(runner, "validate_realized_bootstraps", realized)
    for terminal in sorted(runner.TERMINALS):
        result = runner.make_result_fixture(terminal)
        result["raw_evidence"] = {}
        if terminal == "invalid_instrument":
            scored = []
        elif terminal == "select_invalid_instrument":
            scored = ["FIT"]
        else:
            scored = list(result["evaluated_splits"])
        result["split_scores"] = {
            split: {
                "bootstrap_realization": {
                    "count": 124, "cell_ids_sha256": split.lower() + "-planted"
                }
            }
            for split in scored
        }
        runner.validate_result(result)
        unexpected = copy.deepcopy(result)
        extra = "SELECT" if "SELECT" not in scored else "unexpected"
        unexpected["split_scores"][extra] = {
            "bootstrap_realization": {"count": 124, "cell_ids_sha256": "extra"}
        }
        with pytest.raises(ValueError, match="split-score phase census"):
            runner.validate_result(unexpected)


def test_previous_membership_bootstrap_and_nonfinite_attacks_are_closed(
    runner, tmp_path
):
    execution = runner.build_execution_authority()
    identities = runner.expected_evidence_identities(["FIT"], execution)
    invented = copy.deepcopy(identities)
    invented["endpoints"][0] = "invented-endpoint"
    with pytest.raises(ValueError, match="frozen authority"):
        runner.validate_evidence_membership(
            ["FIT"], invented["endpoints"], invented["directed_arms"],
            invented["factors"], execution,
        )

    planted = runner.planted_intervention_records(execution)
    scales = runner.compute_fit_scales(planted, execution["manifests"])
    report, _ = runner.score_split(
        planted, "FIT", execution["manifests"], scales, replicates=2
    )
    assert report["bootstrap_realization"]["count"] == 124
    incomplete = copy.deepcopy(report)

    def drop_first_bootstrap(value):
        if type(value) is dict:
            for key, child in list(value.items()):
                if type(child) is dict and "cell_id" in child:
                    del value[key]
                    return True
                if drop_first_bootstrap(child):
                    return True
        elif type(value) is list:
            for index, child in enumerate(list(value)):
                if type(child) is dict and "cell_id" in child:
                    del value[index]
                    return True
                if drop_first_bootstrap(child):
                    return True
        return False

    assert drop_first_bootstrap(incomplete)
    with pytest.raises(RuntimeError, match="bootstrap.*census"):
        runner.validate_realized_bootstraps(
            incomplete, "FIT", execution["manifests"]
        )

    nonfinite = runner.make_result_fixture("factor_capacity_null")
    nonfinite["elapsed_seconds"] = float("nan")
    stage = runner.create_stage_root(tmp_path)
    with pytest.raises(ValueError, match="nonfinite"):
        runner._finish_result(nonfinite, stage)
    assert [path.name for path in stage.iterdir()] == [runner.STAGE_MARKER_NAME]


def test_managed_recovery_precedes_guard_and_refuses_unsafe_states(
    runner, adapter, tmp_path, monkeypatch
):
    source = inspect.getsource(adapter.preflight)
    assert source.index("recover_publication_preflight") < source.index(
        "require_unused_namespaces"
    )
    monkeypatch.setattr(adapter, "verify_frozen_bytes", lambda: {})

    recognized = tmp_path / "recognized"
    recognized.mkdir()
    out = recognized / "result.json"
    receipt = recognized / "receipt.json"
    evidence = recognized / "evidence"
    stage = runner.create_stage_root(recognized)
    partial_result = runner.make_result_fixture("factor_capacity_null")
    out.write_text(json.dumps(partial_result, sort_keys=True, allow_nan=False))

    def recover_recognized():
        runner.recover_stale_publication(
            root=recognized, out=out, receipt=receipt, evidence=evidence
        )

    with pytest.raises(RuntimeError, match="recovered incomplete"):
        adapter.dispatch(
            {"BQLIB_DRYRUN": "1"}, recovery_function=recover_recognized,
            namespace_paths=(out, receipt, evidence),
            dry_validator=lambda: pytest.fail("dry run continued after recovery"),
        )
    assert not stage.exists()
    recovered = list(recognized.glob(runner.RECOVERY_PREFIX + "*"))
    assert len(recovered) == 1
    assert any(path.name.startswith("partial-result-") for path in recovered[0].iterdir())

    arbitrary = tmp_path / "arbitrary"
    arbitrary.mkdir()
    arbitrary_out = arbitrary / "result.json"
    arbitrary_out.write_text("unrelated bytes\n")
    arbitrary_stage = runner.create_stage_root(arbitrary)

    def recover_arbitrary():
        runner.recover_stale_publication(
            root=arbitrary, out=arbitrary_out,
            receipt=arbitrary / "receipt.json", evidence=arbitrary / "evidence"
        )

    with pytest.raises(RuntimeError, match="unrecognized"):
        adapter.dispatch(
            {"BQLIB_DRYRUN": "1"}, recovery_function=recover_arbitrary,
            namespace_paths=(
                arbitrary_out, arbitrary / "receipt.json", arbitrary / "evidence"
            ),
        )
    assert arbitrary_out.read_text() == "unrelated bytes\n"
    assert arbitrary_stage.exists()

    complete = tmp_path / "complete"
    complete.mkdir()
    complete_out = complete / "result.json"
    complete_receipt = complete / "receipt.json"
    complete_evidence = complete / "evidence"
    complete_out.write_text("leave result\n")
    complete_receipt.write_text("leave receipt\n")
    complete_evidence.mkdir()

    def refuse_complete():
        runner.recover_stale_publication(
            root=complete, out=complete_out, receipt=complete_receipt,
            evidence=complete_evidence,
        )

    with pytest.raises(RuntimeError, match="complete output namespace"):
        adapter.dispatch(
            {"BQLIB_DRYRUN": "1"}, recovery_function=refuse_complete,
            namespace_paths=(complete_out, complete_receipt, complete_evidence),
        )
    assert complete_out.read_text() == "leave result\n"
    assert complete_receipt.read_text() == "leave receipt\n"
    assert complete_evidence.is_dir()


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


def _miniature_complete_result(
    runner, tmp_path, monkeypatch, *, bad_semantics=False,
    bad_vectors=False, bad_primitives=False, bad_endpoint=False,
):
    endpoint = {
        "split": "FIT", "endpoint_id": "endpoint-0",
        "token_ids": [1, 2, 3, 4, 5], "length": 5, "final_position": 4,
        "source_positions": [0, 2], "payload_positions": [1, 3],
        "condition": "s0p0",
        "answer_id": 3, "other_answer_id": 4,
    }
    direction = {
        "split": "FIT", "directed_id": "direction-0",
        "recipient_endpoint_id": "endpoint-0", "donor_endpoint_id": "endpoint-0",
    }
    execution = {
        "endpoints": [endpoint], "directions": [direction], "manifests": {},
    }
    operations = sorted([
        {
            "split": "FIT", "endpoint_id": "endpoint-0",
            "site": site, "role": role,
        }
        for site in runner.TERM_NAMES for role in runner.ROLES
    ], key=lambda row: (row["split"], row["endpoint_id"], row["site"], row["role"]))
    contract = {
        "evaluated_splits": ["FIT"],
        "endpoint_count": 1,
        "direction_count": 1,
        "directed_arm_record_count": 3,
        "factor_exactness_count": 4,
        "array_shapes": {
            "native_e.npy": [1, 4, 2],
            "native_u.npy": [1, 4, 2, 3],
            "canonical_term.npy": [1, 4, 3],
            "native_head_output.npy": [1, 4, 3],
            "non_equality_remainder.npy": [1, 4, 3],
            "live_removed.npy": [3, 4, 3],
            "hook_delta.npy": [3, 4, 3],
        },
        "jsonl_counts": {
            "endpoint_measurements.jsonl": 1,
            "directed_arm_measurements.jsonl": 3,
            "factor_exactness.jsonl": 4,
        },
    }
    monkeypatch.setattr(runner, "phase_evidence_contract", lambda splits: contract)
    monkeypatch.setattr(runner, "build_execution_authority", lambda: execution)
    monkeypatch.setattr(
        runner, "build_endpoint_site_role_operations", lambda endpoints: operations
    )
    monkeypatch.setattr(runner, "EXPECTED_OPERATION_COUNTS", {"FIT": 8})

    e = np.ones((1, 4, 2), dtype="<f4")
    u = np.ones((1, 4, 2, 3), dtype="<f4")
    canonical = np.sum(e[:, :, :, None] * u, axis=2)
    remainder = np.ones_like(canonical)
    live = np.repeat(canonical, 3, axis=0)
    delta = np.zeros_like(live)
    if bad_vectors:
        live[:] = 0
    arrays = {
        "native_e.npy": e,
        "native_u.npy": u,
        "canonical_term.npy": canonical,
        "native_head_output.npy": canonical + remainder,
        "non_equality_remainder.npy": remainder,
        "live_removed.npy": live,
        "hook_delta.npy": delta,
    }
    endpoint_rows = [dict(endpoint)]
    if bad_endpoint:
        endpoint_rows[0]["token_ids"] = [99, 98, 97]
    directed_rows = []
    for arm in sorted(runner.ARMS):
        row = {
            **direction, "arm": arm,
            "answer_logit": 2.0, "other_logit": 1.0,
            "correct_margin": 1.0, "log_normalizer": 3.0,
            "correct_ce": 1.0, "vocab_squared_difference_sum": 3.0,
            "vocab_size": 3, "vocab_rms": 1.0,
            "live_factor_max_error": 0.0, "hook_delta_sum_max_error": 0.0,
        }
        if bad_semantics:
            row["recipient_endpoint_id"] = "invented-recipient"
            row["donor_endpoint_id"] = "invented-donor"
        if bad_primitives:
            row["correct_margin"] = -999.0
        directed_rows.append(row)
    factor_rows = [{
        "split": "FIT", "endpoint_id": "endpoint-0", "site": site,
        "equality_factor_max_abs": 0.0,
        "equality_plus_independent_remainder_max_abs": 0.0,
    } for site in sorted(runner.TERM_NAMES)]
    endpoint_order = ["endpoint-0"]
    directed_order = [["direction-0", arm] for arm in sorted(runner.ARMS)]
    factor_order = [["endpoint-0", site] for site in sorted(runner.TERM_NAMES)]
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
    result = runner.make_result_fixture("invalid_instrument")
    result["raw_evidence"] = {
        "schema": runner.EVIDENCE_SCHEMA,
        "endpoint_count": 1,
        "directed_arm_record_count": 3,
        "endpoint_site_role_operation_counts": {"FIT": 8},
        "endpoint_site_role_operation_sha256": runner.content_sha256(operations),
        "realized_endpoint_site_role_operations": {
            "FIT": {"count": 8, "sha256": runner.content_sha256(operations)}
        },
        "instrument_maxima": {
            "native_attention_reconstruction_max_abs": 0.0,
            "equality_factor_max_abs": 0.0,
            "equality_plus_independent_remainder_max_abs": 0.0,
            "replay_native_logit_max_abs": 0.0,
            "padding_tripwire_active_lengths": [],
        },
    }
    result["evidence_files"] = descriptors
    result["split_scores"] = {}
    resolver = lambda logical: tmp_path / logical.name
    return result, resolver


@EVIDENCE_BINDING_BLOCKER
def test_completed_evidence_rejects_wrong_endpoint_semantic_coordinates(
    runner, tmp_path, monkeypatch
):
    result, resolver = _miniature_complete_result(
        runner, tmp_path, monkeypatch, bad_endpoint=True
    )
    with pytest.raises(ValueError, match="endpoint.*authority|semantic"):
        runner.validate_result(result, artifact_path_resolver=resolver)


@EVIDENCE_BINDING_BLOCKER
def test_completed_evidence_rejects_wrong_recipient_and_donor_membership(
    runner, tmp_path, monkeypatch
):
    result, resolver = _miniature_complete_result(
        runner, tmp_path, monkeypatch, bad_semantics=True
    )
    with pytest.raises(ValueError, match="recipient|donor|direction authority"):
        runner.validate_result(result, artifact_path_resolver=resolver)


@EVIDENCE_BINDING_BLOCKER
def test_completed_evidence_recomputes_inserted_term_from_live_plus_delta(
    runner, tmp_path, monkeypatch
):
    result, resolver = _miniature_complete_result(
        runner, tmp_path, monkeypatch, bad_vectors=True
    )
    with pytest.raises(ValueError, match="live|delta|inserted term"):
        runner.validate_result(result, artifact_path_resolver=resolver)


@EVIDENCE_BINDING_BLOCKER
def test_completed_evidence_recomputes_primitive_logit_identities(
    runner, tmp_path, monkeypatch
):
    result, resolver = _miniature_complete_result(
        runner, tmp_path, monkeypatch, bad_primitives=True
    )
    with pytest.raises(ValueError, match="primitive|margin"):
        runner.validate_result(result, artifact_path_resolver=resolver)
