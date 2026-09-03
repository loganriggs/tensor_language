"""Independent CPU-only attacks for immutable R585 iteration 4.

The producer is loaded from Git commit 8e1cadec4, never from the moving
working tree.  The strict xfail is a prospective execution blocker: the
preregistered structural control is an equality of final vocabulary logits,
not merely an equality of the local vectors inserted by the intervention.
"""

# BQLANE: cpu

from __future__ import annotations

import hashlib
import importlib.util
import inspect
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

CANDIDATE_COMMIT = "8e1cadec43a6b0203f10aa1a3c15cb494093b6b7"
CANDIDATE_HASHES = {
    PRODUCER: "29650364b386269267dc663154c81e8413edfe2abae2ce9b7b93524760692cb4",
    OWNER_TEST: "2842f58c54c953885c3b78263ab8bbfd1ddef3b46062fd71fd39d2ea133b289b",
    DRYRUN: "b17cd142bfe4c5d5516b95a06a177bc15fc1e5452b0e234ad7ad5d0c5ed76c1c",
    ADAPTER: "e98d8c2a6b562fd638690bdabf159d761836c1057114af7c0fb8e7149b7332a2",
    ADAPTER_TEST: "0c13c70aab4cf3775754d5ce42cf21626394d9ab06655051daa814ad744a2d41",
}
FULL_LOGIT_BLOCKER = pytest.mark.xfail(
    strict=True,
    reason=(
        "8e1cadec4 replaces the preregistered full-vocabulary structural "
        "identity check with a local inserted-vector check"
    ),
)
REGISTERED_PREDICATES = {
    "pred_a_instrument_failures_are_derived":
        "saved instrument failures are exactly rederived in stable order",
    "pred_b_hard_integrity_failures_abort":
        "unreconstructible native and replay integrity failures abort",
    "pred_c_structural_controls_match_preregistration":
        "structural controls remain full-vocabulary final-logit equalities",
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
    descriptor, name = tempfile.mkstemp(
        prefix=".r585-iteration4-review-", suffix=".py", dir=OPS
    )
    os.close(descriptor)
    path = Path(name)
    path.write_bytes(_git_blob(PRODUCER))
    module = _load(path, "r585_iteration4_review_producer")
    try:
        yield module
    finally:
        path.unlink(missing_ok=True)


def test_exact_candidate_commit_and_blobs():
    assert subprocess.check_output(
        ["git", "rev-parse", CANDIDATE_COMMIT], cwd=REPO, text=True
    ).strip() == CANDIDATE_COMMIT
    for path, expected in CANDIDATE_HASHES.items():
        assert hashlib.sha256(_git_blob(path)).hexdigest() == expected


def test_phase_prices_and_split_closure_are_unchanged(runner):
    assert runner.EXPECTED_PHASE_PRICE == {"FIT": 459, "SELECT": 231}
    assert runner.EXPECTED_TOTAL_PRICE == 690
    assert runner.SPLITS == ("FIT", "SELECT")
    fit = runner.phase_evidence_contract(["FIT"])
    full = runner.phase_evidence_contract(["FIT", "SELECT"])
    assert (fit["endpoint_count"], fit["direction_count"]) == (1_728, 3_744)
    assert fit["directed_arm_record_count"] == 11_232
    assert fit["factor_exactness_count"] == 6_912
    assert (full["endpoint_count"], full["direction_count"]) == (2_592, 5_616)
    assert full["directed_arm_record_count"] == 16_848
    assert full["factor_exactness_count"] == 10_368


def _structural_fixture(runner, torch):
    directed_id = "FIT|direction-0"
    cell_id = "FIT|cell-0"
    records = [{
        "directed_id": directed_id,
        "arm": "score",
        "recipient_endpoint_id": "recipient-0",
    }]
    vectors = [
        {"directed_id": directed_id, "arm": "score", "full_logits": torch.zeros(5)},
        {"directed_id": directed_id, "arm": "joint", "full_logits": torch.ones(5)},
    ]
    manifests = {
        "target_cells": [{
            "cell_id": cell_id, "split": "FIT", "directed_ids": [directed_id]
        }],
        "control_cells": [],
        "structural_identities": [{
            "cell_id": cell_id, "left_arm": "joint", "right_arm": "score"
        }],
    }
    frozen = {
        (directed_id, arm, site): torch.zeros(3)
        for arm in ("joint", "score") for site in runner.TERM_NAMES
    }
    return records, vectors, manifests, frozen


def test_attack_distinguishes_full_logits_from_local_insertions(runner):
    torch = pytest.importorskip("torch")
    records, vectors, manifests, frozen = _structural_fixture(runner, torch)
    full_failures, full_evidence = runner.structural_identity_failures(
        records, vectors, manifests, {}
    )
    local_failures, local_evidence = runner.structural_identity_failures(
        records, vectors, manifests, {}, frozen_insertions=frozen
    )
    assert len(full_failures) == 1
    assert full_evidence[0]["max_abs"] == 1.0
    assert local_failures == []
    assert local_evidence[0]["max_abs"] == 0.0


@FULL_LOGIT_BLOCKER
def test_runtime_retains_preregistered_full_vocabulary_identity(runner):
    torch = pytest.importorskip("torch")
    records, vectors, manifests, frozen = _structural_fixture(runner, torch)
    expected_failures, _ = runner.structural_identity_failures(
        records, vectors, manifests, {}
    )
    runtime_failures, _ = runner.structural_identity_failures(
        records, vectors, manifests, {}, frozen_insertions=frozen
    )
    assert runtime_failures == expected_failures
    science = inspect.getsource(runner.run_science)
    assert "frozen_insertions=frozen_insertions" not in science


def test_invalid_instrument_derivation_is_sorted_and_hard_errors_abort(
    runner, monkeypatch,
):
    monkeypatch.setattr(
        runner, "validate_primitive_logit_identities", lambda rows: ["primitive:z"]
    )
    endpoint_rows = [{
        "split": "FIT", "endpoint_id": "endpoint-0", "length": 5,
        "replay_padding_length": 5, "native_padding_length": 5,
        "replay_native_logit_max_abs": 0.0,
    }]
    factor_rows = [{
        "split": "FIT", "endpoint_id": "endpoint-0", "site": "l8h3_qk_score",
        "equality_factor_max_abs": 1e-3,
        "equality_plus_independent_remainder_max_abs": 2e-3,
    }]
    structural = [{
        "cell_id": "FIT|cell-0", "directed_id": "direction-0",
        "left_arm": "joint", "right_arm": "score", "max_abs": 3e-3,
    }]
    arrays = {
        "native_e.npy": np.zeros((1, 4, 2), dtype="<f4"),
        "native_u.npy": np.zeros((1, 4, 2, 3), dtype="<f4"),
        "canonical_term.npy": np.ones((1, 4, 3), dtype="<f4"),
    }
    execution = {"directions": [{
        "split": "FIT", "directed_id": "direction-0",
        "recipient_endpoint_id": "endpoint-0",
    }]}
    maxima = {"native_attention_reconstruction_max_abs": 0.0}
    failures = runner.derive_saved_instrument_failures(
        "FIT", endpoint_rows, [], factor_rows, arrays, structural, maxima, execution
    )
    assert failures == sorted(failures)
    assert "canonical_factor:endpoint-0:l8h3_qk_score" in failures
    assert "head_reconstruction:endpoint-0:l8h3_qk_score" in failures
    assert "primitive:z" in failures
    assert any(value.startswith("frozen_replay_canonical:") for value in failures)
    assert any(value.startswith("structural_identity:") for value in failures)

    bad_native = {"native_attention_reconstruction_max_abs": 1.0}
    with pytest.raises(ValueError, match="native-attention failure"):
        runner.derive_saved_instrument_failures(
            "FIT", endpoint_rows, [], factor_rows, arrays, structural,
            bad_native, execution,
        )
    endpoint_rows[0]["replay_native_logit_max_abs"] = 1.0
    with pytest.raises(ValueError, match="replay/native full-logit failure"):
        runner.derive_saved_instrument_failures(
            "FIT", endpoint_rows, [], factor_rows, arrays, structural,
            maxima, execution,
        )
