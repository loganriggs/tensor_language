"""Independent CPU-only review attacks for immutable R585 iteration 5.

All producer behavior is loaded from commit e63fa74b7 rather than the moving
working tree. Strict xfails identify prospective execution blockers.
"""

# BQLANE: cpu

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile

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

CANDIDATE_COMMIT = "e63fa74b70f722e7d993d6bc2d4b03372e98f7ce"
CANDIDATE_HASHES = {
    PRODUCER: "33b5cbbc26e5ba62bb60a5bf62d69a1ef7ea51d1bf64e51fd3b95049e55f4327",
    OWNER_TEST: "1a3419e3aa19abc2b03424d02ff5c474296472811e780dd10bbde4cc34f410d7",
    DRYRUN: "ac02054a22452911150e173792f28902351fdf1b04d04b87007a570837cf026d",
    ADAPTER: "b156bf741dfbc7dd57e669bc4a9dc981092b7308d14a1c45c05e91a2e5944f1b",
    ADAPTER_TEST: "6a8ce51bc139b5a070adb72ffd2abaf7e811b427a2a0b7189259e6cc20de4bb0",
}
SPLIT_SCOPE_BLOCKER = pytest.mark.xfail(
    strict=True,
    reason=(
        "e63fa74b7 treats directions from the other registered split as missing "
        "within each split-scoped runtime call"
    ),
)
REGISTERED_PREDICATES = {
    "pred_a_full_logit_identity_is_live":
        "the full-vocabulary structural identity hard-aborts mismatches",
    "pred_b_capture_set_is_manifest_derived":
        "all and only non-replay structural arms are captured",
    "pred_c_runtime_identity_is_split_scoped":
        "each split validates its own identities and no directions from another split",
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
        prefix=".r585-iteration5-review-", suffix=".py", dir=OPS
    )
    os.close(descriptor)
    path = Path(name)
    path.write_bytes(_git_blob(PRODUCER))
    module = _load(path, "r585_iteration5_review_producer")
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


def _one_identity(split, directed_id, *, left="joint", right="score"):
    cell_id = f"{split}|cell"
    cell = {
        "cell_id": cell_id, "split": split, "directed_ids": [directed_id]
    }
    identity = {"cell_id": cell_id, "left_arm": left, "right_arm": right}
    return cell, identity


def _frozen(runner, torch, directed_id, arms):
    return {
        (directed_id, arm, site): torch.zeros(3)
        for arm in arms for site in runner.TERM_NAMES
    }


def test_iteration4_full_vocabulary_counterexample_now_hard_aborts(runner):
    torch = pytest.importorskip("torch")
    cell, identity = _one_identity("FIT", "fit-direction")
    manifests = {
        "target_cells": [cell], "control_cells": [],
        "structural_identities": [identity],
    }
    records = [{
        "directed_id": "fit-direction", "arm": "score",
        "recipient_endpoint_id": "recipient",
    }]
    vectors = [
        {
            "directed_id": "fit-direction", "arm": "score",
            "full_logits": torch.zeros(5),
        },
        {
            "directed_id": "fit-direction", "arm": "joint",
            "full_logits": torch.ones(5),
        },
    ]
    frozen = _frozen(runner, torch, "fit-direction", ("score", "joint"))
    with pytest.raises(RuntimeError, match="full-vocabulary identity"):
        runner.structural_identity_failures(
            records, vectors, manifests, {}, frozen_insertions=frozen
        )


def test_manifest_derived_capture_set_is_exact_and_affordable(runner):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        execution = runner.build_execution_authority()
    manifests = execution["manifests"]
    cells = {
        cell["cell_id"]: cell
        for cell in (*manifests["target_cells"], *manifests["control_cells"])
    }
    expected = {
        (directed_id, arm)
        for identity in manifests["structural_identities"]
        for directed_id in cells[identity["cell_id"]]["directed_ids"]
        for arm in (identity["left_arm"], identity["right_arm"])
        if arm != "replay"
    }
    observed = runner.required_structural_full_logit_pairs(manifests)
    assert observed == expected
    assert len(observed) == 5_184
    direction_split = {
        row["directed_id"]: row["split"] for row in execution["directions"]
    }
    assert sum(direction_split[key] == "FIT" for key, _ in observed) == 3_456
    assert sum(direction_split[key] == "SELECT" for key, _ in observed) == 1_728
    assert len(observed) * 50_257 * 4 < 2**30


@SPLIT_SCOPE_BLOCKER
def test_split_scoped_runtime_ignores_registered_other_split(runner):
    torch = pytest.importorskip("torch")
    fit_cell, fit_identity = _one_identity("FIT", "fit-direction")
    select_cell, select_identity = _one_identity("SELECT", "select-direction")
    manifests = {
        "target_cells": [fit_cell, select_cell], "control_cells": [],
        "structural_identities": [fit_identity, select_identity],
    }
    records = [{
        "directed_id": "fit-direction", "arm": "score",
        "recipient_endpoint_id": "recipient",
    }]
    vectors = [
        {
            "directed_id": "fit-direction", "arm": arm,
            "full_logits": torch.zeros(5),
        }
        for arm in ("score", "joint")
    ]
    frozen = _frozen(runner, torch, "fit-direction", ("score", "joint"))
    failures, evidence = runner.structural_identity_failures(
        records, vectors, manifests, {}, frozen_insertions=frozen
    )
    assert failures == []
    assert [row["cell_id"] for row in evidence] == [fit_cell["cell_id"]]


def test_missing_direction_within_selected_manifest_still_hard_aborts(runner):
    torch = pytest.importorskip("torch")
    cell, identity = _one_identity("FIT", "fit-direction")
    manifests = {
        "target_cells": [cell], "control_cells": [],
        "structural_identities": [identity],
    }
    with pytest.raises(RuntimeError, match="direction missing"):
        runner.structural_identity_failures(
            [], [], manifests, {}, frozen_insertions={}
        )


def test_saved_inserted_identity_has_a_distinct_schema_name(runner):
    source = _git_blob(PRODUCER).decode()
    assert '"structural_inserted_term_identity_checks": structural_evidence' in source
    assert "raw.get(\"structural_inserted_term_identity_checks\")" in source
    assert '"structural_identity_checks": structural_evidence' not in source
