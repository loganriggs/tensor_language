"""Independent CPU-only acceptance attacks for immutable R585 iteration 6."""

# BQLANE: cpu

from __future__ import annotations

import hashlib
import importlib.util
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

CANDIDATE_COMMIT = "62680bfc78a9c119c40aca8e8a7f5c1eec30ec87"
CANDIDATE_HASHES = {
    PRODUCER: "3963ac0e666874c4d5f35d7be79d1834d0b88b003643acd9950d504dca29e2a1",
    OWNER_TEST: "237946ac4fa7ef5a65a5c6269ad7cfd064195aef993e469df2b06b9b78600024",
    DRYRUN: "c56f0feee2060966fa3fd4210dac0bdc7c945c779eb8c3c5f3068aa6c3fd6a5c",
    ADAPTER: "b3f80585e5b18657ad52604c722f4cf1a492480efebed2b55d572785098ed8b4",
    ADAPTER_TEST: "8df90f0ea8160ad167fa6fcb77462ddfc5e4a29068f81fa5bb44bf0fff86d931",
}
REGISTERED_PREDICATES = {
    "pred_a_cross_split_scope_is_exact":
        "FIT and SELECT validate only their own identities in either manifest order",
    "pred_b_missing_current_evidence_aborts":
        "missing current-split direction, arm, or replay logits hard-abort",
    "pred_c_full_and_local_checks_remain_distinct":
        "live vocabulary logits and saved inserted terms retain separate checks",
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
        prefix=".r585-iteration6-final-review-", suffix=".py", dir=OPS
    )
    os.close(descriptor)
    path = Path(name)
    path.write_bytes(_git_blob(PRODUCER))
    module = _load(path, "r585_iteration6_final_review_producer")
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


def _two_split_manifest(order):
    cells = {
        split: {
            "cell_id": f"{split}|cell", "split": split,
            "directed_ids": [f"{split.lower()}-direction"],
        }
        for split in ("FIT", "SELECT")
    }
    identities = {
        split: {
            "cell_id": cells[split]["cell_id"],
            "left_arm": "joint", "right_arm": "score",
        }
        for split in cells
    }
    return {
        "target_cells": [cells["FIT"], cells["SELECT"]],
        "control_cells": [],
        "structural_identities": [identities[split] for split in order],
    }


def _selected_evidence(runner, torch, split):
    directed_id = f"{split.lower()}-direction"
    records = [{
        "directed_id": directed_id, "arm": "score",
        "recipient_endpoint_id": f"{split.lower()}-recipient",
    }]
    vectors = [
        {
            "directed_id": directed_id, "arm": arm,
            "full_logits": torch.zeros(5),
        }
        for arm in ("joint", "score")
    ]
    frozen = {
        (directed_id, arm, site): torch.zeros(3)
        for arm in ("joint", "score") for site in runner.TERM_NAMES
    }
    return records, vectors, frozen


def test_both_split_orders_validate_only_the_selected_split(runner):
    torch = pytest.importorskip("torch")
    for order in (("FIT", "SELECT"), ("SELECT", "FIT")):
        manifests = _two_split_manifest(order)
        for split in ("FIT", "SELECT"):
            records, vectors, frozen = _selected_evidence(runner, torch, split)
            failures, evidence = runner.structural_identity_failures(
                records, vectors, manifests, {}, split=split,
                frozen_insertions=frozen,
            )
            assert failures == []
            assert len(evidence) == 1
            assert evidence[0]["cell_id"] == f"{split}|cell"
            assert evidence[0]["directed_id"] == f"{split.lower()}-direction"


def test_missing_current_split_direction_hard_aborts(runner):
    manifests = _two_split_manifest(("SELECT", "FIT"))
    with pytest.raises(RuntimeError, match="direction missing"):
        runner.structural_identity_failures(
            [], [], manifests, {}, split="FIT", frozen_insertions={}
        )


def test_missing_current_split_arm_full_logits_hard_aborts(runner):
    torch = pytest.importorskip("torch")
    manifests = _two_split_manifest(("FIT", "SELECT"))
    records, vectors, frozen = _selected_evidence(runner, torch, "FIT")
    vectors = [row for row in vectors if row["arm"] == "score"]
    with pytest.raises(RuntimeError, match="full-logit evidence missing"):
        runner.structural_identity_failures(
            records, vectors, manifests, {}, split="FIT",
            frozen_insertions=frozen,
        )


def test_missing_current_split_replay_full_logits_hard_aborts(runner):
    torch = pytest.importorskip("torch")
    cell = {
        "cell_id": "FIT|cell", "split": "FIT",
        "directed_ids": ["fit-direction"],
    }
    manifests = {
        "target_cells": [cell], "control_cells": [],
        "structural_identities": [{
            "cell_id": "FIT|cell", "left_arm": "payload", "right_arm": "replay",
        }],
    }
    records = [{
        "directed_id": "fit-direction", "arm": "score",
        "recipient_endpoint_id": "fit-recipient",
    }]
    vectors = [{
        "directed_id": "fit-direction", "arm": "payload",
        "full_logits": torch.zeros(5),
    }]
    frozen = {
        ("fit-direction", arm, site): torch.zeros(3)
        for arm in ("payload", "replay") for site in runner.TERM_NAMES
    }
    with pytest.raises(RuntimeError, match="full-logit evidence missing"):
        runner.structural_identity_failures(
            records, vectors, manifests, {}, split="FIT",
            frozen_insertions=frozen,
        )


def test_full_logit_mismatch_precedes_equal_local_insertions(runner):
    torch = pytest.importorskip("torch")
    manifests = _two_split_manifest(("FIT", "SELECT"))
    records, vectors, frozen = _selected_evidence(runner, torch, "FIT")
    vectors[0]["full_logits"] = torch.ones(5)
    with pytest.raises(RuntimeError, match="full-vocabulary identity"):
        runner.structural_identity_failures(
            records, vectors, manifests, {}, split="FIT",
            frozen_insertions=frozen,
        )


def test_saved_inserted_identity_uses_distinct_schema(runner):
    source = _git_blob(PRODUCER).decode()
    assert '"structural_inserted_term_identity_checks": structural_evidence' in source
    assert "raw.get(\"structural_inserted_term_identity_checks\")" in source
    assert "split=split, frozen_insertions=frozen_insertions" in source
