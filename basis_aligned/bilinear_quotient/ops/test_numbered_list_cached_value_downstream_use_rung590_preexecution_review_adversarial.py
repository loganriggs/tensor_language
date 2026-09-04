"""Independent CPU-only attacks on immutable R590 commit cf00f555d.

These tests deliberately bind the reviewed working files to Git blobs before
importing them.  They never inspect an R584 or R590 scientific result.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import math
from pathlib import Path
import subprocess

import pytest


COMMIT = "cf00f555d"
ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "ops"
REPO = ROOT.parents[1]

PRODUCER = OPS / "numbered_list_cached_value_downstream_use_rung590.py"
ADAPTER = OPS / "execute_numbered_list_cached_value_downstream_use_rung590.py"

EXPECTED = {
    PRODUCER: "74b565fe835ee69a73ed1bdcdc103df3b2f4aa94931796ca1b96a4080639062e",
    ADAPTER: "34899a771279cda55e674df2da3de7cf8321a787b26e986d2601d8bbdd6b3479",
}


def _blob(path: Path) -> bytes:
    relative = path.relative_to(REPO)
    return subprocess.check_output(
        ["git", "show", f"{COMMIT}:{relative.as_posix()}"], cwd=REPO
    )


def _load_exact(path: Path, name: str):
    blob = _blob(path)
    assert hashlib.sha256(blob).hexdigest() == EXPECTED[path]
    # Refuse to import a moving worktree file unless it is byte-identical to
    # the immutable reviewed blob at the import boundary.
    assert path.read_bytes() == blob
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


r590 = _load_exact(PRODUCER, "r590_independent_exact_target")
adapter = _load_exact(ADAPTER, "r590_independent_exact_adapter")


@pytest.fixture(scope="module")
def fit_null_evidence():
    return r590.evidence_from_legacy_payload(
        r590.r588.make_fixture(held=False, replicates=8)
    )


def test_exact_threshold_is_strictly_hard_aborted(fit_null_evidence):
    changed = copy.deepcopy(fit_null_evidence)
    over = math.nextafter(r590.EXACT_BAR, math.inf)
    by_row = changed["fit_capture_raw"][0][
        "native_replay_relative_squared_error_by_row"
    ]
    by_row["source_present"] = over
    by_row["maximum"] = over
    changed["fit_exactness"]["native_replay_relative_squared_error"] = over
    with pytest.raises(r590.UnretainedInstrumentError, match="publishable evidence"):
        r590.derive_scientific_summary(changed, replicates=8)


def test_internally_rehashed_phase_reorder_still_fails(fit_null_evidence):
    changed = copy.deepcopy(fit_null_evidence)
    fit = changed["phase_support_census"]["splits"]["FIT"]
    fit["ordered_row_ids"] = fit["ordered_row_ids"][1:] + fit["ordered_row_ids"][:1]
    fit["ordered_row_ids_sha256"] = r590.canonical_sha256(fit["ordered_row_ids"])
    changed["phase_support_census_sha256"] = r590.canonical_sha256(
        changed["phase_support_census"]
    )
    with pytest.raises(RuntimeError, match="phase_support"):
        r590.derive_scientific_summary(changed, replicates=8)


def test_correlated_result_and_receipt_rewrite_cannot_change_decision(
    fit_null_evidence,
):
    result = r590.build_result(
        fit_null_evidence,
        evidence_sha256=r590.canonical_sha256(fit_null_evidence),
        checkpoint_sha256=r590.CHECKPOINT_SHA256,
        elapsed_seconds=1.0,
        replicates=8,
    )
    changed = copy.deepcopy(result)
    changed["decision"] = "downstream_use_component_held"
    changed["next_step"] = "publish_changed_claim"
    # Recomputing outer hashes cannot make a non-evidence-derived terminal valid.
    changed_bytes = r590.canonical_bytes(changed)
    receipt = r590.make_receipt(
        changed_bytes, r590.canonical_bytes(fit_null_evidence), changed
    )
    r590.validate_receipt(
        receipt, changed_bytes, r590.canonical_bytes(fit_null_evidence), changed
    )
    with pytest.raises(RuntimeError):
        r590.validate_result_against_evidence(
            changed, fit_null_evidence, replicates=8
        )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BLOCK: managed adapter verifies the producer but not its executable "
        "dependency closure before importing it"
    ),
)
def test_adapter_prepins_executable_dependency_closure_before_import():
    required = {
        r590.R584_RUNNER,
        r590.R588_AUDITOR,
        r590.RESULT_CONTRACT,
        r590.r588.FACADE,
        r590.r588.R576_RUNNER,
        r590.r588.R573_RUNNER,
        r590.R582_HELPER,
    }
    assert required <= set(adapter.FROZEN_HASHES)

