"""Independent model-free attacks on exact prospective R590 commit 3eb52938b.

The tests bind every reviewed producer/adapter byte to an immutable Git blob,
use only planted R588 evidence, and never open an R584 or R590 outcome.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import math
from pathlib import Path
import subprocess
import sys

import pytest


COMMIT = "3eb52938b3641f067d8f8eb9e654f461cbd61ad0"
ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "ops"
REPO = ROOT.parents[1]
PRODUCER = OPS / "numbered_list_cached_value_downstream_use_rung590.py"
OWNER_TEST = OPS / "test_numbered_list_cached_value_downstream_use_rung590.py"
DRYRUN = ROOT / "numbered_list_cached_value_downstream_use_rung590_dryrun.json"
ADAPTER = OPS / "execute_numbered_list_cached_value_downstream_use_rung590.py"
ADAPTER_TEST = OPS / "test_execute_numbered_list_cached_value_downstream_use_rung590.py"
NOTE = ROOT.parent / "polynomial_causal" / (
    "NUMBERED_LIST_CACHED_VALUE_DOWNSTREAM_USE_RUNG590_"
    "PROSPECTIVE_CONTRACT_REPLICATION.md"
)

EXPECTED = {
    PRODUCER: "c38654506f36fcf111f3a34f356893240548c3cfbf4eded58efb04d31fdb2e36",
    OWNER_TEST: "49f6f7a998bfb69331c36391f5d3c16d9b702c1fd60b4da5b09c920f3832e5b0",
    DRYRUN: "3ebada19f74906ba3e7cd1637fc1cd6cdff84936124dee01cb058875432d3b95",
    ADAPTER: "c525cad078935ef0552214fba13c16a5d56483c8e3048bbec4d6ab9ef3f17885",
    ADAPTER_TEST: "17d51c8e7df667ecf1cc146b1ac00e34f658e97759ee149ddb254f7d9317f07e",
    NOTE: "dae72b4aee35030f31ce42674d9535d6bff6c857b9beb8633a8ac809edaf031b",
}


def _git_blob(path: Path) -> bytes:
    relative = path.relative_to(REPO).as_posix()
    return subprocess.check_output(["git", "show", f"{COMMIT}:{relative}"], cwd=REPO)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_adapter():
    name = "r590_final_independent_review_adapter"
    spec = importlib.util.spec_from_file_location(name, ADAPTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def exact_candidate():
    adapter = _load_adapter()
    snapshot = adapter.capture_frozen_bytes()
    loaded_names = [name for name, _, _ in adapter.EXECUTABLE_LOAD_ORDER]
    loaded_names.append("r590_managed_producer")
    previous = {name: sys.modules.get(name) for name in loaded_names}
    producer = adapter.load_frozen_producer(snapshot)
    try:
        yield adapter, snapshot, producer
    finally:
        for name, prior in previous.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior


def test_exact_git_packet_and_dependency_closure(exact_candidate):
    adapter, snapshot, producer = exact_candidate
    for path, expected in EXPECTED.items():
        blob = _git_blob(path)
        assert _digest(blob) == expected
        assert path.read_bytes() == blob

    executable = {path for _, path, _ in adapter.EXECUTABLE_LOAD_ORDER}
    assert executable == {
        adapter.R584_RUNNER,
        adapter.R588_AUDITOR,
        adapter.RESULT_CONTRACT,
        adapter.FACADE,
        adapter.R576_RUNNER,
        adapter.R573_RUNNER,
        adapter.R582_HELPER,
        adapter.JACCLUST_PACKAGE,
        adapter.TT_MODEL,
    }
    assert executable <= set(adapter.FROZEN_HASHES)
    assert producer.r584.r582 is sys.modules[
        "numbered_list_cached_value_downstream_use_rung582"
    ]
    assert producer.r588.load_r582_helper() is producer.r584.r582
    assert _digest(snapshot[adapter.PRODUCER]) == EXPECTED[PRODUCER]


def test_call_price_and_phase_support_are_independently_reconstructed(exact_candidate):
    _, _, producer = exact_candidate
    rows = producer.load_outcome_blind_authority()
    assert len(rows) == 1_440
    expected_rows = {"FIT": 576, "SELECT": 288, "FINAL_TEST": 288, "OOD": 288}
    assert {
        split: sum(row["split"] == split for row in rows)
        for split in expected_rows
    } == expected_rows

    # Reconstruct the batches directly from row lengths and the frozen batch-24
    # grammar, rather than trusting the saved dry-run totals.
    def batch_count(split: str, *, null_only: bool = False) -> int:
        selected = [row for row in rows if row["split"] == split]
        if null_only:
            selected = [
                row for row in selected
                if row["condition"] in producer.r588.ELIGIBLE_CONDITIONS
            ]
        counts: dict[int, int] = {}
        for row in selected:
            counts[len(row["ids"])] = counts.get(len(row["ids"]), 0) + 1
        return sum((count + producer.BATCH - 1) // producer.BATCH for count in counts.values())

    fit_batches = batch_count("FIT")
    select_batches = batch_count("SELECT")
    fit_null_batches = batch_count("FIT", null_only=True)
    select_null_batches = batch_count("SELECT", null_only=True)
    assert (fit_batches, select_batches, fit_null_batches, select_null_batches) == (27, 14, 20, 10)
    # Capture uses two trajectories per batch plus one native smoke for the
    # first batch; each component/null arm then uses one call per batch.
    fit_always = 2 * fit_batches + 1 + 12 * fit_batches
    fit_provisional = fit_always + 2 * fit_null_batches
    select = 2 * select_batches + 1 + 3 * select_batches + 2 * select_null_batches
    assert (fit_always, fit_provisional, fit_provisional + select) == (379, 419, 510)

    census = producer.frozen_phase_support_census(rows, producer.AUTHORIZED_SPLITS)
    for split, count in expected_rows.items():
        assert census["splits"][split]["row_count"] == count
        assert census["splits"][split]["cell_count"] == 36


def test_model_free_plan_cannot_reach_prior_outcomes(exact_candidate, monkeypatch):
    _, _, producer = exact_candidate
    forbidden = {
        producer.r584.r582.R576_RESULT.resolve(),
        producer.r584.r582.R579_AUDIT.resolve(),
    }
    original_read_bytes = Path.read_bytes
    original_read_text = Path.read_text

    def reject_broad_authority(*_args, **_kwargs):
        raise AssertionError("R590 dry run reached a prior-outcome authority loader")

    def guarded_read_bytes(path):
        if path.resolve() in forbidden:
            raise AssertionError(f"R590 dry run opened prior outcome bytes: {path}")
        return original_read_bytes(path)

    def guarded_read_text(path, *args, **kwargs):
        if path.resolve() in forbidden:
            raise AssertionError(f"R590 dry run opened prior outcome text: {path}")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(producer.r588, "verify_preoutcome_authority", reject_broad_authority)
    monkeypatch.setattr(producer.r588, "load_authority", reject_broad_authority)
    monkeypatch.setattr(producer.r584, "load_authority", reject_broad_authority)
    monkeypatch.setattr(producer.r584.r582, "validate_authorities", reject_broad_authority)
    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    plan = producer.run_dryrun()
    assert plan["model_loaded"] is False and plan["cuda_opened"] is False
    assert plan["model_forwards"] == 0 and plan["model_backwards"] == 0


def test_nextafter_exactness_failure_is_not_a_scientific_null(exact_candidate):
    _, _, producer = exact_candidate
    evidence = producer.evidence_from_legacy_payload(
        producer.r588.make_fixture(held=False, replicates=8)
    )
    changed = copy.deepcopy(evidence)
    over = math.nextafter(producer.EXACT_BAR, math.inf)
    replay = changed["fit_capture_raw"][0][
        "native_replay_relative_squared_error_by_row"
    ]
    replay["source_present"] = over
    replay["maximum"] = over
    changed["fit_exactness"]["native_replay_relative_squared_error"] = over
    with pytest.raises(producer.UnretainedInstrumentError, match="publishable evidence"):
        producer.derive_scientific_summary(changed, replicates=8)


def test_rehashed_support_and_correlated_terminal_rewrites_fail(exact_candidate):
    _, _, producer = exact_candidate
    evidence = producer.evidence_from_legacy_payload(
        producer.r588.make_fixture(held=False, replicates=8)
    )
    support_attack = copy.deepcopy(evidence)
    fit = support_attack["phase_support_census"]["splits"]["FIT"]
    fit["ordered_row_ids"] = fit["ordered_row_ids"][1:] + fit["ordered_row_ids"][:1]
    fit["ordered_row_ids_sha256"] = producer.canonical_sha256(fit["ordered_row_ids"])
    support_attack["phase_support_census_sha256"] = producer.canonical_sha256(
        support_attack["phase_support_census"]
    )
    with pytest.raises(RuntimeError, match="phase_support"):
        producer.derive_scientific_summary(support_attack, replicates=8)

    result = producer.build_result(
        evidence,
        evidence_sha256=producer.canonical_sha256(evidence),
        checkpoint_sha256=producer.CHECKPOINT_SHA256,
        elapsed_seconds=1.0,
        replicates=8,
    )
    changed = copy.deepcopy(result)
    changed["decision"] = "downstream_use_component_held"
    changed["next_step"] = "invented_followup"
    changed_bytes = producer.canonical_bytes(changed)
    evidence_bytes = producer.canonical_bytes(evidence)
    receipt = producer.make_receipt(changed_bytes, evidence_bytes, changed)
    producer.validate_receipt(receipt, changed_bytes, evidence_bytes, changed)
    with pytest.raises(RuntimeError):
        producer.validate_result_against_evidence(changed, evidence, replicates=8)


def test_held_and_null_terminals_have_exact_phase_closure(exact_candidate):
    _, _, producer = exact_candidate
    cases = (
        (producer.r588.make_fixture(held=False, replicates=8), 379, ["FIT"], None),
        (producer.r588.make_fit_null_failure_fixture(replicates=8), 419, ["FIT"], None),
        (
            producer.r588.make_fixture(held=True, replicates=8),
            510,
            ["FIT", "SELECT"],
            "downstream_use_component_held",
        ),
    )
    for payload, calls, splits, held_decision in cases:
        evidence = producer.evidence_from_legacy_payload(payload)
        summary = producer.derive_scientific_summary(evidence, replicates=8)
        assert summary["model_forwards"] == calls
        assert summary["evaluated_splits"] == splits
        assert set(summary["phase_support_census"]["splits"]) == set(splits)
        if held_decision is not None:
            assert summary["decision"] == held_decision
        assert "FINAL_TEST" not in splits and "OOD" not in splits


def test_owned_scientific_namespaces_remain_absent(exact_candidate):
    adapter, _, _ = exact_candidate
    assert all(not path.exists() for path in adapter.OUTCOME_NAMESPACES)
