"""Independent pre-execution adversarial tests for committed R585.

The tests named ``test_blocker_*`` are deliberately failing repair contracts
for commit a4e7c46c6.  They must pass on a reviewed repair or adapter before any
model execution.  This file never loads the model or touches R585 outcomes.
"""

# BQLANE: cpu

from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest


OPS = Path(__file__).resolve().parent
ROOT = OPS.parent
RUNNER_PATH = OPS / "induction_selector_payload_frozen_factor_rung585.py"
OWNER_TEST_PATH = OPS / "test_induction_selector_payload_frozen_factor_rung585.py"
DRYRUN_PATH = ROOT / "induction_selector_payload_frozen_factor_rung585_dryrun.json"
ADAPTER_PATH = OPS / "execute_induction_selector_payload_frozen_factor_rung585.py"

RUNNER_SHA256 = "4911200ae12dd9c27a609879fded8aab1b5704ef1116f25079b5df7a40162ff3"
OWNER_TEST_SHA256 = "71eab693b578478d39201c267cbea7311972602aec739de19de85acab59ca67e"
DRYRUN_SHA256 = "9b1b8c7c6e66a6b4835fa9ad10219fee16583f34d8a72c41a803cf6be5bfab7d"
BLOCKED_REPAIR = pytest.mark.xfail(
    strict=True,
    reason="pre-execution repair contract intentionally unmet by a4e7c46c6",
)
REGISTERED_PREDICATES = {
    "pred_a_scientific_operation_exact":
        "the reviewed equality and non-equality operations are independently exact",
    "pred_b_realized_censuses_bound":
        "runtime operation and bootstrap censuses equal their frozen manifests",
    "pred_c_artifacts_fail_closed":
        "held evidence, provenance, finiteness, and receipt bindings fail closed",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_runner():
    name = "r585_implementation_adversarial_target"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return _load_runner()


@pytest.fixture(scope="module")
def execution(runner):
    return runner.build_execution_authority()


def test_exact_committed_producer_bytes(runner):
    assert _sha256(RUNNER_PATH) == RUNNER_SHA256
    assert _sha256(OWNER_TEST_PATH) == OWNER_TEST_SHA256
    assert _sha256(DRYRUN_PATH) == DRYRUN_SHA256
    assert runner.verify_authorities()


def test_exact_semantic_coordinates_and_unequal_lengths(runner, execution):
    rows = runner.strict_load_json(runner.ROWS)["rows"]
    included = [row for row in rows if row["family_id"] in runner.load_manifest().INCLUDED_FAMILIES
                and row["split"] in runner.SPLITS]
    assert len(included) == 2_808
    for row in included:
        for prefix in ("base", "donor"):
            ids = row[prefix + "_ids"]
            structure = row[prefix + "_structure"]
            assert [ids[index] for index in structure["source_positions"]] == \
                structure["source_ids"]
            assert [ids[index] for index in structure["payload_positions"]] == \
                structure["payload_ids"]
            assert ids[structure["query_position"]] == structure["query_id"]
            assert len(set(structure["source_positions"])) == 2
            assert len(set(structure["source_ids"])) == 2
            assert sum(source == structure["query_id"]
                       for source in structure["source_ids"]) in (0, 1)
    assert {row["length"] for row in execution["endpoints"]} == \
        {19, 20, 21, 22, 27, 28, 29, 30}


def test_frozen_terms_are_built_before_interventions_and_use_both_roles(runner):
    source = inspect.getsource(runner.run_science)
    assert source.index("collect_capture_replay") < source.index("build_frozen_insertion_cache")
    assert source.index("build_frozen_insertion_cache") < source.index("collect_intervention_arm")
    torch = pytest.importorskip("torch")
    one = torch.ones(1152)
    recipient = {"e": (2.0, 3.0), "u": (one, 10 * one)}
    donor = {"e": (5.0, 7.0), "u": (100 * one, 1000 * one)}
    expected = {"replay": 32, "score": 75, "payload": 3200, "joint": 7500}
    for arm, scalar in expected.items():
        observed = runner.combine_frozen_term(
            recipient, donor, arm, torch=torch, device="cpu"
        )
        assert torch.equal(observed, scalar * one)


def test_live_removal_and_same_state_l8_transaction_are_explicit(runner):
    factor_source = inspect.getsource(runner.factorize_attention_event)
    arm_source = inspect.getsource(runner.collect_intervention_arm)
    assert "live_removed = live[\"canonical\"]" in arm_source
    assert "delta = inserted - live_removed" in arm_source
    assert "for site, head in SITES" in factor_source
    assert "if site != event.site" in factor_source
    assert arm_source.count("factorize_attention_event(") == 1
    assert runner.SITES == ((5, 5), (7, 3), (8, 3), (8, 4))


def test_typed_fit_scales_and_full_control_lookup(runner, execution):
    planted = runner.planted_intervention_records(execution)
    scales = runner.compute_fit_scales(planted, execution["manifests"])
    assert len(scales) == 12
    assert all(set(scale) == {
        "target_cell_id", "insertion", "margin", "vocabulary", "valid"
    } for scale in scales.values())
    assert len(execution["control_scale_lookup"]) == 192
    assert all(scale["valid"] and scale["insertion"] > 0
               and scale["margin"] > 0 and scale["vocabulary"] > 0
               for scale in scales.values())


def test_bootstrap_trace_matches_independent_sha_draw(runner):
    cell = "FIT|family|variant|s0p0|base_to_donor|score|numerator_mean"
    values = {"g2": [3.0], "g0": [1.0], "g1": [2.0]}
    observed = runner.bootstrap_mean(values, cell, replicates=8)
    groups = tuple(sorted(values))
    draws = np.empty((8, 3), dtype=">u2")
    stats = np.empty(8, dtype=np.float64)
    namespace = runner.load_manifest().BOOTSTRAP_NAMESPACE
    for replicate in range(8):
        sample = []
        for draw in range(3):
            payload = f"{namespace}:{cell}:{replicate}:{draw}".encode()
            index = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 3
            draws[replicate, draw] = index
            sample.extend(values[groups[index]])
        stats[replicate] = sum(sample) / len(sample)
    assert observed["draw_sha256"] == hashlib.sha256(draws.tobytes()).hexdigest()
    assert observed["statistic_sha256"] == hashlib.sha256(
        stats.astype(">f8").tobytes()
    ).hexdigest()


def test_intact_score_split_realizes_the_frozen_124_bootstrap_ids(
    runner, execution, monkeypatch
):
    """The current formulas happen to realize the frozen census on intact input."""
    planted = runner.planted_intervention_records(execution)
    scales = runner.compute_fit_scales(planted, execution["manifests"])
    realized = []
    original = runner.bootstrap_mean

    def recording_bootstrap(values_by_group, cell_id, *, replicates=runner.BOOTSTRAPS):
        realized.append(cell_id)
        return original(values_by_group, cell_id, replicates=replicates)

    monkeypatch.setattr(runner, "bootstrap_mean", recording_bootstrap)
    runner.score_split(
        planted, "FIT", execution["manifests"], scales, replicates=2
    )
    expected = sorted(
        row["cell_id"] for row in execution["bootstrap_cells"]
        if row["cell_id"].startswith("FIT|")
    )
    assert len(realized) == len(set(realized)) == 124
    assert sorted(realized) == expected


def test_fit_first_prices_and_terminal_precedence(runner, execution):
    assert len(runner.endpoint_schedules(execution, "FIT")["capture"]) == 54
    assert len(runner.direction_batches(execution, "FIT")) == 117
    assert len(runner.endpoint_schedules(execution, "SELECT")["capture"]) == 27
    assert len(runner.direction_batches(execution, "SELECT")) == 59
    assert 54 + 3 * 117 + 54 == 459
    assert 27 + 3 * 59 + 27 == 231
    failures = {
        "invalid_instrument": ["i"],
        "factor_capacity_null": ["c"],
        "broad_contextual_equality_write": ["b"],
    }
    assert runner.terminal_from_failures(["FIT"], failures) == "invalid_instrument"


def test_current_ram_and_disk_cover_literal_upper_bound():
    available_ram = int(Path("/proc/meminfo").read_text().split("MemAvailable:")[1]
                        .splitlines()[0].split()[0]) * 1024
    disk = __import__("shutil").disk_usage("/workspace")
    # Review upper bounds: under 4 GiB transient CPU cache/serialization and
    # under 2 GiB atomic evidence staging plus final artifact.
    assert available_ram > 4 * 1024**3
    assert disk.free > 2 * 1024**3


def test_blocker_managed_no_arg_execution_adapter_exists_and_pins_producer():
    assert ADAPTER_PATH.is_file(), (
        "bqrunner executes queued files with no args; a reviewed adapter is required"
    )
    source = ADAPTER_PATH.read_text()
    assert RUNNER_SHA256 in source and OWNER_TEST_SHA256 in source and DRYRUN_SHA256 in source
    assert "--execute-science" in source
    assert 'environment.get("BQLIB_DRYRUN")' in source


@BLOCKED_REPAIR
def test_blocker_non_equality_remainder_is_independently_computed(runner):
    source = inspect.getsource(runner.factorize_attention_event)
    assert "contract_without_induction_fetch" in source
    assert "remainder = head_output - canonical_term" not in source


@BLOCKED_REPAIR
def test_blocker_explicit_endpoint_site_role_operation_manifest_is_frozen():
    dryrun = json.loads(DRYRUN_PATH.read_text())
    assert dryrun["census"]["endpoint_site_role_operations"] == {
        "FIT": 13_824, "SELECT": 6_912,
    }
    assert len(dryrun["manifest_hashes"]["endpoint_site_role_operation_sha256"]) == 64


@BLOCKED_REPAIR
def test_blocker_score_split_rejects_realized_bootstrap_census_omission(
    runner, execution
):
    """Expected metadata alone must not license an incomplete realized score."""
    planted = runner.planted_intervention_records(execution)
    scales = runner.compute_fit_scales(planted, execution["manifests"])
    incomplete = copy.deepcopy(execution["manifests"])
    first_fit_target = next(
        cell for cell in incomplete["target_cells"] if cell["split"] == "FIT"
    )
    incomplete["target_cells"].remove(first_fit_target)
    with pytest.raises((RuntimeError, ValueError), match="bootstrap.*census|census.*bootstrap"):
        runner.score_split(planted, "FIT", incomplete, scales, replicates=2)


@BLOCKED_REPAIR
def test_blocker_held_result_requires_complete_evidence(runner):
    result = runner.make_result_fixture("held_operational_selector_payload_factorization")
    result["evidence_files"] = []
    result["raw_evidence"] = {}
    with pytest.raises((ValueError, TypeError)):
        runner.validate_result(result)


@BLOCKED_REPAIR
def test_blocker_result_rejects_wrong_checkpoint_hash(runner):
    result = runner.make_result_fixture("factor_capacity_null")
    result["checkpoint_weights_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checkpoint"):
        runner.validate_result(result)


@BLOCKED_REPAIR
def test_blocker_receipt_rejects_wrong_result_path(runner):
    result = runner.make_result_fixture("factor_capacity_null")
    receipt = runner.make_receipt_fixture(result)
    receipt["result_path"] = "basis_aligned/bilinear_quotient/not_r585.json"
    with pytest.raises(ValueError, match="result_path"):
        runner.validate_receipt(receipt, result)


@BLOCKED_REPAIR
def test_blocker_primitive_identity_validator_rejects_nonfinite(runner):
    row = {
        "directed_id": "planted", "arm": "score",
        "answer_logit": float("nan"), "other_logit": 1.0,
        "correct_margin": 0.0, "log_normalizer": 2.0, "correct_ce": 1.0,
        "vocab_squared_difference_sum": 1.0, "vocab_size": 1,
        "vocab_rms": 1.0, "live_factor_max_error": 0.0,
        "hook_delta_sum_max_error": 0.0,
    }
    assert runner.validate_primitive_logit_identities([row]) != []
