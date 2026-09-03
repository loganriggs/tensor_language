"""CPU-only owner tests for the prospective R585 implementation."""

from __future__ import annotations

import copy
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT = Path(__file__).with_name("induction_selector_payload_frozen_factor_rung585.py")


def load_runner():
    name = "r585_owner_test_target"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return load_runner()


@pytest.fixture(scope="module")
def execution(runner):
    return runner.build_execution_authority()


@pytest.fixture(scope="module")
def planted(runner, execution):
    return runner.planted_intervention_records(execution)


def test_all_authorities_and_held_dependency_are_hash_pinned(runner):
    observed = runner.verify_authorities()
    assert observed == {str(path): digest for path, digest in runner.AUTHORITY_HASHES.items()}
    assert runner.AUTHORITY_HASHES[runner.DEPENDENCY_LOCK] == (
        "908826844336fe7a073ae16a5ef9123434514c21a73f8d3b331b4bab6e9f49b7"
    )
    assert runner.AUTHORITY_HASHES[runner.R586_RESULT] == (
        "14e7414bc7cf6b4a6a221079ac378752602b021b8b411124149dcc2c311666b8"
    )
    assert runner.AUTHORITY_HASHES[runner.R587_AUDIT] == (
        "72f0261fe32aa3d048c442ea1c08af932af6a368894610833e79aaaabf98bfe9"
    )


def test_exact_semantic_authority_and_canonical_census(execution):
    assert len(execution["endpoints"]) == 2_592
    assert len(execution["directions"]) == 5_616
    assert len({row["endpoint_id"] for row in execution["endpoints"]}) == 2_592
    for split, endpoint_count, direction_count in (("FIT", 1728, 3744), ("SELECT", 864, 1872)):
        assert sum(row["split"] == split for row in execution["endpoints"]) == endpoint_count
        assert sum(row["split"] == split for row in execution["directions"]) == direction_count
    assert all(len(row["source_positions"]) == len(row["payload_positions"]) == 2 for row in execution["endpoints"])
    assert all(
        all(payload == source + 1 for source, payload in zip(row["source_positions"], row["payload_positions"]))
        for row in execution["endpoints"]
    )
    assert all(row["final_position"] == row["length"] - 1 for row in execution["endpoints"])
    assert len(execution["manifests"]["target_cells"]) == 40
    assert len(execution["manifests"]["control_cells"]) == 64
    assert len(execution["manifests"]["coverage_keys"]) == 48
    assert len(execution["manifests"]["eligible_control_arm_cells"]) == 176
    assert len(execution["bootstrap_cells"]) == 248
    assert len(execution["control_scale_lookup"]) == 192
    assert len(execution["control_scale_lookup_sha256"]) == 64


def test_semantic_roles_survive_physical_pair_permutations(execution):
    # The R578 rows deliberately move A/C/N physically.  A/C coordinates remain
    # the first and second entries in source_positions/payload_positions.
    lengths = {row["length"] for row in execution["endpoints"]}
    assert lengths == {19, 20, 21, 22, 27, 28, 29, 30}
    match = [row for row in execution["directions"] if row["family"] == "match_break_payload_preserved"]
    assert {row["donor_coherence_sign"] for row in match if row["direction"] == "base_to_donor"} == {-1}
    assert {row["donor_coherence_sign"] for row in match if row["direction"] == "donor_to_base"} == {1}


def test_exact_batch_prices_and_padding_tripwires(runner, execution):
    for split, endpoint_calls, direction_calls in (("FIT", 54, 117), ("SELECT", 27, 59)):
        schedules = runner.endpoint_schedules(execution, split)
        assert len(schedules["capture"]) == len(schedules["comparator"]) == endpoint_calls
        assert len(runner.direction_batches(execution, split)) == direction_calls
    assert 54 + 3 * 117 + 54 == 459
    assert 27 + 3 * 59 + 27 == 231
    assert 459 + 231 == 690


def test_four_cached_factor_combinations_sum_both_roles(runner):
    torch = pytest.importorskip("torch")
    one = torch.ones(1152)
    recipient = {"e": (2.0, 3.0), "u": (one, 10 * one)}
    donor = {"e": (5.0, 7.0), "u": (100 * one, 1000 * one)}
    expected = {
        "replay": 2 * 1 + 3 * 10,
        "score": 5 * 1 + 7 * 10,
        "payload": 2 * 100 + 3 * 1000,
        "joint": 5 * 100 + 7 * 1000,
    }
    for arm, scalar in expected.items():
        value = runner.combine_frozen_term(recipient, donor, arm, torch=torch, device="cpu")
        assert torch.equal(value, scalar * one)
    direction = {
        "directed_id": "d", "recipient_endpoint_id": "x", "donor_endpoint_id": "y"
    }
    factors = {}
    for name in runner.TERM_NAMES:
        factors[("x", name)] = {**recipient, "canonical": expected["replay"] * one}
        factors[("y", name)] = {**donor, "canonical": expected["joint"] * one}
    frozen, failures = runner.build_frozen_insertion_cache([direction], factors, torch=torch)
    assert failures == []
    assert len(frozen) == 4 * 4
    assert torch.equal(frozen[("d", "joint", "L8H4")], expected["joint"] * one)


def test_recovery_is_ratio_of_cell_summaries_not_rowwise_ratios(runner):
    rows = [
        {"group_id": "a", "n": 1.0, "d": 1.0},
        {"group_id": "b", "n": 9.0, "d": 99.0},
    ]
    report = runner.recovery_summary(rows, "FIT|f|v|s0p0|base_to_donor|score", replicates=16)
    assert report["mean_recovery"] == pytest.approx(10 / 100)
    assert report["mean_recovery"] != pytest.approx((1 + 9 / 99) / 2)
    assert report["median_recovery"] == pytest.approx(5 / 50)


def test_distinct_scales_and_full_planted_gate(runner, execution, planted):
    scales = runner.compute_fit_scales(planted, execution["manifests"])
    assert len(scales) == 12
    assert all(set(row) == {"target_cell_id", "insertion", "margin", "vocabulary", "valid"} for row in scales.values())
    assert all(row["valid"] for row in scales.values())
    for split in runner.SPLITS:
        _, failures = runner.score_split(
            planted, split, execution["manifests"], scales, replicates=16
        )
        assert not any(failures.values())


def test_planted_scientific_null_fails_capacity_without_threshold_search(runner, execution):
    rows = runner.planted_intervention_records(execution, null=True)
    scales = runner.compute_fit_scales(rows, execution["manifests"])
    _, failures = runner.score_split(rows, "FIT", execution["manifests"], scales, replicates=16)
    assert failures["factor_capacity_null"]
    assert not failures["invalid_instrument"]


def test_terminal_precedence_and_fit_first_gate(runner):
    failures = {
        "factor_capacity_null": ["capacity"],
        "invalid_instrument": ["instrument"],
        "broad_contextual_equality_write": ["broad"],
    }
    assert runner.terminal_from_failures(["FIT"], failures) == "invalid_instrument"
    select = {"select_factor_capacity_null": ["select-capacity"]}
    assert runner.terminal_from_failures(["FIT", "SELECT"], select) == "select_factor_capacity_null"
    assert runner.terminal_from_failures(["FIT", "SELECT"], {}) == (
        "held_operational_selector_payload_factorization"
    )
    with pytest.raises(ValueError, match="SELECT"):
        runner.terminal_from_failures(["FIT"], {})


@pytest.mark.parametrize("terminal", [
    "held_operational_selector_payload_factorization",
    "factor_capacity_null",
    "invalid_instrument",
])
def test_strict_held_null_and_instrument_result_receipts(runner, terminal):
    result = runner.make_result_fixture(terminal)
    receipt = runner.make_receipt_fixture(result)
    runner.validate_result(result)
    runner.validate_receipt(receipt, result)
    json.dumps(result, allow_nan=False)
    json.dumps(receipt, allow_nan=False)


@pytest.mark.parametrize("mutation", ["tuple_next", "nan", "opened_ood", "price", "terminal"])
def test_result_schema_fails_closed(runner, mutation):
    result = runner.make_result_fixture("factor_capacity_null")
    if mutation == "tuple_next":
        result["next_step"] = (result["next_step"],)
    elif mutation == "nan":
        result["elapsed_seconds"] = float("nan")
    elif mutation == "opened_ood":
        result["forbidden_splits_opened"] = ["OOD"]
    elif mutation == "price":
        result["model_forwards"] = 460
    elif mutation == "terminal":
        result["terminal"] = "held_operational_selector_payload_factorization"
    with pytest.raises((ValueError, TypeError)):
        runner.validate_result(result)


def test_bootstrap_trace_is_sha_defined_and_big_endian(runner):
    cell = "FIT|family|variant|s0p0|base_to_donor|score|numerator_mean"
    values = {"g0": [1.0], "g1": [2.0], "g2": [3.0]}
    first = runner.bootstrap_mean(values, cell, replicates=8)
    second = runner.bootstrap_mean(dict(reversed(list(values.items()))), cell, replicates=8)
    assert first == second
    assert len(first["draw_sha256"]) == len(first["statistic_sha256"]) == 64


def test_primitive_logit_and_vocab_identities(runner):
    row = {
        "directed_id": "x", "arm": "score", "answer_logit": 3.0,
        "other_logit": 1.0, "correct_margin": 2.0, "log_normalizer": 4.0,
        "correct_ce": 1.0, "vocab_squared_difference_sum": 9.0,
        "vocab_size": 9, "vocab_rms": 1.0,
    }
    assert runner.validate_primitive_logit_identities([row]) == []
    broken = copy.deepcopy(row)
    broken["correct_margin"] = 99.0
    assert runner.validate_primitive_logit_identities([broken]) == ["primitive_margin:x:score"]


def test_scientific_execution_is_explicit_opt_in_and_model_import_is_lazy(runner):
    source = SCRIPT.read_text()
    assert "--execute-science" in source
    tree = ast.parse(source)
    top_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert not any(
        (isinstance(node, ast.Import) and any(alias.name == "torch" for alias in node.names))
        or (isinstance(node, ast.ImportFrom) and node.module == "torch")
        for node in top_imports
    )
    assert runner.OUT.name not in {runner.R586_RESULT.name, runner.R587_AUDIT.name}


def test_managed_enqueue_dryrun_and_registered_predictions(runner):
    assert set(runner.REGISTERED_PREDICATES) == {
        "pred_a_exact_factor_instrument",
        "pred_b_complete_joint_capacity",
        "pred_c_selector_payload_factorization",
        "pred_d_active_control_selectivity",
    }
    assert all(len(text) >= 12 for text in runner.REGISTERED_PREDICATES.values())
    source = SCRIPT.read_text()
    assert 'os.environ.get("BQLIB_DRYRUN") == "1"' in source


def test_deterministic_dryrun_is_model_free_and_split_closed(runner):
    dryrun = runner.run_dryrun()
    saved = json.loads(runner.DRYRUN.read_text())
    assert saved == dryrun
    assert dryrun["status"] == "deterministic_cpu_dryrun_passed"
    assert dryrun["model_loaded"] is False
    assert dryrun["cuda_opened"] is False
    assert dryrun["outcomes_opened"] == []
    assert dryrun["upstream_dependency_records_parsed"] == [
        str(runner.R586_RESULT), str(runner.R586_RECEIPT), str(runner.R587_AUDIT)
    ]
    assert dryrun["price"] == {
        "FIT": 459, "SELECT": 231, "maximum": 690, "backwards": 0, "updates": 0
    }
    assert dryrun["planted_terminals"] == {
        "held": "held_operational_selector_payload_factorization",
        "scientific_null": "factor_capacity_null",
        "instrument_failure": "invalid_instrument",
    }
