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
    assert len(execution["endpoint_site_role_operations"]) == 20_736
    assert execution["endpoint_site_role_operation_sha256"] == (
        "82169667d6f658b993f882b7b9951e07ae93149e5d5138fce548f6205e88cc5e"
    )
    assert {
        split: sum(row["split"] == split for row in execution["endpoint_site_role_operations"])
        for split in ("FIT", "SELECT")
    } == {"FIT": 13_824, "SELECT": 6_912}


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


@pytest.mark.parametrize("terminal", ["factor_capacity_null", "invalid_instrument"])
def test_strict_held_null_and_instrument_result_receipts(runner, terminal):
    result = runner.make_result_fixture(terminal)
    receipt = runner.make_receipt_fixture(result)
    runner.validate_result(result, allow_model_free_fixture=True)
    runner.validate_receipt(receipt, result)
    json.dumps(result, allow_nan=False)
    json.dumps(receipt, allow_nan=False)


def test_held_fixture_without_complete_evidence_is_rejected(runner):
    result = runner.make_result_fixture("held_operational_selector_payload_factorization")
    with pytest.raises(ValueError, match="model-free fixture"):
        runner.validate_result(result)


@pytest.mark.parametrize(
    "terminal",
    ["factor_capacity_null", "invalid_instrument", "select_factor_capacity_null"],
)
def test_completed_terminal_cannot_use_fixture_only_or_empty_evidence(runner, terminal):
    result = runner.make_result_fixture(terminal)
    assert result["evidence_files"] == []
    with pytest.raises(ValueError, match="model-free fixture"):
        runner.validate_result(result)


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
        runner.validate_result(result, allow_model_free_fixture=True)


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
    broken = copy.deepcopy(row)
    broken["answer_logit"] = float("nan")
    assert runner.validate_primitive_logit_identities([broken])[0].startswith(
        "nonfinite_primitive:"
    )


def test_independent_remainder_and_realized_operation_omission_fail_closed(runner, execution):
    source = SCRIPT.read_text()
    assert "contract_without_induction_fetch" in source
    assert "remainder = head_output - canonical_term" not in source
    expected = execution["endpoint_site_role_operations"]
    realized = [row for row in expected if row["split"] == "FIT"]
    assert runner.validate_realized_operations(expected, realized, "FIT") == {
        "count": 13_824,
        "sha256": runner.content_sha256(realized),
    }
    with pytest.raises(RuntimeError, match="operation census"):
        runner.validate_realized_operations(expected, realized[:-1], "FIT")


def test_phase_evidence_census_and_frozen_membership(runner, execution):
    fit = runner.phase_evidence_contract(["FIT"])
    assert fit["endpoint_count"] == 1_728
    assert fit["direction_count"] == 3_744
    assert fit["directed_arm_record_count"] == 11_232
    assert fit["factor_exactness_count"] == 6_912
    full = runner.phase_evidence_contract(["FIT", "SELECT"])
    assert full["endpoint_count"] == 2_592
    assert full["directed_arm_record_count"] == 16_848
    assert full["factor_exactness_count"] == 10_368
    identities = runner.expected_evidence_identities(["FIT"], execution)
    runner.validate_evidence_membership(
        ["FIT"], identities["endpoints"], identities["directed_arms"],
        identities["factors"], execution,
    )
    invented = copy.deepcopy(identities)
    invented["endpoints"][0] = "invented-self-consistent-endpoint"
    with pytest.raises(ValueError, match="frozen authority"):
        runner.validate_evidence_membership(
            ["FIT"], invented["endpoints"], invented["directed_arms"],
            invented["factors"], execution,
        )


def _write_mini_complete_evidence(
    runner, tmp_path, monkeypatch, *, mutation=None, terminal="invalid_instrument",
):
    endpoint = {
        "split": "FIT", "endpoint_id": "endpoint-0", "token_ids": [1, 2, 3, 4, 5],
        "length": 5, "final_position": 4, "source_positions": [0, 2],
        "payload_positions": [1, 3], "condition": "s0p0", "answer_id": 3,
        "other_answer_id": 4,
    }
    direction = {
        "split": "FIT", "directed_id": "direction-0", "row_id": "row-0",
        "group_id": "group-0", "family": "mini-family", "variant": "mini-variant",
        "direction": "base_to_donor", "recipient_condition": "s0p0",
        "recipient_is_coherent": True, "donor_is_coherent": True,
        "donor_coherence_sign": 1, "answer_changes": False, "control_kind": None,
        "recipient_endpoint_id": "endpoint-0", "donor_endpoint_id": "endpoint-0",
        "recipient_answer_id": 3, "donor_answer_id": 3,
        "recipient_other_answer_id": 4, "donor_other_answer_id": 4,
    }
    execution = {"endpoints": [endpoint], "directions": [direction], "manifests": {}}
    operations = sorted([
        {"split": "FIT", "endpoint_id": "endpoint-0", "site": site, "role": role}
        for site in runner.TERM_NAMES for role in runner.ROLES
    ], key=lambda row: (row["split"], row["endpoint_id"], row["site"], row["role"]))
    contract = {
        "evaluated_splits": ["FIT"], "endpoint_count": 1, "direction_count": 1,
        "directed_arm_record_count": 3, "factor_exactness_count": 4,
        "array_shapes": {
            "native_e.npy": [1, 4, 2], "native_u.npy": [1, 4, 2, 3],
            "canonical_term.npy": [1, 4, 3], "native_head_output.npy": [1, 4, 3],
            "non_equality_remainder.npy": [1, 4, 3],
            "live_removed.npy": [3, 4, 3], "hook_delta.npy": [3, 4, 3],
        },
        "jsonl_counts": {
            "endpoint_measurements.jsonl": 1,
            "directed_arm_measurements.jsonl": 3,
            "factor_exactness.jsonl": 4,
        },
    }
    monkeypatch.setattr(runner, "phase_evidence_contract", lambda _: contract)
    monkeypatch.setattr(runner, "build_execution_authority", lambda: execution)
    monkeypatch.setattr(runner, "build_endpoint_site_role_operations", lambda _: operations)
    monkeypatch.setattr(runner, "EXPECTED_OPERATION_COUNTS", {"FIT": 8})

    e = np.ones((1, 4, 2), dtype="<f4")
    u = np.ones((1, 4, 2, 3), dtype="<f4")
    canonical = np.sum(e[:, :, :, None] * u, axis=2)
    remainder = np.ones_like(canonical)
    live = np.repeat(canonical, 3, axis=0)
    delta = np.zeros_like(live)
    if mutation == "vectors":
        live[:] = 0
    arrays = {
        "native_e.npy": e, "native_u.npy": u, "canonical_term.npy": canonical,
        "native_head_output.npy": canonical + remainder,
        "non_equality_remainder.npy": remainder,
        "live_removed.npy": live, "hook_delta.npy": delta,
    }
    measurement = {
        "answer_id": 3, "other_answer_id": 4, "answer_logit": 2.0,
        "other_logit": 1.0, "correct_margin": 1.0, "log_normalizer": 3.0,
        "correct_ce": 1.0,
    }
    endpoint_rows = [{**endpoint, "replay": measurement, "native": measurement}]
    if mutation == "endpoint":
        endpoint_rows[0]["token_ids"] = [99, 98, 97]
    directed_rows = []
    for arm in sorted(runner.ARMS):
        row = {
            **{key: direction[key] for key in (
                "split", "directed_id", "row_id", "group_id", "family", "variant",
                "recipient_condition", "direction", "control_kind", "answer_changes",
                "recipient_endpoint_id", "donor_endpoint_id", "recipient_answer_id",
                "donor_answer_id",
            )},
            "other_answer_id": 4, "arm": arm, "replay_correct_margin": 1.0,
            "replay_correct_ce": 1.0, "correct_margin": 1.0, "correct_ce": 1.0,
            "n": 0.0, "d": 0.0, "q": 0.0, "insertion_activity": 0.0,
            "per_site_delta_norms": [0.0] * 4, "live_factor_max_error": 0.0,
            "hook_delta_sum_max_error": 0.0, "vocab_squared_difference_sum": 3.0,
            "vocab_size": 3, "vocab_rms": 1.0, "answer_logit": 2.0,
            "other_logit": 1.0, "log_normalizer": 3.0,
        }
        if mutation == "direction":
            row["donor_endpoint_id"] = "invented-donor"
        if mutation == "primitive":
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
        path.write_text("".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows
        ))
        descriptors.append({
            "path": str((runner.EVIDENCE_DIR / name).relative_to(runner.ROOT.parent.parent)),
            "sha256": runner.sha256(path), "bytes": path.stat().st_size,
            "dtype": "jsonl", "shape": [len(rows)],
            "row_order_sha256": runner.content_sha256(order),
        })
    result = runner.make_result_fixture(terminal)
    result["raw_evidence"] = {
        "schema": runner.EVIDENCE_SCHEMA, "endpoint_count": 1,
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
            "padding_tripwire_active_lengths": (
                [] if terminal == "invalid_instrument" else [19, 20, 21, 22, 27, 28, 29]
            ),
        },
        "fit_scales": {},
    }
    result["evidence_files"] = descriptors
    if terminal == "invalid_instrument":
        result["split_scores"] = {}
    return result, lambda logical: tmp_path / logical.name


def test_miniature_complete_evidence_joins_authority_and_computation(runner, tmp_path, monkeypatch):
    result, resolver = _write_mini_complete_evidence(runner, tmp_path, monkeypatch)
    runner.validate_result(result, artifact_path_resolver=resolver)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("endpoint", "endpoint semantic"),
        ("direction", "recipient/donor"),
        ("vectors", "live plus hook_delta"),
        ("primitive", "primitive logit"),
    ],
)
def test_miniature_complete_evidence_rejects_semantic_join_attacks(
    runner, tmp_path, monkeypatch, mutation, message,
):
    result, resolver = _write_mini_complete_evidence(
        runner, tmp_path, monkeypatch, mutation=mutation
    )
    with pytest.raises(ValueError, match=message):
        runner.validate_result(result, artifact_path_resolver=resolver)


def test_saved_score_report_is_recomputed_from_directed_primitives(runner, monkeypatch):
    rows = [{"split": "FIT", "n": 1.0}]
    scales = {"frozen": {"insertion": 1.0}}

    monkeypatch.setattr(runner, "compute_fit_scales", lambda records, manifests: scales)
    monkeypatch.setattr(
        runner, "score_split",
        lambda records, split, manifests, fit_scales, replicates: (
            {"primitive_projection": runner.content_sha256(records)},
            {
                "invalid_instrument": [], "native_denominator_or_scale_null": [],
                "factor_capacity_null": ["planted:factor_capacity_null"],
                "factorization_not_identified": [],
                "insufficient_active_controls": [], "broad_contextual_equality_write": [],
            },
        ),
    )
    result = runner.make_result_fixture("factor_capacity_null")
    result["raw_evidence"] = {"fit_scales": scales}
    result["split_scores"] = {
        "FIT": {"primitive_projection": runner.content_sha256(rows)}
    }
    runner._validate_saved_score_reports(result, rows, {"manifests": {}})
    result["split_scores"]["FIT"]["primitive_projection"] = "swapped-summary"
    with pytest.raises(ValueError, match="score report disagrees"):
        runner._validate_saved_score_reports(result, rows, {"manifests": {}})


def test_real_score_report_rebuild_rejects_summary_swap(
    runner, execution, planted, monkeypatch,
):
    monkeypatch.setattr(runner, "BOOTSTRAPS", 16)
    fit_rows = [row for row in planted if row["split"] == "FIT"]
    scales = runner.compute_fit_scales(fit_rows, execution["manifests"])
    report, failures = runner.score_split(
        fit_rows, "FIT", execution["manifests"], scales, replicates=16
    )
    result = runner.make_result_fixture("factor_capacity_null")
    result["raw_evidence"] = {"fit_scales": scales}
    result["split_scores"] = {"FIT": report}
    for label, clauses in failures.items():
        result["failure_classes"][label] = clauses
    runner._validate_saved_score_reports(result, fit_rows, execution)
    changed = copy.deepcopy(result)
    first_cell = next(iter(changed["split_scores"]["FIT"]["targets"]))
    changed["split_scores"]["FIT"]["targets"][first_cell]["passes"] ^= True
    with pytest.raises(ValueError, match="score report disagrees"):
        runner._validate_saved_score_reports(changed, fit_rows, execution)


def test_complete_scored_result_rejects_report_swap_with_fixed_jsonl(
    runner, tmp_path, monkeypatch,
):
    result, resolver = _write_mini_complete_evidence(
        runner, tmp_path, monkeypatch, terminal="factor_capacity_null"
    )
    scales = {"mini": {"valid": True}}
    failures = {
        "invalid_instrument": [], "native_denominator_or_scale_null": [],
        "factor_capacity_null": ["planted:factor_capacity_null"],
        "factorization_not_identified": [], "insufficient_active_controls": [],
        "broad_contextual_equality_write": [],
    }

    def recompute(records, split, manifests, fit_scales, replicates):
        return {
            "primitive_projection": runner.content_sha256(records),
            "bootstrap_realization": {"count": 1, "cell_ids_sha256": "mini"},
        }, failures

    monkeypatch.setattr(runner, "compute_fit_scales", lambda records, manifests: scales)
    monkeypatch.setattr(runner, "score_split", recompute)
    monkeypatch.setattr(
        runner, "validate_realized_bootstraps",
        lambda report, split, manifests: report["bootstrap_realization"],
    )
    directed = json.loads((tmp_path / "directed_arm_measurements.jsonl").read_text().splitlines()[0])
    all_directed = runner._strict_jsonl(tmp_path / "directed_arm_measurements.jsonl")
    assert directed in all_directed
    report, _ = recompute(all_directed, "FIT", {}, scales, runner.BOOTSTRAPS)
    result["raw_evidence"]["fit_scales"] = scales
    result["split_scores"] = {"FIT": report}
    runner.validate_result(result, artifact_path_resolver=resolver)
    result["split_scores"]["FIT"]["primitive_projection"] = "swapped-report"
    with pytest.raises(ValueError, match="score report disagrees"):
        runner.validate_result(result, artifact_path_resolver=resolver)


def test_realized_bootstrap_omission_and_provenance_fail_closed(runner, execution, planted):
    scales = runner.compute_fit_scales(planted, execution["manifests"])
    incomplete = copy.deepcopy(execution["manifests"])
    cell = next(row for row in incomplete["target_cells"] if row["split"] == "FIT")
    incomplete["target_cells"].remove(cell)
    with pytest.raises(RuntimeError, match="bootstrap.*census|census.*bootstrap"):
        runner.score_split(planted, "FIT", incomplete, scales, replicates=2)
    result = runner.make_result_fixture("factor_capacity_null")
    result["checkpoint_weights_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checkpoint"):
        runner.validate_result(result)
    result = runner.make_result_fixture("factor_capacity_null")
    receipt = runner.make_receipt_fixture(result)
    receipt["result_path"] = "basis_aligned/bilinear_quotient/not-r585.json"
    with pytest.raises(ValueError, match="result_path"):
        runner.validate_receipt(receipt, result)


@pytest.mark.parametrize("crash_after", ["evidence", "result", "receipt"])
def test_staged_publication_rolls_back_every_injected_crash(runner, tmp_path, crash_after):
    stage = runner.create_stage_root(tmp_path)
    (stage / "evidence").mkdir()
    (stage / "evidence" / "proof").write_text("complete\n")
    (stage / "result.json").write_text("{}")
    (stage / "receipt.json").write_text("{}")
    out = tmp_path / "result.json"
    receipt = tmp_path / "receipt.json"
    evidence = tmp_path / "evidence"

    def crash(label):
        if label == crash_after:
            raise RuntimeError(f"crash-after-{label}")

    with pytest.raises(RuntimeError, match="crash-after"):
        runner.publish_staged_package(
            stage, out=out, receipt=receipt, evidence=evidence, crash_injector=crash
        )
    assert not out.exists() and not receipt.exists() and not evidence.exists()
    assert (stage / "result.json").is_file()
    assert (stage / "receipt.json").is_file()
    assert (stage / "evidence" / "proof").is_file()


def test_stale_partial_publication_is_quarantined_not_deleted(runner, tmp_path):
    out = tmp_path / "result.json"
    receipt = tmp_path / "receipt.json"
    evidence = tmp_path / "evidence"
    stage = runner.create_stage_root(tmp_path)
    result = runner.make_result_fixture("factor_capacity_null")
    out.write_text(json.dumps(result, sort_keys=True, allow_nan=False))
    with pytest.raises(RuntimeError, match="recovered incomplete"):
        runner.recover_stale_publication(
            root=tmp_path, out=out, receipt=receipt, evidence=evidence
        )
    assert not out.exists() and not receipt.exists() and not evidence.exists()
    recovered = list(tmp_path.glob(runner.RECOVERY_PREFIX + "*"))
    assert len(recovered) == 1
    assert any(path.name.startswith("partial-result-")
               for path in recovered[0].iterdir() if path.is_file())


def test_stale_recovery_refuses_arbitrary_or_complete_namespaces(runner, tmp_path):
    out = tmp_path / "result.json"
    receipt = tmp_path / "receipt.json"
    evidence = tmp_path / "evidence"
    out.write_text("arbitrary scientific-looking bytes\n")
    stage = runner.create_stage_root(tmp_path)
    with pytest.raises(RuntimeError, match="unrecognized"):
        runner.recover_stale_publication(
            root=tmp_path, out=out, receipt=receipt, evidence=evidence
        )
    assert out.read_text() == "arbitrary scientific-looking bytes\n"
    assert stage.exists()
    assert not list(tmp_path.glob(runner.RECOVERY_PREFIX + "*"))

    out.unlink()
    stage.rename(tmp_path / "not-a-stage-anymore")
    out.write_text("result")
    receipt.write_text("receipt")
    evidence.mkdir()
    with pytest.raises(RuntimeError, match="complete output namespace"):
        runner.recover_stale_publication(
            root=tmp_path, out=out, receipt=receipt, evidence=evidence
        )
    assert out.read_text() == "result" and receipt.read_text() == "receipt"
    assert evidence.is_dir()


@pytest.mark.parametrize("label", ["staged-result-write", "staged-receipt-write"])
def test_staged_file_write_crash_is_injected_after_fsync(runner, tmp_path, label):
    path = tmp_path / (label + ".json")

    def crash(observed):
        assert observed == label
        raise RuntimeError("planted-write-crash")

    with pytest.raises(RuntimeError, match="planted-write-crash"):
        runner._write_bytes_fsync(
            path, b"{}", label=label, crash_injector=crash
        )
    assert path.read_bytes() == b"{}"


def test_evidence_write_crash_never_touches_final_namespace(runner, tmp_path):
    torch = pytest.importorskip("torch")
    endpoint = {"split": "FIT", "endpoint_id": "planted-endpoint"}
    one = torch.ones(1152)
    factors = {}
    for name in runner.TERM_NAMES:
        factors[(endpoint["endpoint_id"], name)] = {
            "e": (1.0, 2.0), "u": (one, one),
            "canonical": 3 * one, "head_output": 4 * one,
            "remainder": one, "factor_error": 0.0,
            "reconstruction_error": 0.0,
        }
    stage_evidence = tmp_path / "stage-evidence"

    def crash(label):
        if label == "evidence-write:native_e.npy":
            raise RuntimeError("planted-evidence-crash")

    with pytest.raises(RuntimeError, match="planted-evidence-crash"):
        runner.write_evidence(
            {"endpoints": [endpoint]}, factors, [], [], {}, {},
            evidence_dir=stage_evidence, crash_injector=crash,
        )
    assert (stage_evidence / "native_e.npy").is_file()
    assert not runner.EVIDENCE_DIR.exists()


def test_nonfinite_result_is_rejected_before_any_staged_or_final_write(runner, tmp_path):
    result = runner.make_result_fixture("factor_capacity_null")
    result["elapsed_seconds"] = float("nan")
    stage = runner.create_stage_root(tmp_path)
    with pytest.raises(ValueError, match="nonfinite"):
        runner._finish_result(result, stage)
    assert [path.name for path in stage.iterdir()] == [runner.STAGE_MARKER_NAME]
    assert not runner.OUT.exists() and not runner.RECEIPT.exists() and not runner.EVIDENCE_DIR.exists()


def test_receipt_binds_canonical_path_and_exact_result_bytes(runner, tmp_path):
    result = runner.make_result_fixture("factor_capacity_null")
    receipt = runner.make_receipt_fixture(result)
    exact = tmp_path / "result.json"
    exact.write_bytes(json.dumps(
        result, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode())
    runner.validate_receipt(receipt, result, result_file=exact)
    exact.write_text(json.dumps(result, indent=2, sort_keys=True))
    with pytest.raises(ValueError, match="exact result bytes"):
        runner.validate_receipt(receipt, result, result_file=exact)


def test_nonfinite_staged_array_cannot_validate_even_for_null_terminal(runner, tmp_path):
    evidence = tmp_path / "poison.npy"
    np.save(evidence, np.asarray([float("nan")], dtype="<f4"), allow_pickle=False)
    result = runner.make_result_fixture("factor_capacity_null")
    result["evidence_files"] = [{
        "path": "basis_aligned/bilinear_quotient/poison.npy",
        "sha256": runner.sha256(evidence),
        "bytes": evidence.stat().st_size,
        "dtype": "<f4",
        "shape": [1],
        "row_order_sha256": runner.content_sha256(["planted"]),
    }]
    with pytest.raises(ValueError, match="nonfinite evidence array"):
        runner.validate_result(
            result, artifact_path_resolver=lambda _: evidence,
            allow_model_free_fixture=True,
        )


def test_complete_staged_package_publishes_receipt_last(runner, tmp_path):
    stage = runner.create_stage_root(tmp_path)
    (stage / "evidence").mkdir()
    (stage / "evidence" / "proof").write_text("complete\n")
    (stage / "result.json").write_text("result\n")
    (stage / "receipt.json").write_text("receipt\n")
    out = tmp_path / "final-result.json"
    receipt = tmp_path / "final-receipt.json"
    evidence = tmp_path / "final-evidence"
    order = []
    runner.publish_staged_package(
        stage, out=out, receipt=receipt, evidence=evidence,
        crash_injector=order.append,
    )
    assert order == ["evidence", "result", "receipt"]
    assert out.read_text() == "result\n"
    assert receipt.read_text() == "receipt\n"
    assert (evidence / "proof").read_text() == "complete\n"
    assert not stage.exists()


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
    assert dryrun["evidence_contract"]["independent_remainder"] == (
        "contract_without_induction_fetch"
    )
    assert dryrun["evidence_contract"]["endpoint_semantics_equal_frozen_authority"] is True
    assert dryrun["evidence_contract"]["directed_semantics_equal_frozen_authority"] is True
    assert dryrun["evidence_contract"]["inserted_term_reconstructed_from_saved_factors"] is True
    assert dryrun["evidence_contract"]["primitive_and_sufficient_statistics_recomputed"] is True
    assert dryrun["evidence_contract"]["scored_reports_recomputed_from_directed_rows"] is True
    assert dryrun["evidence_contract"]["finite_before_final_write"] is True
    assert dryrun["publication_contract"]["atomic_renames"] == [
        "evidence", "result", "receipt",
    ]
    assert dryrun["publication_contract"]["receipt_is_commit_marker"] is True
    assert dryrun["planted_terminals"] == {
        "held": "held_operational_selector_payload_factorization",
        "scientific_null": "factor_capacity_null",
        "instrument_failure": "invalid_instrument",
    }
