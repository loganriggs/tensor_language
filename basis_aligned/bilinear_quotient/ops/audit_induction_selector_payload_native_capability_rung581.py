#!/usr/bin/env python3
"""R581: independent, model-free audit of the future R580 result."""

# BQLANE: cpu

from __future__ import annotations

import collections
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
ROWS = ROOT / "induction_selector_payload_three_source_rows_rung578.json"
ROWS_RECEIPT = ROOT / "induction_selector_payload_three_source_rows_rung578_receipt.json"
ROWS_PREREG = POLY / "INDUCTION_SELECTOR_PAYLOAD_THREE_SOURCE_ROWS_RUNG578_PREREGISTRATION.md"
ROWS_BUILDER = ROOT / "ops" / "induction_selector_payload_three_source_rows_rung578.py"
ROWS_TEST = ROOT / "ops" / "test_induction_selector_payload_three_source_rows_rung578.py"
R580_PREREG = POLY / "INDUCTION_SELECTOR_PAYLOAD_NATIVE_CAPABILITY_RUNG580_PREREGISTRATION.md"
R580_SCRIPT = ROOT / "ops" / "induction_selector_payload_native_capability_rung580.py"
R580_TEST = ROOT / "ops" / "test_induction_selector_payload_native_capability_rung580.py"
R580_DRYRUN = ROOT / "induction_selector_payload_native_capability_rung580_dryrun.json"
R580_RESULT = ROOT / "induction_selector_payload_native_capability_rung580_results.json"
R580_RECEIPT = ROOT / "induction_selector_payload_native_capability_rung580_receipt.json"
PREREG = POLY / "INDUCTION_SELECTOR_PAYLOAD_NATIVE_CAPABILITY_AUDIT_RUNG581_PREREGISTRATION.md"
SCRIPT = Path(__file__)
TEST = SCRIPT.with_name("test_audit_induction_selector_payload_native_capability_rung581.py")
OUT = ROOT / "induction_selector_payload_native_capability_audit_rung581.json"
DRYRUN = ROOT / "induction_selector_payload_native_capability_audit_rung581_dryrun.json"

AUTHORITY_HASHES = {
    ROWS: "8893ff83ea6080ad704f38376715d19be8971867178a4edc3bfd61fe025b39b6",
    ROWS_RECEIPT: "9e4e63ebd98503d6aa5daa27617a20fea595829c5a372f27b1ce4371d7c05b45",
    ROWS_PREREG: "276d801bbf5795e6421488dd4971b3a2d2dcb56e4fc7c4bc7ecdd2f61a73e9ce",
    ROWS_BUILDER: "d47bb3d46bd2c6061132c13b356e58ba9dfe2a56a2629f8b49a03f280d290bbd",
    ROWS_TEST: "9d795df358dfef9c5d17a539307f8e781f2a4debeb4909078858a242b3dfc512",
    R580_PREREG: "8f80926d0a90360a66ebce605732d32ff3e283a3428eb7245f4813a521d12580",
    R580_SCRIPT: "62d11395d845d663257433936773780dd4bb9ddbcb9286400c420dadd3a73249",
    R580_TEST: "9f166a61409c12d6a4a58e16640af654378151f99c05597f9c63dbb2dec64550",
    R580_DRYRUN: "3d21b62972aa0794598860228554068035af10fd743e8958bfc7a05d56d68588",
    PREREG: "d2989383791cb179fecfa930742812cf8036a85bb9d2f3cfdd6555bb00640887",
}
R580_INPUT_HASHES = {
    str(path): digest
    for path, digest in AUTHORITY_HASHES.items()
    if path not in {R580_SCRIPT, R580_TEST, R580_DRYRUN, PREREG}
}
CHECKPOINT_SHA256 = "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
SPLITS = ("FIT", "SELECT")
CONDITIONS = ("s0p0", "s0p1", "s1p0", "s1p1")
CONTROL_FAMILIES = {
    "neutral_source": ("irrelevant_source_edit", None),
    "neutral_payload": ("irrelevant_payload_edit", None),
    "filler": ("copy_relation_preserved_nuisance_change", "filler_change"),
    "lag": ("copy_relation_preserved_nuisance_change", "lag_extension"),
}
BOOTSTRAP_NAMESPACE = "a8-r580-group-bootstrap-v1"
BOOTSTRAPS = 2_000
EXPECTED_GROUPS = 108
EXPECTED_ROWS = 3_240
EXPECTED_SEQUENCES = 3_024
EXPECTED_FORWARDS = 95
ABS_TOLERANCE = 1e-12
BOOTSTRAP_ALGORITHM = {
    "namespace": BOOTSTRAP_NAMESPACE,
    "group_order": "lexicographic group_id",
    "payload": "{namespace}:{cell_id}:{replicate}:{draw}",
    "index": "uint64_big_endian(SHA256(payload)[0:8]) modulo group_count",
    "cluster_rule": "selected group contributes every observation in that cell",
    "lower_quantile": "numpy.quantile(0.025, method=lower)",
    "upper_quantile": "numpy.quantile(0.975, method=higher); contrast cells only",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def content_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def sequence_id(ids: Sequence[int]) -> str:
    return content_sha256({"token_ids": list(ids)})


def load_authority() -> tuple[list[dict], list[dict], dict[str, dict]]:
    for path, digest in AUTHORITY_HASHES.items():
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"frozen authority mismatch: {path}")
    document = json.loads(ROWS.read_text())
    receipt = json.loads(ROWS_RECEIPT.read_text())
    r580_dryrun = json.loads(R580_DRYRUN.read_text())
    if receipt["rows_sha256"] != AUTHORITY_HASHES[ROWS]:
        raise RuntimeError("R578 receipt does not bind the rows")
    if document["model_loaded"] is not False or document["outcomes_opened"] != []:
        raise RuntimeError("R578 construction opened model evidence")
    if not (r580_dryrun["status"] == "dryrun_passed"
            and r580_dryrun["implementation_sha256"] == AUTHORITY_HASHES[R580_SCRIPT]
            and r580_dryrun["test_sha256"] == AUTHORITY_HASHES[R580_TEST]
            and r580_dryrun["preregistration_sha256"] == AUTHORITY_HASHES[R580_PREREG]
            and r580_dryrun["model_loaded"] is False
            and r580_dryrun["model_forwards"] == 0):
        raise RuntimeError("R580 dry-run does not bind the reviewed instrument")
    groups = [item for item in document["groups"] if item["split"] in SPLITS]
    rows = [item for item in document["rows"] if item["split"] in SPLITS]
    by_group = {item["group_id"]: item for item in groups}
    if len(groups) != len(by_group) or len(groups) != EXPECTED_GROUPS:
        raise RuntimeError("group census mismatch")
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError("row census mismatch")
    if {item["split"] for item in groups} != set(SPLITS):
        raise RuntimeError("FIT/SELECT authority mismatch")
    if any(item["group_id"] not in by_group or item["split"] != by_group[item["group_id"]]["split"]
           for item in rows):
        raise RuntimeError("row/group membership mismatch")
    return groups, rows, by_group


def expected_sequence_specs(groups: Sequence[dict], rows: Sequence[dict]) -> list[dict]:
    owners: dict[tuple[int, ...], str] = {}
    answers: dict[tuple[int, ...], int] = {}

    def add(group_id: str, ids_raw: Sequence[int], answer_id: int) -> None:
        ids = tuple(ids_raw)
        if ids in owners and owners[ids] != group_id:
            raise RuntimeError("sequence crosses groups")
        if ids in answers and answers[ids] != answer_id:
            raise RuntimeError("sequence has inconsistent answers")
        owners[ids], answers[ids] = group_id, answer_id

    by_group = {item["group_id"]: item for item in groups}
    for group in groups:
        for cell in group["factorial_conditions"].values():
            add(group["group_id"], cell["ids"], cell["answer_id"])
    for row in rows:
        add(row["group_id"], row["base_ids"], row["base_answer_id"])
        add(row["group_id"], row["donor_ids"], row["donor_answer_id"])
    ordered = sorted(owners, key=lambda ids: (len(ids), ids))
    if len(ordered) != EXPECTED_SEQUENCES:
        raise RuntimeError("sequence census mismatch")
    output = []
    for ids in ordered:
        group = by_group[owners[ids]]
        output.append({
            "sequence_id": sequence_id(ids), "group_id": owners[ids], "split": group["split"],
            "length": len(ids), "final_position": len(ids) - 1,
            "registered_answer_id": answers[ids],
            "token_b_id": group["variable_token_ids"]["B"],
            "token_d_id": group["variable_token_ids"]["D"],
        })
    if len({item["sequence_id"] for item in output}) != EXPECTED_SEQUENCES:
        raise RuntimeError("sequence ID collision")
    return output


def correct_margin(item: Mapping[str, object], answer_id: int) -> float:
    sign = 1.0 if answer_id == item["token_b_id"] else -1.0
    if answer_id not in (item["token_b_id"], item["token_d_id"]):
        raise RuntimeError("answer is outside B/D")
    return sign * (float(item["logit_b"]) - float(item["logit_d"]))


def correct_ce(item: Mapping[str, object], answer_id: int) -> float:
    if answer_id == item["token_b_id"]:
        return float(item["ce_b"])
    if answer_id == item["token_d_id"]:
        return float(item["ce_d"])
    raise RuntimeError("answer is outside B/D")


def condition_name(group: Mapping[str, object], condition_id: str) -> str:
    names = [name for name, item in group["factorial_conditions"].items()
             if item["condition_id"] == condition_id]
    if len(names) != 1:
        raise RuntimeError("base condition does not identify one factorial cell")
    return names[0]


def validate_sequence_measurements(specs: Sequence[dict], measurements: Sequence[dict]) -> None:
    if len(measurements) != EXPECTED_SEQUENCES:
        raise RuntimeError("sequence measurement census mismatch")
    expected = {item["sequence_id"]: item for item in specs}
    observed = {item["sequence_id"]: item for item in measurements}
    if len(observed) != EXPECTED_SEQUENCES or set(observed) != set(expected):
        raise RuntimeError("sequence membership mismatch")
    keys = ("sequence_id", "group_id", "split", "length", "final_position",
            "registered_answer_id", "token_b_id", "token_d_id")
    for identity, spec in expected.items():
        item = observed[identity]
        if any(item[key] != spec[key] for key in keys):
            raise RuntimeError(f"sequence metadata mismatch: {identity}")
        numbers = [item[key] for key in ("logit_b", "logit_d", "log_normalizer", "ce_b", "ce_d")]
        if not all(math.isfinite(float(value)) for value in numbers):
            raise RuntimeError(f"nonfinite sequence measurement: {identity}")
        if abs(float(item["ce_b"]) - (float(item["log_normalizer"]) - float(item["logit_b"]))) > ABS_TOLERANCE:
            raise RuntimeError(f"ce_b identity mismatch: {identity}")
        if abs(float(item["ce_d"]) - (float(item["log_normalizer"]) - float(item["logit_d"]))) > ABS_TOLERANCE:
            raise RuntimeError(f"ce_d identity mismatch: {identity}")


def reconstruct_raw(groups: Sequence[dict], rows: Sequence[dict], specs: Sequence[dict],
                    measurements: Sequence[dict]) -> dict:
    validate_sequence_measurements(specs, measurements)
    by_group = {item["group_id"]: item for item in groups}
    by_sequence = {item["sequence_id"]: item for item in measurements}
    row_measurements, row_index = [], {}
    for row in rows:
        group = by_group[row["group_id"]]
        condition = condition_name(group, row["base_condition_id"])
        base_id, donor_id = sequence_id(row["base_ids"]), sequence_id(row["donor_ids"])
        base, donor = by_sequence[base_id], by_sequence[donor_id]
        base_margin = correct_margin(base, row["base_answer_id"])
        donor_margin = correct_margin(donor, row["donor_answer_id"])
        item = {
            "row_id": row["row_id"], "group_id": row["group_id"], "split": row["split"],
            "family_id": row["family_id"], "family_variant": row["family_variant"],
            "condition": condition, "base_sequence_id": base_id, "donor_sequence_id": donor_id,
            "base_answer_id": row["base_answer_id"], "donor_answer_id": row["donor_answer_id"],
            "base_margin": base_margin, "donor_margin": donor_margin,
            "base_ce": correct_ce(base, row["base_answer_id"]),
            "donor_ce": correct_ce(donor, row["donor_answer_id"]),
            "donor_minus_base_margin": donor_margin - base_margin,
            "answer_changes": row["answer_changes"],
        }
        row_measurements.append(item)
        key = (row["group_id"], row["family_id"], condition)
        if key in row_index:
            key = (*key, row["family_variant"].split(":")[-1])
        if key in row_index:
            raise RuntimeError("duplicate row index")
        row_index[key] = item

    factorial, condition_effects = [], []
    for group in groups:
        cells = {}
        for condition in CONDITIONS:
            authority = group["factorial_conditions"][condition]
            item = by_sequence[sequence_id(authority["ids"])]
            cells[condition] = {
                "sequence_id": item["sequence_id"], "answer_id": authority["answer_id"],
                "correct_margin": correct_margin(item, authority["answer_id"]),
                "correct_ce": correct_ce(item, authority["answer_id"]),
                "z_b_minus_d": float(item["logit_b"]) - float(item["logit_d"]),
            }
        factorial.append({
            "group_id": group["group_id"], "split": group["split"], "cells": cells,
            "selector_payload_interaction": (
                cells["s0p0"]["z_b_minus_d"] - cells["s1p0"]["z_b_minus_d"]
                - cells["s0p1"]["z_b_minus_d"] + cells["s1p1"]["z_b_minus_d"]
            ) / 4.0,
        })
        for condition in CONDITIONS:
            prefix = (group["group_id"],)
            selected = row_index[prefix + ("match_break_payload_preserved", condition)]
            neutral_source = row_index[prefix + ("irrelevant_source_edit", condition)]
            neutral_payload = row_index[prefix + ("irrelevant_payload_edit", condition)]
            contrast = row_index[prefix + ("contrast_target_source_edit", condition)]
            base_ids = {item["base_sequence_id"] for item in
                        (selected, neutral_source, neutral_payload, contrast)}
            if len(base_ids) != 1:
                raise RuntimeError("condition controls do not share one base")
            base_margin = selected["base_margin"]
            selected_drop = base_margin - selected["donor_margin"]
            source_effect = abs(base_margin - neutral_source["donor_margin"])
            payload_effect = abs(base_margin - neutral_payload["donor_margin"])
            condition_effects.append({
                "group_id": group["group_id"], "split": group["split"], "condition": condition,
                "selected_match_drop": selected_drop,
                "neutral_source_absolute_effect": source_effect,
                "neutral_payload_absolute_effect": payload_effect,
                "selected_vs_neutral_gap": selected_drop - max(source_effect, payload_effect),
                "contrast_source_signed_margin_change": contrast["donor_margin"] - base_margin,
            })
    return {
        "sequence_measurements": list(measurements),
        "row_measurements": row_measurements,
        "group_factorial_measurements": factorial,
        "group_condition_effect_measurements": condition_effects,
    }


def bootstrap_index(cell_id: str, replicate: int, draw: int, count: int) -> int:
    payload = f"{BOOTSTRAP_NAMESPACE}:{cell_id}:{replicate}:{draw}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % count


def bootstrap(values_by_group: Mapping[str, Sequence[float]], cell_id: str,
              traces: dict[str, dict], *, two_sided: bool = False,
              replicates: int = BOOTSTRAPS) -> dict:
    group_ids = tuple(sorted(values_by_group))
    if not group_ids or any(not values_by_group[group] for group in group_ids):
        raise RuntimeError(f"empty bootstrap cell: {cell_id}")
    means = np.empty(replicates, dtype=np.float64)
    draws = np.empty((replicates, len(group_ids)), dtype=np.uint16)
    for replicate in range(replicates):
        total, observations = 0.0, 0
        for draw in range(len(group_ids)):
            index = bootstrap_index(cell_id, replicate, draw, len(group_ids))
            draws[replicate, draw] = index
            values = values_by_group[group_ids[index]]
            total += sum(float(value) for value in values)
            observations += len(values)
        means[replicate] = total / observations
    flat = [float(value) for group in group_ids for value in values_by_group[group]]
    if not all(math.isfinite(value) for value in flat):
        raise RuntimeError(f"nonfinite bootstrap cell: {cell_id}")
    draw_bytes = draws.astype(">u2", copy=False).tobytes(order="C")
    statistic_bytes = means.astype(">f8", copy=False).tobytes(order="C")
    traces[cell_id] = {
        "ordered_group_ids": list(group_ids), "replicates": replicates,
        "draw_count": int(draws.size), "draw_matrix_dtype": "big-endian uint16 row-major",
        "draw_matrix_sha256": hashlib.sha256(draw_bytes).hexdigest(),
        "statistic_dtype": "big-endian float64 row-major",
        "statistic_vector_sha256": hashlib.sha256(statistic_bytes).hexdigest(),
    }
    report = {
        "namespace": BOOTSTRAP_NAMESPACE, "cell_id": cell_id, "replicates": replicates,
        "ordered_group_ids": list(group_ids), "observation_count": len(flat),
        "point_mean": float(np.mean(flat)),
        "lower95": float(np.quantile(means, .025, method="lower")),
    }
    if two_sided:
        report["upper95"] = float(np.quantile(means, .975, method="higher"))
    return report


def accuracy(values: Mapping[str, Sequence[float]], cell_id: str, traces: dict,
             replicates: int) -> dict:
    flat = [float(value) for group in sorted(values) for value in values[group]]
    interval = bootstrap(values, cell_id, traces, replicates=replicates)
    report = {"group_count": len(values), "observation_count": len(flat),
              "correct_fraction": float(np.mean(np.asarray(flat) > 0)),
              "mean_margin": float(np.mean(flat)), "bootstrap": interval}
    report["passes"] = bool(report["correct_fraction"] >= .75 and interval["lower95"] > 0)
    return report


def control_matches(label: str, row: dict) -> bool:
    family, variant = CONTROL_FAMILIES[label]
    return row["family_id"] == family and (variant is None or row["family_variant"].endswith(f":{variant}"))


def score(raw: Mapping[str, object], *, replicates: int = BOOTSTRAPS) -> tuple[dict, dict]:
    factorial_rows = raw["group_factorial_measurements"]
    row_measurements = raw["row_measurements"]
    effects = raw["group_condition_effect_measurements"]
    failures, factorial_reports, interactions = [], {}, {}
    controls, necessity, contrasts, traces = {}, {}, {}, {}
    for split in SPLITS:
        split_factorial = [item for item in factorial_rows if item["split"] == split]
        factorial_reports[split] = {}
        for condition in CONDITIONS:
            values = {item["group_id"]: [item["cells"][condition]["correct_margin"]]
                      for item in split_factorial}
            report = accuracy(values, f"{split}:factorial:{condition}:correct_margin", traces, replicates)
            factorial_reports[split][condition] = report
            if not report["passes"]:
                failures.append(f"factorial:{split}:{condition}")
        values = {item["group_id"]: [item["selector_payload_interaction"]]
                  for item in split_factorial}
        interaction = bootstrap(values, f"{split}:selector_payload_interaction", traces,
                                replicates=replicates)
        interaction["passes"] = interaction["lower95"] > 0
        interactions[split] = interaction
        if not interaction["passes"]:
            failures.append(f"interaction:{split}")

        controls[split] = {}
        for label in CONTROL_FAMILIES:
            controls[split][label] = {}
            for condition in CONDITIONS:
                matching = [item for item in row_measurements if item["split"] == split
                            and item["condition"] == condition and control_matches(label, item)]
                if len(matching) != len(split_factorial):
                    raise RuntimeError(f"control membership mismatch: {split}/{label}/{condition}")
                controls[split][label][condition] = {}
                for endpoint in ("base", "donor"):
                    values = {item["group_id"]: [item[f"{endpoint}_margin"]] for item in matching}
                    cell_id = f"{split}:control:{label}:{condition}:{endpoint}"
                    report = accuracy(values, cell_id, traces, replicates)
                    controls[split][label][condition][endpoint] = report
                    if not report["passes"]:
                        failures.append(f"control:{split}:{label}:{condition}:{endpoint}")

        split_effects = [item for item in effects if item["split"] == split]
        drops, gaps = collections.defaultdict(list), collections.defaultdict(list)
        for item in split_effects:
            drops[item["group_id"]].append(item["selected_match_drop"])
            gaps[item["group_id"]].append(item["selected_vs_neutral_gap"])
        flat_drops = [value for group in sorted(drops) for value in drops[group]]
        drop_interval = bootstrap(drops, f"{split}:selected_match_drop", traces,
                                  replicates=replicates)
        gap_interval = bootstrap(gaps, f"{split}:selected_vs_neutral_gap", traces,
                                 replicates=replicates)
        item = {
            "group_count": len(drops), "observation_count": len(flat_drops),
            "positive_selected_drop_fraction": float(np.mean(np.asarray(flat_drops) > 0)),
            "mean_selected_drop": float(np.mean(flat_drops)),
            "selected_drop_bootstrap": drop_interval,
            "mean_selected_vs_neutral_gap": gap_interval["point_mean"],
            "selected_vs_neutral_gap_bootstrap": gap_interval,
        }
        item["passes_selected_necessity"] = bool(
            item["positive_selected_drop_fraction"] >= .70 and drop_interval["lower95"] > 0)
        item["passes_selected_vs_neutral"] = gap_interval["lower95"] > 0
        item["passes"] = item["passes_selected_necessity"] and item["passes_selected_vs_neutral"]
        necessity[split] = item
        if not item["passes_selected_necessity"]:
            failures.append(f"selected_match_necessity:{split}")
        if not item["passes_selected_vs_neutral"]:
            failures.append(f"selected_vs_neutral:{split}")

        contrasts[split] = {}
        for condition in CONDITIONS:
            values = {item["group_id"]: [item["contrast_source_signed_margin_change"]]
                      for item in split_effects if item["condition"] == condition}
            contrasts[split][condition] = bootstrap(
                values, f"{split}:contrast_source:{condition}", traces,
                two_sided=True, replicates=replicates)
    pred_a = not any(item.startswith(("factorial:", "control:")) for item in failures)
    pred_b = not any(item.startswith("interaction:") for item in failures)
    pred_c = not any(item.startswith(("selected_match_necessity:", "selected_vs_neutral:"))
                     for item in failures)
    reports = {
        "pred_a_native_factorial_and_controls": pred_a,
        "pred_b_selector_payload_interaction": pred_b,
        "pred_c_selected_match_necessity_and_neutral_selectivity": pred_c,
        "factorial_cells": factorial_reports,
        "selector_payload_interaction": interactions,
        "relation_preserving_controls": controls,
        "selected_match_necessity_and_neutral_selectivity": necessity,
        "contrast_source_diagnostics_not_gated": contrasts,
        "failed_scientific_clauses": failures,
        "all_scientific_gates_pass": bool(pred_a and pred_b and pred_c),
        "verdict": "held_capability_screen" if not failures else "scientific_null",
    }
    if len(traces) != 86:
        raise RuntimeError(f"bootstrap cell census mismatch: {len(traces)}")
    return reports, traces


def compare(expected: object, observed: object, path: str, failures: list[str]) -> None:
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(expected) != set(observed):
            failures.append(f"{path}:keys_or_type")
            return
        for key in expected:
            compare(expected[key], observed[key], f"{path}.{key}", failures)
    elif isinstance(expected, list):
        if not isinstance(observed, list) or len(expected) != len(observed):
            failures.append(f"{path}:length_or_type")
            return
        for index, value in enumerate(expected):
            compare(value, observed[index], f"{path}[{index}]", failures)
    elif isinstance(expected, float):
        if not isinstance(observed, (int, float)) or not math.isclose(
                expected, float(observed), rel_tol=0.0, abs_tol=ABS_TOLERANCE):
            failures.append(f"{path}:numeric")
    elif expected != observed:
        failures.append(f"{path}:value")


def audit_payload(result: Mapping[str, object], groups: Sequence[dict], rows: Sequence[dict],
                  specs: Sequence[dict], *, replicates: int = BOOTSTRAPS) -> dict:
    failures = []
    try:
        rebuilt_raw = reconstruct_raw(groups, rows, specs, result["raw_evidence"]
                                      ["sequence_measurements"])
        recomputed, traces = score(rebuilt_raw, replicates=replicates)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        return {"audit_verdict": "failed_independent_audit",
                "audit_failures": [f"raw_reconstruction:{type(error).__name__}:{error}"],
                "independently_recomputed_scientific_verdict": None,
                "bootstrap_cell_count": 0, "bootstrap_trace_hash": None}
    compare(rebuilt_raw, result.get("raw_evidence"), "raw_evidence", failures)
    for key, value in recomputed.items():
        compare(value, result.get(key), f"score.{key}", failures)
    envelope = {
        "schema": "induction_selector_payload_native_capability_rung580_result_v1",
        "instrument_passes": True, "model_forwards": EXPECTED_FORWARDS,
        "model_backwards": 0, "model_weights_updated": False,
        "unique_sequences": EXPECTED_SEQUENCES,
        "checkpoint_weights_sha256": CHECKPOINT_SHA256,
        "implementation_sha256": AUTHORITY_HASHES[R580_SCRIPT],
        "test_sha256": AUTHORITY_HASHES[R580_TEST],
        "input_sha256": R580_INPUT_HASHES,
        "evaluated_splits": list(SPLITS), "forbidden_splits_opened": [],
        "next_step": ("independent_CPU_audit_then_separate_R557_R558_adaptation_preregistration"
                      if recomputed["all_scientific_gates_pass"]
                      else "preserve_scientific_null_and_do_not_search_factor_sites"),
    }
    for key, value in envelope.items():
        compare(value, result.get(key), f"envelope.{key}", failures)
    ordered_traces = {key: traces[key] for key in sorted(traces)}
    memberships = {
        "group_membership_sha256": content_sha256([
            [item["group_id"], item["split"]] for item in groups]),
        "row_membership_sha256": content_sha256([
            [item["row_id"], item["group_id"], item["split"], item["family_id"],
             item["family_variant"]] for item in rows]),
        "sequence_membership_sha256": content_sha256(list(specs)),
    }
    return {
        "audit_verdict": "held_independent_audit" if not failures else "failed_independent_audit",
        "audit_failures": failures,
        "independently_recomputed_scientific_verdict": recomputed["verdict"],
        "independently_recomputed_failed_clauses": recomputed["failed_scientific_clauses"],
        "raw_counts": {"sequences": len(rebuilt_raw["sequence_measurements"]),
                       "rows": len(rebuilt_raw["row_measurements"]),
                       "factorial_groups": len(rebuilt_raw["group_factorial_measurements"]),
                       "condition_effects": len(rebuilt_raw["group_condition_effect_measurements"])},
        "membership_hashes": memberships,
        "bootstrap_cell_count": len(traces),
        "bootstrap_algorithm": BOOTSTRAP_ALGORITHM,
        "bootstrap_algorithm_sha256": content_sha256(BOOTSTRAP_ALGORITHM),
        "bootstrap_trace_hash": content_sha256(ordered_traces),
        "bootstrap_traces": ordered_traces,
        "recomputed_scores": recomputed,
    }


def audit_receipt(result: Mapping[str, object], result_bytes: bytes,
                  receipt: Mapping[str, object]) -> list[str]:
    failures = []
    expected = {
        "schema": "induction_selector_payload_native_capability_rung580_receipt_v1",
        "result_path": "basis_aligned/bilinear_quotient/induction_selector_payload_native_capability_rung580_results.json",
        "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "implementation_sha256": AUTHORITY_HASHES[R580_SCRIPT],
        "test_sha256": AUTHORITY_HASHES[R580_TEST],
        "preregistration_sha256": AUTHORITY_HASHES[R580_PREREG],
        "input_sha256": R580_INPUT_HASHES,
        "checkpoint_weights_sha256": CHECKPOINT_SHA256,
        "verdict": result.get("verdict"), "model_forwards": EXPECTED_FORWARDS,
        "model_backwards": 0, "evaluated_splits": list(SPLITS),
        "forbidden_splits_opened": [],
    }
    compare(expected, receipt, "receipt", failures)
    return failures


def planted_measurements(specs: Sequence[dict], rows: Sequence[dict],
                         *, make_null: bool = False) -> list[dict]:
    margins = {item["sequence_id"]: 4.0 for item in specs}
    donor_margins = {"match_break_payload_preserved": 1.0,
                     "irrelevant_source_edit": 3.9, "irrelevant_payload_edit": 3.85,
                     "contrast_target_source_edit": 3.4,
                     "copy_relation_preserved_nuisance_change": 3.8}
    for row in rows:
        if row["family_id"] in donor_margins:
            margins[sequence_id(row["donor_ids"])] = donor_margins[row["family_id"]]
    output = []
    for spec in specs:
        margin = margins[spec["sequence_id"]]
        z = margin if spec["registered_answer_id"] == spec["token_b_id"] else -margin
        logit_b, logit_d = z / 2.0, -z / 2.0
        normalizer = float(np.logaddexp(logit_b, logit_d) + .1)
        output.append({**spec, "logit_b": logit_b, "logit_d": logit_d,
                       "log_normalizer": normalizer, "ce_b": normalizer - logit_b,
                       "ce_d": normalizer - logit_d})
    if make_null:
        null_ids = {sequence_id(group["factorial_conditions"]["s0p0"]["ids"])
                    for group in load_authority()[0] if group["split"] == "SELECT"}
        for item in output:
            if item["sequence_id"] in null_ids:
                item["logit_b"], item["logit_d"] = -2.0, 2.0
                item["log_normalizer"] = float(np.logaddexp(-2., 2.) + .1)
                item["ce_b"] = item["log_normalizer"] + 2.
                item["ce_d"] = item["log_normalizer"] - 2.
    return output


def fixture_result(groups: Sequence[dict], rows: Sequence[dict], specs: Sequence[dict],
                   *, make_null: bool, replicates: int) -> dict:
    raw = reconstruct_raw(groups, rows, specs, planted_measurements(specs, rows, make_null=make_null))
    reports, _ = score(raw, replicates=replicates)
    return {
        "schema": "induction_selector_payload_native_capability_rung580_result_v1",
        "instrument_passes": True, **reports, "raw_evidence": raw,
        "model_forwards": EXPECTED_FORWARDS, "model_backwards": 0,
        "model_weights_updated": False, "unique_sequences": EXPECTED_SEQUENCES,
        "checkpoint_weights_sha256": CHECKPOINT_SHA256,
        "implementation_sha256": AUTHORITY_HASHES[R580_SCRIPT],
        "test_sha256": AUTHORITY_HASHES[R580_TEST], "input_sha256": R580_INPUT_HASHES,
        "evaluated_splits": list(SPLITS), "forbidden_splits_opened": [],
        "next_step": ("preserve_scientific_null_and_do_not_search_factor_sites" if make_null
                      else "independent_CPU_audit_then_separate_R557_R558_adaptation_preregistration"),
    }


def run_dryrun() -> dict:
    groups, rows, _ = load_authority()
    specs = expected_sequence_specs(groups, rows)
    fixture_replicates = 41
    held = fixture_result(groups, rows, specs, make_null=False, replicates=fixture_replicates)
    null = fixture_result(groups, rows, specs, make_null=True, replicates=fixture_replicates)
    held_audit = audit_payload(held, groups, rows, specs, replicates=fixture_replicates)
    null_audit = audit_payload(null, groups, rows, specs, replicates=fixture_replicates)
    if held_audit["audit_verdict"] != "held_independent_audit":
        raise RuntimeError("held fixture audit failed")
    if null_audit["audit_verdict"] != "held_independent_audit" or \
            null_audit["independently_recomputed_scientific_verdict"] != "scientific_null":
        raise RuntimeError("scientific-null fixture audit failed")
    receipt = {
        "schema": "induction_selector_payload_native_capability_audit_rung581_dryrun_v1",
        "status": "dryrun_passed", "authority_groups": len(groups), "authority_rows": len(rows),
        "authority_sequences": len(specs), "fixture_bootstrap_replicates": fixture_replicates,
        "bootstrap_cells_per_fixture": held_audit["bootstrap_cell_count"],
        "held_fixture_audit_verdict": held_audit["audit_verdict"],
        "held_fixture_scientific_verdict": held_audit["independently_recomputed_scientific_verdict"],
        "null_fixture_audit_verdict": null_audit["audit_verdict"],
        "null_fixture_scientific_verdict": null_audit["independently_recomputed_scientific_verdict"],
        "null_fixture_failed_clauses": null_audit["independently_recomputed_failed_clauses"],
        "future_result_opened": False, "model_loaded": False, "model_forwards": 0,
        "model_backwards": 0, "script_sha256": sha256(SCRIPT),
        "test_sha256": sha256(TEST) if TEST.is_file() else None,
        "preregistration_sha256": sha256(PREREG),
    }
    DRYRUN.write_text(json.dumps(receipt, indent=1) + "\n")
    return receipt


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps(run_dryrun(), indent=2))
        return
    groups, rows, _ = load_authority()
    specs = expected_sequence_specs(groups, rows)
    if not R580_RESULT.is_file() or not R580_RECEIPT.is_file():
        raise RuntimeError("R580 result and receipt are not both available")
    if OUT.exists():
        raise RuntimeError("R581 audit namespace already exists")
    result_bytes = R580_RESULT.read_bytes()
    result = json.loads(result_bytes)
    receipt = json.loads(R580_RECEIPT.read_text())
    audit = audit_payload(result, groups, rows, specs)
    receipt_failures = audit_receipt(result, result_bytes, receipt)
    audit["audit_failures"].extend(receipt_failures)
    if receipt_failures:
        audit["audit_verdict"] = "failed_independent_audit"
    audit.update({
        "schema": "induction_selector_payload_native_capability_audit_rung581_v1",
        "rung": 581, "source_result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "source_receipt_sha256": sha256(R580_RECEIPT),
        "authority_sha256": {str(path): digest for path, digest in AUTHORITY_HASHES.items()},
        "model_loaded": False, "model_forwards": 0, "model_backwards": 0,
        "evaluated_splits": list(SPLITS), "forbidden_splits_opened": [],
    })
    OUT.write_text(json.dumps(audit, indent=1) + "\n")
    print(json.dumps({key: audit[key] for key in (
        "audit_verdict", "audit_failures", "independently_recomputed_scientific_verdict",
        "independently_recomputed_failed_clauses", "bootstrap_cell_count",
        "bootstrap_trace_hash", "model_forwards")}, indent=2))


if __name__ == "__main__":
    main()
