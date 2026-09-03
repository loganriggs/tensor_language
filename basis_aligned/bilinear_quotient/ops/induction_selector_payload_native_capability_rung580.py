#!/usr/bin/env python3
"""R580: audited native capability for repaired selector x payload rows.

The dry run is CPU-only and model-free.  The scientific path evaluates each
unique FIT/SELECT prompt once, saves raw sequence/row/group measurements, and
writes scientific gate failures as a terminal null rather than raising.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import collections
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


os.environ["BQLIB_NO_MODEL"] = "1"
ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
ROWS = ROOT / "induction_selector_payload_three_source_rows_rung578.json"
ROWS_RECEIPT = ROOT / "induction_selector_payload_three_source_rows_rung578_receipt.json"
ROWS_PREREG = POLY / "INDUCTION_SELECTOR_PAYLOAD_THREE_SOURCE_ROWS_RUNG578_PREREGISTRATION.md"
ROWS_BUILDER = ROOT / "ops" / "induction_selector_payload_three_source_rows_rung578.py"
ROWS_TEST = ROOT / "ops" / "test_induction_selector_payload_three_source_rows_rung578.py"
PREREG = POLY / "INDUCTION_SELECTOR_PAYLOAD_NATIVE_CAPABILITY_RUNG580_PREREGISTRATION.md"
SCRIPT = Path(__file__)
TEST = SCRIPT.with_name("test_induction_selector_payload_native_capability_rung580.py")
OUT = ROOT / "induction_selector_payload_native_capability_rung580_results.json"
OUT_RECEIPT = ROOT / "induction_selector_payload_native_capability_rung580_receipt.json"
DRYRUN = ROOT / "induction_selector_payload_native_capability_rung580_dryrun.json"
HASHES = {
    ROWS: "8893ff83ea6080ad704f38376715d19be8971867178a4edc3bfd61fe025b39b6",
    ROWS_RECEIPT: "9e4e63ebd98503d6aa5daa27617a20fea595829c5a372f27b1ce4371d7c05b45",
    ROWS_PREREG: "276d801bbf5795e6421488dd4971b3a2d2dcb56e4fc7c4bc7ecdd2f61a73e9ce",
    ROWS_BUILDER: "d47bb3d46bd2c6061132c13b356e58ba9dfe2a56a2629f8b49a03f280d290bbd",
    ROWS_TEST: "9d795df358dfef9c5d17a539307f8e781f2a4debeb4909078858a242b3dfc512",
    PREREG: "8f80926d0a90360a66ebce605732d32ff3e283a3428eb7245f4813a521d12580",
}
SPLITS = ("FIT", "SELECT")
FORBIDDEN_SPLITS = ("FINAL_TEST", "OOD")
CONDITIONS = ("s0p0", "s0p1", "s1p0", "s1p1")
CONTROL_FAMILIES = {
    "neutral_source": ("irrelevant_source_edit", None),
    "neutral_payload": ("irrelevant_payload_edit", None),
    "filler": ("copy_relation_preserved_nuisance_change", "filler_change"),
    "lag": ("copy_relation_preserved_nuisance_change", "lag_extension"),
}
BOOTSTRAP_NAMESPACE = "a8-r580-group-bootstrap-v1"
BOOTSTRAPS = 2_000
BATCH = 32
EXPECTED_GROUPS = 108
EXPECTED_ROWS = 3_240
EXPECTED_SEQUENCES = 3_024
EXPECTED_FORWARDS = math.ceil(EXPECTED_SEQUENCES / BATCH)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def content_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_authority() -> tuple[list[dict], list[dict], dict[str, dict]]:
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen input mismatch: {path}")
    document = json.loads(ROWS.read_text())
    receipt = json.loads(ROWS_RECEIPT.read_text())
    if receipt["rows_sha256"] != HASHES[ROWS]:
        raise RuntimeError("R578 receipt does not bind the frozen rows")
    if document["model_loaded"] is not False or document["outcomes_opened"] != []:
        raise RuntimeError("R578 construction opened an outcome")
    groups = [group for group in document["groups"] if group["split"] in SPLITS]
    rows = [row for row in document["rows"] if row["split"] in SPLITS]
    by_group = {group["group_id"]: group for group in groups}
    if len(groups) != EXPECTED_GROUPS or len(by_group) != EXPECTED_GROUPS:
        raise RuntimeError("FIT/SELECT group census changed")
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError("FIT/SELECT row census changed")
    if {group["split"] for group in groups} != set(SPLITS):
        raise RuntimeError("required FIT/SELECT split is absent")
    if any(row["split"] in FORBIDDEN_SPLITS for row in rows):
        raise RuntimeError("forbidden split entered the capability rows")
    return groups, rows, by_group


def _sequence_id(ids: Sequence[int]) -> str:
    return content_sha256({"token_ids": list(ids)})


def collect_sequence_specs(
    groups: Sequence[dict], rows: Sequence[dict]
) -> list[dict]:
    owner: dict[tuple[int, ...], str] = {}
    answer: dict[tuple[int, ...], int] = {}
    by_group = {group["group_id"]: group for group in groups}

    def register(group_id: str, ids_raw: Sequence[int], answer_id: int) -> None:
        ids = tuple(ids_raw)
        if ids in owner and owner[ids] != group_id:
            raise RuntimeError("a prompt sequence crosses semantic groups")
        if ids in answer and answer[ids] != answer_id:
            raise RuntimeError("one prompt has inconsistent registered answers")
        owner[ids] = group_id
        answer[ids] = answer_id

    for group in groups:
        for condition in group["factorial_conditions"].values():
            register(group["group_id"], condition["ids"], condition["answer_id"])
    for row in rows:
        register(row["group_id"], row["base_ids"], row["base_answer_id"])
        register(row["group_id"], row["donor_ids"], row["donor_answer_id"])
    ordered = sorted(owner, key=lambda ids: (len(ids), ids))
    if len(ordered) != EXPECTED_SEQUENCES:
        raise RuntimeError(
            f"unique sequence census changed: {len(ordered)} != {EXPECTED_SEQUENCES}"
        )
    specs = []
    for ids in ordered:
        group = by_group[owner[ids]]
        token_b = group["variable_token_ids"]["B"]
        token_d = group["variable_token_ids"]["D"]
        if answer[ids] not in (token_b, token_d):
            raise RuntimeError("registered answer is outside the target payload pair")
        specs.append(
            {
                "sequence_id": _sequence_id(ids),
                "group_id": owner[ids],
                "split": group["split"],
                "token_ids": list(ids),
                "length": len(ids),
                "final_position": len(ids) - 1,
                "registered_answer_id": answer[ids],
                "token_b_id": token_b,
                "token_d_id": token_d,
            }
        )
    if len({spec["sequence_id"] for spec in specs}) != len(specs):
        raise RuntimeError("sequence hash collision")
    return specs


def native_logits(model: torch.nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    """Exact observed-model forward returning clipped logits."""
    x = model.transformer.wte(tokens)
    x = F.rms_norm(x, (x.size(-1),))
    x0, v1 = x, None
    for block in model.transformer.h:
        x, v1 = block(x, v1, x0)
    logits = model.lm_head(F.rms_norm(x, (x.size(-1),)))
    return (30.0 * torch.tanh(logits / 30.0)).float()


def evaluate_unique_sequences(
    model: torch.nn.Module, specs: Sequence[dict]
) -> tuple[list[dict], int]:
    device = next(model.parameters()).device
    measurements = []
    calls = 0
    with torch.inference_mode():
        for start in range(0, len(specs), BATCH):
            chunk = specs[start : start + BATCH]
            length = max(spec["length"] for spec in chunk)
            tokens = torch.full(
                (len(chunk), length), 50256, dtype=torch.long, device=device
            )
            for index, spec in enumerate(chunk):
                ids = torch.tensor(spec["token_ids"], dtype=torch.long, device=device)
                tokens[index, : ids.numel()] = ids
            logits = native_logits(model, tokens)
            calls += 1
            for index, spec in enumerate(chunk):
                final_logits = logits[index, spec["final_position"]]
                logit_b = float(final_logits[spec["token_b_id"]].cpu())
                logit_d = float(final_logits[spec["token_d_id"]].cpu())
                log_normalizer = float(torch.logsumexp(final_logits, dim=-1).cpu())
                measurements.append(
                    {
                        **{key: spec[key] for key in (
                            "sequence_id", "group_id", "split", "length",
                            "final_position", "registered_answer_id", "token_b_id",
                            "token_d_id",
                        )},
                        "logit_b": logit_b,
                        "logit_d": logit_d,
                        "log_normalizer": log_normalizer,
                        "ce_b": log_normalizer - logit_b,
                        "ce_d": log_normalizer - logit_d,
                    }
                )
    if len(measurements) != EXPECTED_SEQUENCES:
        raise RuntimeError("not every unique sequence was measured exactly once")
    if len({item["sequence_id"] for item in measurements}) != len(measurements):
        raise RuntimeError("a sequence was evaluated more than once")
    return measurements, calls


def _correct_margin(measurement: Mapping[str, object], answer_id: int) -> float:
    if answer_id == measurement["token_b_id"]:
        return float(measurement["logit_b"]) - float(measurement["logit_d"])
    if answer_id == measurement["token_d_id"]:
        return float(measurement["logit_d"]) - float(measurement["logit_b"])
    raise RuntimeError("answer token is neither B nor D")


def _correct_ce(measurement: Mapping[str, object], answer_id: int) -> float:
    if answer_id == measurement["token_b_id"]:
        return float(measurement["ce_b"])
    if answer_id == measurement["token_d_id"]:
        return float(measurement["ce_d"])
    raise RuntimeError("answer token is neither B nor D")


def _condition_name(group: Mapping[str, object], condition_id: str) -> str:
    matches = [
        name
        for name, condition in group["factorial_conditions"].items()
        if condition["condition_id"] == condition_id
    ]
    if len(matches) != 1:
        raise RuntimeError("row base does not identify exactly one factorial cell")
    return matches[0]


def build_raw_evidence(
    groups: Sequence[dict],
    rows: Sequence[dict],
    sequence_measurements: Sequence[dict],
) -> dict:
    by_group = {group["group_id"]: group for group in groups}
    by_sequence = {item["sequence_id"]: item for item in sequence_measurements}
    if len(by_sequence) != EXPECTED_SEQUENCES:
        raise RuntimeError("raw sequence evidence is incomplete")

    row_measurements = []
    row_index = {}
    for row in rows:
        group = by_group[row["group_id"]]
        condition = _condition_name(group, row["base_condition_id"])
        base_id = _sequence_id(row["base_ids"])
        donor_id = _sequence_id(row["donor_ids"])
        base = by_sequence[base_id]
        donor = by_sequence[donor_id]
        base_margin = _correct_margin(base, row["base_answer_id"])
        donor_margin = _correct_margin(donor, row["donor_answer_id"])
        item = {
            "row_id": row["row_id"],
            "group_id": row["group_id"],
            "split": row["split"],
            "family_id": row["family_id"],
            "family_variant": row["family_variant"],
            "condition": condition,
            "base_sequence_id": base_id,
            "donor_sequence_id": donor_id,
            "base_answer_id": row["base_answer_id"],
            "donor_answer_id": row["donor_answer_id"],
            "base_margin": base_margin,
            "donor_margin": donor_margin,
            "base_ce": _correct_ce(base, row["base_answer_id"]),
            "donor_ce": _correct_ce(donor, row["donor_answer_id"]),
            "donor_minus_base_margin": donor_margin - base_margin,
            "answer_changes": row["answer_changes"],
        }
        row_measurements.append(item)
        key = (row["group_id"], row["family_id"], condition)
        if key in row_index:
            # Nuisance rows have distinct filler/lag variants within a condition.
            key = (*key, row["family_variant"].split(":")[-1])
        row_index[key] = item
    if len(row_measurements) != EXPECTED_ROWS:
        raise RuntimeError("raw row evidence is incomplete")

    factorial = []
    group_effects = []
    for group in groups:
        cells = {}
        for condition_name in CONDITIONS:
            condition = group["factorial_conditions"][condition_name]
            measurement = by_sequence[_sequence_id(condition["ids"])]
            cells[condition_name] = {
                "sequence_id": measurement["sequence_id"],
                "answer_id": condition["answer_id"],
                "correct_margin": _correct_margin(measurement, condition["answer_id"]),
                "correct_ce": _correct_ce(measurement, condition["answer_id"]),
                "z_b_minus_d": float(measurement["logit_b"])
                - float(measurement["logit_d"]),
            }
        interaction = (
            cells["s0p0"]["z_b_minus_d"]
            - cells["s1p0"]["z_b_minus_d"]
            - cells["s0p1"]["z_b_minus_d"]
            + cells["s1p1"]["z_b_minus_d"]
        ) / 4.0
        factorial.append(
            {
                "group_id": group["group_id"],
                "split": group["split"],
                "cells": cells,
                "selector_payload_interaction": interaction,
            }
        )
        for condition in CONDITIONS:
            selected = row_index[(
                group["group_id"], "match_break_payload_preserved", condition
            )]
            neutral_source = row_index[(
                group["group_id"], "irrelevant_source_edit", condition
            )]
            neutral_payload = row_index[(
                group["group_id"], "irrelevant_payload_edit", condition
            )]
            contrast = row_index[(
                group["group_id"], "contrast_target_source_edit", condition
            )]
            if not (
                selected["base_sequence_id"]
                == neutral_source["base_sequence_id"]
                == neutral_payload["base_sequence_id"]
                == contrast["base_sequence_id"]
            ):
                raise RuntimeError("paired controls do not share one factorial base")
            base_margin = selected["base_margin"]
            selected_drop = base_margin - selected["donor_margin"]
            source_effect = abs(base_margin - neutral_source["donor_margin"])
            payload_effect = abs(base_margin - neutral_payload["donor_margin"])
            group_effects.append(
                {
                    "group_id": group["group_id"],
                    "split": group["split"],
                    "condition": condition,
                    "selected_match_drop": selected_drop,
                    "neutral_source_absolute_effect": source_effect,
                    "neutral_payload_absolute_effect": payload_effect,
                    "selected_vs_neutral_gap": selected_drop
                    - max(source_effect, payload_effect),
                    "contrast_source_signed_margin_change": contrast["donor_margin"]
                    - base_margin,
                }
            )
    return {
        "sequence_measurements": list(sequence_measurements),
        "row_measurements": row_measurements,
        "group_factorial_measurements": factorial,
        "group_condition_effect_measurements": group_effects,
    }


def _bootstrap_index(cell_id: str, replicate: int, draw: int, count: int) -> int:
    payload = f"{BOOTSTRAP_NAMESPACE}:{cell_id}:{replicate}:{draw}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % count


def bootstrap_summary(
    values_by_group: Mapping[str, Sequence[float]],
    cell_id: str,
    *,
    two_sided: bool = False,
) -> dict:
    group_ids = tuple(sorted(values_by_group))
    if not group_ids or any(not values_by_group[group] for group in group_ids):
        raise RuntimeError(f"bootstrap cell {cell_id!r} has no group values")
    if any(
        not math.isfinite(float(value))
        for group in group_ids
        for value in values_by_group[group]
    ):
        raise RuntimeError(f"bootstrap cell {cell_id!r} contains nonfinite values")
    means = np.empty(BOOTSTRAPS, dtype=np.float64)
    for replicate in range(BOOTSTRAPS):
        total = 0.0
        observations = 0
        for draw in range(len(group_ids)):
            group = group_ids[
                _bootstrap_index(cell_id, replicate, draw, len(group_ids))
            ]
            values = values_by_group[group]
            total += sum(float(value) for value in values)
            observations += len(values)
        means[replicate] = total / observations
    all_values = [float(value) for group in group_ids for value in values_by_group[group]]
    report = {
        "namespace": BOOTSTRAP_NAMESPACE,
        "cell_id": cell_id,
        "replicates": BOOTSTRAPS,
        "ordered_group_ids": list(group_ids),
        "observation_count": len(all_values),
        "point_mean": float(np.mean(all_values)),
        "lower95": float(np.quantile(means, 0.025, method="lower")),
    }
    if two_sided:
        report["upper95"] = float(np.quantile(means, 0.975, method="higher"))
    return report


def _accuracy_report(
    values_by_group: Mapping[str, Sequence[float]], cell_id: str
) -> dict:
    values = [float(value) for group in sorted(values_by_group) for value in values_by_group[group]]
    bootstrap = bootstrap_summary(values_by_group, cell_id)
    report = {
        "group_count": len(values_by_group),
        "observation_count": len(values),
        "correct_fraction": float(np.mean(np.asarray(values) > 0)),
        "mean_margin": float(np.mean(values)),
        "bootstrap": bootstrap,
    }
    report["passes"] = bool(
        report["correct_fraction"] >= 0.75 and bootstrap["lower95"] > 0
    )
    return report


def _control_variant_matches(label: str, variant: str | None, row: dict) -> bool:
    if row["family_id"] != CONTROL_FAMILIES[label][0]:
        return False
    if variant is None:
        return True
    return row["family_variant"].endswith(f":{variant}")


def score_raw_evidence(raw: Mapping[str, object]) -> dict:
    factorial_rows = raw["group_factorial_measurements"]
    row_measurements = raw["row_measurements"]
    effects = raw["group_condition_effect_measurements"]
    failures = []
    factorial_reports = {}
    interaction_reports = {}
    control_reports = {}
    necessity_reports = {}
    contrast_reports = {}

    for split in SPLITS:
        factorial_reports[split] = {}
        split_factorial = [row for row in factorial_rows if row["split"] == split]
        for condition in CONDITIONS:
            values = {
                row["group_id"]: [row["cells"][condition]["correct_margin"]]
                for row in split_factorial
            }
            cell_id = f"{split}:factorial:{condition}:correct_margin"
            report = _accuracy_report(values, cell_id)
            factorial_reports[split][condition] = report
            if not report["passes"]:
                failures.append(f"factorial:{split}:{condition}")
        interactions = {
            row["group_id"]: [row["selector_payload_interaction"]]
            for row in split_factorial
        }
        interaction = bootstrap_summary(
            interactions, f"{split}:selector_payload_interaction"
        )
        interaction["passes"] = interaction["lower95"] > 0
        interaction_reports[split] = interaction
        if not interaction["passes"]:
            failures.append(f"interaction:{split}")

        control_reports[split] = {}
        for label, (_family, variant) in CONTROL_FAMILIES.items():
            control_reports[split][label] = {}
            for condition in CONDITIONS:
                matching = [
                    row
                    for row in row_measurements
                    if row["split"] == split
                    and row["condition"] == condition
                    and _control_variant_matches(label, variant, row)
                ]
                if len(matching) != len(split_factorial):
                    raise RuntimeError(
                        f"control cell {split}/{label}/{condition} lacks one row per group"
                    )
                control_reports[split][label][condition] = {}
                for endpoint in ("base", "donor"):
                    values = {
                        row["group_id"]: [row[f"{endpoint}_margin"]]
                        for row in matching
                    }
                    cell_id = f"{split}:control:{label}:{condition}:{endpoint}"
                    report = _accuracy_report(values, cell_id)
                    control_reports[split][label][condition][endpoint] = report
                    if not report["passes"]:
                        failures.append(
                            f"control:{split}:{label}:{condition}:{endpoint}"
                        )

        split_effects = [row for row in effects if row["split"] == split]
        drops = collections.defaultdict(list)
        gaps = collections.defaultdict(list)
        for row in split_effects:
            drops[row["group_id"]].append(row["selected_match_drop"])
            gaps[row["group_id"]].append(row["selected_vs_neutral_gap"])
        drop_values = [value for group in sorted(drops) for value in drops[group]]
        drop_bootstrap = bootstrap_summary(drops, f"{split}:selected_match_drop")
        gap_bootstrap = bootstrap_summary(gaps, f"{split}:selected_vs_neutral_gap")
        necessity = {
            "group_count": len(drops),
            "observation_count": len(drop_values),
            "positive_selected_drop_fraction": float(
                np.mean(np.asarray(drop_values) > 0)
            ),
            "mean_selected_drop": float(np.mean(drop_values)),
            "selected_drop_bootstrap": drop_bootstrap,
            "mean_selected_vs_neutral_gap": gap_bootstrap["point_mean"],
            "selected_vs_neutral_gap_bootstrap": gap_bootstrap,
        }
        necessity["passes_selected_necessity"] = bool(
            necessity["positive_selected_drop_fraction"] >= 0.70
            and drop_bootstrap["lower95"] > 0
        )
        necessity["passes_selected_vs_neutral"] = gap_bootstrap["lower95"] > 0
        necessity["passes"] = bool(
            necessity["passes_selected_necessity"]
            and necessity["passes_selected_vs_neutral"]
        )
        necessity_reports[split] = necessity
        if not necessity["passes_selected_necessity"]:
            failures.append(f"selected_match_necessity:{split}")
        if not necessity["passes_selected_vs_neutral"]:
            failures.append(f"selected_vs_neutral:{split}")

        contrast_reports[split] = {}
        for condition in CONDITIONS:
            values = {
                row["group_id"]: [row["contrast_source_signed_margin_change"]]
                for row in split_effects
                if row["condition"] == condition
            }
            contrast_reports[split][condition] = bootstrap_summary(
                values,
                f"{split}:contrast_source:{condition}",
                two_sided=True,
            )

    pred_a = not any(
        clause.startswith(("factorial:", "control:")) for clause in failures
    )
    pred_b = not any(clause.startswith("interaction:") for clause in failures)
    pred_c = not any(
        clause.startswith(("selected_match_necessity:", "selected_vs_neutral:"))
        for clause in failures
    )
    return {
        "pred_a_native_factorial_and_controls": pred_a,
        "pred_b_selector_payload_interaction": pred_b,
        "pred_c_selected_match_necessity_and_neutral_selectivity": pred_c,
        "factorial_cells": factorial_reports,
        "selector_payload_interaction": interaction_reports,
        "relation_preserving_controls": control_reports,
        "selected_match_necessity_and_neutral_selectivity": necessity_reports,
        "contrast_source_diagnostics_not_gated": contrast_reports,
        "failed_scientific_clauses": failures,
        "all_scientific_gates_pass": bool(pred_a and pred_b and pred_c),
        "verdict": "held_capability_screen" if not failures else "scientific_null",
    }


def planted_sequence_measurements(
    specs: Sequence[dict], rows: Sequence[dict]
) -> list[dict]:
    margin_by_sequence = {spec["sequence_id"]: 4.0 for spec in specs}
    donor_margin = {
        "match_break_payload_preserved": 1.0,
        "irrelevant_source_edit": 3.9,
        "irrelevant_payload_edit": 3.85,
        "contrast_target_source_edit": 3.4,
        "copy_relation_preserved_nuisance_change": 3.8,
    }
    for row in rows:
        if row["family_id"] in donor_margin:
            margin_by_sequence[_sequence_id(row["donor_ids"])] = donor_margin[
                row["family_id"]
            ]
    result = []
    for spec in specs:
        margin = margin_by_sequence[spec["sequence_id"]]
        answer_is_b = spec["registered_answer_id"] == spec["token_b_id"]
        z = margin if answer_is_b else -margin
        logit_b, logit_d = z / 2.0, -z / 2.0
        log_normalizer = float(np.logaddexp(logit_b, logit_d) + 0.1)
        result.append(
            {
                **{key: spec[key] for key in (
                    "sequence_id", "group_id", "split", "length", "final_position",
                    "registered_answer_id", "token_b_id", "token_d_id",
                )},
                "logit_b": logit_b,
                "logit_d": logit_d,
                "log_normalizer": log_normalizer,
                "ce_b": log_normalizer - logit_b,
                "ce_d": log_normalizer - logit_d,
            }
        )
    return result


def make_planted_scientific_null(
    measurements: Sequence[dict], groups: Sequence[dict]
) -> list[dict]:
    result = copy.deepcopy(list(measurements))
    group_by_id = {group["group_id"]: group for group in groups}
    target_ids = {
        _sequence_id(group["factorial_conditions"]["s0p0"]["ids"])
        for group in groups
        if group["split"] == "SELECT"
    }
    for measurement in result:
        if measurement["sequence_id"] not in target_ids:
            continue
        group = group_by_id[measurement["group_id"]]
        if measurement["registered_answer_id"] != group["variable_token_ids"]["B"]:
            raise RuntimeError("planted null targeted the wrong factorial answer")
        measurement["logit_b"] = -2.0
        measurement["logit_d"] = 2.0
        measurement["log_normalizer"] = float(np.logaddexp(-2.0, 2.0) + 0.1)
        measurement["ce_b"] = measurement["log_normalizer"] + 2.0
        measurement["ce_d"] = measurement["log_normalizer"] - 2.0
    return result


def _write_scientific_result(result: dict) -> None:
    encoded = (json.dumps(result, indent=1) + "\n").encode()
    OUT.write_bytes(encoded)
    receipt = {
        "schema": "induction_selector_payload_native_capability_rung580_receipt_v1",
        "result_path": str(OUT.relative_to(ROOT.parent.parent)),
        "result_sha256": hashlib.sha256(encoded).hexdigest(),
        "implementation_sha256": sha256(SCRIPT),
        "test_sha256": sha256(TEST),
        "preregistration_sha256": sha256(PREREG),
        "input_sha256": {str(path): expected for path, expected in HASHES.items()},
        "checkpoint_weights_sha256": result["checkpoint_weights_sha256"],
        "verdict": result["verdict"],
        "model_forwards": result["model_forwards"],
        "model_backwards": result["model_backwards"],
        "evaluated_splits": result["evaluated_splits"],
        "forbidden_splits_opened": result["forbidden_splits_opened"],
    }
    OUT_RECEIPT.write_text(json.dumps(receipt, indent=1) + "\n")


def run_dryrun() -> dict:
    groups, rows, _ = load_authority()
    specs = collect_sequence_specs(groups, rows)
    planted = planted_sequence_measurements(specs, rows)
    raw_pass = build_raw_evidence(groups, rows, planted)
    pass_score = score_raw_evidence(raw_pass)
    raw_null = build_raw_evidence(
        groups, rows, make_planted_scientific_null(planted, groups)
    )
    null_score = score_raw_evidence(raw_null)
    if not pass_score["all_scientific_gates_pass"]:
        raise RuntimeError("planted passing dry run failed")
    if null_score["verdict"] != "scientific_null":
        raise RuntimeError("planted scientific null did not return a terminal null")
    receipt = {
        "schema": "induction_selector_payload_native_capability_rung580_dryrun_v1",
        "status": "dryrun_passed",
        "implementation_sha256": sha256(SCRIPT),
        "test_sha256": sha256(TEST) if TEST.is_file() else None,
        "preregistration_sha256": sha256(PREREG),
        "input_sha256": {str(path): expected for path, expected in HASHES.items()},
        "groups": len(groups),
        "rows": len(rows),
        "unique_sequences": len(specs),
        "batch_size": BATCH,
        "literal_expected_forwards": EXPECTED_FORWARDS,
        "literal_expected_backwards": 0,
        "passing_fixture_verdict": pass_score["verdict"],
        "null_fixture_verdict": null_score["verdict"],
        "null_fixture_failed_clauses": null_score["failed_scientific_clauses"],
        "raw_sequence_measurement_count": len(raw_pass["sequence_measurements"]),
        "raw_row_measurement_count": len(raw_pass["row_measurements"]),
        "raw_group_factorial_count": len(raw_pass["group_factorial_measurements"]),
        "raw_group_condition_effect_count": len(
            raw_pass["group_condition_effect_measurements"]
        ),
        "bootstrap_namespace": BOOTSTRAP_NAMESPACE,
        "bootstrap_replicates": BOOTSTRAPS,
        "model_loaded": False,
        "model_forwards": 0,
        "model_backwards": 0,
        "evaluated_splits": list(SPLITS),
        "forbidden_splits_opened": [],
    }
    DRYRUN.write_text(json.dumps(receipt, indent=1) + "\n")
    return receipt


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps(run_dryrun(), indent=2))
        return
    started = time.time()
    groups, rows, _ = load_authority()
    specs = collect_sequence_specs(groups, rows)
    for search_path in (ROOT, ROOT / "ops", POLY):
        if str(search_path) not in sys.path:
            sys.path.insert(0, str(search_path))
    import bilin18_observed_model_facade as facade

    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.float32, verify_weights_sha256=True
    )
    measurements, calls = evaluate_unique_sequences(model, specs)
    if calls != EXPECTED_FORWARDS:
        raise RuntimeError(f"forward price changed: {calls} != {EXPECTED_FORWARDS}")
    raw = build_raw_evidence(groups, rows, measurements)
    scores = score_raw_evidence(raw)
    result = {
        "schema": "induction_selector_payload_native_capability_rung580_result_v1",
        "rung": 580,
        "stage": "repaired_induction_selector_payload_native_capability",
        "instrument_passes": True,
        **scores,
        "raw_evidence": raw,
        "model_forwards": calls,
        "model_backwards": 0,
        "model_weights_updated": False,
        "unique_sequences": len(measurements),
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "implementation_sha256": sha256(SCRIPT),
        "test_sha256": sha256(TEST),
        "input_sha256": {str(path): expected for path, expected in HASHES.items()},
        "evaluated_splits": list(SPLITS),
        "forbidden_splits_opened": [],
        "elapsed_seconds": time.time() - started,
        "next_step": (
            "independent_CPU_audit_then_separate_R557_R558_adaptation_preregistration"
            if scores["all_scientific_gates_pass"]
            else "preserve_scientific_null_and_do_not_search_factor_sites",
        ),
    }
    _write_scientific_result(result)
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "failed_scientific_clauses": result["failed_scientific_clauses"],
                "model_forwards": calls,
                "unique_sequences": len(measurements),
                "next_step": result["next_step"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
