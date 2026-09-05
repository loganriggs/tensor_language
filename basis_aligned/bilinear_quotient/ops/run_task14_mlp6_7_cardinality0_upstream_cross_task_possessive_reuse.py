#!/usr/bin/env python3
"""Test Task14 upstream-number writes on adjacent possessive agreement."""

# BQGATE: EXPERIMENT pred_a_authorities_and_program_validate pred_b_native_capability_and_noop pred_c_correct_write_moves_possessive_margin pred_d_each_direction_construction_transfers pred_e_direction_assignment_is_necessary pred_f_literal_cross_task_reuse
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

import circuit_fast_screen_candidate_possessive_adjacent as authority
import circuit_fast_screen_managed_runner as managed
import circuit_fast_screen_producer as producer
import run_task14_mlp6_7_direction_cardinality_prototype_cross_circuit_collateral as projected


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/task14_mlp6_7_cardinality0_upstream_cross_task_possessive_reuse_v1.json"
PROTOTYPES = ROOT / "circuits/fast_screens/task14_mlp6_7_direction_cardinality_prototype_artifact_v1.json"
TASK14_VALIDATION = ROOT / "circuits/fast_screens/task14_mlp6_7_direction_cardinality_prototype_causal_validation_v1_result.json"
POSSESSIVE_RESULT = ROOT / "circuits/fast_screens/possessive_number_adjacent_antecedent_v1_result.json"
OUT = ROOT / "circuits/fast_screens/task14_mlp6_7_cardinality0_upstream_cross_task_possessive_reuse_v1_result.json"
PRIOR_ART_SHA256 = "85be699531f66bcfa8c4000bb684a2fcd1977c89f95caefb10567e607ec41fc0"
PROTOTYPE_SHA256 = "cce00d8f2309b0f9e0329c094238505819f2cf8a21c04e276af2085e068c1d07"
TASK14_VALIDATION_SHA256 = "9a488259efdb85477f35c612696368e8a3e372338a11253615a7b53650c88fe0"
POSSESSIVE_RESULT_SHA256 = "3a08562d9d9ce57bef0027bc0d2e5ba1ca1d85a5d4ae462f489b5830a07f8dc7"
AUTHORITY_FILE_SHA256 = "258dfd83967d3023c0d52f9703fa46e428f0009b6890e1c98b0102772359b8f8"
MAX_NOOP_ERROR = 1e-4
MAX_INSTALL_ERROR = 5e-5
MIN_POSITIVE_FRACTION = 0.75
MIN_MEDIAN_CHANGE = 0.005
MIN_CELL_POSITIVE_FRACTION = 0.625
MIN_CORRECT_BEATS_SWAPPED_FRACTION = 0.65
MIN_MEDIAN_CORRECT_ADVANTAGE = 0.005
PRED_KEYS = (
    "pred_a_authorities_and_program_validate",
    "pred_b_native_capability_and_noop",
    "pred_c_correct_write_moves_possessive_margin",
    "pred_d_each_direction_construction_transfers",
    "pred_e_direction_assignment_is_necessary",
    "pred_f_literal_cross_task_reuse",
)


class ReuseError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_vectors() -> dict[str, list[float]]:
    if _sha256(PROTOTYPES) != PROTOTYPE_SHA256:
        raise ReuseError("prototype artifact changed")
    artifact = json.loads(PROTOTYPES.read_text())
    keys = ("plural_to_singular.cardinality_0", "singular_to_plural.cardinality_0")
    vectors = {key: artifact["prototypes"][key]["coordinates"] for key in keys}
    if any(len(vector) != 1152 or not all(math.isfinite(float(x)) for x in vector) for vector in vectors.values()):
        raise ReuseError("cardinality-0 vectors invalid")
    return vectors


def validate_preflight() -> None:
    for path, expected, label in (
        (PRIOR_ART, PRIOR_ART_SHA256, "prior art"),
        (TASK14_VALIDATION, TASK14_VALIDATION_SHA256, "Task14 validation"),
        (POSSESSIVE_RESULT, POSSESSIVE_RESULT_SHA256, "possessive result"),
        (Path(authority.__file__), AUTHORITY_FILE_SHA256, "possessive authority"),
    ):
        if _sha256(path) != expected:
            raise ReuseError(f"{label} changed")
    if json.loads(POSSESSIVE_RESULT.read_text()).get("terminal") != "screen":
        raise ReuseError("possessive authority did not pass")
    _load_vectors()


def derive_price() -> dict[str, int]:
    return {
        "physical_model_forwards": 4, "example_evaluations": 256,
        "correct_direction_installations": 64, "swapped_direction_installations": 64,
        "zero_add_replays": 64, "native_evaluations": 64,
        "backwards": 0, "parameter_updates": 0,
    }


def selected_rows():
    rows = [row for row in authority.build_rows() if row["transform_id"] in {"A1", "A2"}]
    if len(rows) != 64 or len({row["row_id"] for row in rows}) != 64:
        raise ReuseError("selected possessive census changed")
    return rows


def compile_plan() -> dict[str, object]:
    validate_preflight()
    rows = selected_rows()
    cells = {(row["direction_id"], row["construction_id"]) for row in rows}
    if len(cells) != 4:
        raise ReuseError("direction-construction cells changed")
    return {
        "schema": "task14_mlp6_7_cardinality0_upstream_cross_task_possessive_reuse_plan_v1",
        "candidate_id": "subject_verb.number_agreement.mlp6_7_cardinality0_upstream_cross_task_possessive_reuse_v1",
        "split": "POSSESSIVE_ADJACENT_FIT_ALL_A1_A2",
        "row_count": 64, "cells": sorted(f"{direction}.{construction}" for direction, construction in cells),
        "prototype_keys": ["plural_to_singular.cardinality_0", "singular_to_plural.cardinality_0"],
        "prior_art_sha256": PRIOR_ART_SHA256, "prototype_artifact_sha256": PROTOTYPE_SHA256,
        "possessive_authority_sha256": authority.authority_sha256(),
        "bars": {
            "maximum_noop_absolute_logit_error": MAX_NOOP_ERROR,
            "maximum_install_absolute_error": MAX_INSTALL_ERROR,
            "minimum_positive_fraction": MIN_POSITIVE_FRACTION,
            "minimum_median_donorward_change": MIN_MEDIAN_CHANGE,
            "minimum_cell_positive_fraction": MIN_CELL_POSITIVE_FRACTION,
            "minimum_correct_beats_swapped_fraction": MIN_CORRECT_BEATS_SWAPPED_FRACTION,
            "minimum_median_correct_advantage": MIN_MEDIAN_CORRECT_ADVANTAGE,
        },
        "fit_operations": 0, "price": derive_price(),
    }


def _batch(rows) -> producer.ModelBatch:
    return producer.ModelBatch(
        row_ids=tuple(str(row["row_id"]) for row in rows), side="base",
        token_rows=tuple(tuple(int(token) for token in row["base_ids"]) for row in rows),
        answer_ids=tuple(int(row["donor_answer_id"]) for row in rows),
        foil_ids=tuple(int(row["base_answer_id"]) for row in rows),
        semantic_positions=tuple(int(row["base_semantic_position"]) for row in rows),
    )


def score(rows, native_pairs, noop_pairs, correct_pairs, swapped_pairs, install_error):
    evidence = []
    for row, native, noop, correct, swapped in zip(rows, native_pairs, noop_pairs, correct_pairs, swapped_pairs):
        native_margin = native[0] - native[1]
        correct_change = (correct[0] - correct[1]) - native_margin
        swapped_change = (swapped[0] - swapped[1]) - native_margin
        evidence.append({
            "row_id": row["row_id"], "direction": row["direction_id"],
            "construction": row["construction_id"], "native_donor_minus_base_margin": native_margin,
            "correct_donorward_change": correct_change, "swapped_donorward_change": swapped_change,
            "correct_advantage": correct_change - swapped_change,
        })
    noop_error = max(abs(x - y) for native, noop in zip(native_pairs, noop_pairs) for x, y in zip(native, noop))
    correct = [item["correct_donorward_change"] for item in evidence]
    overall = {
        "positive_fraction": sum(value > 0 for value in correct) / len(correct),
        "median_donorward_change": statistics.median(correct),
        "mean_donorward_change": statistics.mean(correct),
    }
    cells = {}
    for direction in ("plural_to_singular", "singular_to_plural"):
        for construction in ("conjunct_frame", "notes_frame"):
            values = [item["correct_donorward_change"] for item in evidence if item["direction"] == direction and item["construction"] == construction]
            cells[f"{direction}.{construction}"] = {
                "count": len(values), "positive_fraction": sum(value > 0 for value in values) / len(values),
                "median_donorward_change": statistics.median(values), "mean_donorward_change": statistics.mean(values),
            }
    advantages = [item["correct_advantage"] for item in evidence]
    assignment = {
        "correct_beats_swapped_fraction": sum(value > 0 for value in advantages) / len(advantages),
        "median_correct_advantage": statistics.median(advantages),
        "mean_correct_advantage": statistics.mean(advantages),
    }
    pred_b = all(item["native_donor_minus_base_margin"] < 0 for item in evidence) and noop_error <= MAX_NOOP_ERROR and install_error <= MAX_INSTALL_ERROR
    pred_c = overall["positive_fraction"] >= MIN_POSITIVE_FRACTION and overall["median_donorward_change"] >= MIN_MEDIAN_CHANGE
    pred_d = all(item["count"] == 16 and item["positive_fraction"] >= MIN_CELL_POSITIVE_FRACTION and item["median_donorward_change"] > 0 for item in cells.values())
    pred_e = assignment["correct_beats_swapped_fraction"] >= MIN_CORRECT_BEATS_SWAPPED_FRACTION and assignment["median_correct_advantage"] >= MIN_MEDIAN_CORRECT_ADVANTAGE
    predictions = dict(zip(PRED_KEYS, (True, pred_b, pred_c, pred_d, pred_e, len(evidence) == 64)))
    return {
        "noop_max_absolute_logit_error": noop_error, "maximum_install_absolute_error": install_error,
        "overall_correct_write": overall, "by_direction_construction": cells,
        "direction_assignment": assignment, "predictions": predictions, "evidence": evidence,
    }


def evaluate(executor):
    rows = selected_rows()
    batch = _batch(rows)
    native = executor.native(batch, capture=False).answer_foil
    noop, noop_install_error = projected._projected_add(executor, batch, None)
    vectors = _load_vectors()
    opposite = {"plural_to_singular": "singular_to_plural", "singular_to_plural": "plural_to_singular"}
    torch = executor.torch
    correct_vectors = torch.tensor([vectors[f"{row['direction_id']}.cardinality_0"] for row in rows], dtype=torch.float32, device=executor.device)
    swapped_vectors = torch.tensor([vectors[f"{opposite[row['direction_id']]}.cardinality_0"] for row in rows], dtype=torch.float32, device=executor.device)
    correct, correct_error = projected._projected_add(executor, batch, correct_vectors)
    swapped, swapped_error = projected._projected_add(executor, batch, swapped_vectors)
    return score(rows, native, noop, correct, swapped, max(noop_install_error, correct_error, swapped_error))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    plan = compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True))
        return
    if OUT.exists():
        raise ReuseError(f"refusing overwrite {OUT}")
    checker = __import__("run_circuit_fast_screen_task14_head11_3_cross_circuit_collateral")
    checker._verify_checkpoint()
    executor = producer.Bilin18TorchBackend.load("cuda")
    scored = evaluate(executor)
    instrument = scored["predictions"][PRED_KEYS[0]] and scored["predictions"][PRED_KEYS[1]] and scored["predictions"][PRED_KEYS[5]]
    terminal = "screen" if all(scored["predictions"].values()) else "null" if instrument else "invalid"
    reason = "cross_task_number_write_reuse" if terminal == "screen" else "task14_write_does_not_transfer_to_possessive" if terminal == "null" else "instrument_failed"
    payload = managed.atomic_create_json(OUT, {
        "schema": "task14_mlp6_7_cardinality0_upstream_cross_task_possessive_reuse_result_v1",
        "candidate_id": plan["candidate_id"], "terminal": terminal, "reason": reason,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "plan": plan, "score": scored,
        "limits": "A positive screen establishes reuse on one related local-controller task, not a universal number variable.",
    })
    print(json.dumps({"terminal": terminal, "reason": reason, "predictions": scored["predictions"], "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
