#!/usr/bin/env python3
"""Native-only staged capability license for fresh matched-natural Task14 text."""

# BQGATE: EXPERIMENT pred_a_fit_all_cells_pass pred_b_holdout_all_cells_pass pred_c_license_issued

from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import circuit_fast_screen_candidate_task14_fresh_matched_natural_native_capability as authority
import circuit_fast_screen_managed_runner as managed
import native_capability_license as licensing
import run_task14_head11_3_subject_attractor_score_payload_factorial as model_helpers


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/task14_fresh_matched_natural_native_capability_v1.json"
RESULT = ROOT / "circuits/fast_screens/task14_fresh_matched_natural_native_capability_v1_result.json"
CAPABILITY_RESULT = ROOT / "circuits/fast_screens/task14_fresh_matched_natural_native_capability_v1_license_result.json"
LICENSE = ROOT / "circuits/fast_screens/task14_fresh_matched_natural_native_capability_v1_license.json"
PRIOR_ART_SHA256 = "a0efc20022dfc96611b3bfb02a113c391c4d3b1f8c0533a3e6776c2026e2be5c"
AUTHORITY_FILE_SHA256 = "3d02183780bdae4f4b317b3cf04410ac2184af83d69d9c517fba10522d4f3449"
AUTHORITY_LOGICAL_SHA256 = "8862dc84c10a28c0857cb7b201adab46e72f2e7063069fcf1a09f50ec21947d7"
MINIMUM_ACCURACY = .875


class CapabilityRunnerError(ValueError):
    """The frozen native capability contract was violated."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cell_id(row: Mapping[str, object], role: str) -> str:
    return "/".join((str(row["phase"]), str(row["template_id"]),
                     str(row["direction_id"]), role))


def _phase_rows(phase: str) -> list[dict]:
    if phase not in {"FIT", "HOLDOUT"}:
        raise CapabilityRunnerError("phase is invalid")
    return [row for row in authority.build_rows() if row["phase"] == phase]


def build_gate() -> licensing.CapabilityGate:
    if authority.validate_rows(authority.build_rows()) != AUTHORITY_LOGICAL_SHA256:
        raise CapabilityRunnerError("authority logical hash changed")
    counts = Counter(
        _cell_id(row, role) for row in authority.build_rows() for role in authority.ROLES)
    if len(counts) != 24 or set(counts.values()) != {4}:
        raise CapabilityRunnerError("registered capability cells changed")
    gate = licensing.CapabilityGate(
        capability_id=authority.CAPABILITY_ID,
        authority_path=Path(authority.__file__),
        expected_authority_file_sha256=AUTHORITY_FILE_SHA256,
        authority_logical_sha256=AUTHORITY_LOGICAL_SHA256,
        cells=tuple(licensing.CapabilityCell(cell_id, count, MINIMUM_ACCURACY)
                    for cell_id, count in sorted(counts.items())),
    )
    licensing.validate_gate(gate)
    return gate


def compile_plan() -> dict:
    if _sha256(PRIOR_ART) != PRIOR_ART_SHA256 \
            or _sha256(Path(authority.__file__)) != AUTHORITY_FILE_SHA256:
        raise CapabilityRunnerError("frozen prior-art receipt or authority changed")
    gate = build_gate()
    return {
        "schema": "task14_fresh_matched_natural_native_capability_plan_v1",
        "candidate_id": authority.CAPABILITY_ID,
        "causal_candidate_id": authority.CAUSAL_CANDIDATE_ID,
        "native_only": True, "causal_interventions": 0,
        "model_loaded": False, "gpu_accessed": False, "queue_touched": False,
        "prior_art_sha256": PRIOR_ART_SHA256,
        "authority_file_sha256": AUTHORITY_FILE_SHA256,
        "authority_logical_sha256": AUTHORITY_LOGICAL_SHA256,
        "registered_cells_sha256": licensing.cells_sha256(gate),
        "fit_rows": len(_phase_rows("FIT")), "fit_endpoint_evaluations": 48,
        "holdout_rows": len(_phase_rows("HOLDOUT")),
        "maximum_holdout_endpoint_evaluations": 48,
        "minimum_accuracy_each_cell": MINIMUM_ACCURACY,
        "fit_policy": "All twelve 4-example FIT cells must pass before HOLDOUT opens.",
        "holdout_policy": "All twelve 4-example HOLDOUT cells must pass before licensing.",
        "metrics": ["correct", "answer_minus_foil_margin", "full_vocab_CE"],
        "maximum_price": {"model_forwards": 2, "example_evaluations": 96,
                          "backwards": 0, "parameter_updates": 0},
    }


def evaluate_native(model, rows, torch, F):
    examples = [(row, role) for row in rows for role in authority.ROLES]
    device = next(model.parameters()).device
    tokens = torch.tensor(
        [row["endpoints"][role]["ids"] for row, role in examples],
        dtype=torch.long, device=device)
    logits = model_helpers._native_logits(model, tokens, torch, F)
    evidence = []
    for index, (row, role) in enumerate(examples):
        endpoint = row["endpoints"][role]
        answer, foil = int(endpoint["answer_id"]), int(endpoint["foil_id"])
        margin = float(logits[index, 8, answer] - logits[index, 8, foil])
        ce = float(-torch.log_softmax(logits[index, 8], dim=-1)[answer])
        evidence.append({
            "example_id": f"{row['row_id']}:{role}",
            "cell_id": _cell_id(row, role),
            "correct": bool(margin > 0),
            "full_vocab_CE": ce,
            "answer_minus_foil_margin": margin,
        })
    return evidence


def _expected_examples(phase: str) -> dict[str, str]:
    return {
        f"{row['row_id']}:{role}": _cell_id(row, role)
        for row in _phase_rows(phase) for role in authority.ROLES
    }


def summarize_phase(evidence: Sequence[Mapping[str, object]], phase: str) -> dict:
    expected = _expected_examples(phase)
    if not isinstance(evidence, (list, tuple)) or len(evidence) != len(expected):
        raise CapabilityRunnerError(f"{phase} evidence count changed")
    grouped = defaultdict(list)
    seen = set()
    required = {"example_id", "cell_id", "correct", "full_vocab_CE",
                "answer_minus_foil_margin"}
    for item in evidence:
        if set(item) != required or type(item["correct"]) is not bool:
            raise CapabilityRunnerError("native evidence fields or correctness changed")
        example_id = item["example_id"]
        if not isinstance(example_id, str) or example_id in seen \
                or expected.get(example_id) != item["cell_id"]:
            raise CapabilityRunnerError("native evidence identity changed or duplicated")
        for metric in ("full_vocab_CE", "answer_minus_foil_margin"):
            if type(item[metric]) not in (int, float) \
                    or not math.isfinite(float(item[metric])):
                raise CapabilityRunnerError("native metric is not finite")
        seen.add(example_id)
        grouped[str(item["cell_id"])].append(item)
    if seen != set(expected) or len(grouped) != 12 \
            or any(len(items) != 4 for items in grouped.values()):
        raise CapabilityRunnerError("native evidence does not cover exact frozen cells")
    cells, passed = {}, True
    for cell_id, items in sorted(grouped.items()):
        accuracy = sum(bool(item["correct"]) for item in items) / len(items)
        cell_passed = accuracy >= MINIMUM_ACCURACY
        cells[cell_id] = {
            "count": len(items), "accuracy": accuracy,
            "minimum_accuracy": MINIMUM_ACCURACY, "passed": cell_passed,
            "mean_answer_minus_foil_margin": sum(
                float(item["answer_minus_foil_margin"]) for item in items) / len(items),
            "mean_full_vocab_CE": sum(
                float(item["full_vocab_CE"]) for item in items) / len(items),
        }
        passed &= cell_passed
    return {"passed": passed, "cells": cells}


def run_two_stage(model, torch, F):
    fit_evidence = evaluate_native(model, _phase_rows("FIT"), torch, F)
    fit = summarize_phase(fit_evidence, "FIT")
    if not fit["passed"]:
        return fit, fit_evidence, None, [], None, None
    holdout_evidence = evaluate_native(model, _phase_rows("HOLDOUT"), torch, F)
    holdout = summarize_phase(holdout_evidence, "HOLDOUT")
    gate = build_gate()
    capability, capability_sha = licensing.finalize_native_capability(
        gate, fit_evidence + holdout_evidence, CAPABILITY_RESULT)
    license_record = {"value": None, "sha256": None}
    if capability["terminal"] == "pass":
        value, digest = licensing.issue_capability_license(
            gate, CAPABILITY_RESULT, LICENSE,
            causal_candidate_id=authority.CAUSAL_CANDIDATE_ID)
        licensing.validate_causal_preflight(
            gate, CAPABILITY_RESULT, LICENSE, expected_license_sha256=digest,
            causal_candidate_id=authority.CAUSAL_CANDIDATE_ID)
        license_record = {"value": value, "sha256": digest}
    return fit, fit_evidence, holdout, holdout_evidence, capability_sha, license_record


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for name in ("BQLIB_DRYRUN", "BQLIB_NO_MODEL"):
        if os.environ.get(name) not in {None, "1"}:
            raise CapabilityRunnerError(f"{name} must be absent or exactly 1")
    plan = compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1" \
            or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True))
        return
    if any(path.exists() for path in (RESULT, CAPABILITY_RESULT, LICENSE)):
        raise CapabilityRunnerError("refusing to overwrite capability artifacts")
    torch, F, facade = model_helpers._dependencies()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    with torch.no_grad():
        fit, fit_evidence, holdout, holdout_evidence, capability_sha, license_record = \
            run_two_stage(model, torch, F)
    if not fit["passed"]:
        terminal, forwards, evaluations = "fit_failed", 1, 48
    elif not holdout["passed"]:
        terminal, forwards, evaluations = "holdout_failed", 2, 96
    else:
        terminal, forwards, evaluations = "licensed", 2, 96
    result = {
        "schema": "task14_fresh_matched_natural_native_capability_result_v1",
        "candidate_id": authority.CAPABILITY_ID, "terminal": terminal,
        "plan": plan, "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "fit": fit, "fit_evidence": fit_evidence,
        "holdout": holdout, "holdout_evidence": holdout_evidence,
        "capability_result_sha256": capability_sha, "license": license_record,
        "predictions": {
            "pred_a_fit_all_cells_pass": bool(fit["passed"]),
            "pred_b_holdout_all_cells_pass": bool(holdout and holdout["passed"]),
            "pred_c_license_issued": bool(license_record and license_record["sha256"]),
        },
        "active_price": {"model_forwards": forwards, "example_evaluations": evaluations,
                         "backwards": 0, "parameter_updates": 0},
        "causal_interventions": 0,
    }
    payload = managed.atomic_create_json(RESULT, result)
    print(json.dumps({"terminal": terminal, "result_sha256": hashlib.sha256(payload).hexdigest(),
                      "license_sha256": license_record["sha256"] if license_record else None},
                     sort_keys=True))


if __name__ == "__main__":
    main()
