#!/usr/bin/env python3
"""Run the frozen fresh-lexicon A1/A2 narrative native capability gate."""

# BQGATE: EXPERIMENT pred_a_fit_all_cells_pass pred_b_holdout_all_cells_pass pred_c_license_issued

from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
from typing import Mapping, Sequence

import circuit_fast_screen_candidate_narrative_tense_newlex_a1_a2_authority as authority
import circuit_fast_screen_managed_runner as managed
import native_capability_license as licensing
import run_task14_head11_3_subject_attractor_score_payload_factorial as model_helpers


ROOT = Path(__file__).resolve().parent.parent
PREREG = ROOT / "circuits/prior_art/NARRATIVE_TENSE_NEWLEX_A1_A2_NATIVE_CAPABILITY_V1_PREREGISTRATION.md"
RESULT = ROOT / "circuits/fast_screens/narrative_tense_newlex_a1_a2_native_capability_v1_result.json"
CAPABILITY_RESULT = ROOT / "circuits/fast_screens/narrative_tense_newlex_a1_a2_native_capability_v1_capability.json"
LICENSE = ROOT / "circuits/fast_screens/narrative_tense_newlex_a1_a2_native_capability_v1_license.json"
PREREG_SHA256 = "39a4a69c66b24db1a3a302df26d53a52b5e7410fb169cca0ef6d4c0d28a1368d"
AUTHORITY_FILE_SHA256 = "5128fe4277fe366a4b2a48b338f7fe9cfc9808236a9402bafde2c37c985d1b41"
AUTHORITY_LOGICAL_SHA256 = "8527c9d13f964d694ada01384bf179ec2e6a245708eb4cf1da056b5acc7dbc76"
MINIMUM_ACCURACY = .875


class CapabilityError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cell_id(row: Mapping[str, object], side: str) -> str:
    return "/".join((str(row["phase"]), str(row["family"]),
                     str(row["direction_id"]), side))


def build_gate() -> licensing.CapabilityGate:
    if _sha256(PREREG) != PREREG_SHA256 \
            or _sha256(Path(authority.__file__)) != AUTHORITY_FILE_SHA256 \
            or authority.authority_sha256() != AUTHORITY_LOGICAL_SHA256:
        raise CapabilityError("frozen preregistration or authority changed")
    counts = defaultdict(int)
    for row in authority.build_rows():
        for side in ("base", "donor"):
            counts[_cell_id(row, side)] += 1
    gate = licensing.CapabilityGate(
        capability_id=authority.CAPABILITY_ID,
        authority_path=Path(authority.__file__),
        expected_authority_file_sha256=AUTHORITY_FILE_SHA256,
        authority_logical_sha256=AUTHORITY_LOGICAL_SHA256,
        cells=tuple(licensing.CapabilityCell(cell, count, MINIMUM_ACCURACY)
                    for cell, count in sorted(counts.items())),
    )
    if len(gate.cells) != 32 or any(cell.expected_count != 4 for cell in gate.cells):
        raise CapabilityError("registered capability cell layout changed")
    licensing.validate_gate(gate)
    return gate


def compile_plan() -> dict:
    gate = build_gate()
    return {
        "schema": "narrative_tense_newlex_a1_a2_native_capability_plan_v1",
        "candidate_id": authority.CAPABILITY_ID,
        "model_loaded": False, "gpu_accessed": False, "queue_touched": False,
        "native_only": True, "causal_interventions": 0,
        "authority_file_sha256": AUTHORITY_FILE_SHA256,
        "authority_logical_sha256": AUTHORITY_LOGICAL_SHA256,
        "preregistration_sha256": PREREG_SHA256,
        "fit_groups": list(authority.FIT_GROUPS),
        "holdout_groups": list(authority.HOLDOUT_GROUPS),
        "registered_cells": len(gate.cells),
        "minimum_accuracy_each_cell": MINIMUM_ACCURACY,
        "maximum_price": {"model_forwards": 2, "example_evaluations": 128,
                          "backwards": 0, "parameter_updates": 0},
        "interpretation_limit": "Native dataset capability only; no carrier evidence.",
    }


def _pad(rows, torch, device):
    endpoints = [(row, side) for row in rows for side in ("base", "donor")]
    length = max(len(row[f"{side}_ids"]) for row, side in endpoints)
    tokens = torch.full((len(endpoints), length), 50256, dtype=torch.long, device=device)
    finals = []
    for index, (row, side) in enumerate(endpoints):
        ids = row[f"{side}_ids"]
        tokens[index, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        finals.append(len(ids) - 1)
    return endpoints, tokens, torch.tensor(finals, dtype=torch.long, device=device)


def evaluate(model, rows, torch, F):
    endpoints, tokens, finals = _pad(rows, torch, next(model.parameters()).device)
    logits = model_helpers._native_logits(model, tokens, torch, F)
    evidence = []
    for index, (row, side) in enumerate(endpoints):
        query = int(finals[index])
        answer, foil = int(row[f"{side}_answer_id"]), int(row[f"{side}_foil_id"])
        margin = float(logits[index, query, answer] - logits[index, query, foil])
        ce = float(-torch.log_softmax(logits[index, query], dim=-1)[answer])
        evidence.append({
            "example_id": f"{row['row_id']}:{side}",
            "cell_id": _cell_id(row, side), "correct": bool(margin > 0),
            "full_vocab_CE": ce, "answer_minus_foil_margin": margin,
        })
    return evidence


def _summary(evidence):
    grouped = defaultdict(list)
    for item in evidence:
        grouped[item["cell_id"]].append(item)
    return {cell: {"count": len(items),
                   "accuracy": statistics.fmean(x["correct"] for x in items),
                   "mean_margin": statistics.fmean(x["answer_minus_foil_margin"] for x in items),
                   "mean_full_vocab_CE": statistics.fmean(x["full_vocab_CE"] for x in items)}
            for cell, items in sorted(grouped.items())}


def _phase_pass(evidence, phase: str) -> bool:
    summary = _summary(evidence)
    selected = {cell: value for cell, value in summary.items()
                if cell.startswith(phase + "/")}
    return len(selected) == 16 and all(value["count"] == 4
                                       and value["accuracy"] >= MINIMUM_ACCURACY
                                       for value in selected.values())


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    plan = compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1" \
            or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True)); return
    if any(path.exists() for path in (RESULT, CAPABILITY_RESULT, LICENSE)):
        raise CapabilityError("refusing to overwrite an output artifact")
    torch, F, facade = model_helpers._dependencies()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    rows = authority.build_rows()
    with torch.no_grad():
        fit = evaluate(model, [row for row in rows if row["phase"] == "FIT"], torch, F)
        fit_pass = _phase_pass(fit, "FIT")
        holdout = (evaluate(model, [row for row in rows if row["phase"] == "HOLDOUT"],
                            torch, F) if fit_pass else [])
    capability = capability_sha = license_value = license_sha = None
    if fit_pass:
        gate = build_gate()
        capability, capability_sha = licensing.finalize_native_capability(
            gate, fit + holdout, CAPABILITY_RESULT)
        if capability["terminal"] == "pass":
            license_value, license_sha = licensing.issue_capability_license(
                gate, CAPABILITY_RESULT, LICENSE,
                causal_candidate_id=authority.CAUSAL_CANDIDATE_ID)
            licensing.validate_causal_preflight(
                gate, CAPABILITY_RESULT, LICENSE, expected_license_sha256=license_sha,
                causal_candidate_id=authority.CAUSAL_CANDIDATE_ID)
    holdout_pass = bool(capability and capability["terminal"] == "pass")
    terminal = "licensed" if holdout_pass else "holdout_failed" if fit_pass else "fit_failed"
    result = {
        "schema": "narrative_tense_newlex_a1_a2_native_capability_result_v1",
        "candidate_id": authority.CAPABILITY_ID, "terminal": terminal, "plan": plan,
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "fit_evidence": fit, "fit_summary": _summary(fit),
        "holdout_evidence": holdout, "holdout_summary": _summary(holdout),
        "capability_result_sha256": capability_sha,
        "license": {"value": license_value, "sha256": license_sha},
        "predictions": {"pred_a_fit_all_cells_pass": fit_pass,
                        "pred_b_holdout_all_cells_pass": holdout_pass,
                        "pred_c_license_issued": license_sha is not None},
        "active_price": {"model_forwards": 2 if fit_pass else 1,
                         "example_evaluations": 128 if fit_pass else 64,
                         "backwards": 0, "parameter_updates": 0},
        "causal_interventions": 0,
    }
    digest = hashlib.sha256(managed.atomic_create_json(RESULT, result)).hexdigest()
    print(json.dumps({"terminal": terminal, "result_sha256": digest,
                      "license_sha256": license_sha}, sort_keys=True))


if __name__ == "__main__":
    main()
