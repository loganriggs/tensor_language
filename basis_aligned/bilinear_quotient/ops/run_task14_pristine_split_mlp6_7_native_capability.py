#!/usr/bin/env python3
"""Native-only capability license for pristine absolute MLP6--7 composition transfer."""

# BQGATE: EXPERIMENT pred_a_authority_valid pred_b_native_capability_pass pred_c_license_issued
from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
import os
from pathlib import Path

import circuit_fast_screen_candidate_task14_pristine_split_mlp6_7_absolute_composition as authority
import native_capability_license as licensing
import run_task14_head11_3_subject_attractor_score_payload_factorial as model_helpers


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/task14_pristine_split_mlp6_7_absolute_composition_transfer_v1.json"
RESULT = ROOT / "circuits/fast_screens/task14_pristine_split_mlp6_7_native_capability_v1_result.json"
LICENSE = ROOT / "circuits/fast_screens/task14_pristine_split_mlp6_7_absolute_composition_v1_capability_license.json"
PRIOR_ART_SHA256 = "4a06619140c165c99b7ec5930e68bd8d2d3c6f5205722f31997ef1ee9cf8a06e"
AUTHORITY_FILE_SHA256 = "429124ece9c746f425b5bdd340280bbb4ac1ccb256302a92a4c45c5dba405ead"
MINIMUM_ACCURACY = .75


class PristineCapabilityError(ValueError):
    pass


def _cell_id(row, role):
    return f"{row['phase']}__{row['direction_id']}__{row['template_id']}__{role}"


def build_gate():
    if hashlib.sha256(PRIOR_ART.read_bytes()).hexdigest() != PRIOR_ART_SHA256:
        raise PristineCapabilityError("prior-art receipt changed")
    rows = authority.build_rows()
    counts = Counter(_cell_id(row, role) for row in rows for role in authority.ROLES)
    if len(counts) != 18 or sorted(counts.values()) != [4]*6 + [8]*12:
        raise PristineCapabilityError(f"capability cells changed: {counts}")
    value = licensing.CapabilityGate(capability_id=authority.CAPABILITY_ID,
        authority_path=Path(authority.__file__),
        expected_authority_file_sha256=AUTHORITY_FILE_SHA256,
        authority_logical_sha256=authority.EXPECTED_AUTHORITY_SHA256,
        cells=tuple(licensing.CapabilityCell(cell, count, MINIMUM_ACCURACY)
                    for cell, count in sorted(counts.items())))
    licensing.validate_gate(value); return value


def compile_plan():
    value = build_gate()
    return {"schema": "task14_pristine_split_mlp6_7_native_capability_plan_v1",
        "capability_id": authority.CAPABILITY_ID,
        "causal_candidate_id": authority.CAUSAL_CANDIDATE_ID,
        "split": "NEW_FIT_AND_UNTOUCHED_HOLDOUT_TEXT_NATIVE_ONLY",
        "native_only": True, "row_count": 40, "endpoint_evaluations": 120,
        "minimum_accuracy_each_phase_direction_template_role_cell": MINIMUM_ACCURACY,
        "prior_art_sha256": PRIOR_ART_SHA256,
        "authority_file_sha256": AUTHORITY_FILE_SHA256,
        "authority_logical_sha256": authority.EXPECTED_AUTHORITY_SHA256,
        "registered_cells_sha256": licensing.cells_sha256(value),
        "predictions": {"pred_a_authority_valid": "frozen pristine authority passes validation",
            "pred_b_native_capability_pass": "all eighteen cells reach 0.75 accuracy",
            "pred_c_license_issued": "candidate-scoped license exists only after a pass"},
        "price": {"model_forwards": 1, "example_evaluations": 120,
                  "causal_interventions": 0, "backwards": 0,
                  "parameter_updates": 0}}


def evaluate_native(model, torch, F):
    rows = authority.build_rows(); examples = [(row, role) for row in rows for role in authority.ROLES]
    device = next(model.parameters()).device
    tokens = torch.tensor([row["endpoints"][role]["ids"] for row, role in examples],
                          dtype=torch.long, device=device)
    logits = model_helpers._native_logits(model, tokens, torch, F)
    evidence = []
    for i, (row, role) in enumerate(examples):
        endpoint = row["endpoints"][role]; answer = endpoint["answer_id"]; foil = endpoint["foil_id"]
        margin = float(logits[i, authority.SUBJECT_POSITION, answer]
                       - logits[i, authority.SUBJECT_POSITION, foil])
        ce = float(-torch.log_softmax(logits[i, authority.SUBJECT_POSITION], dim=-1)[answer])
        evidence.append({"example_id": f"{row['row_id']}:{role}",
            "cell_id": _cell_id(row, role), "correct": bool(margin > 0),
            "full_vocab_CE": ce, "answer_minus_foil_margin": margin})
    return evidence


def finalize(evidence):
    if RESULT.exists() or LICENSE.exists():
        raise PristineCapabilityError("refusing to overwrite capability result or license")
    value = build_gate()
    result, result_sha = licensing.finalize_native_capability(value, evidence, RESULT)
    if result["terminal"] != "pass": return result, result_sha, None, None
    license_value, license_sha = licensing.issue_capability_license(
        value, RESULT, LICENSE, causal_candidate_id=authority.CAUSAL_CANDIDATE_ID)
    licensing.validate_causal_preflight(value, RESULT, LICENSE,
        expected_license_sha256=license_sha,
        causal_candidate_id=authority.CAUSAL_CANDIDATE_ID)
    return result, result_sha, license_value, license_sha


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv); plan = compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True)); return
    torch, F, facade = model_helpers._dependencies()
    model, _ = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    with torch.no_grad(): evidence = evaluate_native(model, torch, F)
    result, result_sha, _, license_sha = finalize(evidence)
    print(json.dumps({"terminal": result["terminal"],
        "capability_result_sha256": result_sha, "license_sha256": license_sha}, sort_keys=True))


if __name__ == "__main__": main()
