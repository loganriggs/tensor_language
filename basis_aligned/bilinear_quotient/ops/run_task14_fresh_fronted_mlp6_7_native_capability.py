#!/usr/bin/env python3
"""Issue a scoped native-capability license for fresh-fronted MLP6--7 composition."""

# BQGATE: EXPERIMENT pred_a_authority_valid pred_b_native_capability_pass pred_c_license_issued
from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
import os
from pathlib import Path

import circuit_fast_screen_candidate_task14_fresh_fronted_natural_qk_number_specificity as authority
import native_capability_license as licensing
import run_task14_head11_3_subject_attractor_score_payload_factorial as model_helpers


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/task14_fresh_fronted_mlp6_7_background_composition_transfer_v1.json"
RESULT = ROOT / "circuits/fast_screens/task14_fresh_fronted_mlp6_7_native_capability_v1_result.json"
LICENSE = ROOT / "circuits/fast_screens/task14_fresh_fronted_mlp6_7_background_composition_v1_capability_license.json"
PRIOR_ART_SHA256 = "1dbcb5bfbb09e6bf485c3a81c5b2912b5e5af41e596b8331d638d5e71f992cf1"
AUTHORITY_FILE_SHA256 = "245c0551b6d0988143e5caa7e9638623e33e0585af7c9cb3d4db52c9e72de652"
AUTHORITY_LOGICAL_SHA256 = authority.EXPECTED_ROWS_SHA256
CAPABILITY_ID = "subject_verb.number_agreement.fresh_fronted_mlp6_7_native_capability_v1"
CAUSAL_CANDIDATE_ID = "subject_verb.number_agreement.fresh_fronted_mlp6_7_background_composition_transfer_v1"
ROLES = ("recipient", "opposite_same_lemma", "same_number_different_lemma")
ROLE_SOURCE = {"recipient": "base", "opposite_same_lemma": "opposite",
               "same_number_different_lemma": "same"}
MINIMUM_ACCURACY = .875
SUBJECT_POSITION = 8


class FreshFrontedCapabilityError(ValueError):
    pass


def _cell_id(row, role):
    return f"{row['cell_id']}__{role}"


def build_gate():
    if hashlib.sha256(PRIOR_ART.read_bytes()).hexdigest() != PRIOR_ART_SHA256:
        raise FreshFrontedCapabilityError("prior-art receipt changed")
    rows = authority.build_rows()
    counts = Counter(_cell_id(row, role) for row in rows for role in ROLES)
    if len(counts) != 12 or set(counts.values()) != {8}:
        raise FreshFrontedCapabilityError(f"capability cells changed: {counts}")
    value = licensing.CapabilityGate(
        capability_id=CAPABILITY_ID, authority_path=Path(authority.__file__),
        expected_authority_file_sha256=AUTHORITY_FILE_SHA256,
        authority_logical_sha256=AUTHORITY_LOGICAL_SHA256,
        cells=tuple(licensing.CapabilityCell(cell, count, MINIMUM_ACCURACY)
                    for cell, count in sorted(counts.items())))
    licensing.validate_gate(value)
    return value


def compile_plan():
    value = build_gate()
    return {"schema": "task14_fresh_fronted_mlp6_7_native_capability_plan_v1",
            "capability_id": CAPABILITY_ID,
            "causal_candidate_id": CAUSAL_CANDIDATE_ID,
            "split": "FRESH_TEXT_REUSE_NATIVE_ONLY",
            "data_status": "text/native outcomes previously opened; clean candidate-scoped native-only gate",
            "native_only": True, "row_count": 32, "endpoint_evaluations": 96,
            "minimum_accuracy_each_direction_template_role_cell": MINIMUM_ACCURACY,
            "prior_art_sha256": PRIOR_ART_SHA256,
            "authority_file_sha256": AUTHORITY_FILE_SHA256,
            "authority_logical_sha256": AUTHORITY_LOGICAL_SHA256,
            "registered_cells_sha256": licensing.cells_sha256(value),
            "predictions": {"pred_a_authority_valid": "frozen authority passes static validation",
                "pred_b_native_capability_pass": "all twelve cells reach 0.875 native accuracy",
                "pred_c_license_issued": "license exists only after a complete pass"},
            "price": {"model_forwards": 1, "example_evaluations": 96,
                      "causal_interventions": 0, "backwards": 0,
                      "parameter_updates": 0}}


def evaluate_native(model, torch, F):
    rows = authority.build_rows()
    examples = [(row, role) for row in rows for role in ROLES]
    device = next(model.parameters()).device
    tokens = torch.tensor([row[f"{ROLE_SOURCE[role]}_ids"] for row, role in examples],
                          dtype=torch.long, device=device)
    logits = model_helpers._native_logits(model, tokens, torch, F)
    evidence = []
    for index, (row, role) in enumerate(examples):
        source = ROLE_SOURCE[role]
        answer = int(row[f"{source}_answer_id"])
        foil = 389 if answer == 318 else 318
        margin = float(logits[index, SUBJECT_POSITION, answer]
                       - logits[index, SUBJECT_POSITION, foil])
        ce = float(-torch.log_softmax(logits[index, SUBJECT_POSITION], dim=-1)[answer])
        evidence.append({"example_id": f"{row['row_id']}:{role}",
                         "cell_id": _cell_id(row, role), "correct": bool(margin > 0),
                         "full_vocab_CE": ce, "answer_minus_foil_margin": margin})
    return evidence


def finalize(evidence):
    if RESULT.exists() or LICENSE.exists():
        raise FreshFrontedCapabilityError("refusing to overwrite capability result or license")
    value = build_gate()
    result, result_sha = licensing.finalize_native_capability(value, evidence, RESULT)
    if result["terminal"] != "pass":
        return result, result_sha, None, None
    license_value, license_sha = licensing.issue_capability_license(
        value, RESULT, LICENSE, causal_candidate_id=CAUSAL_CANDIDATE_ID)
    licensing.validate_causal_preflight(
        value, RESULT, LICENSE, expected_license_sha256=license_sha,
        causal_candidate_id=CAUSAL_CANDIDATE_ID)
    return result, result_sha, license_value, license_sha


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv); plan = compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True)); return
    torch, F, facade = model_helpers._dependencies()
    model, _checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32,
                                             verify_weights_sha256=True)
    with torch.no_grad():
        evidence = evaluate_native(model, torch, F)
    result, result_sha, _license, license_sha = finalize(evidence)
    print(json.dumps({"terminal": result["terminal"],
                      "capability_result_sha256": result_sha,
                      "license_sha256": license_sha}, sort_keys=True))


if __name__ == "__main__":
    main()
