#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_authority_valid pred_b_native_capability_pass pred_c_license_issued
"""Native-only capability gate for the fourth subject-number corpus."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import circuit_fast_screen_candidate_subject_number_response_weighted_fresh as authority
import native_capability_license as licensing
import run_task14_head11_3_subject_attractor_score_payload_factorial as helpers


RUNNER = Path(__file__).resolve()
ROOT = RUNNER.parent.parent
PRIOR_ART = ROOT.parent / "polynomial_causal/SUBJECT_NUMBER_RESPONSE_WEIGHTED_PROTOTYPE_FROZEN_V1_ARTIFACT.json"
RESULT = ROOT / "circuits/fast_screens/subject_number_response_weighted_fresh_capability_v1_result.json"
LICENSE = ROOT / "circuits/fast_screens/subject_number_response_weighted_fresh_v1_license.json"
PRIOR_ART_SHA256 = "85d7a86f4e45438ab57970f92a8612b9e2ca01b4fbab46610cea7a4f6e8aa778"
AUTHORITY_FILE_SHA256 = "b05b7086f917d1344ecf1bd7d74350141252afb771e695e38e336b46f15ab302"
MINIMUM_ACCURACY = .75
PREDICTION_REGISTRY = {"pred_a_authority_valid": None,
                       "pred_b_native_capability_pass": None,
                       "pred_c_license_issued": None}


class CapabilityError(ValueError):
    pass


def _cell(row, role):
    return f"{row['direction_id']}__{row['template_id']}__{role}"


def build_gate():
    if hashlib.sha256(PRIOR_ART.read_bytes()).hexdigest() != PRIOR_ART_SHA256:
        raise CapabilityError("frozen prototype artifact changed")
    rows = authority.build_rows()
    counts = Counter(_cell(row, role) for row in rows for role in authority.ROLES)
    if len(counts) != 12 or set(counts.values()) != {8}:
        raise CapabilityError("capability cells changed")
    gate = licensing.CapabilityGate(
        capability_id=authority.CAPABILITY_ID,
        authority_path=Path(authority.__file__),
        expected_authority_file_sha256=AUTHORITY_FILE_SHA256,
        authority_logical_sha256=authority.EXPECTED_AUTHORITY_SHA256,
        cells=tuple(licensing.CapabilityCell(key, count, MINIMUM_ACCURACY)
                    for key, count in sorted(counts.items())))
    licensing.validate_gate(gate); return gate


def compile_plan():
    gate = build_gate()
    return {"schema": "subject_number_response_weighted_fresh_capability_v1_plan",
            "capability_id": authority.CAPABILITY_ID,
            "causal_candidate_id": authority.CAUSAL_CANDIDATE_ID,
            "split": "FOURTH_CORPUS_RESPONSE_WEIGHTED_SUBJECT_NUMBER_NATIVE_ONLY",
            "native_only": True, "row_count": 32, "endpoint_evaluations": 96,
            "minimum_accuracy_each_direction_template_role_cell": MINIMUM_ACCURACY,
            "prior_art_sha256": PRIOR_ART_SHA256,
            "authority_file_sha256": AUTHORITY_FILE_SHA256,
            "authority_logical_sha256": authority.EXPECTED_AUTHORITY_SHA256,
            "registered_cells_sha256": licensing.cells_sha256(gate),
            "price": {"model_forwards": 1, "example_evaluations": 96,
                      "causal_interventions": 0, "backwards": 0, "parameter_updates": 0}}


def evaluate(model, torch, F):
    rows = authority.build_rows(); examples = [(row, role) for row in rows for role in authority.ROLES]
    device = next(model.parameters()).device
    tokens = torch.tensor([row["endpoints"][role]["ids"] for row, role in examples],
                          dtype=torch.long, device=device)
    logits = helpers._native_logits(model, tokens, torch, F)
    evidence = []
    for index, (row, role) in enumerate(examples):
        endpoint = row["endpoints"][role]
        margin = float(logits[index, authority.SUBJECT_POSITION, endpoint["answer_id"]]
                       - logits[index, authority.SUBJECT_POSITION, endpoint["foil_id"]])
        ce = float(-torch.log_softmax(logits[index, authority.SUBJECT_POSITION], dim=-1)[endpoint["answer_id"]])
        evidence.append({"example_id": f"{row['row_id']}:{role}", "cell_id": _cell(row, role),
                         "correct": bool(margin > 0), "full_vocab_CE": ce,
                         "answer_minus_foil_margin": margin})
    return evidence


def finalize(evidence):
    if RESULT.exists() or LICENSE.exists():
        raise CapabilityError("refusing overwrite")
    gate = build_gate()
    result, result_sha = licensing.finalize_native_capability(gate, evidence, RESULT)
    if result["terminal"] != "pass":
        return result, result_sha, None
    _, license_sha = licensing.issue_capability_license(
        gate, RESULT, LICENSE, causal_candidate_id=authority.CAUSAL_CANDIDATE_ID)
    licensing.validate_causal_preflight(gate, RESULT, LICENSE,
        expected_license_sha256=license_sha, causal_candidate_id=authority.CAUSAL_CANDIDATE_ID)
    return result, result_sha, license_sha


def main():
    plan = compile_plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, sort_keys=True)); return
    torch, F, facade = helpers._dependencies()
    model, _ = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    with torch.no_grad():
        evidence = evaluate(model, torch, F)
    result, result_sha, license_sha = finalize(evidence)
    predictions = {"pred_a_authority_valid": True,
                   "pred_b_native_capability_pass": result["terminal"] == "pass",
                   "pred_c_license_issued": license_sha is not None}
    print(json.dumps({"terminal": result["terminal"], "predictions": predictions,
                      "capability_result_sha256": result_sha, "license_sha256": license_sha}, sort_keys=True))


if __name__ == "__main__":
    main()
