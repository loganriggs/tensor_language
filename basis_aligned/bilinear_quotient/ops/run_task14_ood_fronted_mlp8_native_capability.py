#!/usr/bin/env python3
"""Issue the scoped native-capability license for the OOD MLP8 experiment."""

# BQGATE: EXPERIMENT pred_a_authority_valid pred_b_native_capability_pass pred_c_license_issued
from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
import os
from pathlib import Path

import circuit_fast_screen_candidate_task14_ood_fronted_mlp8_polarized_response as authority
import native_capability_license as licensing
import run_task14_head11_3_subject_attractor_score_payload_factorial as model_helpers


ROOT = Path(__file__).resolve().parent.parent
RESULT = ROOT / "circuits/fast_screens/task14_ood_fronted_mlp8_native_capability_v1_result.json"
LICENSE = ROOT / "circuits/fast_screens/task14_ood_fronted_mlp8_polarized_response_v1_capability_license.json"
AUTHORITY_FILE_SHA256 = "9945ef76cb65fe3717f54be478dab2d7444f92738c4075c3ea47b54ab252cccb"
AUTHORITY_LOGICAL_SHA256 = authority.EXPECTED_AUTHORITY_SHA256
MINIMUM_ACCURACY = .875


class OODMLP8CapabilityError(ValueError):
    pass


def _cell_id(row, role):
    return f"{row['direction_id']}__{role}"


def build_gate():
    rows = authority.build_rows()
    counts = Counter(_cell_id(row, role) for row in rows for role in authority.ROLES)
    if len(counts) != 6 or set(counts.values()) != {8}:
        raise OODMLP8CapabilityError(f"capability cells changed: {counts}")
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


def compile_plan():
    gate = build_gate()
    return {
        "schema": "task14_ood_fronted_mlp8_native_capability_plan_v1",
        "capability_id": authority.CAPABILITY_ID,
        "causal_candidate_id": authority.CAUSAL_CANDIDATE_ID,
        "split": authority.SPLIT,
        "data_status": "previously opened OOD text; scoped native gate for a new MLP8 intervention",
        "native_only": True,
        "row_count": 16,
        "endpoint_evaluations": 48,
        "minimum_accuracy_each_direction_role_cell": MINIMUM_ACCURACY,
        "authority_file_sha256": AUTHORITY_FILE_SHA256,
        "authority_logical_sha256": AUTHORITY_LOGICAL_SHA256,
        "registered_cells_sha256": licensing.cells_sha256(gate),
        "predictions": {
            "pred_a_authority_valid": "the exact 16-row authority passes all static validation",
            "pred_b_native_capability_pass": "all six direction-role cells reach 0.875 native accuracy",
            "pred_c_license_issued": "a candidate-scoped license is issued only after a complete pass",
        },
        "result_path": str(RESULT),
        "license_path": str(LICENSE),
        "price": {"model_forwards": 1, "example_evaluations": 48,
                  "causal_interventions": 0, "backwards": 0, "parameter_updates": 0},
    }


def evaluate_native(model, torch, F):
    rows = authority.build_rows()
    examples = [(row, role) for row in rows for role in authority.ROLES]
    device = next(model.parameters()).device
    tokens = torch.tensor([row["endpoints"][role]["ids"] for row, role in examples],
                          dtype=torch.long, device=device)
    logits = model_helpers._native_logits(model, tokens, torch, F)
    evidence = []
    for index, (row, role) in enumerate(examples):
        endpoint = row["endpoints"][role]
        answer, foil = int(endpoint["answer_id"]), int(endpoint["foil_id"])
        margin = float(logits[index, authority.SUBJECT_POSITION, answer]
                       - logits[index, authority.SUBJECT_POSITION, foil])
        ce = float(-torch.log_softmax(
            logits[index, authority.SUBJECT_POSITION], dim=-1)[answer])
        evidence.append({
            "example_id": f"{row['row_id']}:{role}",
            "cell_id": _cell_id(row, role),
            "correct": bool(margin > 0),
            "full_vocab_CE": ce,
            "answer_minus_foil_margin": margin,
        })
    return evidence


def finalize(evidence):
    if RESULT.exists() or LICENSE.exists():
        raise OODMLP8CapabilityError("refusing to overwrite capability result or license")
    gate = build_gate()
    result, result_sha = licensing.finalize_native_capability(gate, evidence, RESULT)
    if result["terminal"] != "pass":
        return result, result_sha, None, None
    license_value, license_sha = licensing.issue_capability_license(
        gate, RESULT, LICENSE, causal_candidate_id=authority.CAUSAL_CANDIDATE_ID)
    licensing.validate_causal_preflight(
        gate, RESULT, LICENSE, expected_license_sha256=license_sha,
        causal_candidate_id=authority.CAUSAL_CANDIDATE_ID)
    return result, result_sha, license_value, license_sha


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    plan = compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1" \
            or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True))
        return
    torch, F, facade = model_helpers._dependencies()
    model, _checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    with torch.no_grad():
        evidence = evaluate_native(model, torch, F)
    result, result_sha, _license, license_sha = finalize(evidence)
    print(json.dumps({"terminal": result["terminal"],
                      "capability_result_sha256": result_sha,
                      "license_sha256": license_sha}, sort_keys=True))


if __name__ == "__main__":
    main()
