#!/usr/bin/env python3
"""Native-only capability gate for fresh subject-number rank-one confirmation."""

# BQGATE: EXPERIMENT pred_a_authority_valid pred_b_native_capability_pass pred_c_license_issued
from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
import os
from pathlib import Path

import circuit_fast_screen_candidate_subject_number_rank1_fresh_confirmation as authority
import native_capability_license as licensing
import run_task14_head11_3_subject_attractor_score_payload_factorial as helpers


RUNNER = Path(__file__).resolve()
ROOT = RUNNER.parent.parent
PRIOR_ART = ROOT / "../polynomial_causal/SUBJECT_NUMBER_DIRECTION_CARDINALITY_RANK1_V1_ARTIFACT.json"
RESULT = ROOT / "circuits/fast_screens/subject_number_rank1_fresh_confirmation_native_capability_v1_result.json"
LICENSE = ROOT / "circuits/fast_screens/subject_number_rank1_fresh_confirmation_v1_license.json"
PRIOR_ART_SHA256 = "f73126268794342809c2953b388ed83f6d1d0e2e245ce9ea78f2d665a0e87c1f"
AUTHORITY_FILE_SHA256 = "01d990ccdaa71d96d6e8fa7d496c6159a255f624ff036fb4605c91936cdbca61"
MINIMUM_ACCURACY = 0.75
PREDICTION_REGISTRY = {
    "pred_a_authority_valid": None,
    "pred_b_native_capability_pass": None,
    "pred_c_license_issued": None,
}


class CapabilityError(ValueError):
    pass


def _cell(row: dict[str, object], role: str) -> str:
    return f"{row['direction_id']}__{row['template_id']}__{role}"


def build_gate() -> licensing.CapabilityGate:
    if hashlib.sha256(PRIOR_ART.read_bytes()).hexdigest() != PRIOR_ART_SHA256:
        raise CapabilityError("prior art changed")
    rows = authority.build_rows()
    counts = Counter(_cell(row, role) for row in rows for role in authority.ROLES)
    if len(counts) != 12 or set(counts.values()) != {8}:
        raise CapabilityError("capability cells changed")
    gate = licensing.CapabilityGate(
        capability_id=authority.CAPABILITY_ID,
        authority_path=Path(authority.__file__),
        expected_authority_file_sha256=AUTHORITY_FILE_SHA256,
        authority_logical_sha256=authority.EXPECTED_AUTHORITY_SHA256,
        cells=tuple(licensing.CapabilityCell(key, count, MINIMUM_ACCURACY) for key, count in sorted(counts.items())),
    )
    licensing.validate_gate(gate)
    return gate


def compile_plan() -> dict[str, object]:
    gate = build_gate()
    return {
        "schema": "subject_number_rank1_fresh_confirmation_native_capability_plan_v1",
        "capability_id": authority.CAPABILITY_ID,
        "causal_candidate_id": authority.CAUSAL_CANDIDATE_ID,
        "split": "FRESH_SUBJECT_NUMBER_RANK1_CONFIRMATION_TEXT_NATIVE_ONLY",
        "native_only": True,
        "row_count": 32,
        "endpoint_evaluations": 96,
        "minimum_accuracy_each_direction_template_role_cell": MINIMUM_ACCURACY,
        "prior_art_sha256": PRIOR_ART_SHA256,
        "authority_file_sha256": AUTHORITY_FILE_SHA256,
        "authority_logical_sha256": authority.EXPECTED_AUTHORITY_SHA256,
        "registered_cells_sha256": licensing.cells_sha256(gate),
        "predictions": {
            "pred_a_authority_valid": "frozen third-corpus authority passes structural and novelty checks",
            "pred_b_native_capability_pass": "all twelve direction-template-role cells reach 0.75 accuracy",
            "pred_c_license_issued": "candidate-scoped license exists only after pass",
        },
        "price": {"model_forwards": 1, "example_evaluations": 96, "causal_interventions": 0, "backwards": 0, "parameter_updates": 0},
    }


def evaluate(model, torch, F) -> list[dict[str, object]]:
    rows = authority.build_rows()
    examples = [(row, role) for row in rows for role in authority.ROLES]
    device = next(model.parameters()).device
    tokens = torch.tensor([row["endpoints"][role]["ids"] for row, role in examples], dtype=torch.long, device=device)
    logits = helpers._native_logits(model, tokens, torch, F)
    evidence = []
    for index, (row, role) in enumerate(examples):
        endpoint = row["endpoints"][role]
        margin = float(logits[index, authority.SUBJECT_POSITION, endpoint["answer_id"]] - logits[index, authority.SUBJECT_POSITION, endpoint["foil_id"]])
        ce = float(-torch.log_softmax(logits[index, authority.SUBJECT_POSITION], dim=-1)[endpoint["answer_id"]])
        evidence.append({
            "example_id": f"{row['row_id']}:{role}", "cell_id": _cell(row, role),
            "correct": bool(margin > 0), "full_vocab_CE": ce,
            "answer_minus_foil_margin": margin,
        })
    return evidence


def finalize(evidence: list[dict[str, object]]):
    if RESULT.exists() or LICENSE.exists():
        raise CapabilityError("refusing overwrite")
    gate = build_gate()
    result, result_sha = licensing.finalize_native_capability(gate, evidence, RESULT)
    if result["terminal"] != "pass":
        return result, result_sha, None
    _, license_sha = licensing.issue_capability_license(gate, RESULT, LICENSE, causal_candidate_id=authority.CAUSAL_CANDIDATE_ID)
    licensing.validate_causal_preflight(gate, RESULT, LICENSE, expected_license_sha256=license_sha, causal_candidate_id=authority.CAUSAL_CANDIDATE_ID)
    return result, result_sha, license_sha


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    plan = compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True))
        return
    torch, F, facade = helpers._dependencies()
    model, _ = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    with torch.no_grad():
        evidence = evaluate(model, torch, F)
    result, result_sha, license_sha = finalize(evidence)
    print(json.dumps({"terminal": result["terminal"], "capability_result_sha256": result_sha, "license_sha256": license_sha}, sort_keys=True))


if __name__ == "__main__":
    main()
