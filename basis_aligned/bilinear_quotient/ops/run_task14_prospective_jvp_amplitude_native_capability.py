#!/usr/bin/env python3
"""Native-only gate for prospective Task14 downstream-JVP amplitude validation."""

# BQGATE: EXPERIMENT pred_a_authority_valid pred_b_native_capability_pass pred_c_license_issued
from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
import os
from pathlib import Path

import circuit_fast_screen_candidate_task14_prospective_jvp_amplitude as authority
import native_capability_license as licensing
import run_task14_head11_3_subject_attractor_score_payload_factorial as model_helpers


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/task14_prospective_mlp6_7_downstream_midpoint_margin_jvp_amplitude_v1.json"
RESULT = ROOT / "circuits/fast_screens/task14_prospective_jvp_amplitude_native_capability_v1_result.json"
LICENSE = ROOT / "circuits/fast_screens/task14_prospective_mlp6_7_downstream_midpoint_margin_jvp_amplitude_v1_capability_license.json"
PRIOR_ART_SHA256 = "b8e10492f622bb08cb0a2ae4370267e6782f9a4028ed907272d5f2ad04bad030"
AUTHORITY_FILE_SHA256 = "ca0b743dce3ec636924b25e260fd635af306d99b91cf93984e25f1e015e79f12"
MINIMUM_ACCURACY = .75


class ProspectiveCapabilityError(ValueError):
    pass


def _cell_id(row, role):
    return f"{row['direction_id']}__{row['template_id']}__{role}"


def build_gate():
    if hashlib.sha256(PRIOR_ART.read_bytes()).hexdigest() != PRIOR_ART_SHA256:
        raise ProspectiveCapabilityError("prior-art receipt changed")
    rows = authority.build_rows()
    counts = Counter(_cell_id(row, role) for row in rows for role in authority.ROLES)
    if len(counts) != 12 or set(counts.values()) != {8}:
        raise ProspectiveCapabilityError(f"capability cells changed: {counts}")
    gate = licensing.CapabilityGate(capability_id=authority.CAPABILITY_ID,
        authority_path=Path(authority.__file__),
        expected_authority_file_sha256=AUTHORITY_FILE_SHA256,
        authority_logical_sha256=authority.EXPECTED_AUTHORITY_SHA256,
        cells=tuple(licensing.CapabilityCell(cell, count, MINIMUM_ACCURACY)
                    for cell, count in sorted(counts.items())))
    licensing.validate_gate(gate); return gate


def compile_plan():
    gate = build_gate()
    return {"schema": "task14_prospective_jvp_amplitude_native_capability_plan_v1",
        "capability_id": authority.CAPABILITY_ID,
        "causal_candidate_id": authority.CAUSAL_CANDIDATE_ID,
        "split": "NEW_PROSPECTIVE_TEXT_NATIVE_ONLY", "native_only": True,
        "row_count": 32, "endpoint_evaluations": 96,
        "minimum_accuracy_each_direction_template_role_cell": MINIMUM_ACCURACY,
        "prior_art_sha256": PRIOR_ART_SHA256,
        "authority_file_sha256": AUTHORITY_FILE_SHA256,
        "authority_logical_sha256": authority.EXPECTED_AUTHORITY_SHA256,
        "registered_cells_sha256": licensing.cells_sha256(gate),
        "price": {"model_forwards": 1, "example_evaluations": 96,
            "causal_interventions": 0, "backwards": 0, "parameter_updates": 0}}


def evaluate_native(model, torch, F):
    rows = authority.build_rows()
    examples = [(row, role) for row in rows for role in authority.ROLES]
    device = next(model.parameters()).device
    tokens = torch.tensor([row["endpoints"][role]["ids"] for row, role in examples],
                          dtype=torch.long, device=device)
    logits = model_helpers._native_logits(model, tokens, torch, F)
    evidence = []
    for i, (row, role) in enumerate(examples):
        endpoint = row["endpoints"][role]
        margin = float(logits[i, authority.SUBJECT_POSITION, endpoint["answer_id"]]
                       - logits[i, authority.SUBJECT_POSITION, endpoint["foil_id"]])
        ce = float(-torch.log_softmax(logits[i, authority.SUBJECT_POSITION], dim=-1)[endpoint["answer_id"]])
        evidence.append({"example_id": f"{row['row_id']}:{role}",
            "cell_id": _cell_id(row, role), "correct": bool(margin > 0),
            "full_vocab_CE": ce, "answer_minus_foil_margin": margin})
    return evidence


def finalize(evidence):
    if RESULT.exists() or LICENSE.exists():
        raise ProspectiveCapabilityError("refusing to overwrite result or license")
    gate = build_gate()
    result, result_sha = licensing.finalize_native_capability(gate, evidence, RESULT)
    if result["terminal"] != "pass":
        return result, result_sha, None
    _, license_sha = licensing.issue_capability_license(
        gate, RESULT, LICENSE, causal_candidate_id=authority.CAUSAL_CANDIDATE_ID)
    licensing.validate_causal_preflight(gate, RESULT, LICENSE,
        expected_license_sha256=license_sha, causal_candidate_id=authority.CAUSAL_CANDIDATE_ID)
    return result, result_sha, license_sha


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv); plan = compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True)); return
    torch, F, facade = model_helpers._dependencies()
    model, _ = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    with torch.no_grad():
        evidence = evaluate_native(model, torch, F)
    result, result_sha, license_sha = finalize(evidence)
    print(json.dumps({"terminal": result["terminal"],
        "capability_result_sha256": result_sha, "license_sha256": license_sha}, sort_keys=True))


if __name__ == "__main__":
    main()
