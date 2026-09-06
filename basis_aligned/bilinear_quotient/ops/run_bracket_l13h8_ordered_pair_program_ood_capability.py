#!/usr/bin/env python3
"""Capability-only gate for the still-closed bracket OOD authority."""

# BQGATE: EXPERIMENT pred_a_ood_authority_complete pred_b_native_target_capability pred_c_native_control_capability pred_d_capability_only_price
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

import circuit_fast_screen_candidate_bracket_l13h8_ordered_pair_displacement_program as authority
import circuit_fast_screen_managed_runner as managed
import run_bracket_l13h8_source_region_payload_factorial as exact


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "circuits/fast_screens/bracket_l13h8_ordered_pair_displacement_artifact_v1.json"
OUT = ROOT / "circuits/fast_screens/bracket_l13h8_ordered_pair_program_ood_capability_v1_result.json"
ARTIFACT_SHA256 = "531434535daf526ebf584564afbfd5834c75df7bd636014ea233f9ae71206db0"
MINIMUM_ACCURACY = .75


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_plan():
    rows = authority.build_ood_rows()
    if _sha(ARTIFACT) != ARTIFACT_SHA256:
        raise ValueError("committed prototype artifact changed")
    artifact = json.loads(ARTIFACT.read_text())
    if artifact.get("terminal") != "prototype_artifact" or not all(artifact["predictions"].values()):
        raise ValueError("prototype artifact is not licensed")
    return {
        "schema": "bracket_l13h8_ordered_pair_program_ood_capability_plan_v1",
        "candidate_id": authority.CANDIDATE_ID,
        "prior_art_sha256": authority.PRIOR_ART_SHA256,
        "artifact_sha256": ARTIFACT_SHA256,
        "split": "OOD", "rows": len(rows), "endpoints": 2 * len(rows),
        "causal_interventions": 0, "program_vectors_installed": 0,
        "minimum_accuracy_each_cell": MINIMUM_ACCURACY,
        "price": {"model_forwards": 1, "example_evaluations": 360,
                  "backwards": 0, "parameter_updates": 0},
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
    return endpoints, tokens, torch.tensor(finals, device=device)


def score(evidence):
    groups = defaultdict(list)
    for item in evidence:
        groups[item["cell_id"]].append(item)
    reports = {}
    for key, items in sorted(groups.items()):
        reports[key] = {
            "n": len(items),
            "accuracy": sum(item["correct"] for item in items) / len(items),
            "mean_closer_margin": statistics.fmean(item["closer_margin"] for item in items),
        }
        reports[key]["passed"] = reports[key]["accuracy"] >= MINIMUM_ACCURACY
    target = {key: value for key, value in reports.items() if key.startswith("target|")}
    control = {key: value for key, value in reports.items() if key.startswith("control|")}
    predictions = {
        "pred_a_ood_authority_complete": len(evidence) == 360 and len(target) == 12 and len(control) == 6
                                          and {x["n"] for x in target.values()} == {12}
                                          and {x["n"] for x in control.values()} == {36},
        "pred_b_native_target_capability": bool(target) and all(x["passed"] for x in target.values()),
        "pred_c_native_control_capability": bool(control) and all(x["passed"] for x in control.values()),
        "pred_d_capability_only_price": True,
    }
    return {"cell_reports": reports, "predictions": predictions,
            "terminal": "capability_pass" if all(predictions.values()) else "capability_null"}


def evaluate(model, torch, F):
    rows = authority.build_ood_rows()
    endpoints, tokens, finals = _pad(rows, torch, next(model.parameters()).device)
    logits = exact.native_logits(model, tokens, torch, F)
    evidence = []
    for index, (row, side) in enumerate(endpoints):
        q = int(finals[index]); answer = row[f"{side}_answer_id"]
        if row["program_role"] == "target":
            other = "donor" if side == "base" else "base"
            cell = f"target|{row['family_id']}|{answer}->{row[f'{other}_answer_id']}"
        else:
            cell = f"control|{row['family_id']}|{side}"
        margin = exact.closer_margin(logits[index, q], answer)
        evidence.append({"row_id": row["row_id"], "family_id": row["family_id"],
                         "program_role": row["program_role"], "side": side,
                         "cell_id": cell, "answer_id": answer,
                         "closer_margin": margin, "correct": bool(margin > 0)})
    return evidence


def main():
    plan = compile_plan()
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1" \
            or "--dry-run" in sys.argv:
        print(json.dumps(plan, sort_keys=True)); return
    if OUT.exists():
        raise RuntimeError(f"refusing overwrite {OUT}")
    torch, F, facade = exact._dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32,
                                             verify_weights_sha256=True)
    with torch.no_grad():
        evidence = evaluate(model, torch, F)
    scored = score(evidence)
    payload = managed.atomic_create_json(OUT, {
        "schema": "bracket_l13h8_ordered_pair_program_ood_capability_result_v1",
        "candidate_id": authority.CANDIDATE_ID,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "plan": plan, "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "evidence": evidence, **scored,
    })
    print(json.dumps({"terminal": scored["terminal"],
                      "predictions": scored["predictions"],
                      "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
