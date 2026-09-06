#!/usr/bin/env python3
"""Prospective OOD causal validation of the fixed six-vector bracket program."""

# BQGATE: EXPERIMENT pred_a_export_capability_and_instrument pred_b_fixed_program_substitutes_exact_term pred_c_each_family_and_ordered_pair_recur pred_d_control_zero_dispatch pred_e_fixed_program_and_price
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

import circuit_fast_screen_candidate_bracket_l13h8_ordered_pair_displacement_program as authority
import circuit_fast_screen_managed_runner as managed
import run_bracket_l13h8_ordered_pair_program_ood_capability as capability
import run_bracket_l13h8_source_region_payload_factorial as exact


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "circuits/fast_screens/bracket_l13h8_ordered_pair_displacement_artifact_v1.json"
CAPABILITY = ROOT / "circuits/fast_screens/bracket_l13h8_ordered_pair_program_ood_capability_v1_result.json"
OUT = ROOT / "circuits/followups/bracket_l13h8_ordered_pair_displacement_program_ood_validation_v1_result.json"
ARTIFACT_SHA256 = "531434535daf526ebf584564afbfd5834c75df7bd636014ea233f9ae71206db0"
CAPABILITY_SHA256 = "a1bb465af45b6d7c4059370629d117017aa01e028ed0a985b58d8bbb46da5622"
MAX_ERROR = 1e-4
BARS = {
    "minimum_overall_cosine": .65,
    "maximum_overall_relative_l2_error": .90,
    "minimum_overall_sign_agreement": .75,
    "minimum_overall_norm_ratio": .25,
    "maximum_overall_norm_ratio": 2.0,
    "minimum_family_cosine": .50,
    "minimum_family_sign_agreement": .70,
    "minimum_ordered_pair_positive_fraction": .70,
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_artifact():
    if _sha(ARTIFACT) != ARTIFACT_SHA256 or _sha(CAPABILITY) != CAPABILITY_SHA256:
        raise ValueError("committed artifact or capability receipt changed")
    artifact, gate = json.loads(ARTIFACT.read_text()), json.loads(CAPABILITY.read_text())
    if artifact.get("terminal") != "prototype_artifact" or not all(artifact["predictions"].values()):
        raise ValueError("prototype artifact invalid")
    if gate.get("terminal") != "capability_pass" or not all(gate["predictions"].values()):
        raise ValueError("OOD capability did not pass")
    return artifact, gate


def compile_plan():
    _load_artifact()
    rows = authority.build_ood_rows()
    return {
        "schema": "bracket_l13h8_ordered_pair_displacement_program_ood_validation_plan_v1",
        "candidate_id": authority.CANDIDATE_ID,
        "prior_art_sha256": authority.PRIOR_ART_SHA256,
        "artifact_sha256": ARTIFACT_SHA256,
        "capability_sha256": CAPABILITY_SHA256,
        "split": "OOD", "rows": len(rows), "endpoints": 2 * len(rows),
        "conditions": ["native_factor_replay", "exact_donor_term_swap", "fixed_ordered_pair_vector"],
        "target_installations": 144, "control_zero_dispatches": 216,
        "bars": dict(BARS),
        "stage_price": {"model_forwards": 3, "example_evaluations": 1080,
                        "backwards": 0, "parameter_updates": 0},
        "total_export_capability_causal_price": {
            "model_forwards": 5, "example_evaluations": 1584,
            "backwards": 0, "parameter_updates": 0,
        },
        "fit_operations": 0, "vector_changes": 0,
    }


def _pad(rows, torch, device):
    endpoints = [(row, side) for row in rows for side in ("base", "donor")]
    length = max(len(row[f"{side}_ids"]) for row, side in endpoints)
    tokens = torch.full((len(endpoints), length), 50256, dtype=torch.long, device=device)
    finals, sources = [], []
    for index, (row, side) in enumerate(endpoints):
        ids = row[f"{side}_ids"]
        tokens[index, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        finals.append(len(ids) - 1)
        sources.append(row[f"{side}_open_position"])
    return endpoints, tokens, torch.tensor(finals, device=device), torch.tensor(sources, device=device)


def evaluate(model, torch, F, facade):
    artifact, gate = _load_artifact()
    rows = authority.build_ood_rows()
    endpoints, tokens, finals, sources = _pad(rows, torch, next(model.parameters()).device)
    replay, factors = exact.factor_forward(model, tokens, finals, {}, torch, F, facade)
    arange = torch.arange(len(endpoints), device=tokens.device)
    native_terms = factors["p"][arange, sources].unsqueeze(-1) * factors["u"][arange, sources]
    donor_terms = native_terms[torch.arange(len(endpoints), device=tokens.device) ^ 1]
    vectors = {key: torch.tensor(value["coordinates"], dtype=torch.float32, device=tokens.device)
               for key, value in artifact["prototypes"].items()}
    program_terms, dispatch = [], []
    for index, (row, side) in enumerate(endpoints):
        recipient = row[f"{side}_answer_id"]
        other = "donor" if side == "base" else "base"
        donor = row[f"{other}_answer_id"]
        if row["program_role"] == "target":
            vector = vectors[f"{recipient}->{donor}"]
            dispatch.append(f"{recipient}->{donor}")
        else:
            vector = torch.zeros_like(native_terms[index])
            dispatch.append("zero")
        program_terms.append(native_terms[index] + vector)
    program_terms = torch.stack(program_terms)
    exact_logits = exact.factor_forward(model, tokens, finals, {}, torch, F, facade,
                                        replacement_terms=donor_terms, source_positions=sources)[0]
    program_logits = exact.factor_forward(model, tokens, finals, {}, torch, F, facade,
                                          replacement_terms=program_terms, source_positions=sources)[0]
    native_margin = {(item["row_id"], item["side"]): item["closer_margin"] for item in gate["evidence"]}
    records = []
    for index, (row, side) in enumerate(endpoints):
        q = int(finals[index]); recipient = row[f"{side}_answer_id"]
        other = "donor" if side == "base" else "base"
        donor = row[f"{other}_answer_id"]
        replay_margin_error = abs(exact.closer_margin(replay[index, q], recipient)
                                  - native_margin[(row["row_id"], side)])
        record = {"row_id": row["row_id"], "family_id": row["family_id"],
                  "program_role": row["program_role"], "side": side,
                  "ordered_pair": f"{recipient}->{donor}", "dispatch": dispatch[index],
                  "native_margin_replay_absolute_error": replay_margin_error,
                  "program_max_absolute_logit_change": float((program_logits[index, q] - replay[index, q]).abs().max()),
                  "program_recipient_correct": bool(exact.closer_margin(program_logits[index, q], recipient) > 0)}
        if row["program_role"] == "target":
            record["exact_donorward_effect"] = exact.endpoint_change(
                replay[index, q], exact_logits[index, q], row,
                "base_to_donor" if side == "base" else "donor_to_base")
            record["program_donorward_effect"] = exact.endpoint_change(
                replay[index, q], program_logits[index, q], row,
                "base_to_donor" if side == "base" else "donor_to_base")
        records.append(record)
    return records


def _stats(rows):
    actual = [float(row["exact_donorward_effect"]) for row in rows]
    predicted = [float(row["program_donorward_effect"]) for row in rows]
    dot = sum(a * p for a, p in zip(actual, predicted))
    an = math.sqrt(sum(a * a for a in actual)); pn = math.sqrt(sum(p * p for p in predicted))
    return {"count": len(rows), "cosine": dot / max(an * pn, 1e-30),
            "relative_l2_error": math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted))) / max(an, 1e-30),
            "sign_agreement": sum((a > 0) == (p > 0) for a, p in zip(actual, predicted)) / len(rows),
            "predicted_to_actual_norm_ratio": pn / max(an, 1e-30)}


def score(records):
    targets = [row for row in records if row["program_role"] == "target"]
    controls = [row for row in records if row["program_role"] == "control"]
    overall = _stats(targets)
    by_family = {family: _stats([row for row in targets if row["family_id"] == family])
                 for family in authority.TARGET_FAMILIES}
    by_pair = {}
    for a, b in authority.ORDERED_PAIRS:
        key = f"{a}->{b}"; rows = [row for row in targets if row["ordered_pair"] == key]
        by_pair[key] = {"count": len(rows), "positive_fraction":
                        sum(row["program_donorward_effect"] > 0 for row in rows) / len(rows),
                        **_stats(rows)}
    instrument = (len(records) == 360 and len(targets) == 144 and len(controls) == 216
                  and max(row["native_margin_replay_absolute_error"] for row in records) <= MAX_ERROR
                  and sum(row["dispatch"] != "zero" for row in targets) == 144
                  and sum(row["dispatch"] == "zero" for row in controls) == 216)
    overall_pass = (overall["cosine"] >= BARS["minimum_overall_cosine"]
                    and overall["relative_l2_error"] <= BARS["maximum_overall_relative_l2_error"]
                    and overall["sign_agreement"] >= BARS["minimum_overall_sign_agreement"]
                    and BARS["minimum_overall_norm_ratio"] <= overall["predicted_to_actual_norm_ratio"]
                    <= BARS["maximum_overall_norm_ratio"])
    recurrence = (all(value["cosine"] >= BARS["minimum_family_cosine"]
                      and value["sign_agreement"] >= BARS["minimum_family_sign_agreement"]
                      for value in by_family.values())
                  and all(value["count"] == 24 and value["positive_fraction"]
                          >= BARS["minimum_ordered_pair_positive_fraction"] for value in by_pair.values()))
    control_pass = (max(row["program_max_absolute_logit_change"] for row in controls) <= MAX_ERROR
                    and all(row["program_recipient_correct"] for row in controls))
    price_pass = compile_plan()["total_export_capability_causal_price"] == {
        "model_forwards": 5, "example_evaluations": 1584,
        "backwards": 0, "parameter_updates": 0}
    predictions = {
        "pred_a_export_capability_and_instrument": instrument,
        "pred_b_fixed_program_substitutes_exact_term": instrument and overall_pass,
        "pred_c_each_family_and_ordered_pair_recur": instrument and recurrence,
        "pred_d_control_zero_dispatch": instrument and control_pass,
        "pred_e_fixed_program_and_price": price_pass,
    }
    valid = predictions["pred_a_export_capability_and_instrument"] \
        and predictions["pred_d_control_zero_dispatch"] and predictions["pred_e_fixed_program_and_price"]
    terminal = "program_screen" if all(predictions.values()) else "null" if valid else "invalid"
    return {"overall": overall, "by_family": by_family, "by_ordered_pair": by_pair,
            "control_max_absolute_logit_change": max(row["program_max_absolute_logit_change"] for row in controls),
            "native_margin_replay_max_absolute_error": max(row["native_margin_replay_absolute_error"] for row in records),
            "predictions": predictions, "terminal": terminal}


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
        records = evaluate(model, torch, F, facade)
    scored = score(records)
    payload = managed.atomic_create_json(OUT, {
        "schema": "bracket_l13h8_ordered_pair_displacement_program_ood_validation_result_v1",
        "candidate_id": authority.CANDIDATE_ID,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "plan": plan, "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "score": scored, "evidence": records, "terminal": scored["terminal"],
    })
    print(json.dumps({"terminal": scored["terminal"],
                      "predictions": scored["predictions"],
                      "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
