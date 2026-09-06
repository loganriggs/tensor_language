#!/usr/bin/env python3
"""Calibrate once, then prospectively test the sole licensed L13H8 direct readout."""

# BQGATE: EXPERIMENT pred_a_temporal_seal_exact_instrument_and_capability pred_b_fixed_direct_readout_transfers pred_c_native_baseline_precision pred_d_absolute_counterfactual_precision pred_e_price_and_scope
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import circuit_fast_screen_managed_runner as managed
import run_bracket_l13h8_source_region_payload_factorial as exact
from run_task14_bracket_native_baseline_semantic_linear_feasibility_v1 import bracket_features

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/bracket_l13h8_direct_readout_baseline_program_v1.json"
CALIBRATION_ROWS = ROOT / "circuits/prior_art/bracket_native_baseline_fresh_corpus_v1_rows.json"
FRESH_ROWS = ROOT / "circuits/prior_art/bracket_l13h8_direct_readout_fresh_corpus_v1_rows.json"
CEILING = ROOT / "circuits/followups/bracket_native_baseline_l13h8_causal_ceiling_newest_v1_result.json"
EFFECTS = ROOT / "circuits/followups/task14_bracket_counterfactual_margin_actuator_v4_artifact.json"
VECTORS = ROOT / "circuits/fast_screens/bracket_l13h8_ordered_pair_displacement_artifact_v1.json"
FACTOR_RUNNER = ROOT / "ops/run_bracket_l13h8_source_region_payload_factorial.py"
FEATURE_RUNNER = ROOT / "ops/run_task14_bracket_native_baseline_semantic_linear_feasibility_v1.py"
ARTIFACT = ROOT / "circuits/followups/bracket_l13h8_direct_readout_baseline_program_v1_artifact.json"
OUT = ROOT / "circuits/followups/bracket_l13h8_direct_readout_baseline_program_v1_result.json"
CANDIDATE_ID = "bracket.pending_opener.l13h8_direct_readout_baseline_program_v1"
EXPECTED = {
    CEILING: "670ec697b68c0e74f7ee8b11d33dace33c16a726acd7272738d9ef876f214cf1",
    CALIBRATION_ROWS: "ad246a0ab2affd0a351b971c100c27c2ad09597d0d9e7b84b636e1eb4c8fb399",
    FRESH_ROWS: "09424b15ad797491b4968bdb9e84b3f81f7062a6b42b057e85c725b25c1b4f8c",
    EFFECTS: "85c5cc0549421fc1575d96ce621d0677ea4b0cc2d154b2c0bf7af90f4148bd4c",
    VECTORS: "531434535daf526ebf584564afbfd5834c75df7bd636014ea233f9ae71206db0",
    FACTOR_RUNNER: "b4897139e6bc8451909c09c038bf46dae80ccdca5931e62fd8b7d7ed66c7f53e",
    FEATURE_RUNNER: "cf7eebb392e9207fea0421172824b9e0234417a025e696db86602cd1e89d2620",
}
V2_BASELINE_RELATIVE_L2 = 0.2879385749265087


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load() -> tuple[dict, list[dict], list[dict], dict, dict]:
    prior = json.loads(PRIOR.read_text())
    for path, expected in EXPECTED.items():
        if sha(path) != expected:
            raise ValueError(f"immutable authority changed: {path}")
    if prior["authority"] != {path.name: digest for path, digest in EXPECTED.items()}:
        raise ValueError("prior authority mismatch")
    calibration = json.loads(CALIBRATION_ROWS.read_text())
    fresh = json.loads(FRESH_ROWS.read_text())
    if fresh["status"] != "rows_frozen_outcomes_unopened" or fresh["outcomes_opened"] or len(fresh["rows"]) != 36:
        raise ValueError("prospective row seal invalid")
    ceiling = json.loads(CEILING.read_text())
    if ceiling["terminal"] != "screen" or ceiling["score"]["licensed_next_action"] != "one_fixed_direct_readout_compression":
        raise ValueError("direct-readout action was not licensed")
    return prior, calibration["rows"], fresh["rows"], json.loads(EFFECTS.read_text()), json.loads(VECTORS.read_text())


def compile_plan() -> dict:
    prior, calibration, fresh, _, _ = load()
    return {
        "schema": "bracket_l13h8_direct_readout_baseline_program_plan_v1",
        "candidate_id": CANDIDATE_ID, "prior_art_sha256": sha(PRIOR),
        "calibration_rows": len(calibration), "prospective_rows": len(fresh),
        "calibration_endpoints": 2 * len(calibration), "prospective_endpoints": 2 * len(fresh),
        "features": ["intercept", "recipient_is_1", "recipient_is_8", "donor_is_1", "donor_is_8", "checkpoint_fixed_direct_opener_readout"],
        "bars": prior["bars"], "price": prior["price"],
    }


def prepare(rows: list[dict], torch, device):
    endpoints = [(row, side) for row in rows for side in ("base", "donor")]
    length = max(len(row[f"{side}_ids"]) for row, side in endpoints)
    tokens = torch.full((len(endpoints), length), 50256, dtype=torch.long, device=device)
    finals, sources = [], []
    for index, (row, side) in enumerate(endpoints):
        ids = row[f"{side}_ids"]
        other = row["donor_ids" if side == "base" else "base_ids"]
        differences = [i for i, (a, b) in enumerate(zip(ids, other)) if a != b]
        if len(ids) != len(other) or len(differences) != 1:
            raise ValueError("unaligned opener substitution")
        tokens[index, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        finals.append(len(ids) - 1); sources.append(differences[0])
    return endpoints, tokens, torch.tensor(finals, device=device), torch.tensor(sources, device=device)


def direct_features(model, terms, endpoints, torch, F):
    carried = terms.float()
    for layer in range(14, len(model.transformer.h)):
        carried = model.transformer.h[layer].lambdas[0].float() * carried
    raw = F.linear(carried, model.lm_head.weight.float(), None)
    values = []
    for index, (row, side) in enumerate(endpoints):
        other = "donor" if side == "base" else "base"
        values.append(float(raw[index, row[f"{other}_answer_id"]] - raw[index, row[f"{side}_answer_id"]]))
    return values


def collect(model, rows, vectors, torch, F, facade, *, interventions: bool, program: bool):
    endpoints, tokens, finals, sources = prepare(rows, torch, next(model.parameters()).device)
    replay, factors = exact.factor_forward(model, tokens, finals, {}, torch, F, facade)
    arange = torch.arange(len(endpoints), device=tokens.device)
    terms = factors["p"][arange, sources].unsqueeze(-1) * factors["u"][arange, sources]
    direct = direct_features(model, terms, endpoints, torch, F)
    zero_logits = None; program_logits = None
    if interventions:
        zero_logits = exact.factor_forward(model, tokens, finals, {}, torch, F, facade, replacement_terms=torch.zeros_like(terms), source_positions=sources)[0]
        if program:
            installs = []
            for index, (row, side) in enumerate(endpoints):
                other = "donor" if side == "base" else "base"
                pair = f'{row[f"{side}_answer_id"]}->{row[f"{other}_answer_id"]}'
                installs.append(terms[index] + torch.tensor(vectors["prototypes"][pair]["coordinates"], dtype=torch.float32, device=tokens.device))
            program_logits = exact.factor_forward(model, tokens, finals, {}, torch, F, facade, replacement_terms=torch.stack(installs), source_positions=sources)[0]
    records = []
    for index, (row, side) in enumerate(endpoints):
        other = "donor" if side == "base" else "base"
        recipient, donor = row[f"{side}_answer_id"], row[f"{other}_answer_id"]
        q = int(finals[index]); pair = f"{recipient}->{donor}"
        native = float(replay[index, q, donor] - replay[index, q, recipient])
        item = {"row_id": row["row_id"], "side": side, "ordered_pair": pair, "recipient_closer_id": recipient, "donor_closer_id": donor, "direct_readout": direct[index], "native_donorward_margin": native, "native_recipient_correct": native < 0}
        if zero_logits is not None:
            zeroed = float(zero_logits[index, q, donor] - zero_logits[index, q, recipient])
            item.update({"semantic_zero_donorward_margin": zeroed, "semantic_term_donorward_contribution": native - zeroed})
            if program_logits is not None:
                edited = float(program_logits[index, q, donor] - program_logits[index, q, recipient])
                item.update({"actual_counterfactual_margin": edited, "actual_program_effect": edited - native})
        records.append(item)
    return records


def fit(calibration: list[dict]) -> dict:
    design = np.asarray([bracket_features(row) for row in calibration], dtype=np.float64)
    residual = np.asarray([row["semantic_zero_donorward_margin"] for row in calibration], dtype=np.float64)
    direct = np.asarray([row["direct_readout"] for row in calibration], dtype=np.float64)
    damage = np.asarray([row["semantic_term_donorward_contribution"] for row in calibration], dtype=np.float64)
    residual_coefficients = np.linalg.lstsq(design, residual, rcond=None)[0]
    causal_gain = float(np.dot(direct, damage) / np.dot(direct, direct))
    return {"semantic_zero_residual_coefficients": residual_coefficients.tolist(), "direct_readout_causal_gain": causal_gain}


def predict(records: list[dict], coefficients: dict, effects: dict) -> None:
    for row in records:
        residual = sum(a * b for a, b in zip(bracket_features(row), coefficients["semantic_zero_residual_coefficients"]))
        damage = coefficients["direct_readout_causal_gain"] * row["direct_readout"]
        row["predicted_semantic_zero_donorward_margin"] = residual
        row["predicted_semantic_term_donorward_contribution"] = damage
        row["predicted_native_donorward_margin"] = residual + damage
        row["predicted_program_effect"] = effects["effects"]["bracket"][row["ordered_pair"]]
        row["predicted_counterfactual_margin"] = row["predicted_native_donorward_margin"] + row["predicted_program_effect"]


def metrics(rows: list[dict], actual_key: str, predicted_key: str) -> dict:
    actual = [row[actual_key] for row in rows]; predicted = [row[predicted_key] for row in rows]
    an = math.sqrt(sum(value * value for value in actual)); pn = math.sqrt(sum(value * value for value in predicted))
    return {"count": len(rows), "cosine": sum(a * p for a, p in zip(actual, predicted)) / (an * pn), "relative_l2_error": math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted))) / an, "predicted_to_actual_norm_ratio": pn / an, "sign_agreement": sum((a > 0) == (p > 0) for a, p in zip(actual, predicted)) / len(rows)}


def score(records, calibration, bars, price):
    direct = metrics(records, "semantic_term_donorward_contribution", "predicted_semantic_term_donorward_contribution")
    baseline = metrics(records, "native_donorward_margin", "predicted_native_donorward_margin")
    counter = metrics(records, "actual_counterfactual_margin", "predicted_counterfactual_margin")
    pairs = {pair: {"baseline": metrics([row for row in records if row["ordered_pair"] == pair], "native_donorward_margin", "predicted_native_donorward_margin"), "counterfactual": metrics([row for row in records if row["ordered_pair"] == pair], "actual_counterfactual_margin", "predicted_counterfactual_margin")} for pair in sorted({row["ordered_pair"] for row in records})}
    instrument = len(records) == 72 and len(calibration) == 72 and len(pairs) == 6 and min(sum(row["native_recipient_correct"] for row in rows) / len(rows) for rows in (calibration, records)) >= bars["minimum_native_capability"]
    direct_ok = direct["cosine"] >= bars["minimum_prospective_direct_damage_cosine"] and direct["relative_l2_error"] <= bars["maximum_prospective_direct_damage_relative_l2"] and direct["sign_agreement"] >= bars["minimum_prospective_direct_damage_sign_agreement"]
    baseline_ok = baseline["cosine"] >= bars["minimum_prospective_baseline_cosine"] and baseline["relative_l2_error"] <= bars["maximum_prospective_baseline_relative_l2"] and baseline["sign_agreement"] >= bars["minimum_prospective_baseline_sign_agreement"] and baseline["relative_l2_error"] <= bars["maximum_baseline_relative_l2_fraction_of_semantic_linear_v2"] * V2_BASELINE_RELATIVE_L2 and all(value["baseline"]["cosine"] >= bars["minimum_each_pair_baseline_cosine"] and value["baseline"]["relative_l2_error"] <= bars["maximum_each_pair_baseline_relative_l2"] for value in pairs.values())
    counter_ok = counter["cosine"] >= bars["minimum_prospective_counterfactual_cosine"] and counter["relative_l2_error"] <= bars["maximum_prospective_counterfactual_relative_l2"] and counter["sign_agreement"] >= bars["minimum_prospective_counterfactual_sign_agreement"] and all(value["counterfactual"]["sign_agreement"] >= bars["minimum_each_pair_counterfactual_sign_agreement"] for value in pairs.values())
    price_ok = price == {"physical_model_forwards": 5, "endpoint_evaluations": 360, "causal_installations": 216, "fits": 2, "fitted_fp32_scalars": 6, "backwards": 0, "parameter_updates": 0}
    predictions = {"pred_a_temporal_seal_exact_instrument_and_capability": instrument, "pred_b_fixed_direct_readout_transfers": direct_ok, "pred_c_native_baseline_precision": baseline_ok, "pred_d_absolute_counterfactual_precision": counter_ok, "pred_e_price_and_scope": price_ok}
    terminal = "program_screen" if all(predictions.values()) else "null" if instrument and price_ok else "invalid"
    return {"prospective_direct_damage": direct, "prospective_native_baseline": baseline, "prospective_absolute_counterfactual": counter, "by_ordered_pair": pairs, "semantic_linear_v2_baseline_relative_l2": V2_BASELINE_RELATIVE_L2, "baseline_relative_l2_fraction_of_v2": baseline["relative_l2_error"] / V2_BASELINE_RELATIVE_L2, "predictions": predictions, "terminal": terminal, "dependency_boundary": {"removed": ["native bracket output margin", "layers 14-17 execution"], "retained": ["upstream execution through L13H8 semantic-opener term", "semantic opener position", "checkpoint-fixed residual lambdas and unembedding", "closer ids", "edit specification"], "classification": "circuit_conditioned_baseline_and_counterfactual_margin_program_not_standalone_not_whole_model"}}


def main() -> None:
    plan = compile_plan()
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1" or "--dry-run" in sys.argv:
        print(json.dumps(plan, sort_keys=True)); return
    if ARTIFACT.exists() or OUT.exists():
        raise ValueError("refusing overwrite")
    prior, calibration_rows, fresh_rows, effects, vectors = load()
    torch, F, facade = exact._dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    with torch.no_grad():
        calibration = collect(model, calibration_rows, vectors, torch, F, facade, interventions=True, program=False)
        coefficients = fit(calibration)
        artifact_value = {"schema": "bracket_l13h8_direct_readout_baseline_program_artifact_v1", "candidate_id": CANDIDATE_ID, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "checkpoint_weights_sha256": checkpoint.weights_sha256, "features": plan["features"], "coefficients": coefficients, "stored_fp32_scalars": 6, "equations": prior["program"], "terminal": "frozen_before_prospective_forward"}
        artifact_bytes = managed.atomic_create_json(ARTIFACT, artifact_value)
        prospective = collect(model, fresh_rows, vectors, torch, F, facade, interventions=True, program=True)
    predict(calibration, coefficients, effects); predict(prospective, coefficients, effects)
    scored = score(prospective, calibration, prior["bars"], prior["price"])
    payload = managed.atomic_create_json(OUT, {"schema": "bracket_l13h8_direct_readout_baseline_program_result_v1", "candidate_id": CANDIDATE_ID, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "plan": plan, "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(), "checkpoint_weights_sha256": checkpoint.weights_sha256, "score": scored, "calibration_evidence": calibration, "prospective_evidence": prospective, "terminal": scored["terminal"]})
    print(json.dumps({"terminal": scored["terminal"], "predictions": scored["predictions"], "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(), "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
