#!/usr/bin/env python3
"""Four-corner MLP15+17 mediation of the fixed Task14 prototype program."""

# BQGATE: EXPERIMENT pred_a_immutable_inputs_and_complete_instrument pred_b_mlp15_17_materially_mediate_program pred_c_mediation_generalizes_across_registered_cells pred_d_direct_residual_route pred_e_fixed_program_and_price
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import circuit_fast_screen_managed_runner as managed
import run_task14_mlp6_7_direction_cardinality_prototype_causal_validation as program
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent
import run_task14_ood_fronted_mlp6_7_eauw_background_gate_factorial as factor_gate


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/task14_mlp6_7_direction_cardinality_program_mlp15_17_mediation_v2.json"
PROGRAM_RESULT = ROOT / "circuits/fast_screens/task14_mlp6_7_direction_cardinality_prototype_causal_validation_v1_result.json"
NATURAL_PATH_RESULT = ROOT / "circuits/followups/task14_head11_3_mlp15_17_vs_mlp16_factorial_v1_result.json"
OUT = ROOT / "circuits/followups/task14_mlp6_7_direction_cardinality_program_mlp15_17_mediation_v2_result.json"
PRIOR_ART_SHA256 = "8d67858b0e571a42bd76ed3b90d6d884e3fc65c29628adc96cc43232e4040332"
PROGRAM_RESULT_SHA256 = "9a488259efdb85477f35c612696368e8a3e372338a11253615a7b53650c88fe0"
NATURAL_PATH_RESULT_SHA256 = "e7765dbfd0269a32ab3e4ea8ccfeda1d4ceccabcce98e1d97d3dfd08c0ad8747"
CANDIDATE_ID = "subject_verb.number_agreement.mlp6_7_direction_cardinality_program_mlp15_17_mediation_v2"
SUBSETS = factor_gate.BACKGROUND_SUBSETS
ARMS = ("base_native", "program_native", "base_replayed", "program_clamped")
SPEC_CHUNK = 64
MAX_ERROR = 5e-5
BARS = {
    "minimum_mediated_cosine": .50,
    "minimum_mediated_norm_ratio": .15,
    "maximum_mediated_norm_ratio": 1.50,
    "minimum_mediated_sign_agreement": .65,
    "minimum_group_mediated_cosine": .25,
    "minimum_group_mediated_norm_ratio": .10,
    "minimum_cardinality_mediated_norm_ratio": .10,
    "minimum_direct_cosine": .95,
    "maximum_direct_relative_l2_error": .15,
    "minimum_direct_sign_agreement": .90,
}
PRED_KEYS = (
    "pred_a_immutable_inputs_and_complete_instrument",
    "pred_b_mlp15_17_materially_mediate_program",
    "pred_c_mediation_generalizes_across_registered_cells",
    "pred_d_direct_residual_route",
    "pred_e_fixed_program_and_price",
)


class MediationError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive_price() -> dict[str, int]:
    cells = 32 * len(SUBSETS)
    return {
        "physical_model_forwards": 1 + math.ceil(cells / SPEC_CHUNK),
        "example_evaluations": 32 * 3 + cells * len(ARMS),
        "causal_installations": cells * 3,
        "mediator_clamps": cells * 2 * 2,
        "backwards": 0,
        "parameter_updates": 0,
        "maximum_forward_batch": SPEC_CHUNK * len(ARMS),
    }


def validate_preflight() -> None:
    for path, expected, label in (
        (PRIOR_ART, PRIOR_ART_SHA256, "prior art"),
        (program.PROTOTYPES, program.PROTOTYPE_SHA256, "prototype artifact"),
        (PROGRAM_RESULT, PROGRAM_RESULT_SHA256, "program validation"),
        (NATURAL_PATH_RESULT, NATURAL_PATH_RESULT_SHA256, "natural path screen"),
    ):
        if _sha256(path) != expected:
            raise MediationError(f"{label} changed")
    parent = json.loads(PROGRAM_RESULT.read_text())
    natural = json.loads(NATURAL_PATH_RESULT.read_text())
    if parent.get("terminal") != "valid_causal_screen" or not all(parent.get("score", {}).get("predictions", {}).values()):
        raise MediationError("prototype program is not a passing immutable parent")
    if natural.get("terminal") != "mlp15_17_core_path_screen" or natural.get("predictions", {}).get("pred_b_mlp15_17_explain_combined_path") is not True:
        raise MediationError("natural MLP15+17 localization is not licensed")


def compile_plan() -> dict[str, object]:
    validate_preflight()
    return {
        "schema": "task14_mlp6_7_direction_cardinality_program_mlp15_17_mediation_plan_v2",
        "candidate_id": CANDIDATE_ID,
        "split": "PROSPECTIVE_THIRD_CORPUS_COMPLETE_LATTICE",
        "row_count": 32,
        "background_subsets": list(SUBSETS),
        "arms": list(ARMS),
        "mediators": ["mlp:15", "mlp:17"],
        "equation": "mediated_q=(program_native-base_native)-(program_clamped-base_replayed)",
        "within_batch_clamp": "For each cell, base arm MLP output supplies both clamp arms in the same forward.",
        "prior_art_sha256": PRIOR_ART_SHA256,
        "program_result_sha256": PROGRAM_RESULT_SHA256,
        "natural_path_result_sha256": NATURAL_PATH_RESULT_SHA256,
        "bars": dict(BARS),
        "price": derive_price(),
        "fit_operations": 0,
        "program_changes": 0,
    }


def _stats(actual, predicted):
    dot = sum(x * y for x, y in zip(actual, predicted))
    an = math.sqrt(sum(x * x for x in actual))
    pn = math.sqrt(sum(x * x for x in predicted))
    return {
        "count": len(actual),
        "cosine": dot / max(an * pn, 1e-30),
        "relative_l2_error": math.sqrt(sum((x - y) ** 2 for x, y in zip(actual, predicted))) / max(an, 1e-30),
        "predicted_to_actual_norm_ratio": pn / max(an, 1e-30),
        "sign_agreement": sum((x > 0) == (y > 0) for x, y in zip(actual, predicted)) / len(actual),
    }


def _install_within_batch_clamps(model, finals, torch):
    """Clamp arms 2/3 to arm 0's final-token output for every four-row cell."""
    if finals.numel() % len(ARMS):
        raise MediationError("four-corner batch is incomplete")
    base = torch.arange(0, finals.numel(), len(ARMS), device=finals.device)
    targets = torch.cat((base + 2, base + 3))
    sources = torch.cat((base, base))
    handles = []
    for layer in (15, 17):
        def clamp(_module, _arguments, output, *, src=sources, dst=targets):
            if not isinstance(output, torch.Tensor):
                raise MediationError("MLP hook output is not a tensor")
            changed = output.clone()
            changed[dst, finals[dst]] = output[src, finals[src]].to(changed.dtype)
            return changed
        handles.append(model.transformer.h[layer].mlp.register_forward_hook(clamp))
    return handles


def evaluate(model, torch, F, facade):
    artifact, _ = program._load_artifacts()
    rows = program.authority.build_rows()
    count = len(rows)
    parent = tangent.parent
    device = next(model.parameters()).device
    role_tokens, role_finals = parent.downstream.depth.parent.v1._role_batch(rows, torch, device)
    _, captured, projection, role_closure, inputs = parent._decomposed_forward(model, role_tokens, role_finals, torch, F, facade)
    roles = {"recipient": tangent._role_slice(captured, 0, count), "opposite": tangent._role_slice(captured, count, 2 * count)}
    input_roles = {"recipient": tangent._role_slice(inputs, 0, count), "opposite": tangent._role_slice(inputs, count, 2 * count)}
    function = tangent._head_function(model, roles["recipient"], roles["opposite"], model.transformer.h[parent.LAYER].attn, projection, torch, F)
    vectors = {key: torch.tensor(value["coordinates"], dtype=torch.float32, device=device) for key, value in artifact["prototypes"].items() if ".cardinality_" in key}
    cells = []
    with torch.no_grad():
        for subset in SUBSETS:
            base_heads = function(factor_gate._raw_for(input_roles["recipient"], input_roles["opposite"], subset, F)).detach()
            for index, row in enumerate(rows):
                cells.append((index, subset, base_heads[index], base_heads[index] + vectors[f'{row["direction_id"]}.cardinality_{len(subset)}']))
        margins = {}
        closures = []
        for start in range(0, len(cells), SPEC_CHUNK):
            chunk = cells[start:start + SPEC_CHUNK]
            row_indices, heads, specs = [], [], []
            for index, subset, base_head, program_head in chunk:
                for arm in ARMS:
                    row_indices.append(index)
                    heads.append(program_head if arm in {"program_native", "program_clamped"} else base_head)
                    specs.append((index, subset, arm))
            index_tensor = torch.tensor(row_indices, dtype=torch.long, device=device)
            tokens = role_tokens[:count][index_tensor]
            finals = torch.full_like(index_tensor, parent.SUBJECT_POSITION)
            replacement = torch.stack(heads)
            mask = torch.zeros(len(specs), dtype=torch.bool, device=device)
            handles = _install_within_batch_clamps(model, finals, torch)
            try:
                logits, _, _, closure = parent.downstream._decomposed_forward(
                    model, tokens, finals, torch, F, facade,
                    replacement_heads=replacement, native_reinstall_mask=mask)
            finally:
                for handle in handles:
                    handle.remove()
            closures.append(closure)
            for local, (index, subset, arm) in enumerate(specs):
                endpoint = rows[index]["endpoints"]["opposite_same_lemma"]
                margins[(index, subset, arm)] = float(logits[local, parent.SUBJECT_POSITION, endpoint["answer_id"]] - logits[local, parent.SUBJECT_POSITION, endpoint["foil_id"]])
    evidence = []
    for index, row in enumerate(rows):
        for subset in SUBSETS:
            values = {arm: margins[(index, subset, arm)] for arm in ARMS}
            full = values["program_native"] - values["base_native"]
            clamped = values["program_clamped"] - values["base_replayed"]
            evidence.append({
                "row_id": row["row_id"], "direction": row["direction_id"], "template": row["template_id"],
                "background": subset, "cardinality": len(subset), **values,
                "full_program_q": full, "clamped_program_q": clamped, "mediated_q": full - clamped,
            })
    exactness = {
        "role_state_closure_max_absolute_error": role_closure["input_state_closure_max_absolute_error"],
        "role_normalized_closure_max_absolute_error": role_closure["input_normalized_closure_max_absolute_error"],
        "downstream_state_closure_max_absolute_error": max(x["state_sum_max_absolute_error"] for x in closures),
        "downstream_normalized_closure_max_absolute_error": max(x["normalized_state_max_absolute_error"] for x in closures),
    }
    return evidence, exactness


def score(evidence, exactness):
    prior = {(x["row_id"], x["background"]): x["cardinality_prototype_q"] for x in json.loads(PROGRAM_RESULT.read_text())["causal_evidence"]}
    replay_error = max(abs(x["full_program_q"] - prior[(x["row_id"], x["background"])]) for x in evidence)
    base_replay_error = max(abs(x["base_native"] - x["base_replayed"]) for x in evidence)
    full = [x["full_program_q"] for x in evidence]
    mediated = [x["mediated_q"] for x in evidence]
    clamped = [x["clamped_program_q"] for x in evidence]
    mediation = _stats(full, mediated)
    direct = _stats(full, clamped)
    groups = {}
    for direction in ("singular_to_plural", "plural_to_singular"):
        for template in ("near_beyond", "beyond_near"):
            chosen = [x for x in evidence if x["direction"] == direction and x["template"] == template]
            groups[f"{direction}/{template}"] = _stats([x["full_program_q"] for x in chosen], [x["mediated_q"] for x in chosen])
    cardinalities = {}
    for cardinality in range(5):
        chosen = [x for x in evidence if x["cardinality"] == cardinality]
        cardinalities[str(cardinality)] = _stats([x["full_program_q"] for x in chosen], [x["mediated_q"] for x in chosen])
    instrument = len(evidence) == 512 and len({(x["row_id"], x["background"]) for x in evidence}) == 512 and replay_error <= MAX_ERROR and base_replay_error <= MAX_ERROR and all(v <= MAX_ERROR for v in exactness.values())
    pred_b = mediation["cosine"] >= BARS["minimum_mediated_cosine"] and BARS["minimum_mediated_norm_ratio"] <= mediation["predicted_to_actual_norm_ratio"] <= BARS["maximum_mediated_norm_ratio"] and mediation["sign_agreement"] >= BARS["minimum_mediated_sign_agreement"]
    pred_c = all(v["cosine"] >= BARS["minimum_group_mediated_cosine"] and v["predicted_to_actual_norm_ratio"] >= BARS["minimum_group_mediated_norm_ratio"] for v in groups.values()) and all(v["predicted_to_actual_norm_ratio"] >= BARS["minimum_cardinality_mediated_norm_ratio"] for v in cardinalities.values())
    pred_d = direct["cosine"] >= BARS["minimum_direct_cosine"] and direct["relative_l2_error"] <= BARS["maximum_direct_relative_l2_error"] and direct["sign_agreement"] >= BARS["minimum_direct_sign_agreement"]
    pred_e = derive_price()["physical_model_forwards"] <= 16 and derive_price()["example_evaluations"] <= 2560
    predictions = dict(zip(PRED_KEYS, (instrument, instrument and pred_b, instrument and pred_c, instrument and pred_d, pred_e)))
    terminal = "invalid" if not (predictions[PRED_KEYS[0]] and predictions[PRED_KEYS[4]]) else "mediation_screen" if predictions[PRED_KEYS[1]] and predictions[PRED_KEYS[2]] else "direct_route_screen" if predictions[PRED_KEYS[3]] else "inconclusive"
    return {**exactness, "prior_program_effect_replay_max_absolute_error": replay_error, "base_mediator_replay_max_absolute_error": base_replay_error, "overall_mediation": mediation, "overall_direct_route": direct, "mediation_by_direction_template": groups, "mediation_by_cardinality": cardinalities, "predictions": predictions, "terminal": terminal}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    plan = compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True))
        return
    if OUT.exists():
        raise MediationError(f"refusing overwrite {OUT}")
    torch, F, facade = tangent.parent.factors._dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    evidence, exactness = evaluate(model, torch, F, facade)
    scored = score(evidence, exactness)
    payload = managed.atomic_create_json(OUT, {
        "schema": "task14_mlp6_7_direction_cardinality_program_mlp15_17_mediation_result_v2",
        "candidate_id": CANDIDATE_ID, "terminal": scored["terminal"],
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "plan": plan, "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "score": scored, "evidence": evidence,
    })
    print(json.dumps({"terminal": scored["terminal"], "predictions": scored["predictions"], "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
