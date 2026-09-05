#!/usr/bin/env python3
"""Split fixed-program mediation into exact MLP15 and MLP17 effects."""

# BQGATE: EXPERIMENT pred_a_complete_exact_factorial pred_b_additive_distributed_pair pred_c_single_dominant_mediator pred_d_distribution_recurs_across_cells pred_e_fixed_program_and_price
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import circuit_fast_screen_managed_runner as managed
import run_task14_mlp6_7_direction_cardinality_program_mlp15_17_mediation as grouped
import run_task14_mlp6_7_direction_cardinality_prototype_causal_validation as program
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent
import run_task14_ood_fronted_mlp6_7_eauw_background_gate_factorial as factor_gate


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/task14_mlp6_7_direction_cardinality_program_mlp15_vs_mlp17_mediation_v1.json"
NATURAL_PAIR = ROOT / "circuits/followups/task14_head11_3_mlp15_mlp17_interaction_v1_result.json"
GROUPED_RESULT = grouped.OUT
OUT = ROOT / "circuits/followups/task14_mlp6_7_direction_cardinality_program_mlp15_vs_mlp17_mediation_v1_result.json"
PRIOR_ART_SHA256 = "1fd9b9816c66535668de53757217368dc38d54b1130a6a2d553471d36a19754b"
NATURAL_PAIR_SHA256 = "fdae100bd42177371b8429372f1d02ce24c1bb5eb0b29ac6a00e9511286968de"
GROUPED_RESULT_SHA256 = "430f315a2223d564f8bd3eb2f0d12de43bdea4e6ebe950c1094b66a06a245ff2"
CANDIDATE_ID = "subject_verb.number_agreement.mlp6_7_direction_cardinality_program_mlp15_vs_mlp17_mediation_v1"
SUBSETS = factor_gate.BACKGROUND_SUBSETS
ARMS = ("base_empty", "program_empty", "base_15", "program_15", "base_17", "program_17", "base_both", "program_both")
SPEC_CHUNK = 32
MAX_ERROR = 5e-5
BARS = {
    "minimum_singleton_full_norm_ratio": .05,
    "maximum_interaction_joint_norm_ratio": .25,
    "minimum_additive_reconstruction_cosine": .90,
    "maximum_additive_reconstruction_relative_l2": .25,
    "minimum_dominant_joint_norm_ratio": .80,
    "maximum_minor_joint_norm_ratio": .10,
    "minimum_dominant_joint_cosine": .90,
    "minimum_group_singleton_full_norm_ratio": .03,
    "minimum_groups_material": 3,
    "minimum_cardinalities_material": 4,
    "maximum_group_interaction_joint_norm_ratio": .35,
}
PRED_KEYS = ("pred_a_complete_exact_factorial", "pred_b_additive_distributed_pair", "pred_c_single_dominant_mediator", "pred_d_distribution_recurs_across_cells", "pred_e_fixed_program_and_price")


class SplitMediationError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive_price():
    cells = 32 * len(SUBSETS)
    return {"physical_model_forwards": 1 + math.ceil(cells / SPEC_CHUNK), "example_evaluations": 32 * 3 + cells * len(ARMS), "causal_installations": cells * 7, "mediator_clamps": cells * 2 * 4, "backwards": 0, "parameter_updates": 0, "maximum_forward_batch": SPEC_CHUNK * len(ARMS)}


def validate_preflight():
    for path, expected, label in ((PRIOR_ART, PRIOR_ART_SHA256, "prior art"), (program.PROTOTYPES, program.PROTOTYPE_SHA256, "program artifact"), (grouped.PROGRAM_RESULT, grouped.PROGRAM_RESULT_SHA256, "program validation"), (NATURAL_PAIR, NATURAL_PAIR_SHA256, "natural pair"), (GROUPED_RESULT, GROUPED_RESULT_SHA256, "grouped mediation")):
        if _sha256(path) != expected:
            raise SplitMediationError(f"{label} changed")
    if json.loads(NATURAL_PAIR.read_text()).get("terminal") != "additive_pair_screen":
        raise SplitMediationError("natural pair screen is not additive")
    if json.loads(GROUPED_RESULT.read_text()).get("terminal") != "mediation_screen":
        raise SplitMediationError("grouped program mediation is not licensed")


def compile_plan():
    validate_preflight()
    return {"schema": "task14_mlp6_7_direction_cardinality_program_mlp15_vs_mlp17_mediation_plan_v1", "candidate_id": CANDIDATE_ID, "split": "PROSPECTIVE_THIRD_CORPUS_COMPLETE_LATTICE", "row_count": 32, "background_subsets": list(SUBSETS), "arms": list(ARMS), "definitions": {"m15": "q_empty-q_15", "m17": "q_empty-q_17", "m_both": "q_empty-q_both", "interaction": "m_both-m15-m17"}, "prior_art_sha256": PRIOR_ART_SHA256, "grouped_result_sha256": GROUPED_RESULT_SHA256, "natural_pair_sha256": NATURAL_PAIR_SHA256, "bars": dict(BARS), "price": derive_price(), "fit_operations": 0, "program_changes": 0}


def _install_clamps(model, finals, torch):
    if finals.numel() % len(ARMS):
        raise SplitMediationError("eight-corner batch is incomplete")
    base = torch.arange(0, finals.numel(), len(ARMS), device=finals.device)
    target_offsets = {15: (2, 3, 6, 7), 17: (4, 5, 6, 7)}
    handles = []
    for layer, offsets in target_offsets.items():
        targets = torch.cat(tuple(base + offset for offset in offsets))
        sources = torch.cat(tuple(base for _ in offsets))
        def clamp(_module, _arguments, output, *, dst=targets, src=sources):
            if not isinstance(output, torch.Tensor):
                raise SplitMediationError("MLP hook output is not a tensor")
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
        margins, closures = {}, []
        for start in range(0, len(cells), SPEC_CHUNK):
            chunk = cells[start:start + SPEC_CHUNK]
            row_indices, heads, specs = [], [], []
            for index, subset, base_head, program_head in chunk:
                for arm in ARMS:
                    row_indices.append(index)
                    heads.append(program_head if arm.startswith("program") else base_head)
                    specs.append((index, subset, arm))
            index_tensor = torch.tensor(row_indices, dtype=torch.long, device=device)
            tokens = role_tokens[:count][index_tensor]
            finals = torch.full_like(index_tensor, parent.SUBJECT_POSITION)
            handles = _install_clamps(model, finals, torch)
            try:
                logits, _, _, closure = parent.downstream._decomposed_forward(model, tokens, finals, torch, F, facade, replacement_heads=torch.stack(heads), native_reinstall_mask=torch.zeros(len(specs), dtype=torch.bool, device=device))
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
            q = {name: values[f"program_{name}"] - values[f"base_{name}"] for name in ("empty", "15", "17", "both")}
            m15, m17, mboth = q["empty"] - q["15"], q["empty"] - q["17"], q["empty"] - q["both"]
            evidence.append({"row_id": row["row_id"], "direction": row["direction_id"], "template": row["template_id"], "background": subset, "cardinality": len(subset), **values, **{f"q_{k}": v for k, v in q.items()}, "m15": m15, "m17": m17, "m_both": mboth, "interaction": mboth - m15 - m17})
    exactness = {"role_state_closure_max_absolute_error": role_closure["input_state_closure_max_absolute_error"], "role_normalized_closure_max_absolute_error": role_closure["input_normalized_closure_max_absolute_error"], "downstream_state_closure_max_absolute_error": max(x["state_sum_max_absolute_error"] for x in closures), "downstream_normalized_closure_max_absolute_error": max(x["normalized_state_max_absolute_error"] for x in closures)}
    return evidence, exactness


def _norm(values):
    return math.sqrt(sum(x * x for x in values))


def _summarize(items):
    full = [x["q_empty"] for x in items]
    m15 = [x["m15"] for x in items]
    m17 = [x["m17"] for x in items]
    both = [x["m_both"] for x in items]
    interaction = [x["interaction"] for x in items]
    reconstruction = [a + b for a, b in zip(m15, m17)]
    full_norm, both_norm = max(_norm(full), 1e-30), max(_norm(both), 1e-30)
    return {"m15_to_full_norm_ratio": _norm(m15) / full_norm, "m17_to_full_norm_ratio": _norm(m17) / full_norm, "m15_to_joint_norm_ratio": _norm(m15) / both_norm, "m17_to_joint_norm_ratio": _norm(m17) / both_norm, "interaction_to_joint_norm_ratio": _norm(interaction) / both_norm, "m15_to_joint": grouped._stats(both, m15), "m17_to_joint": grouped._stats(both, m17), "additive_reconstruction": grouped._stats(both, reconstruction)}


def score(evidence, exactness):
    parent = {(x["row_id"], x["background"]): x for x in json.loads(GROUPED_RESULT.read_text())["evidence"]}
    empty_replay = max(abs(x["q_empty"] - parent[(x["row_id"], x["background"])]["full_program_q"]) for x in evidence)
    both_replay = max(abs(x["q_both"] - parent[(x["row_id"], x["background"])]["clamped_program_q"]) for x in evidence)
    base_replay = max(abs(x["base_empty"] - x[f"base_{site}"]) for x in evidence for site in ("15", "17", "both"))
    overall = _summarize(evidence)
    groups = {f"{d}/{t}": _summarize([x for x in evidence if x["direction"] == d and x["template"] == t]) for d in ("singular_to_plural", "plural_to_singular") for t in ("near_beyond", "beyond_near")}
    cards = {str(k): _summarize([x for x in evidence if x["cardinality"] == k]) for k in range(5)}
    instrument = len(evidence) == 512 and len({(x["row_id"], x["background"]) for x in evidence}) == 512 and max(empty_replay, both_replay, base_replay, *exactness.values()) <= MAX_ERROR
    pred_b = overall["m15_to_full_norm_ratio"] >= BARS["minimum_singleton_full_norm_ratio"] and overall["m17_to_full_norm_ratio"] >= BARS["minimum_singleton_full_norm_ratio"] and overall["interaction_to_joint_norm_ratio"] <= BARS["maximum_interaction_joint_norm_ratio"] and overall["additive_reconstruction"]["cosine"] >= BARS["minimum_additive_reconstruction_cosine"] and overall["additive_reconstruction"]["relative_l2_error"] <= BARS["maximum_additive_reconstruction_relative_l2"]
    pred_c = any(v[f"m{dominant}_to_joint_norm_ratio"] >= BARS["minimum_dominant_joint_norm_ratio"] and v[f"m{minor}_to_joint_norm_ratio"] <= BARS["maximum_minor_joint_norm_ratio"] and v[f"m{dominant}_to_joint"]["cosine"] >= BARS["minimum_dominant_joint_cosine"] for dominant, minor, v in (("15", "17", overall), ("17", "15", overall)))
    group_material = {site: sum(v[f"m{site}_to_full_norm_ratio"] >= BARS["minimum_group_singleton_full_norm_ratio"] for v in groups.values()) for site in ("15", "17")}
    card_material = {site: sum(v[f"m{site}_to_full_norm_ratio"] >= BARS["minimum_group_singleton_full_norm_ratio"] for v in cards.values()) for site in ("15", "17")}
    pred_d = all(n >= BARS["minimum_groups_material"] for n in group_material.values()) and all(n >= BARS["minimum_cardinalities_material"] for n in card_material.values()) and all(v["interaction_to_joint_norm_ratio"] <= BARS["maximum_group_interaction_joint_norm_ratio"] for v in groups.values())
    price = derive_price()
    pred_e = price["physical_model_forwards"] <= 17 and price["example_evaluations"] <= 4192
    predictions = dict(zip(PRED_KEYS, (instrument, instrument and pred_b, instrument and pred_c, instrument and pred_d, pred_e)))
    terminal = "invalid" if not (predictions[PRED_KEYS[0]] and predictions[PRED_KEYS[4]]) else "additive_distributed_screen" if predictions[PRED_KEYS[1]] and predictions[PRED_KEYS[3]] else "single_dominant_screen" if predictions[PRED_KEYS[2]] else "inconclusive"
    return {**exactness, "empty_effect_replay_max_absolute_error": empty_replay, "both_clamp_effect_replay_max_absolute_error": both_replay, "base_replay_max_absolute_error": base_replay, "overall": overall, "by_direction_template": groups, "by_cardinality": cards, "material_group_counts": group_material, "material_cardinality_counts": card_material, "predictions": predictions, "terminal": terminal}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    plan = compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True)); return
    if OUT.exists():
        raise SplitMediationError(f"refusing overwrite {OUT}")
    torch, F, facade = tangent.parent.factors._dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    evidence, exactness = evaluate(model, torch, F, facade)
    scored = score(evidence, exactness)
    payload = managed.atomic_create_json(OUT, {"schema": "task14_mlp6_7_direction_cardinality_program_mlp15_vs_mlp17_mediation_result_v1", "candidate_id": CANDIDATE_ID, "terminal": scored["terminal"], "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "plan": plan, "checkpoint_weights_sha256": checkpoint.weights_sha256, "score": scored, "evidence": evidence})
    print(json.dumps({"terminal": scored["terminal"], "predictions": scored["predictions"], "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
