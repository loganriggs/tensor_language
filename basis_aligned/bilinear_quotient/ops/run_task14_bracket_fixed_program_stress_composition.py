#!/usr/bin/env python3
"""Exact four-corner stress composition of the frozen Task14 and bracket programs."""

# BQGATE: EXPERIMENT pred_a_immutable_programs_and_exact_corners pred_b_foreign_stress_is_live pred_c_task14_survives_bracket_program pred_d_bracket_survives_task14_program pred_e_fixed_price_and_no_refit
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import circuit_fast_screen_candidate_bracket_l13h8_ordered_pair_displacement_program as bracket_authority
import circuit_fast_screen_managed_runner as managed
import run_bracket_l13h8_ordered_pair_displacement_program_ood_validation as bracket_program
import run_bracket_l13h8_source_region_payload_factorial as bracket_exact
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent
import run_task14_mlp6_7_direction_cardinality_prototype_causal_validation as task14_program
import run_task14_ood_fronted_mlp6_7_eauw_background_gate_factorial as factor_gate


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/task14_bracket_fixed_program_stress_composition_v1.json"
TASK14_ARTIFACT = task14_program.PROTOTYPES
TASK14_RESULT = ROOT / "circuits/fast_screens/task14_mlp6_7_direction_cardinality_prototype_causal_validation_v1_result.json"
BRACKET_ARTIFACT = bracket_program.ARTIFACT
BRACKET_RESULT = ROOT / "circuits/followups/bracket_l13h8_ordered_pair_displacement_program_ood_validation_v1_result.json"
OUT = ROOT / "circuits/followups/task14_bracket_fixed_program_stress_composition_v1_result.json"
CANDIDATE_ID = "cross_behavior.task14_bracket_fixed_program_stress_composition_v1"
EXPECTED_SHA256 = {
    PRIOR: "882940031d66c68d98746ef92342be52a1ba9e6ba58e54265255259870f13bcf",
    TASK14_ARTIFACT: "cce00d8f2309b0f9e0329c094238505819f2cf8a21c04e276af2085e068c1d07",
    TASK14_RESULT: "9a488259efdb85477f35c612696368e8a3e372338a11253615a7b53650c88fe0",
    BRACKET_ARTIFACT: "531434535daf526ebf584564afbfd5834c75df7bd636014ea233f9ae71206db0",
    BRACKET_RESULT: "3b267f069647824fb7557e9784c63becb0366f94fe4d274fea343ae2bc802e5f",
}
ARMS = ("base", "own", "stress", "both")
TASK14_CHUNK_CELLS = 64
MAX_REPLAY_ERROR = 1e-4
BARS = {"minimum_stress_to_own_norm_ratio": .05,
        "minimum_preservation_cosine": .90,
        "maximum_preservation_relative_l2_error": .40,
        "minimum_preservation_sign_agreement": .90,
        "maximum_interaction_to_own_norm_ratio": .40}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_artifacts():
    for path, expected in EXPECTED_SHA256.items():
        if _sha(path) != expected:
            raise ValueError(f"immutable composition input changed: {path}")
    ta, tr = json.loads(TASK14_ARTIFACT.read_text()), json.loads(TASK14_RESULT.read_text())
    ba, br = json.loads(BRACKET_ARTIFACT.read_text()), json.loads(BRACKET_RESULT.read_text())
    if ta.get("terminal") != "prototype_artifact" or tr.get("terminal") != "valid_causal_screen":
        raise ValueError("Task14 parent is not licensed")
    if ba.get("terminal") != "prototype_artifact" or br.get("terminal") != "program_screen":
        raise ValueError("bracket parent is not licensed")
    return ta, tr, ba, br


def _choice(key: str, choices):
    digest = hashlib.sha256(key.encode()).digest()
    return choices[int.from_bytes(digest[:8], "big") % len(choices)]


def derive_price():
    task14_cells = 32 * len(factor_gate.BACKGROUND_SUBSETS)
    bracket_endpoints = 2 * sum(row["program_role"] == "target"
                                for row in bracket_authority.build_ood_rows())
    return {"physical_model_forwards": 1 + math.ceil(task14_cells / TASK14_CHUNK_CELLS) + 1,
            "example_evaluations": 32 * 3 + task14_cells * len(ARMS)
                                   + bracket_endpoints * len(ARMS),
            "task14_cells": task14_cells, "bracket_endpoints": bracket_endpoints,
            "backwards": 0, "fits": 0, "parameter_updates": 0, "vector_changes": 0}


def compile_plan():
    _load_artifacts()
    price = derive_price()
    if price != {"physical_model_forwards": 10, "example_evaluations": 2720,
                 "task14_cells": 512, "bracket_endpoints": 144,
                 "backwards": 0, "fits": 0, "parameter_updates": 0, "vector_changes": 0}:
        raise ValueError("composition price changed")
    return {"schema": "task14_bracket_fixed_program_stress_composition_plan_v1",
            "candidate_id": CANDIDATE_ID,
            "prior_art_sha256": EXPECTED_SHA256[PRIOR],
            "input_sha256": {str(path.relative_to(ROOT)): digest
                              for path, digest in EXPECTED_SHA256.items()},
            "arms": list(ARMS),
            "assignment": "SHA256(row_id + panel cell key), first 8 bytes big-endian modulo sorted foreign keys",
            "interaction": "both-own-stress+base",
            "scope": "active cross-program stress composition, not joint semantic instantiation",
            "bars": dict(BARS), "price": price}


def _attention_add_hook(finals, vectors, mask, torch):
    def hook(_module, _arguments, output):
        write, first_value = output
        changed = write.clone()
        rows = torch.nonzero(mask, as_tuple=False).flatten()
        if rows.numel():
            changed[rows, finals[rows]] += vectors[rows].to(changed.dtype)
        return changed, first_value
    return hook


def _task14_panel(model, torch, F, facade, task14_artifact, bracket_artifact):
    rows = task14_program.authority.build_rows(); count = len(rows); parent = tangent.parent
    device = next(model.parameters()).device
    role_tokens, role_finals = parent.downstream.depth.parent.v1._role_batch(rows, torch, device)
    _, captured, projection, role_closure, inputs = parent._decomposed_forward(
        model, role_tokens, role_finals, torch, F, facade)
    roles = {"recipient": tangent._role_slice(captured, 0, count),
             "opposite": tangent._role_slice(captured, count, 2 * count)}
    input_roles = {"recipient": tangent._role_slice(inputs, 0, count),
                   "opposite": tangent._role_slice(inputs, count, 2 * count)}
    function = tangent._head_function(model, roles["recipient"], roles["opposite"],
                                      model.transformer.h[parent.LAYER].attn, projection, torch, F)
    task_vectors = {key: torch.tensor(value["coordinates"], dtype=torch.float32, device=device)
                    for key, value in task14_artifact["prototypes"].items() if ".cardinality_" in key}
    bracket_vectors = {key: torch.tensor(value["coordinates"], dtype=torch.float32, device=device)
                       for key, value in bracket_artifact["prototypes"].items()}
    bracket_keys = sorted(bracket_vectors)
    cells = []
    for subset in factor_gate.BACKGROUND_SUBSETS:
        base_heads = function(factor_gate._raw_for(input_roles["recipient"],
                                                   input_roles["opposite"], subset, F)).detach()
        for index, row in enumerate(rows):
            own = task_vectors[f'{row["direction_id"]}.cardinality_{len(subset)}']
            stress_key = _choice(f'task14|{row["row_id"]}|{subset}', bracket_keys)
            cells.append((index, subset, base_heads[index], base_heads[index] + own,
                          stress_key, bracket_vectors[stress_key]))
    margins = {}; closures = []
    for start in range(0, len(cells), TASK14_CHUNK_CELLS):
        chunk = cells[start:start + TASK14_CHUNK_CELLS]
        row_indices, heads, stress_vectors, stress_mask, specs = [], [], [], [], []
        for index, subset, base_head, own_head, stress_key, stress_vector in chunk:
            for arm in ARMS:
                row_indices.append(index); heads.append(own_head if arm in {"own", "both"} else base_head)
                stress_vectors.append(stress_vector); stress_mask.append(arm in {"stress", "both"})
                specs.append((index, subset, arm, stress_key))
        index_tensor = torch.tensor(row_indices, dtype=torch.long, device=device)
        tokens = role_tokens[:count][index_tensor]
        finals = torch.full_like(index_tensor, parent.SUBJECT_POSITION)
        stress_tensor = torch.stack(stress_vectors)
        mask_tensor = torch.tensor(stress_mask, dtype=torch.bool, device=device)
        handle = model.transformer.h[13].attn.register_forward_hook(
            _attention_add_hook(finals, stress_tensor, mask_tensor, torch))
        try:
            logits, _, _, closure = parent.downstream._decomposed_forward(
                model, tokens, finals, torch, F, facade,
                replacement_heads=torch.stack(heads),
                native_reinstall_mask=torch.zeros(len(specs), dtype=torch.bool, device=device))
        finally:
            handle.remove()
        closures.append(closure)
        for local, (index, subset, arm, stress_key) in enumerate(specs):
            endpoint = rows[index]["endpoints"]["opposite_same_lemma"]
            margins[(index, subset, arm)] = float(
                logits[local, parent.SUBJECT_POSITION, endpoint["answer_id"]]
                - logits[local, parent.SUBJECT_POSITION, endpoint["foil_id"]])
    evidence = []
    for index, row in enumerate(rows):
        for subset in factor_gate.BACKGROUND_SUBSETS:
            value = {arm: margins[(index, subset, arm)] for arm in ARMS}
            evidence.append({"row_id": row["row_id"], "background": subset,
                             "direction": row["direction_id"], "template": row["template_id"],
                             **value, "isolated_own": value["own"] - value["base"],
                             "own_under_stress": value["both"] - value["stress"],
                             "foreign_stress": value["stress"] - value["base"],
                             "interaction": value["both"] - value["own"] - value["stress"] + value["base"]})
    exactness = {"role_state": role_closure["input_state_closure_max_absolute_error"],
                 "role_normalized": role_closure["input_normalized_closure_max_absolute_error"],
                 "suffix_state": max(x["state_sum_max_absolute_error"] for x in closures),
                 "suffix_normalized": max(x["normalized_state_max_absolute_error"] for x in closures)}
    return evidence, exactness


def _bracket_panel(model, torch, F, facade, task14_artifact, bracket_artifact):
    rows = [row for row in bracket_authority.build_ood_rows() if row["program_role"] == "target"]
    endpoints, tokens0, finals0 = bracket_program.capability._pad(
        rows, torch, next(model.parameters()).device)
    device = tokens0.device
    task_vectors = {key: torch.tensor(value["coordinates"], dtype=torch.float32, device=device)
                    for key, value in task14_artifact["prototypes"].items() if ".cardinality_" in key}
    bracket_vectors = {key: torch.tensor(value["coordinates"], dtype=torch.float32, device=device)
                       for key, value in bracket_artifact["prototypes"].items()}
    task_keys = sorted(task_vectors)
    indices, own_vectors, stress_vectors, own_mask, stress_mask, specs = [], [], [], [], [], []
    for index, (row, side) in enumerate(endpoints):
        other = "donor" if side == "base" else "base"
        pair = f'{row[f"{side}_answer_id"]}->{row[f"{other}_answer_id"]}'
        stress_key = _choice(f'bracket|{row["row_id"]}|{side}', task_keys)
        for arm in ARMS:
            indices.append(index); own_vectors.append(bracket_vectors[pair]); stress_vectors.append(task_vectors[stress_key])
            own_mask.append(arm in {"own", "both"}); stress_mask.append(arm in {"stress", "both"})
            specs.append((index, arm, pair, stress_key))
    index_tensor = torch.tensor(indices, dtype=torch.long, device=device)
    tokens, finals = tokens0[index_tensor], finals0[index_tensor]
    own_tensor, stress_tensor = torch.stack(own_vectors), torch.stack(stress_vectors)
    own_mask_tensor = torch.tensor(own_mask, dtype=torch.bool, device=device)
    stress_mask_tensor = torch.tensor(stress_mask, dtype=torch.bool, device=device)
    def attention(event):
        write, first_value = event.block.attn(event.state, event.first_value)
        if event.site == 11:
            rows_live = torch.nonzero(stress_mask_tensor, as_tuple=False).flatten(); write = write.clone()
            write[rows_live, finals[rows_live]] += stress_tensor[rows_live].to(write.dtype)
        elif event.site == 13:
            rows_live = torch.nonzero(own_mask_tensor, as_tuple=False).flatten(); write = write.clone()
            write[rows_live, finals[rows_live]] += own_tensor[rows_live].to(write.dtype)
        return write, first_value
    logits = facade.forward_with_dispatch(model, tokens, attention,
                                          lambda event: event.block.mlp(event.state),
                                          require_production=False).float()
    margins = {}
    for local, (index, arm, _pair, _stress_key) in enumerate(specs):
        row, side = endpoints[index]; other = "donor" if side == "base" else "base"
        recipient, donor = row[f"{side}_answer_id"], row[f"{other}_answer_id"]
        q = int(finals[local]); margins[(index, arm)] = float(
            (logits[local, q, donor] - logits[local, q, recipient]))
    evidence = []
    for index, (row, side) in enumerate(endpoints):
        value = {arm: margins[(index, arm)] for arm in ARMS}
        evidence.append({"row_id": row["row_id"], "side": side, "family_id": row["family_id"],
                         **value, "isolated_own": value["own"] - value["base"],
                         "own_under_stress": value["both"] - value["stress"],
                         "foreign_stress": value["stress"] - value["base"],
                         "interaction": value["both"] - value["own"] - value["stress"] + value["base"]})
    return evidence


def _stats(actual, predicted):
    dot = sum(a * p for a, p in zip(actual, predicted)); an = math.sqrt(sum(a*a for a in actual)); pn = math.sqrt(sum(p*p for p in predicted))
    return {"count": len(actual), "cosine": dot / max(an*pn, 1e-30),
            "relative_l2_error": math.sqrt(sum((a-p)**2 for a,p in zip(actual,predicted))) / max(an,1e-30),
            "sign_agreement": sum((a>0)==(p>0) for a,p in zip(actual,predicted))/len(actual),
            "predicted_to_actual_norm_ratio": pn/max(an,1e-30)}


def _panel_score(rows):
    own = [x["isolated_own"] for x in rows]; under = [x["own_under_stress"] for x in rows]
    stress = [x["foreign_stress"] for x in rows]; interaction = [x["interaction"] for x in rows]
    own_norm = math.sqrt(sum(x*x for x in own))
    return {"preservation": _stats(own, under),
            "foreign_stress_to_own_norm_ratio": math.sqrt(sum(x*x for x in stress))/max(own_norm,1e-30),
            "interaction_to_own_norm_ratio": math.sqrt(sum(x*x for x in interaction))/max(own_norm,1e-30)}


def _passes(value):
    p = value["preservation"]
    return (p["cosine"] >= BARS["minimum_preservation_cosine"]
            and p["relative_l2_error"] <= BARS["maximum_preservation_relative_l2_error"]
            and p["sign_agreement"] >= BARS["minimum_preservation_sign_agreement"]
            and value["interaction_to_own_norm_ratio"] <= BARS["maximum_interaction_to_own_norm_ratio"])


def score(task_rows, bracket_rows, exactness):
    _, task_parent, _, bracket_parent = _load_artifacts()
    task_prior = {(x["row_id"], x["background"]): x["cardinality_prototype_q"] for x in task_parent["causal_evidence"]}
    bracket_prior = {(x["row_id"], x["side"]): x["program_donorward_effect"]
                     for x in bracket_parent["evidence"] if x["program_role"] == "target"}
    task_replay = max(abs(x["isolated_own"]-task_prior[(x["row_id"],x["background"])]) for x in task_rows)
    bracket_replay = max(abs(x["isolated_own"]-bracket_prior[(x["row_id"],x["side"])]) for x in bracket_rows)
    instrument = (len(task_rows)==512 and len(bracket_rows)==144 and task_replay<=MAX_REPLAY_ERROR
                  and bracket_replay<=MAX_REPLAY_ERROR and all(x<=MAX_REPLAY_ERROR for x in exactness.values()))
    task_score, bracket_score = _panel_score(task_rows), _panel_score(bracket_rows)
    live = (task_score["foreign_stress_to_own_norm_ratio"] >= BARS["minimum_stress_to_own_norm_ratio"]
            and bracket_score["foreign_stress_to_own_norm_ratio"] >= BARS["minimum_stress_to_own_norm_ratio"])
    price = derive_price(); price_pass = price["physical_model_forwards"]<=20 and price["example_evaluations"]<=3000 and not any(price[k] for k in ("backwards","fits","parameter_updates","vector_changes"))
    predictions = {"pred_a_immutable_programs_and_exact_corners": instrument,
                   "pred_b_foreign_stress_is_live": instrument and live,
                   "pred_c_task14_survives_bracket_program": instrument and _passes(task_score),
                   "pred_d_bracket_survives_task14_program": instrument and _passes(bracket_score),
                   "pred_e_fixed_price_and_no_refit": price_pass}
    valid = instrument and price_pass
    terminal = "composition_screen" if all(predictions.values()) else "inconclusive" if valid and not live else "interaction_null" if valid else "invalid"
    return {"task14": task_score, "bracket": bracket_score,
            "task14_parent_replay_max_absolute_error": task_replay,
            "bracket_parent_replay_max_absolute_error": bracket_replay,
            "exactness": exactness, "predictions": predictions, "terminal": terminal}


def main():
    plan = compile_plan()
    if os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1" or "--dry-run" in sys.argv:
        print(json.dumps(plan,sort_keys=True)); return
    if OUT.exists(): raise RuntimeError(f"refusing overwrite {OUT}")
    ta, _, ba, _ = _load_artifacts(); torch,F,facade=bracket_exact._dependencies()
    model,checkpoint=facade.load_bilin18(device="cuda",dtype=torch.float32,verify_weights_sha256=True)
    with torch.no_grad():
        task_rows,exactness=_task14_panel(model,torch,F,facade,ta,ba)
        bracket_rows=_bracket_panel(model,torch,F,facade,ta,ba)
    scored=score(task_rows,bracket_rows,exactness)
    payload=managed.atomic_create_json(OUT,{"schema":"task14_bracket_fixed_program_stress_composition_result_v1","candidate_id":CANDIDATE_ID,"created_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"plan":plan,"checkpoint_weights_sha256":checkpoint.weights_sha256,"score":scored,"task14_evidence":task_rows,"bracket_evidence":bracket_rows,"terminal":scored["terminal"]})
    print(json.dumps({"terminal":scored["terminal"],"predictions":scored["predictions"],"result_sha256":hashlib.sha256(payload).hexdigest()},sort_keys=True))


if __name__=="__main__": main()
