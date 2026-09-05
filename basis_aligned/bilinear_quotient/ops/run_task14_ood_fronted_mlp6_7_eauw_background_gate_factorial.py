#!/usr/bin/env python3
"""Exact OOD E/A/U/W-background by grouped-MLP6--7 composition lattice."""

# BQGATE: EXPERIMENT pred_a_instrument_and_endpoint_closure pred_b_single_background_factor_gate pred_c_distributed_background_gate pred_d_higher_order_background_gate pred_e_direction_reversal pred_f_lexical_effect_absolutely_small

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

import circuit_fast_screen_managed_runner as managed
import run_task14_ood_fronted_mlp6_7_contextual_midpoint_tangent as ood_tangent
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/task14_ood_fronted_mlp6_7_eauw_background_gate_factorial_v1.json"
PARENT_RESULT = ROOT / "circuits/fast_screens/task14_ood_fronted_mlp6_7_contextual_midpoint_tangent_v1_result.json"
OUT = ROOT / "circuits/fast_screens/task14_ood_fronted_mlp6_7_eauw_background_gate_factorial_v1_result.json"
PRIOR_ART_SHA256 = "d32a267f0c887c209c3abe65b3a745d153a4cb7a3cd682276cf2e902a429f6c6"
PARENT_RESULT_SHA256 = "d22b73e6da04b90ca55ebaf3df209164628fd6916af1a7e4dcd50981086d9b68"
CANDIDATE_ID = "subject_verb.number_agreement.ood_fronted_mlp6_7_eauw_background_gate_factorial_v1"
BACKGROUND_FACTORS = ("E", "A", "U", "W")
FACTORS = (*BACKGROUND_FACTORS, "X")
BACKGROUND_SUBSETS = tuple("".join(parts) for size in range(5)
                           for parts in combinations(BACKGROUND_FACTORS, size))
METHODS = ("base", "exact")
PATCH_CHUNK_ROWS = 256
BARS = {
    "maximum_numerical_absolute_error": 5e-5,
    "minimum_single_factor_absolute_share": .65,
    "minimum_distributed_factor_absolute_share": .20,
    "minimum_distributed_factor_count": 2,
    "minimum_higher_order_absolute_mass_share": .35,
    "plural_to_singular_maximum_context_shift": -.05,
    "singular_to_plural_minimum_context_shift": .05,
    "maximum_absolute_lexical_margin_effect": .02,
}


class OODBackgroundGateError(ValueError):
    pass


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_rows():
    return ood_tangent.build_rows()


def validate_preflight():
    if _sha256(PRIOR_ART) != PRIOR_ART_SHA256:
        raise OODBackgroundGateError("prior-art receipt changed")
    if _sha256(PARENT_RESULT) != PARENT_RESULT_SHA256:
        raise OODBackgroundGateError("parent OOD tangent result changed")
    result = json.loads(PARENT_RESULT.read_text())
    predictions = result.get("score", {}).get("predictions", {})
    if result.get("terminal") != "valid_causal_screen" \
            or predictions.get("pred_a_instrument_live") is not True \
            or predictions.get("pred_b_ood_midpoint_geometry") is not True \
            or predictions.get("pred_c_ood_midpoint_task_prediction") is not True:
        raise OODBackgroundGateError("parent no longer licenses background-gate test")


def compile_plan():
    validate_preflight()
    return {
        "schema": "task14_ood_fronted_mlp6_7_eauw_background_gate_factorial_plan_v1",
        "candidate_id": CANDIDATE_ID,
        "split": "OOD_TEXT_REUSE_NEW_EAUW_BY_MLP6_7_INTERVENTION",
        "data_status": "already-open OOD text; complete background-by-X intervention is prospective",
        "row_count": 16, "sources": list(tangent.SOURCES),
        "background_factors": list(BACKGROUND_FACTORS),
        "background_subsets": list(BACKGROUND_SUBSETS), "methods": list(METHODS),
        "condition_count": 64, "prior_art_sha256": PRIOR_ART_SHA256,
        "parent_result_sha256": PARENT_RESULT_SHA256, "bars": dict(BARS),
        "price": {"physical_model_forwards": 10, "example_evaluations": 2144,
                  "causal_installations": 1024, "backwards": 0,
                  "parameter_updates": 0, "maximum_patch_chunk_rows": PATCH_CHUNK_ROWS},
        "closed_claims": ["pristine_OOD_confirmation", "semantic_uniqueness",
                          "MLP6_versus_MLP7_identity", "rank", "compression",
                          "activation_reconstruction",
                          "necessity_outside_fixed_L11H3_interface"],
    }


def _canonical_subset(parts):
    chosen = set(parts)
    return "".join(factor for factor in tangent.parent.FAMILIES if factor in chosen)


def _raw_for(recipient, source, subset, torch_F):
    if not subset:
        return recipient["raw_state"][:, tangent.parent.SUBJECT_POSITION]
    _, raw, _ = tangent.parent._hybrid_input(recipient, source, subset, torch_F)
    return raw[:, tangent.parent.SUBJECT_POSITION]


def _compile_patch(tokens, heads, rows, torch):
    indices, replacements, masks, specs = [], [], [], []
    for row_index, row in enumerate(rows):
        for source in tangent.SOURCES:
            for background in BACKGROUND_SUBSETS:
                for method in METHODS:
                    key = (source, background, method)
                    indices.append(row_index); replacements.append(heads[key][row_index])
                    masks.append(background == "" and method == "base")
                    specs.append((row_index, source, background, method,
                                  f"{row['direction_id']}__{row['template_id']}"))
    index = torch.tensor(indices, dtype=torch.long, device=tokens.device)
    return {"tokens": tokens[index],
            "finals": torch.full_like(index, tangent.parent.SUBJECT_POSITION),
            "replacement_heads": torch.stack(replacements),
            "native_reinstall_mask": torch.tensor(masks, dtype=torch.bool,
                                                    device=tokens.device),
            "specs": specs}


def evaluate(model, torch, F, facade):
    parent = tangent.parent
    rows = build_rows(); n = len(rows); device = next(model.parameters()).device
    tokens, finals = parent.downstream.depth.parent.v1._role_batch(rows, torch, device)
    native_roles = parent.factors._native_logits(model, tokens, torch, F)
    replay, captured, projection, role_closure, inputs = parent._decomposed_forward(
        model, tokens, finals, torch, F, facade)
    roles = {"recipient": tangent._role_slice(captured, 0, n),
             "opposite": tangent._role_slice(captured, n, 2*n)}
    input_roles = {"recipient": tangent._role_slice(inputs, 0, n),
                   "opposite": tangent._role_slice(inputs, n, 2*n),
                   "lexical": tangent._role_slice(inputs, 2*n, 3*n)}
    function = tangent._head_function(
        model, roles["recipient"], roles["opposite"],
        model.transformer.h[parent.LAYER].attn, projection, torch, F)
    heads = {}
    with torch.no_grad():
        for source in tangent.SOURCES:
            for background in BACKGROUND_SUBSETS:
                base_subset = _canonical_subset(background)
                exact_subset = _canonical_subset(background + "YZ")
                base_raw = _raw_for(input_roles["recipient"], input_roles[source],
                                    base_subset, F)
                exact_raw = _raw_for(input_roles["recipient"], input_roles[source],
                                     exact_subset, F)
                heads[(source, background, "base")] = function(base_raw).detach()
                heads[(source, background, "exact")] = function(exact_raw).detach()
    patch = _compile_patch(tokens[:n], heads, rows, torch)
    native_chunks, patched_chunks, closures = [], [], []
    noop_error = 0.0
    for start in range(0, len(patch["specs"]), PATCH_CHUNK_ROWS):
        stop = min(start + PATCH_CHUNK_ROWS, len(patch["specs"]))
        chunk_tokens = patch["tokens"][start:stop]
        native = parent.factors._native_logits(model, chunk_tokens, torch, F)
        patched, _, _, closure = parent.downstream._decomposed_forward(
            model, chunk_tokens, patch["finals"][start:stop], torch, F, facade,
            replacement_heads=patch["replacement_heads"][start:stop],
            native_reinstall_mask=patch["native_reinstall_mask"][start:stop])
        mask = patch["native_reinstall_mask"][start:stop]
        if bool(mask.any()):
            noop_error = max(noop_error, float((patched[mask]-native[mask]).abs().max()))
        native_chunks.append(native[:, parent.SUBJECT_POSITION].detach().cpu())
        patched_chunks.append(patched[:, parent.SUBJECT_POSITION].detach().cpu())
        closures.append(closure)
    native_patch = torch.cat(native_chunks); patched = torch.cat(patched_chunks)
    exactness = {
        "native_role_replay_max_absolute_logit_error": float((native_roles-replay).abs().max()),
        "role_state_closure_max_absolute_error": role_closure["input_state_closure_max_absolute_error"],
        "role_normalized_closure_max_absolute_error": role_closure["input_normalized_closure_max_absolute_error"],
        "downstream_state_closure_max_absolute_error": max(
            x["state_sum_max_absolute_error"] for x in closures),
        "downstream_normalized_closure_max_absolute_error": max(
            x["normalized_state_max_absolute_error"] for x in closures),
        "recipient_noop_full_logit_max_absolute_error": noop_error,
    }
    evidence = []
    for out_index, (row_index, source, background, method, cell_id) in enumerate(patch["specs"]):
        base = parent.grandparent._both_metrics(native_patch[out_index], rows[row_index], torch)
        value = parent.grandparent._both_metrics(patched[out_index], rows[row_index], torch)
        evidence.append({"row_id": rows[row_index]["row_id"], "cell_id": cell_id,
                         "source": source, "background": background, "method": method,
                         "target_margin": value[f"{source}_target_margin"],
                         "target_CE": value[f"{source}_target_CE"],
                         "target_margin_improvement": value[f"{source}_target_margin"]
                             - base[f"{source}_target_margin"],
                         "target_CE_improvement": base[f"{source}_target_CE"]
                             - value[f"{source}_target_CE"]})
    return evidence, exactness


def _mobius(values):
    result = {}
    for size in range(1, len(FACTORS)+1):
        for parts in combinations(FACTORS, size):
            subset = "".join(parts); total = 0.0
            for inner_size in range(size+1):
                for inner in combinations(parts, inner_size):
                    key = "".join(factor for factor in FACTORS if factor in inner)
                    total += (-1) ** (size-inner_size) * values[key]
            result[subset] = total
    return result


def score(evidence, exactness, bars=BARS):
    expected = {(row["row_id"], source, background, method)
                for row in build_rows() for source in tangent.SOURCES
                for background in BACKGROUND_SUBSETS for method in METHODS}
    observed = {(x["row_id"], x["source"], x["background"], x["method"])
                for x in evidence}
    if len(evidence) != len(expected) or observed != expected:
        raise OODBackgroundGateError("incomplete or duplicate background lattice")
    numeric = ("target_margin", "target_CE", "target_margin_improvement",
               "target_CE_improvement")
    if not all(math.isfinite(float(x[k])) for x in evidence for k in numeric):
        raise OODBackgroundGateError("non-finite task evidence")
    parent_by = {(x["row_id"], x["source"], x["background"], x["method"]): x
                 for x in json.loads(PARENT_RESULT.read_text())["evidence"]}
    endpoint_error = 0.0
    for item in evidence:
        if item["background"] not in {"", "EAUW"}:
            continue
        parent_background = "recipient" if item["background"] == "" else "donor_context"
        prior = parent_by[(item["row_id"], item["source"], parent_background, item["method"])]
        for metric in ("target_margin", "target_CE"):
            endpoint_error = max(endpoint_error, abs(float(item[metric])-float(prior[metric])))
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for item in evidence:
        grouped[item["cell_id"]][item["source"]][(item["background"], item["method"])].append(item)
    cells = {}
    mobius_closure = 0.0
    for cell_id, sources in grouped.items():
        cells[cell_id] = {}
        for source, conditions in sources.items():
            by_metric = {}
            for metric in ("margin", "CE"):
                key = f"target_{metric}_improvement"
                values = {}
                q = {}
                for background in BACKGROUND_SUBSETS:
                    base = statistics.fmean(float(x[key]) for x in conditions[(background, "base")])
                    exact = statistics.fmean(float(x[key]) for x in conditions[(background, "exact")])
                    values[background] = base
                    values[background + "X"] = exact
                    q[background] = exact-base
                terms = _mobius(values)
                contextual = {subset: terms[subset+"X"]
                              for size in range(1, 5)
                              for subset in ("".join(parts)
                                             for parts in combinations(BACKGROUND_FACTORS, size))}
                attribution = {factor: sum(value/len(subset)
                    for subset, value in contextual.items() if factor in subset)
                    for factor in BACKGROUND_FACTORS}
                total_shift = q["EAUW"] - q[""]
                mobius_closure = max(mobius_closure,
                                     abs(sum(attribution.values())-total_shift))
                mass = sum(abs(value) for value in attribution.values())
                shares = {factor: abs(value)/max(mass, 1e-12)
                          for factor, value in attribution.items()}
                term_mass = sum(abs(value) for value in contextual.values())
                higher_mass = sum(abs(value) for subset, value in contextual.items()
                                  if len(subset) >= 2)
                by_metric[metric] = {"q": q, "mobius_X_terms": contextual,
                                     "shapley_attribution": attribution,
                                     "absolute_attribution_share": shares,
                                     "context_shift": total_shift,
                                     "higher_order_absolute_mass_share":
                                         higher_mass/max(term_mass, 1e-12)}
            cells[cell_id][source] = by_metric
    exactness["parent_endpoint_margin_CE_max_absolute_error"] = endpoint_error
    exactness["mobius_shapley_closure_max_absolute_error"] = mobius_closure
    instrument = all(float(value) <= bars["maximum_numerical_absolute_error"]
                     for value in exactness.values())
    opposite_margin = {cell_id: cell["opposite"]["margin"]
                       for cell_id, cell in cells.items()}
    winners = {}
    for cell_id, entry in opposite_margin.items():
        shares = entry["absolute_attribution_share"]
        winner = max(shares, key=shares.get)
        winners[cell_id] = winner if shares[winner] >= bars[
            "minimum_single_factor_absolute_share"] \
            and entry["shapley_attribution"][winner]*entry["context_shift"] > 0 else None
    single = instrument and len(set(winners.values())) == 1 and None not in winners.values()
    distributed = instrument and not single and all(
        sum(share >= bars["minimum_distributed_factor_absolute_share"]
            for share in entry["absolute_attribution_share"].values())
        >= bars["minimum_distributed_factor_count"]
        for entry in opposite_margin.values())
    higher_order = instrument and any(entry["higher_order_absolute_mass_share"]
                                      >= bars["minimum_higher_order_absolute_mass_share"]
                                      for entry in opposite_margin.values())
    direction_shift = {cell_id.split("__", 1)[0]: entry["context_shift"]
                       for cell_id, entry in opposite_margin.items()}
    reversal = instrument \
        and direction_shift["plural_to_singular"] <= bars[
            "plural_to_singular_maximum_context_shift"] \
        and direction_shift["singular_to_plural"] >= bars[
            "singular_to_plural_minimum_context_shift"]
    maximum_lexical = max(abs(value) for cell in cells.values()
                          for value in cell["lexical"]["margin"]["q"].values())
    lexical = instrument and maximum_lexical <= bars[
        "maximum_absolute_lexical_margin_effect"]
    return {**exactness, "cells": cells, "single_factor_winners": winners,
            "direction_context_shift": direction_shift,
            "maximum_absolute_lexical_margin_effect": maximum_lexical,
            "predictions": {
                "pred_a_instrument_and_endpoint_closure": bool(instrument),
                "pred_b_single_background_factor_gate": bool(single),
                "pred_c_distributed_background_gate": bool(distributed),
                "pred_d_higher_order_background_gate": bool(higher_order),
                "pred_e_direction_reversal": bool(reversal),
                "pred_f_lexical_effect_absolutely_small": bool(lexical)}}


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv); plan = compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True)); return
    if OUT.exists():
        raise OODBackgroundGateError(f"refusing to overwrite {OUT}")
    torch, F, facade = tangent.parent.factors._dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32,
                                            verify_weights_sha256=True)
    evidence, exactness = evaluate(model, torch, F, facade)
    scored = score(evidence, exactness)
    terminal = "valid_causal_screen" if scored["predictions"][
        "pred_a_instrument_and_endpoint_closure"] else "invalid"
    result = {"schema": "task14_ood_fronted_mlp6_7_eauw_background_gate_factorial_result_v1",
              "candidate_id": CANDIDATE_ID, "terminal": terminal, "plan": plan,
              "checkpoint_weights_sha256": checkpoint.weights_sha256,
              "score": scored, "evidence": evidence,
              "evaluated_splits": ["OOD_TEXT_REUSE_NEW_EAUW_BY_MLP6_7_INTERVENTION"],
              "forbidden_splits_opened": []}
    payload = managed.atomic_create_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": scored["predictions"],
                      "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
