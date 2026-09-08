#!/usr/bin/env python3
"""Causal module/head/factor atlas for normalized-weight reader nominations."""

# BQGATE: EXPERIMENT pred_a_authority_replay_geometry_closure_finiteness_and_price pred_b_a_complete_layer12_or13_module_mediates_the_program pred_c_a_stable_complete_head_mediates pred_d_weight_nominated_factor_mediates pred_e_factorial_is_control_selective pred_f_singleton_heads_compose_to_modules
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import attention_factor_mediation_executor as factor_executor
import circuit_candidate_tense_auxiliary_is_was_v15_aligned_controls_v1 as v15
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import entry12_finite_router_contract as finite_router
import entry12_response_basis_contract as basis_contract
import head_response_mediation_scorer as scorer
import residual_state_mediation_executor as state_executor
import run_temporal_iswas_v15_construction_oracle_attention15_dependency_factorial_v1 as dependency
import run_temporal_iswas_v15_head_response_target_feasible_regularized_das_v1 as parent
from two_by_two_dependency_contract import vector_metrics

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v15_entry12_rank2_attention12_13_factor_mediation_atlas_v1.json"
WEIGHT_AUDIT = ROOT / "circuits/followups/temporal_iswas_v15_entry12_rank2_normalized_weight_reader_atlas_v2_price_audit_result.json"
BASIS_RESULT = ROOT / "circuits/followups/temporal_iswas_v15_entry12_shared_target_control_response_basis_v1_result.json"
TOKEN_RESULT = ROOT / "circuits/followups/temporal_iswas_v15_entry12_unordered_token_pair_router_v1_result.json"
ORACLES = ROOT / "circuits/followups/temporal_iswas_v15_construction_oracle_projective_bisector_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_iswas_v15_entry12_rank2_attention12_13_factor_mediation_atlas_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.iswas_v15_entry12_rank2_attention12_13_factor_mediation_atlas_v1"
EXPECTED = {
    "prior": "24054773d85e49c2dc5ad7f3f618ac6297066148e0e10c5c3590021d43fe6052",
    "weight_audit": "afd58acc948a4338ca7fa15a296e765ad29020320fe403d0c749b7ff423b2f58",
    "basis_result": "ffaec3f0bede415f7b9b1664ae7ddf1667978dc26f42cfa2f81e9c82f0f77467",
    "token_result": "582c19b22017fd0bb1070260b4737b8b046c54c3838a0a646a85254d77135be0",
    "oracles": "dde8cc793f2ba2d1f0bf7439dd9c62cb53ad979ea4ab3363928015b8cbe7e224",
    "factor_executor": "a8f4e003611af0dc82a04999865408abdd1e7a7cbd0caf5cbb2f8dd045bd4e35",
    "finite_router": "48007c4950bd9116601afbfe66e5e4a6d9bb7dbef65f90afc2dd405714809222",
    "basis_contract": "cc05ed73baf02c7ec6f0ec2b9e64a831a1fff6462ab4d8a5ba1f39bc5f70dab1",
    "state_executor": "ae3dc83c4a1954575778fa88744f964efc712e8bc483de078e366a90afee9720",
    "scorer": "9d0b56bd68a75e939d6cd0cc900de82d905d4d7131b56261d17759e83a28ba2f",
    "dependency": "4d435dfa6c6f29a34de2b5ba7679aafb3b622cdf0fbf2ea779c58bea982b1fe3",
    "parent": "0b7d92038283cad0fd48e398739474c4004de0da8e361d5aaaf0f68a4db6595e",
    "v15": "7027e128ea3e68a35e92f451d0f62453a937e64a8dafda24d841aa1f1d29e645",
}
FILES = {
    "prior": PRIOR, "weight_audit": WEIGHT_AUDIT, "basis_result": BASIS_RESULT,
    "token_result": TOKEN_RESULT, "oracles": ORACLES,
    "factor_executor": ROOT / "ops/attention_factor_mediation_executor.py",
    "finite_router": ROOT / "ops/entry12_finite_router_contract.py",
    "basis_contract": ROOT / "ops/entry12_response_basis_contract.py",
    "state_executor": ROOT / "ops/residual_state_mediation_executor.py",
    "scorer": ROOT / "ops/head_response_mediation_scorer.py",
    "dependency": ROOT / "ops/run_temporal_iswas_v15_construction_oracle_attention15_dependency_factorial_v1.py",
    "parent": ROOT / "ops/run_temporal_iswas_v15_head_response_target_feasible_regularized_das_v1.py",
    "v15": ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_v15_aligned_controls_v1.py",
}
PRICE_MAX = {"native_capture_forwards": 10, "differentiable_transformer_forwards": 156,
             "transformer_backward_forwards": 0, "model_updates": 0,
             "example_evaluations": 12000, "fit_parameters": 0}
NOMINATED_FACTORS = ("L12H0:v", "L12H4:q", "L12H4:k", "L13H2:q")


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict): return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)): return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def maxdiff(left, right): return float((left - right).abs().max())


def stack_prefix(torch, tensor, context, panel):
    indices = context["panel_indices"][panel].detach().cpu().tolist()
    return torch.cat([tensor[index, :int(context["base_batch"].semantic_positions[index]) + 1].float()
                      for index in indices], dim=0)


def panel_vector(context, output, background, panel):
    indices = context["panel_indices"][panel]
    margin = lambda item: item["logits"][context["index"], context["answer"]] - item["logits"][context["index"], context["foil"]]
    return (margin(output) - margin(background))[indices].detach().cpu().tolist()


def control(backend, context, changed, background, panel):
    indices = context["panel_indices"][panel]
    lp = backend.F.log_softmax(changed["logits"][indices], -1)
    lq = backend.F.log_softmax(background["logits"][indices], -1)
    kl = (lq.exp() * (lq - lp)).sum(-1)
    flips = changed["logits"][indices].argmax(-1) != background["logits"][indices].argmax(-1)
    return {"mean_kl": float(kl.mean()), "top1_flip_count": int(flips.sum())}


def mediators():
    result = {f"module{layer}": factor_executor.mediator_sites(layer) for layer in (12, 13)}
    for layer in (12, 13):
        for head in range(9):
            result[f"L{layer}H{head}"] = factor_executor.mediator_sites(layer, head)
    for layer, head in ((12, 0), (12, 4), (13, 2)):
        for factor in factor_executor.FACTORS:
            result[f"L{layer}H{head}:{factor}"] = factor_executor.mediator_sites(layer, head, factor)
    if len(result) != 35:
        raise RuntimeError("mediator inventory changed")
    return result


def main():
    observed = {name: sha(path) for name, path in FILES.items()}
    weight_audit = json.loads(WEIGHT_AUDIT.read_text())
    basis_result = json.loads(BASIS_RESULT.read_text())
    token_result = json.loads(TOKEN_RESULT.read_text())
    authority = bool(observed == EXPECTED
                     and weight_audit.get("terminal") == "normalized_reader_candidate_price_corrected"
                     and basis_result.get("predictions", {}).get("pred_b_construction_union_is_target_sufficient") is True
                     and token_result.get("terminal") == "token_pair_router_candidate")
    inventory = mediators()
    dry = {"candidate_id": CANDIDATE_ID, "dryrun": True, "authority_ok": authority,
           "mediator_count": len(inventory), "expected_differentiable_forwards": 156,
           "price_max": PRICE_MAX}
    if not authority: raise RuntimeError(f"authority changed: {observed}")
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)

    started = time.perf_counter(); backend = producer.Bilin18TorchBackend.load("cuda")
    for parameter in backend.model.parameters(): parameter.requires_grad_(False)
    counters = {name: 0 for name in PRICE_MAX}; native = backend.native
    def counted(batch, *, capture):
        counters["native_capture_forwards"] += 1; counters["example_evaluations"] += len(batch.row_ids)
        return native(batch, capture=capture)
    backend.native = counted
    rows = v15.build_rows(); bank = parent.capture_bank(backend, rows, counters, factors=False)
    contexts = {parity: parent.attach_references(
        backend, parent.subset_context(bank, parent.row_indices(rows, parity=parity)), counters)
        for parity in (0, 1)}
    off_state = {parity: state_executor.execute(parent, backend, context, counters, {}, capture=True)
                 for parity, context in contexts.items()}
    oracle_data = json.loads(ORACLES.read_text()); expert_state = {panel: {} for panel in ("A1", "A2")}
    for panel, expert in (("A1", "A1_oracle"), ("A2", "A2_oracle")):
        bases = dependency.bases_for_evaluation(backend.torch, backend.device, oracle_data, expert)
        for parity, context in contexts.items():
            expert_state[panel][parity] = state_executor.execute(
                parent, backend, context, counters, bases[parity], capture=True)

    reports, controls, compositions, geometry = {}, {}, {}, {}
    replay_error = geometry_error = orthogonality = closure = 0.0
    target_sufficiency = {}
    for held in (0, 1):
        train = 1 - held; train_context = contexts[train]
        base_entry = off_state[train][1]["entry12"]
        deltas = {panel: expert_state[panel][train][1]["entry12"].float() - base_entry.float()
                  for panel in ("A1", "A2")}
        target_a1 = stack_prefix(backend.torch, deltas["A1"], train_context, "A1").mean(0)
        target_a2 = stack_prefix(backend.torch, deltas["A2"], train_context, "A2").mean(0)
        p_rows = backend.torch.cat([stack_prefix(backend.torch, deltas[panel], train_context, "P")
                                    for panel in ("A1", "A2")], dim=0)
        bases, geom = basis_contract.fit_bases(
            backend.torch, target_a1, target_a2, p_rows, relative_threshold=1e-6)
        union = bases["construction_union_rank_le_2"]
        geometry[str(held)] = {"rank": int(union.shape[1]), **geom}
        orthogonality = max(orthogonality, float((union.T @ union - backend.torch.eye(2, device=union.device)).abs().max()))
        expected_singular = basis_result["fits"][str(held)]["union_singular_values"]
        geometry_error = max(geometry_error, max(abs(a - b) / max(abs(b), 1e-30)
            for a, b in zip(geom["union_singular_values"], expected_singular)))

        context = contexts[held]; off_output = off_state[held][0]
        expert_entries = {panel: expert_state[panel][held][1]["entry12"] for panel in ("A1", "A2")}
        truth = finite_router.labels_for_rows(backend.torch, context["rows"], device=backend.device)
        gold_entry = finite_router.routed_absolute(
            backend.torch, off_state[held][1]["entry12"], expert_entries, union, truth,
            context["base_batch"].semantic_positions)
        gold_reference = state_executor.execute(parent, backend, context, counters, {}, absolute={"entry12": gold_entry})
        off_factor, factors_off = factor_executor.execute(parent, backend, context, counters)
        on_factor, factors_on = factor_executor.execute(parent, backend, context, counters, entry12=gold_entry)
        replay_error = max(replay_error, maxdiff(off_factor["logits"], off_output["logits"]),
                           maxdiff(on_factor["logits"], gold_reference["logits"]))
        target_sufficiency[str(held)] = {}
        for panel in ("A1", "A2"):
            full = panel_vector(context, expert_state[panel][held][0], off_output, panel)
            routed = panel_vector(context, on_factor, off_output, panel)
            target_sufficiency[str(held)][panel] = vector_metrics(routed, full)

        reports[str(held)] = {}; controls[str(held)] = {}
        for name, sites in inventory.items():
            rescue_abs = factor_executor.absolute_for(
                factors_off, factors_on, sites, context["base_batch"].semantic_positions)
            reset_abs = factor_executor.absolute_for(
                factors_on, factors_off, sites, context["base_batch"].semantic_positions)
            rescue, _ = factor_executor.execute(parent, backend, context, counters, factor_absolute=rescue_abs)
            reset, _ = factor_executor.execute(parent, backend, context, counters,
                                               entry12=gold_entry, factor_absolute=reset_abs)
            cells = {"00": off_factor, "01": rescue, "10": reset, "11": on_factor}
            item = {}
            for panel in ("A1", "A2"):
                vectors = {cell: panel_vector(context, output, off_output, panel)
                           for cell, output in cells.items()}
                target = vectors["11"]
                item[panel] = scorer.score_cells(vectors, target)
                closure = max(closure, item[panel]["closure_max_abs_error"])
            reports[str(held)][name] = item
            if name.startswith("module"):
                controls[str(held)][name] = {panel: {
                    "rescue": control(backend, context, rescue, off_factor, panel),
                    "reset": control(backend, context, reset, on_factor, panel),
                } for panel in ("P", "C")}
        compositions[str(held)] = {}
        for layer in (12, 13):
            compositions[str(held)][str(layer)] = {panel: scorer.singleton_module_composition(
                {f"L{layer}H{head}": reports[str(held)][f"L{layer}H{head}"][panel] for head in range(9)},
                reports[str(held)][f"module{layer}"][panel]) for panel in ("A1", "A2")}

    counters["fit_parameters"] = 0
    native_closure = max(context["manual_native_max_abs_error"] for context in contexts.values())
    sufficient = all(target_sufficiency[str(fold)][panel]["signed_projection"] >= .75
                     and target_sufficiency[str(fold)][panel]["direction_fraction"] >= .875
                     for fold in (0, 1) for panel in ("A1", "A2"))
    A = bool(authority and native_closure <= 1e-4 and replay_error <= 1e-4
             and geometry_error <= 1e-5 and orthogonality <= 1e-5 and closure <= 1e-12
             and sufficient and finite({"reports": reports, "controls": controls, "compositions": compositions})
             and counters["differentiable_transformer_forwards"] == 156
             and all(counters[name] <= PRICE_MAX[name] for name in PRICE_MAX))
    passes = lambda fold, name, bar: all(
        reports[str(fold)][name][panel]["metrics"][component]["signed_projection"] >= bar
        and reports[str(fold)][name][panel]["metrics"][component]["direction_fraction"] >= .875
        for panel in ("A1", "A2") for component in ("head_reset_loss", "head_rescue"))
    B = any(all(passes(fold, f"module{layer}", .5) for fold in (0, 1)) for layer in (12, 13))
    C = any(all(passes(fold, f"L{layer}H{head}", .1) for fold in (0, 1))
            for layer in (12, 13) for head in range(9))
    D = any(all(passes(fold, name, .05) for fold in (0, 1)) for name in NOMINATED_FACTORS)
    E = all(controls[str(fold)][f"module{layer}"][panel][arm]["top1_flip_count"] == 0
            and controls[str(fold)][f"module{layer}"][panel][arm]["mean_kl"] <= .02
            for fold in (0, 1) for layer in (12, 13) for panel in ("P", "C") for arm in ("reset", "rescue"))
    F = all(compositions[str(fold)][str(layer)][panel]["metrics"]["cosine"] >= .95
            and compositions[str(fold)][str(layer)][panel]["metrics"]["relative_l2_error"] <= .25
            for fold in (0, 1) for layer in (12, 13) for panel in ("A1", "A2"))
    predictions = dict(zip((
        "pred_a_authority_replay_geometry_closure_finiteness_and_price",
        "pred_b_a_complete_layer12_or13_module_mediates_the_program",
        "pred_c_a_stable_complete_head_mediates",
        "pred_d_weight_nominated_factor_mediates",
        "pred_e_factorial_is_control_selective",
        "pred_f_singleton_heads_compose_to_modules"), map(bool, (A, B, C, D, E, F))))
    terminal = ("invalid" if not A else "factor_reader_candidate" if B and C and D
                else "distributed_module_reader" if B else "weight_reader_bypass_null")
    result = {"schema": "temporal_iswas_v15_entry12_rank2_attention12_13_factor_mediation_atlas_result_v1",
              "candidate_id": CANDIDATE_ID, "started_utc": datetime.now(timezone.utc).isoformat(),
              "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
              "target_sufficiency": target_sufficiency, "geometry": geometry,
              "reports": reports, "controls": controls, "singleton_module_composition": compositions,
              "instrument": {"manual_native_max_abs_error": native_closure,
                             "off_on_replay_max_abs_error": replay_error,
                             "geometry_relative_max_error": geometry_error,
                             "orthogonality_max_abs_error": orthogonality,
                             "factorial_closure_max_abs_error": closure,
                             "mediator_count": len(inventory)},
              "predictions": predictions, "terminal": terminal,
              "price": {**counters, "maxima": PRICE_MAX}}
    atomic_create_json(OUT, result)
    print(json.dumps({"predictions": predictions, "terminal": terminal,
                      "target_sufficiency": target_sufficiency, "instrument": result["instrument"],
                      "controls": controls, "singleton_module_composition": compositions,
                      "price": result["price"]}, sort_keys=True))


if __name__ == "__main__": main()
