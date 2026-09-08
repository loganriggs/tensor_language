#!/usr/bin/env python3
"""Direct residual-identity route from the selective entry12 program to the final head."""

# BQGATE: EXPERIMENT pred_a_authority_replay_geometry_closure_finiteness_and_price pred_b_suffix_write_bank_is_minor pred_c_direct_residual_route_is_sufficient pred_d_direct_state_obeys_recurrent_product_law pred_e_route_is_control_selective
from __future__ import annotations

from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

import circuit_candidate_tense_auxiliary_is_was_v15_aligned_controls_v1 as v15
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import entry12_finite_router_contract as finite_router
import entry12_response_basis_contract as basis_contract
import head_response_mediation_scorer as scorer
import module_write_mediation_contract as write_contract
import residual_state_mediation_executor as state_executor
import run_temporal_iswas_v15_construction_oracle_attention15_dependency_factorial_v1 as dependency
import run_temporal_iswas_v15_head_response_target_feasible_regularized_das_v1 as parent
from two_by_two_dependency_contract import vector_metrics

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v15_entry12_rank2_direct_residual_final_head_route_v1.json"
FACTOR_ATLAS = ROOT / "circuits/followups/temporal_iswas_v15_entry12_rank2_attention12_13_factor_mediation_atlas_v1_result.json"
BASIS_RESULT = ROOT / "circuits/followups/temporal_iswas_v15_entry12_shared_target_control_response_basis_v1_result.json"
ORACLES = ROOT / "circuits/followups/temporal_iswas_v15_construction_oracle_projective_bisector_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_iswas_v15_entry12_rank2_direct_residual_final_head_route_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.iswas_v15_entry12_rank2_direct_residual_final_head_route_v1"
EXPECTED = {
    "prior": "b755b2437d6117cd648e0ef6c590d0ab831a21c2d7fd34e35de8557efc7afa5f",
    "factor_atlas": "f029dbefe062d49362987081b0404ea00b3db4120cb384f899df8bbf5d3482c0",
    "basis_result": "ffaec3f0bede415f7b9b1664ae7ddf1667978dc26f42cfa2f81e9c82f0f77467",
    "oracles": "dde8cc793f2ba2d1f0bf7439dd9c62cb53ad979ea4ab3363928015b8cbe7e224",
    "write_contract": "43181f1f69512899bd3f13ea159bceb7bcef6f892246b879d1c42d830c5f39c3",
    "finite_router": "48007c4950bd9116601afbfe66e5e4a6d9bb7dbef65f90afc2dd405714809222",
    "basis_contract": "cc05ed73baf02c7ec6f0ec2b9e64a831a1fff6462ab4d8a5ba1f39bc5f70dab1",
    "state_executor": "ae3dc83c4a1954575778fa88744f964efc712e8bc483de078e366a90afee9720",
    "scorer": "9d0b56bd68a75e939d6cd0cc900de82d905d4d7131b56261d17759e83a28ba2f",
    "dependency": "4d435dfa6c6f29a34de2b5ba7679aafb3b622cdf0fbf2ea779c58bea982b1fe3",
    "parent": "0b7d92038283cad0fd48e398739474c4004de0da8e361d5aaaf0f68a4db6595e",
    "v15": "7027e128ea3e68a35e92f451d0f62453a937e64a8dafda24d841aa1f1d29e645",
}
FILES = {
    "prior": PRIOR, "factor_atlas": FACTOR_ATLAS, "basis_result": BASIS_RESULT, "oracles": ORACLES,
    "write_contract": ROOT / "ops/module_write_mediation_contract.py",
    "finite_router": ROOT / "ops/entry12_finite_router_contract.py",
    "basis_contract": ROOT / "ops/entry12_response_basis_contract.py",
    "state_executor": ROOT / "ops/residual_state_mediation_executor.py",
    "scorer": ROOT / "ops/head_response_mediation_scorer.py",
    "dependency": ROOT / "ops/run_temporal_iswas_v15_construction_oracle_attention15_dependency_factorial_v1.py",
    "parent": ROOT / "ops/run_temporal_iswas_v15_head_response_target_feasible_regularized_das_v1.py",
    "v15": ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_v15_aligned_controls_v1.py",
}
PRICE_MAX = {"native_capture_forwards": 10, "differentiable_transformer_forwards": 24,
             "transformer_backward_forwards": 0, "model_updates": 0,
             "example_evaluations": 2000, "fit_parameters": 0}


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


def module_sites(model):
    sites = {}
    for layer in range(12, 18):
        block = model.transformer.h[layer]
        sites[f"attn{layer}"] = (block.attn, "attention")
        sites[f"mlp{layer}"] = (block.mlp, "mlp")
    return sites


def exact_head(backend, state):
    return 30.0 * backend.torch.tanh(
        backend.model.lm_head(backend.F.rms_norm(state, (backend.model.config.n_embd,))) / 30.0)


def main():
    observed = {name: sha(path) for name, path in FILES.items()}
    factor_atlas = json.loads(FACTOR_ATLAS.read_text()); basis_result = json.loads(BASIS_RESULT.read_text())
    authority = bool(observed == EXPECTED and factor_atlas.get("terminal") == "weight_reader_bypass_null"
                     and basis_result.get("predictions", {}).get("pred_b_construction_union_is_target_sufficient") is True)
    dry = {"candidate_id": CANDIDATE_ID, "dryrun": True, "authority_ok": authority,
           "expected_differentiable_forwards": 24, "suffix_write_count": 12, "price_max": PRICE_MAX}
    if not authority: raise RuntimeError(f"authority changed: {observed}")
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)

    start = time.perf_counter(); backend = producer.Bilin18TorchBackend.load("cuda")
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
    oracles = json.loads(ORACLES.read_text()); expert_state = {panel: {} for panel in ("A1", "A2")}
    for panel, expert in (("A1", "A1_oracle"), ("A2", "A2_oracle")):
        bases = dependency.bases_for_evaluation(backend.torch, backend.device, oracles, expert)
        for parity, context in contexts.items():
            expert_state[panel][parity] = state_executor.execute(parent, backend, context, counters, bases[parity], capture=True)

    reports, controls, direct_state, geometry = {}, {}, {}, {}
    replay_error = head_replay = closure = geometry_error = orthogonality = 0.0
    sites = module_sites(backend.model)
    for held in (0, 1):
        train = 1 - held; train_context = contexts[train]; base_entry = off_state[train][1]["entry12"]
        deltas = {panel: expert_state[panel][train][1]["entry12"].float() - base_entry.float()
                  for panel in ("A1", "A2")}
        target_a1 = stack_prefix(backend.torch, deltas["A1"], train_context, "A1").mean(0)
        target_a2 = stack_prefix(backend.torch, deltas["A2"], train_context, "A2").mean(0)
        p_rows = backend.torch.cat([stack_prefix(backend.torch, deltas[panel], train_context, "P")
                                    for panel in ("A1", "A2")], dim=0)
        bases, geom = basis_contract.fit_bases(backend.torch, target_a1, target_a2, p_rows, relative_threshold=1e-6)
        union = bases["construction_union_rank_le_2"]; geometry[str(held)] = {"rank": 2, **geom}
        orthogonality = max(orthogonality, float((union.T @ union - backend.torch.eye(2, device=union.device)).abs().max()))
        geometry_error = max(geometry_error, max(abs(a-b)/max(abs(b),1e-30) for a,b in zip(
            geom["union_singular_values"], basis_result["fits"][str(held)]["union_singular_values"])))

        context = contexts[held]; off_output = off_state[held][0]
        entries = {panel: expert_state[panel][held][1]["entry12"] for panel in ("A1", "A2")}
        truth = finite_router.labels_for_rows(backend.torch, context["rows"], device=backend.device)
        gold_entry = finite_router.routed_absolute(backend.torch, off_state[held][1]["entry12"], entries,
                                                   union, truth, context["base_batch"].semantic_positions)
        gold_reference = state_executor.execute(parent, backend, context, counters, {}, absolute={"entry12": gold_entry})
        off, off_writes = write_contract.capture_writes(
            sites, lambda: state_executor.execute(parent, backend, context, counters, {}))
        on, on_writes = write_contract.capture_writes(
            sites, lambda: state_executor.execute(parent, backend, context, counters, {}, absolute={"entry12": gold_entry}))
        off_replay = write_contract.execute_with_writes(
            sites, off_writes, lambda: state_executor.execute(parent, backend, context, counters, {}))
        on_replay = write_contract.execute_with_writes(
            sites, on_writes, lambda: state_executor.execute(parent, backend, context, counters, {}, absolute={"entry12": gold_entry}))
        replay_error = max(replay_error, maxdiff(off["logits"], off_output["logits"]),
                           maxdiff(on["logits"], gold_reference["logits"]),
                           maxdiff(off_replay["logits"], off["logits"]), maxdiff(on_replay["logits"], on["logits"]))
        positions = context["base_batch"].semantic_positions
        rescue_bank = write_contract.hybrid_bank(off_writes, on_writes, sites, positions)
        reset_bank = write_contract.hybrid_bank(on_writes, off_writes, sites, positions)
        rescue = write_contract.execute_with_writes(
            sites, rescue_bank, lambda: state_executor.execute(parent, backend, context, counters, {}))
        reset = write_contract.execute_with_writes(
            sites, reset_bank, lambda: state_executor.execute(parent, backend, context, counters, {}, absolute={"entry12": gold_entry}))
        cells = {"00": off, "01": rescue, "10": reset, "11": on}
        reports[str(held)] = {}
        for panel in ("A1", "A2"):
            vectors = {cell: panel_vector(context, output, off, panel) for cell, output in cells.items()}
            reports[str(held)][panel] = scorer.score_cells(vectors, vectors["11"])
            closure = max(closure, reports[str(held)][panel]["closure_max_abs_error"])
        controls[str(held)] = {panel: {
            "rescue": control(backend, context, rescue, off, panel),
            "reset": control(backend, context, reset, on, panel)} for panel in ("P", "C")}

        _, lengths = backend._tensor_batch(context["base_batch"])
        semantic_is_final = all(int(stop) == int(length)-1 for stop, length in zip(positions, lengths))
        index = backend.torch.arange(len(lengths), device=backend.device)
        pos = backend.torch.tensor(positions, device=backend.device)
        entry_delta = gold_entry[index, pos].float() - off_state[held][1]["entry12"][index, pos].float()
        gamma = backend.torch.stack([backend.model.transformer.h[layer].lambdas[0].detach().float()
                                     for layer in range(12, 18)]).prod()
        observed_delta = reset["state"].float() - off["state"].float()
        relative = float((observed_delta - gamma * entry_delta).norm() /
                         (gamma * entry_delta).norm().clamp_min(1e-30))
        direct_state[str(held)] = {"semantic_position_is_final_token": semantic_is_final,
                                   "recurrent_product": float(gamma), "relative_l2_error": relative}
        for output in cells.values(): head_replay = max(head_replay, maxdiff(exact_head(backend, output["state"]), output["logits"]))

    counters["fit_parameters"] = 0
    native_closure = max(context["manual_native_max_abs_error"] for context in contexts.values())
    A = bool(authority and native_closure <= 1e-4 and replay_error <= 1e-4 and head_replay <= 1e-4
             and closure <= 1e-12 and geometry_error <= 1e-5 and orthogonality <= 1e-5
             and finite({"reports": reports, "controls": controls, "direct_state": direct_state})
             and counters["differentiable_transformer_forwards"] == 24
             and all(counters[name] <= PRICE_MAX[name] for name in PRICE_MAX))
    B = all(reports[str(fold)][panel]["metrics"][component]["signed_projection"] < .25
            for fold in (0,1) for panel in ("A1","A2") for component in ("head_reset_loss","head_rescue"))
    C = all(reports[str(fold)][panel]["metrics"]["head_bypass"]["signed_projection"] >= .75
            and reports[str(fold)][panel]["metrics"]["head_bypass"]["direction_fraction"] >= .875
            for fold in (0,1) for panel in ("A1","A2"))
    D = all(item["semantic_position_is_final_token"] and item["relative_l2_error"] <= 1e-5
            for item in direct_state.values())
    E = all(controls[str(fold)][panel][arm]["top1_flip_count"] == 0
            and controls[str(fold)][panel][arm]["mean_kl"] <= .02
            for fold in (0,1) for panel in ("P","C") for arm in ("reset","rescue"))
    predictions = dict(zip(("pred_a_authority_replay_geometry_closure_finiteness_and_price",
        "pred_b_suffix_write_bank_is_minor", "pred_c_direct_residual_route_is_sufficient",
        "pred_d_direct_state_obeys_recurrent_product_law", "pred_e_route_is_control_selective"),
        map(bool,(A,B,C,D,E))))
    terminal = "invalid" if not A else "direct_residual_final_head_route" if all((B,C,D,E)) else "direct_route_null"
    result = {"schema":"temporal_iswas_v15_entry12_rank2_direct_residual_final_head_route_result_v1",
              "candidate_id":CANDIDATE_ID,"started_utc":datetime.now(timezone.utc).isoformat(),
              "serial_seconds":time.perf_counter()-start,"authority_sha256":EXPECTED,
              "geometry":geometry,"reports":reports,"controls":controls,"direct_state_law":direct_state,
              "instrument":{"manual_native_max_abs_error":native_closure,"write_and_program_replay_max_abs_error":replay_error,
                            "exact_final_head_replay_max_abs_error":head_replay,"factorial_closure_max_abs_error":closure,
                            "geometry_relative_max_error":geometry_error,"orthogonality_max_abs_error":orthogonality},
              "predictions":predictions,"terminal":terminal,"price":{**counters,"maxima":PRICE_MAX}}
    atomic_create_json(OUT,result); print(json.dumps({k:result[k] for k in ("predictions","terminal","direct_state_law","reports","controls","instrument","price")},sort_keys=True))


if __name__ == "__main__": main()
