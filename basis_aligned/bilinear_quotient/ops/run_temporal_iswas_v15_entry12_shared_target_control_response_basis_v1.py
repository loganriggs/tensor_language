#!/usr/bin/env python3
"""Cross-fitted shared target/control response bases at the causal entry12 state."""

# BQGATE: EXPERIMENT pred_a_authority_replay_scope_geometry_finiteness_and_price pred_b_construction_union_is_target_sufficient pred_c_p_complement_preserves_target pred_d_p_complement_removes_collision pred_e_shared_rank1_is_construction_general
from __future__ import annotations

from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

import circuit_candidate_tense_auxiliary_is_was_v15_aligned_controls_v1 as v15
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import entry12_response_basis_contract as basis_contract
import residual_state_mediation_executor as state_executor
import run_temporal_iswas_v15_construction_oracle_attention15_dependency_factorial_v1 as dependency
import run_temporal_iswas_v15_head_response_target_feasible_regularized_das_v1 as parent
from two_by_two_dependency_contract import vector_metrics

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v15_entry12_shared_target_control_response_basis_v1.json"
STATE_RESULT = ROOT / "circuits/followups/temporal_iswas_v15_residual_state_boundary_mediation_atlas_v1_result.json"
ORACLES = ROOT / "circuits/followups/temporal_iswas_v15_construction_oracle_projective_bisector_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_iswas_v15_entry12_shared_target_control_response_basis_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.iswas_v15_entry12_shared_target_control_response_basis_v1"
EXPECTED = {
    "prior": "ae4bc9caf7383df7ab6070c1fcd81f57a9647b2c85c9e76270116acad682fc39",
    "state_result": "b2c667ef8f795d85ca729ccd6b97210ff5754a787fa2426aaab517176d7df862",
    "oracles": "dde8cc793f2ba2d1f0bf7439dd9c62cb53ad979ea4ab3363928015b8cbe7e224",
    "basis_contract": "cc05ed73baf02c7ec6f0ec2b9e64a831a1fff6462ab4d8a5ba1f39bc5f70dab1",
    "state_executor": "ae3dc83c4a1954575778fa88744f964efc712e8bc483de078e366a90afee9720",
    "parent": "0b7d92038283cad0fd48e398739474c4004de0da8e361d5aaaf0f68a4db6595e",
    "v15": "7027e128ea3e68a35e92f451d0f62453a937e64a8dafda24d841aa1f1d29e645",
}
FILES = {
    "prior": PRIOR,
    "state_result": STATE_RESULT,
    "oracles": ORACLES,
    "basis_contract": ROOT / "ops/entry12_response_basis_contract.py",
    "state_executor": ROOT / "ops/residual_state_mediation_executor.py",
    "parent": ROOT / "ops/run_temporal_iswas_v15_head_response_target_feasible_regularized_das_v1.py",
    "v15": ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_v15_aligned_controls_v1.py",
}
EXPERTS = {"A1_oracle": "A1", "A2_oracle": "A2"}
BASES = ("shared_dim_rank1", "construction_union_rank_le_2",
         "p_complement_union_rank_le_2")
PRICE_MAX = {"native_capture_forwards": 10, "differentiable_transformer_forwards": 32,
             "transformer_backward_forwards": 0, "model_updates": 0,
             "example_evaluations": 3000, "fit_parameters": 0}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def panel_vector(context, output, background, panel):
    indices = context["panel_indices"][panel]
    logits, base_logits = output["logits"], background["logits"]
    response = ((logits[context["index"], context["answer"]]
                 - logits[context["index"], context["foil"]])
                - (base_logits[context["index"], context["answer"]]
                   - base_logits[context["index"], context["foil"]]))
    return response[indices].detach().cpu().tolist()


def pair_control(backend, context, changed, background, panel):
    F = backend.F
    indices = context["panel_indices"][panel]
    changed_lp = F.log_softmax(changed["logits"][indices], -1)
    background_lp = F.log_softmax(background["logits"][indices], -1)
    kl = (background_lp.exp() * (background_lp - changed_lp)).sum(-1)
    flips = changed["logits"][indices].argmax(-1) != background["logits"][indices].argmax(-1)
    return {"mean_kl": float(kl.mean()), "top1_flip_count": int(flips.sum())}


def stack_prefix(torch, delta, context, panel):
    indices = context["panel_indices"][panel].detach().cpu().tolist()
    pieces = [delta[index, :int(context["base_batch"].semantic_positions[index]) + 1].float()
              for index in indices]
    if not pieces:
        raise RuntimeError(f"empty panel {panel}")
    return torch.cat(pieces, dim=0)


def maxdiff(left, right):
    return float((left - right).abs().max())


def main():
    observed = {name: sha(path) for name, path in FILES.items()}
    prior = json.loads(PRIOR.read_text())
    state_result = json.loads(STATE_RESULT.read_text())
    authority = (observed == EXPECTED and prior.get("candidate_id") == CANDIDATE_ID
                 and state_result.get("terminal") == "nonselective_residual_lockin")
    dry = {"candidate_id": CANDIDATE_ID, "dryrun": True, "authority_ok": authority,
           "bases": list(BASES), "expected_differentiable_forwards": 32,
           "price_max": PRICE_MAX}
    if not authority:
        raise RuntimeError(f"authority changed: {observed}")
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(OUT)

    start = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    for parameter in backend.model.parameters():
        parameter.requires_grad_(False)
    counters = {name: 0 for name in PRICE_MAX}
    native = backend.native

    def counted(batch, *, capture):
        counters["native_capture_forwards"] += 1
        counters["example_evaluations"] += len(batch.row_ids)
        return native(batch, capture=capture)

    backend.native = counted
    rows = v15.build_rows()
    bank = parent.capture_bank(backend, rows, counters, factors=False)
    contexts = {parity: parent.attach_references(
        backend, parent.subset_context(bank, parent.row_indices(rows, parity=parity)), counters)
        for parity in (0, 1)}
    oracles = json.loads(ORACLES.read_text())
    captures = {expert: {} for expert in EXPERTS}
    off_cache = {}
    zero_replay = 0.0
    for parity, context in contexts.items():
        off_output, states = state_executor.execute(
            parent, backend, context, counters, {}, capture=True)
        off_cache[parity] = off_output, states["entry12"]
        zero = state_executor.execute(
            parent, backend, context, counters, {}, absolute={"entry12": states["entry12"]})
        zero_replay = max(zero_replay, maxdiff(zero["logits"], off_output["logits"]))
    on_self_replay = 0.0
    full_replay = 0.0
    for expert in EXPERTS:
        bases_by_parity = dependency.bases_for_evaluation(
            backend.torch, backend.device, oracles, expert)
        for parity, context in contexts.items():
            bases = bases_by_parity[parity]
            on_output, states = state_executor.execute(
                parent, backend, context, counters, bases, capture=True)
            on_state = states["entry12"]
            on_self = state_executor.execute(
                parent, backend, context, counters, bases, absolute={"entry12": on_state})
            on_self_replay = max(on_self_replay, maxdiff(on_self["logits"], on_output["logits"]))
            off_output, off_state = off_cache[parity]
            full_absolute = state_executor.hybrid_state(
                off_state, on_state, context["base_batch"].semantic_positions)
            full = state_executor.execute(
                parent, backend, context, counters, {}, absolute={"entry12": full_absolute})
            full_replay = max(full_replay, maxdiff(full["logits"], on_output["logits"]))
            captures[expert][parity] = {"bases": bases, "on_output": on_output,
                                        "on_state": on_state}

    fits = {}
    reports = {expert: {} for expert in EXPERTS}
    orthogonality = 0.0
    for held_parity in (0, 1):
        train_parity = 1 - held_parity
        context = contexts[train_parity]
        off_state = off_cache[train_parity][1]
        deltas = {expert: captures[expert][train_parity]["on_state"].float()
                  - off_state.float() for expert in EXPERTS}
        target_a1 = stack_prefix(backend.torch, deltas["A1_oracle"], context, "A1").mean(0)
        target_a2 = stack_prefix(backend.torch, deltas["A2_oracle"], context, "A2").mean(0)
        p_rows = backend.torch.cat([
            stack_prefix(backend.torch, deltas[expert], context, "P") for expert in EXPERTS], dim=0)
        bases, geometry = basis_contract.fit_bases(
            backend.torch, target_a1, target_a2, p_rows, relative_threshold=1e-6)
        geometry["ranks"] = {name: int(basis.shape[1]) for name, basis in bases.items()}
        for basis in bases.values():
            if basis.shape[1]:
                identity = backend.torch.eye(basis.shape[1], device=basis.device)
                orthogonality = max(orthogonality, float((basis.T @ basis - identity).abs().max()))
        fits[str(held_parity)] = geometry
        for expert, own_panel in EXPERTS.items():
            held_context = contexts[held_parity]
            off_output, off_state = off_cache[held_parity]
            on_output = captures[expert][held_parity]["on_output"]
            on_state = captures[expert][held_parity]["on_state"]
            full_targets = {panel: panel_vector(held_context, on_output, off_output, panel)
                            for panel in ("A1", "A2", "P", "C")}
            expert_report = {}
            for name, basis in bases.items():
                absolute = basis_contract.projected_absolute(
                    backend.torch, off_state, on_state, basis,
                    held_context["base_batch"].semantic_positions)
                output = state_executor.execute(
                    parent, backend, held_context, counters, {}, absolute={"entry12": absolute})
                item = {}
                for panel in ("A1", "A2", "P", "C"):
                    response = panel_vector(held_context, output, off_output, panel)
                    item[panel] = {"response": response,
                                   "metrics": vector_metrics(response, full_targets[panel])}
                item["controls"] = {panel: pair_control(
                    backend, held_context, output, off_output, panel) for panel in ("P", "C")}
                expert_report[name] = item
            reports[expert][str(held_parity)] = expert_report

    counters["fit_parameters"] = 0
    native_closure = max(context["manual_native_max_abs_error"] for context in contexts.values())
    row_coverage = all(len(context["panel_indices"][panel]) == 8
                       for context in contexts.values() for panel in ("A1", "A2", "P", "C"))
    fit_disjoint = all(set(contexts[p]["base_batch"].row_ids).isdisjoint(
        contexts[1 - p]["base_batch"].row_ids) for p in (0, 1))
    A = (authority and row_coverage and fit_disjoint and native_closure <= 1e-4
         and zero_replay <= 1e-4 and on_self_replay <= 1e-4 and full_replay <= 1e-4
         and orthogonality <= 1e-5 and finite({"fits": fits, "reports": reports})
         and counters["differentiable_transformer_forwards"] == 32
         and all(counters[name] <= PRICE_MAX[name] for name in PRICE_MAX))
    metric = lambda expert, parity, basis: reports[expert][str(parity)][basis][EXPERTS[expert]]["metrics"]
    B = all(metric(expert, parity, "construction_union_rank_le_2")["signed_projection"] >= .75
            and metric(expert, parity, "construction_union_rank_le_2")["direction_fraction"] >= .875
            for expert in EXPERTS for parity in (0, 1))
    C = all(metric(expert, parity, "p_complement_union_rank_le_2")["signed_projection"] >= .675
            and metric(expert, parity, "p_complement_union_rank_le_2")["signed_projection"]
                >= .90 * metric(expert, parity, "construction_union_rank_le_2")["signed_projection"]
            for expert in EXPERTS for parity in (0, 1))
    D = all(reports[expert][str(parity)]["p_complement_union_rank_le_2"]["controls"][panel]["top1_flip_count"] == 0
            and reports[expert][str(parity)]["p_complement_union_rank_le_2"]["controls"][panel]["mean_kl"] <= .02
            for expert in EXPERTS for parity in (0, 1) for panel in ("P", "C"))
    E = all(metric(expert, parity, "shared_dim_rank1")["signed_projection"] >= .75
            and metric(expert, parity, "shared_dim_rank1")["direction_fraction"] >= .875
            for expert in EXPERTS for parity in (0, 1))
    predictions = {
        "pred_a_authority_replay_scope_geometry_finiteness_and_price": A,
        "pred_b_construction_union_is_target_sufficient": B,
        "pred_c_p_complement_preserves_target": C,
        "pred_d_p_complement_removes_collision": D,
        "pred_e_shared_rank1_is_construction_general": E,
    }
    terminal = ("invalid" if not A else "no_shared_linear_state" if not B
                else "p_complement_destroys_target" if not C
                else "p_collateral_outside_training_span" if not D
                else "selective_shared_rank1_state" if E else "selective_entry12_union")
    result = {
        "schema": "temporal_iswas_v15_entry12_shared_target_control_response_basis_result_v1",
        "candidate_id": CANDIDATE_ID,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "serial_seconds": time.perf_counter() - start,
        "authority_sha256": EXPECTED,
        "fits": fits,
        "reports": reports,
        "instrument": {"manual_native_max_abs_error": native_closure,
                       "zero_replay_max_abs_error": zero_replay,
                       "on_self_replay_max_abs_error": on_self_replay,
                       "full_state_replay_max_abs_error": full_replay,
                       "orthogonality_max_abs_error": orthogonality,
                       "row_coverage_ok": row_coverage, "train_held_disjoint": fit_disjoint},
        "predictions": predictions,
        "terminal": terminal,
        "price": {**counters, "maxima": PRICE_MAX},
    }
    atomic_create_json(OUT, result)
    print(json.dumps({"predictions": predictions, "terminal": terminal,
                      "instrument": result["instrument"], "price": result["price"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
