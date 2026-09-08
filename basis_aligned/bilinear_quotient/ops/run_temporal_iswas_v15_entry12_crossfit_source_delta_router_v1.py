#!/usr/bin/env python3
"""Held-parity source-delta A1/A2/off router for the entry12 construction union."""

# BQGATE: EXPERIMENT pred_a_authority_replay_scope_geometry_finiteness_and_price pred_b_source_router_identifies_held_branches pred_c_routed_union_preserves_both_targets pred_d_routed_union_is_control_selective pred_e_router_retains_ungated_union_effect
from __future__ import annotations

from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

import circuit_candidate_tense_auxiliary_is_was_v15_aligned_controls_v1 as v15
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import entry12_finite_router_contract as router
import entry12_response_basis_contract as basis_contract
import residual_state_mediation_executor as state_executor
import run_temporal_iswas_v15_construction_oracle_attention15_dependency_factorial_v1 as dependency
import run_temporal_iswas_v15_head_response_target_feasible_regularized_das_v1 as parent
from two_by_two_dependency_contract import vector_metrics

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v15_entry12_crossfit_source_delta_router_v1.json"
NATIVE_ROUTER_RESULT = ROOT / "circuits/followups/temporal_iswas_v15_entry12_crossfit_finite_router_v1_result.json"
ORACLES = ROOT / "circuits/followups/temporal_iswas_v15_construction_oracle_projective_bisector_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_iswas_v15_entry12_crossfit_source_delta_router_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.iswas_v15_entry12_crossfit_source_delta_router_v1"
EXPECTED = {
    "prior": "a1775faff7d0fb1f44851357c5cdd3206d734aee884f3a73bff0dc79305693bf",
    "native_router_result": "a449e3fa784d733775e2d6f0e9ad371ae2cbffbcaeca5b1c35047a909674d697",
    "oracles": "dde8cc793f2ba2d1f0bf7439dd9c62cb53ad979ea4ab3363928015b8cbe7e224",
    "router": "48007c4950bd9116601afbfe66e5e4a6d9bb7dbef65f90afc2dd405714809222",
    "basis_contract": "cc05ed73baf02c7ec6f0ec2b9e64a831a1fff6462ab4d8a5ba1f39bc5f70dab1",
    "state_executor": "ae3dc83c4a1954575778fa88744f964efc712e8bc483de078e366a90afee9720",
    "parent": "0b7d92038283cad0fd48e398739474c4004de0da8e361d5aaaf0f68a4db6595e",
    "v15": "7027e128ea3e68a35e92f451d0f62453a937e64a8dafda24d841aa1f1d29e645",
}
FILES = {
    "prior": PRIOR, "native_router_result": NATIVE_ROUTER_RESULT, "oracles": ORACLES,
    "router": ROOT / "ops/entry12_finite_router_contract.py",
    "basis_contract": ROOT / "ops/entry12_response_basis_contract.py",
    "state_executor": ROOT / "ops/residual_state_mediation_executor.py",
    "parent": ROOT / "ops/run_temporal_iswas_v15_head_response_target_feasible_regularized_das_v1.py",
    "v15": ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_v15_aligned_controls_v1.py",
}
EXPERTS = {"A1_oracle": "A1", "A2_oracle": "A2"}
PRICE_MAX = {"native_capture_forwards": 10, "differentiable_transformer_forwards": 24,
             "transformer_backward_forwards": 0, "model_updates": 0,
             "example_evaluations": 2000, "fit_parameters": 0}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict): return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)): return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def maxdiff(left, right): return float((left - right).abs().max())


def stack_prefix(torch, delta, context, panel):
    indices = context["panel_indices"][panel].detach().cpu().tolist()
    return torch.cat([delta[index, :int(context["base_batch"].semantic_positions[index]) + 1].float()
                      for index in indices], dim=0)


def panel_vector(context, output, background, panel):
    indices = context["panel_indices"][panel]
    def margins(item):
        return (item["logits"][context["index"], context["answer"]]
                - item["logits"][context["index"], context["foil"]])
    return (margins(output) - margins(background))[indices].detach().cpu().tolist()


def pair_control(backend, context, changed, background, panel):
    F, indices = backend.F, context["panel_indices"][panel]
    lp = F.log_softmax(changed["logits"][indices], -1)
    lq = F.log_softmax(background["logits"][indices], -1)
    kl = (lq.exp() * (lq - lp)).sum(-1)
    flips = changed["logits"][indices].argmax(-1) != background["logits"][indices].argmax(-1)
    return {"mean_kl": float(kl.mean()), "top1_flip_count": int(flips.sum())}


def program_report(backend, context, output, off_output, full_targets):
    result = {}
    for panel in ("A1", "A2"):
        response = panel_vector(context, output, off_output, panel)
        result[panel] = {"response": response, "metrics": vector_metrics(response, full_targets[panel])}
    result["controls"] = {panel: pair_control(backend, context, output, off_output, panel)
                           for panel in ("P", "C")}
    return result


def main():
    observed = {name: sha(path) for name, path in FILES.items()}
    prior, native_router_result = json.loads(PRIOR.read_text()), json.loads(NATIVE_ROUTER_RESULT.read_text())
    authority = (observed == EXPECTED and prior.get("candidate_id") == CANDIDATE_ID
                 and native_router_result.get("terminal") == "router_identification_failure")
    dry = {"candidate_id": CANDIDATE_ID, "dryrun": True, "authority_ok": authority,
           "router_classes": list(router.CLASSES), "expected_differentiable_forwards": 24,
           "price_max": PRICE_MAX}
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
    rows = v15.build_rows()
    base_text_impossibility = all(
        next(row["base_text"] for row in rows if row["group_number"] == group and row["transform_id"] == "A1")
        == next(row["base_text"] for row in rows if row["group_number"] == group and row["transform_id"] == "P")
        for group in range(16))
    bank = parent.capture_bank(backend, rows, counters, factors=False)
    contexts = {parity: parent.attach_references(
        backend, parent.subset_context(bank, parent.row_indices(rows, parity=parity)), counters)
        for parity in (0, 1)}
    off_cache = {}; parent_replay = zero_replay = full_replay = 0.0
    for parity, context in contexts.items():
        output, states = state_executor.execute(parent, backend, context, counters, {}, capture=True)
        entry = states["entry12"]; off_cache[parity] = output, entry
        parent_replay = max(parent_replay, maxdiff(output["logits"], context["base"]["logits"]))
        zero = state_executor.execute(parent, backend, context, counters, {}, absolute={"entry12": entry})
        zero_replay = max(zero_replay, maxdiff(zero["logits"], output["logits"]))
    oracles = json.loads(ORACLES.read_text()); on_cache = {panel: {} for panel in ("A1", "A2")}
    for expert, panel in EXPERTS.items():
        bases_by_parity = dependency.bases_for_evaluation(backend.torch, backend.device, oracles, expert)
        for parity, context in contexts.items():
            output, states = state_executor.execute(
                parent, backend, context, counters, bases_by_parity[parity], capture=True)
            entry = states["entry12"]; on_cache[panel][parity] = output, entry
            off_output, off_entry = off_cache[parity]
            absolute = state_executor.hybrid_state(
                off_entry, entry, context["base_batch"].semantic_positions)
            replay = state_executor.execute(parent, backend, context, counters, {},
                                            absolute={"entry12": absolute})
            full_replay = max(full_replay, maxdiff(replay["logits"], output["logits"]))

    fits, reports = {}, {}
    orthogonality = 0.0
    for held in (0, 1):
        train = 1 - held; train_context = contexts[train]
        train_off = off_cache[train][1]
        deltas = {panel: on_cache[panel][train][1].float() - train_off.float()
                  for panel in ("A1", "A2")}
        target_a1 = stack_prefix(backend.torch, deltas["A1"], train_context, "A1").mean(0)
        target_a2 = stack_prefix(backend.torch, deltas["A2"], train_context, "A2").mean(0)
        p_rows = backend.torch.cat([stack_prefix(backend.torch, deltas[panel], train_context, "P")
                                    for panel in ("A1", "A2")], dim=0)
        bases, basis_geometry = basis_contract.fit_bases(
            backend.torch, target_a1, target_a2, p_rows, relative_threshold=1e-6)
        union = bases["construction_union_rank_le_2"]
        identity = backend.torch.eye(union.shape[1], device=union.device)
        orthogonality = max(orthogonality, float((union.T @ union - identity).abs().max()))
        train_features = router.source_delta_features(
            backend.torch, train_off,
            {panel: on_cache[panel][train][1] for panel in ("A1", "A2")},
            train_context["base_batch"].semantic_positions)
        train_labels = router.labels_for_rows(backend.torch, train_context["rows"], device=backend.device)
        router_fit = router.fit_centroids(backend.torch, train_features, train_labels)
        held_context = contexts[held]; off_output, off_entry = off_cache[held]
        held_features = router.source_delta_features(
            backend.torch, off_entry,
            {panel: on_cache[panel][held][1] for panel in ("A1", "A2")},
            held_context["base_batch"].semantic_positions)
        truth = router.labels_for_rows(backend.torch, held_context["rows"], device=backend.device)
        predicted, scores = router.predict(backend.torch, held_features, router_fit)
        classification = router.classification_report(backend.torch, truth, predicted)
        fits[str(held)] = {"basis": {"rank": int(union.shape[1]), **basis_geometry},
                           "router": {**classification, "predictions": predicted.cpu().tolist(),
                                      "truth": truth.cpu().tolist(), "scores": scores.cpu().tolist()}}
        expert_states = {panel: on_cache[panel][held][1] for panel in ("A1", "A2")}
        full_targets = {panel: panel_vector(
            held_context, on_cache[panel][held][0], off_output, panel) for panel in ("A1", "A2")}
        ungated = {}
        for panel in ("A1", "A2"):
            absolute = basis_contract.projected_absolute(
                backend.torch, off_entry, expert_states[panel], union,
                held_context["base_batch"].semantic_positions)
            output = state_executor.execute(parent, backend, held_context, counters, {},
                                            absolute={"entry12": absolute})
            ungated[panel] = program_report(backend, held_context, output, off_output, full_targets)
        learned_absolute = router.routed_absolute(
            backend.torch, off_entry, expert_states, union, predicted,
            held_context["base_batch"].semantic_positions)
        gold_absolute = router.routed_absolute(
            backend.torch, off_entry, expert_states, union, truth,
            held_context["base_batch"].semantic_positions)
        learned = state_executor.execute(parent, backend, held_context, counters, {},
                                         absolute={"entry12": learned_absolute})
        gold = state_executor.execute(parent, backend, held_context, counters, {},
                                      absolute={"entry12": gold_absolute})
        reports[str(held)] = {"ungated": ungated,
                              "learned_router": program_report(
                                  backend, held_context, learned, off_output, full_targets),
                              "gold_router": program_report(
                                  backend, held_context, gold, off_output, full_targets)}

    counters["fit_parameters"] = 0
    native_closure = max(context["manual_native_max_abs_error"] for context in contexts.values())
    disjoint = set(contexts[0]["base_batch"].row_ids).isdisjoint(contexts[1]["base_batch"].row_ids)
    A = (authority and base_text_impossibility and disjoint
         and native_closure <= 1e-4 and parent_replay <= 1e-4
         and zero_replay <= 1e-4 and full_replay <= 1e-4 and orthogonality <= 1e-5
         and finite({"fits": fits, "reports": reports})
         and counters["differentiable_transformer_forwards"] == 24
         and all(counters[name] <= PRICE_MAX[name] for name in PRICE_MAX))
    B = all(fits[str(parity)]["router"]["macro_accuracy"] >= .875
            and fits[str(parity)]["router"]["control_predicted_target_count"] == 0 for parity in (0, 1))
    C = all(reports[str(parity)]["learned_router"][panel]["metrics"]["signed_projection"] >= .75
            and reports[str(parity)]["learned_router"][panel]["metrics"]["direction_fraction"] >= .875
            for parity in (0, 1) for panel in ("A1", "A2"))
    D = all(reports[str(parity)]["learned_router"]["controls"][panel]["top1_flip_count"] == 0
            and reports[str(parity)]["learned_router"]["controls"][panel]["mean_kl"] <= .02
            for parity in (0, 1) for panel in ("P", "C"))
    E = all(reports[str(parity)]["learned_router"][panel]["metrics"]["signed_projection"]
            >= .90 * reports[str(parity)]["ungated"][panel][panel]["metrics"]["signed_projection"]
            for parity in (0, 1) for panel in ("A1", "A2"))
    predictions = {
        "pred_a_authority_replay_scope_geometry_finiteness_and_price": A,
        "pred_b_source_router_identifies_held_branches": B,
        "pred_c_routed_union_preserves_both_targets": C,
        "pred_d_routed_union_is_control_selective": D,
        "pred_e_router_retains_ungated_union_effect": E,
    }
    gold_pass = all(reports[str(parity)]["gold_router"][panel]["metrics"]["signed_projection"] >= .75
                    for parity in (0, 1) for panel in ("A1", "A2"))
    terminal = ("invalid" if not A else "source_delta_router_candidate" if all((B, C, D, E))
                else "router_identification_failure" if gold_pass and not B
                else "routed_union_execution_failure" if B and not C
                else "routed_control_leakage" if C and not D else "finite_router_null")
    result = {"schema": "temporal_iswas_v15_entry12_crossfit_source_delta_router_result_v1",
              "candidate_id": CANDIDATE_ID, "started_utc": datetime.now(timezone.utc).isoformat(),
              "serial_seconds": time.perf_counter() - start, "authority_sha256": EXPECTED,
              "fits": fits, "reports": reports, "gold_router_target_pass": gold_pass,
              "instrument": {"manual_native_max_abs_error": native_closure,
                             "parent_off_replay_max_abs_error": parent_replay,
                             "zero_replay_max_abs_error": zero_replay,
                             "full_state_replay_max_abs_error": full_replay,
                             "orthogonality_max_abs_error": orthogonality,
                             "train_held_disjoint": disjoint,
                             "paired_a1_p_base_texts_identical": base_text_impossibility},
              "predictions": predictions, "terminal": terminal,
              "price": {**counters, "maxima": PRICE_MAX}}
    atomic_create_json(OUT, result)
    print(json.dumps({"predictions": predictions, "terminal": terminal,
                      "gold_router_target_pass": gold_pass, "instrument": result["instrument"],
                      "price": result["price"]}, sort_keys=True))


if __name__ == "__main__": main()
