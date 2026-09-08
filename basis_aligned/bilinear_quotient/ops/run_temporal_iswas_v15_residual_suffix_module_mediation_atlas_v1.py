#!/usr/bin/env python3
"""Complete layer-12--17 module-write mediation atlas for the v15 oracle route."""

# BQGATE: EXPERIMENT pred_a_authority_replay_closure_finiteness_and_price pred_b_joint_suffix_write_bank_mediates_bypass pred_c_stable_singleton_module_mediator_exists pred_d_singleton_module_effects_compose pred_e_selected_module_mediation_is_control_selective
from __future__ import annotations

from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

import circuit_candidate_tense_auxiliary_is_was_v15_aligned_controls_v1 as v15
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import head_response_mediation_scorer as scorer
import module_write_mediation_contract as mediation
import run_temporal_iswas_v15_construction_oracle_attention15_dependency_factorial_v1 as dependency
import run_temporal_iswas_v15_head_response_target_feasible_regularized_das_v1 as parent

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v15_residual_suffix_module_mediation_atlas_v1.json"
ATTENTION_ATLAS = ROOT / "circuits/followups/temporal_iswas_v15_attention15_head_module_mediation_atlas_v1_result.json"
ORACLES = ROOT / "circuits/followups/temporal_iswas_v15_construction_oracle_projective_bisector_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_iswas_v15_residual_suffix_module_mediation_atlas_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.iswas_v15_residual_suffix_module_mediation_atlas_v1"
EXPECTED = {
    "prior": "97b9702b286070639310c7f0db6e2ba77510d2f69637934dce67890838559baa",
    "attention_atlas": "006fdb1871ee188aa9b3500cc09af0b3e6cb418c898003a8c08bb7d36b4dd3b8",
    "oracles": "dde8cc793f2ba2d1f0bf7439dd9c62cb53ad979ea4ab3363928015b8cbe7e224",
    "mediation": "43181f1f69512899bd3f13ea159bceb7bcef6f892246b879d1c42d830c5f39c3",
    "scorer": "9d0b56bd68a75e939d6cd0cc900de82d905d4d7131b56261d17759e83a28ba2f",
    "parent": "0b7d92038283cad0fd48e398739474c4004de0da8e361d5aaaf0f68a4db6595e",
    "v15": "7027e128ea3e68a35e92f451d0f62453a937e64a8dafda24d841aa1f1d29e645",
}
FILES = {
    "prior": PRIOR,
    "attention_atlas": ATTENTION_ATLAS,
    "oracles": ORACLES,
    "mediation": ROOT / "ops/module_write_mediation_contract.py",
    "scorer": ROOT / "ops/head_response_mediation_scorer.py",
    "parent": ROOT / "ops/run_temporal_iswas_v15_head_response_target_feasible_regularized_das_v1.py",
    "v15": ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_v15_aligned_controls_v1.py",
}
EXPERTS = {"A1_oracle": "A1", "A2_oracle": "A2"}
SINGLETONS = tuple(f"{kind}{layer}" for layer in range(12, 18) for kind in ("attn", "mlp"))
MEDIATORS = ("joint",) + SINGLETONS
PRICE_MAX = {"native_capture_forwards": 10, "differentiable_transformer_forwards": 120,
             "transformer_backward_forwards": 0, "model_updates": 0,
             "example_evaluations": 10000, "fit_parameters": 0}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def module_sites(model):
    sites = {}
    for layer in range(12, 18):
        block = model.transformer.h[layer]
        sites[f"attn{layer}"] = (block.attn, "attention")
        sites[f"mlp{layer}"] = (block.mlp, "mlp")
    return sites


def run_parent(backend, context, counters, bases):
    return parent.manual_forward(backend, context["base_batch"], counters,
                                 context=context, raw_by_site=bases, complete15=False)


def execute_cells(backend, context, counters, sites, bases, off_output, off_writes,
                  on_output, on_writes, names):
    selected = {name: sites[name] for name in names}
    positions = context["base_batch"].semantic_positions
    rescue = mediation.hybrid_bank(off_writes, on_writes, names, positions)
    reset = mediation.hybrid_bank(on_writes, off_writes, names, positions)
    return {
        "00": off_output,
        "01": mediation.execute_with_writes(
            selected, rescue, lambda: run_parent(backend, context, counters, {})),
        "10": mediation.execute_with_writes(
            selected, reset, lambda: run_parent(backend, context, counters, bases)),
        "11": on_output,
    }


def margin(context, output, panel):
    logits = output["logits"]
    raw = logits[context["index"], context["answer"]] - logits[context["index"], context["foil"]]
    indices = context["panel_indices"][panel]
    return (raw[indices] - context["base_margin"][indices]).detach().cpu().tolist()


def maxdiff(left, right):
    return float((left - right).abs().max())


def pair_control(backend, context, changed, background, panel):
    F = backend.F
    indices = context["panel_indices"][panel]
    changed_lp = F.log_softmax(changed["logits"][indices], -1)
    background_lp = F.log_softmax(background["logits"][indices], -1)
    kl = (background_lp.exp() * (background_lp - changed_lp)).sum(-1)
    flips = changed["logits"][indices].argmax(-1) != background["logits"][indices].argmax(-1)
    return {"mean_kl": float(kl.mean()), "top1_flip_count": int(flips.sum())}


def main():
    observed = {name: sha(path) for name, path in FILES.items()}
    prior = json.loads(PRIOR.read_text())
    attention = json.loads(ATTENTION_ATLAS.read_text())
    authority = (observed == EXPECTED and prior.get("candidate_id") == CANDIDATE_ID
                 and attention.get("terminal") == "attention15_bypass")
    dry = {"candidate_id": CANDIDATE_ID, "dryrun": True, "authority_ok": authority,
           "mediators": list(MEDIATORS), "expected_differentiable_forwards": 120,
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
    sites = module_sites(backend.model)
    reports = {}
    self_replay = 0.0
    closure = 0.0

    # Upstream-off executions are expert-independent and are shared prospectively.
    off_cache = {}
    for parity, context in contexts.items():
        off_output, off_writes = mediation.capture_writes(
            sites, lambda context=context: run_parent(backend, context, counters, {}))
        off_replay = mediation.execute_with_writes(
            sites, off_writes, lambda context=context: run_parent(backend, context, counters, {}))
        self_replay = max(self_replay, maxdiff(off_replay["logits"], off_output["logits"]))
        off_cache[parity] = (off_output, off_writes)

    oracles = json.loads(ORACLES.read_text())
    for expert, own_panel in EXPERTS.items():
        bases_by_parity = dependency.bases_for_evaluation(
            backend.torch, backend.device, oracles, expert)
        reports[expert] = {}
        for parity, context in contexts.items():
            bases = bases_by_parity[parity]
            off_output, off_writes = off_cache[parity]
            on_output, on_writes = mediation.capture_writes(
                sites, lambda context=context, bases=bases: run_parent(
                    backend, context, counters, bases))
            on_replay = mediation.execute_with_writes(
                sites, on_writes, lambda context=context, bases=bases: run_parent(
                    backend, context, counters, bases))
            self_replay = max(self_replay, maxdiff(on_replay["logits"], on_output["logits"]))
            mediator_reports = {}
            for mediator in MEDIATORS:
                names = SINGLETONS if mediator == "joint" else (mediator,)
                cells = execute_cells(backend, context, counters, sites, bases,
                                      off_output, off_writes, on_output, on_writes, names)
                item = {}
                for panel in ("A1", "A2", "P", "C"):
                    vectors = {cell: margin(context, output, panel)
                               for cell, output in cells.items()}
                    target = [on - off for on, off in zip(vectors["11"], vectors["00"])]
                    item[panel] = scorer.score_cells(vectors, target)
                    closure = max(closure, item[panel]["closure_max_abs_error"])
                item["controls"] = {panel: {
                    "rescue": pair_control(backend, context, cells["01"], cells["00"], panel),
                    "reset": pair_control(backend, context, cells["10"], cells["11"], panel),
                } for panel in ("P", "C")}
                mediator_reports[mediator] = item
            own_singletons = {name: mediator_reports[name][own_panel] for name in SINGLETONS}
            mediator_reports["rank_reset"] = scorer.deterministic_head_ranking(
                own_singletons, "head_reset_loss")
            mediator_reports["rank_rescue"] = scorer.deterministic_head_ranking(
                own_singletons, "head_rescue")
            mediator_reports["composition"] = scorer.singleton_module_composition(
                own_singletons, mediator_reports["joint"][own_panel])
            reports[expert][str(parity)] = mediator_reports

    counters["fit_parameters"] = 0
    own = lambda expert, parity, mediator: reports[expert][str(parity)][mediator][EXPERTS[expert]]
    native_closure = max(context["manual_native_max_abs_error"] for context in contexts.values())
    row_coverage = all(len(context["panel_indices"][panel]) == 8
                       for context in contexts.values() for panel in ("A1", "A2", "P", "C"))
    A = (authority and row_coverage and native_closure <= 1e-4 and self_replay <= 1e-4
         and closure <= 1e-12 and finite(reports)
         and all(counters[name] <= PRICE_MAX[name] for name in PRICE_MAX)
         and counters["differentiable_transformer_forwards"] == 120)
    B = all(own(expert, parity, "joint")["metrics"][component]["signed_projection"] >= .50
            and own(expert, parity, "joint")["metrics"][component]["direction_fraction"] >= .875
            for expert in EXPERTS for parity in (0, 1)
            for component in ("head_reset_loss", "head_rescue"))
    stable_singletons = [name for name in SINGLETONS if all(
        own(expert, parity, name)["metrics"]["head_reset_loss"]["signed_projection"] >= .10
        and own(expert, parity, name)["metrics"]["head_reset_loss"]["direction_fraction"] >= .75
        for expert in EXPERTS for parity in (0, 1))]
    C = bool(stable_singletons)
    D = all(reports[expert][str(parity)]["composition"]["metrics"]["cosine"] >= .90
            and reports[expert][str(parity)]["composition"]["metrics"]["relative_l2_error"] <= .25
            for expert in EXPERTS for parity in (0, 1))
    E = bool(stable_singletons) and all(
        reports[expert][str(parity)][name]["controls"][panel][arm]["top1_flip_count"] == 0
        and reports[expert][str(parity)][name]["controls"][panel][arm]["mean_kl"] <= .02
        for name in stable_singletons for expert in EXPERTS for parity in (0, 1)
        for panel in ("P", "C") for arm in ("rescue", "reset"))
    predictions = {
        "pred_a_authority_replay_closure_finiteness_and_price": A,
        "pred_b_joint_suffix_write_bank_mediates_bypass": B,
        "pred_c_stable_singleton_module_mediator_exists": C,
        "pred_d_singleton_module_effects_compose": D,
        "pred_e_selected_module_mediation_is_control_selective": E,
    }
    terminal = ("invalid" if not A else "direct_residual_carry" if not B
                else "distributed_write_mediation" if not C
                else "interaction_conditioned_module_mediation" if not D
                else "nonselective_module_mediation" if not E
                else "greedy_module_union_licensed")
    result = {
        "schema": "temporal_iswas_v15_residual_suffix_module_mediation_atlas_result_v1",
        "candidate_id": CANDIDATE_ID,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "serial_seconds": time.perf_counter() - start,
        "authority_sha256": EXPECTED,
        "reports": reports,
        "stable_singleton_modules": stable_singletons,
        "instrument": {"manual_native_max_abs_error": native_closure,
                       "self_clamp_replay_max_abs_error": self_replay,
                       "factorial_closure_max_abs_error": closure,
                       "row_coverage_ok": row_coverage},
        "predictions": predictions,
        "terminal": terminal,
        "price": {**counters, "maxima": PRICE_MAX},
    }
    atomic_create_json(OUT, result)
    print(json.dumps({"predictions": predictions, "terminal": terminal,
                      "stable_singleton_modules": stable_singletons,
                      "instrument": result["instrument"], "price": result["price"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
