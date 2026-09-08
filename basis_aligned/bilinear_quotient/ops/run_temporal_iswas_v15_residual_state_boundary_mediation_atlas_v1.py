#!/usr/bin/env python3
"""Ordered complete-residual boundary mediation atlas for fixed v15 oracle routes."""

# BQGATE: EXPERIMENT pred_a_authority_replay_closure_finiteness_and_price pred_b_entry12_residual_state_carries_upstream_effect pred_c_final_residual_state_closes_logit_route pred_d_stable_preterminal_lockin_boundary_exists pred_e_lockin_boundary_is_control_selective
from __future__ import annotations

from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

import circuit_candidate_tense_auxiliary_is_was_v15_aligned_controls_v1 as v15
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import head_response_mediation_scorer as scorer
import residual_state_mediation_executor as executor
import run_temporal_iswas_v15_construction_oracle_attention15_dependency_factorial_v1 as dependency
import run_temporal_iswas_v15_head_response_target_feasible_regularized_das_v1 as parent

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v15_residual_state_boundary_mediation_atlas_v1.json"
SUFFIX = ROOT / "circuits/followups/temporal_iswas_v15_residual_suffix_module_mediation_atlas_v1_result.json"
ORACLES = ROOT / "circuits/followups/temporal_iswas_v15_construction_oracle_projective_bisector_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_iswas_v15_residual_state_boundary_mediation_atlas_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.iswas_v15_residual_state_boundary_mediation_atlas_v1"
EXPECTED = {
    "prior": "db7beef501930a04367836a4faac1d4a9d1ed9d356f83c911e24bf293dd38d1b",
    "suffix": "a8d990bc3aeec08193f8cd90ec64c6581e15927308c24fa6d6e72f879c248936",
    "oracles": "dde8cc793f2ba2d1f0bf7439dd9c62cb53ad979ea4ab3363928015b8cbe7e224",
    "executor": "ae3dc83c4a1954575778fa88744f964efc712e8bc483de078e366a90afee9720",
    "scorer": "9d0b56bd68a75e939d6cd0cc900de82d905d4d7131b56261d17759e83a28ba2f",
    "parent": "0b7d92038283cad0fd48e398739474c4004de0da8e361d5aaaf0f68a4db6595e",
    "v15": "7027e128ea3e68a35e92f451d0f62453a937e64a8dafda24d841aa1f1d29e645",
}
FILES = {
    "prior": PRIOR,
    "suffix": SUFFIX,
    "oracles": ORACLES,
    "executor": ROOT / "ops/residual_state_mediation_executor.py",
    "scorer": ROOT / "ops/head_response_mediation_scorer.py",
    "parent": ROOT / "ops/run_temporal_iswas_v15_head_response_target_feasible_regularized_das_v1.py",
    "v15": ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_v15_aligned_controls_v1.py",
}
EXPERTS = {"A1_oracle": "A1", "A2_oracle": "A2"}
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


def boundary_passes(reports, boundary, projection=.75, direction=.875):
    return all(
        reports[expert][str(parity)][boundary][EXPERTS[expert]]["metrics"][component]["signed_projection"] >= projection
        and reports[expert][str(parity)][boundary][EXPERTS[expert]]["metrics"][component]["direction_fraction"] >= direction
        for expert in EXPERTS for parity in (0, 1)
        for component in ("head_reset_loss", "head_rescue"))


def lockin_boundary(reports):
    candidates = executor.BOUNDARIES[:executor.BOUNDARIES.index("post_mlp16") + 1]
    for index, boundary in enumerate(candidates):
        suffix = executor.BOUNDARIES[executor.BOUNDARIES.index(boundary):]
        if all(boundary_passes(reports, later) for later in suffix):
            return boundary
    return None


def main():
    observed = {name: sha(path) for name, path in FILES.items()}
    prior = json.loads(PRIOR.read_text())
    suffix = json.loads(SUFFIX.read_text())
    authority = (observed == EXPECTED and prior.get("candidate_id") == CANDIDATE_ID
                 and suffix.get("terminal") == "direct_residual_carry")
    dry = {"candidate_id": CANDIDATE_ID, "dryrun": True, "authority_ok": authority,
           "boundaries": list(executor.BOUNDARIES),
           "expected_differentiable_forwards": 120, "price_max": PRICE_MAX}
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
    reports = {}
    parent_replay = 0.0
    self_replay = 0.0
    closure = 0.0
    off_cache = {}
    for parity, context in contexts.items():
        off_output, off_states = executor.execute(
            parent, backend, context, counters, {}, capture=True)
        parent_replay = max(parent_replay, maxdiff(off_output["logits"], context["base"]["logits"]))
        replay = executor.execute(parent, backend, context, counters, {}, absolute=off_states)
        self_replay = max(self_replay, maxdiff(replay["logits"], off_output["logits"]))
        off_cache[parity] = off_output, off_states

    oracles = json.loads(ORACLES.read_text())
    for expert, own_panel in EXPERTS.items():
        bases_by_parity = dependency.bases_for_evaluation(
            backend.torch, backend.device, oracles, expert)
        reports[expert] = {}
        for parity, context in contexts.items():
            bases = bases_by_parity[parity]
            off_output, off_states = off_cache[parity]
            on_output, on_states = executor.execute(
                parent, backend, context, counters, bases, capture=True)
            replay = executor.execute(
                parent, backend, context, counters, bases, absolute=on_states)
            self_replay = max(self_replay, maxdiff(replay["logits"], on_output["logits"]))
            boundary_reports = {}
            for boundary in executor.BOUNDARIES:
                cells = executor.boundary_cells(
                    parent, backend, context, counters, bases, off_output, off_states,
                    on_output, on_states, boundary)
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
                boundary_reports[boundary] = item
            reports[expert][str(parity)] = boundary_reports

    counters["fit_parameters"] = 0
    own = lambda expert, parity, boundary: reports[expert][str(parity)][boundary][EXPERTS[expert]]
    native_closure = max(context["manual_native_max_abs_error"] for context in contexts.values())
    row_coverage = all(len(context["panel_indices"][panel]) == 8
                       for context in contexts.values() for panel in ("A1", "A2", "P", "C"))
    A = (authority and row_coverage and native_closure <= 1e-4 and parent_replay <= 1e-4
         and self_replay <= 1e-4 and closure <= 1e-12 and finite(reports)
         and counters["differentiable_transformer_forwards"] == 120
         and all(counters[name] <= PRICE_MAX[name] for name in PRICE_MAX))
    B = boundary_passes(reports, "entry12")
    C = all(own(expert, parity, "post_mlp17")["metrics"][component]["signed_projection"] >= .95
            and own(expert, parity, "post_mlp17")["metrics"][component]["direction_fraction"] >= .875
            for expert in EXPERTS for parity in (0, 1)
            for component in ("head_reset_loss", "head_rescue"))
    lockin = lockin_boundary(reports)
    D = lockin is not None
    E = lockin is not None and all(
        reports[expert][str(parity)][lockin]["controls"][panel][arm]["top1_flip_count"] == 0
        and reports[expert][str(parity)][lockin]["controls"][panel][arm]["mean_kl"] <= .02
        for expert in EXPERTS for parity in (0, 1)
        for panel in ("P", "C") for arm in ("rescue", "reset"))
    predictions = {
        "pred_a_authority_replay_closure_finiteness_and_price": A,
        "pred_b_entry12_residual_state_carries_upstream_effect": B,
        "pred_c_final_residual_state_closes_logit_route": C,
        "pred_d_stable_preterminal_lockin_boundary_exists": D,
        "pred_e_lockin_boundary_is_control_selective": E,
    }
    terminal = ("invalid" if not A or not C else "diffuse_residual_reconstruction" if not D
                else "delayed_residual_lockin" if not B
                else "nonselective_residual_lockin" if not E
                else "residual_lockin_boundary_identified")
    result = {
        "schema": "temporal_iswas_v15_residual_state_boundary_mediation_atlas_result_v1",
        "candidate_id": CANDIDATE_ID,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "serial_seconds": time.perf_counter() - start,
        "authority_sha256": EXPECTED,
        "reports": reports,
        "lockin_boundary": lockin,
        "instrument": {"manual_native_max_abs_error": native_closure,
                       "parent_off_replay_max_abs_error": parent_replay,
                       "self_clamp_replay_max_abs_error": self_replay,
                       "factorial_closure_max_abs_error": closure,
                       "row_coverage_ok": row_coverage},
        "predictions": predictions,
        "terminal": terminal,
        "price": {**counters, "maxima": PRICE_MAX},
    }
    atomic_create_json(OUT, result)
    print(json.dumps({"predictions": predictions, "terminal": terminal,
                      "lockin_boundary": lockin, "instrument": result["instrument"],
                      "price": result["price"]}, sort_keys=True))


if __name__ == "__main__":
    main()
