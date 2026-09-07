#!/usr/bin/env python3
"""Joint full-component patch factorial for atlas-selected task source sets."""
# BQGATE: EXPERIMENT pred_a_authority_identity_full_replay_finiteness_and_price pred_b_own_top80_sets_are_jointly_sufficient pred_c_own_source_sets_beat_cross_task_sets pred_d_union_is_sufficient_and_selective pred_e_physical_complements_are_insufficient
from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import compiled_response_ood as oodctx
import pooled_response_projector as pooled
import run_temporal_iswas_rank46_task_mode_complete_rank_ladder_v1 as ladderrun
import run_temporal_iswas_rank46_task_typed_mode_causal_factorial_v1 as factorial
import run_temporal_iswas_rank46_task_rank4_upstream_weight_edge_atlas_v1 as edgeimpl
import run_temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1 as atlasrun
import run_temporal_five_mlp_family_feasible_kl_refinement_v1 as klfit
import run_temporal_five_mlp_generic_subspace_complement_program_v1 as comp

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_rank46_task_rank4_joint_source_factorial_v1.json"
ATLAS = ROOT / "circuits/followups/temporal_iswas_rank46_task_rank4_upstream_weight_edge_atlas_v2_result.json"
EDGE_IMPL = ROOT / "ops/run_temporal_iswas_rank46_task_rank4_upstream_weight_edge_atlas_v1.py"
RANK46 = factorial.RANK46
LADDER = ROOT / "circuits/followups/temporal_iswas_rank46_task_mode_complete_rank_ladder_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_iswas_rank46_task_rank4_joint_source_factorial_v1_result.json"
EXPECTED = {"prior": "819367d400c4da884b622052297344edaf83c957ab60b313478a56006a95e594",
    "atlas": "658069e39a5ce1a68da3b0e812c635364e0aa0fcff82bc4713afbccba5c1df51",
    "edge_impl": "341fcf3d9b0fe1df029f2da6a3ddcb90e57cfc46ea21893a085c1883dfa095b6",
    "rank46": "dc8b66d826acede98bde996babe42420dd9e805981f6b81de458d568f29eea1d",
    "ladder": "345ecd0542890c69181efe067729560375543c694f2183319ff824300028e6d8"}
TEMPORAL = ("MLP1", "MLP0", "MLP2", "MLP3", "L0H3", "MLP4", "MLP8", "MLP7", "MLP6")
ISWAS = ("MLP1", "MLP2", "MLP0", "MLP3", "MLP4", "MLP5", "L2H2", "MLP6", "MLP7", "MLP8", "L4H7")
MAX_FORWARDS = 32


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def response_report(backend, fresh, base, donor, changed, qs, modes):
    torch = backend.torch; width = int(backend.model.config.n_embd // backend.model.config.n_head); output = {}
    for task in ("temporal", "iswas"):
        ids = [i for i, row in enumerate(fresh["rows"]) if klfit.task_name(row) == task]
        num = den = residual = 0.0; sites = {}
        for site in edgeimpl.RESPONSE_SITES:
            rb = qs[site] @ modes[site][task]
            b = edgeimpl.valid_rows(torch, edgeimpl.raw_response(base, site, width), fresh["batch"], ids)
            d = edgeimpl.valid_rows(torch, edgeimpl.raw_response(donor, site, width), fresh["batch"], ids) - b
            x = edgeimpl.valid_rows(torch, edgeimpl.raw_response(changed, site, width), fresh["batch"], ids) - b
            dr, xr = d @ rb, x @ rb; sd = float(dr.square().sum()); sn = float((xr*dr).sum()); sr = float((xr-dr).square().sum())
            sites[site] = {"signed_projection": sn/max(1e-30, sd), "relative_squared_error": sr/max(1e-30, sd)}
            num += sn; den += sd; residual += sr
        output[task] = {"signed_projection": num/max(1e-30, den), "relative_squared_error": residual/max(1e-30, den), "sites": sites}
    return output


def behavior_report(backend, rows, base_state, donor_state, changed_state):
    torch = backend.torch; base = comp.margins(backend, base_state, rows); target = comp.margins(backend, donor_state, rows)-base
    changed = comp.margins(backend, changed_state, rows)-base; output = {}
    for task in ("temporal", "iswas"):
        ix = torch.as_tensor([i for i, row in enumerate(rows) if klfit.task_name(row) == task], device=backend.device)
        output[task] = float((changed[ix]*target[ix]).sum()/target[ix].square().sum().clamp_min(1e-30))
    return output


def main():
    observed = {key: sha(path) for key, path in {"prior": PRIOR, "atlas": ATLAS, "edge_impl": EDGE_IMPL, "rank46": RANK46, "ladder": LADDER}.items()}
    if observed != EXPECTED: raise RuntimeError(f"joint-source authority changed: {observed}")
    support = tuple(json.loads(RANK46.read_text())["selected_support"]); intersection = tuple(x for x in support if x in set(TEMPORAL)&set(ISWAS))
    union = tuple(x for x in support if x in set(TEMPORAL)|set(ISWAS)); arms = {"full46": support, "temporal_top80": TEMPORAL,
        "iswas_top80": ISWAS, "intersection": intersection, "union": union,
        "temporal_complement": tuple(x for x in support if x not in TEMPORAL),
        "iswas_complement": tuple(x for x in support if x not in ISWAS)}
    dry = {"candidate_id": "temporal_auxiliary.iswas_rank46_task_rank4_joint_source_factorial_v1", "dryrun": True,
        "gpu_accessed": False, "model_loaded": False, "queue_touched": False, "arms": {k: list(v) for k,v in arms.items()},
        "model_forwards_max": MAX_FORWARDS, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1": print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    started, tic = now(), time.perf_counter(); backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch
    fitted = pooled.fit(backend); qs = fitted["projector"]; task_rows, full_modes = ladderrun.fit_full_modes(backend, fitted)
    modes = {site: {task: full_modes[site][task][:, :4] for task in ("temporal", "iswas")} for site in edgeimpl.RESPONSE_SITES}
    fresh = oodctx.capture(backend, factorial.TCAP, factorial.ICAP); _unused, donor_full = atlasrun.capture_native(backend, fresh["donor_batch"])
    base_output, base = edgeimpl.capture_with_source_patch(backend, fresh["batch"], fresh["base_full"], ())
    donor_output, donor = edgeimpl.capture_with_source_patch(backend, fresh["donor_batch"], donor_full, ())
    base_state = atlasrun.states(torch, backend, base_output, fresh["rows"]); donor_state = atlasrun.states(torch, backend, donor_output, fresh["rows"])
    reports = {}; finite = []
    for name, sites in arms.items():
        output, response = edgeimpl.capture_with_source_patch(backend, fresh["batch"], donor_full, sites)
        rr = response_report(backend, fresh, base, donor, response, qs, modes)
        br = behavior_report(backend, fresh["rows"], base_state, donor_state, atlasrun.states(torch, backend, output, fresh["rows"]))
        reports[name] = {"sources": list(sites), "response": rr, "behavior_signed_projection": br}
        finite += list(br.values()) + [x for task in rr.values() for x in (task["signed_projection"], task["relative_squared_error"])]
    _cunused, control_donor_full = atlasrun.capture_native(backend, fresh["control_donor_batch"])
    control_output, _control_response = edgeimpl.capture_with_source_patch(backend, fresh["control_batch"], control_donor_full, union)
    control_ctx = {"rows": fresh["controls"], "base_logits": das.head_logits(backend, fresh["control_base_state"]).float()}
    controls = klfit.control_metrics(backend, control_ctx, das.head_logits(backend, atlasrun.states(torch, backend, control_output, fresh["controls"])).float())
    finite += list(controls["margin_rms_fraction"].values()) + [controls["median_kl"], controls["max_kl"], controls["top1_flip_fraction"]]
    fit_ids = {row["row_id"] for rows in task_rows.values() for row in rows} | {row["row_id"] for row in fitted["control_rows"]}
    eval_ids = {row["row_id"] for row in fresh["rows"] + fresh["controls"]}
    prior = json.loads(PRIOR.read_text())["authority"]
    pa = (len(support)==46 and list(TEMPORAL)==prior["temporal_top80_sources"] and list(ISWAS)==prior["iswas_top80_sources"]
          and not(fit_ids & eval_ids) and min(reports["full46"]["behavior_signed_projection"].values()) >= .8
          and min(x["signed_projection"] for x in reports["full46"]["response"].values()) >= .8 and all(math.isfinite(float(x)) for x in finite))
    own = {"temporal": reports["temporal_top80"], "iswas": reports["iswas_top80"]}
    cross = {"temporal": reports["iswas_top80"], "iswas": reports["temporal_top80"]}
    pb = all(own[t]["response"][t]["signed_projection"] >= .7 and own[t]["response"][t]["relative_squared_error"] <= .3
             and own[t]["behavior_signed_projection"][t] >= .6 for t in own)
    advantages = {t: own[t]["response"][t]["signed_projection"]-cross[t]["response"][t]["signed_projection"] for t in own}
    pc = min(advantages.values()) >= .1
    ur = reports["union"]; pd = (all(ur["response"][t]["signed_projection"] >= .8 and ur["response"][t]["relative_squared_error"] <= .2
        and ur["behavior_signed_projection"][t] >= .75 for t in own) and controls["median_kl"] <= .02 and controls["top1_flip_fraction"] == 0.0)
    pe = (reports["temporal_complement"]["response"]["temporal"]["signed_projection"] <= .5
          and reports["iswas_complement"]["response"]["iswas"]["signed_projection"] <= .5)
    predictions = {"pred_a_authority_identity_full_replay_finiteness_and_price": bool(pa),
        "pred_b_own_top80_sets_are_jointly_sufficient": bool(pb), "pred_c_own_source_sets_beat_cross_task_sets": bool(pc),
        "pred_d_union_is_sufficient_and_selective": bool(pd), "pred_e_physical_complements_are_insufficient": bool(pe)}
    terminal = "invalid" if not pa else "joint_task_typed_source_program" if all(predictions.values()) else "joint_source_interaction_boundary"
    result = {"schema": "temporal_iswas_rank46_task_rank4_joint_source_factorial_result_v1", "started_utc": started, "finished_utc": now(),
        "serial_seconds": time.perf_counter()-tic, "authority_sha256": EXPECTED, "arms": {k:list(v) for k,v in arms.items()},
        "reports": reports, "own_minus_cross_response": advantages, "union_controls": controls, "predictions": predictions,
        "terminal": terminal, "price": {"model_forwards_max": MAX_FORWARDS, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}}
    atomic_create_json(OUT, result); print(json.dumps({k:result[k] for k in ("reports", "own_minus_cross_response", "union_controls", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
