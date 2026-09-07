#!/usr/bin/env python3
"""Causally execute the factor-pruned form of the frozen eight-site response program."""
# BQGATE: EXPERIMENT pred_a_authority_hash_factor_reconstruction_finiteness_and_price pred_b_factor_pruned_program_is_target_sufficient pred_c_factor_pruned_program_is_selective pred_d_attention_and_mlp_factor_branches_compose pred_e_mlp1_iswas_interaction_is_causally_required
from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_iswas_three_mlp_response_program_fresh_v12_v1 as population
import run_temporal_five_mlp_target_contrast_response_basis_v1 as v1
import run_temporal_five_mlp_generic_subspace_complement_program_v1 as comp
import run_temporal_five_mlp_matched_control_upstream_atlas_v2 as matched
import run_temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1 as atlasrun
import run_temporal_five_mlp_upstream_tensor_ranked_greedy_program_v1 as greedy
import run_temporal_five_mlp_crossfit_response_weight_interface_v1 as interface
import run_temporal_five_mlp_family_feasible_kl_refinement_v1 as klfit
import run_temporal_five_mlp_frozen_response_ood_regularization_v2 as ood
import run_temporal_five_mlp_projector_coefficient_factor_atlas_v1 as factor

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_five_mlp_factor_pruned_causal_program_v1.json"
FACTOR_RESULT = ROOT / "circuits/followups/temporal_five_mlp_projector_coefficient_factor_atlas_v1_result.json"
FACTOR_RUNNER = ROOT / "ops/run_temporal_five_mlp_projector_coefficient_factor_atlas_v1.py"
LITERAL_RESULT = ROOT / "circuits/followups/temporal_five_mlp_literal_projector_weight_execution_v1_result.json"
INTERFACE = ROOT / "circuits/followups/temporal_five_mlp_crossfit_response_weight_interface_v1_result.json"
KLFIT = ROOT / "circuits/followups/temporal_five_mlp_family_feasible_kl_refinement_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_five_mlp_factor_pruned_causal_program_v1_result.json"
EXPECTED = {
    "prior": "98a1d29b3782e47b0c884ed35df2fbe9106373a9faaa380b4669d9b733b5b5ee",
    "factor_result": "a092ffd51aa8b4964f68860c7082e2cbf273680ae867bb6f3543b10c1478326b",
    "factor_runner": "8619c04c59f8cc8a7188c36240c1625b6edfc6b9e9cc8569d623df8962cf783a",
    "literal_result": "0cc7e19a287b433511127692a0a13c1d86447a7818d9efd8aa0e6325d52fedd1",
    "interface": "578fa690f2f3962409bb50bd40a1c4c6f0b5ce7f88828e2e889a3a47bb20f013",
    "klfit": "347d9ba45de6c5fdd54743c55f119eb3da529fcb8e25a1658b0549fe3b9c1853",
}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def capture_everything(backend, batch):
    """One native forward yielding factor inputs plus the ordinary selected-site cache."""
    mlp_outputs = {}; handles = []
    layers = sorted(atlasrun.site_parts(site)[1] for site in comp.SITES if atlasrun.site_parts(site)[0] == "mlp")
    for layer in layers:
        def save(_module, _arguments, output, layer=layer):
            mlp_outputs[layer] = output.detach().float().clone()
        handles.append(backend.model.transformer.h[layer].mlp.register_forward_hook(save))
    try:
        output, attention, inputs, reconstruction = factor.capture_all(backend, batch)
    finally:
        for handle in handles: handle.remove()
    if set(mlp_outputs) != set(layers): raise RuntimeError("incomplete MLP output capture")
    cache = {
        "attention": {layer: item["head_output"].reshape(item["head_output"].shape[0], item["head_output"].shape[1], -1)
                      for layer, item in attention.items()},
        "mlp": mlp_outputs,
    }
    return output, cache, attention, inputs, reconstruction


def factor_deltas(backend, rows, base_attention, donor_attention, base_inputs, donor_inputs, *, include_exception):
    """Frozen factor rule: suffix base-pattern/value change plus MLP linear terms."""
    torch = backend.torch; cues = factor.cue_positions(rows); attention = {}; mlps = {}
    for site in comp.SITES:
        kind, layer, head = atlasrun.site_parts(site)
        if kind == "attn":
            p0 = base_attention[layer]["pattern"][:, head].float()
            dv = donor_attention[layer]["value"][:, :, head].float() - base_attention[layer]["value"][:, :, head].float()
            delta = dv.new_zeros(dv.shape)
            for i, query_pos in enumerate([int(r["base_semantic_position"]) for r in rows]):
                cue = cues[i]
                for query in range(query_pos + 1):
                    if query >= cue:
                        delta[i, query] = (p0[i, query, cue:query + 1, None] * dv[i, cue:query + 1]).sum(0)
            attention[site] = delta
        else:
            module = backend.model.transformer.h[layer].mlp
            xb, xd = base_inputs[layer], donor_inputs[layer]
            with torch.no_grad():
                l0, l1 = module.Left(xb).float(), module.Left(xd).float()
                r0, r1 = module.Right(xb).float(), module.Right(xd).float()
                dl, dr = l1 - l0, r1 - r0
                hidden = dl * r0 + l0 * dr
                if include_exception and site == "MLP1":
                    ids = [i for i, row in enumerate(rows) if klfit.task_name(row) == "iswas"]
                    if ids:
                        ix = torch.as_tensor(ids, device=backend.device)
                        hidden[ix] = hidden[ix] + dl[ix] * dr[ix]
                mlps[site] = hidden @ module.Down.weight.detach().float().T
    return attention, mlps


def run_factors(backend, batch, base, projectors, attention_delta, mlp_delta, *, use_attention, use_mlp):
    torch = backend.torch; width = backend.model.config.n_embd // backend.model.config.n_head; handles = []
    if use_attention:
        for site, raw in attention_delta.items():
            _kind, layer, head = atlasrun.site_parts(site); q = projectors[site][:, :8]
            source = base["attention"][layer]; a, z = head * width, (head + 1) * width
            def patch(_module, arguments, raw=raw, source=source, q=q, a=a, z=z):
                changed = arguments[0].clone()
                for i, pos in enumerate(batch.semantic_positions):
                    stop = int(pos) + 1
                    projected = (raw[i, :stop].to(changed).float() @ q) @ q.T
                    changed[i, :stop, a:z] = source[i, :stop, a:z].to(changed) + projected.to(changed)
                return (changed,) + tuple(arguments[1:])
            handles.append(backend.model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(patch))
    if use_mlp:
        for site, raw in mlp_delta.items():
            _kind, layer, _head = atlasrun.site_parts(site); q = projectors[site][:, :8]; source = base["mlp"][layer]
            def patch(_module, _arguments, output, raw=raw, source=source, q=q):
                changed = output.clone()
                for i, pos in enumerate(batch.semantic_positions):
                    stop = int(pos) + 1
                    projected = (raw[i, :stop].to(changed).float() @ q) @ q.T
                    changed[i, :stop] = source[i, :stop].to(changed) + projected.to(changed)
                return changed
            handles.append(backend.model.transformer.h[layer].mlp.register_forward_hook(patch))
    try: return backend.native(batch, capture=True)
    finally:
        for handle in handles: handle.remove()


def serial_report(backend, rows, output, base_state, full_state, reader, temporal_n):
    state = atlasrun.states(backend.torch, backend, output, rows)
    return ood.target_report(backend, rows, base_state, full_state, state, reader, temporal_n)


def main():
    observed = {"prior": sha(PRIOR), "factor_result": sha(FACTOR_RESULT), "factor_runner": sha(FACTOR_RUNNER),
                "literal_result": sha(LITERAL_RESULT), "interface": sha(INTERFACE), "klfit": sha(KLFIT)}
    if observed != EXPECTED: raise RuntimeError(f"factor-pruned authority changed: {observed}")
    dry = {"candidate_id": "temporal_auxiliary.five_mlp_factor_pruned_causal_program_v1", "dryrun": True,
           "gpu_accessed": False, "model_loaded": False, "queue_touched": False, "sites": comp.SITES,
           "projectors": ("even_fit", "odd_fit"), "model_forwards_max": 29, "fit_updates": 0,
           "model_updates": 0, "transformer_backwards": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1": print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    atlas = json.loads(FACTOR_RESULT.read_text()); literal = json.loads(LITERAL_RESULT.read_text())
    if atlas["terminal"] != "mixed_coefficient_mechanisms" or literal["terminal"] != "literal_weight_executable_response_program": raise RuntimeError("upstream terminal changed")
    started, tic = now(), time.perf_counter(); backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch
    reader, orientation, reader_ok = greedy.physical_reader(backend, json.loads(comp.WEIGHTS.read_text()))

    old_tcap, old_icap = json.loads(comp.TCAP.read_text()), json.loads(comp.ICAP.read_text())
    old_tr, old_ir, *_ = greedy.rows_and_controls(old_tcap, old_icap); controls = matched.control_rows()
    te, to = old_tr[::2] + old_ir[::2], old_tr[1::2] + old_ir[1::2]
    ceb, _, ce0, _, ce1, _, _ = interface.cap_inputs(backend, controls["discovery"])
    cob, _, co0, _, co1, _, _ = interface.cap_inputs(backend, controls["validation"])
    teb, _, te0, _, te1, _, _ = interface.cap_inputs(backend, te)
    tob, _, to0, _, to1, _, _ = interface.cap_inputs(backend, to)
    cbe, _ = comp.fit_bases(backend, ceb, ce0, ce1); cbo, _ = comp.fit_bases(backend, cob, co0, co1)
    qe, _ = v1.fit_target(backend, teb, te0, te1, cbe); qo, _ = v1.fit_target(backend, tob, to0, to1, cbo)
    projectors = {"even_fit": qe, "odd_fit": qo}; iface = json.loads(INTERFACE.read_text())
    hashes_ok = all([interface.thash(qe[s]), interface.thash(qo[s])] == iface["records"][s]["fit_basis_sha256"] for s in comp.SITES)

    stcap, sicap = json.loads(klfit.STC.read_text()), json.loads(klfit.SIC.read_text())
    tr = sum((population.capable_rows(klfit.sealed_t, stcap, p, 12) for p in ("A1", "A2")), [])
    ir = sum((population.capable_rows(klfit.sealed_i, sicap, p, 12) for p in ("A1", "A2")), [])
    rows = tr + ir; crows = [r for r in klfit.sealed_t.build_rows() if r["transform_id"] == "P"][:16]
    tb = das._batch(backend, rows, side="base"); td = das._batch(backend, rows, side="donor")
    cb = das._batch(backend, crows, side="base"); cd = das._batch(backend, crows, side="donor")
    tbo, tbase, tba, tbi, terr0 = capture_everything(backend, tb)
    _tdo, tdonor, tda, tdi, terr1 = capture_everything(backend, td)
    cbo, cbase, cba, cbi, cerr0 = capture_everything(backend, cb)
    _cdo, cdonor, cda, cdi, cerr1 = capture_everything(backend, cd)
    base_target = atlasrun.states(torch, backend, tbo, rows); base_control = atlasrun.states(torch, backend, cbo, crows)
    full_out, _ = atlasrun.run_patch(backend, tb, tdonor, comp.SITES); full_target = atlasrun.states(torch, backend, full_out, rows)
    tdeltas = factor_deltas(backend, rows, tba, tda, tbi, tdi, include_exception=True)
    tdeltas_noexc = factor_deltas(backend, rows, tba, tda, tbi, tdi, include_exception=False)
    cdeltas = factor_deltas(backend, crows, cba, cda, cbi, cdi, include_exception=True)
    control_ctx = {"rows": crows, "base_logits": das.head_logits(backend, base_control).float()}

    reports = {}
    for label, bases in projectors.items():
        complete_t = ood.run_project(backend, tb, tbase, tdonor, bases)
        complete_c = ood.run_project(backend, cb, cbase, cdonor, bases)
        pruned_t = run_factors(backend, tb, tbase, bases, *tdeltas, use_attention=True, use_mlp=True)
        pruned_c = run_factors(backend, cb, cbase, bases, *cdeltas, use_attention=True, use_mlp=True)
        attention_t = run_factors(backend, tb, tbase, bases, *tdeltas, use_attention=True, use_mlp=False)
        mlp_t = run_factors(backend, tb, tbase, bases, *tdeltas, use_attention=False, use_mlp=True)
        noexc_t = run_factors(backend, tb, tbase, bases, *tdeltas_noexc, use_attention=True, use_mlp=True)
        arms = {"complete_projector": complete_t, "factor_pruned": pruned_t, "attention_only": attention_t,
                "mlp_only": mlp_t, "factor_pruned_without_mlp1_exception": noexc_t}
        target = {name: serial_report(backend, rows, output, base_target, full_target, reader, len(tr)) for name, output in arms.items()}
        reports[label] = {"target": target,
                          "controls": {"complete_projector": klfit.control_metrics(backend, control_ctx, das.head_logits(backend, atlasrun.states(torch, backend, complete_c, crows)).float()),
                                       "factor_pruned": klfit.control_metrics(backend, control_ctx, das.head_logits(backend, atlasrun.states(torch, backend, pruned_c, crows)).float())}}

    finite = [orientation, terr0, terr1, cerr0, cerr1]
    for report in reports.values():
        for arm in report["target"].values(): finite += list(arm["cells"].values()) + list(arm["behavior_signed_projection"].values())
        for control in report["controls"].values(): finite += list(control["margin_rms_fraction"].values()) + [control["median_kl"], control["max_kl"], control["top1_flip_fraction"]]
    pa = hashes_ok and reader_ok and orientation <= 1e-6 and atlas["summary"]["max_factor_closure_rse"] <= 1e-10 and max(terr0, terr1, cerr0, cerr1) <= 5e-4 and all(math.isfinite(float(x)) for x in finite)
    pb = all(r["target"]["factor_pruned"]["worst_target_residual"] <= .15 and min(r["target"]["factor_pruned"]["behavior_signed_projection"].values()) >= .8 for r in reports.values())
    pc = all(max(r["controls"]["factor_pruned"]["margin_rms_fraction"].values()) <= .1 and r["controls"]["factor_pruned"]["median_kl"] <= .02 and r["controls"]["factor_pruned"]["top1_flip_fraction"] <= .05 for r in reports.values())
    composition = {}
    for label, report in reports.items():
        projection = {arm: item["behavior_signed_projection"] for arm, item in report["target"].items()}
        attention_gain = max(projection["factor_pruned"][task] - projection["attention_only"][task] for task in ("temporal", "iswas"))
        mlp_gain = max(projection["factor_pruned"][task] - projection["mlp_only"][task] for task in ("temporal", "iswas"))
        composition[label] = {"gain_over_attention_only_max": attention_gain, "gain_over_mlp_only_max": mlp_gain}
    pd = all(x["gain_over_attention_only_max"] >= .05 and x["gain_over_mlp_only_max"] >= .05 for x in composition.values())
    exception = {}
    for label, report in reports.items():
        yes, no = report["target"]["factor_pruned"], report["target"]["factor_pruned_without_mlp1_exception"]
        residual_ratio = no["cells"]["iswas_behavior"] / max(yes["cells"]["iswas_behavior"], 1e-30)
        projection_loss = yes["behavior_signed_projection"]["iswas"] - no["behavior_signed_projection"]["iswas"]
        temporal_projection_change = abs(yes["behavior_signed_projection"]["temporal"] - no["behavior_signed_projection"]["temporal"])
        temporal_residual_change = abs(yes["cells"]["temporal_behavior"] - no["cells"]["temporal_behavior"])
        exception[label] = {"iswas_behavior_residual_ratio": residual_ratio, "iswas_projection_loss": projection_loss,
                            "temporal_projection_abs_change": temporal_projection_change, "temporal_behavior_residual_abs_change": temporal_residual_change}
    pe = all((x["iswas_behavior_residual_ratio"] >= 1.2 or x["iswas_projection_loss"] >= .02) and x["temporal_projection_abs_change"] <= .01 and x["temporal_behavior_residual_abs_change"] <= .01 for x in exception.values())
    predictions = {"pred_a_authority_hash_factor_reconstruction_finiteness_and_price": bool(pa),
                   "pred_b_factor_pruned_program_is_target_sufficient": bool(pb),
                   "pred_c_factor_pruned_program_is_selective": bool(pc),
                   "pred_d_attention_and_mlp_factor_branches_compose": bool(pd),
                   "pred_e_mlp1_iswas_interaction_is_causally_required": bool(pe)}
    terminal = "invalid" if not pa else "factor_pruned_weight_program" if all(predictions.values()) else "factor_pruning_incomplete"
    summary = {"factor_pruned_worst_target_max": max(r["target"]["factor_pruned"]["worst_target_residual"] for r in reports.values()),
               "factor_pruned_projection_min": min(min(r["target"]["factor_pruned"]["behavior_signed_projection"].values()) for r in reports.values()),
               "factor_pruned_control_margin_max": max(max(r["controls"]["factor_pruned"]["margin_rms_fraction"].values()) for r in reports.values()),
               "factor_pruned_control_median_kl_max": max(r["controls"]["factor_pruned"]["median_kl"] for r in reports.values()),
               "factor_pruned_control_flip_max": max(r["controls"]["factor_pruned"]["top1_flip_fraction"] for r in reports.values()),
               "composition": composition, "mlp1_exception": exception}
    result = {"schema": "temporal_five_mlp_factor_pruned_causal_program_result_v1", "started_utc": started, "finished_utc": now(),
              "serial_seconds": time.perf_counter() - tic, "authority_sha256": EXPECTED, "reports": reports, "summary": summary,
              "predictions": predictions, "terminal": terminal,
              "price": {"model_forwards": 27, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}}
    atomic_create_json(OUT, result); print(json.dumps({k: result[k] for k in ("summary", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
