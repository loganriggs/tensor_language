#!/usr/bin/env python3
"""Compile selected attention value responses into local and layer-0 c_v weight reads."""
# BQGATE: EXPERIMENT pred_a_authority_hash_exact_value_coordinate_closure_finiteness_and_price pred_b_compiled_attention_execution_matches_captured_factor_program pred_c_fully_weight_compiled_program_is_sufficient_and_selective pred_d_current_and_cached_value_branches_compose pred_e_attention_input_weight_maps_are_crossfit_stable
from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

import circuit_candidate_temporal_auxiliary_fresh_cues_v13 as fresh_t
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v12 as fresh_i
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
import run_temporal_five_mlp_factor_pruned_causal_program_v1 as pruned
import run_temporal_five_mlp_mlp_input_covector_crossfit_v1 as covector
import run_temporal_five_mlp_mlp_input_covector_ood_v2 as covood

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_five_mlp_attention_value_weight_pullback_v1.json"
OOD_RESULT = ROOT / "circuits/followups/temporal_five_mlp_mlp_input_covector_ood_v2_result.json"
OOD_RUNNER = ROOT / "ops/run_temporal_five_mlp_mlp_input_covector_ood_v2.py"
PRUNED_RESULT = ROOT / "circuits/followups/temporal_five_mlp_factor_pruned_causal_program_v1_result.json"
INTERFACE = ROOT / "circuits/followups/temporal_five_mlp_crossfit_response_weight_interface_v1_result.json"
TB = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v13.py"
TC = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v13_capability_v1_result.json"
IB = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v12.py"
IC = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v12_capability_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_five_mlp_attention_value_weight_pullback_v1_result.json"
EXPECTED = {
    "prior": "46f2bcfb80451ac967c4eade46c54e66ce6d99e0a0ef0e6b3839051bfe08255f",
    "ood_result": "10a6e169348cffb001f14d299ce05c3f33d80bd50e574dced2cac1228d436947",
    "ood_runner": "ca5ac0b74533d110c98415d1fa147e61f0f1c04036260898d181fddf8528c221",
    "pruned_result": "44cd07106b7b68c14d9cc068d620b28fdb527b65caf64c536978b277754bb2d7",
    "interface": "578fa690f2f3962409bb50bd40a1c4c6f0b5ce7f88828e2e889a3a47bb20f013",
    "temporal_builder": "3f738bf2fb2d4a5425dba85eaf948d7ed888e4b891666ff77a51c8638acd2509",
    "temporal_capability": "e053d3381680ce5a933356a060448466d7567e3079c4a2b9a5bff262bd98b9c1",
    "iswas_builder": "2734cbeceb4e6979dab22fe5b24870386874ac2f905ae426e6e689548e43e8a2",
    "iswas_capability": "67cb3efbd1ea86f98f94a826922928229a7c7b0a247f218778fc4960a6e8c6f4",
}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
def cosine(a, b): return float((a.flatten() @ b.flatten()) / (a.norm() * b.norm()).clamp_min(1e-30))


def capture_with_attention_inputs(backend, batch):
    inputs = {}; handles = []
    layers = sorted({0} | {atlasrun.site_parts(site)[1] for site in comp.SITES if atlasrun.site_parts(site)[0] == "attn"})
    for layer in layers:
        def save(_module, arguments, layer=layer): inputs[layer] = arguments[0].detach().clone()
        handles.append(backend.model.transformer.h[layer].attn.register_forward_pre_hook(save))
    try: values = pruned.capture_everything(backend, batch)
    finally:
        for handle in handles: handle.remove()
    if set(inputs) != set(layers): raise RuntimeError("incomplete attention-input capture")
    return (*values, inputs)


def compiled_attention(backend, rows, qs, base_attention, base_inputs, donor_inputs, *, current, cached):
    torch = backend.torch; cues = pruned.factor.cue_positions(rows); output = {}
    heads = backend.model.config.n_head; width = backend.model.config.n_embd // heads
    layer0 = backend.model.transformer.h[0].attn
    with torch.no_grad():
        v0b = layer0.c_v(base_inputs[0]).view(len(rows), -1, heads, width)
        v0d = layer0.c_v(donor_inputs[0]).view(len(rows), -1, heads, width)
    for site, q in qs.items():
        kind, layer, head = atlasrun.site_parts(site)
        if kind != "attn": continue
        module = backend.model.transformer.h[layer].attn
        with torch.no_grad():
            vlb = module.c_v(base_inputs[layer]).view(len(rows), -1, heads, width)
            vld = module.c_v(donor_inputs[layer]).view(len(rows), -1, heads, width)
            coordinate = q.new_zeros((len(rows), vlb.shape[1], q.shape[1]))
            if current: coordinate += (1.0 - module.lamb.float()) * ((vld[:, :, head].float() - vlb[:, :, head].float()) @ q)
            if cached: coordinate += module.lamb.float() * ((v0d[:, :, head].float() - v0b[:, :, head].float()) @ q)
        raw = coordinate.new_zeros((len(rows), coordinate.shape[1], width))
        pattern = base_attention[layer]["pattern"][:, head].float()
        for i, stop_position in enumerate([int(row["base_semantic_position"]) + 1 for row in rows]):
            cue = cues[i]
            for query in range(stop_position):
                if query >= cue:
                    response_coordinate = (pattern[i, query, cue:query + 1, None] * coordinate[i, cue:query + 1]).sum(0)
                    raw[i, query] = response_coordinate @ q.T
        output[site] = raw
    return output


def comparison(backend, rows, left, right):
    torch = backend.torch; a = atlasrun.states(torch, backend, left, rows); b = atlasrun.states(torch, backend, right, rows)
    la, lb = das.head_logits(backend, a).float(), das.head_logits(backend, b).float()
    ca, cb = la - la.mean(-1, keepdim=True), lb - lb.mean(-1, keepdim=True)
    return {"state_rse": float((b-a).square().sum() / a.square().sum().clamp_min(1e-30)),
            "centered_logits_rse": float((cb-ca).square().sum() / ca.square().sum().clamp_min(1e-30)),
            "state_max_abs": float((b-a).abs().max()), "logits_max_abs": float((lb-la).abs().max())}


def main():
    paths = {"prior": PRIOR, "ood_result": OOD_RESULT, "ood_runner": OOD_RUNNER, "pruned_result": PRUNED_RESULT,
             "interface": INTERFACE, "temporal_builder": TB, "temporal_capability": TC,
             "iswas_builder": IB, "iswas_capability": IC}
    observed = {key: sha(path) for key, path in paths.items()}
    if observed != EXPECTED: raise RuntimeError(f"attention pullback authority changed: {observed}")
    dry = {"candidate_id": "temporal_auxiliary.five_mlp_attention_value_weight_pullback_v1", "dryrun": True,
           "gpu_accessed": False, "model_loaded": False, "queue_touched": False, "sites": comp.SITES,
           "model_forwards_max": 28, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1": print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    if json.loads(OOD_RESULT.read_text())["terminal"] != "ood_weight_derived_input_reader": raise RuntimeError("MLP input reader not promoted")
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

    train_tcap, train_icap = json.loads(klfit.STC.read_text()), json.loads(klfit.SIC.read_text())
    train_t = sum((population.capable_rows(klfit.sealed_t, train_tcap, panel, 12) for panel in ("A1", "A2")), [])
    train_i = sum((population.capable_rows(klfit.sealed_i, train_icap, panel, 12) for panel in ("A1", "A2")), [])
    train_rows = train_t + train_i; train_batch = das._batch(backend, train_rows, side="base")
    _tro, _trcache, _tra, train_mlp_inputs, train_error, _train_attn_inputs = capture_with_attention_inputs(backend, train_batch)
    train_panels = {panel: [i for i, row in enumerate(train_rows) if row["transform_id"] == panel] for panel in ("A1", "A2")}

    tcap, icap = json.loads(TC.read_text()), json.loads(IC.read_text())
    test_t = sum((population.capable_rows(fresh_t, tcap, panel, 12) for panel in ("A1", "A2")), [])
    test_i = sum((population.capable_rows(fresh_i, icap, panel, 12) for panel in ("A1", "A2")), [])
    rows = test_t + test_i; crows = [row for row in fresh_t.build_rows() if row["transform_id"] == "P"][:16]
    tb, td = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
    cb, cd = das._batch(backend, crows, side="base"), das._batch(backend, crows, side="donor")
    tbo, tbase, tba, tbi, terr0, tai0 = capture_with_attention_inputs(backend, tb)
    _tdo, tdonor, tda, tdi, terr1, tai1 = capture_with_attention_inputs(backend, td)
    cbo, cbase, cba, cbi, cerr0, cai0 = capture_with_attention_inputs(backend, cb)
    _cdo, _cdonor, cda, cdi, cerr1, cai1 = capture_with_attention_inputs(backend, cd)
    base_target = atlasrun.states(torch, backend, tbo, rows); base_control = atlasrun.states(torch, backend, cbo, crows)
    full_out, _ = atlasrun.run_patch(backend, tb, tdonor, comp.SITES); full_target = atlasrun.states(torch, backend, full_out, rows)
    captured_t_attention, _captured_t_mlp = pruned.factor_deltas(backend, rows, tba, tda, tbi, tdi, include_exception=False)
    captured_c_attention, _captured_c_mlp = pruned.factor_deltas(backend, crows, cba, cda, cbi, cdi, include_exception=False)
    control_ctx = {"rows": crows, "base_logits": das.head_logits(backend, base_control).float()}

    reports = {}; closure = []; execution = []
    for label, qs in projectors.items():
        train_ids = train_panels["A1" if label == "even_fit" else "A2"]; mlp_bases = {}
        for site, q in qs.items():
            if atlasrun.site_parts(site)[0] != "mlp": continue
            amap = covector.effective_map(backend, site, q, train_mlp_inputs)
            mlp_bases[site], _energy = covector.fit_input_basis(torch, train_batch, amap, train_ids, 64)
        target_mlp = covood.compiled_mlps(backend, qs, tbi, tdi, mlp_bases)
        control_mlp = covood.compiled_mlps(backend, qs, cbi, cdi, mlp_bases)
        target_both = compiled_attention(backend, rows, qs, tba, tai0, tai1, current=True, cached=True)
        target_current = compiled_attention(backend, rows, qs, tba, tai0, tai1, current=True, cached=False)
        target_cached = compiled_attention(backend, rows, qs, tba, tai0, tai1, current=False, cached=True)
        control_both = compiled_attention(backend, crows, qs, cba, cai0, cai1, current=True, cached=True)
        for site, q in qs.items():
            if atlasrun.site_parts(site)[0] == "attn":
                closure.append(covector.coefficient_rse(torch, tb, captured_t_attention[site] @ q, target_both[site] @ q, list(range(len(rows)))))
                closure.append(covector.coefficient_rse(torch, cb, captured_c_attention[site] @ q, control_both[site] @ q, list(range(len(crows)))))
        captured_t = pruned.run_factors(backend, tb, tbase, qs, captured_t_attention, target_mlp, use_attention=True, use_mlp=True)
        captured_c = pruned.run_factors(backend, cb, cbase, qs, captured_c_attention, control_mlp, use_attention=True, use_mlp=True)
        compiled_t = pruned.run_factors(backend, tb, tbase, qs, target_both, target_mlp, use_attention=True, use_mlp=True)
        compiled_c = pruned.run_factors(backend, cb, cbase, qs, control_both, control_mlp, use_attention=True, use_mlp=True)
        current_t = pruned.run_factors(backend, tb, tbase, qs, target_current, target_mlp, use_attention=True, use_mlp=True)
        cached_t = pruned.run_factors(backend, tb, tbase, qs, target_cached, target_mlp, use_attention=True, use_mlp=True)
        target = {"compiled_both": pruned.serial_report(backend, rows, compiled_t, base_target, full_target, reader, len(test_t)),
                  "current_only": pruned.serial_report(backend, rows, current_t, base_target, full_target, reader, len(test_t)),
                  "cached_only": pruned.serial_report(backend, rows, cached_t, base_target, full_target, reader, len(test_t))}
        target_comparison = comparison(backend, rows, captured_t, compiled_t); control_comparison = comparison(backend, crows, captured_c, compiled_c)
        execution += [target_comparison["state_rse"], target_comparison["centered_logits_rse"], control_comparison["state_rse"], control_comparison["centered_logits_rse"]]
        control_state = atlasrun.states(torch, backend, compiled_c, crows)
        reports[label] = {"target": target, "control": klfit.control_metrics(backend, control_ctx, das.head_logits(backend, control_state).float()),
                          "captured_vs_compiled_target": target_comparison, "captured_vs_compiled_control": control_comparison}

    map_stability = {}
    heads = backend.model.config.n_head; width = backend.model.config.n_embd // heads
    for site in comp.SITES:
        kind, layer, head = atlasrun.site_parts(site)
        if kind != "attn": continue
        a, b = qe[site], qo[site]; u, _s, vh = torch.linalg.svd(b.T @ a); rotation = u @ vh
        sl = slice(head * width, (head + 1) * width)
        local_weight = backend.model.transformer.h[layer].attn.c_v.weight.detach().float()[sl].T
        cache_weight = backend.model.transformer.h[0].attn.c_v.weight.detach().float()[sl].T
        map_stability[site] = {"current": cosine(local_weight @ a, (local_weight @ b) @ rotation),
                               "cached": cosine(cache_weight @ a, (cache_weight @ b) @ rotation)}

    finite = [orientation, train_error, terr0, terr1, cerr0, cerr1] + closure + execution
    for report in reports.values():
        for target in report["target"].values(): finite += list(target["cells"].values()) + list(target["behavior_signed_projection"].values())
        finite += list(report["control"]["margin_rms_fraction"].values()) + [report["control"]["median_kl"], report["control"]["max_kl"], report["control"]["top1_flip_fraction"]]
    finite += [value for site in map_stability.values() for value in site.values()]
    pa = hashes_ok and reader_ok and orientation <= 1e-6 and max(closure) <= 1e-10 and max(train_error, terr0, terr1, cerr0, cerr1) <= 5e-4 and all(math.isfinite(float(x)) for x in finite)
    pb = max(execution) <= 1e-10
    pc = all(report["target"]["compiled_both"]["worst_target_residual"] <= .15 and min(report["target"]["compiled_both"]["behavior_signed_projection"].values()) >= .8 and max(report["control"]["margin_rms_fraction"].values()) <= .1 and report["control"]["median_kl"] <= .02 and report["control"]["top1_flip_fraction"] <= .05 for report in reports.values())
    composition = {}
    for label, report in reports.items():
        projections = {arm: item["behavior_signed_projection"] for arm, item in report["target"].items()}
        composition[label] = {branch: max(projections["compiled_both"][task] - projections[branch][task] for task in ("temporal", "iswas")) for branch in ("current_only", "cached_only")}
    pd = all(item["current_only"] >= .02 and item["cached_only"] >= .02 for item in composition.values())
    pe = all(value >= .8 for site in map_stability.values() for value in site.values())
    predictions = {"pred_a_authority_hash_exact_value_coordinate_closure_finiteness_and_price": bool(pa),
                   "pred_b_compiled_attention_execution_matches_captured_factor_program": bool(pb),
                   "pred_c_fully_weight_compiled_program_is_sufficient_and_selective": bool(pc),
                   "pred_d_current_and_cached_value_branches_compose": bool(pd),
                   "pred_e_attention_input_weight_maps_are_crossfit_stable": bool(pe)}
    terminal = "invalid" if not pa else "fully_weight_compiled_response_program" if all(predictions.values()) else "attention_weight_pullback_mixed"
    summary = {"max_value_coordinate_closure_rse": max(closure), "max_captured_vs_compiled_execution_rse": max(execution),
               "target_worst_max": max(report["target"]["compiled_both"]["worst_target_residual"] for report in reports.values()),
               "target_projection_min": min(min(report["target"]["compiled_both"]["behavior_signed_projection"].values()) for report in reports.values()),
               "control_margin_max": max(max(report["control"]["margin_rms_fraction"].values()) for report in reports.values()),
               "control_median_kl_max": max(report["control"]["median_kl"] for report in reports.values()),
               "control_flip_max": max(report["control"]["top1_flip_fraction"] for report in reports.values()),
               "composition_gain": composition, "min_weight_map_cosine": min(value for site in map_stability.values() for value in site.values())}
    result = {"schema": "temporal_five_mlp_attention_value_weight_pullback_result_v1", "started_utc": started, "finished_utc": now(),
              "serial_seconds": time.perf_counter() - tic, "authority_sha256": EXPECTED, "reports": reports,
              "weight_map_stability": map_stability, "summary": summary, "predictions": predictions, "terminal": terminal,
              "price": {"model_forwards": 26, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}}
    atomic_create_json(OUT, result); print(json.dumps({k: result[k] for k in ("summary", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
