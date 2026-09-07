#!/usr/bin/env python3
"""End-to-end cue source -> restricted upstream graph -> compiled response program."""
# BQGATE: EXPERIMENT pred_a_authority_top20_self_clamp_embedding_scope_finiteness_and_price pred_b_source_clamped_graph_generates_compiled_coordinates pred_c_source_clamped_compiled_program_is_target_sufficient pred_d_source_clamped_compiled_program_is_selective pred_e_source_clamped_graph_is_crossfit_stable
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
import run_temporal_five_mlp_attention_value_weight_pullback_v1 as pullback
import run_temporal_five_mlp_compiled_coordinate_upstream_atlas_v1 as atlas

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_five_mlp_source_clamped_compiled_graph_v1.json"
EMBED = ROOT / "circuits/followups/temporal_five_mlp_compiled_coordinate_embedding_source_v1_result.json"
ATLAS = ROOT / "circuits/followups/temporal_five_mlp_compiled_coordinate_upstream_atlas_v1_result.json"
ATTENTION_AUDIT = ROOT / "circuits/followups/temporal_five_mlp_attention_value_weight_pullback_v2_tolerance_audit_result.json"
MLP_OOD = ROOT / "circuits/followups/temporal_five_mlp_mlp_input_covector_ood_v2_result.json"
INTERFACE = ROOT / "circuits/followups/temporal_five_mlp_crossfit_response_weight_interface_v1_result.json"
TC = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v13_capability_v1_result.json"
IC = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v12_capability_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_five_mlp_source_clamped_compiled_graph_v1_result.json"
EXPECTED = {
    "prior": "40eca783c8782dd14ebf14b211cb5bc306c8b19de2c81296fbd73703b7171e89",
    "embedding_source": "b02305e9b92166d795b8afddb99cf73192970050d422a5e77d4977e4e37553cc",
    "atlas": "0b96284ebf7bd89ccdbf1dbc0fbdacd54e58eb16841865c48b985aa1a3c2da53",
    "attention_audit": "42f9db9cf44e63a1930f454798b9d25b115cea94e4c45a8344694da6982637b4",
    "mlp_ood": "10a6e169348cffb001f14d299ce05c3f33d80bd50e574dced2cac1228d436947",
    "interface": "578fa690f2f3962409bb50bd40a1c4c6f0b5ce7f88828e2e889a3a47bb20f013",
    "temporal_capability": "e053d3381680ce5a933356a060448466d7567e3079c4a2b9a5bff262bd98b9c1",
    "iswas_capability": "67cb3efbd1ea86f98f94a826922928229a7c7b0a247f218778fc4960a6e8c6f4",
}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def restricted_graph(backend, base_batch, donor_batch, base_full, top, *, source):
    torch = backend.torch; width = backend.model.config.n_embd // backend.model.config.n_head; handles = []; counts = []
    top = set(top)
    if source:
        base_tokens, _ = backend._tensor_batch(base_batch); donor_tokens, _ = backend._tensor_batch(donor_batch)
        mask = base_tokens != donor_tokens; counts = [int(row.sum()) for row in mask]
        with torch.no_grad(): donor_embedding = backend.model.transformer.wte(donor_tokens).detach().clone()
        def embedding_hook(_module, _arguments, output):
            changed = output.clone(); changed[mask] = donor_embedding[mask].to(changed); return changed
        handles.append(backend.model.transformer.wte.register_forward_hook(embedding_hook))
    for layer in range(11):
        keep_heads = {head for head in range(9) if f"L{layer}H{head}" in top}; clamp_heads = tuple(head for head in range(9) if head not in keep_heads)
        base_attention = base_full["attention"][layer]
        def attention_hook(_module, arguments, layer=layer, clamp_heads=clamp_heads, base_attention=base_attention):
            changed = arguments[0].clone()
            for i, pos in enumerate(base_batch.semantic_positions):
                stop = int(pos) + 1
                for head in clamp_heads:
                    a, z = head * width, (head + 1) * width
                    changed[i, :stop, a:z] = base_attention[i, :stop, a:z].to(changed)
            return (changed,) + tuple(arguments[1:])
        handles.append(backend.model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(attention_hook))
        if f"MLP{layer}" not in top:
            base_mlp = base_full["mlp"][layer]
            def mlp_hook(_module, _arguments, output, base_mlp=base_mlp):
                changed = output.clone()
                for i, pos in enumerate(base_batch.semantic_positions):
                    stop = int(pos) + 1; changed[i, :stop] = base_mlp[i, :stop].to(changed)
                return changed
            handles.append(backend.model.transformer.h[layer].mlp.register_forward_hook(mlp_hook))
    try: output, attention, mlps = atlas.patch_and_capture_inputs(backend, base_batch, base_full, ())
    finally:
        for handle in handles: handle.remove()
    return output, attention, mlps, counts


def main():
    paths = {"prior": PRIOR, "embedding_source": EMBED, "atlas": ATLAS, "attention_audit": ATTENTION_AUDIT,
             "mlp_ood": MLP_OOD, "interface": INTERFACE, "temporal_capability": TC, "iswas_capability": IC}
    observed = {key: sha(path) for key, path in paths.items()}
    if observed != EXPECTED: raise RuntimeError(f"source-clamped authority changed: {observed}")
    dry = {"candidate_id": "temporal_auxiliary.five_mlp_source_clamped_compiled_graph_v1", "dryrun": True,
           "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
           "model_forwards_max": 24, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1": print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    upstream = json.loads(ATLAS.read_text()); top = upstream["top20"]["even_fit"]["sites"]
    top_agree = top == upstream["top20"]["odd_fit"]["sites"] and len(top) == 20
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
    _tro, _trcache, _tra, train_inputs, train_error, _trai = pullback.capture_with_attention_inputs(backend, train_batch)
    train_panels = {panel: [i for i, row in enumerate(train_rows) if row["transform_id"] == panel] for panel in ("A1", "A2")}

    tcap, icap = json.loads(TC.read_text()), json.loads(IC.read_text())
    test_t = sum((population.capable_rows(fresh_t, tcap, panel, 12) for panel in ("A1", "A2")), [])
    test_i = sum((population.capable_rows(fresh_i, icap, panel, 12) for panel in ("A1", "A2")), [])
    rows = test_t + test_i; crows = [row for row in fresh_t.build_rows() if row["transform_id"] == "P"][:16]
    batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
    control_batch, control_donor_batch = das._batch(backend, crows, side="base"), das._batch(backend, crows, side="donor")
    base_output, _bcache, base_attention, base_mlp, base_error, base_ai = pullback.capture_with_attention_inputs(backend, batch)
    _donor_output, donor_selected, _da, donor_mlp, donor_error, donor_ai = pullback.capture_with_attention_inputs(backend, donor_batch)
    control_base_output, _cbcache, control_attention, control_mlp, control_error0, control_ai = pullback.capture_with_attention_inputs(backend, control_batch)
    _control_donor_output, _cdcache, _cda, _control_donor_mlp, control_error1, _control_donor_ai = pullback.capture_with_attention_inputs(backend, control_donor_batch)
    _base_full_output, base_full = atlasrun.capture_native(backend, batch); _control_full_output, control_base_full = atlasrun.capture_native(backend, control_batch)
    base_state = atlasrun.states(torch, backend, base_output, rows); control_base_state = atlasrun.states(torch, backend, control_base_output, crows)
    full_output, _ = atlasrun.run_patch(backend, batch, donor_selected, comp.SITES); full_state = atlasrun.states(torch, backend, full_output, rows)

    self_output, self_ai, self_mlp, _self_counts = restricted_graph(backend, batch, donor_batch, base_full, top, source=False)
    graph_output, graph_ai, graph_mlp, source_counts = restricted_graph(backend, batch, donor_batch, base_full, top, source=True)
    _control_graph_output, control_graph_ai, control_graph_mlp, control_counts = restricted_graph(
        backend, control_batch, control_donor_batch, control_base_full, top, source=True)
    self_state = atlasrun.states(torch, backend, self_output, rows)
    self_error = max(float((self_state-base_state).abs().max()),
                     max(float((self_ai[layer]-base_ai[layer]).abs().max()) for layer in self_ai),
                     max(float((self_mlp[layer]-base_mlp[layer]).abs().max()) for layer in self_mlp))
    control_ctx = {"rows": crows, "base_logits": das.head_logits(backend, control_base_state).float()}

    reports = {}; coordinate_reports = {}; finite = [orientation, train_error, base_error, donor_error, control_error0, control_error1, self_error]
    for label, qs in projectors.items():
        train_ids = train_panels["A1" if label == "even_fit" else "A2"]; bases = {}
        for site, q in qs.items():
            if atlasrun.site_parts(site)[0] != "mlp": continue
            amap = covector.effective_map(backend, site, q, train_inputs)
            bases[site], _energy = covector.fit_input_basis(torch, train_batch, amap, train_ids, 64)
        full_coordinates = atlas.compiled_coordinates(backend, rows, qs, base_attention, base_ai, donor_ai, base_mlp, donor_mlp, bases)
        generated = atlas.compiled_coordinates(backend, rows, qs, base_attention, base_ai, graph_ai, base_mlp, graph_mlp, bases)
        coordinate = atlas.coordinate_report(torch, atlas.vectors(backend, batch, qs, full_coordinates), atlas.vectors(backend, batch, qs, generated))
        target_output = pruned.run_factors(backend, batch, _bcache, qs,
                                           {site: value for site, value in generated.items() if atlasrun.site_parts(site)[0] == "attn"},
                                           {site: value for site, value in generated.items() if atlasrun.site_parts(site)[0] == "mlp"},
                                           use_attention=True, use_mlp=True)
        target = ood.target_report(backend, rows, base_state, full_state, atlasrun.states(torch, backend, target_output, rows), reader, len(test_t))
        control_generated = atlas.compiled_coordinates(backend, crows, qs, control_attention, control_ai, control_graph_ai,
                                                       control_mlp, control_graph_mlp, bases)
        control_output = pruned.run_factors(backend, control_batch, _cbcache, qs,
                                            {site: value for site, value in control_generated.items() if atlasrun.site_parts(site)[0] == "attn"},
                                            {site: value for site, value in control_generated.items() if atlasrun.site_parts(site)[0] == "mlp"},
                                            use_attention=True, use_mlp=True)
        control_state = atlasrun.states(torch, backend, control_output, crows)
        control = klfit.control_metrics(backend, control_ctx, das.head_logits(backend, control_state).float())
        reports[label] = {"target": target, "control": control}; coordinate_reports[label] = coordinate
        finite += list(coordinate["signed_projection"].values()) + list(coordinate["residual"].values()) + list(target["cells"].values()) + list(target["behavior_signed_projection"].values()) + list(control["margin_rms_fraction"].values()) + [control["median_kl"], control["max_kl"], control["top1_flip_fraction"]]

    pa = hashes_ok and reader_ok and orientation <= 1e-6 and top_agree and self_error <= 1e-4 and set(source_counts) == {1} and all(math.isfinite(float(value)) for value in finite)
    pb = all(min(report["signed_projection"].values()) >= .75 and report["mean_residual"] <= .2 for report in coordinate_reports.values())
    pc = all(report["target"]["worst_target_residual"] <= .15 and min(report["target"]["behavior_signed_projection"].values()) >= .8 for report in reports.values())
    pd = all(max(report["control"]["margin_rms_fraction"].values()) <= .1 and report["control"]["median_kl"] <= .02 and report["control"]["top1_flip_fraction"] <= .05 for report in reports.values())
    mincoord = {label: min(report["signed_projection"].values()) for label, report in coordinate_reports.items()}
    projection_diff = max(abs(reports["even_fit"]["target"]["behavior_signed_projection"][task] - reports["odd_fit"]["target"]["behavior_signed_projection"][task]) for task in ("temporal", "iswas"))
    pe = abs(mincoord["even_fit"]-mincoord["odd_fit"]) <= .05 and projection_diff <= .05
    predictions = {"pred_a_authority_top20_self_clamp_embedding_scope_finiteness_and_price": bool(pa),
                   "pred_b_source_clamped_graph_generates_compiled_coordinates": bool(pb),
                   "pred_c_source_clamped_compiled_program_is_target_sufficient": bool(pc),
                   "pred_d_source_clamped_compiled_program_is_selective": bool(pd),
                   "pred_e_source_clamped_graph_is_crossfit_stable": bool(pe)}
    terminal = "invalid" if not pa else "source_to_reader_compiled_tensor_program" if all(predictions.values()) else "source_clamped_graph_incomplete"
    summary = {"self_clamp_max_abs": self_error, "source_change_count": {"min": min(source_counts), "max": max(source_counts)},
               "control_source_change_count": {"min": min(control_counts), "max": max(control_counts)},
               "coordinate_projection_min": min(mincoord.values()), "coordinate_mean_residual_max": max(report["mean_residual"] for report in coordinate_reports.values()),
               "target_worst_max": max(report["target"]["worst_target_residual"] for report in reports.values()),
               "target_projection_min": min(min(report["target"]["behavior_signed_projection"].values()) for report in reports.values()),
               "control_margin_max": max(max(report["control"]["margin_rms_fraction"].values()) for report in reports.values()),
               "control_median_kl_max": max(report["control"]["median_kl"] for report in reports.values()),
               "control_flip_max": max(report["control"]["top1_flip_fraction"] for report in reports.values()),
               "crossfit_projection_max_abs_diff": projection_diff}
    result = {"schema": "temporal_five_mlp_source_clamped_compiled_graph_result_v1", "started_utc": started, "finished_utc": now(),
              "serial_seconds": time.perf_counter()-tic, "authority_sha256": EXPECTED, "top20": top,
              "coordinate_reports": coordinate_reports, "reports": reports, "summary": summary, "predictions": predictions,
              "terminal": terminal, "price": {"model_forwards": 23, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}}
    atomic_create_json(OUT, result); print(json.dumps({k: result[k] for k in ("summary", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
