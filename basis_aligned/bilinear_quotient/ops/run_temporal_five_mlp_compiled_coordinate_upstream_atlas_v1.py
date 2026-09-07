#!/usr/bin/env python3
"""Full component-patch atlas scored against the weight-compiled response coordinates."""
# BQGATE: EXPERIMENT pred_a_authority_alignment_self_patch_finiteness_and_price pred_b_compiled_coordinate_instrument_closes_on_full_donor pred_c_known_response_frontier_is_enriched pred_d_compiled_coordinate_incidence_predicts_native_causality pred_e_top20_joint_patch_is_coordinate_sufficient
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
import run_temporal_five_mlp_factor_pruned_causal_program_v1 as pruned
import run_temporal_five_mlp_mlp_input_covector_crossfit_v1 as covector
import run_temporal_five_mlp_mlp_input_covector_ood_v2 as covood
import run_temporal_five_mlp_attention_value_weight_pullback_v1 as pullback

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_five_mlp_compiled_coordinate_upstream_atlas_v1.json"
ATTENTION_AUDIT = ROOT / "circuits/followups/temporal_five_mlp_attention_value_weight_pullback_v2_tolerance_audit_result.json"
MLP_OOD = ROOT / "circuits/followups/temporal_five_mlp_mlp_input_covector_ood_v2_result.json"
PULLBACK_RUNNER = ROOT / "ops/run_temporal_five_mlp_attention_value_weight_pullback_v1.py"
COVECTOR_RUNNER = ROOT / "ops/run_temporal_five_mlp_mlp_input_covector_ood_v2.py"
INTERFACE = ROOT / "circuits/followups/temporal_five_mlp_crossfit_response_weight_interface_v1_result.json"
TC = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v13_capability_v1_result.json"
IC = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v12_capability_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_five_mlp_compiled_coordinate_upstream_atlas_v1_result.json"
EXPECTED = {
    "prior": "8e0bc300111c008c76ece4a3880e7f7c2573174622d94d6b93a800e0e4798faf",
    "attention_audit": "42f9db9cf44e63a1930f454798b9d25b115cea94e4c45a8344694da6982637b4",
    "mlp_ood": "10a6e169348cffb001f14d299ce05c3f33d80bd50e574dced2cac1228d436947",
    "pullback_runner": "f60d07e8a52b4bb3cb0228ecf41d044548f063f1266f53328c5443206eed7dfc",
    "covector_runner": "ca5ac0b74533d110c98415d1fa147e61f0f1c04036260898d181fddf8528c221",
    "interface": "578fa690f2f3962409bb50bd40a1c4c6f0b5ce7f88828e2e889a3a47bb20f013",
    "temporal_capability": "e053d3381680ce5a933356a060448466d7567e3079c4a2b9a5bff262bd98b9c1",
    "iswas_capability": "67cb3efbd1ea86f98f94a826922928229a7c7b0a247f218778fc4960a6e8c6f4",
}
CANDIDATES = tuple(site for layer in range(11) for site in tuple(f"L{layer}H{head}" for head in range(9)) + (f"MLP{layer}",))
KNOWN = ("L8H1", "L9H1", "L9H4", "MLP1", "MLP3", "MLP4", "MLP6")


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def patch_and_capture_inputs(backend, batch, source, sites):
    attention = {}; mlps = {}; handles = []
    attn_layers = sorted({0} | {atlasrun.site_parts(site)[1] for site in comp.SITES if atlasrun.site_parts(site)[0] == "attn"})
    mlp_layers = sorted({atlasrun.site_parts(site)[1] for site in comp.SITES if atlasrun.site_parts(site)[0] == "mlp"})
    for layer in attn_layers:
        def save_attention(_module, arguments, layer=layer): attention[layer] = arguments[0].detach().clone()
        handles.append(backend.model.transformer.h[layer].attn.register_forward_pre_hook(save_attention))
    for layer in mlp_layers:
        def save_mlp(_module, arguments, layer=layer): mlps[layer] = arguments[0].detach().clone()
        handles.append(backend.model.transformer.h[layer].mlp.register_forward_pre_hook(save_mlp))
    try: output, _io = atlasrun.run_patch(backend, batch, source, sites)
    finally:
        for handle in handles: handle.remove()
    if set(attention) != set(attn_layers) or set(mlps) != set(mlp_layers): raise RuntimeError("incomplete patched input capture")
    return output, attention, mlps


def compiled_coordinates(backend, rows, qs, base_attention, base_attn_inputs, changed_attn_inputs,
                         base_mlp_inputs, changed_mlp_inputs, mlp_bases):
    attention = pullback.compiled_attention(backend, rows, qs, base_attention, base_attn_inputs,
                                            changed_attn_inputs, current=True, cached=False)
    mlps = covood.compiled_mlps(backend, qs, base_mlp_inputs, changed_mlp_inputs, mlp_bases)
    return {**attention, **mlps}


def vectors(backend, batch, qs, coordinates):
    output = {}
    for site, q in qs.items():
        parts = []
        for i, pos in enumerate(batch.semantic_positions): parts.append(coordinates[site][i, :int(pos)+1] @ q)
        output[site] = backend.torch.cat(parts)
    return output


def coordinate_report(torch, reference, changed):
    cells = {}; projections = {}
    for site in reference:
        a, b = reference[site], changed[site]; den = a.square().sum().clamp_min(1e-30)
        projections[site] = float((b * a).sum() / den)
        cells[site] = float((b-a).square().sum() / den)
    return {"signed_projection": projections, "residual": cells,
            "magnitude": sum(abs(value) for value in projections.values()),
            "mean_residual": sum(cells.values()) / len(cells), "worst_residual": max(cells.values())}


def causal_report(backend, rows, base_state, donor_state, state):
    torch = backend.torch; base = comp.margins(backend, base_state, rows); full = comp.margins(backend, donor_state, rows) - base
    changed = comp.margins(backend, state, rows) - base; values = {}
    for task in ("temporal", "iswas"):
        ids = torch.as_tensor([i for i, row in enumerate(rows) if klfit.task_name(row) == task], device=backend.device)
        values[task] = float((changed[ids] * full[ids]).sum() / full[ids].square().sum().clamp_min(1e-30))
    return {"signed_projection": values, "magnitude": sum(abs(value) for value in values.values())}


def main():
    paths = {"prior": PRIOR, "attention_audit": ATTENTION_AUDIT, "mlp_ood": MLP_OOD,
             "pullback_runner": PULLBACK_RUNNER, "covector_runner": COVECTOR_RUNNER,
             "interface": INTERFACE, "temporal_capability": TC, "iswas_capability": IC}
    observed = {key: sha(path) for key, path in paths.items()}
    if observed != EXPECTED: raise RuntimeError(f"compiled-coordinate atlas authority changed: {observed}")
    dry = {"candidate_id": "temporal_auxiliary.five_mlp_compiled_coordinate_upstream_atlas_v1", "dryrun": True,
           "gpu_accessed": False, "model_loaded": False, "queue_touched": False, "candidate_count": len(CANDIDATES),
           "model_forwards_max": 128, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1": print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    if json.loads(ATTENTION_AUDIT.read_text())["terminal"] != "attention_current_value_weight_program" or json.loads(MLP_OOD.read_text())["terminal"] != "ood_weight_derived_input_reader": raise RuntimeError("compiled coordinate authorities not promoted")
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
    _tro, _trcache, _tra, train_mlp_inputs, train_error, _train_attn_inputs = pullback.capture_with_attention_inputs(backend, train_batch)
    train_panels = {panel: [i for i, row in enumerate(train_rows) if row["transform_id"] == panel] for panel in ("A1", "A2")}

    tcap, icap = json.loads(TC.read_text()), json.loads(IC.read_text())
    test_t = sum((population.capable_rows(fresh_t, tcap, panel, 12) for panel in ("A1", "A2")), [])
    test_i = sum((population.capable_rows(fresh_i, icap, panel, 12) for panel in ("A1", "A2")), [])
    rows = test_t + test_i; batch = das._batch(backend, rows, side="base"); donor_batch = das._batch(backend, rows, side="donor")
    base_output, _base_selected, base_attention, base_mlp_inputs, base_error, base_attn_inputs = pullback.capture_with_attention_inputs(backend, batch)
    donor_output, _donor_selected, _donor_attention, donor_mlp_inputs, donor_error, donor_attn_inputs = pullback.capture_with_attention_inputs(backend, donor_batch)
    _base_full_output, base_full = atlasrun.capture_native(backend, batch); _donor_full_output, donor_full = atlasrun.capture_native(backend, donor_batch)
    base_state = atlasrun.states(torch, backend, base_output, rows); donor_state = atlasrun.states(torch, backend, donor_output, rows)

    self_output, self_ai, self_mi = patch_and_capture_inputs(backend, batch, base_full, CANDIDATES)
    self_state = atlasrun.states(torch, backend, self_output, rows)
    self_error = max(float((self_state-base_state).abs().max()),
                     max(float((self_ai[layer]-base_attn_inputs[layer]).abs().max()) for layer in self_ai),
                     max(float((self_mi[layer]-base_mlp_inputs[layer]).abs().max()) for layer in self_mi))
    reports = {label: {} for label in projectors}; rankings = {}; top_reports = {}; all_scores = []; all_causal = []; reference_norms = {}; fit_data = {}
    for label, qs in projectors.items():
        train_ids = train_panels["A1" if label == "even_fit" else "A2"]; mlp_bases = {}
        for site, q in qs.items():
            if atlasrun.site_parts(site)[0] != "mlp": continue
            amap = covector.effective_map(backend, site, q, train_mlp_inputs)
            mlp_bases[site], _energy = covector.fit_input_basis(torch, train_batch, amap, train_ids, 64)
        full_coordinates = compiled_coordinates(backend, rows, qs, base_attention, base_attn_inputs, donor_attn_inputs,
                                                base_mlp_inputs, donor_mlp_inputs, mlp_bases)
        reference = vectors(backend, batch, qs, full_coordinates)
        reference_norms[label] = {site: float(value.square().sum()) for site, value in reference.items()}
        fit_data[label] = {"qs": qs, "mlp_bases": mlp_bases, "reference": reference}
    for candidate in CANDIDATES:
        output, changed_ai, changed_mi = patch_and_capture_inputs(backend, batch, donor_full, (candidate,))
        state = atlasrun.states(torch, backend, output, rows); causal = causal_report(backend, rows, base_state, donor_state, state)
        for label, data in fit_data.items():
            qs, mlp_bases, reference = data["qs"], data["mlp_bases"], data["reference"]
            coordinates = compiled_coordinates(backend, rows, qs, base_attention, base_attn_inputs, changed_ai,
                                               base_mlp_inputs, changed_mi, mlp_bases)
            creport = coordinate_report(torch, reference, vectors(backend, batch, qs, coordinates))
            reports[label][candidate] = {"coordinate": creport, "native_causal": causal}
        del output, changed_ai, changed_mi, state
    for label, data in fit_data.items():
        qs, mlp_bases, reference = data["qs"], data["mlp_bases"], data["reference"]
        site_records = reports[label]
        ranking = sorted(CANDIDATES, key=lambda site: (-site_records[site]["coordinate"]["magnitude"], CANDIDATES.index(site)))
        top = tuple(ranking[:20]); top_output, top_ai, top_mi = patch_and_capture_inputs(backend, batch, donor_full, top)
        top_coordinates = compiled_coordinates(backend, rows, qs, base_attention, base_attn_inputs, top_ai,
                                               base_mlp_inputs, top_mi, mlp_bases)
        top_coordinate = coordinate_report(torch, reference, vectors(backend, batch, qs, top_coordinates))
        top_causal = causal_report(backend, rows, base_state, donor_state, atlasrun.states(torch, backend, top_output, rows))
        rankings[label] = ranking; top_reports[label] = {"sites": list(top), "coordinate": top_coordinate, "native_causal": top_causal}
        all_scores += [site_records[site]["coordinate"]["magnitude"] for site in CANDIDATES]
        all_causal += [site_records[site]["native_causal"]["magnitude"] for site in CANDIDATES]

    finite = [orientation, train_error, base_error, donor_error, self_error, *all_scores, *all_causal]
    pa = hashes_ok and reader_ok and orientation <= 1e-6 and self_error <= 1e-4 and all(math.isfinite(float(value)) for value in finite)
    pb = all(value > 0 for fit in reference_norms.values() for value in fit.values())
    pc = all(sum(site in ranking[:20] for site in KNOWN) >= 4 for ranking in rankings.values())
    correlation = atlasrun.spearman(all_scores, all_causal); pd = correlation >= .5
    pe = all(min(report["coordinate"]["signed_projection"].values()) >= .8 and report["coordinate"]["mean_residual"] <= .2 for report in top_reports.values())
    predictions = {"pred_a_authority_alignment_self_patch_finiteness_and_price": bool(pa),
                   "pred_b_compiled_coordinate_instrument_closes_on_full_donor": bool(pb),
                   "pred_c_known_response_frontier_is_enriched": bool(pc),
                   "pred_d_compiled_coordinate_incidence_predicts_native_causality": bool(pd),
                   "pred_e_top20_joint_patch_is_coordinate_sufficient": bool(pe)}
    terminal = "invalid" if not pa else "compiled_coordinate_upstream_pool" if all(predictions.values()) else "compiled_coordinate_writer_atlas_mixed"
    summary = {"candidate_count": len(CANDIDATES), "self_patch_max_abs": self_error, "known_in_top20": {label: [site for site in ranking[:20] if site in KNOWN] for label, ranking in rankings.items()},
               "coordinate_native_causal_spearman": correlation,
               "top20_projection_min": min(min(report["coordinate"]["signed_projection"].values()) for report in top_reports.values()),
               "top20_mean_residual_max": max(report["coordinate"]["mean_residual"] for report in top_reports.values()),
               "top20_overlap": len(set(top_reports["even_fit"]["sites"]) & set(top_reports["odd_fit"]["sites"]))}
    result = {"schema": "temporal_five_mlp_compiled_coordinate_upstream_atlas_result_v1", "started_utc": started, "finished_utc": now(),
              "serial_seconds": time.perf_counter()-tic, "authority_sha256": EXPECTED, "rankings": rankings, "top20": top_reports,
              "site_reports": reports, "summary": summary, "predictions": predictions, "terminal": terminal,
              "price": {"model_forwards": 126, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}}
    atomic_create_json(OUT, result); print(json.dumps({k: result[k] for k in ("summary", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
