#!/usr/bin/env python3
"""Test the changed cue embedding as the omitted direct source of the MLP1 coordinate."""
# BQGATE: EXPERIMENT pred_a_authority_top20_agreement_embedding_scope_finiteness_and_price pred_b_cue_embedding_closes_missing_mlp1_coordinate pred_c_top20_plus_embedding_is_coordinate_sufficient pred_d_top20_plus_embedding_is_behaviorally_sufficient_and_selective pred_e_embedding_is_a_material_direct_mlp1_source
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
import run_temporal_five_mlp_upstream_tensor_ranked_greedy_program_v1 as greedy
import run_temporal_five_mlp_crossfit_response_weight_interface_v1 as interface
import run_temporal_five_mlp_family_feasible_kl_refinement_v1 as klfit
import run_temporal_five_mlp_attention_value_weight_pullback_v1 as pullback
import run_temporal_five_mlp_mlp_input_covector_crossfit_v1 as covector
import run_temporal_five_mlp_compiled_coordinate_upstream_atlas_v1 as atlas

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_five_mlp_compiled_coordinate_embedding_source_v1.json"
ATLAS_RESULT = ROOT / "circuits/followups/temporal_five_mlp_compiled_coordinate_upstream_atlas_v1_result.json"
ATLAS_RUNNER = ROOT / "ops/run_temporal_five_mlp_compiled_coordinate_upstream_atlas_v1.py"
ATTENTION_AUDIT = ROOT / "circuits/followups/temporal_five_mlp_attention_value_weight_pullback_v2_tolerance_audit_result.json"
MLP_OOD = ROOT / "circuits/followups/temporal_five_mlp_mlp_input_covector_ood_v2_result.json"
INTERFACE = ROOT / "circuits/followups/temporal_five_mlp_crossfit_response_weight_interface_v1_result.json"
TC = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v13_capability_v1_result.json"
IC = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v12_capability_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_five_mlp_compiled_coordinate_embedding_source_v1_result.json"
EXPECTED = {
    "prior": "430e8d7afea2d67f4e0c6051130ab550215a6d647714b33bf918b36e96740c0b",
    "atlas_result": "0b96284ebf7bd89ccdbf1dbc0fbdacd54e58eb16841865c48b985aa1a3c2da53",
    "atlas_runner": "25732b607a8190a391668e824b0bd4183f7aa9182f53e80d9ca0cb96cfeceec5",
    "attention_audit": "42f9db9cf44e63a1930f454798b9d25b115cea94e4c45a8344694da6982637b4",
    "mlp_ood": "10a6e169348cffb001f14d299ce05c3f33d80bd50e574dced2cac1228d436947",
    "interface": "578fa690f2f3962409bb50bd40a1c4c6f0b5ce7f88828e2e889a3a47bb20f013",
    "temporal_capability": "e053d3381680ce5a933356a060448466d7567e3079c4a2b9a5bff262bd98b9c1",
    "iswas_capability": "67cb3efbd1ea86f98f94a826922928229a7c7b0a247f218778fc4960a6e8c6f4",
}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def patch_with_embedding(backend, base_batch, donor_batch, source, sites, *, embedding):
    handle = None; counts = []
    if embedding:
        base_tokens, _ = backend._tensor_batch(base_batch); donor_tokens, _ = backend._tensor_batch(donor_batch)
        mask = base_tokens != donor_tokens; counts = [int(row.sum()) for row in mask]
        with backend.torch.no_grad(): donor_embedding = backend.model.transformer.wte(donor_tokens).detach().clone()
        def hook(_module, _arguments, output):
            changed = output.clone(); changed[mask] = donor_embedding[mask].to(changed); return changed
        handle = backend.model.transformer.wte.register_forward_hook(hook)
    try: output, attention, mlps = atlas.patch_and_capture_inputs(backend, base_batch, source, sites)
    finally:
        if handle is not None: handle.remove()
    return output, attention, mlps, counts


def main():
    paths = {"prior": PRIOR, "atlas_result": ATLAS_RESULT, "atlas_runner": ATLAS_RUNNER,
             "attention_audit": ATTENTION_AUDIT, "mlp_ood": MLP_OOD, "interface": INTERFACE,
             "temporal_capability": TC, "iswas_capability": IC}
    observed = {key: sha(path) for key, path in paths.items()}
    if observed != EXPECTED: raise RuntimeError(f"embedding-source authority changed: {observed}")
    dry = {"candidate_id": "temporal_auxiliary.five_mlp_compiled_coordinate_embedding_source_v1", "dryrun": True,
           "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
           "model_forwards_max": 24, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1": print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    upstream = json.loads(ATLAS_RESULT.read_text()); top = upstream["top20"]["even_fit"]["sites"]
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
    donor_output, _dcache, _da, donor_mlp, donor_error, donor_ai = pullback.capture_with_attention_inputs(backend, donor_batch)
    control_base_output, _cbcache, _cba, _cbm, control_error0, _cbai = pullback.capture_with_attention_inputs(backend, control_batch)
    _control_donor_output, _cdcache, _cda, _cdm, control_error1, _cdai = pullback.capture_with_attention_inputs(backend, control_donor_batch)
    _donor_full_output, donor_full = atlasrun.capture_native(backend, donor_batch)
    _control_full_output, control_donor_full = atlasrun.capture_native(backend, control_donor_batch)
    base_state = atlasrun.states(torch, backend, base_output, rows); donor_state = atlasrun.states(torch, backend, donor_output, rows)
    control_base_state = atlasrun.states(torch, backend, control_base_output, crows)
    control_ctx = {"rows": crows, "base_logits": das.head_logits(backend, control_base_state).float()}

    target_runs = {}
    for name, sites, embedding in (("top20", top, False), ("embedding_only", (), True), ("top20_plus_embedding", top, True)):
        target_runs[name] = patch_with_embedding(backend, batch, donor_batch, donor_full, sites, embedding=embedding)
    control_output, _control_ai, _control_mi, control_counts = patch_with_embedding(
        backend, control_batch, control_donor_batch, control_donor_full, top, embedding=True)
    control_state = atlasrun.states(torch, backend, control_output, crows)
    control = klfit.control_metrics(backend, control_ctx, das.head_logits(backend, control_state).float())

    reports = {}; finite = [orientation, train_error, base_error, donor_error, control_error0, control_error1]
    target_causal = atlas.causal_report(backend, rows, base_state, donor_state,
                                        atlasrun.states(torch, backend, target_runs["top20_plus_embedding"][0], rows))
    for label, qs in projectors.items():
        train_ids = train_panels["A1" if label == "even_fit" else "A2"]; bases = {}
        for site, q in qs.items():
            if atlasrun.site_parts(site)[0] != "mlp": continue
            amap = covector.effective_map(backend, site, q, train_inputs)
            bases[site], _energy = covector.fit_input_basis(torch, train_batch, amap, train_ids, 64)
        full = atlas.vectors(backend, batch, qs, atlas.compiled_coordinates(
            backend, rows, qs, base_attention, base_ai, donor_ai, base_mlp, donor_mlp, bases))
        arms = {}
        for name, (_output, changed_ai, changed_mlp, counts) in target_runs.items():
            coordinates = atlas.compiled_coordinates(backend, rows, qs, base_attention, base_ai, changed_ai,
                                                     base_mlp, changed_mlp, bases)
            arms[name] = atlas.coordinate_report(torch, full, atlas.vectors(backend, batch, qs, coordinates))
            finite += list(arms[name]["signed_projection"].values()) + list(arms[name]["residual"].values())
        reports[label] = arms

    target_counts = target_runs["embedding_only"][3]
    pa = hashes_ok and reader_ok and orientation <= 1e-6 and top_agree and target_counts and set(target_counts) == {1} and all(math.isfinite(float(value)) for value in finite)
    pb = all(report["top20_plus_embedding"]["signed_projection"]["MLP1"] - report["top20"]["signed_projection"]["MLP1"] >= .15 and report["top20_plus_embedding"]["signed_projection"]["MLP1"] >= .85 for report in reports.values())
    pc = all(min(report["top20_plus_embedding"]["signed_projection"].values()) >= .8 and report["top20_plus_embedding"]["mean_residual"] <= .1 for report in reports.values())
    pd = min(target_causal["signed_projection"].values()) >= .9 and max(control["margin_rms_fraction"].values()) <= .1 and control["median_kl"] <= .02 and control["top1_flip_fraction"] <= .05
    pe = all(report["embedding_only"]["signed_projection"]["MLP1"] >= .15 for report in reports.values())
    predictions = {"pred_a_authority_top20_agreement_embedding_scope_finiteness_and_price": bool(pa),
                   "pred_b_cue_embedding_closes_missing_mlp1_coordinate": bool(pb),
                   "pred_c_top20_plus_embedding_is_coordinate_sufficient": bool(pc),
                   "pred_d_top20_plus_embedding_is_behaviorally_sufficient_and_selective": bool(pd),
                   "pred_e_embedding_is_a_material_direct_mlp1_source": bool(pe)}
    terminal = "invalid" if not pa else "embedding_complete_upstream_pool" if all(predictions.values()) else "embedding_source_null"
    summary = {"target_embedding_change_count": {"min": min(target_counts), "max": max(target_counts)},
               "control_embedding_change_count": {"min": min(control_counts), "max": max(control_counts)},
               "mlp1_projection": {label: {arm: report[arm]["signed_projection"]["MLP1"] for arm in report} for label, report in reports.items()},
               "combined_projection_min": min(min(report["top20_plus_embedding"]["signed_projection"].values()) for report in reports.values()),
               "combined_mean_residual_max": max(report["top20_plus_embedding"]["mean_residual"] for report in reports.values()),
               "native_target_projection_min": min(target_causal["signed_projection"].values()),
               "control_margin_max": max(control["margin_rms_fraction"].values()), "control_median_kl": control["median_kl"],
               "control_flip_fraction": control["top1_flip_fraction"]}
    result = {"schema": "temporal_five_mlp_compiled_coordinate_embedding_source_result_v1", "started_utc": started, "finished_utc": now(),
              "serial_seconds": time.perf_counter()-tic, "authority_sha256": EXPECTED, "top20": top, "reports": reports,
              "native_target": target_causal, "control": control, "summary": summary, "predictions": predictions, "terminal": terminal,
              "price": {"model_forwards": 19, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}}
    atomic_create_json(OUT, result); print(json.dumps({k: result[k] for k in ("summary", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
