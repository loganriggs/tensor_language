#!/usr/bin/env python3
"""Prospective OOD confirmation of the frozen rank-64 MLP input-covector program."""
# BQGATE: EXPERIMENT pred_a_authority_hash_disjointness_exact_closure_finiteness_and_price pred_b_rank64_ood_coefficient_prediction pred_c_rank64_ood_program_is_target_sufficient pred_d_rank64_ood_program_is_selective pred_e_rank64_ood_program_survives_input_noise
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

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_five_mlp_mlp_input_covector_ood_v2.json"
COV_RESULT = ROOT / "circuits/followups/temporal_five_mlp_mlp_input_covector_crossfit_v1_result.json"
COV_RUNNER = ROOT / "ops/run_temporal_five_mlp_mlp_input_covector_crossfit_v1.py"
TB = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v13.py"
TC = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v13_capability_v1_result.json"
IB = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v12.py"
IC = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v12_capability_v1_result.json"
INTERFACE = ROOT / "circuits/followups/temporal_five_mlp_crossfit_response_weight_interface_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_five_mlp_mlp_input_covector_ood_v2_result.json"
EXPECTED = {
    "prior": "d684abb3956f4de35680939c4726c0914f3d3a24a121f1ae5cff3ec98b7cb9eb",
    "covector_result": "bcd1078234e569114b3278f8cb4626513ef5f883995789e623a0369ab79e4214",
    "covector_runner": "07040ed211fef63c51103d8a18976b81ab94e38301d8c651084249408946aee1",
    "temporal_builder": "3f738bf2fb2d4a5425dba85eaf948d7ed888e4b891666ff77a51c8638acd2509",
    "temporal_capability": "e053d3381680ce5a933356a060448466d7567e3079c4a2b9a5bff262bd98b9c1",
    "iswas_builder": "2734cbeceb4e6979dab22fe5b24870386874ac2f905ae426e6e689548e43e8a2",
    "iswas_capability": "67cb3efbd1ea86f98f94a826922928229a7c7b0a247f218778fc4960a6e8c6f4",
    "interface": "578fa690f2f3962409bb50bd40a1c4c6f0b5ce7f88828e2e889a3a47bb20f013",
}
NOISE_SEEDS = (2901, 2902, 2903)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def compiled_mlps(backend, qs, base_inputs, donor_inputs, bases, *, noise_seed=None):
    torch = backend.torch; output = {}
    for site, q in qs.items():
        if atlasrun.site_parts(site)[0] != "mlp": continue
        layer = atlasrun.site_parts(site)[1]; delta = donor_inputs[layer] - base_inputs[layer]
        if noise_seed is not None:
            generator = torch.Generator(device=backend.device).manual_seed(noise_seed + 97 * layer)
            scale = delta.float().square().mean().sqrt().clamp_min(1e-12) * .10
            delta = delta + torch.randn(delta.shape, device=delta.device, dtype=delta.dtype, generator=generator) * scale
        amap = covector.effective_map(backend, site, q, base_inputs)
        output[site] = covector.coordinates(torch, delta, amap, bases[site]) @ q.T
    return output


def main():
    paths = {"prior": PRIOR, "covector_result": COV_RESULT, "covector_runner": COV_RUNNER,
             "temporal_builder": TB, "temporal_capability": TC, "iswas_builder": IB,
             "iswas_capability": IC, "interface": INTERFACE}
    observed = {key: sha(path) for key, path in paths.items()}
    if observed != EXPECTED: raise RuntimeError(f"rank64 OOD authority changed: {observed}")
    dry = {"candidate_id": "temporal_auxiliary.five_mlp_mlp_input_covector_ood_v2", "dryrun": True,
           "gpu_accessed": False, "model_loaded": False, "queue_touched": False, "rank": 64,
           "noise_seeds": NOISE_SEEDS, "model_forwards_max": 30, "fit_updates": 0,
           "model_updates": 0, "transformer_backwards": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1": print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    if json.loads(COV_RESULT.read_text())["terminal"] != "input_covector_compression_incomplete": raise RuntimeError("covector frontier terminal changed")
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
    _train_out, _train_cache, _train_attention, train_inputs, train_error = pruned.capture_everything(backend, train_batch)
    train_panels = {panel: [i for i, row in enumerate(train_rows) if row["transform_id"] == panel] for panel in ("A1", "A2")}

    tcap, icap = json.loads(TC.read_text()), json.loads(IC.read_text())
    test_t = sum((population.capable_rows(fresh_t, tcap, panel, 12) for panel in ("A1", "A2")), [])
    test_i = sum((population.capable_rows(fresh_i, icap, panel, 12) for panel in ("A1", "A2")), [])
    rows = test_t + test_i; crows = [row for row in fresh_t.build_rows() if row["transform_id"] == "P"][:16]
    train_text = {row[key] for row in train_rows for key in ("base_text", "donor_text")}
    test_text = {row[key] for row in rows for key in ("base_text", "donor_text")}
    disjoint = train_text.isdisjoint(test_text)
    tb, td = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
    cb, cd = das._batch(backend, crows, side="base"), das._batch(backend, crows, side="donor")
    tbo, tbase, tba, tbi, terr0 = pruned.capture_everything(backend, tb)
    _tdo, tdonor, tda, tdi, terr1 = pruned.capture_everything(backend, td)
    cbo, cbase, cba, cbi, cerr0 = pruned.capture_everything(backend, cb)
    _cdo, _cdonor, cda, cdi, cerr1 = pruned.capture_everything(backend, cd)
    base_target = atlasrun.states(torch, backend, tbo, rows); base_control = atlasrun.states(torch, backend, cbo, crows)
    full_out, _ = atlasrun.run_patch(backend, tb, tdonor, comp.SITES); full_target = atlasrun.states(torch, backend, full_out, rows)
    exact_attention, exact_mlp = pruned.factor_deltas(backend, rows, tba, tda, tbi, tdi, include_exception=False)
    control_attention, _control_mlp = pruned.factor_deltas(backend, crows, cba, cda, cbi, cdi, include_exception=False)
    control_ctx = {"rows": crows, "base_logits": das.head_logits(backend, base_control).float()}

    reports = {}; closure = []
    for label, qs in projectors.items():
        train_ids = train_panels["A1" if label == "even_fit" else "A2"]; bases = {}; coefficient = {}
        for site, q in qs.items():
            if atlasrun.site_parts(site)[0] != "mlp": continue
            train_map = covector.effective_map(backend, site, q, train_inputs)
            basis, energy = covector.fit_input_basis(torch, train_batch, train_map, train_ids, 64); bases[site] = basis
            layer = atlasrun.site_parts(site)[1]; amap = covector.effective_map(backend, site, q, tbi); delta = tdi[layer] - tbi[layer]
            exact = covector.coordinates(torch, delta, amap); predicted = covector.coordinates(torch, delta, amap, basis)
            closure.append(covector.coefficient_rse(torch, tb, exact_mlp[site] @ q, exact, list(range(len(rows)))))
            coefficient[site] = {"training_energy": energy, "ood_rse": {
                task: covector.coefficient_rse(torch, tb, exact, predicted, [i for i, row in enumerate(rows) if klfit.task_name(row) == task])
                for task in ("temporal", "iswas")}}
        clean_mlp = compiled_mlps(backend, qs, tbi, tdi, bases)
        clean_out = pruned.run_factors(backend, tb, tbase, qs, exact_attention, clean_mlp, use_attention=True, use_mlp=True)
        clean = pruned.serial_report(backend, rows, clean_out, base_target, full_target, reader, len(test_t))
        noise = {}
        for seed in NOISE_SEEDS:
            noisy_mlp = compiled_mlps(backend, qs, tbi, tdi, bases, noise_seed=seed)
            noisy_out = pruned.run_factors(backend, tb, tbase, qs, exact_attention, noisy_mlp, use_attention=True, use_mlp=True)
            noise[str(seed)] = pruned.serial_report(backend, rows, noisy_out, base_target, full_target, reader, len(test_t))
        control_mlp = compiled_mlps(backend, qs, cbi, cdi, bases)
        control_out = pruned.run_factors(backend, cb, cbase, qs, control_attention, control_mlp, use_attention=True, use_mlp=True)
        control_state = atlasrun.states(torch, backend, control_out, crows)
        control = klfit.control_metrics(backend, control_ctx, das.head_logits(backend, control_state).float())
        reports[label] = {"training_panel": "A1" if label == "even_fit" else "A2", "coefficient": coefficient,
                          "clean_target": clean, "noise_target": noise, "control": control}

    finite = [orientation, train_error, terr0, terr1, cerr0, cerr1] + closure
    for report in reports.values():
        for item in report["coefficient"].values(): finite += [item["training_energy"]] + list(item["ood_rse"].values())
        for target in [report["clean_target"], *report["noise_target"].values()]: finite += list(target["cells"].values()) + list(target["behavior_signed_projection"].values())
        finite += list(report["control"]["margin_rms_fraction"].values()) + [report["control"]["median_kl"], report["control"]["max_kl"], report["control"]["top1_flip_fraction"]]
    pa = hashes_ok and reader_ok and orientation <= 1e-6 and disjoint and max(closure) <= 1e-10 and max(train_error, terr0, terr1, cerr0, cerr1) <= 5e-4 and all(math.isfinite(float(x)) for x in finite)
    pb = all(item["ood_rse"][task] <= .1 for report in reports.values() for item in report["coefficient"].values() for task in ("temporal", "iswas"))
    pc = all(report["clean_target"]["worst_target_residual"] <= .15 and min(report["clean_target"]["behavior_signed_projection"].values()) >= .8 for report in reports.values())
    pd = all(max(report["control"]["margin_rms_fraction"].values()) <= .1 and report["control"]["median_kl"] <= .02 and report["control"]["top1_flip_fraction"] <= .05 for report in reports.values())
    pe = all(target["worst_target_residual"] <= .2 and min(target["behavior_signed_projection"].values()) >= .75 for report in reports.values() for target in report["noise_target"].values())
    predictions = {"pred_a_authority_hash_disjointness_exact_closure_finiteness_and_price": bool(pa),
                   "pred_b_rank64_ood_coefficient_prediction": bool(pb),
                   "pred_c_rank64_ood_program_is_target_sufficient": bool(pc),
                   "pred_d_rank64_ood_program_is_selective": bool(pd),
                   "pred_e_rank64_ood_program_survives_input_noise": bool(pe)}
    terminal = "invalid" if not pa else "ood_weight_derived_input_reader" if all(predictions.values()) else "rank64_ood_input_reader_null"
    summary = {"max_exact_covector_closure_rse": max(closure),
               "max_ood_coefficient_rse": max(item["ood_rse"][task] for report in reports.values() for item in report["coefficient"].values() for task in ("temporal", "iswas")),
               "clean_target_worst_max": max(report["clean_target"]["worst_target_residual"] for report in reports.values()),
               "clean_target_projection_min": min(min(report["clean_target"]["behavior_signed_projection"].values()) for report in reports.values()),
               "noise_target_worst_max": max(target["worst_target_residual"] for report in reports.values() for target in report["noise_target"].values()),
               "noise_target_projection_min": min(min(target["behavior_signed_projection"].values()) for report in reports.values() for target in report["noise_target"].values()),
               "control_margin_max": max(max(report["control"]["margin_rms_fraction"].values()) for report in reports.values()),
               "control_median_kl_max": max(report["control"]["median_kl"] for report in reports.values()),
               "control_flip_max": max(report["control"]["top1_flip_fraction"] for report in reports.values())}
    result = {"schema": "temporal_five_mlp_mlp_input_covector_ood_result_v2", "started_utc": started, "finished_utc": now(),
              "serial_seconds": time.perf_counter() - tic, "authority_sha256": EXPECTED, "disjoint_text": disjoint,
              "reports": reports, "summary": summary, "predictions": predictions, "terminal": terminal,
              "price": {"model_forwards": 24, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}}
    atomic_create_json(OUT, result); print(json.dumps({k: result[k] for k in ("summary", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
