#!/usr/bin/env python3
"""Cross-fit causal compilation of MLP response factors into residual-input covectors."""
# BQGATE: EXPERIMENT pred_a_authority_hash_exact_effective_covector_closure_finiteness_and_price pred_b_rank32_crosspanel_coefficient_prediction pred_c_rank32_compiled_program_is_target_sufficient pred_d_rank32_compiled_program_is_selective pred_e_weight_covectors_admit_material_low_rank_compression
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
import run_temporal_five_mlp_factor_pruned_causal_program_v1 as pruned

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_five_mlp_mlp_input_covector_crossfit_v1.json"
PRUNED_RESULT = ROOT / "circuits/followups/temporal_five_mlp_factor_pruned_causal_program_v1_result.json"
PRUNED_RUNNER = ROOT / "ops/run_temporal_five_mlp_factor_pruned_causal_program_v1.py"
FACTOR_RESULT = ROOT / "circuits/followups/temporal_five_mlp_projector_coefficient_factor_atlas_v1_result.json"
INTERFACE = ROOT / "circuits/followups/temporal_five_mlp_crossfit_response_weight_interface_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_five_mlp_mlp_input_covector_crossfit_v1_result.json"
EXPECTED = {
    "prior": "daf684b121f2b2a60b41911e7554a9978c550a81d7ccbef9d2238572e19a7876",
    "pruned_result": "44cd07106b7b68c14d9cc068d620b28fdb527b65caf64c536978b277754bb2d7",
    "pruned_runner": "4e0325a1d9b5612275ce9fecec138862096d4651473e2f305cd7f2b8efabd463",
    "factor_result": "a092ffd51aa8b4964f68860c7082e2cbf273680ae867bb6f3543b10c1478326b",
    "interface": "578fa690f2f3962409bb50bd40a1c4c6f0b5ce7f88828e2e889a3a47bb20f013",
}
RANKS = (8, 16, 32, 64)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def effective_map(backend, site, q, base_inputs):
    """A(x0) such that (x1-x0) A equals the retained linear MLP response coordinates."""
    torch = backend.torch; _kind, layer, _head = atlasrun.site_parts(site)
    module = backend.model.transformer.h[layer].mlp; xb = base_inputs[layer]
    with torch.no_grad():
        left0, right0 = module.Left(xb).float(), module.Right(xb).float()
        downmap = module.Down.weight.detach().float().T @ q
        left = torch.einsum("hi,bth,hk->btik", module.Left.weight.detach().float(), right0, downmap)
        right = torch.einsum("hi,bth,hk->btik", module.Right.weight.detach().float(), left0, downmap)
    return left + right


def valid_rows(batch, tensor, ids):
    parts = []
    for i in ids:
        stop = int(batch.semantic_positions[i]) + 1
        parts.append(tensor[i, :stop])
    return backend_empty(tensor) if not parts else __import__("torch").cat(parts)


def backend_empty(tensor): return tensor.new_empty((0,) + tuple(tensor.shape[2:]))


def fit_input_basis(torch, batch, amap, ids, rank):
    chunks = []
    for i in ids:
        stop = int(batch.semantic_positions[i]) + 1
        chunks.append(amap[i, :stop].permute(0, 2, 1).reshape(-1, amap.shape[2]))
    matrix = torch.cat(chunks)
    _u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    used = min(rank, vh.shape[0]); basis = vh[:used].T.contiguous()
    energy = float(singular[:used].square().sum() / singular.square().sum().clamp_min(1e-30))
    return basis, energy


def coordinates(torch, delta_input, amap, basis=None):
    if basis is None: return torch.einsum("bti,btik->btk", delta_input, amap)
    reduced_input = delta_input @ basis
    reduced_map = torch.einsum("ir,btik->btrk", basis, amap)
    return torch.einsum("btr,btrk->btk", reduced_input, reduced_map)


def coefficient_rse(torch, batch, exact, predicted, ids):
    a, b = valid_rows(batch, exact, ids), valid_rows(batch, predicted, ids)
    return float((b-a).square().sum() / a.square().sum().clamp_min(1e-30))


def subset_target_report(backend, rows, ids, base_state, full_state, output, reader):
    torch = backend.torch; ix = torch.as_tensor(ids, device=backend.device)
    subrows = [rows[i] for i in ids]; temporal_n = sum(klfit.task_name(row) == "temporal" for row in subrows)
    state = atlasrun.states(torch, backend, output, rows)
    return ood.target_report(backend, subrows, base_state[ix], full_state[ix], state[ix], reader, temporal_n)


def main():
    paths = {"prior": PRIOR, "pruned_result": PRUNED_RESULT, "pruned_runner": PRUNED_RUNNER,
             "factor_result": FACTOR_RESULT, "interface": INTERFACE}
    observed = {key: sha(path) for key, path in paths.items()}
    if observed != EXPECTED: raise RuntimeError(f"MLP-input-covector authority changed: {observed}")
    dry = {"candidate_id": "temporal_auxiliary.five_mlp_mlp_input_covector_crossfit_v1", "dryrun": True,
           "gpu_accessed": False, "model_loaded": False, "queue_touched": False, "sites": comp.SITES,
           "ranks": RANKS, "model_forwards_max": 25, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1": print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    if json.loads(PRUNED_RESULT.read_text())["terminal"] != "factor_pruning_incomplete": raise RuntimeError("factor-pruned terminal changed")
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
    tb, td = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
    cb, cd = das._batch(backend, crows, side="base"), das._batch(backend, crows, side="donor")
    tbo, tbase, tba, tbi, terr0 = pruned.capture_everything(backend, tb)
    _tdo, tdonor, tda, tdi, terr1 = pruned.capture_everything(backend, td)
    cbo, cbase, cba, cbi, cerr0 = pruned.capture_everything(backend, cb)
    _cdo, _cdonor, cda, cdi, cerr1 = pruned.capture_everything(backend, cd)
    base_target = atlasrun.states(torch, backend, tbo, rows); base_control = atlasrun.states(torch, backend, cbo, crows)
    full_out, _ = atlasrun.run_patch(backend, tb, tdonor, comp.SITES); full_target = atlasrun.states(torch, backend, full_out, rows)
    exact_t_attention, exact_t_mlp = pruned.factor_deltas(backend, rows, tba, tda, tbi, tdi, include_exception=False)
    exact_c_attention, _exact_c_mlp = pruned.factor_deltas(backend, crows, cba, cda, cbi, cdi, include_exception=False)
    panels = {panel: [i for i, row in enumerate(rows) if row["transform_id"] == panel] for panel in ("A1", "A2")}
    fit_panel = {"even_fit": "A1", "odd_fit": "A2"}; eval_panel = {"even_fit": "A2", "odd_fit": "A1"}
    control_ctx = {"rows": crows, "base_logits": das.head_logits(backend, base_control).float()}

    reports = {}; closure = []
    for label, qs in projectors.items():
        train_ids, test_ids = panels[fit_panel[label]], panels[eval_panel[label]]
        bases_by_rank = {rank: {} for rank in RANKS}; site_records = {}
        for site, q in qs.items():
            if atlasrun.site_parts(site)[0] != "mlp": continue
            amap = effective_map(backend, site, q, tbi); delta_input = tdi[atlasrun.site_parts(site)[1]] - tbi[atlasrun.site_parts(site)[1]]
            exact = coordinates(torch, delta_input, amap)
            direct = exact_t_mlp[site] @ q
            closure.append(coefficient_rse(torch, tb, direct, exact, list(range(len(rows)))))
            ranks = {}
            for rank in RANKS:
                basis, energy = fit_input_basis(torch, tb, amap, train_ids, rank); bases_by_rank[rank][site] = basis
                predicted = coordinates(torch, delta_input, amap, basis)
                task_rse = {}
                for task in ("temporal", "iswas"):
                    ids = [i for i in test_ids if klfit.task_name(rows[i]) == task]
                    task_rse[task] = coefficient_rse(torch, tb, exact, predicted, ids)
                ranks[str(rank)] = {"train_effective_covector_energy": energy, "heldout_coefficient_rse": task_rse}
            site_records[site] = ranks

        target_arms = {}
        for rank in RANKS:
            compressed = {}
            for site, q in qs.items():
                if atlasrun.site_parts(site)[0] != "mlp": continue
                layer = atlasrun.site_parts(site)[1]; amap = effective_map(backend, site, q, tbi)
                delta_input = tdi[layer] - tbi[layer]
                compressed[site] = coordinates(torch, delta_input, amap, bases_by_rank[rank][site]) @ q.T
            output = pruned.run_factors(backend, tb, tbase, qs, exact_t_attention, compressed, use_attention=True, use_mlp=True)
            target_arms[str(rank)] = subset_target_report(backend, rows, test_ids, base_target, full_target, output, reader)

        compressed_control = {}
        for site, q in qs.items():
            if atlasrun.site_parts(site)[0] != "mlp": continue
            layer = atlasrun.site_parts(site)[1]; amap = effective_map(backend, site, q, cbi)
            delta_input = cdi[layer] - cbi[layer]
            compressed_control[site] = coordinates(torch, delta_input, amap, bases_by_rank[32][site]) @ q.T
        control_output = pruned.run_factors(backend, cb, cbase, qs, exact_c_attention, compressed_control, use_attention=True, use_mlp=True)
        control_state = atlasrun.states(torch, backend, control_output, crows)
        control = klfit.control_metrics(backend, control_ctx, das.head_logits(backend, control_state).float())
        reports[label] = {"fit_panel": fit_panel[label], "evaluation_panel": eval_panel[label], "sites": site_records,
                          "target_by_rank": target_arms, "rank32_control": control}

    finite = [orientation, terr0, terr1, cerr0, cerr1] + closure
    for report in reports.values():
        for site in report["sites"].values():
            for rank in site.values(): finite += [rank["train_effective_covector_energy"]] + list(rank["heldout_coefficient_rse"].values())
        for target in report["target_by_rank"].values(): finite += list(target["cells"].values()) + list(target["behavior_signed_projection"].values())
        finite += list(report["rank32_control"]["margin_rms_fraction"].values()) + [report["rank32_control"]["median_kl"], report["rank32_control"]["max_kl"], report["rank32_control"]["top1_flip_fraction"]]
    pa = hashes_ok and reader_ok and orientation <= 1e-6 and max(closure) <= 1e-10 and max(terr0, terr1, cerr0, cerr1) <= 5e-4 and all(math.isfinite(float(x)) for x in finite)
    pb = all(rank["heldout_coefficient_rse"][task] <= .1 for report in reports.values() for site in report["sites"].values() for rank in (site["32"],) for task in ("temporal", "iswas"))
    pc = all(report["target_by_rank"]["32"]["worst_target_residual"] <= .15 and min(report["target_by_rank"]["32"]["behavior_signed_projection"].values()) >= .8 for report in reports.values())
    pd = all(max(report["rank32_control"]["margin_rms_fraction"].values()) <= .1 and report["rank32_control"]["median_kl"] <= .02 and report["rank32_control"]["top1_flip_fraction"] <= .05 for report in reports.values())
    compression = {}
    for label, report in reports.items():
        mean8 = sum(x["8"]["heldout_coefficient_rse"][task] for x in report["sites"].values() for task in ("temporal", "iswas")) / (2 * len(report["sites"]))
        mean32 = sum(x["32"]["heldout_coefficient_rse"][task] for x in report["sites"].values() for task in ("temporal", "iswas")) / (2 * len(report["sites"]))
        min_energy = min(x["32"]["train_effective_covector_energy"] for x in report["sites"].values())
        compression[label] = {"rank8_mean_heldout_coefficient_rse": mean8, "rank32_mean_heldout_coefficient_rse": mean32,
                              "rank32_min_training_energy": min_energy, "rank32_relative_error_reduction_vs_rank8": 1.0 - mean32 / max(mean8, 1e-30)}
    pe = all(x["rank32_min_training_energy"] >= .9 and x["rank32_relative_error_reduction_vs_rank8"] >= .1 for x in compression.values())
    predictions = {"pred_a_authority_hash_exact_effective_covector_closure_finiteness_and_price": bool(pa),
                   "pred_b_rank32_crosspanel_coefficient_prediction": bool(pb),
                   "pred_c_rank32_compiled_program_is_target_sufficient": bool(pc),
                   "pred_d_rank32_compiled_program_is_selective": bool(pd),
                   "pred_e_weight_covectors_admit_material_low_rank_compression": bool(pe)}
    terminal = "invalid" if not pa else "crossfit_input_covector_program" if all(predictions.values()) else "input_covector_compression_incomplete"
    summary = {"max_exact_covector_closure_rse": max(closure),
               "rank32_worst_heldout_coefficient_rse": max(site["32"]["heldout_coefficient_rse"][task] for report in reports.values() for site in report["sites"].values() for task in ("temporal", "iswas")),
               "rank32_target_worst_max": max(report["target_by_rank"]["32"]["worst_target_residual"] for report in reports.values()),
               "rank32_target_projection_min": min(min(report["target_by_rank"]["32"]["behavior_signed_projection"].values()) for report in reports.values()),
               "rank32_control_margin_max": max(max(report["rank32_control"]["margin_rms_fraction"].values()) for report in reports.values()),
               "rank32_control_median_kl_max": max(report["rank32_control"]["median_kl"] for report in reports.values()),
               "rank32_control_flip_max": max(report["rank32_control"]["top1_flip_fraction"] for report in reports.values()),
               "compression": compression}
    result = {"schema": "temporal_five_mlp_mlp_input_covector_crossfit_result_v1", "started_utc": started, "finished_utc": now(),
              "serial_seconds": time.perf_counter() - tic, "authority_sha256": EXPECTED, "reports": reports, "summary": summary,
              "predictions": predictions, "terminal": terminal,
              "price": {"model_forwards": 23, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}}
    atomic_create_json(OUT, result); print(json.dumps({k: result[k] for k in ("summary", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
