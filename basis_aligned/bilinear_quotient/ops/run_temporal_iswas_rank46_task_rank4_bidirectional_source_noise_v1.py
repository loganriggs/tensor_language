#!/usr/bin/env python3
"""Bidirectional source-noise validation of task-conditioned rank-four responses."""
# BQGATE: EXPERIMENT pred_a_authority_hash_noise_tripwire_finiteness_and_exact_price pred_b_all_noisy_rank4_coordinates_transfer pred_c_all_noisy_rank4_behaviors_transfer pred_d_all_noisy_rank4_controls_are_selective pred_e_rank4_seed_and_direction_stability
from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import compiled_response_ood as oodctx
import pooled_response_projector as pooled
import run_temporal_five_mlp_rank48_pooled_bidirectional_source_noise_ood_v1 as noise
import run_temporal_iswas_rank46_task_typed_mode_causal_factorial_v1 as factor
import run_temporal_iswas_rank46_task_mode_complete_rank_ladder_v1 as ladder
import run_temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1 as atlasrun
import run_temporal_five_mlp_upstream_tensor_ranked_greedy_program_v1 as greedy
import run_temporal_five_mlp_crossfit_response_weight_interface_v1 as interface
import run_temporal_five_mlp_family_feasible_kl_refinement_v1 as klfit
import run_temporal_five_mlp_compiled_coordinate_upstream_atlas_v1 as atlas
import run_temporal_five_mlp_source_clamped_layer_band_addback_v1 as addback
import run_temporal_five_mlp_factor_pruned_causal_program_v1 as pruned
import run_temporal_five_mlp_frozen_response_ood_regularization_v2 as ood
import run_temporal_five_mlp_generic_subspace_complement_program_v1 as comp

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_rank46_task_rank4_bidirectional_source_noise_v1.json"
REVERSE = ROOT / "circuits/followups/temporal_iswas_rank46_task_rank4_reverse_ood_v1_result.json"
RANK46_NOISE = ROOT / "circuits/followups/temporal_five_mlp_rank46_pooled_bidirectional_source_noise_ood_v1_result.json"
LADDER = ROOT / "circuits/followups/temporal_iswas_rank46_task_mode_complete_rank_ladder_v1_result.json"
RANK46 = factor.RANK46; TCAP = factor.TCAP; ICAP = factor.ICAP
OUT = ROOT / "circuits/followups/temporal_iswas_rank46_task_rank4_bidirectional_source_noise_v1_result.json"
EXPECTED = {"reverse": "a77f34c55a0701f2ccbd0ce9e07f561855ad1b764d99ad5fdd246ded72d4829b",
    "rank46_noise": "1f8afe2f2e01e7b669ea8cb4343283d1fc246b8d07c6d70486af6649ef1cc824",
    "ladder": "345ecd0542890c69181efe067729560375543c694f2183319ff824300028e6d8",
    "rank46": factor.EXPECTED["rank46"]}
SEEDS = (1729, 2718, 3141)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def reverse_report(backend, fresh, support, qs, bases, rank4_modes, reader, donor_full, control_donor_full, reverse_full_state, seed):
    torch = backend.torch
    gai, gmi, tr = noise.noisy_graph(backend, fresh["donor_batch"], fresh["batch"], donor_full, support, seed)
    cai, cmi, cr = noise.noisy_graph(backend, fresh["control_donor_batch"], fresh["control_batch"], control_donor_full, support, seed + 100000)
    raw = atlas.compiled_coordinates(backend, fresh["rows"], qs, fresh["donor"][2], fresh["donor"][5], gai, fresh["donor"][3], gmi, bases)
    craw = atlas.compiled_coordinates(backend, fresh["controls"], qs, fresh["control_donor"][2], fresh["control_donor"][5], cai, fresh["control_donor"][3], cmi, bases)
    values = factor.filter_coordinates(backend, fresh, raw, qs, rank4_modes, "own")
    cvalues = factor.filter_coordinates(backend, fresh, craw, qs, rank4_modes, "own", controls=True)
    output = pruned.run_factors(backend, fresh["donor_batch"], fresh["donor"][1], qs,
        {s: v for s, v in values.items() if atlasrun.site_parts(s)[0] == "attn"}, {s: v for s, v in values.items() if atlasrun.site_parts(s)[0] == "mlp"}, use_attention=True, use_mlp=True)
    coutput = pruned.run_factors(backend, fresh["control_donor_batch"], fresh["control_donor"][1], qs,
        {s: v for s, v in cvalues.items() if atlasrun.site_parts(s)[0] == "attn"}, {s: v for s, v in cvalues.items() if atlasrun.site_parts(s)[0] == "mlp"}, use_attention=True, use_mlp=True)
    coordinate = factor.coordinate_by_task(fresh, raw, values)
    donor_state = atlasrun.states(torch, backend, fresh["donor"][0], fresh["rows"])
    target = ood.target_report(backend, fresh["rows"], donor_state, reverse_full_state, atlasrun.states(torch, backend, output, fresh["rows"]), reader, fresh["temporal_n"])
    cstate = atlasrun.states(torch, backend, fresh["control_donor"][0], fresh["controls"])
    control = klfit.control_metrics(backend, {"rows": fresh["controls"], "base_logits": das.head_logits(backend, cstate).float()},
        das.head_logits(backend, atlasrun.states(torch, backend, coutput, fresh["controls"])).float())
    return {"coordinate": coordinate, "target": target, "control": control}, tr + cr


def main():
    paths = {"reverse": REVERSE, "rank46_noise": RANK46_NOISE, "ladder": LADDER, "rank46": RANK46}
    observed = {key: sha(path) for key, path in paths.items()}
    if observed != EXPECTED: raise RuntimeError(f"rank4 noise authority changed: {observed}")
    dry = {"candidate_id": "temporal_auxiliary.iswas_rank46_task_rank4_bidirectional_source_noise_v1", "dryrun": True,
        "gpu_accessed": False, "model_loaded": False, "queue_touched": False, "noise_seeds": SEEDS,
        "model_forwards_exact": 43, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1": print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    started, tic = now(), time.perf_counter(); backend = producer.Bilin18TorchBackend.load("cuda")
    fitted = pooled.fit(backend); qs = fitted["projector"]; task_rows, modes = ladder.fit_full_modes(backend, fitted)
    rank4_modes = {site: {task: value[:, :4] for task, value in task_modes.items()} for site, task_modes in modes.items()}
    training, bases = pooled.fit_mlp_input_bases(backend, qs); fresh = oodctx.capture(backend, TCAP, ICAP)
    support = json.loads(RANK46.read_text())["selected_support"]
    _do, donor_full = atlasrun.capture_native(backend, fresh["donor_batch"]); _co, control_donor_full = atlasrun.capture_native(backend, fresh["control_donor_batch"])
    reverse_full_output, _ = atlasrun.run_patch(backend, fresh["donor_batch"], fresh["base"][1], comp.SITES)
    reverse_full_state = atlasrun.states(backend.torch, backend, reverse_full_output, fresh["rows"])
    reader, orientation, reader_ok = greedy.physical_reader(backend, json.loads(comp.WEIGHTS.read_text()))
    reports = {}; ratios = []; finite = [orientation, training["capture_error"]]
    for seed in SEEDS:
        gai, gmi, tr = noise.noisy_graph(backend, fresh["batch"], fresh["donor_batch"], fresh["base_full"], support, seed)
        cai, cmi, cr = noise.noisy_graph(backend, fresh["control_batch"], fresh["control_donor_batch"], fresh["control_base_full"], support, seed + 100000)
        raw = atlas.compiled_coordinates(backend, fresh["rows"], qs, fresh["base"][2], fresh["base"][5], gai, fresh["base"][3], gmi, bases)
        craw = atlas.compiled_coordinates(backend, fresh["controls"], qs, fresh["control_base"][2], fresh["control_base"][5], cai, fresh["control_base"][3], cmi, bases)
        values = factor.filter_coordinates(backend, fresh, raw, qs, rank4_modes, "own")
        cvalues = factor.filter_coordinates(backend, fresh, craw, qs, rank4_modes, "own", controls=True)
        forward = addback.fit_arm(backend, {"fresh": fresh, "reader": reader}, support, "rank4", qs, bases, values, cvalues)
        forward["coordinate"] = factor.coordinate_by_task(fresh, raw, values)
        reverse, rr = reverse_report(backend, fresh, support, qs, bases, rank4_modes, reader, donor_full, control_donor_full, reverse_full_state, seed)
        reports[str(seed)] = {"forward": forward, "reverse": reverse}; ratios += tr + cr + rr
        for report in (forward, reverse):
            finite += list(report["target"]["cells"].values()) + list(report["target"]["behavior_signed_projection"].values())
            finite += list(report["control"]["margin_rms_fraction"].values()) + [report["control"]["median_kl"], report["control"]["max_kl"], report["control"]["top1_flip_fraction"]]
            finite += [x for task in report["coordinate"].values() for x in (task["mean_signed_projection"], task["worst_residual"])]
    flat = [reports[str(seed)][direction] for seed in SEEDS for direction in ("forward", "reverse")]
    fit_ids = {row["row_id"] for rows in task_rows.values() for row in rows} | {row["row_id"] for row in fitted["control_rows"]}
    eval_ids = {row["row_id"] for row in fresh["rows"] + fresh["controls"]}; hashes = {site: interface.thash(q) for site, q in qs.items()}
    pa = (reader_ok and orientation <= 1e-6 and hashes == json.loads(REVERSE.read_text())["pooled_projector_sha256"] and not (fit_ids & eval_ids)
          and len(support) == 46 and min(ratios) >= .099 and max(ratios) <= .101 and all(math.isfinite(float(v)) for v in finite + ratios))
    pb = all(min(task["mean_signed_projection"] for task in r["coordinate"].values()) >= .95 and max(task["worst_residual"] for task in r["coordinate"].values()) <= .20 for r in flat)
    pc = all(min(r["target"]["behavior_signed_projection"].values()) >= .8 and r["target"]["worst_target_residual"] <= .15 for r in flat)
    pd = all(max(r["control"]["margin_rms_fraction"].values()) <= .1 and r["control"]["median_kl"] <= .02 and r["control"]["top1_flip_fraction"] == 0.0 for r in flat)
    ranges = {}; direction_gap = 0.0
    for direction in ("forward", "reverse"):
        ranges[direction] = {}
        for task in ("temporal", "iswas"):
            values = [reports[str(seed)][direction]["target"]["behavior_signed_projection"][task] for seed in SEEDS]
            ranges[direction][task] = max(values) - min(values)
    for seed in SEEDS:
        for task in ("temporal", "iswas"):
            direction_gap = max(direction_gap, abs(reports[str(seed)]["forward"]["target"]["behavior_signed_projection"][task] - reports[str(seed)]["reverse"]["target"]["behavior_signed_projection"][task]))
    pe = max(v for values in ranges.values() for v in values.values()) <= .08 and direction_gap <= .08
    predictions = {"pred_a_authority_hash_noise_tripwire_finiteness_and_exact_price": bool(pa),
        "pred_b_all_noisy_rank4_coordinates_transfer": bool(pb), "pred_c_all_noisy_rank4_behaviors_transfer": bool(pc),
        "pred_d_all_noisy_rank4_controls_are_selective": bool(pd), "pred_e_rank4_seed_and_direction_stability": bool(pe)}
    summary = {"noise_ratio_min": min(ratios), "noise_ratio_max": max(ratios),
        "coordinate_projection_min": min(min(t["mean_signed_projection"] for t in r["coordinate"].values()) for r in flat),
        "coordinate_worst_residual_max": max(max(t["worst_residual"] for t in r["coordinate"].values()) for r in flat),
        "target_projection_min": min(min(r["target"]["behavior_signed_projection"].values()) for r in flat),
        "target_worst_max": max(r["target"]["worst_target_residual"] for r in flat),
        "control_margin_max": max(max(r["control"]["margin_rms_fraction"].values()) for r in flat),
        "control_median_kl_max": max(r["control"]["median_kl"] for r in flat), "control_flip_max": max(r["control"]["top1_flip_fraction"] for r in flat),
        "seed_ranges": ranges, "direction_gap_max": direction_gap}
    terminal = "invalid" if not pa else "noise_robust_bidirectional_task_rank4_rank46_program" if all(predictions.values()) else "task_rank4_source_noise_boundary"
    result = {"schema": "temporal_iswas_rank46_task_rank4_bidirectional_source_noise_result_v1", "started_utc": started, "finished_utc": now(),
        "serial_seconds": time.perf_counter() - tic, "authority_sha256": EXPECTED, "pooled_projector_sha256": hashes, "support_count": len(support),
        "noise_seeds": SEEDS, "reports": reports, "summary": summary, "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": 43, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}}
    atomic_create_json(OUT, result); print(json.dumps({"summary": summary, "predictions": predictions, "terminal": terminal, "price": result["price"]}, sort_keys=True))


if __name__ == "__main__": main()
