#!/usr/bin/env python3
"""Bidirectional source-noise assay of the pooled rank48 response interface."""
# BQGATE: EXPERIMENT pred_a_authority_hash_noise_tripwire_finiteness_and_exact_price pred_b_all_noisy_bidirectional_coordinates_transfer pred_c_all_noisy_bidirectional_behaviors_transfer pred_d_all_noisy_bidirectional_controls_are_selective pred_e_seed_and_direction_stability
from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import compiled_response_ood as oodctx
import pooled_response_projector as pooled
import run_temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1 as atlasrun
import run_temporal_five_mlp_upstream_tensor_ranked_greedy_program_v1 as greedy
import run_temporal_five_mlp_crossfit_response_weight_interface_v1 as interface
import run_temporal_five_mlp_family_feasible_kl_refinement_v1 as klfit
import run_temporal_five_mlp_compiled_coordinate_upstream_atlas_v1 as atlas
import run_temporal_five_mlp_source_clamped_compiled_graph_v1 as source
import run_temporal_five_mlp_source_clamped_layer_band_addback_v1 as addback
import run_temporal_five_mlp_factor_pruned_causal_program_v1 as pruned
import run_temporal_five_mlp_frozen_response_ood_regularization_v2 as ood
import run_temporal_five_mlp_generic_subspace_complement_program_v1 as comp

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_five_mlp_rank48_pooled_bidirectional_source_noise_ood_v1.json"
FORWARD = ROOT / "circuits/followups/temporal_five_mlp_rank48_pooled_projector_forward_ood_v1_result.json"
REVERSE = ROOT / "circuits/followups/temporal_five_mlp_rank48_pooled_projector_reverse_ood_v2_result.json"
OLD_NOISE = ROOT / "circuits/followups/temporal_five_mlp_rank48_source_noise_ood_v1_result.json"
SUPPORT = ROOT / "circuits/followups/temporal_five_mlp_rank49_joint_single_deletion_v1_result.json"
TCAP = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v12_capability_v1_result.json"
ICAP = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v11_capability_v1_result.json"
HELPER = ROOT / "ops/pooled_response_projector.py"
OUT = ROOT / "circuits/followups/temporal_five_mlp_rank48_pooled_bidirectional_source_noise_ood_v1_result.json"
EXPECTED = {
    "forward": "c9fb774866c09391ddde6fa68842e3770b7c1a944744cc454191e0a23a141d40",
    "reverse": "2791135e5abf88ceb5b76a3e0da44c1bb5625a5031ae58b0aba351f1232fb9b2",
    "old_noise": "c37fb4d86d1524544d9060ac33c7751df5deb3369c6a52fa989626ac7ba41f9b",
    "support": "92384a4496c7e3ceb5dbccecfd8ef65a61e7f5c219e52b5cf919b91ef1ba5fe2",
    "helper": "09a7ea5504722a1765367f8422cbbb858373bcc5e910d5aae75ccba46cb124ed",
}
SEEDS = (1729, 2718, 3141)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def noisy_graph(backend, base_batch, donor_batch, base_full, support, seed):
    """Exchange one cue token through the restricted graph plus norm-matched noise."""
    torch = backend.torch
    base_tokens, _ = backend._tensor_batch(base_batch); donor_tokens, _ = backend._tensor_batch(donor_batch)
    mask = base_tokens != donor_tokens
    with torch.no_grad():
        base_embedding = backend.model.transformer.wte(base_tokens).detach()
        donor_embedding = backend.model.transformer.wte(donor_tokens).detach()
        generator = torch.Generator(device=backend.device).manual_seed(seed)
        noise = torch.zeros_like(donor_embedding); ratios = []
        for row in range(base_tokens.shape[0]):
            indices = mask[row].nonzero(as_tuple=False).flatten()
            if len(indices) != 1: raise RuntimeError(f"row {row} has {len(indices)} changed source tokens")
            pos = int(indices[0]); delta = donor_embedding[row, pos] - base_embedding[row, pos]
            raw = torch.randn(delta.shape, device=delta.device, dtype=delta.dtype, generator=generator)
            scaled = raw * (.1 * delta.float().norm() / raw.float().norm().clamp_min(1e-30)).to(raw.dtype)
            noise[row, pos] = scaled
            ratios.append(float(scaled.float().norm() / delta.float().norm().clamp_min(1e-30)))
    def hook(_module, _arguments, output):
        changed = output.clone(); changed[mask] = (donor_embedding + noise)[mask].to(changed); return changed
    handle = backend.model.transformer.wte.register_forward_hook(hook)
    try: result = source.restricted_graph(backend, base_batch, donor_batch, base_full, support, source=False)
    finally: handle.remove()
    return result[1], result[2], ratios


def reverse_report(backend, fresh, support, qs, bases, reader, donor_full, control_donor_full, reverse_full_state, seed):
    torch = backend.torch
    gai, gmi, tr = noisy_graph(backend, fresh["donor_batch"], fresh["batch"], donor_full, support, seed)
    cai, cmi, cr = noisy_graph(backend, fresh["control_donor_batch"], fresh["control_batch"], control_donor_full, support, seed + 100000)
    reference = atlas.compiled_coordinates(backend, fresh["rows"], qs, fresh["donor"][2], fresh["donor"][5], fresh["base"][5], fresh["donor"][3], fresh["base"][3], bases)
    generated = atlas.compiled_coordinates(backend, fresh["rows"], qs, fresh["donor"][2], fresh["donor"][5], gai, fresh["donor"][3], gmi, bases)
    coordinate = atlas.coordinate_report(torch, atlas.vectors(backend, fresh["donor_batch"], qs, reference), atlas.vectors(backend, fresh["donor_batch"], qs, generated))
    output = pruned.run_factors(backend, fresh["donor_batch"], fresh["donor"][1], qs,
        {s: v for s, v in generated.items() if atlasrun.site_parts(s)[0] == "attn"},
        {s: v for s, v in generated.items() if atlasrun.site_parts(s)[0] == "mlp"}, use_attention=True, use_mlp=True)
    target = ood.target_report(backend, fresh["rows"], atlasrun.states(torch, backend, fresh["donor"][0], fresh["rows"]),
        reverse_full_state, atlasrun.states(torch, backend, output, fresh["rows"]), reader, fresh["temporal_n"])
    cgenerated = atlas.compiled_coordinates(backend, fresh["controls"], qs, fresh["control_donor"][2], fresh["control_donor"][5], cai, fresh["control_donor"][3], cmi, bases)
    coutput = pruned.run_factors(backend, fresh["control_donor_batch"], fresh["control_donor"][1], qs,
        {s: v for s, v in cgenerated.items() if atlasrun.site_parts(s)[0] == "attn"},
        {s: v for s, v in cgenerated.items() if atlasrun.site_parts(s)[0] == "mlp"}, use_attention=True, use_mlp=True)
    cstate = atlasrun.states(torch, backend, fresh["control_donor"][0], fresh["controls"])
    control_ctx = {"rows": fresh["controls"], "base_logits": das.head_logits(backend, cstate).float()}
    control = klfit.control_metrics(backend, control_ctx, das.head_logits(backend, atlasrun.states(torch, backend, coutput, fresh["controls"])).float())
    return {"coordinate": coordinate, "target": target, "control": control}, tr + cr


def main():
    observed = {key: sha(path) for key, path in {"forward": FORWARD, "reverse": REVERSE, "old_noise": OLD_NOISE, "support": SUPPORT, "helper": HELPER}.items()}
    if observed != EXPECTED: raise RuntimeError(f"pooled bidirectional noise authority changed: {observed}")
    dry = {"candidate_id": "temporal_auxiliary.five_mlp_rank48_pooled_bidirectional_source_noise_ood_v1", "dryrun": True,
           "gpu_accessed": False, "model_loaded": False, "queue_touched": False, "noise_fraction": .1,
           "noise_seeds": SEEDS, "model_forwards_exact": 39, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1": print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    started, tic = now(), time.perf_counter(); backend = producer.Bilin18TorchBackend.load("cuda")
    fitted = pooled.fit(backend); qs = fitted["projector"]; training, bases = pooled.fit_mlp_input_bases(backend, qs)
    fresh = oodctx.capture(backend, TCAP, ICAP); support = json.loads(SUPPORT.read_text())["selected_support"]
    reader, orientation, reader_ok = greedy.physical_reader(backend, json.loads(comp.WEIGHTS.read_text()))
    _do, donor_full = atlasrun.capture_native(backend, fresh["donor_batch"])
    _co, control_donor_full = atlasrun.capture_native(backend, fresh["control_donor_batch"])
    reverse_full_output, _ = atlasrun.run_patch(backend, fresh["donor_batch"], fresh["base"][1], comp.SITES)
    reverse_full_state = atlasrun.states(backend.torch, backend, reverse_full_output, fresh["rows"])
    clean = json.loads(FORWARD.read_text()); hashes = {site: interface.thash(q) for site, q in qs.items()}
    reports = {}; ratios = []; finite = [orientation, training["capture_error"]]
    for seed in SEEDS:
        gai, gmi, tr = noisy_graph(backend, fresh["batch"], fresh["donor_batch"], fresh["base_full"], support, seed)
        cai, cmi, cr = noisy_graph(backend, fresh["control_batch"], fresh["control_donor_batch"], fresh["control_base_full"], support, seed + 100000)
        generated = atlas.compiled_coordinates(backend, fresh["rows"], qs, fresh["base"][2], fresh["base"][5], gai, fresh["base"][3], gmi, bases)
        cgenerated = atlas.compiled_coordinates(backend, fresh["controls"], qs, fresh["control_base"][2], fresh["control_base"][5], cai, fresh["control_base"][3], cmi, bases)
        forward = addback.fit_arm(backend, {"fresh": fresh, "reader": reader}, support, "pooled", qs, bases, generated, cgenerated)
        reverse, rr = reverse_report(backend, fresh, support, qs, bases, reader, donor_full, control_donor_full, reverse_full_state, seed)
        reports[str(seed)] = {"forward": forward, "reverse": reverse}; ratios += tr + cr + rr
        for report in (forward, reverse):
            finite += list(report["coordinate"]["signed_projection"].values()) + list(report["coordinate"]["residual"].values())
            finite += list(report["target"]["cells"].values()) + list(report["target"]["behavior_signed_projection"].values())
            finite += list(report["control"]["margin_rms_fraction"].values()) + [report["control"]["median_kl"], report["control"]["max_kl"], report["control"]["top1_flip_fraction"]]
    flat = [reports[str(seed)][direction] for seed in SEEDS for direction in ("forward", "reverse")]
    fit_ids = {row["row_id"] for row in fitted["target_rows"] + fitted["control_rows"]}; eval_ids = {row["row_id"] for row in fresh["rows"] + fresh["controls"]}
    pa = (reader_ok and orientation <= 1e-6 and hashes == clean["pooled_projector_sha256"] and not (fit_ids & eval_ids)
          and len(support) == 48 and min(ratios) >= .099 and max(ratios) <= .101 and all(math.isfinite(float(v)) for v in finite + ratios))
    pb = all(min(r["coordinate"]["signed_projection"].values()) >= .75 and r["coordinate"]["mean_residual"] <= .2 and r["coordinate"]["worst_residual"] <= .2 for r in flat)
    pc = all(min(r["target"]["behavior_signed_projection"].values()) >= .8 and r["target"]["worst_target_residual"] <= .15 for r in flat)
    pd = all(max(r["control"]["margin_rms_fraction"].values()) <= .1 and r["control"]["median_kl"] <= .02 and r["control"]["top1_flip_fraction"] == 0.0 for r in flat)
    ranges = {}; direction_gap = 0.0
    for direction in ("forward", "reverse"):
        coordinate = [min(reports[str(seed)][direction]["coordinate"]["signed_projection"].values()) for seed in SEEDS]
        ranges[direction] = {"coordinate": max(coordinate) - min(coordinate)}
        for task in ("temporal", "iswas"):
            values = [reports[str(seed)][direction]["target"]["behavior_signed_projection"][task] for seed in SEEDS]
            ranges[direction][task] = max(values) - min(values)
    for seed in SEEDS:
        for task in ("temporal", "iswas"):
            direction_gap = max(direction_gap, abs(reports[str(seed)]["forward"]["target"]["behavior_signed_projection"][task] - reports[str(seed)]["reverse"]["target"]["behavior_signed_projection"][task]))
    pe = max(v for values in ranges.values() for v in values.values()) <= .08 and direction_gap <= .08
    predictions = {"pred_a_authority_hash_noise_tripwire_finiteness_and_exact_price": bool(pa),
        "pred_b_all_noisy_bidirectional_coordinates_transfer": bool(pb), "pred_c_all_noisy_bidirectional_behaviors_transfer": bool(pc),
        "pred_d_all_noisy_bidirectional_controls_are_selective": bool(pd), "pred_e_seed_and_direction_stability": bool(pe)}
    summary = {"noise_ratio_min": min(ratios), "noise_ratio_max": max(ratios),
        "coordinate_projection_min": min(min(r["coordinate"]["signed_projection"].values()) for r in flat),
        "coordinate_mean_residual_max": max(r["coordinate"]["mean_residual"] for r in flat),
        "target_projection_min": min(min(r["target"]["behavior_signed_projection"].values()) for r in flat),
        "target_worst_max": max(r["target"]["worst_target_residual"] for r in flat),
        "control_margin_max": max(max(r["control"]["margin_rms_fraction"].values()) for r in flat),
        "control_median_kl_max": max(r["control"]["median_kl"] for r in flat),
        "control_flip_max": max(r["control"]["top1_flip_fraction"] for r in flat), "seed_ranges": ranges, "direction_gap_max": direction_gap}
    terminal = "invalid" if not pa else "pooled_bidirectional_source_noise_robust_rank48_program" if all(predictions.values()) else "pooled_bidirectional_source_noise_failure"
    result = {"schema": "temporal_five_mlp_rank48_pooled_bidirectional_source_noise_ood_result_v1", "started_utc": started, "finished_utc": now(),
        "serial_seconds": time.perf_counter() - tic, "authority_sha256": EXPECTED, "pooled_projector_sha256": hashes,
        "noise_fraction": .1, "noise_seeds": SEEDS, "reports": reports, "summary": summary, "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": 39, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}}
    atomic_create_json(OUT, result); print(json.dumps({k: result[k] for k in ("summary", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
