#!/usr/bin/env python3
"""End-to-end cue-source noise robustness of the frozen rank48 OOD circuit."""
# BQGATE: EXPERIMENT pred_a_authority_noise_tripwire_finiteness_and_price pred_b_noisy_ood_coordinates_transfer pred_c_noisy_ood_behavior_transfer pred_d_noisy_ood_controls_are_selective pred_e_noise_seed_stability
from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import compiled_response_ood as oodctx
import run_temporal_five_mlp_source_clamped_compiled_graph_v1 as source
import run_temporal_five_mlp_source_clamped_layer_band_addback_v1 as addback
import run_temporal_five_mlp_compiled_coordinate_upstream_atlas_v1 as atlas

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_five_mlp_rank48_source_noise_ood_v1.json"
SUPPORT = ROOT / "circuits/followups/temporal_five_mlp_rank49_joint_single_deletion_v1_result.json"
CLEAN = ROOT / "circuits/followups/temporal_five_mlp_rank48_lexical_ood_v1_result.json"
TCAP = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v12_capability_v1_result.json"
ICAP = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v11_capability_v1_result.json"
HELPER = ROOT / "ops/compiled_response_ood.py"
OUT = ROOT / "circuits/followups/temporal_five_mlp_rank48_source_noise_ood_v1_result.json"
EXPECTED = {"support": "92384a4496c7e3ceb5dbccecfd8ef65a61e7f5c219e52b5cf919b91ef1ba5fe2",
            "clean_ood": "a19d8f07f1233e7ee607ee18d15d114afda429b396c405f17330681294e7b08b",
            "helper": "96b9e3aef3b64364c7fc4c49d0af9ff35827eadb4ca5bfbdf5d2271cb26ad808"}
SEEDS = (1729, 2718, 3141)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def noisy_graph(backend, base_batch, donor_batch, base_full, support, seed):
    torch = backend.torch; base_tokens, _ = backend._tensor_batch(base_batch); donor_tokens, _ = backend._tensor_batch(donor_batch)
    mask = base_tokens != donor_tokens
    with torch.no_grad():
        base_embedding = backend.model.transformer.wte(base_tokens).detach(); donor_embedding = backend.model.transformer.wte(donor_tokens).detach()
        generator = torch.Generator(device=backend.device).manual_seed(seed); noise = torch.zeros_like(donor_embedding)
        ratios = []
        for row in range(base_tokens.shape[0]):
            indices = mask[row].nonzero(as_tuple=False).flatten()
            if len(indices) != 1: raise RuntimeError(f"row {row} has {len(indices)} changed source tokens")
            pos = int(indices[0]); delta = donor_embedding[row, pos]-base_embedding[row, pos]
            raw = torch.randn(delta.shape, device=delta.device, dtype=delta.dtype, generator=generator)
            scaled = raw * (.1 * delta.float().norm() / raw.float().norm().clamp_min(1e-30)).to(raw.dtype)
            noise[row, pos] = scaled; ratios.append(float(scaled.float().norm()/delta.float().norm().clamp_min(1e-30)))
    def hook(_module, _arguments, output):
        changed = output.clone(); changed[mask] = (donor_embedding+noise)[mask].to(changed); return changed
    handle = backend.model.transformer.wte.register_forward_hook(hook)
    try: result = source.restricted_graph(backend, base_batch, donor_batch, base_full, support, source=False)
    finally: handle.remove()
    return (*result[:3], ratios)


def main():
    observed = {"support": sha(SUPPORT), "clean_ood": sha(CLEAN), "helper": sha(HELPER)}
    if observed != EXPECTED: raise RuntimeError(f"rank48 source-noise authority changed: {observed}")
    dry = {"candidate_id": "temporal_auxiliary.five_mlp_rank48_source_noise_ood_v1", "dryrun": True,
           "gpu_accessed": False, "model_loaded": False, "queue_touched": False, "noise_fraction": .1,
           "noise_seeds": SEEDS, "model_forwards_max": 34, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1": print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    started, tic = now(), time.perf_counter(); backend = producer.Bilin18TorchBackend.load("cuda")
    ctx = oodctx.prepare(backend, TCAP, ICAP); fresh = ctx["fresh"]; support = json.loads(SUPPORT.read_text())["selected_support"]
    reports = {}; ratios = []; finite = [ctx["reader_orientation"], ctx["training"]["capture_error"]]
    for seed in SEEDS:
        _go, gai, gmi, tr = noisy_graph(backend, fresh["batch"], fresh["donor_batch"], fresh["base_full"], support, seed)
        _co, cai, cmi, cr = noisy_graph(backend, fresh["control_batch"], fresh["control_donor_batch"], fresh["control_base_full"], support, seed+100000)
        ratios += tr + cr; seed_reports = {}
        for label, qs in ctx["projectors"].items():
            generated = atlas.compiled_coordinates(backend, fresh["rows"], qs, fresh["base"][2], fresh["base"][5], gai, fresh["base"][3], gmi, ctx["rank64_bases"][label])
            cgenerated = atlas.compiled_coordinates(backend, fresh["controls"], qs, fresh["control_base"][2], fresh["control_base"][5], cai, fresh["control_base"][3], cmi, ctx["rank64_bases"][label])
            report = addback.fit_arm(backend, ctx, support, label, qs, ctx["rank64_bases"][label], generated, cgenerated); seed_reports[label] = report
            finite += list(report["coordinate"]["signed_projection"].values()) + list(report["coordinate"]["residual"].values())
            finite += list(report["target"]["cells"].values()) + list(report["target"]["behavior_signed_projection"].values())
            finite += list(report["control"]["margin_rms_fraction"].values()) + [report["control"]["median_kl"], report["control"]["max_kl"], report["control"]["top1_flip_fraction"]]
        reports[str(seed)] = seed_reports
    all_reports = [report for seed in reports.values() for report in seed.values()]
    pa = ctx["hashes_ok"] and ctx["reader_ok"] and ctx["reader_orientation"] <= 1e-6 and len(support) == 48 and min(ratios) >= .099 and max(ratios) <= .101 and all(math.isfinite(float(v)) for v in finite+ratios)
    pb = all(min(r["coordinate"]["signed_projection"].values()) >= .7 and r["coordinate"]["mean_residual"] <= .25 for r in all_reports)
    pc = all(min(r["target"]["behavior_signed_projection"].values()) >= .75 and r["target"]["worst_target_residual"] <= .2 for r in all_reports)
    pd = all(max(r["control"]["margin_rms_fraction"].values()) <= .1 and r["control"]["median_kl"] <= .02 and r["control"]["top1_flip_fraction"] <= .05 for r in all_reports)
    ranges = {}
    for label in ctx["projectors"]:
        coordinate = [min(reports[str(seed)][label]["coordinate"]["signed_projection"].values()) for seed in SEEDS]
        ranges[label] = {"coordinate": max(coordinate)-min(coordinate)}
        for task in ("temporal", "iswas"):
            values = [reports[str(seed)][label]["target"]["behavior_signed_projection"][task] for seed in SEEDS]
            ranges[label][task] = max(values)-min(values)
    pe = max(value for fit in ranges.values() for value in fit.values()) <= .08
    predictions = {"pred_a_authority_noise_tripwire_finiteness_and_price": bool(pa), "pred_b_noisy_ood_coordinates_transfer": bool(pb),
                   "pred_c_noisy_ood_behavior_transfer": bool(pc), "pred_d_noisy_ood_controls_are_selective": bool(pd), "pred_e_noise_seed_stability": bool(pe)}
    summary = {"noise_ratio_min": min(ratios), "noise_ratio_max": max(ratios),
               "coordinate_projection_min": min(min(r["coordinate"]["signed_projection"].values()) for r in all_reports),
               "coordinate_mean_residual_max": max(r["coordinate"]["mean_residual"] for r in all_reports),
               "target_projection_min": min(min(r["target"]["behavior_signed_projection"].values()) for r in all_reports),
               "target_worst_max": max(r["target"]["worst_target_residual"] for r in all_reports),
               "control_margin_max": max(max(r["control"]["margin_rms_fraction"].values()) for r in all_reports),
               "control_median_kl_max": max(r["control"]["median_kl"] for r in all_reports),
               "control_flip_max": max(r["control"]["top1_flip_fraction"] for r in all_reports), "seed_ranges": ranges}
    terminal = "invalid" if not pa else "noise_robust_ood_rank48_program" if all(predictions.values()) else "rank48_source_noise_failure"
    result = {"schema": "temporal_five_mlp_rank48_source_noise_ood_result_v1", "started_utc": started, "finished_utc": now(),
              "serial_seconds": time.perf_counter()-tic, "authority_sha256": EXPECTED, "noise_fraction": .1, "noise_seeds": SEEDS,
              "support": support, "reports": reports, "summary": summary, "predictions": predictions, "terminal": terminal,
              "price": {"model_forwards": 34, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}}
    atomic_create_json(OUT, result); print(json.dumps({k: result[k] for k in ("summary", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
