#!/usr/bin/env python3
"""Reverse OOD manipulation of the frozen rank48 source-to-reader program."""
# BQGATE: EXPERIMENT pred_a_authority_bidirectional_population_finiteness_and_price pred_b_reverse_ood_coordinates_transfer pred_c_reverse_ood_behavior_is_necessary pred_d_reverse_ood_controls_are_selective pred_e_reverse_crossfit_is_stable
from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import compiled_response_ood as oodctx
import run_temporal_five_mlp_source_clamped_compiled_graph_v1 as source
import run_temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1 as atlasrun
import run_temporal_five_mlp_compiled_coordinate_upstream_atlas_v1 as atlas
import run_temporal_five_mlp_factor_pruned_causal_program_v1 as pruned
import run_temporal_five_mlp_frozen_response_ood_regularization_v2 as ood
import run_temporal_five_mlp_family_feasible_kl_refinement_v1 as klfit
import run_temporal_five_mlp_generic_subspace_complement_program_v1 as comp

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_five_mlp_rank48_reverse_ood_v1.json"
SUPPORT = ROOT / "circuits/followups/temporal_five_mlp_rank49_joint_single_deletion_v1_result.json"
FORWARD = ROOT / "circuits/followups/temporal_five_mlp_rank48_lexical_ood_v1_result.json"
NOISE = ROOT / "circuits/followups/temporal_five_mlp_rank48_source_noise_ood_v1_result.json"
TCAP = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v12_capability_v1_result.json"
ICAP = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v11_capability_v1_result.json"
HELPER = ROOT / "ops/compiled_response_ood.py"
OUT = ROOT / "circuits/followups/temporal_five_mlp_rank48_reverse_ood_v1_result.json"
EXPECTED = {"support": "92384a4496c7e3ceb5dbccecfd8ef65a61e7f5c219e52b5cf919b91ef1ba5fe2",
            "forward": "a19d8f07f1233e7ee607ee18d15d114afda429b396c405f17330681294e7b08b",
            "noise": "c37fb4d86d1524544d9060ac33c7751df5deb3369c6a52fa989626ac7ba41f9b",
            "helper": "96b9e3aef3b64364c7fc4c49d0af9ff35827eadb4ca5bfbdf5d2271cb26ad808"}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def main():
    paths = {"support": SUPPORT, "forward": FORWARD, "noise": NOISE, "helper": HELPER}; observed = {k: sha(v) for k, v in paths.items()}
    if observed != EXPECTED: raise RuntimeError(f"rank48 reverse authority changed: {observed}")
    dry = {"candidate_id": "temporal_auxiliary.five_mlp_rank48_reverse_ood_v1", "dryrun": True,
           "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
           "model_forwards_max": 25, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1": print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    started, tic = now(), time.perf_counter(); backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch
    ctx = oodctx.prepare(backend, TCAP, ICAP); fresh = ctx["fresh"]; support = json.loads(SUPPORT.read_text())["selected_support"]
    _donor_native, donor_full = atlasrun.capture_native(backend, fresh["donor_batch"])
    _control_donor_native, control_donor_full = atlasrun.capture_native(backend, fresh["control_donor_batch"])
    reverse_full_output, _ = atlasrun.run_patch(backend, fresh["donor_batch"], fresh["base"][1], comp.SITES)
    _go, gai, gmi, counts = source.restricted_graph(backend, fresh["donor_batch"], fresh["batch"], donor_full, support, source=True)
    _co, cai, cmi, ccounts = source.restricted_graph(backend, fresh["control_donor_batch"], fresh["control_batch"], control_donor_full, support, source=True)
    donor_state = atlasrun.states(torch, backend, fresh["donor"][0], fresh["rows"])
    reverse_full_state = atlasrun.states(torch, backend, reverse_full_output, fresh["rows"])
    control_donor_state = atlasrun.states(torch, backend, fresh["control_donor"][0], fresh["controls"])
    control_ctx = {"rows": fresh["controls"], "base_logits": das.head_logits(backend, control_donor_state).float()}
    reports = {}; finite = [ctx["reader_orientation"], ctx["training"]["capture_error"]]
    for label, qs in ctx["projectors"].items():
        reference = atlas.compiled_coordinates(backend, fresh["rows"], qs, fresh["donor"][2], fresh["donor"][5], fresh["base"][5], fresh["donor"][3], fresh["base"][3], ctx["rank64_bases"][label])
        generated = atlas.compiled_coordinates(backend, fresh["rows"], qs, fresh["donor"][2], fresh["donor"][5], gai, fresh["donor"][3], gmi, ctx["rank64_bases"][label])
        coordinate = atlas.coordinate_report(torch, atlas.vectors(backend, fresh["donor_batch"], qs, reference), atlas.vectors(backend, fresh["donor_batch"], qs, generated))
        target_output = pruned.run_factors(backend, fresh["donor_batch"], fresh["donor"][1], qs,
            {s: v for s, v in generated.items() if atlasrun.site_parts(s)[0] == "attn"},
            {s: v for s, v in generated.items() if atlasrun.site_parts(s)[0] == "mlp"}, use_attention=True, use_mlp=True)
        target = ood.target_report(backend, fresh["rows"], donor_state, reverse_full_state,
                                   atlasrun.states(torch, backend, target_output, fresh["rows"]), ctx["reader"], fresh["temporal_n"])
        cgenerated = atlas.compiled_coordinates(backend, fresh["controls"], qs, fresh["control_donor"][2], fresh["control_donor"][5], cai, fresh["control_donor"][3], cmi, ctx["rank64_bases"][label])
        control_output = pruned.run_factors(backend, fresh["control_donor_batch"], fresh["control_donor"][1], qs,
            {s: v for s, v in cgenerated.items() if atlasrun.site_parts(s)[0] == "attn"},
            {s: v for s, v in cgenerated.items() if atlasrun.site_parts(s)[0] == "mlp"}, use_attention=True, use_mlp=True)
        control = klfit.control_metrics(backend, control_ctx, das.head_logits(backend, atlasrun.states(torch, backend, control_output, fresh["controls"])).float())
        reports[label] = {"coordinate": coordinate, "target": target, "control": control}
        finite += list(coordinate["signed_projection"].values()) + list(coordinate["residual"].values())
        finite += list(target["cells"].values()) + list(target["behavior_signed_projection"].values())
        finite += list(control["margin_rms_fraction"].values()) + [control["median_kl"], control["max_kl"], control["top1_flip_fraction"]]
    forward = json.loads(FORWARD.read_text()); noise = json.loads(NOISE.read_text())
    pa = (forward["terminal"] == "ood_rank48_source_to_reader_program" and noise["terminal"] == "noise_robust_ood_rank48_program"
          and ctx["hashes_ok"] and ctx["reader_ok"] and ctx["reader_orientation"] <= 1e-6 and len(support) == 48
          and set(counts) == {1} and set(ccounts) == {1} and all(math.isfinite(float(v)) for v in finite))
    pb = all(min(r["coordinate"]["signed_projection"].values()) >= .75 and r["coordinate"]["mean_residual"] <= .2 for r in reports.values())
    pc = all(min(r["target"]["behavior_signed_projection"].values()) >= .8 and r["target"]["worst_target_residual"] <= .15 for r in reports.values())
    pd = all(max(r["control"]["margin_rms_fraction"].values()) <= .1 and r["control"]["median_kl"] <= .02 and r["control"]["top1_flip_fraction"] <= .05 for r in reports.values())
    em, om = min(reports["even_fit"]["coordinate"]["signed_projection"].values()), min(reports["odd_fit"]["coordinate"]["signed_projection"].values())
    task_diff = max(abs(reports["even_fit"]["target"]["behavior_signed_projection"][task]-reports["odd_fit"]["target"]["behavior_signed_projection"][task]) for task in ("temporal", "iswas"))
    pe = abs(em-om) <= .05 and task_diff <= .05
    predictions = {"pred_a_authority_bidirectional_population_finiteness_and_price": bool(pa), "pred_b_reverse_ood_coordinates_transfer": bool(pb),
                   "pred_c_reverse_ood_behavior_is_necessary": bool(pc), "pred_d_reverse_ood_controls_are_selective": bool(pd), "pred_e_reverse_crossfit_is_stable": bool(pe)}
    summary = {"coordinate_projection_min": min(em, om), "coordinate_mean_residual_max": max(r["coordinate"]["mean_residual"] for r in reports.values()),
               "target_projection_min": min(min(r["target"]["behavior_signed_projection"].values()) for r in reports.values()),
               "target_worst_max": max(r["target"]["worst_target_residual"] for r in reports.values()),
               "control_margin_max": max(max(r["control"]["margin_rms_fraction"].values()) for r in reports.values()),
               "control_median_kl_max": max(r["control"]["median_kl"] for r in reports.values()),
               "control_flip_max": max(r["control"]["top1_flip_fraction"] for r in reports.values())}
    terminal = "invalid" if not pa else "bidirectional_ood_rank48_program" if all(predictions.values()) else "reverse_necessity_failure"
    result = {"schema": "temporal_five_mlp_rank48_reverse_ood_result_v1", "started_utc": started, "finished_utc": now(),
              "serial_seconds": time.perf_counter()-tic, "authority_sha256": EXPECTED, "support": support, "reports": reports,
              "summary": summary, "predictions": predictions, "terminal": terminal,
              "price": {"model_forwards": 25, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}}
    atomic_create_json(OUT, result); print(json.dumps({k: result[k] for k in ("summary", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
