#!/usr/bin/env python3
"""Frozen dose response for the rank48 reverse-OOD response interface."""
# BQGATE: EXPERIMENT pred_a_authority_reproduction_finiteness_and_price pred_b_reverse_behavior_has_monotone_dose_response pred_c_fixed_125_gain_calibrates_reverse_behavior pred_d_fixed_125_gain_remains_selective pred_e_calibrated_crossfit_is_stable
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
PRIOR = ROOT / "circuits/prior_art/temporal_five_mlp_rank48_reverse_gain_dose_v1.json"
REVERSE = ROOT / "circuits/followups/temporal_five_mlp_rank48_reverse_ood_v1_result.json"
SUPPORT = ROOT / "circuits/followups/temporal_five_mlp_rank49_joint_single_deletion_v1_result.json"
TCAP = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v12_capability_v1_result.json"
ICAP = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v11_capability_v1_result.json"
HELPER = ROOT / "ops/compiled_response_ood.py"
OUT = ROOT / "circuits/followups/temporal_five_mlp_rank48_reverse_gain_dose_v1_result.json"
EXPECTED = {
    "reverse": "68e7fab4d108d365771105441ff7ac4d67a5533122dc9592f1260d984870824b",
    "support": "92384a4496c7e3ceb5dbccecfd8ef65a61e7f5c219e52b5cf919b91ef1ba5fe2",
    "helper": "96b9e3aef3b64364c7fc4c49d0af9ff35827eadb4ca5bfbdf5d2271cb26ad808",
}
GAINS = (0.75, 1.0, 1.25, 1.5)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
def scaled(items, gain): return {site: value * gain for site, value in items.items()}


def main():
    observed = {"reverse": sha(REVERSE), "support": sha(SUPPORT), "helper": sha(HELPER)}
    if observed != EXPECTED: raise RuntimeError(f"reverse gain authority changed: {observed}")
    dry = {"candidate_id": "temporal_auxiliary.five_mlp_rank48_reverse_gain_dose_v1", "dryrun": True,
           "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
           "gains": GAINS, "model_forwards_max": 21, "fit_updates": 0, "model_updates": 0,
           "transformer_backwards": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1": print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    started, tic = now(), time.perf_counter(); backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch
    ctx = oodctx.prepare(backend, TCAP, ICAP); fresh = ctx["fresh"]
    support = json.loads(SUPPORT.read_text())["selected_support"]
    prior = json.loads(REVERSE.read_text())
    _do, donor_full = atlasrun.capture_native(backend, fresh["donor_batch"])
    _co, control_donor_full = atlasrun.capture_native(backend, fresh["control_donor_batch"])
    reverse_full_output, _ = atlasrun.run_patch(backend, fresh["donor_batch"], fresh["base"][1], comp.SITES)
    _go, gai, gmi, counts = source.restricted_graph(backend, fresh["donor_batch"], fresh["batch"], donor_full, support, source=True)
    _gx, cai, cmi, ccounts = source.restricted_graph(backend, fresh["control_donor_batch"], fresh["control_batch"], control_donor_full, support, source=True)
    donor_state = atlasrun.states(torch, backend, fresh["donor"][0], fresh["rows"])
    reverse_full_state = atlasrun.states(torch, backend, reverse_full_output, fresh["rows"])
    control_donor_state = atlasrun.states(torch, backend, fresh["control_donor"][0], fresh["controls"])
    control_ctx = {"rows": fresh["controls"], "base_logits": das.head_logits(backend, control_donor_state).float()}
    reports = {}; finite = [ctx["reader_orientation"], ctx["training"]["capture_error"]]
    for label, qs in ctx["projectors"].items():
        generated = atlas.compiled_coordinates(backend, fresh["rows"], qs, fresh["donor"][2], fresh["donor"][5], gai,
                                               fresh["donor"][3], gmi, ctx["rank64_bases"][label])
        cgenerated = atlas.compiled_coordinates(backend, fresh["controls"], qs, fresh["control_donor"][2], fresh["control_donor"][5], cai,
                                                fresh["control_donor"][3], cmi, ctx["rank64_bases"][label])
        reports[label] = {}
        for gain in GAINS:
            gd = scaled(generated, gain); cgd = scaled(cgenerated, gain)
            target_output = pruned.run_factors(backend, fresh["donor_batch"], fresh["donor"][1], qs,
                {s: v for s, v in gd.items() if atlasrun.site_parts(s)[0] == "attn"},
                {s: v for s, v in gd.items() if atlasrun.site_parts(s)[0] == "mlp"}, use_attention=True, use_mlp=True)
            target = ood.target_report(backend, fresh["rows"], donor_state, reverse_full_state,
                                       atlasrun.states(torch, backend, target_output, fresh["rows"]), ctx["reader"], fresh["temporal_n"])
            control_output = pruned.run_factors(backend, fresh["control_donor_batch"], fresh["control_donor"][1], qs,
                {s: v for s, v in cgd.items() if atlasrun.site_parts(s)[0] == "attn"},
                {s: v for s, v in cgd.items() if atlasrun.site_parts(s)[0] == "mlp"}, use_attention=True, use_mlp=True)
            control = klfit.control_metrics(backend, control_ctx, das.head_logits(backend, atlasrun.states(torch, backend, control_output, fresh["controls"])).float())
            reports[label][str(gain)] = {"target": target, "control": control}
            finite += list(target["cells"].values()) + list(target["behavior_signed_projection"].values())
            finite += list(control["margin_rms_fraction"].values()) + [control["median_kl"], control["max_kl"], control["top1_flip_fraction"]]
    reproduced = max(abs(reports[label]["1.0"]["target"]["behavior_signed_projection"][task] -
                         prior["reports"][label]["target"]["behavior_signed_projection"][task])
                     for label in reports for task in ("temporal", "iswas"))
    pa = (ctx["hashes_ok"] and ctx["reader_ok"] and len(support) == 48 and set(counts) == {1} and set(ccounts) == {1}
          and reproduced <= .002 and all(math.isfinite(float(v)) for v in finite))
    pb = all(all(reports[label][str(a)]["target"]["behavior_signed_projection"][task] <
                    reports[label][str(b)]["target"]["behavior_signed_projection"][task]
                    for a, b in zip(GAINS, GAINS[1:])) for label in reports for task in ("temporal", "iswas"))
    calibrated = {label: reports[label]["1.25"] for label in reports}
    pc = all(.9 <= r["target"]["behavior_signed_projection"][task] <= 1.1
             for r in calibrated.values() for task in ("temporal", "iswas")) and all(r["target"]["worst_target_residual"] <= .15 for r in calibrated.values())
    pd = all(max(r["control"]["margin_rms_fraction"].values()) <= .1 and r["control"]["median_kl"] <= .02
             and r["control"]["top1_flip_fraction"] <= .05 for r in calibrated.values())
    pe = max(abs(calibrated["even_fit"]["target"]["behavior_signed_projection"][task] -
                 calibrated["odd_fit"]["target"]["behavior_signed_projection"][task]) for task in ("temporal", "iswas")) <= .05
    predictions = {"pred_a_authority_reproduction_finiteness_and_price": bool(pa),
                   "pred_b_reverse_behavior_has_monotone_dose_response": bool(pb),
                   "pred_c_fixed_125_gain_calibrates_reverse_behavior": bool(pc),
                   "pred_d_fixed_125_gain_remains_selective": bool(pd),
                   "pred_e_calibrated_crossfit_is_stable": bool(pe)}
    summary = {"gain1_reproduction_max_abs": reproduced,
               "gain125_behavior_min": min(r["target"]["behavior_signed_projection"][t] for r in calibrated.values() for t in ("temporal", "iswas")),
               "gain125_behavior_max": max(r["target"]["behavior_signed_projection"][t] for r in calibrated.values() for t in ("temporal", "iswas")),
               "gain125_worst_target_residual": max(r["target"]["worst_target_residual"] for r in calibrated.values()),
               "gain125_control_margin_max": max(max(r["control"]["margin_rms_fraction"].values()) for r in calibrated.values()),
               "gain125_control_median_kl_max": max(r["control"]["median_kl"] for r in calibrated.values()),
               "gain125_control_flip_max": max(r["control"]["top1_flip_fraction"] for r in calibrated.values())}
    terminal = "invalid" if not pa else "scalar_reverse_interface_undergain" if all(predictions.values()) else "structural_reverse_miss"
    result = {"schema": "temporal_five_mlp_rank48_reverse_gain_dose_result_v1", "started_utc": started, "finished_utc": now(),
              "serial_seconds": time.perf_counter()-tic, "authority_sha256": EXPECTED, "gains": GAINS, "reports": reports,
              "summary": summary, "predictions": predictions, "terminal": terminal,
              "price": {"model_forwards": 21, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}}
    atomic_create_json(OUT, result); print(json.dumps({k: result[k] for k in ("summary", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
