#!/usr/bin/env python3
"""Confirm frozen rank48 reverse gain 1.25 on temporal-v13/iswas-v12."""
# BQGATE: EXPERIMENT pred_a_authority_population_finiteness_and_price pred_b_gain125_reverse_behavior_transfers pred_c_reverse_coordinates_remain_accurate pred_d_gain125_controls_remain_selective pred_e_crossfit_is_stable
from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

import circuit_candidate_temporal_auxiliary_fresh_cues_v13 as fresh_t
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v12 as fresh_i
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
PRIOR = ROOT / "circuits/prior_art/temporal_five_mlp_rank48_reverse_gain125_fresh_v1.json"
DOSE = ROOT / "circuits/followups/temporal_five_mlp_rank48_reverse_gain_dose_v1_result.json"
CORRECTION = ROOT / "circuits/followups/temporal_five_mlp_rank48_reverse_gain_dose_v1_price_correction.json"
SUPPORT = ROOT / "circuits/followups/temporal_five_mlp_rank49_joint_single_deletion_v1_result.json"
TCAP = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v13_capability_v1_result.json"
ICAP = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v12_capability_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_five_mlp_rank48_reverse_gain125_fresh_v1_result.json"
EXPECTED = {"dose": "a1f00d208ec1a370d9e9d995de0620de70c6202ef242d46019608635552b9aa9",
            "correction": "f3410f081becb2951db3b214af2330d58db638ab3b5e7cfb543a9d971188c8a7",
            "support": "92384a4496c7e3ceb5dbccecfd8ef65a61e7f5c219e52b5cf919b91ef1ba5fe2",
            "tcap": "e053d3381680ce5a933356a060448466d7567e3079c4a2b9a5bff262bd98b9c1",
            "icap": "67cb3efbd1ea86f98f94a826922928229a7c7b0a247f218778fc4960a6e8c6f4",
            "tbuilder": "3f738bf2fb2d4a5425dba85eaf948d7ed888e4b891666ff77a51c8638acd2509",
            "ibuilder": "2734cbeceb4e6979dab22fe5b24870386874ac2f905ae426e6e689548e43e8a2"}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def main():
    paths = {"dose": DOSE, "correction": CORRECTION, "support": SUPPORT, "tcap": TCAP, "icap": ICAP,
             "tbuilder": Path(fresh_t.__file__), "ibuilder": Path(fresh_i.__file__)}
    observed = {key: sha(path) for key, path in paths.items()}
    if observed != EXPECTED: raise RuntimeError(f"fresh gain authority changed: {observed}")
    dry = {"candidate_id": "temporal_auxiliary.five_mlp_rank48_reverse_gain125_fresh_v1", "dryrun": True,
           "gpu_accessed": False, "model_loaded": False, "queue_touched": False, "gain": 1.25,
           "model_forwards_exact": 25, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1": print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    started, tic = now(), time.perf_counter(); backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch
    oodctx.ood_t, oodctx.ood_i = fresh_t, fresh_i
    ctx = oodctx.prepare(backend, TCAP, ICAP); fresh = ctx["fresh"]; support = json.loads(SUPPORT.read_text())["selected_support"]
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
        reference = atlas.compiled_coordinates(backend, fresh["rows"], qs, fresh["donor"][2], fresh["donor"][5], fresh["base"][5], fresh["donor"][3], fresh["base"][3], ctx["rank64_bases"][label])
        generated = atlas.compiled_coordinates(backend, fresh["rows"], qs, fresh["donor"][2], fresh["donor"][5], gai, fresh["donor"][3], gmi, ctx["rank64_bases"][label])
        coordinate = atlas.coordinate_report(torch, atlas.vectors(backend, fresh["donor_batch"], qs, reference), atlas.vectors(backend, fresh["donor_batch"], qs, generated))
        gd = {site: value * 1.25 for site, value in generated.items()}
        target_output = pruned.run_factors(backend, fresh["donor_batch"], fresh["donor"][1], qs,
            {s: v for s, v in gd.items() if atlasrun.site_parts(s)[0] == "attn"}, {s: v for s, v in gd.items() if atlasrun.site_parts(s)[0] == "mlp"}, use_attention=True, use_mlp=True)
        target = ood.target_report(backend, fresh["rows"], donor_state, reverse_full_state, atlasrun.states(torch, backend, target_output, fresh["rows"]), ctx["reader"], fresh["temporal_n"])
        cgenerated = atlas.compiled_coordinates(backend, fresh["controls"], qs, fresh["control_donor"][2], fresh["control_donor"][5], cai, fresh["control_donor"][3], cmi, ctx["rank64_bases"][label])
        cgd = {site: value * 1.25 for site, value in cgenerated.items()}
        control_output = pruned.run_factors(backend, fresh["control_donor_batch"], fresh["control_donor"][1], qs,
            {s: v for s, v in cgd.items() if atlasrun.site_parts(s)[0] == "attn"}, {s: v for s, v in cgd.items() if atlasrun.site_parts(s)[0] == "mlp"}, use_attention=True, use_mlp=True)
        control = klfit.control_metrics(backend, control_ctx, das.head_logits(backend, atlasrun.states(torch, backend, control_output, fresh["controls"])).float())
        reports[label] = {"coordinate": coordinate, "target": target, "control": control}
        finite += list(coordinate["signed_projection"].values()) + list(coordinate["residual"].values()) + list(target["cells"].values()) + list(target["behavior_signed_projection"].values())
        finite += list(control["margin_rms_fraction"].values()) + [control["median_kl"], control["max_kl"], control["top1_flip_fraction"]]
    caps = json.loads(TCAP.read_text()), json.loads(ICAP.read_text())
    pa = (caps[0]["terminal"] == "manifest" and caps[1]["terminal"] == "screen" and ctx["hashes_ok"] and ctx["reader_ok"]
          and len(support) == 48 and set(counts) == {1} and set(ccounts) == {1} and all(math.isfinite(float(v)) for v in finite))
    pb = all(.9 <= r["target"]["behavior_signed_projection"][task] <= 1.1 for r in reports.values() for task in ("temporal", "iswas")) and all(r["target"]["worst_target_residual"] <= .15 for r in reports.values())
    pc = all(min(r["coordinate"]["signed_projection"].values()) >= .75 and r["coordinate"]["mean_residual"] <= .2 for r in reports.values())
    pd = all(max(r["control"]["margin_rms_fraction"].values()) <= .1 and r["control"]["median_kl"] <= .02 and r["control"]["top1_flip_fraction"] <= .05 for r in reports.values())
    pe = (max(abs(reports["even_fit"]["target"]["behavior_signed_projection"][task]-reports["odd_fit"]["target"]["behavior_signed_projection"][task]) for task in ("temporal", "iswas")) <= .05
          and abs(min(reports["even_fit"]["coordinate"]["signed_projection"].values())-min(reports["odd_fit"]["coordinate"]["signed_projection"].values())) <= .05)
    predictions = {"pred_a_authority_population_finiteness_and_price": bool(pa), "pred_b_gain125_reverse_behavior_transfers": bool(pb),
                   "pred_c_reverse_coordinates_remain_accurate": bool(pc), "pred_d_gain125_controls_remain_selective": bool(pd), "pred_e_crossfit_is_stable": bool(pe)}
    summary = {"coordinate_projection_min": min(min(r["coordinate"]["signed_projection"].values()) for r in reports.values()),
               "behavior_projection_min": min(r["target"]["behavior_signed_projection"][t] for r in reports.values() for t in ("temporal", "iswas")),
               "behavior_projection_max": max(r["target"]["behavior_signed_projection"][t] for r in reports.values() for t in ("temporal", "iswas")),
               "worst_target_residual": max(r["target"]["worst_target_residual"] for r in reports.values()),
               "control_margin_max": max(max(r["control"]["margin_rms_fraction"].values()) for r in reports.values()),
               "control_median_kl_max": max(r["control"]["median_kl"] for r in reports.values()), "control_flip_max": max(r["control"]["top1_flip_fraction"] for r in reports.values())}
    terminal = "invalid" if not pa else "fresh_calibrated_bidirectional_rank48_program" if all(predictions.values()) else "gain_calibration_transfer_failure"
    result = {"schema": "temporal_five_mlp_rank48_reverse_gain125_fresh_result_v1", "started_utc": started, "finished_utc": now(), "serial_seconds": time.perf_counter()-tic,
              "authority_sha256": EXPECTED, "scope": {"temporal": "fresh_v13", "iswas": "corroborative_v12"}, "gain": 1.25, "reports": reports, "summary": summary,
              "predictions": predictions, "terminal": terminal, "price": {"model_forwards": 25, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}}
    atomic_create_json(OUT, result); print(json.dumps({k: result[k] for k in ("summary", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
