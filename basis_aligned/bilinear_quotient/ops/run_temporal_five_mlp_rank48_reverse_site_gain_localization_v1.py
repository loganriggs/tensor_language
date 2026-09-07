#!/usr/bin/env python3
"""Localize the rare fresh control flip by reverting one response-site gain."""
# BQGATE: EXPERIMENT pred_a_authority_replay_finiteness_and_exact_price pred_b_one_site_localizes_the_control_flip pred_c_localized_reversion_preserves_target_calibration pred_d_localization_is_crossfit_stable pred_e_localized_site_is_mechanistically_specific
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
PRIOR = ROOT / "circuits/prior_art/temporal_five_mlp_rank48_reverse_site_gain_localization_v1.json"
PARENT = ROOT / "circuits/followups/temporal_five_mlp_rank48_reverse_gain125_fresh_v1_result.json"
DOSE = ROOT / "circuits/followups/temporal_five_mlp_rank48_reverse_gain_dose_v1_result.json"
CORRECTION = ROOT / "circuits/followups/temporal_five_mlp_rank48_reverse_gain_dose_v1_price_correction.json"
SUPPORT = ROOT / "circuits/followups/temporal_five_mlp_rank49_joint_single_deletion_v1_result.json"
TCAP = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v13_capability_v1_result.json"
ICAP = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v12_capability_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_five_mlp_rank48_reverse_site_gain_localization_v1_result.json"
EXPECTED = {
    "parent": "6fee15243ad80cd7b21b86c0001474d11b5789d955d1cd90b793fc65584dd3d9",
    "dose": "a1f00d208ec1a370d9e9d995de0620de70c6202ef242d46019608635552b9aa9",
    "correction": "f3410f081becb2951db3b214af2330d58db638ab3b5e7cfb543a9d971188c8a7",
    "support": "92384a4496c7e3ceb5dbccecfd8ef65a61e7f5c219e52b5cf919b91ef1ba5fe2",
    "tcap": "e053d3381680ce5a933356a060448466d7567e3079c4a2b9a5bff262bd98b9c1",
    "icap": "67cb3efbd1ea86f98f94a826922928229a7c7b0a247f218778fc4960a6e8c6f4",
}
SITES = tuple(sorted(comp.SITES))
ARMS = ("all_125",) + tuple(f"revert_{site}" for site in SITES)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def gains(values, reverted):
    return {site: value * (1.0 if site == reverted else 1.25) for site, value in values.items()}


def target_calibrated(record):
    return (all(.9 <= record["target"]["behavior_signed_projection"][task] <= 1.1 for task in ("temporal", "iswas"))
            and record["target"]["worst_target_residual"] <= .15)


def control_clear(record):
    return (record["control"]["top1_flip_fraction"] == 0.0
            and max(record["control"]["margin_rms_fraction"].values()) <= .1
            and record["control"]["median_kl"] <= .02)


def main():
    observed = {key: sha(path) for key, path in {"parent": PARENT, "dose": DOSE, "correction": CORRECTION,
                                                "support": SUPPORT, "tcap": TCAP, "icap": ICAP}.items()}
    if observed != EXPECTED: raise RuntimeError(f"site-gain authority changed: {observed}")
    dry = {"candidate_id": "temporal_auxiliary.five_mlp_rank48_reverse_site_gain_localization_v1",
           "dryrun": True, "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
           "arms": ARMS, "model_forwards_exact": 57, "fit_updates": 0, "model_updates": 0,
           "transformer_backwards": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    started, tic = now(), time.perf_counter(); backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch
    oodctx.ood_t, oodctx.ood_i = fresh_t, fresh_i
    ctx = oodctx.prepare(backend, TCAP, ICAP); fresh = ctx["fresh"]
    support = json.loads(SUPPORT.read_text())["selected_support"]; parent = json.loads(PARENT.read_text())
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
        for arm in ARMS:
            reverted = None if arm == "all_125" else arm.removeprefix("revert_")
            gd, cgd = gains(generated, reverted), gains(cgenerated, reverted)
            target_output = pruned.run_factors(backend, fresh["donor_batch"], fresh["donor"][1], qs,
                {s: v for s, v in gd.items() if atlasrun.site_parts(s)[0] == "attn"},
                {s: v for s, v in gd.items() if atlasrun.site_parts(s)[0] == "mlp"}, use_attention=True, use_mlp=True)
            target = ood.target_report(backend, fresh["rows"], donor_state, reverse_full_state,
                                       atlasrun.states(torch, backend, target_output, fresh["rows"]), ctx["reader"], fresh["temporal_n"])
            control_output = pruned.run_factors(backend, fresh["control_donor_batch"], fresh["control_donor"][1], qs,
                {s: v for s, v in cgd.items() if atlasrun.site_parts(s)[0] == "attn"},
                {s: v for s, v in cgd.items() if atlasrun.site_parts(s)[0] == "mlp"}, use_attention=True, use_mlp=True)
            control = klfit.control_metrics(backend, control_ctx, das.head_logits(backend, atlasrun.states(torch, backend, control_output, fresh["controls"])).float())
            reports[label][arm] = {"target": target, "control": control}
            finite += list(target["cells"].values()) + list(target["behavior_signed_projection"].values())
            finite += list(control["margin_rms_fraction"].values()) + [control["median_kl"], control["max_kl"], control["top1_flip_fraction"]]
    replay_error = max(
        max(abs(reports[label]["all_125"]["target"]["behavior_signed_projection"][task] - parent["reports"][label]["target"]["behavior_signed_projection"][task])
            for label in reports for task in ("temporal", "iswas")),
        max(abs(reports[label]["all_125"]["control"][metric] - parent["reports"][label]["control"][metric])
            for label in reports for metric in ("median_kl", "max_kl", "top1_flip_fraction")))
    clearing = [site for site in SITES if all(control_clear(reports[label][f"revert_{site}"]) for label in reports)]
    selected = clearing[0] if clearing else None
    pa = (ctx["hashes_ok"] and ctx["reader_ok"] and len(support) == 48 and set(counts) == {1} and set(ccounts) == {1}
          and replay_error <= .002 and all(math.isfinite(float(value)) for value in finite))
    pb = bool(clearing)
    pc = bool(selected) and all(target_calibrated(reports[label][f"revert_{selected}"]) for label in reports)
    pd = bool(selected) and all(control_clear(reports[label][f"revert_{selected}"]) and target_calibrated(reports[label][f"revert_{selected}"]) for label in reports)
    pe = any(not (all(control_clear(reports[label][f"revert_{site}"]) for label in reports)
                      and all(target_calibrated(reports[label][f"revert_{site}"]) for label in reports)) for site in SITES if site != selected)
    predictions = {"pred_a_authority_replay_finiteness_and_exact_price": bool(pa),
                   "pred_b_one_site_localizes_the_control_flip": bool(pb),
                   "pred_c_localized_reversion_preserves_target_calibration": bool(pc),
                   "pred_d_localization_is_crossfit_stable": bool(pd),
                   "pred_e_localized_site_is_mechanistically_specific": bool(pe)}
    arm_summary = {site: {label: {"behavior": reports[label][f"revert_{site}"]["target"]["behavior_signed_projection"],
                                        "worst_residual": reports[label][f"revert_{site}"]["target"]["worst_target_residual"],
                                        "flip_fraction": reports[label][f"revert_{site}"]["control"]["top1_flip_fraction"]}
                               for label in reports} for site in SITES}
    terminal = "invalid" if not pa else "site_specific_gain_map" if all(predictions.values()) else "distributed_gain_threshold"
    result = {"schema": "temporal_five_mlp_rank48_reverse_site_gain_localization_result_v1",
              "started_utc": started, "finished_utc": now(), "serial_seconds": time.perf_counter()-tic,
              "authority_sha256": EXPECTED, "sites": SITES, "clearing_sites": clearing, "selected_site": selected,
              "replay_max_abs": replay_error, "arm_summary": arm_summary, "reports": reports,
              "predictions": predictions, "terminal": terminal,
              "price": {"model_forwards": 57, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}}
    atomic_create_json(OUT, result)
    print(json.dumps({"clearing_sites": clearing, "selected_site": selected, "replay_max_abs": replay_error,
                      "predictions": predictions, "terminal": terminal, "price": result["price"]}, sort_keys=True))


if __name__ == "__main__": main()
