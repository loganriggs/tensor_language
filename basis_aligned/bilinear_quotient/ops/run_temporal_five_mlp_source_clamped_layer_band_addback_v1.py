#!/usr/bin/env python3
"""Frozen layer-band addback for the source-driven compiled response graph."""
# BQGATE: EXPERIMENT pred_a_authority_self_clamp_finiteness_and_price pred_b_a_non_all_arm_is_jointly_feasible pred_c_selected_arm_is_crossfit_stable pred_d_selected_arm_is_strictly_pruned pred_e_addback_localizes_a_required_band
from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import compiled_response_graph as graph
import run_temporal_five_mlp_source_clamped_compiled_graph_v1 as source
import run_temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1 as atlasrun
import run_temporal_five_mlp_generic_subspace_complement_program_v1 as comp
import run_temporal_five_mlp_frozen_response_ood_regularization_v2 as ood
import run_temporal_five_mlp_factor_pruned_causal_program_v1 as pruned
import run_temporal_five_mlp_family_feasible_kl_refinement_v1 as klfit
import run_temporal_five_mlp_compiled_coordinate_upstream_atlas_v1 as atlas

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_five_mlp_source_clamped_layer_band_addback_v1.json"
UPSTREAM = ROOT / "circuits/followups/temporal_five_mlp_compiled_coordinate_upstream_atlas_v1_result.json"
SOURCE = ROOT / "circuits/followups/temporal_five_mlp_source_clamped_compiled_graph_v1_result.json"
ATTENTION = ROOT / "circuits/followups/temporal_five_mlp_attention_value_weight_pullback_v2_tolerance_audit_result.json"
MLP = ROOT / "circuits/followups/temporal_five_mlp_mlp_input_covector_ood_v2_result.json"
INTERFACE = ROOT / "circuits/followups/temporal_five_mlp_crossfit_response_weight_interface_v1_result.json"
TC = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v13_capability_v1_result.json"
IC = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v12_capability_v1_result.json"
HELPER = ROOT / "ops/compiled_response_graph.py"
OUT = ROOT / "circuits/followups/temporal_five_mlp_source_clamped_layer_band_addback_v1_result.json"
EXPECTED = {
    "source": "904ff7f22b3b264457c5e7ec8a1126f941a46a9f422edd7c0a1d63692fa25243",
    "upstream": "0b96284ebf7bd89ccdbf1dbc0fbdacd54e58eb16841865c48b985aa1a3c2da53",
    "attention": "42f9db9cf44e63a1930f454798b9d25b115cea94e4c45a8344694da6982637b4",
    "mlp": "10a6e169348cffb001f14d299ce05c3f33d80bd50e574dced2cac1228d436947",
    "interface": "578fa690f2f3962409bb50bd40a1c4c6f0b5ce7f88828e2e889a3a47bb20f013",
    "temporal_capability": "e053d3381680ce5a933356a060448466d7567e3079c4a2b9a5bff262bd98b9c1",
    "iswas_capability": "67cb3efbd1ea86f98f94a826922928229a7c7b0a247f218778fc4960a6e8c6f4",
    "helper": "2de68cab2af7e9f1619bedc9c0692780574e015a64c3c2aaf7fbc53fd5d0f32a"
}
ARMS = [
    ("top_only", ()), ("terminal", ("terminal",)), ("early", ("early",)),
    ("middle", ("middle",)), ("late", ("late",)),
    ("early_middle", ("early", "middle")),
    ("early_middle_late", ("early", "middle", "late")),
    ("early_middle_terminal", ("early", "middle", "terminal")),
    ("all", ("early", "middle", "late", "terminal"))]


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def fit_arm(backend, ctx, allowed, label, qs, bases, generated, control_generated):
    torch = backend.torch; fresh = ctx["fresh"]; rows = fresh["rows"]; crows = fresh["controls"]
    full_coords = atlas.compiled_coordinates(
        backend, rows, qs, fresh["base"][2], fresh["base"][5], fresh["donor"][5],
        fresh["base"][3], fresh["donor"][3], bases)
    coord = atlas.coordinate_report(
        torch, atlas.vectors(backend, fresh["batch"], qs, full_coords),
        atlas.vectors(backend, fresh["batch"], qs, generated))
    target_output = pruned.run_factors(
        backend, fresh["batch"], fresh["base"][1], qs,
        {s: v for s, v in generated.items() if atlasrun.site_parts(s)[0] == "attn"},
        {s: v for s, v in generated.items() if atlasrun.site_parts(s)[0] == "mlp"},
        use_attention=True, use_mlp=True)
    target = ood.target_report(
        backend, rows, fresh["base_state"], fresh["full_state"],
        atlasrun.states(torch, backend, target_output, rows), ctx["reader"], fresh["temporal_n"])
    control_output = pruned.run_factors(
        backend, fresh["control_batch"], fresh["control_base"][1], qs,
        {s: v for s, v in control_generated.items() if atlasrun.site_parts(s)[0] == "attn"},
        {s: v for s, v in control_generated.items() if atlasrun.site_parts(s)[0] == "mlp"},
        use_attention=True, use_mlp=True)
    control_ctx = {"rows": crows, "base_logits": das.head_logits(backend, fresh["control_base_state"]).float()}
    control = klfit.control_metrics(
        backend, control_ctx, das.head_logits(backend, atlasrun.states(torch, backend, control_output, crows)).float())
    return {"coordinate": coord, "target": target, "control": control}


def feasible(report):
    for fit in report["fits"].values():
        if min(fit["coordinate"]["signed_projection"].values()) < .75 or fit["coordinate"]["mean_residual"] > .2: return False
        if min(fit["target"]["behavior_signed_projection"].values()) < .8 or fit["target"]["worst_target_residual"] > .15: return False
        if max(fit["control"]["margin_rms_fraction"].values()) > .1 or fit["control"]["median_kl"] > .02 or fit["control"]["top1_flip_fraction"] > .05: return False
    return True


def main():
    paths = {"source": SOURCE, "upstream": UPSTREAM, "attention": ATTENTION, "mlp": MLP,
             "interface": INTERFACE, "temporal_capability": TC, "iswas_capability": IC, "helper": HELPER}
    observed = {key: sha(path) for key, path in paths.items()}
    if observed != EXPECTED: raise RuntimeError(f"layer-band authority changed: {observed}")
    dry = {"candidate_id": "temporal_auxiliary.five_mlp_source_clamped_layer_band_addback_v1", "dryrun": True,
           "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
           "model_forwards_max": 72, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1": print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    started, tic = now(), time.perf_counter(); backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch
    ctx = graph.prepare(backend); fresh = ctx["fresh"]; upstream = json.loads(UPSTREAM.read_text())
    top = upstream["top20"]["even_fit"]["sites"]
    top_agree = top == upstream["top20"]["odd_fit"]["sites"] and len(top) == 20
    self_output, self_ai, self_mlp, _ = source.restricted_graph(
        backend, fresh["batch"], fresh["donor_batch"], fresh["base_full"], top, source=False)
    self_state = atlasrun.states(torch, backend, self_output, fresh["rows"])
    self_error = max(float((self_state-fresh["base_state"]).abs().max()),
                     max(float((self_ai[i]-fresh["base"][5][i]).abs().max()) for i in self_ai),
                     max(float((self_mlp[i]-fresh["base"][3][i]).abs().max()) for i in self_mlp))
    reports = {}; finite = [ctx["reader_orientation"], ctx["training"]["capture_error"], self_error]
    for name, bands in ARMS:
        allowed = graph.expanded_pool(top, bands)
        _go, gai, gmi, source_counts = source.restricted_graph(
            backend, fresh["batch"], fresh["donor_batch"], fresh["base_full"], allowed, source=True)
        _co, cai, cmi, control_counts = source.restricted_graph(
            backend, fresh["control_batch"], fresh["control_donor_batch"], fresh["control_base_full"], allowed, source=True)
        arm = {"bands": list(bands), "allowed_count": len(allowed), "allowed_sites": list(allowed),
               "source_change_count": sorted(set(source_counts)), "control_source_change_count": sorted(set(control_counts)), "fits": {}}
        for label, qs in ctx["projectors"].items():
            generated = atlas.compiled_coordinates(
                backend, fresh["rows"], qs, fresh["base"][2], fresh["base"][5], gai,
                fresh["base"][3], gmi, ctx["rank64_bases"][label])
            control_generated = atlas.compiled_coordinates(
                backend, fresh["controls"], qs, fresh["control_base"][2], fresh["control_base"][5], cai,
                fresh["control_base"][3], cmi, ctx["rank64_bases"][label])
            arm["fits"][label] = fit_arm(backend, ctx, allowed, label, qs, ctx["rank64_bases"][label], generated, control_generated)
        arm["jointly_feasible"] = feasible(arm); reports[name] = arm
        for fit in arm["fits"].values():
            finite += list(fit["coordinate"]["signed_projection"].values()) + list(fit["coordinate"]["residual"].values())
            finite += list(fit["target"]["cells"].values()) + list(fit["target"]["behavior_signed_projection"].values())
            finite += list(fit["control"]["margin_rms_fraction"].values()) + [fit["control"]["median_kl"], fit["control"]["max_kl"], fit["control"]["top1_flip_fraction"]]
    candidates = [(report["allowed_count"], index, name) for index, (name, _) in enumerate(ARMS)
                  if name != "all" and (report := reports[name])["jointly_feasible"]]
    selected = min(candidates)[2] if candidates else None
    stable = False
    if selected:
        fits = reports[selected]["fits"]; emin = min(fits["even_fit"]["coordinate"]["signed_projection"].values()); omin = min(fits["odd_fit"]["coordinate"]["signed_projection"].values())
        task_diff = max(abs(fits["even_fit"]["target"]["behavior_signed_projection"][task]-fits["odd_fit"]["target"]["behavior_signed_projection"][task]) for task in ("temporal", "iswas"))
        stable = abs(emin-omin) <= .05 and task_diff <= .05
    top_min = {label: min(reports["top_only"]["fits"][label]["coordinate"]["signed_projection"].values()) for label in ctx["projectors"]}
    single_names = ("terminal", "early", "middle", "late")
    localized = any(all(min(reports[name]["fits"][label]["coordinate"]["signed_projection"].values())-top_min[label] >= .15 for label in ctx["projectors"]) for name in single_names)
    pa = ctx["hashes_ok"] and ctx["reader_ok"] and ctx["reader_orientation"] <= 1e-6 and top_agree and self_error <= 1e-4 and all(math.isfinite(float(v)) for v in finite)
    predictions = {"pred_a_authority_self_clamp_finiteness_and_price": bool(pa),
                   "pred_b_a_non_all_arm_is_jointly_feasible": selected is not None,
                   "pred_c_selected_arm_is_crossfit_stable": bool(stable),
                   "pred_d_selected_arm_is_strictly_pruned": selected is not None and reports[selected]["allowed_count"] < len(graph.ALL_UPSTREAM),
                   "pred_e_addback_localizes_a_required_band": bool(localized)}
    compact = {name: {"allowed_count": report["allowed_count"], "jointly_feasible": report["jointly_feasible"],
                      "coordinate_projection_min": min(min(f["coordinate"]["signed_projection"].values()) for f in report["fits"].values()),
                      "target_projection_min": min(min(f["target"]["behavior_signed_projection"].values()) for f in report["fits"].values()),
                      "target_worst_max": max(f["target"]["worst_target_residual"] for f in report["fits"].values()),
                      "control_margin_max": max(max(f["control"]["margin_rms_fraction"].values()) for f in report["fits"].values()),
                      "control_median_kl_max": max(f["control"]["median_kl"] for f in report["fits"].values()),
                      "control_flip_max": max(f["control"]["top1_flip_fraction"] for f in report["fits"].values())} for name, report in reports.items()}
    terminal = "invalid" if not pa else "dependency_complete_pruned_graph" if all(predictions.values()) else "band_addback_localization"
    result = {"schema": "temporal_five_mlp_source_clamped_layer_band_addback_result_v1", "started_utc": started, "finished_utc": now(),
              "serial_seconds": time.perf_counter()-tic, "authority_sha256": EXPECTED, "top20": top, "arms": reports,
              "selected_arm": selected, "arm_summary": compact, "predictions": predictions, "terminal": terminal,
              "price": {"model_forwards": 71, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}}
    atomic_create_json(OUT, result); print(json.dumps({k: result[k] for k in ("selected_arm", "arm_summary", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
