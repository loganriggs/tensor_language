#!/usr/bin/env python3
"""Complete the four-band subset lattice for the source-driven compiled graph."""
# BQGATE: EXPERIMENT pred_a_authority_self_clamp_finiteness_and_price pred_b_complete_lattice_contains_a_pruned_feasible_arm pred_c_selected_arm_is_crossfit_stable pred_d_late_and_terminal_are_jointly_needed pred_e_complete_lattice_is_monotone_at_ceiling
from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import compiled_response_graph as graph
import run_temporal_five_mlp_source_clamped_compiled_graph_v1 as source
import run_temporal_five_mlp_source_clamped_layer_band_addback_v1 as addback
import run_temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1 as atlasrun
import run_temporal_five_mlp_compiled_coordinate_upstream_atlas_v1 as atlas

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_five_mlp_source_clamped_band_lattice_completion_v1.json"
V1 = ROOT / "circuits/followups/temporal_five_mlp_source_clamped_layer_band_addback_v1_result.json"
HELPER = ROOT / "ops/compiled_response_graph.py"
OUT = ROOT / "circuits/followups/temporal_five_mlp_source_clamped_band_lattice_completion_v1_result.json"
EXPECTED = {"v1": "558e7037f89abc1d2f14d370f5cbb48a0a4587a58501aa25394f153626473409",
            "helper": "2de68cab2af7e9f1619bedc9c0692780574e015a64c3c2aaf7fbc53fd5d0f32a"}
ARMS = [("early_late", ("early", "late")), ("early_terminal", ("early", "terminal")),
        ("middle_late", ("middle", "late")), ("middle_terminal", ("middle", "terminal")),
        ("late_terminal", ("late", "terminal")),
        ("early_late_terminal", ("early", "late", "terminal")),
        ("middle_late_terminal", ("middle", "late", "terminal"))]


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def compact(report):
    fits = report["fits"]
    return {"bands": report["bands"], "allowed_count": report["allowed_count"], "jointly_feasible": report["jointly_feasible"],
            "coordinate_projection_min": min(min(f["coordinate"]["signed_projection"].values()) for f in fits.values()),
            "target_projection_min": min(min(f["target"]["behavior_signed_projection"].values()) for f in fits.values()),
            "target_worst_max": max(f["target"]["worst_target_residual"] for f in fits.values()),
            "control_margin_max": max(max(f["control"]["margin_rms_fraction"].values()) for f in fits.values()),
            "control_median_kl_max": max(f["control"]["median_kl"] for f in fits.values()),
            "control_flip_max": max(f["control"]["top1_flip_fraction"] for f in fits.values())}


def main():
    observed = {"v1": sha(V1), "helper": sha(HELPER)}
    if observed != EXPECTED: raise RuntimeError(f"band-lattice authority changed: {observed}")
    dry = {"candidate_id": "temporal_auxiliary.five_mlp_source_clamped_band_lattice_completion_v1", "dryrun": True,
           "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
           "model_forwards_max": 59, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1": print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    started, tic = now(), time.perf_counter(); backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch
    ctx = graph.prepare(backend); fresh = ctx["fresh"]; old = json.loads(V1.read_text()); top = old["top20"]
    self_output, self_ai, self_mlp, _ = source.restricted_graph(
        backend, fresh["batch"], fresh["donor_batch"], fresh["base_full"], top, source=False)
    self_error = max(float((atlasrun.states(torch, backend, self_output, fresh["rows"])-fresh["base_state"]).abs().max()),
                     max(float((self_ai[i]-fresh["base"][5][i]).abs().max()) for i in self_ai),
                     max(float((self_mlp[i]-fresh["base"][3][i]).abs().max()) for i in self_mlp))
    reports = {}; finite = [ctx["reader_orientation"], ctx["training"]["capture_error"], self_error]
    for name, bands in ARMS:
        allowed = graph.expanded_pool(top, bands)
        _go, gai, gmi, source_counts = source.restricted_graph(backend, fresh["batch"], fresh["donor_batch"], fresh["base_full"], allowed, source=True)
        _co, cai, cmi, control_counts = source.restricted_graph(backend, fresh["control_batch"], fresh["control_donor_batch"], fresh["control_base_full"], allowed, source=True)
        report = {"bands": list(bands), "allowed_count": len(allowed), "allowed_sites": list(allowed),
                  "source_change_count": sorted(set(source_counts)), "control_source_change_count": sorted(set(control_counts)), "fits": {}}
        for label, qs in ctx["projectors"].items():
            generated = atlas.compiled_coordinates(backend, fresh["rows"], qs, fresh["base"][2], fresh["base"][5], gai, fresh["base"][3], gmi, ctx["rank64_bases"][label])
            cgenerated = atlas.compiled_coordinates(backend, fresh["controls"], qs, fresh["control_base"][2], fresh["control_base"][5], cai, fresh["control_base"][3], cmi, ctx["rank64_bases"][label])
            report["fits"][label] = addback.fit_arm(backend, ctx, allowed, label, qs, ctx["rank64_bases"][label], generated, cgenerated)
        report["jointly_feasible"] = addback.feasible(report); reports[name] = report
        for fit in report["fits"].values():
            finite += list(fit["coordinate"]["signed_projection"].values()) + list(fit["coordinate"]["residual"].values())
            finite += list(fit["target"]["cells"].values()) + list(fit["target"]["behavior_signed_projection"].values())
            finite += list(fit["control"]["margin_rms_fraction"].values()) + [fit["control"]["median_kl"], fit["control"]["max_kl"], fit["control"]["top1_flip_fraction"]]
    combined = {name: {**summary, "bands": old["arms"][name]["bands"]} for name, summary in old["arm_summary"].items()}
    combined.update({name: compact(report) for name, report in reports.items()})
    feasible_names = [name for name, report in combined.items() if name != "all" and report["jointly_feasible"]]
    selected = min(feasible_names, key=lambda name: (combined[name]["allowed_count"], name)) if feasible_names else None
    stable = False
    if selected in reports:
        fits = reports[selected]["fits"]; emin = min(fits["even_fit"]["coordinate"]["signed_projection"].values()); omin = min(fits["odd_fit"]["coordinate"]["signed_projection"].values())
        task_diff = max(abs(fits["even_fit"]["target"]["behavior_signed_projection"][task]-fits["odd_fit"]["target"]["behavior_signed_projection"][task]) for task in ("temporal", "iswas"))
        stable = abs(emin-omin) <= .05 and task_diff <= .05
    late_terminal_needed = bool(feasible_names) and all({"late", "terminal"}.issubset(combined[name]["bands"]) for name in feasible_names)
    predecessors = ("early_middle_late", "early_middle_terminal", "early_late_terminal", "middle_late_terminal")
    monotone = all(combined["all"][metric] >= max(combined[name][metric] for name in predecessors) for metric in ("coordinate_projection_min", "target_projection_min"))
    pa = ctx["hashes_ok"] and ctx["reader_ok"] and ctx["reader_orientation"] <= 1e-6 and self_error <= 1e-4 and all(math.isfinite(float(v)) for v in finite)
    predictions = {"pred_a_authority_self_clamp_finiteness_and_price": bool(pa),
                   "pred_b_complete_lattice_contains_a_pruned_feasible_arm": selected is not None,
                   "pred_c_selected_arm_is_crossfit_stable": bool(stable),
                   "pred_d_late_and_terminal_are_jointly_needed": bool(late_terminal_needed),
                   "pred_e_complete_lattice_is_monotone_at_ceiling": bool(monotone)}
    terminal = "invalid" if not pa else "pruned_band_dependency_support" if all(predictions.values()) else "all_bands_required_at_coarse_resolution"
    result = {"schema": "temporal_five_mlp_source_clamped_band_lattice_completion_result_v1", "started_utc": started, "finished_utc": now(),
              "serial_seconds": time.perf_counter()-tic, "authority_sha256": EXPECTED, "new_arms": reports, "complete_lattice_summary": combined,
              "selected_arm": selected, "predictions": predictions, "terminal": terminal,
              "price": {"model_forwards": 59, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}}
    atomic_create_json(OUT, result); print(json.dumps({k: result[k] for k in ("selected_arm", "complete_lattice_summary", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
