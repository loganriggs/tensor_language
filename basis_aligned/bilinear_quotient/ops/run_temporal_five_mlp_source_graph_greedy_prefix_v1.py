#!/usr/bin/env python3
"""Nested source-graph construction from the frozen component-deletion ranking."""
# BQGATE: EXPERIMENT pred_a_authority_self_clamp_finiteness_and_price pred_b_a_strictly_pruned_prefix_is_jointly_feasible pred_c_importance_pool_or_modest_extension_suffices pred_d_selected_arm_contains_frozen_90pct_pool pred_e_selected_arm_is_crossfit_stable
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
PRIOR = ROOT / "circuits/prior_art/temporal_five_mlp_source_graph_greedy_prefix_v1.json"
DELETION = ROOT / "circuits/followups/temporal_five_mlp_source_graph_component_deletion_atlas_v1_result.json"
HELPER = ROOT / "ops/compiled_response_graph.py"
OUT = ROOT / "circuits/followups/temporal_five_mlp_source_graph_greedy_prefix_v1_result.json"
EXPECTED = {"deletion": "b674de9220b3fb3b13ded3f48d908614ba363c10ffcc34faf9a6a8637cc3b10c",
            "helper": "2de68cab2af7e9f1619bedc9c0692780574e015a64c3c2aaf7fbc53fd5d0f32a"}
PREFIXES = [("top20", 0), ("rank27", 27), ("rank35", 35), ("rank45", 45), ("rank55", 55),
            ("rank65", 65), ("rank75", 75), ("rank85", 85), ("rank95", 95), ("rank110", 110)]


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def main():
    observed = {"deletion": sha(DELETION), "helper": sha(HELPER)}
    if observed != EXPECTED: raise RuntimeError(f"greedy-prefix authority changed: {observed}")
    dry = {"candidate_id": "temporal_auxiliary.five_mlp_source_graph_greedy_prefix_v1", "dryrun": True,
           "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
           "model_forwards_max": 77, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1": print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    started, tic = now(), time.perf_counter(); backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch
    ctx = graph.prepare(backend); fresh = ctx["fresh"]; deletion = json.loads(DELETION.read_text())
    top20, ranking, frozen_pool = deletion["top20"], deletion["ranking"], deletion["candidate_pool"]
    self_output, self_ai, self_mlp, _ = source.restricted_graph(backend, fresh["batch"], fresh["donor_batch"], fresh["base_full"], top20, source=False)
    self_error = max(float((atlasrun.states(torch, backend, self_output, fresh["rows"])-fresh["base_state"]).abs().max()),
                     max(float((self_ai[i]-fresh["base"][5][i]).abs().max()) for i in self_ai),
                     max(float((self_mlp[i]-fresh["base"][3][i]).abs().max()) for i in self_mlp))
    reports = {}; finite = [ctx["reader_orientation"], ctx["training"]["capture_error"], self_error]
    for name, n in PREFIXES:
        chosen = set(top20) | set(ranking[:n]); allowed = [site for site in graph.ALL_UPSTREAM if site in chosen]
        _go, gai, gmi, source_counts = source.restricted_graph(backend, fresh["batch"], fresh["donor_batch"], fresh["base_full"], allowed, source=True)
        _co, cai, cmi, control_counts = source.restricted_graph(backend, fresh["control_batch"], fresh["control_donor_batch"], fresh["control_base_full"], allowed, source=True)
        report = {"rank_prefix": n, "allowed_count": len(allowed), "allowed_sites": allowed,
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
    feasible_names = [name for name, _ in PREFIXES if reports[name]["jointly_feasible"]]
    selected = feasible_names[0] if feasible_names else None
    stable = False
    if selected:
        fits = reports[selected]["fits"]; emin = min(fits["even_fit"]["coordinate"]["signed_projection"].values()); omin = min(fits["odd_fit"]["coordinate"]["signed_projection"].values())
        task_diff = max(abs(fits["even_fit"]["target"]["behavior_signed_projection"][task]-fits["odd_fit"]["target"]["behavior_signed_projection"][task]) for task in ("temporal", "iswas"))
        stable = abs(emin-omin) <= .05 and task_diff <= .05
    pa = ctx["hashes_ok"] and ctx["reader_ok"] and ctx["reader_orientation"] <= 1e-6 and self_error <= 1e-4 and all(math.isfinite(float(v)) for v in finite)
    selected_sites = set(reports[selected]["allowed_sites"]) if selected else set()
    predictions = {"pred_a_authority_self_clamp_finiteness_and_price": bool(pa),
                   "pred_b_a_strictly_pruned_prefix_is_jointly_feasible": selected is not None and reports[selected]["allowed_count"] < 110,
                   "pred_c_importance_pool_or_modest_extension_suffices": selected is not None and reports[selected]["allowed_count"] <= 55,
                   "pred_d_selected_arm_contains_frozen_90pct_pool": selected is not None and set(frozen_pool).issubset(selected_sites),
                   "pred_e_selected_arm_is_crossfit_stable": bool(stable)}
    compact = {name: {"rank_prefix": report["rank_prefix"], "allowed_count": report["allowed_count"], "jointly_feasible": report["jointly_feasible"],
                      "coordinate_projection_min": min(min(f["coordinate"]["signed_projection"].values()) for f in report["fits"].values()),
                      "target_projection_min": min(min(f["target"]["behavior_signed_projection"].values()) for f in report["fits"].values()),
                      "target_worst_max": max(f["target"]["worst_target_residual"] for f in report["fits"].values()),
                      "control_margin_max": max(max(f["control"]["margin_rms_fraction"].values()) for f in report["fits"].values()),
                      "control_median_kl_max": max(f["control"]["median_kl"] for f in report["fits"].values()),
                      "control_flip_max": max(f["control"]["top1_flip_fraction"] for f in report["fits"].values())} for name, report in reports.items()}
    terminal = "invalid" if not pa else "greedy_pruned_source_to_reader_program" if all(predictions.values()) else "greedy_prefix_partial_or_null"
    result = {"schema": "temporal_five_mlp_source_graph_greedy_prefix_result_v1", "started_utc": started, "finished_utc": now(),
              "serial_seconds": time.perf_counter()-tic, "authority_sha256": EXPECTED, "selected_arm": selected,
              "reports": reports, "arm_summary": compact, "predictions": predictions, "terminal": terminal,
              "price": {"model_forwards": 77, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}}
    atomic_create_json(OUT, result); print(json.dumps({k: result[k] for k in ("selected_arm", "arm_summary", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
