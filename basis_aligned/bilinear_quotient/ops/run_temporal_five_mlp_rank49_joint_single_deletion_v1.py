#!/usr/bin/env python3
"""Jointly test the ten lowest-impact single deletions from rank49."""
# BQGATE: EXPERIMENT pred_a_authority_self_clamp_finiteness_and_price pred_b_at_least_one_rank48_arm_is_jointly_feasible pred_c_at_least_two_rank48_arms_are_jointly_feasible pred_d_selected_deletion_is_outside_donor_top20 pred_e_selected_arm_is_crossfit_stable
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
PRIOR = ROOT / "circuits/prior_art/temporal_five_mlp_rank49_joint_single_deletion_v1.json"
ATLAS = ROOT / "circuits/followups/temporal_five_mlp_rank49_backward_deletion_atlas_v1_result.json"
BOUNDARY = ROOT / "circuits/followups/temporal_five_mlp_source_graph_greedy_boundary_v1_result.json"
HELPER = ROOT / "ops/compiled_response_graph.py"
OUT = ROOT / "circuits/followups/temporal_five_mlp_rank49_joint_single_deletion_v1_result.json"
EXPECTED = {"atlas": "a686fa1a3f10c554e25a779faa5ec56a5c5808194f5fcb924b9208d7dd071cd2",
            "boundary": "1023702de0660085b03e4d1d81dce5ff820d67f317c68bc94293cfd177d02ecb",
            "helper": "2de68cab2af7e9f1619bedc9c0692780574e015a64c3c2aaf7fbc53fd5d0f32a"}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def main():
    paths = {"atlas": ATLAS, "boundary": BOUNDARY, "helper": HELPER}; observed = {k: sha(v) for k, v in paths.items()}
    if observed != EXPECTED: raise RuntimeError(f"joint-deletion authority changed: {observed}")
    dry = {"candidate_id": "temporal_auxiliary.five_mlp_rank49_joint_single_deletion_v1", "dryrun": True,
           "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
           "model_forwards_max": 77, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1": print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    started, tic = now(), time.perf_counter(); backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch
    ctx = graph.prepare(backend); fresh = ctx["fresh"]; local = json.loads(ATLAS.read_text()); boundary = json.loads(BOUNDARY.read_text())
    support = local["support"]; arms = local["removal_order"][:10]; top20 = set(json.loads(ROOT.joinpath("circuits/followups/temporal_five_mlp_source_graph_component_deletion_atlas_v1_result.json").read_text())["top20"])
    self_output, self_ai, self_mlp, _ = source.restricted_graph(backend, fresh["batch"], fresh["donor_batch"], fresh["base_full"], support, source=False)
    self_error = max(float((atlasrun.states(torch, backend, self_output, fresh["rows"])-fresh["base_state"]).abs().max()),
                     max(float((self_ai[i]-fresh["base"][5][i]).abs().max()) for i in self_ai),
                     max(float((self_mlp[i]-fresh["base"][3][i]).abs().max()) for i in self_mlp))
    reports = {}; finite = [ctx["reader_orientation"], ctx["training"]["capture_error"], self_error]
    for removed in arms:
        allowed = [site for site in support if site != removed]
        _go, gai, gmi, source_counts = source.restricted_graph(backend, fresh["batch"], fresh["donor_batch"], fresh["base_full"], allowed, source=True)
        _co, cai, cmi, control_counts = source.restricted_graph(backend, fresh["control_batch"], fresh["control_donor_batch"], fresh["control_base_full"], allowed, source=True)
        report = {"removed": removed, "allowed_count": len(allowed), "allowed_sites": allowed,
                  "source_change_count": sorted(set(source_counts)), "control_source_change_count": sorted(set(control_counts)), "fits": {}}
        for label, qs in ctx["projectors"].items():
            generated = atlas.compiled_coordinates(backend, fresh["rows"], qs, fresh["base"][2], fresh["base"][5], gai, fresh["base"][3], gmi, ctx["rank64_bases"][label])
            cgenerated = atlas.compiled_coordinates(backend, fresh["controls"], qs, fresh["control_base"][2], fresh["control_base"][5], cai, fresh["control_base"][3], cmi, ctx["rank64_bases"][label])
            report["fits"][label] = addback.fit_arm(backend, ctx, allowed, label, qs, ctx["rank64_bases"][label], generated, cgenerated)
        report["jointly_feasible"] = addback.feasible(report); reports[removed] = report
        for fit in report["fits"].values():
            finite += list(fit["coordinate"]["signed_projection"].values()) + list(fit["coordinate"]["residual"].values())
            finite += list(fit["target"]["cells"].values()) + list(fit["target"]["behavior_signed_projection"].values())
            finite += list(fit["control"]["margin_rms_fraction"].values()) + [fit["control"]["median_kl"], fit["control"]["max_kl"], fit["control"]["top1_flip_fraction"]]
    feasible = [site for site in arms if reports[site]["jointly_feasible"]]; selected = feasible[0] if feasible else None
    stable = False
    if selected:
        fits = reports[selected]["fits"]; emin = min(fits["even_fit"]["coordinate"]["signed_projection"].values()); omin = min(fits["odd_fit"]["coordinate"]["signed_projection"].values())
        task_diff = max(abs(fits["even_fit"]["target"]["behavior_signed_projection"][task]-fits["odd_fit"]["target"]["behavior_signed_projection"][task]) for task in ("temporal", "iswas"))
        stable = abs(emin-omin) <= .05 and task_diff <= .05
    pa = ctx["hashes_ok"] and ctx["reader_ok"] and ctx["reader_orientation"] <= 1e-6 and self_error <= 1e-4 and all(math.isfinite(float(v)) for v in finite)
    predictions = {"pred_a_authority_self_clamp_finiteness_and_price": bool(pa),
                   "pred_b_at_least_one_rank48_arm_is_jointly_feasible": len(feasible) >= 1,
                   "pred_c_at_least_two_rank48_arms_are_jointly_feasible": len(feasible) >= 2,
                   "pred_d_selected_deletion_is_outside_donor_top20": selected is not None and selected not in top20,
                   "pred_e_selected_arm_is_crossfit_stable": bool(stable)}
    compact = {site: {"jointly_feasible": report["jointly_feasible"],
                      "coordinate_projection_min": min(min(f["coordinate"]["signed_projection"].values()) for f in report["fits"].values()),
                      "target_projection_min": min(min(f["target"]["behavior_signed_projection"].values()) for f in report["fits"].values()),
                      "target_worst_max": max(f["target"]["worst_target_residual"] for f in report["fits"].values()),
                      "control_flip_max": max(f["control"]["top1_flip_fraction"] for f in report["fits"].values())} for site, report in reports.items()}
    terminal = "invalid" if not pa else "rank48_source_to_reader_program" if all(predictions.values()) else "rank49_single_deletion_boundary"
    result = {"schema": "temporal_five_mlp_rank49_joint_single_deletion_result_v1", "started_utc": started, "finished_utc": now(),
              "serial_seconds": time.perf_counter()-tic, "authority_sha256": EXPECTED, "tested_deletions": arms,
              "feasible_deletions": feasible, "selected_deletion": selected, "selected_support": reports[selected]["allowed_sites"] if selected else None,
              "reports": reports, "arm_summary": compact, "predictions": predictions, "terminal": terminal,
              "price": {"model_forwards": 77, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}}
    atomic_create_json(OUT, result); print(json.dumps({k: result[k] for k in ("feasible_deletions", "selected_deletion", "arm_summary", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
