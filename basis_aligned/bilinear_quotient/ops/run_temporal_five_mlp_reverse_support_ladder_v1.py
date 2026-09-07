#!/usr/bin/env python3
"""Frozen support-size ladder for the reverse OOD compiled response graph."""
# BQGATE: EXPERIMENT pred_a_authority_rank48_replay_finiteness_and_price pred_b_rank49_repairs_reverse_necessity pred_c_a_pruned_reverse_support_exists pred_d_reverse_behavior_is_monotone_through_rank55 pred_e_selected_support_is_crossfit_stable
from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import compiled_response_graph as graph
import compiled_response_ood as oodctx
import compiled_response_reverse as reverse

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_five_mlp_reverse_support_ladder_v1.json"
R48 = ROOT / "circuits/followups/temporal_five_mlp_rank49_joint_single_deletion_v1_result.json"
R49 = ROOT / "circuits/followups/temporal_five_mlp_source_graph_greedy_boundary_v1_result.json"
R55 = ROOT / "circuits/followups/temporal_five_mlp_source_graph_greedy_prefix_v1_result.json"
OLD = ROOT / "circuits/followups/temporal_five_mlp_rank48_reverse_ood_v1_result.json"
TCAP = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v12_capability_v1_result.json"
ICAP = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v11_capability_v1_result.json"
OOD_HELPER = ROOT / "ops/compiled_response_ood.py"; REVERSE_HELPER = ROOT / "ops/compiled_response_reverse.py"
OUT = ROOT / "circuits/followups/temporal_five_mlp_reverse_support_ladder_v1_result.json"
EXPECTED = {"rank48": "92384a4496c7e3ceb5dbccecfd8ef65a61e7f5c219e52b5cf919b91ef1ba5fe2",
            "rank49": "1023702de0660085b03e4d1d81dce5ff820d67f317c68bc94293cfd177d02ecb",
            "rank55": "5bc58ebb5b1f79bb4b9d4b24a5a341b95ad1ab21402ed76f679b2be5a0cf9d56",
            "old_reverse": "68e7fab4d108d365771105441ff7ac4d67a5533122dc9592f1260d984870824b",
            "ood_helper": "96b9e3aef3b64364c7fc4c49d0af9ff35827eadb4ca5bfbdf5d2271cb26ad808",
            "reverse_helper": "6ca850c14f94d16d249ee2d46f6a41e52204871425f8210e80f8de30410bec15"}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def main():
    paths = {"rank48": R48, "rank49": R49, "rank55": R55, "old_reverse": OLD,
             "ood_helper": OOD_HELPER, "reverse_helper": REVERSE_HELPER}; observed = {k: sha(v) for k, v in paths.items()}
    if observed != EXPECTED: raise RuntimeError(f"reverse ladder authority changed: {observed}")
    dry = {"candidate_id": "temporal_auxiliary.five_mlp_reverse_support_ladder_v1", "dryrun": True,
           "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
           "model_forwards_max": 43, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1": print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    started, tic = now(), time.perf_counter(); backend = producer.Bilin18TorchBackend.load("cuda")
    ctx = oodctx.prepare(backend, TCAP, ICAP); reverse_ctx = reverse.prepare(backend, ctx)
    supports = {"rank48": json.loads(R48.read_text())["selected_support"],
                "rank49": json.loads(R49.read_text())["reports"]["49"]["allowed_sites"],
                "rank55": json.loads(R55.read_text())["reports"]["rank55"]["allowed_sites"],
                "all110": list(graph.ALL_UPSTREAM)}
    reports = {}; summaries = {}; finite = [ctx["reader_orientation"], ctx["training"]["capture_error"]]
    for name, support in supports.items():
        fit_reports, counts, ccounts = reverse.evaluate(backend, ctx, reverse_ctx, support)
        reports[name] = {"allowed_count": len(support), "allowed_sites": support, "fits": fit_reports,
                         "source_change_count": sorted(set(counts)), "control_source_change_count": sorted(set(ccounts)),
                         "jointly_feasible": reverse.feasible(fit_reports)}
        summaries[name] = {"allowed_count": len(support), "jointly_feasible": reports[name]["jointly_feasible"], **reverse.summary(fit_reports)}
        for fit in fit_reports.values():
            finite += list(fit["coordinate"]["signed_projection"].values()) + list(fit["coordinate"]["residual"].values())
            finite += list(fit["target"]["cells"].values()) + list(fit["target"]["behavior_signed_projection"].values())
            finite += list(fit["control"]["margin_rms_fraction"].values()) + [fit["control"]["median_kl"], fit["control"]["max_kl"], fit["control"]["top1_flip_fraction"]]
    old = json.loads(OLD.read_text())["summary"]
    replay_error = max(abs(summaries["rank48"][key]-old[key]) for key in old)
    order = ("rank48", "rank49", "rank55", "all110"); feasible = [name for name in order if reports[name]["jointly_feasible"]]
    selected = feasible[0] if feasible else None
    stable = False
    if selected:
        fits = reports[selected]["fits"]; em = min(fits["even_fit"]["coordinate"]["signed_projection"].values()); om = min(fits["odd_fit"]["coordinate"]["signed_projection"].values())
        task_diff = max(abs(fits["even_fit"]["target"]["behavior_signed_projection"][task]-fits["odd_fit"]["target"]["behavior_signed_projection"][task]) for task in ("temporal", "iswas"))
        stable = abs(em-om) <= .05 and task_diff <= .05
    behavior = [summaries[name]["target_projection_min"] for name in ("rank48", "rank49", "rank55")]
    pa = ctx["hashes_ok"] and ctx["reader_ok"] and ctx["reader_orientation"] <= 1e-6 and replay_error <= 1e-6 and all(math.isfinite(float(v)) for v in finite)
    predictions = {"pred_a_authority_rank48_replay_finiteness_and_price": bool(pa),
                   "pred_b_rank49_repairs_reverse_necessity": reports["rank49"]["jointly_feasible"],
                   "pred_c_a_pruned_reverse_support_exists": selected is not None and reports[selected]["allowed_count"] <= 55,
                   "pred_d_reverse_behavior_is_monotone_through_rank55": behavior[0] <= behavior[1] <= behavior[2],
                   "pred_e_selected_support_is_crossfit_stable": bool(stable)}
    terminal = "invalid" if not pa else "pruned_bidirectional_ood_program" if all(predictions.values()) else "reverse_support_or_interface_null"
    result = {"schema": "temporal_five_mlp_reverse_support_ladder_result_v1", "started_utc": started, "finished_utc": now(),
              "serial_seconds": time.perf_counter()-tic, "authority_sha256": EXPECTED, "rank48_replay_max_abs": replay_error,
              "selected_support": selected, "reports": reports, "summary": summaries, "predictions": predictions, "terminal": terminal,
              "price": {"model_forwards": 43, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}}
    atomic_create_json(OUT, result); print(json.dumps({k: result[k] for k in ("selected_support", "summary", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
