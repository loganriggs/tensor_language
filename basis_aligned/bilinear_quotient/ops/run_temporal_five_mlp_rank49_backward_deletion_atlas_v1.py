#!/usr/bin/env python3
"""Leave-one-out redundancy atlas inside the passing rank49 source graph."""
# BQGATE: EXPERIMENT pred_a_authority_baseline_replay_finiteness_and_price pred_b_at_least_ten_sites_are_individually_removable pred_c_removability_spans_all_four_bands pred_d_early_mlp_chain_is_individually_required pred_e_removability_is_crossfit_stable
from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import compiled_response_graph as graph
import run_temporal_five_mlp_source_clamped_compiled_graph_v1 as source
import run_temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1 as atlasrun
import run_temporal_five_mlp_compiled_coordinate_upstream_atlas_v1 as atlas
import run_temporal_five_mlp_source_graph_component_deletion_atlas_v1 as deletion_tools

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_five_mlp_rank49_backward_deletion_atlas_v1.json"
BOUNDARY = ROOT / "circuits/followups/temporal_five_mlp_source_graph_greedy_boundary_v1_result.json"
HELPER = ROOT / "ops/compiled_response_graph.py"
OUT = ROOT / "circuits/followups/temporal_five_mlp_rank49_backward_deletion_atlas_v1_result.json"
EXPECTED = {"boundary": "1023702de0660085b03e4d1d81dce5ff820d67f317c68bc94293cfd177d02ecb",
            "helper": "2de68cab2af7e9f1619bedc9c0692780574e015a64c3c2aaf7fbc53fd5d0f32a"}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def main():
    observed = {"boundary": sha(BOUNDARY), "helper": sha(HELPER)}
    if observed != EXPECTED: raise RuntimeError(f"rank49 deletion authority changed: {observed}")
    dry = {"candidate_id": "temporal_auxiliary.five_mlp_rank49_backward_deletion_atlas_v1", "dryrun": True,
           "gpu_accessed": False, "model_loaded": False, "queue_touched": False, "candidate_count": 49,
           "model_forwards_max": 66, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1": print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    started, tic = now(), time.perf_counter(); backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch
    ctx = graph.prepare(backend); fresh = ctx["fresh"]; boundary = json.loads(BOUNDARY.read_text()); support = boundary["reports"]["49"]["allowed_sites"]
    baseline_output, baseline_ai, baseline_mlp, counts = source.restricted_graph(
        backend, fresh["batch"], fresh["donor_batch"], fresh["base_full"], support, source=True)
    baseline_state = atlasrun.states(torch, backend, baseline_output, fresh["rows"]); references = {}
    for label, qs in ctx["projectors"].items():
        coords = atlas.compiled_coordinates(backend, fresh["rows"], qs, fresh["base"][2], fresh["base"][5], baseline_ai, fresh["base"][3], baseline_mlp, ctx["rank64_bases"][label])
        references[label] = atlas.vectors(backend, fresh["batch"], qs, coords)
    records = {}; finite = [ctx["reader_orientation"], ctx["training"]["capture_error"]]; scores = {label: [] for label in ctx["projectors"]}
    for removed in support:
        allowed = [site for site in support if site != removed]
        output, changed_ai, changed_mlp, _ = source.restricted_graph(backend, fresh["batch"], fresh["donor_batch"], fresh["base_full"], allowed, source=True)
        native = atlas.causal_report(backend, fresh["rows"], fresh["base_state"], baseline_state, atlasrun.states(torch, backend, output, fresh["rows"]))
        native_impairment = max(0.0, 1.0-min(native["signed_projection"].values())); record = {"native": native, "fits": {}}
        for label, qs in ctx["projectors"].items():
            coords = atlas.compiled_coordinates(backend, fresh["rows"], qs, fresh["base"][2], fresh["base"][5], changed_ai, fresh["base"][3], changed_mlp, ctx["rank64_bases"][label])
            report = atlas.coordinate_report(torch, references[label], atlas.vectors(backend, fresh["batch"], qs, coords))
            impairment = max(0.0, 1.0-min(report["signed_projection"].values())); score = max(impairment, native_impairment); scores[label].append(score)
            record["fits"][label] = {"coordinate": report, "coordinate_impairment": impairment, "native_impairment": native_impairment, "score": score,
                                     "most_impaired_coordinate": min(report["signed_projection"], key=lambda s: (report["signed_projection"][s], s))}
            finite += list(report["signed_projection"].values()) + list(report["residual"].values()) + [impairment, score]
        record["score"] = max(f["score"] for f in record["fits"].values()); records[removed] = record
        finite += list(native["signed_projection"].values()) + [native_impairment]
    removal_order = sorted(support, key=lambda site: (records[site]["score"], site)); required_order = list(reversed(removal_order))
    removable = [site for site in support if all(min(records[site]["fits"][label]["coordinate"]["signed_projection"].values()) >= .99 for label in ctx["projectors"]) and min(records[site]["native"]["signed_projection"].values()) >= .99]
    corr = deletion_tools.correlation(deletion_tools.ranks(scores["even_fit"]), deletion_tools.ranks(scores["odd_fit"]))
    even_required = sorted(support, key=lambda s: (-records[s]["fits"]["even_fit"]["score"], s))[:20]
    odd_required = sorted(support, key=lambda s: (-records[s]["fits"]["odd_fit"]["score"], s))[:20]
    overlap = len(set(even_required) & set(odd_required)); bands = {name for name, layers in graph.BANDS.items() if any(graph.site_layer(site) in layers for site in removable)}
    early_required = all(all(records[site]["fits"][label]["score"] >= .05 for label in ctx["projectors"]) for site in ("MLP0", "MLP1", "MLP2", "MLP3", "MLP4"))
    pa = ctx["hashes_ok"] and ctx["reader_ok"] and ctx["reader_orientation"] <= 1e-6 and len(support) == 49 and set(counts) == {1} and all(math.isfinite(float(v)) for v in finite)
    predictions = {"pred_a_authority_baseline_replay_finiteness_and_price": bool(pa),
                   "pred_b_at_least_ten_sites_are_individually_removable": len(removable) >= 10,
                   "pred_c_removability_spans_all_four_bands": bands == set(graph.BANDS),
                   "pred_d_early_mlp_chain_is_individually_required": bool(early_required),
                   "pred_e_removability_is_crossfit_stable": corr >= .9 and overlap >= 18}
    terminal = "invalid" if not pa else "rank49_backward_deletion_order" if all(predictions.values()) else "rank49_interacting_support"
    result = {"schema": "temporal_five_mlp_rank49_backward_deletion_atlas_result_v1", "started_utc": started, "finished_utc": now(),
              "serial_seconds": time.perf_counter()-tic, "authority_sha256": EXPECTED, "support": support, "records": records,
              "removal_order": removal_order, "required_order": required_order, "individually_removable": removable,
              "removable_bands": sorted(bands), "rank_spearman": corr, "top20_required_overlap": overlap,
              "predictions": predictions, "terminal": terminal,
              "price": {"model_forwards": 66, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}}
    atomic_create_json(OUT, result); print(json.dumps({"removable_count": len(removable), "first15_removals": removal_order[:15],
        "last15_required": required_order[:15], "removable_bands": sorted(bands), "rank_spearman": corr,
        "top20_required_overlap": overlap, "predictions": predictions, "terminal": terminal, "price": result["price"]}, sort_keys=True))


if __name__ == "__main__": main()
