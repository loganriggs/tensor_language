#!/usr/bin/env python3
"""Individual head/module deletion atlas for the unrestricted cue-source graph."""
# BQGATE: EXPERIMENT pred_a_authority_all_source_equivalence_finiteness_and_price pred_b_individual_dependencies_are_detectable pred_c_dependency_importance_is_concentrated pred_d_dependencies_extend_beyond_the_donor_top20 pred_e_weight_coordinate_edges_are_crossfit_stable
from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import compiled_response_graph as graph
import run_temporal_five_mlp_source_clamped_compiled_graph_v1 as source
import run_temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1 as atlasrun
import run_temporal_five_mlp_compiled_coordinate_upstream_atlas_v1 as atlas

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_five_mlp_source_graph_component_deletion_atlas_v1.json"
LATTICE = ROOT / "circuits/followups/temporal_five_mlp_source_clamped_band_lattice_completion_v1_result.json"
HELPER = ROOT / "ops/compiled_response_graph.py"
OUT = ROOT / "circuits/followups/temporal_five_mlp_source_graph_component_deletion_atlas_v1_result.json"
EXPECTED = {"lattice": "83893ad394691dadc98f48d336f7f0b90e779f23be66dfca7124eb8110c45ee8",
            "helper": "2de68cab2af7e9f1619bedc9c0692780574e015a64c3c2aaf7fbc53fd5d0f32a"}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: (values[i], i)); output = [0.0] * len(values)
    for rank, index in enumerate(order): output[index] = float(rank)
    return output


def correlation(xs, ys):
    mx, my = sum(xs)/len(xs), sum(ys)/len(ys)
    dx, dy = [x-mx for x in xs], [y-my for y in ys]
    return sum(x*y for x, y in zip(dx, dy)) / max(1e-30, math.sqrt(sum(x*x for x in dx)*sum(y*y for y in dy)))


def main():
    observed = {"lattice": sha(LATTICE), "helper": sha(HELPER)}
    if observed != EXPECTED: raise RuntimeError(f"component-deletion authority changed: {observed}")
    dry = {"candidate_id": "temporal_auxiliary.five_mlp_source_graph_component_deletion_atlas_v1", "dryrun": True,
           "gpu_accessed": False, "model_loaded": False, "queue_touched": False, "candidate_count": 110,
           "model_forwards_max": 128, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1": print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    started, tic = now(), time.perf_counter(); backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch
    ctx = graph.prepare(backend); fresh = ctx["fresh"]; top = json.loads(LATTICE.read_text())["complete_lattice_summary"]
    top20 = json.loads(ROOT.joinpath("circuits/followups/temporal_five_mlp_compiled_coordinate_upstream_atlas_v1_result.json").read_text())["top20"]["even_fit"]["sites"]
    all_sites = list(graph.ALL_UPSTREAM)
    all_output, all_ai, all_mlp, counts = source.restricted_graph(
        backend, fresh["batch"], fresh["donor_batch"], fresh["base_full"], all_sites, source=True)
    donor_state = atlasrun.states(torch, backend, fresh["donor"][0], fresh["rows"])
    all_state = atlasrun.states(torch, backend, all_output, fresh["rows"])
    equivalence = max(float((all_state-donor_state).abs().max()),
                      max(float((all_ai[i]-fresh["donor"][5][i]).abs().max()) for i in all_ai),
                      max(float((all_mlp[i]-fresh["donor"][3][i]).abs().max()) for i in all_mlp))
    references = {}
    for label, qs in ctx["projectors"].items():
        coords = atlas.compiled_coordinates(backend, fresh["rows"], qs, fresh["base"][2], fresh["base"][5], all_ai, fresh["base"][3], all_mlp, ctx["rank64_bases"][label])
        references[label] = atlas.vectors(backend, fresh["batch"], qs, coords)
    records = {}; finite = [ctx["reader_orientation"], ctx["training"]["capture_error"], equivalence]
    fit_scores = {label: [] for label in ctx["projectors"]}
    for removed in all_sites:
        allowed = [site for site in all_sites if site != removed]
        output, changed_ai, changed_mlp, _ = source.restricted_graph(
            backend, fresh["batch"], fresh["donor_batch"], fresh["base_full"], allowed, source=True)
        native = atlas.causal_report(backend, fresh["rows"], fresh["base_state"], donor_state,
                                     atlasrun.states(torch, backend, output, fresh["rows"]))
        record = {"native": native, "fits": {}}
        native_impairment = max(0.0, 1.0-min(native["signed_projection"].values()))
        for label, qs in ctx["projectors"].items():
            coords = atlas.compiled_coordinates(backend, fresh["rows"], qs, fresh["base"][2], fresh["base"][5], changed_ai, fresh["base"][3], changed_mlp, ctx["rank64_bases"][label])
            report = atlas.coordinate_report(torch, references[label], atlas.vectors(backend, fresh["batch"], qs, coords))
            projections = report["signed_projection"]; coordinate_impairment = max(0.0, 1.0-min(projections.values()))
            most_impaired = min(projections, key=lambda site: (projections[site], site))
            score = max(coordinate_impairment, native_impairment); fit_scores[label].append(score)
            record["fits"][label] = {"coordinate": report, "coordinate_impairment": coordinate_impairment,
                                     "native_impairment": native_impairment, "score": score,
                                     "most_impaired_coordinate": most_impaired}
            finite += list(projections.values()) + list(report["residual"].values()) + [coordinate_impairment, score]
        record["score"] = max(fit["score"] for fit in record["fits"].values()); records[removed] = record
        finite += list(native["signed_projection"].values()) + [native_impairment]
    ranking = sorted(all_sites, key=lambda site: (-records[site]["score"], site))
    total = sum(records[site]["score"] for site in ranking); running = 0.0; prefix90 = []
    for site in ranking:
        if running >= .9*total and prefix90: break
        prefix90.append(site); running += records[site]["score"]
    pool = [site for site in all_sites if site in set(top20) | set(prefix90)]
    rank_corr = correlation(ranks(fit_scores["even_fit"]), ranks(fit_scores["odd_fit"]))
    top_overlap = len(set(sorted(all_sites, key=lambda s: (-records[s]["fits"]["even_fit"]["score"], s))[:20]) &
                      set(sorted(all_sites, key=lambda s: (-records[s]["fits"]["odd_fit"]["score"], s))[:20]))
    both_detect = any(all(records[site]["fits"][label]["score"] >= .05 for label in ctx["projectors"]) for site in all_sites)
    prefix80 = 0; running = 0.0
    for site in ranking:
        if running >= .8*total: break
        running += records[site]["score"]; prefix80 += 1
    pa = ctx["hashes_ok"] and ctx["reader_ok"] and ctx["reader_orientation"] <= 1e-6 and equivalence <= 1e-4 and set(counts) == {1} and all(math.isfinite(float(v)) for v in finite)
    predictions = {"pred_a_authority_all_source_equivalence_finiteness_and_price": bool(pa),
                   "pred_b_individual_dependencies_are_detectable": bool(both_detect),
                   "pred_c_dependency_importance_is_concentrated": prefix80 <= 30,
                   "pred_d_dependencies_extend_beyond_the_donor_top20": any(site not in top20 for site in ranking[:10]),
                   "pred_e_weight_coordinate_edges_are_crossfit_stable": rank_corr >= .75 and top_overlap >= 15}
    terminal = "invalid" if not pa else "component_dependency_atlas" if all(predictions.values()) else "distributed_or_interacting_dependencies"
    result = {"schema": "temporal_five_mlp_source_graph_component_deletion_atlas_result_v1", "started_utc": started, "finished_utc": now(),
              "serial_seconds": time.perf_counter()-tic, "authority_sha256": EXPECTED, "all_source_donor_max_abs": equivalence,
              "records": records, "ranking": ranking, "top20": top20, "prefix80_count": prefix80, "prefix90": prefix90,
              "candidate_pool": pool, "rank_spearman": rank_corr, "top20_crossfit_overlap": top_overlap,
              "predictions": predictions, "terminal": terminal,
              "price": {"model_forwards": 127, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}}
    atomic_create_json(OUT, result); print(json.dumps({"top15": ranking[:15], "prefix80_count": prefix80, "prefix90_count": len(prefix90),
        "candidate_pool_count": len(pool), "rank_spearman": rank_corr, "top20_crossfit_overlap": top_overlap,
        "predictions": predictions, "terminal": terminal, "price": result["price"]}, sort_keys=True))


if __name__ == "__main__": main()
