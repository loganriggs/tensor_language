#!/usr/bin/env python3
"""Prospective lexical/construction transfer of the frozen rank48 source graph."""
# BQGATE: EXPERIMENT pred_a_authority_population_finiteness_and_price pred_b_ood_coordinates_transfer pred_c_ood_behavior_transfer pred_d_ood_controls_are_selective pred_e_ood_crossfit_is_stable
from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

import circuit_candidate_temporal_auxiliary_fresh_cues_v12 as ood_t
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v11 as ood_i
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import compiled_response_graph as graph
import run_temporal_iswas_three_mlp_response_program_fresh_v12_v1 as population
import run_temporal_five_mlp_source_clamped_compiled_graph_v1 as source
import run_temporal_five_mlp_source_clamped_layer_band_addback_v1 as addback
import run_temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1 as atlasrun
import run_temporal_five_mlp_compiled_coordinate_upstream_atlas_v1 as atlas
import run_temporal_five_mlp_attention_value_weight_pullback_v1 as pullback
import run_temporal_five_mlp_generic_subspace_complement_program_v1 as comp

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_five_mlp_rank48_lexical_ood_v1.json"
SUPPORT = ROOT / "circuits/followups/temporal_five_mlp_rank49_joint_single_deletion_v1_result.json"
TCAP = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v12_capability_v1_result.json"
ICAP = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v11_capability_v1_result.json"
TB = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v12.py"
IB = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v11.py"
HELPER = ROOT / "ops/compiled_response_graph.py"
OUT = ROOT / "circuits/followups/temporal_five_mlp_rank48_lexical_ood_v1_result.json"
EXPECTED = {"support": "92384a4496c7e3ceb5dbccecfd8ef65a61e7f5c219e52b5cf919b91ef1ba5fe2",
            "temporal_capability": "4758b02cd026c85289dc3eaf352cc496d238057c6f8b52dfc6fe49ae17893324",
            "iswas_capability": "6dd757b066304d1f81ea1e52e0db601fea05adeac516a49cc84ab42bc73a86a2",
            "temporal_builder": "4cf4624361ff2bd3c87cd987a7c4d16a1eeefb1c7f78287b78319862ab8d8de9",
            "iswas_builder": "fbd47713fafcb87fc30ba339d175f7d06770ce36b93b6035e8848455529344ec",
            "helper": "2de68cab2af7e9f1619bedc9c0692780574e015a64c3c2aaf7fbc53fd5d0f32a"}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def main():
    paths = {"support": SUPPORT, "temporal_capability": TCAP, "iswas_capability": ICAP,
             "temporal_builder": TB, "iswas_builder": IB, "helper": HELPER}; observed = {k: sha(v) for k, v in paths.items()}
    if observed != EXPECTED: raise RuntimeError(f"rank48 OOD authority changed: {observed}")
    dry = {"candidate_id": "temporal_auxiliary.five_mlp_rank48_lexical_ood_v1", "dryrun": True,
           "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
           "model_forwards_max": 22, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1": print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    started, tic = now(), time.perf_counter(); backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch
    replay = graph.reconstruct_projectors(backend); training = graph.training_context(backend)
    replay["training"] = training; replay["rank64_bases"] = graph.rank64_bases(backend, replay["projectors"], training)
    tcap, icap = json.loads(TCAP.read_text()), json.loads(ICAP.read_text())
    temporal = sum((population.capable_rows(ood_t, tcap, panel, 12) for panel in ("A1", "A2")), [])
    iswas = sum((population.capable_rows(ood_i, icap, panel, 12) for panel in ("A1", "A2")), [])
    rows = temporal + iswas; controls = [row for row in ood_t.build_rows() if row["transform_id"] == "P"][:16]
    batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
    cbatch, cdonor_batch = das._batch(backend, controls, side="base"), das._batch(backend, controls, side="donor")
    base = pullback.capture_with_attention_inputs(backend, batch); donor = pullback.capture_with_attention_inputs(backend, donor_batch)
    cbase = pullback.capture_with_attention_inputs(backend, cbatch); cdonor = pullback.capture_with_attention_inputs(backend, cdonor_batch)
    _bo, base_full = atlasrun.capture_native(backend, batch); _co, control_base_full = atlasrun.capture_native(backend, cbatch)
    full_output, _ = atlasrun.run_patch(backend, batch, donor[1], comp.SITES)
    support_receipt = json.loads(SUPPORT.read_text()); support = support_receipt["selected_support"]
    _go, gai, gmi, counts = source.restricted_graph(backend, batch, donor_batch, base_full, support, source=True)
    _cgo, cgai, cgmi, ccounts = source.restricted_graph(backend, cbatch, cdonor_batch, control_base_full, support, source=True)
    fresh = {"rows": rows, "temporal_n": len(temporal), "controls": controls, "batch": batch, "donor_batch": donor_batch,
             "control_batch": cbatch, "control_donor_batch": cdonor_batch, "base": base, "donor": donor,
             "control_base": cbase, "control_donor": cdonor, "base_full": base_full, "control_base_full": control_base_full,
             "base_state": atlasrun.states(torch, backend, base[0], rows), "full_state": atlasrun.states(torch, backend, full_output, rows),
             "control_base_state": atlasrun.states(torch, backend, cbase[0], controls)}
    ctx = {**replay, "fresh": fresh}; reports = {}; coordinates = {}; finite = [replay["reader_orientation"], training["capture_error"]]
    for label, qs in replay["projectors"].items():
        generated = atlas.compiled_coordinates(backend, rows, qs, base[2], base[5], gai, base[3], gmi, replay["rank64_bases"][label])
        cgenerated = atlas.compiled_coordinates(backend, controls, qs, cbase[2], cbase[5], cgai, cbase[3], cgmi, replay["rank64_bases"][label])
        report = addback.fit_arm(backend, ctx, support, label, qs, replay["rank64_bases"][label], generated, cgenerated)
        reports[label] = report; coordinates[label] = report["coordinate"]
        finite += list(report["coordinate"]["signed_projection"].values()) + list(report["coordinate"]["residual"].values())
        finite += list(report["target"]["cells"].values()) + list(report["target"]["behavior_signed_projection"].values())
        finite += list(report["control"]["margin_rms_fraction"].values()) + [report["control"]["median_kl"], report["control"]["max_kl"], report["control"]["top1_flip_fraction"]]
    pa = replay["hashes_ok"] and replay["reader_ok"] and replay["reader_orientation"] <= 1e-6 and len(support) == 48 and set(counts) == {1} and set(ccounts) == {1} and all(math.isfinite(float(v)) for v in finite)
    pb = all(min(r["coordinate"]["signed_projection"].values()) >= .75 and r["coordinate"]["mean_residual"] <= .2 for r in reports.values())
    pc = all(min(r["target"]["behavior_signed_projection"].values()) >= .8 and r["target"]["worst_target_residual"] <= .15 for r in reports.values())
    pd = all(max(r["control"]["margin_rms_fraction"].values()) <= .1 and r["control"]["median_kl"] <= .02 and r["control"]["top1_flip_fraction"] <= .05 for r in reports.values())
    em, om = min(coordinates["even_fit"]["signed_projection"].values()), min(coordinates["odd_fit"]["signed_projection"].values())
    task_diff = max(abs(reports["even_fit"]["target"]["behavior_signed_projection"][task]-reports["odd_fit"]["target"]["behavior_signed_projection"][task]) for task in ("temporal", "iswas"))
    pe = abs(em-om) <= .05 and task_diff <= .05
    predictions = {"pred_a_authority_population_finiteness_and_price": bool(pa), "pred_b_ood_coordinates_transfer": bool(pb),
                   "pred_c_ood_behavior_transfer": bool(pc), "pred_d_ood_controls_are_selective": bool(pd), "pred_e_ood_crossfit_is_stable": bool(pe)}
    summary = {"coordinate_projection_min": min(em, om), "coordinate_mean_residual_max": max(r["coordinate"]["mean_residual"] for r in reports.values()),
               "target_projection_min": min(min(r["target"]["behavior_signed_projection"].values()) for r in reports.values()),
               "target_worst_max": max(r["target"]["worst_target_residual"] for r in reports.values()),
               "control_margin_max": max(max(r["control"]["margin_rms_fraction"].values()) for r in reports.values()),
               "control_median_kl_max": max(r["control"]["median_kl"] for r in reports.values()),
               "control_flip_max": max(r["control"]["top1_flip_fraction"] for r in reports.values())}
    terminal = "invalid" if not pa else "ood_rank48_source_to_reader_program" if all(predictions.values()) else "rank48_ood_transfer_failure"
    result = {"schema": "temporal_five_mlp_rank48_lexical_ood_result_v1", "started_utc": started, "finished_utc": now(),
              "serial_seconds": time.perf_counter()-tic, "authority_sha256": EXPECTED, "support": support, "reports": reports,
              "summary": summary, "predictions": predictions, "terminal": terminal,
              "price": {"model_forwards": 22, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}}
    atomic_create_json(OUT, result); print(json.dumps({k: result[k] for k in ("summary", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
