#!/usr/bin/env python3
"""Complete-family pooled response projector on fresh rank48 reverse execution."""
# BQGATE: EXPERIMENT pred_a_authority_disjointness_finiteness_and_exact_price pred_b_pooled_reverse_coordinates_transfer pred_c_pooled_reverse_behavior_passes pred_d_pooled_reverse_controls_are_selective pred_e_pooling_resolves_the_split_failure
from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import compiled_response_graph as graph
import run_temporal_iswas_three_mlp_response_program_fresh_v12_v1 as population
import run_temporal_five_mlp_target_contrast_response_basis_v1 as v1
import run_temporal_five_mlp_generic_subspace_complement_program_v1 as comp
import run_temporal_five_mlp_matched_control_upstream_atlas_v2 as matched
import run_temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1 as atlasrun
import run_temporal_five_mlp_upstream_tensor_ranked_greedy_program_v1 as greedy
import run_temporal_five_mlp_crossfit_response_weight_interface_v1 as interface
import run_temporal_five_mlp_family_feasible_kl_refinement_v1 as klfit
import run_temporal_five_mlp_mlp_input_covector_crossfit_v1 as covector
import run_temporal_five_mlp_compiled_coordinate_upstream_atlas_v1 as atlas
import run_temporal_five_mlp_source_clamped_compiled_graph_v1 as source
import run_temporal_five_mlp_factor_pruned_causal_program_v1 as pruned
import run_temporal_five_mlp_frozen_response_ood_regularization_v2 as ood

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_five_mlp_rank48_pooled_projector_reverse_fresh_v1.json"
CURVE = ROOT / "circuits/followups/temporal_five_mlp_rank48_reverse_fresh_gain_curve_v1_result.json"
SUPPORT = ROOT / "circuits/followups/temporal_five_mlp_rank49_joint_single_deletion_v1_result.json"
WEIGHT = ROOT / "circuits/followups/temporal_five_mlp_crossfit_response_weight_interface_v1_result.json"
GRAPH = ROOT / "ops/compiled_response_graph.py"
OUT = ROOT / "circuits/followups/temporal_five_mlp_rank48_pooled_projector_reverse_fresh_v1_result.json"
EXPECTED = {"curve": "158d0b73d6acc2122802caa5965ece1085ba1de57c46cfd7809addd32a47bd7f",
            "support": "92384a4496c7e3ceb5dbccecfd8ef65a61e7f5c219e52b5cf919b91ef1ba5fe2",
            "weight": "578fa690f2f3962409bb50bd40a1c4c6f0b5ce7f88828e2e889a3a47bb20f013",
            "graph": "2de68cab2af7e9f1619bedc9c0692780574e015a64c3c2aaf7fbc53fd5d0f32a"}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def main():
    observed = {key: sha(path) for key, path in {"curve": CURVE, "support": SUPPORT, "weight": WEIGHT, "graph": GRAPH}.items()}
    if observed != EXPECTED: raise RuntimeError(f"pooled-projector authority changed: {observed}")
    dry = {"candidate_id": "temporal_auxiliary.five_mlp_rank48_pooled_projector_reverse_fresh_v1",
           "dryrun": True, "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
           "model_forwards_exact": 19, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    started, tic = now(), time.perf_counter(); backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch

    old_tcap, old_icap = json.loads(comp.TCAP.read_text()), json.loads(comp.ICAP.read_text())
    temporal, iswas, *_ = greedy.rows_and_controls(old_tcap, old_icap)
    target_rows = temporal + iswas
    controls = matched.control_rows(); control_rows = controls["discovery"] + controls["validation"]
    cbatch, _, cbase, _, cdonor, _, _ = interface.cap_inputs(backend, control_rows)
    tbatch, _, tbase, _, tdonor, _, _ = interface.cap_inputs(backend, target_rows)
    control_basis, _ = comp.fit_bases(backend, cbatch, cbase, cdonor)
    pooled_q, _ = v1.fit_target(backend, tbatch, tbase, tdonor, control_basis)

    training = graph.training_context(backend); ids = list(range(len(training["rows"]))); mlp_bases = {}
    for site, q in pooled_q.items():
        if atlasrun.site_parts(site)[0] != "mlp": continue
        amap = covector.effective_map(backend, site, q, training["mlp_inputs"])
        mlp_bases[site], _energy = covector.fit_input_basis(torch, training["batch"], amap, ids, 64)

    fresh = graph.fresh_context(backend); support = json.loads(SUPPORT.read_text())["selected_support"]
    _do, donor_full = atlasrun.capture_native(backend, fresh["donor_batch"])
    _co, control_donor_full = atlasrun.capture_native(backend, fresh["control_donor_batch"])
    reverse_full_output, _ = atlasrun.run_patch(backend, fresh["donor_batch"], fresh["base"][1], comp.SITES)
    _go, gai, gmi, counts = source.restricted_graph(backend, fresh["donor_batch"], fresh["batch"], donor_full, support, source=True)
    _gx, cai, cmi, ccounts = source.restricted_graph(backend, fresh["control_donor_batch"], fresh["control_batch"], control_donor_full, support, source=True)
    donor_state = atlasrun.states(torch, backend, fresh["donor"][0], fresh["rows"])
    reverse_full_state = atlasrun.states(torch, backend, reverse_full_output, fresh["rows"])
    control_donor_state = atlasrun.states(torch, backend, fresh["control_donor"][0], fresh["controls"])
    control_ctx = {"rows": fresh["controls"], "base_logits": das.head_logits(backend, control_donor_state).float()}

    reference = atlas.compiled_coordinates(backend, fresh["rows"], pooled_q, fresh["donor"][2], fresh["donor"][5], fresh["base"][5],
                                           fresh["donor"][3], fresh["base"][3], mlp_bases)
    generated = atlas.compiled_coordinates(backend, fresh["rows"], pooled_q, fresh["donor"][2], fresh["donor"][5], gai,
                                           fresh["donor"][3], gmi, mlp_bases)
    coordinate = atlas.coordinate_report(torch, atlas.vectors(backend, fresh["donor_batch"], pooled_q, reference),
                                         atlas.vectors(backend, fresh["donor_batch"], pooled_q, generated))
    reader, reader_orientation, reader_ok = greedy.physical_reader(backend, json.loads(comp.WEIGHTS.read_text()))
    target_output = pruned.run_factors(backend, fresh["donor_batch"], fresh["donor"][1], pooled_q,
        {site: value for site, value in generated.items() if atlasrun.site_parts(site)[0] == "attn"},
        {site: value for site, value in generated.items() if atlasrun.site_parts(site)[0] == "mlp"}, use_attention=True, use_mlp=True)
    target = ood.target_report(backend, fresh["rows"], donor_state, reverse_full_state,
                               atlasrun.states(torch, backend, target_output, fresh["rows"]),
                               reader, fresh["temporal_n"])
    cgenerated = atlas.compiled_coordinates(backend, fresh["controls"], pooled_q, fresh["control_donor"][2], fresh["control_donor"][5], cai,
                                            fresh["control_donor"][3], cmi, mlp_bases)
    control_output = pruned.run_factors(backend, fresh["control_donor_batch"], fresh["control_donor"][1], pooled_q,
        {site: value for site, value in cgenerated.items() if atlasrun.site_parts(site)[0] == "attn"},
        {site: value for site, value in cgenerated.items() if atlasrun.site_parts(site)[0] == "mlp"}, use_attention=True, use_mlp=True)
    control = klfit.control_metrics(backend, control_ctx, das.head_logits(backend, atlasrun.states(torch, backend, control_output, fresh["controls"])).float())

    fit_ids = {row["row_id"] for row in target_rows + control_rows}; eval_ids = {row["row_id"] for row in fresh["rows"] + fresh["controls"]}
    finite = list(coordinate["signed_projection"].values()) + list(coordinate["residual"].values())
    finite += list(target["cells"].values()) + list(target["behavior_signed_projection"].values())
    finite += list(control["margin_rms_fraction"].values()) + [control["median_kl"], control["max_kl"], control["top1_flip_fraction"]]
    pa = (not (fit_ids & eval_ids) and reader_ok and reader_orientation <= 1e-6
          and len(support) == 48 and set(counts) == {1} and set(ccounts) == {1}
          and all(math.isfinite(float(value)) for value in finite))
    pb = min(coordinate["signed_projection"].values()) >= .75 and coordinate["mean_residual"] <= .2 and coordinate["worst_residual"] <= .2
    pc = min(target["behavior_signed_projection"].values()) >= .8 and target["worst_target_residual"] <= .15
    pd = max(control["margin_rms_fraction"].values()) <= .1 and control["median_kl"] <= .02 and control["top1_flip_fraction"] == 0.0
    pe = min(target["behavior_signed_projection"].values()) > .7962746 and control["top1_flip_fraction"] == 0.0
    predictions = {"pred_a_authority_disjointness_finiteness_and_exact_price": bool(pa),
                   "pred_b_pooled_reverse_coordinates_transfer": bool(pb),
                   "pred_c_pooled_reverse_behavior_passes": bool(pc),
                   "pred_d_pooled_reverse_controls_are_selective": bool(pd),
                   "pred_e_pooling_resolves_the_split_failure": bool(pe)}
    terminal = "invalid" if not pa else "pooled_reverse_rank48_program" if all(predictions.values()) else "pooled_projector_failure"
    result = {"schema": "temporal_five_mlp_rank48_pooled_projector_reverse_fresh_result_v1",
              "started_utc": started, "finished_utc": now(), "serial_seconds": time.perf_counter()-tic,
              "authority_sha256": EXPECTED, "fit_row_count": len(target_rows), "control_fit_row_count": len(control_rows),
              "fresh_row_count": len(fresh["rows"]), "fresh_control_count": len(fresh["controls"]),
              "pooled_projector_sha256": {site: interface.thash(q) for site, q in pooled_q.items()},
              "coordinate": coordinate, "target": target, "control": control,
              "predictions": predictions, "terminal": terminal,
              "price": {"model_forwards": 19, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}}
    atomic_create_json(OUT, result)
    print(json.dumps({"coordinate": coordinate, "target": target, "control": control,
                      "predictions": predictions, "terminal": terminal, "price": result["price"]}, sort_keys=True))


if __name__ == "__main__": main()
