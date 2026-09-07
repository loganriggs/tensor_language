#!/usr/bin/env python3
"""Reverse OOD validation of the task-conditioned rank-four response programs."""
# BQGATE: EXPERIMENT pred_a_authority_disjointness_hashes_finiteness_and_exact_price pred_b_full_rank46_reverse_arm_replays pred_c_own_task_rank4_reverse_is_sufficient pred_d_own_task_rank4_beats_cross_task_rank4 pred_e_own_task_rank4_reverse_is_selective
from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import compiled_response_ood as oodctx
import pooled_response_projector as pooled
import run_temporal_iswas_rank46_task_typed_mode_causal_factorial_v1 as factor
import run_temporal_iswas_rank46_task_mode_complete_rank_ladder_v1 as ladder
import run_temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1 as atlasrun
import run_temporal_five_mlp_upstream_tensor_ranked_greedy_program_v1 as greedy
import run_temporal_five_mlp_crossfit_response_weight_interface_v1 as interface
import run_temporal_five_mlp_family_feasible_kl_refinement_v1 as klfit
import run_temporal_five_mlp_compiled_coordinate_upstream_atlas_v1 as atlas
import run_temporal_five_mlp_source_clamped_compiled_graph_v1 as source
import run_temporal_five_mlp_factor_pruned_causal_program_v1 as pruned
import run_temporal_five_mlp_frozen_response_ood_regularization_v2 as ood
import run_temporal_five_mlp_generic_subspace_complement_program_v1 as comp

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_rank46_task_rank4_reverse_ood_v1.json"
LADDER = ROOT / "circuits/followups/temporal_iswas_rank46_task_mode_complete_rank_ladder_v1_result.json"
LADDER_RUNNER = ROOT / "ops/run_temporal_iswas_rank46_task_mode_complete_rank_ladder_v1.py"
RANK46 = factor.RANK46; CORRECTION = factor.CORRECTION; NOISE = factor.NOISE
TCAP = factor.TCAP; ICAP = factor.ICAP
OUT = ROOT / "circuits/followups/temporal_iswas_rank46_task_rank4_reverse_ood_v1_result.json"
EXPECTED = {"ladder": "345ecd0542890c69181efe067729560375543c694f2183319ff824300028e6d8",
    "ladder_runner": "291dc68eddf827bf97c1fdcae01b43a2b4bd2df953acbbb9efe08caff8867473",
    "rank46": factor.EXPECTED["rank46"], "correction": factor.EXPECTED["correction"], "noise": factor.EXPECTED["noise"]}
ARMS = ("full", "own", "cross")


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def main():
    paths = {"ladder": LADDER, "ladder_runner": LADDER_RUNNER, "rank46": RANK46, "correction": CORRECTION, "noise": NOISE}
    observed = {key: sha(path) for key, path in paths.items()}
    if observed != EXPECTED: raise RuntimeError(f"rank4 reverse authority changed: {observed}")
    dry = {"candidate_id": "temporal_auxiliary.iswas_rank46_task_rank4_reverse_ood_v1", "dryrun": True,
        "gpu_accessed": False, "model_loaded": False, "queue_touched": False, "arms": ARMS,
        "model_forwards_exact": 27, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1": print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    started, tic = now(), time.perf_counter(); backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch
    fitted = pooled.fit(backend); qs = fitted["projector"]; task_rows, modes = ladder.fit_full_modes(backend, fitted)
    training, bases = pooled.fit_mlp_input_bases(backend, qs); fresh = oodctx.capture(backend, TCAP, ICAP)
    support = json.loads(RANK46.read_text())["selected_support"]
    _do, donor_full = atlasrun.capture_native(backend, fresh["donor_batch"]); _co, control_donor_full = atlasrun.capture_native(backend, fresh["control_donor_batch"])
    reverse_full_output, _ = atlasrun.run_patch(backend, fresh["donor_batch"], fresh["base"][1], comp.SITES)
    reverse_full_state = atlasrun.states(torch, backend, reverse_full_output, fresh["rows"])
    _go, gai, gmi, counts = source.restricted_graph(backend, fresh["donor_batch"], fresh["batch"], donor_full, support, source=True)
    _co, cai, cmi, ccounts = source.restricted_graph(backend, fresh["control_donor_batch"], fresh["control_batch"], control_donor_full, support, source=True)
    generated = atlas.compiled_coordinates(backend, fresh["rows"], qs, fresh["donor"][2], fresh["donor"][5], gai, fresh["donor"][3], gmi, bases)
    cgenerated = atlas.compiled_coordinates(backend, fresh["controls"], qs, fresh["control_donor"][2], fresh["control_donor"][5], cai, fresh["control_donor"][3], cmi, bases)
    reader, orientation, reader_ok = greedy.physical_reader(backend, json.loads(comp.WEIGHTS.read_text()))
    donor_state = atlasrun.states(torch, backend, fresh["donor"][0], fresh["rows"])
    control_donor_state = atlasrun.states(torch, backend, fresh["control_donor"][0], fresh["controls"])
    control_ctx = {"rows": fresh["controls"], "base_logits": das.head_logits(backend, control_donor_state).float()}
    rank4_modes = {site: {task: value[:, :4] for task, value in task_modes.items()} for site, task_modes in modes.items()}
    reports = {}; finite = [orientation, training["capture_error"]]
    for arm in ARMS:
        if arm == "full": values, cvalues = generated, cgenerated
        else:
            values = factor.filter_coordinates(backend, fresh, generated, qs, rank4_modes, arm)
            cvalues = factor.filter_coordinates(backend, fresh, cgenerated, qs, rank4_modes, arm, controls=True)
        output = pruned.run_factors(backend, fresh["donor_batch"], fresh["donor"][1], qs,
            {s: v for s, v in values.items() if atlasrun.site_parts(s)[0] == "attn"}, {s: v for s, v in values.items() if atlasrun.site_parts(s)[0] == "mlp"}, use_attention=True, use_mlp=True)
        coutput = pruned.run_factors(backend, fresh["control_donor_batch"], fresh["control_donor"][1], qs,
            {s: v for s, v in cvalues.items() if atlasrun.site_parts(s)[0] == "attn"}, {s: v for s, v in cvalues.items() if atlasrun.site_parts(s)[0] == "mlp"}, use_attention=True, use_mlp=True)
        coordinate = factor.coordinate_by_task(fresh, generated, values)
        target = ood.target_report(backend, fresh["rows"], donor_state, reverse_full_state, atlasrun.states(torch, backend, output, fresh["rows"]), reader, fresh["temporal_n"])
        control = klfit.control_metrics(backend, control_ctx, das.head_logits(backend, atlasrun.states(torch, backend, coutput, fresh["controls"])).float())
        reports[arm] = {"coordinate": coordinate, "target": target, "control": control}
        finite += list(target["cells"].values()) + list(target["behavior_signed_projection"].values())
        finite += list(control["margin_rms_fraction"].values()) + [control["median_kl"], control["max_kl"], control["top1_flip_fraction"]]
        finite += [x for task in coordinate.values() for x in (task["mean_signed_projection"], task["worst_residual"])]
    fit_ids = {row["row_id"] for rows in task_rows.values() for row in rows} | {row["row_id"] for row in fitted["control_rows"]}
    eval_ids = {row["row_id"] for row in fresh["rows"] + fresh["controls"]}; hashes = {site: interface.thash(q) for site, q in qs.items()}
    pa = (reader_ok and orientation <= 1e-6 and hashes == json.loads(NOISE.read_text())["pooled_projector_sha256"] and not (fit_ids & eval_ids)
          and len(support) == 46 and set(counts) == {1} and set(ccounts) == {1} and all(math.isfinite(float(v)) for v in finite))
    full, own, cross = (reports[arm] for arm in ARMS)
    pb = min(full["target"]["behavior_signed_projection"].values()) >= .8 and full["target"]["worst_target_residual"] <= .15 and full["control"]["top1_flip_fraction"] == 0.0
    pc = (min(own["target"]["behavior_signed_projection"].values()) >= .8 and own["target"]["worst_target_residual"] <= .15
          and min(v["mean_signed_projection"] for v in own["coordinate"].values()) >= .95)
    advantages = {task: own["target"]["behavior_signed_projection"][task] - cross["target"]["behavior_signed_projection"][task] for task in ("temporal", "iswas")}
    pd = min(advantages.values()) >= .15
    pe = max(own["control"]["margin_rms_fraction"].values()) <= .1 and own["control"]["median_kl"] <= .02 and own["control"]["top1_flip_fraction"] == 0.0
    predictions = {"pred_a_authority_disjointness_hashes_finiteness_and_exact_price": bool(pa),
        "pred_b_full_rank46_reverse_arm_replays": bool(pb), "pred_c_own_task_rank4_reverse_is_sufficient": bool(pc),
        "pred_d_own_task_rank4_beats_cross_task_rank4": bool(pd), "pred_e_own_task_rank4_reverse_is_selective": bool(pe)}
    terminal = "invalid" if not pa else "bidirectional_task_rank4_rank46_program" if all(predictions.values()) else "task_rank4_reverse_null"
    result = {"schema": "temporal_iswas_rank46_task_rank4_reverse_ood_result_v1", "started_utc": started, "finished_utc": now(),
        "serial_seconds": time.perf_counter() - tic, "authority_sha256": EXPECTED, "pooled_projector_sha256": hashes,
        "support_count": len(support), "own_minus_cross_behavior": advantages, "reports": reports, "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": 27, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}}
    atomic_create_json(OUT, result); print(json.dumps({"own_minus_cross_behavior": advantages, "reports": reports, "predictions": predictions, "terminal": terminal, "price": result["price"]}, sort_keys=True))


if __name__ == "__main__": main()
