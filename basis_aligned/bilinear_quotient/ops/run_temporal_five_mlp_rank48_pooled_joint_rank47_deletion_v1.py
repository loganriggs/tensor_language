#!/usr/bin/env python3
"""Joint bidirectional deletion test from the stable pooled rank48 interface."""
# BQGATE: EXPERIMENT pred_a_authority_hash_disjointness_finiteness_and_exact_price pred_b_at_least_one_rank47_arm_has_bidirectional_coordinate_fidelity pred_c_at_least_one_rank47_arm_has_bidirectional_behavior_fidelity pred_d_at_least_one_rank47_arm_is_bidirectionally_selective pred_e_selected_rank47_arm_is_direction_stable
from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import compiled_response_ood as oodctx
import pooled_response_projector as pooled
import run_temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1 as atlasrun
import run_temporal_five_mlp_upstream_tensor_ranked_greedy_program_v1 as greedy
import run_temporal_five_mlp_crossfit_response_weight_interface_v1 as interface
import run_temporal_five_mlp_family_feasible_kl_refinement_v1 as klfit
import run_temporal_five_mlp_compiled_coordinate_upstream_atlas_v1 as atlas
import run_temporal_five_mlp_source_clamped_compiled_graph_v1 as source
import run_temporal_five_mlp_source_clamped_layer_band_addback_v1 as addback
import run_temporal_five_mlp_factor_pruned_causal_program_v1 as pruned
import run_temporal_five_mlp_frozen_response_ood_regularization_v2 as ood
import run_temporal_five_mlp_generic_subspace_complement_program_v1 as comp

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_five_mlp_rank48_pooled_joint_rank47_deletion_v1.json"
RANK48 = ROOT / "circuits/followups/temporal_five_mlp_rank49_joint_single_deletion_v1_result.json"
FORWARD = ROOT / "circuits/followups/temporal_five_mlp_rank48_pooled_projector_forward_ood_v1_result.json"
REVERSE = ROOT / "circuits/followups/temporal_five_mlp_rank48_pooled_projector_reverse_ood_v2_result.json"
NOISE = ROOT / "circuits/followups/temporal_five_mlp_rank48_pooled_bidirectional_source_noise_ood_v1_result.json"
TCAP = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v12_capability_v1_result.json"
ICAP = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v11_capability_v1_result.json"
HELPER = ROOT / "ops/pooled_response_projector.py"
OUT = ROOT / "circuits/followups/temporal_five_mlp_rank48_pooled_joint_rank47_deletion_v1_result.json"
EXPECTED = {"rank48": "92384a4496c7e3ceb5dbccecfd8ef65a61e7f5c219e52b5cf919b91ef1ba5fe2",
    "forward": "c9fb774866c09391ddde6fa68842e3770b7c1a944744cc454191e0a23a141d40",
    "reverse": "2791135e5abf88ceb5b76a3e0da44c1bb5625a5031ae58b0aba351f1232fb9b2",
    "noise": "a811d10364deeaa8b0e8bb9c5a747413e0df6473999362ddf24d101dc206e46c",
    "helper": "09a7ea5504722a1765367f8422cbbb858373bcc5e910d5aae75ccba46cb124ed"}
REMOVALS = ("L1H3", "L3H7", "L10H5")
CANDIDATE = "temporal_auxiliary.five_mlp_rank48_pooled_joint_rank47_deletion_v1"
RESULT_SCHEMA = "temporal_five_mlp_rank48_pooled_joint_rank47_deletion_result_v1"
SUCCESS_TERMINAL = "pooled_bidirectional_rank47_program"
FAILURE_TERMINAL = "rank48_single_deletion_boundary"


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def reverse_arm(backend, fresh, support, qs, bases, reader, donor_full, control_donor_full, reverse_full_state):
    torch = backend.torch
    _go, gai, gmi, counts = source.restricted_graph(backend, fresh["donor_batch"], fresh["batch"], donor_full, support, source=True)
    _co, cai, cmi, ccounts = source.restricted_graph(backend, fresh["control_donor_batch"], fresh["control_batch"], control_donor_full, support, source=True)
    reference = atlas.compiled_coordinates(backend, fresh["rows"], qs, fresh["donor"][2], fresh["donor"][5], fresh["base"][5], fresh["donor"][3], fresh["base"][3], bases)
    generated = atlas.compiled_coordinates(backend, fresh["rows"], qs, fresh["donor"][2], fresh["donor"][5], gai, fresh["donor"][3], gmi, bases)
    coordinate = atlas.coordinate_report(torch, atlas.vectors(backend, fresh["donor_batch"], qs, reference), atlas.vectors(backend, fresh["donor_batch"], qs, generated))
    output = pruned.run_factors(backend, fresh["donor_batch"], fresh["donor"][1], qs,
        {s: v for s, v in generated.items() if atlasrun.site_parts(s)[0] == "attn"},
        {s: v for s, v in generated.items() if atlasrun.site_parts(s)[0] == "mlp"}, use_attention=True, use_mlp=True)
    target = ood.target_report(backend, fresh["rows"], atlasrun.states(torch, backend, fresh["donor"][0], fresh["rows"]), reverse_full_state,
        atlasrun.states(torch, backend, output, fresh["rows"]), reader, fresh["temporal_n"])
    cgenerated = atlas.compiled_coordinates(backend, fresh["controls"], qs, fresh["control_donor"][2], fresh["control_donor"][5], cai, fresh["control_donor"][3], cmi, bases)
    coutput = pruned.run_factors(backend, fresh["control_donor_batch"], fresh["control_donor"][1], qs,
        {s: v for s, v in cgenerated.items() if atlasrun.site_parts(s)[0] == "attn"},
        {s: v for s, v in cgenerated.items() if atlasrun.site_parts(s)[0] == "mlp"}, use_attention=True, use_mlp=True)
    control_state = atlasrun.states(torch, backend, fresh["control_donor"][0], fresh["controls"])
    control = klfit.control_metrics(backend, {"rows": fresh["controls"], "base_logits": das.head_logits(backend, control_state).float()},
        das.head_logits(backend, atlasrun.states(torch, backend, coutput, fresh["controls"])).float())
    return {"coordinate": coordinate, "target": target, "control": control, "source_counts": sorted(set(counts)), "control_source_counts": sorted(set(ccounts))}


def main():
    observed = {key: sha(path) for key, path in {"rank48": RANK48, "forward": FORWARD, "reverse": REVERSE, "noise": NOISE, "helper": HELPER}.items()}
    if observed != EXPECTED: raise RuntimeError(f"rank47 authority changed: {observed}")
    exact_forwards = 15 + 8 * len(REMOVALS)
    dry = {"candidate_id": CANDIDATE, "dryrun": True,
        "gpu_accessed": False, "model_loaded": False, "queue_touched": False, "candidate_count": len(REMOVALS),
        "model_forwards_exact": exact_forwards, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1": print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    started, tic = now(), time.perf_counter(); backend = producer.Bilin18TorchBackend.load("cuda")
    fitted = pooled.fit(backend); qs = fitted["projector"]; training, bases = pooled.fit_mlp_input_bases(backend, qs)
    fresh = oodctx.capture(backend, TCAP, ICAP); rank48 = json.loads(RANK48.read_text()); base_support = rank48["selected_support"]
    reader, orientation, reader_ok = greedy.physical_reader(backend, json.loads(comp.WEIGHTS.read_text()))
    _do, donor_full = atlasrun.capture_native(backend, fresh["donor_batch"]); _co, control_donor_full = atlasrun.capture_native(backend, fresh["control_donor_batch"])
    reverse_full_output, _ = atlasrun.run_patch(backend, fresh["donor_batch"], fresh["base"][1], comp.SITES)
    reverse_full_state = atlasrun.states(backend.torch, backend, reverse_full_output, fresh["rows"])
    hashes = {site: interface.thash(q) for site, q in qs.items()}; clean_hashes = json.loads(FORWARD.read_text())["pooled_projector_sha256"]
    reports = {}; finite = [orientation, training["capture_error"]]
    for removed in REMOVALS:
        support = [site for site in base_support if site != removed]
        _go, gai, gmi, counts = source.restricted_graph(backend, fresh["batch"], fresh["donor_batch"], fresh["base_full"], support, source=True)
        _co, cai, cmi, ccounts = source.restricted_graph(backend, fresh["control_batch"], fresh["control_donor_batch"], fresh["control_base_full"], support, source=True)
        generated = atlas.compiled_coordinates(backend, fresh["rows"], qs, fresh["base"][2], fresh["base"][5], gai, fresh["base"][3], gmi, bases)
        cgenerated = atlas.compiled_coordinates(backend, fresh["controls"], qs, fresh["control_base"][2], fresh["control_base"][5], cai, fresh["control_base"][3], cmi, bases)
        forward = addback.fit_arm(backend, {"fresh": fresh, "reader": reader}, support, "pooled", qs, bases, generated, cgenerated)
        forward["source_counts"], forward["control_source_counts"] = sorted(set(counts)), sorted(set(ccounts))
        reverse = reverse_arm(backend, fresh, support, qs, bases, reader, donor_full, control_donor_full, reverse_full_state)
        arm = {"removed": removed, "support": support, "forward": forward, "reverse": reverse}; reports[removed] = arm
        for report in (forward, reverse):
            finite += list(report["coordinate"]["signed_projection"].values()) + list(report["coordinate"]["residual"].values())
            finite += list(report["target"]["cells"].values()) + list(report["target"]["behavior_signed_projection"].values())
            finite += list(report["control"]["margin_rms_fraction"].values()) + [report["control"]["median_kl"], report["control"]["max_kl"], report["control"]["top1_flip_fraction"]]
    fit_ids = {row["row_id"] for row in fitted["target_rows"] + fitted["control_rows"]}; eval_ids = {row["row_id"] for row in fresh["rows"] + fresh["controls"]}
    pa = reader_ok and orientation <= 1e-6 and hashes == clean_hashes and not (fit_ids & eval_ids) and len(base_support) == 48 and all(math.isfinite(float(v)) for v in finite)
    def coords(arm): return all(min(r["coordinate"]["signed_projection"].values()) >= .75 and r["coordinate"]["mean_residual"] <= .2 and r["coordinate"]["worst_residual"] <= .2 and r["source_counts"] == [1] and r["control_source_counts"] == [1] for r in (arm["forward"], arm["reverse"]))
    def behavior(arm): return all(min(r["target"]["behavior_signed_projection"].values()) >= .8 and r["target"]["worst_target_residual"] <= .15 for r in (arm["forward"], arm["reverse"]))
    def selective(arm): return all(max(r["control"]["margin_rms_fraction"].values()) <= .1 and r["control"]["median_kl"] <= .02 and r["control"]["top1_flip_fraction"] == 0.0 for r in (arm["forward"], arm["reverse"]))
    feasible = [removed for removed in REMOVALS if coords(reports[removed]) and behavior(reports[removed]) and selective(reports[removed])]
    selected = feasible[0] if feasible else None
    gaps = {removed: max(abs(reports[removed]["forward"]["target"]["behavior_signed_projection"][task] - reports[removed]["reverse"]["target"]["behavior_signed_projection"][task]) for task in ("temporal", "iswas")) for removed in REMOVALS}
    predictions = {"pred_a_authority_hash_disjointness_finiteness_and_exact_price": bool(pa),
        "pred_b_at_least_one_rank47_arm_has_bidirectional_coordinate_fidelity": any(coords(arm) for arm in reports.values()),
        "pred_c_at_least_one_rank47_arm_has_bidirectional_behavior_fidelity": selected is not None,
        "pred_d_at_least_one_rank47_arm_is_bidirectionally_selective": selected is not None,
        "pred_e_selected_rank47_arm_is_direction_stable": selected is not None and gaps[selected] <= .05}
    compact = {removed: {"jointly_feasible": removed in feasible, "direction_gap": gaps[removed],
        "coordinate_projection_min": min(min(reports[removed][d]["coordinate"]["signed_projection"].values()) for d in ("forward", "reverse")),
        "target_projection_min": min(min(reports[removed][d]["target"]["behavior_signed_projection"].values()) for d in ("forward", "reverse")),
        "control_flip_max": max(reports[removed][d]["control"]["top1_flip_fraction"] for d in ("forward", "reverse"))} for removed in REMOVALS}
    terminal = "invalid" if not pa else SUCCESS_TERMINAL if all(predictions.values()) else FAILURE_TERMINAL
    result = {"schema": RESULT_SCHEMA, "candidate_id": CANDIDATE, "started_utc": started, "finished_utc": now(),
        "serial_seconds": time.perf_counter() - tic, "authority_sha256": EXPECTED, "pooled_projector_sha256": hashes, "tested_deletions": REMOVALS,
        "feasible_deletions": feasible, "selected_deletion": selected, "selected_support": reports[selected]["support"] if selected else None,
        "reports": reports, "arm_summary": compact, "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": exact_forwards, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}}
    atomic_create_json(OUT, result); print(json.dumps({k: result[k] for k in ("feasible_deletions", "selected_deletion", "arm_summary", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
