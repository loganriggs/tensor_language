#!/usr/bin/env python3
"""Causal own/cross/union factorial for task-typed modes in the pooled interface."""
# BQGATE: EXPERIMENT pred_a_authority_disjointness_hashes_finiteness_and_exact_price pred_b_full_rank46_arm_replays_the_licensed_program pred_c_own_task_modes_are_causally_sufficient pred_d_own_task_modes_beat_cross_task_modes pred_e_union_modes_restore_the_full_program_selectively
from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import compiled_response_ood as oodctx
import pooled_response_projector as pooled
import run_temporal_iswas_joint_projector_task_usage_v1 as usage
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
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_rank46_task_typed_mode_causal_factorial_v1.json"
RANK46 = ROOT / "circuits/followups/temporal_five_mlp_rank47_pooled_greedy_rank46_deletion_v1_result.json"
CORRECTION = ROOT / "circuits/followups/temporal_five_mlp_rank47_pooled_greedy_rank46_deletion_v1_count_correction.json"
NOISE = ROOT / "circuits/followups/temporal_five_mlp_rank46_pooled_bidirectional_source_noise_ood_v1_result.json"
MLP_USAGE = ROOT / "circuits/followups/temporal_iswas_joint_projector_task_usage_v1_result.json"
ATTN_USAGE = ROOT / "circuits/followups/temporal_iswas_attention_ov_task_usage_v1_result.json"
TCAP = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v12_capability_v1_result.json"
ICAP = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v11_capability_v1_result.json"
HELPER = ROOT / "ops/pooled_response_projector.py"
OUT = ROOT / "circuits/followups/temporal_iswas_rank46_task_typed_mode_causal_factorial_v1_result.json"
EXPECTED = {"rank46": "dc8b66d826acede98bde996babe42420dd9e805981f6b81de458d568f29eea1d",
    "correction": "fc1b3ac442e93a841131996bd73f826a18279c80947089deac4ed7060a988c50",
    "noise": "1f8afe2f2e01e7b669ea8cb4343283d1fc246b8d07c6d70486af6649ef1cc824",
    "mlp_usage": "c446f90beeb2f86856b062faf85fe8d9218985cabee0670cd2f1002a2985934c",
    "attn_usage": "4a1e42a1ab490f2f22f8c7ae9dc9636d836876812bab60cff647985861c7947e",
    "helper": "09a7ea5504722a1765367f8422cbbb858373bcc5e910d5aae75ccba46cb124ed"}
ARMS = ("full", "own", "cross", "union")


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def task_capture(backend, rows):
    batch, _bo, base, _do, donor, _bi, _di = interface.cap_inputs(backend, rows)
    return batch, base, donor


def response_delta(backend, batch, base, donor, site):
    kind, layer, head = atlasrun.site_parts(site)
    if kind == "mlp": return interface.valid_delta(batch, base["mlp"][layer], donor["mlp"][layer]).to(backend.device).float()
    width = backend.model.config.n_embd // backend.model.config.n_head
    sl = slice(head * width, (head + 1) * width)
    return interface.valid_delta(batch, base["attention"][layer], donor["attention"][layer], sl).to(backend.device).float()


def fit_modes(backend, fitted):
    torch = backend.torch
    task_rows = {task: [row for row in fitted["target_rows"] if klfit.task_name(row) == task] for task in ("temporal", "iswas")}
    captures = {task: task_capture(backend, rows) for task, rows in task_rows.items()}
    modes = {}
    for site, q in fitted["projector"].items():
        modes[site] = {}
        for task, (batch, base, donor) in captures.items():
            z = response_delta(backend, batch, base, donor, site) @ q
            modes[site][task], _energy = usage.top_usage(torch, z)
        joined = torch.cat((modes[site]["temporal"], modes[site]["iswas"]), dim=1)
        modes[site]["union"] = torch.linalg.qr(joined, mode="reduced").Q
    return task_rows, modes


def filter_coordinates(backend, fresh, raw, modes, arm, *, controls=False):
    output = {}
    for site, value in raw.items():
        changed = value.clone()
        for i, row in enumerate(fresh["controls"] if controls else fresh["rows"]):
            task = "temporal" if controls else klfit.task_name(row)
            if arm == "full": continue
            key = "union" if arm == "union" else task if arm == "own" else ("iswas" if task == "temporal" else "temporal")
            v = modes[site][key].to(changed); changed[i] = (changed[i] @ v) @ v.T
        output[site] = changed
    return output


def coordinate_by_task(fresh, reference, changed):
    output = {}
    for task in ("temporal", "iswas"):
        ids = [i for i, row in enumerate(fresh["rows"]) if klfit.task_name(row) == task]
        per_site = {}
        for site in reference:
            apart, bpart = [], []
            for i in ids:
                stop = int(fresh["batch"].semantic_positions[i]) + 1
                apart.append(reference[site][i, :stop].float()); bpart.append(changed[site][i, :stop].float())
            a = __import__('torch').cat(apart); b = __import__('torch').cat(bpart); den = a.square().sum().clamp_min(1e-30)
            per_site[site] = {"signed_projection": float((a * b).sum() / den), "residual": float((b-a).square().sum() / den)}
        output[task] = {"mean_signed_projection": sum(x["signed_projection"] for x in per_site.values()) / len(per_site),
                        "worst_residual": max(x["residual"] for x in per_site.values()), "sites": per_site}
    return output


def main():
    paths = {"rank46": RANK46, "correction": CORRECTION, "noise": NOISE, "mlp_usage": MLP_USAGE, "attn_usage": ATTN_USAGE, "helper": HELPER}
    observed = {key: sha(path) for key, path in paths.items()}
    if observed != EXPECTED: raise RuntimeError(f"task-mode causal authority changed: {observed}")
    dry = {"candidate_id": "temporal_auxiliary.iswas_rank46_task_typed_mode_causal_factorial_v1", "dryrun": True,
        "gpu_accessed": False, "model_loaded": False, "queue_touched": False, "arms": ARMS,
        "model_forwards_exact": 26, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1": print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    started, tic = now(), time.perf_counter(); backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch
    fitted = pooled.fit(backend); qs = fitted["projector"]; task_rows, modes = fit_modes(backend, fitted)
    training, bases = pooled.fit_mlp_input_bases(backend, qs); fresh = oodctx.capture(backend, TCAP, ICAP)
    support = json.loads(RANK46.read_text())["selected_support"]
    _go, gai, gmi, counts = source.restricted_graph(backend, fresh["batch"], fresh["donor_batch"], fresh["base_full"], support, source=True)
    _co, cai, cmi, ccounts = source.restricted_graph(backend, fresh["control_batch"], fresh["control_donor_batch"], fresh["control_base_full"], support, source=True)
    generated = atlas.compiled_coordinates(backend, fresh["rows"], qs, fresh["base"][2], fresh["base"][5], gai, fresh["base"][3], gmi, bases)
    cgenerated = atlas.compiled_coordinates(backend, fresh["controls"], qs, fresh["control_base"][2], fresh["control_base"][5], cai, fresh["control_base"][3], cmi, bases)
    reader, orientation, reader_ok = greedy.physical_reader(backend, json.loads(comp.WEIGHTS.read_text()))
    control_ctx = {"rows": fresh["controls"], "base_logits": das.head_logits(backend, fresh["control_base_state"]).float()}
    reports = {}; finite = [orientation, training["capture_error"]]
    for arm in ARMS:
        values = filter_coordinates(backend, fresh, generated, modes, arm)
        cvalues = filter_coordinates(backend, fresh, cgenerated, modes, arm, controls=True)
        output = pruned.run_factors(backend, fresh["batch"], fresh["base"][1], qs,
            {s: v for s, v in values.items() if atlasrun.site_parts(s)[0] == "attn"},
            {s: v for s, v in values.items() if atlasrun.site_parts(s)[0] == "mlp"}, use_attention=True, use_mlp=True)
        coutput = pruned.run_factors(backend, fresh["control_batch"], fresh["control_base"][1], qs,
            {s: v for s, v in cvalues.items() if atlasrun.site_parts(s)[0] == "attn"},
            {s: v for s, v in cvalues.items() if atlasrun.site_parts(s)[0] == "mlp"}, use_attention=True, use_mlp=True)
        target = ood.target_report(backend, fresh["rows"], fresh["base_state"], fresh["full_state"], atlasrun.states(torch, backend, output, fresh["rows"]), reader, fresh["temporal_n"])
        control = klfit.control_metrics(backend, control_ctx, das.head_logits(backend, atlasrun.states(torch, backend, coutput, fresh["controls"])).float())
        coordinate = coordinate_by_task(fresh, generated, values); reports[arm] = {"coordinate": coordinate, "target": target, "control": control}
        finite += list(target["cells"].values()) + list(target["behavior_signed_projection"].values())
        finite += list(control["margin_rms_fraction"].values()) + [control["median_kl"], control["max_kl"], control["top1_flip_fraction"]]
        finite += [x for task in coordinate.values() for x in (task["mean_signed_projection"], task["worst_residual"])]
    fit_ids = {row["row_id"] for rows in task_rows.values() for row in rows} | {row["row_id"] for row in fitted["control_rows"]}
    eval_ids = {row["row_id"] for row in fresh["rows"] + fresh["controls"]}
    hashes = {site: interface.thash(q) for site, q in qs.items()}; noise = json.loads(NOISE.read_text())
    pa = (reader_ok and orientation <= 1e-6 and hashes == noise["pooled_projector_sha256"] and not (fit_ids & eval_ids)
          and len(support) == 46 and set(counts) == {1} and set(ccounts) == {1} and all(math.isfinite(float(v)) for v in finite))
    full, own, cross, union = (reports[arm] for arm in ARMS)
    def selective(r):
        return (max(r["control"]["margin_rms_fraction"].values()) <= .1 and r["control"]["median_kl"] <= .02
                and r["control"]["top1_flip_fraction"] == 0.0)
    pb = min(full["target"]["behavior_signed_projection"].values()) >= .8 and full["target"]["worst_target_residual"] <= .15 and selective(full)
    pc = all(own["target"]["behavior_signed_projection"][task] >= .65 and own["coordinate"][task]["mean_signed_projection"] >= .65 for task in ("temporal", "iswas"))
    advantages = {task: own["target"]["behavior_signed_projection"][task] - cross["target"]["behavior_signed_projection"][task] for task in ("temporal", "iswas")}
    pd = min(advantages.values()) >= .15
    pe = min(union["target"]["behavior_signed_projection"].values()) >= .8 and union["target"]["worst_target_residual"] <= .15 and selective(union)
    predictions = {"pred_a_authority_disjointness_hashes_finiteness_and_exact_price": bool(pa),
        "pred_b_full_rank46_arm_replays_the_licensed_program": bool(pb), "pred_c_own_task_modes_are_causally_sufficient": bool(pc),
        "pred_d_own_task_modes_beat_cross_task_modes": bool(pd), "pred_e_union_modes_restore_the_full_program_selectively": bool(pe)}
    terminal = "invalid" if not pa else "causal_task_typed_rank46_interface" if all(predictions.values()) else "task_mode_causal_factorial_null"
    result = {"schema": "temporal_iswas_rank46_task_typed_mode_causal_factorial_result_v1", "started_utc": started, "finished_utc": now(),
        "serial_seconds": time.perf_counter() - tic, "authority_sha256": EXPECTED, "pooled_projector_sha256": hashes,
        "fit_row_counts": {task: len(rows) for task, rows in task_rows.items()}, "evaluation_row_count": len(fresh["rows"]),
        "support_count": len(support), "mode_ranks": {site: {key: value.shape[1] for key, value in task.items()} for site, task in modes.items()},
        "reports": reports, "own_minus_cross_behavior": advantages, "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": 26, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}}
    atomic_create_json(OUT, result); print(json.dumps({"reports": reports, "own_minus_cross_behavior": advantages, "predictions": predictions, "terminal": terminal, "price": result["price"]}, sort_keys=True))


if __name__ == "__main__": main()
