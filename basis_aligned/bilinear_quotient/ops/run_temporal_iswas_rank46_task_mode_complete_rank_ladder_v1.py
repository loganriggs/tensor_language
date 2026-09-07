#!/usr/bin/env python3
"""Complete causal rank ladder for task-typed modes in the pooled rank46 interface."""
# BQGATE: EXPERIMENT pred_a_authority_disjointness_finiteness_and_exact_price pred_b_nested_coordinate_capture_is_monotone pred_c_an_own_task_rank_at_most_four_is_program_sufficient pred_d_a_union_task_rank_at_most_three_is_program_sufficient pred_e_selected_compression_is_selective_and_strict
from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import compiled_response_ood as oodctx
import pooled_response_projector as pooled
import run_temporal_iswas_rank46_task_typed_mode_causal_factorial_v1 as priorrun
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
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_rank46_task_mode_complete_rank_ladder_v1.json"
FACTORIAL = ROOT / "circuits/followups/temporal_iswas_rank46_task_typed_mode_causal_factorial_v1_result.json"
FACTORIAL_RUNNER = ROOT / "ops/run_temporal_iswas_rank46_task_typed_mode_causal_factorial_v1.py"
RANK46 = priorrun.RANK46; CORRECTION = priorrun.CORRECTION; NOISE = priorrun.NOISE
TCAP = priorrun.TCAP; ICAP = priorrun.ICAP
OUT = ROOT / "circuits/followups/temporal_iswas_rank46_task_mode_complete_rank_ladder_v1_result.json"
EXPECTED = {"factorial": "301377a9eded372a3c7a624892ae88089618690336e1da67f573d4e29ae87de4",
    "factorial_runner": "aab635003748cf2164c8b6b467c10b35e4214ec7cb99dac451e78170b664f940",
    "rank46": priorrun.EXPECTED["rank46"], "correction": priorrun.EXPECTED["correction"], "noise": priorrun.EXPECTED["noise"]}
OWN_RANKS = tuple(range(1, 9)); UNION_RANKS = tuple(range(1, 5))


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def fit_full_modes(backend, fitted):
    torch = backend.torch
    task_rows = {task: [row for row in fitted["target_rows"] if klfit.task_name(row) == task] for task in ("temporal", "iswas")}
    captures = {task: priorrun.task_capture(backend, rows) for task, rows in task_rows.items()}
    modes = {}
    for site, q in fitted["projector"].items():
        modes[site] = {}
        for task, (batch, base, donor) in captures.items():
            z = priorrun.response_delta(backend, batch, base, donor, site) @ q
            modes[site][task] = torch.linalg.svd(z, full_matrices=False).Vh.T.contiguous()
    return task_rows, modes


def filtered(raw, qs, modes, family, rank, fresh, *, controls=False):
    output = {}
    rows = fresh["controls"] if controls else fresh["rows"]
    for site, value in raw.items():
        changed = value.clone()
        for i, row in enumerate(rows):
            task = "temporal" if controls else klfit.task_name(row)
            if family == "own": coordinate_basis = modes[site][task][:, :rank]
            else:
                joined = __import__('torch').cat((modes[site]["temporal"][:, :rank], modes[site]["iswas"][:, :rank]), dim=1)
                coordinate_basis = __import__('torch').linalg.qr(joined, mode="reduced").Q
            response_basis = qs[site].to(changed) @ coordinate_basis.to(changed)
            changed[i] = (changed[i] @ response_basis) @ response_basis.T
        output[site] = changed
    return output


def passes(report):
    return (min(report["target"]["behavior_signed_projection"].values()) >= .8 and report["target"]["worst_target_residual"] <= .15
            and max(report["control"]["margin_rms_fraction"].values()) <= .1 and report["control"]["median_kl"] <= .02
            and report["control"]["top1_flip_fraction"] == 0.0)


def main():
    paths = {"factorial": FACTORIAL, "factorial_runner": FACTORIAL_RUNNER, "rank46": RANK46, "correction": CORRECTION, "noise": NOISE}
    observed = {key: sha(path) for key, path in paths.items()}
    if observed != EXPECTED: raise RuntimeError(f"task-mode rank-ladder authority changed: {observed}")
    dry = {"candidate_id": "temporal_auxiliary.iswas_rank46_task_mode_complete_rank_ladder_v1", "dryrun": True,
        "gpu_accessed": False, "model_loaded": False, "queue_touched": False, "own_ranks": OWN_RANKS, "union_ranks": UNION_RANKS,
        "model_forwards_exact": 42, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1": print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    started, tic = now(), time.perf_counter(); backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch
    fitted = pooled.fit(backend); qs = fitted["projector"]; task_rows, modes = fit_full_modes(backend, fitted)
    training, bases = pooled.fit_mlp_input_bases(backend, qs); fresh = oodctx.capture(backend, TCAP, ICAP)
    support = json.loads(RANK46.read_text())["selected_support"]
    _go, gai, gmi, counts = source.restricted_graph(backend, fresh["batch"], fresh["donor_batch"], fresh["base_full"], support, source=True)
    _co, cai, cmi, ccounts = source.restricted_graph(backend, fresh["control_batch"], fresh["control_donor_batch"], fresh["control_base_full"], support, source=True)
    generated = atlas.compiled_coordinates(backend, fresh["rows"], qs, fresh["base"][2], fresh["base"][5], gai, fresh["base"][3], gmi, bases)
    cgenerated = atlas.compiled_coordinates(backend, fresh["controls"], qs, fresh["control_base"][2], fresh["control_base"][5], cai, fresh["control_base"][3], cmi, bases)
    reader, orientation, reader_ok = greedy.physical_reader(backend, json.loads(comp.WEIGHTS.read_text()))
    control_ctx = {"rows": fresh["controls"], "base_logits": das.head_logits(backend, fresh["control_base_state"]).float()}
    reports = {"own": {}, "union": {}}; finite = [orientation, training["capture_error"]]
    for family, ranks in (("own", OWN_RANKS), ("union", UNION_RANKS)):
        for rank in ranks:
            values = filtered(generated, qs, modes, family, rank, fresh)
            cvalues = filtered(cgenerated, qs, modes, family, rank, fresh, controls=True)
            output = pruned.run_factors(backend, fresh["batch"], fresh["base"][1], qs,
                {s: v for s, v in values.items() if atlasrun.site_parts(s)[0] == "attn"}, {s: v for s, v in values.items() if atlasrun.site_parts(s)[0] == "mlp"}, use_attention=True, use_mlp=True)
            coutput = pruned.run_factors(backend, fresh["control_batch"], fresh["control_base"][1], qs,
                {s: v for s, v in cvalues.items() if atlasrun.site_parts(s)[0] == "attn"}, {s: v for s, v in cvalues.items() if atlasrun.site_parts(s)[0] == "mlp"}, use_attention=True, use_mlp=True)
            coordinate = priorrun.coordinate_by_task(fresh, generated, values)
            target = ood.target_report(backend, fresh["rows"], fresh["base_state"], fresh["full_state"], atlasrun.states(torch, backend, output, fresh["rows"]), reader, fresh["temporal_n"])
            control = klfit.control_metrics(backend, control_ctx, das.head_logits(backend, atlasrun.states(torch, backend, coutput, fresh["controls"])).float())
            installed_rank = rank if family == "own" else min(2 * rank, 8)
            report = {"task_rank": rank, "installed_rank": installed_rank, "coordinate": coordinate, "target": target, "control": control, "passes": False}
            report["passes"] = passes(report); reports[family][str(rank)] = report
            finite += list(target["cells"].values()) + list(target["behavior_signed_projection"].values())
            finite += list(control["margin_rms_fraction"].values()) + [control["median_kl"], control["max_kl"], control["top1_flip_fraction"]]
            finite += [x for task in coordinate.values() for x in (task["mean_signed_projection"], task["worst_residual"])]
    selected = {family: next((rank for rank in ranks if reports[family][str(rank)]["passes"]), None) for family, ranks in (("own", OWN_RANKS), ("union", UNION_RANKS))}
    fit_ids = {row["row_id"] for rows in task_rows.values() for row in rows} | {row["row_id"] for row in fitted["control_rows"]}
    eval_ids = {row["row_id"] for row in fresh["rows"] + fresh["controls"]}
    pa = reader_ok and orientation <= 1e-6 and not (fit_ids & eval_ids) and len(support) == 46 and set(counts) == {1} and set(ccounts) == {1} and all(math.isfinite(float(v)) for v in finite)
    monotone = True
    for family, ranks in (("own", OWN_RANKS), ("union", UNION_RANKS)):
        for task in ("temporal", "iswas"):
            values = [reports[family][str(rank)]["coordinate"][task]["mean_signed_projection"] for rank in ranks]
            monotone &= all(b + .01 >= a for a, b in zip(values, values[1:]))
    predictions = {"pred_a_authority_disjointness_finiteness_and_exact_price": bool(pa),
        "pred_b_nested_coordinate_capture_is_monotone": bool(monotone),
        "pred_c_an_own_task_rank_at_most_four_is_program_sufficient": selected["own"] is not None and selected["own"] <= 4,
        "pred_d_a_union_task_rank_at_most_three_is_program_sufficient": selected["union"] is not None and selected["union"] <= 3,
        "pred_e_selected_compression_is_selective_and_strict": any(selected[f] is not None and reports[f][str(selected[f])]["installed_rank"] < 8 for f in selected)}
    terminal = "invalid" if not pa else "compressed_task_typed_rank46_program" if all(predictions.values()) else "complete_task_mode_rank_boundary"
    result = {"schema": "temporal_iswas_rank46_task_mode_complete_rank_ladder_result_v1", "started_utc": started, "finished_utc": now(),
        "serial_seconds": time.perf_counter() - tic, "authority_sha256": EXPECTED, "support_count": len(support), "selected_rank": selected,
        "reports": reports, "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": 42, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}}
    atomic_create_json(OUT, result); print(json.dumps({"selected_rank": selected, "reports": reports, "predictions": predictions, "terminal": terminal, "price": result["price"]}, sort_keys=True))


if __name__ == "__main__": main()
