#!/usr/bin/env python3
"""Causal test of static-weight plus suffix-Jacobian union modes."""

# BQGATE: EXPERIMENT pred_a_authority_replay_capability_projection_finiteness_and_exact_price pred_b_union_modes_are_sufficient pred_c_union_complements_are_secondary pred_d_union_is_still_dimensionally_compressive pred_e_no_causal_fit_or_model_update
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import attention_source_group_eval as source_groups
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import positioned_component_program_eval as positioned
import run_aspectual_tense_carrier_component_greedy_program_v1 as greedy
import run_aspectual_tense_l9h1h4_source_position_weight_validation_v1 as source_rank
import run_aspectual_tense_suffix_jacobian_mode_program_v1 as jacobian
import run_aspectual_tense_weight_readable_mode_program_v1 as static
import subspace_weight_atlas as atlas


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_tense_weight_jacobian_union_mode_program_v1.json"
JACOBIAN_RESULT = ROOT / "circuits/followups/aspectual_tense_suffix_jacobian_mode_program_v1_result.json"
JACOBIAN_RUNNER = ROOT / "ops/run_aspectual_tense_suffix_jacobian_mode_program_v1.py"
STATIC_RESULT = ROOT / "circuits/followups/aspectual_tense_weight_readable_mode_program_v1_result.json"
STATIC_RUNNER = ROOT / "ops/run_aspectual_tense_weight_readable_mode_program_v1.py"
SUBSPACES = ROOT / "circuits/followups/aspectual_tense_l9h1h4_shared_value_weight_subspace_v2_result.json"
ATLAS = ROOT / "ops/subspace_weight_atlas.py"
OUT = ROOT / "circuits/followups/aspectual_tense_weight_jacobian_union_mode_program_v1_result.json"
CANDIDATE_ID = "aspectual_tense.weight_jacobian_union_mode_program_v1"
EXPECTED = {"prior": "8e954d90b26b4b59135703aef47a9eae20a813b9554014288648e16c2db5fad4",
            "jacobian_result": "94dca39621701afcc87a17b0006d36b5378b6c67cfffd1287172c1c3a67b10a3",
            "jacobian_runner": "24b6c554960f2ace5e920944b0d0bffbb0aece185e81aa39ea9ff0b6aa0031c5",
            "static_result": "a49f9d6b06cb37a1759e449b72fb1446b93addae80fda521c0bd6d9af61bed49",
            "static_runner": "e72ec560692b2924c506f592ce83037f353b0bebc5cfa1e3705f1c2aab1a1c8f",
            "subspaces": "0ae262ee932d6ecb93d1df028ac080f3a4597861620da79b7ad45a0f3ae2d16e",
            "atlas": "2e7d3a546813a6029eca6fae455ad5abd03b429fcee92432ca6fe06e835e83f5"}
PRICE = {"model_forwards": 26, "example_evaluations": 407, "records": 201,
         "transformer_backwards": 2, "model_updates": 0}


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def union_basis(torch, left, right):
    combined = torch.cat([left, right], dim=1).float()
    u, singular, _vh = torch.linalg.svd(combined, full_matrices=False)
    keep = singular > singular.max().clamp_min(1e-30) * 1e-6
    return torch.linalg.qr(u[:, keep].double()).Q.float().to(combined.device)


def validate_static():
    paths = {"prior": PRIOR, "jacobian_result": JACOBIAN_RESULT,
             "jacobian_runner": JACOBIAN_RUNNER, "static_result": STATIC_RESULT,
             "static_runner": STATIC_RUNNER, "subspaces": SUBSPACES, "atlas": ATLAS}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("authority hash changed")
    jacobian_result, static_result = [json.loads(path.read_text())
                                      for path in (JACOBIAN_RESULT, STATIC_RESULT)]
    _old, splits, chosen, pools, programs = jacobian.validate_static()
    _splits2, _chosen2, _pools2, programs2, subspaces = static.validate_static()
    if (jacobian_result.get("terminal") != "null" or static_result.get("terminal") != "null"
            or programs != programs2):
        raise ExperimentError("parent null pattern or programs changed")
    return splits, chosen, pools, programs, subspaces


def main():
    splits, chosen, pools, programs, subspaces = validate_static()
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
              "model_loaded": False, "queue_touched": False, "gradient_energy": jacobian.ENERGY,
              "fit_counts": {"has": 16, "is": 8}, **PRICE}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch, model = backend.torch, backend.model
    projections, rank_reports, replay_errors = {}, {}, []
    all_capable, forwards, evaluations, backwards = True, 0, 0, 0
    for task in ("has", "is"):
        rows, components = splits[f"{task}_fit"], greedy.program_specs(programs[task], pools[task])
        base_batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
        banks = source_rank.carrier_banks(task, base_batch, donor_batch)
        native = backend.native(base_batch, capture=False)
        gradient_output, gradients = jacobian.gradient_forward(backend, base_batch, components, banks)
        donor_output, _cache = positioned.capture_full_components(
            backend, donor_batch, source_rank.capture_specs(chosen[task]))
        replay_errors.append(float((torch.tensor(native.answer_foil, device=backend.device)
                                    - torch.tensor(gradient_output.answer_foil, device=backend.device)).abs().max()))
        all_capable = all_capable and jacobian.capable_output(native) and jacobian.capable_output(
            gradient_output) and jacobian.capable_output(donor_output)
        task_basis = static.stored_basis(torch, subspaces["subspaces"][task]["basis"]).cuda()
        read = atlas.head_bank_value_read_map(model.transformer.h[9].attn, static.HEADS, task_basis)
        residual_static = static.row_basis(torch, read)
        projections[task], rank_reports[task] = {}, {}
        for component in components:
            name = static.label(component)
            gradient_q, _singular, captured = jacobian.energy_basis(torch, gradients[name])
            if component.kind == "mlp":
                static_q = residual_static
            else:
                mapped = atlas.attention_writer_to_read_map(
                    model.transformer.h[component.layer].attn, component.heads[0], read)["contraction"]
                static_q = static.row_basis(torch, mapped)
            q = union_basis(torch, static_q, gradient_q)
            projections[task][name] = q
            rank_reports[task][name] = {"static_rank": int(static_q.shape[1]),
                "jacobian_rank": int(gradient_q.shape[1]), "union_rank": int(q.shape[1]),
                "width": int(q.shape[0]), "jacobian_captured_energy": captured,
                "gram_max_abs_error": float((q.T @ q - torch.eye(q.shape[1], device=q.device)).abs().max())}
        forwards += 3
        evaluations += 3 * len(rows)
        backwards += 1
    records, metrics, bank_widths = [], {"has": {}, "is": {}}, {}
    for task in ("has", "is"):
        components = greedy.program_specs(programs[task], pools[task])
        for panel in ("heldout", "a2"):
            split, rows = f"{task}_{panel}", splits[f"{task}_{panel}"]
            base_batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
            base_output = backend.native(base_batch, capture=False)
            donor_output, cache = positioned.capture_full_components(
                backend, donor_batch, source_rank.capture_specs(chosen[task]))
            banks = source_rank.carrier_banks(task, base_batch, donor_batch)
            bank_widths[split] = sorted(set(map(len, banks)))
            all_capable = all_capable and jacobian.capable_output(base_output) and jacobian.capable_output(donor_output)
            outputs = {"full": positioned.patch_positioned_components(
                    backend, base_batch, donor_batch, components, cache, banks, banks),
                "union": static.patch_projected(backend, base_batch, donor_batch, components,
                    cache, banks, banks, projections[task], complement=False),
                "complement": static.patch_projected(backend, base_batch, donor_batch, components,
                    cache, banks, banks, projections[task], complement=True)}
            metrics[task][panel] = {}
            for arm, output in outputs.items():
                arm_records = greedy.tagged(source_groups.recovery_records(
                    rows, base_output, donor_output, output, arm=arm), split, task)
                records.extend(arm_records)
                metrics[task][panel][arm] = source_groups.summarize(arm_records)
            forwards += 5
            evaluations += 5 * len(rows)
    price = {"model_forwards": forwards, "example_evaluations": evaluations,
             "records": len(records), "transformer_backwards": backwards, "model_updates": 0}
    reports = [report for task in rank_reports.values() for report in task.values()]
    finite = all(math.isfinite(float(record["recovery"])) for record in records)
    pred_a = bool(max(replay_errors) <= 1e-4 and all_capable and finite and price == PRICE
                  and max(report["gram_max_abs_error"] for report in reports) <= 1e-5
                  and bank_widths == {"has_heldout": [3], "has_a2": [3],
                                      "is_heldout": [2], "is_a2": [2]})
    pred_b = all(metrics[task][panel]["union"]["mean_recovery"]
                 >= 0.95 * metrics[task][panel]["full"]["mean_recovery"]
                 and metrics[task][panel]["union"]["direction_fraction"] == 1.0
                 for task in ("has", "is") for panel in ("heldout", "a2"))
    pred_c = all(abs(metrics[task][panel]["complement"]["mean_recovery"])
                 <= 0.20 * abs(metrics[task][panel]["full"]["mean_recovery"])
                 for task in ("has", "is") for panel in ("heldout", "a2"))
    pred_d = all(report["union_rank"] < report["width"]
                 and report["union_rank"] <= (66 if task == "has" else 19)
                 for task, task_reports in rank_reports.items() for report in task_reports.values())
    pred_e = backwards == 2 and price["model_updates"] == 0
    predictions = {
        "pred_a_authority_replay_capability_projection_finiteness_and_exact_price": pred_a,
        "pred_b_union_modes_are_sufficient": pred_b,
        "pred_c_union_complements_are_secondary": pred_c,
        "pred_d_union_is_still_dimensionally_compressive": pred_d,
        "pred_e_no_causal_fit_or_model_update": pred_e,
    }
    terminal = "invalid" if not pred_a or not pred_e else ("screen" if all(predictions.values()) else "null")
    result = {"schema": "aspectual_tense_weight_jacobian_union_mode_program_result_v1",
              "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
              "started_utc": started_utc, "finished_utc": utc_now(),
              "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
              "programs": programs, "rank_reports": rank_reports,
              "gradient_forward_replay_max_abs_error": max(replay_errors),
              "metrics": metrics, "bank_widths": bank_widths,
              "predictions": predictions, "price": price, "records": records,
              "terminal": terminal}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "rank_reports",
        "gradient_forward_replay_max_abs_error", "metrics", "predictions", "price", "terminal")},
        sort_keys=True))


if __name__ == "__main__":
    main()
