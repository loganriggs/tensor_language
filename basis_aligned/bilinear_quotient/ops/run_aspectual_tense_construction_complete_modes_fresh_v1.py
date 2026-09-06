#!/usr/bin/env python3
"""Construction-complete endpoint modes tested on a fourth fresh lexicon."""

# BQGATE: EXPERIMENT pred_a_authority_novelty_capability_projection_finiteness_and_price pred_b_full_programs_generalize pred_c_construction_complete_modes_are_sufficient pred_d_construction_complete_complements_are_secondary pred_e_mlp_modes_remain_compressive_without_causal_fit
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import attention_source_group_eval as source_groups
import circuit_candidate_aspectual_tense_matched_fresh_lexicon_v4 as fresh
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import positioned_component_program_eval as positioned
import run_aspectual_tense_carrier_component_greedy_program_v1 as greedy
import run_aspectual_tense_endpoint_jacobian_union_mode_program_v1 as endpoint
import run_aspectual_tense_l9h1h4_source_position_weight_validation_v1 as source_rank
import run_aspectual_tense_suffix_jacobian_mode_program_v1 as jacobian
import run_aspectual_tense_weight_jacobian_union_mode_program_v1 as union
import run_aspectual_tense_weight_readable_mode_program_v1 as static
import subspace_weight_atlas as atlas


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_tense_construction_complete_modes_fresh_v1.json"
ENDPOINT_RESULT = ROOT / "circuits/followups/aspectual_tense_endpoint_jacobian_union_mode_program_v1_result.json"
ENDPOINT_RUNNER = ROOT / "ops/run_aspectual_tense_endpoint_jacobian_union_mode_program_v1.py"
BUILDER = ROOT / "ops/circuit_candidate_aspectual_tense_matched_fresh_lexicon_v4.py"
OUT = ROOT / "circuits/followups/aspectual_tense_construction_complete_modes_fresh_v1_result.json"
CANDIDATE_ID = "aspectual_tense.construction_complete_modes_fresh_v1"
EXPECTED = {"prior": "7f4e2e6d418cd736db0a868249de399b4789ffd53c594a268649701965261c4a",
            "endpoint_result": "edba80528947b591c7d483d0edff90207835659d5036582b13af3020b25c5f78",
            "endpoint_runner": "f9ef86bb4ba8256da54c892a48d47c524b7451f678f92325e46dc20a7daa1aa6",
            "builder": "1b78f075b0a8d5e50578793fd512a45b635b6d353b85441ccaf3bf851efc4b05"}
ROW_DIGESTS = {"has_had": "d3a8926058996f6ef8e150a0a66812b1c3feaf7b268ba7f6de13be0e7946b7e7",
               "is_was": "620d54aa95727bb29d02534974a23edb51bc8fdafc8fe63f00dc1f6c75a366d6"}
MAX_PRICE = {"model_forwards": 28, "example_evaluations": 566, "records": 192,
             "transformer_backwards": 4, "model_updates": 0}


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def validate_static():
    paths = {"prior": PRIOR, "endpoint_result": ENDPOINT_RESULT,
             "endpoint_runner": ENDPOINT_RUNNER, "builder": BUILDER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("authority hash changed")
    endpoint_result = json.loads(ENDPOINT_RESULT.read_text())
    splits, chosen, pools, programs, subspaces = endpoint.validate_static()
    rows_by_bank = fresh.build_rows_by_bank()
    if (endpoint_result.get("terminal") != "null"
            or fresh.validate_rows_by_bank(rows_by_bank) != ROW_DIGESTS):
        raise ExperimentError("endpoint null or fresh authority changed")
    return splits, chosen, pools, programs, subspaces, rows_by_bank


def subset_output(output, indices):
    return producer.BatchOutput(tuple(output.answer_foil[index] for index in indices), {})


def main():
    splits, chosen, pools, programs, subspaces, rows_by_bank = validate_static()
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
              "model_loaded": False, "queue_touched": False,
              "development_counts": {"has": 62, "is": 29},
              "fresh_rows_per_task_panel": 16, "maximum_price": MAX_PRICE}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch, model = backend.torch, backend.model
    projections, rank_reports = {}, {}
    all_capable, forwards, evaluations, backwards = True, 0, 0, 0
    for task in ("has", "is"):
        rows = [*splits[f"{task}_fit"], *splits[f"{task}_heldout"], *splits[f"{task}_a2"]]
        components = greedy.program_specs(programs[task], pools[task])
        base_batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
        banks = source_rank.carrier_banks(task, base_batch, donor_batch)
        base_output, base_gradients = jacobian.gradient_forward(backend, base_batch, components, banks)
        donor_output, donor_gradients = jacobian.gradient_forward(backend, donor_batch, components, banks)
        all_capable = all_capable and jacobian.capable_output(base_output) and jacobian.capable_output(donor_output)
        task_basis = static.stored_basis(torch, subspaces["subspaces"][task]["basis"]).cuda()
        read = atlas.head_bank_value_read_map(model.transformer.h[9].attn, static.HEADS, task_basis)
        residual_static = static.row_basis(torch, read)
        projections[task], rank_reports[task] = {}, {}
        for component in components:
            name = static.label(component)
            endpoint_q, singular = endpoint.complete_basis(
                torch, torch.cat([base_gradients[name], donor_gradients[name]], dim=0))
            if component.kind == "mlp":
                static_q = residual_static
            else:
                mapped = atlas.attention_writer_to_read_map(
                    model.transformer.h[component.layer].attn, component.heads[0], read)["contraction"]
                static_q = static.row_basis(torch, mapped)
            q = union.union_basis(torch, static_q, endpoint_q)
            projections[task][name] = q
            rank_reports[task][name] = {"static_rank": int(static_q.shape[1]),
                "endpoint_jacobian_rank": int(endpoint_q.shape[1]), "union_rank": int(q.shape[1]),
                "width": int(q.shape[0]), "gradient_rows": int(base_gradients[name].shape[0]
                    + donor_gradients[name].shape[0]),
                "singular_values": [float(value) for value in singular.detach().cpu()],
                "gram_max_abs_error": float((q.T @ q - torch.eye(q.shape[1], device=q.device)).abs().max())}
        forwards += 2
        evaluations += 2 * len(rows)
        backwards += 2
    panels = {"has": {family: [row for row in rows_by_bank["has_had"]
                               if row["transform_id"] == family] for family in ("A1", "A2")},
              "is": {family: [row for row in rows_by_bank["is_was"]
                              if row["transform_id"] == family] for family in ("A1", "A2")}}
    records, metrics, capability, bank_widths = [], {"has": {}, "is": {}}, {}, {}
    for task in ("has", "is"):
        components = greedy.program_specs(programs[task], pools[task])
        for family in ("A1", "A2"):
            raw_rows = panels[task][family]
            raw_base, raw_donor = das._batch(backend, raw_rows, side="base"), das._batch(backend, raw_rows, side="donor")
            raw_base_output = backend.native(raw_base, capture=False)
            raw_donor_output = backend.native(raw_donor, capture=False)
            indices = [index for index, (base_pair, donor_pair) in enumerate(zip(
                raw_base_output.answer_foil, raw_donor_output.answer_foil))
                if float(base_pair[0]) - float(base_pair[1]) > 0
                and float(donor_pair[0]) - float(donor_pair[1]) > 0]
            rows = [raw_rows[index] for index in indices]
            directions = {direction: sum(row["direction_id"] == direction for row in rows)
                          for direction in {row["direction_id"] for row in raw_rows}}
            panel = f"{task}_{family}"
            capability[panel] = {"total": 16, "jointly_capable": len(rows),
                                 "direction_counts": directions,
                                 "passed": len(rows) >= 12 and min(directions.values()) >= 6}
            if not rows:
                raise ExperimentError("fresh capability selection is empty")
            base_batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
            base_output = subset_output(raw_base_output, indices)
            donor_output, cache = positioned.capture_full_components(
                backend, donor_batch, source_rank.capture_specs(chosen[task]))
            banks = source_rank.carrier_banks(task, base_batch, donor_batch)
            bank_widths[panel] = sorted(set(map(len, banks)))
            outputs = {"full": positioned.patch_positioned_components(
                    backend, base_batch, donor_batch, components, cache, banks, banks),
                "construction_union": static.patch_projected(backend, base_batch, donor_batch,
                    components, cache, banks, banks, projections[task], complement=False),
                "complement": static.patch_projected(backend, base_batch, donor_batch,
                    components, cache, banks, banks, projections[task], complement=True)}
            metrics[task][family] = {}
            for arm, output in outputs.items():
                arm_records = greedy.tagged(source_groups.recovery_records(
                    rows, base_output, donor_output, output, arm=arm), panel, task)
                records.extend(arm_records)
                metrics[task][family][arm] = source_groups.summarize(arm_records)
            forwards += 6
            evaluations += 32 + 4 * len(rows)
    price = {"model_forwards": forwards, "example_evaluations": evaluations,
             "records": len(records), "transformer_backwards": backwards, "model_updates": 0}
    reports = [report for task in rank_reports.values() for report in task.values()]
    finite = all(math.isfinite(float(record["recovery"])) for record in records) and all(
        math.isfinite(value) for report in reports for value in report["singular_values"])
    pred_a = bool(all(item["passed"] for item in capability.values()) and all_capable and finite
                  and max(report["gram_max_abs_error"] for report in reports) <= 1e-5
                  and all(price[key] <= MAX_PRICE[key] for key in ("model_forwards", "example_evaluations", "records"))
                  and bank_widths == {"has_A1": [3], "has_A2": [3], "is_A1": [2], "is_A2": [2]})
    pred_b = all(metrics[task][family]["full"]["mean_recovery"] >= 0.60
                 and metrics[task][family]["full"]["direction_fraction"] == 1.0
                 for task in ("has", "is") for family in ("A1", "A2"))
    pred_c = all(metrics[task][family]["construction_union"]["mean_recovery"]
                 >= 0.95 * metrics[task][family]["full"]["mean_recovery"]
                 and metrics[task][family]["construction_union"]["direction_fraction"] == 1.0
                 for task in ("has", "is") for family in ("A1", "A2"))
    pred_d = all(abs(metrics[task][family]["complement"]["mean_recovery"])
                 <= 0.20 * abs(metrics[task][family]["full"]["mean_recovery"])
                 for task in ("has", "is") for family in ("A1", "A2"))
    pred_e = all(report["union_rank"] < 1152 and report["union_rank"] <= (390 if task == "has" else 119)
                 for task, task_reports in rank_reports.items() for name, report in task_reports.items()
                 if name.startswith("MLP")) and backwards == 4 and price["model_updates"] == 0
    predictions = {
        "pred_a_authority_novelty_capability_projection_finiteness_and_price": pred_a,
        "pred_b_full_programs_generalize": pred_b,
        "pred_c_construction_complete_modes_are_sufficient": pred_c,
        "pred_d_construction_complete_complements_are_secondary": pred_d,
        "pred_e_mlp_modes_remain_compressive_without_causal_fit": pred_e,
    }
    terminal = "invalid" if not pred_a or not pred_e else ("screen" if all(predictions.values()) else "null")
    result = {"schema": "aspectual_tense_construction_complete_modes_fresh_result_v1",
              "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
              "started_utc": started_utc, "finished_utc": utc_now(),
              "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
              "fresh_row_digests": ROW_DIGESTS, "programs": programs,
              "rank_reports": rank_reports, "capability": capability,
              "metrics": metrics, "bank_widths": bank_widths,
              "predictions": predictions, "price": price, "records": records,
              "terminal": terminal}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "capability", "rank_reports",
        "metrics", "predictions", "price", "terminal")}, sort_keys=True))


if __name__ == "__main__":
    main()
