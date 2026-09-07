#!/usr/bin/env python3
"""No-reselection temporal-v15/is-was-v15 confirmation of source mask 26."""
# BQGATE: EXPERIMENT pred_a_authority_disjointness_replay_closure_finiteness_and_price pred_b_full_parent_remains_functional pred_c_mask26_is_functional_and_full_equivalent pred_d_mask26_adds_no_parent_collateral pred_e_each_selected_site_is_jointly_necessary
import hashlib
import json
import math
import os
from pathlib import Path

import circuit_candidate_temporal_auxiliary_fresh_cues_v15 as temporal
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v15 as iswas
import circuit_das_subspace as das
import run_temporal_iswas_three_mlp_response_program_fresh_v12_v1 as rowslib
import run_temporal_iswas_rank16_source_task_pair_weight_groups_v1 as base
import run_temporal_iswas_rank16_source_task_pair_weight_groups_v2 as old_population

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_rank16_source_mask26_construction_holdout_v1.json"
COMPOSITION = ROOT / "circuits/followups/temporal_iswas_rank16_source_complement_composition_lattice_v1_result.json"
TCAP = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v15_capability_v1_result.json"
ICAP = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v15_capability_v2_audit_result.json"
ICAP_ROWS = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v15_capability_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_iswas_rank16_source_mask26_construction_holdout_v1_result.json"
EXPECTED = {
    "prior": "2528ea30e32b2d384c1519c44ddf54919a8e2313bd10885ecbec4c2e3a9697ac",
    "composition_result": "176ea2d57508fb42e83159c8f17a21044c7fcbda7b4e847113c92b15b441bea3",
    "composition_runner": "a0fa322f30fa7de04c26ad8bec6f3bd363c5bf3c0e9d0c92ab2b3c80e83fba79",
    "temporal_builder": "e11f490895dcbfaf06039640b6b980c9d0ef7f8e25d823a4b95803fd67617418",
    "temporal_capability": "5d60c5733147419ea3a988b842c5bee96652a0f6f939910cd03e1687b5478867",
    "iswas_builder": "e5774cd5e93c564ad3bdb3169c482c2e1ddb295b85cc7c9885a7b077493a8343",
    "iswas_capability": "fbe395731de4fa7850999864f67bd45ec709e5da3e3e237e9dcae352c1211c2f",
    "iswas_capability_rows": "ecc34f5d089f31269de29250d8d63843ce49c622f5cc0161cbea3afb9042d7f0",
    "weight_group_runner": "8ec435c1e551c37d5ba86cf4be7cd4541de8a7c6cb1f798707309c4119f9adb7",
    "old_population_runner": "7389053e5746120f4fa8611b4e182191363a29509088a4551542cf56c1b647fa",
}
MASKS = (0, 31, 26, 24, 18, 10)
MASK_ARMS = tuple(f"addback_mask_{mask:02d}" for mask in MASKS)
BASE_ARMS = (
    "full_rank16", "shared_mean", "task_contrast", "mean_plus_contrast",
    "own_task", "cross_task", "mean_plus_contrast_complement",
)
EXECUTED_ARMS = BASE_ARMS + MASK_ARMS
MAX_FORWARDS = 64
TOLERANCE = .015
POPULATION = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def target_gap(report, full):
    return max(
        *[abs(report["behavior_signed_projection"][task] - full["behavior_signed_projection"][task]) for task in base.TASKS],
        *[abs(report["response"][task]["signed_projection"] - full["response"][task]["signed_projection"]) for task in base.TASKS],
    )


def control_ok(report, full):
    left, right = report["control"], full["control"]
    return (
        set(left["flipped_row_ids"]).issubset(right["flipped_row_ids"])
        and left["median_kl"] <= right["median_kl"] + .002
        and set(left["margin_rms_fraction"]) == set(right["margin_rms_fraction"])
        and all(left["margin_rms_fraction"][task] <= right["margin_rms_fraction"][task] + .02 for task in right["margin_rms_fraction"])
    )


def capable(builder, capability, panel):
    return rowslib.capable_rows(builder, capability, panel)


def holdout_capture(backend, *_unused):
    tcap, icap = json.loads(TCAP.read_text()), json.loads(ICAP_ROWS.read_text())
    temporal_rows = sum((capable(temporal, tcap, panel) for panel in ("A1", "A2")), [])
    iswas_rows = sum((capable(iswas, icap, panel) for panel in ("A1", "A2")), [])
    temporal_all, iswas_all = temporal.build_rows(), iswas.build_rows()
    temporal_controls = [row for row in temporal_all if row["transform_id"] == "C"][:8]
    iswas_controls = [row for row in iswas_all if row["transform_id"] == "C"][:8]
    rows, controls = temporal_rows + iswas_rows, temporal_controls + iswas_controls
    POPULATION.update({
        "temporal_rows": len(temporal_rows), "iswas_rows": len(iswas_rows),
        "temporal_controls": len(temporal_controls), "iswas_controls": len(iswas_controls),
        "row_ids": [row["row_id"] for row in rows],
    })
    batch = das._batch(backend, rows, side="base")
    donor_batch = das._batch(backend, rows, side="donor")
    control_batch = das._batch(backend, controls, side="base")
    control_donor_batch = das._batch(backend, controls, side="donor")
    _base_output, base_full = base.atlas.capture_native(backend, batch)
    control_output, control_base_full = base.atlas.capture_native(backend, control_batch)
    return {
        "rows": rows, "controls": controls, "batch": batch, "donor_batch": donor_batch,
        "control_batch": control_batch, "control_donor_batch": control_donor_batch,
        "base_full": base_full, "control_base_full": control_base_full,
        "control_base_state": base.atlas.states(backend.torch, backend, control_output, controls),
    }


def main():
    candidate_id = "temporal_auxiliary.iswas_rank16_source_mask26_construction_holdout_v1"
    dry = {
        "candidate_id": candidate_id, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "masks": MASKS,
        "population": "temporal-v15/iswas-v15", "model_forwards_max": MAX_FORWARDS,
        "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(OUT)
    paths = {
        "prior": PRIOR, "composition_result": COMPOSITION,
        "composition_runner": ROOT / "ops/run_temporal_iswas_rank16_source_complement_composition_lattice_v1.py",
        "temporal_builder": ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v15.py",
        "temporal_capability": TCAP,
        "iswas_builder": ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v15.py",
        "iswas_capability": ICAP,
        "iswas_capability_rows": ICAP_ROWS,
        "weight_group_runner": ROOT / "ops/run_temporal_iswas_rank16_source_task_pair_weight_groups_v1.py",
        "old_population_runner": ROOT / "ops/run_temporal_iswas_rank16_source_task_pair_weight_groups_v2.py",
    }
    observed = {name: sha(path) for name, path in paths.items()}
    if observed != EXPECTED:
        raise RuntimeError(f"mask26 holdout authority changed: {observed}")
    composition = json.loads(COMPOSITION.read_text())
    tcap, icap = json.loads(TCAP.read_text()), json.loads(ICAP.read_text())
    if composition["terminal"] != "minimal_source_complement_program" or composition["selected_mask"] != 26:
        raise RuntimeError("composition selection changed")
    if tcap["terminal"] != "manifest" or icap["terminal"] != "manifest":
        raise RuntimeError("holdout capability manifest changed")
    old_tcap = json.loads(old_population.transfer.TCAP.read_text())
    old_icap = json.loads(old_population.transfer.ICAP.read_text())
    old_row_ids = {
        row_id for capability in (old_tcap, old_icap)
        for row_ids in capability["jointly_capable_row_ids"].values() for row_id in row_ids
    }

    base.PRIOR = PRIOR
    base.OUT = OUT
    base.MAX_FORWARDS = MAX_FORWARDS
    base.ARMS = EXECUTED_ARMS
    base.EXPECTED_CONTROL_TASKS = set(base.TASKS)
    base.EXPECTED = {
        "prior": EXPECTED["prior"], "weight_compiler_audit": sha(base.AUDIT),
        "mask30_holdout": sha(base.HOLDOUT), "mask30_control_attribution": sha(base.ATTRIBUTION),
        "task_mode_helper": sha(base.HELPER), "mathematical_review": sha(base.MATH_REVIEW),
    }
    base.oodctx.capture = holdout_capture
    original_block_for, original_run_blocks = base.block_for, base.run_blocks
    original_write = base.atomic_create_json
    cache = {}

    def block_for(blocks, site, arm, task):
        if arm.startswith("addback_mask_"):
            mask = int(arm.rsplit("_", 1)[1])
            bit = base.SUPPORT.index(site)
            return blocks[site]["full"] if mask & (1 << bit) else blocks[site]["union"]
        return original_block_for(blocks, site, arm, task)

    def run_blocks(backend, batch, rows, base_out, delta_hidden, bases, maps, blocks, arm, *, control=False):
        canonical = {"addback_mask_00": "mean_plus_contrast", "addback_mask_31": "full_rank16"}.get(arm, arm)
        key = (id(batch), canonical, control)
        if key not in cache:
            cache[key] = original_run_blocks(
                backend, batch, rows, base_out, delta_hidden, bases, maps, blocks, canonical, control=control,
            )
        return cache[key]

    def write_result(_path, result):
        reports = result["reports"]
        full, union = reports["full_rank16"], reports["mean_plus_contrast"]
        mask_reports = {mask: reports[f"addback_mask_{mask:02d}"] for mask in MASKS}
        replay = max(target_gap(mask_reports[0], union), target_gap(mask_reports[31], full))
        disjoint = not (set(POPULATION["row_ids"]) & old_row_ids)

        def eligible(mask):
            report = mask_reports[mask]
            return base.source.functional(report) and target_gap(report, full) <= TOLERANCE and control_ok(report, full)

        pred_a = bool(
            disjoint and replay <= 1e-6 and max(result["union_complement_closure_rse_by_site"].values()) <= 1e-6
            and finite(mask_reports) and result["price"]["model_forwards_observed"] <= MAX_FORWARDS
            and POPULATION["temporal_rows"] == 62 and POPULATION["iswas_rows"] == 32
            and POPULATION["temporal_controls"] == POPULATION["iswas_controls"] == 8
        )
        pred_b = base.source.functional(full)
        pred_c = eligible(26)
        pred_d = control_ok(mask_reports[26], full)
        pred_e = all(not eligible(mask) for mask in (24, 18, 10))
        predictions = {
            "pred_a_authority_disjointness_replay_closure_finiteness_and_price": bool(pred_a),
            "pred_b_full_parent_remains_functional": bool(pred_b),
            "pred_c_mask26_is_functional_and_full_equivalent": bool(pred_c),
            "pred_d_mask26_adds_no_parent_collateral": bool(pred_d),
            "pred_e_each_selected_site_is_jointly_necessary": bool(pred_e),
        }
        if not pred_a:
            terminal = "invalid"
        elif not pred_b:
            terminal = "parent_construction_failure"
        elif not pred_c or not pred_d:
            terminal = "construction_specific_source_mask"
        elif not pred_e:
            terminal = "transfers_but_not_minimal"
        else:
            terminal = "stable_minimal_source_complement_program"
        result.update({
            "schema": "temporal_iswas_rank16_source_mask26_construction_holdout_result_v1",
            "candidate_id": candidate_id, "authority_sha256": EXPECTED,
            "population": {key: value for key, value in POPULATION.items() if key != "row_ids"},
            "population_row_ids": POPULATION["row_ids"], "selection_population_disjoint": disjoint,
            "selected_mask": 26, "selected_sites": ["MLP1", "MLP3", "MLP6"],
            "leave_one_out_masks": [24, 18, 10], "mask_reports": mask_reports,
            "mask26_worst_target_gap": target_gap(mask_reports[26], full),
            "mask_replay_max_abs_error": replay, "predictions": predictions, "terminal": terminal,
        })
        original_write(OUT, result)

    base.block_for = block_for
    base.run_blocks = run_blocks
    base.atomic_create_json = write_result
    base.main()


if __name__ == "__main__":
    main()
