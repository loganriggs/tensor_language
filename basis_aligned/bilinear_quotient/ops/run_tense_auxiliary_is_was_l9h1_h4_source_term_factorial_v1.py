#!/usr/bin/env python3
"""Exact L9H1/H4 source-term factorial on aligned is/was rows."""

# BQGATE: EXPERIMENT pred_a_authority_capability_exact_instrument_and_coverage pred_b_full_h1_h4_pair_recurrence pred_c_literal_cue_is_not_sufficient pred_d_downstream_moment_determiner_context pred_e_self_negative_and_exact_price
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import aspectual_tense_dual_eval as evaluator
import circuit_candidate_aspectual_tense_matched_fresh_lexicon_v2 as rows_builder
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_aspectual_anchor_l9h1_h4_source_term_factorial_v1 as source_instrument


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/tense_auxiliary_is_was_l9h1_h4_source_term_factorial_v1.json"
OUT = ROOT / "circuits/followups/tense_auxiliary_is_was_l9h1_h4_source_term_factorial_v1_result.json"
CANDIDATE_ID = "tense_auxiliary.is_vs_was.l9h1_h4_source_term_factorial_v1"
PATHS = {
    "cross_task_head_reuse": ROOT / "circuits/followups/tense_auxiliary_is_was_l9h1_h4_cross_task_reader_reuse_v1_result.json",
    "source_instrument": ROOT / "ops/run_aspectual_anchor_l9h1_h4_source_term_factorial_v1.py",
    "matched_v2_builder": ROOT / "ops/circuit_candidate_aspectual_tense_matched_fresh_lexicon_v2.py",
}
EXPECTED_PRIOR_SHA256 = "529ad19556b47fb24da73886d2dfeb6c7d7fd6bdff1a0f1347b445f5078ee685"
EXPECTED = {
    "cross_task_head_reuse": "ca3139b6eba33f3d06c6d79c5b772f8ecf568e16918d2e2211c282847d577070",
    "source_instrument": "8e890efd3520cfbece1d71f3ffb58397c732d8fc9c9446c74af9ac0380f2ca01",
    "matched_v2_builder": "1f4b29bda3e26af3ee0102316ab0af166e317d1646e8b0b51332061245e606d6",
}
EXPECTED_ROWS_SHA256 = "2efd47b9a89d0f092688a96d75bbc33e5b89991a8e5de28723c714319b9ccceb"
ARMS = {"full_pair": "full_pair", "cue": "cue_joint", "moment": "last_joint", "determiner": "period_joint", "self": "self_joint"}


class ExperimentError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair_error(first, second):
    return max(abs(float(a) - float(b)) for left, right in zip(first.answer_foil, second.answer_foil) for a, b in zip(left, right))


def recovery_records(rows, base_output, donor_output, patched_output):
    records = []
    for row, base_pair, donor_pair, patched_pair in zip(rows, base_output.answer_foil, donor_output.answer_foil, patched_output.answer_foil):
        base_native = float(base_pair[0]) - float(base_pair[1])
        donor_target = float(donor_pair[0]) - float(donor_pair[1])
        patched_target = -(float(patched_pair[0]) - float(patched_pair[1]))
        base_target = -base_native
        denominator = donor_target - base_target
        if denominator == 0.0 or not all(math.isfinite(value) for value in (base_native, donor_target, patched_target)):
            raise ExperimentError("zero or nonfinite recovery input")
        records.append({"family": row["family"], "direction": evaluator.direction_for(row), "row_id": str(row["row_id"]), "recovery": (patched_target - base_target) / denominator})
    return records


def prediction_record(a, b, c, d, e):
    return {
        "pred_a_authority_capability_exact_instrument_and_coverage": a,
        "pred_b_full_h1_h4_pair_recurrence": b,
        "pred_c_literal_cue_is_not_sufficient": c,
        "pred_d_downstream_moment_determiner_context": d,
        "pred_e_self_negative_and_exact_price": e,
    }


def validate_static():
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256 or {name: sha(path) for name, path in PATHS.items()} != EXPECTED:
        raise ExperimentError("prior or authority hash changed")
    prior = json.loads(PRIOR.read_text())
    parent = json.loads(PATHS["cross_task_head_reuse"].read_text())
    rows_by_bank = rows_builder.build_rows_by_bank()
    rows_sha = rows_builder.validate_rows_by_bank(rows_by_bank)
    rows = [row for row in rows_by_bank["is_was"] if row["family"] in ("A1", "A2")]
    expected_authorities = {"cross_task_head_reuse_sha256": EXPECTED["cross_task_head_reuse"], "source_instrument_sha256": EXPECTED["source_instrument"], "matched_v2_builder_sha256": EXPECTED["matched_v2_builder"], "q_is_v8_rows_sha256": EXPECTED_ROWS_SHA256}
    aligned = all(len(row["base_ids"]) == len(row["donor_ids"]) and sum(base != donor for base, donor in zip(row["base_ids"], row["donor_ids"])) == 1 for row in rows)
    ok = prior.get("candidate_id") == CANDIDATE_ID and prior.get("authorities") == expected_authorities and parent.get("terminal") == "null" and parent["predictions"]["pred_c_q_has_h1_h4_cross_task_reader_reuse"] is True and rows_sha["is_was"] == EXPECTED_ROWS_SHA256 and len(rows) == 32 and aligned and evaluator.verify_contract()
    if not ok:
        raise ExperimentError("candidate, parent reuse, rows, alignment, or shared contract changed")
    return rows


def main():
    rows = validate_static()
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False, "model_loaded": False, "rows": 32, "causal_arms": 5, "model_forwards": 18, "example_evaluations": 288, "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    started = time.perf_counter()
    backend = source_instrument.SourceBackend.load("cuda")
    family_data = {}
    capability_cells = []
    manual_logit_error, reconstruction_error = 0.0, 0.0
    for family in ("A1", "A2"):
        family_rows = [row for row in rows if row["family"] == family]
        base_batch, donor_batch = das._batch(backend, family_rows, side="base"), das._batch(backend, family_rows, side="donor")
        references, captures = {}, {}
        for side, batch in (("base", base_batch), ("donor", donor_batch)):
            reference = backend.native(batch, capture=False)
            manual, capture = backend.manual_forward(batch)
            references[side] = reference
            captures[side] = capture
            manual_logit_error = max(manual_logit_error, pair_error(reference, manual))
            reconstruction_error = max(reconstruction_error, float(capture["reconstruction_max_abs"]))
            for direction in evaluator.DIRECTIONS:
                indices = [index for index, row in enumerate(family_rows) if evaluator.direction_for(row) == direction]
                accuracy = sum(float(reference.answer_foil[index][0]) > float(reference.answer_foil[index][1]) for index in indices) / len(indices)
                capability_cells.append({"family": family, "direction": direction, "side": side, "count": len(indices), "accuracy": accuracy, "threshold": 0.85, "passed": accuracy >= 0.85})
        family_data[family] = {"rows": family_rows, "base_batch": base_batch, "donor_batch": donor_batch, "references": references, "donor_capture": captures["donor"]}
    capability_pass = all(cell["passed"] for cell in capability_cells)

    arm_records = {name: [] for name in ARMS}
    for arm_name, instrument_arm in ARMS.items():
        for family in ("A1", "A2"):
            item = family_data[family]
            output, _ = backend.manual_forward(item["base_batch"], donor_batch=item["donor_batch"], donor_capture=item["donor_capture"], arm=instrument_arm)
            arm_records[arm_name].extend(recovery_records(item["rows"], item["references"]["base"], item["references"]["donor"], output))
    summaries = {}
    for arm_name, records in arm_records.items():
        summaries[arm_name] = {family: evaluator.metric_summary([record for record in records if record["family"] == family], "recovery") for family in ("A1", "A2")}
    full = summaries["full_pair"]
    cue_fraction = {family: abs(summaries["cue"][family]["mean_recovery"]) / full[family]["mean_recovery"] for family in ("A1", "A2")}
    self_fraction = {family: summaries["self"][family]["mean_absolute_recovery"] / full[family]["mean_recovery"] for family in ("A1", "A2")}
    downstream = {}
    for family in ("A1", "A2"):
        moment, determiner = summaries["moment"][family], summaries["determiner"][family]
        candidates = (("moment", moment), ("determiner", determiner))
        best_name, best = max(candidates, key=lambda item: item[1]["mean_recovery"])
        downstream[family] = {"summed_mean_fraction_of_full": (moment["mean_recovery"] + determiner["mean_recovery"]) / full[family]["mean_recovery"], "best_source": best_name, "best_mean_fraction_of_full": best["mean_recovery"] / full[family]["mean_recovery"], "best_direction_fraction": best["direction_fraction"]}
    pred_a = capability_pass and manual_logit_error <= 1e-4 and reconstruction_error <= 1e-4 and all(len(records) == 32 for records in arm_records.values())
    pred_b = all(full[family]["mean_recovery"] >= 0.30 and full[family]["direction_fraction"] >= 0.80 for family in ("A1", "A2"))
    pred_c = all(cue_fraction[family] <= 0.25 for family in ("A1", "A2"))
    pred_d = all(downstream[family]["summed_mean_fraction_of_full"] >= 0.50 and downstream[family]["best_mean_fraction_of_full"] >= 0.25 and downstream[family]["best_direction_fraction"] >= 0.80 for family in ("A1", "A2"))
    pred_e = all(self_fraction[family] <= 0.25 for family in ("A1", "A2"))
    predictions = prediction_record(pred_a, pred_b, pred_c, pred_d, pred_e)
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and pred_b else "invalid")
    reason = {"screen": "shared_contextualized_downstream_source_reader_architecture", "null": "literal_cue_downstream_context_or_self_source_prediction_misses", "invalid": "authority_alignment_capability_instrument_full_pair_coverage_finiteness_or_price_invalid"}[terminal]
    result = {
        "schema": "tense_auxiliary_is_was_l9h1_h4_source_term_factorial_result_v1", "candidate_id": CANDIDATE_ID,
        "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "authority_sha256": EXPECTED, "rows_sha256": EXPECTED_ROWS_SHA256, "capability_cells": capability_cells,
        "instrument": {"manual_selected_logit_max_abs_error": manual_logit_error, "selected_head_source_reconstruction_max_abs_error": reconstruction_error},
        "source_labels": {"cue": "this/that", "moment": "cue+1", "determiner": "cue+2", "self": "final occupation"},
        "summaries": summaries, "cue_fraction_of_full": cue_fraction, "self_absolute_fraction_of_full": self_fraction, "downstream_context": downstream,
        "arm_records": arm_records, "predictions": predictions,
        "price": {"model_forwards": 18, "example_evaluations": 288, "rows": 32, "causal_arms": 5, "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0},
        "terminal": terminal, "reason": reason, "serial_seconds": time.perf_counter() - started,
        "scope_boundary": "P insertion is not source aligned and remains unresolved.",
        "next_action": "construct an alignment-preserving P source test inside the same fixed heads" if terminal == "screen" else "retain only head-level reuse and do not alter the source inventory post hoc",
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "capability_cells", "instrument", "summaries", "cue_fraction_of_full", "self_absolute_fraction_of_full", "downstream_context", "predictions", "price", "terminal", "reason", "scope_boundary", "next_action")}, sort_keys=True))


if __name__ == "__main__":
    main()
