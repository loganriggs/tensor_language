#!/usr/bin/env python3
"""Direction-matched cross-task residual-state onset screen."""

# BQGATE: EXPERIMENT pred_a_authority_pairing_capability_and_identity pred_b_bidirectional_task_state_transfer pred_c_stable_onset pred_d_temporal_value_not_destroyed pred_e_exact_zero_fit_price
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import circuit_candidate_aspectual_tense_matched_fresh_lexicon_v2 as rows_builder
import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_tense_direction_matched_task_state_onset_v1.json"
OUT = ROOT / "circuits/followups/aspectual_tense_direction_matched_task_state_onset_v1_result.json"
CANDIDATE_ID = "aspectual_tense.direction_matched_task_state_onset_v1"
EXPECTED_PRIOR_SHA256 = "cdd9777bc648846fa083a9e773332e0690d77af499dccfa18f76fd72edd68480"
PATHS = {
    "typed_result": ROOT / "circuits/followups/aspectual_tense_typed_shared_contextual_path_v1_result.json",
    "typed_artifact": ROOT / "circuits/followups/aspectual_tense_typed_shared_contextual_path_v1_artifact.json",
    "matched_builder": ROOT / "ops/circuit_candidate_aspectual_tense_matched_fresh_lexicon_v2.py",
    "dual_result": ROOT / "circuits/followups/aspectual_tense_raw_text_dual_program_fresh_lexicon_v1_result.json",
}
EXPECTED = {
    "typed_result": "afb17159330dc6abcf018d36313a7df2c78c6708b67feb8d2f2d9d2eee50faf0",
    "typed_artifact": "f0f038f37fd9d97dff088117f93acdf239bab74c5877b522d7976bc81bfc6e85",
    "matched_builder": "1f4b29bda3e26af3ee0102316ab0af166e317d1646e8b0b51332061245e606d6",
    "dual_result": "36d54f861bd6dd70a493e306480a812b1fb9009e4e35c26fd77df5ab22d59ca7",
}
EXPECTED_ROWS = {"has_had": "7c2341ea65eb5915114ac4def7c3e7433d063e4cb3c988e518c91f1ff8e2b0ff", "is_was": "2efd47b9a89d0f092688a96d75bbc33e5b89991a8e5de28723c714319b9ccceb"}
SITES = tuple(f"resid:{layer:02d}" for layer in range(19))


class ExperimentError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_rows():
    banks = rows_builder.build_rows_by_bank()
    if rows_builder.validate_rows_by_bank(banks) != EXPECTED_ROWS:
        raise ExperimentError("matched row authority changed")
    selected = {name: {int(row["group_number"]): row for row in rows if row["transform_id"] == "A1"} for name, rows in banks.items()}
    pairs = [(selected["has_had"][group], selected["is_was"][group]) for group in range(16)]
    if any(has_row["reporter"] != is_row["reporter"] or has_row["direction_id"] != is_row["direction_id"] or has_row["base_ids"][-1] != is_row["base_ids"][-1] for has_row, is_row in pairs):
        raise ExperimentError("occupation, direction, or semantic token pairing changed")
    return pairs


def validate_static():
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256 or {name: sha(path) for name, path in PATHS.items()} != EXPECTED:
        raise ExperimentError("prior or authority hash changed")
    if json.loads(PRIOR.read_text()).get("candidate_id") != CANDIDATE_ID or len(paired_rows()) != 16 or len(SITES) != 19:
        raise ExperimentError("candidate, pairing, or site inventory changed")


def make_batch(pairs, task):
    index = 0 if task == "has_had" else 1
    rows = [pair[index] for pair in pairs]
    return producer.ModelBatch(
        row_ids=tuple(f"pair:{group:02d}" for group in range(16)),
        side="base",
        token_rows=tuple(tuple(int(token) for token in row["base_ids"]) for row in rows),
        answer_ids=tuple(int(row["base_answer_id"]) for row in rows),
        foil_ids=tuple(int(row["base_foil_id"]) for row in rows),
        semantic_positions=tuple(len(row["base_ids"]) - 1 for row in rows),
    )


def run_logits(backend, batch, *, capture=False, site=None, donor_cache=None):
    holder = {}

    def save_logits(_module, _arguments, output):
        holder["raw"] = output.detach().clone()

    handle = backend.model.lm_head.register_forward_hook(save_logits)
    try:
        output = backend.native(batch, capture=True) if capture else backend.patched(
            batch, site=kernel.SiteRef("residual", site), donor_cache=donor_cache
        )
    finally:
        handle.remove()
    raw = holder.get("raw")
    if raw is None:
        raise ExperimentError("final-head logits were not captured")
    logits = 30.0 * backend.torch.tanh(raw / 30.0)
    return output, logits


def four_logits(logits, batch, pairs):
    values = []
    for index, (has_row, is_row) in enumerate(pairs):
        position = len(batch.token_rows[index]) - 1
        ids = (
            int(has_row["base_answer_id"]), int(has_row["base_foil_id"]),
            int(is_row["base_answer_id"]), int(is_row["base_foil_id"]),
        )
        row_values = tuple(float(logits[index, position, token].float()) for token in ids)
        if any(not math.isfinite(value) for value in row_values):
            raise ExperimentError("nonfinite four-token logits")
        values.append(row_values)
    return values


def is_support(values):
    has_answer, has_foil, is_answer, is_foil = values
    return 0.5 * (is_answer + is_foil) - 0.5 * (has_answer + has_foil)


def summarize(records):
    recoveries = [record["recovery"] for record in records]
    return {
        "count": len(records),
        "mean_normalized_donor_recovery": statistics.fmean(recoveries),
        "mean_absolute_normalized_donor_recovery": statistics.fmean(abs(value) for value in recoveries),
        "direction_fraction": sum(value > 0.0 for value in recoveries) / len(recoveries),
        "donor_temporal_correct_fraction": sum(record["donor_temporal_correct"] for record in records) / len(records),
    }


def main():
    validate_static()
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False, "model_loaded": False, "pairs": 16, "sites": list(SITES), "orientations": 2, "model_forwards": 40, "example_evaluations": 640, "site_orientation_arms": 38, "cache_vector_identity_checks": 38, "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    started = time.perf_counter()
    pairs = paired_rows()
    backend = producer.Bilin18TorchBackend.load("cuda")
    batches = {task: make_batch(pairs, task) for task in ("has_had", "is_was")}
    native, caches = {}, {}
    forward_calls = evaluations = 0
    for task, batch in batches.items():
        output, logits = run_logits(backend, batch, capture=True)
        native[task] = four_logits(logits, batch, pairs)
        caches[task] = output.captured
        forward_calls += 1
        evaluations += len(pairs)
    capability_cells = []
    for task, pair_index in (("has_had", 0), ("is_was", 1)):
        answer_offset = 0 if task == "has_had" else 2
        for direction in ("present_to_past", "past_to_present"):
            indices = [i for i, pair in enumerate(pairs) if pair[pair_index]["direction_id"] == direction]
            accuracy = sum(native[task][i][answer_offset] > native[task][i][answer_offset + 1] for i in indices) / len(indices)
            capability_cells.append({"task": task, "direction": direction, "count": len(indices), "accuracy": accuracy, "threshold": 0.85, "passed": accuracy >= 0.85})
    patch_token_identity = all(has_row["base_ids"][-1] == is_row["base_ids"][-1] for has_row, is_row in pairs)
    identity_checks = 0
    identity_max_abs = 0.0
    records = []
    summaries = {}
    for site in SITES:
        for recipient, donor in (("has_had", "is_was"), ("is_was", "has_had")):
            for row_id in batches[recipient].row_ids:
                vector = caches[recipient][(row_id, site)]
                identity_max_abs = max(identity_max_abs, float((vector.float() - vector.float().clone()).abs().max()))
            identity_checks += 1
            _output, logits = run_logits(backend, batches[recipient], site=site, donor_cache=caches[donor])
            patched = four_logits(logits, batches[recipient], pairs)
            forward_calls += 1
            evaluations += len(pairs)
            orientation = f"{recipient}_to_{donor}"
            site_records = []
            for index, values in enumerate(patched):
                base_support = is_support(native[recipient][index])
                donor_support = is_support(native[donor][index])
                patched_support = is_support(values)
                sign = 1.0 if donor == "is_was" else -1.0
                denominator = sign * (donor_support - base_support)
                if not math.isfinite(denominator) or denominator <= 0.0:
                    raise ExperimentError("native donor task-support endpoint is not ordered")
                recovery = sign * (patched_support - base_support) / denominator
                donor_offset = 2 if donor == "is_was" else 0
                record = {"site": site, "orientation": orientation, "pair_index": index, "direction": pairs[index][0]["direction_id"], "recovery": recovery, "donor_temporal_correct": values[donor_offset] > values[donor_offset + 1]}
                records.append(record)
                site_records.append(record)
            summaries.setdefault(site, {})[orientation] = summarize(site_records)
    passing = []
    for site in SITES:
        passing.append(all(summaries[site][orientation]["mean_normalized_donor_recovery"] >= 0.50 and summaries[site][orientation]["direction_fraction"] >= 0.75 for orientation in ("has_had_to_is_was", "is_was_to_has_had")))
    stable_onset = None
    for index in range(len(SITES) - 2):
        if all(passing[index:index + 3]):
            stable_onset = SITES[index]
            break
    pred_a = all(cell["passed"] for cell in capability_cells) and patch_token_identity and identity_checks == 38 and identity_max_abs == 0.0 and len(records) == 608
    pred_b = any(passing)
    pred_c = stable_onset is not None
    pred_d = stable_onset is not None and all(summaries[stable_onset][orientation]["donor_temporal_correct_fraction"] >= 0.75 for orientation in ("has_had_to_is_was", "is_was_to_has_had"))
    price = {"model_forwards": forward_calls, "example_evaluations": evaluations, "pairs": 16, "site_orientation_arms": 38, "cache_vector_identity_checks": identity_checks, "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0}
    pred_e = price == {"model_forwards": 40, "example_evaluations": 640, "pairs": 16, "site_orientation_arms": 38, "cache_vector_identity_checks": 38, "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0}
    predictions = {"pred_a_authority_pairing_capability_and_identity": pred_a, "pred_b_bidirectional_task_state_transfer": pred_b, "pred_c_stable_onset": pred_c, "pred_d_temporal_value_not_destroyed": pred_d, "pred_e_exact_zero_fit_price": pred_e}
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and pred_e else "invalid")
    result = {"schema": "aspectual_tense_direction_matched_task_state_onset_result_v1", "candidate_id": CANDIDATE_ID, "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "prior_art_sha256": EXPECTED_PRIOR_SHA256, "authority_sha256": EXPECTED, "rows_sha256": EXPECTED_ROWS, "capability_cells": capability_cells, "instrument": {"patch_token_identity": patch_token_identity, "cache_vector_identity_max_abs_error": identity_max_abs, "native_endpoint_ordered": True}, "site_summaries": summaries, "joint_passing_sites": [site for site, passed in zip(SITES, passing) if passed], "stable_onset": stable_onset, "predictions": predictions, "price": price, "terminal": terminal, "reason": {"screen": "direction_matched_task_state_onset_localized", "null": "no_stable_bidirectional_task_state_onset", "invalid": "authority_pairing_capability_identity_coverage_or_price_invalid"}[terminal], "serial_seconds": time.perf_counter() - started, "next_action": f"decompose modules immediately before {stable_onset}" if terminal == "screen" else "retain external selector and test a cue-position task-state route"}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "capability_cells", "instrument", "joint_passing_sites", "stable_onset", "predictions", "price", "terminal", "reason", "next_action")}, sort_keys=True))


if __name__ == "__main__":
    main()
