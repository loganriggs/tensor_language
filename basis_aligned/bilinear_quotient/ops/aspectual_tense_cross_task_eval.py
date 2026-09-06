#!/usr/bin/env python3
"""Pure/shared contracts for direction-matched has/had versus is/was interventions."""

# BQGATE: LIBRARY
from __future__ import annotations

import math
import statistics

import circuit_candidate_aspectual_tense_matched_fresh_lexicon_v2 as rows_builder
import circuit_fast_screen_producer as producer


EXPECTED_ROWS = {"has_had": "7c2341ea65eb5915114ac4def7c3e7433d063e4cb3c988e518c91f1ff8e2b0ff", "is_was": "2efd47b9a89d0f092688a96d75bbc33e5b89991a8e5de28723c714319b9ccceb"}
TASKS = ("has_had", "is_was")
ORIENTATIONS = (("has_had", "is_was"), ("is_was", "has_had"))


class CrossTaskContractError(RuntimeError):
    pass


def paired_rows():
    banks = rows_builder.build_rows_by_bank()
    if rows_builder.validate_rows_by_bank(banks) != EXPECTED_ROWS:
        raise CrossTaskContractError("matched row authority changed")
    selected = {name: {int(row["group_number"]): row for row in rows if row["transform_id"] == "A1"} for name, rows in banks.items()}
    pairs = tuple((selected["has_had"][group], selected["is_was"][group]) for group in range(16))
    if any(has_row["reporter"] != is_row["reporter"] or has_row["direction_id"] != is_row["direction_id"] or has_row["base_ids"][-1] != is_row["base_ids"][-1] for has_row, is_row in pairs):
        raise CrossTaskContractError("occupation, direction, or semantic token pairing changed")
    return pairs


def make_batch(pairs, task):
    task_index = 0 if task == "has_had" else 1
    rows = [pair[task_index] for pair in pairs]
    return producer.ModelBatch(row_ids=tuple(f"pair:{group:02d}" for group in range(len(pairs))), side="base", token_rows=tuple(tuple(int(token) for token in row["base_ids"]) for row in rows), answer_ids=tuple(int(row["base_answer_id"]) for row in rows), foil_ids=tuple(int(row["base_foil_id"]) for row in rows), semantic_positions=tuple(len(row["base_ids"]) - 1 for row in rows))


def capture_softcapped_logits(backend, call):
    holder = {}
    def hook(_module, _arguments, output):
        holder["raw"] = output.detach().clone()
    handle = backend.model.lm_head.register_forward_hook(hook)
    try:
        result = call()
    finally:
        handle.remove()
    if "raw" not in holder:
        raise CrossTaskContractError("final-head logits were not captured")
    return result, 30.0 * backend.torch.tanh(holder["raw"] / 30.0)


def four_logits(logits, batch, pairs):
    values = []
    for index, (has_row, is_row) in enumerate(pairs):
        position = batch.semantic_positions[index]
        ids = (int(has_row["base_answer_id"]), int(has_row["base_foil_id"]), int(is_row["base_answer_id"]), int(is_row["base_foil_id"]))
        row = tuple(float(logits[index, position, token].float()) for token in ids)
        if any(not math.isfinite(value) for value in row):
            raise CrossTaskContractError("nonfinite four-token logits")
        values.append(row)
    return tuple(values)


def is_support(values):
    has_answer, has_foil, is_answer, is_foil = values
    return 0.5 * (is_answer + is_foil) - 0.5 * (has_answer + has_foil)


def intervention_record(native, patched, recipient, donor, index, **extra):
    base_support = is_support(native[recipient][index])
    donor_support = is_support(native[donor][index])
    patched_support = is_support(patched)
    sign = 1.0 if donor == "is_was" else -1.0
    denominator = sign * (donor_support - base_support)
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise CrossTaskContractError("native task-support endpoints are not ordered")
    offset = 2 if donor == "is_was" else 0
    return {**extra, "orientation": f"{recipient}_to_{donor}", "pair_index": index, "recovery": sign * (patched_support - base_support) / denominator, "donor_temporal_correct": patched[offset] > patched[offset + 1]}


def summarize(records):
    values = [float(record["recovery"]) for record in records]
    if not values or any(not math.isfinite(value) for value in values):
        raise CrossTaskContractError("missing or nonfinite recoveries")
    return {"count": len(values), "mean_normalized_donor_recovery": statistics.fmean(values), "mean_absolute_normalized_donor_recovery": statistics.fmean(abs(value) for value in values), "direction_fraction": sum(value > 0.0 for value in values) / len(values), "donor_temporal_correct_fraction": sum(bool(record["donor_temporal_correct"]) for record in records) / len(records)}


def capability_cells(native, pairs):
    cells = []
    for task, pair_index, offset in (("has_had", 0, 0), ("is_was", 1, 2)):
        for direction in ("present_to_past", "past_to_present"):
            indices = [index for index, pair in enumerate(pairs) if pair[pair_index]["direction_id"] == direction]
            accuracy = sum(native[task][index][offset] > native[task][index][offset + 1] for index in indices) / len(indices)
            cells.append({"task": task, "direction": direction, "count": len(indices), "accuracy": accuracy, "threshold": 0.85, "passed": accuracy >= 0.85})
    return cells


def verify_contract():
    native = {"has_had": ((3.0, 1.0, -1.0, -1.0),), "is_was": ((-1.0, -1.0, 3.0, 1.0),)}
    forward = intervention_record(native, native["is_was"][0], "has_had", "is_was", 0)
    reverse = intervention_record(native, native["has_had"][0], "is_was", "has_had", 0)
    if forward["recovery"] != 1.0 or reverse["recovery"] != 1.0 or summarize([forward])["direction_fraction"] != 1.0:
        raise CrossTaskContractError("endpoint normalization contract failed")
    return True


verify_contract()
