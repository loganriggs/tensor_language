#!/usr/bin/env python3
# BQGATE: five frozen science predictions are emitted by this targeted managed runner.
"""Run held-out literal removal of Task14 head 11.3 and attention 11."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Mapping, Protocol, Sequence

import circuit_fast_screen_candidate_task14_select_head11_3_zero_removal as candidate
import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_ledger as ledger
import circuit_fast_screen_managed_runner as managed
import circuit_fast_screen_producer as producer


ROOT = Path(__file__).resolve().parent.parent
REQUEST_ID = "task14-subject-verb-agreement-select-head11-3-zero-removal-v1"
EXPERIMENT_ID = "fast-screen-task14-select-head11-3-zero-removal-v1"
RESULT_RELATIVE = Path(
    "circuits/fast_screens/task14_subject_verb_agreement_select_head11_3_zero_removal_v1_result.json"
)
RESULT = ROOT / RESULT_RELATIVE
LEDGER = ROOT / "circuits/fast_screen_ledger.jsonl"
PRIOR_ART_SHA256 = "e889eae18a5d68e759f2a80592e4655a185456936f3c126103fd9b276f703aae"
EXPECTED_AUTHORITY_SHA256 = "bee6689a2b09f13417ddfc3d0ff0102834f415b51c9fb9679f17dee9cb0c3565"
CHECKPOINT_SHA256 = "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
CONFIG_SHA256 = "428042bfd807ba36f8b4326395440fbbebe52cd3d040212e6fef14a4fdf2d83c"

REGISTERED_PREDICTIONS = (
    "pred_a_native_capability",
    "pred_b_native_head_replay",
    "pred_c_attention11_removal_is_live",
    "pred_d_head11_3_is_individually_necessary",
    "pred_e_head11_3_removal_is_selective_within_task14",
)


class RemovalBackend(Protocol):
    def native(self, batch: producer.ModelBatch, *, capture: bool) -> producer.BatchOutput: ...
    def patched(self, batch: producer.ModelBatch, *, site: kernel.SiteRef,
                donor_cache: Mapping[tuple[str, str], object]) -> producer.BatchOutput: ...


class RemovalRunError(ValueError):
    pass


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_text(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _verify_checkpoint() -> None:
    import fastload
    _config, blob, source = fastload._paths()
    if hashlib.sha256((source.SNAP / "config.json").read_bytes()).hexdigest() != CONFIG_SHA256:
        raise RemovalRunError("checkpoint config hash changed")
    with open(blob, "rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != CHECKPOINT_SHA256:
            raise RemovalRunError("checkpoint weights hash changed")


def _chunks(rows: Sequence[Mapping[str, object]]) -> list[list[Mapping[str, object]]]:
    return [list(rows[start:start + candidate.BATCH_SIZE])
            for start in range(0, len(rows), candidate.BATCH_SIZE)]


def _batch(rows: Sequence[Mapping[str, object]]) -> producer.ModelBatch:
    return producer.ModelBatch(
        row_ids=tuple(str(row["row_id"]) for row in rows), side="base",
        token_rows=tuple(tuple(int(token) for token in row["ids"]) for row in rows),
        answer_ids=tuple(int(row["answer_id"]) for row in rows),
        foil_ids=tuple(int(row["foil_id"]) for row in rows),
        semantic_positions=tuple(int(row["semantic_position"]) for row in rows),
    )


def _pair(value: object) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2 or any(
        type(item) not in {int, float} or not math.isfinite(float(item)) for item in value
    ):
        raise RemovalRunError("backend returned a malformed logit pair")
    return float(value[0]), float(value[1])


def _margin(pair: tuple[float, float]) -> float:
    return pair[0] - pair[1]


def _zero_cache(
    captured: Mapping[tuple[str, str], object], rows: Sequence[Mapping[str, object]], site_id: str,
) -> dict[tuple[str, str], object]:
    output = {}
    for row in rows:
        key = (str(row["row_id"]), site_id)
        value = captured.get(key)
        if value is None or not hasattr(value, "mul"):
            raise RemovalRunError(f"native capture lacks tensor {key}")
        output[key] = value.mul(0)
    return output


def _capability(rows, native) -> list[dict[str, object]]:
    grouped: dict[str, list[bool]] = {}
    family_by_cell = {}
    for row in rows:
        cell = str(row["cell_id"])
        family = str(row["family"])
        grouped.setdefault(cell, []).append(_margin(native[str(row["row_id"])]) > 0)
        family_by_cell[cell] = family
    output = []
    for cell, values in sorted(grouped.items()):
        family = family_by_cell[cell]
        accuracy = sum(values) / len(values)
        minimum = candidate.MIN_NATIVE_ACCURACY[family]
        output.append({"cell_id": cell, "family": family, "row_count": len(values),
                       "accuracy": accuracy, "minimum_accuracy": minimum,
                       "passed": accuracy >= minimum})
    return output


def _score(rows, native, zero_head, zero_attention, replay) -> dict[str, object]:
    target_margins = [abs(_margin(native[str(row["row_id"])]))
                      for row in rows if row["family"] in {"A1", "A2"}]
    target_scale = statistics.median(target_margins)
    if target_scale <= candidate.MIN_TARGET_SCALE:
        raise RemovalRunError("native target scale is too small")
    evidence = []
    grouped: dict[tuple[str, str], list[dict[str, float]]] = {}
    replay_error = 0.0
    for row in rows:
        row_id = str(row["row_id"])
        native_pair, replay_pair = native[row_id], replay[row_id]
        replay_error = max(replay_error, *(abs(a - b) for a, b in zip(native_pair, replay_pair)))
        record = {
            "head_damage": (_margin(native_pair) - _margin(zero_head[row_id])) / target_scale,
            "attention_damage": (_margin(native_pair) - _margin(zero_attention[row_id])) / target_scale,
        }
        evidence.append({"row_id": row_id, "family": row["family"],
                         "cell_id": row["cell_id"], **record})
        grouped.setdefault((str(row["family"]), str(row["cell_id"])), []).append(record)
    cells = []
    for (family, cell_id), records in sorted(grouped.items()):
        head = [item["head_damage"] for item in records]
        attention = [item["attention_damage"] for item in records]
        cells.append({
            "family": family, "cell_id": cell_id, "row_count": len(records),
            "mean_head_damage": statistics.fmean(head),
            "head_damage_direction_fraction": sum(value > 0 for value in head) / len(head),
            "mean_absolute_head_damage": statistics.fmean(abs(value) for value in head),
            "mean_attention_damage": statistics.fmean(attention),
            "attention_damage_direction_fraction": sum(value > 0 for value in attention) / len(attention),
        })
    target_cells = [cell for cell in cells if cell["family"] in {"A1", "A2"}]
    full_live = all(
        cell["mean_attention_damage"] >= candidate.MIN_TARGET_DAMAGE
        and cell["attention_damage_direction_fraction"] >= candidate.MIN_TARGET_DIRECTION
        for cell in target_cells
    )
    head_necessary = all(
        cell["mean_head_damage"] >= candidate.MIN_TARGET_DAMAGE
        and cell["head_damage_direction_fraction"] >= candidate.MIN_TARGET_DIRECTION
        for cell in target_cells
    )
    controls = {
        family: statistics.fmean(
            cell["mean_absolute_head_damage"] for cell in cells if cell["family"] == family
        ) for family in ("P", "C")
    }
    selective = (controls["P"] <= candidate.MAX_P_ABSOLUTE_DAMAGE
                 and controls["C"] <= candidate.MAX_C_ABSOLUTE_DAMAGE)
    return {"target_scale": target_scale, "replay_max_abs_logit_error": replay_error,
            "cells": cells, "control_mean_absolute_head_damage": controls,
            "full_attention_removal_live": full_live,
            "head11_3_individually_necessary": head_necessary,
            "head11_3_within_task_selective": selective, "evidence": evidence}


def run_science(
    *, backend: RemovalBackend | None = None, device: str = "cuda",
    wall_clock=_utc_now, monotonic_clock=time.perf_counter,
) -> dict[str, object]:
    rows = candidate.build_rows()
    authority_sha = candidate.validate_rows(rows)
    if authority_sha != EXPECTED_AUTHORITY_SHA256:
        raise RemovalRunError("reviewed authority hash changed")
    plan = candidate.compile_plan(rows)
    if backend is None:
        _verify_checkpoint()
    executor = backend if backend is not None else producer.Bilin18TorchBackend.load(device)
    started_utc, started = wall_clock(), monotonic_clock()
    native, zero_head, zero_attention, replay = {}, {}, {}, {}
    captured = {}
    forwards = evaluations = 0
    chunks = _chunks(rows)
    for chunk in chunks:
        output = executor.native(_batch(chunk), capture=True)
        forwards += 1; evaluations += len(chunk); captured.update(output.captured)
        native.update({row_id: _pair(pair) for row_id, pair in zip(_batch(chunk).row_ids, output.answer_foil)})
    head_zero_cache = _zero_cache(captured, rows, candidate.HEAD_SITE_ID)
    attention_zero_cache = _zero_cache(captured, rows, candidate.ATTENTION_SITE_ID)
    conditions = (
        (candidate.HEAD_SITE_ID, head_zero_cache, zero_head),
        (candidate.ATTENTION_SITE_ID, attention_zero_cache, zero_attention),
        (candidate.HEAD_SITE_ID, captured, replay),
    )
    for site_id, cache, destination in conditions:
        kind = "head" if ":head:" in site_id else "module"
        site = kernel.SiteRef(kind, site_id)  # type: ignore[arg-type]
        for chunk in chunks:
            batch = _batch(chunk)
            output = executor.patched(batch, site=site, donor_cache=cache)
            forwards += 1; evaluations += len(chunk)
            destination.update({row_id: _pair(pair)
                                for row_id, pair in zip(batch.row_ids, output.answer_foil)})
    capability = _capability(rows, native)
    scores = _score(rows, native, zero_head, zero_attention, replay)
    capability_passed = all(cell["passed"] for cell in capability)
    replay_passed = scores["replay_max_abs_logit_error"] <= candidate.MAX_REPLAY_LOGIT_ERROR
    predictions = {
        REGISTERED_PREDICTIONS[0]: capability_passed,
        REGISTERED_PREDICTIONS[1]: replay_passed,
        REGISTERED_PREDICTIONS[2]: scores["full_attention_removal_live"],
        REGISTERED_PREDICTIONS[3]: scores["head11_3_individually_necessary"],
        REGISTERED_PREDICTIONS[4]: scores["head11_3_within_task_selective"],
    }
    if not replay_passed:
        terminal, reason = "invalid", "native_head_replay_failed"
    elif not capability_passed:
        terminal, reason = "null", "native_select_endpoints_incapable"
    elif not scores["full_attention_removal_live"]:
        terminal, reason = "inconclusive", "whole_attention_removal_positive_control_failed"
    elif not scores["head11_3_individually_necessary"]:
        terminal, reason = "null", "head11_3_not_individually_necessary"
    elif not scores["head11_3_within_task_selective"]:
        terminal, reason = "null", "head11_3_zero_removal_not_selective_within_task14"
    else:
        terminal, reason = "screen", "head11_3_selective_necessity_passed"
    finished, finished_utc = monotonic_clock(), wall_clock()
    return {
        "schema": "task14_select_head11_3_zero_removal_result_v1",
        "request_id": REQUEST_ID, "experiment_id": EXPERIMENT_ID,
        "candidate_id": candidate.TASK_ID, "screen_tier_only": True,
        "execution_policy": "managed_queue_only", "create_only": True,
        "phase": candidate.PHASE, "partition": candidate.PARTITION,
        "authority_sha256": authority_sha, "plan_sha256": plan["compiled_sha256"],
        "source_sha256": candidate.EXPECTED_SOURCE_SHA256,
        "checkpoint": {"weights_sha256": CHECKPOINT_SHA256,
                       "config_sha256": CONFIG_SHA256,
                       "verified_before_model_load": backend is None},
        "started_utc": _utc_text(started_utc), "finished_utc": _utc_text(finished_utc),
        "serial_seconds": finished - started, "terminal": terminal, "reason": reason,
        "predictions": predictions, "bars": plan["bars"], "capability_cells": capability,
        **scores,
        "active_price": {"forward_calls": forwards, "example_evaluations": evaluations,
                         "backward_calls": 0, "model_updates": 0,
                         "raw_numeric_evidence_bytes": evaluations * 8},
        "maximum_price": plan["price"],
        "limits": (
            "P/C are within-Task14 answer-preserving controls. This does not yet measure "
            "collateral effects on unrelated registered behaviors."
        ),
    }


def _publish(result: Mapping[str, object]) -> dict[str, object]:
    payload = managed.atomic_create_json(RESULT, result)
    result_sha = hashlib.sha256(payload).hexdigest()
    terminal = str(result["terminal"])
    entry = {
        "request_id": REQUEST_ID, "candidate_id": candidate.TASK_ID,
        "started_utc": result["started_utc"], "finished_utc": result["finished_utc"],
        "serial_seconds": result["serial_seconds"], "prior_art_sha256": PRIOR_ART_SHA256,
        "spec_sha256": result["plan_sha256"], "authority_sha256": result["authority_sha256"],
        "result_path": RESULT_RELATIVE.as_posix(), "result_sha256": result_sha,
        "terminal": terminal,
        "reasons": [] if terminal == "screen" else [str(result["reason"])],
        "selected_site_id": candidate.HEAD_SITE_ID if terminal == "screen" else None,
        "active_forward_calls": result["active_price"]["forward_calls"],
        "active_example_evaluations": result["active_price"]["example_evaluations"],
        "active_evidence_bytes": result["active_price"]["raw_numeric_evidence_bytes"],
        "max_forward_calls": result["maximum_price"]["forward_calls"],
        "max_example_evaluations": result["maximum_price"]["example_evaluations"],
        "max_evidence_bytes": result["maximum_price"]["raw_numeric_evidence_bytes"],
        "relation": "extension",
        "novelty": (
            "Held-out literal head-11.3 zero removal with whole-attention positive control, "
            "native replay, and P/C within-task collateral measurements."
        ),
    }
    ledger.append_entry(LEDGER, entry, result_root=ROOT)
    return {"terminal": terminal, "reason": result["reason"],
            "result_path": RESULT_RELATIVE.as_posix(), "result_sha256": result_sha,
            "active_price": result["active_price"]}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    flags = {name: os.environ.get(name) for name in ("BQLIB_DRYRUN", "BQLIB_NO_MODEL")}
    if any(value not in {None, "1"} for value in flags.values()):
        raise RemovalRunError("dry-run environment flags must be absent or exactly 1")
    if args.dry_run or "1" in flags.values():
        print(json.dumps(candidate.compile_plan(), sort_keys=True)); return
    print(json.dumps(_publish(run_science()), sort_keys=True))


if __name__ == "__main__":
    main()
