#!/usr/bin/env python3
# BQGATE: three frozen science predictions are emitted by this targeted runner.
"""Measure the one missing Task-14 attention-11 head/complement factorial arm.

The immutable v2 screen already contains native, head-11.3-only, and whole
attention-11 logits.  This runner performs only donor capture plus recipient
patching of the other eight pre-c_proj head slices.  It fits nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Mapping, Protocol, Sequence

import circuit_fast_screen_candidate_task14_agreement as candidate
import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_managed_runner as managed
import circuit_fast_screen_producer as producer


ROOT = Path(__file__).resolve().parent.parent
V2_RESULT = ROOT / "circuits/fast_screens/task14_subject_verb_agreement_full_state_v2_result.json"
RESULT = ROOT / "circuits/followups/task14_attention11_head_complement_factorial_v1_result.json"
EXPECTED_V2_SHA256 = "3c87e3973e1a7627f504ce26dfdaa3d7c48536f27a522e36c9e85741f09555c1"
EXPECTED_AUTHORITY_SHA256 = "9b8ede7d17b0358467438b7f8fda7703bba1c93c9c594d55454404c1bb6e21cc"
LAYER = 11
HEAD = 3
COMPLEMENT_HEADS = tuple(head for head in range(9) if head != HEAD)
BATCH_SIZE = 32
MAX_ABS_COMPLEMENT_TARGET = 0.10
MAX_ABS_INTERACTION_TARGET = 0.10
MAX_CONTROL_MOVEMENT = 0.10
DISTRIBUTED_COMPLEMENT_TARGET = 0.25
DISTRIBUTED_INTERACTION_TARGET = 0.20


class ComplementBackend(Protocol):
    def native(self, batch: producer.ModelBatch, *, capture: bool) -> producer.BatchOutput: ...

    def patched_heads(
        self,
        batch: producer.ModelBatch,
        *,
        layer: int,
        heads: Sequence[int],
        donor_cache: Mapping[tuple[str, str], object],
    ) -> producer.BatchOutput: ...


class ComplementScreenError(ValueError):
    """The frozen closure or new evidence is inconsistent."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pair(value: object) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2 or any(
        type(item) not in {int, float} or not math.isfinite(float(item)) for item in value
    ):
        raise ComplementScreenError("backend returned a malformed logit pair")
    return float(value[0]), float(value[1])


def _margin(record: Mapping[str, object]) -> float:
    return float(record["answer_logit"]) - float(record["foil_logit"])


def _load_closure() -> tuple[list[dict[str, object]], dict[str, object]]:
    if _sha256(V2_RESULT) != EXPECTED_V2_SHA256:
        raise ComplementScreenError("immutable v2 evidence hash changed")
    rows = candidate.build_rows()
    if candidate.validate_rows(rows) != EXPECTED_AUTHORITY_SHA256:
        raise ComplementScreenError("immutable Task 14 authority changed")
    old = json.loads(V2_RESULT.read_text())
    if old.get("authority_sha256") != EXPECTED_AUTHORITY_SHA256 or old.get("terminal") != "screen":
        raise ComplementScreenError("v2 evidence does not bind the expected passing authority")
    row_ids = {str(row["row_id"]) for row in rows}
    native = old.get("run", {}).get("native_logits", [])
    interventions = old.get("run", {}).get("intervention_logits", [])
    if len(native) != 2 * len(rows):
        raise ComplementScreenError("v2 native evidence lost exact coverage")
    for site_id in ("attn:11", "attn:11:head:03"):
        observed = [
            item for item in interventions
            if item.get("site", {}).get("site_id") == site_id
        ]
        if len(observed) != len(rows) or {str(item["row_id"]) for item in observed} != row_ids:
            raise ComplementScreenError(f"v2 evidence lost exact {site_id} coverage")
    return rows, old


def compile_dryrun() -> dict[str, object]:
    rows, _old = _load_closure()
    calls = 2 * math.ceil(len(rows) / BATCH_SIZE)
    return {
        "schema": "task14_attention11_head_complement_factorial_dryrun_v1",
        "model_loaded": False,
        "gpu_accessed": False,
        "queue_touched": False,
        "authority_sha256": EXPECTED_AUTHORITY_SHA256,
        "v2_result_sha256": EXPECTED_V2_SHA256,
        "layer": LAYER,
        "head": HEAD,
        "complement_heads": list(COMPLEMENT_HEADS),
        "only_new_arm": "joint donor replacement of attention-11 heads other than head 3",
        "maximum_price": {
            "forward_calls": calls,
            "example_evaluations": 2 * len(rows),
            "backward_calls": 0,
            "model_updates": 0,
            "raw_numeric_evidence_bytes": 2 * len(rows) * 4,
        },
        "bars": {
            "clean_split_max_abs_target_cell_complement": MAX_ABS_COMPLEMENT_TARGET,
            "clean_split_max_abs_target_cell_interaction": MAX_ABS_INTERACTION_TARGET,
            "clean_split_max_control_movement": MAX_CONTROL_MOVEMENT,
            "distributed_min_abs_target_cell_complement": DISTRIBUTED_COMPLEMENT_TARGET,
            "distributed_min_abs_target_cell_interaction": DISTRIBUTED_INTERACTION_TARGET,
        },
    }


def _batch(rows: Sequence[Mapping[str, object]], side: str) -> producer.ModelBatch:
    return producer.ModelBatch(
        row_ids=tuple(str(row["row_id"]) for row in rows),
        side=side,  # type: ignore[arg-type]
        token_rows=tuple(tuple(int(x) for x in row[f"{side}_ids"]) for row in rows),
        answer_ids=tuple(int(row[f"{side}_answer_id"]) for row in rows),
        foil_ids=tuple(int(row[f"{side}_foil_id"]) for row in rows),
        semantic_positions=tuple(int(row[f"{side}_semantic_position"]) for row in rows),
    )


def _chunks(rows: Sequence[dict[str, object]]) -> list[list[dict[str, object]]]:
    return [list(rows[start:start + BATCH_SIZE]) for start in range(0, len(rows), BATCH_SIZE)]


def run_science(
    *, backend: ComplementBackend | None = None, device: str = "cuda",
    clock=time.perf_counter,
) -> dict[str, object]:
    rows, old = _load_closure()
    executor = backend if backend is not None else producer.Bilin18TorchBackend.load(device)
    donor_cache: dict[tuple[str, str], object] = {}
    patched: dict[str, tuple[float, float]] = {}
    forwards = evaluations = 0
    started = clock()
    for chunk in _chunks(rows):
        output = executor.native(_batch(chunk, "donor"), capture=True)
        forwards += 1
        evaluations += len(chunk)
        donor_cache.update(output.captured)
    required = {
        (str(row["row_id"]), f"attn:{LAYER:02d}:head:{head:02d}")
        for row in rows for head in COMPLEMENT_HEADS
    }
    if not required.issubset(donor_cache):
        raise ComplementScreenError("donor capture lacks a complement head slice")
    for chunk in _chunks(rows):
        batch = _batch(chunk, "base")
        output = executor.patched_heads(
            batch, layer=LAYER, heads=COMPLEMENT_HEADS, donor_cache=donor_cache
        )
        forwards += 1
        evaluations += len(chunk)
        if len(output.answer_foil) != len(chunk):
            raise ComplementScreenError("patched output count differs from its batch")
        patched.update({row_id: _pair(pair) for row_id, pair in zip(batch.row_ids, output.answer_foil)})

    native = {
        (str(item["row_id"]), str(item["side"])): item
        for item in old["run"]["native_logits"]
    }
    prior = {
        (str(item["row_id"]), str(item["site"]["site_id"])): item
        for item in old["run"]["intervention_logits"]
        if item["site"]["site_id"] in {"attn:11", "attn:11:head:03"}
    }
    denominators = [
        _margin(native[(str(row["row_id"]), "donor")])
        + _margin(native[(str(row["row_id"]), "base")])
        for row in rows if row["transform_id"] in {"A1", "A2"}
    ]
    target_scale = statistics.median(denominators)
    if target_scale <= kernel.MIN_DONOR_DENOMINATOR:
        raise ComplementScreenError("frozen target scale is invalid")

    evidence = []
    by_cell: dict[str, list[dict[str, float]]] = {}
    control_abs: dict[str, list[float]] = {"P": [], "C": []}
    for row in rows:
        row_id = str(row["row_id"])
        family = str(row["transform_id"])
        base_margin = _margin(native[(row_id, "base")])
        donor_margin = _margin(native[(row_id, "donor")])
        margins = {
            "empty": base_margin,
            "head": _margin(prior[(row_id, "attn:11:head:03")]),
            "complement": patched[row_id][0] - patched[row_id][1],
            "full": _margin(prior[(row_id, "attn:11")]),
        }
        if bool(row["answer_changes"]):
            f = {
                name: kernel.signed_pairwise_donor_recovery(
                    -base_margin, donor_margin, -margin
                )
                for name, margin in margins.items()
            }
        else:
            f = {name: (margin - base_margin) / target_scale for name, margin in margins.items()}
            control_abs[family].append(
                kernel.normalized_same_answer_effect(
                    base_margin, margins["complement"], target_scale
                )
            )
        interaction = f["full"] - f["head"] - f["complement"] + f["empty"]
        record = {
            "row_id": row_id, "family": family,
            "empty": f["empty"], "head": f["head"],
            "complement": f["complement"], "full": f["full"],
            "interaction": interaction,
        }
        evidence.append(record)
        if family in {"A1", "A2"}:
            by_cell.setdefault(str(row["capability_cell_id"]), []).append(record)

    cells = []
    for cell_id, records in sorted(by_cell.items()):
        complement = statistics.fmean(item["complement"] for item in records)
        interaction = statistics.fmean(item["interaction"] for item in records)
        cells.append({
            "cell_id": cell_id, "row_count": len(records),
            "mean_complement_recovery": complement,
            "mean_interaction_recovery": interaction,
            "clean_split": (
                abs(complement) <= MAX_ABS_COMPLEMENT_TARGET
                and abs(interaction) <= MAX_ABS_INTERACTION_TARGET
            ),
            "distributed_or_interactive": (
                abs(complement) >= DISTRIBUTED_COMPLEMENT_TARGET
                or abs(interaction) >= DISTRIBUTED_INTERACTION_TARGET
            ),
        })
    controls = {
        family: statistics.fmean(values) for family, values in control_abs.items()
    }
    clean = all(cell["clean_split"] for cell in cells) and all(
        value <= MAX_CONTROL_MOVEMENT for value in controls.values()
    )
    distributed = any(cell["distributed_or_interactive"] for cell in cells)
    terminal = "clean_split" if clean else ("distributed_or_interactive" if distributed else "inconclusive")
    return {
        "schema": "task14_attention11_head_complement_factorial_result_v1",
        "screen_tier_only": True,
        "execution_policy": "managed_queue_only",
        "authority_sha256": EXPECTED_AUTHORITY_SHA256,
        "v2_result_sha256": EXPECTED_V2_SHA256,
        "terminal": terminal,
        "predictions": {
            "pred_a_exact_factorial_partition": True,
            "pred_b_clean_head11_3_split": clean,
            "pred_c_distributed_or_interactive_complement": distributed,
        },
        "target_scale": target_scale,
        "cells": cells,
        "control_mean_absolute_movement": controls,
        "evidence": evidence,
        "active_price": {
            "forward_calls": forwards, "example_evaluations": evaluations,
            "backward_calls": 0, "model_updates": 0,
            "raw_numeric_evidence_bytes": 2 * len(rows) * 4,
        },
        "serial_seconds": clock() - started,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    environment = os.environ
    for name in ("BQLIB_DRYRUN", "BQLIB_NO_MODEL"):
        if environment.get(name) not in {None, "1"}:
            raise ComplementScreenError(f"{name} must be absent or exactly 1")
    managed_preflight = (
        environment.get("BQLIB_DRYRUN") == "1"
        or environment.get("BQLIB_NO_MODEL") == "1"
    )
    if args.dry_run or managed_preflight:
        print(json.dumps(compile_dryrun(), sort_keys=True))
        return
    result = run_science()
    payload = managed.atomic_create_json(RESULT, result)
    print(json.dumps({
        "terminal": result["terminal"],
        "result_path": RESULT.relative_to(ROOT).as_posix(),
        "result_sha256": hashlib.sha256(payload).hexdigest(),
        "active_price": result["active_price"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
