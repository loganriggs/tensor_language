#!/usr/bin/env python3
# BQGATE: three frozen science predictions are emitted by this targeted runner.
"""Split the Task-14 MLP15--17 path into {MLP15, MLP17} versus {MLP16}.

Let F(S) be head-11.3 donor recovery after restoring the recipient-native
outputs of the MLPs in S. Immutable prior results already contain F(empty),
F(16), and F(15,16,17). This screen computes only F(15,17), completing

    I = F(15,16,17) - F(15,17) - F(16) + F(empty).

Prediction A: native replay error is at most 1e-4, otherwise the instrument is
invalid. Prediction B: MLP15+17 explain the combined path if their four-cell
loss vector is within 25% relative L2 error of the combined loss, MLP16 alone
has at most 3% RMS loss, and interaction has at most 3% RMS. Prediction C:
MLP16 is interaction-dependent if interaction has at least 5% RMS, at least
two cells have magnitude at least 3%, and its norm is at least 35% of the
combined loss norm. Every scientific terminal also requires every control
term to be at most 10%. The gap is inconclusive. This is a causal grouping
screen, not rank reduction or semantic identification. Maximum new price:
12 forwards, 384 examples, no backward calls or updates, and 1,024 retained
raw-logit bytes. GPU execution is managed-queue only.
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

import circuit_fast_screen_producer as producer
import run_task14_head11_3_attention_mlp_path_factorial as path_parent
import run_task14_head11_3_downstream_module_reader_screen as reader


ROOT = Path(__file__).resolve().parent.parent
READER_RESULT = ROOT / "circuits/followups/task14_head11_3_downstream_module_reader_screen_v1_result.json"
LATE_RESULT = ROOT / "circuits/followups/task14_head11_3_late_mlp_halves_factorial_v1_result.json"
RESULT = ROOT / "circuits/followups/task14_head11_3_mlp15_17_vs_mlp16_factorial_v1_result.json"
READER_RESULT_SHA256 = "677e5e2eccdc13fcc7bd2053be5d9c450d2af22d36d20760334d9bd2f50a0ffa"
LATE_RESULT_SHA256 = "ecea48f3b050af9a7cf23f06622045b87ef4e68428ca3ac320aee34338e8f35a"
PRIOR_ART_SHA256 = "00468dd4de08b468cc16b97cae4ae27571077983536fa4f7613e5ccc0e68f9e5"
CORE = ("mlp:15", "mlp:17")
REPLAY_ATOL = 1e-4
CORE_RELATIVE_L2_MAX = 0.25
MLP16_RMS_MAX = 0.03
CORE_INTERACTION_RMS_MAX = 0.03
NONLINEAR_INTERACTION_RMS_MIN = 0.05
NONLINEAR_CELL_ABS_MIN = 0.03
NONLINEAR_MIN_CELLS = 2
NONLINEAR_NORM_RATIO_MIN = 0.35
CONTROL_TERM_MAX = 0.10


class CoreSplitError(ValueError):
    """Frozen evidence or execution violated the registered contract."""


class CoreSplitBackend(Protocol):
    def native(self, batch: producer.ModelBatch, *, capture: bool) -> producer.BatchOutput: ...

    def induce_and_restore(
        self, batch: producer.ModelBatch, *, restore_sites: Sequence[str],
        donor_cache: Mapping[tuple[str, str], object],
        recipient_cache: Mapping[tuple[str, str], object],
    ) -> producer.BatchOutput: ...


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load():
    rows, native, head = path_parent._load()
    expected = ((READER_RESULT, READER_RESULT_SHA256), (LATE_RESULT, LATE_RESULT_SHA256))
    for path, digest in expected:
        if _sha256(path) != digest:
            raise CoreSplitError(f"immutable source changed: {path.name}")
    reader_result = json.loads(READER_RESULT.read_text())
    late_result = json.loads(LATE_RESULT.read_text())
    if any(source.get("authority_sha256") != reader.AUTHORITY_SHA256
           for source in (reader_result, late_result)):
        raise CoreSplitError("source authority changed")
    single = {
        str(item["row_id"]): item for item in reader_result.get("evidence", [])
        if item.get("site_id") == "mlp:16"
    }
    late = {str(item["row_id"]): item for item in late_result.get("evidence", [])}
    row_ids = {str(row["row_id"]) for row in rows}
    if set(single) != row_ids or set(late) != row_ids:
        raise CoreSplitError("source results lack exact row coverage")
    prior = {}
    for row_id in sorted(row_ids):
        empty = float(single[row_id]["head_only_recovery"])
        if abs(empty - float(late[row_id]["empty"])) > 1e-12:
            raise CoreSplitError("source F(empty) corners disagree")
        prior[row_id] = {
            "empty": empty,
            "mlp16": float(single[row_id]["restored_recovery"]),
            "all": float(late[row_id]["mlp15_17"]),
        }
    return rows, native, head, prior


def compile_dryrun() -> dict[str, object]:
    rows, _native, _head, _prior = _load()
    chunks = len(reader._chunks(rows))
    calls = 3 * chunks
    return {
        "schema": "task14_head11_3_mlp15_17_vs_mlp16_factorial_dryrun_v1",
        "model_loaded": False, "gpu_accessed": False, "queue_touched": False,
        "authority_sha256": reader.AUTHORITY_SHA256,
        "reader_result_sha256": READER_RESULT_SHA256,
        "late_result_sha256": LATE_RESULT_SHA256,
        "prior_art_sha256": PRIOR_ART_SHA256,
        "new_arm": list(CORE),
        "reused_corners": ["F(empty)", "F(16)", "F(15,16,17)"],
        "equation": "I=F(15,16,17)-F(15,17)-F(16)+F(empty)",
        "maximum_new_price": {
            "forward_calls": calls, "example_evaluations": calls * reader.BATCH_SIZE,
            "backward_calls": 0, "model_updates": 0,
            "raw_numeric_evidence_bytes": len(rows) * 2 * 4,
        },
        "bars": {
            "native_replay_atol": REPLAY_ATOL,
            "core_relative_l2_max": CORE_RELATIVE_L2_MAX,
            "mlp16_rms_max": MLP16_RMS_MAX,
            "core_interaction_rms_max": CORE_INTERACTION_RMS_MAX,
            "nonlinear_interaction_rms_min": NONLINEAR_INTERACTION_RMS_MIN,
            "nonlinear_cell_abs_min": NONLINEAR_CELL_ABS_MIN,
            "nonlinear_min_cells": NONLINEAR_MIN_CELLS,
            "nonlinear_norm_ratio_min": NONLINEAR_NORM_RATIO_MIN,
            "all_control_terms_max": CONTROL_TERM_MAX,
        },
    }


def _norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def run_science(
    *, backend: CoreSplitBackend | None = None, device: str = "cuda", clock=time.perf_counter,
) -> dict[str, object]:
    rows, native, _head, prior = _load()
    executor = backend if backend is not None else path_parent.Task14PathTorchBackend.load(device)
    recipient_cache: dict[tuple[str, str], object] = {}
    donor_cache: dict[tuple[str, str], object] = {}
    replay_error = 0.0
    forwards = evaluations = 0
    started = clock()
    for side, cache in (("base", recipient_cache), ("donor", donor_cache)):
        for chunk in reader._chunks(rows):
            batch = reader._batch(chunk, side)
            output = executor.native(batch, capture=True)
            forwards += 1
            evaluations += len(chunk)
            if len(output.answer_foil) != len(chunk):
                raise CoreSplitError("native output count differs from batch")
            cache.update(output.captured)
            for row_id, observed in zip(batch.row_ids, output.answer_foil):
                replay_error = max(
                    replay_error,
                    *(abs(a - b) for a, b in zip(reader._pair(observed), native[(row_id, side)])),
                )
    required_donor = {(str(row["row_id"]), reader.HEAD_SITE) for row in rows}
    required_recipient = {(str(row["row_id"]), site) for row in rows for site in CORE}
    if not required_donor.issubset(donor_cache) or not required_recipient.issubset(recipient_cache):
        raise CoreSplitError("native capture lacks the head or MLP15/17 output")
    core_pairs: dict[str, tuple[float, float]] = {}
    for chunk in reader._chunks(rows):
        batch = reader._batch(chunk, "base")
        output = executor.induce_and_restore(
            batch, restore_sites=CORE, donor_cache=donor_cache,
            recipient_cache=recipient_cache,
        )
        forwards += 1
        evaluations += len(chunk)
        if len(output.answer_foil) != len(chunk):
            raise CoreSplitError("MLP15+17 output count differs from batch")
        core_pairs.update({
            row_id: reader._pair(pair) for row_id, pair in zip(batch.row_ids, output.answer_foil)
        })
    scale = statistics.median(
        reader._margin(native[(str(row["row_id"]), "donor")])
        + reader._margin(native[(str(row["row_id"]), "base")])
        for row in rows if row["transform_id"] in {"A1", "A2"}
    )
    targets: dict[str, list[dict[str, object]]] = {}
    controls: dict[str, list[dict[str, object]]] = {"P": [], "C": []}
    evidence = []
    for row in rows:
        row_id = str(row["row_id"])
        family = str(row["transform_id"])
        core = reader._recovery(
            family, native[(row_id, "base")], native[(row_id, "donor")],
            core_pairs[row_id], scale,
        )
        values = {**prior[row_id], "mlp15_17": core}
        values["interaction"] = values["all"] - core - values["mlp16"] + values["empty"]
        values["all_loss"] = values["empty"] - values["all"]
        values["mlp16_loss"] = values["empty"] - values["mlp16"]
        values["mlp15_17_loss"] = values["empty"] - core
        record = {"row_id": row_id, "family": family, **values}
        evidence.append(record)
        if family in {"A1", "A2"}:
            targets.setdefault(str(row["capability_cell_id"]), []).append(record)
        else:
            controls[family].append(record)
    keys = ("all_loss", "mlp16_loss", "mlp15_17_loss", "interaction")
    cells = {
        cell: {key: statistics.fmean(float(item[key]) for item in records) for key in keys}
        for cell, records in sorted(targets.items())
    }
    control_keys = ("empty", "mlp16", "mlp15_17", "all") + keys
    control = {
        family: {
            key: statistics.fmean(abs(float(item[key])) for item in records)
            for key in control_keys
        }
        for family, records in controls.items()
    }
    max_control = max(value for family in control.values() for value in family.values())
    all_vector = [float(cell["all_loss"]) for cell in cells.values()]
    core_vector = [float(cell["mlp15_17_loss"]) for cell in cells.values()]
    mlp16_vector = [float(cell["mlp16_loss"]) for cell in cells.values()]
    interaction_vector = [float(cell["interaction"]) for cell in cells.values()]
    all_norm = max(_norm(all_vector), 1e-12)
    core_relative_l2 = _norm([a - b for a, b in zip(core_vector, all_vector)]) / all_norm
    mlp16_rms = _norm(mlp16_vector) / 2
    interaction_rms = _norm(interaction_vector) / 2
    interaction_ratio = _norm(interaction_vector) / all_norm
    control_ok = max_control <= CONTROL_TERM_MAX
    valid = replay_error <= REPLAY_ATOL
    core_explains = (
        core_relative_l2 <= CORE_RELATIVE_L2_MAX
        and mlp16_rms <= MLP16_RMS_MAX
        and interaction_rms <= CORE_INTERACTION_RMS_MAX
        and control_ok
    )
    nonlinear = (
        interaction_rms >= NONLINEAR_INTERACTION_RMS_MIN
        and sum(abs(value) >= NONLINEAR_CELL_ABS_MIN for value in interaction_vector)
        >= NONLINEAR_MIN_CELLS
        and interaction_ratio >= NONLINEAR_NORM_RATIO_MIN
        and control_ok
    )
    terminal = (
        "invalid" if not valid else
        "mlp15_17_core_path_screen" if core_explains else
        "mlp16_interaction_screen" if nonlinear else
        "inconclusive"
    )
    return {
        "schema": "task14_head11_3_mlp15_17_vs_mlp16_factorial_result_v1",
        "screen_tier_only": True, "execution_policy": "managed_queue_only",
        "authority_sha256": reader.AUTHORITY_SHA256,
        "reader_result_sha256": READER_RESULT_SHA256,
        "late_result_sha256": LATE_RESULT_SHA256,
        "prior_art_sha256": PRIOR_ART_SHA256,
        "terminal": terminal,
        "predictions": {
            "pred_a_native_replay": valid,
            "pred_b_mlp15_17_explain_combined_path": core_explains,
            "pred_c_mlp16_interaction_dependent": nonlinear,
        },
        "native_replay_max_abs_error": replay_error,
        "target_scale": scale,
        "target_cells": cells,
        "core_relative_l2_to_all": core_relative_l2,
        "mlp16_rms": mlp16_rms,
        "interaction_rms": interaction_rms,
        "interaction_norm_ratio_to_all": interaction_ratio,
        "control_mean_absolute_terms": control,
        "evidence": evidence,
        "active_new_price": {
            "forward_calls": forwards, "example_evaluations": evaluations,
            "backward_calls": 0, "model_updates": 0,
            "raw_numeric_evidence_bytes": len(rows) * 2 * 4,
        },
        "serial_seconds": clock() - started,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for name in ("BQLIB_DRYRUN", "BQLIB_NO_MODEL"):
        if os.environ.get(name) not in {None, "1"}:
            raise CoreSplitError(f"{name} must be absent or exactly 1")
    if args.dry_run or any(os.environ.get(name) == "1" for name in ("BQLIB_DRYRUN", "BQLIB_NO_MODEL")):
        print(json.dumps(compile_dryrun(), sort_keys=True))
        return
    if RESULT.exists():
        raise CoreSplitError(f"refusing to overwrite {RESULT}")
    result = run_science()
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({key: result[key] for key in (
        "terminal", "predictions", "active_new_price", "serial_seconds",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
