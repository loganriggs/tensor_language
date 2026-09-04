#!/usr/bin/env python3
# BQGATE: emits pred_a_source_identity, pred_b_additive_pair, pred_c_nonlinear_pair.
"""Compute the exact MLP15-by-MLP17 Task-14 interaction from landed evidence.

No model execution is needed. The downstream reader result supplies F(empty),
F(15), and F(17); the MLP15+17 result supplies F(15,17). Per row,

    I = F(15,17) - F(15) - F(17) + F(empty).

Additive prediction: four-cell interaction RMS <= 0.03 and every P/C term
<= 0.10. Nonlinear prediction: interaction RMS >= 0.08, at least two cells
have |I| >= 0.05, its norm is >= 50% of the joint-loss norm, and every P/C
term <= 0.10. These bars were copied from the preceding late-MLP factorial.
The gap is inconclusive. This is a FIT causal grouping screen, not rank
reduction, held-out identification, or adoption.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Mapping, Sequence

import run_task14_head11_3_downstream_module_reader_screen as reader


ROOT = Path(__file__).resolve().parent.parent
READER_RESULT = ROOT / "circuits/followups/task14_head11_3_downstream_module_reader_screen_v1_result.json"
JOINT_RESULT = ROOT / "circuits/followups/task14_head11_3_mlp15_17_vs_mlp16_factorial_v1_result.json"
RESULT = ROOT / "circuits/followups/task14_head11_3_mlp15_mlp17_interaction_v1_result.json"
READER_RESULT_SHA256 = "677e5e2eccdc13fcc7bd2053be5d9c450d2af22d36d20760334d9bd2f50a0ffa"
JOINT_RESULT_SHA256 = "e7765dbfd0269a32ab3e4ea8ccfeda1d4ceccabcce98e1d97d3dfd08c0ad8747"
PRIOR_ART_SHA256 = "716cd1637afc16d4cd390c2ba8e9a557cab33387cd7c1bdcee5aa0fc00241c42"
SOURCE_EMPTY_ATOL = 1e-12
ADDITIVE_INTERACTION_RMS_MAX = 0.03
NONLINEAR_INTERACTION_RMS_MIN = 0.08
NONLINEAR_CELL_ABS_MIN = 0.05
NONLINEAR_MIN_CELLS = 2
NONLINEAR_NORM_RATIO_MIN = 0.50
CONTROL_TERM_MAX = 0.10


class PairAnalysisError(ValueError):
    """The immutable sources or exact factorial contract are invalid."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _load():
    rows, _parent = reader._load()
    for path, expected in (
        (READER_RESULT, READER_RESULT_SHA256),
        (JOINT_RESULT, JOINT_RESULT_SHA256),
    ):
        if _sha256(path) != expected:
            raise PairAnalysisError(f"immutable source changed: {path.name}")
    reader_result = json.loads(READER_RESULT.read_text())
    joint_result = json.loads(JOINT_RESULT.read_text())
    if any(source.get("authority_sha256") != reader.AUTHORITY_SHA256
           for source in (reader_result, joint_result)):
        raise PairAnalysisError("source authority changed")
    by_site = {
        site: {
            str(item["row_id"]): item for item in reader_result.get("evidence", [])
            if item.get("site_id") == site
        }
        for site in ("mlp:15", "mlp:17")
    }
    joint = {str(item["row_id"]): item for item in joint_result.get("evidence", [])}
    row_ids = {str(row["row_id"]) for row in rows}
    if any(set(items) != row_ids for items in (*by_site.values(), joint)):
        raise PairAnalysisError("source results lack identical 128-row coverage")
    corners = {}
    max_empty_error = 0.0
    for row in rows:
        row_id = str(row["row_id"])
        items = (by_site["mlp:15"][row_id], by_site["mlp:17"][row_id], joint[row_id])
        if any(str(item["family"]) != str(row["transform_id"]) for item in items):
            raise PairAnalysisError("source family labels disagree")
        empty_values = (
            float(items[0]["head_only_recovery"]),
            float(items[1]["head_only_recovery"]),
            float(items[2]["empty"]),
        )
        max_empty_error = max(max_empty_error, max(empty_values) - min(empty_values))
        corners[row_id] = {
            "empty": empty_values[0],
            "mlp15": float(items[0]["restored_recovery"]),
            "mlp17": float(items[1]["restored_recovery"]),
            "both": float(items[2]["mlp15_17"]),
        }
    if max_empty_error > SOURCE_EMPTY_ATOL:
        raise PairAnalysisError("source F(empty) corners disagree")
    return rows, corners, max_empty_error


def compile_dryrun() -> dict[str, object]:
    rows, _corners, max_empty_error = _load()
    return {
        "schema": "task14_head11_3_mlp15_mlp17_interaction_dryrun_v1",
        "model_loaded": False, "gpu_accessed": False, "queue_touched": False,
        "authority_sha256": reader.AUTHORITY_SHA256,
        "reader_result_sha256": READER_RESULT_SHA256,
        "joint_result_sha256": JOINT_RESULT_SHA256,
        "prior_art_sha256": PRIOR_ART_SHA256,
        "row_count": len(rows), "source_empty_max_abs_error": max_empty_error,
        "equation": "I=F(15,17)-F(15)-F(17)+F(empty)",
        "maximum_new_execution_price": {
            "forward_calls": 0, "example_evaluations": 0,
            "backward_calls": 0, "model_updates": 0, "raw_numeric_evidence_bytes": 0,
        },
        "bars": {
            "source_empty_atol": SOURCE_EMPTY_ATOL,
            "additive_interaction_rms_max": ADDITIVE_INTERACTION_RMS_MAX,
            "nonlinear_interaction_rms_min": NONLINEAR_INTERACTION_RMS_MIN,
            "nonlinear_cell_abs_min": NONLINEAR_CELL_ABS_MIN,
            "nonlinear_min_cells": NONLINEAR_MIN_CELLS,
            "nonlinear_norm_ratio_min": NONLINEAR_NORM_RATIO_MIN,
            "all_control_terms_max": CONTROL_TERM_MAX,
        },
    }


def _norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _score(rows: Sequence[Mapping[str, object]], corners: Mapping[str, Mapping[str, float]]):
    targets: dict[str, list[dict[str, object]]] = {}
    controls: dict[str, list[dict[str, object]]] = {"P": [], "C": []}
    evidence = []
    for row in rows:
        row_id = str(row["row_id"])
        family = str(row["transform_id"])
        values = {key: float(value) for key, value in corners[row_id].items()}
        values["interaction"] = (
            values["both"] - values["mlp15"] - values["mlp17"] + values["empty"]
        )
        values["mlp15_loss"] = values["empty"] - values["mlp15"]
        values["mlp17_loss"] = values["empty"] - values["mlp17"]
        values["both_loss"] = values["empty"] - values["both"]
        record = {"row_id": row_id, "family": family, **values}
        evidence.append(record)
        if family in {"A1", "A2"}:
            targets.setdefault(str(row["capability_cell_id"]), []).append(record)
        else:
            controls[family].append(record)
    keys = ("mlp15_loss", "mlp17_loss", "both_loss", "interaction")
    cells = {
        cell: {key: statistics.fmean(float(item[key]) for item in items) for key in keys}
        for cell, items in sorted(targets.items())
    }
    control_keys = ("empty", "mlp15", "mlp17", "both") + keys
    control = {
        family: {
            key: statistics.fmean(abs(float(item[key])) for item in items)
            for key in control_keys
        }
        for family, items in controls.items()
    }
    interaction_vector = [float(cell["interaction"]) for cell in cells.values()]
    joint_vector = [float(cell["both_loss"]) for cell in cells.values()]
    interaction_rms = _norm(interaction_vector) / math.sqrt(len(interaction_vector))
    interaction_ratio = _norm(interaction_vector) / max(_norm(joint_vector), 1e-12)
    max_control = max(value for family in control.values() for value in family.values())
    control_ok = max_control <= CONTROL_TERM_MAX
    additive = interaction_rms <= ADDITIVE_INTERACTION_RMS_MAX and control_ok
    nonlinear = (
        interaction_rms >= NONLINEAR_INTERACTION_RMS_MIN
        and sum(abs(value) >= NONLINEAR_CELL_ABS_MIN for value in interaction_vector)
        >= NONLINEAR_MIN_CELLS
        and interaction_ratio >= NONLINEAR_NORM_RATIO_MIN
        and control_ok
    )
    return {
        "evidence": evidence, "target_cells": cells,
        "control_mean_absolute_terms": control,
        "interaction_rms": interaction_rms,
        "interaction_norm_ratio_to_joint_loss": interaction_ratio,
        "maximum_control_term": max_control,
        "control_ok": control_ok, "additive": additive, "nonlinear": nonlinear,
    }


def run_analysis(*, clock=time.perf_counter) -> dict[str, object]:
    started_utc = _utc_now()
    started = clock()
    rows, corners, max_empty_error = _load()
    scored = _score(rows, corners)
    terminal = (
        "additive_pair_screen" if scored["additive"] else
        "nonlinear_pair_screen" if scored["nonlinear"] else
        "inconclusive"
    )
    elapsed = clock() - started
    return {
        "schema": "task14_head11_3_mlp15_mlp17_interaction_result_v1",
        "screen_tier_only": True, "execution_policy": "zero_gpu_immutable_evidence_only",
        "authority_sha256": reader.AUTHORITY_SHA256,
        "reader_result_sha256": READER_RESULT_SHA256,
        "joint_result_sha256": JOINT_RESULT_SHA256,
        "prior_art_sha256": PRIOR_ART_SHA256,
        "blinding_status": "not_blinded_due_to_disclosed_repository_search; bars mechanically inherited",
        "terminal": terminal,
        "predictions": {
            "pred_a_source_identity": True,
            "pred_b_additive_pair": bool(scored["additive"]),
            "pred_c_nonlinear_pair": bool(scored["nonlinear"]),
        },
        "source_row_count": len(rows),
        "source_empty_max_abs_error": max_empty_error,
        "interaction_rms": scored["interaction_rms"],
        "interaction_norm_ratio_to_joint_loss": scored["interaction_norm_ratio_to_joint_loss"],
        "maximum_control_term": scored["maximum_control_term"],
        "target_cells": scored["target_cells"],
        "control_mean_absolute_terms": scored["control_mean_absolute_terms"],
        "evidence": scored["evidence"],
        "active_new_execution_price": {
            "forward_calls": 0, "example_evaluations": 0,
            "backward_calls": 0, "model_updates": 0, "raw_numeric_evidence_bytes": 0,
        },
        "fast_screen_ledger_status": "not_appended: schema requires positive model-execution price for a screen",
        "started_utc": started_utc, "finished_utc": _utc_now(), "serial_seconds": elapsed,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps(compile_dryrun(), sort_keys=True))
        return
    if RESULT.exists():
        raise PairAnalysisError(f"refusing to overwrite {RESULT}")
    result = run_analysis()
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({key: result[key] for key in (
        "terminal", "predictions", "interaction_rms",
        "interaction_norm_ratio_to_joint_loss", "maximum_control_term", "serial_seconds",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
