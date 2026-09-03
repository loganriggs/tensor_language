#!/usr/bin/env python3
"""Summarize rung-522 optimizer health from its immutable frame archive.

This is a CPU-only diagnostic.  It does not change scientific thresholds and
does not score TEST.  Lower objective changes are improvements.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Mapping, Sequence


OPS = Path(__file__).resolve().parent
ROOT = OPS.parent
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))

import attention8_selective_shared_projector_rung522_archive as archive  # noqa: E402


DEFAULT_ARCHIVE = ROOT / "attention8_selective_shared_projector_rung522_work/frames_pretest.pt"
DEFAULT_OUTPUT = ROOT / "attention8_selective_shared_projector_rung522_health_analysis.json"


def _group_target(record: archive.ArchivedFrameRecord) -> str:
    spec = record.spec
    if spec.family in {"real_leave_one_out", "recovery_only", "label_null"}:
        if spec.omitted_target is None:
            raise ValueError(f"{spec.frame_id}: omitted target is absent")
        return spec.omitted_target
    if spec.family == "target_oracle":
        if spec.oracle_target is None:
            raise ValueError(f"{spec.frame_id}: oracle target is absent")
        return spec.oracle_target
    if spec.family == "all_three":
        return "all_three_fitted_targets"
    raise ValueError(f"{spec.frame_id}: unknown frame family {spec.family!r}")


def _finite_number(payload: Mapping[str, object], name: str) -> float:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"health payload has no numeric {name}")
    return float(value)


def _summarize_group(
    records: Sequence[archive.ArchivedFrameRecord],
) -> dict[str, object]:
    if not records:
        raise ValueError("cannot summarize an empty frame group")
    failures = Counter(
        failure for record in records for failure in record.health_failures
    )
    healthy = sum(record.healthy for record in records)
    initial_validation = [
        _finite_number(record.health_record_payload, "initial_validation_objective")
        for record in records
    ]
    final_validation = [
        _finite_number(record.health_record_payload, "final_validation_objective")
        for record in records
    ]
    initial_window = [
        _finite_number(record.health_record_payload, "initial_window_mean")
        for record in records
    ]
    final_window = [
        _finite_number(record.health_record_payload, "final_window_mean")
        for record in records
    ]
    distance = [
        _finite_number(record.health_record_payload, "projector_distance_from_initialization")
        for record in records
    ]
    orthonormality = [
        _finite_number(record.health_record_payload, "orthonormality_error")
        for record in records
    ]
    histories = []
    for record in records:
        raw_history = record.fit_record_payload.get("loss_history")
        if not isinstance(raw_history, list) or len(raw_history) != 200:
            raise ValueError("fit record has no exact 200-update loss history")
        histories.append([float(value) for value in raw_history])
    maxima = [max(history) for history in histories]
    medians = [statistics.median(history) for history in histories]
    best_window_starts = []
    for history in histories:
        windows = [sum(history[start:start + 20]) / 20 for start in range(181)]
        best_window_starts.append(min(range(len(windows)), key=windows.__getitem__))
    return {
        "frame_count": len(records),
        "healthy_count": healthy,
        "healthy_percent": 100.0 * healthy / len(records),
        "failure_reason_counts": dict(sorted(failures.items())),
        "median_initial_validation_objective": statistics.median(initial_validation),
        "median_final_validation_objective": statistics.median(final_validation),
        "median_validation_change_final_minus_initial": statistics.median(
            final - initial for initial, final in zip(
                initial_validation, final_validation, strict=True
            )
        ),
        "median_initial_training_window": statistics.median(initial_window),
        "median_final_training_window": statistics.median(final_window),
        "median_training_window_change_final_minus_initial": statistics.median(
            final - initial for initial, final in zip(
                initial_window, final_window, strict=True
            )
        ),
        "median_projector_distance_from_initialization": statistics.median(distance),
        "maximum_orthonormality_error": max(orthonormality),
        "median_frame_maximum_loss": statistics.median(maxima),
        "maximum_loss": max(maxima),
        "median_frame_maximum_to_median_loss_ratio": statistics.median(
            maximum / max(median, 1e-30)
            for maximum, median in zip(maxima, medians, strict=True)
        ),
        "frames_with_loss_above_10": sum(maximum > 10 for maximum in maxima),
        "frames_with_loss_above_100": sum(maximum > 100 for maximum in maxima),
        "frames_with_loss_above_1000": sum(maximum > 1000 for maximum in maxima),
        "median_best_20_update_start": statistics.median(best_window_starts),
        "best_20_update_start_minimum": min(best_window_starts),
        "best_20_update_start_maximum": max(best_window_starts),
        "frames_with_first_window_mean_above_twice_first_window_median": sum(
            (sum(history[:20]) / 20) > 2 * max(statistics.median(history[:20]), 1e-30)
            for history in histories
        ),
    }


def summarize_archive_health(
    loaded: archive.LoadedFrameArchive,
) -> dict[str, object]:
    """Return family and target-stratified health facts without reinterpretation."""
    records = list(loaded.records.values())
    if len(records) != 103:
        raise ValueError("rung522 health analysis requires the complete 103-frame archive")
    by_family: dict[str, list[archive.ArchivedFrameRecord]] = {}
    by_family_target: dict[str, list[archive.ArchivedFrameRecord]] = {}
    for record in records:
        by_family.setdefault(record.spec.family, []).append(record)
        key = f"{record.spec.family}:{_group_target(record)}"
        by_family_target.setdefault(key, []).append(record)
    return {
        "schema": "rung522-health-analysis-v1",
        "archive_file_sha256": loaded.file_sha256,
        "archive_content_sha256": loaded.content_sha256,
        "overall": _summarize_group(records),
        "by_family": {
            name: _summarize_group(group) for name, group in sorted(by_family.items())
        },
        "by_family_and_target": {
            name: _summarize_group(group)
            for name, group in sorted(by_family_target.items())
        },
        "interpretation_rule": {
            "final_window_not_below_initial_window": (
                "the rotating FIT objective did not improve from its first to last 20 updates"
            ),
            "validation_not_better_than_initialization": (
                "the fixed VALIDATION objective did not improve over the seeded initial frame"
            ),
            "projector_did_not_move": "the learned projector stayed too close to initialization",
            "orthonormality": "the stored columns ceased to define a valid orthonormal frame",
            "nonfinite_loss": "at least one optimization loss was not finite",
        },
        "scientific_scope": (
            "optimizer-health diagnosis only; an unhealthy frame is an invalid instrument, "
            "not evidence that the circuit hypothesis is false"
        ),
    }


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite health analysis: {path}")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as sink:
            json.dump(payload, sink, indent=2, sort_keys=True, allow_nan=False)
            sink.write("\n")
            sink.flush()
            os.fsync(sink.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = summarize_archive_health(archive.load_frame_archive(args.archive))
    _atomic_json(args.output, result)
    print(json.dumps({
        "output": str(args.output),
        "archive_file_sha256": result["archive_file_sha256"],
        "overall": result["overall"],
    }, indent=2, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    main()
