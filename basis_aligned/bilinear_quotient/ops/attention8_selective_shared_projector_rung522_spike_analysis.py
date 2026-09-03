#!/usr/bin/env python3
"""Locate rung-522 loss spikes in the frozen row/map scheduler.

This CPU-only diagnostic asks whether extreme losses recur on particular
target/member-row/donor-map combinations.  It does not alter a fit or threshold.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Mapping


OPS = Path(__file__).resolve().parent
ROOT = OPS.parent
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))

import attention8_selective_shared_projector_rung522_archive as archive  # noqa: E402


DEFAULT_ARCHIVE = ROOT / "attention8_selective_shared_projector_rung522_work/frames_pretest.pt"
DEFAULT_OUTPUT = ROOT / "attention8_selective_shared_projector_rung522_spike_analysis.json"


def _rate_table(visits: Counter, spikes: Counter) -> dict[str, dict[str, float | int]]:
    return {
        str(key): {
            "visits": visits[key],
            "spikes": spikes[key],
            "spike_percent": 100.0 * spikes[key] / visits[key],
        }
        for key in sorted(visits, key=str)
    }


def analyze_spikes(
    records: Mapping[str, archive.ArchivedFrameRecord], *, threshold: float = 100.0
) -> dict[str, object]:
    if not records:
        raise ValueError("spike analysis requires at least one archived fit")
    if threshold <= 0:
        raise ValueError("spike threshold must be positive")
    map_visits: Counter = Counter()
    map_spikes: Counter = Counter()
    target_visits: Counter = Counter()
    target_spikes: Counter = Counter()
    family_visits: Counter = Counter()
    family_spikes: Counter = Counter()
    pattern_visits: Counter = Counter()
    pattern_spikes: Counter = Counter()
    pattern_fits: dict[tuple[object, ...], set[str]] = defaultdict(set)
    pattern_spike_fits: dict[tuple[object, ...], set[str]] = defaultdict(set)
    pattern_seeds: dict[tuple[object, ...], set[int]] = defaultdict(set)
    pattern_spike_seeds: dict[tuple[object, ...], set[int]] = defaultdict(set)
    first_spike_updates = []

    for record in records.values():
        history = record.fit_record_payload.get("loss_history")
        maximizing = record.fit_record_payload.get("maximizing_targets")
        roles = record.fit_scheduler_payload.get("roles")
        if not isinstance(history, list) or len(history) != 200:
            raise ValueError("fit history is not the frozen 200-update sequence")
        if not isinstance(maximizing, list) or len(maximizing) != 200:
            raise ValueError("maximizing-target history is not the frozen sequence")
        if not isinstance(roles, list):
            raise ValueError("fit scheduler has no role list")
        first = None
        for update, (loss, target) in enumerate(zip(history, maximizing, strict=True)):
            donor_map = update % 4
            member_rows = tuple(
                role["permutation"][update % len(role["permutation"])]
                for role in roles
                if role["target"] == target and role["kind"] == "member"
            )
            if not member_rows:
                raise ValueError("maximizing target has no scheduled member row")
            control_rows = tuple(
                role["permutation"][update % len(role["permutation"])]
                for role in roles
                if role["target"] == target and role["kind"] == "control"
            )
            if not control_rows:
                raise ValueError("maximizing target has no scheduled control row")
            pattern = (str(target), donor_map, member_rows, control_rows)
            map_visits[donor_map] += 1
            target_visits[str(target)] += 1
            family_visits[record.spec.family] += 1
            pattern_visits[pattern] += 1
            pattern_fits[pattern].add(record.spec.frame_id)
            pattern_seeds[pattern].add(record.spec.seed)
            if float(loss) > threshold:
                map_spikes[donor_map] += 1
                target_spikes[str(target)] += 1
                family_spikes[record.spec.family] += 1
                pattern_spikes[pattern] += 1
                pattern_spike_fits[pattern].add(record.spec.frame_id)
                pattern_spike_seeds[pattern].add(record.spec.seed)
                if first is None:
                    first = update
        if first is not None:
            first_spike_updates.append(first)

    repeated = []
    for pattern, spike_count in pattern_spikes.items():
        visits = pattern_visits[pattern]
        if visits < 2:
            continue
        target, donor_map, member_rows, control_rows = pattern
        repeated.append({
            "target": target,
            "donor_map_index": donor_map,
            "member_rows": list(member_rows),
            "control_rows": list(control_rows),
            "visits": visits,
            "spikes": spike_count,
            "spike_percent": 100.0 * spike_count / visits,
            "fits_with_pattern": len(pattern_fits[pattern]),
            "fits_with_a_spike": len(pattern_spike_fits[pattern]),
            "seeds_with_pattern": len(pattern_seeds[pattern]),
            "seeds_with_a_spike": len(pattern_spike_seeds[pattern]),
        })
    repeated.sort(
        key=lambda value: (-value["spike_percent"], -value["spikes"], value["target"],
                           value["donor_map_index"], value["member_rows"],
                           value["control_rows"])
    )
    total_events = sum(map_visits.values())
    total_spikes = sum(map_spikes.values())
    return {
        "schema": "rung522-scheduled-spike-analysis-v1",
        "threshold_strictly_above": threshold,
        "fit_count": len(records),
        "event_count": total_events,
        "spike_count": total_spikes,
        "spike_percent": 100.0 * total_spikes / total_events,
        "fits_with_a_spike": len(first_spike_updates),
        "first_spike_update_median": (
            None if not first_spike_updates else statistics.median(first_spike_updates)
        ),
        "first_spike_update_minimum": (
            None if not first_spike_updates else min(first_spike_updates)
        ),
        "first_spike_update_maximum": (
            None if not first_spike_updates else max(first_spike_updates)
        ),
        "by_donor_map": _rate_table(map_visits, map_spikes),
        "by_maximizing_target": _rate_table(target_visits, target_spikes),
        "by_fit_family": _rate_table(family_visits, family_spikes),
        "repeated_target_map_member_row_patterns": repeated,
        "repeated_patterns_with_100_percent_spike_rate": sum(
            value["spikes"] == value["visits"] for value in repeated
        ),
        "patterns_spiking_in_multiple_fits": sum(
            value["fits_with_a_spike"] >= 2 for value in repeated
        ),
        "patterns_spiking_in_every_fit_that_saw_them": sum(
            value["fits_with_a_spike"] == value["fits_with_pattern"]
            and value["fits_with_pattern"] >= 2
            for value in repeated
        ),
        "patterns_spiking_in_multiple_seeds": sum(
            value["seeds_with_a_spike"] >= 2 for value in repeated
        ),
        "patterns_spiking_in_every_seed_that_saw_them": sum(
            value["seeds_with_a_spike"] == value["seeds_with_pattern"]
            and value["seeds_with_pattern"] >= 2
            for value in repeated
        ),
        "interpretation": (
            "recurrence across different fits on the same target/member-row/control-row/map "
            "combination supports a row-scale or donor-specific objective problem; recurrence "
            "within one fit can instead reflect optimizer state. Neither observation by itself "
            "distinguishes a small full-response denominator from an extreme projected response"
        ),
    }


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite spike analysis: {path}")
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
    parser.add_argument("--threshold", type=float, default=100.0)
    args = parser.parse_args()
    loaded = archive.load_frame_archive(args.archive)
    result = {
        "archive_file_sha256": loaded.file_sha256,
        "archive_content_sha256": loaded.content_sha256,
        **analyze_spikes(loaded.records, threshold=args.threshold),
    }
    _atomic_json(args.output, result)
    print(json.dumps({
        "output": str(args.output),
        "spike_count": result["spike_count"],
        "event_count": result["event_count"],
        "fits_with_a_spike": result["fits_with_a_spike"],
        "repeated_patterns_with_100_percent_spike_rate": result[
            "repeated_patterns_with_100_percent_spike_rate"
        ],
        "patterns_spiking_in_multiple_fits": result["patterns_spiking_in_multiple_fits"],
        "patterns_spiking_in_multiple_seeds": result["patterns_spiking_in_multiple_seeds"],
    }, indent=2, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    main()
