#!/usr/bin/env python3
"""Independently audit rung 522's expected pre-TEST terminal failure."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Mapping


OPS = Path(__file__).resolve().parent
ROOT = OPS.parent
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))

import attention8_selective_shared_projector_rung522_archive as archive  # noqa: E402


DEFAULT_RESULT = ROOT / "attention8_selective_shared_projector_rung522_results.json"
DEFAULT_ARCHIVE = ROOT / "attention8_selective_shared_projector_rung522_work/frames_pretest.pt"
DEFAULT_OUTPUT = ROOT / "attention8_selective_shared_projector_rung522_terminal_audit.json"
EXPECTED_BUCKETS = {
    "all_three_selection_and_test": 180,
    "fit_d0_full_attention8": 95,
    "fit_health": 206,
    "full_attention8_comparator": 36,
    "haar": 720,
    "label_null_fit_health": 0,
    "native_capture": 131,
    "native_replay": 131,
    "prediction_a": 2_988,
    "recovery_only": 540,
    "self_donor": 2,
}
# label_null_fit_health is represented by fit_health rather than a distinct
# runtime bucket; keep only physically present keys for exact comparison.
EXPECTED_BUCKETS.pop("label_null_fit_health")
EXPECTED_INFERENCE = sum(EXPECTED_BUCKETS.values())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _ledger(value: Mapping[str, object], context: str) -> dict[str, object]:
    required = {
        "optimization_forwards", "optimization_backwards", "inference_forwards",
        "removal_forwards", "inference_by_bucket",
    }
    if not required.issubset(value):
        raise ValueError(f"{context} is missing ledger fields")
    observed = {key: value[key] for key in required}
    expected = {
        "optimization_forwards": 20_600,
        "optimization_backwards": 20_600,
        "inference_forwards": EXPECTED_INFERENCE,
        "removal_forwards": 0,
        "inference_by_bucket": EXPECTED_BUCKETS,
    }
    if observed != expected:
        raise ValueError(f"{context} changed: {observed} != {expected}")
    return observed


def audit_terminal_result(
    result: Mapping[str, object], loaded: archive.LoadedFrameArchive
) -> dict[str, object]:
    if result.get("rung") != 522:
        raise ValueError("result is not rung 522")
    if result.get("status") != "terminal_pretest_validation_failure":
        raise ValueError("rung 522 did not end at the registered pre-TEST failure")
    required_flags = {
        "test_opened": False,
        "test_closed": False,
        "pretest_manifest_created": False,
    }
    for name, expected in required_flags.items():
        if result.get(name) is not expected:
            raise ValueError(f"{name} changed or is not a literal {expected}")
    forbidden = {
        "capture_test", "test_outputs", "final_validation_test_decision",
        "prediction_c", "prediction_d", "removal",
    }
    present = sorted(forbidden.intersection(result))
    if present:
        raise ValueError(f"pre-TEST result contains TEST-stage fields: {present}")
    predictions = result.get("predictions")
    if not isinstance(predictions, Mapping):
        raise ValueError("prediction record is absent")
    if predictions.get("c") is not None or predictions.get("d") is not None:
        raise ValueError("pre-TEST result scored C or D")
    provisional = result.get("provisional_validation_decision")
    if not isinstance(provisional, Mapping) or provisional.get("pretest_passes") is not False:
        raise ValueError("provisional decision did not explicitly fail pre-TEST")

    pretest_ledger = result.get("pretest_call_ledger")
    execution = result.get("execution_price")
    if not isinstance(pretest_ledger, Mapping) or not isinstance(execution, Mapping):
        raise ValueError("terminal ledgers are absent")
    ledger = _ledger(pretest_ledger, "pretest_call_ledger")
    execution_ledger = _ledger(execution, "execution_price")
    if ledger != execution_ledger:
        raise ValueError("pretest and terminal execution ledgers differ")

    result_archive = result.get("frame_archive")
    if not isinstance(result_archive, Mapping):
        raise ValueError("frame archive receipt is absent")
    if result_archive.get("file_sha256") != loaded.file_sha256:
        raise ValueError("result names a different frame-archive file hash")
    if result_archive.get("content_sha256") != loaded.content_sha256:
        raise ValueError("result names a different frame-archive content hash")
    if result_archive.get("frame_count") != 103 or len(loaded.records) != 103:
        raise ValueError("frame census changed")

    health = result.get("frame_health")
    if not isinstance(health, Mapping) or set(health) != set(loaded.records):
        raise ValueError("result/archive frame-health keys differ")
    family_total: Counter = Counter()
    family_healthy: Counter = Counter()
    failure_counts: Counter = Counter()
    for frame_id, record in loaded.records.items():
        reported = health[frame_id]
        if not isinstance(reported, Mapping):
            raise ValueError(f"malformed health entry {frame_id}")
        if reported.get("healthy") is not record.healthy:
            raise ValueError(f"health boolean differs from archive for {frame_id}")
        if tuple(reported.get("failures", ())) != record.health_failures:
            raise ValueError(f"health failures differ from archive for {frame_id}")
        if reported.get("frame_sha256") != record.tensor_sha256:
            raise ValueError(f"frame hash differs from archive for {frame_id}")
        family_total[record.spec.family] += 1
        family_healthy[record.spec.family] += int(record.healthy)
        failure_counts.update(record.health_failures)
    if family_healthy["real_leave_one_out"] != 0:
        raise ValueError("expected invalid-instrument boundary changed: a real fit is healthy")
    return {
        "schema": "rung522-terminal-pretest-audit-v1",
        "passes": True,
        "status": result["status"],
        "test_opened": False,
        "pretest_manifest_created": False,
        "frame_archive_file_sha256": loaded.file_sha256,
        "frame_archive_content_sha256": loaded.content_sha256,
        "frame_count": len(loaded.records),
        "healthy_fit_count": sum(family_healthy.values()),
        "family_health": {
            family: {
                "healthy": family_healthy[family],
                "total": family_total[family],
                "healthy_percent": 100.0 * family_healthy[family] / family_total[family],
            }
            for family in sorted(family_total)
        },
        "failure_counts": dict(sorted(failure_counts.items())),
        "exact_call_ledger": ledger,
        "interpretation": "invalid optimizer instrument; no circuit null and no TEST evidence",
    }


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite audit: {path}")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as sink:
            json.dump(value, sink, indent=2, sort_keys=True, allow_nan=False)
            sink.write("\n")
            sink.flush()
            os.fsync(sink.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    with args.result.open(encoding="utf-8") as source:
        result = json.load(source)
    loaded = archive.load_frame_archive(args.archive)
    audit = {
        "result_file_sha256": _file_sha256(args.result),
        **audit_terminal_result(result, loaded),
    }
    _atomic_json(args.output, audit)
    print(json.dumps({
        "output": str(args.output),
        "passes": audit["passes"],
        "status": audit["status"],
        "test_opened": audit["test_opened"],
        "healthy_fit_count": audit["healthy_fit_count"],
    }, indent=2, sort_keys=True), flush=True)
    return audit


if __name__ == "__main__":
    main()
