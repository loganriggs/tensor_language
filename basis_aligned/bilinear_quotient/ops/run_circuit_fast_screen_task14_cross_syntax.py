#!/usr/bin/env python3
# BQGATE: three frozen science predictions are emitted by this targeted managed runner.
"""Run the frozen Task 14 PP<->relative interchange at two preselected sites.

Dry-run imports no model and touches neither GPU nor queue.  Real execution is
intended only through the repository's hash-bound managed queue.  The result is
create-only and the ordinary append-only fast-screen ledger receives its hash.
"""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Callable, Mapping, Sequence
from types import ModuleType

import circuit_fast_screen_candidate_task14_cross_syntax as candidate
import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_ledger as ledger
import circuit_fast_screen_managed_runner as managed
import circuit_fast_screen_producer as producer


ROOT = Path(__file__).resolve().parent.parent
REQUEST_ID = "task14-subject-verb-agreement-cross-syntax-v1"
EXPERIMENT_ID = "fast-screen-task14-subject-verb-agreement-cross-syntax-v1"
RESULT_RELATIVE = Path(
    "circuits/fast_screens/task14_subject_verb_agreement_cross_syntax_v1_result.json"
)
RESULT = ROOT / RESULT_RELATIVE
LEDGER = ROOT / "circuits/fast_screen_ledger.jsonl"
PRIOR_ART_SHA256 = (
    "fa6a1c53136601d527c9efa2c667fc70b624e8f7b8ce3544bf19342615af649a"
)
EXPECTED_AUTHORITY_SHA256 = (
    "9ec9730ffdaa00e3ed43909cf09d355e6e42e693f7d63a0176988da78ed16b95"
)

REGISTERED_PREDICTIONS = (
    (
        "pred_a_native_capability",
        "Every PP-to-relative and relative-to-PP ordered number cell has at "
        "least 85% correct native target and donor endpoints.",
    ),
    (
        "pred_b_attention11_cross_syntax",
        "Exact attention-11 output interchange has at least 75% correct "
        "direction and 40% mean recovery in every cross-syntax direction cell.",
    ),
    (
        "pred_c_head11_3_cross_syntax",
        "Exact head-11.3 pre-projection interchange has at least 75% correct "
        "direction and 40% mean recovery in every cross-syntax direction cell.",
    ),
)


@dataclass(frozen=True)
class TargetedCrossSyntaxProtocol:
    candidate: ModuleType
    request_id: str
    experiment_id: str
    result_relative: Path
    prior_art_sha256: str
    expected_authority_sha256: str
    result_schema: str
    phase: str
    partition: str
    validation_scope: str
    expected_cell_count: int
    limits: str
    novelty: str
    checkpoint_sha256: str | None = None
    config_sha256: str | None = None


DEFAULT_PROTOCOL = TargetedCrossSyntaxProtocol(
    candidate=candidate, request_id=REQUEST_ID, experiment_id=EXPERIMENT_ID,
    result_relative=RESULT_RELATIVE, prior_art_sha256=PRIOR_ART_SHA256,
    expected_authority_sha256=EXPECTED_AUTHORITY_SHA256,
    result_schema="task14_cross_syntax_interchange_result_v1",
    phase="FIT", partition="VALIDATION",
    validation_scope="new_cross_syntax_relations_not_unseen_text",
    expected_cell_count=16,
    limits=("This run tests cross-syntax transfer only. It has no unrelated "
            "is/are endpoint control and does not identify a selective grammar state."),
    novelty=("Literal PP-to-relative and relative-to-PP donor interchange at "
             "preselected attention 11 and head 11.3; prior v2 donors stayed "
             "within each construction."),
)


def _verify_checkpoint(protocol: TargetedCrossSyntaxProtocol) -> None:
    """Bind a promoted confirmation to the exact files the lazy loader will use."""
    if protocol.checkpoint_sha256 is None or protocol.config_sha256 is None:
        return
    import fastload
    _config, blob, source = fastload._paths()
    config_path = source.SNAP / "config.json"
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != protocol.config_sha256:
        raise CrossSyntaxRunError("checkpoint config hash differs from protocol")
    with open(blob, "rb") as handle:
        observed = hashlib.file_digest(handle, "sha256").hexdigest()
    if observed != protocol.checkpoint_sha256:
        raise CrossSyntaxRunError("checkpoint weights hash differs from protocol")


class CrossSyntaxRunError(ValueError):
    """Execution evidence differs from the frozen targeted plan."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_text(value: datetime) -> str:
    if value.tzinfo != timezone.utc:
        raise CrossSyntaxRunError("timestamp must be UTC")
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _chunks(
    rows: Sequence[Mapping[str, object]], candidate_module: ModuleType = candidate,
) -> list[list[Mapping[str, object]]]:
    return [
        list(rows[start:start + candidate_module.BATCH_SIZE])
        for start in range(0, len(rows), candidate_module.BATCH_SIZE)
    ]


def _batch(
    rows: Sequence[Mapping[str, object]], side: str,
) -> producer.ModelBatch:
    if side not in {"base", "donor"}:
        raise CrossSyntaxRunError("batch side must be base or donor")
    sequence = f"{side}_ids"
    answer = f"{side}_answer_id"
    foil = f"{side}_foil_id"
    position = f"{side}_semantic_position"
    return producer.ModelBatch(
        row_ids=tuple(str(row["row_id"]) for row in rows),
        side=side,  # type: ignore[arg-type]
        token_rows=tuple(tuple(int(token) for token in row[sequence]) for row in rows),
        answer_ids=tuple(int(row[answer]) for row in rows),
        foil_ids=tuple(int(row[foil]) for row in rows),
        semantic_positions=tuple(int(row[position]) for row in rows),
    )


def _finite_pair(value: object) -> tuple[float, float]:
    if not isinstance(value, tuple) or len(value) != 2 or any(
        type(item) not in {int, float} or not math.isfinite(float(item))
        for item in value
    ):
        raise CrossSyntaxRunError("backend returned a malformed logit pair")
    return float(value[0]), float(value[1])


def _site(site_id: str) -> kernel.SiteRef:
    if site_id == "attn:11":
        return kernel.SiteRef("module", site_id)
    if site_id == "attn:11:head:03":
        return kernel.SiteRef("head", site_id)
    raise CrossSyntaxRunError(f"undeclared site: {site_id}")


def _cell_capability(
    rows: Sequence[Mapping[str, object]],
    native: Mapping[tuple[str, str], tuple[float, float]],
    candidate_module: ModuleType = candidate,
    expected_cell_count: int = 16,
) -> list[dict[str, object]]:
    grouped: dict[str, list[tuple[bool, bool]]] = {}
    for row in rows:
        row_id = str(row["row_id"])
        target = native[(row_id, "base")]
        donor = native[(row_id, "donor")]
        grouped.setdefault(str(row["cell_id"]), []).append(
            (target[0] > target[1], donor[0] > donor[1])
        )
    output = []
    for cell_id, values in sorted(grouped.items()):
        count = len(values)
        target_correct = sum(target for target, _donor in values)
        donor_correct = sum(donor for _target, donor in values)
        target_accuracy = target_correct / count
        donor_accuracy = donor_correct / count
        output.append({
            "cell_id": cell_id,
            "row_count": count,
            "target_correct_count": target_correct,
            "donor_correct_count": donor_correct,
            "target_accuracy": target_accuracy,
            "donor_accuracy": donor_accuracy,
            "minimum_accuracy": candidate_module.MIN_NATIVE_CELL_ACCURACY,
            "passed": (
                target_accuracy >= candidate_module.MIN_NATIVE_CELL_ACCURACY
                and donor_accuracy >= candidate_module.MIN_NATIVE_CELL_ACCURACY
            ),
        })
    if len(output) != 4 or any(item["row_count"] != expected_cell_count for item in output):
        raise CrossSyntaxRunError("native capability cells lost exact balance")
    return output


def _score_site(
    site_id: str,
    rows: Sequence[Mapping[str, object]],
    native: Mapping[tuple[str, str], tuple[float, float]],
    patched: Mapping[str, tuple[float, float]],
    candidate_module: ModuleType = candidate,
    expected_cell_count: int = 16,
) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = {}
    evidence = []
    for row in rows:
        row_id = str(row["row_id"])
        target_pair = native[(row_id, "base")]
        donor_pair = native[(row_id, "donor")]
        patched_pair = patched[row_id]
        target_margin = target_pair[0] - target_pair[1]
        donor_margin = donor_pair[0] - donor_pair[1]
        patched_target_margin = patched_pair[0] - patched_pair[1]
        denominator = target_margin + donor_margin
        if not math.isfinite(denominator) \
                or denominator <= candidate_module.MIN_DONOR_DENOMINATOR:
            raise CrossSyntaxRunError(f"invalid native donor effect: {row_id}")
        recovery = (target_margin - patched_target_margin) / denominator
        if not math.isfinite(recovery):
            raise CrossSyntaxRunError(f"nonfinite recovery: {row_id}")
        record = {
            "row_id": row_id,
            "cell_id": str(row["cell_id"]),
            "target_endpoint_id": str(row["target_endpoint_id"]),
            "donor_endpoint_id": str(row["donor_endpoint_id"]),
            "target_native_margin": target_margin,
            "donor_native_margin": donor_margin,
            "patched_target_margin": patched_target_margin,
            "native_donor_effect": denominator,
            "recovery": recovery,
            "toward_donor": recovery > 0.0,
        }
        evidence.append(record)
        grouped.setdefault(str(row["cell_id"]), []).append(record)
    cells = []
    for cell_id, records in sorted(grouped.items()):
        recoveries = [float(record["recovery"]) for record in records]
        direction = sum(value > 0.0 for value in recoveries) / len(recoveries)
        mean_recovery = statistics.fmean(recoveries)
        cells.append({
            "cell_id": cell_id,
            "row_count": len(records),
            "direction_fraction": direction,
            "mean_recovery": mean_recovery,
            "minimum_direction_fraction": candidate_module.MIN_CELL_DIRECTION_FRACTION,
            "minimum_mean_recovery": candidate_module.MIN_CELL_MEAN_RECOVERY,
            "passed": (
                direction >= candidate_module.MIN_CELL_DIRECTION_FRACTION
                and mean_recovery >= candidate_module.MIN_CELL_MEAN_RECOVERY
            ),
        })
    passed = len(cells) == 4 and all(
        cell["row_count"] == expected_cell_count and cell["passed"] for cell in cells
    )
    return {
        "site_id": site_id,
        "passed": passed,
        "row_count": len(evidence),
        "overall_direction_fraction": (
            sum(bool(record["toward_donor"]) for record in evidence) / len(evidence)
        ),
        "overall_mean_recovery": statistics.fmean(
            float(record["recovery"]) for record in evidence
        ),
        "cells": cells,
        "evidence": evidence,
    }


def run_science(
    *,
    protocol: TargetedCrossSyntaxProtocol = DEFAULT_PROTOCOL,
    backend: producer.ExecutionBackend | None = None,
    device: str = "cuda",
    wall_clock: Callable[[], datetime] = _utc_now,
    monotonic_clock: Callable[[], float] = time.perf_counter,
) -> dict[str, object]:
    """Execute at most 8 forwards and return a strict literal result tree."""
    candidate_module = protocol.candidate
    for attribute, expected in (
        ("PHASE", protocol.phase), ("PARTITION", protocol.partition),
        ("VALIDATION_SCOPE", protocol.validation_scope),
    ):
        if getattr(candidate_module, attribute, None) != expected:
            raise CrossSyntaxRunError(
                f"candidate {attribute} differs from the frozen run protocol"
            )
    rows = candidate_module.build_rows()
    authority_sha256 = candidate_module.validate_rows(rows)
    if authority_sha256 != protocol.expected_authority_sha256:
        raise CrossSyntaxRunError(
            "derived authority differs from the reviewed runner constant"
        )
    plan = candidate_module.compile_plan(rows)
    if backend is None:
        _verify_checkpoint(protocol)
    executor = backend if backend is not None else producer.Bilin18TorchBackend.load(device)
    started_utc = wall_clock()
    started = monotonic_clock()
    forward_calls = 0
    evaluations = 0
    native: dict[tuple[str, str], tuple[float, float]] = {}
    native_evidence = []
    donor_cache: dict[tuple[str, str], object] = {}

    for side in ("base", "donor"):
        for chunk in _chunks(rows, candidate_module):
            batch = _batch(chunk, side)
            output = executor.native(batch, capture=side == "donor")
            forward_calls += 1
            evaluations += len(chunk)
            if len(output.answer_foil) != len(chunk):
                raise CrossSyntaxRunError("native output count differs from batch")
            donor_cache.update(output.captured)
            for row_id, pair in zip(batch.row_ids, output.answer_foil):
                finite = _finite_pair(pair)
                native[(row_id, side)] = finite
                native_evidence.append({
                    "row_id": row_id,
                    "side": "target" if side == "base" else "donor",
                    "answer_logit": finite[0],
                    "foil_logit": finite[1],
                })

    capability = _cell_capability(
        rows, native, candidate_module, protocol.expected_cell_count,
    )
    capability_passed = all(cell["passed"] for cell in capability)
    missing_cache = [
        f"{row['row_id']}/{site_id}"
        for row in rows
        for site_id in candidate_module.SITE_IDS
        if (str(row["row_id"]), site_id) not in donor_cache
    ]
    if missing_cache:
        raise CrossSyntaxRunError(f"donor cache lacks {missing_cache[0]}")

    site_results = []
    if capability_passed:
        for site_id in candidate_module.SITE_IDS:
            patched: dict[str, tuple[float, float]] = {}
            site = _site(site_id)
            for chunk in _chunks(rows, candidate_module):
                batch = _batch(chunk, "base")
                output = executor.patched(batch, site=site, donor_cache=donor_cache)
                forward_calls += 1
                evaluations += len(chunk)
                if len(output.answer_foil) != len(chunk):
                    raise CrossSyntaxRunError("patched output count differs from batch")
                for row_id, pair in zip(batch.row_ids, output.answer_foil):
                    patched[row_id] = _finite_pair(pair)
            site_results.append(_score_site(
                site_id, rows, native, patched, candidate_module,
                protocol.expected_cell_count,
            ))

    results_by_site = {str(item["site_id"]): item for item in site_results}
    predictions = {
        REGISTERED_PREDICTIONS[0][0]: capability_passed,
        REGISTERED_PREDICTIONS[1][0]: bool(
            results_by_site.get("attn:11", {}).get("passed")
        ),
        REGISTERED_PREDICTIONS[2][0]: bool(
            results_by_site.get("attn:11:head:03", {}).get("passed")
        ),
    }
    if not capability_passed:
        terminal, reason = "null", "native_cross_syntax_endpoints_incapable"
    elif any(result["passed"] for result in site_results):
        terminal, reason = "screen", "literal_cross_syntax_interchange_passed"
    else:
        terminal, reason = "null", "preselected_sites_failed_cross_syntax_interchange"
    finished = monotonic_clock()
    finished_utc = wall_clock()
    active_evidence_bytes = evaluations * 8
    result = {
        "schema": protocol.result_schema,
        "request_id": protocol.request_id,
        "experiment_id": protocol.experiment_id,
        "candidate_id": candidate_module.TASK_ID,
        "screen_tier_only": True,
        "execution_policy": "managed_queue_only",
        "create_only": True,
        "phase": protocol.phase,
        "partition": protocol.partition,
        "validation_scope": protocol.validation_scope,
        "correction": plan["correction"],
        "limits": protocol.limits,
        "source_sha256": dict(candidate_module.EXPECTED_SOURCE_SHA256),
        "authority_sha256": authority_sha256,
        "plan_sha256": str(plan["compiled_sha256"]),
        "started_utc": _utc_text(started_utc),
        "finished_utc": _utc_text(finished_utc),
        "serial_seconds": finished - started,
        "terminal": terminal,
        "reason": reason,
        "predictions": predictions,
        "bars": plan["score"],
        "active_price": {
            "forward_calls": forward_calls,
            "example_evaluations": evaluations,
            "backward_calls": 0,
            "model_updates": 0,
            "raw_numeric_evidence_bytes": active_evidence_bytes,
        },
        "maximum_price": plan["price"],
        "capability_cells": capability,
        "native_evidence": native_evidence,
        "site_results": site_results,
    }
    if protocol.checkpoint_sha256 is not None:
        result["checkpoint"] = {
            "weights_sha256": protocol.checkpoint_sha256,
            "config_sha256": protocol.config_sha256,
            "verified_before_model_load": backend is None,
        }
    return result


def _publish(
    result: Mapping[str, object],
    protocol: TargetedCrossSyntaxProtocol = DEFAULT_PROTOCOL,
) -> dict[str, object]:
    result_path = ROOT / protocol.result_relative
    payload = managed.atomic_create_json(result_path, result)
    result_sha256 = hashlib.sha256(payload).hexdigest()
    terminal = str(result["terminal"])
    passing = [
        str(item["site_id"])
        for item in result["site_results"]
        if item["passed"]
    ]
    selected = (
        "attn:11:head:03" if "attn:11:head:03" in passing
        else ("attn:11" if "attn:11" in passing else None)
    )
    maximum = result["maximum_price"]
    active = result["active_price"]
    entry = {
        "request_id": protocol.request_id,
        "candidate_id": protocol.candidate.TASK_ID,
        "started_utc": result["started_utc"],
        "finished_utc": result["finished_utc"],
        "serial_seconds": result["serial_seconds"],
        "prior_art_sha256": protocol.prior_art_sha256,
        "spec_sha256": result["plan_sha256"],
        "authority_sha256": result["authority_sha256"],
        "result_path": protocol.result_relative.as_posix(),
        "result_sha256": result_sha256,
        "terminal": terminal,
        "reasons": [] if terminal == "screen" else [str(result["reason"])],
        "selected_site_id": selected if terminal == "screen" else None,
        "active_forward_calls": active["forward_calls"],
        "active_example_evaluations": active["example_evaluations"],
        "active_evidence_bytes": active["raw_numeric_evidence_bytes"],
        "max_forward_calls": maximum["forward_calls"],
        "max_example_evaluations": maximum["example_evaluations"],
        "max_evidence_bytes": maximum["raw_numeric_evidence_bytes"],
        "relation": "extension",
        "novelty": protocol.novelty,
    }
    ledger.append_entry(LEDGER, entry, result_root=ROOT)
    return {
        "terminal": terminal,
        "reason": result["reason"],
        "result_path": protocol.result_relative.as_posix(),
        "result_sha256": result_sha256,
        "active_price": active,
    }


def main(protocol: TargetedCrossSyntaxProtocol = DEFAULT_PROTOCOL) -> None:
    environment = os.environ
    candidate_module = protocol.candidate
    if environment.get("BQLIB_DRYRUN") == "1" \
            or environment.get("BQLIB_NO_MODEL") == "1":
        rows = candidate_module.build_rows()
        digest = candidate_module.validate_rows(rows)
        if digest != protocol.expected_authority_sha256:
            raise CrossSyntaxRunError("reviewed authority digest changed")
        print(json.dumps(candidate_module.compile_plan(rows), sort_keys=True))
        return
    if environment.get("BQLIB_DRYRUN") is not None \
            or environment.get("BQLIB_NO_MODEL") is not None:
        raise CrossSyntaxRunError("dry-run flags must be absent or exactly 1")
    print(json.dumps(_publish(run_science(protocol=protocol), protocol), sort_keys=True))


def cli(protocol: TargetedCrossSyntaxProtocol = DEFAULT_PROTOCOL) -> None:
    """Expose a real dry-run flag; never silently ignore command-line arguments."""
    arguments = sys.argv[1:]
    if arguments == ["--dry-run"]:
        if os.environ.get("BQLIB_DRYRUN") not in {None, "1"}:
            raise CrossSyntaxRunError("conflicting dry-run environment")
        os.environ["BQLIB_DRYRUN"] = "1"
    elif arguments:
        raise CrossSyntaxRunError(f"unknown command-line arguments: {arguments}")
    main(protocol)


if __name__ == "__main__":
    cli()
