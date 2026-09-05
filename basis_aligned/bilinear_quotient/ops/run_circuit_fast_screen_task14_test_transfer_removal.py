#!/usr/bin/env python3
# BQGATE: frozen TEST transfer, removal, capability, and replay predictions are emitted here.
"""Open Task14 TEST once for head-11.3 transfer and literal removal."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from types import ModuleType
from typing import Mapping, Protocol, Sequence

import circuit_fast_screen_candidate_task14_test_cross_syntax as candidate
import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_ledger as ledger
import circuit_fast_screen_managed_runner as managed
import circuit_fast_screen_producer as producer
import run_circuit_fast_screen_task14_cross_syntax as transfer


ROOT = Path(__file__).resolve().parent.parent
REQUEST_ID = "task14-test-cross-noun-head11-3-transfer-removal-v1"
EXPERIMENT_ID = "fast-screen-task14-test-cross-noun-head11-3-transfer-removal-v1"
RESULT_RELATIVE = Path(
    "circuits/fast_screens/task14_test_cross_noun_head11_3_transfer_removal_v1_result.json"
)
RESULT = ROOT / RESULT_RELATIVE
LEDGER = ROOT / "circuits/fast_screen_ledger.jsonl"
PRIOR_ART_SHA256 = "db6e5bd9c7a8592345fa75f467e60e8fddfe140c7f48a1355b75fef5a7165b49"
HEAD_SITE_ID = "attn:11:head:03"
CHECKPOINT_SHA256 = "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
CONFIG_SHA256 = "428042bfd807ba36f8b4326395440fbbebe52cd3d040212e6fef14a4fdf2d83c"


@dataclass(frozen=True)
class RunProtocol:
    candidate: ModuleType
    request_id: str
    experiment_id: str
    result_relative: Path
    prior_art_sha256: str
    result_schema: str
    novelty: str
    limits: str


TEST_PROTOCOL = RunProtocol(
    candidate=candidate,
    request_id=REQUEST_ID,
    experiment_id=EXPERIMENT_ID,
    result_relative=RESULT_RELATIVE,
    prior_art_sha256=PRIOR_ART_SHA256,
    result_schema="task14_test_cross_noun_head11_3_transfer_removal_result_v1",
    novelty="First unopened TEST cross-noun transfer plus literal removal for preselected Task14 head11.3.",
    limits="TEST only. OOD remains unopened and unchanged; unrelated-behavior selectivity comes from the prior collateral event.",
)


class Backend(Protocol):
    def native(self, batch: producer.ModelBatch, *, capture: bool) -> producer.BatchOutput: ...
    def patched(self, batch: producer.ModelBatch, *, site: kernel.SiteRef,
                donor_cache: Mapping[tuple[str, str], object]) -> producer.BatchOutput: ...


class TestRunError(ValueError):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _utc(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _verify_checkpoint() -> None:
    import fastload
    _config, blob, source = fastload._paths()
    if hashlib.sha256((source.SNAP / "config.json").read_bytes()).hexdigest() != CONFIG_SHA256:
        raise TestRunError("checkpoint config hash changed")
    with open(blob, "rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != CHECKPOINT_SHA256:
            raise TestRunError("checkpoint weights hash changed")


def _pair(value: object) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2 or any(
        type(item) not in {int, float} or not math.isfinite(float(item)) for item in value
    ):
        raise TestRunError("backend returned malformed logit pair")
    return float(value[0]), float(value[1])


def _margin(value: tuple[float, float]) -> float:
    return value[0] - value[1]


def _norm(value: object) -> float:
    if not hasattr(value, "float") or not hasattr(value, "norm"):
        raise TestRunError("captured head is not a tensor")
    result = float(value.float().norm())
    if not math.isfinite(result):
        raise TestRunError("captured head norm is nonfinite")
    return result


def _collect_native(executor, rows, side, candidate_module):
    pairs, cache = {}, {}
    calls = evaluations = 0
    for chunk in transfer._chunks(rows, candidate_module):
        batch = transfer._batch(chunk, side)
        output = executor.native(batch, capture=True)
        calls += 1; evaluations += len(chunk); cache.update(output.captured)
        pairs.update({row_id: _pair(pair) for row_id, pair in zip(batch.row_ids, output.answer_foil)})
    return pairs, cache, calls, evaluations


def _collect_patched(executor, rows, cache, candidate_module):
    pairs = {}; calls = evaluations = 0
    site = kernel.SiteRef("head", HEAD_SITE_ID)
    for chunk in transfer._chunks(rows, candidate_module):
        batch = transfer._batch(chunk, "base")
        output = executor.patched(batch, site=site, donor_cache=cache)
        calls += 1; evaluations += len(chunk)
        pairs.update({row_id: _pair(pair) for row_id, pair in zip(batch.row_ids, output.answer_foil)})
    return pairs, calls, evaluations


def _zero_cache(rows, cache):
    zeros, norms = {}, {}
    for row in rows:
        row_id = str(row["row_id"]); key = (row_id, HEAD_SITE_ID)
        value = cache.get(key)
        if value is None or not hasattr(value, "mul"):
            raise TestRunError(f"native base capture lacks {key}")
        norms[row_id] = _norm(value)
        zeros[key] = value.mul(0)
        if _norm(zeros[key]) != 0.0:
            raise TestRunError("zero replacement cache is not exactly zero")
    return zeros, norms


def _removal_score(rows, native_base, removed, plan, candidate_module):
    scale = statistics.median(abs(_margin(native_base[str(row["row_id"])])) for row in rows)
    if scale <= candidate_module.MIN_DONOR_DENOMINATOR:
        raise TestRunError("native removal scale is too small")
    grouped, evidence = {}, []
    for row in rows:
        row_id = str(row["row_id"])
        native_margin = _margin(native_base[row_id]); removed_margin = _margin(removed[row_id])
        damage = (native_margin - removed_margin) / scale
        record = {"row_id": row_id, "cell_id": row["cell_id"],
                  "native_margin": native_margin, "removed_margin": removed_margin,
                  "normalized_removal_damage": damage}
        evidence.append(record); grouped.setdefault(str(row["cell_id"]), []).append(damage)
    bars = plan["score"]
    cells = []
    for cell_id, values in sorted(grouped.items()):
        median_damage = statistics.median(values)
        positive = sum(value > 0 for value in values) / len(values)
        cells.append({"cell_id": cell_id, "row_count": len(values),
                      "median_normalized_damage": median_damage,
                      "positive_damage_fraction": positive,
                      "passed": median_damage >= bars["minimum_cell_median_normalized_removal_damage"]
                      and positive >= bars["minimum_cell_positive_removal_fraction"]})
    if len(cells) != 4 or any(cell["row_count"] != 16 for cell in cells):
        raise TestRunError("removal cells lost exact balance")
    return {"target_scale": scale, "cells": cells,
            "passed": all(cell["passed"] for cell in cells), "evidence": evidence}


def run_science(*, protocol: RunProtocol = TEST_PROTOCOL,
                backend: Backend | None = None, device: str = "cuda",
                wall_clock=_now, monotonic_clock=time.perf_counter) -> dict[str, object]:
    candidate_module = protocol.candidate
    rows = candidate_module.build_rows(); authority_sha = candidate_module.validate_rows(rows)
    plan = candidate_module.compile_plan(rows)
    if backend is None:
        _verify_checkpoint()
    executor = backend if backend is not None else producer.Bilin18TorchBackend.load(device)
    started_utc, started = wall_clock(), monotonic_clock()
    native_base, base_cache, calls, evaluations = _collect_native(
        executor, rows, "base", candidate_module,
    )
    native_donor, donor_cache, c, e = _collect_native(
        executor, rows, "donor", candidate_module,
    )
    calls += c; evaluations += e
    interchanged, c, e = _collect_patched(executor, rows, donor_cache, candidate_module)
    calls += c; evaluations += e
    zeros, norms = _zero_cache(rows, base_cache)
    removed, c, e = _collect_patched(executor, rows, zeros, candidate_module)
    calls += c; evaluations += e
    replay, c, e = _collect_patched(executor, rows, base_cache, candidate_module)
    calls += c; evaluations += e
    native_for_shared = {(row_id, "base"): value for row_id, value in native_base.items()}
    native_for_shared.update({(row_id, "donor"): value for row_id, value in native_donor.items()})
    capability = transfer._cell_capability(rows, native_for_shared, candidate_module, 16)
    transfer_score = transfer._score_site(
        HEAD_SITE_ID, rows, native_for_shared, interchanged, candidate_module, 16,
    )
    removal = _removal_score(rows, native_base, removed, plan, candidate_module)
    replay_error = max(abs(a-b) for row in rows for a,b in zip(
        native_base[str(row["row_id"])], replay[str(row["row_id"])],
    ))
    capability_passed = all(cell["passed"] for cell in capability)
    transfer_passed = transfer_score["passed"]
    replay_passed = replay_error <= plan["score"]["maximum_native_head_replay_absolute_logit_error"]
    hook_live = min(norms.values()) > 0.0
    predictions = {
        "pred_a_native_capability": capability_passed,
        "pred_b_head11_3_cross_noun_transfer": transfer_passed,
        "pred_c_head11_3_literal_removal": removal["passed"],
        "pred_d_native_head_replay": replay_passed,
        "pred_e_head_hook_live": hook_live,
    }
    if not replay_passed:
        terminal, reason = "invalid", "native_head_replay_failed"
    elif not hook_live:
        terminal, reason = "invalid", "head11_3_hook_inactive"
    elif not capability_passed:
        terminal, reason = "null", "TEST_native_capability_failed"
    elif not transfer_passed:
        terminal, reason = "null", "TEST_cross_noun_transfer_failed"
    elif not removal["passed"]:
        terminal, reason = "null", "TEST_literal_removal_failed"
    else:
        terminal, reason = "screen", "TEST_transfer_and_removal_passed"
    finished, finished_utc = monotonic_clock(), wall_clock()
    active_price = {"forward_calls": calls, "example_evaluations": evaluations,
                    "backward_calls": 0, "model_updates": 0,
                    "raw_numeric_evidence_bytes": evaluations * 8}
    if active_price != plan["price"]:
        raise TestRunError(f"active price differs from frozen plan: {active_price}")
    return {
        "schema": protocol.result_schema,
        "request_id": protocol.request_id, "experiment_id": protocol.experiment_id,
        "candidate_id": candidate_module.TASK_ID, "screen_tier_only": True,
        "execution_policy": "managed_queue_only", "create_only": True,
        "phase": candidate_module.PHASE, "partition": candidate_module.PARTITION,
        "validation_scope": candidate_module.VALIDATION_SCOPE,
        "authority_sha256": authority_sha, "plan_sha256": plan["compiled_sha256"],
        "source_sha256": candidate_module.EXPECTED_SOURCE_SHA256,
        "checkpoint": {"weights_sha256": CHECKPOINT_SHA256, "config_sha256": CONFIG_SHA256,
                       "verified_before_model_load": backend is None},
        "started_utc": _utc(started_utc), "finished_utc": _utc(finished_utc),
        "serial_seconds": finished-started, "terminal": terminal, "reason": reason,
        "predictions": predictions, "bars": plan["score"], "capability_cells": capability,
        "transfer": transfer_score, "removal": removal,
        "replay_max_abs_logit_error": replay_error,
        "minimum_native_head_norm": min(norms.values()),
        "native_base": native_base, "native_donor": native_donor,
        "active_price": active_price, "maximum_price": plan["price"],
        "limits": protocol.limits,
    }


def _publish(result: Mapping[str, object], protocol: RunProtocol = TEST_PROTOCOL) -> dict[str, object]:
    result_path = ROOT / protocol.result_relative
    payload = managed.atomic_create_json(result_path, result)
    result_sha = hashlib.sha256(payload).hexdigest()
    terminal = str(result["terminal"])
    entry = {
        "request_id": protocol.request_id, "candidate_id": protocol.candidate.TASK_ID,
        "started_utc": result["started_utc"], "finished_utc": result["finished_utc"],
        "serial_seconds": result["serial_seconds"], "prior_art_sha256": protocol.prior_art_sha256,
        "spec_sha256": result["plan_sha256"], "authority_sha256": result["authority_sha256"],
        "result_path": protocol.result_relative.as_posix(), "result_sha256": result_sha,
        "terminal": terminal, "reasons": [] if terminal == "screen" else [str(result["reason"])],
        "selected_site_id": HEAD_SITE_ID if terminal == "screen" else None,
        "active_forward_calls": result["active_price"]["forward_calls"],
        "active_example_evaluations": result["active_price"]["example_evaluations"],
        "active_evidence_bytes": result["active_price"]["raw_numeric_evidence_bytes"],
        "max_forward_calls": result["maximum_price"]["forward_calls"],
        "max_example_evaluations": result["maximum_price"]["example_evaluations"],
        "max_evidence_bytes": result["maximum_price"]["raw_numeric_evidence_bytes"],
        "relation": "extension",
        "novelty": protocol.novelty,
    }
    ledger.append_entry(LEDGER, entry, result_root=ROOT)
    return {"terminal": terminal, "reason": result["reason"],
            "result_path": protocol.result_relative.as_posix(), "result_sha256": result_sha,
            "active_price": result["active_price"]}


def cli(protocol: RunProtocol, argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    flags = {name: os.environ.get(name) for name in ("BQLIB_DRYRUN", "BQLIB_NO_MODEL")}
    if any(value not in {None, "1"} for value in flags.values()):
        raise TestRunError("dry-run environment flags must be absent or exactly 1")
    if args.dry_run or "1" in flags.values():
        print(json.dumps(protocol.candidate.compile_plan(), sort_keys=True)); return
    if protocol.prior_art_sha256 == "PENDING_PREREGISTRATION_HASH":
        raise TestRunError("prior-art receipt hash was not frozen before execution")
    print(json.dumps(_publish(run_science(protocol=protocol), protocol), sort_keys=True))


def main(argv: Sequence[str] | None = None) -> None:
    cli(TEST_PROTOCOL, argv)


if __name__ == "__main__":
    main()
