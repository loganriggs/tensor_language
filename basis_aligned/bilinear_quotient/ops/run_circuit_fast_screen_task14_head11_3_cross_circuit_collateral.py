#!/usr/bin/env python3
# BQGATE: frozen cross-circuit removal predictions are emitted by this targeted managed runner.
"""Measure collateral from literal Task14 head-11.3 removal on two other behaviors."""

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

import circuit_fast_screen_candidate_task14_head11_3_cross_circuit_collateral as candidate
import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_ledger as ledger
import circuit_fast_screen_managed_runner as managed
import circuit_fast_screen_producer as producer


ROOT = Path(__file__).resolve().parent.parent
REQUEST_ID = "task14-head11-3-cross-circuit-collateral-v1"
EXPERIMENT_ID = "fast-screen-task14-head11-3-cross-circuit-collateral-v1"
RESULT_RELATIVE = Path("circuits/fast_screens/task14_head11_3_cross_circuit_collateral_v1_result.json")
RESULT = ROOT / RESULT_RELATIVE
LEDGER = ROOT / "circuits/fast_screen_ledger.jsonl"
PRIOR_ART_SHA256 = "ecadfcb634710bcdb8e2061d6b85d911547034d3ead5f587e6b4156d568d613d"
CHECKPOINT_SHA256 = "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
CONFIG_SHA256 = "428042bfd807ba36f8b4326395440fbbebe52cd3d040212e6fef14a4fdf2d83c"


class CollateralBackend(Protocol):
    def native(self, batch: producer.ModelBatch, *, capture: bool) -> producer.BatchOutput: ...
    def patched(self, batch: producer.ModelBatch, *, site: kernel.SiteRef,
                donor_cache: Mapping[tuple[str, str], object]) -> producer.BatchOutput: ...


class CollateralRunError(ValueError):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _utc(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _verify_checkpoint() -> None:
    import fastload
    _config, blob, source = fastload._paths()
    if hashlib.sha256((source.SNAP / "config.json").read_bytes()).hexdigest() != CONFIG_SHA256:
        raise CollateralRunError("checkpoint config hash changed")
    with open(blob, "rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != CHECKPOINT_SHA256:
            raise CollateralRunError("checkpoint weights hash changed")


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
        raise CollateralRunError("backend returned malformed logit pair")
    return float(value[0]), float(value[1])


def _margin(value: tuple[float, float]) -> float:
    return value[0] - value[1]


def _norm(value: object) -> float:
    if not hasattr(value, "float") or not hasattr(value, "norm"):
        raise CollateralRunError("captured head is not a tensor")
    result = float(value.float().norm())
    if not math.isfinite(result):
        raise CollateralRunError("captured head norm is nonfinite")
    return result


def _zero_cache(rows, captured) -> tuple[dict[tuple[str, str], object], dict[str, float]]:
    zeros, norms = {}, {}
    for row in rows:
        row_id = str(row["row_id"])
        key = (row_id, candidate.SITE_ID)
        value = captured.get(key)
        if value is None or not hasattr(value, "mul"):
            raise CollateralRunError(f"native capture lacks {key}")
        norms[row_id] = _norm(value)
        zero = value.mul(0)
        if _norm(zero) != 0.0:
            raise CollateralRunError("zero replacement cache is not exactly zero")
        zeros[key] = zero
    return zeros, norms


def _score(rows, native, removed, replay, norms) -> dict[str, object]:
    replay_error = max(
        abs(a - b)
        for row in rows
        for a, b in zip(native[str(row["row_id"])], replay[str(row["row_id"])])
    )
    behavior_results, evidence = [], []
    for behavior in candidate.BEHAVIORS:
        subset = [row for row in rows if row["behavior"] == behavior]
        native_margins = [_margin(native[str(row["row_id"])]) for row in subset]
        scale = statistics.median(native_margins)
        if not math.isfinite(scale) or scale <= candidate.MIN_NATIVE_MARGIN:
            raise CollateralRunError(f"invalid native scale for {behavior}")
        effects, flips = [], 0
        for row, native_margin in zip(subset, native_margins):
            row_id = str(row["row_id"])
            removed_margin = _margin(removed[row_id])
            effect = abs(removed_margin - native_margin) / scale
            effects.append(effect)
            flips += removed_margin <= 0
            evidence.append({
                "row_id": row_id, "behavior": behavior,
                "native_margin": native_margin, "removed_margin": removed_margin,
                "normalized_absolute_effect": effect, "answer_flipped": removed_margin <= 0,
                "native_head_norm": norms[row_id],
            })
        median_effect = statistics.median(effects)
        rows_under = sum(effect <= candidate.MAX_ROW_EFFECT for effect in effects)
        passed = (
            median_effect <= candidate.MAX_MEDIAN_EFFECT
            and rows_under >= candidate.MIN_ROWS_UNDER_EFFECT
            and flips <= candidate.MAX_ANSWER_FLIPS
        )
        behavior_results.append({
            "behavior": behavior, "row_count": len(subset), "native_scale": scale,
            "minimum_native_margin": min(native_margins),
            "median_normalized_absolute_effect": median_effect,
            "rows_at_or_below_0_25": rows_under, "answer_flips": flips,
            "passed_preservation": passed,
        })
    return {
        "replay_max_abs_logit_error": replay_error,
        "minimum_native_head_norm": min(norms.values()),
        "behavior_results": behavior_results,
        "evidence": evidence,
    }


def run_science(*, backend: CollateralBackend | None = None, device: str = "cuda",
                wall_clock=_now, monotonic_clock=time.perf_counter) -> dict[str, object]:
    rows = candidate.build_rows()
    authority_sha = candidate.validate_rows(rows)
    plan = candidate.compile_plan(rows)
    if backend is None:
        _verify_checkpoint()
    executor = backend if backend is not None else producer.Bilin18TorchBackend.load(device)
    batch = _batch(rows)
    started_utc, started = wall_clock(), monotonic_clock()
    native_output = executor.native(batch, capture=True)
    native = {row_id: _pair(pair) for row_id, pair in zip(batch.row_ids, native_output.answer_foil)}
    zeros, norms = _zero_cache(rows, native_output.captured)
    site = kernel.SiteRef("head", candidate.SITE_ID)
    removed_output = executor.patched(batch, site=site, donor_cache=zeros)
    replay_output = executor.patched(batch, site=site, donor_cache=native_output.captured)
    removed = {row_id: _pair(pair) for row_id, pair in zip(batch.row_ids, removed_output.answer_foil)}
    replay = {row_id: _pair(pair) for row_id, pair in zip(batch.row_ids, replay_output.answer_foil)}
    if any(len(values) != len(rows) for values in (native, removed, replay)):
        raise CollateralRunError("backend output count differs from authority")
    scores = _score(rows, native, removed, replay, norms)
    native_capable = all(item["minimum_native_margin"] > candidate.MIN_NATIVE_MARGIN
                         for item in scores["behavior_results"])
    replay_passed = scores["replay_max_abs_logit_error"] <= candidate.MAX_REPLAY_LOGIT_ERROR
    hook_live = scores["minimum_native_head_norm"] >= candidate.MIN_NATIVE_HEAD_NORM
    preserved = all(item["passed_preservation"] for item in scores["behavior_results"])
    predictions = {
        "pred_a_native_capability": native_capable,
        "pred_b_native_head_replay": replay_passed,
        "pred_c_head_hook_is_live": hook_live,
        "pred_d_numbered_list_preserved": scores["behavior_results"][0]["passed_preservation"],
        "pred_e_bracket_preserved": scores["behavior_results"][1]["passed_preservation"],
    }
    if not replay_passed:
        terminal, reason = "invalid", "native_head_replay_failed"
    elif not native_capable:
        terminal, reason = "null", "collateral_behavior_native_capability_failed"
    elif not hook_live:
        terminal, reason = "inconclusive", "head11_3_hook_not_live_on_collateral_rows"
    elif preserved:
        terminal, reason = "screen", "both_unrelated_behaviors_preserved"
    else:
        terminal, reason = "null", "head11_3_removal_has_cross_circuit_collateral"
    finished, finished_utc = monotonic_clock(), wall_clock()
    return {
        "schema": "task14_head11_3_cross_circuit_collateral_result_v1",
        "request_id": REQUEST_ID, "experiment_id": EXPERIMENT_ID,
        "candidate_id": "subject_verb.number_agreement.head11_3_cross_circuit_collateral",
        "screen_tier_only": True,
        "execution_policy": "managed_queue_only", "create_only": True,
        "phase": candidate.PHASE, "partition": candidate.PARTITION,
        "authority_sha256": authority_sha, "plan_sha256": plan["compiled_sha256"],
        "source_sha256": candidate.EXPECTED_SOURCE_SHA256,
        "checkpoint": {"weights_sha256": CHECKPOINT_SHA256, "config_sha256": CONFIG_SHA256,
                       "verified_before_model_load": backend is None},
        "started_utc": _utc(started_utc), "finished_utc": _utc(finished_utc),
        "serial_seconds": finished - started, "terminal": terminal, "reason": reason,
        "predictions": predictions, "bars": plan["scoring"], **scores,
        "active_price": plan["price"],
        "maximum_price": plan["price"],
        "limits": "Two unrelated behaviors establish narrow collateral breadth, not universal selectivity.",
    }


def _publish(result: Mapping[str, object]) -> dict[str, object]:
    payload = managed.atomic_create_json(RESULT, result)
    result_sha = hashlib.sha256(payload).hexdigest()
    terminal = str(result["terminal"])
    entry = {
        "request_id": REQUEST_ID,
        "candidate_id": "subject_verb.number_agreement.head11_3_cross_circuit_collateral",
        "started_utc": result["started_utc"], "finished_utc": result["finished_utc"],
        "serial_seconds": result["serial_seconds"], "prior_art_sha256": PRIOR_ART_SHA256,
        "spec_sha256": result["plan_sha256"], "authority_sha256": result["authority_sha256"],
        "result_path": RESULT_RELATIVE.as_posix(), "result_sha256": result_sha,
        "terminal": terminal, "reasons": [] if terminal == "screen" else [str(result["reason"])],
        "selected_site_id": candidate.SITE_ID if terminal == "screen" else None,
        "active_forward_calls": 3, "active_example_evaluations": 96,
        "active_evidence_bytes": 768, "max_forward_calls": 3,
        "max_example_evaluations": 96, "max_evidence_bytes": 768,
        "relation": "extension",
        "novelty": "Exact head11.3 zero-removal collateral on held-out numbered-list and bracket behaviors.",
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
        raise CollateralRunError("dry-run environment flags must be absent or exactly 1")
    if args.dry_run or "1" in flags.values():
        print(json.dumps(candidate.compile_plan(), sort_keys=True)); return
    print(json.dumps(_publish(run_science()), sort_keys=True))


if __name__ == "__main__":
    main()
