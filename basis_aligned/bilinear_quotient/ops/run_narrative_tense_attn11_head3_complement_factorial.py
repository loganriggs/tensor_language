#!/usr/bin/env python3
"""Split narrative tense's frozen attention-11 effect into H3 and its complement."""

# BQGATE: EXPERIMENT pred_a_instrument_live pred_b_shared_copular_service pred_c_task_split

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

import circuit_fast_screen_candidate_narrative_tense as candidate
import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_managed_runner as managed
import circuit_fast_screen_producer as producer


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/narrative_tense_attn11_head3_complement_factorial_v1.json"
PARENT = ROOT / "circuits/fast_screens/narrative_tense_past_vs_present_v2_result.json"
OUT = ROOT / "circuits/fast_screens/narrative_tense_attn11_head3_complement_factorial_v1_result.json"
EXPECTED_PRIOR_ART_SHA256 = "704c1207c110e7f2384db1827d9a4dd5b03385b65586e7f5a87aaa7914b933e2"
EXPECTED_PARENT_SHA256 = "5466980e1aa0a59538e4e8fcfb29457814c01e91cbe39bf41a2d42140fc7e71a"
EXPECTED_AUTHORITY_SHA256 = "745910973b77cfec0dd945920f72b2c19c06a6e51464df11bd969165233d4a77"
LAYER = 11
HEAD = 3
COMPLEMENT = tuple(head for head in range(9) if head != HEAD)
BATCH_SIZE = 32
MIN_HEAD_FRACTION_OF_FULL = 0.50
MIN_HEAD_DIRECTION_FRACTION = 0.75


class FactorialError(ValueError):
    """Frozen evidence or newly produced evidence violated the screen contract."""


class Backend(Protocol):
    def native(self, batch: producer.ModelBatch, *, capture: bool) -> producer.BatchOutput: ...

    def patched_heads(
        self, batch: producer.ModelBatch, *, layer: int, heads: Sequence[int],
        donor_cache: Mapping[tuple[str, str], object],
    ) -> producer.BatchOutput: ...


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _margin(item: Mapping[str, object]) -> float:
    return float(item["answer_logit"]) - float(item["foil_logit"])


def _pair(value: object) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2 or any(
        type(x) not in {int, float} or not math.isfinite(float(x)) for x in value
    ):
        raise FactorialError("backend returned a malformed answer/foil pair")
    return float(value[0]), float(value[1])


def _binary_ce_from_margin(margin: float, *, answer_is_first: bool) -> float:
    """Two-answer contrast CE; this is not full-vocabulary CE."""
    signed = margin if answer_is_first else -margin
    return math.log1p(math.exp(-abs(signed))) + max(-signed, 0.0)


def _load_closure() -> tuple[list[dict[str, object]], dict[str, object]]:
    if _sha256(PRIOR_ART) != EXPECTED_PRIOR_ART_SHA256:
        raise FactorialError("prior-art receipt hash changed")
    if _sha256(PARENT) != EXPECTED_PARENT_SHA256:
        raise FactorialError("frozen narrative result hash changed")
    rows = candidate.build_rows()
    if candidate.validate_rows(rows) != EXPECTED_AUTHORITY_SHA256:
        raise FactorialError("frozen narrative authority changed")
    old = json.loads(PARENT.read_text())
    if old.get("authority_sha256") != EXPECTED_AUTHORITY_SHA256 \
            or old.get("terminal") != "screen" \
            or not all(old.get("predictions", {}).values()):
        raise FactorialError("parent result is not the frozen capable screen")
    ids = {str(row["row_id"]) for row in rows}
    native = old.get("run", {}).get("native_logits", [])
    full = [
        item for item in old.get("run", {}).get("intervention_logits", [])
        if item.get("site", {}).get("site_id") == "attn:11"
    ]
    if len(native) != 2 * len(rows) or {
        (str(item["row_id"]), str(item["side"])) for item in native
    } != {(row_id, side) for row_id in ids for side in ("base", "donor")}:
        raise FactorialError("parent native logits lost exact row/side coverage")
    if len(full) != len(rows) or {str(item["row_id"]) for item in full} != ids:
        raise FactorialError("parent full-attention-11 logits lost exact row coverage")
    cells = old.get("run", {}).get("capability_cells", [])
    if not cells or not all(item.get("passed") is True for item in cells):
        raise FactorialError("parent native capability is not live in every cell")
    return rows, old


def compile_plan() -> dict[str, object]:
    rows, _old = _load_closure()
    chunks = math.ceil(len(rows) / BATCH_SIZE)
    return {
        "schema": "narrative_tense_attn11_head3_complement_factorial_plan_v1",
        "candidate_id": "narrative_tense.attn11_head3_complement_factorial",
        "model_loaded": False,
        "gpu_accessed": False,
        "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_ART_SHA256,
        "parent_result_sha256": EXPECTED_PARENT_SHA256,
        "authority_sha256": EXPECTED_AUTHORITY_SHA256,
        "layer": LAYER,
        "head": HEAD,
        "complement_heads": list(COMPLEMENT),
        "corners": {"empty": "frozen native", "head3": "new", "other8": "new", "full": "frozen attn11"},
        "bars": {
            "minimum_head3_fraction_of_full_per_target_cell": MIN_HEAD_FRACTION_OF_FULL,
            "minimum_head3_donor_direction_fraction_per_target_cell": MIN_HEAD_DIRECTION_FRACTION,
            "maximum_head3_control_movement": "the frozen full-attention-11 movement in the same control family",
        },
        "price": {
            "model_forwards": 3 * chunks,
            "example_evaluations": 3 * len(rows),
            "backwards": 0,
            "parameter_updates": 0,
        },
        "interpretation_limit": (
            "A positive result licenses below-head tests but does not establish a shared "
            "agreement/tense state or treat a native attention head as the final basis."
        ),
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


def _chunks(rows: Sequence[dict[str, object]]):
    for start in range(0, len(rows), BATCH_SIZE):
        yield list(rows[start:start + BATCH_SIZE])


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise FactorialError("attempted to summarize an empty cell")
    return statistics.fmean(values)


def score(
    rows: Sequence[Mapping[str, object]], old: Mapping[str, object],
    head_pairs: Mapping[str, tuple[float, float]],
    complement_pairs: Mapping[str, tuple[float, float]],
) -> dict[str, object]:
    native = {
        (str(item["row_id"]), str(item["side"])): item
        for item in old["run"]["native_logits"]  # type: ignore[index]
    }
    full = {
        str(item["row_id"]): item
        for item in old["run"]["intervention_logits"]  # type: ignore[index]
        if item["site"]["site_id"] == "attn:11"
    }
    evidence: list[dict[str, object]] = []
    target_groups: dict[str, list[dict[str, object]]] = {}
    control_groups: dict[str, list[dict[str, object]]] = {"P": [], "C": []}
    for row in rows:
        row_id = str(row["row_id"])
        family = str(row["transform_id"])
        base_margin = _margin(native[(row_id, "base")])
        donor_margin = _margin(native[(row_id, "donor")])
        corners = {
            "empty": base_margin,
            "head3": head_pairs[row_id][0] - head_pairs[row_id][1],
            "other8": complement_pairs[row_id][0] - complement_pairs[row_id][1],
            "full": _margin(full[row_id]),
        }
        if bool(row["answer_changes"]):
            recoveries = {
                name: kernel.signed_pairwise_donor_recovery(-base_margin, donor_margin, -margin)
                for name, margin in corners.items()
            }
            base_donor_ce = _binary_ce_from_margin(base_margin, answer_is_first=False)
            donor_ce = _binary_ce_from_margin(donor_margin, answer_is_first=True)
            ce_denominator = base_donor_ce - donor_ce
            if ce_denominator <= 0:
                raise FactorialError("native donor-answer contrast CE denominator is not positive")
            ce_recoveries = {
                name: (base_donor_ce - _binary_ce_from_margin(margin, answer_is_first=False)) / ce_denominator
                for name, margin in corners.items()
            }
        else:
            scale = abs(base_margin) + abs(donor_margin)
            if scale <= kernel.MIN_DONOR_DENOMINATOR:
                raise FactorialError("control normalization scale is zero")
            recoveries = {name: (margin - base_margin) / scale for name, margin in corners.items()}
            ce_recoveries = {}
        interaction = recoveries["full"] - recoveries["head3"] - recoveries["other8"] + recoveries["empty"]
        ce_interaction = None if not ce_recoveries else (
            ce_recoveries["full"] - ce_recoveries["head3"]
            - ce_recoveries["other8"] + ce_recoveries["empty"]
        )
        item = {
            "row_id": row_id,
            "family": family,
            "cell_id": str(row["capability_cell_id"]),
            "margins": corners,
            "margin_recovery": recoveries,
            "margin_interaction": interaction,
            "binary_contrast_ce_recovery": ce_recoveries or None,
            "binary_contrast_ce_interaction": ce_interaction,
        }
        evidence.append(item)
        if family in {"A1", "A2"}:
            target_groups.setdefault(str(row["capability_cell_id"]), []).append(item)
        else:
            control_groups[family].append(item)

    cells = []
    for cell_id, items in sorted(target_groups.items()):
        means = {
            corner: _mean([float(item["margin_recovery"][corner]) for item in items])
            for corner in ("empty", "head3", "other8", "full")
        }
        full_mean = means["full"]
        ratio = means["head3"] / full_mean if full_mean > kernel.MIN_DONOR_DENOMINATOR else float("-inf")
        direction = _mean([float(item["margin_recovery"]["head3"]) > 0 for item in items])
        ce_means = {
            corner: _mean([float(item["binary_contrast_ce_recovery"][corner]) for item in items])
            for corner in ("empty", "head3", "other8", "full")
        }
        passed = ratio >= MIN_HEAD_FRACTION_OF_FULL and direction >= MIN_HEAD_DIRECTION_FRACTION
        cells.append({
            "cell_id": cell_id,
            "row_count": len(items),
            "mean_margin_recovery": means,
            "head3_fraction_of_full": ratio,
            "head3_donor_direction_fraction": direction,
            "mean_margin_interaction": _mean([float(item["margin_interaction"]) for item in items]),
            "mean_binary_contrast_ce_recovery": ce_means,
            "mean_binary_contrast_ce_interaction": _mean([
                float(item["binary_contrast_ce_interaction"]) for item in items
            ]),
            "shared_service_cell_passed": passed,
        })

    controls: dict[str, object] = {}
    controls_pass = True
    for family, items in control_groups.items():
        head = _mean([abs(float(item["margin_recovery"]["head3"])) for item in items])
        full_move = _mean([abs(float(item["margin_recovery"]["full"])) for item in items])
        passed = head <= full_move + 1e-12
        controls[family] = {
            "row_count": len(items),
            "head3_mean_absolute_normalized_movement": head,
            "full_attn11_mean_absolute_normalized_movement": full_move,
            "head3_no_larger_than_full": passed,
        }
        controls_pass &= passed
    shared = all(cell["shared_service_cell_passed"] for cell in cells) and controls_pass
    return {
        "cells": cells,
        "controls": controls,
        "evidence": evidence,
        "predictions": {
            "pred_a_instrument_live": True,
            "pred_b_shared_copular_service": shared,
            "pred_c_task_split": not shared,
        },
        "terminal": "shared_copular_service" if shared else "task_split",
    }


def run_science(*, backend: Backend | None = None, device: str = "cuda", clock=time.perf_counter) -> dict[str, object]:
    rows, old = _load_closure()
    executor = backend if backend is not None else producer.Bilin18TorchBackend.load(device)
    cache: dict[tuple[str, str], object] = {}
    head_pairs: dict[str, tuple[float, float]] = {}
    complement_pairs: dict[str, tuple[float, float]] = {}
    forwards = evaluations = 0
    started = clock()
    for chunk in _chunks(rows):
        batch = _batch(chunk, "donor")
        output = executor.native(batch, capture=True)
        forwards += 1
        evaluations += len(chunk)
        cache.update(output.captured)
    required = {
        (str(row["row_id"]), f"attn:{LAYER:02d}:head:{head:02d}")
        for row in rows for head in range(9)
    }
    if not required.issubset(cache):
        raise FactorialError("donor capture lacks one or more attention-11 head slices")
    for chunk in _chunks(rows):
        batch = _batch(chunk, "base")
        for heads, sink in (((HEAD,), head_pairs), (COMPLEMENT, complement_pairs)):
            output = executor.patched_heads(batch, layer=LAYER, heads=heads, donor_cache=cache)
            forwards += 1
            evaluations += len(chunk)
            if len(output.answer_foil) != len(chunk):
                raise FactorialError("patched output count differs from batch")
            sink.update({row_id: _pair(pair) for row_id, pair in zip(batch.row_ids, output.answer_foil)})
    scored = score(rows, old, head_pairs, complement_pairs)
    return {
        "schema": "narrative_tense_attn11_head3_complement_factorial_result_v1",
        "candidate_id": "narrative_tense.attn11_head3_complement_factorial",
        "screen_tier_only": True,
        "execution_policy": "managed_queue_only",
        "prior_art_sha256": EXPECTED_PRIOR_ART_SHA256,
        "parent_result_sha256": EXPECTED_PARENT_SHA256,
        "authority_sha256": EXPECTED_AUTHORITY_SHA256,
        **scored,
        "active_price": {
            "model_forwards": forwards,
            "example_evaluations": evaluations,
            "backwards": 0,
            "parameter_updates": 0,
        },
        "ce_note": "CE fields are exact two-answer contrast CE, not unavailable full-vocabulary CE.",
        "interpretation_limit": compile_plan()["interpretation_limit"],
        "serial_seconds": clock() - started,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for name in ("BQLIB_DRYRUN", "BQLIB_NO_MODEL"):
        if os.environ.get(name) not in {None, "1"}:
            raise FactorialError(f"{name} must be absent or exactly 1")
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1" \
            or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(compile_plan(), sort_keys=True))
        return
    result = run_science()
    payload = managed.atomic_create_json(OUT, result)
    print(json.dumps({
        "terminal": result["terminal"],
        "result_path": OUT.relative_to(ROOT).as_posix(),
        "result_sha256": hashlib.sha256(payload).hexdigest(),
        "active_price": result["active_price"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
