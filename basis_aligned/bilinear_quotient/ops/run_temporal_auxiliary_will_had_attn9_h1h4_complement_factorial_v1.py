#!/usr/bin/env python3
"""Prospective H1/H4 versus complement factorial for will/had attention 9."""

# BQGATE: EXPERIMENT pred_a_authority_capability_and_exact_partition pred_b_shared_h1h4_reader_group pred_c_controls_selective pred_d_exact_zero_fit_price
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import circuit_fast_screen_candidate_temporal_auxiliary as candidate
import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_attn9_h1h4_complement_factorial_v1.json"
PARENT = ROOT / "circuits/fast_screens/temporal_auxiliary_will_vs_had_v1_result.json"
LOCAL_NULL = ROOT / "circuits/followups/temporal_auxiliary_will_had_local_l9_value_reuse_v1_result.json"
BUILDER = ROOT / "ops/circuit_fast_screen_candidate_temporal_auxiliary.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_attn9_h1h4_complement_factorial_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.attn9_h1h4_complement_factorial_v1"
EXPECTED = {
    "prior": "e2f777649a31e0c990dfede7b13a30e3062d57ff4fb3996ca082e2b2a07890c0",
    "parent": "71a833c52c39b98be6f576d11749522e2725c540a29d61a2352e315978684ec9",
    "local_null": "52026d5df5994c75501ea005078fd649cf5685a5ac03f9ae680f87623b2ba4e7",
    "builder": "4d23947375504edaa51e4ef057ccd65f042c505728d1909bc06edf526633bc58",
}
EXPECTED_ROWS_SHA256 = "b74ff65cf6b59142bf38172dae7f70a25e5f471bbeaa038f9b3971caedf913da"
LAYER = 9
H1H4 = (1, 4)
COMPLEMENT = (0, 2, 3, 5, 6, 7, 8)
BATCH_SIZE = 32


class ExperimentError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def margin(item):
    return float(item["answer_logit"]) - float(item["foil_logit"])


def pair(value):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ExperimentError("malformed answer/foil pair")
    result = float(value[0]), float(value[1])
    if any(not math.isfinite(item) for item in result):
        raise ExperimentError("nonfinite answer/foil pair")
    return result


def load_closure():
    observed = {
        "prior": sha(PRIOR),
        "parent": sha(PARENT),
        "local_null": sha(LOCAL_NULL),
        "builder": sha(BUILDER),
    }
    if observed != EXPECTED:
        raise ExperimentError(f"authority hash changed: {observed}")
    rows = candidate.build_rows()
    if candidate.validate_rows(rows) != EXPECTED_ROWS_SHA256 or len(rows) != 128:
        raise ExperimentError("row authority changed")
    parent = json.loads(PARENT.read_text())
    local_null = json.loads(LOCAL_NULL.read_text())
    if (
        parent.get("terminal") != "screen"
        or parent.get("selected_site_id") != "resid:18"
        or not all(parent.get("predictions", {}).values())
        or local_null.get("terminal") != "null"
        or local_null.get("predictions", {}).get("pred_c_local_value_edge_reuses") is not True
    ):
        raise ExperimentError("parent or local-value boundary changed")
    row_ids = {str(row["row_id"]) for row in rows}
    native = parent.get("run", {}).get("native_logits", [])
    full = [
        item
        for item in parent.get("run", {}).get("intervention_logits", [])
        if item.get("site", {}).get("site_id") == "attn:09"
    ]
    if len(native) != 256 or {
        (str(item["row_id"]), str(item["side"])) for item in native
    } != {(row_id, side) for row_id in row_ids for side in ("base", "donor")}:
        raise ExperimentError("frozen native coverage changed")
    if len(full) != 128 or {str(item["row_id"]) for item in full} != row_ids:
        raise ExperimentError("frozen attention-9 coverage changed")
    cells = parent.get("run", {}).get("capability_cells", [])
    if not cells or not all(cell.get("passed") is True for cell in cells):
        raise ExperimentError("native capability boundary changed")
    return rows, parent


def chunks(rows):
    return [list(rows[start : start + BATCH_SIZE]) for start in range(0, len(rows), BATCH_SIZE)]


def make_batch(rows, side):
    return producer.ModelBatch(
        row_ids=tuple(str(row["row_id"]) for row in rows),
        side=side,
        token_rows=tuple(tuple(int(item) for item in row[f"{side}_ids"]) for row in rows),
        answer_ids=tuple(int(row[f"{side}_answer_id"]) for row in rows),
        foil_ids=tuple(int(row[f"{side}_foil_id"]) for row in rows),
        semantic_positions=tuple(int(row[f"{side}_semantic_position"]) for row in rows),
    )


def summarize(rows, parent, h1h4_pairs, complement_pairs):
    native = {
        (str(item["row_id"]), str(item["side"])): item
        for item in parent["run"]["native_logits"]
    }
    full = {
        str(item["row_id"]): item
        for item in parent["run"]["intervention_logits"]
        if item["site"]["site_id"] == "attn:09"
    }
    target_denominators = [
        margin(native[(str(row["row_id"]), "base")])
        + margin(native[(str(row["row_id"]), "donor")])
        for row in rows
        if row["transform_id"] in {"A1", "A2"}
    ]
    target_scale = statistics.median(target_denominators)
    if target_scale <= kernel.MIN_DONOR_DENOMINATOR:
        raise ExperimentError("target scale is invalid")
    evidence = []
    target_groups = {}
    control_groups = {"P": [], "C": []}
    for row in rows:
        row_id = str(row["row_id"])
        family = str(row["transform_id"])
        base_margin = margin(native[(row_id, "base")])
        donor_margin = margin(native[(row_id, "donor")])
        margins = {
            "empty": base_margin,
            "h1h4": h1h4_pairs[row_id][0] - h1h4_pairs[row_id][1],
            "complement": complement_pairs[row_id][0] - complement_pairs[row_id][1],
            "full": margin(full[row_id]),
        }
        if bool(row["answer_changes"]):
            recoveries = {
                name: kernel.signed_pairwise_donor_recovery(
                    -base_margin, donor_margin, -value
                )
                for name, value in margins.items()
            }
        else:
            recoveries = {
                name: (value - base_margin) / target_scale for name, value in margins.items()
            }
        interaction = (
            recoveries["full"]
            - recoveries["h1h4"]
            - recoveries["complement"]
            + recoveries["empty"]
        )
        record = {
            "row_id": row_id,
            "family": family,
            "cell_id": str(row["capability_cell_id"]),
            "recoveries": recoveries,
            "interaction": interaction,
        }
        evidence.append(record)
        if family in {"A1", "A2"}:
            target_groups.setdefault(record["cell_id"], []).append(record)
        else:
            control_groups[family].append(record)
    cells = []
    for cell_id, records in sorted(target_groups.items()):
        means = {
            arm: statistics.fmean(record["recoveries"][arm] for record in records)
            for arm in ("empty", "h1h4", "complement", "full")
        }
        ratio = means["h1h4"] / means["full"] if means["full"] > 1e-9 else math.nan
        direction = sum(record["recoveries"]["h1h4"] > 0.0 for record in records) / len(records)
        cells.append(
            {
                "cell_id": cell_id,
                "row_count": len(records),
                "mean_recovery": means,
                "h1h4_fraction_of_full": ratio,
                "h1h4_direction_fraction": direction,
                "mean_interaction": statistics.fmean(record["interaction"] for record in records),
                "passed": math.isfinite(ratio) and ratio >= 0.50 and direction >= 0.75,
            }
        )
    controls = {}
    for family, records in control_groups.items():
        h1h4 = statistics.fmean(abs(record["recoveries"]["h1h4"]) for record in records)
        full_move = statistics.fmean(abs(record["recoveries"]["full"]) for record in records)
        controls[family] = {
            "row_count": len(records),
            "h1h4_mean_absolute_normalized_movement": h1h4,
            "full_attn9_mean_absolute_normalized_movement": full_move,
            "passed": h1h4 <= full_move + 1e-12,
        }
    return target_scale, cells, controls, evidence


def main():
    rows, parent = load_closure()
    dryrun = {
        "candidate_id": CANDIDATE_ID,
        "dryrun": True,
        "gpu_accessed": False,
        "model_loaded": False,
        "rows": 128,
        "batch_size": 32,
        "new_patch_arms": 2,
        "model_forwards": 12,
        "example_evaluations": 384,
        "fitted_scalars": 0,
        "transformer_backwards": 0,
        "model_updates": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    started = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    donor_cache = {}
    h1h4_pairs = {}
    complement_pairs = {}
    forward_calls = 0
    evaluations = 0
    for chunk in chunks(rows):
        output = backend.native(make_batch(chunk, "donor"), capture=True)
        donor_cache.update(output.captured)
        forward_calls += 1
        evaluations += len(chunk)
    required = {
        (str(row["row_id"]), f"attn:{LAYER:02d}:head:{head:02d}")
        for row in rows
        for head in range(9)
    }
    missing = required.difference(donor_cache)
    if missing:
        raise ExperimentError(f"donor cache lacks {len(missing)} head slices")
    for chunk in chunks(rows):
        batch = make_batch(chunk, "base")
        for heads, sink in ((H1H4, h1h4_pairs), (COMPLEMENT, complement_pairs)):
            output = backend.patched_heads(
                batch, layer=LAYER, heads=heads, donor_cache=donor_cache
            )
            forward_calls += 1
            evaluations += len(chunk)
            if len(output.answer_foil) != len(chunk):
                raise ExperimentError("patched output coverage changed")
            sink.update(
                {row_id: pair(value) for row_id, value in zip(batch.row_ids, output.answer_foil)}
            )
    target_scale, cells, controls, evidence = summarize(
        rows, parent, h1h4_pairs, complement_pairs
    )
    pred_a = (
        len(h1h4_pairs) == 128
        and len(complement_pairs) == 128
        and len(evidence) == 128
        and all(cell.get("passed") is True for cell in parent["run"]["capability_cells"])
    )
    pred_b = all(cell["passed"] for cell in cells)
    pred_c = all(item["passed"] for item in controls.values())
    price = {
        "model_forwards": forward_calls,
        "example_evaluations": evaluations,
        "rows": len(rows),
        "new_patch_arms": 2,
        "batch_size": BATCH_SIZE,
        "fitted_scalars": 0,
        "transformer_backwards": 0,
        "model_updates": 0,
    }
    pred_d = price == {
        "model_forwards": 12,
        "example_evaluations": 384,
        "rows": 128,
        "new_patch_arms": 2,
        "batch_size": 32,
        "fitted_scalars": 0,
        "transformer_backwards": 0,
        "model_updates": 0,
    }
    predictions = {
        "pred_a_authority_capability_and_exact_partition": pred_a,
        "pred_b_shared_h1h4_reader_group": pred_b,
        "pred_c_controls_selective": pred_c,
        "pred_d_exact_zero_fit_price": pred_d,
    }
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and pred_d else "invalid")
    result = {
        "schema": "temporal_auxiliary_will_had_attn9_h1h4_complement_factorial_result_v1",
        "candidate_id": CANDIDATE_ID,
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "prior_art_sha256": EXPECTED["prior"],
        "authority_sha256": EXPECTED,
        "rows_sha256": EXPECTED_ROWS_SHA256,
        "target_scale": target_scale,
        "cells": cells,
        "controls": controls,
        "evidence": evidence,
        "predictions": predictions,
        "price": price,
        "terminal": terminal,
        "reason": {
            "screen": "h1h4_reader_group_generalizes_to_will_had_attention9",
            "null": "h1h4_target_fraction_or_control_selectivity_misses",
            "invalid": "authority_capability_cache_coverage_finiteness_or_price_invalid",
        }[terminal],
        "scope_boundary": "Head grouping only; upstream sources and the full will/had circuit remain untested.",
        "serial_seconds": time.perf_counter() - started,
        "next_action": (
            "localize the non-MLP4 source terms entering the reused H1/H4 reader group"
            if terminal == "screen"
            else "follow the seven-head complement and attention-11 route rather than extending H1/H4"
        ),
    }
    atomic_create_json(OUT, result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "candidate_id",
                    "target_scale",
                    "cells",
                    "controls",
                    "predictions",
                    "price",
                    "terminal",
                    "reason",
                    "scope_boundary",
                    "next_action",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
