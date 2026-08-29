#!/usr/bin/env python3
"""Evaluate a static matcher score and distance table as L8 copy-edge gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bilin18_observed_model_facade as facade
from discover_copy_source_edges import (
    CELLS,
    HEADS,
    LAYER,
    ROWS_FILE_SHA256,
    ROWS_PATH,
    ROWS_TENSOR_SHA256,
    file_sha256,
    nearest_repeat_policy,
    tensor_sha256,
)
from run_copy_edge_constant_scalar import _cell_summary, _record
from terminal_copy_attention_adapter import OwnedPerHeadTensorAttention


HERE = Path(__file__).resolve().parent
PREREG = HERE / "COPY_EDGE_SIMPLE_GATE_PREREGISTRATION.md"
BASELINE = HERE / "copy_edge_constant_scalar_results.json"
BASELINE_SHA256 = "3da06d79c0d28bbb6f4d13082aa8c0dcc1bd3315a5ef9ec485e347136774603f"
OUTPUT = HERE / "copy_edge_simple_gate_results.json"
FIT_STOP = 32
EVAL_START = 32
EVAL_STOP = 128
MATCHERS = ((2, 5), (3, 8))
DISTANCE_EDGES = (1, 9, 33, 65, 129)
NEW_ARMS = (
    "static_match_affine_broadcast",
    "static_match_shift_control",
    "distance_bin_broadcast",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fit_affine(score: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    design = torch.stack((score.double(), torch.ones_like(score, dtype=torch.double)), 1)
    return torch.linalg.lstsq(design, target.double()).solution


def _predict_affine(score: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    return (
        score.double()[..., None] * coefficients[0] + coefficients[1]
    ).float()


def _r2(target: torch.Tensor, predicted: torch.Tensor) -> list[float]:
    target, predicted = target.double(), predicted.double()
    residual = ((target - predicted) ** 2).sum(0)
    total = ((target - target.mean(0)) ** 2).sum(0).clamp_min(1e-30)
    return (1 - residual / total).tolist()


def _fit_distance_table(
    distance: torch.Tensor, target: torch.Tensor,
) -> tuple[torch.Tensor, list[int]]:
    table, counts = [], []
    for lower, upper in zip(DISTANCE_EDGES[:-1], DISTANCE_EDGES[1:]):
        mask = (distance >= lower) & (distance < upper)
        if not bool(mask.any()):
            raise RuntimeError(f"empty distance fit bin [{lower},{upper})")
        table.append(target[mask].mean(0))
        counts.append(int(mask.sum()))
    return torch.stack(table), counts


def _distance_prediction(distance: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    output = torch.empty(*distance.shape, len(HEADS), device=distance.device)
    assigned = torch.zeros_like(distance, dtype=torch.bool)
    for index, (lower, upper) in enumerate(zip(DISTANCE_EDGES[:-1], DISTANCE_EDGES[1:])):
        mask = (distance >= lower) & (distance < upper)
        output[mask] = table[index].to(output.device)
        assigned |= mask
    output[~assigned] = 0
    return output


def _summaries(stats: dict[str, dict[str, torch.Tensor]]) -> dict[str, Any]:
    return {
        arm: {cell: _cell_summary(value) for cell, value in cells.items()}
        for arm, cells in stats.items()
    }


@torch.no_grad()
def run() -> dict[str, Any]:
    started = time.time()
    if _sha256(BASELINE) != BASELINE_SHA256:
        raise RuntimeError("constant-scalar baseline changed")
    baseline = json.loads(BASELINE.read_text())
    if (
        baseline["fit_rows"] != [0, FIT_STOP]
        or baseline["evaluation_rows"] != [EVAL_START, EVAL_STOP]
        or baseline["schema"] != "copy_edge_constant_scalar_v1"
    ):
        raise RuntimeError("constant-scalar baseline split or schema changed")
    if file_sha256(ROWS_PATH) != ROWS_FILE_SHA256:
        raise RuntimeError("cached selection-role bytes changed")
    payload = torch.load(ROWS_PATH, map_location="cpu", weights_only=True)
    rows = payload["rows"]
    if tensor_sha256(rows) != ROWS_TENSOR_SHA256 or tuple(rows.shape) != (192, 257):
        raise RuntimeError("cached selection rows changed")
    rows = rows[:EVAL_STOP].contiguous()
    policy = nearest_repeat_policy(rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = facade.load_bilin18(device=device, dtype=torch.bfloat16)
    if checkpoint.__dict__ != baseline["checkpoint"]:
        raise RuntimeError("checkpoint differs from frozen baseline")
    l8_adapter = OwnedPerHeadTensorAttention.from_native(model.transformer.h[LAYER].attn)
    matcher_adapters = {
        layer: OwnedPerHeadTensorAttention.from_native(model.transformer.h[layer].attn)
        for layer, _ in MATCHERS
    }
    width = model.transformer.wte.weight.shape[1]

    def native_mlp(event: facade.EarlyMLPEvent):
        return event.block.mlp(event.state)

    def static_match_score(
        tokens: torch.Tensor, batch_policy: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        state = F.rms_norm(model.transformer.wte(tokens), (width,))
        score = torch.zeros_like(batch_policy["source"], dtype=torch.float32)
        for layer, head in MATCHERS:
            with matcher_adapters[layer].begin(state) as transaction:
                component = transaction.source_pattern(
                    (head,), batch_policy["source"], batch_policy["eligible"],
                ).squeeze(2)
            score -= component.float()
        return score

    fit_scores: list[torch.Tensor] = []
    fit_targets: list[torch.Tensor] = []
    fit_distances: list[torch.Tensor] = []
    closure_errors: list[float] = []
    for start in range(0, FIT_STOP, 4):
        batch_rows = rows[start:start + 4]
        tokens = batch_rows[:, :-1].to(device).contiguous()
        batch_policy = {
            key: value[start:start + len(batch_rows)].to(device)
            for key, value in policy.items()
        }
        score = static_match_score(tokens, batch_policy)
        positions = torch.arange(tokens.shape[1], device=device)[None, :]
        distance = positions - batch_policy["source"]
        captured: list[torch.Tensor] = []

        def capture_attention(event: facade.AttentionEvent):
            if event.site != LAYER:
                return event.block.attn(event.state, event.first_value)
            with l8_adapter.begin(event.state, event.first_value) as transaction:
                native = transaction.native_full_write()
                bus = transaction.first_value_bus()
                captured.append(transaction.source_pattern(
                    HEADS, batch_policy["successor"], batch_policy["eligible"],
                ))
            closure_errors.append(
                transaction.closure.all_head_recomposition_relative_error
            )
            return native, bus

        facade.forward_with_dispatch(model, tokens, capture_attention, native_mlp)
        mask = batch_policy["eligible"]
        fit_scores.append(score[mask].cpu())
        fit_targets.append(captured[0][mask].float().cpu())
        fit_distances.append(distance[mask].cpu())

    fit_score = torch.cat(fit_scores)
    fit_target = torch.cat(fit_targets)
    fit_distance = torch.cat(fit_distances)
    affine = _fit_affine(fit_score, fit_target)
    fit_affine_prediction = _predict_affine(fit_score, affine)
    distance_table, distance_counts = _fit_distance_table(fit_distance, fit_target)
    fit_distance_prediction = _distance_prediction(fit_distance, distance_table)

    eval_count = EVAL_STOP - EVAL_START
    stats = {
        arm: {cell: torch.zeros(eval_count, 5, dtype=torch.float64) for cell in CELLS}
        for arm in ("native_recomputed",) + NEW_ARMS
    }
    eval_targets: list[torch.Tensor] = []
    eval_affine_predictions: list[torch.Tensor] = []
    eval_distance_predictions: list[torch.Tensor] = []
    for start in range(EVAL_START, EVAL_STOP, 4):
        local_start = start - EVAL_START
        batch_rows = rows[start:start + 4]
        tokens = batch_rows[:, :-1].to(device).contiguous()
        targets = batch_rows[:, 1:].to(device)
        batch_policy = {
            key: value[start:start + len(batch_rows)].to(device)
            for key, value in policy.items()
        }
        score = static_match_score(tokens, batch_policy)
        positions = torch.arange(tokens.shape[1], device=device)[None, :]
        distance = positions - batch_policy["source"]
        predictions = {
            "static_match_affine_broadcast": _predict_affine(score, affine.to(device)),
            "static_match_shift_control": _predict_affine(
                torch.roll(score, shifts=1, dims=1), affine.to(device),
            ),
            "distance_bin_broadcast": _distance_prediction(
                distance, distance_table.to(device),
            ),
        }
        captured: list[torch.Tensor] = []

        def capture_attention(event: facade.AttentionEvent):
            if event.site != LAYER:
                return event.block.attn(event.state, event.first_value)
            with l8_adapter.begin(event.state, event.first_value) as transaction:
                native = transaction.native_full_write()
                bus = transaction.first_value_bus()
                captured.append(transaction.source_pattern(
                    HEADS, batch_policy["successor"], batch_policy["eligible"],
                ))
            closure_errors.append(
                transaction.closure.all_head_recomposition_relative_error
            )
            return native, bus

        native_logits = facade.forward_with_dispatch(
            model, tokens, capture_attention, native_mlp,
        )
        native_logprob = F.log_softmax(native_logits.float(), dim=-1)
        native_nll = -native_logprob.gather(2, targets[..., None]).squeeze(2)
        masks = {cell: batch_policy[cell] for cell in CELLS}
        _record(
            stats["native_recomputed"], local_start, native_logprob, native_nll,
            None, targets, masks,
        )
        del native_logits
        eligible = batch_policy["eligible"]
        eval_targets.append(captured[0][eligible].float().cpu())
        eval_affine_predictions.append(
            predictions["static_match_affine_broadcast"][eligible].float().cpu()
        )
        eval_distance_predictions.append(
            predictions["distance_bin_broadcast"][eligible].float().cpu()
        )

        for arm in NEW_ARMS:
            def intervened_attention(
                event: facade.AttentionEvent, arm: str = arm,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                if event.site != LAYER:
                    return event.block.attn(event.state, event.first_value)
                with l8_adapter.begin(event.state, event.first_value) as transaction:
                    native = transaction.native_full_write()
                    bus = transaction.first_value_bus()
                    removed = transaction.source_write(
                        HEADS, batch_policy["successor"], eligible, route="mixed",
                    )
                    replacement = transaction.source_write(
                        HEADS,
                        batch_policy["successor"],
                        eligible,
                        route="broadcast",
                        pattern_override=predictions[arm],
                    )
                closure_errors.append(
                    transaction.closure.all_head_recomposition_relative_error
                )
                return native - removed + replacement, bus

            candidate_logits = facade.forward_with_dispatch(
                model, tokens, intervened_attention, native_mlp,
            )
            _record(
                stats[arm], local_start, native_logprob, native_nll,
                candidate_logits, targets, masks,
            )
            del candidate_logits
        del native_logprob, native_nll

    new_summaries = _summaries(stats)
    baseline_arms = baseline["summary"]["arms"]
    baseline_native_per_doc = torch.tensor(
        baseline["per_document"]["native"]["all_scored"], dtype=torch.float64,
    )
    if not torch.equal(stats["native_recomputed"]["all_scored"], baseline_native_per_doc):
        maximum = float((
            stats["native_recomputed"]["all_scored"] - baseline_native_per_doc
        ).abs().max())
        raise RuntimeError(f"recomputed native baseline differs by {maximum}")
    arms = {
        "native": baseline_arms["native"],
        "edge_removed": baseline_arms["edge_removed"],
        "native_pattern_broadcast": baseline_arms["native_pattern_broadcast"],
        "eligible_constant_broadcast": baseline_arms["fit_eligible_broadcast"],
        **{arm: new_summaries[arm] for arm in NEW_ARMS},
    }
    deletion = arms["edge_removed"]["copy_positive"]["delta_ce"]
    recoveries = {
        arm: 1 - values["copy_positive"]["delta_ce"] / deletion
        for arm, values in arms.items() if arm != "native"
    }
    primary = recoveries["static_match_affine_broadcast"]
    distance_recovery = recoveries["distance_bin_broadcast"]
    repeat_delta = abs(
        arms["static_match_affine_broadcast"]["repeat_negative"]["delta_ce"]
    )
    nonrepeat_delta = abs(
        arms["static_match_affine_broadcast"]["nonrepeat"]["delta_ce"]
    )
    gates = {
        "g1_matcher_gate_useful": primary >= 0.70,
        "g2_beats_unconditional_constant": (
            primary - recoveries["eligible_constant_broadcast"] >= 0.20
        ),
        "g3_beats_shifted_association_control": (
            primary - recoveries["static_match_shift_control"] >= 0.20
        ),
        "g4_selective": (
            repeat_delta <= min(0.02, 0.25 * deletion)
            and nonrepeat_delta <= min(0.02, 0.25 * deletion)
        ),
        "g5_distance_cheap_alternative": (
            distance_recovery >= 0.60 and abs(distance_recovery - primary) <= 0.10
        ),
    }
    eval_target = torch.cat(eval_targets)
    eval_affine = torch.cat(eval_affine_predictions)
    eval_distance = torch.cat(eval_distance_predictions)
    qk_values_per_matcher = 4 * width * (width // model.transformer.h[0].attn.n_head)
    writer_values = width * len(HEADS) * (width // model.transformer.h[0].attn.n_head)
    return {
        "schema": "copy_edge_simple_gate_v1",
        "status": "exploratory_disjoint_fit_eval_on_exposed_selection_rows",
        "fit_rows": [0, FIT_STOP],
        "evaluation_rows": [EVAL_START, EVAL_STOP],
        "baseline_sha256": BASELINE_SHA256,
        "preregistration_sha256": file_sha256(PREREG),
        "runner_sha256": _sha256(Path(__file__)),
        "checkpoint": checkpoint.__dict__,
        "affine_coefficients_rows_alpha_beta_columns_h3_h4": affine.tolist(),
        "distance_edges_half_open": list(DISTANCE_EDGES),
        "distance_fit_counts": distance_counts,
        "distance_table_rows_columns_h3_h4": distance_table.tolist(),
        "fit_scalar_r2_h3_h4": {
            "static_match_affine": _r2(fit_target, fit_affine_prediction),
            "distance_bins": _r2(fit_target, fit_distance_prediction),
        },
        "evaluation_scalar_r2_h3_h4": {
            "static_match_affine": _r2(eval_target, eval_affine),
            "distance_bins": _r2(eval_target, eval_distance),
        },
        "summary": {
            "arms": arms,
            "copy_positive_recovery_relative_to_edge_deletion": recoveries,
            "gates": gates,
        },
        "price": {
            "static_match_affine": {
                "qk_projection_slice_values": 2 * qk_values_per_matcher,
                "affine_scalars": 4,
                "writer_projection_slice_values": writer_values,
                "uses_embedding_table": True,
                "uses_live_contextual_state": False,
            },
            "distance_bins": {
                "gate_scalars": len(HEADS) * (len(DISTANCE_EDGES) - 1),
                "interval_boundaries": len(DISTANCE_EDGES),
                "writer_projection_slice_values": writer_values,
                "uses_embedding_table": False,
                "uses_live_contextual_state": False,
            },
        },
        "maximum_attention_recomposition_relative_error": max(closure_errors),
        "new_per_document": {
            arm: {cell: value.tolist() for cell, value in cells.items()}
            for arm, cells in stats.items()
        },
        "runtime_seconds": time.time() - started,
        "claim_boundary": (
            "Disjoint fit/evaluation within an exposed cache. Baseline arms are "
            "hash-pinned reuse; only three gate arms were newly forwarded."
        ),
    }


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"output already exists: {OUTPUT}")
    result = run()
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "affine": result["affine_coefficients_rows_alpha_beta_columns_h3_h4"],
        "distance_table": result["distance_table_rows_columns_h3_h4"],
        "r2": {
            "fit": result["fit_scalar_r2_h3_h4"],
            "eval": result["evaluation_scalar_r2_h3_h4"],
        },
        "recoveries": result["summary"][
            "copy_positive_recovery_relative_to_edge_deletion"
        ],
        "gates": result["summary"]["gates"],
    }, indent=2, sort_keys=True), flush=True)
    print(f"wrote {OUTPUT} in {result['runtime_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()

