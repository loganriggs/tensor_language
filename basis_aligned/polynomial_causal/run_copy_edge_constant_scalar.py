#!/usr/bin/env python3
"""Fit two L8 copy-edge scalars on 32 cached rows and evaluate on the next 96."""

from __future__ import annotations

import hashlib
import json
import math
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
from terminal_copy_attention_adapter import OwnedPerHeadTensorAttention


HERE = Path(__file__).resolve().parent
PREREG = HERE / "COPY_EDGE_CONSTANT_SCALAR_PREREGISTRATION.md"
OUTPUT = HERE / "copy_edge_constant_scalar_results.json"
FIT_ROWS = 32
EVAL_START = 32
EVAL_STOP = 128
HISTORICAL = (-0.119, 0.190)
ARMS = (
    "native",
    "edge_removed",
    "native_pattern_broadcast",
    "fit_eligible_mixed",
    "fit_eligible_broadcast",
    "fit_positive_broadcast",
    "historical_broadcast",
    "wrong_source_fit_broadcast",
)


def _empty_stats(rows: int) -> dict[str, dict[str, torch.Tensor]]:
    return {
        arm: {cell: torch.zeros(rows, 5, dtype=torch.float64) for cell in CELLS}
        for arm in ARMS
    }


def _record(
    destination: dict[str, torch.Tensor],
    row_offset: int,
    native_logprob: torch.Tensor,
    native_nll: torch.Tensor,
    candidate_logits: torch.Tensor | None,
    targets: torch.Tensor,
    masks: dict[str, torch.Tensor],
) -> None:
    if candidate_logits is None:
        delta_nll = torch.zeros_like(native_nll)
        point_kl = torch.zeros_like(native_nll)
        candidate_correct = native_logprob.argmax(2) == targets
    else:
        candidate_logprob = F.log_softmax(candidate_logits.float(), dim=-1)
        candidate_nll = -candidate_logprob.gather(2, targets[..., None]).squeeze(2)
        delta_nll = candidate_nll - native_nll
        point_kl = (
            native_logprob.exp() * (native_logprob - candidate_logprob)
        ).sum(2).clamp_min(0)
        candidate_correct = candidate_logits.argmax(2) == targets
    for cell in CELLS:
        mask = masks[cell]
        rows = slice(row_offset, row_offset + len(targets))
        destination[cell][rows] = torch.stack((
            mask.sum(1),
            (native_nll * mask).sum(1),
            (delta_nll * mask).sum(1),
            (point_kl * mask).sum(1),
            (candidate_correct & mask).sum(1),
        ), dim=1).double().cpu()


def _cell_summary(values: torch.Tensor) -> dict[str, float | int | None]:
    count = int(values[:, 0].sum())
    supported = values[:, 0] > 0
    document_effect = values[supported, 2] / values[supported, 0]
    return {
        "count": count,
        "supporting_documents": int(supported.sum()),
        "native_ce": float(values[:, 1].sum() / max(count, 1)),
        "arm_ce": float((values[:, 1] + values[:, 2]).sum() / max(count, 1)),
        "delta_ce": float(values[:, 2].sum() / max(count, 1)),
        "native_to_arm_kl": float(values[:, 3].sum() / max(count, 1)),
        "arm_accuracy": float(values[:, 4].sum() / max(count, 1)),
        "document_mean_delta_ce": (
            float(document_effect.mean()) if len(document_effect) else None
        ),
        "document_se_delta_ce": (
            float(document_effect.std(unbiased=True) / math.sqrt(len(document_effect)))
            if len(document_effect) > 1 else None
        ),
    }


def _summarize(stats: dict[str, dict[str, torch.Tensor]]) -> dict[str, Any]:
    arms = {
        arm: {cell: _cell_summary(values) for cell, values in cells.items()}
        for arm, cells in stats.items()
    }
    deletion = arms["edge_removed"]["copy_positive"]["delta_ce"]

    def recovery(arm: str) -> float | None:
        if deletion <= 0:
            return None
        return 1 - arms[arm]["copy_positive"]["delta_ce"] / deletion

    recoveries = {arm: recovery(arm) for arm in ARMS if arm != "native"}
    primary = recoveries["fit_eligible_broadcast"]
    positive = recoveries["fit_positive_broadcast"]
    wrong = recoveries["wrong_source_fit_broadcast"]
    repeat_delta = abs(arms["fit_eligible_broadcast"]["repeat_negative"]["delta_ce"])
    nonrepeat_delta = abs(arms["fit_eligible_broadcast"]["nonrepeat"]["delta_ce"])
    gates = {
        "c1_useful_constant_compiler": primary is not None and primary >= 0.70,
        "c2_scalar_itself_simple": (
            recoveries["fit_eligible_mixed"] is not None
            and recoveries["fit_eligible_mixed"] >= 0.85
        ),
        "c3_selective_behavior": (
            repeat_delta <= min(0.02, 0.25 * deletion)
            and nonrepeat_delta <= min(0.02, 0.25 * deletion)
        ),
        "c4_directional_source": (
            primary is not None and wrong is not None and wrong <= 0.5 * primary
        ),
        "c5_fit_not_outcome_dependent": (
            primary is not None and positive is not None
            and abs(primary - positive) <= 0.10
        ),
    }
    return {
        "arms": arms,
        "copy_positive_recovery_relative_to_edge_deletion": recoveries,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def _distribution(values: torch.Tensor) -> dict[str, list[float] | int]:
    if values.ndim != 2 or values.shape[1] != len(HEADS) or not len(values):
        raise ValueError("pattern fit sample is malformed")
    values = values.float()
    return {
        "count": len(values),
        "mean": values.mean(0).tolist(),
        "std": values.std(0, unbiased=True).tolist(),
        "q05": torch.quantile(values, 0.05, dim=0).tolist(),
        "median": torch.quantile(values, 0.50, dim=0).tolist(),
        "q95": torch.quantile(values, 0.95, dim=0).tolist(),
    }


@torch.no_grad()
def run() -> dict[str, Any]:
    started = time.time()
    if file_sha256(ROWS_PATH) != ROWS_FILE_SHA256:
        raise RuntimeError("cached selection-role bytes changed")
    payload = torch.load(ROWS_PATH, map_location="cpu", weights_only=True)
    all_rows = payload["rows"]
    if tensor_sha256(all_rows) != ROWS_TENSOR_SHA256 or tuple(all_rows.shape) != (192, 257):
        raise RuntimeError("cached selection rows changed")
    rows = all_rows[:EVAL_STOP].contiguous()
    policy = nearest_repeat_policy(rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = facade.load_bilin18(device=device, dtype=torch.bfloat16)
    adapter = OwnedPerHeadTensorAttention.from_native(model.transformer.h[LAYER].attn)

    def native_attention(event: facade.AttentionEvent):
        return event.block.attn(event.state, event.first_value)

    def native_mlp(event: facade.EarlyMLPEvent):
        return event.block.mlp(event.state)

    fit_samples: dict[str, list[torch.Tensor]] = {"eligible": [], "positive": []}
    fit_closure_errors: list[float] = []
    for start in range(0, FIT_ROWS, 4):
        batch_rows = rows[start:start + 4]
        tokens = batch_rows[:, :-1].to(device).contiguous()
        batch_policy = {
            key: value[start:start + len(batch_rows)].to(device)
            for key, value in policy.items()
        }

        def capture_attention(event: facade.AttentionEvent):
            if event.site != LAYER:
                return event.block.attn(event.state, event.first_value)
            with adapter.begin(event.state, event.first_value) as transaction:
                native = transaction.native_full_write()
                bus = transaction.first_value_bus()
                pattern = transaction.source_pattern(
                    HEADS, batch_policy["successor"], batch_policy["eligible"],
                )
            fit_closure_errors.append(
                transaction.closure.all_head_recomposition_relative_error
            )
            fit_samples["eligible"].append(
                pattern[batch_policy["eligible"]].float().cpu()
            )
            fit_samples["positive"].append(
                pattern[batch_policy["copy_positive"]].float().cpu()
            )
            return native, bus

        facade.forward_with_dispatch(model, tokens, capture_attention, native_mlp)

    fit_values = {key: torch.cat(value) for key, value in fit_samples.items()}
    fit_distributions = {key: _distribution(value) for key, value in fit_values.items()}
    constants = {
        "fit_eligible": torch.tensor(
            fit_distributions["eligible"]["mean"], device=device,
        ),
        "fit_positive": torch.tensor(
            fit_distributions["positive"]["mean"], device=device,
        ),
        "historical": torch.tensor(HISTORICAL, device=device),
    }

    eval_rows = EVAL_STOP - EVAL_START
    stats = _empty_stats(eval_rows)
    eval_closure_errors: list[float] = []
    for start in range(EVAL_START, EVAL_STOP, 4):
        local_start = start - EVAL_START
        batch_rows = rows[start:start + 4]
        tokens = batch_rows[:, :-1].to(device).contiguous()
        targets = batch_rows[:, 1:].to(device)
        batch_policy = {
            key: value[start:start + len(batch_rows)].to(device)
            for key, value in policy.items()
        }
        masks = {cell: batch_policy[cell] for cell in CELLS}
        native_logits = facade.forward_with_dispatch(
            model, tokens, native_attention, native_mlp,
        )
        native_logprob = F.log_softmax(native_logits.float(), dim=-1)
        native_nll = -native_logprob.gather(2, targets[..., None]).squeeze(2)
        _record(
            stats["native"], local_start, native_logprob, native_nll, None,
            targets, masks,
        )
        del native_logits

        for arm in ARMS[1:]:
            def intervened_attention(
                event: facade.AttentionEvent, arm: str = arm,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                if event.site != LAYER:
                    return event.block.attn(event.state, event.first_value)
                with adapter.begin(event.state, event.first_value) as transaction:
                    native = transaction.native_full_write()
                    bus = transaction.first_value_bus()
                    removed = transaction.source_write(
                        HEADS,
                        batch_policy["successor"],
                        batch_policy["eligible"],
                        route="mixed",
                    )
                    if arm == "edge_removed":
                        replacement = torch.zeros_like(removed)
                    elif arm == "native_pattern_broadcast":
                        replacement = transaction.source_write(
                            HEADS,
                            batch_policy["successor"],
                            batch_policy["eligible"],
                            route="broadcast",
                        )
                    else:
                        source = (
                            batch_policy["source"]
                            if arm == "wrong_source_fit_broadcast"
                            else batch_policy["successor"]
                        )
                        fit_name = {
                            "fit_eligible_mixed": "fit_eligible",
                            "fit_eligible_broadcast": "fit_eligible",
                            "fit_positive_broadcast": "fit_positive",
                            "historical_broadcast": "historical",
                            "wrong_source_fit_broadcast": "fit_eligible",
                        }[arm]
                        route = "mixed" if arm == "fit_eligible_mixed" else "broadcast"
                        replacement = transaction.source_write(
                            HEADS,
                            source,
                            batch_policy["eligible"],
                            route=route,
                            pattern_override=constants[fit_name],
                        )
                eval_closure_errors.append(
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

    summary = _summarize(stats)
    return {
        "schema": "copy_edge_constant_scalar_v1",
        "status": "exploratory_disjoint_fit_eval_on_exposed_selection_rows",
        "fit_rows": [0, FIT_ROWS],
        "evaluation_rows": [EVAL_START, EVAL_STOP],
        "fit_document_ids": [
            record["document_id"] for record in payload["records"][:FIT_ROWS]
        ],
        "evaluation_document_ids": [
            record["document_id"] for record in payload["records"][EVAL_START:EVAL_STOP]
        ],
        "rows_file_sha256": ROWS_FILE_SHA256,
        "rows_tensor_sha256": ROWS_TENSOR_SHA256,
        "preregistration_sha256": file_sha256(PREREG),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checkpoint": checkpoint.__dict__,
        "fit_pattern_distributions_head_order_h3_h4": fit_distributions,
        "frozen_constants_head_order_h3_h4": {
            key: value.float().cpu().tolist() for key, value in constants.items()
        },
        "fit_maximum_attention_recomposition_relative_error": max(fit_closure_errors),
        "evaluation_maximum_attention_recomposition_relative_error": max(
            eval_closure_errors
        ),
        "summary": summary,
        "per_document": {
            arm: {cell: values.tolist() for cell, values in cells.items()}
            for arm, cells in stats.items()
        },
        "runtime_seconds": time.time() - started,
        "claim_boundary": (
            "Disjoint 32-row fit and 96-row evaluation, but both reuse an exposed "
            "selection cache. This is an exploratory local compiler test, not fresh "
            "confirmation or a standalone whole-model extraction."
        ),
    }


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"output already exists: {OUTPUT}")
    result = run()
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "constants": result["frozen_constants_head_order_h3_h4"],
        "summary": result["summary"],
    }, indent=2, sort_keys=True), flush=True)
    print(f"wrote {OUTPUT} in {result['runtime_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()

