#!/usr/bin/env python3
"""Causal rank curve for SVD-factorized L8 H3/H4 copy-edge Q/K gates."""

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
PREREG = HERE / "COPY_EDGE_LOWRANK_QK_PREREGISTRATION.md"
BASELINE = HERE / "copy_edge_constant_scalar_results.json"
BASELINE_SHA256 = "3da06d79c0d28bbb6f4d13082aa8c0dcc1bd3315a5ef9ec485e347136774603f"
OUTPUT = HERE / "copy_edge_lowrank_qk_results.json"
EVAL_START = 32
EVAL_STOP = 128
RANKS = (8, 16, 32, 64, 96, 128)
PROJECTIONS = ("q", "k", "q2", "k2")
HEAD_DIM = 128


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LowRankSourcePattern:
    """Owned SVD factors for only the two L8 heads' Q/K pattern computation."""

    def __init__(self, adapter: OwnedPerHeadTensorAttention) -> None:
        if adapter.head_dim != HEAD_DIM or tuple(HEADS) != (3, 4):
            raise ValueError("low-rank copy gate topology changed")
        self.device = adapter.q.device
        self.dtype = adapter.q.dtype
        self.width = adapter.width
        self.inv_freq = adapter.inv_freq.detach().clone()
        self.factors: dict[int, dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]]] = {
            rank: {} for rank in RANKS
        }
        self.spectra: dict[str, list[float]] = {}
        for name in PROJECTIONS:
            full = getattr(adapter, name)
            for head in HEADS:
                weight = full[head * HEAD_DIM:(head + 1) * HEAD_DIM].float()
                u, singular, vh = torch.linalg.svd(weight, full_matrices=False)
                self.spectra[f"{name}_h{head}"] = singular.detach().cpu().tolist()
                for rank in RANKS:
                    left = (u[:, :rank] * singular[:rank]).to(self.dtype)
                    right = vh[:rank].to(self.dtype)
                    self.factors[rank][(name, head)] = (left, right)

    def _project(
        self, state: torch.Tensor, name: str, head: int, rank: int,
    ) -> torch.Tensor:
        left, right = self.factors[rank][(name, head)]
        latent = F.linear(state, right)
        return F.linear(latent, left)

    def _rotate(self, value: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(
            value.shape[1], device=value.device, dtype=self.inv_freq.dtype,
        )
        angles = torch.outer(positions, self.inv_freq.to(value.device))
        cosine = angles.cos().bfloat16()[None, :, :]
        sine = angles.sin().bfloat16()[None, :, :]
        value = F.rms_norm(value, (HEAD_DIM,))
        first, second = value[..., :HEAD_DIM // 2], value[..., HEAD_DIM // 2:]
        return torch.cat((
            first * cosine + second * sine,
            first * (-sine) + second * cosine,
        ), dim=-1).to(value.dtype)

    def source_pattern(
        self, state: torch.Tensor, source_indices: torch.Tensor, rank: int,
    ) -> torch.Tensor:
        if rank not in self.factors:
            raise ValueError("unregistered low-rank gate")
        if (
            state.ndim != 3 or state.shape[-1] != self.width
            or source_indices.shape != state.shape[:2]
            or source_indices.dtype != torch.long
            or source_indices.device != state.device
        ):
            raise ValueError("low-rank source-pattern input is malformed")
        batch, sequence = source_indices.shape
        output = []
        gather = source_indices[..., None].expand(batch, sequence, HEAD_DIM)
        for head in HEADS:
            query = self._rotate(self._project(state, "q", head, rank))
            key = self._rotate(self._project(state, "k", head, rank))
            query2 = self._rotate(self._project(state, "q2", head, rank))
            key2 = self._rotate(self._project(state, "k2", head, rank))
            key_at_source = torch.gather(key, 1, gather)
            key2_at_source = torch.gather(key2, 1, gather)
            first = torch.einsum("btd,btd->bt", query, key_at_source) / HEAD_DIM
            second = torch.einsum("btd,btd->bt", query2, key2_at_source) / HEAD_DIM
            output.append(first * second)
        return torch.stack(output, dim=2)


def _r2(target: torch.Tensor, predicted: torch.Tensor) -> list[float]:
    target, predicted = target.double(), predicted.double()
    residual = ((target - predicted) ** 2).sum(0)
    total = ((target - target.mean(0)) ** 2).sum(0).clamp_min(1e-30)
    return (1 - residual / total).tolist()


def _correlation(target: torch.Tensor, predicted: torch.Tensor) -> list[float]:
    target = target.double() - target.double().mean(0)
    predicted = predicted.double() - predicted.double().mean(0)
    numerator = (target * predicted).sum(0)
    denominator = (target.square().sum(0) * predicted.square().sum(0)).sqrt().clamp_min(1e-30)
    return (numerator / denominator).tolist()


@torch.no_grad()
def run() -> dict[str, Any]:
    started = time.time()
    if _sha256(BASELINE) != BASELINE_SHA256:
        raise RuntimeError("constant-scalar baseline changed")
    baseline = json.loads(BASELINE.read_text())
    if (
        baseline["evaluation_rows"] != [EVAL_START, EVAL_STOP]
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
    adapter = OwnedPerHeadTensorAttention.from_native(model.transformer.h[LAYER].attn)
    lowrank = LowRankSourcePattern(adapter)

    def native_mlp(event: facade.EarlyMLPEvent):
        return event.block.mlp(event.state)

    eval_count = EVAL_STOP - EVAL_START
    arm_names = ("native_recomputed",) + tuple(f"rank_{rank}" for rank in RANKS)
    stats = {
        arm: {cell: torch.zeros(eval_count, 5, dtype=torch.float64) for cell in CELLS}
        for arm in arm_names
    }
    native_patterns: list[torch.Tensor] = []
    approximate_patterns: dict[int, list[torch.Tensor]] = {rank: [] for rank in RANKS}
    closure_errors: list[float] = []
    for start in range(EVAL_START, EVAL_STOP, 4):
        local_start = start - EVAL_START
        batch_rows = rows[start:start + 4]
        tokens = batch_rows[:, :-1].to(device).contiguous()
        targets = batch_rows[:, 1:].to(device)
        batch_policy = {
            key: value[start:start + len(batch_rows)].to(device)
            for key, value in policy.items()
        }
        eligible = batch_policy["eligible"]
        batch_approximations: dict[int, torch.Tensor] = {}
        captured_native: list[torch.Tensor] = []

        def capture_attention(event: facade.AttentionEvent):
            if event.site != LAYER:
                return event.block.attn(event.state, event.first_value)
            with adapter.begin(event.state, event.first_value) as transaction:
                native = transaction.native_full_write()
                bus = transaction.first_value_bus()
                captured_native.append(transaction.source_pattern(
                    HEADS, batch_policy["successor"], eligible,
                ))
            closure_errors.append(
                transaction.closure.all_head_recomposition_relative_error
            )
            for rank in RANKS:
                batch_approximations[rank] = lowrank.source_pattern(
                    event.state, batch_policy["successor"], rank,
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
        native_patterns.append(captured_native[0][eligible].float().cpu())
        for rank in RANKS:
            approximate_patterns[rank].append(
                batch_approximations[rank][eligible].float().cpu()
            )

        for rank in RANKS:
            arm = f"rank_{rank}"

            def intervened_attention(
                event: facade.AttentionEvent, rank: int = rank,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                if event.site != LAYER:
                    return event.block.attn(event.state, event.first_value)
                with adapter.begin(event.state, event.first_value) as transaction:
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
                        pattern_override=batch_approximations[rank],
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

    baseline_native = torch.tensor(
        baseline["per_document"]["native"]["all_scored"], dtype=torch.float64,
    )
    if not torch.equal(stats["native_recomputed"]["all_scored"], baseline_native):
        maximum = float((
            stats["native_recomputed"]["all_scored"] - baseline_native
        ).abs().max())
        raise RuntimeError(f"recomputed native baseline differs by {maximum}")
    baseline_arms = baseline["summary"]["arms"]
    arms: dict[str, Any] = {
        "native": baseline_arms["native"],
        "edge_removed": baseline_arms["edge_removed"],
        "native_pattern_broadcast": baseline_arms["native_pattern_broadcast"],
    }
    for rank in RANKS:
        arm = f"rank_{rank}"
        arms[arm] = {
            cell: _cell_summary(value) for cell, value in stats[arm].items()
        }
    deletion = arms["edge_removed"]["copy_positive"]["delta_ce"]
    recoveries = {
        arm: 1 - value["copy_positive"]["delta_ce"] / deletion
        for arm, value in arms.items() if arm != "native"
    }
    gates = {
        "q1_rank128_executable_control": recoveries["rank_128"] >= 0.90,
        "q2_rank64_half_price": recoveries["rank_64"] >= 0.80,
        "q3_rank32_quarter_price": recoveries["rank_32"] >= 0.70,
        "q4_rank64_selective": (
            abs(arms["rank_64"]["repeat_negative"]["delta_ce"])
            <= min(0.02, 0.25 * deletion)
            and abs(arms["rank_64"]["nonrepeat"]["delta_ce"])
            <= min(0.02, 0.25 * deletion)
        ),
        "q5_curve_coherent": (
            recoveries["rank_96"] >= recoveries["rank_64"] - 0.05
            and recoveries["rank_64"] >= recoveries["rank_32"] - 0.05
        ),
    }
    selected_rank = None
    for rank in RANKS:
        arm = arms[f"rank_{rank}"]
        if (
            recoveries[f"rank_{rank}"] >= 0.90
            and arm["all_scored"]["delta_ce"] <= 0.001
            and abs(arm["repeat_negative"]["delta_ce"]) <= 0.01
            and abs(arm["nonrepeat"]["delta_ce"]) <= 0.01
        ):
            selected_rank = rank
            break
    native_pattern = torch.cat(native_patterns)
    scalar_metrics = {}
    width = adapter.width
    original_values = len(PROJECTIONS) * len(HEADS) * HEAD_DIM * width
    prices = {}
    for rank in RANKS:
        approximate = torch.cat(approximate_patterns[rank])
        scalar_metrics[str(rank)] = {
            "r2_h3_h4": _r2(native_pattern, approximate),
            "correlation_h3_h4": _correlation(native_pattern, approximate),
            "mean_absolute_error_h3_h4": (
                native_pattern - approximate
            ).abs().mean(0).tolist(),
        }
        values = len(PROJECTIONS) * len(HEADS) * rank * (width + HEAD_DIM)
        prices[str(rank)] = {
            "factor_values": values,
            "fraction_of_native_qk_slice_values": values / original_values,
            "native_qk_slice_values": original_values,
            "writer_projection_slice_values": width * len(HEADS) * HEAD_DIM,
        }
    return {
        "schema": "copy_edge_lowrank_qk_v1",
        "status": "exploratory_weights_only_factorization_on_exposed_eval_rows",
        "evaluation_rows": [EVAL_START, EVAL_STOP],
        "ranks": list(RANKS),
        "baseline_sha256": BASELINE_SHA256,
        "preregistration_sha256": file_sha256(PREREG),
        "runner_sha256": _sha256(Path(__file__)),
        "checkpoint": checkpoint.__dict__,
        "weight_singular_values": lowrank.spectra,
        "scalar_metrics": scalar_metrics,
        "prices": prices,
        "summary": {
            "arms": arms,
            "copy_positive_recovery_relative_to_edge_deletion": recoveries,
            "gates": gates,
            "selected_smallest_rank": selected_rank,
        },
        "maximum_attention_recomposition_relative_error": max(closure_errors),
        "new_per_document": {
            arm: {cell: value.tolist() for cell, value in cells.items()}
            for arm, cells in stats.items()
        },
        "runtime_seconds": time.time() - started,
        "claim_boundary": (
            "Weights-only SVD factors and disjoint exposed evaluation rows. The "
            "replacement still consumes the native L8 contextual input state."
        ),
    }


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"output already exists: {OUTPUT}")
    result = run()
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "recoveries": result["summary"][
            "copy_positive_recovery_relative_to_edge_deletion"
        ],
        "gates": result["summary"]["gates"],
        "selected_rank": result["summary"]["selected_smallest_rank"],
        "scalar_metrics": result["scalar_metrics"],
    }, indent=2, sort_keys=True), flush=True)
    print(f"wrote {OUTPUT} in {result['runtime_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()

