#!/usr/bin/env python3
"""Raw and norm-canonical shared-input HOSVD curves for the L8 copy gate."""

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
from run_copy_edge_lowrank_qk import _correlation, _r2
from terminal_copy_attention_adapter import OwnedPerHeadTensorAttention


HERE = Path(__file__).resolve().parent
PREREG = HERE / "COPY_EDGE_SHARED_HOSVD_PREREGISTRATION.md"
BASELINE = HERE / "copy_edge_constant_scalar_results.json"
BASELINE_SHA256 = "3da06d79c0d28bbb6f4d13082aa8c0dcc1bd3315a5ef9ec485e347136774603f"
OUTPUT = HERE / "copy_edge_shared_hosvd_results.json"
EVAL_START = 32
EVAL_STOP = 128
RANKS = (64, 128, 192, 256, 320, 384, 512, 1024)
VARIANTS = ("raw", "canonical")
PROJECTIONS = ("q", "k", "q2", "k2")
HEAD_DIM = 128


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SharedInputSourcePattern:
    """Own two HOSVD input bases and their eight original-scale cores."""

    def __init__(self, adapter: OwnedPerHeadTensorAttention) -> None:
        if adapter.head_dim != HEAD_DIM:
            raise ValueError("shared copy-gate topology changed")
        self.dtype = adapter.q.dtype
        self.width = adapter.width
        self.inv_freq = adapter.inv_freq.detach().clone()
        self.slice_order = tuple(
            (name, head) for name in PROJECTIONS for head in HEADS
        )
        slices = {
            (name, head): getattr(adapter, name)[
                head * HEAD_DIM:(head + 1) * HEAD_DIM
            ].float()
            for name, head in self.slice_order
        }
        norms = {key: value.norm() for key, value in slices.items()}
        stacks = {
            "raw": torch.cat([slices[key] for key in self.slice_order], 0),
            "canonical": torch.cat([
                slices[key] / norms[key] for key in self.slice_order
            ], 0),
        }
        self.norms = {
            f"{name}_h{head}": float(norms[(name, head)])
            for name, head in self.slice_order
        }
        self.right: dict[str, torch.Tensor] = {}
        self.cores: dict[str, dict[tuple[str, int], torch.Tensor]] = {}
        self.spectra: dict[str, list[float]] = {}
        for variant in VARIANTS:
            _, singular, vh = torch.linalg.svd(stacks[variant], full_matrices=False)
            self.spectra[variant] = singular.detach().cpu().tolist()
            self.right[variant] = vh.to(self.dtype)
            basis = vh.transpose(0, 1)
            self.cores[variant] = {
                key: (slices[key] @ basis).to(self.dtype) for key in self.slice_order
            }

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

    def all_source_patterns(
        self, state: torch.Tensor, source_indices: torch.Tensor,
    ) -> dict[tuple[str, int], torch.Tensor]:
        if (
            state.ndim != 3 or state.shape[-1] != self.width
            or source_indices.shape != state.shape[:2]
            or source_indices.dtype != torch.long
            or source_indices.device != state.device
        ):
            raise ValueError("shared source-pattern input is malformed")
        batch, sequence = source_indices.shape
        gather = source_indices[..., None].expand(batch, sequence, HEAD_DIM)
        output: dict[tuple[str, int], torch.Tensor] = {}
        for variant in VARIANTS:
            latent = F.linear(state, self.right[variant])
            for rank in RANKS:
                projected = {
                    key: self._rotate(F.linear(
                        latent[..., :rank], self.cores[variant][key][:, :rank],
                    ))
                    for key in self.slice_order
                }
                patterns = []
                for head in HEADS:
                    key = torch.gather(projected[("k", head)], 1, gather)
                    key2 = torch.gather(projected[("k2", head)], 1, gather)
                    first = torch.einsum(
                        "btd,btd->bt", projected[("q", head)], key,
                    ) / HEAD_DIM
                    second = torch.einsum(
                        "btd,btd->bt", projected[("q2", head)], key2,
                    ) / HEAD_DIM
                    patterns.append(first * second)
                output[(variant, rank)] = torch.stack(patterns, 2)
        return output


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
    shared = SharedInputSourcePattern(adapter)

    def native_mlp(event: facade.EarlyMLPEvent):
        return event.block.mlp(event.state)

    eval_count = EVAL_STOP - EVAL_START
    candidates = tuple(
        (variant, rank) for variant in VARIANTS for rank in RANKS
    )
    arm_names = ("native_recomputed",) + tuple(
        f"{variant}_{rank}" for variant, rank in candidates
    )
    stats = {
        arm: {cell: torch.zeros(eval_count, 5, dtype=torch.float64) for cell in CELLS}
        for arm in arm_names
    }
    native_patterns: list[torch.Tensor] = []
    approximations: dict[tuple[str, int], list[torch.Tensor]] = {
        candidate: [] for candidate in candidates
    }
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
        captured_native: list[torch.Tensor] = []
        batch_approximations: dict[tuple[str, int], torch.Tensor] = {}

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
            batch_approximations.update(shared.all_source_patterns(
                event.state, batch_policy["successor"],
            ))
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
        for candidate in candidates:
            approximations[candidate].append(
                batch_approximations[candidate][eligible].float().cpu()
            )

        for variant, rank in candidates:
            arm = f"{variant}_{rank}"

            def intervened_attention(
                event: facade.AttentionEvent,
                candidate: tuple[str, int] = (variant, rank),
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
                        pattern_override=batch_approximations[candidate],
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
    for variant, rank in candidates:
        arm = f"{variant}_{rank}"
        arms[arm] = {
            cell: _cell_summary(value) for cell, value in stats[arm].items()
        }
    deletion = arms["edge_removed"]["copy_positive"]["delta_ce"]
    recoveries = {
        arm: 1 - value["copy_positive"]["delta_ce"] / deletion
        for arm, value in arms.items() if arm != "native"
    }
    gates = {
        "h1_full_rank_controls": (
            recoveries["raw_1024"] >= 0.90
            and recoveries["canonical_1024"] >= 0.90
        ),
        "h2_canonical256_at_least_80pct": recoveries["canonical_256"] >= 0.80,
        "h3_canonical256_at_least_90pct": recoveries["canonical_256"] >= 0.90,
        "h4_canonicalization_improves_256": (
            recoveries["canonical_256"] - recoveries["raw_256"] >= 0.02
        ),
        "h5_canonical256_selective": (
            abs(arms["canonical_256"]["repeat_negative"]["delta_ce"])
            <= min(0.02, 0.25 * deletion)
            and abs(arms["canonical_256"]["nonrepeat"]["delta_ce"])
            <= min(0.02, 0.25 * deletion)
        ),
    }
    passing = []
    for variant, rank in candidates:
        arm = arms[f"{variant}_{rank}"]
        price = 2176 * rank
        if (
            price <= 655_360
            and recoveries[f"{variant}_{rank}"] >= 0.90
            and arm["all_scored"]["delta_ce"] <= 0.001
            and abs(arm["repeat_negative"]["delta_ce"]) <= 0.01
            and abs(arm["nonrepeat"]["delta_ce"]) <= 0.01
        ):
            passing.append((price, -recoveries[f"{variant}_{rank}"], variant, rank))
    gates["h6_improves_independent_rank64_frontier"] = bool(passing)
    selected = None
    if passing:
        price, negative_recovery, variant, rank = min(passing)
        selected = {
            "variant": variant,
            "rank": rank,
            "factor_values": price,
            "copy_recovery": -negative_recovery,
        }
    native_pattern = torch.cat(native_patterns)
    scalar_metrics = {}
    prices = {}
    for variant, rank in candidates:
        approximate = torch.cat(approximations[(variant, rank)])
        key = f"{variant}_{rank}"
        scalar_metrics[key] = {
            "r2_h3_h4": _r2(native_pattern, approximate),
            "correlation_h3_h4": _correlation(native_pattern, approximate),
            "mean_absolute_error_h3_h4": (
                native_pattern - approximate
            ).abs().mean(0).tolist(),
        }
        values = 2176 * rank
        prices[key] = {
            "factor_values": values,
            "fraction_of_native_qk_slice_values": values / 1_179_648,
            "fraction_of_independent_rank64_values": values / 655_360,
            "writer_projection_slice_values": 294_912,
        }
    return {
        "schema": "copy_edge_shared_hosvd_v1",
        "status": "exploratory_weights_only_factorization_on_exposed_eval_rows",
        "evaluation_rows": [EVAL_START, EVAL_STOP],
        "ranks": list(RANKS),
        "variants": list(VARIANTS),
        "baseline_sha256": BASELINE_SHA256,
        "preregistration_sha256": file_sha256(PREREG),
        "runner_sha256": _sha256(Path(__file__)),
        "checkpoint": checkpoint.__dict__,
        "slice_frobenius_norms": shared.norms,
        "shared_input_singular_values": shared.spectra,
        "scalar_metrics": scalar_metrics,
        "prices": prices,
        "summary": {
            "arms": arms,
            "copy_positive_recovery_relative_to_edge_deletion": recoveries,
            "canonical_minus_raw_recovery_by_rank": {
                str(rank): recoveries[f"canonical_{rank}"] - recoveries[f"raw_{rank}"]
                for rank in RANKS
            },
            "gates": gates,
            "selected_price_frontier_program": selected,
        },
        "maximum_attention_recomposition_relative_error": max(closure_errors),
        "new_per_document": {
            arm: {cell: value.tolist() for cell, value in cells.items()}
            for arm, cells in stats.items()
        },
        "runtime_seconds": time.time() - started,
        "claim_boundary": (
            "Weights-only shared factorization on exposed evaluation rows. The "
            "replacement still consumes the native L8 contextual input state."
        ),
    }


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"output already exists: {OUTPUT}")
    result = run()
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "recoveries": {
            key: value for key, value in result["summary"][
                "copy_positive_recovery_relative_to_edge_deletion"
            ].items() if key.startswith(("raw_", "canonical_"))
        },
        "canonical_minus_raw": result["summary"][
            "canonical_minus_raw_recovery_by_rank"
        ],
        "gates": result["summary"]["gates"],
        "selected": result["summary"]["selected_price_frontier_program"],
    }, indent=2, sort_keys=True), flush=True)
    print(f"wrote {OUTPUT} in {result['runtime_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()

