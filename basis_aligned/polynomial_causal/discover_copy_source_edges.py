#!/usr/bin/env python3
"""Fast cached-data discovery of exact L8 H3/H4 copy-source contributions."""

from __future__ import annotations

import argparse
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
from terminal_copy_attention_adapter import OwnedPerHeadTensorAttention


HERE = Path(__file__).resolve().parent
ROWS_PATH = (
    HERE.parent / "bilinear_quotient" /
    ".rowcache_terminal_copy_induction_v2/selection_natural.pt"
)
ROWS_FILE_SHA256 = "cc8e1c3e468b7bc249e0cf8fc00640955ae17251c7f0c7640350f65a86202cac"
ROWS_TENSOR_SHA256 = "625258ae1128823194fd27c94c241bd197dfd8daba77cfa2d1a0156ae1daaf8a"
PREREG = HERE / "COPY_SOURCE_EDGE_DISCOVERY_PREREGISTRATION.md"
HEADS = (3, 4)
LAYER = 8
WINDOW = 128
SCORE_START = 64
ARMS = (
    "native", "edge_mixed", "edge_fresh", "edge_broadcast", "edge_wrong",
    "heads_full",
)
CELLS = ("copy_positive", "repeat_negative", "nonrepeat", "all_scored")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def nearest_repeat_policy(
    rows: torch.Tensor, *, window: int = WINDOW, score_start: int = SCORE_START,
) -> dict[str, torch.Tensor]:
    """Construct the input-only nearest-repeat source and outcome-only score cells."""

    if (
        not torch.is_tensor(rows) or rows.dtype != torch.long or rows.ndim != 2
        or rows.shape[1] < 3 or window <= 0 or not 0 <= score_start < rows.shape[1] - 1
    ):
        raise ValueError("copy-source rows or policy dimensions are malformed")
    inputs, targets = rows[:, :-1], rows[:, 1:]
    batch, sequence = inputs.shape
    positions = torch.arange(sequence, device=rows.device)
    source = torch.full((batch, sequence), -1, device=rows.device, dtype=torch.long)
    for distance in range(1, min(window, sequence - 1) + 1):
        candidate = (positions - distance).clamp_min(0)
        candidate_token = inputs.gather(1, candidate.expand(batch, -1))
        valid_position = positions >= distance
        choose = (
            (source < 0) & valid_position.unsqueeze(0) & (inputs == candidate_token)
        )
        source[choose] = candidate.expand(batch, -1)[choose]
    eligible = source >= 0
    safe_source = source.clamp_min(0)
    successor = (safe_source + 1).clamp_max(sequence - 1)
    successor_token = inputs.gather(1, successor)
    scored = positions.unsqueeze(0) >= score_start
    positive = eligible & scored & (targets == successor_token)
    repeat_negative = eligible & scored & ~positive
    nonrepeat = scored & ~eligible
    return {
        "source": safe_source,
        "successor": successor,
        "eligible": eligible,
        "copy_positive": positive,
        "repeat_negative": repeat_negative,
        "nonrepeat": nonrepeat,
        "all_scored": scored.expand(batch, -1),
    }


def _empty_stats(rows: int) -> dict[str, dict[str, torch.Tensor]]:
    return {
        arm: {
            # count, native NLL, candidate-native NLL, KL(native||candidate), correct
            cell: torch.zeros(rows, 5, dtype=torch.float64)
            for cell in CELLS
        }
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


def _summarize(stats: dict[str, dict[str, torch.Tensor]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for arm in ARMS:
        output[arm] = {}
        for cell in CELLS:
            values = stats[arm][cell]
            count = int(values[:, 0].sum())
            supported = values[:, 0] > 0
            document_effect = values[supported, 2] / values[supported, 0]
            document_mean = float(document_effect.mean()) if len(document_effect) else None
            document_se = (
                float(document_effect.std(unbiased=True) / math.sqrt(len(document_effect)))
                if len(document_effect) > 1 else None
            )
            output[arm][cell] = {
                "count": count,
                "supporting_documents": int(supported.sum()),
                "native_ce": float(values[:, 1].sum() / max(count, 1)),
                "arm_ce": float((values[:, 1] + values[:, 2]).sum() / max(count, 1)),
                "delta_ce": float(values[:, 2].sum() / max(count, 1)),
                "native_to_arm_kl": float(values[:, 3].sum() / max(count, 1)),
                "arm_accuracy": float(values[:, 4].sum() / max(count, 1)),
                "document_mean_delta_ce": document_mean,
                "document_se_delta_ce": document_se,
            }
    mixed = output["edge_mixed"]["copy_positive"]["delta_ce"]
    full = output["heads_full"]["copy_positive"]["delta_ce"]
    wrong = output["edge_wrong"]["copy_positive"]["delta_ce"]
    negative = output["edge_mixed"]["repeat_negative"]["delta_ce"]
    nonrepeat = output["edge_mixed"]["nonrepeat"]["delta_ce"]
    share = mixed / full if full > 0 else None
    gates = {
        "p1_consequential_edge": mixed >= 0.05,
        "p2_share_at_least_25pct": share is not None and share >= 0.25,
        "p3_directional_source": wrong <= 0.5 * mixed,
        "p4_input_policy_specificity": (
            mixed - negative >= 0.03 and nonrepeat <= 0.25 * mixed
        ),
    }
    return {
        "arms": output,
        "derived": {
            "edge_mixed_share_of_full_head_damage": share,
            "fresh_minus_broadcast_copy_delta_ce": (
                output["edge_fresh"]["copy_positive"]["delta_ce"]
                - output["edge_broadcast"]["copy_positive"]["delta_ce"]
            ),
            "copy_minus_repeat_negative_delta_ce": mixed - negative,
            "copy_to_nonrepeat_delta_ce_ratio": (
                mixed / nonrepeat if nonrepeat > 0 else None
            ),
        },
        "gates": gates,
        "escalate_to_128": gates["p1_consequential_edge"] and gates[
            "p2_share_at_least_25pct"
        ],
    }


@torch.no_grad()
def run(row_count: int) -> dict[str, Any]:
    if row_count not in (32, 128):
        raise ValueError("discovery row count must be 32 or 128")
    started = time.time()
    if file_sha256(ROWS_PATH) != ROWS_FILE_SHA256:
        raise RuntimeError("cached selection-role bytes changed")
    payload = torch.load(ROWS_PATH, map_location="cpu", weights_only=True)
    rows = payload["rows"]
    if tensor_sha256(rows) != ROWS_TENSOR_SHA256 or tuple(rows.shape) != (192, 257):
        raise RuntimeError("cached selection rows changed")
    rows = rows[:row_count].contiguous()
    records = payload["records"][:row_count]
    document_ids = [record["document_id"] for record in records]
    policy = nearest_repeat_policy(rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = facade.load_bilin18(device=device, dtype=torch.bfloat16)
    adapter = OwnedPerHeadTensorAttention.from_native(model.transformer.h[LAYER].attn)
    stats = _empty_stats(row_count)
    closure_errors: list[float] = []

    def native_attention(event: facade.AttentionEvent):
        return event.block.attn(event.state, event.first_value)

    def native_mlp(event: facade.EarlyMLPEvent):
        return event.block.mlp(event.state)

    for start in range(0, row_count, 4):
        batch_rows = rows[start:start + 4]
        tokens = batch_rows[:, :-1].to(device).contiguous()
        targets = batch_rows[:, 1:].to(device)
        batch_policy = {
            key: value[start:start + len(batch_rows)].to(device)
            for key, value in policy.items()
        }
        native_logits = facade.forward_with_dispatch(
            model, tokens, native_attention, native_mlp,
        )
        native_logprob = F.log_softmax(native_logits.float(), dim=-1)
        native_nll = -native_logprob.gather(2, targets[..., None]).squeeze(2)
        masks = {cell: batch_policy[cell] for cell in CELLS}
        _record(
            stats["native"], start, native_logprob, native_nll, None, targets, masks,
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
                    if arm == "heads_full":
                        removed = transaction.select(HEADS) * batch_policy[
                            "eligible"
                        ][..., None].to(native.dtype)
                    else:
                        source = (
                            batch_policy["source"] if arm == "edge_wrong"
                            else batch_policy["successor"]
                        )
                        route = {
                            "edge_mixed": "mixed",
                            "edge_fresh": "fresh",
                            "edge_broadcast": "broadcast",
                            "edge_wrong": "mixed",
                        }[arm]
                        removed = transaction.source_write(
                            HEADS, source, batch_policy["eligible"], route=route,
                        )
                closure_errors.append(
                    transaction.closure.all_head_recomposition_relative_error
                )
                return native - removed, bus

            candidate_logits = facade.forward_with_dispatch(
                model, tokens, intervened_attention, native_mlp,
            )
            _record(
                stats[arm], start, native_logprob, native_nll, candidate_logits,
                targets, masks,
            )
            del candidate_logits
        del native_logprob, native_nll

    summary = _summarize(stats)
    return {
        "schema": "copy_source_edge_discovery_v1",
        "status": "exploratory_reused_exposed_selection_rows",
        "row_count": row_count,
        "document_ids": document_ids,
        "rows_file_sha256": ROWS_FILE_SHA256,
        "rows_tensor_sha256": ROWS_TENSOR_SHA256,
        "preregistration_sha256": file_sha256(PREREG),
        "checkpoint": checkpoint.__dict__,
        "policy": {
            "window": WINDOW,
            "score_start": SCORE_START,
            "input_only_nearest_equal_token_source": True,
            "eligible_positions_all": int(policy["eligible"].sum()),
            "copy_positive_scored": int(policy["copy_positive"].sum()),
            "repeat_negative_scored": int(policy["repeat_negative"].sum()),
        },
        "maximum_attention_recomposition_relative_error": max(closure_errors),
        "summary": summary,
        "per_document": {
            arm: {cell: values.tolist() for cell, values in cells.items()}
            for arm, cells in stats.items()
        },
        "runtime_seconds": time.time() - started,
        "claim_boundary": (
            "Discovery on an already exposed selection role. This tests an exact L8 "
            "source-edge intervention, not a standalone extraction or fresh-data claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, choices=(32, 128), default=32)
    args = parser.parse_args()
    output = HERE / f"copy_source_edge_discovery_{args.rows}_results.json"
    if output.exists():
        raise RuntimeError(f"discovery output already exists: {output}")
    result = run(args.rows)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True), flush=True)
    print(f"wrote {output} in {result['runtime_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
