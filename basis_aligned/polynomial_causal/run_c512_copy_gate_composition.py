#!/usr/bin/env python3
"""Cross frozen C512 MLP0 with the selected shared-rank-256 L8 copy gate."""

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
from mlp0_native_down_program import load_program
from run_copy_edge_constant_scalar import _cell_summary, _record
from run_copy_edge_lowrank_qk import _r2
from run_copy_edge_shared_hosvd import SharedInputSourcePattern
from terminal_copy_attention_adapter import OwnedPerHeadTensorAttention


HERE = Path(__file__).resolve().parent
BQ = HERE.parent / "bilinear_quotient"
PREREG = HERE / "C512_COPY_GATE_COMPOSITION_PREREGISTRATION.md"
OUTPUT = HERE / "c512_copy_gate_composition_results.json"
C512_PATH = BQ / "mlp0_native_down_hierarchy_v1_programs/C512_at_C512.bin"
C512_SHA256 = "3ecf43b485d343bc5413e817dbd4236e5ce6cdaa7a3e0e653214e812b84ce470"
C512_RECEIPT = BQ / "mlp0_native_down_hierarchy_v1_fit_receipt.json"
C512_RECEIPT_SHA256 = "79d0069864e9df521a99fc36531dd86c7ed31106f58f029d681fb1788a269f82"
HOSVD_RESULT = HERE / "copy_edge_shared_hosvd_results.json"
HOSVD_RESULT_SHA256 = "8e2e27a7231472bce1389167414898c343ebf8ac2e1bcac6b78f220fe7b5801e"
EVAL_START = 32
EVAL_STOP = 128
RANK = 256
ARMS = ("NN", "CN", "NH", "CH", "ZN")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state_metrics(
    native: torch.Tensor, candidate: torch.Tensor, zero: torch.Tensor,
) -> dict[str, float]:
    native, candidate, zero = native.double(), candidate.double(), zero.double()
    error = (candidate - native).square().sum()
    centered = (native - native.mean(0, keepdim=True)).square().sum().clamp_min(1e-30)
    zero_error = (zero - native).square().sum().clamp_min(1e-30)
    cosine = F.cosine_similarity(candidate, native, dim=1).mean()
    relative_rms = (
        (candidate - native).square().mean() / native.square().mean().clamp_min(1e-30)
    ).sqrt()
    return {
        "r2": float(1 - error / centered),
        "mean_position_cosine": float(cosine),
        "relative_rms_error": float(relative_rms),
        "fraction_zero_mlp0_gap_removed": float(1 - error / zero_error),
    }


@torch.no_grad()
def run() -> dict[str, Any]:
    started = time.time()
    for path, expected in (
        (C512_PATH, C512_SHA256),
        (C512_RECEIPT, C512_RECEIPT_SHA256),
        (HOSVD_RESULT, HOSVD_RESULT_SHA256),
    ):
        if _sha256(path) != expected:
            raise RuntimeError(f"frozen input changed: {path}")
    receipt = json.loads(C512_RECEIPT.read_text())
    if receipt["programs"]["C512_at_C512"]["sha256"] != C512_SHA256:
        raise RuntimeError("C512 fit receipt no longer binds the program")
    hosvd_result = json.loads(HOSVD_RESULT.read_text())
    if hosvd_result["summary"]["selected_price_frontier_program"] != {
        "variant": "canonical", "rank": RANK, "factor_values": 557056,
        "copy_recovery": hosvd_result["summary"][
            "copy_positive_recovery_relative_to_edge_deletion"
        ]["canonical_256"],
    }:
        raise RuntimeError("selected HOSVD program changed")
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
    adapter = OwnedPerHeadTensorAttention.from_native(model.transformer.h[LAYER].attn)
    shared = SharedInputSourcePattern(adapter)
    c512 = load_program(C512_PATH)
    c512_tensors = {
        name: c512[name].to(device=device, dtype=torch.bfloat16)
        for name in ("intercept", "left", "right")
    }

    def c512_write(event: facade.EarlyMLPEvent) -> torch.Tensor:
        mlp = event.block.mlp
        hidden = mlp.Left(event.state) * mlp.Right(event.state)
        latent = F.linear(hidden, c512_tensors["right"])
        down = F.linear(latent, c512_tensors["left"], c512_tensors["intercept"])
        return down + mlp.Down_bias.to(down.dtype)

    def canonical_pattern_and_z(
        state: torch.Tensor, source: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        right = shared.right["canonical"][:RANK]
        latent = F.linear(state, right)
        projected = {
            key: shared._rotate(F.linear(
                latent, shared.cores["canonical"][key][:, :RANK],
            ))
            for key in shared.slice_order
        }
        batch, sequence = source.shape
        gather = source[..., None].expand(batch, sequence, 128)
        patterns = []
        for head in HEADS:
            key = torch.gather(projected[("k", head)], 1, gather)
            key2 = torch.gather(projected[("k2", head)], 1, gather)
            first = torch.einsum(
                "btd,btd->bt", projected[("q", head)], key,
            ) / 128
            second = torch.einsum(
                "btd,btd->bt", projected[("q2", head)], key2,
            ) / 128
            patterns.append(first * second)
        return torch.stack(patterns, 2), latent

    eval_count = EVAL_STOP - EVAL_START
    stats = {
        arm: {cell: torch.zeros(eval_count, 5, dtype=torch.float64) for cell in CELLS}
        for arm in ARMS
    }
    z_values: dict[str, dict[str, list[torch.Tensor]]] = {
        arm: {"all_scored": [], "copy_positive": []} for arm in ARMS
    }
    native_patterns: dict[str, list[torch.Tensor]] = {arm: [] for arm in ARMS}
    hosvd_patterns: dict[str, list[torch.Tensor]] = {arm: [] for arm in ("NH", "CH")}
    closure_errors: list[float] = []
    state_identity_errors = {"NN_vs_NH": 0.0, "CN_vs_CH": 0.0}

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
        batch_captures: dict[str, dict[str, torch.Tensor]] = {}
        logits_by_arm: dict[str, torch.Tensor] = {}

        for arm in ARMS:
            upstream, gate = arm
            captured: dict[str, torch.Tensor] = {}

            def attention_dispatcher(
                event: facade.AttentionEvent, gate: str = gate,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                if event.site != LAYER:
                    return event.block.attn(event.state, event.first_value)
                with adapter.begin(event.state, event.first_value) as transaction:
                    native = transaction.native_full_write()
                    bus = transaction.first_value_bus()
                    native_pattern = transaction.source_pattern(
                        HEADS, batch_policy["successor"], batch_policy["eligible"],
                    )
                    approximate, z = canonical_pattern_and_z(
                        event.state, batch_policy["successor"],
                    )
                    captured.update({
                        "z": z.clone(),
                        "native_pattern": native_pattern.clone(),
                        "hosvd_pattern": approximate.clone(),
                    })
                    if gate == "H":
                        removed = transaction.source_write(
                            HEADS,
                            batch_policy["successor"],
                            batch_policy["eligible"],
                            route="mixed",
                        )
                        replacement = transaction.source_write(
                            HEADS,
                            batch_policy["successor"],
                            batch_policy["eligible"],
                            route="broadcast",
                            pattern_override=approximate,
                        )
                        native = native - removed + replacement
                closure_errors.append(
                    transaction.closure.all_head_recomposition_relative_error
                )
                return native, bus

            def mlp_dispatcher(
                event: facade.EarlyMLPEvent, upstream: str = upstream,
            ) -> torch.Tensor:
                if event.site == 0 and upstream == "C":
                    return c512_write(event)
                if event.site == 0 and upstream == "Z":
                    return torch.zeros_like(event.state)
                return event.block.mlp(event.state)

            logits_by_arm[arm] = facade.forward_with_dispatch(
                model, tokens, attention_dispatcher, mlp_dispatcher,
            )
            batch_captures[arm] = captured

        state_identity_errors["NN_vs_NH"] = max(
            state_identity_errors["NN_vs_NH"],
            float((batch_captures["NN"]["z"] - batch_captures["NH"]["z"]).abs().max()),
        )
        state_identity_errors["CN_vs_CH"] = max(
            state_identity_errors["CN_vs_CH"],
            float((batch_captures["CN"]["z"] - batch_captures["CH"]["z"]).abs().max()),
        )
        native_logprob = F.log_softmax(logits_by_arm["NN"].float(), dim=-1)
        native_nll = -native_logprob.gather(2, targets[..., None]).squeeze(2)
        for arm in ARMS:
            _record(
                stats[arm], local_start, native_logprob, native_nll,
                None if arm == "NN" else logits_by_arm[arm], targets, masks,
            )
            capture = batch_captures[arm]
            for cell in ("all_scored", "copy_positive"):
                z_values[arm][cell].append(capture["z"][masks[cell]].float().cpu())
            eligible = batch_policy["eligible"]
            native_patterns[arm].append(
                capture["native_pattern"][eligible].float().cpu()
            )
            if arm in hosvd_patterns:
                hosvd_patterns[arm].append(
                    capture["hosvd_pattern"][eligible].float().cpu()
                )
        del logits_by_arm, native_logprob, native_nll

    if any(value != 0 for value in state_identity_errors.values()):
        raise RuntimeError(f"L8 pre-gate state changed across gate arms: {state_identity_errors}")
    arms = {
        arm: {cell: _cell_summary(value) for cell, value in cells.items()}
        for arm, cells in stats.items()
    }
    interactions = {
        cell: (
            arms["CH"][cell]["delta_ce"]
            - arms["CN"][cell]["delta_ce"]
            - arms["NH"][cell]["delta_ce"]
        ) for cell in CELLS
    }
    state_metrics = {}
    for cell in ("all_scored", "copy_positive"):
        state_metrics[cell] = _state_metrics(
            torch.cat(z_values["NN"][cell]),
            torch.cat(z_values["CN"][cell]),
            torch.cat(z_values["ZN"][cell]),
        )
    pattern_metrics = {}
    for arm in ("NH", "CH"):
        target = torch.cat(native_patterns[arm])
        predicted = torch.cat(hosvd_patterns[arm])
        pattern_metrics[arm] = {
            "r2_h3_h4": _r2(target, predicted),
            "mean_absolute_error_h3_h4": (target - predicted).abs().mean(0).tolist(),
        }
    native_copy_accuracy = arms["NN"]["copy_positive"]["arm_accuracy"]
    gates = {
        "j1_c512_observational": arms["CN"]["all_scored"]["delta_ce"] <= 0.0075,
        "j2_c512_preserves_copy": (
            arms["CN"]["copy_positive"]["delta_ce"] <= 0.02
            and native_copy_accuracy - arms["CN"]["copy_positive"]["arm_accuracy"] <= 0.01
        ),
        "j3_downstream_state": (
            state_metrics["all_scored"]["r2"] >= 0.90
            and state_metrics["all_scored"]["fraction_zero_mlp0_gap_removed"] >= 0.75
        ),
        "j4_joint_composition": (
            arms["CH"]["all_scored"]["delta_ce"] <= 0.0075
            and arms["CH"]["copy_positive"]["delta_ce"] <= 0.03
            and native_copy_accuracy - arms["CH"]["copy_positive"]["arm_accuracy"] <= 0.01
        ),
        "j5_bounded_interaction": (
            abs(interactions["all_scored"]) <= 0.002
            and abs(interactions["copy_positive"]) <= 0.01
        ),
        "j6_hosvd_stable_under_c512": all(
            value >= 0.90 for value in pattern_metrics["CH"]["r2_h3_h4"]
        ),
    }
    return {
        "schema": "c512_copy_gate_composition_v1",
        "status": "exploratory_composition_on_exposed_eval_rows",
        "evaluation_rows": [EVAL_START, EVAL_STOP],
        "preregistration_sha256": file_sha256(PREREG),
        "runner_sha256": _sha256(Path(__file__)),
        "c512_sha256": C512_SHA256,
        "c512_receipt_sha256": C512_RECEIPT_SHA256,
        "hosvd_result_sha256": HOSVD_RESULT_SHA256,
        "checkpoint": checkpoint.__dict__,
        "summary": {
            "arms": arms,
            "ce_composition_interaction": interactions,
            "state_metrics": state_metrics,
            "hosvd_pattern_metrics": pattern_metrics,
            "gates": gates,
            "all_gates_pass": all(gates.values()),
        },
        "state_identity_errors": state_identity_errors,
        "maximum_attention_recomposition_relative_error": max(closure_errors),
        "per_document": {
            arm: {cell: value.tolist() for cell, value in cells.items()}
            for arm, cells in stats.items()
        },
        "runtime_seconds": time.time() - started,
        "claim_boundary": (
            "Exploratory composition on exposed rows. C512 is frozen authoritative "
            "input, but this cross does not alter its prior equivalence authority."
        ),
    }


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"output already exists: {OUTPUT}")
    result = run()
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True), flush=True)
    print(f"wrote {OUTPUT} in {result['runtime_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()

