#!/usr/bin/env python3
"""Exact MLP0 token/context tensor split and no-fit causal factorial.

The three fixed branches are token-token (TT), symmetric token/context cross (X),
and context-context (CC). Hard routing is absent. The physical factorial is a
diagnostic: every arm computes native MLP0 and subtracts omitted analytical branches,
so it earns no executable-compression credit.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time
from typing import Mapping

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BQ = HERE.parent / "bilinear_quotient"
for root in (ROOT, HERE, BQ):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import bilin18_observed_model_facade as facade
import run_mlp1_sparse_c512_continue_factorial_v1_fit as rows_parent


ROWS_RECEIPT = BQ / "mlp1_sparse_c512_continue_factorial_v1_rows_receipt.json"
PREREGISTRATION = HERE / "MLP0_TOKEN_CONTEXT_TENSOR_FACTORIAL_DISCOVERY_PREREGISTRATION.md"
OUTPUT = HERE / "mlp0_token_context_tensor_factorial_discovery.json"

BRANCHES = ("TT", "X", "CC")
DOCUMENT_BATCH = 4
SCORING = slice(64, 256)
D = 1152
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 84_119


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def arm_name(subset: frozenset[str]) -> str:
    if not subset:
        return "EMPTY"
    return "+".join(branch for branch in BRANCHES if branch in subset)


ARMS = tuple(
    frozenset(branch for index, branch in enumerate(BRANCHES) if mask & (1 << index))
    for mask in range(1 << len(BRANCHES))
)


def split_normalized_state(
    normalized: torch.Tensor,
    token_base: torch.Tensor,
    context_write: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split observed normalized state into a token ray plus exact residual context."""

    if normalized.shape != token_base.shape or normalized.shape != context_write.shape \
            or normalized.shape[-1] != D:
        raise ValueError("MLP0 token/context split shapes changed")
    z = normalized.float()
    token = token_base.float()
    raw = token + context_write.float()
    denominator = raw.square().sum(-1, keepdim=True).clamp_min(1e-30)
    scale = (z * raw).sum(-1, keepdim=True) / denominator
    token_part = scale * token
    context_part = z - token_part
    collinearity_error = (z - scale * raw).square().sum(-1) \
        / z.square().sum(-1).clamp_min(1e-30)
    return token_part, context_part, collinearity_error


def quadratic_tensor_branches(
    token_part: torch.Tensor,
    context_part: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    down: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Apply the exact four-term bilinear expansion, grouping the two cross terms."""

    if token_part.shape != context_part.shape or token_part.shape[-1] != left.shape[1] \
            or left.shape != right.shape or down.shape[1] != left.shape[0]:
        raise ValueError("quadratic branch shapes changed")
    left_token = F.linear(token_part, left)
    right_token = F.linear(token_part, right)
    left_context = F.linear(context_part, left)
    right_context = F.linear(context_part, right)
    return {
        "TT": F.linear(left_token * right_token, down),
        "X": F.linear(
            left_token * right_context + left_context * right_token, down,
        ),
        "CC": F.linear(left_context * right_context, down),
    }


def full_float_quadratic(
    state: torch.Tensor, left: torch.Tensor, right: torch.Tensor, down: torch.Tensor,
) -> torch.Tensor:
    state = state.float()
    return F.linear(F.linear(state, left) * F.linear(state, right), down)


def mobius_dividends(performance: Mapping[frozenset[str], float]) -> dict[str, float]:
    if set(performance) != set(ARMS):
        raise ValueError("factorial performance cube changed")
    output = {}
    for subset in ARMS:
        value = 0.0
        items = tuple(subset)
        for size in range(len(items) + 1):
            for chosen in itertools.combinations(items, size):
                child = frozenset(chosen)
                value += (-1.0) ** (len(subset) - len(child)) * performance[child]
        output[arm_name(subset)] = value
    return output


def shapley_values(performance: Mapping[frozenset[str], float]) -> dict[str, float]:
    if set(performance) != set(ARMS):
        raise ValueError("factorial performance cube changed")
    n = len(BRANCHES)
    output = {}
    for branch in BRANCHES:
        total = 0.0
        others = tuple(item for item in BRANCHES if item != branch)
        for size in range(len(others) + 1):
            coefficient = math.factorial(size) * math.factorial(n - size - 1) \
                / math.factorial(n)
            for chosen in itertools.combinations(others, size):
                subset = frozenset(chosen)
                total += coefficient * (
                    performance[subset | {branch}] - performance[subset]
                )
        output[branch] = total
    return output


def lexical_dag_decomposition(
    token_table: torch.Tensor,
    incidence: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mean + overlapping-DAG least-squares term + exact token-private residual."""

    if token_table.ndim != 2 or incidence.ndim != 2 \
            or len(token_table) != len(incidence):
        raise ValueError("lexical DAG shapes changed")
    if weights is None:
        weights = torch.ones(len(token_table), dtype=token_table.dtype, device=token_table.device)
    if weights.shape != (len(token_table),) or not bool((weights > 0).all()):
        raise ValueError("lexical DAG weights changed")
    mean = (weights[:, None] * token_table).sum(0) / weights.sum()
    centered = token_table - mean
    weighted_incidence = incidence * weights.sqrt()[:, None]
    weighted_target = centered * weights.sqrt()[:, None]
    atoms = torch.linalg.lstsq(weighted_incidence, weighted_target).solution
    shared = incidence @ atoms
    private = centered - shared
    return mean, atoms, shared, private


def bootstrap_mean_interval(
    values: torch.Tensor, *, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    values = values.detach().cpu().double().flatten()
    if len(values) < 2 or not bool(torch.isfinite(values).all()):
        raise ValueError("bootstrap values changed")
    generator = torch.Generator().manual_seed(seed)
    means = []
    for start in range(0, draws, 1_000):
        count = min(1_000, draws - start)
        indices = torch.randint(len(values), (count, len(values)), generator=generator)
        means.append(values[indices].mean(1))
    samples = torch.cat(means).sort().values
    return {
        "mean": float(values.mean()),
        "bootstrap_95_low": float(samples[int(math.floor(0.025 * (draws - 1)))]),
        "bootstrap_95_high": float(samples[int(math.ceil(0.975 * (draws - 1)))]),
        "documents": len(values),
        "draws": draws,
    }


@torch.no_grad()
def score_role(model, rows: torch.Tensor, device: torch.device) -> dict[str, object]:
    left = model.transformer.h[0].mlp.Left.weight.detach().float()
    right = model.transformer.h[0].mlp.Right.weight.detach().float()
    down = model.transformer.h[0].mlp.Down.weight.detach().float()
    arm_labels = {subset: arm_name(subset) for subset in ARMS}
    document_ce = {label: [] for label in arm_labels.values()}
    calls = {
        label: {
            "forwards": 0, "attention": 0, "site0": 0, "other_mlp": 0,
            "native_left": 0, "native_right": 0, "native_down": 0,
        }
        for label in arm_labels.values()
    }
    diagnostics = {
        "collinearity_numerator": 0.0,
        "collinearity_denominator": 0,
        "analytic_identity_numerator": 0.0,
        "analytic_identity_denominator": 0.0,
        "deployed_residual_numerator": 0.0,
        "deployed_residual_denominator": 0.0,
        "maximum_state_collinearity_error": 0.0,
    }
    gram = torch.zeros(3, 3, dtype=torch.float64)
    gram_tokens = 0
    current_arm = {"label": None}

    def hook(name: str):
        def count(_module, _inputs, _output):
            label = current_arm["label"]
            if label is None:
                raise RuntimeError("MLP0 native submodule called outside an arm")
            calls[label][name] += 1
        return count

    hooks = [
        model.transformer.h[0].mlp.Left.register_forward_hook(hook("native_left")),
        model.transformer.h[0].mlp.Right.register_forward_hook(hook("native_right")),
        model.transformer.h[0].mlp.Down.register_forward_hook(hook("native_down")),
    ]
    try:
        for start in range(0, len(rows), DOCUMENT_BATCH):
            batch = rows[start:start + DOCUMENT_BATCH]
            tokens = batch[:, :-1].to(device)
            targets = batch[:, 1:].to(device)
            x0 = F.rms_norm(model.transformer.wte(tokens), (D,))
            block0 = model.transformer.h[0]
            token_base = (block0.lambdas[0] + block0.lambdas[1]) * x0
            for subset in ARMS:
                label = arm_labels[subset]
                current_arm["label"] = label

                def attention(event: facade.AttentionEvent, label=label):
                    calls[label]["attention"] += 1
                    return event.block.attn(event.state, event.first_value)

                def mlp(event: facade.EarlyMLPEvent, subset=subset, label=label):
                    nonlocal gram_tokens
                    if event.site != 0:
                        calls[label]["other_mlp"] += 1
                        return event.block.mlp(event.state)
                    calls[label]["site0"] += 1
                    native = event.block.mlp(event.state)
                    token_part, context_part, collinearity = split_normalized_state(
                        event.state, token_base, event.attention_write,
                    )
                    branches = quadratic_tensor_branches(
                        token_part, context_part, left, right, down,
                    )
                    omitted = sum(
                        (branches[name] for name in BRANCHES if name not in subset),
                        start=torch.zeros_like(branches["TT"]),
                    )
                    if len(subset) == len(BRANCHES):
                        analytical = sum(branches.values())
                        direct = full_float_quadratic(event.state, left, right, down)
                        bias = event.block.mlp.Down_bias.detach().float()
                        deployed = native.float() - bias
                        diagnostics["collinearity_numerator"] += float(collinearity.sum())
                        diagnostics["collinearity_denominator"] += collinearity.numel()
                        diagnostics["maximum_state_collinearity_error"] = max(
                            diagnostics["maximum_state_collinearity_error"],
                            float(collinearity.max()),
                        )
                        diagnostics["analytic_identity_numerator"] += float(
                            (analytical.double() - direct.double()).square().sum()
                        )
                        diagnostics["analytic_identity_denominator"] += float(
                            direct.double().square().sum()
                        )
                        diagnostics["deployed_residual_numerator"] += float(
                            (analytical.double() - deployed.double()).square().sum()
                        )
                        diagnostics["deployed_residual_denominator"] += float(
                            deployed.double().square().sum()
                        )
                        flat = torch.stack(
                            [branches[name].double().reshape(-1, D) for name in BRANCHES],
                        )
                        gram.add_(torch.einsum("and,bnd->ab", flat, flat).cpu())
                        gram_tokens += flat.shape[1]
                    return native - omitted.to(native.dtype)

                logits = facade.forward_with_dispatch(model, tokens, attention, mlp)
                losses = F.cross_entropy(
                    logits[:, SCORING].transpose(1, 2), targets[:, SCORING], reduction="none",
                ).mean(1)
                document_ce[label].extend(float(loss) for loss in losses)
                calls[label]["forwards"] += 1
                current_arm["label"] = None
    finally:
        current_arm["label"] = None
        for handle in hooks:
            handle.remove()

    expected = len(rows) // DOCUMENT_BATCH
    wanted = {
        "forwards": expected, "attention": 18 * expected, "site0": expected,
        "other_mlp": 17 * expected, "native_left": expected,
        "native_right": expected, "native_down": expected,
    }
    for label in document_ce:
        if calls[label] != wanted or len(document_ce[label]) != len(rows):
            raise RuntimeError(f"MLP0 factorial census changed for {label}: {calls[label]}")

    pooled_ce = {
        label: float(torch.tensor(values, dtype=torch.float64).mean())
        for label, values in document_ce.items()
    }
    performance = {
        subset: -pooled_ce[arm_labels[subset]] for subset in ARMS
    }
    pooled_shapley = shapley_values(performance)
    document_shapley = {branch: [] for branch in BRANCHES}
    for document in range(len(rows)):
        doc_performance = {
            subset: -document_ce[arm_labels[subset]][document] for subset in ARMS
        }
        values = shapley_values(doc_performance)
        for branch in BRANCHES:
            document_shapley[branch].append(values[branch])
    diagonal = gram.diag().clamp_min(1e-30).sqrt()
    correlations = gram / (diagonal[:, None] * diagonal[None, :])
    diagnostics_out = {
        "mean_state_collinearity_relative_mse": (
            diagnostics["collinearity_numerator"]
            / max(diagnostics["collinearity_denominator"], 1)
        ),
        "maximum_state_collinearity_relative_mse": diagnostics[
            "maximum_state_collinearity_error"
        ],
        "analytical_branch_sum_relative_mse_vs_float32_direct": (
            diagnostics["analytic_identity_numerator"]
            / max(diagnostics["analytic_identity_denominator"], 1e-30)
        ),
        "analytical_branch_sum_relative_mse_vs_deployed_bf16": (
            diagnostics["deployed_residual_numerator"]
            / max(diagnostics["deployed_residual_denominator"], 1e-30)
        ),
    }
    if diagnostics_out[
        "analytical_branch_sum_relative_mse_vs_float32_direct"
    ] > 1e-8:
        raise RuntimeError(
            "MLP0 analytical TT+X+CC identity exceeded the frozen real-state gate"
        )
    return {
        "pooled_ce": pooled_ce,
        "full_minus_empty_ce_benefit": pooled_ce["EMPTY"] - pooled_ce["TT+X+CC"],
        "pooled_shapley_ce_benefit": pooled_shapley,
        "document_shapley_ce_benefit": {
            branch: bootstrap_mean_interval(
                torch.tensor(values), seed=BOOTSTRAP_SEED + index,
            )
            for index, (branch, values) in enumerate(document_shapley.items())
        },
        "mobius_dividends_of_negative_ce": mobius_dividends(performance),
        "branch_gram_per_token": (gram / max(gram_tokens, 1)).tolist(),
        "branch_gram_correlations": correlations.tolist(),
        "diagnostics": diagnostics_out,
        "calls": calls,
    }


def main() -> None:
    started = time.time()
    if OUTPUT.exists():
        raise RuntimeError(f"MLP0 tensor-factorial namespace already exists: {OUTPUT}")
    receipt = json.loads(ROWS_RECEIPT.read_text())
    fit_rows = rows_parent.load_role(receipt["entries"]["FIT"])
    select_rows = rows_parent.load_role(receipt["entries"]["SELECT"])
    device = torch.device("cuda")
    model, checkpoint = facade.load_bilin18(device=device, dtype=torch.bfloat16)
    role_results = {
        "FIT": score_role(model, fit_rows, device),
        "SELECT": score_role(model, select_rows, device),
    }
    fit_shapley = role_results["FIT"]["pooled_shapley_ce_benefit"]
    select_shapley = role_results["SELECT"]["pooled_shapley_ce_benefit"]
    fit_order = sorted(BRANCHES, key=lambda name: fit_shapley[name], reverse=True)
    select_order = sorted(BRANCHES, key=lambda name: select_shapley[name], reverse=True)
    transport = {
        "shapley_sign_matches": {
            branch: (fit_shapley[branch] >= 0) == (select_shapley[branch] >= 0)
            for branch in BRANCHES
        },
        "shapley_rank_order_fit": fit_order,
        "shapley_rank_order_select": select_order,
        "shapley_rank_order_matches": fit_order == select_order,
    }
    output = {
        "schema": "mlp0_token_context_tensor_factorial_discovery_v1",
        "status": "discovery_complete",
        "claim_boundary": (
            "No-fit algebraic and causal branch census on already-opened FIT/SELECT; "
            "native MLP0 oracle used in every arm; no executable compression, FINAL, "
            "OOD, semantic, extraction, removal, or promotion claim."
        ),
        "branches": {
            "TT": "token-token numerator with shared RMS scale",
            "X": "token-context plus context-token cross numerator",
            "CC": "context-context numerator",
        },
        "arms": [arm_name(subset) for subset in ARMS],
        "documents": {"FIT": len(fit_rows), "SELECT": len(select_rows), "FINAL_opened": 0},
        "positions_per_document": SCORING.stop - SCORING.start,
        "roles": role_results,
        "transport": transport,
        "checkpoint": checkpoint.__dict__,
        "runtime_seconds": time.time() - started,
        "parents": {
            "runner_sha256": file_sha256(Path(__file__).resolve()),
            "preregistration_sha256": file_sha256(PREREGISTRATION),
            "rows_receipt_sha256": file_sha256(ROWS_RECEIPT),
        },
    }
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
