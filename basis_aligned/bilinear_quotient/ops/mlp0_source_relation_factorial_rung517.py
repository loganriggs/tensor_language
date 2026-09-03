"""RUNG517 -- cross-head source-relation factorial for MLP0.

The scientific path is deliberately incomplete until every exact-source and
full/empty replay gate is implemented.  The currently executable dry run tests
the frozen five-way partition and the Boolean-lattice/Mobius/Shapley algebra on
eight planted functions without loading the model or opening outcome rows.

BQGATE: EXPERIMENT
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path("/workspace/tensor_language")
POLY = ROOT / "basis_aligned/polynomial_causal"
OUT = ROOT / "basis_aligned/bilinear_quotient/mlp0_source_relation_factorial_rung517_results.json"
PREREG = POLY / "MLP0_SOURCE_RELATION_FACTORIAL_RUNG517_PREREGISTRATION.md"
PREREG_SHA256 = "a0ff4160af15b57c549c3998e24010d7f60f14d34b2de811fd5f5a5824bde56c"

GROUPS = ("SELF", "PREVIOUS", "NEAR", "DISTANT_SAME", "DISTANT_OTHER")
N_GROUPS = len(GROUPS)
N_ARMS = 1 << N_GROUPS
PLANTED_SEEDS = tuple(range(51700, 51708))
CONTROL_SEEDS = tuple(range(517100, 517108))
DOCUMENT_BATCH = 4
D = 1152
N_HEAD = 9
HEAD_DIM = 128
SCORING = slice(64, 256)
ROWS_RECEIPT = ROOT / "basis_aligned/bilinear_quotient/mlp1_sparse_c512_continue_factorial_v1_rows_receipt.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def registered_structured_rows(census) -> torch.Tensor:
    """Select the registered documents and the model's 256+target token window."""
    rows = census.fineweb_rows(80)
    if rows.ndim != 2 or rows.shape[0] != 80 or rows.shape[1] < 257:
        raise RuntimeError(f"unexpected structured-row shape: {tuple(rows.shape)}")
    selected = rows[16:80, :257].contiguous()
    if selected.shape != (64, 257):
        raise RuntimeError(f"registered structured rows have shape {tuple(selected.shape)}")
    return selected


def source_group_masks(tokens: torch.Tensor) -> torch.Tensor:
    """Return the five exhaustive masks [group,batch,query,source]."""
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape [batch, position]")
    batch, length = tokens.shape
    q = torch.arange(length, device=tokens.device)[:, None]
    s = torch.arange(length, device=tokens.device)[None, :]
    lag = q - s
    causal = lag >= 0
    self_mask = lag == 0
    previous = lag == 1
    near = (lag >= 2) & (lag <= 7)
    same = tokens[:, :, None].eq(tokens[:, None, :])
    distant_same = (lag[None] >= 8) & same
    distant_other = causal[None] & ~(
        self_mask[None] | previous[None] | near[None] | distant_same)
    masks = torch.stack((
        self_mask.expand(batch, -1, -1),
        previous.expand(batch, -1, -1),
        near.expand(batch, -1, -1),
        distant_same,
        distant_other,
    ))
    membership = masks.to(torch.int8).sum(0)
    if not torch.equal(membership, causal.expand(batch, -1, -1).to(torch.int8)):
        raise RuntimeError("source-relation groups are not an exact causal partition")
    return masks


@torch.no_grad()
def attention0_source_writes(block, state: torch.Tensor, tokens: torch.Tensor) -> dict:
    """Split the deployed layer-0 attention write by source relation.

    The native write is authoritative.  Semantic groups are evaluated with the
    deployed Q/K normalization, rotary map, bilinear score product, values, and
    output projection.  Their finite-precision closing remainder is retained as
    a separate numerical term so the split is exact by construction.
    """
    if state.shape != (*tokens.shape, D):
        raise ValueError("state and token shapes do not match the bilin18 contract")
    attention = block.attn
    batch, length, _ = state.shape
    native_write, first_value = attention(state, None)

    q = attention.c_q(state).view(batch, length, N_HEAD, HEAD_DIM)
    k = attention.c_k(state).view(batch, length, N_HEAD, HEAD_DIM)
    q2 = attention.c_q2(state).view(batch, length, N_HEAD, HEAD_DIM)
    k2 = attention.c_k2(state).view(batch, length, N_HEAD, HEAD_DIM)
    raw_value = attention.c_v(state).view(batch, length, N_HEAD, HEAD_DIM)
    value = ((1 - attention.lamb) * raw_value
             + attention.lamb * first_value.view_as(raw_value))
    cos, sin = attention.rotary(q)
    module = sys.modules[type(attention).__module__]
    q = module.apply_rotary_emb(F.rms_norm(q, (HEAD_DIM,)), cos, sin)
    k = module.apply_rotary_emb(F.rms_norm(k, (HEAD_DIM,)), cos, sin)
    q2 = module.apply_rotary_emb(F.rms_norm(q2, (HEAD_DIM,)), cos, sin)
    k2 = module.apply_rotary_emb(F.rms_norm(k2, (HEAD_DIM,)), cos, sin)
    score1 = torch.einsum("bqhd,bkhd->bhqk", q, k) / HEAD_DIM
    score2 = torch.einsum("bqhd,bkhd->bhqk", q2, k2) / HEAD_DIM
    pattern = score1 * score2
    masks = source_group_masks(tokens)
    group_writes = []
    group_preprojection = []
    for group_mask in masks:
        selected = pattern * group_mask[:, None].to(pattern.dtype)
        head_output = torch.einsum("bhqk,bkhd->bhqd", selected, value)
        joined = head_output.transpose(1, 2).contiguous().view(batch, length, D)
        group_preprojection.append(joined)
        group_writes.append(attention.c_proj(joined).float())
    group_writes = torch.stack(group_writes)
    group_preprojection = torch.stack(group_preprojection)
    semantic_sum = group_writes.sum(0)
    numerical_remainder = native_write.float() - semantic_sum
    direct_joined = group_preprojection.sum(0)
    direct_write = attention.c_proj(direct_joined)
    denominator = native_write.double().square().sum().clamp_min(1e-30)
    return {
        "native_write": native_write,
        "first_value": first_value,
        "group_writes": group_writes,
        "numerical_remainder": numerical_remainder,
        "partition_masks": masks,
        "pattern": pattern,
        "value": value,
        "diagnostics": {
            "semantic_plus_remainder_relative_mse": float(
                (semantic_sum.double() + numerical_remainder.double()
                 - native_write.double()).square().sum() / denominator),
            "joint_preprojection_relative_mse": float(
                (direct_write.double() - native_write.double()).square().sum() / denominator),
            "numerical_remainder_relative_energy": float(
                numerical_remainder.double().square().sum() / denominator),
        },
    }


@torch.no_grad()
def project_source_mask(block, pattern: torch.Tensor, value: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
    """Output-project the attention contribution from an arbitrary source mask."""
    batch, _, query, source = pattern.shape
    if mask.shape != (batch, query, source):
        raise ValueError("arbitrary source mask has the wrong shape")
    selected = pattern * mask[:, None].to(pattern.dtype)
    head_output = torch.einsum("bhqk,bkhd->bhqd", selected, value)
    joined = head_output.transpose(1, 2).contiguous().view(batch, query, D)
    return block.attn.c_proj(joined).float()


def subset_context(split: dict, mask: int) -> torch.Tensor:
    """The attention context exposed only to MLP0 for one registered subset."""
    if mask < 0 or mask >= N_ARMS:
        raise ValueError("subset mask is outside the five-group Boolean lattice")
    if mask == N_ARMS - 1:
        return split["native_write"]
    selected = torch.zeros_like(split["numerical_remainder"])
    for group in range(N_GROUPS):
        if mask & (1 << group):
            selected = selected + split["group_writes"][group]
    return (selected + split["numerical_remainder"]).to(split["native_write"].dtype)


def mobius_from_subset_values(values: torch.Tensor) -> torch.Tensor:
    """Boolean-lattice Mobius coefficients; leading axis enumerates bitmasks."""
    if values.shape[0] != N_ARMS:
        raise ValueError(f"expected {N_ARMS} subset arms")
    coefficients = values.clone()
    for bit in range(N_GROUPS):
        for mask in range(N_ARMS):
            if mask & (1 << bit):
                coefficients[mask] = coefficients[mask] - coefficients[mask ^ (1 << bit)]
    return coefficients


def subset_values_from_mobius(coefficients: torch.Tensor) -> torch.Tensor:
    if coefficients.shape[0] != N_ARMS:
        raise ValueError(f"expected {N_ARMS} Mobius coefficients")
    values = coefficients.clone()
    for bit in range(N_GROUPS):
        for mask in range(N_ARMS):
            if mask & (1 << bit):
                values[mask] = values[mask] + values[mask ^ (1 << bit)]
    return values


def shapley_from_mobius(coefficients: torch.Tensor) -> torch.Tensor:
    """Equal division of each nonempty interaction among its members."""
    result = torch.zeros((N_GROUPS,) + coefficients.shape[1:], dtype=coefficients.dtype)
    for mask in range(1, N_ARMS):
        members = [bit for bit in range(N_GROUPS) if mask & (1 << bit)]
        share = coefficients[mask] / len(members)
        for bit in members:
            result[bit] = result[bit] + share
    return result


def planted_suite() -> dict:
    cases = []
    for seed in PLANTED_SEEDS:
        generator = torch.Generator().manual_seed(seed)
        coefficients = torch.zeros(N_ARMS, 7, dtype=torch.float64)
        coefficients[0] = torch.randn(7, generator=generator, dtype=torch.float64)
        support = sorted(torch.randperm(N_ARMS - 1, generator=generator)[:9].add(1).tolist())
        coefficients[support] = torch.randn(
            len(support), 7, generator=generator, dtype=torch.float64)
        values = subset_values_from_mobius(coefficients)
        recovered = mobius_from_subset_values(values)
        reconstructed = subset_values_from_mobius(recovered)
        planted_shapley = shapley_from_mobius(coefficients)
        recovered_shapley = shapley_from_mobius(recovered)
        cases.append({
            "seed": seed,
            "support": support,
            "max_coefficient_error": float((recovered - coefficients).abs().max()),
            "max_subset_reconstruction_error": float((reconstructed - values).abs().max()),
            "max_shapley_error": float((recovered_shapley - planted_shapley).abs().max()),
        })
    holds = all(
        max(case["max_coefficient_error"], case["max_subset_reconstruction_error"],
            case["max_shapley_error"]) <= 1e-10
        for case in cases)
    return {"cases": cases, "all_eight_exact": holds}


def _rank(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(len(values), dtype=torch.float64)
    return ranks


def spearman(left, right) -> float:
    left = _rank(torch.as_tensor(left, dtype=torch.float64))
    right = _rank(torch.as_tensor(right, dtype=torch.float64))
    left -= left.mean()
    right -= right.mean()
    denominator = left.norm() * right.norm()
    return float((left @ right) / denominator) if float(denominator) else 0.0


def proportional_metrics(left, right) -> dict:
    """Unsigned scale and signed direction between shared-coordinate profiles."""
    left = torch.as_tensor(left, dtype=torch.float64)
    right = torch.as_tensor(right, dtype=torch.float64)
    beta = float((left @ right) / right.square().sum().clamp_min(1e-30))
    left_norm = float(left.norm())
    right_norm = float(right.norm())
    cosine = float((left @ right) / (left.norm() * right.norm()).clamp_min(1e-30))
    residual = float((left - beta * right).norm() / left.norm().clamp_min(1e-30))
    return {
        "beta_left_from_right": beta, "cosine": cosine,
        "relative_residual": residual, "left_rms": left_norm / math.sqrt(left.numel()),
        "right_rms": right_norm / math.sqrt(right.numel()),
    }


@torch.no_grad()
def group_reference(model, rows: torch.Tensor, device: torch.device) -> dict:
    block0 = model.transformer.h[0]
    sums = torch.zeros(N_GROUPS, D, dtype=torch.float64)
    epsilon_sum = torch.zeros(D, dtype=torch.float64)
    support = torch.zeros(N_GROUPS, dtype=torch.long)
    positions = 0
    partition_num = partition_den = joint_num = 0.0
    for start in range(0, len(rows), DOCUMENT_BATCH):
        tokens = rows[start:start + DOCUMENT_BATCH, :-1].to(device)
        raw_token = F.rms_norm(model.transformer.wte(tokens), (D,))
        token_base = (block0.lambdas[0] + block0.lambdas[1]) * raw_token
        state = F.rms_norm(token_base, (D,))
        split = attention0_source_writes(block0, state, tokens)
        sums += split["group_writes"].double().sum((1, 2)).cpu()
        epsilon_sum += split["numerical_remainder"].double().sum((0, 1)).cpu()
        support += split["partition_masks"][:, :, SCORING].sum((1, 2, 3)).cpu()
        positions += tokens.shape[0] * tokens.shape[1]
        denominator = float(split["native_write"].double().square().sum())
        partition_num += split["diagnostics"]["semantic_plus_remainder_relative_mse"] * denominator
        joint_num += split["diagnostics"]["joint_preprojection_relative_mse"] * denominator
        partition_den += denominator
    return {
        "group_means": (sums / positions).float().to(device),
        "epsilon_mean": (epsilon_sum / positions).float().to(device),
        "fit_positions": positions,
        "fit_source_support": support.tolist(),
        "semantic_plus_remainder_relative_mse": partition_num / max(partition_den, 1e-30),
        "joint_preprojection_relative_mse": joint_num / max(partition_den, 1e-30),
    }


def _algebra_accumulator() -> dict:
    return {
        "i_group_energy": torch.zeros(N_GROUPS, dtype=torch.float64),
        "c_diagonal_energy": torch.zeros(N_GROUPS, dtype=torch.float64),
        "c_cross_energy": torch.zeros(N_GROUPS, N_GROUPS, dtype=torch.float64),
        "i_parent_energy": 0.0, "c_parent_energy": 0.0,
        "i_closing_energy": 0.0, "c_closing_energy": 0.0,
        "i_reconstruction_error": 0.0, "c_reconstruction_error": 0.0,
    }


@torch.no_grad()
def accumulate_algebra(accumulator: dict, base, token_base: torch.Tensor,
                       split: dict, normalized: torch.Tensor, reference: dict,
                       group_means: torch.Tensor, left: torch.Tensor,
                       right: torch.Tensor, down: torch.Tensor) -> None:
    _retained, branches, _g, _gain, _collinearity = base._components(
        token_base, split["native_write"], normalized, reference, left, right, down)
    token_delta = token_base.float() - reference["token_mean"]
    contexts = [split["group_writes"][group] - group_means[group]
                for group in range(N_GROUPS)]
    total_mean = reference["token_mean"] + reference["context_mean"]
    gain = reference["gain_mean"]
    i_terms = []
    c_diagonal = []
    c_cross = {}
    for group, context in enumerate(contexts):
        i_term = gain * (
            base._T(token_delta, context, left, right, down)
            + base._T(context, token_delta, left, right, down))
        c_term = gain * (
            base._T(context, total_mean, left, right, down)
            + base._T(total_mean, context, left, right, down)
            + base._T(context, context, left, right, down))
        i_terms.append(i_term)
        c_diagonal.append(c_term)
        accumulator["i_group_energy"][group] += i_term[:, SCORING].double().square().sum().cpu()
        accumulator["c_diagonal_energy"][group] += c_term[:, SCORING].double().square().sum().cpu()
    for first in range(N_GROUPS):
        for second in range(first + 1, N_GROUPS):
            term = gain * (
                base._T(contexts[first], contexts[second], left, right, down)
                + base._T(contexts[second], contexts[first], left, right, down))
            c_cross[(first, second)] = term
            energy = term[:, SCORING].double().square().sum().cpu()
            accumulator["c_cross_energy"][first, second] += energy
            accumulator["c_cross_energy"][second, first] += energy
    i_named = torch.stack(i_terms).sum(0)
    c_named = torch.stack(c_diagonal).sum(0) + torch.stack(list(c_cross.values())).sum(0)
    i_closing = branches["I"] - i_named
    c_closing = branches["C"] - c_named
    accumulator["i_parent_energy"] += float(branches["I"][:, SCORING].double().square().sum())
    accumulator["c_parent_energy"] += float(branches["C"][:, SCORING].double().square().sum())
    accumulator["i_closing_energy"] += float(i_closing[:, SCORING].double().square().sum())
    accumulator["c_closing_energy"] += float(c_closing[:, SCORING].double().square().sum())
    accumulator["i_reconstruction_error"] += float(
        (i_named[:, SCORING].double() + i_closing[:, SCORING].double()
         - branches["I"][:, SCORING].double()).square().sum())
    accumulator["c_reconstruction_error"] += float(
        (c_named[:, SCORING].double() + c_closing[:, SCORING].double()
         - branches["C"][:, SCORING].double()).square().sum())


def summarize_algebra(accumulator: dict) -> dict:
    i_den = max(accumulator["i_parent_energy"], 1e-30)
    c_den = max(accumulator["c_parent_energy"], 1e-30)
    return {
        "i_group_energy_fraction_of_parent": (accumulator["i_group_energy"] / i_den).tolist(),
        "c_diagonal_energy_fraction_of_parent": (accumulator["c_diagonal_energy"] / c_den).tolist(),
        "c_cross_energy_fraction_of_parent": (accumulator["c_cross_energy"] / c_den).tolist(),
        "i_closing_energy_fraction": accumulator["i_closing_energy"] / i_den,
        "c_closing_energy_fraction": accumulator["c_closing_energy"] / c_den,
        "i_reconstruction_relative_mse": accumulator["i_reconstruction_error"] / i_den,
        "c_reconstruction_relative_mse": accumulator["c_reconstruction_error"] / c_den,
    }


@torch.no_grad()
def score_role(model, rows: torch.Tensor, device: torch.device, reference: dict,
               group_ref: dict, base) -> dict:
    sys.path.insert(0, str(POLY))
    import bilin18_observed_model_facade as facade

    block0 = model.transformer.h[0]
    left = block0.mlp.Left.weight.detach().float()
    right = block0.mlp.Right.weight.detach().float()
    down = block0.mlp.Down.weight.detach().float()
    losses = {mask: [] for mask in range(N_ARMS)}
    consumer_sq = {
        mask: {name: torch.zeros(SCORING.stop - SCORING.start, dtype=torch.float64)
               for name in ("MLP0", "ATTENTION1", "MLP1")}
        for mask in range(N_ARMS)}
    calls = {mask: {"attention": 0, "mlp": 0, "forwards": 0}
             for mask in range(N_ARMS)}
    edit_energy = torch.zeros(N_ARMS, dtype=torch.float64)
    source_support = torch.zeros(N_GROUPS, dtype=torch.long)
    algebra = _algebra_accumulator()
    diagnostic = {
        "partition_num": 0.0, "partition_den": 0.0,
        "joint_num": 0.0, "full_mlp0_max_abs": 0.0,
        "full_suffix_max_abs": 0.0, "empty_suffix_max_abs_replay": 0.0,
    }

    def run_arm(tokens, site0_attention, first_value, site0_write, mask):
        capture = {}

        def attention_dispatch(event):
            calls[mask]["attention"] += 1
            if event.site == 0:
                return site0_attention, first_value
            result = event.block.attn(event.state, event.first_value)
            if event.site == 1:
                capture["ATTENTION1"] = result[0].detach().float()
            return result

        def mlp_dispatch(event):
            calls[mask]["mlp"] += 1
            if event.site == 0:
                capture["MLP0"] = site0_write.detach().float()
                return site0_write
            result = event.block.mlp(event.state)
            if event.site == 1:
                capture["MLP1"] = result.detach().float()
            return result

        logits = facade.forward_with_dispatch(
            model, tokens, attention_dispatch, mlp_dispatch)
        calls[mask]["forwards"] += 1
        return logits, capture

    for start in range(0, len(rows), DOCUMENT_BATCH):
        batch = rows[start:start + DOCUMENT_BATCH]
        tokens = batch[:, :-1].to(device)
        targets = batch[:, 1:].to(device)
        raw_token = F.rms_norm(model.transformer.wte(tokens), (D,))
        token_base = (block0.lambdas[0] + block0.lambdas[1]) * raw_token
        state = F.rms_norm(token_base, (D,))
        split = attention0_source_writes(block0, state, tokens)
        normalized = F.rms_norm(token_base + split["native_write"], (D,))
        native0 = block0.mlp(normalized)
        source_support += split["partition_masks"][:, :, SCORING].sum((1, 2, 3)).cpu()
        accumulate_algebra(
            algebra, base, token_base, split, normalized, reference,
            group_ref["group_means"], left, right, down)
        denominator = float(split["native_write"].double().square().sum())
        diagnostic["partition_num"] += (
            split["diagnostics"]["semantic_plus_remainder_relative_mse"] * denominator)
        diagnostic["joint_num"] += split["diagnostics"]["joint_preprojection_relative_mse"] * denominator
        diagnostic["partition_den"] += denominator

        site0_writes = []
        for mask in range(N_ARMS):
            context = subset_context(split, mask)
            site0_writes.append(block0.mlp(F.rms_norm(token_base + context, (D,))))
        diagnostic["full_mlp0_max_abs"] = max(
            diagnostic["full_mlp0_max_abs"],
            float((site0_writes[-1].float() - native0.float()).abs().max()))

        full_logits, full_capture = run_arm(
            tokens, split["native_write"], split["first_value"], site0_writes[-1], N_ARMS - 1)
        full_loss = F.cross_entropy(
            full_logits[:, SCORING].transpose(1, 2), targets[:, SCORING], reduction="none")
        losses[N_ARMS - 1].append(full_loss.cpu())

        native_logits = facade.forward_with_dispatch(
            model, tokens,
            lambda event: event.block.attn(event.state, event.first_value),
            lambda event: event.block.mlp(event.state))
        diagnostic["full_suffix_max_abs"] = max(
            diagnostic["full_suffix_max_abs"], float((native_logits - full_logits).abs().max()))

        for mask in range(N_ARMS - 1):
            logits, capture = run_arm(
                tokens, split["native_write"], split["first_value"], site0_writes[mask], mask)
            loss = F.cross_entropy(
                logits[:, SCORING].transpose(1, 2), targets[:, SCORING], reduction="none")
            losses[mask].append(loss.cpu())
            for name in consumer_sq[mask]:
                difference = capture[name][:, SCORING] - full_capture[name][:, SCORING]
                consumer_sq[mask][name] += difference.double().square().sum((0, 2)).cpu()
            difference = site0_writes[mask][:, SCORING].float() - native0[:, SCORING].float()
            edit_energy[mask] += difference.double().square().sum().cpu()
            if mask == 0:
                repeat_logits, _repeat_capture = run_arm(
                    tokens, split["native_write"], split["first_value"], site0_writes[mask], mask)
                # Remove the extra replay from the registered arm call census.
                calls[mask]["attention"] -= 18
                calls[mask]["mlp"] -= 18
                calls[mask]["forwards"] -= 1
                diagnostic["empty_suffix_max_abs_replay"] = max(
                    diagnostic["empty_suffix_max_abs_replay"],
                    float((repeat_logits - logits).abs().max()))

    per_mask_loss = {mask: torch.cat(parts).double() for mask, parts in losses.items()}
    n_documents = len(rows)
    pooled_ce = torch.tensor(
        [float(per_mask_loss[mask].mean()) for mask in range(N_ARMS)], dtype=torch.float64)
    performance = -pooled_ce
    mobius = mobius_from_subset_values(performance)
    shapley = shapley_from_mobius(mobius)
    position_loss = torch.stack([per_mask_loss[mask].mean(0) for mask in range(N_ARMS)])
    endpoint = []
    endpoint_position = []
    singleton = []
    removal = []
    consumer_endpoint = {name: [] for name in ("MLP0", "ATTENTION1", "MLP1")}
    for group in range(N_GROUPS):
        single_mask = 1 << group
        drop_mask = (N_ARMS - 1) ^ single_mask
        single_value = float(pooled_ce[0] - pooled_ce[single_mask])
        removal_value = float(pooled_ce[drop_mask] - pooled_ce[-1])
        singleton.append(single_value)
        removal.append(removal_value)
        endpoint.append((single_value + removal_value) / 2)
        endpoint_position.append((
            position_loss[0] - position_loss[single_mask]
            + position_loss[drop_mask] - position_loss[-1]) / 2)
        for name in consumer_endpoint:
            single_profile = torch.sqrt(
                consumer_sq[single_mask][name] / max(n_documents * D, 1))
            drop_profile = torch.sqrt(
                consumer_sq[drop_mask][name] / max(n_documents * D, 1))
            consumer_endpoint[name].append((single_profile + drop_profile) / 2)
    positive = torch.tensor(endpoint).clamp_min(0)
    positive_denominator = float(positive.sum())
    expected_batches = len(rows) // DOCUMENT_BATCH
    live_calls = all(
        calls[mask] == {"attention": 18 * expected_batches,
                        "mlp": 18 * expected_batches, "forwards": expected_batches}
        for mask in range(N_ARMS))
    mobius_reconstruction = subset_values_from_mobius(mobius)
    return {
        "documents": len(rows),
        "pooled_ce": pooled_ce.tolist(),
        "full_minus_empty_ce_benefit": float(pooled_ce[0] - pooled_ce[-1]),
        "shapley_ce_benefit": shapley.tolist(),
        "mobius_negative_ce": mobius.tolist(),
        "singleton_benefit": singleton,
        "removal_benefit": removal,
        "endpoint_average_benefit": endpoint,
        "positive_endpoint_share": (
            (positive / positive_denominator).tolist() if positive_denominator > 0
            else [0.0] * N_GROUPS),
        "endpoint_position_ce_profiles": torch.stack(endpoint_position).tolist(),
        "consumer_endpoint_rms_profiles": {
            name: torch.stack(profiles).tolist() for name, profiles in consumer_endpoint.items()},
        "source_support": source_support.tolist(),
        "algebra": summarize_algebra(algebra),
        "diagnostics": {
            "semantic_plus_remainder_relative_mse": diagnostic["partition_num"]
            / max(diagnostic["partition_den"], 1e-30),
            "joint_preprojection_relative_mse": diagnostic["joint_num"]
            / max(diagnostic["partition_den"], 1e-30),
            "full_mlp0_max_abs_error": diagnostic["full_mlp0_max_abs"],
            "full_suffix_max_abs_error": diagnostic["full_suffix_max_abs"],
            "empty_suffix_max_abs_replay": diagnostic["empty_suffix_max_abs_replay"],
            "all_nonfull_edits_live": bool((edit_energy[:-1] > 0).all()),
            "minimum_nonfull_edit_energy": float(edit_energy[:-1].min()),
            "live_call_census": live_calls,
            "mobius_reconstruction_max_abs_error": float(
                (mobius_reconstruction - performance).abs().max()),
        },
        "calls": {str(mask): value for mask, value in calls.items()},
    }


def random_same_count_mask(tokens: torch.Tensor, original: torch.Tensor, seed: int,
                           document_offset: int, *, distant_only: bool = False) -> torch.Tensor:
    """Fixed same-count source randomization, independently reproducible per row/query."""
    batch, length = tokens.shape
    if original.shape != (batch, length, length):
        raise ValueError("original source group mask has the wrong shape")
    output = torch.zeros_like(original, device="cpu")
    q_grid = torch.arange(length)[:, None]
    s_grid = torch.arange(length)[None, :]
    lag = q_grid - s_grid
    pool_template = (lag >= 8) if distant_only else (lag >= 0)
    original_cpu = original.cpu()
    for local in range(batch):
        document = document_offset + local
        for query in range(length):
            count = int(original_cpu[local, query].sum())
            if count == 0:
                continue
            pool = pool_template[query].nonzero().flatten()
            if len(pool) < count:
                raise RuntimeError("same-count control pool is too small")
            generator = torch.Generator().manual_seed(
                seed * 1_000_003 + document * 521 + query * 17)
            chosen = pool[torch.randperm(len(pool), generator=generator)[:count]]
            output[local, query, chosen] = True
    if not torch.equal(output.sum(-1), original_cpu.sum(-1)):
        raise RuntimeError("randomized source mask failed same-count preservation")
    return output.to(tokens.device)


@torch.no_grad()
def control_role(model, rows: torch.Tensor, device: torch.device, group: int,
                 seed: int, *, distant_only: bool = False) -> dict:
    """Matched random-source endpoint profiles for one FIT-selected relation."""
    sys.path.insert(0, str(POLY))
    import bilin18_observed_model_facade as facade

    block0 = model.transformer.h[0]
    consumers = ("ATTENTION1", "MLP1")
    endpoint_sq = {name: {"singleton": torch.zeros(192, dtype=torch.float64),
                          "removal": torch.zeros(192, dtype=torch.float64)}
                   for name in consumers}
    calls = 0
    for start in range(0, len(rows), DOCUMENT_BATCH):
        tokens = rows[start:start + DOCUMENT_BATCH, :-1].to(device)
        raw_token = F.rms_norm(model.transformer.wte(tokens), (D,))
        token_base = (block0.lambdas[0] + block0.lambdas[1]) * raw_token
        state = F.rms_norm(token_base, (D,))
        split = attention0_source_writes(block0, state, tokens)
        randomized = random_same_count_mask(
            tokens, split["partition_masks"][group], seed, start,
            distant_only=distant_only)
        random_write = project_source_mask(
            block0, split["pattern"], split["value"], randomized)
        contexts = {
            "FULL": split["native_write"],
            "EMPTY": split["numerical_remainder"].to(split["native_write"].dtype),
            "SINGLE": (split["numerical_remainder"] + random_write).to(
                split["native_write"].dtype),
            "DROP": (split["native_write"].float() - random_write).to(
                split["native_write"].dtype),
        }
        captures = {}
        for label in ("FULL", "EMPTY", "SINGLE", "DROP"):
            site0_write = block0.mlp(F.rms_norm(token_base + contexts[label], (D,)))
            capture = {}

            def attention_dispatch(event):
                if event.site == 0:
                    return split["native_write"], split["first_value"]
                result = event.block.attn(event.state, event.first_value)
                if event.site == 1:
                    capture["ATTENTION1"] = result[0].detach().float()
                return result

            def mlp_dispatch(event):
                if event.site == 0:
                    return site0_write
                result = event.block.mlp(event.state)
                if event.site == 1:
                    capture["MLP1"] = result.detach().float()
                return result

            facade.forward_with_dispatch(model, tokens, attention_dispatch, mlp_dispatch)
            captures[label] = capture
            calls += 1
        for name in consumers:
            singleton = captures["SINGLE"][name][:, SCORING] - captures["EMPTY"][name][:, SCORING]
            removal = captures["DROP"][name][:, SCORING] - captures["FULL"][name][:, SCORING]
            endpoint_sq[name]["singleton"] += singleton.double().square().sum((0, 2)).cpu()
            endpoint_sq[name]["removal"] += removal.double().square().sum((0, 2)).cpu()
    profiles = {}
    for name in consumers:
        singleton = torch.sqrt(endpoint_sq[name]["singleton"] / max(len(rows) * D, 1))
        removal = torch.sqrt(endpoint_sq[name]["removal"] / max(len(rows) * D, 1))
        profiles[name] = ((singleton + removal) / 2).tolist()
    return {"profiles": profiles, "forwards": calls,
            "same_count_preserved": True, "distant_only": distant_only}


def analyze_transport(fit: dict, select: dict) -> dict:
    fit_endpoint = fit["endpoint_average_benefit"]
    select_endpoint = select["endpoint_average_benefit"]
    fit_profiles = torch.tensor(fit["endpoint_position_ce_profiles"], dtype=torch.float64)
    select_profiles = torch.tensor(select["endpoint_position_ce_profiles"], dtype=torch.float64)
    profile_metrics = [proportional_metrics(fit_profiles[group], select_profiles[group])
                       for group in range(N_GROUPS)]
    promoted = []
    fit_eligible = []
    for group in range(N_GROUPS):
        fit_rms = float(fit_profiles[group].square().mean().sqrt())
        select_rms = float(select_profiles[group].square().mean().sqrt())
        support_fit = fit["source_support"][group]
        support_select = select["source_support"][group]
        repeated_ok_fit = group != GROUPS.index("DISTANT_SAME") or support_fit >= 1000
        repeated_ok_both = repeated_ok_fit and (
            group != GROUPS.index("DISTANT_SAME") or support_select >= 1000)
        if fit_endpoint[group] > 0 and fit_rms >= .001 and repeated_ok_fit:
            fit_eligible.append(group)
        if (fit_endpoint[group] > 0 and select_endpoint[group] >= .005
                and fit_rms >= .001 and select_rms >= .001 and repeated_ok_both):
            promoted.append(group)
    selected_for_controls = (
        max(fit_eligible, key=lambda group: fit_endpoint[group]) if fit_eligible else None)
    return {
        "endpoint_spearman": spearman(fit_endpoint, select_endpoint),
        "fit_top_group": GROUPS[int(torch.tensor(fit_endpoint).argmax())],
        "select_top_group": GROUPS[int(torch.tensor(select_endpoint).argmax())],
        "profile_metrics": {GROUPS[group]: profile_metrics[group] for group in range(N_GROUPS)},
        "promoted_groups": [GROUPS[group] for group in promoted],
        "fit_selected_control_group": (
            GROUPS[selected_for_controls] if selected_for_controls is not None else None),
        "selected_control_group_index": selected_for_controls,
    }


def dry_run() -> dict:
    # Repeats and edge positions exercise all five groups and exact coverage.
    tokens = torch.tensor([
        [3, 5, 3, 7, 8, 9, 10, 11, 3, 12],
        [2, 2, 4, 6, 8, 10, 12, 14, 16, 2],
    ])
    if sha256(PREREG) != PREREG_SHA256:
        raise RuntimeError("rung517 preregistration changed after the source was frozen")
    masks = source_group_masks(tokens)
    causal_count = tokens.shape[0] * tokens.shape[1] * (tokens.shape[1] + 1) // 2
    partition_counts = {name: int(masks[index].sum()) for index, name in enumerate(GROUPS)}
    planted = planted_suite()
    return {
        "status": "dry_run_passed",
        "rung": 517,
        "outcomes_opened": False,
        "model_loaded": False,
        "groups": list(GROUPS),
        "arms": N_ARMS,
        "causal_edges": causal_count,
        "partition_counts": partition_counts,
        "partition_total": sum(partition_counts.values()),
        "partition_exact": sum(partition_counts.values()) == causal_count,
        "planted_recovery": planted,
        "preregistration_sha256": sha256(PREREG),
    }


@torch.no_grad()
def gpu_smoke() -> None:
    """One-batch managed CUDA path check; retain no task or semantic outcomes."""
    if sha256(PREREG) != PREREG_SHA256:
        raise RuntimeError("rung517 preregistration changed after the source was frozen")
    started = time.time()
    sys.path.insert(0, str(POLY))
    sys.path.insert(0, str(ROOT / "basis_aligned/bilinear_quotient/ops"))
    import bilin18_observed_model_facade as facade
    import run_mlp1_sparse_c512_continue_factorial_v1_fit as rows_parent

    receipt = json.loads(ROWS_RECEIPT.read_text())
    rows = rows_parent.load_role(receipt["entries"]["FIT"])[:DOCUMENT_BATCH]
    tokens = rows[:, :-1].cuda()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.bfloat16)
    block0 = model.transformer.h[0]
    raw_token = F.rms_norm(model.transformer.wte(tokens), (D,))
    token_base = (block0.lambdas[0] + block0.lambdas[1]) * raw_token
    state = F.rms_norm(token_base, (D,))
    split = attention0_source_writes(block0, state, tokens)
    native0 = block0.mlp(F.rms_norm(token_base + split["native_write"], (D,)))
    writes = []
    for mask in range(N_ARMS):
        context = subset_context(split, mask)
        writes.append(block0.mlp(F.rms_norm(token_base + context, (D,))))
    full_error = float((writes[-1].float() - native0.float()).abs().max())
    live = [float((write.float() - native0.float()).square().mean().sqrt()) for write in writes[:-1]]

    captured = {}
    calls = {"attention": 0, "mlp": 0}
    for label, site0_write in (("FULL", writes[-1]), ("EMPTY", writes[0])):
        def attention_dispatch(event, label=label):
            calls["attention"] += 1
            if event.site == 0:
                return split["native_write"], split["first_value"]
            result = event.block.attn(event.state, event.first_value)
            if event.site == 1:
                captured[f"{label}_attention1"] = result[0].detach().float()
            return result

        def mlp_dispatch(event, label=label, site0_write=site0_write):
            calls["mlp"] += 1
            if event.site == 0:
                return site0_write
            result = event.block.mlp(event.state)
            if event.site == 1:
                captured[f"{label}_mlp1"] = result.detach().float()
            return result

        captured[f"{label}_logits"] = facade.forward_with_dispatch(
            model, tokens, attention_dispatch, mlp_dispatch)

    native_logits = captured["FULL_logits"]
    full_replay = float((native_logits - facade.forward_with_dispatch(
        model, tokens,
        lambda event: event.block.attn(event.state, event.first_value),
        lambda event: event.block.mlp(event.state))).abs().max())
    empty_logit_rms = float((captured["EMPTY_logits"] - native_logits).square().mean().sqrt())
    attention1_rms = float((captured["EMPTY_attention1"] - captured["FULL_attention1"])
                           .square().mean().sqrt())
    mlp1_rms = float((captured["EMPTY_mlp1"] - captured["FULL_mlp1"])
                     .square().mean().sqrt())
    checks = {
        "source_partition_exact": split["diagnostics"]["semantic_plus_remainder_relative_mse"] <= 1e-8,
        "full_mlp0_replay_exact": full_error == 0.0,
        "all_nonfull_subset_edits_live": min(live) > 0,
        "full_suffix_replay_exact": full_replay == 0.0,
        "empty_suffix_and_consumers_live": min(empty_logit_rms, attention1_rms, mlp1_rms) > 0,
        "call_census_exact": calls == {"attention": 36, "mlp": 36},
        "planted_recovery_exact": planted_suite()["all_eight_exact"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"rung517 managed smoke failed: {checks}")
    print(json.dumps({
        "status": "gpu_smoke_passed", "rung": 517,
        "scientific_outcomes_retained": False,
        "checkpoint": checkpoint.__dict__, "checks": checks,
        "diagnostics": split["diagnostics"],
        "full_mlp0_max_abs_error": full_error,
        "full_suffix_max_abs_error": full_replay,
        "nonfull_minimum_edit_rms": min(live),
        "calls": calls, "runtime_s": time.time() - started,
    }, indent=2, sort_keys=True))


@torch.no_grad()
def scientific_main() -> None:
    if sha256(PREREG) != PREREG_SHA256:
        raise RuntimeError("rung517 preregistration changed after the source was frozen")
    started = time.time()
    sys.path.insert(0, str(POLY))
    sys.path.insert(0, str(ROOT / "basis_aligned/bilinear_quotient"))
    sys.path.insert(0, str(ROOT / "basis_aligned/bilinear_quotient/ops"))
    import bilin18_observed_model_facade as facade
    import census_lib as census
    import mlp0_centered_context_anova_factorial as base
    import run_mlp1_sparse_c512_continue_factorial_v1_fit as rows_parent

    receipt = json.loads(ROWS_RECEIPT.read_text())
    prose_fit = rows_parent.load_role(receipt["entries"]["FIT"])
    prose_select = rows_parent.load_role(receipt["entries"]["SELECT"])
    structured = registered_structured_rows(census)
    corpora = {
        "PROSE": {"FIT": prose_fit, "SELECT": prose_select},
        "STRUCTURED": {"FIT": structured[:32], "SELECT": structured[32:]},
    }
    row_hashes = {
        corpus: {role: rows_parent.rows_life.base.tensor_sha256(rows)
                 for role, rows in splits.items()}
        for corpus, splits in corpora.items()}
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.bfloat16)
    device = torch.device("cuda")
    results = {}
    group_references = {}
    for corpus, splits in corpora.items():
        reference = base._reference_moments(model, splits["FIT"], device)
        group_ref = group_reference(model, splits["FIT"], device)
        reconstructed_context_mean = (
            group_ref["group_means"].sum(0) + group_ref["epsilon_mean"])
        group_ref["reference_mean_relative_mse"] = float(
            (reconstructed_context_mean - reference["context_mean"]).double()
            .square().sum()
            / reference["context_mean"].double().square().sum().clamp_min(1e-30))
        group_references[corpus] = group_ref
        roles = {
            role: score_role(model, rows, device, reference, group_ref, base)
            for role, rows in splits.items()
        }
        transport = analyze_transport(roles["FIT"], roles["SELECT"])
        results[corpus] = {"roles": roles, "transport": transport}

    control_results = {}
    control_forwards = 0
    for corpus, splits in corpora.items():
        transport = results[corpus]["transport"]
        group = transport["selected_control_group_index"]
        if group is None or GROUPS[group] not in transport["promoted_groups"]:
            control_results[corpus] = {
                "opened": False, "reason": "fit_selected_group_not_promoted_on_select"}
            continue
        rows_by_seed = []
        seeds = list(CONTROL_SEEDS)
        if group == GROUPS.index("DISTANT_SAME"):
            seeds.append(517199)
        for seed in seeds:
            distant_only = seed == 517199
            fit_control = control_role(
                model, splits["FIT"], device, group, seed, distant_only=distant_only)
            select_control = control_role(
                model, splits["SELECT"], device, group, seed, distant_only=distant_only)
            control_forwards += fit_control["forwards"] + select_control["forwards"]
            metrics = {
                consumer: proportional_metrics(
                    fit_control["profiles"][consumer], select_control["profiles"][consumer])
                for consumer in ("ATTENTION1", "MLP1")}
            rows_by_seed.append({
                "seed": seed, "distant_only": distant_only,
                "metrics": metrics, "same_count_preserved": True,
            })
        real_profiles_fit = results[corpus]["roles"]["FIT"]["consumer_endpoint_rms_profiles"]
        real_profiles_select = results[corpus]["roles"]["SELECT"]["consumer_endpoint_rms_profiles"]
        real_metrics = {
            consumer: proportional_metrics(
                real_profiles_fit[consumer][group], real_profiles_select[consumer][group])
            for consumer in ("ATTENTION1", "MLP1")}
        max_control_cosine = {
            consumer: max(row["metrics"][consumer]["cosine"] for row in rows_by_seed)
            for consumer in ("ATTENTION1", "MLP1")}
        select_rms = {
            consumer: float(torch.tensor(real_profiles_select[consumer][group]).square().mean().sqrt())
            for consumer in ("ATTENTION1", "MLP1")}
        consumer_ratio = max(select_rms.values()) / max(min(select_rms.values()), 1e-30)
        winning_consumers = [
            consumer for consumer in ("ATTENTION1", "MLP1")
            if real_metrics[consumer]["cosine"] >= max_control_cosine[consumer] + .15]
        control_results[corpus] = {
            "opened": True, "group": GROUPS[group], "group_index": group,
            "real_metrics": real_metrics, "controls": rows_by_seed,
            "maximum_control_cosine": max_control_cosine,
            "select_total_rms": select_rms,
            "consumer_rms_ratio": consumer_ratio,
            "winning_consumers": winning_consumers,
            "passes": bool(winning_consumers and consumer_ratio >= 1.5),
        }

    planted = planted_suite()
    pred_a = bool(planted["all_eight_exact"])
    for corpus in results:
        pred_a = pred_a and (
            group_references[corpus]["semantic_plus_remainder_relative_mse"] <= 1e-8
            and group_references[corpus]["reference_mean_relative_mse"] <= 1e-8)
        for role in ("FIT", "SELECT"):
            diagnostics = results[corpus]["roles"][role]["diagnostics"]
            algebra = results[corpus]["roles"][role]["algebra"]
            pred_a = pred_a and all((
                diagnostics["semantic_plus_remainder_relative_mse"] <= 1e-8,
                diagnostics["full_mlp0_max_abs_error"] == 0,
                diagnostics["full_suffix_max_abs_error"] == 0,
                diagnostics["empty_suffix_max_abs_replay"] == 0,
                diagnostics["all_nonfull_edits_live"],
                diagnostics["live_call_census"],
                diagnostics["mobius_reconstruction_max_abs_error"] <= 1e-10,
                algebra["i_reconstruction_relative_mse"] <= 1e-10,
                algebra["c_reconstruction_relative_mse"] <= 1e-10,
            ))

    prose = results["PROSE"]["roles"]["SELECT"]
    structured_select = results["STRUCTURED"]["roles"]["SELECT"]
    self_index = GROUPS.index("SELF")
    previous_index = GROUPS.index("PREVIOUS")
    near_index = GROUPS.index("NEAR")
    pred_b = bool(
        int(torch.tensor(prose["endpoint_average_benefit"]).argmax()) == previous_index
        and sum(prose["positive_endpoint_share"][index]
                for index in (self_index, previous_index)) >= .70
        and prose["singleton_benefit"][self_index] > 0
        and prose["singleton_benefit"][previous_index] > 0
        and prose["removal_benefit"][self_index] > 0
        and prose["removal_benefit"][previous_index] > 0)
    prose_local_share = sum(prose["positive_endpoint_share"][index]
                            for index in (self_index, previous_index))
    structured_local_share = sum(
        structured_select["positive_endpoint_share"][index]
        for index in (self_index, previous_index))
    structured_near_share = structured_local_share + structured_select[
        "positive_endpoint_share"][near_index]
    pred_c = bool(
        prose_local_share - structured_local_share >= .10
        and structured_near_share >= .70)
    pred_d = True
    for corpus in results:
        transport = results[corpus]["transport"]
        pred_d = pred_d and (
            transport["endpoint_spearman"] >= .70
            and transport["fit_top_group"] == transport["select_top_group"])
        for group_name in transport["promoted_groups"]:
            metrics = transport["profile_metrics"][group_name]
            pred_d = pred_d and metrics["cosine"] >= .70 and metrics["relative_residual"] <= .65
    pred_e = any(row.get("passes", False) for row in control_results.values())
    strong_null = not (pred_a and (pred_b or pred_c) and pred_d)
    if not pred_a:
        next_step = "repair_exact_source_or_replay_instrument_only"
    elif pred_b and pred_c and pred_d:
        next_step = "expand_dominant_relation_by_token_semantics_and_physical_interchange"
    elif pred_c and pred_d:
        next_step = "retain_register_dependent_local_relation_grammar"
    elif not pred_d:
        next_step = "leave_source_relation_basis_for_different_mlp0_object"
    else:
        next_step = "retain_diagnostic_only_and_choose_new_program_gap"

    base_batches = sum(len(rows) // DOCUMENT_BATCH for splits in corpora.values()
                       for rows in splits.values())
    result = {
        "status": "complete", "rung": 517,
        "claim_level": "source_relation_causal_factorial_diagnostic_not_circuit_or_compression",
        "groups": list(GROUPS), "arms": N_ARMS,
        "source_hashes": {"preregistration": sha256(PREREG),
                          "source": sha256(Path(sys.argv[0]).resolve())},
        "checkpoint": checkpoint.__dict__, "row_hashes": row_hashes,
        "data": {"PROSE": {"FIT": 96, "SELECT": 96, "FINAL_opened": 0},
                 "STRUCTURED": {"FIT": 32, "SELECT": 32,
                                "excluded_old_width_rows": 16}},
        "group_references": {
            corpus: {key: value for key, value in reference.items()
                     if not torch.is_tensor(value)}
            for corpus, reference in group_references.items()},
        "corpora": results, "permutation_controls": control_results,
        "planted_recovery": planted,
        'pred_a_exact_live_instrument': bool(pred_a),
        'pred_b_prose_localization': bool(pred_b),
        'pred_c_structured_text_widening': bool(pred_c),
        'pred_d_split_stable_source_roles': bool(pred_d),
        'pred_e_downstream_specificity_screen': bool(pred_e),
        "strong_null": bool(strong_null), "next_step": next_step,
        "execution_price": {
            "base_full_model_forwards": 34 * base_batches,
            "conditional_control_forwards": control_forwards,
            "full_model_forwards": 34 * base_batches + control_forwards,
            "backwards": 0, "deployed_parameters_added": 0,
            "deployed_parameters_saved": 0,
        },
        "claim_limits": [
            "diagnostic_source_relations_not_a_circuit",
            "shapley_is_an_all_orders_allocation_convention",
            "no_rank_sae_reconstruction_or_compression_claim",
            "no_final_prose_rows_opened",
            "physical_cross_context_interchange_not_yet_tested",
        ],
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": "complete", "rung": 517,
        "pred_a": pred_a, "pred_b": pred_b, "pred_c": pred_c,
        "pred_d": pred_d, "pred_e": pred_e, "strong_null": strong_null,
        "prose_endpoint": prose["endpoint_average_benefit"],
        "structured_endpoint": structured_select["endpoint_average_benefit"],
        "promoted": {corpus: results[corpus]["transport"]["promoted_groups"]
                     for corpus in results},
        "next_step": next_step, "runtime_s": result["runtime_s"],
    }, indent=2, sort_keys=True))


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1" or "--dry-run" in sys.argv:
        result = dry_run()
        if not result["partition_exact"] or not result["planted_recovery"]["all_eight_exact"]:
            raise RuntimeError("rung517 dry-run identification gate failed")
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if os.environ.get("BQLIB_GPU_SMOKE") == "1":
        gpu_smoke()
        return
    scientific_main()


if __name__ == "__main__":
    main()
