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

GROUPS = ("SELF", "PREVIOUS", "NEAR", "DISTANT_SAME", "DISTANT_OTHER")
N_GROUPS = len(GROUPS)
N_ARMS = 1 << N_GROUPS
PLANTED_SEEDS = tuple(range(51700, 51708))
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
    value = attention.c_v(state).view(batch, length, N_HEAD, HEAD_DIM)
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


def dry_run() -> dict:
    # Repeats and edge positions exercise all five groups and exact coverage.
    tokens = torch.tensor([
        [3, 5, 3, 7, 8, 9, 10, 11, 3, 12],
        [2, 2, 4, 6, 8, 10, 12, 14, 16, 2],
    ])
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
        'pred_a_exact_live_instrument': None,
        'pred_b_prose_localization': None,
        'pred_c_structured_text_widening': None,
        'pred_d_split_stable_source_roles': None,
        'pred_e_downstream_specificity_screen': None,
    }


@torch.no_grad()
def gpu_smoke() -> None:
    """One-batch managed CUDA path check; retain no task or semantic outcomes."""
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
    raise RuntimeError(
        "Rung517 scientific execution is fail-closed: exact attention-source construction, "
        "full/empty native replay, corpus hashes, and downstream capture are still being implemented."
    )


if __name__ == "__main__":
    main()
