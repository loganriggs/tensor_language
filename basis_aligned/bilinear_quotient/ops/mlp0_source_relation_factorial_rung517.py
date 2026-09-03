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
from pathlib import Path

import torch


ROOT = Path("/workspace/tensor_language")
POLY = ROOT / "basis_aligned/polynomial_causal"
OUT = ROOT / "basis_aligned/bilinear_quotient/mlp0_source_relation_factorial_rung517_results.json"
PREREG = POLY / "MLP0_SOURCE_RELATION_FACTORIAL_RUNG517_PREREGISTRATION.md"

GROUPS = ("SELF", "PREVIOUS", "NEAR", "DISTANT_SAME", "DISTANT_OTHER")
N_GROUPS = len(GROUPS)
N_ARMS = 1 << N_GROUPS
PLANTED_SEEDS = tuple(range(51700, 51708))


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


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1" or "--dry-run" in sys.argv:
        result = dry_run()
        if not result["partition_exact"] or not result["planted_recovery"]["all_eight_exact"]:
            raise RuntimeError("rung517 dry-run identification gate failed")
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    raise RuntimeError(
        "Rung517 scientific execution is fail-closed: exact attention-source construction, "
        "full/empty native replay, corpus hashes, and downstream capture are still being implemented."
    )


if __name__ == "__main__":
    main()
