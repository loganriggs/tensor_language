"""CPU mathematics for the frozen MLP2 centered-response selectors."""

from __future__ import annotations

import hashlib
from typing import Sequence

import torch
import torch.nn.functional as F


def _canonical_data(
    mean: torch.Tensor, variance: torch.Tensor, down: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        not torch.is_tensor(mean) or not torch.is_tensor(variance)
        or not torch.is_tensor(down) or mean.ndim != 1 or variance.shape != mean.shape
        or down.ndim != 2 or down.shape[1] != mean.numel()
        or not mean.is_floating_point() or not variance.is_floating_point()
        or not down.is_floating_point() or not bool(torch.isfinite(mean).all())
        or not bool(torch.isfinite(variance).all()) or not bool(torch.isfinite(down).all())
        or bool((variance <= 0).any())
    ):
        raise ValueError("canonical channel inputs are malformed")
    mean64, variance64, down64 = mean.double(), variance.double(), down.double()
    standard_deviation = variance64.sqrt()
    scaled_down = down64 * standard_deviation[None, :]
    pivots = scaled_down.abs().argmax(0)
    pivot_values = scaled_down.gather(0, pivots[None, :]).squeeze(0)
    if bool((pivot_values == 0).any()):
        raise ValueError("every canonical Down column must be nonzero")
    orientation = torch.where(
        pivot_values < 0, -torch.ones_like(pivot_values), torch.ones_like(pivot_values),
    )
    canonical_mean = mean64 / standard_deviation * orientation
    canonical_down = scaled_down * orientation[None, :]
    return standard_deviation, orientation, canonical_mean, canonical_down


def canonical_hash_random_support(
    mean: torch.Tensor, variance: torch.Tensor, down: torch.Tensor,
    count: int, seed: int,
) -> tuple[int, ...]:
    gates = mean.numel() if torch.is_tensor(mean) and mean.ndim == 1 else -1
    if type(count) is not int or type(seed) is not int or not 0 < count <= gates or seed < 0:
        raise ValueError("hash-random constants are malformed")
    _, _, canonical_mean, canonical_down = _canonical_data(mean, variance, down)
    prefix = f"{seed}:".encode()
    records: list[tuple[bytes, bytes, bytes, int]] = []
    for index in range(gates):
        mean_bytes = canonical_mean[index:index + 1].contiguous().numpy().tobytes()
        down_bytes = canonical_down[:, index].contiguous().numpy().tobytes()
        digest = hashlib.sha256(prefix + mean_bytes + down_bytes).digest()
        records.append((digest, mean_bytes, down_bytes, index))
    return tuple(item[3] for item in sorted(records)[:count])


def canonical_derangement(
    mean: torch.Tensor, variance: torch.Tensor, down: torch.Tensor, seed: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    """Canonical scale/sign data and a permutation-equivariant cyclic derangement."""
    if type(seed) is not int or seed < 0:
        raise ValueError("canonical derangement inputs are malformed")
    standard_deviation, orientation, canonical_mean, canonical_down = _canonical_data(
        mean, variance, down,
    )
    prefix = f"{seed}:".encode()
    records: list[tuple[bytes, bytes, bytes, int]] = []
    for index in range(mean.numel()):
        mean_bytes = canonical_mean[index:index + 1].contiguous().numpy().tobytes()
        down_bytes = canonical_down[:, index].contiguous().numpy().tobytes()
        digest = hashlib.sha256(prefix + mean_bytes + down_bytes).digest()
        records.append((digest, mean_bytes, down_bytes, index))
    order = tuple(item[3] for item in sorted(records))
    permutation = [-1] * len(order)
    for location, source in enumerate(order):
        permutation[source] = order[(location + 1) % len(order)]
    if sorted(permutation) != list(range(len(order))) or any(
        source == target for source, target in enumerate(permutation)
    ):
        raise RuntimeError("canonical pairing is not a fixed-point-free permutation")
    return standard_deviation.contiguous(), orientation.contiguous(), tuple(permutation)


def centered_dual_write(
    product: torch.Tensor, mean: torch.Tensor, down: torch.Tensor, bias: torch.Tensor,
    alpha: torch.Tensor, beta: torch.Tensor, standard_deviation: torch.Tensor,
    orientation: torch.Tensor, permutation: Sequence[int],
) -> torch.Tensor:
    """Exact native baseline plus centered real and zero-baseline deranged leaves."""
    hidden = product.shape[-1] if product.ndim == 3 else -1
    perm = tuple(permutation)
    if (
        hidden <= 1 or mean.shape != (hidden,) or standard_deviation.shape != (hidden,)
        or orientation.shape != (hidden,) or down.ndim != 2 or down.shape[1] != hidden
        or bias.shape != (down.shape[0],) or alpha.shape != (product.shape[0], hidden)
        or beta.shape != alpha.shape or len(perm) != hidden
        or sorted(perm) != list(range(hidden)) or any(i == j for i, j in enumerate(perm))
        or bool((standard_deviation <= 0).any())
    ):
        raise ValueError("centered dual-write inputs are malformed")
    dtype, device = product.dtype, product.device
    mean_d = mean.to(device=device, dtype=dtype)
    std_d = standard_deviation.to(device=device, dtype=dtype)
    orient_d = orientation.to(device=device, dtype=dtype)
    down_d = down.to(device=device, dtype=dtype)
    bias_d = bias.to(device=device, dtype=dtype)
    centered = product - mean_d
    native = F.linear(product, down_d, bias_d)
    real_delta = F.linear(centered * (alpha.to(dtype) - 1)[:, None, :], down_d)
    canonical_product = centered / std_d * orient_d
    canonical_down = down_d * std_d[None, :] * orient_d[None, :]
    index = torch.tensor(perm, device=device, dtype=torch.long)
    control = F.linear(canonical_product * beta.to(dtype)[:, None, :], canonical_down[:, index])
    return native + real_delta + control


def mapped_permutation(
    permutation: Sequence[int], reorder: Sequence[int],
) -> tuple[int, ...]:
    """Expected derangement after arrays are reordered as ``new[i]=old[reorder[i]]``."""
    old_perm, order = tuple(permutation), tuple(reorder)
    if sorted(old_perm) != list(range(len(old_perm))) or sorted(order) != list(
        range(len(old_perm))
    ):
        raise ValueError("permutation replay inputs are malformed")
    inverse = {old: new for new, old in enumerate(order)}
    return tuple(inverse[old_perm[order[new]]] for new in range(len(order)))


def support_jaccard(first: Sequence[int], second: Sequence[int]) -> float:
    left, right = set(first), set(second)
    if not left or not right:
        raise ValueError("support Jaccard requires nonempty supports")
    return len(left & right) / len(left | right)


def spearman(first: torch.Tensor, second: torch.Tensor) -> float | None:
    if first.ndim != 1 or second.shape != first.shape or not bool(
        torch.isfinite(first).all() and torch.isfinite(second).all()
    ):
        raise ValueError("Spearman inputs are malformed")

    def ranks(value: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(value.double(), stable=True)
        result = torch.empty_like(value, dtype=torch.float64)
        result[order] = torch.arange(value.numel(), dtype=torch.float64)
        return result

    left, right = ranks(first), ranks(second)
    left, right = left - left.mean(), right - right.mean()
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) == 0:
        return None
    return float((torch.dot(left, right) / denominator).clamp(-1, 1))
