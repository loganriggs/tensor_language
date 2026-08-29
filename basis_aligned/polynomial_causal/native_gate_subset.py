"""Shared native product-gate subsets for polarized bilinear block programs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from grouped_block_coefficient_screen import balance_product_gauge


TERM_NAMES = ("uu", "uv", "vu", "vv")
MASKS: Mapping[str, tuple[str, ...]] = {
    "all": TERM_NAMES,
    "no_vv": ("uu", "uv", "vu"),
    "no_cross": ("uu", "vv"),
    "cross_only": ("uv", "vu"),
    "uu_only": ("uu",),
}


def typed_gate_features(
    left: torch.Tensor, right: torch.Tensor, u: torch.Tensor, v: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if left.ndim != 2 or right.shape != left.shape or u.shape != v.shape or (
        u.shape[-1] != left.shape[1]
    ) or any(value.dtype != left.dtype or value.device != left.device for value in (
        right, u, v,
    )):
        raise ValueError("native gate factors and typed inputs are incompatible")
    balanced_left, balanced_right, _ = balance_product_gauge(left, right)
    lu = torch.nn.functional.linear(u, balanced_left)
    ru = torch.nn.functional.linear(u, balanced_right)
    lv = torch.nn.functional.linear(v, balanced_left)
    rv = torch.nn.functional.linear(v, balanced_right)
    return {"uu": lu * ru, "uv": lu * rv, "vu": lv * ru, "vv": lv * rv}


def contribution_energy(
    features: Mapping[str, torch.Tensor], down: torch.Tensor,
) -> torch.Tensor:
    if set(features) != set(TERM_NAMES) or down.ndim != 2:
        raise ValueError("typed feature bank or Down map is malformed")
    first = features[TERM_NAMES[0]]
    if any(value.shape != first.shape for value in features.values()) or (
        first.shape[-1] != down.shape[1]
    ) or down.dtype != first.dtype or down.device != first.device:
        raise ValueError("typed features do not share the Down gate interface")
    gate_energy = sum(value.square().sum(dim=tuple(range(value.ndim - 1))) for value in features.values())
    return gate_energy * down.square().sum(dim=0)


def stack_features_and_writes(
    features: Mapping[str, torch.Tensor], down: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if set(features) != set(TERM_NAMES):
        raise ValueError("typed feature bank is incomplete")
    ordered = [features[name].reshape(-1, features[name].shape[-1]) for name in TERM_NAMES]
    matrix = torch.cat(ordered, dim=0)
    if down.ndim != 2 or down.shape[1] != matrix.shape[1] or down.dtype != matrix.dtype or (
        down.device != matrix.device
    ):
        raise ValueError("Down map is incompatible with typed features")
    return matrix, torch.nn.functional.linear(matrix, down)


def sufficient_statistics(
    features: torch.Tensor, writes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if features.ndim != 2 or writes.ndim != 2 or features.shape[0] != writes.shape[0] or (
        features.dtype != writes.dtype or features.device != writes.device
    ) or not bool(torch.isfinite(features).all() and torch.isfinite(writes).all()):
        raise ValueError("joint decoder statistics require finite aligned matrices")
    return features.T @ features, features.T @ writes


def _ridge(gram: torch.Tensor, relative_ridge: float) -> torch.Tensor:
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or not (
        isinstance(relative_ridge, float) and 0 <= relative_ridge < 1
    ):
        raise ValueError("Gram or ridge is malformed")
    scale = gram.diagonal().sum() / max(gram.shape[0], 1)
    return relative_ridge * scale


def fit_joint_decoder(
    gram: torch.Tensor, cross: torch.Tensor, *, relative_ridge: float = 1e-6,
) -> torch.Tensor:
    if gram.ndim != 2 or cross.ndim != 2 or gram.shape[0] != gram.shape[1] or (
        cross.shape[0] != gram.shape[0]
    ) or gram.dtype != cross.dtype or gram.device != cross.device or not bool(
        torch.isfinite(gram).all() and torch.isfinite(cross).all()
    ):
        raise ValueError("decoder sufficient statistics are malformed")
    ridge = _ridge(gram, relative_ridge)
    identity = torch.eye(len(gram), dtype=gram.dtype, device=gram.device)
    coefficient = torch.linalg.solve(gram + ridge * identity, cross)
    return coefficient.T.contiguous()


def batch_simultaneous_omp(
    gram: torch.Tensor,
    cross: torch.Tensor,
    energy: torch.Tensor,
    *,
    budget: int,
    prefilter: int,
    batch_size: int = 16,
    relative_ridge: float = 1e-6,
) -> torch.Tensor:
    """Deterministic batched multi-output OMP from sufficient statistics."""

    gates = gram.shape[0] if gram.ndim == 2 else -1
    if gram.shape != (gates, gates) or cross.ndim != 2 or cross.shape[0] != gates or (
        energy.shape != (gates,)
    ) or any(value.dtype != gram.dtype or value.device != gram.device for value in (
        cross, energy,
    )) or not (1 <= budget <= prefilter <= gates) or not (
        type(batch_size) is int and batch_size > 0
    ):
        raise ValueError("OMP statistics or budgets are incompatible")
    # Stable sort makes the original gate index the frozen tie breaker.
    candidates = torch.argsort(energy, descending=True, stable=True)[:prefilter]
    local_gram = gram[candidates][:, candidates]
    local_cross = cross[candidates]
    selected: list[int] = []
    available = torch.ones(prefilter, dtype=torch.bool, device=gram.device)
    diagonal = local_gram.diagonal().clamp_min(torch.finfo(gram.dtype).tiny)
    while len(selected) < budget:
        if selected:
            index = torch.tensor(selected, dtype=torch.long, device=gram.device)
            decoder_t = fit_joint_decoder(
                local_gram[index][:, index], local_cross[index],
                relative_ridge=relative_ridge,
            ).T
            residual_cross = local_cross - local_gram[:, index] @ decoder_t
        else:
            residual_cross = local_cross
        score = residual_cross.square().sum(dim=1) / diagonal
        score[~available] = -torch.inf
        take = min(batch_size, budget - len(selected))
        chosen = torch.argsort(score, descending=True, stable=True)[:take]
        if not bool(torch.isfinite(score[chosen]).all()):
            raise RuntimeError("OMP exhausted finite candidates before its budget")
        selected.extend(int(value) for value in chosen)
        available[chosen] = False
    local = torch.tensor(selected, dtype=torch.long, device=gram.device)
    return candidates[local]


@dataclass(frozen=True, slots=True)
class NativeGateSubsetProgram:
    indices: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor
    decoder: torch.Tensor
    bias: torch.Tensor

    def __post_init__(self) -> None:
        if self.indices.ndim != 1 or self.indices.dtype != torch.long or len(
            self.indices
        ) == 0 or len(torch.unique(self.indices)) != len(self.indices) or self.left.ndim != 2 or (
            self.right.shape != self.left.shape
        ) or self.decoder.shape != (self.left.shape[1], self.left.shape[0]) or (
            self.bias.shape != (self.left.shape[1],)
        ) or any(value.dtype != self.left.dtype or value.device != self.left.device for value in (
            self.right, self.decoder, self.bias,
        )) or self.indices.device != self.left.device:
            raise ValueError("native gate-subset program is malformed")

    @property
    def gates(self) -> int:
        return len(self.indices)

    @property
    def width(self) -> int:
        return self.left.shape[1]

    @property
    def float_parameter_count(self) -> int:
        return 3 * self.width * self.gates + self.width

    @property
    def product_count_per_token(self) -> int:
        return self.gates

    def terms(self, u: torch.Tensor, v: torch.Tensor) -> dict[str, torch.Tensor]:
        if u.shape != v.shape or u.shape[-1] != self.width or any(
            value.dtype != self.left.dtype or value.device != self.left.device
            for value in (u, v)
        ):
            raise ValueError("program typed inputs are incompatible")
        # Program factors are already sealed in the positive balanced gauge.
        # Rebalancing here would change their literal float32 execution bytes.
        lu = torch.nn.functional.linear(u, self.left)
        ru = torch.nn.functional.linear(u, self.right)
        lv = torch.nn.functional.linear(v, self.left)
        rv = torch.nn.functional.linear(v, self.right)
        features = {"uu": lu * ru, "uv": lu * rv, "vu": lv * ru, "vv": lv * rv}
        return {
            name: torch.nn.functional.linear(value, self.decoder)
            for name, value in features.items()
        }

    def write(self, value: torch.Tensor) -> torch.Tensor:
        """Deployable K-product bilinear write, including the native bias once."""

        left = torch.nn.functional.linear(value, self.left)
        right = torch.nn.functional.linear(value, self.right)
        output = torch.nn.functional.linear(left * right, self.decoder)
        shape = (1,) * (output.ndim - 1) + (self.width,)
        return output + self.bias.reshape(shape)

    def masked_write(self, u: torch.Tensor, v: torch.Tensor, mask: str = "all") -> torch.Tensor:
        if mask not in MASKS:
            raise ValueError("unknown typed mask")
        if mask == "all":
            # Polarization is a diagnostic coordinate system, not the deployed
            # execution plan.  Bilinearity makes this exactly K products rather
            # than four separate K-product typed banks.
            return self.write(u + v)
        terms = self.terms(u, v)
        output = sum((terms[name] for name in MASKS[mask]), torch.zeros_like(terms["uu"]))
        shape = (1,) * (output.ndim - 1) + (self.width,)
        return output + self.bias.reshape(shape)


def build_program(
    left: torch.Tensor,
    right: torch.Tensor,
    bias: torch.Tensor,
    indices: torch.Tensor,
    decoder: torch.Tensor,
) -> NativeGateSubsetProgram:
    balanced_left, balanced_right, _ = balance_product_gauge(left, right)
    if indices.ndim != 1 or indices.dtype != torch.long or indices.device != left.device or (
        decoder.shape != (left.shape[1], len(indices))
    ):
        raise ValueError("selected indices or decoder are incompatible")
    return NativeGateSubsetProgram(
        indices=indices.clone(),
        left=balanced_left[indices].contiguous(),
        right=balanced_right[indices].contiguous(),
        decoder=decoder.contiguous(),
        bias=bias.contiguous(),
    )
