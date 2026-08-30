"""Pure-CPU math for causal-response factorization v1.

This module never opens a model, corpus, FIT bundle, or EVAL artifact. It defines the
signed response, prospective document/anchor splits, multilinear shared/private
program, literal pricing, and held-out document-code inference used by the frozen
analysis. Production artifact loading and publication belong in a separate lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping, Sequence

import torch


DOC_SALT = "causal-response-factorization-v1-doc"
ANCHOR_SALT = "causal-response-factorization-v1-anchor"
FIT_TRAIN_DOCUMENTS = 229
ANCHOR_CELLS = 384


def _exact_cpu_tensor(value: object, dtype: torch.dtype, label: str) -> torch.Tensor:
    if type(value) is not torch.Tensor or value.dtype != dtype or (
        value.device.type != "cpu" or not value.is_contiguous()
    ):
        raise TypeError(f"{label} must be an exact contiguous CPU {dtype} tensor")
    if dtype.is_floating_point and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} contains a nonfinite value")
    return value


def signed_response_from_sums(
    statistics: Mapping[str, torch.Tensor],
    member_count: torch.Tensor,
    off_count: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return signed response [phase,source,target,document] and validity mask.

    Counts have shape [target,document]. Unsupported member cells remain invalid and
    are filled with zero only so downstream masked arithmetic stays finite.
    """

    if set(statistics) != {
        "member_signed_sum", "member_abs_sum", "off_signed_sum", "off_abs_sum"
    }:
        raise ValueError("response statistics have missing or unexpected keys")
    member_count = _exact_cpu_tensor(member_count, torch.int64, "member_count")
    off_count = _exact_cpu_tensor(off_count, torch.int64, "off_count")
    if member_count.ndim != 2 or off_count.shape != member_count.shape:
        raise ValueError("counts must align as [target,document]")
    if bool((member_count < 0).any()) or bool((off_count <= 0).any()):
        raise ValueError("member counts must be nonnegative and off counts positive")
    member = _exact_cpu_tensor(
        statistics["member_signed_sum"], torch.float64, "member_signed_sum"
    )
    off = _exact_cpu_tensor(
        statistics["off_signed_sum"], torch.float64, "off_signed_sum"
    )
    expected = member.shape[:3] + (member_count.shape[1],)
    if member.ndim != 4 or member.shape != expected or off.shape != expected or (
        member.shape[2] != member_count.shape[0]
    ):
        raise ValueError("signed sums do not align with [phase,source,target,document]")
    valid_td = member_count > 0
    valid = valid_td[None, None].expand_as(member)
    expanded_member_count = member_count[None, None].expand_as(member)
    expanded_off_count = off_count[None, None].expand_as(off)
    response = torch.zeros_like(member)
    response[valid] = (
        member[valid] / expanded_member_count[valid]
        - off[valid] / expanded_off_count[valid]
    )
    return response.contiguous(), valid.contiguous()


def prospective_document_split(
    document_ids: torch.Tensor, *, train_documents: int = FIT_TRAIN_DOCUMENTS
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return outcome-blind train/validation index vectors in hash order."""

    document_ids = _exact_cpu_tensor(document_ids, torch.int64, "document_ids")
    if document_ids.ndim != 1 or torch.unique(document_ids).numel() != document_ids.numel():
        raise ValueError("document IDs must be a unique vector")
    if not 0 < train_documents < document_ids.numel():
        raise ValueError("train_documents must leave both roles nonempty")
    keyed = []
    for index, value in enumerate(document_ids.tolist()):
        digest = hashlib.sha256(f"{DOC_SALT}|{value}".encode()).digest()
        keyed.append((digest, index))
    order = [index for _, index in sorted(keyed)]
    return (
        torch.tensor(order[:train_documents], dtype=torch.int64),
        torch.tensor(order[train_documents:], dtype=torch.int64),
    )


def prospective_anchor_mask(
    phases: int, sources: int, targets: int, *, anchors: int = ANCHOR_CELLS
) -> torch.Tensor:
    """Return fixed outcome-blind anchor mask over flattened (phase,source,target)."""

    total = phases * sources * targets
    if min(phases, sources, targets) <= 0 or not 0 < anchors < total:
        raise ValueError("anchor count must lie strictly inside the observation count")
    keyed = []
    for p in range(phases):
        for s in range(sources):
            for t in range(targets):
                digest = hashlib.sha256(f"{ANCHOR_SALT}|{p}|{s}|{t}".encode()).digest()
                keyed.append((digest, (p * sources + s) * targets + t))
    mask = torch.zeros(total, dtype=torch.bool)
    mask[torch.tensor([index for _, index in sorted(keyed)[:anchors]])] = True
    return mask


@dataclass(frozen=True)
class ResponseProgram:
    """Canonical factors for a global CP parent plus owner-private CP children."""

    global_phase: torch.Tensor
    global_source: torch.Tensor
    global_target: torch.Tensor
    private_phase: tuple[torch.Tensor, ...]
    private_source: tuple[torch.Tensor, ...]
    private_target: tuple[torch.Tensor, ...]
    source_groups: torch.Tensor

    def __post_init__(self) -> None:
        tensors = (self.global_phase, self.global_source, self.global_target)
        for label, value in zip(("global_phase", "global_source", "global_target"), tensors):
            _exact_cpu_tensor(value, torch.float64, label)
            if value.ndim != 2:
                raise ValueError(f"{label} must be a matrix")
        p, k0 = self.global_phase.shape
        s, k0s = self.global_source.shape
        t, k0t = self.global_target.shape
        if k0 != k0s or k0 != k0t:
            raise ValueError("global factors must share rank")
        groups = _exact_cpu_tensor(self.source_groups, torch.int64, "source_groups")
        if groups.shape != (s,) or groups.min() < 0:
            raise ValueError("source_groups must assign every source")
        group_count = int(groups.max()) + 1
        if not (
            len(self.private_phase) == len(self.private_source)
            == len(self.private_target) == group_count
        ):
            raise ValueError("private factor tuples must match source-group count")
        for group in range(group_count):
            ap = _exact_cpu_tensor(self.private_phase[group], torch.float64, "private_phase")
            bs = _exact_cpu_tensor(self.private_source[group], torch.float64, "private_source")
            ct = _exact_cpu_tensor(self.private_target[group], torch.float64, "private_target")
            group_sources = int((groups == group).sum())
            if ap.ndim != 2 or bs.ndim != 2 or ct.ndim != 2 or (
                ap.shape[0] != p or bs.shape[0] != group_sources or ct.shape[0] != t
                or ap.shape[1] != bs.shape[1] or ap.shape[1] != ct.shape[1]
            ):
                raise ValueError("private factors have inconsistent dimensions")

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.global_phase.shape[0], self.global_source.shape[0], self.global_target.shape[0]

    @property
    def code_dimension(self) -> int:
        return self.global_phase.shape[1] + sum(value.shape[1] for value in self.private_phase)

    @property
    def persistent_values(self) -> int:
        total = sum(value.numel() for value in (
            self.global_phase, self.global_source, self.global_target
        ))
        for family in (self.private_phase, self.private_source, self.private_target):
            total += sum(value.numel() for value in family)
        return total

    def basis(self) -> torch.Tensor:
        """Materialize observation-by-code basis only for fitting/scoring tests."""

        p, s, t = self.shape
        columns = []
        if self.global_phase.shape[1]:
            global_tensor = torch.einsum(
                "pk,sk,tk->pstk", self.global_phase, self.global_source, self.global_target
            )
            columns.append(global_tensor.reshape(p * s * t, -1))
        for group, (ap, bs, ct) in enumerate(zip(
            self.private_phase, self.private_source, self.private_target
        )):
            if ap.shape[1] == 0:
                continue
            local = torch.einsum("pk,sk,tk->pstk", ap, bs, ct)
            full = torch.zeros((p, s, t, ap.shape[1]), dtype=torch.float64)
            full[:, self.source_groups == group] = local
            columns.append(full.reshape(p * s * t, -1))
        if not columns:
            return torch.empty((p * s * t, 0), dtype=torch.float64)
        return torch.cat(columns, dim=1).contiguous()


def infer_document_codes(
    basis: torch.Tensor,
    responses: torch.Tensor,
    valid: torch.Tensor,
    anchors: torch.Tensor,
    *,
    ridge: float = 1e-8,
    minimum_anchor_ratio: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Infer one code per document from valid anchors using ridge least squares."""

    basis = _exact_cpu_tensor(basis, torch.float64, "basis")
    responses = _exact_cpu_tensor(responses, torch.float64, "responses")
    valid = _exact_cpu_tensor(valid, torch.bool, "valid")
    anchors = _exact_cpu_tensor(anchors, torch.bool, "anchors")
    if basis.ndim != 2 or responses.ndim != 2 or valid.shape != responses.shape or (
        basis.shape[0] != responses.shape[0] or anchors.shape != (basis.shape[0],)
    ):
        raise ValueError("basis, responses, validity, and anchors do not align")
    if ridge < 0 or minimum_anchor_ratio < 1:
        raise ValueError("ridge and minimum_anchor_ratio are invalid")
    k = basis.shape[1]
    codes = torch.zeros((responses.shape[1], k), dtype=torch.float64)
    supported = torch.zeros(responses.shape[1], dtype=torch.bool)
    if k == 0:
        return codes, supported
    eye = torch.eye(k, dtype=torch.float64)
    for document in range(responses.shape[1]):
        mask = anchors & valid[:, document]
        if int(mask.sum()) < minimum_anchor_ratio * k:
            continue
        design = basis[mask]
        target = responses[mask, document]
        codes[document] = torch.linalg.solve(
            design.T @ design + ridge * eye, design.T @ target
        )
        supported[document] = True
    return codes, supported


def predict_from_codes(basis: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    basis = _exact_cpu_tensor(basis, torch.float64, "basis")
    codes = _exact_cpu_tensor(codes, torch.float64, "codes")
    if basis.ndim != 2 or codes.ndim != 2 or basis.shape[1] != codes.shape[1]:
        raise ValueError("basis and codes do not align")
    return (basis @ codes.T).contiguous()


@dataclass(frozen=True)
class FitResult:
    program: ResponseProgram
    document_codes: torch.Tensor
    initial_mse: float
    final_mse: float
    improvement_fraction: float
    steps: int
    seed: int


def _canonicalize_block(
    factors: Sequence[torch.Tensor], codes: torch.Tensor
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Fix continuous scale/sign and discrete permutation without changing output."""

    normalized = [factor.detach().clone() for factor in factors]
    canonical_codes = codes.detach().clone()
    rank = canonical_codes.shape[1]
    for factor_index, factor in enumerate(normalized):
        norms = factor.norm(dim=0)
        if bool((~torch.isfinite(norms)).any()) or bool((norms == 0).any()):
            raise RuntimeError("a fitted CP factor column is zero or nonfinite")
        normalized[factor_index] = factor / norms
        canonical_codes = canonical_codes * norms
        factor = normalized[factor_index]
        pivots = factor.abs().argmax(dim=0)
        signs = factor[pivots, torch.arange(rank)].sign()
        signs[signs == 0] = 1
        normalized[factor_index] = factor * signs
        canonical_codes = canonical_codes * signs
    keys = []
    for column in range(rank):
        payload = torch.cat([factor[:, column] for factor in normalized]).numpy().tobytes()
        keys.append((hashlib.sha256(payload).digest(), column))
    order = torch.tensor([column for _, column in sorted(keys)], dtype=torch.int64)
    return (
        tuple(factor[:, order].contiguous() for factor in normalized),  # type: ignore[return-value]
        canonical_codes[:, order].contiguous(),
    )


def fit_shared_private_program(
    response: torch.Tensor,
    valid: torch.Tensor,
    source_groups: torch.Tensor,
    *,
    global_rank: int,
    private_rank: int,
    seed: int,
    steps: int = 2_000,
    learning_rate: float = 0.03,
) -> FitResult:
    """Fit the frozen shared-parent/component-private CP family with Adam.

    This is a deterministic CPU float64 optimizer given its inputs and seed. It is a
    mathematical fitter only: it performs no candidate selection or artifact I/O.
    """

    response = _exact_cpu_tensor(response, torch.float64, "response")
    valid = _exact_cpu_tensor(valid, torch.bool, "valid")
    source_groups = _exact_cpu_tensor(source_groups, torch.int64, "source_groups")
    if response.ndim != 4 or valid.shape != response.shape:
        raise ValueError("response and validity must align as [phase,source,target,document]")
    p, s, t, d = response.shape
    if source_groups.shape != (s,) or source_groups.min() < 0:
        raise ValueError("source_groups must assign every source")
    if global_rank < 0 or private_rank < 0 or global_rank + private_rank == 0:
        raise ValueError("at least one nonnegative shared/private rank is required")
    if steps < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("optimizer controls are invalid")
    if not bool(valid.any()):
        raise ValueError("at least one response cell must be valid")
    group_count = int(source_groups.max()) + 1
    if set(source_groups.tolist()) != set(range(group_count)):
        raise ValueError("source group labels must be contiguous")

    generator = torch.Generator().manual_seed(seed)

    def parameter(shape: tuple[int, ...]) -> torch.nn.Parameter:
        value = 0.35 * torch.randn(shape, generator=generator, dtype=torch.float64)
        return torch.nn.Parameter(value)

    global_factors = [
        parameter((p, global_rank)), parameter((s, global_rank)),
        parameter((t, global_rank)), parameter((d, global_rank)),
    ] if global_rank else []
    private_factors: list[list[torch.nn.Parameter]] = []
    for group in range(group_count):
        group_sources = int((source_groups == group).sum())
        private_factors.append([
            parameter((p, private_rank)), parameter((group_sources, private_rank)),
            parameter((t, private_rank)), parameter((d, private_rank)),
        ] if private_rank else [])
    parameters = list(global_factors)
    for group in private_factors:
        parameters.extend(group)
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    def prediction() -> torch.Tensor:
        estimate = torch.zeros_like(response)
        if global_rank:
            estimate = estimate + torch.einsum(
                "pk,sk,tk,dk->pstd", *global_factors
            )
        if private_rank:
            for group, factors in enumerate(private_factors):
                estimate[:, source_groups == group] += torch.einsum(
                    "pk,sk,tk,dk->pstd", *factors
                )
        return estimate

    with torch.no_grad():
        initial = float(((prediction()[valid] - response[valid]) ** 2).mean())
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        estimate = prediction()
        loss = ((estimate[valid] - response[valid]) ** 2).mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("shared/private optimizer became nonfinite")
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final = float(((prediction()[valid] - response[valid]) ** 2).mean())
    if not math.isfinite(final):
        raise RuntimeError("shared/private optimizer ended nonfinite")

    if global_rank:
        global_block, global_codes = _canonicalize_block(
            global_factors[:3], global_factors[3]
        )
    else:
        global_block = (
            torch.empty((p, 0), dtype=torch.float64),
            torch.empty((s, 0), dtype=torch.float64),
            torch.empty((t, 0), dtype=torch.float64),
        )
        global_codes = torch.empty((d, 0), dtype=torch.float64)
    private_blocks = []
    private_codes = []
    for group, factors in enumerate(private_factors):
        if private_rank:
            block, codes = _canonicalize_block(factors[:3], factors[3])
        else:
            block = (
                torch.empty((p, 0), dtype=torch.float64),
                torch.empty((int((source_groups == group).sum()), 0), dtype=torch.float64),
                torch.empty((t, 0), dtype=torch.float64),
            )
            codes = torch.empty((d, 0), dtype=torch.float64)
        private_blocks.append(block)
        private_codes.append(codes)
    program = make_program_from_factors(global_block, private_blocks, source_groups)
    all_codes = torch.cat([global_codes, *private_codes], dim=1).contiguous()
    replay = predict_from_codes(program.basis(), all_codes).reshape(p, s, t, d)
    replay_final = float(((replay[valid] - response[valid]) ** 2).mean())
    if not math.isclose(replay_final, final, rel_tol=1e-10, abs_tol=1e-12):
        raise RuntimeError("canonical factor program does not replay fitted loss")
    improvement = (initial - final) / max(initial, torch.finfo(torch.float64).tiny)
    return FitResult(
        program=program,
        document_codes=all_codes,
        initial_mse=initial,
        final_mse=final,
        improvement_fraction=float(improvement),
        steps=steps,
        seed=seed,
    )


def make_program_from_factors(
    global_factors: Sequence[torch.Tensor],
    private_factors: Sequence[Sequence[torch.Tensor]],
    source_groups: torch.Tensor,
) -> ResponseProgram:
    """Construct a program from `(phase,source,target)` factor triples."""

    if len(global_factors) != 3 or any(len(group) != 3 for group in private_factors):
        raise ValueError("every CP block needs phase, source, and target factors")
    return ResponseProgram(
        *(value.detach().to(dtype=torch.float64, device="cpu").contiguous()
          for value in global_factors),
        private_phase=tuple(group[0].detach().double().cpu().contiguous() for group in private_factors),
        private_source=tuple(group[1].detach().double().cpu().contiguous() for group in private_factors),
        private_target=tuple(group[2].detach().double().cpu().contiguous() for group in private_factors),
        source_groups=source_groups.detach().to(dtype=torch.int64, device="cpu").contiguous(),
    )
