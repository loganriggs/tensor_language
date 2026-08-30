"""Accelerated optimizer for the frozen shared/private response program.

Inputs, canonical factors, replay, and reported losses remain CPU float64. Only the
fixed Adam optimization loop may run on CUDA float32. Initialization is generated on
CPU float64 before casting, so seeds have one device-independent mathematical
preimage. This module performs no artifact I/O, candidate selection, or validation.
"""

from __future__ import annotations

import math

import torch

from causal_response_factorization_v1 import (
    FitResult,
    _canonicalize_block,
    _exact_cpu_tensor,
    make_program_from_factors,
    predict_from_codes,
)


def _seeded_initial_factors(
    shape: tuple[int, int, int, int],
    source_groups: torch.Tensor,
    *,
    global_rank: int,
    private_rank: int,
    seed: int,
) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
    """Return the one device-independent CPU-float64 optimizer preimage."""

    p, s, t, d = shape
    group_count = int(source_groups.max()) + 1
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def master(value_shape: tuple[int, ...]) -> torch.Tensor:
        return 0.35 * torch.randn(
            value_shape, generator=generator, dtype=torch.float64,
        )

    global_factors = [
        master((p, global_rank)), master((s, global_rank)),
        master((t, global_rank)), master((d, global_rank)),
    ] if global_rank else []
    private_factors = []
    for group in range(group_count):
        group_sources = int((source_groups == group).sum())
        private_factors.append([
            master((p, private_rank)), master((group_sources, private_rank)),
            master((t, private_rank)), master((d, private_rank)),
        ] if private_rank else [])
    return global_factors, private_factors


def seeded_initial_mse(
    response: torch.Tensor,
    valid: torch.Tensor,
    source_groups: torch.Tensor,
    *,
    global_rank: int,
    private_rank: int,
    seed: int,
) -> float:
    """Replay the optimizer initialization loss entirely on CPU float64."""

    response = _exact_cpu_tensor(response, torch.float64, "response")
    valid = _exact_cpu_tensor(valid, torch.bool, "valid")
    source_groups = _exact_cpu_tensor(source_groups, torch.int64, "source_groups")
    if response.ndim != 4 or valid.shape != response.shape or not bool(valid.any()):
        raise ValueError("initial-loss response and validity are malformed")
    if source_groups.shape != (response.shape[1],) or source_groups.min() < 0 or (
        global_rank < 0 or private_rank < 0 or global_rank + private_rank == 0
    ):
        raise ValueError("initial-loss topology or ranks are malformed")
    global_factors, private_factors = _seeded_initial_factors(
        tuple(response.shape), source_groups, global_rank=global_rank,
        private_rank=private_rank, seed=seed,
    )
    estimate = torch.zeros_like(response)
    if global_rank:
        estimate += torch.einsum("pk,sk,tk,dk->pstd", *global_factors)
    if private_rank:
        for group, factors in enumerate(private_factors):
            estimate[:, source_groups == group] += torch.einsum(
                "pk,sk,tk,dk->pstd", *factors,
            )
    return float(((estimate[valid] - response[valid]) ** 2).mean())


def fit_shared_private_program_accelerated(
    response: torch.Tensor,
    valid: torch.Tensor,
    source_groups: torch.Tensor,
    *,
    global_rank: int,
    private_rank: int,
    seed: int,
    steps: int = 2_000,
    learning_rate: float = 0.03,
    optimizer_device: str = "cuda",
) -> FitResult:
    """Fit the exact shared/private topology with a device-local Adam loop."""

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
    device = torch.device(optimizer_device)
    if device.type not in ("cpu", "cuda") or (
        device.type == "cuda" and not torch.cuda.is_available()
    ):
        raise RuntimeError("requested accelerated optimizer device is unavailable")
    dtype = torch.float32
    response_work = response.to(device=device, dtype=dtype)
    valid_work = valid.to(device=device)
    groups_work = source_groups.to(device=device)
    group_masks = tuple(groups_work == group for group in range(group_count))
    global_master, private_master = _seeded_initial_factors(
        (p, s, t, d), source_groups, global_rank=global_rank,
        private_rank=private_rank, seed=seed,
    )

    def parameter(master: torch.Tensor) -> torch.nn.Parameter:
        return torch.nn.Parameter(master.to(device=device, dtype=dtype))

    global_factors = [parameter(value) for value in global_master]
    private_factors: list[list[torch.nn.Parameter]] = [
        [parameter(value) for value in block] for block in private_master
    ]
    parameters = list(global_factors)
    for group in private_factors:
        parameters.extend(group)
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    def prediction() -> torch.Tensor:
        estimate = torch.zeros_like(response_work)
        if global_rank:
            estimate = estimate + torch.einsum(
                "pk,sk,tk,dk->pstd", *global_factors
            )
        if private_rank:
            for group, factors in enumerate(private_factors):
                estimate[:, group_masks[group]] += torch.einsum(
                    "pk,sk,tk,dk->pstd", *factors
                )
        return estimate

    initial = seeded_initial_mse(
        response, valid, source_groups, global_rank=global_rank,
        private_rank=private_rank, seed=seed,
    )
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        estimate = prediction()
        loss = ((estimate[valid_work] - response_work[valid_work]) ** 2).mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("accelerated shared/private optimizer became nonfinite")
        loss.backward()
        optimizer.step()

    def cpu_double(value: torch.Tensor) -> torch.Tensor:
        return value.detach().to(device="cpu", dtype=torch.float64).contiguous()

    if global_rank:
        global_block, global_codes = _canonicalize_block(
            [cpu_double(value) for value in global_factors[:3]],
            cpu_double(global_factors[3]),
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
            block, codes = _canonicalize_block(
                [cpu_double(value) for value in factors[:3]], cpu_double(factors[3])
            )
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
    replay = predict_from_codes(program.basis(), all_codes).reshape_as(response)
    final = float(((replay[valid] - response[valid]) ** 2).mean())
    if not math.isfinite(final):
        raise RuntimeError("accelerated canonical replay ended nonfinite")
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
