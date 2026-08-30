#!/usr/bin/env python3
"""FIT-only empirical-fourth-moment optimization of the MLP1 pre-gate router.

This discovery runner reuses the exact P512 bundle and its already-opened disjoint
FIT/SELECT roles. It never requests FINAL. Coefficient-Frobenius signed eigenmodes are
the fixed initialization and matched control; only real-state score error supplies
gradients. Frozen candidates are then evaluated through the native model suffix.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BQ = HERE.parent / "bilinear_quotient"
for root in (ROOT, HERE, BQ):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import bilin18_observed_model_facade as facade
import explore_mlp1_pregate_quadratic_router as control
import mlp1_sparse_down_program_v1 as sparse
import run_mlp1_sparse_c512_continue_factorial_v1_fit as base


BUNDLE = HERE / "mlp1_sparse_c512_continue_factorial_v2_fit_bundle.pt"
P512_RESULT = HERE / "mlp1_sparse_c512_continue_factorial_v2_fit_result.json"
ROWS_RECEIPT = BQ / "mlp1_sparse_c512_continue_factorial_v1_rows_receipt.json"
PREREGISTRATION = HERE / "MLP1_PREGATE_EMPIRICAL_M4_ROUTER_DISCOVERY_PREREGISTRATION.md"
OUTPUT = HERE / "mlp1_pregate_empirical_m4_router_discovery.json"

RANKS = (8,)
STEPS = 1_200
BATCH_SIZE = 256
LEARNING_RATE = 0.003
FINAL_LEARNING_RATE = 0.0003
SEED = 73_031
MONITOR_POSITIONS = 2_048
MONITOR_EVERY = 200
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 91_173


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def empirical_score_loss(
    states: torch.Tensor,
    targets: torch.Tensor,
    factors: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """Sample fourth-moment loss, normalized by exact uncentered score energy."""

    predictions = control.quadratic_scores(states, factors, values, factors.shape[2])
    numerator = (predictions.double() - targets.double()).square().mean()
    denominator = targets.double().square().mean().clamp_min(1e-30)
    return numerator / denominator


def explicit_fourth_moment_loss(
    states: torch.Tensor, error_matrices: torch.Tensor,
) -> torch.Tensor:
    """Explicit M4 contraction for small known-answer tests only."""

    if states.ndim != 2 or error_matrices.ndim != 3 \
            or error_matrices.shape[1:] != (states.shape[1], states.shape[1]):
        raise ValueError("state/error-matrix shapes disagree")
    moment4 = torch.einsum("ni,nj,nk,nl->ijkl", states, states, states, states) / len(states)
    return torch.einsum("aij,ijkl,akl->", error_matrices, moment4, error_matrices) \
        / len(error_matrices)


class EmpiricalFactorBank(nn.Module):
    """Independent signed-square grammars for the prospectively frozen ranks."""

    def __init__(self, initial: Mapping[int, tuple[torch.Tensor, torch.Tensor]]) -> None:
        super().__init__()
        if not initial or any(type(rank) is not int or rank <= 0 for rank in initial):
            raise ValueError("initial rank bank changed")
        self.ranks = tuple(sorted(initial))
        self.raw_factors = nn.ParameterDict()
        self.signed_values = nn.ParameterDict()
        for rank in self.ranks:
            factors, values = initial[rank]
            if factors.ndim != 3 or factors.shape[2] != rank \
                    or values.shape != (factors.shape[0], rank):
                raise ValueError("initial empirical factor shapes changed")
            self.raw_factors[str(rank)] = nn.Parameter(factors.float().clone())
            self.signed_values[str(rank)] = nn.Parameter(values.float().clone())

    def factors(self, rank: int) -> torch.Tensor:
        return self.raw_factors[str(rank)]

    def scores(self, states: torch.Tensor, rank: int) -> torch.Tensor:
        return control.quadratic_scores(
            states, self.factors(rank), self.signed_values[str(rank)], rank,
        )

    @torch.no_grad()
    def renormalize_(self) -> None:
        """Fix vector scale gauge while preserving every represented quadratic."""

        for rank in self.ranks:
            raw = self.raw_factors[str(rank)]
            norms = raw.norm(dim=1).clamp_min(1e-12)
            raw.div_(norms[:, None, :])
            self.signed_values[str(rank)].mul_(norms.square())

    @torch.no_grad()
    def export(self) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        return {
            rank: canonicalize_signed_squares(
                self.factors(rank).detach(),
                self.signed_values[str(rank)].detach(),
            )
            for rank in self.ranks
        }


@torch.no_grad()
def canonicalize_signed_squares(
    factors: torch.Tensor, values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Canonical orthonormal eigensquares for the same batched quadratic forms."""

    if factors.ndim != 3 or values.shape != (factors.shape[0], factors.shape[2]):
        raise ValueError("canonical factor shapes changed")
    basis, triangular = torch.linalg.qr(factors, mode="reduced")
    small = (triangular * values[:, None, :]) @ triangular.transpose(1, 2)
    small = 0.5 * (small + small.transpose(1, 2))
    eigenvalues, rotations = torch.linalg.eigh(small)
    order = eigenvalues.abs().argsort(dim=1, descending=True)
    eigenvalues = eigenvalues.gather(1, order)
    rotations = rotations.gather(
        2, order[:, None, :].expand(-1, rotations.shape[1], -1),
    )
    canonical = basis @ rotations
    pivot_indices = canonical.abs().argmax(dim=1, keepdim=True)
    pivots = canonical.gather(1, pivot_indices).squeeze(1)
    signs = torch.where(pivots < 0, -torch.ones_like(pivots), torch.ones_like(pivots))
    canonical = canonical * signs[:, None, :]
    return canonical.cpu().contiguous(), eigenvalues.cpu().contiguous()


@torch.no_grad()
def monitor_losses(
    bank: EmpiricalFactorBank, states: torch.Tensor, targets: torch.Tensor,
) -> dict[str, float]:
    return {
        str(rank): float(empirical_score_loss(
            states, targets, bank.factors(rank), bank.signed_values[str(rank)],
        ))
        for rank in bank.ranks
    }


def fit_empirical_factors(
    states: torch.Tensor,
    targets: torch.Tensor,
    initial: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    steps: int = STEPS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    final_learning_rate: float = FINAL_LEARNING_RATE,
    seed: int = SEED,
    monitor_positions: int = MONITOR_POSITIONS,
    monitor_every: int = MONITOR_EVERY,
) -> tuple[dict[int, tuple[torch.Tensor, torch.Tensor]], list[dict[str, object]]]:
    """Optimize all fixed ranks on FIT scores only, without early stopping."""

    if states.ndim != 2 or targets.ndim != 2 or len(states) != len(targets) \
            or not len(states) or steps <= 0 or not 0 < batch_size <= len(states):
        raise ValueError("empirical fit inputs changed")
    if monitor_every <= 0 or steps % monitor_every:
        raise ValueError("monitor cadence must divide fixed steps")
    bank = EmpiricalFactorBank(initial).to(device)
    bank.renormalize_()
    optimizer = torch.optim.Adam(bank.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=final_learning_rate,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    monitor_count = min(monitor_positions, len(states))
    monitor_x = states[:monitor_count].to(device)
    monitor_y = targets[:monitor_count].to(device)
    curve: list[dict[str, object]] = [{
        "step": 0,
        "learning_rate": learning_rate,
        "fit_monitor_relative_score_mse": monitor_losses(bank, monitor_x, monitor_y),
    }]
    for step in range(1, steps + 1):
        indices = torch.randint(len(states), (batch_size,), generator=generator)
        x = states[indices].to(device)
        y = targets[indices].to(device)
        denominator = y.double().square().mean().clamp_min(1e-30)
        losses = []
        for rank in bank.ranks:
            prediction = bank.scores(x, rank)
            losses.append((prediction.double() - y.double()).square().mean() / denominator)
        loss = torch.stack(losses).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bank.parameters(), 100.0)
        optimizer.step()
        bank.renormalize_()
        scheduler.step()
        if step % monitor_every == 0:
            row = {
                "step": step,
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "batch_total_normalized_loss": float(loss.detach()),
                "fit_monitor_relative_score_mse": monitor_losses(
                    bank, monitor_x, monitor_y,
                ),
            }
            curve.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    frozen = bank.export()
    with torch.no_grad():
        replay = {}
        for rank in bank.ranks:
            before = bank.scores(monitor_x, rank)
            factors, values = frozen[rank]
            after = control.quadratic_scores(
                monitor_x, factors.to(device), values.to(device), rank,
            )
            replay[str(rank)] = float(
                (after.double() - before.double()).square().sum()
                / before.double().square().sum().clamp_min(1e-30)
            )
            if replay[str(rank)] > 1e-10:
                raise RuntimeError("canonical factor replay changed scores")
        curve[-1]["canonicalization_relative_score_mse"] = replay
    return frozen, curve


@torch.no_grad()
def exact_scores(
    states: torch.Tensor, left: torch.Tensor, right: torch.Tensor,
    encoder: torch.Tensor, device: torch.device,
) -> torch.Tensor:
    output = []
    for start in range(0, len(states), control.STATE_CHUNK):
        x = states[start:start + control.STATE_CHUNK].to(device)
        output.append((((x @ left.T) * (x @ right.T)) @ encoder.T).cpu())
    return torch.cat(output)


@torch.no_grad()
def capture_fit_references(
    model, rows: torch.Tensor, exact_program: sparse.SparseDownProgram,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """Capture states and finite-precision deployed P512 scores in one native pass."""

    states, deployed_scores = [], []
    calls = {"forwards": 0, "attention": 0, "site1": 0, "other_mlp": 0}
    for start in range(0, len(rows), control.DOCUMENT_BATCH):
        tokens = rows[start:start + control.DOCUMENT_BATCH, :-1].to(device)

        def attention(event: facade.AttentionEvent):
            calls["attention"] += 1
            return event.block.attn(event.state, event.first_value)

        def mlp(event: facade.EarlyMLPEvent):
            if event.site != 1:
                calls["other_mlp"] += 1
                return event.block.mlp(event.state)
            state = event.state
            left = event.block.mlp.Left(state)
            right = event.block.mlp.Right(state)
            gate = left * right
            flat_gate = gate[:, control.SCORING].float().reshape(-1, control.H)
            states.append(
                state[:, control.SCORING].detach().float().cpu().reshape(-1, control.D)
            )
            deployed_scores.append(
                (flat_gate @ exact_program.encoder.T).detach().float().cpu()
            )
            calls["site1"] += 1
            return event.block.mlp.Down(gate) + event.block.mlp.Down_bias

        facade.forward_with_dispatch(model, tokens, attention, mlp)
        calls["forwards"] += 1
    expected = len(rows) // control.DOCUMENT_BATCH
    wanted = {
        "forwards": expected,
        "attention": 18 * expected,
        "site1": expected,
        "other_mlp": 17 * expected,
    }
    if calls != wanted:
        raise RuntimeError(f"FIT reference capture census changed: {calls}")
    captured_states = torch.cat(states)
    captured_scores = torch.cat(deployed_scores)
    if tuple(captured_states.shape) != (len(rows) * 192, control.D) \
            or tuple(captured_scores.shape) != (len(rows) * 192, control.P):
        raise RuntimeError("FIT reference capture shapes changed")
    return captured_states, captured_scores, calls


def bootstrap_mean_interval(
    values: torch.Tensor, *, draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Fixed-seed equal-document percentile interval for a paired mean."""

    values = values.detach().cpu().double().flatten()
    if len(values) < 2 or not bool(torch.isfinite(values).all()) or draws <= 0:
        raise ValueError("bootstrap values changed")
    generator = torch.Generator().manual_seed(seed)
    samples = []
    for start in range(0, draws, 1_000):
        count = min(1_000, draws - start)
        indices = torch.randint(len(values), (count, len(values)), generator=generator)
        samples.append(values[indices].mean(1))
    means = torch.cat(samples).sort().values
    low = means[int(math.floor(0.025 * (draws - 1)))]
    high = means[int(math.ceil(0.975 * (draws - 1)))]
    return {
        "mean": float(values.mean()),
        "bootstrap_95_low": float(low),
        "bootstrap_95_high": float(high),
        "documents": len(values),
        "draws": draws,
    }


@torch.no_grad()
def physical_ce_comparison(
    model,
    rows: torch.Tensor,
    exact_program: sparse.SparseDownProgram,
    coefficient: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
    empirical: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
    decoder: torch.Tensor,
    intercept: torch.Tensor,
    folded_bias: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, list[float]], dict[str, dict[str, int]]]:
    """Score matched router grammars and retain paired document-level CE."""

    arms = (
        "NATIVE", "ZERO", "P512_EXACT",
        *(f"COEFFICIENT_RANK_{rank}" for rank in RANKS),
        *(f"EMPIRICAL_RANK_{rank}" for rank in RANKS),
    )
    coefficient = {
        rank: (factor.to(device), value.to(device))
        for rank, (factor, value) in coefficient.items()
    }
    empirical = {
        rank: (factor.to(device), value.to(device))
        for rank, (factor, value) in empirical.items()
    }
    document_ce = {arm: [] for arm in arms}
    calls = {
        arm: {
            "forwards": 0, "attention": 0, "site1_native": 0,
            "site1_replacement": 0, "other_mlp": 0,
            "native_left": 0, "native_right": 0, "native_down": 0,
        }
        for arm in arms
    }
    current_arm = {"name": None}

    def count_submodule(name: str):
        def hook(_module, _inputs, _output):
            arm = current_arm["name"]
            if arm is None:
                raise RuntimeError("native MLP1 submodule called outside scored arm")
            calls[arm][name] += 1
        return hook

    hooks = [
        model.transformer.h[1].mlp.Left.register_forward_hook(count_submodule("native_left")),
        model.transformer.h[1].mlp.Right.register_forward_hook(count_submodule("native_right")),
        model.transformer.h[1].mlp.Down.register_forward_hook(count_submodule("native_down")),
    ]
    try:
        for start in range(0, len(rows), control.DOCUMENT_BATCH):
            batch = rows[start:start + control.DOCUMENT_BATCH]
            tokens = batch[:, :-1].to(device)
            targets = batch[:, 1:].to(device)
            for arm in arms:
                current_arm["name"] = arm

                def attention(event: facade.AttentionEvent, arm=arm):
                    calls[arm]["attention"] += 1
                    return event.block.attn(event.state, event.first_value)

                def mlp(event: facade.EarlyMLPEvent, arm=arm):
                    if event.site != 1:
                        calls[arm]["other_mlp"] += 1
                        return event.block.mlp(event.state)
                    if arm == "NATIVE":
                        calls[arm]["site1_native"] += 1
                        return event.block.mlp(event.state)
                    calls[arm]["site1_replacement"] += 1
                    if arm == "ZERO":
                        action = torch.zeros_like(event.state)
                        return action + event.block.mlp.Down_bias
                    elif arm == "P512_EXACT":
                        gate = event.block.mlp.Left(event.state) \
                            * event.block.mlp.Right(event.state)
                        action = exact_program(gate)
                        return action + event.block.mlp.Down_bias
                    else:
                        rank = int(arm.rsplit("_", 1)[1])
                        bank = empirical if arm.startswith("EMPIRICAL") else coefficient
                        factors, values = bank[rank]
                        flat = event.state.float().reshape(-1, control.D)
                        scores = control.quadratic_scores(flat, factors, values, rank)
                        action = (
                            sparse.topk_relu(scores) @ decoder.T + folded_bias
                        ).reshape_as(event.state).to(event.state.dtype)
                        return action

                logits = facade.forward_with_dispatch(model, tokens, attention, mlp)
                losses = F.cross_entropy(
                    logits[:, control.SCORING].transpose(1, 2),
                    targets[:, control.SCORING], reduction="none",
                ).mean(1)
                document_ce[arm].extend(float(value) for value in losses)
                calls[arm]["forwards"] += 1
                current_arm["name"] = None
    finally:
        current_arm["name"] = None
        for hook in hooks:
            hook.remove()

    expected = len(rows) // control.DOCUMENT_BATCH
    for arm in arms:
        native_gate = arm in ("NATIVE", "P512_EXACT")
        wanted = {
            "forwards": expected,
            "attention": 18 * expected,
            "site1_native": expected if arm == "NATIVE" else 0,
            "site1_replacement": 0 if arm == "NATIVE" else expected,
            "other_mlp": 17 * expected,
            "native_left": expected if native_gate else 0,
            "native_right": expected if native_gate else 0,
            "native_down": expected if arm == "NATIVE" else 0,
        }
        if calls[arm] != wanted or len(document_ce[arm]) != len(rows):
            raise RuntimeError(f"physical comparison census changed for {arm}: {calls[arm]}")
    pooled = {
        arm: float(torch.tensor(values, dtype=torch.float64).mean())
        for arm, values in document_ce.items()
    }
    return pooled, document_ce, calls


def route_metric_bank(
    states: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    encoder: torch.Tensor,
    bank: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
    decoder: torch.Tensor,
    intercept: torch.Tensor,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Reuse the audited metric routine while preserving independently fit ranks."""

    result = {}
    for rank in RANKS:
        factors, values = bank[rank]
        metrics = control.route_metrics(
            states, left, right, encoder, factors.to(device), values.to(device),
            decoder, intercept, device,
        )[str(rank)]
        result[str(rank)] = metrics
    return result


def main() -> None:
    started = time.time()
    if OUTPUT.exists():
        raise RuntimeError(f"discovery namespace already exists: {OUTPUT}")
    receipt = json.loads(ROWS_RECEIPT.read_text())
    fit_rows = base.load_role(receipt["entries"]["FIT"])
    select_rows = base.load_role(receipt["entries"]["SELECT"])
    bundle = torch.load(BUNDLE, map_location="cpu", weights_only=True)
    state = sparse.validate_state(bundle["program"])
    device = torch.device("cuda")
    model, checkpoint = facade.load_bilin18(device=device, dtype=torch.bfloat16)
    left = model.transformer.h[1].mlp.Left.weight.detach().float()
    right = model.transformer.h[1].mlp.Right.weight.detach().float()
    encoder = state["encoder"].to(device)
    decoder = state["decoder"].to(device)
    intercept = state["intercept"].to(device)
    folded_bias = intercept + model.transformer.h[1].mlp.Down_bias.detach().float()
    exact_program = sparse.SparseDownProgram(state, device).eval()

    coefficient_factors, coefficient_values, coefficient_diagnostics = \
        control.implicit_quadratic_factors(left, right, encoder)
    coefficient = {
        rank: (
            coefficient_factors[:, :, :rank].clone(),
            coefficient_values[:, :rank].clone(),
        )
        for rank in RANKS
    }
    fit_states, fit_deployed_targets, fit_capture_calls = capture_fit_references(
        model, fit_rows, exact_program, device,
    )
    select_states, select_capture_calls = control.capture_states(model, select_rows, device)
    fit_targets = exact_scores(fit_states, left, right, encoder, device)
    deployed_target_discrepancy = float(
        (fit_deployed_targets.double() - fit_targets.double()).square().mean()
        / fit_targets.double().square().mean().clamp_min(1e-30)
    )

    empirical, training_curve = fit_empirical_factors(
        fit_states, fit_targets, coefficient, device=device,
    )
    coefficient_metrics = route_metric_bank(
        select_states, left, right, encoder, coefficient, decoder, intercept, device,
    )
    empirical_metrics = route_metric_bank(
        select_states, left, right, encoder, empirical, decoder, intercept, device,
    )
    ce, document_ce, score_calls = physical_ce_comparison(
        model, select_rows, exact_program, coefficient, empirical,
        decoder, intercept, folded_bias, device,
    )

    anchors = json.loads(P512_RESULT.read_text())["select_ce"]
    for actual, expected, label in (
        (ce["NATIVE"], anchors["NATIVE"], "native"),
        (ce["ZERO"], anchors["ZERO"], "zero"),
        (ce["P512_EXACT"], anchors["SPARSE"], "P512"),
    ):
        if abs(actual - expected) > 2e-6:
            raise RuntimeError(f"{label} physical CE anchor did not replay")

    deletion_stake = ce["ZERO"] - ce["NATIVE"]
    p512_stake = ce["ZERO"] - ce["P512_EXACT"]
    comparisons = {}
    for rank in RANKS:
        empirical_arm = f"EMPIRICAL_RANK_{rank}"
        coefficient_arm = f"COEFFICIENT_RANK_{rank}"
        emp_docs = torch.tensor(document_ce[empirical_arm])
        coefficient_docs = torch.tensor(document_ce[coefficient_arm])
        p512_docs = torch.tensor(document_ce["P512_EXACT"])
        comparisons[str(rank)] = {
            "coefficient_ce": ce[coefficient_arm],
            "empirical_ce": ce[empirical_arm],
            "coefficient_deletion_recovery": (
                ce["ZERO"] - ce[coefficient_arm]
            ) / deletion_stake,
            "empirical_deletion_recovery": (
                ce["ZERO"] - ce[empirical_arm]
            ) / deletion_stake,
            "coefficient_retention_of_p512": (
                ce["ZERO"] - ce[coefficient_arm]
            ) / p512_stake,
            "empirical_retention_of_p512": (
                ce["ZERO"] - ce[empirical_arm]
            ) / p512_stake,
            "empirical_minus_coefficient_document_ce": bootstrap_mean_interval(
                emp_docs - coefficient_docs, seed=BOOTSTRAP_SEED + rank,
            ),
            "empirical_minus_p512_document_ce": bootstrap_mean_interval(
                emp_docs - p512_docs, seed=BOOTSTRAP_SEED + 100 + rank,
            ),
            "price": control.router_price(rank),
            "factor_sha256": tensor_sha256(empirical[rank][0]),
            "signed_value_sha256": tensor_sha256(empirical[rank][1]),
        }
    rank8 = comparisons["8"]
    rank8_vs_coefficient = rank8["empirical_minus_coefficient_document_ce"]
    rank8_vs_p512 = rank8["empirical_minus_p512_document_ce"]
    gates = {
        "rank8_retention_of_p512_ge_0p98": (
            rank8["empirical_retention_of_p512"] >= 0.98
        ),
        "rank8_empirical_minus_p512_upper_le_0p02": (
            rank8_vs_p512["bootstrap_95_high"] <= 0.02
        ),
        "strong_same_grammar_pass": (
            rank8["empirical_retention_of_p512"] >= 0.98
            and rank8_vs_p512["bootstrap_95_high"] <= 0.02
        ),
        "rank8_empirical_ce_better_than_coefficient_with_95pct_upper_below_zero": (
            rank8_vs_coefficient["bootstrap_95_high"] < 0.0
        ),
        "rank8_empirical_score_mse_better_than_coefficient": (
            empirical_metrics["8"]["relative_score_mse"]
            < coefficient_metrics["8"]["relative_score_mse"]
        ),
        "rank8_empirical_write_mse_better_than_coefficient": (
            empirical_metrics["8"]["relative_frozen_decoder_write_mse"]
            < coefficient_metrics["8"]["relative_frozen_decoder_write_mse"]
        ),
        "rank8_retention_of_p512_lt_0p90": (
            rank8["empirical_retention_of_p512"] < 0.90
        ),
    }
    output = {
        "schema": "mlp1_pregate_empirical_m4_router_discovery_v1",
        "status": "discovery_complete",
        "claim_boundary": (
            "FIT-optimized and already-opened SELECT discovery only; FINAL opened zero, "
            "no promotion, composition, OOD, semantic, extraction, or removal claim."
        ),
        "documents": {"FIT": len(fit_rows), "SELECT": len(select_rows), "FINAL_opened": 0},
        "positions_per_document": control.SCORING.stop - control.SCORING.start,
        "optimization": {
            "objective": "empirical fourth-moment exact-score MSE",
            "ranks": list(RANKS),
            "steps": STEPS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "final_learning_rate": FINAL_LEARNING_RATE,
            "seed": SEED,
            "monitor_positions": MONITOR_POSITIONS,
            "monitor_every": MONITOR_EVERY,
            "no_select_gradient": True,
            "no_ce_gradient": True,
            "normalize_vectors_and_absorb_squared_norm_after_each_step": True,
            "final_qr_small_eigendecomposition_canonicalization": True,
            "analytical_float32_target_relative_mse_vs_deployed_bf16_target": (
                deployed_target_discrepancy
            ),
        },
        "training_curve": training_curve,
        "coefficient_factorization_diagnostics": coefficient_diagnostics,
        "select_route_metrics": {
            "coefficient": coefficient_metrics,
            "empirical": empirical_metrics,
        },
        "physical_ce": ce,
        "comparisons": comparisons,
        "gates": gates,
        "calls": {
            "FIT_capture": fit_capture_calls,
            "SELECT_capture": select_capture_calls,
            "SELECT_score": score_calls,
        },
        "checkpoint": checkpoint.__dict__,
        "runtime_seconds": time.time() - started,
        "parents": {
            "runner_sha256": file_sha256(Path(__file__).resolve()),
            "preregistration_sha256": file_sha256(PREREGISTRATION),
            "p512_bundle_sha256": file_sha256(BUNDLE),
            "p512_result_sha256": file_sha256(P512_RESULT),
            "rows_receipt_sha256": file_sha256(ROWS_RECEIPT),
        },
    }
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
