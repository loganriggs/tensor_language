"""Fit-only A--E program construction for state-complete compiler v2."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch

import early_mlp_affine_compiler_v1 as affine_v1
import early_mlp_state_complete_compiler_v2 as compiler
import state_complete_compiler_solver_v2 as native_solver


FIT_SEED = 271828
CAUSAL_EPOCHS = 16
CAUSAL_BATCH = 1024
CAUSAL_LEARNING_RATE = 0.003
CAUSAL_GRADIENT_CLIP = 5.0


def document_block_permutation(document_ids: Sequence[str], seed: int) -> torch.Tensor:
    """Permute whole fit-document row blocks only within equal-count strata."""

    groups: dict[str, list[int]] = {}
    for index, document in enumerate(document_ids):
        if not isinstance(document, str) or not document:
            raise ValueError("document IDs must be nonempty strings")
        groups.setdefault(document, []).append(index)
    strata: dict[int, list[str]] = {}
    for document, indices in groups.items():
        strata.setdefault(len(indices), []).append(document)
    generator = torch.Generator().manual_seed(seed)
    output = torch.arange(len(document_ids), dtype=torch.long)
    moved = 0
    for count, documents in sorted(strata.items()):
        documents = sorted(documents)
        if len(documents) < 2:
            continue
        order = torch.randperm(len(documents), generator=generator).tolist()
        if order == list(range(len(documents))):
            order = order[1:] + order[:1]
        for target_index, source_index in enumerate(order):
            target_rows = groups[documents[target_index]]
            source_rows = groups[documents[source_index]]
            output[torch.tensor(target_rows)] = torch.tensor(source_rows)
            moved += sum(a != b for a, b in zip(target_rows, source_rows, strict=True))
    if moved == 0 or sorted(output.tolist()) != list(range(len(document_ids))):
        raise RuntimeError("fit document-block permutation is degenerate")
    return output


def expand_capture_permutation(row_permutation: torch.Tensor) -> torch.Tensor:
    offsets = torch.arange(64, dtype=torch.long)
    return (row_permutation[:, None] * 64 + offsets[None, :]).reshape(-1)


def clip_fit_adjoints(adjoint: torch.Tensor, quantile: float = 0.99) -> tuple[torch.Tensor, float]:
    adjoint = adjoint.double()
    if adjoint.ndim != 2 or not torch.isfinite(adjoint).all():
        raise ValueError("fit adjoints must be a finite matrix")
    if not 0.0 < quantile <= 1.0:
        raise ValueError("adjoint clip quantile must lie in (0,1]")
    norms = adjoint.norm(dim=1)
    threshold = float(torch.quantile(norms, quantile))
    if threshold <= 0.0:
        raise ValueError("fit adjoints have zero clipping threshold")
    scale = (threshold / norms.clamp_min(1e-30)).clamp_max(1.0)
    return adjoint * scale[:, None], threshold


def _normalized_fit(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = x.double()
    mean = x.mean(dim=0)
    centered = x - mean
    scale = centered.square().mean(dim=0).sqrt().clamp_min(1e-6)
    return centered / scale, mean, scale


def euclidean_affine_states(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    lambdas: Sequence[float] = affine_v1.LAMBDA_GRID,
    ranks: Sequence[int] = affine_v1.RANK_GRID,
    interface: str,
    family: str,
) -> dict[tuple[float, int], dict[str, Any]]:
    """Construct every analytic ridge/rank state without validation selection."""

    x, mean, scale = _normalized_fit(train_x)
    y = train_y.double()
    if y.ndim != 2 or x.shape[0] != y.shape[0] or not torch.isfinite(y).all():
        raise ValueError("affine fit target must align and be finite")
    bias = y.mean(dim=0)
    centered_y = y - bias
    gram = x.T @ x / x.shape[0]
    cross = x.T @ centered_y / x.shape[0]
    eigvals, eigvecs = torch.linalg.eigh(gram)
    states = {}
    for ridge in lambdas:
        full = affine_v1._ridge_weight(eigvals, eigvecs, cross, float(ridge))
        u, singular, vh = torch.linalg.svd(full, full_matrices=False)
        for rank in ranks:
            root = singular[:rank].clamp_min(0.0).sqrt()
            left, right = affine_v1._canonicalize_svd_signs(
                u[:, :rank] * root, root[:, None] * vh[:rank]
            )
            states[(float(ridge), int(rank))] = {
                "grammar": "affine", "interface": interface, "family": family,
                "mean": mean.float().cpu(), "scale": scale.float().cpu(),
                "bias": bias.float().cpu(), "left": left.float().cpu(),
                "right": right.float().cpu(), "lambda": float(ridge),
                "rank": int(rank),
            }
    return states


def _affine_predict_device(
    normalized_x: torch.Tensor, left: torch.Tensor, right: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return (normalized_x @ left) @ right + bias


def causal_affine_states(
    train_x: torch.Tensor,
    train_p: torch.Tensor,
    train_adjoint: torch.Tensor,
    *,
    lambdas: Sequence[float] = affine_v1.LAMBDA_GRID,
    ranks: Sequence[int] = affine_v1.RANK_GRID,
    epochs: int = CAUSAL_EPOCHS,
    token_batch: int = CAUSAL_BATCH,
    learning_rate: float = CAUSAL_LEARNING_RATE,
    seed: int = FIT_SEED,
    device: str | torch.device = "cpu",
) -> tuple[dict[tuple[float, int], dict[str, Any]], list[dict[str, Any]]]:
    """Fit C with deterministic low-rank AdamW under the fit-only causal loss."""

    if epochs <= 0 or token_batch <= 0 or learning_rate <= 0.0:
        raise ValueError("causal affine optimizer settings must be positive")
    normalized, mean, scale = _normalized_fit(train_x)
    p = train_p.double()
    g = train_adjoint.double()
    if p.shape != g.shape or normalized.shape[0] != p.shape[0]:
        raise ValueError("causal affine fit arrays do not align")
    initial = euclidean_affine_states(
        train_x, train_p, lambdas=lambdas, ranks=ranks,
        interface="state_complete_p", family="C_state_complete_affine_causal",
    )
    x_device, p_device, g_device = (
        normalized.to(device), p.to(device), g.to(device)
    )
    generator = torch.Generator().manual_seed(seed)
    output: dict[tuple[float, int], dict[str, Any]] = {}
    diagnostics: list[dict[str, Any]] = []
    for ridge in lambdas:
        for rank in ranks:
            start = initial[(float(ridge), int(rank))]
            left = start["left"].double().to(device).requires_grad_(True)
            right = start["right"].double().to(device).requires_grad_(True)
            bias = start["bias"].double().to(device).requires_grad_(True)
            optimizer = torch.optim.AdamW(
                [left, right, bias], lr=learning_rate, weight_decay=float(ridge)
            )
            with torch.no_grad():
                initial_error = _affine_predict_device(
                    x_device, left, right, bias
                ) - p_device
                initial_loss = float(compiler.empirical_fisher_loss(initial_error, g_device))
            curve = [initial_loss]
            best_loss = initial_loss
            best_epoch = 0
            best = (left.detach().clone(), right.detach().clone(), bias.detach().clone())
            for epoch in range(epochs):
                order = torch.randperm(x_device.shape[0], generator=generator)
                epoch_numerator = 0.0
                epoch_count = 0
                for start_index in range(0, len(order), token_batch):
                    index = order[start_index:start_index + token_batch].to(device)
                    prediction = _affine_predict_device(
                        x_device.index_select(0, index), left, right, bias
                    )
                    loss = compiler.empirical_fisher_loss(
                        prediction - p_device.index_select(0, index),
                        g_device.index_select(0, index),
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([left, right, bias], CAUSAL_GRADIENT_CLIP)
                    optimizer.step()
                    epoch_numerator += float(loss.detach()) * index.numel()
                    epoch_count += index.numel()
                with torch.no_grad():
                    full_error = _affine_predict_device(
                        x_device, left, right, bias
                    ) - p_device
                    full_loss = float(compiler.empirical_fisher_loss(full_error, g_device))
                curve.append(full_loss)
                if full_loss < best_loss:
                    best_loss = full_loss
                    best_epoch = epoch + 1
                    best = (
                        left.detach().clone(), right.detach().clone(), bias.detach().clone()
                    )
            best_left, best_right, best_bias = best
            state = {
                "grammar": "affine", "interface": "state_complete_p",
                "family": "C_state_complete_affine_causal",
                "mean": mean.float().cpu(), "scale": scale.float().cpu(),
                "bias": best_bias.float().cpu(),
                "left": best_left.float().cpu(),
                "right": best_right.float().cpu(),
                "lambda": float(ridge), "rank": int(rank),
                "selected_fit_epoch": int(best_epoch),
            }
            output[(float(ridge), int(rank))] = state
            diagnostics.append({"lambda": float(ridge), "rank": int(rank),
                                "full_fit_loss_initial_then_epochs": curve,
                                "selected_fit_epoch": int(best_epoch),
                                "selected_fit_loss": best_loss})
    return output, diagnostics


def native_features(
    z: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    token_batch: int = 1024,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Compute fit-only phi without retaining a graph or checkpoint pointer."""

    if z.ndim != 2 or left.ndim != 2 or right.shape != left.shape:
        raise ValueError("native feature dimensions are invalid")
    if z.shape[1] != left.shape[1] or token_batch <= 0:
        raise ValueError("native feature input dimensions or batch are invalid")
    left_device = left.float().to(device)
    right_device = right.float().to(device)
    parts = []
    with torch.no_grad():
        for start in range(0, z.shape[0], token_batch):
            batch = z[start:start + token_batch].float().to(device)
            part = (batch @ left_device.T) * (batch @ right_device.T)
            parts.append(part.detach().cpu().contiguous())
    return torch.cat(parts)


def native_states(
    phi: torch.Tensor,
    target_p: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    projected_decoder: torch.Tensor,
    *,
    adjoint: torch.Tensor | None,
    family: str,
    k_grid: Sequence[int] = compiler.NATIVE_K_GRID,
    device: str | torch.device = "cpu",
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Fit D or E from exact sufficient statistics and serialize standalone terms."""

    hessian, linear, intercept, offset = native_solver.native_quadratic_statistics(
        phi.to(device), projected_decoder.to(device), target_p.to(device),
        adjoint=None if adjoint is None else adjoint.to(device),
    )
    path = native_solver.fista_l1_path(hessian, linear)
    frontier = native_solver.select_refit_frontier(hessian, linear, path, k_grid)
    states = {}
    for k, selected in frontier.items():
        support = selected["support"].detach().cpu().long()
        amplitudes = selected["amplitudes"].detach().cpu().double()
        q_selected = projected_decoder.double().index_select(0, support)
        q_selected = q_selected * amplitudes[:, None]
        selected_left, selected_right, selected_q = compiler.canonicalize_native_terms(
            left.double().index_select(0, support),
            right.double().index_select(0, support), q_selected,
        )
        beta = native_solver.materialize_native_intercept(
            intercept.detach().cpu(), offset.detach().cpu(), projected_decoder,
            support, amplitudes,
        )
        states[int(k)] = {
            "grammar": "native", "interface": "state_complete_p",
            "family": family, "k": int(k),
            "left": selected_left.float().cpu(),
            "right": selected_right.float().cpu(),
            "projected_decoder": selected_q.float().cpu(),
            "beta": beta.float().cpu(), "indices": support,
            "source_lambda_ratio": selected["source_lambda_ratio"],
            "fit_smooth_objective_without_constant": (
                selected["fit_smooth_objective_without_constant"]
            ),
        }
    diagnostics = {
        "path": [{key: value for key, value in row.items() if key != "gates"}
                 for row in path],
        "feature_shape": list(phi.shape),
        "objective": "causal" if adjoint is not None else "euclidean",
    }
    return states, diagnostics


def constant_state(target_p: torch.Tensor) -> dict[str, Any]:
    if target_p.ndim != 2 or not torch.isfinite(target_p).all():
        raise ValueError("constant target must be a finite matrix")
    return {"grammar": "constant", "interface": "state_complete_p",
            "family": "fit_mean_control", "bias": target_p.float().mean(dim=0).cpu()}
