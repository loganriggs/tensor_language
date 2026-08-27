"""Pure fitting and causal-scoring contract for affine compiler v1."""

from __future__ import annotations

from itertools import product
import math
from typing import Any, Mapping, Sequence

import torch


D_MODEL = 1152
COEFFICIENT_DIM = 64
LAMBDA_GRID = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
RANK_GRID = (8, 16, 32, 64)
SELECTION_SLACK = 1.01
FIDELITY_FRACTION = 0.50
ARM_STATES = tuple(product(("N", "Q", "O"), ("N", "Q", "O"), ("N", "E")))
BASELINE_ARM = ("N", "N", "N")


def arm_name(arm: tuple[str, str, str]) -> str:
    if arm not in ARM_STATES:
        raise ValueError(f"unregistered compiler arm: {arm}")
    return "".join(arm)


def _canonicalize_svd_signs(
    left: torch.Tensor, right: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    left = left.clone()
    right = right.clone()
    for column in range(left.shape[1]):
        pivot = int(left[:, column].abs().argmax())
        if float(left[pivot, column]) < 0.0:
            left[:, column].neg_()
            right[column].neg_()
    return left, right


def balanced_factors(
    weight: torch.Tensor, rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic balanced rank-r factors of an input-by-output map."""

    if weight.ndim != 2 or not torch.isfinite(weight).all():
        raise ValueError("weight must be a finite matrix")
    if not 1 <= rank <= min(weight.shape):
        raise ValueError("rank is outside the matrix dimensions")
    u, singular, vh = torch.linalg.svd(weight.double(), full_matrices=False)
    root = singular[:rank].clamp_min(0.0).sqrt()
    left = u[:, :rank] * root
    right = root[:, None] * vh[:rank]
    return _canonicalize_svd_signs(left, right)


def _ridge_weight(
    eigvals: torch.Tensor,
    eigvecs: torch.Tensor,
    cross: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    if ridge < 0.0 or not math.isfinite(ridge):
        raise ValueError("ridge must be finite and nonnegative")
    rotated = eigvecs.T @ cross
    if ridge == 0.0:
        cutoff = max(float(eigvals.max()), 1.0) * 1e-12
        inverse = torch.where(eigvals > cutoff, eigvals.reciprocal(), 0.0)
    else:
        inverse = (eigvals + ridge).reciprocal()
    return eigvecs @ (inverse[:, None] * rotated)


def affine_predict(x: torch.Tensor, state: Mapping[str, Any]) -> torch.Tensor:
    required = ("mean", "scale", "bias", "left", "right")
    if any(key not in state for key in required):
        raise ValueError("affine state is incomplete")
    normalized = (x.double() - state["mean"].double()) / state["scale"].double()
    return (normalized @ state["left"].double()) @ state["right"].double() + state[
        "bias"
    ].double()


def transport_output_gauge(
    state: Mapping[str, Any], basis: torch.Tensor, rotation: torch.Tensor
) -> tuple[dict[str, Any], torch.Tensor]:
    """Transport a predictor through an orthogonal output-basis gauge."""

    basis = basis.double()
    rotation = rotation.double()
    if basis.ndim != 2 or rotation.shape != (basis.shape[1], basis.shape[1]):
        raise ValueError("basis and rotation dimensions do not align")
    identity = torch.eye(rotation.shape[0], dtype=torch.float64, device=rotation.device)
    if not torch.allclose(rotation.T @ rotation, identity, atol=2e-6, rtol=2e-6):
        raise ValueError("output gauge must be orthogonal")
    transported = dict(state)
    transported["right"] = state["right"].double() @ rotation
    transported["bias"] = state["bias"].double() @ rotation
    return transported, basis @ rotation


def affine_program_price(rank: int, *, include_basis: bool) -> dict[str, Any]:
    """Registered real/FLOP price for one executable site."""

    if rank not in RANK_GRID:
        raise ValueError("rank is outside the registered frontier")
    basis_reals = D_MODEL * COEFFICIENT_DIM if include_basis else 0
    predictor_reals = (
        2 * D_MODEL + COEFFICIENT_DIM
        + D_MODEL * rank + rank * COEFFICIENT_DIM
    )
    original_reals = 3 * (4 * D_MODEL * D_MODEL) + D_MODEL
    inference_multiplies = (
        D_MODEL * rank + rank * COEFFICIENT_DIM
        + COEFFICIENT_DIM * D_MODEL
    )
    return {
        "rank": rank,
        "include_basis": include_basis,
        "basis_reals": basis_reals,
        "predictor_reals": predictor_reals,
        "total_reals": basis_reals + predictor_reals,
        "float32_bits": 32 * (basis_reals + predictor_reals),
        "counterfactual_float16_bits": 16 * (basis_reals + predictor_reals),
        "inference_multiplies_per_token": inference_multiplies,
        "native_hadamard_products_per_token": 0,
        "original_mlp_reals": original_reals,
        "fraction_of_original_reals": (basis_reals + predictor_reals) / original_reals,
        "original_native_hadamard_products_per_token": 4 * D_MODEL,
    }


def coefficient_metrics(
    prediction: torch.Tensor, target: torch.Tensor
) -> dict[str, float]:
    prediction = prediction.double()
    target = target.double()
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must be aligned matrices")
    error = prediction - target
    mse = float(error.square().mean())
    energy = float(target.square().mean())
    centered = target - target.mean(dim=0)
    centered_energy = float(centered.square().mean())
    dot = (prediction * target).sum(dim=1)
    denominator = prediction.norm(dim=1) * target.norm(dim=1)
    cosine = torch.where(denominator > 0.0, dot / denominator, 0.0)
    return {
        "mse": mse,
        "target_energy": energy,
        "nmse": mse / max(energy, 1e-30),
        "r2_centered": 1.0 - mse / max(centered_energy, 1e-30),
        "mean_row_cosine": float(cosine.mean()),
        # B has orthonormal columns, so ||(c_hat-c)B^T||_F^2 equals the
        # coefficient squared norm.  ``mse`` averages over 64 coefficients;
        # physical RMS averages the same total error over 1152 stream channels.
        "physical_rms": math.sqrt(mse * target.shape[1] / D_MODEL),
    }


def fit_ridge_frontier(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    validation_x: torch.Tensor,
    validation_y: torch.Tensor,
    *,
    lambdas: Sequence[float] = LAMBDA_GRID,
    ranks: Sequence[int] = RANK_GRID,
    selection_slack: float = SELECTION_SLACK,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fit the preregistered ridge/SVD frontier and select by validation NMSE."""

    train_x = train_x.double()
    train_y = train_y.double()
    validation_x = validation_x.double()
    validation_y = validation_y.double()
    if train_x.ndim != 2 or train_y.ndim != 2:
        raise ValueError("training arrays must be matrices")
    if validation_x.ndim != 2 or validation_y.ndim != 2:
        raise ValueError("validation arrays must be matrices")
    if train_x.shape[0] != train_y.shape[0] or validation_x.shape[0] != validation_y.shape[0]:
        raise ValueError("feature and target row counts differ")
    if train_x.shape[1] != validation_x.shape[1]:
        raise ValueError("train/validation feature dimensions differ")
    if train_y.shape[1] != validation_y.shape[1]:
        raise ValueError("train/validation target dimensions differ")
    if not all(torch.isfinite(value).all() for value in (
        train_x, train_y, validation_x, validation_y
    )):
        raise ValueError("fit arrays must be finite")
    if selection_slack < 1.0:
        raise ValueError("selection slack must be at least one")

    mean = train_x.mean(dim=0)
    centered_x = train_x - mean
    scale = centered_x.square().mean(dim=0).sqrt().clamp_min(1e-6)
    normalized_x = centered_x / scale
    bias = train_y.mean(dim=0)
    centered_y = train_y - bias
    n = train_x.shape[0]
    gram = normalized_x.T @ normalized_x / n
    cross = normalized_x.T @ centered_y / n
    eigvals, eigvecs = torch.linalg.eigh(gram)

    frontier: list[dict[str, Any]] = []
    states: dict[tuple[float, int], dict[str, Any]] = {}
    for ridge in lambdas:
        full_weight = _ridge_weight(eigvals, eigvecs, cross, float(ridge))
        u, singular, vh = torch.linalg.svd(full_weight, full_matrices=False)
        for rank in ranks:
            if not 1 <= rank <= min(full_weight.shape):
                raise ValueError(f"invalid registered rank: {rank}")
            root = singular[:rank].clamp_min(0.0).sqrt()
            left, right = _canonicalize_svd_signs(
                u[:, :rank] * root, root[:, None] * vh[:rank]
            )
            state = {
                "mean": mean,
                "scale": scale,
                "bias": bias,
                "left": left,
                "right": right,
                "rank": int(rank),
                "lambda": float(ridge),
            }
            metrics = coefficient_metrics(affine_predict(validation_x, state), validation_y)
            standalone_reals_without_basis = (
                2 * train_x.shape[1] + train_y.shape[1]
                + train_x.shape[1] * rank + rank * train_y.shape[1]
            )
            row = {
                "lambda": float(ridge),
                "rank": int(rank),
                "standalone_reals_without_basis": int(standalone_reals_without_basis),
                "validation": metrics,
            }
            frontier.append(row)
            states[(float(ridge), int(rank))] = state

    best_nmse = min(row["validation"]["nmse"] for row in frontier)
    eligible = [
        row for row in frontier
        if row["validation"]["nmse"] <= selection_slack * best_nmse
    ]
    chosen = min(
        eligible,
        key=lambda row: (
            row["standalone_reals_without_basis"], row["rank"],
            -row["lambda"], row["validation"]["nmse"],
        ),
    )
    state = states[(chosen["lambda"], chosen["rank"])]
    state = {
        **state,
        "selection_slack": float(selection_slack),
        "best_validation_nmse": float(best_nmse),
        "selected_validation": dict(chosen["validation"]),
        "standalone_reals_without_basis": chosen["standalone_reals_without_basis"],
    }
    serial_frontier = sorted(frontier, key=lambda row: (row["lambda"], row["rank"]))
    return state, serial_frontier


def _ci95(values: torch.Tensor) -> list[float]:
    return [float(torch.quantile(values, 0.025)), float(torch.quantile(values, 0.975))]


def _summary(point: float, samples: torch.Tensor) -> dict[str, Any]:
    return {
        "point_estimate": float(point),
        "bootstrap_mean": float(samples.mean()),
        "ci95": _ci95(samples),
    }


def compiler_lattice_analysis(
    row_ce_by_arm: Mapping[tuple[str, str, str], Sequence[float]],
    document_ids: Sequence[str],
    *,
    mean_control_rows: Sequence[float],
    shuffle_control_rows: Sequence[float],
    draws: int = 2000,
    seed: int = 31415926,
) -> dict[str, Any]:
    """Shared row-weighted document-cluster bootstrap for every registered gate."""

    if set(row_ce_by_arm) != set(ARM_STATES):
        raise ValueError("compiler bootstrap requires the complete 18-arm lattice")
    values = torch.tensor(
        [[float(value) for value in row_ce_by_arm[arm]] for arm in ARM_STATES],
        dtype=torch.float64,
    )
    controls = torch.tensor(
        [[float(value) for value in mean_control_rows],
         [float(value) for value in shuffle_control_rows]], dtype=torch.float64,
    )
    if values.ndim != 2 or values.shape[1] <= 0 or controls.shape != (2, values.shape[1]):
        raise ValueError("compiler bootstrap rows are not paired")
    if not torch.isfinite(values).all() or not torch.isfinite(controls).all():
        raise ValueError("compiler bootstrap values must be finite")
    if len(document_ids) != values.shape[1] or not all(
        isinstance(document, str) and document for document in document_ids
    ):
        raise ValueError("document IDs must align with compiler rows")
    unique_documents = list(dict.fromkeys(document_ids))
    if len(unique_documents) < 2 or draws < 2:
        raise ValueError("compiler bootstrap needs at least two documents and draws")

    all_values = torch.cat([values, controls], dim=0)
    baseline = values[ARM_STATES.index(BASELINE_ARM)]
    gain_rows = baseline.unsqueeze(0) - all_values
    point_gain = gain_rows.mean(dim=1)
    document_index = {document: index for index, document in enumerate(unique_documents)}
    cluster = torch.tensor([document_index[row] for row in document_ids], dtype=torch.long)
    document_sums = torch.zeros(
        all_values.shape[0], len(unique_documents), dtype=torch.float64
    )
    document_sums.index_add_(1, cluster, gain_rows)
    counts = torch.bincount(cluster, minlength=len(unique_documents)).double()
    generator = torch.Generator().manual_seed(seed)
    sampled = torch.randint(
        len(unique_documents), (draws, len(unique_documents)), generator=generator
    )
    boot_gain = document_sums[:, sampled].sum(dim=2).T / counts[sampled].sum(dim=1)[:, None]
    arm_index = {arm: index for index, arm in enumerate(ARM_STATES)}

    def point(arm: tuple[str, str, str]) -> torch.Tensor:
        return point_gain[arm_index[arm]]

    def boot(arm: tuple[str, str, str]) -> torch.Tensor:
        return boot_gain[:, arm_index[arm]]

    def contrast(candidate, background) -> tuple[float, torch.Tensor]:
        return float(point(candidate) - point(background)), boot(candidate) - boot(background)

    core: dict[str, Any] = {}
    reuse: dict[str, Any] = {}
    fidelity: dict[str, Any] = {}
    for site in (0, 1):
        neighbors = ("N", "Q")
        for neighbor in neighbors:
            for mlp2 in ("N", "E"):
                if site == 0:
                    candidate = ("Q", neighbor, mlp2)
                    background = ("N", neighbor, mlp2)
                    oracle = ("O", neighbor, mlp2)
                else:
                    candidate = (neighbor, "Q", mlp2)
                    background = (neighbor, "N", mlp2)
                    oracle = (neighbor, "O", mlp2)
                name = f"mlp{site}_neighbor_{neighbor}_mlp2_{mlp2}"
                candidate_point, candidate_boot = contrast(candidate, background)
                oracle_point, oracle_boot = contrast(oracle, background)
                margin_point = candidate_point - FIDELITY_FRACTION * oracle_point
                margin_boot = candidate_boot - FIDELITY_FRACTION * oracle_boot
                core[name] = _summary(candidate_point, candidate_boot)
                fidelity[name] = {
                    **_summary(margin_point, margin_boot),
                    "predicted_effect": candidate_point,
                    "oracle_effect": oracle_point,
                    "descriptive_ratio_if_identified": (
                        candidate_point / oracle_point if oracle_point > 0.0 else None
                    ),
                }
        for mlp2 in ("N", "E"):
            if site == 0:
                candidate, background = ("Q", "O", mlp2), ("N", "O", mlp2)
            else:
                candidate, background = ("O", "Q", mlp2), ("O", "N", mlp2)
            estimate, samples = contrast(candidate, background)
            reuse[f"mlp{site}_oracle_neighbor_mlp2_{mlp2}"] = _summary(estimate, samples)

    qnn, nqn = point(("Q", "N", "N")), point(("N", "Q", "N"))
    composition_n_samples = boot(("Q", "Q", "N")) - torch.maximum(
        boot(("Q", "N", "N")), boot(("N", "Q", "N"))
    )
    composition_n_point = float(point(("Q", "Q", "N")) - torch.maximum(qnn, nqn))
    e_points = torch.stack([
        point(("Q", "N", "E")), point(("N", "Q", "E")), point(("N", "N", "E"))
    ])
    e_samples = torch.stack([
        boot(("Q", "N", "E")), boot(("N", "Q", "E")), boot(("N", "N", "E"))
    ], dim=1)
    composition_e_point = float(point(("Q", "Q", "E")) - e_points.max())
    composition_e_samples = boot(("Q", "Q", "E")) - e_samples.max(dim=1).values

    joint_n_point = float(
        point(("Q", "Q", "N")) - FIDELITY_FRACTION * point(("O", "O", "N"))
    )
    joint_n_samples = (
        boot(("Q", "Q", "N")) - FIDELITY_FRACTION * boot(("O", "O", "N"))
    )
    joint_e_point = float(
        (point(("Q", "Q", "E")) - point(("N", "N", "E")))
        - FIDELITY_FRACTION
        * (point(("O", "O", "E")) - point(("N", "N", "E")))
    )
    joint_e_samples = (
        (boot(("Q", "Q", "E")) - boot(("N", "N", "E")))
        - FIDELITY_FRACTION
        * (boot(("O", "O", "E")) - boot(("N", "N", "E")))
    )
    mean_index, shuffle_index = len(ARM_STATES), len(ARM_STATES) + 1
    q_gain_point = point(("Q", "Q", "N"))
    q_gain_boot = boot(("Q", "Q", "N"))
    controls_out = {
        "QQN_beats_mean": _summary(
            float(q_gain_point - point_gain[mean_index]),
            q_gain_boot - boot_gain[:, mean_index],
        ),
        "QQN_beats_shuffle": _summary(
            float(q_gain_point - point_gain[shuffle_index]),
            q_gain_boot - boot_gain[:, shuffle_index],
        ),
    }
    downstream_point, downstream_samples = contrast(("Q", "Q", "E"), ("Q", "Q", "N"))
    output = {
        "core_no_free_rider": core,
        "oracle_neighbor_reuse": reuse,
        "same_background_fidelity": fidelity,
        "composition": {
            "QQN_minus_best_singleton": _summary(composition_n_point, composition_n_samples),
            "QQE_minus_best_fixed_background": _summary(
                composition_e_point, composition_e_samples
            ),
        },
        "joint_fidelity": {
            "QQN_half_OON_margin": _summary(joint_n_point, joint_n_samples),
            "QQE_half_OOE_fixed_E_margin": _summary(joint_e_point, joint_e_samples),
        },
        "controls": controls_out,
        "mlp2_compatibility_diagnostic": _summary(downstream_point, downstream_samples),
    }
    positive = lambda row: row["point_estimate"] > 0.0 and row["ci95"][0] > 0.0
    oracle_positive = all(row["oracle_effect"] > 0.0 for row in fidelity.values())
    output["decisions"] = {
        "core_no_free_rider": all(positive(row) for row in core.values()),
        "oracle_neighbor_reuse": all(positive(row) for row in reuse.values()),
        "same_background_fidelity": oracle_positive and all(
            positive(row) for row in fidelity.values()
        ),
        "composition": all(positive(row) for row in output["composition"].values()),
        "joint_fidelity": all(positive(row) for row in output["joint_fidelity"].values()),
        "controls": all(positive(row) for row in controls_out.values()),
    }
    output["decisions"]["all_statistical_gates"] = all(output["decisions"].values())
    return output
