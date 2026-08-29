"""Pure DESIGN-only predictor fitting for the MLP2 error-Rayleigh pilot."""

from __future__ import annotations

from typing import Any, Mapping

import torch

import mlp2_error_rayleigh_collector_core as core


RIDGE_GRID = (0.0, 1e-6, 1e-4, 1e-2, 1.0, 100.0)
FAMILIES = {
    "LOCAL": ("local",),
    "FINAL": ("linear", "qlogit"),
    "FULL": ("linear", "qlogit", "q5", "q6"),
}
EPS = 1e-12


def _indices() -> dict[str, int]:
    return {name: index for index, name in enumerate(core.FEATURE_NAMES)}


def averaged_features(features: torch.Tensor) -> Mapping[str, torch.Tensor]:
    """Return [program, background, control, document] frozen scalar features."""
    expected = (3, 2, 3, 32, len(core.FEATURE_NAMES))
    if not isinstance(features, torch.Tensor) or features.dtype != torch.float64 \
            or tuple(features.shape) != expected or not torch.isfinite(features).all():
        raise ValueError("collector feature tensor changed")
    i = _indices()
    return {
        "local": features[..., i["local_mse"]],
        "linear": 0.5 * (features[..., i["ce_jvp_h16"]]
                         + features[..., i["ce_jvp_h8"]]),
        "qlogit": 0.5 * (features[..., i["qlogit_h16"]]
                         + features[..., i["qlogit_h8"]]),
        "q5": 0.5 * (features[..., i["q5_h16"]]
                     + features[..., i["q5_h8"]]),
        "q6": 0.5 * (features[..., i["q6_h16"]]
                     + features[..., i["q6_h8"]]),
    }


def regression_units(
    features: torch.Tensor, finite: torch.Tensor, control_index: int,
) -> tuple[Mapping[str, torch.Tensor], torch.Tensor]:
    """Build [document, program, feature] contrasts and the finite interaction target."""
    expected = (3, 2, 32, len(core.FINITE_NAMES))
    if not isinstance(finite, torch.Tensor) or finite.dtype != torch.float64 \
            or tuple(finite.shape) != expected or not torch.isfinite(finite).all() \
            or control_index not in range(3):
        raise ValueError("finite collector tensor or control index changed")
    values = averaged_features(features)
    # Stored axes are [program, background, control, document]. Move to the explicit
    # inference order [document, program, feature]. Background 1 is C512, 0 native.
    contrasts = {
        name: (value[:, 1, control_index] - value[:, 0, control_index]).T.contiguous()
        for name, value in values.items()
    }
    injected = core.FINITE_NAMES.index("injected_dce")
    target = (finite[:, 1, :, injected] - finite[:, 0, :, injected]).T.contiguous()
    if any(value.shape != (32, 3) for value in contrasts.values()) \
            or target.shape != (32, 3):
        raise RuntimeError("regression unit layout changed")
    return contrasts, target


def family_matrix(contrasts: Mapping[str, torch.Tensor], family: str) -> torch.Tensor:
    names = FAMILIES.get(family)
    if names is None or any(name not in contrasts for name in names):
        raise ValueError("predictor family changed")
    value = torch.stack([contrasts[name] for name in names], dim=-1)
    if tuple(value.shape[:2]) != (32, 3) or not torch.isfinite(value).all():
        raise RuntimeError("predictor family matrix is malformed")
    return value.double()


def fit_normalizer(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flat = matrix.flatten(0, 1)
    mean = flat.mean(0)
    scale = flat.std(0, unbiased=False)
    if bool((scale <= EPS).any()):
        raise ValueError("DESIGN feature has zero variance")
    return mean, scale


def normalize(matrix: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if matrix.shape[-1] != len(mean) or mean.shape != scale.shape \
            or bool((scale <= EPS).any()):
        raise ValueError("predictor normalizer changed")
    return (matrix - mean) / scale


def ridge_fit(matrix: torch.Tensor, target: torch.Tensor, penalty: float) -> torch.Tensor:
    if matrix.ndim != 2 or target.ndim != 1 or len(matrix) != len(target) \
            or penalty not in RIDGE_GRID:
        raise ValueError("ridge fit inputs changed")
    design = torch.cat((torch.ones(len(matrix), 1, dtype=torch.float64), matrix), dim=1)
    gram = design.T @ design
    regularizer = torch.eye(design.shape[1], dtype=torch.float64) * float(penalty)
    regularizer[0, 0] = 0.0
    rhs = design.T @ target
    try:
        return torch.linalg.solve(gram + regularizer, rhs)
    except torch.linalg.LinAlgError:
        return torch.linalg.lstsq(gram + regularizer, rhs.unsqueeze(1)).solution[:, 0]


def predict(matrix: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    design = torch.cat((torch.ones(*matrix.shape[:-1], 1, dtype=torch.float64), matrix), -1)
    if coefficients.shape != (design.shape[-1],):
        raise ValueError("predictor coefficient shape changed")
    return design @ coefficients


def select_ridge(matrix: torch.Tensor, target: torch.Tensor) -> Mapping[str, Any]:
    if tuple(matrix.shape[:2]) != (32, 3) or target.shape != (32, 3):
        raise ValueError("clustered ridge units changed")
    losses = {}
    for penalty in RIDGE_GRID:
        fold = []
        for document in range(32):
            keep = torch.ones(32, dtype=torch.bool); keep[document] = False
            mean, scale = fit_normalizer(matrix[keep])
            coefficients = ridge_fit(
                normalize(matrix[keep], mean, scale).flatten(0, 1), target[keep].flatten(),
                penalty,
            )
            prediction = predict(normalize(matrix[document], mean, scale), coefficients)
            fold.append((prediction - target[document]).square().mean())
        losses[penalty] = float(torch.stack(fold).mean())
    # Larger penalty wins only an exact floating tie.
    selected = min(RIDGE_GRID, key=lambda penalty: (losses[penalty], -penalty))
    return {"selected": selected, "clustered_lodo_mse": losses}


def fit_family(matrix: torch.Tensor, target: torch.Tensor) -> Mapping[str, Any]:
    selection = select_ridge(matrix, target)
    mean, scale = fit_normalizer(matrix)
    coefficients = ridge_fit(
        normalize(matrix, mean, scale).flatten(0, 1), target.flatten(),
        selection["selected"],
    )
    return {
        "ridge": selection, "mean": mean, "scale": scale,
        "coefficients": coefficients,
        "design_prediction": predict(normalize(matrix, mean, scale), coefficients),
    }


def fit_design(features: torch.Tensor, finite: torch.Tensor) -> Mapping[str, Any]:
    actual, target = regression_units(features, finite, 0)
    models = {}
    null_predictions = {"DERANGED": {}, "COV_RANDOM": {}}
    for family in FAMILIES:
        model = fit_family(family_matrix(actual, family), target)
        models[family] = model
        for control_index, control in enumerate(core.CONTROL_NAMES[1:], start=1):
            null, null_target = regression_units(features, finite, control_index)
            if not torch.equal(null_target, target):
                raise RuntimeError("control changed the finite true-error target")
            matrix = family_matrix(null, family)
            null_predictions[control][family] = predict(
                normalize(matrix, model["mean"], model["scale"]), model["coefficients"],
            )
    return {"target": target, "models": models, "null_predictions": null_predictions}
