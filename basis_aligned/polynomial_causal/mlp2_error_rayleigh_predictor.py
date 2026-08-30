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
    if penalty == 0:
        # The unregularized design can be rank-deficient.  ``solve(X.T @ X)`` may
        # return an unstable answer without raising, so use the declared SVD least-
        # squares solution directly.  ``gelsd`` is CPU-only and deterministic here.
        return torch.linalg.lstsq(
            design, target.unsqueeze(1), driver="gelsd",
        ).solution[:, 0]
    gram = design.T @ design
    regularizer = torch.eye(design.shape[1], dtype=torch.float64) * float(penalty)
    regularizer[0, 0] = 0.0
    rhs = design.T @ target
    return torch.linalg.solve(gram + regularizer, rhs)


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


def serialize_fit(value: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical, deployable representation of a DESIGN fit."""
    models = {}
    for name, model in value["models"].items():
        models[name] = {
            "ridge_selected": model["ridge"]["selected"],
            "clustered_lodo_mse": model["ridge"]["clustered_lodo_mse"],
            "mean": model["mean"].clone(), "scale": model["scale"].clone(),
            "coefficients": model["coefficients"].clone(),
            "design_prediction": model["design_prediction"].clone(),
        }
    return {
        "schema": "mlp2_error_rayleigh_v1_design_predictor_bundle",
        "target": value["target"].clone(), "models": models,
        "null_predictions": {
            control: {family: prediction.clone() for family, prediction in families.items()}
            for control, families in value["null_predictions"].items()
        },
        "families": {name: list(features) for name, features in FAMILIES.items()},
        "ridge_grid": list(RIDGE_GRID),
        "unit": "source_document_by_program",
        "program_identity_feature": False,
        "directional_amplitude_reduction": "arithmetic_mean_h16_h8",
    }


def exact_nested_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) \
            and left.dtype == right.dtype and left.shape == right.shape \
            and torch.equal(left, right)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return isinstance(left, Mapping) and isinstance(right, Mapping) \
            and set(left) == set(right) \
            and all(exact_nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) \
            and all(exact_nested_equal(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def validate_frozen_bundle(value: Any) -> Mapping[str, Any]:
    """Validate the exact receipt-bound predictor serialization used to unlock HELDOUT."""
    required = {
        "schema", "target", "models", "null_predictions", "families", "ridge_grid",
        "unit", "program_identity_feature", "directional_amplitude_reduction",
    }
    if not isinstance(value, dict) or set(value) != required \
            or value.get("schema") != "mlp2_error_rayleigh_v1_design_predictor_bundle" \
            or value.get("families") != {
                name: list(features) for name, features in FAMILIES.items()
            } or value.get("ridge_grid") != list(RIDGE_GRID) \
            or value.get("unit") != "source_document_by_program" \
            or value.get("program_identity_feature") is not False \
            or value.get("directional_amplitude_reduction") != "arithmetic_mean_h16_h8" \
            or not isinstance(value.get("target"), torch.Tensor) \
            or value["target"].dtype != torch.float64 or value["target"].shape != (32, 3) \
            or not torch.isfinite(value["target"]).all() \
            or set(value.get("models", {})) != set(FAMILIES) \
            or set(value.get("null_predictions", {})) != {"DERANGED", "COV_RANDOM"}:
        raise RuntimeError("DESIGN predictor bundle metadata changed")
    for family, model in value["models"].items():
        width = len(FAMILIES[family])
        if set(model) != {
            "ridge_selected", "clustered_lodo_mse", "mean", "scale", "coefficients",
            "design_prediction",
        } or model["ridge_selected"] not in RIDGE_GRID \
                or set(model["clustered_lodo_mse"]) != set(RIDGE_GRID) \
                or model["mean"].shape != (width,) or model["scale"].shape != (width,) \
                or model["coefficients"].shape != (width + 1,) \
                or model["design_prediction"].shape != (32, 3) \
                or any(not torch.isfinite(model[key]).all() for key in (
                    "mean", "scale", "coefficients", "design_prediction",
                )) or bool((model["scale"] <= 0).any()):
            raise RuntimeError(f"DESIGN predictor {family} changed")
        if any(not isinstance(loss, (int, float)) or not torch.isfinite(torch.tensor(loss))
               for loss in model["clustered_lodo_mse"].values()):
            raise RuntimeError(f"DESIGN predictor {family} CV loss changed")
    for families in value["null_predictions"].values():
        if set(families) != set(FAMILIES) or any(
            prediction.shape != (32, 3) or prediction.dtype != torch.float64
            or not torch.isfinite(prediction).all() for prediction in families.values()
        ):
            raise RuntimeError("DESIGN null prediction changed")
    return value
