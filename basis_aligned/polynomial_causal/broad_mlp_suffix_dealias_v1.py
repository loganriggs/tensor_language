"""Pure registry and scorer math for broad-MLP suffix de-alias v1.

No artifact, model, CUDA, or git I/O is permitted here.  The module implements the
frozen 8-cell mask registry, exact factorial contrasts, document bootstrap, and
registered gates from BROAD_MLP_SUFFIX_DEALIAS_V1_PREREGISTRATION.md.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np

import early_mlp_context_cross_v1 as parent


SCHEMA_VERSION = 1
ROLE_NAMES = ("skip7000", "skip11000")
PREFIX_MASKS = parent.PREFIX_MASKS
MLP_SUFFIX = tuple(("mlp", layer) for layer in range(3, 9))
REQUEST_MASKS = tuple((*prefix, *MLP_SUFFIX) for prefix in PREFIX_MASKS)
CELL_COUNT = 8
BOOTSTRAP_DRAWS = 2_000
BOOTSTRAP_SEEDS = {"skip7000": 2026082903, "skip11000": 2026082904}
SINGLETON_ROWS = (1, 3, 5)
PAIR_ROWS = (2, 6, 7)
TRIPLE_ROWS = (4,)
NONEMPTY_ROWS = tuple(range(1, 8))


def logical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def validate_registry() -> None:
    if len(PREFIX_MASKS) != CELL_COUNT or len(set(PREFIX_MASKS)) != CELL_COUNT or (
        MLP_SUFFIX != tuple(("mlp", layer) for layer in range(3, 9))
    ) or any(len(mask) != len(set(mask)) for mask in REQUEST_MASKS) or any(
        kind == "attn" for mask in REQUEST_MASKS for kind, _layer in mask
    ) or any(
        set(mask) != set(PREFIX_MASKS[ordinal]) | set(MLP_SUFFIX)
        for ordinal, mask in enumerate(REQUEST_MASKS)
    ):
        raise RuntimeError("broad-MLP suffix registry changed")


validate_registry()


@dataclass(frozen=True, slots=True)
class RoleArrays:
    """Per-document CE or correct-count sufficient statistics on common support."""

    role: str
    e: np.ndarray
    a: np.ndarray
    m: np.ndarray
    am: np.ndarray
    token_count: np.ndarray

    def __post_init__(self) -> None:
        arrays = tuple(np.asarray(getattr(self, name)) for name in ("e", "a", "m", "am"))
        tokens = np.asarray(self.token_count)
        if self.role not in ROLE_NAMES or not arrays or any(
            value.ndim != 2 or value.shape[1] != CELL_COUNT for value in arrays
        ) or len({value.shape for value in arrays}) != 1 or tokens.shape != (
            arrays[0].shape[0],
        ) or arrays[0].shape[0] == 0 or any(
            not np.all(np.isfinite(value)) for value in arrays
        ) or not np.all(np.isfinite(tokens)) or np.any(tokens <= 0):
            raise ValueError("role sufficient statistics are malformed")
        for name, value in zip(("e", "a", "m", "am"), arrays, strict=True):
            object.__setattr__(self, name, value.astype(np.float64, copy=True))
        object.__setattr__(self, "token_count", tokens.astype(np.float64, copy=True))

    @property
    def document_count(self) -> int:
        return len(self.token_count)


def aggregate(data: RoleArrays, multiplicity: np.ndarray | None = None) -> dict[str, np.ndarray]:
    if multiplicity is None:
        multiplicity = np.ones(data.document_count, dtype=np.float64)
    multiplicity = np.asarray(multiplicity, dtype=np.float64)
    if multiplicity.shape != (data.document_count,) or np.any(multiplicity < 0) or (
        not np.all(np.isfinite(multiplicity))
    ) or float(multiplicity.sum()) <= 0:
        raise ValueError("document multiplicity is malformed")
    denominator = float(multiplicity @ data.token_count)
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("aggregate token denominator is zero or nonfinite")
    return {
        name: multiplicity @ getattr(data, name) / denominator
        for name in ("e", "a", "m", "am")
    }


def contrasts(cost: dict[str, np.ndarray]) -> dict[str, np.ndarray | float]:
    if set(cost) != {"e", "a", "m", "am"} or any(
        np.asarray(value).shape != (CELL_COUNT,) or not np.all(np.isfinite(value))
        for value in cost.values()
    ):
        raise ValueError("cost grid is malformed")
    e, a, m, am = (np.asarray(cost[name], dtype=np.float64) for name in ("e", "a", "m", "am"))

    def early_suffix_interaction(suffix: np.ndarray) -> np.ndarray:
        return suffix - e - suffix[0] + e[0]

    d_a = early_suffix_interaction(a)
    d_m = early_suffix_interaction(m)
    d_am = early_suffix_interaction(am)
    prediction = d_am - d_a
    r = am - a - m + e
    q = r - r[0]
    if not np.allclose(q, d_am - d_a - d_m, atol=1e-12, rtol=1e-12):
        raise RuntimeError("factorial identity failed")
    return {
        "d_a": d_a,
        "d_m": d_m,
        "d_am": d_am,
        "prediction": prediction,
        "r": r,
        "q": q,
        "standalone_m_marginal": float(m[0] - e[0]),
    }


def descriptive_cosine(left: np.ndarray, right: np.ndarray) -> float | str:
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0:
        return "undefined_zero_norm"
    value = float(left @ right / denominator)
    return value if np.isfinite(value) else "undefined_nonfinite"


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(truth - prediction))))


def decision_metrics(contrast: dict[str, np.ndarray | float]) -> dict[str, Any]:
    truth = np.asarray(contrast["d_m"], dtype=np.float64)[1:]
    prediction = np.asarray(contrast["prediction"], dtype=np.float64)[1:]
    error = prediction - truth
    zero_rmse = float(np.sqrt(np.mean(np.square(truth))))
    denominator_r2 = float(np.sum(np.square(truth - np.mean(truth))))
    rmse = _rmse(truth, prediction)
    nre = rmse / zero_rmse if zero_rmse > 0 else float("nan")
    r2 = 1.0 - float(np.sum(np.square(error))) / denominator_r2 if denominator_r2 > 0 else float("nan")
    sign_agreement = int(np.sum(
        ((np.sign(prediction) == np.sign(truth)) & ((prediction != 0) | (truth == 0)))
    ))
    subgroup = {}
    for name, rows in (
        ("singletons", SINGLETON_ROWS), ("pairs", PAIR_ROWS), ("triple", TRIPLE_ROWS),
    ):
        indices = np.asarray([row - 1 for row in rows], dtype=np.int64)
        subgroup_truth, subgroup_prediction = truth[indices], prediction[indices]
        subgroup_zero = float(np.sqrt(np.mean(np.square(subgroup_truth))))
        subgroup_rmse = _rmse(subgroup_truth, subgroup_prediction)
        subgroup[name] = {
            "rmse": subgroup_rmse,
            "zero_rmse": subgroup_zero,
            "nre": subgroup_rmse / subgroup_zero if subgroup_zero > 0 else float("nan"),
            "no_worse_than_zero": bool(subgroup_rmse <= subgroup_zero),
        }
    d_m = np.asarray(contrast["d_m"], dtype=np.float64)
    r = np.asarray(contrast["r"], dtype=np.float64)
    q = np.asarray(contrast["q"], dtype=np.float64)
    r_nonempty = r[1:]
    return {
        "rmse": rmse,
        "zero_rmse": zero_rmse,
        "nre": nre,
        "r2": r2,
        "max_abs_error": float(np.max(np.abs(error))),
        "max_abs_truth": float(np.max(np.abs(truth))),
        "sign_agreement": sign_agreement,
        "subgroups": subgroup,
        "norms": {
            "d_m": float(np.linalg.norm(d_m[1:])),
            "r": float(np.linalg.norm(r_nonempty)),
            "q": float(np.linalg.norm(q[1:])),
        },
        "cosines": {
            "d_m_r": descriptive_cosine(d_m[1:], r_nonempty),
            "d_m_q": descriptive_cosine(d_m[1:], q[1:]),
            "r_q": descriptive_cosine(r_nonempty, q[1:]),
        },
    }


def type7(values: np.ndarray, probability: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not 0 <= probability <= 1:
        raise ValueError("invalid type-7 quantile input")
    return float(np.quantile(values, probability, method="linear"))


def summarize_draws(draws: list[dict[str, Any]]) -> dict[str, Any]:
    decision_names = ("rmse", "zero_rmse", "nre", "r2", "max_abs_error", "max_abs_truth")
    finite_every_draw = all(
        all(np.isfinite(float(draw[name])) for name in decision_names)
        and all(
            np.isfinite(float(group[key]))
            for group in draw["subgroups"].values()
            for key in ("rmse", "zero_rmse", "nre")
        )
        for draw in draws
    )
    summary: dict[str, Any] = {"finite_every_draw": finite_every_draw, "draw_count": len(draws)}
    for name in decision_names:
        values = np.asarray([draw[name] for draw in draws], dtype=np.float64)
        summary[name] = (
            {
                "q025": type7(values, 0.025), "q05": type7(values, 0.05),
                "q95": type7(values, 0.95), "q975": type7(values, 0.975),
            }
            if np.all(np.isfinite(values)) else None
        )
    return summary


def registered_gates(point: dict[str, Any], bootstrap: dict[str, Any]) -> dict[str, bool]:
    finite_point = all(np.isfinite(float(point[name])) for name in (
        "rmse", "zero_rmse", "nre", "r2", "max_abs_error", "max_abs_truth",
    )) and all(
        np.isfinite(float(group[key]))
        for group in point["subgroups"].values()
        for key in ("rmse", "zero_rmse", "nre")
    )
    return {
        "finite_decision_metrics": bool(finite_point and bootstrap["finite_every_draw"]),
        "point_nre_below_half": bool(point["nre"] < 0.5),
        "q95_nre_below_one": bool(
            bootstrap["nre"] is not None and bootstrap["nre"]["q95"] < 1.0
        ),
        "positive_r2": bool(
            point["r2"] > 0.5 and bootstrap["r2"] is not None
            and bootstrap["r2"]["q025"] > 0
        ),
        "sign_agreement": bool(point["sign_agreement"] >= 6),
        "every_subgroup_no_worse_than_zero": all(
            group["no_worse_than_zero"] for group in point["subgroups"].values()
        ),
        "max_error_below_max_truth": bool(point["max_abs_error"] < point["max_abs_truth"]),
    }


def score_role(data: RoleArrays, *, draws: int = BOOTSTRAP_DRAWS) -> dict[str, Any]:
    if type(draws) is not int or draws <= 0:
        raise ValueError("bootstrap draw count is invalid")
    point_contrast = contrasts(aggregate(data))
    point = decision_metrics(point_contrast)
    rng = np.random.default_rng(BOOTSTRAP_SEEDS[data.role])
    bootstrap_metrics = []
    for _ in range(draws):
        indices = rng.integers(0, data.document_count, size=data.document_count)
        multiplicity = np.bincount(indices, minlength=data.document_count)
        bootstrap_metrics.append(decision_metrics(contrasts(aggregate(data, multiplicity))))
    bootstrap = summarize_draws(bootstrap_metrics)
    gates = registered_gates(point, bootstrap)
    return {
        "role": data.role,
        "point": point,
        "bootstrap": bootstrap,
        "gates": gates,
        "useful_pass": all(gates.values()),
        "contrasts": {
            name: value.tolist() if isinstance(value, np.ndarray) else value
            for name, value in point_contrast.items()
        },
    }


def score_cross_role(
    source: RoleArrays, target: RoleArrays, *, draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    source_prediction = np.asarray(
        contrasts(aggregate(source))["prediction"], dtype=np.float64,
    )[1:]

    def target_metrics(multiplicity: np.ndarray | None) -> dict[str, float]:
        truth = np.asarray(contrasts(aggregate(target, multiplicity))["d_m"], dtype=np.float64)[1:]
        rmse = _rmse(truth, source_prediction)
        zero = float(np.sqrt(np.mean(np.square(truth))))
        return {"rmse": rmse, "zero_rmse": zero, "nre": rmse / zero if zero > 0 else float("nan")}

    point = target_metrics(None)
    rng = np.random.default_rng(BOOTSTRAP_SEEDS[target.role])
    samples = []
    for _ in range(draws):
        indices = rng.integers(0, target.document_count, size=target.document_count)
        samples.append(target_metrics(np.bincount(indices, minlength=target.document_count)))
    nre = np.asarray([sample["nre"] for sample in samples], dtype=np.float64)
    finite = bool(np.all(np.isfinite(nre)) and all(
        np.isfinite(value) for value in point.values()
    ))
    q95 = type7(nre, 0.95) if finite else None
    gates = {
        "finite": finite,
        "point_nre_below_half": bool(finite and point["nre"] < 0.5),
        "q95_nre_below_one": bool(finite and q95 is not None and q95 < 1),
    }
    return {
        "source_role": source.role,
        "target_role": target.role,
        "source_prediction_fixed_at_point": True,
        "point": point,
        "bootstrap": {"draw_count": draws, "q95_nre": q95, "finite_every_draw": finite},
        "gates": gates,
        "useful_pass": all(gates.values()),
    }
