"""CPU contracts for joint live-oracle corrections at MLP0--2.

The runtime correction map is frozen before a forward pass.  The analyzer accepts
paired per-row CE for the complete Boolean cube and reports gains with the useful
sign convention: positive means lower CE than the deployed-ship baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping, MutableMapping, Sequence

from factorial_causal_attribution import analyze_cube, powerset


EARLY_MLP_GROUPS = ("mlp0", "mlp1", "mlp2")


@dataclass(frozen=True)
class OracleCorrectionSpec:
    """Read-only correction configuration for one live site.

    ``basis=None`` denotes the unrestricted exact original-minus-plank residual.
    Tensor payloads are treated as read-only; freezing prevents topology, basis
    reference, or strength changes while an arm is being evaluated.
    """

    basis: Any
    scale: float = 1.0


def _validated_spec(value: OracleCorrectionSpec | Mapping[str, Any]) -> OracleCorrectionSpec:
    if isinstance(value, OracleCorrectionSpec):
        spec = value
    elif isinstance(value, Mapping):
        unknown = set(value).difference(("basis", "scale"))
        if unknown:
            raise ValueError(f"unknown correction fields: {sorted(unknown)}")
        spec = OracleCorrectionSpec(value.get("basis"), float(value.get("scale", 1.0)))
    else:
        raise TypeError("correction must be OracleCorrectionSpec or a mapping")
    if not math.isfinite(spec.scale):
        raise ValueError("correction scale must be finite")
    return spec


def freeze_oracle_corrections(
    corrections: Mapping[int, OracleCorrectionSpec | Mapping[str, Any]],
) -> Mapping[int, OracleCorrectionSpec]:
    """Validate and freeze a per-site correction map.

    The mapping and specs are immutable.  Sites are restricted to the registered
    early block so a mistyped later-layer intervention fails before execution.
    """

    frozen: dict[int, OracleCorrectionSpec] = {}
    for raw_site, value in corrections.items():
        if isinstance(raw_site, bool) or not isinstance(raw_site, int):
            raise TypeError("oracle correction sites must be integer layer indices")
        if raw_site not in (0, 1, 2):
            raise ValueError(f"joint early-MLP oracle does not admit site {raw_site}")
        frozen[raw_site] = _validated_spec(value)
    return MappingProxyType(frozen)


def configure_oracle_corrections(
    state: MutableMapping[str, Any],
    corrections: Mapping[int, OracleCorrectionSpec | Mapping[str, Any]],
) -> Mapping[int, OracleCorrectionSpec]:
    """Install an immutable joint map into the legacy mutable runtime state."""

    frozen = freeze_oracle_corrections(corrections)
    state.update({
        "on": bool(frozen),
        "site": None,
        "basis": None,
        "scale": 1.0,
        "corrections": frozen,
    })
    return frozen


def clear_oracle_corrections(state: MutableMapping[str, Any]) -> None:
    """Return the shared runtime state to its inert legacy-compatible form."""

    state.update({
        "on": False,
        "site": None,
        "basis": None,
        "scale": 1.0,
        "corrections": None,
    })


def resolve_oracle_correction(
    state: Mapping[str, Any], site: int,
) -> OracleCorrectionSpec | None:
    """Resolve one site's correction while preserving the singleton API.

    An explicitly selected legacy ``site`` takes precedence.  Thus existing calls
    which update only ``on/site/basis/scale`` retain their exact behavior even if a
    stale joint map is present in a long-lived process.
    """

    if not state.get("on", False):
        return None
    legacy_site = state.get("site")
    if legacy_site is not None:
        if legacy_site != site:
            return None
        return _validated_spec({
            "basis": state.get("basis"),
            "scale": state.get("scale", 1.0),
        })
    corrections = state.get("corrections")
    if corrections is None:
        return None
    return corrections.get(site)


def _arm_name(arm: Sequence[str]) -> str:
    return "+".join(arm) if arm else "baseline"


def analyze_full_live_subset_rows(
    row_ce_by_arm: Mapping[tuple[str, ...], Sequence[float]],
    *,
    mlp012_residual_nats: float | None = None,
) -> dict[str, Any]:
    """Analyze paired row CE for all eight exact-live restoration subsets.

    ``row_ce_by_arm`` keys are canonicalized by the shared factorial library.  All
    arms must contain the same positive number of finite paired row scores.  The
    optional residual denominator must use the same realization, rows, mask, and CE
    currency; callers unable to supply that denominator leave the recovery fraction
    absent rather than importing a number from another run.
    """

    expected = set(powerset(EARLY_MLP_GROUPS))
    supplied = set(row_ce_by_arm)
    if supplied != expected:
        raise ValueError(
            f"full-live subset cube mismatch: missing={sorted(expected - supplied)}, "
            f"extra={sorted(supplied - expected)}"
        )
    rows = {arm: [float(value) for value in values]
            for arm, values in row_ce_by_arm.items()}
    lengths = {len(values) for values in rows.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) <= 0:
        raise ValueError("all subset arms must contain the same positive row count")
    if any(not math.isfinite(value) for values in rows.values() for value in values):
        raise ValueError("subset row CE must be finite")
    if mlp012_residual_nats is not None:
        mlp012_residual_nats = float(mlp012_residual_nats)
        if not math.isfinite(mlp012_residual_nats) or mlp012_residual_nats <= 0.0:
            raise ValueError("MLP0-2 residual denominator must be finite and positive")

    baseline = rows[()]
    row_count = len(baseline)
    gain_cube = {
        arm: sum(base - score for base, score in zip(baseline, rows[arm])) / row_count
        for arm in powerset(EARLY_MLP_GROUPS)
    }
    factorial = analyze_cube(EARLY_MLP_GROUPS, gain_cube)
    full_arm = EARLY_MLP_GROUPS
    upstream_arm = ("mlp0", "mlp1")
    joint_gain = gain_cube[full_arm]
    singleton_sum = sum(gain_cube[(group,)] for group in EARLY_MLP_GROUPS)
    pair_names = ("mlp0+mlp1", "mlp0+mlp2", "mlp1+mlp2")

    return {
        "groups": list(EARLY_MLP_GROUPS),
        "row_count": row_count,
        "currency": "paired mean CE gain; positive lowers CE versus deployed ship",
        "gain_by_arm": {_arm_name(arm): gain_cube[arm]
                        for arm in powerset(EARLY_MLP_GROUPS)},
        "joint_gain": joint_gain,
        "singleton_gain_sum": singleton_sum,
        "joint_minus_singleton_sum": joint_gain - singleton_sum,
        "joint_gain_fraction_of_mlp012_residual": (
            joint_gain / mlp012_residual_nats
            if mlp012_residual_nats is not None else None
        ),
        "mlp012_residual_nats": mlp012_residual_nats,
        "pairwise_mobius": {name: factorial["mobius"][name] for name in pair_names},
        "triple_mobius": factorial["mobius"]["mlp0+mlp1+mlp2"],
        "mlp2_conditional_marginal_after_mlp0_mlp1": (
            gain_cube[full_arm] - gain_cube[upstream_arm]
        ),
        "shapley": factorial["shapley"],
        "shapley_closure_error": factorial["shapley_closure_error"],
        "interaction_l1": factorial["interaction_l1"],
        "interaction_l1_fraction_of_joint_gain": (
            factorial["interaction_l1_fraction_of_total"]
        ),
    }
