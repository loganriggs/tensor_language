"""Canonical semantic-to-physical action plans for suffix final execution.

This pure module loads no rows, model, checkpoint, or program artifact.  It tells an
observed executor which site-0 program, site-1 program, cross map, and early-MLP path
each registered final arm denotes.  Keeping one exhaustive mapping prevents an arm
such as ``r0_l1`` from silently executing the jointly fitted ``R`` pair.
"""

from __future__ import annotations

from dataclasses import dataclass


BACKGROUNDS = ("N", "E")
BASE_ARMS = (
    "qq", "ll", "s0_l1", "l0_s1", "rr", "r0_l1", "l0_r1", "lt",
    "zero_a", *(f"a_null_{index:02d}" for index in range(20)),
    "shuffled_l", "shuffled_r", "n_n", "o_o", "new_fit_mean",
)
RESPONSE_ARMS = frozenset({
    "ll", "lt", *(f"a_null_{index:02d}" for index in range(20)),
})
CODE_RESPONSE_ARMS = frozenset({"ll", "lt"})

_EXECUTION_KINDS = {
    "projected_program", "deployed_baseline", "native_baseline", "mean_program",
}
_SITE_SOURCES = {
    "inherited_q0", "inherited_q1", "true_l0", "true_l1", "true_s0",
    "true_s1", "true_r0", "true_r1", "mapped_shuffled_l0",
    "mapped_shuffled_l1", "mapped_shuffled_r0", "mapped_shuffled_r1",
    "new_fit_mean0", "new_fit_mean1",
}


@dataclass(frozen=True, slots=True)
class FinalArmPlan:
    """One base arm's exact physical program composition."""

    arm: str
    execution_kind: str
    site0_source: str | None
    site1_source: str | None
    cross_source: str | None
    identity_route: str | None
    identity_control: str | None

    def __post_init__(self) -> None:
        if self.arm not in BASE_ARMS or self.execution_kind not in _EXECUTION_KINDS:
            raise ValueError("final arm plan header is outside the registered lattice")
        if self.execution_kind in {"projected_program", "mean_program"}:
            if self.site0_source not in _SITE_SOURCES or self.site1_source not in (
                _SITE_SOURCES
            ) or self.identity_route not in {"Q", "L", "R", "S0", "S1", "T"}:
                raise ValueError("program arm lacks two typed site sources and a route")
        elif any(value is not None for value in (
            self.site0_source, self.site1_source, self.cross_source,
            self.identity_route, self.identity_control,
        )):
            raise ValueError("native/deployed baseline acquired a projected program")
        if self.execution_kind == "projected_program" and self.identity_control is None:
            raise ValueError("projected arm lacks a typed control identity")
        if self.execution_kind == "mean_program" and (
            self.identity_control != "new_fit_mean" or self.cross_source is not None
        ):
            raise ValueError("new-fit mean program identity changed")
        if self.cross_source is not None and self.identity_route != "T":
            raise ValueError("a cross map is licensed only on the T physical route")

    @property
    def expects_logit_response(self) -> bool:
        return self.arm in RESPONSE_ARMS

    @property
    def expects_code_response(self) -> bool:
        return self.arm in CODE_RESPONSE_ARMS


def _program(
    arm: str, site0: str, site1: str, *, route: str, control: str,
    cross: str | None = None,
) -> FinalArmPlan:
    return FinalArmPlan(
        arm=arm, execution_kind="projected_program",
        site0_source=site0, site1_source=site1, cross_source=cross,
        identity_route=route, identity_control=control,
    )


_PLAN_BY_ARM = {
    "qq": _program(
        "qq", "inherited_q0", "inherited_q1", route="Q", control="inherited_q",
    ),
    "ll": _program("ll", "true_l0", "true_l1", route="L", control="true"),
    "s0_l1": _program(
        "s0_l1", "true_s0", "true_l1", route="S0", control="hybrid_s0_l1",
    ),
    "l0_s1": _program(
        "l0_s1", "true_l0", "true_s1", route="S1", control="hybrid_l0_s1",
    ),
    "rr": _program("rr", "true_r0", "true_r1", route="R", control="true"),
    "r0_l1": _program(
        "r0_l1", "true_r0", "true_l1", route="R", control="hybrid_r0_l1",
    ),
    "l0_r1": _program(
        "l0_r1", "true_l0", "true_r1", route="R", control="hybrid_l0_r1",
    ),
    "lt": _program(
        "lt", "true_l0", "true_l1", route="T", control="true",
        cross="true_t_cross",
    ),
    "zero_a": _program(
        "zero_a", "true_l0", "true_l1", route="T", control="zero_A",
        cross="zero_cross",
    ),
    **{
        f"a_null_{index:02d}": _program(
            f"a_null_{index:02d}", "true_l0", "true_l1", route="T",
            control=f"A_null_{index:02d}",
            cross=f"mapped_A_null_{index:02d}_cross",
        )
        for index in range(20)
    },
    "shuffled_l": _program(
        "shuffled_l", "mapped_shuffled_l0", "mapped_shuffled_l1", route="L",
        control="document_shuffle",
    ),
    "shuffled_r": _program(
        "shuffled_r", "mapped_shuffled_r0", "mapped_shuffled_r1", route="R",
        control="document_shuffle",
    ),
    "n_n": FinalArmPlan(
        arm="n_n", execution_kind="deployed_baseline", site0_source=None,
        site1_source=None, cross_source=None, identity_route=None,
        identity_control=None,
    ),
    "o_o": FinalArmPlan(
        arm="o_o", execution_kind="native_baseline", site0_source=None,
        site1_source=None, cross_source=None, identity_route=None,
        identity_control=None,
    ),
    "new_fit_mean": FinalArmPlan(
        arm="new_fit_mean", execution_kind="mean_program",
        site0_source="new_fit_mean0", site1_source="new_fit_mean1",
        cross_source=None, identity_route="L", identity_control="new_fit_mean",
    ),
}

if tuple(_PLAN_BY_ARM) != BASE_ARMS:
    raise RuntimeError("canonical final arm plan is incomplete or reordered")

CANONICAL_ARM_PLANS = tuple(_PLAN_BY_ARM[arm] for arm in BASE_ARMS)


@dataclass(frozen=True, slots=True)
class FinalActionPlan:
    """One arm plan combined with its preregistered MLP2 background."""

    arm_plan: FinalArmPlan
    background: str

    def __post_init__(self) -> None:
        if type(self.arm_plan) is not FinalArmPlan or self.background not in BACKGROUNDS:
            raise ValueError("final action plan is outside the registered lattice")

    @property
    def key(self) -> str:
        return f"{self.arm_plan.arm}/{self.background}"

    @property
    def mlp2_source(self) -> str:
        return "deployed_mlp2" if self.background == "N" else "exact_mlp2"

    @property
    def emits_registered_primary(self) -> bool:
        return self.background == "N"

    @property
    def emits_logit_response(self) -> bool:
        return self.background == "N" and self.arm_plan.expects_logit_response

    @property
    def emits_code_response(self) -> bool:
        return self.background == "N" and self.arm_plan.expects_code_response


CANONICAL_ACTION_PLANS = tuple(
    FinalActionPlan(arm_plan=arm, background=background)
    for arm in CANONICAL_ARM_PLANS for background in BACKGROUNDS
)
CANONICAL_ACTION_KEYS = tuple(plan.key for plan in CANONICAL_ACTION_PLANS)


def plan_for(arm: str, background: str) -> FinalActionPlan:
    """Resolve one typed semantic key without accepting aliases or fallbacks."""

    if arm not in _PLAN_BY_ARM or background not in BACKGROUNDS:
        raise ValueError("final action key is outside the registered lattice")
    return FinalActionPlan(arm_plan=_PLAN_BY_ARM[arm], background=background)
