"""Canonical semantic-to-physical action plans for suffix final execution.

This pure module loads no rows, model, checkpoint, or program artifact.  It tells an
observed executor which site-0 program, site-1 program, cross map, and early-MLP path
each registered final arm denotes.  Keeping one exhaustive mapping prevents an arm
such as ``r0_l1`` from silently executing the jointly fitted ``R`` pair.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch

import early_mlp_suffix_transport_v1_runtime as runtime


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
SOURCE_PROGRAM_KEYS = (
    "inherited_q", "true/L", "true/R", "true/S0", "true/S1", "true/T",
    "mapped/document_shuffle/L", "mapped/document_shuffle/R",
    *(f"mapped/A_null_{index:02d}/T" for index in range(20)),
    "new_fit_mean",
)

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

    @property
    def sha256(self) -> str:
        return runtime.logical_identity_sha256({
            field: getattr(self, field) for field in self.__dataclass_fields__
        })


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

    @property
    def sha256(self) -> str:
        return runtime.logical_identity_sha256({
            "arm_plan_sha256": self.arm_plan.sha256,
            "background": self.background,
            "mlp2_source": self.mlp2_source,
            "emits_registered_primary": self.emits_registered_primary,
            "emits_logit_response": self.emits_logit_response,
            "emits_code_response": self.emits_code_response,
        })


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


def _program_sha256(program: runtime.JointAffineProgram) -> str:
    if not isinstance(program, runtime.JointAffineProgram):
        raise TypeError("final source is not a joint affine program")
    return runtime.program_snapshot_sha256(program)


class FinalProgramSourceBank:
    """Immutable, complete physical sources for all projected final arms."""

    __slots__ = ("__expected", "__programs", "__sealed")

    def __init__(self, programs: Mapping[str, runtime.JointAffineProgram]) -> None:
        object.__setattr__(self, "_FinalProgramSourceBank__sealed", False)
        if not isinstance(programs, Mapping) or tuple(programs) != SOURCE_PROGRAM_KEYS:
            raise ValueError("final program source bank is incomplete or reordered")
        expected_routes = {
            "inherited_q": "L", "true/L": "L", "true/R": "R",
            "true/S0": "S0", "true/S1": "S1", "true/T": "T",
            "mapped/document_shuffle/L": "L", "mapped/document_shuffle/R": "R",
            **{f"mapped/A_null_{index:02d}/T": "T" for index in range(20)},
            "new_fit_mean": "L",
        }
        masters: dict[str, runtime.JointAffineProgram] = {}
        identities: dict[str, str] = {}
        for key in SOURCE_PROGRAM_KEYS:
            value = programs[key]
            if not isinstance(value, runtime.JointAffineProgram) or value.route != (
                expected_routes[key]
            ):
                raise ValueError(f"final program source {key} has the wrong route")
            masters[key] = copy.deepcopy(value).cpu()
            identities[key] = _program_sha256(masters[key])
        object.__setattr__(
            self, "_FinalProgramSourceBank__programs", MappingProxyType(masters),
        )
        object.__setattr__(
            self, "_FinalProgramSourceBank__expected", MappingProxyType(identities),
        )
        object.__setattr__(self, "_FinalProgramSourceBank__sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_FinalProgramSourceBank__sealed", False):
            raise AttributeError("final program source bank is sealed")
        object.__setattr__(self, name, value)

    def __copy__(self):
        raise RuntimeError("final program source banks cannot be copied")

    def __deepcopy__(self, memo):
        raise RuntimeError("final program source banks cannot be copied")

    def __reduce__(self):
        raise RuntimeError("final program source banks cannot be serialized")

    @property
    def sha256(self) -> str:
        self._require_pristine()
        return runtime.logical_identity_sha256(dict(self.__expected))

    def _require_pristine(self) -> None:
        for key in SOURCE_PROGRAM_KEYS:
            if _program_sha256(self.__programs[key]) != self.__expected[key]:
                raise RuntimeError(f"final program source {key} mutated")

    def clone(self, key: str) -> runtime.JointAffineProgram:
        self._require_pristine()
        if key not in self.__programs:
            raise ValueError("final program source key is unknown")
        value = copy.deepcopy(self.__programs[key]).cpu()
        if _program_sha256(value) != self.__expected[key]:
            raise RuntimeError("final program source clone changed identity")
        return value

    def source_sha256(self, key: str) -> str:
        self._require_pristine()
        if key not in self.__expected:
            raise ValueError("final program source key is unknown")
        return self.__expected[key]


def source_bank_from_validated(
    validated_program_bank: Mapping[str, Any],
    *, inherited_q: runtime.JointAffineProgram,
) -> FinalProgramSourceBank:
    """Mint all physical sources only from the replay-validated canonical bank."""

    import early_mlp_suffix_transport_v1_programs as programs

    required = {
        "true_programs", "mapped_programs", "new_fit_mean", "payload_sha256",
        "validation_baseline", "validation_execution", "transport_geometry",
        "teacher_calibration",
    }
    if not isinstance(validated_program_bank, Mapping) or set(
        validated_program_bank
    ) != required or not runtime._sha256_text(
        validated_program_bank["payload_sha256"]
    ) or not isinstance(inherited_q, runtime.JointAffineProgram) or (
        inherited_q.route != "L"
    ):
        raise ValueError("final sources require the replay-validated canonical bank and Q")
    true = validated_program_bank["true_programs"]
    mapped = validated_program_bank["mapped_programs"]
    mean = validated_program_bank["new_fit_mean"]
    if not isinstance(true, Mapping) or set(true) != set(programs.SELECTABLE_ROUTES) or (
        not isinstance(mapped, Mapping)
    ) or tuple(mapped) != programs.required_mapped_control_keys() or not isinstance(
        mean, programs.FrozenNewFitMeanProgram
    ):
        raise ValueError("validated final program families are incomplete")
    values: dict[str, runtime.JointAffineProgram] = {"inherited_q": inherited_q}
    for route in programs.SELECTABLE_ROUTES:
        frozen = true[route]
        if not isinstance(frozen, programs.FrozenProgram) or frozen.route != route:
            raise ValueError("validated true final program family changed")
        values[f"true/{route}"] = frozen.make_program()
    for route in ("L", "R"):
        key = f"document_shuffle/{route}"
        frozen = mapped[key]
        if not isinstance(frozen, programs.FrozenMappedProgram) or frozen.key != key:
            raise ValueError("validated shuffled final program family changed")
        values[f"mapped/{key}"] = frozen.make_program()
    for index in range(20):
        key = f"A_null_{index:02d}/T"
        frozen = mapped[key]
        if not isinstance(frozen, programs.FrozenMappedProgram) or frozen.key != key:
            raise ValueError("validated A-null final program family changed")
        values[f"mapped/{key}"] = frozen.make_program()
    values["new_fit_mean"] = mean.make_program()
    if tuple(values) != SOURCE_PROGRAM_KEYS:
        raise RuntimeError("validated final source ordering changed")
    return FinalProgramSourceBank(values)


_SITE_SOURCE = {
    "inherited_q0": ("inherited_q", 0), "inherited_q1": ("inherited_q", 1),
    "true_l0": ("true/L", 0), "true_l1": ("true/L", 1),
    "true_s0": ("true/S0", 0), "true_s1": ("true/S1", 1),
    "true_r0": ("true/R", 0), "true_r1": ("true/R", 1),
    "mapped_shuffled_l0": ("mapped/document_shuffle/L", 0),
    "mapped_shuffled_l1": ("mapped/document_shuffle/L", 1),
    "mapped_shuffled_r0": ("mapped/document_shuffle/R", 0),
    "mapped_shuffled_r1": ("mapped/document_shuffle/R", 1),
    "new_fit_mean0": ("new_fit_mean", 0),
    "new_fit_mean1": ("new_fit_mean", 1),
}


class MaterializedFinalAction:
    """Sealed physical program or baseline bound to one semantic final action."""

    __slots__ = (
        "__component_sha256s", "__expected_program_sha256", "__plan", "__program",
        "__sealed", "__source_bank_sha256",
    )

    def __init__(
        self, *, plan: FinalActionPlan, source_bank_sha256: str,
        component_sha256s: Mapping[str, str],
        program: runtime.JointAffineProgram | None,
    ) -> None:
        object.__setattr__(self, "_MaterializedFinalAction__sealed", False)
        if type(plan) is not FinalActionPlan or not runtime._sha256_text(
            source_bank_sha256
        ) or not isinstance(component_sha256s, Mapping) or not component_sha256s or any(
            not isinstance(key, str) or not key or not runtime._sha256_text(value)
            for key, value in component_sha256s.items()
        ):
            raise ValueError("materialized final action identity is malformed")
        requires_program = plan.arm_plan.execution_kind in {
            "projected_program", "mean_program",
        }
        if requires_program != isinstance(program, runtime.JointAffineProgram):
            raise ValueError("materialized final action program/baseline kind changed")
        master = None if program is None else copy.deepcopy(program).cpu()
        expected = None if master is None else _program_sha256(master)
        object.__setattr__(self, "_MaterializedFinalAction__plan", plan)
        object.__setattr__(
            self, "_MaterializedFinalAction__source_bank_sha256", source_bank_sha256,
        )
        object.__setattr__(
            self, "_MaterializedFinalAction__component_sha256s",
            MappingProxyType(dict(sorted(component_sha256s.items()))),
        )
        object.__setattr__(self, "_MaterializedFinalAction__program", master)
        object.__setattr__(
            self, "_MaterializedFinalAction__expected_program_sha256", expected,
        )
        object.__setattr__(self, "_MaterializedFinalAction__sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_MaterializedFinalAction__sealed", False):
            raise AttributeError("materialized final action is sealed")
        object.__setattr__(self, name, value)

    @property
    def plan(self) -> FinalActionPlan:
        return self.__plan

    @property
    def program_sha256(self) -> str | None:
        self._require_pristine()
        return self.__expected_program_sha256

    @property
    def component_sha256s(self) -> Mapping[str, str]:
        self._require_pristine()
        return self.__component_sha256s

    @property
    def sha256(self) -> str:
        self._require_pristine()
        return runtime.logical_identity_sha256({
            "action_plan_sha256": self.__plan.sha256,
            "source_bank_sha256": self.__source_bank_sha256,
            "component_sha256s": dict(self.__component_sha256s),
            "program_sha256": self.__expected_program_sha256,
        })

    def _require_pristine(self) -> None:
        if self.__program is not None and _program_sha256(self.__program) != (
            self.__expected_program_sha256
        ):
            raise RuntimeError("materialized final program mutated")

    def make_program(self) -> runtime.JointAffineProgram:
        self._require_pristine()
        if self.__program is None:
            raise RuntimeError("baseline final action has no projected program")
        value = copy.deepcopy(self.__program).cpu()
        if _program_sha256(value) != self.__expected_program_sha256:
            raise RuntimeError("materialized final program clone changed identity")
        return value


def _affine_component_sha256(value: runtime.AffineCodeProgram) -> str:
    return runtime.logical_identity_sha256({
        name: runtime.tensor_identity_sha256(getattr(value, name))
        for name in ("mean", "scale", "weight", "bias")
    })


def materialize(
    plan: FinalActionPlan, sources: FinalProgramSourceBank,
) -> MaterializedFinalAction:
    """Compose the exact registered site/cross sources without model or row access."""

    if type(plan) is not FinalActionPlan or not isinstance(
        sources, FinalProgramSourceBank
    ):
        raise TypeError("final materialization requires a typed plan and source bank")
    arm = plan.arm_plan
    if arm.execution_kind in {"deployed_baseline", "native_baseline"}:
        return MaterializedFinalAction(
            plan=plan, source_bank_sha256=sources.sha256,
            component_sha256s={"baseline": runtime.logical_identity_sha256({
                "execution_kind": arm.execution_kind,
            })}, program=None,
        )

    site0_key, site0 = _SITE_SOURCE[arm.site0_source]
    site1_key, site1 = _SITE_SOURCE[arm.site1_source]
    source0 = sources.clone(site0_key)
    source1 = sources.clone(site1_key)
    affine0 = copy.deepcopy(source0.site0 if site0 == 0 else source0.site1)
    affine1 = copy.deepcopy(source1.site0 if site1 == 0 else source1.site1)
    topology_route = "L" if arm.identity_route == "Q" else arm.identity_route
    program = runtime.JointAffineProgram(affine0, affine1, route=topology_route)
    components = {
        "site0": _affine_component_sha256(affine0),
        "site1": _affine_component_sha256(affine1),
        "site0_source_program": sources.source_sha256(site0_key),
        "site1_source_program": sources.source_sha256(site1_key),
    }
    if arm.cross_source is not None:
        if program.cross is None:
            raise RuntimeError("cross-bearing final action lost T topology")
        if arm.cross_source == "zero_cross":
            cross = torch.zeros_like(program.cross)
            cross_source_identity = runtime.logical_identity_sha256({
                "kind": "registered_zero_cross", "shape": list(cross.shape),
                "dtype": str(cross.dtype),
            })
        elif arm.cross_source == "true_t_cross":
            source_key = "true/T"
            source = sources.clone(source_key)
            cross = source.cross
            cross_source_identity = sources.source_sha256(source_key)
        else:
            control = arm.identity_control
            source_key = f"mapped/{control}/T"
            source = sources.clone(source_key)
            cross = source.cross
            cross_source_identity = sources.source_sha256(source_key)
        if cross is None:
            raise RuntimeError("cross-bearing final source lacks its cross tensor")
        with torch.no_grad():
            program.cross.copy_(cross)
        components["cross"] = runtime.tensor_identity_sha256(program.cross)
        components["cross_source_program"] = cross_source_identity
    return MaterializedFinalAction(
        plan=plan, source_bank_sha256=sources.sha256,
        component_sha256s=components, program=program,
    )


@dataclass(frozen=True, slots=True)
class FinalActionBatchIdentity:
    """Final-only trace binding semantic action to physical materialization and rows."""

    action_key: str
    action_plan_sha256: str
    materialization_sha256: str
    source_commit: str
    inherited_snapshot_sha256: str
    rows_receipt_sha256: str
    final_role_tensor_sha256: str
    program_payload_sha256: str
    common_support_sha256: str
    ordered_batch_indices_sha256: str
    ordered_input_tokens_sha256: str
    batch_ordinal: int
    batch_rows: int = runtime.BATCH_SIZE
    sequence_length: int = runtime.SEQUENCE_LENGTH

    def __post_init__(self) -> None:
        if self.action_key not in CANONICAL_ACTION_KEYS or not runtime._sha256_text(
            self.action_plan_sha256
        ) or not runtime._sha256_text(self.materialization_sha256) or not isinstance(
            self.source_commit, str
        ) or len(self.source_commit) != 40 or any(
            character not in "0123456789abcdef" for character in self.source_commit
        ) or any(not runtime._sha256_text(value) for value in (
            self.inherited_snapshot_sha256, self.rows_receipt_sha256,
            self.final_role_tensor_sha256, self.program_payload_sha256,
            self.common_support_sha256, self.ordered_batch_indices_sha256,
            self.ordered_input_tokens_sha256,
        )):
            raise ValueError("final action batch identity is malformed")
        if type(self.batch_ordinal) is not int or not 0 <= self.batch_ordinal < (
            192 // runtime.BATCH_SIZE
        ) or self.batch_rows != runtime.BATCH_SIZE or self.sequence_length != (
            runtime.SEQUENCE_LENGTH
        ):
            raise ValueError("final action batch schedule changed")
        arm, background = self.action_key.split("/")
        if plan_for(arm, background).sha256 != self.action_plan_sha256:
            raise ValueError("final action batch plan identity changed")

    @property
    def sha256(self) -> str:
        return runtime.logical_identity_sha256({
            field: getattr(self, field) for field in self.__dataclass_fields__
        })

    @classmethod
    def from_inputs(
        cls, *, materialized: MaterializedFinalAction, inputs: torch.Tensor,
        ordered_batch_indices: Sequence[int], batch_ordinal: int,
        source_commit: str, inherited_snapshot_sha256: str,
        rows_receipt_sha256: str, final_role_tensor_sha256: str,
        program_payload_sha256: str, common_support_sha256: str,
    ) -> "FinalActionBatchIdentity":
        if not isinstance(materialized, MaterializedFinalAction) or not torch.is_tensor(
            inputs
        ) or inputs.dtype != torch.long or tuple(inputs.shape) != (
            runtime.BATCH_SIZE, runtime.SEQUENCE_LENGTH
        ):
            raise ValueError("final action batch inputs/materialization are malformed")
        indices = tuple(ordered_batch_indices)
        expected = tuple(range(
            batch_ordinal * runtime.BATCH_SIZE,
            (batch_ordinal + 1) * runtime.BATCH_SIZE,
        ))
        if indices != expected:
            raise ValueError("final action batch indices are not canonical")
        return cls(
            action_key=materialized.plan.key,
            action_plan_sha256=materialized.plan.sha256,
            materialization_sha256=materialized.sha256,
            source_commit=source_commit,
            inherited_snapshot_sha256=inherited_snapshot_sha256,
            rows_receipt_sha256=rows_receipt_sha256,
            final_role_tensor_sha256=final_role_tensor_sha256,
            program_payload_sha256=program_payload_sha256,
            common_support_sha256=common_support_sha256,
            ordered_batch_indices_sha256=runtime.logical_identity_sha256(list(indices)),
            ordered_input_tokens_sha256=runtime.tensor_identity_sha256(inputs),
            batch_ordinal=batch_ordinal,
        )

    def require(
        self, *, materialized: MaterializedFinalAction, inputs: torch.Tensor,
        ordered_batch_indices: Sequence[int],
    ) -> None:
        if not isinstance(materialized, MaterializedFinalAction) or (
            materialized.sha256 != self.materialization_sha256
        ) or materialized.plan.sha256 != self.action_plan_sha256 or (
            materialized.plan.key != self.action_key
        ) or runtime.tensor_identity_sha256(inputs) != self.ordered_input_tokens_sha256 or (
            runtime.logical_identity_sha256(list(ordered_batch_indices))
            != self.ordered_batch_indices_sha256
        ):
            raise RuntimeError("final action batch differs from its sealed identity")
