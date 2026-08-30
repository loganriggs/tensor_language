"""Outcome-blind fixed-tensor scaffold for the newline attention crew.

Behavior masks are CPU-only scoring metadata.  Replacement callbacks receive no mask
and execute complete stored squared-attention layers with a constant head projector.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

import torch
import torch.nn as nn

import circuit_campaign_runtime as runtime
import circuit_campaign_statistics as statistics
from tensor_preserving_attention import PROJECTION_NAMES, TensorPreservingSquaredAttention


PRODUCTION_SITES = 18
PRODUCTION_HEADS = 9
PRODUCTION_WIDTH = 1152
CANARY_HEAD = (12, 6)
FIVE_HEAD_CREW = ((7, 2), (8, 2), (10, 2), (11, 0), (12, 6))
FIVE_HEAD_CONTROL = tuple((site, (head + 1) % PRODUCTION_HEADS) for site, head in FIVE_HEAD_CREW)


class NewlineScope(str, Enum):
    CANARY = "l12h6_canary"
    FIVE_HEAD = "five_head"


class NewlineArm(str, Enum):
    NATIVE = "native"
    EXACT = "exact"
    REMOVE = "remove"
    HEAD_LABEL_CONTROL = "head_label_control"


def _heads(scope: NewlineScope, *, control: bool = False) -> tuple[tuple[int, int], ...]:
    if type(scope) is not NewlineScope:
        raise ValueError("scope must be a NewlineScope")
    if scope is NewlineScope.CANARY:
        site, head = CANARY_HEAD
        return ((site, (head + 1) % PRODUCTION_HEADS),) if control else (CANARY_HEAD,)
    return FIVE_HEAD_CONTROL if control else FIVE_HEAD_CREW


def replacement_sites(scope: NewlineScope) -> tuple[int, ...]:
    return tuple(site for site, _head in _heads(scope))


def head_weights(
    scope: NewlineScope,
    arm: NewlineArm,
    site: int,
    *,
    n_head: int = PRODUCTION_HEADS,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the immutable global head projector for one replaced site."""

    if type(arm) is not NewlineArm or arm is NewlineArm.NATIVE:
        raise ValueError("head weights exist only for a nonnative NewlineArm")
    sites = replacement_sites(scope)
    if type(site) is not int or site not in sites:
        raise ValueError("site is outside the selected newline scope")
    if type(n_head) is not int or n_head <= max(
        head for _site, head in (*FIVE_HEAD_CREW, *FIVE_HEAD_CONTROL)
    ):
        raise ValueError("n_head does not cover the frozen head indices")
    if not dtype.is_floating_point:
        raise ValueError("head weights require a floating dtype")
    weights = torch.ones(n_head, dtype=dtype)
    if arm is NewlineArm.REMOVE:
        head = dict(_heads(scope))[site]
        weights[head] = 0
    elif arm is NewlineArm.HEAD_LABEL_CONTROL:
        head = dict(_heads(scope, control=True))[site]
        weights[head] = 0
    elif arm is not NewlineArm.EXACT:
        raise AssertionError("unreachable NewlineArm")
    return weights


def replacement_id(scope: NewlineScope, arm: NewlineArm, site: int) -> str:
    if site not in replacement_sites(scope) or arm is NewlineArm.NATIVE:
        raise ValueError("replacement ID requires a selected nonnative site")
    return f"newline_v1:{scope.value}:{arm.value}:attention:{site}"


def build_newline_plan(scope: NewlineScope) -> runtime.CircuitPlan:
    """Build the complete immutable native/exact/remove/control topology."""

    native = runtime.ArmPlan.build(
        NewlineArm.NATIVE.value, runtime.ArmKind.NATIVE,
        site_count=PRODUCTION_SITES,
    )
    candidates = []
    for arm in (NewlineArm.EXACT, NewlineArm.REMOVE, NewlineArm.HEAD_LABEL_CONTROL):
        replacements = {
            site: replacement_id(scope, arm, site) for site in replacement_sites(scope)
        }
        candidates.append(runtime.ArmPlan.build(
            arm.value, runtime.ArmKind.CANDIDATE,
            site_count=PRODUCTION_SITES,
            attention_replacements=replacements,
        ))
    return runtime.CircuitPlan(
        f"newline_fixed_crew_v1:{scope.value}",
        PRODUCTION_SITES,
        (native, *candidates),
    )


@dataclass(frozen=True)
class NewlinePrice:
    scope: NewlineScope
    replaced_sites: int
    stored_values_per_site: int
    total_stored_values: int
    operations_per_site: int
    total_operations: int
    token_table_values: int
    native_calls_at_replaced_sites: int
    total_input_support: bool


def newline_price(
    scope: NewlineScope,
    *,
    width: int = PRODUCTION_WIDTH,
    n_head: int = PRODUCTION_HEADS,
    sequence: int = 256,
    batch: int = 1,
) -> NewlinePrice:
    """Literal dense factor, projector, and contraction price for every arm."""

    values = (width, n_head, sequence, batch)
    if any(type(value) is not int or value <= 0 for value in values) or width % n_head:
        raise ValueError("price dimensions must be positive and head-compatible")
    count = len(replacement_sites(scope))
    per_site = 6 * width * width + 1 + width // (2 * n_head) + n_head
    operations = batch * (
        6 * sequence * width * width
        + 3 * sequence * sequence * width
        + sequence * width
    )
    return NewlinePrice(
        scope=scope,
        replaced_sites=count,
        stored_values_per_site=per_site,
        total_stored_values=count * per_site,
        operations_per_site=operations,
        total_operations=count * operations,
        token_table_values=0,
        native_calls_at_replaced_sites=0,
        total_input_support=True,
    )


def expected_call_ledger(
    scope: NewlineScope, arm: NewlineArm,
) -> tuple[runtime.SiteCallLedger, ...]:
    """Expected one-batch component calls for preregistered lifecycle validation."""

    if type(arm) is not NewlineArm:
        raise ValueError("arm must be a NewlineArm")
    replaced = set() if arm is NewlineArm.NATIVE else set(replacement_sites(scope))
    return tuple(
        runtime.SiteCallLedger(
            site=site,
            native_attention_calls=0 if site in replaced else 1,
            replacement_attention_calls=1 if site in replaced else 0,
            native_mlp_calls=1,
            replacement_mlp_calls=0,
        )
        for site in range(PRODUCTION_SITES)
    )


def newline_coordinate_specs(
    scope: NewlineScope,
) -> tuple[statistics.CoordinateSpec, ...]:
    """Return the frozen higher-is-better campaign coordinates for this scope."""

    _heads(scope)
    role = "CANARY_SELECT" if scope is NewlineScope.CANARY else "FINAL"
    prefix = scope.value

    def role_specs(current_role: str, *, include_control: bool) -> list[statistics.CoordinateSpec]:
        draw_group = f"newline_v1:{current_role}"
        specs = [
            statistics.CoordinateSpec(
                f"{prefix}:{current_role}:removal_target",
                statistics.CoordinateKind.TARGET_DAMAGE,
                current_role, "newline_target",
                candidate_arm=NewlineArm.REMOVE.value,
                draw_group=draw_group,
            ),
            statistics.CoordinateSpec(
                f"{prefix}:{current_role}:removal_specificity",
                statistics.CoordinateKind.SPECIFICITY,
                current_role, "newline_target",
                candidate_arm=NewlineArm.REMOVE.value,
                comparison_cell="position_jitter",
                draw_group=draw_group,
            ),
            statistics.CoordinateSpec(
                f"{prefix}:{current_role}:collateral_margin",
                statistics.CoordinateKind.COLLATERAL,
                current_role, "global_off_target",
                candidate_arm=NewlineArm.REMOVE.value,
                limit=0.01,
                draw_group=draw_group,
            ),
            statistics.CoordinateSpec(
                f"{prefix}:{current_role}:extraction_recovery",
                statistics.CoordinateKind.EXTRACTION_RECOVERY,
                current_role, "newline_target",
                candidate_arm=NewlineArm.EXACT.value,
                stake_arm=NewlineArm.REMOVE.value,
                draw_group=draw_group,
            ),
        ]
        if include_control:
            # CE(remove)-CE(head-label-control), expressed in the generic directed
            # damage currency without adding a special scorer or changing signs.
            specs.append(statistics.CoordinateSpec(
                f"{prefix}:{current_role}:removal_over_head_label_control",
                statistics.CoordinateKind.TARGET_DAMAGE,
                current_role, "newline_target",
                native_arm=NewlineArm.HEAD_LABEL_CONTROL.value,
                candidate_arm=NewlineArm.REMOVE.value,
                draw_group=draw_group,
            ))
        return specs

    answer = role_specs(role, include_control=True)
    if scope is NewlineScope.FIVE_HEAD:
        answer.extend(role_specs("OOD", include_control=True))
    return tuple(answer)


@dataclass(frozen=True)
class NewlineExecutorClosure:
    sites: tuple[tuple[int, int], ...]
    ordered: bool
    complete: bool
    closed: bool


class NewlineAttentionExecutor(nn.Module):
    """One-use ordered callback owner for stored newline attention programs."""

    def __init__(
        self,
        *,
        scope: NewlineScope,
        arm: NewlineArm,
        programs: Mapping[int, TensorPreservingSquaredAttention],
    ) -> None:
        super().__init__()
        if arm is NewlineArm.NATIVE:
            raise ValueError("native arm has no replacement executor")
        expected = replacement_sites(scope)
        if tuple(sorted(programs)) != expected or any(
            not isinstance(program, TensorPreservingSquaredAttention)
            for program in programs.values()
        ):
            raise ValueError("newline programs do not exactly cover the selected sites")
        self.scope = scope
        self.arm = arm
        self.programs = nn.ModuleDict({str(site): programs[site] for site in expected})
        self._expected = expected
        self._next = 0
        self._closed = False

    @classmethod
    def from_model(
        cls, model: nn.Module, *, scope: NewlineScope, arm: NewlineArm,
    ) -> "NewlineAttentionExecutor":
        try:
            blocks = tuple(model.transformer.h)
            parameter = next(model.parameters())
        except (AttributeError, StopIteration) as error:
            raise ValueError("model does not expose the required attention topology") from error
        if len(blocks) != PRODUCTION_SITES:
            raise ValueError("newline executor requires exactly 18 model blocks")
        programs = {
            site: TensorPreservingSquaredAttention.from_native(
                blocks[site].attn,
                ranks={name: None for name in PROJECTION_NAMES},
                head_weights=head_weights(
                    scope, arm, site,
                    n_head=int(blocks[site].attn.n_head),
                    dtype=parameter.dtype,
                ),
            )
            for site in replacement_sites(scope)
        }
        return cls(scope=scope, arm=arm, programs=programs).to(
            device=parameter.device, dtype=parameter.dtype,
        )

    def _call(self, event: runtime.AttentionReplacementEvent):
        if self._closed or self._next >= len(self._expected):
            raise RuntimeError("newline attention executor is closed")
        site = self._expected[self._next]
        if type(event) is not runtime.AttentionReplacementEvent or event.site != site:
            self._closed = True
            raise RuntimeError("newline replacement site is skipped, repeated, or reordered")
        write, bus = self.programs[str(site)](event.state, event.first_value)
        self._next += 1
        if self._next == len(self._expected):
            self._closed = True
        return write, bus

    def callbacks(self) -> Mapping[str, runtime.AttentionReplacement]:
        if self._closed or self._next != 0:
            raise RuntimeError("newline attention executor is already active or closed")
        return MappingProxyType({
            replacement_id(self.scope, self.arm, site): self._call
            for site in self._expected
        })

    @property
    def closure(self) -> NewlineExecutorClosure:
        if not self._closed:
            raise RuntimeError("newline executor has not completed")
        sites = tuple((site, int(index < self._next)) for index, site in enumerate(self._expected))
        complete = self._next == len(self._expected)
        return NewlineExecutorClosure(sites, complete, complete, True)


@dataclass(frozen=True)
class NewlineMaskSpec:
    newline_token_ids: tuple[int, ...]
    punctuation_token_ids: tuple[int, ...]
    capitalized_token_ids: tuple[int, ...]
    quote_bracket_token_ids: tuple[int, ...]
    first_prediction: int = 64
    jitter_offsets: tuple[int, ...] = tuple(
        offset for distance in range(2, 33) for offset in (distance, -distance)
    )
    random_seed: int = 2_026_083_000

    def __post_init__(self) -> None:
        groups = (
            self.newline_token_ids, self.punctuation_token_ids,
            self.capitalized_token_ids, self.quote_bracket_token_ids,
        )
        if any(not group or len(group) != len(set(group)) for group in groups):
            raise ValueError("every mask token group must be nonempty and unique")
        if any(type(token) is not int or token < 0 for group in groups for token in group):
            raise ValueError("mask token IDs must be nonnegative integers")
        flattened = [token for group in groups for token in group]
        if len(flattened) != len(set(flattened)):
            raise ValueError("named newline mask token groups must be disjoint")
        if type(self.first_prediction) is not int or self.first_prediction < 0:
            raise ValueError("first_prediction must be nonnegative")
        if not self.jitter_offsets or 0 in self.jitter_offsets or len(
            self.jitter_offsets
        ) != len(set(self.jitter_offsets)):
            raise ValueError("jitter offsets must be unique and nonzero")
        if type(self.random_seed) is not int or self.random_seed < 0:
            raise ValueError("random_seed must be a nonnegative integer")


@dataclass(frozen=True)
class NewlineMasks:
    newline_target: torch.Tensor
    position_jitter: torch.Tensor
    matched_random: torch.Tensor
    punctuation: torch.Tensor
    capitalized: torch.Tensor
    quote_bracket: torch.Tensor
    global_off_target: torch.Tensor

    def as_mapping(self) -> Mapping[str, torch.Tensor]:
        return MappingProxyType({
            "newline_target": self.newline_target,
            "position_jitter": self.position_jitter,
            "matched_random": self.matched_random,
            "punctuation": self.punctuation,
            "capitalized": self.capitalized,
            "quote_bracket": self.quote_bracket,
            "global_off_target": self.global_off_target,
        })


def _member(values: torch.Tensor, token_ids: Sequence[int]) -> torch.Tensor:
    result = torch.zeros_like(values, dtype=torch.bool)
    for token in token_ids:
        result |= values.eq(token)
    return result


def _matched_controls(
    target: torch.Tensor,
    forbidden: torch.Tensor,
    offsets: Sequence[int],
    random_seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    jitter = torch.zeros_like(target)
    random = torch.zeros_like(target)
    width = target.shape[1]
    generator = torch.Generator().manual_seed(random_seed)
    for row in range(target.shape[0]):
        target_positions = target[row].nonzero(as_tuple=False).flatten().tolist()
        available = (~forbidden[row]).clone()
        for position in target_positions:
            chosen = None
            for offset in offsets:
                candidate = position + offset
                if 0 <= candidate < width and bool(available[candidate]):
                    chosen = candidate
                    break
            if chosen is None:
                raise RuntimeError("document cannot supply exact position-jitter support")
            jitter[row, chosen] = True
            available[chosen] = False
        candidates = available.nonzero(as_tuple=False).flatten()
        if len(candidates) < len(target_positions):
            raise RuntimeError("document cannot supply exact random-control support")
        if target_positions:
            order = torch.randperm(len(candidates), generator=generator)[:len(target_positions)]
            random[row, candidates[order]] = True
    return jitter, random


def build_newline_masks(rows: torch.Tensor, spec: NewlineMaskSpec) -> NewlineMasks:
    """Construct document-matched CPU masks; these values never enter callbacks."""

    if not torch.is_tensor(rows) or rows.device.type != "cpu" or rows.dtype != torch.long or (
        rows.ndim != 2 or rows.shape[1] < 2
    ):
        raise ValueError("rows must be a rank-2 CPU int64 tensor")
    prediction_count = rows.shape[1] - 1
    if spec.first_prediction >= prediction_count:
        raise ValueError("first_prediction is outside the row")
    targets = rows[:, 1:]
    scored = torch.zeros_like(targets, dtype=torch.bool)
    scored[:, spec.first_prediction:] = True
    newline = _member(targets, spec.newline_token_ids) & scored
    punctuation = _member(targets, spec.punctuation_token_ids) & scored
    capitalized = _member(targets, spec.capitalized_token_ids) & scored
    quote_bracket = _member(targets, spec.quote_bracket_token_ids) & scored
    named = newline | punctuation | capitalized | quote_bracket | ~scored
    jitter, random = _matched_controls(
        newline, named, spec.jitter_offsets, spec.random_seed,
    )
    masks = NewlineMasks(
        newline_target=newline,
        position_jitter=jitter,
        matched_random=random,
        punctuation=punctuation,
        capitalized=capitalized,
        quote_bracket=quote_bracket,
        global_off_target=scored & ~newline,
    )
    disjoint = torch.stack([
        masks.newline_target, masks.position_jitter, masks.matched_random,
        masks.punctuation, masks.capitalized, masks.quote_bracket,
    ]).to(torch.int8).sum(0)
    if bool((disjoint > 1).any()):
        raise AssertionError("named newline score cells overlap")
    if not torch.equal(masks.position_jitter.sum(1), masks.newline_target.sum(1)) or not (
        torch.equal(masks.matched_random.sum(1), masks.newline_target.sum(1))
    ):
        raise AssertionError("newline controls are not document-count matched")
    return masks


__all__ = (
    "CANARY_HEAD",
    "FIVE_HEAD_CONTROL",
    "FIVE_HEAD_CREW",
    "NewlineArm",
    "NewlineAttentionExecutor",
    "NewlineExecutorClosure",
    "NewlineMaskSpec",
    "NewlineMasks",
    "NewlinePrice",
    "NewlineScope",
    "build_newline_masks",
    "build_newline_plan",
    "expected_call_ledger",
    "head_weights",
    "newline_coordinate_specs",
    "newline_price",
    "replacement_id",
    "replacement_sites",
)
