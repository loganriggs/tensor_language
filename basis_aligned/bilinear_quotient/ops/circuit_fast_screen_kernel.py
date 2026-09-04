"""Model-independent scoring kernel for cheap causal-circuit screens.

The kernel consumes scalar evidence produced by a model adapter; it does not
know how residual, module, or head interventions are executed.  Scores use one
common convention:

* A1 and A2 are answer-changing target families.
* P is a same-answer invariance family.
* C is a registered active negative control.  It may either change an
  unrelated answer or preserve the answer while changing a known distractor.
  Movement at C is evidence against selectivity in both cases.

All answer-changing records must express ``base_score``, ``donor_score``, and
``intervened_score`` on the same donor-oriented axis.  Recovery is deliberately
unclipped.  A weak or sign-reversed donor denominator is invalid evidence, not
something to repair with an epsilon clamp.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Collection, Literal, Protocol, Sequence, runtime_checkable


Family = Literal["A1", "A2", "P", "C"]
EvidenceKind = Literal["residual", "module", "head"]
Terminal = Literal["screen", "null", "invalid"]


# These are screen gates, not fit parameters.  Callers cannot override them.
MIN_DONOR_DENOMINATOR = 1.0e-6
MIN_A1_CAPABILITY_ACCURACY = 0.85
MIN_A2_CAPABILITY_ACCURACY = 0.85
MIN_P_CAPABILITY_ACCURACY = 0.85
MIN_C_CAPABILITY_ACCURACY = 0.75
MIN_TARGET_DIRECTION_FRACTION = 0.80
MIN_TARGET_FAMILY_RECOVERY = 0.50
MAX_P_INVARIANCE_EFFECT = 0.20
MAX_C_ABSOLUTE_RECOVERY = 0.35


@dataclass(frozen=True)
class FixedBars:
    minimum_donor_denominator: float = MIN_DONOR_DENOMINATOR
    minimum_a1_capability_accuracy: float = MIN_A1_CAPABILITY_ACCURACY
    minimum_a2_capability_accuracy: float = MIN_A2_CAPABILITY_ACCURACY
    minimum_p_capability_accuracy: float = MIN_P_CAPABILITY_ACCURACY
    minimum_c_capability_accuracy: float = MIN_C_CAPABILITY_ACCURACY
    minimum_target_direction_fraction: float = MIN_TARGET_DIRECTION_FRACTION
    minimum_target_family_recovery: float = MIN_TARGET_FAMILY_RECOVERY
    maximum_p_invariance_effect: float = MAX_P_INVARIANCE_EFFECT
    maximum_c_absolute_recovery: float = MAX_C_ABSOLUTE_RECOVERY


FIXED_BARS = FixedBars()


class InvalidEvidenceError(ValueError):
    """A scalar cannot support a scientific screen."""


@dataclass(frozen=True, order=True)
class SiteRef:
    """A model adapter's stable name for one intervention site."""

    evidence_kind: EvidenceKind
    site_id: str


@runtime_checkable
class InterventionEvidenceRecord(Protocol):
    """Structural interface implemented by model-specific evidence records."""

    @property
    def record_id(self) -> str: ...

    @property
    def pair_id(self) -> str: ...

    @property
    def family(self) -> Family: ...

    @property
    def evidence_kind(self) -> EvidenceKind: ...

    @property
    def site_id(self) -> str: ...

    @property
    def base_score(self) -> float: ...

    @property
    def donor_score(self) -> float | None: ...

    @property
    def intervened_score(self) -> float: ...

    @property
    def effect_scale(self) -> float | None: ...

    @property
    def complete(self) -> bool: ...


class ResidualEvidenceRecord(InterventionEvidenceRecord, Protocol):
    """Adapter protocol for a residual-stream intervention record."""

    @property
    def evidence_kind(self) -> Literal["residual"]: ...


class ModuleEvidenceRecord(InterventionEvidenceRecord, Protocol):
    """Adapter protocol for a whole-module intervention record."""

    @property
    def evidence_kind(self) -> Literal["module"]: ...


class HeadEvidenceRecord(InterventionEvidenceRecord, Protocol):
    """Adapter protocol for an attention-head intervention record."""

    @property
    def evidence_kind(self) -> Literal["head"]: ...


@dataclass(frozen=True)
class ScalarInterventionEvidence:
    """Convenient immutable implementation of the adapter protocol.

    ``effect_scale`` is required for same-answer P records and same-answer C
    records.  It is a prospectively registered positive task-effect unit,
    never inferred from the control pair. ``donor_score`` is required for A1,
    A2, and answer-changing C records.
    """

    record_id: str
    pair_id: str
    family: Family
    evidence_kind: EvidenceKind
    site_id: str
    base_score: float
    donor_score: float | None
    intervened_score: float
    effect_scale: float | None = None
    complete: bool = True


@dataclass(frozen=True)
class FamilyCapabilityEvidence:
    """Native capability and exact coverage for one ordered construction cell."""

    family: Family
    correct_count: int
    observed_count: int
    expected_count: int
    complete: bool = True
    cell_id: str = "all"


@dataclass(frozen=True)
class CapabilityEvidence:
    """Complete, separately thresholded native evidence for all families."""

    families: tuple[FamilyCapabilityEvidence, ...]
    complete: bool = True


@dataclass(frozen=True)
class FamilyCapabilityScore:
    family: Family
    cell_id: str
    correct_count: int
    expected_count: int
    accuracy: float
    minimum_accuracy: float


@dataclass(frozen=True)
class FamilyScore:
    family: Family
    record_count: int
    mean_effect: float
    mean_absolute_effect: float
    direction_fraction: float | None


@dataclass(frozen=True)
class SiteScreenResult:
    site: SiteRef
    terminal: Terminal
    reasons: tuple[str, ...]
    evidence_count: int
    capability: tuple[FamilyCapabilityScore, ...] | None
    a1: FamilyScore | None
    a2: FamilyScore | None
    target_recovery: float | None
    p_invariance_effect: float | None
    c_absolute_recovery: float | None
    c_signed_recovery: float | None
    c_direction_fraction: float | None


@dataclass(frozen=True)
class RankedSite:
    rank: int
    result: SiteScreenResult


def _finite_number(value: object, label: str) -> float:
    if type(value) not in {int, float}:
        raise InvalidEvidenceError(f"{label} must be a real scalar")
    converted = float(value)
    if not math.isfinite(converted):
        raise InvalidEvidenceError(f"{label} must be finite")
    return converted


def signed_pairwise_donor_recovery(
    base_score: float,
    donor_score: float,
    intervened_score: float,
) -> float:
    """Return signed donor recovery without clipping or denominator repair.

    Scores must already share a donor-oriented axis.  Thus the native donor
    effect ``donor_score - base_score`` must be strictly greater than the fixed
    denominator floor.  Negative, zero, and weak positive denominators all
    raise :class:`InvalidEvidenceError`.
    """

    base = _finite_number(base_score, "base_score")
    donor = _finite_number(donor_score, "donor_score")
    intervened = _finite_number(intervened_score, "intervened_score")
    denominator = donor - base
    if not math.isfinite(denominator):
        raise InvalidEvidenceError("donor denominator must be finite")
    if denominator <= MIN_DONOR_DENOMINATOR:
        raise InvalidEvidenceError(
            "donor denominator must be positive and greater than 1e-6"
        )
    recovery = (intervened - base) / denominator
    if not math.isfinite(recovery):  # Defensive against finite overflow.
        raise InvalidEvidenceError("donor recovery must be finite")
    return recovery


def normalized_same_answer_effect(
    base_score: float,
    intervened_score: float,
    effect_scale: float,
) -> float:
    """Return absolute P movement in a prospectively registered effect unit."""

    base = _finite_number(base_score, "base_score")
    intervened = _finite_number(intervened_score, "intervened_score")
    scale = _finite_number(effect_scale, "effect_scale")
    if scale <= MIN_DONOR_DENOMINATOR:
        raise InvalidEvidenceError("effect_scale must be greater than 1e-6")
    effect = abs(intervened - base) / scale
    if not math.isfinite(effect):
        raise InvalidEvidenceError("same-answer effect must be finite")
    return effect


def _invalid(site: SiteRef, evidence_count: int, reason: str) -> SiteScreenResult:
    return SiteScreenResult(
        site=site,
        terminal="invalid",
        reasons=(reason,),
        evidence_count=evidence_count,
        capability=None,
        a1=None,
        a2=None,
        target_recovery=None,
        p_invariance_effect=None,
        c_absolute_recovery=None,
        c_signed_recovery=None,
        c_direction_fraction=None,
    )


def _validate_site(site: SiteRef) -> None:
    if not isinstance(site, SiteRef):
        raise InvalidEvidenceError("site must be a SiteRef")
    if site.evidence_kind not in {"residual", "module", "head"}:
        raise InvalidEvidenceError("site evidence_kind is invalid")
    if type(site.site_id) is not str or not site.site_id:
        raise InvalidEvidenceError("site_id must be nonempty")


def _capability_scores(
    capability: CapabilityEvidence,
) -> tuple[FamilyCapabilityScore, ...]:
    if not isinstance(capability, CapabilityEvidence):
        raise InvalidEvidenceError("capability must be CapabilityEvidence")
    if type(capability.complete) is not bool or not capability.complete:
        raise InvalidEvidenceError("capability evidence is incomplete")
    if type(capability.families) is not tuple:
        raise InvalidEvidenceError("capability families must be a tuple")
    thresholds = {
        "A1": MIN_A1_CAPABILITY_ACCURACY,
        "A2": MIN_A2_CAPABILITY_ACCURACY,
        "P": MIN_P_CAPABILITY_ACCURACY,
        "C": MIN_C_CAPABILITY_ACCURACY,
    }
    cells: dict[tuple[Family, str], FamilyCapabilityEvidence] = {}
    for cell in capability.families:
        if not isinstance(cell, FamilyCapabilityEvidence) or cell.family not in thresholds:
            raise InvalidEvidenceError("capability family evidence is malformed")
        if type(cell.cell_id) is not str or not cell.cell_id:
            raise InvalidEvidenceError("capability cell_id must be nonempty")
        key = (cell.family, cell.cell_id)
        if key in cells:
            raise InvalidEvidenceError(
                f"duplicate capability cell {cell.family}/{cell.cell_id}"
            )
        if type(cell.complete) is not bool or not cell.complete:
            raise InvalidEvidenceError(f"capability family {cell.family} is incomplete")
        for name in ("correct_count", "observed_count", "expected_count"):
            if type(getattr(cell, name)) is not int:
                raise InvalidEvidenceError(
                    f"capability {cell.family} {name} must be an integer"
                )
        if cell.expected_count <= 0:
            raise InvalidEvidenceError(
                f"capability {cell.family} expected_count must be positive"
            )
        if cell.observed_count != cell.expected_count:
            raise InvalidEvidenceError(f"capability {cell.family} coverage is incomplete")
        if not 0 <= cell.correct_count <= cell.observed_count:
            raise InvalidEvidenceError(
                f"capability {cell.family} correct_count is out of range"
            )
        cells[key] = cell
    if {family for family, _ in cells} != set(thresholds):
        raise InvalidEvidenceError("capability must cover A1, A2, P, and C")
    return tuple(
        FamilyCapabilityScore(
            cell.family,
            cell.cell_id,
            cell.correct_count,
            cell.expected_count,
            cell.correct_count / cell.expected_count,
            thresholds[cell.family],
        )
        for family in ("A1", "A2", "P", "C")
        for _, cell in sorted(
            (item for item in cells.items() if item[0][0] == family),
            key=lambda item: item[0][1],
        )
    )


def _family_score(
    family: Family, effects: Sequence[float], *, direction_defined: bool = True,
) -> FamilyScore:
    if not effects:
        raise InvalidEvidenceError(f"missing required family {family}")
    mean = math.fsum(effects) / len(effects)
    mean_absolute = math.fsum(abs(value) for value in effects) / len(effects)
    if not math.isfinite(mean) or not math.isfinite(mean_absolute):
        raise InvalidEvidenceError(f"{family} mean effect must be finite")
    direction = (
        sum(value > 0.0 for value in effects) / len(effects)
        if direction_defined else None
    )
    return FamilyScore(family, len(effects), mean, mean_absolute, direction)


def score_site(
    site: SiteRef,
    *,
    evidence: Sequence[InterventionEvidenceRecord],
    expected_record_ids: Collection[str],
    capability: CapabilityEvidence,
    c_answer_changes: bool = True,
) -> SiteScreenResult:
    """Validate and score one site, returning only screen/null/invalid.

    Expected record identities are mandatory so a missing family, dropped hard
    example, duplicate, or unexpected model output cannot silently change an
    aggregate.  Invalid evidence suppresses all scientific score fields.
    """

    try:
        _validate_site(site)
        if type(c_answer_changes) is not bool:
            raise InvalidEvidenceError("c_answer_changes must be a literal bool")
        capability_scores = _capability_scores(capability)
        if type(expected_record_ids) in {str, bytes}:
            raise InvalidEvidenceError("expected_record_ids must be a collection of IDs")
        expected = tuple(expected_record_ids)
        if not expected or any(type(item) is not str or not item for item in expected):
            raise InvalidEvidenceError("expected record IDs must be nonempty strings")
        if len(expected) != len(set(expected)):
            raise InvalidEvidenceError("expected record IDs contain duplicates")

        records = tuple(evidence)
        observed_ids: list[str] = []
        observed_pairs: set[tuple[Family, str]] = set()
        effects: dict[Family, list[float]] = {"A1": [], "A2": [], "P": [], "C": []}
        for record in records:
            record_id = record.record_id
            if type(record_id) is not str or not record_id:
                raise InvalidEvidenceError("record_id must be nonempty")
            observed_ids.append(record_id)
            if type(record.pair_id) is not str or not record.pair_id:
                raise InvalidEvidenceError(f"record {record_id} has invalid pair_id")
            if type(record.complete) is not bool or not record.complete:
                raise InvalidEvidenceError(f"record {record_id} is incomplete")
            if record.evidence_kind != site.evidence_kind or record.site_id != site.site_id:
                raise InvalidEvidenceError(f"record {record_id} belongs to a different site")
            family = record.family
            if family not in effects:
                raise InvalidEvidenceError(f"record {record_id} has invalid family")
            pair_key = (family, record.pair_id)
            if pair_key in observed_pairs:
                raise InvalidEvidenceError(
                    f"family {family} contains duplicate pair_id {record.pair_id}"
                )
            observed_pairs.add(pair_key)
            same_answer = family == "P" or (family == "C" and not c_answer_changes)
            if same_answer:
                if record.donor_score is not None:
                    raise InvalidEvidenceError(
                        f"same-answer record {record_id} must not define donor_score"
                    )
                if record.effect_scale is None:
                    raise InvalidEvidenceError(
                        f"same-answer record {record_id} lacks effect_scale"
                    )
                effect = normalized_same_answer_effect(
                    record.base_score, record.intervened_score, record.effect_scale
                )
            else:
                if record.donor_score is None:
                    raise InvalidEvidenceError(f"record {record_id} lacks donor_score")
                if record.effect_scale is not None:
                    raise InvalidEvidenceError(
                        f"answer-changing record {record_id} must not define effect_scale"
                    )
                effect = signed_pairwise_donor_recovery(
                    record.base_score, record.donor_score, record.intervened_score
                )
            effects[family].append(effect)

        if len(observed_ids) != len(set(observed_ids)):
            raise InvalidEvidenceError("observed evidence contains duplicate record IDs")
        if set(observed_ids) != set(expected) or len(observed_ids) != len(expected):
            missing = sorted(set(expected) - set(observed_ids))
            extra = sorted(set(observed_ids) - set(expected))
            raise InvalidEvidenceError(
                f"evidence coverage mismatch; missing={missing}, extra={extra}"
            )

        a1 = _family_score("A1", effects["A1"])
        a2 = _family_score("A2", effects["A2"])
        p = _family_score("P", effects["P"], direction_defined=False)
        c = _family_score(
            "C", effects["C"], direction_defined=c_answer_changes
        )
        assert a1.direction_fraction is not None
        assert a2.direction_fraction is not None

        # Equal family weighting prevents a larger A1 or A2 cell from silently
        # dominating the target decision.
        target_recovery = math.fsum((a1.mean_effect, a2.mean_effect)) / 2.0
        if not math.isfinite(target_recovery):
            raise InvalidEvidenceError("target recovery must be finite")
        failures: list[str] = []
        for cell in capability_scores:
            if cell.accuracy < cell.minimum_accuracy:
                failures.append(
                    f"{cell.family}/{cell.cell_id}_capability_below_fixed_bar"
                )
        if a1.direction_fraction < MIN_TARGET_DIRECTION_FRACTION:
            failures.append("A1_direction_below_fixed_bar")
        if a2.direction_fraction < MIN_TARGET_DIRECTION_FRACTION:
            failures.append("A2_direction_below_fixed_bar")
        if a1.mean_effect < MIN_TARGET_FAMILY_RECOVERY:
            failures.append("A1_recovery_below_fixed_bar")
        if a2.mean_effect < MIN_TARGET_FAMILY_RECOVERY:
            failures.append("A2_recovery_below_fixed_bar")
        if p.mean_effect > MAX_P_INVARIANCE_EFFECT:
            failures.append("P_invariance_above_fixed_bar")
        if c.mean_absolute_effect > MAX_C_ABSOLUTE_RECOVERY:
            failures.append("C_absolute_recovery_above_fixed_bar")

        return SiteScreenResult(
            site=site,
            terminal="null" if failures else "screen",
            reasons=tuple(failures),
            evidence_count=len(records),
            capability=capability_scores,
            a1=a1,
            a2=a2,
            target_recovery=target_recovery,
            p_invariance_effect=p.mean_absolute_effect,
            c_absolute_recovery=c.mean_absolute_effect,
            c_signed_recovery=c.mean_effect,
            c_direction_fraction=c.direction_fraction,
        )
    except (AttributeError, OverflowError, TypeError, InvalidEvidenceError) as error:
        try:
            evidence_count = len(evidence)
        except TypeError:
            evidence_count = 0
        return _invalid(site, evidence_count, str(error))


def rank_sites(results: Sequence[SiteScreenResult]) -> tuple[RankedSite, ...]:
    """Rank sites deterministically, with invalid instruments last.

    Scientific order is terminal, target recovery (descending), P invariance
    (ascending), absolute C transfer (ascending), then residual/module/head and site ID.
    The final two fields are the stable exact-tie break.
    """

    values = tuple(results)
    sites = [item.site for item in values]
    if len(sites) != len(set(sites)):
        raise ValueError("site results must be unique")
    terminal_order = {"screen": 0, "null": 1, "invalid": 2}
    kind_order = {"residual": 0, "module": 1, "head": 2}

    def descending(value: float | None) -> float:
        return math.inf if value is None else -value

    def ascending(value: float | None) -> float:
        return math.inf if value is None else value

    ordered = sorted(
        values,
        key=lambda item: (
            terminal_order[item.terminal],
            descending(item.target_recovery),
            ascending(item.p_invariance_effect),
            ascending(item.c_absolute_recovery),
            kind_order[item.site.evidence_kind],
            item.site.site_id,
        ),
    )
    return tuple(RankedSite(index + 1, item) for index, item in enumerate(ordered))
