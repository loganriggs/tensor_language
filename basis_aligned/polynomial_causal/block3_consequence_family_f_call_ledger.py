"""Pure call-accounting contract for the frozen Block-3 Family-F fit.

The numerical runner records physical calls against named phase/arm cells.  Closing
the ledger requires both every cell and every derived global/site total to equal the
prospectively frozen schedule.  This module performs no file, row, model, or GPU I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


SCHEMA = "block3_consequence_family_f_v1_call_ledger"

SCORE_ARMS = (
    "teacher",
    "teacher_row_reversal",
    "teacher_document_derangement",
)

AFFINE_ARMS = (
    "teacher_F_k512",
    "family_A_k512",
    "random_k512",
    "same_support_permuted_cross_k512",
)

REPORT_SHARED_ARM = "shared_native_teacher"
OUTER_REPLAY_ARM = "ordinary_full_model_raw_logit_replay"
REPORT_STUDENT_ARMS = (
    "continuous_teacher_F1",
    "real_F_binary_native_down_k256",
    "real_F_post_refit_k256",
    "random_post_refit_k256",
    "same_support_permuted_cross_post_refit_k256",
    "row_reversal_selector_post_refit_k256",
    "document_derangement_selector_post_refit_k256",
    "real_F_binary_native_down_k512",
    "real_F_post_refit_k512",
    "random_post_refit_k512",
    "same_support_permuted_cross_post_refit_k512",
    "row_reversal_selector_post_refit_k512",
    "document_derangement_selector_post_refit_k512",
    "family_A_uncalibrated_k512",
    "affine_teacher_F_k512",
    "affine_family_A_k512",
    "affine_random_k512",
    "affine_same_support_permuted_cross_k512",
)


@dataclass(frozen=True, slots=True)
class CellCounts:
    """Literal physical work attributed to one registered phase/arm cell."""

    optimizer_steps: int = 0
    two_row_backwards: int = 0
    prefixes: int = 0
    teacher_suffixes: int = 0
    student_suffixes: int = 0
    student_native_mlp3_calls: int = 0
    donor_prefixes: int = 0
    projections: int = 0
    outer_model_calls: int = 0
    outer_model_returns: int = 0

    def __post_init__(self) -> None:
        for name in self.field_names():
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative literal integer")

    @staticmethod
    def field_names() -> tuple[str, ...]:
        return (
            "optimizer_steps",
            "two_row_backwards",
            "prefixes",
            "teacher_suffixes",
            "student_suffixes",
            "student_native_mlp3_calls",
            "donor_prefixes",
            "projections",
            "outer_model_calls",
            "outer_model_returns",
        )

    def plus(self, other: "CellCounts") -> "CellCounts":
        return CellCounts(**{
            name: getattr(self, name) + getattr(other, name)
            for name in self.field_names()
        })

    def as_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.field_names()}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CellCounts":
        if not isinstance(value, Mapping) or set(value) != set(cls.field_names()):
            raise ValueError("call-ledger cell schema changed")
        return cls(**{name: value[name] for name in cls.field_names()})


def expected_cells() -> dict[tuple[str, str], CellCounts]:
    """Return a fresh copy of the exact prospectively frozen cell schedule."""

    cells: dict[tuple[str, str], CellCounts] = {}
    for arm in SCORE_ARMS:
        cells[("score_fit", arm)] = CellCounts(
            optimizer_steps=480,
            two_row_backwards=1_920,
            prefixes=960 if arm == "teacher_document_derangement" else 480,
            teacher_suffixes=480,
            student_suffixes=1_920,
            donor_prefixes=480 if arm == "teacher_document_derangement" else 0,
            projections=480,
        )
    for arm in AFFINE_ARMS:
        cells[("affine_fit", arm)] = CellCounts(
            optimizer_steps=240,
            two_row_backwards=960,
            prefixes=240,
            teacher_suffixes=240,
            student_suffixes=960,
        )
    cells[("postfit_report", REPORT_SHARED_ARM)] = CellCounts(
        prefixes=60, teacher_suffixes=60,
    )
    for arm in REPORT_STUDENT_ARMS:
        cells[("postfit_report", arm)] = CellCounts(student_suffixes=60)
    cells[("known_answer", OUTER_REPLAY_ARM)] = CellCounts(
        outer_model_calls=1, outer_model_returns=1,
    )
    return cells


def _cell_key(phase: str, arm: str) -> tuple[str, str]:
    if type(phase) is not str or type(arm) is not str:
        raise ValueError("phase and arm must be literal strings")
    key = (phase, arm)
    if key not in expected_cells():
        raise ValueError(f"unregistered Family-F phase/arm cell: {phase}/{arm}")
    return key


def _sum_cells(cells: Mapping[tuple[str, str], CellCounts]) -> CellCounts:
    total = CellCounts()
    for counts in cells.values():
        total = total.plus(counts)
    return total


def _derived_totals(cells: Mapping[tuple[str, str], CellCounts]) -> dict[str, Any]:
    raw = _sum_cells(cells)
    suffix_returns = raw.teacher_suffixes + raw.student_suffixes
    attention_by_site = {
        str(site): (
            raw.prefixes if site <= 3 else suffix_returns
        ) + raw.outer_model_calls
        for site in range(18)
    }
    mlp_by_site = {
        str(site): (
            raw.prefixes + raw.student_native_mlp3_calls
            if site == 3 else raw.prefixes if site < 3 else suffix_returns
        ) + raw.outer_model_calls
        for site in range(18)
    }
    return {
        **raw.as_dict(),
        "suffix_returns": suffix_returns,
        "raw_logit_returns": suffix_returns + raw.outer_model_returns,
        "attention_calls_by_site": attention_by_site,
        "mlp_calls_by_site": mlp_by_site,
    }


def expected_totals() -> dict[str, Any]:
    """Return a fresh, mutation-independent copy of the frozen global census."""

    return {
        "optimizer_steps": 2_400,
        "two_row_backwards": 9_600,
        "prefixes": 2_940,
        "teacher_suffixes": 2_460,
        "student_suffixes": 10_680,
        "student_native_mlp3_calls": 0,
        "donor_prefixes": 480,
        "projections": 1_440,
        "outer_model_calls": 1,
        "outer_model_returns": 1,
        "suffix_returns": 13_140,
        "raw_logit_returns": 13_141,
        "attention_calls_by_site": {
            str(site): 2_941 if site <= 3 else 13_141 for site in range(18)
        },
        "mlp_calls_by_site": {
            str(site): 2_941 if site <= 3 else 13_141 for site in range(18)
        },
    }


# Convenience snapshot only. Validation always calls expected_totals() afresh.
EXPECTED_TOTALS = expected_totals()


class FamilyFCallLedger:
    """Mutable during measurement, then validated and sealed exactly once."""

    __slots__ = ("_cells", "_closed")

    def __init__(self) -> None:
        self._cells = {key: CellCounts() for key in expected_cells()}
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def record(
        self,
        phase: str,
        arm: str,
        *,
        optimizer_steps: int = 0,
        two_row_backwards: int = 0,
        prefixes: int = 0,
        teacher_suffixes: int = 0,
        student_suffixes: int = 0,
        student_native_mlp3_calls: int = 0,
        donor_prefixes: int = 0,
        projections: int = 0,
        outer_model_calls: int = 0,
        outer_model_returns: int = 0,
    ) -> None:
        if self._closed:
            raise RuntimeError("Family-F call ledger is already closed")
        key = _cell_key(phase, arm)
        increment = CellCounts(
            optimizer_steps=optimizer_steps,
            two_row_backwards=two_row_backwards,
            prefixes=prefixes,
            teacher_suffixes=teacher_suffixes,
            student_suffixes=student_suffixes,
            student_native_mlp3_calls=student_native_mlp3_calls,
            donor_prefixes=donor_prefixes,
            projections=projections,
            outer_model_calls=outer_model_calls,
            outer_model_returns=outer_model_returns,
        )
        updated = self._cells[key].plus(increment)
        expected = expected_cells()[key]
        if any(
            getattr(updated, name) > getattr(expected, name)
            for name in CellCounts.field_names()
        ):
            raise RuntimeError(f"Family-F call cell exceeded protocol: {phase}/{arm}")
        self._cells[key] = updated

    def record_prefix(self, phase: str, arm: str, *, donor: bool = False) -> None:
        """Record one prefix; donor prefixes are legal only for the derangement arm."""

        if type(donor) is not bool:
            raise ValueError("donor must be a literal bool")
        key = _cell_key(phase, arm)
        if donor and key != ("score_fit", "teacher_document_derangement"):
            raise RuntimeError("donor prefix used outside the document-derangement arm")
        self.record(phase, arm, prefixes=1, donor_prefixes=int(donor))

    def record_teacher_suffix(self, phase: str, arm: str) -> None:
        self.record(phase, arm, teacher_suffixes=1)

    def record_student_suffix(self, phase: str, arm: str) -> None:
        self.record(phase, arm, student_suffixes=1)

    def record_optimizer_step(
        self, phase: str, arm: str, *, backwards: int = 4,
        projection: bool = False,
    ) -> None:
        """Record one Adam update after its registered microbatch backwards."""

        if type(projection) is not bool:
            raise ValueError("projection must be a literal bool")
        self.record(
            phase, arm, optimizer_steps=1, two_row_backwards=backwards,
            projections=int(projection),
        )

    def record_outer_replay(self) -> None:
        """Record the one independent ordinary full-model raw-logit replay."""

        self.record(
            "known_answer", OUTER_REPLAY_ARM,
            outer_model_calls=1, outer_model_returns=1,
        )

    def partial_receipt(self) -> dict[str, Any]:
        """Return a nonauthoritative snapshot suitable for a failure record."""

        return self._receipt(status="partial_nonauthoritative", complete=False)

    def close(self) -> dict[str, Any]:
        expected = expected_cells()
        if self._cells != expected:
            mismatches = [
                f"{phase}/{arm}"
                for (phase, arm), counts in self._cells.items()
                if counts != expected[(phase, arm)]
            ]
            raise RuntimeError(
                "Family-F call ledger is incomplete or misattributed: "
                + ", ".join(mismatches)
            )
        totals = _derived_totals(self._cells)
        if totals != expected_totals():
            raise RuntimeError("Family-F derived physical call census changed")
        self._closed = True
        return self._receipt(status="complete_exact_frozen_census", complete=True)

    def validate_exact(self) -> dict[str, Any]:
        """Validate every cell and derived total, then seal the ledger."""

        if self._closed:
            return self.receipt()
        return self.close()

    def receipt(self) -> dict[str, Any]:
        if not self._closed:
            raise RuntimeError("Family-F call ledger has not closed")
        return self._receipt(status="complete_exact_frozen_census", complete=True)

    def _receipt(self, *, status: str, complete: bool) -> dict[str, Any]:
        cells = {
            f"{phase}/{arm}": counts.as_dict()
            for (phase, arm), counts in sorted(self._cells.items())
        }
        return {
            "schema": SCHEMA,
            "status": status,
            "complete": complete,
            "cells": cells,
            "totals": _derived_totals(self._cells),
        }

    @classmethod
    def replay_complete_receipt(cls, receipt: Mapping[str, Any]) -> "FamilyFCallLedger":
        if not isinstance(receipt, Mapping) or set(receipt) != {
            "schema", "status", "complete", "cells", "totals",
        } or receipt.get("schema") != SCHEMA or receipt.get(
            "status"
        ) != "complete_exact_frozen_census" or receipt.get("complete") is not True:
            raise RuntimeError("Family-F complete call receipt schema changed")
        expected = expected_cells()
        expected_keys = {f"{phase}/{arm}" for phase, arm in expected}
        raw_cells = receipt.get("cells")
        if not isinstance(raw_cells, Mapping) or set(raw_cells) != expected_keys:
            raise RuntimeError("Family-F complete call receipt cells changed")
        observed = {
            key: CellCounts.from_mapping(raw_cells[f"{key[0]}/{key[1]}"])
            for key in expected
        }
        if observed != expected or receipt.get("totals") != expected_totals():
            raise RuntimeError("Family-F complete call receipt failed semantic replay")
        ledger = cls()
        ledger._cells = observed
        ledger._closed = True
        return ledger


def record_frozen_schedule(ledger: FamilyFCallLedger) -> None:
    """Test/known-answer helper: record each frozen cell as one aggregate update."""

    for (phase, arm), counts in expected_cells().items():
        ledger.record(phase, arm, **counts.as_dict())
