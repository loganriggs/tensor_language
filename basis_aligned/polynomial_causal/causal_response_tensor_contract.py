"""Contract for additive, split-able circuit intervention response tensors.

The existing substrate artifacts store concentration ratios of absolute CE changes.
Those are useful localization scores but are not additive causal response cells.  This
module defines the smallest per-document sufficient statistics needed for a lawful
response tensor and records exact counterexamples to ratio-only identifiability.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DocumentResponse:
    document_id: int
    member_signed_sum: float
    member_abs_sum: float
    member_count: int
    off_signed_sum: float
    off_abs_sum: float
    off_count: int

    def validate(self) -> None:
        if self.document_id < 0:
            raise ValueError("document_id must be nonnegative")
        if self.member_count <= 0 or self.off_count <= 0:
            raise ValueError("member/off counts must be positive")
        if self.member_abs_sum < abs(self.member_signed_sum) - 1e-12:
            raise ValueError("member absolute sum cannot be below absolute signed sum")
        if self.off_abs_sum < abs(self.off_signed_sum) - 1e-12:
            raise ValueError("off absolute sum cannot be below absolute signed sum")

    @property
    def signed_contrast(self) -> float:
        """Linear functional of dCE: member mean minus off-slice mean."""
        self.validate()
        return self.member_signed_sum / self.member_count - self.off_signed_sum / self.off_count

    @property
    def absolute_concentration(self) -> float:
        """Legacy localization ratio, retained only as a diagnostic."""
        self.validate()
        denominator = self.off_abs_sum / self.off_count
        if denominator == 0:
            raise ValueError("absolute concentration denominator is zero")
        return (self.member_abs_sum / self.member_count) / denominator


def summarize_document(
    document_id: int,
    dce: np.ndarray,
    member_mask: np.ndarray,
    off_mask: np.ndarray,
) -> DocumentResponse:
    dce = np.asarray(dce, dtype=np.float64)
    member_mask = np.asarray(member_mask, dtype=bool)
    off_mask = np.asarray(off_mask, dtype=bool)
    if dce.ndim != 1 or member_mask.shape != dce.shape or off_mask.shape != dce.shape:
        raise ValueError("dce and masks must be same-length vectors")
    if np.any(member_mask & off_mask):
        raise ValueError("member and off masks must be disjoint")
    if not member_mask.any() or not off_mask.any():
        raise ValueError("both masks must contain positions")
    response = DocumentResponse(
        document_id=document_id,
        member_signed_sum=float(dce[member_mask].sum()),
        member_abs_sum=float(np.abs(dce[member_mask]).sum()),
        member_count=int(member_mask.sum()),
        off_signed_sum=float(dce[off_mask].sum()),
        off_abs_sum=float(np.abs(dce[off_mask]).sum()),
        off_count=int(off_mask.sum()),
    )
    response.validate()
    return response


def validate_response_records(records: list[DocumentResponse]) -> None:
    if len(records) < 2:
        raise ValueError("at least two documents are required for split/bootstrap evidence")
    seen: set[int] = set()
    for record in records:
        record.validate()
        if record.document_id in seen:
            raise ValueError("document IDs must be unique within a response cell")
        seen.add(record.document_id)


def identifiability_counterexamples() -> dict[str, object]:
    # Same ratio, arbitrarily different additive signed contrast.
    first = DocumentResponse(0, 2.0, 2.0, 1, 1.0, 1.0, 1)
    scaled = DocumentResponse(1, 20.0, 20.0, 1, 10.0, 10.0, 1)

    # Ratios do not add when raw effects are pooled/composed.
    left = DocumentResponse(2, 2.0, 2.0, 1, 1.0, 1.0, 1)
    right = DocumentResponse(3, 2.0, 2.0, 1, 2.0, 2.0, 1)
    pooled = DocumentResponse(4, 4.0, 4.0, 1, 3.0, 3.0, 1)

    # Absolute values erase the sign/cancellation needed for CE prediction.
    cancelling = summarize_document(
        5, np.asarray([1.0, -1.0, 0.0]), np.asarray([1, 1, 0]), np.asarray([0, 0, 1])
    )
    aligned = summarize_document(
        6, np.asarray([1.0, 1.0, 0.0]), np.asarray([1, 1, 0]), np.asarray([0, 0, 1])
    )

    return {
        "same_ratio_different_contrast": {
            "ratio_first": first.absolute_concentration,
            "ratio_scaled": scaled.absolute_concentration,
            "signed_contrast_first": first.signed_contrast,
            "signed_contrast_scaled": scaled.signed_contrast,
        },
        "ratio_nonadditivity": {
            "left_ratio": left.absolute_concentration,
            "right_ratio": right.absolute_concentration,
            "pooled_ratio": pooled.absolute_concentration,
            "sum_of_ratios": left.absolute_concentration + right.absolute_concentration,
        },
        "absolute_value_sign_loss": {
            "member_abs_mean_cancelling": cancelling.member_abs_sum / cancelling.member_count,
            "member_abs_mean_aligned": aligned.member_abs_sum / aligned.member_count,
            "member_signed_mean_cancelling": (
                cancelling.member_signed_sum / cancelling.member_count
            ),
            "member_signed_mean_aligned": aligned.member_signed_sum / aligned.member_count,
        },
    }


def build_receipt() -> dict[str, object]:
    started = time.monotonic()
    return {
        "schema": "causal_response_tensor_contract_v1",
        "claim_boundary": (
            "CPU mathematical/serialization contract only. No model, rows, protected "
            "outcomes, response collection, tensor fit, or circuit promotion."
        ),
        "lawful_cell": (
            "For each intervention-phase/source/target/document, store signed and "
            "absolute member/off-slice sums plus counts. The fitted response is the "
            "document-level signed mean contrast; absolute concentration is diagnostic."
        ),
        "required_split": (
            "Direction fitting and response evaluation use disjoint documents; retain "
            "document IDs so model selection and bootstrap resample whole documents."
        ),
        "counterexamples": identifiability_counterexamples(),
        "runtime_seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "causal_response_tensor_contract_receipt.json",
    )
    args = parser.parse_args()
    receipt = build_receipt()
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
