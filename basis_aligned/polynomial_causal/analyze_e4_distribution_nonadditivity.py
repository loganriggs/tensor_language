#!/usr/bin/env python3
"""Receipt-bound descriptive KL nonadditivity analysis for the E4 copy screen.

The earlier CE contrast establishes behavioral nonadditivity.  This companion asks
whether the *whole output distribution* also moves nonadditively.  It cannot identify
the residual-stream displacement magnitude because the sealed E4 ledger contains only
downstream sufficient statistics, not activations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

import analyze_e4_interaction_excess as ce_analysis


HERE = Path(__file__).resolve().parent
RESULT = HERE / "e4_four_head_distribution_nonadditivity_descriptive.json"
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 2026082915


def _candidate_document_kl(
    ledger: Mapping[str, Any], candidate: str, cell: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    numerators, denominators = [], []
    for document in ledger["ordered_document_ids"]:
        entry = ledger["candidates"][candidate][document][cell]
        numerators.append(entry["native_to_ablated_kl_sum"])
        denominators.append(entry["n"])
    return (
        torch.tensor(numerators, dtype=torch.float64),
        torch.tensor(denominators, dtype=torch.float64),
    )


def analyze(
    ledger: Mapping[str, Any], *, draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    ce_analysis.validate_shared_support(ledger)
    documents = len(ledger["ordered_document_ids"])
    generator = torch.Generator().manual_seed(seed)
    samples = torch.randint(documents, (draws, documents), generator=generator)
    weights = torch.zeros(draws, documents, dtype=torch.float64)
    weights.scatter_add_(1, samples, torch.ones_like(samples, dtype=torch.float64))
    weights = torch.cat((torch.ones(1, documents, dtype=torch.float64), weights), 0)

    kl: dict[str, dict[str, torch.Tensor]] = {}
    candidates = (*ce_analysis.SINGLETONS, ce_analysis.JOINT)
    for candidate in candidates:
        kl[candidate] = {}
        for cell in ce_analysis.CELLS:
            numerator, denominator = _candidate_document_kl(ledger, candidate, cell)
            pooled_denominator = weights @ denominator
            if not bool((pooled_denominator > 0).all()):
                raise RuntimeError("E4 KL contrast has empty bootstrap support")
            kl[candidate][cell] = (weights @ numerator) / pooled_denominator

    excess = {
        cell: kl[ce_analysis.JOINT][cell]
        - sum(kl[candidate][cell] for candidate in ce_analysis.SINGLETONS)
        for cell in ce_analysis.CELLS
    }
    excess["copy_specificity"] = excess["positive"] - excess["matched_negative"]
    coordinates = ("positive", "matched_negative", "off_target", "copy_specificity")
    series = torch.stack(tuple(excess[cell] for cell in coordinates), dim=1)
    point, replicas = series[0], series[1:]
    upper_radius = torch.quantile((replicas - point).max(1).values, 0.95)
    lower_radius = torch.quantile((point - replicas).max(1).values, 0.95)

    cell_summary = {}
    for cell in ce_analysis.CELLS:
        joint = kl[ce_analysis.JOINT][cell][0]
        singleton_sum = sum(
            kl[candidate][cell][0] for candidate in ce_analysis.SINGLETONS
        )
        cell_summary[cell] = {
            "joint_kl": float(joint),
            "singleton_kl_sum": float(singleton_sum),
            "joint_minus_singleton_sum": float(joint - singleton_sum),
            "joint_to_singleton_sum_ratio": float(joint / singleton_sum),
        }

    return {
        "coordinates": list(coordinates),
        "point": point.tolist(),
        "simultaneous_q05_lower": (point - lower_radius).tolist(),
        "simultaneous_q95_upper": (point + upper_radius).tolist(),
        "lower_radius": float(lower_radius),
        "upper_radius": float(upper_radius),
        "cells": cell_summary,
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "documents": documents,
    }


def main() -> None:
    if RESULT.exists():
        raise RuntimeError(f"descriptive result already exists: {RESULT}")
    if ce_analysis.file_sha256(ce_analysis.LEDGER) != ce_analysis.LEDGER_SHA256:
        raise RuntimeError("E4 receipt-backed ledger bytes changed")
    ledger = json.loads(ce_analysis.LEDGER.read_text())
    payload = {
        "schema": "e4_four_head_distribution_nonadditivity_descriptive_v1",
        "status": "posthoc_descriptive_not_confirmatory_not_interface_norm",
        "ledger_file_sha256": ce_analysis.LEDGER_SHA256,
        "joint": ce_analysis.JOINT,
        "singletons": list(ce_analysis.SINGLETONS),
        "analysis": analyze(ledger),
        "claim_boundary": (
            "Native-to-replacement output-distribution KL only. This is not a "
            "residual-stream displacement norm, does not identify interaction order, "
            "and cannot by itself distinguish head interaction from downstream "
            "nonlinearity."
        ),
    }
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
