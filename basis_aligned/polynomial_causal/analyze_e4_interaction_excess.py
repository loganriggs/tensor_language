#!/usr/bin/env python3
"""Post-hoc, descriptive nonadditivity analysis of the receipt-backed E4 ledger.

This does not estimate the Boolean-lattice Möbius coefficients: pairs and triples were
not measured.  It computes only the observable excess of the registered four-head arm
over the sum of its four registered singleton arms, with a paired document bootstrap.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch


HERE = Path(__file__).resolve().parent
LEDGER = HERE / "terminal_copy_selection_v1_attempt2_ledger.json"
RESULT = HERE / "e4_four_head_nonadditivity_descriptive.json"
LEDGER_SHA256 = "ca180ec981bfdc68d554740597afb0ae94db8469022f8f090257fe6b9f6f6935"
SINGLETONS = ("L5H5", "L7H3", "L8H3", "L8H4")
JOINT = "registered_four_head_set"
CELLS = ("positive", "matched_negative", "off_target")
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 2026082914


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_document_arrays(
    ledger: Mapping[str, Any], candidate: str, cell: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    documents = ledger["ordered_document_ids"]
    numerators, denominators = [], []
    for document in documents:
        entry = ledger["candidates"][candidate][document][cell]
        numerators.append(entry["ablated_nll_sum"] - entry["native_nll_sum"])
        denominators.append(entry["n"])
    return (
        torch.tensor(numerators, dtype=torch.float64),
        torch.tensor(denominators, dtype=torch.float64),
    )


def validate_shared_support(ledger: Mapping[str, Any]) -> None:
    candidates = (*SINGLETONS, JOINT)
    for document in ledger["ordered_document_ids"]:
        for cell in CELLS:
            reference = ledger["candidates"][JOINT][document][cell]
            for candidate in candidates:
                observed = ledger["candidates"][candidate][document][cell]
                if (
                    observed["n"] != reference["n"]
                    or observed["support_sha256"] != reference["support_sha256"]
                    or observed["native_nll_sum"] != reference["native_nll_sum"]
                ):
                    raise RuntimeError("E4 interaction contrast lacks shared support/baseline")


def bootstrap_interaction_excess(
    ledger: Mapping[str, Any], *, draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    validate_shared_support(ledger)
    documents = len(ledger["ordered_document_ids"])
    generator = torch.Generator().manual_seed(seed)
    samples = torch.randint(documents, (draws, documents), generator=generator)
    weights = torch.zeros(draws, documents, dtype=torch.float64)
    weights.scatter_add_(1, samples, torch.ones_like(samples, dtype=torch.float64))
    weights = torch.cat((torch.ones(1, documents, dtype=torch.float64), weights), 0)

    tau: dict[str, dict[str, torch.Tensor]] = {}
    for candidate in (*SINGLETONS, JOINT):
        tau[candidate] = {}
        for cell in CELLS:
            numerator, denominator = _candidate_document_arrays(ledger, candidate, cell)
            pooled_denominator = weights @ denominator
            if not bool((pooled_denominator > 0).all()):
                raise RuntimeError("E4 interaction contrast has empty bootstrap support")
            tau[candidate][cell] = (weights @ numerator) / pooled_denominator

    excess = {
        cell: tau[JOINT][cell] - sum(tau[candidate][cell] for candidate in SINGLETONS)
        for cell in CELLS
    }
    excess["specificity"] = excess["positive"] - excess["matched_negative"]
    coordinates = ("positive", "matched_negative", "off_target", "specificity")
    series = torch.stack(tuple(excess[cell] for cell in coordinates), dim=1)
    point, replicas = series[0], series[1:]
    upper_radius = torch.quantile((replicas - point).max(1).values, 0.95)
    lower_radius = torch.quantile((point - replicas).max(1).values, 0.95)
    singleton_positive_sum = sum(tau[candidate]["positive"][0] for candidate in SINGLETONS)
    joint_positive = tau[JOINT]["positive"][0]
    return {
        "coordinates": list(coordinates),
        "point": point.tolist(),
        "simultaneous_q05_lower": (point - lower_radius).tolist(),
        "simultaneous_q95_upper": (point + upper_radius).tolist(),
        "lower_radius": float(lower_radius),
        "upper_radius": float(upper_radius),
        "joint_positive_tau": float(joint_positive),
        "singleton_positive_tau_sum": float(singleton_positive_sum),
        "joint_to_singleton_sum_ratio": float(joint_positive / singleton_positive_sum),
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "documents": documents,
    }


def main() -> None:
    if RESULT.exists():
        raise RuntimeError(f"descriptive result already exists: {RESULT}")
    if file_sha256(LEDGER) != LEDGER_SHA256:
        raise RuntimeError("E4 receipt-backed ledger bytes changed")
    ledger = json.loads(LEDGER.read_text())
    analysis = bootstrap_interaction_excess(ledger)
    payload = {
        "schema": "e4_four_head_nonadditivity_descriptive_v1",
        "status": "posthoc_descriptive_not_confirmatory_not_mobius_identified",
        "ledger_file_sha256": LEDGER_SHA256,
        "joint": JOINT,
        "singletons": list(SINGLETONS),
        "analysis": analysis,
        "claim_boundary": (
            "Observable joint-minus-singleton-sum excess only; pairs/triples are absent, "
            "so no unique interaction order or head attribution is identified."
        ),
    }
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

