"""Exact protocol registry for prospective ordered-successor SELECT v2."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import ordered_successor_digit_lexicon_v2 as digits


SCHEMA = "ordered_successor_tensor_select_v2_protocol_registry"
RANK_LADDER = (8, 16, 32, 64, 96, 128)
OMITTED_V1_DIAGNOSTICS = (
    "head8_7_current_only_r128", "head8_7_v1_only_r128",
)
ARM_NAMES = (
    "native", "full_attention8_replay", "head8_7_deleted",
    *(
        f"head8_7_both_r{rank}_{kind}"
        for rank in RANK_LADDER
        for kind in ("true", "spectral_null")
    ),
)
PROMOTIVE_ARMS = tuple(f"head8_7_both_r{rank}_true" for rank in RANK_LADDER)
NULL_BY_TRUE = {
    true: true.removesuffix("_true") + "_spectral_null" for true in PROMOTIVE_ARMS
}
POWERED_CELLS = ("positive_clean", "wrong_source_clean", "no_source_clean")
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 2_026_083_013
MAX_ERROR_ORDER_INDEX = 18_999
SCORER_SOURCE_PATHS = (
    "basis_aligned/polynomial_causal/ORDERED_SUCCESSOR_TENSOR_DISCOVERY_PREREGISTRATION.md",
    "basis_aligned/polynomial_causal/bilin18_observed_model_facade.py",
    "basis_aligned/polynomial_causal/circuit_campaign_runtime.py",
    "basis_aligned/polynomial_causal/circuit_successor_tensor.py",
    "basis_aligned/polynomial_causal/ordered_successor_masks_v1.py",
    "basis_aligned/polynomial_causal/ordered_successor_tensor_discovery_v1.py",
    "basis_aligned/polynomial_causal/ordered_successor_tensor_backend_adapter_v1.py",
    "basis_aligned/polynomial_causal/successor_attention_backend.py",
    "basis_aligned/polynomial_causal/tensor_preserving_attention.py",
    "basis_aligned/polynomial_causal/test_ordered_successor_masks_v1.py",
    "basis_aligned/polynomial_causal/test_ordered_successor_tensor_discovery_v1.py",
    "basis_aligned/polynomial_causal/test_ordered_successor_tensor_backend_adapter_v1.py",
    "basis_aligned/polynomial_causal/test_successor_attention_backend.py",
    "basis_aligned/polynomial_causal/test_bilin18_observed_model_facade.py",
    "basis_aligned/polynomial_causal/test_circuit_campaign_runtime.py",
    "basis_aligned/polynomial_causal/test_circuit_successor_tensor.py",
    "basis_aligned/polynomial_causal/test_tensor_preserving_attention.py",
    "basis_aligned/polynomial_causal/ordered_successor_tensor_select_statistics_v1.py",
    "basis_aligned/polynomial_causal/test_ordered_successor_tensor_select_statistics_v1.py",
    "basis_aligned/polynomial_causal/ordered_successor_tensor_select_registry_v2.py",
)
REGISTRY_SHA256 = "38e9775c8a30e9ed9ac1278ca3940a0b46699527e958b24d145d0978932be7d5"


def registry_payload() -> dict[str, Any]:
    """Return the JSON-domain contract around the unchanged v1 statistics."""

    return {
        "schema": SCHEMA,
        "arm_names": list(ARM_NAMES),
        "promotive_arms": list(PROMOTIVE_ARMS),
        "omitted_nonpromotive_v1_diagnostics": list(OMITTED_V1_DIAGNOSTICS),
        "digit_lexicon_registry_sha256": digits.REGISTRY_SHA256,
        "statistical_procedure": {
            "source": "ordered_successor_tensor_select_statistics_v1.py",
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "order_index": MAX_ERROR_ORDER_INDEX,
            "powered_cells": list(POWERED_CELLS),
        },
    }


def registry_sha256() -> str:
    return hashlib.sha256(json.dumps(
        registry_payload(), sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def validate_registry() -> None:
    if (
        len(ARM_NAMES) != 15
        or ARM_NAMES[:3] != ("native", "full_attention8_replay", "head8_7_deleted")
        or len(PROMOTIVE_ARMS) != 6
        or any(NULL_BY_TRUE[name] not in ARM_NAMES for name in PROMOTIVE_ARMS)
        or any(name in ARM_NAMES for name in OMITTED_V1_DIAGNOSTICS)
        or registry_sha256() != REGISTRY_SHA256
    ):
        raise RuntimeError("ordered-successor v2 protocol registry changed")


__all__ = (
    "ARM_NAMES", "BOOTSTRAP_DRAWS", "BOOTSTRAP_SEED", "MAX_ERROR_ORDER_INDEX",
    "NULL_BY_TRUE", "OMITTED_V1_DIAGNOSTICS", "POWERED_CELLS",
    "PROMOTIVE_ARMS", "RANK_LADDER", "REGISTRY_SHA256", "SCHEMA",
    "SCORER_SOURCE_PATHS", "registry_payload", "registry_sha256",
    "validate_registry",
)
