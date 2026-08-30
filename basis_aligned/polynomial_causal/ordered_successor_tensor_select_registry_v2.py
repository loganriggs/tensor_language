"""Exact protocol registry for prospective ordered-successor SELECT v2."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import ordered_successor_digit_lexicon_v2 as digits
import ordered_successor_tensor_discovery_v1 as discovery
import ordered_successor_tensor_select_statistics_v1 as statistics


SCHEMA = "ordered_successor_tensor_select_v2_protocol_registry"
OMITTED_V1_DIAGNOSTICS = (discovery.CURRENT_ONLY, discovery.V1_ONLY)
ARM_NAMES = tuple(
    name for name in discovery.ARM_NAMES if name not in OMITTED_V1_DIAGNOSTICS
)
PROMOTIVE_ARMS = discovery.PROMOTIVE_ARMS
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
            "bootstrap_draws": discovery.BOOTSTRAP_DRAWS,
            "bootstrap_seed": discovery.BOOTSTRAP_SEED,
            "order_index": discovery.MAX_ERROR_ORDER_INDEX,
            "powered_cells": list(statistics.POWERED_CELLS),
        },
    }


def registry_sha256() -> str:
    return hashlib.sha256(json.dumps(
        registry_payload(), sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def validate_registry() -> None:
    if (
        len(ARM_NAMES) != 15
        or ARM_NAMES != discovery.ARM_NAMES[:-2]
        or PROMOTIVE_ARMS != tuple(
            candidate.arm for candidate in discovery.CANDIDATES
            if candidate.kind is discovery.CandidateKind.TRUE
        )
        or any(name in ARM_NAMES for name in OMITTED_V1_DIAGNOSTICS)
        or registry_sha256() != REGISTRY_SHA256
    ):
        raise RuntimeError("ordered-successor v2 protocol registry changed")


__all__ = (
    "ARM_NAMES", "OMITTED_V1_DIAGNOSTICS", "PROMOTIVE_ARMS", "REGISTRY_SHA256",
    "SCHEMA", "registry_payload", "registry_sha256", "validate_registry",
)
