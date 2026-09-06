#!/usr/bin/env python3
"""Fit-free raw-text selector over the released has/had and is/was programs."""

from __future__ import annotations

import re

import aspectual_anchor_transparent_path_program_v12 as has_program
import tense_auxiliary_is_was_transparent_path_program_v1 as is_program


PROGRAM_ID = "aspectual_tense.raw_text_dual_program_v1"
HAS_CUES = {
    "present_to_past": ("since last",),
    "past_to_present": ("by last",),
}
IS_CUES = {
    "present_to_past": ("this moment", "present moment"),
    "past_to_present": ("that moment", "previous moment"),
}


class SelectorInputError(ValueError):
    pass


def normalize_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise SelectorInputError("text must be a nonempty string")
    return " ".join(text.lower().split())


def select_command(text: str) -> dict[str, str | None]:
    normalized = normalize_text(text)
    padded = f" {normalized} "
    matches = []
    for bank, cues_by_direction in (("has_had", HAS_CUES), ("is_was", IS_CUES)):
        for direction, cues in cues_by_direction.items():
            if any(re.search(rf"(?<!\w){re.escape(cue)}(?!\w)", padded) for cue in cues):
                matches.append((bank, direction))
    unique = sorted(set(matches))
    if not unique:
        return {"bank": "abstain", "direction": None, "normalized_text": normalized}
    if len(unique) != 1:
        raise SelectorInputError("text has ambiguous bank or direction cues")
    bank, direction = unique[0]
    return {"bank": bank, "direction": direction, "normalized_text": normalized}


def actuate(resid10, base_resid18, bases, lm_head, *, text: str):
    command = select_command(text)
    bank, direction = command["bank"], command["direction"]
    if bank == "abstain":
        return {
            "patched_resid18": base_resid18,
            "alpha": 0.0,
            "resid10_unembedding_contrast": None,
            "bank": bank,
            "direction": direction,
        }
    if not isinstance(bases, dict) or bank not in bases:
        raise SelectorInputError(f"bases must contain selected bank {bank!r}")
    if bank == "has_had":
        result = has_program.upstream_carrier_actuation(resid10, base_resid18, bases[bank], lm_head, direction=direction)
    else:
        result = is_program.upstream_writer_actuation(resid10, base_resid18, bases[bank], lm_head, direction=direction)
    return {**result, "bank": bank}


def program_manifest() -> dict[str, object]:
    return {
        "program_id": PROGRAM_ID,
        "selector": {
            "normalization": "lowercase, trim, collapse whitespace",
            "has_had": {key: list(value) for key, value in HAS_CUES.items()},
            "is_was": {key: list(value) for key, value in IS_CUES.items()},
            "default": "abstain",
            "ambiguous": "reject",
            "fitted_values": 0,
        },
        "branches": {
            "has_had": has_program.PROGRAM_ID,
            "is_was": is_program.PROGRAM_ID,
            "abstain": "identity at resid:18",
        },
        "required_runtime_inputs": ("raw_text", "resid10", "base_resid18", "q_has", "q_is", "lm_head"),
        "scope": "registered since/by-last and this/that/present/previous-moment constructions only",
        "internal_task_circuit_claimed": False,
    }
