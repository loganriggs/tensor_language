#!/usr/bin/env python3
"""Hash-bound 22-scalar Task14/bracket transparent margin program."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT / "circuits/followups/task14_standalone_bracket_conditioned_hybrid_v6_artifact.json"
EXPECTED_SHA256 = "c365321035a7cf7886f3038ac29a76659b9f3c968bf044e6ef84bf448dd5218d"
LETTERS = "EAUW"
CLOSERS = {1, 8, 60}


class ProgramError(ValueError):
    pass


def load_artifact(path: Path = DEFAULT_ARTIFACT) -> dict:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_SHA256:
        raise ProgramError("hybrid artifact hash mismatch")
    artifact = json.loads(raw)
    if artifact.get("terminal") != "frozen_hybrid_program" or artifact.get("stored_fp32_scalars") != 22:
        raise ProgramError("hybrid artifact status/inventory mismatch")
    return artifact


def task14(artifact: dict, *, direction: str, background: str, edit: bool) -> dict:
    if direction not in {"singular_to_plural", "plural_to_singular"}:
        raise ProgramError("invalid Task14 direction")
    if not isinstance(background, str) or len(set(background)) != len(background) or any(letter not in LETTERS for letter in background):
        raise ProgramError("background must be a unique subset of EAUW")
    if not isinstance(edit, bool):
        raise ProgramError("edit must be boolean")
    program = artifact["programs"]["task14"]
    features = [1.0, 1.0 if direction == "singular_to_plural" else -1.0] + [1.0 if letter in background else 0.0 for letter in LETTERS]
    native = sum(feature * coefficient for feature, coefficient in zip(features, program["native_margin_coefficients"]))
    key = f"{direction}.cardinality_{len(background)}"
    effect = float(program["intervention_effects"][key]) if edit else 0.0
    return {"behavior": "task14", "predicted_native_donorward_margin": native, "predicted_intervention_effect": effect, "predicted_counterfactual_donorward_margin": native + effect, "edit_key": key if edit else None}


def bracket(artifact: dict, *, native_unedited_donorward_margin: float, recipient_closer_id: int, donor_closer_id: int) -> dict:
    if isinstance(native_unedited_donorward_margin, bool) or not isinstance(native_unedited_donorward_margin, (int, float)) or not math.isfinite(float(native_unedited_donorward_margin)):
        raise ProgramError("native margin must be finite numeric")
    if recipient_closer_id not in CLOSERS or donor_closer_id not in CLOSERS:
        raise ProgramError("invalid closer id")
    key = f"{recipient_closer_id}->{donor_closer_id}"
    effect = 0.0 if recipient_closer_id == donor_closer_id else float(artifact["programs"]["bracket"]["intervention_effects"][key])
    native = float(native_unedited_donorward_margin)
    return {"behavior": "bracket", "native_unedited_donorward_margin": native, "predicted_intervention_effect": effect, "predicted_counterfactual_donorward_margin": native + effect, "edit_key": None if recipient_closer_id == donor_closer_id else key}


def dispatch(artifact: dict, request: dict) -> dict:
    if not isinstance(request, dict):
        raise ProgramError("request must be an object")
    behavior = request.get("behavior")
    if behavior == "task14":
        if set(request) != {"behavior", "direction", "background", "edit"}:
            raise ProgramError("unexpected Task14 fields")
        return task14(artifact, direction=request["direction"], background=request["background"], edit=request["edit"])
    if behavior == "bracket":
        if set(request) != {"behavior", "native_unedited_donorward_margin", "recipient_closer_id", "donor_closer_id"}:
            raise ProgramError("unexpected bracket fields")
        return bracket(artifact, native_unedited_donorward_margin=request["native_unedited_donorward_margin"], recipient_closer_id=request["recipient_closer_id"], donor_closer_id=request["donor_closer_id"])
    raise ProgramError("unknown behavior")


def main() -> None:
    try:
        request = json.loads(sys.stdin.read())
        print(json.dumps(dispatch(load_artifact(), request), sort_keys=True))
    except (ProgramError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
