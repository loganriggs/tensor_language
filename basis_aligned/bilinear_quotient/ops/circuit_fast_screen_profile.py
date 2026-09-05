#!/usr/bin/env python3
# BQLANE: cpu
"""Compare fast screens without letting control choice erase target evidence.

This is deliberately a small post-processing layer.  The existing runner still
does the intervention.  This module separates two questions that its historical
single Boolean verdict combined:

1. Did the intervention move the target behaviour in the donor direction?
2. What happened on each named P/C control family?

The first answer selects candidate sites.  The second is retained as a response
profile; a related control moving is evidence for a shared route, not evidence
that the target carrier did not exist.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "circuit_fast_screen_control_profile_v1"
TARGET_RECOVERY_BAR = 0.5
TARGET_DIRECTION_BAR = 0.8


class ProfileError(ValueError):
    """Input results cannot support a comparable response profile."""


def _finite(value: Any, label: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(value):
        raise ProfileError(f"{label} must be a finite number")
    return float(value)


def _site_id(item: Mapping[str, Any]) -> str:
    site = item.get("site")
    if type(site) is not dict or type(site.get("site_id")) is not str:
        raise ProfileError("site result lacks site.site_id")
    return site["site_id"]


def _target_pass(item: Mapping[str, Any]) -> bool:
    """Apply only the frozen A1/A2 target gates, never the P/C gates."""
    for family in ("a1", "a2"):
        score = item.get(family)
        if type(score) is not dict:
            return False
        if _finite(score.get("mean_effect"), f"{family}.mean_effect") < TARGET_RECOVERY_BAR:
            return False
        if _finite(
            score.get("direction_fraction"), f"{family}.direction_fraction"
        ) < TARGET_DIRECTION_BAR:
            return False
    return True


def _margin(record: Mapping[str, Any]) -> float:
    return _finite(record.get("answer_logit"), "answer_logit") - _finite(
        record.get("foil_logit"), "foil_logit"
    )


def _raw_same_answer_movement(run: Mapping[str, Any], site_id: str, family: str) -> float:
    native: dict[str, float] = {}
    for record in run.get("native_logits", []):
        if record.get("family") == family and record.get("side") == "base":
            row_id = record.get("row_id")
            if type(row_id) is not str or row_id in native:
                raise ProfileError(f"invalid or duplicate native {family} row")
            native[row_id] = _margin(record)
    movements: list[float] = []
    for record in run.get("intervention_logits", []):
        site = record.get("site")
        if (
            record.get("family") == family
            and type(site) is dict
            and site.get("site_id") == site_id
        ):
            row_id = record.get("row_id")
            if row_id not in native:
                raise ProfileError(f"intervened {family} row lacks matching native base")
            movements.append(abs(_margin(record) - native[row_id]))
    if not movements:
        raise ProfileError(f"site {site_id} lacks {family} intervention logits")
    return math.fsum(movements) / len(movements)


def profile_results(
    members: Sequence[tuple[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Return target-only candidates plus every member's named control response."""
    if not members:
        raise ProfileError("at least one result is required")
    labels = [label for label, _ in members]
    if any(type(label) is not str or not label for label in labels) or len(labels) != len(set(labels)):
        raise ProfileError("member labels must be unique nonempty strings")

    indexed: list[tuple[str, Mapping[str, Any], dict[str, Mapping[str, Any]]]] = []
    all_sites: set[str] = set()
    for label, result in members:
        if type(result) is not dict or result.get("schema") != "circuit_fast_screen_result_v1":
            raise ProfileError(f"{label} is not a fast-screen result v1")
        run = result.get("run")
        if type(run) is not dict or type(run.get("site_results")) is not list:
            raise ProfileError(f"{label} lacks run.site_results")
        sites = {_site_id(item): item for item in run["site_results"]}
        if len(sites) != len(run["site_results"]):
            raise ProfileError(f"{label} contains duplicate sites")
        indexed.append((label, run, sites))
        all_sites.update(sites)

    output_sites: list[dict[str, Any]] = []
    for site_id in sorted(all_sites):
        responses: list[dict[str, Any]] = []
        kinds: set[str] = set()
        for label, run, sites in indexed:
            item = sites.get(site_id)
            if item is None:
                responses.append({"member": label, "present": False})
                continue
            kinds.add(item["site"]["evidence_kind"])
            responses.append({
                "member": label,
                "present": True,
                "target_pass": _target_pass(item),
                "a1_recovery": item["a1"]["mean_effect"],
                "a2_recovery": item["a2"]["mean_effect"],
                "p_normalized": item.get("p_invariance_effect"),
                "p_raw_mean_margin_movement": _raw_same_answer_movement(run, site_id, "P"),
                "c_absolute_recovery": item.get("c_absolute_recovery"),
                "c_signed_recovery": item.get("c_signed_recovery"),
            })
        if len(kinds) != 1:
            raise ProfileError(f"site {site_id} has inconsistent evidence kinds")
        present = [response for response in responses if response["present"]]
        target_passes = [response["target_pass"] for response in present]
        output_sites.append({
            "site_id": site_id,
            "evidence_kind": next(iter(kinds)),
            "target_pass_any": any(target_passes),
            "target_pass_all_present": all(target_passes),
            "responses": responses,
        })

    mechanistic = [
        item["site_id"] for item in output_sites
        if item["evidence_kind"] != "residual" and item["target_pass_any"]
    ]
    residual_ceiling = [
        item["site_id"] for item in output_sites
        if item["evidence_kind"] == "residual" and item["target_pass_any"]
    ]
    return {
        "schema": SCHEMA,
        "members": labels,
        "selection_rule": "A1 and A2 each pass recovery and direction bars; P/C never select sites",
        "fixed_bars": {
            "minimum_target_family_recovery": TARGET_RECOVERY_BAR,
            "minimum_target_direction_fraction": TARGET_DIRECTION_BAR,
        },
        "mechanistic_target_candidates": mechanistic,
        "residual_ceiling_sites": residual_ceiling,
        "site_profiles": output_sites,
    }


def _parse_member(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("member must be LABEL=RESULT.json")
    return label, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member", action="append", required=True, type=_parse_member)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    members = [(label, json.loads(path.read_text())) for label, path in args.member]
    payload = json.dumps(profile_results(members), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload)


if __name__ == "__main__":
    main()
