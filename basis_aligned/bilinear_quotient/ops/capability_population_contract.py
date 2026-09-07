#!/usr/bin/env python3
"""Model-free capability-panel count contract.

Use this before registering a capability prior.  Required counts are derived from
the rows that will actually be executed, so a copied denominator cannot make the
prospective gate impossible or silently weaker than intended.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


class CapabilityPopulationContractError(ValueError):
    """The built population and the declared capability contract disagree."""


def derive_panel_contract(
    rows: Iterable[Mapping[str, Any]],
    *,
    panels: Sequence[str],
    minimum_fraction: float,
    panel_field: str = "transform_id",
    row_id_field: str = "row_id",
) -> dict[str, Any]:
    materialized = [dict(row) for row in rows]
    ordered_panels = tuple(str(panel) for panel in panels)
    if not ordered_panels or len(set(ordered_panels)) != len(ordered_panels):
        raise CapabilityPopulationContractError("panels must be nonempty and unique")
    if not 0.0 <= minimum_fraction <= 1.0 or not math.isfinite(minimum_fraction):
        raise CapabilityPopulationContractError("minimum_fraction must be finite and in [0, 1]")

    row_ids = [str(row[row_id_field]) for row in materialized]
    duplicates = sorted(row_id for row_id, count in Counter(row_ids).items() if count != 1)
    if duplicates:
        raise CapabilityPopulationContractError(f"duplicate row IDs: {duplicates[:3]}")

    counts = {
        panel: sum(str(row.get(panel_field)) == panel for row in materialized)
        for panel in ordered_panels
    }
    empty = [panel for panel, count in counts.items() if count == 0]
    if empty:
        raise CapabilityPopulationContractError(f"empty capability panels: {empty}")
    required = {
        panel: int(math.ceil((minimum_fraction * count) - 1e-12))
        for panel, count in counts.items()
    }
    panel_row_ids = {
        panel: [str(row[row_id_field]) for row in materialized if str(row.get(panel_field)) == panel]
        for panel in ordered_panels
    }
    return {
        "panel_field": panel_field,
        "row_id_field": row_id_field,
        "minimum_fraction": minimum_fraction,
        "population_rows": len(materialized),
        "panel_counts": counts,
        "required_jointly_capable_counts": required,
        "panel_row_ids": panel_row_ids,
    }


def assert_declared_counts(
    contract: Mapping[str, Any],
    *,
    declared_denominators: Mapping[str, int],
    declared_required_counts: Mapping[str, int],
) -> None:
    actual_denominators = dict(contract["panel_counts"])
    actual_required = dict(contract["required_jointly_capable_counts"])
    if dict(declared_denominators) != actual_denominators:
        raise CapabilityPopulationContractError(
            f"declared denominators {dict(declared_denominators)} != built counts {actual_denominators}"
        )
    if dict(declared_required_counts) != actual_required:
        raise CapabilityPopulationContractError(
            f"declared required counts {dict(declared_required_counts)} != derived counts {actual_required}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("builder_module", help="importable module exposing build_rows()")
    parser.add_argument("--panels", nargs="+", required=True)
    parser.add_argument("--minimum-fraction", type=float, required=True)
    args = parser.parse_args()
    builder = importlib.import_module(args.builder_module)
    contract = derive_panel_contract(
        builder.build_rows(), panels=args.panels, minimum_fraction=args.minimum_fraction
    )
    print(json.dumps(contract, sort_keys=True))


if __name__ == "__main__":
    main()
