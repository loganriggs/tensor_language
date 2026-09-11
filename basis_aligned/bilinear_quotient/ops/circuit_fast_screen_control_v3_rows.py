#!/usr/bin/env python3
"""Rebuild a spec-authored cell's C panel against a DIFFERENT same-answer control.

WHY. All 88 spec-authored cells draw their C family from one control
(`canonical_same_answer_nocturnal_completion`, answers " night"), because
`circuit_fast_screen_behaviour_spec` imports it directly at line 45 and calls it at line 133. Row 4 -- the C upper
bound -- is the general gate on countable behaviours in this corpus, so every selectivity verdict is selectivity
with respect to that one control. This module builds the SAME panel against a second control so the two can be
compared. It does not modify the spec, the canonical control, or any bar.

HOW IT STAYS HONEST (standing lesson 7: a new module that rebuilds an existing authority must reproduce the OLD
digest through the NEW code path). `rows_for(cell, control)` replicates the spec's own `common` dict and its
`builder._row(**common, transform_id="C", **control.row_kwargs(case_index, forward))` call exactly. Running it with
the CANONICAL control must return rows identical to the cell's own C family, and `verify_against_spec` asserts that
cell by cell. If that check fails the module is wrong and its v3 rows mean nothing.
"""
from __future__ import annotations

from typing import Any

import circuit_fast_screen_candidate_sentence_terminal_context_control as builder
import circuit_fast_screen_behaviour_spec as bs
import circuit_fast_screen_candidates as lex
import circuit_fast_screen_canonical_control_v2 as control_v2

canonical_sha256 = builder.canonical_sha256

# keys the control supplies, and keys the row builder derives from them; both must be dropped before rebuilding
# _row accepts exactly these; everything else in a built row is derived by it and must NOT be passed back.
_CARRIED_KEYS = ("seed", "task_id", "group_number", "group_id", "reporter", "alternate_reporter",
                 "adjective", "object_name", "spec")


def rows_for(cell_module, control, groups: int = bs.DEFAULT_GROUPS,
             seed: int = bs.DEFAULT_SEED) -> list[dict[str, Any]]:
    """The cell's C panel, built against `control` instead of the spec's hardcoded one."""
    spec_obj = cell_module.SPEC
    order = lex._permutation(seed)
    battery_spec = spec_obj.battery_spec()
    out: list[dict[str, Any]] = []
    for group_number in range(groups):
        case_index = order[group_number]
        group_id = f"FIT:{canonical_sha256([bs.SCHEMA, spec_obj.task_id, seed, group_number])[:24]}"
        forward = group_number % 2 == 0
        common = dict(seed=seed, task_id=spec_obj.task_id, group_number=group_number,
                      group_id=group_id, reporter=lex._REPORTERS[case_index][0],
                      alternate_reporter=lex._REPORTERS[case_index][1],
                      adjective=lex._ADJECTIVES[case_index], object_name=lex._OBJECTS[case_index],
                      spec=battery_spec)
        out.append(builder._row(**common, transform_id="C",
                                **control.row_kwargs(case_index, forward)))
    return out


def rows_for_any(cell_module, control) -> list[dict[str, Any]]:
    """As `rows_for`, but for cells with no SPEC attribute.

    The spec-authored path rebuilds the panel from the cell's own `common` dict. Older cells -- including the two
    standing DAS targets, correlative_pair and possessive_adjacent -- predate BehaviourSpec and expose no SPEC, so
    that path cannot run. Here the cell's EXISTING C rows are taken as the carrier of every non-control field, and
    only the control-specific keys are replaced. That is strictly less clever than rebuilding and it is also safer:
    whatever the old cell put in its rows is preserved untouched, and the only thing that changes is the control.
    """
    own = [r for r in cell_module.build_rows()
           if r.get("family", r.get("transform_id")) == "C"]
    out: list[dict[str, Any]] = []
    seed = own[0].get("seed", bs.DEFAULT_SEED) if own else bs.DEFAULT_SEED
    order = lex._permutation(seed)
    for group_number, row in enumerate(own):
        # the case index is the group number mapped through the seed permutation, exactly as the spec path does;
        # reading group_number directly produced the wrong lexicon slot and the known-good check caught it
        case_index = order[row.get("group_number", group_number)]
        forward = row.get("direction_id") == "base_to_donor"
        kwargs = control.row_kwargs(case_index, forward)
        carried = {k: row[k] for k in _CARRIED_KEYS if k in row}
        if "spec" not in carried:
            # older cells do not store the spec object in the row; take it from the module
            carried["spec"] = getattr(cell_module, "TASK_SPEC", None)
        rebuilt = builder._row(**carried, transform_id="C", **kwargs)
        out.append(rebuilt)
    return out


def rows_any(cell_module, control) -> list[dict[str, Any]]:
    """Dispatch: spec-authored cells rebuild from their own `common`, older cells substitute into their C rows."""
    if hasattr(cell_module, "SPEC"):
        return rows_for(cell_module, control)
    return rows_for_any(cell_module, control)


def verify_against_any(cell_module) -> tuple[bool, str]:
    """KNOWN-GOOD CHECK for the no-SPEC path: with the canonical control it must reproduce the cell's own C rows."""
    rebuilt = rows_for_any(cell_module, control_v2)
    own = [r for r in cell_module.build_rows()
           if r.get("family", r.get("transform_id")) == "C"]
    if len(rebuilt) != len(own):
        return False, f"row count {len(rebuilt)} != {len(own)}"
    for i, (a, b) in enumerate(zip(rebuilt, own)):
        if a != b:
            diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            return False, f"row {i} differs on {diff[:8]}"
    return True, f"{len(own)} rows identical"


def verify_against_spec(cell_module) -> tuple[bool, str]:
    """KNOWN-GOOD CHECK: with the canonical control this must reproduce the cell's own C rows."""
    rebuilt = rows_for(cell_module, control_v2)
    own = [r for r in cell_module.build_rows()
           if r.get("family", r.get("transform_id")) == "C"]
    if len(rebuilt) != len(own):
        return False, f"row count {len(rebuilt)} != {len(own)}"
    for i, (a, b) in enumerate(zip(rebuilt, own)):
        if a != b:
            diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            return False, f"row {i} differs on {diff[:6]}"
    return True, f"{len(own)} rows identical"


if __name__ == "__main__":
    import importlib, sys
    for name in sys.argv[1:]:
        m = importlib.import_module(f"circuit_fast_screen_candidate_{name}")
        ok, msg = verify_against_spec(m)
        print(f"{'PASS' if ok else 'FAIL'} {name}: {msg}")
