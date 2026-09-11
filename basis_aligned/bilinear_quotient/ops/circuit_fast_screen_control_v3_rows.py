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
