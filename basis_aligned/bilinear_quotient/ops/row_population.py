#!/usr/bin/env python3
"""Print the ROW POPULATION of a behaviour set before registering any bar on it (Claude's helper, no model).

For each family and parity: number of rows, distinct cue-token pairs per differing column, the base/donor answer
tokens (orientation), and length/position sanity. Written after two same-hour errors of reading ONE row as the
population (07:11 "one token pair" — the set had 16; 07:20 "same base->donor orientation" — quantifier is reversed).
Usage: python ops/row_population.py <set_name> [<set_name> ...]   (names as in v112.BATCH / INSTRUMENT)
"""
from __future__ import annotations

import collections
import importlib
import sys

import circuit_unit_greedy as g
import run_unit_tier3_batch_v112 as v112


def describe(n: str) -> None:
    names = {**v112.BATCH, **v112.INSTRUMENT}
    m = importlib.import_module(f"circuit_fast_screen_candidate_{names[n]}")
    print(f"== {n} ({names[n]})")
    for fam in ("A1", "A2", "P", "C"):
        rows = g.rows_of(m, fam)
        if not rows:
            continue
        for par in (0, 1):
            sub = rows[par::2]
            pairs = collections.defaultdict(set)
            answers = collections.Counter()
            bad = 0
            for r in sub:
                t = r["donor_semantic_position"]
                if len(r["base_ids"]) != len(r["donor_ids"]) or r["base_semantic_position"] != t:
                    bad += 1
                    continue
                for p in range(t):
                    if r["base_ids"][p] != r["donor_ids"][p]:
                        pairs[p].add((r["base_ids"][p], r["donor_ids"][p]))
                answers[(r.get("base_answer"), r.get("donor_answer"))] += 1
            cols = {p: len(s) for p, s in sorted(pairs.items())}
            print(f"  {fam} parity {par}: rows {len(sub)} (unequal/misaligned {bad}); cue columns -> distinct pairs {cols}; "
                  f"base->donor answers {dict(answers)}")


if __name__ == "__main__":
    for name in sys.argv[1:]:
        describe(name)
