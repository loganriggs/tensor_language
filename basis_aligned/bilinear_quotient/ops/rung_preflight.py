#!/usr/bin/env python3
"""Model-free preflight for a rung, for when the CPU smoke cannot run.

WHY THIS EXISTS. The standing gate before enqueueing is a CPU smoke, and it has caught real bugs (a quantifier
error, a KeyError on a missing bar, a vacuous predicate). But while the GPU lane is saturated the CPU smoke is
unusable: on 2026-09-11 one ran its full 900 s timeout and produced a ZERO-BYTE log, never reaching its first
behaviour line. I substituted hand-written checks three times that day and wrote each substitution into the ledger.
This makes the substitution reusable and uniform instead of improvised per rung.

WHAT IT CHECKS, all without loading the model:
  1. the file parses;
  2. every NAMES value resolves to an importable candidate module -- this caught `lexical_number`, whose task_id is
     `lexical_number` but whose module is `lexical_number_pp`, a mismatch that would have errored that cell on GPU;
  3. every cell builds all four families at the expected row count;
  4. no module-level CONSTANT is assigned twice, which is how a stale value from the predecessor shadows a new one;
  5. the predicate names returned by PREDS match the names registered in the docstring.

WHAT IT DOES NOT CHECK, and why it is not a smoke replacement: it never runs the fit, so it cannot catch a
mis-specified bar, an unfalsifiable predicate, or a panel that prepares empty. Those need the smoke or a synthetic
PREDS exercise. Passing this is a licence to enqueue when the smoke cannot run, not a licence to skip thinking.
"""
from __future__ import annotations

import ast
import importlib
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))


def _module_of(path):
    src = open(path).read()
    tree = ast.parse(src)
    return src, tree


def check(path, expect_rows=32, families=("A1", "A2", "P", "C")):
    problems = []
    try:
        src, tree = _module_of(path)
    except SyntaxError as err:
        return [f"does not parse: {err}"]

    dup = [n for n, c in _const_counts(tree).items() if c > 1]
    if dup:
        problems.append(f"module-level constants assigned twice: {', '.join(dup)}")

    names = _names_dict(path)
    if names is None:
        problems.append("could not read a NAMES mapping; skipping cell checks")
        return problems

    import circuit_unit_greedy as g
    for behaviour, module_suffix in names.items():
        try:
            mod = importlib.import_module(f"circuit_fast_screen_candidate_{module_suffix}")
        except Exception as err:
            problems.append(f"{behaviour}: module circuit_fast_screen_candidate_{module_suffix} -> "
                            f"{type(err).__name__}")
            continue
        try:
            counts = {fam: len(g.rows_of(mod, fam)) for fam in families}
        except Exception as err:
            problems.append(f"{behaviour}: rows_of failed with {type(err).__name__}")
            continue
        bad = {f: n for f, n in counts.items() if n != expect_rows}
        if bad:
            problems.append(f"{behaviour}: unexpected row counts {bad}")

    registered = set(re.findall(r"^\s*(pred_[A-Za-z0-9_]+)", src, re.M))
    returned = set(re.findall(r'"(pred_[A-Za-z0-9_]+)"\s*:', src))
    if returned and registered:
        missing = returned - registered
        if missing:
            problems.append(f"returned but not registered in the docstring: {sorted(missing)}")
    return problems


def _const_counts(tree):
    import collections
    seen = collections.Counter()
    def names(target):
        # tuple assignment is how this repo declares most constants:
        # `POOL, TARGET, MIN_GAIN, MAX_UNITS = ...` and `EXT_MIN, REM_MIN, C_UB_MAX = ...`.
        # An earlier version of this check only walked ast.Name and silently missed every one of them.
        if isinstance(target, ast.Name):
            return [target.id] if target.id.isupper() else []
        if isinstance(target, (ast.Tuple, ast.List)):
            return [n for t in target.elts for n in names(t)]
        return []

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                for n in names(t):
                    seen[n] += 1
        elif isinstance(node, ast.AnnAssign):
            for n in names(node.target):
                seen[n] += 1
    return seen


def _names_dict(path):
    spec = importlib.util.spec_from_file_location("_rung", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    except Exception:
        return None
    names = getattr(mod, "NAMES", None)
    return names if isinstance(names, dict) else None


if __name__ == "__main__":
    import importlib.util  # noqa: F401  (used by _names_dict)
    failed = 0
    for path in sys.argv[1:]:
        problems = check(path)
        label = os.path.basename(path)
        if problems:
            failed += 1
            print(f"FAIL {label}")
            for p in problems:
                print(f"     {p}")
        else:
            print(f"PASS {label}")
    sys.exit(1 if failed else 0)
