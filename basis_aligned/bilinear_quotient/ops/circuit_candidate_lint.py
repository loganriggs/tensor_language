"""circuit_candidate_lint -- mechanical pre-review checks for FIT screen candidates. READ-ONLY.

Motivation, measured 14:00-15:04: GPU science is now ~2 s per screen while wall-clock per screen is ~20 min
(Codex's own audit: plan-to-terminal ~42 min vs 1.990 s of compute). The bottleneck is not the machine, it is
the serial design -> review -> fix round-trip between two agents. Both blockers I raised by hand today are
mechanically detectable, so they should not cost a round-trip:

  ENDPOINT_MERGE  the answer and foil are punctuation whose concatenation is a SINGLE token, so the model's
                  natural continuation is invisible to the answer-minus-foil contrast. Found in quote_parity:
                  answer '"' (1), foil '.' (13), but '."' is token 526 -- neither.

  ORDER_PREDICTS  a surface ORDER inside the prompt predicts the answer across every row of a cell, so a model
                  can score by ordinal position without representing the causal variable. Found in the pronoun
                  candidate: `_introduction()` always names the woman first, so "actor mentioned first" <=>
                  " she" in all 128 rows.

This does not replace review; it removes the two classes that are mechanical. It never edits or runs a model.

Usage:  python ops/circuit_candidate_lint.py ops/circuit_fast_screen_candidate_pronoun.py [--groups N]
"""
import importlib.util
import os
import sys
import collections

import tiktoken

ENCODING = tiktoken.get_encoding("gpt2")


def _load(path):
    spec = importlib.util.spec_from_file_location("candidate_under_lint", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["candidate_under_lint"] = mod
    spec.loader.exec_module(mod)
    return mod


def lint_endpoint_merge(rows):
    """Answer/foil surface forms whose concatenation is a single token the score cannot see."""
    findings = []
    seen = set()
    for r in rows:
        a, f = r.get("base_answer_id"), r.get("base_foil_id")
        if a is None or f is None:
            continue
        sa, sf = ENCODING.decode([a]), ENCODING.decode([f])
        for first, second in ((sa, sf), (sf, sa)):
            merged = first + second
            ids = ENCODING.encode(merged)
            if len(ids) == 1 and ids[0] not in (a, f) and merged not in seen:
                seen.add(merged)
                findings.append(
                    f"{merged!r} is a single token ({ids[0]}) but is neither the answer "
                    f"({sa!r}={a}) nor the foil ({sf!r}={f}); a model preferring it scores on neither side")
    return findings


def lint_order_predicts_answer(rows):
    """A constant mention-order that determines the answer across a whole capability cell."""
    ident_fields = [k for k in ("base_antecedent", "base_actor", "antecedent", "actor") if k in (rows[0] or {})]
    label_fields = [k for k, v in (rows[0] or {}).items()
                    if isinstance(v, str) and k.endswith("_label")]
    if not ident_fields or len(label_fields) < 2:
        return ["skipped: no antecedent/actor field plus >=2 *_label fields to order"]
    af = ident_fields[0]
    findings = []
    # Group per cell AND pooled. A confound can be invisible within a cell -- if every row of
    # "A1/active/toward_donor" carries the same answer, no within-cell feature can vary with it -- while
    # being perfect across the dataset. The first version of this lint grouped per cell only and reported
    # "ok" on a candidate whose introduction order I had already shown, by hand, to determine the answer.
    by_cell = collections.defaultdict(list)
    for r in rows:
        by_cell[r.get("capability_cell_id", "?")].append(r)
    by_cell["ALL ROWS POOLED"] = list(rows)
    for cell, rs in sorted(by_cell.items()):
        pairs = set()
        for r in rs:
            text, actor = r.get("base_text", ""), r.get(af)
            positions = {lf: text.find(r[lf]) for lf in label_fields if r.get(lf) and text.find(r[lf]) >= 0}
            if len(positions) < 2 or actor is None:
                continue
            first = min(positions, key=positions.get)
            actor_is_first = r.get(first) == actor
            pairs.add((actor_is_first, r.get("base_answer_id")))
        answers = {a for _, a in pairs}
        by_flag = collections.defaultdict(set)
        for flag, a in pairs:
            by_flag[flag].add(a)
        if len(answers) > 1 and all(len(v) == 1 for v in by_flag.values()) and len(by_flag) > 1:
            findings.append(
                f"cell {cell}: 'actor is the first-mentioned label' PERFECTLY predicts the answer "
                f"({dict((k, sorted(v)) for k, v in by_flag.items())}) -- a model can score by ordinal "
                f"position without representing the causal variable")
    return findings or ["ok: mention order does not by itself determine the answer in any cell"]


CHECKS = (("ENDPOINT_MERGE", lint_endpoint_merge), ("ORDER_PREDICTS", lint_order_predicts_answer))

if __name__ == "__main__":
    path = sys.argv[1]
    groups = int(sys.argv[sys.argv.index("--groups") + 1]) if "--groups" in sys.argv else None
    mod = _load(os.path.abspath(path))
    kw = {} if groups is None else {"groups": groups}
    rows = mod.build_rows(mod.TASK_ID, **kw)   # every candidate module exposes TASK_ID and build_rows
    print(f"{os.path.basename(path)}: {len(rows)} rows, "
          f"{len({r.get('capability_cell_id') for r in rows})} capability cells")
    bad = 0
    for name, fn in CHECKS:
        out = fn(rows)
        for line in out:
            flag = "ok" if line.startswith(("ok:", "skipped:")) else "FLAG"
            if flag == "FLAG":
                bad += 1
            print(f"  {flag:<4} {name:<15} {line}")
    print(f"\n{bad} flag(s)")
    raise SystemExit(1 if bad else 0)
