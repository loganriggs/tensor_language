#!/usr/bin/env python3
"""Answer, before any GPU work, whether a screen's execution would be refused as a duplicate.

The screen ledger refuses an append whose execution key already appears
(`circuit_fast_screen_ledger._execution_key`):

    (candidate_id, prior_art_sha256, spec_sha256, authority_sha256,
     max_forward_calls, max_example_evaluations, max_evidence_bytes)

Not one of those fields is derived from the screen's *output*. Every one is fixed by the
runner's PROTOCOL literals and by compiling the frozen candidate rows -- all CPU, all
available before a single forward call. Yet the check runs at publish time, so a duplicate
costs a full screen's compute plus a human round trip to diagnose it. This tool moves the
same check to the front, in milliseconds.

Case it was built from: 2026-09-05T05:02Z, run_circuit_fast_screen_task14_select_cross_noun
ran to completion and was then refused against ledger entry 25 -- identical in all seven
fields. The fix varied one protocol literal (prior_art_sha256). ~11 s of compute and 4 min of
wall clock, both avoidable by reading the runner.

Usage:
    python ops/screen_preflight.py --runner run_circuit_fast_screen_task14_select_cross_noun
    python ops/screen_preflight.py --all          # every runner that targets the ledger
Exit status is 1 when a duplicate is predicted, so this can gate a run.
"""
from __future__ import annotations

import argparse
import importlib
import json
import signal
import sys
from pathlib import Path

signal.signal(signal.SIGPIPE, signal.SIG_DFL)

OPS = Path(__file__).resolve().parent
ROOT = OPS.parent
LEDGER = ROOT / "circuits/fast_screen_ledger.jsonl"

KEY_FIELDS = (
    "candidate_id", "prior_art_sha256", "spec_sha256", "authority_sha256",
    "max_forward_calls", "max_example_evaluations", "max_evidence_bytes",
)
# The three fields every engine exposes as protocol literals. A partial check compares
# only these; it can prove "not a duplicate" but only suspect "duplicate".
PARTIAL_FIELDS = ("candidate_id", "prior_art_sha256", "authority_sha256")


def _first_attr(obj, *names):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _price(mapping, *names):
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def protocol_of(module):
    """The runner's frozen protocol object, whichever engine it targets (may be None)."""
    return _first_attr(module, "PROTOCOL", "CONFIG", "MANAGED_CONFIG")


def candidate_of(module, protocol):
    """The frozen candidate module.

    Runners name it three ways: hung off the protocol (task14 shared engine), imported
    alongside a CandidateRunConfig (managed engine), or implied by module-level constants.
    Rather than encode those conventions, find the imported module that satisfies the
    candidate contract -- TASK_ID plus build_rows.
    """
    if protocol is not None and getattr(protocol, "candidate", None) is not None:
        return protocol.candidate
    found = [
        value for value in vars(module).values()
        if isinstance(value, type(sys)) and hasattr(value, "TASK_ID") and hasattr(value, "build_rows")
    ]
    if len(found) == 1:
        return found[0]
    if not found:
        return None
    raise LookupError(f"{module.__name__} imports {len(found)} candidate modules; ambiguous")


def _digest(protocol, module, *names):
    """A digest field, taken from the protocol object or from module-level constants."""
    if protocol is not None:
        value = _first_attr(protocol, *names)
        if value is not None:
            return value
    return _first_attr(module, *(n.upper() for n in names))


def execution_key(module):
    """Compute as much of the ledger execution key as CPU-only work allows.

    Returns (key_dict, complete) -- complete is True only when all seven fields were
    derived, which needs the candidate to expose compile_plan().
    """
    protocol = protocol_of(module)
    candidate = candidate_of(module, protocol)
    if candidate is None:
        raise LookupError(f"{module.__name__} names no candidate module")

    key = {
        "candidate_id": candidate.TASK_ID,
        "prior_art_sha256": _digest(
            protocol, module, "prior_art_sha256", "expected_prior_art_sha256"),
        "authority_sha256": _digest(
            protocol, module, "expected_authority_sha256", "authority_sha256"),
    }
    compile_plan = getattr(candidate, "compile_plan", None)
    if compile_plan is None:
        return key, False
    plan = compile_plan(candidate.build_rows())
    price = plan["price"]
    key["spec_sha256"] = str(plan["compiled_sha256"])
    key["max_forward_calls"] = _price(price, "forward_calls")
    key["max_example_evaluations"] = _price(price, "example_evaluations")
    key["max_evidence_bytes"] = _price(price, "raw_numeric_evidence_bytes", "evidence_bytes")
    complete = all(key.get(field) is not None for field in KEY_FIELDS)
    return key, complete


def ledger_entries(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def collisions(key, entries, complete):
    """Ledger entries this execution would collide with."""
    fields = KEY_FIELDS if complete else PARTIAL_FIELDS
    hits = []
    for index, entry in enumerate(entries):
        if all(entry.get(field) == key.get(field) for field in fields):
            hits.append((index, entry))
    return hits


def check(name, ledger_path=LEDGER):
    if str(OPS) not in sys.path:
        sys.path.insert(0, str(OPS))
    module = importlib.import_module(name)
    key, complete = execution_key(module)
    entries = ledger_entries(ledger_path)
    hits = collisions(key, entries, complete)
    return key, complete, hits, entries


def describe(name, key, complete, hits, entries):
    scope = "full 7-field key" if complete else f"{len(PARTIAL_FIELDS)}-of-7 key (candidate lacks compile_plan)"
    print(f"{name}\n  checked: {scope} against {len(entries)} ledger entries")
    for field in KEY_FIELDS:
        value = key.get(field)
        # abbreviate digests only; ids and budget caps are shown whole
        shown = value[:8] if field.endswith("_sha256") and isinstance(value, str) else value
        marker = " " if field in key and value is not None else "?"
        print(f"   {marker} {field:28s} {shown}")
    if not hits:
        print("  CLEAR: no ledger entry matches; this execution would be accepted.")
        return 0
    verdict = "DUPLICATE" if complete else "LIKELY DUPLICATE"
    print(f"  {verdict}: would be refused against {len(hits)} entry/entries:")
    for index, entry in hits:
        print(f"     entry {index}  {entry.get('request_id')}")
    if not complete:
        print("  (partial check: spec_sha256 and the three budget caps were not derivable,")
        print("   so confirm the spec/budget really are identical before treating this as final.)")
    differing = [f for f in (KEY_FIELDS if complete else PARTIAL_FIELDS)]
    print(f"  To proceed, vary one key field -- {', '.join(differing)} --")
    print("  e.g. register a distinguishing prior-art receipt, as the 05:02 cross-noun fix did.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runner", help="runner module name (no .py)")
    parser.add_argument("--all", action="store_true", help="check every ledger-targeting runner")
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    args = parser.parse_args()

    if args.all:
        names = sorted(p.stem for p in OPS.glob("run_circuit_fast_screen_*.py"))
    elif args.runner:
        names = [args.runner.removesuffix(".py")]
    else:
        parser.error("give --runner NAME or --all")

    status = 0
    for name in names:
        try:
            key, complete, hits, entries = check(name, args.ledger)
        except Exception as error:  # a runner that cannot be inspected is reported, not fatal
            print(f"{name}\n  SKIPPED: {type(error).__name__}: {error}")
            continue
        status |= describe(name, key, complete, hits, entries)
        print()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
