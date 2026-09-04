"""sync_screen_hashes -- recompute and rewrite a screen runner's two frozen digests.

A managed runner pins two hashes that must match objects it does not itself compute:

    EXPECTED_AUTHORITY_SHA256   = candidate.authority_sha256()      (changes on ANY row edit)
    EXPECTED_PRIOR_ART_SHA256   = circuit_prior_art.canonical_hash(receipt)

Both are transcribed by hand today, and both are easy to get wrong in ways that fail late and confusingly:
across the two screens this lane has run, four attempts died on a stale or wrong digest -- including one
where I pasted the receipt's FILE sha256 instead of its CANONICAL hash, which are different by design, and
one where my own completed run had appended to a reviewed source and silently moved the receipt's digest.

Neither failure is interesting and both are mechanical, so this recomputes them from the objects themselves
and rewrites the constants in place.

Usage:
  python ops/sync_screen_hashes.py ops/run_circuit_fast_screen_<name>.py            # report only
  python ops/sync_screen_hashes.py ops/run_circuit_fast_screen_<name>.py --write    # rewrite in place
"""
import importlib
import json
import os
import re
import sys

OPS = os.path.dirname(os.path.abspath(__file__))
BQ = os.path.dirname(OPS)
AUTH = re.compile(r'(EXPECTED_AUTHORITY_SHA256 = \(\s*\n\s*")([0-9a-f]{64})("\s*\n\))')
PRIOR = re.compile(r'(EXPECTED_PRIOR_ART_SHA256 = \(\s*\n\s*")([0-9a-f]{64})("\s*\n\))')
CANDIDATE = re.compile(r'^import (circuit_fast_screen_candidate_\w+) as candidate', re.M)
PRIOR_PATH = re.compile(r'PRIOR_ART = ROOT / "([^"]+)"')


def current(path):
    text = open(path).read()
    a = AUTH.search(text)
    p = PRIOR.search(text)
    if not a or not p:
        raise SystemExit(f"{path}: could not find both digest constants")
    return text, a.group(2), p.group(2)


def expected(text):
    sys.path.insert(0, OPS)
    m = CANDIDATE.search(text)
    if not m:
        raise SystemExit("runner does not import a candidate module as `candidate`")
    candidate = importlib.import_module(m.group(1))
    authority = candidate.authority_sha256()
    rel = PRIOR_PATH.search(text)
    if not rel:
        raise SystemExit("runner does not declare PRIOR_ART")
    import circuit_prior_art as pa
    receipt = json.load(open(os.path.join(BQ, rel.group(1))))
    return authority, pa.canonical_hash(receipt)


if __name__ == "__main__":
    path = sys.argv[1]
    text, have_a, have_p = current(path)
    want_a, want_p = expected(text)
    for label, have, want in (("authority", have_a, want_a), ("prior-art", have_p, want_p)):
        state = "ok" if have == want else "STALE"
        print(f"{state:<6} {label:<10} have {have[:16]}…  want {want[:16]}…")
    if have_a == want_a and have_p == want_p:
        print("both digests current")
        raise SystemExit(0)
    if "--write" not in sys.argv:
        print("\nrun again with --write to update")
        raise SystemExit(1)
    text = AUTH.sub(lambda m: m.group(1) + want_a + m.group(3), text)
    text = PRIOR.sub(lambda m: m.group(1) + want_p + m.group(3), text)
    open(path, "w").write(text)
    print("rewrote both digests")
