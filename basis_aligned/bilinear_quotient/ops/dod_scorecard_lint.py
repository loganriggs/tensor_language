#!/usr/bin/env python3
# BQGATE: LIBRARY -- CPU ledger check (review 32): scorecard citations <-> receipt files, both directions.
"""Usage: python dod_scorecard_lint.py [first_version=164]. Reads claude_hourly_review/*_DOD_SCORECARD.md, SHARED_READOUT_COMPONENTS.md and
REVIEW_*.md; collects every cited vNNN; lists (a) citations with no receipt file among circuits/followups/*_vNNN*_result.json on this lane's
prefixes, (b) receipts with version >= first_version on this lane's prefixes that no scorecard / shared log / review cites."""
import re, sys, glob, pathlib
first = int(sys.argv[1]) if len(sys.argv) > 1 else 164
root = pathlib.Path(__file__).resolve().parent.parent; rev = root.parent / "claude_hourly_review"
prefixes = ("pronoun_", "aspectual_dod", "perfect_number", "noun_number", "correlative_", "selection_dod", "person_", "temporal_")
text = "".join(p.read_text() for p in list(rev.glob("*_DOD_SCORECARD.md")) + [rev / "SHARED_READOUT_COMPONENTS.md"] + list(rev.glob("REVIEW_*.md")))
cited = {int(v) for v in re.findall(r"\bv(\d{2,3})b?\b", text)}
receipts = {}
for f in glob.glob(str(root / "circuits/followups/*_result.json")):
    name = pathlib.Path(f).name
    if not name.startswith(prefixes): continue
    m = re.search(r"_v(\d+)b?_result", name)
    if m: receipts.setdefault(int(m.group(1)), []).append(name)
missing_files = sorted(v for v in cited if v >= first and v not in receipts)
uncited = sorted(v for v in receipts if v >= first and v not in cited)
print(f"cited versions >= v{first}: {len([v for v in cited if v >= first])}; receipt versions >= v{first}: {len([v for v in receipts if v >= first])}")
print("cited but no receipt file:", missing_files or "none")
print("receipt but never cited:", uncited or "none")
for v in uncited: print("   ", receipts[v])
