#!/usr/bin/env python3
# BQGATE: LIBRARY -- CPU helper (review 24): insert one scorecard row and one SHARED_READOUT_COMPONENTS line in a single call.
"""Usage: python dod_scorecard_row.py <SCORECARD.md path> "<| n | ... |>" "<shared line>"   or   --json <spec.json> (row, shared and an optional document replacement in one call)  -- the row goes above '## Five-property status'
(appended at the end if that header is absent); the shared line is appended to claude_hourly_review/SHARED_READOUT_COMPONENTS.md.
Nothing is committed; follow with ops/dod_record.sh as usual."""
import sys, pathlib, json
if sys.argv[1] == "--json":          # review 29: {"scorecard": path, "row": str, "shared": str, "doc": path?, "old": str?, "new": str?}
    spec = json.load(open(sys.argv[2])); card, row, shared = pathlib.Path(spec["scorecard"]), spec["row"].rstrip("\n") + "\n", spec["shared"].rstrip("\n") + "\n"
    if spec.get("doc"):
        d = pathlib.Path(spec["doc"]); t = d.read_text(); assert spec["old"] in t, "doc anchor not found"; d.write_text(t.replace(spec["old"], spec["new"], 1)); print("doc updated", d.name)
else:
    card, row, shared = pathlib.Path(sys.argv[1]), sys.argv[2].rstrip("\n") + "\n", sys.argv[3].rstrip("\n") + "\n"
s = card.read_text(); key = "\n## Five-property status"
card.write_text(s.replace(key, row + key, 1) if key in s else s.rstrip("\n") + "\n" + row)
q = card.parent / "SHARED_READOUT_COMPONENTS.md"; q.write_text(q.read_text() + shared)
print("row added to", card.name, "; shared line appended")
