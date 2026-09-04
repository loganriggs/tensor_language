#!/usr/bin/env python3
"""derive.py -- build a child probe script from a parent by the substitution every Claude lane-1 rung has been doing inline.

Added 2026-09-04 00:06Z (ops-efficiency review; proposed 22:06 and 23:06 as the top time sink: each rung was a ~60-line inline
python heredoc doing exactly this). One call does: docstring swap, name/prereg rename, PRIOR line (+ its "# §N" comment),
frozen HASHES of the new prereg + new prior results, BARS/NULLS constants, body splice between an anchor line and the smoke
print, then gate + dry-run. Nothing here touches a registered script's semantics: the child is a NEW file.

Usage:
  python3 ops/derive.py PARENT.py CHILD.py --prereg PREREG.md --prior PRIOR_results.json --prior-sec 2768 \
      --doc doc.txt --bars 'BARS = {...}' --nulls 'NULLS = {...}' --anchor '    def next_frame(s_):' --body body.py \
      [--extra 'OLD=>NEW' ...]
The doc file holds the whole new module docstring (starting at '#!/usr/bin/env python' and ending at the closing triple quote).
--extra applies literal OLD=>NEW replacements after everything else (e.g. smoke-print key lists).
"""
import argparse, hashlib, pathlib, re, subprocess, sys


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parent"); ap.add_argument("child")
    ap.add_argument("--prereg", required=True); ap.add_argument("--prior", required=True); ap.add_argument("--prior-sec", required=True)
    ap.add_argument("--doc", required=True); ap.add_argument("--bars", required=True); ap.add_argument("--nulls", required=True)
    ap.add_argument("--anchor", required=True); ap.add_argument("--body", required=True)
    ap.add_argument("--extra", action="append", default=[]); ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args()
    parent, child = pathlib.Path(a.parent), pathlib.Path(a.child)
    pn, cn = parent.stem, child.stem
    src = parent.read_text()
    src = pathlib.Path(a.doc).read_text().rstrip("\n") + "\n" + src[src.index("import json, os, sys, time"):]
    src = src.replace(pn, cn).replace(pn.upper() + "_PREREGISTRATION", cn.upper() + "_PREREGISTRATION")
    prior = pathlib.Path(a.prior)
    src, n = re.subn(r'PRIOR = ROOT / "[^"]*"   # §\d+\n', f'PRIOR = ROOT / "{prior.name}"   # §{a.prior_sec}\n', src)
    assert n == 1, "PRIOR line not found (expected `PRIOR = ROOT / \"...\"   # §N`)"
    src, n = re.subn(r'HASHES = \{PREREG: "[0-9a-f]*", PRIOR: "[0-9a-f]*",', f'HASHES = {{PREREG: "{sha(a.prereg)}", PRIOR: "{sha(prior)}",', src)
    assert n == 1, "HASHES line not found"
    src, n = re.subn(r'BARS = \{[^\n]*\}\nNULLS = \{[^\n]*\}\n', a.bars.rstrip("\n") + "\n" + a.nulls.rstrip("\n") + "\n", src)
    assert n == 1, "BARS/NULLS lines not found"
    i = src.index(a.anchor); j = src.index("    if smoke:\n        print(json.dumps")
    src = src[:i] + pathlib.Path(a.body).read_text() + src[j:]
    for e in a.extra:
        old, new = e.split("=>", 1); assert old in src, f"--extra OLD not found: {old!r}"; src = src.replace(old, new)
    child.write_text(src)
    print(f"wrote {child} ({len(src.splitlines())} lines)")
    if a.no_check:
        return
    ops = child.parent
    r = subprocess.run([sys.executable, str(ops / "gate.py"), str(child)], capture_output=True, text=True)
    print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip()[-300:])
    if r.returncode != 0:
        sys.exit(1)
    r = subprocess.run([sys.executable, str(child)], capture_output=True, text=True, env={**__import__("os").environ, "BQLIB_DRYRUN": "1", "BQLIB_NO_MODEL": "1"})
    print((r.stdout.strip() or r.stderr.strip())[-200:])
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
