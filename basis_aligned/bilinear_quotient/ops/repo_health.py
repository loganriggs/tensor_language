"""repo_health -- one command that says whether this working tree is in a fit state.

Every check here exists because something went wrong without it being noticed:

  ledger        a ledger section citing a receipt that does not resolve, or a Price line the auditor
                cannot parse, breaks the evidence chain silently (SS2876 is still UNAUDITABLE).
  orphans       after an archive sweep the scan must converge to zero, or the sweep was incomplete.
  renames       `git mv` + a pathspec commit covering one side leaves the other side staged. Hit twice
                on 2026-09-04; the tell is bare `A ` or `D ` lines in `git status --porcelain`.
  bulk          a large untracked directory is one `git add -A` away from being committed (.fitcache
                is 1.6 GB).
  tests         the ops-lane tests are fast; there is no reason for them to be red.
  lanes         a silent lane is invisible in git, because BOTH agents commit as the same author -- a
                per-author recency check returns one name and tells you nothing. The board is the only
                place the lanes are distinguishable, so liveness is read from its entry headers. On
                2026-09-04 one lane went quiet for ~4 h and it was only noticed by inference from an idle
                queue.

  queue         a lane at depth < 2 means the runner is about to idle, historically the single largest
                loss bucket (36% of a measured 48 h span).

Exit 0 = healthy. Exit 1 = at least one check failed. Read-only: it changes nothing.

Usage:  python ops/repo_health.py [--quiet]
"""
import os, re, sys, json, datetime, subprocess

BQ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TL = os.path.dirname(os.path.dirname(BQ))
PY = sys.executable
BULK_MB = 100


def run(cmd, cwd=TL):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def check_ledger():
    r = run([PY, os.path.join(BQ, 'ops', 'audit_ledger_prices.py'), '--since', '1'])
    out = r.stdout + r.stderr
    m = re.search(r'checked (\d+) sections .*?; (\d+) mismatched', out)
    if not m:
        return False, 'auditor produced no parseable summary'
    checked, bad = int(m.group(1)), int(m.group(2))
    un = re.search(r'UNAUDITABLE: (\d+) section', out)
    n_un = int(un.group(1)) if un else 0
    ok = bad == 0
    note = f'{checked} sections, {bad} mismatched'
    if n_un:
        note += f', {n_un} UNAUDITABLE'          # known and tolerated; not a failure by itself
    return ok, note


def check_orphans():
    r = run([PY, os.path.join(BQ, 'ops', 'repo_orphans.py')])
    m = re.search(r'ORPHANS \(nothing references them\): (\d+) files', r.stdout)
    if not m:
        return False, 'orphan scan produced no parseable summary'
    n = int(m.group(1))
    return n == 0, f'{n} unreferenced artefacts at the root'


def check_renames():
    r = run(['git', 'status', '--porcelain', 'basis_aligned/bilinear_quotient'])
    staged = [ln for ln in r.stdout.splitlines() if ln[:1] in 'AMDR' and ln[1:2] == ' ']
    return not staged, (f'{len(staged)} staged-but-uncommitted paths'
                        if staged else 'no half-committed renames')


def _disk_bytes(path):
    """Bytes actually occupied, via st_blocks -- NOT st_size.

    `induction_centered_fixed_geometry_rung592_invalid_evidence/` holds SPARSE .npy files: 4.9 GB apparent,
    11 MB on disk. Summing st_size made this check scream about a directory costing 11 MB, and a health check
    that cries wolf is worse than no health check.
    """
    try:
        if os.path.isfile(path):
            return os.stat(path).st_blocks * 512
        return sum(os.stat(os.path.join(dp, f)).st_blocks * 512
                   for dp, _d, fs in os.walk(path) for f in fs
                   if os.path.exists(os.path.join(dp, f)))
    except OSError:
        return 0


def check_bulk():
    r = run(['git', 'status', '--porcelain', '--ignored=no', 'basis_aligned/bilinear_quotient'])
    big = []
    for ln in r.stdout.splitlines():
        if not ln.startswith('??'):
            continue
        rel = ln[3:].strip()
        sz = _disk_bytes(os.path.join(TL, rel))
        if sz > BULK_MB * 1e6:
            big.append((rel, sz / 1e6))
    return not big, ('untracked bulk: ' + ', '.join(f'{n} ({s:.0f} MB on disk)' for n, s in big)
                     if big else f'no untracked path over {BULK_MB} MB on disk')


def check_tests():
    tests = [os.path.join(BQ, 'ops', t) for t in
             ('test_repo_orphans.py', 'test_frontier_fitcache.py', 'test_armsweep.py',
              'test_frontier_evalarms.py')
             if os.path.exists(os.path.join(BQ, 'ops', t))]
    if not tests:
        return True, 'no ops-lane tests found'
    r = run([PY, '-m', 'pytest', '-q'] + tests, cwd=BQ)
    tail = [l for l in (r.stdout + r.stderr).strip().splitlines() if l.strip()]
    return r.returncode == 0, (tail[-1][:70] if tail else 'no output')


BOARD = os.path.join(TL, "AGENT_BOARD.md")
LANE_HEADER = re.compile(r"^###\s+(\d{4}-\d\d-\d\dT\d\d:\d\d)Z\s+[-—]+\s*([A-Za-z]+)")
LANE_QUIET_MIN = 90
# Only the two live agent lanes gate this check. Board headers also carry "USER DIRECTIVE", historical
# "GPT"/"CLAUDE" entries and similar; counting those made the first version report four silent "lanes",
# three of which are not agents at all. Same rule as the sparse-file bulk alarm: a check that cries wolf
# gets ignored.
AGENT_LANES = ("Claude", "Codex")


def check_lanes():
    """Minutes since each lane last wrote to the board. Both agents share a git author, so this is the
    only lane-distinguishing liveness signal available."""
    if not os.path.exists(BOARD):
        return True, "no board"
    latest = {}
    for line in open(BOARD, errors="ignore"):
        m = LANE_HEADER.match(line)
        if not m:
            continue
        try:
            when = datetime.datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M")
        except ValueError:
            continue
        lane = m.group(2)
        if lane not in AGENT_LANES:
            continue
        if lane not in latest or when > latest[lane]:
            latest[lane] = when
    if not latest:
        return True, "no parseable board entries"
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None, microsecond=0)
    parts, quiet = [], []
    for lane, when in sorted(latest.items(), key=lambda kv: -kv[1].timestamp()):
        mins = (now - when).total_seconds() / 60.0
        parts.append(f"{lane} {mins:.0f}m")
        if mins > LANE_QUIET_MIN:
            quiet.append(f"{lane} silent {mins:.0f}m")
    return not quiet, ", ".join(parts) + (f"  ({'; '.join(quiet)})" if quiet else "")


def check_queue():
    depths = {}
    for name, f in (('lane 1', 'queue.txt'), ('lane 2', 'queue2.txt')):
        p = os.path.join(BQ, f)
        depths[name] = sum(1 for _ in open(p)) if os.path.exists(p) else 0
    note = ', '.join(f'{k}={v}' for k, v in depths.items())
    return depths['lane 1'] >= 2, note + '  (lane 1 wants >= 2)'


CHECKS = [('ledger', check_ledger), ('orphans', check_orphans), ('renames', check_renames),
          ('bulk', check_bulk), ('tests', check_tests), ('lanes', check_lanes), ('queue', check_queue)]

if __name__ == '__main__':
    quiet = '--quiet' in sys.argv
    failed = []
    for name, fn in CHECKS:
        try:
            ok, note = fn()
        except Exception as e:                     # a check that crashes is a failed check
            ok, note = False, f'{type(e).__name__}: {e}'
        if not ok:
            failed.append(name)
        if not quiet or not ok:
            print(f'{"ok  " if ok else "FAIL"}  {name:<9} {note}')
    if failed:
        print(f'\n{len(failed)} check(s) failed: {", ".join(failed)}')
        raise SystemExit(1)
    if not quiet:
        print('\nall checks passed')
