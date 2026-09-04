"""archive_orphans -- move the artefacts `repo_orphans` proves nobody references into archive/<date>/.

Deliberately a SEPARATE script from ops/repo_orphans.py: deciding what is dead and acting on it should never
be the same command. This re-runs the scan at move time (so a file that gained a citation since the last scan
is spared), moves tracked files with `git mv` to keep their history and untracked ones with `os.replace`, and
writes a manifest so every move is reversible by name.

Nothing is moved without --apply; the default prints the plan.

Usage:
  python ops/archive_orphans.py                 # dry run
  python ops/archive_orphans.py --apply         # move, write manifest, print restore instructions
"""
import os, sys, json, time, subprocess, collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repo_orphans import scan, BQ, TL

ARCHIVE = os.path.join(BQ, 'archive')


def tracked_set():
    """One `git ls-files` for the whole directory beats one call per candidate file."""
    r = subprocess.run(['git', '-C', TL, 'ls-files', 'basis_aligned/bilinear_quotient'],
                       capture_output=True, text=True)
    return set(r.stdout.splitlines())


if __name__ == '__main__':
    apply_ = '--apply' in sys.argv
    _, _, orphans, dead, ran_only = scan()
    if not orphans:
        print('no orphans; nothing to do'); raise SystemExit(0)
    day = time.strftime('%Y-%m-%d', time.gmtime())
    dest_root = os.path.join(ARCHIVE, day)
    tiers = {'dead': dead, 'ran-but-uncited': ran_only}
    total = sum(orphans.values())
    by_ext = collections.Counter(os.path.splitext(f)[1] for f in orphans)
    print(f'{len(orphans)} orphans, {total/1e6:.1f} MB -> archive/{day}/')
    for tier, files in tiers.items():
        print(f'   {tier:<16} {len(files):>4} files  {sum(files.values())/1e6:>7.1f} MB')
    for e, n in by_ext.most_common():
        print(f'   {e:>7}  {n:>5}')
    if not apply_:
        print('\ndry run; pass --apply to move'); raise SystemExit(0)

    TRACKED = tracked_set()
    moved, failed = [], []
    tier_of = {}
    for tier, files in tiers.items():
        os.makedirs(os.path.join(dest_root, tier), exist_ok=True)
        for f in files:
            tier_of[f] = tier
    for f in sorted(orphans):
        src = os.path.join(BQ, f)
        rel = os.path.relpath(src, TL)
        dst = os.path.join(dest_root, tier_of[f], f)
        if not os.path.isfile(src):
            continue                                    # vanished since the scan; skip quietly
        try:
            if rel in TRACKED:
                r = subprocess.run(['git', '-C', TL, 'mv', rel,
                                    os.path.relpath(dst, TL)], capture_output=True, text=True)
                if r.returncode != 0:
                    failed.append((f, r.stderr.strip()[:120])); continue
            else:
                os.replace(src, dst)
            moved.append(f)
        except OSError as e:
            failed.append((f, str(e)))
    man = os.path.join(dest_root, 'MANIFEST.json')
    json.dump({'archived_utc': time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime()),
               'rule': 'no ledger section, board entry, backlog note, preregistration, script, '
                       'shell wrapper or queue file mentions the file (self-mentions do not count)',
               'tool': 'ops/repo_orphans.py + ops/archive_orphans.py',
               'tiers': {'dead': 'no reference anywhere and never executed by the runner',
                         'ran-but-uncited': 'the runner executed it, but nothing cites it or its results today'},
               'n_moved': len(moved), 'bytes': total,
               'files': {t_: sorted(f_ for f_ in moved if tier_of[f_] == t_) for t_ in tiers},
               'failed': failed}, open(man, 'w'), indent=1)
    print(f'\nmoved {len(moved)} files ({total/1e6:.1f} MB); manifest at archive/{day}/MANIFEST.json')
    if failed:
        print(f'FAILED {len(failed)}:')
        for f, e in failed[:10]:
            print(f'   {f}: {e}')
    print(f'restore one:  git -C {TL} mv basis_aligned/bilinear_quotient/archive/{day}/<tier>/<name> '
          f'basis_aligned/bilinear_quotient/<name>')
