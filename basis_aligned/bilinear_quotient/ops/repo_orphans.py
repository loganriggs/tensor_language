"""repo_orphans -- find root-level artefacts that NOTHING references. (ops lane, read-only by default.)

The bilinear_quotient root has accumulated ~1400 loose .py and ~2900 loose .json across the campaign. Some are
live evidence cited by ledger sections; most are historical. Archiving by guesswork would break the evidence
chain, so this decides by REFERENCE: an artefact is an orphan only when no ledger section, board entry,
backlog note, preregistration, script, shell wrapper or queue file mentions it -- counting a file's mention of
ITSELF as no evidence at all.

Orphans come in two tiers, because "nothing cites it" and "it never ran" are different facts:
  DEAD      no reference anywhere, and it never appears in runlogs/runner.log
  RAN_ONLY  the runner executed it, but nothing cites it or its results today
Both are archivable -- moving, not deleting -- but they go to different folders so the distinction survives.

Read-only. It prints what it found and writes nothing unless --json is given. Moving files is a separate,
deliberate step so that the decision and the action are never the same command.

Usage:
  python ops/repo_orphans.py                 # summary
  python ops/repo_orphans.py --list 40       # summary + the 40 largest orphans
  python ops/repo_orphans.py --json out.json # machine-readable, for an archiving step to consume
"""
import os, re, sys, json, collections, signal

signal.signal(signal.SIGPIPE, signal.SIG_DFL)   # `| head` should end the program, not raise BrokenPipeError

BQ = '/workspace/tensor_language/basis_aligned/bilinear_quotient'
TL = '/workspace/tensor_language'
PC = os.path.join(TL, 'basis_aligned', 'polynomial_causal')
CAND_EXT = ('.py', '.json', '.pt', '.png', '.html', '.txt', '.npy', '.jsonl')
# never archivable regardless of references: live infrastructure and curated documents
KEEP = {'queue.txt', 'BILIN18_CONNECTION.md', 'BENCHMARK_BACKLOG.md', 'MEMORY.md'}
TOKEN = re.compile(r'[A-Za-z0-9_][A-Za-z0-9_.\-/]*\.(?:py|json|pt|png|html|txt|npy|jsonl)')
WORD = re.compile(r'[A-Za-z0-9_]{6,}')


RUNNER_LOG = os.path.join(BQ, 'runlogs', 'runner.log')


def runner_stems():
    """Stems the runner has actually executed. Not a citation, but not nothing either."""
    try:
        txt = open(RUNNER_LOG, errors='ignore').read()
    except OSError:
        return set()
    return set(WORD.findall(txt))


def corpus_files():
    """Everything that could legitimately cite an artefact. NOT the runner log -- see runner_stems()."""
    out = []
    for d in (BQ, PC, TL):
        for f in sorted(os.listdir(d)):
            if f.endswith('.md') or f == 'queue.txt':
                out.append(os.path.join(d, f))
    for sub in ('ops', 'circuits'):
        p = os.path.join(BQ, sub)
        if not os.path.isdir(p):
            continue
        for f in sorted(os.listdir(p)):
            if f.endswith(('.py', '.sh', '.md', '.json', '.jsonl', '.conf')):
                out.append(os.path.join(p, f))
    for f in sorted(os.listdir(BQ)):
        if f.endswith('.py'):
            out.append(os.path.join(BQ, f))
    # Archived files still COUNT as references. Without this, archiving cascades: every receipt whose only
    # citation was an archived script becomes an orphan on the next scan, and the next, until the root is
    # empty. An artefact that an archived rung produced belongs with that rung, not on a second sweep.
    arch = os.path.join(BQ, 'archive')
    for dirpath, _dirs, files in os.walk(arch):
        for f in files:
            if f.endswith(('.py', '.md', '.json', '.sh', '.jsonl', '.txt')):
                out.append(os.path.join(dirpath, f))
    return out


def scan():
    cands = {}
    for f in sorted(os.listdir(BQ)):
        p = os.path.join(BQ, f)
        if os.path.isfile(p) and f.endswith(CAND_EXT) and f not in KEEP:
            cands[f] = os.path.getsize(p)
    stems = {os.path.splitext(f)[0]: f for f in cands}
    mentions = collections.defaultdict(set)
    for cf in corpus_files():
        try:
            txt = open(cf, 'r', errors='ignore').read()
        except OSError:
            continue
        base = os.path.basename(cf)
        for t in set(TOKEN.findall(txt)):
            b = os.path.basename(t)
            if b in cands and b != base:
                mentions[b].add(cf)
        for w in set(WORD.findall(txt)):
            if w in stems and stems[w] != base:
                mentions[stems[w]].add(cf)
    ran = runner_stems()
    orphans = {f: s for f, s in cands.items() if not mentions[f]}
    dead = {f: s for f, s in orphans.items() if os.path.splitext(f)[0] not in ran}
    ran_only = {f: s for f, s in orphans.items() if os.path.splitext(f)[0] in ran}
    return cands, mentions, orphans, dead, ran_only


if __name__ == '__main__':
    cands, mentions, orphans, dead, ran_only = scan()
    tot = sum(cands.values()); orph = sum(orphans.values())
    by_ext = collections.Counter(os.path.splitext(f)[1] for f in orphans)
    sz_ext = collections.Counter()
    for f, s in orphans.items():
        sz_ext[os.path.splitext(f)[1]] += s
    print(f'root artefacts: {len(cands)} files, {tot/1e6:.1f} MB')
    print(f'   DEAD      (no reference, never ran): {len(dead):>4} files, {sum(dead.values())/1e6:>7.1f} MB')
    print(f'   RAN_ONLY  (ran, nothing cites it)  : {len(ran_only):>4} files, {sum(ran_only.values())/1e6:>7.1f} MB')
    print(f'ORPHANS (nothing references them): {len(orphans)} files, {orph/1e6:.1f} MB '
          f'({100*orph/tot if tot else 0:.0f}% of bytes, {100*len(orphans)/len(cands) if cands else 0:.0f}% of files)')
    for e, n in by_ext.most_common():
        print(f'   {e:>7}  {n:>5} files  {sz_ext[e]/1e6:>8.1f} MB')
    if '--list' in sys.argv:
        k = int(sys.argv[sys.argv.index('--list') + 1])
        print(f'\nlargest {k} orphans:')
        for f, s in sorted(orphans.items(), key=lambda kv: -kv[1])[:k]:
            print(f'   {s/1e6:>8.2f} MB  {f}')
    if '--json' in sys.argv:
        out = sys.argv[sys.argv.index('--json') + 1]
        json.dump({'orphans': sorted(orphans), 'dead': sorted(dead), 'ran_only': sorted(ran_only),
                   'sizes': orphans,
                   'n_candidates': len(cands), 'orphan_bytes': orph},
                  open(out, 'w'), indent=1)
        print(f'\nwrote {out}')
