# BQGATE: LIBRARY  -- a test runner, not an experiment; see ops/gate.py.
#
# ops/test_fast.py -- the fast suite. Target: under 5 seconds, no GPU, no model.
#
# WHY. Every check below encodes a mistake that already cost a real run this session. Lessons in
# LESSONS.md rely on me or Codex remembering them at the moment of writing a fork, and the record shows
# that is not reliable: LESSON 85 (a fabricated per-role triple) was repeated one section after it was
# written, and the covered-input control polarity was inherited backwards four times. A check that runs
# in a second does not depend on anyone's memory.
#
# RUN:   python3 ops/test_fast.py            (exit 0 = all pass)
# It is also invoked by ops/enqueue.sh, so a broken library cannot reach the GPU.
import json
import os
import subprocess
import sys
import tempfile
import time

os.environ['BQLIB_NO_MODEL'] = '1'          # skips the 6.5s model load; see bqlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BQ = os.path.dirname(HERE)

import bqlib as B                                                          # noqa: E402
import torch                                                              # noqa: E402

FAILS = []
T0 = time.time()


def check(name, cond, detail=''):
    if cond:
        print(f'  ok    {name}')
    else:
        print(f'  FAIL  {name}   {detail}')
        FAILS.append(name)


# ---------------------------------------------------------------- bqlib pure helpers

def test_rk_key_is_order_independent():
    """Cache keys must not depend on dict iteration order, or two identical builds get two entries
    and a 'cache HIT' can serve rows built from a different spec."""
    a = B._rk_key({'mlp': 768, 'attn': 256})
    b = B._rk_key({'attn': 256, 'mlp': 768})
    check('_rk_key: dict order does not change the key', a == b, f'{a} != {b}')
    check('_rk_key: None and int pass through unchanged',
          B._rk_key(None) is None and B._rk_key(512) == 512)


def test_inertness_pairs_splits_by_table_rank():
    """S1765/S1936: same table_rank => exactly inert at covered inputs; different => must move them.
    Four forks inherited this control with the polarity backwards (S1946/S1949/S1951/S1955)."""
    plan = (('mix25m512', {'mlp': 768, 'attn': 256}, 'a'),
            ('map512', {'mlp': 768, 'attn': 256}, 'b'),
            ('mix25m512', None, 'c'))
    inert, differ = B.inertness_pairs(plan)
    check('inertness_pairs: same-spec pair is in the INERT list', ('a', 'b') in inert, str(inert))
    check('inertness_pairs: differing-spec pairs are in the DIFFER list',
          {('a', 'c'), ('b', 'c')} == set(differ), str(differ))


def test_inertness_pairs_warns_when_a_side_is_vacuous():
    """S1957 passed pred_d with the inertness half empty -- it had nothing to check and said True."""
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        B.inertness_pairs((('mix25m512', None, 'x'), ('mix25m512', {'mlp': 768, 'attn': 256}, 'y')))
    check('inertness_pairs: warns when one side of the control is empty', 'VACUOUS' in buf.getvalue(),
          repr(buf.getvalue()[:80]))


def test_ref_reads_published_triples():
    """LESSONS 85/87: I typed a per-role triple by hand twice and got an entry wrong both times."""
    p = B.PT + 'ops/coverage_specific_build_results.json'
    if not os.path.exists(p):
        check('ref: artifact present', False, p)
        return
    got = B.ref(p, 'blend_768_384')
    check('ref: returns one value per role', len(got) == len(B.ROLES), str(got))
    with open(p) as fh:
        d = json.load(fh)
    r = d['results'][next(iter(d['results']))]
    want = tuple(r[role]['blend_768_384']['pooled']['overall']['ce_prog'] for role in B.ROLES)
    check('ref: matches the artifact exactly', got == want, f'{got} != {want}')


def test_ref_refuses_to_guess_a_coverage():
    """S1963: ref() silently returned the first coverage of a two-coverage artifact, and the control
    failed by 0.086 nats -- which reads as a data discrepancy and was a helper picking arbitrarily."""
    p = B.PT + 'ops/alpha_reoptimised_results.json'
    if not os.path.exists(p):
        check('ref: two-coverage artifact present', False, p)
        return
    raised = False
    try:
        B.ref(p, 'a25')
    except ValueError:
        raised = True
    check('ref: refuses to guess when an artifact holds two coverages', raised)
    got = B.ref(p, 'a25', coverage='c5419')
    with open(p) as fh:
        d = json.load(fh)
    want = tuple(d['results']['c5419'][role]['a25']['pooled']['overall']['ce_prog'] for role in B.ROLES)
    check('ref: an explicit coverage returns that coverage', got == want, f'{got} != {want}')


def test_paired_t_arithmetic():
    """The instrument S1939 lacked and S1940 used to retract its headline."""
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    b = torch.tensor([0.0, 1.0, 2.0, 3.0])
    p = B.paired_t(a, b)
    check('paired_t: mean of a constant difference', abs(p['mean'] - 1.0) < 1e-12, str(p['mean']))
    check('paired_t: zero variance gives a huge t, not a crash', p['t'] > 1e6, str(p['t']))
    check('paired_t: counts every position and every nonzero', p['n'] == 4 and p['n_nonzero'] == 4)


def test_penalty_accessor():
    """Ctx.penalty is ce_prog - ce_live -- the number S1977-S1983 all published and all re-implemented."""
    res = {'c': {'r': {'a': {'pooled': {'overall': {'ce_prog': 12.5, 'ce_live': 1.5}}}}}}
    x = B.Ctx(res, {}, {}, [('c', '', 0)], ['r'])
    check('penalty: ce_prog - ce_live', abs(x.penalty('c', 'r', 'a') - 11.0) < 1e-9,
          str(x.penalty('c', 'r', 'a')))


def test_composite_arm_grammar():
    """S1983 named the limit: run() applies ONE arm per plan entry, so `mean mlp4 + compiled attn6` --
    the intervention S1980's account actually predicts -- could not be expressed. `A@mlp4+B@attn6` can."""
    check('composite: names its sites', B.composite_sites('meanrow@mlp4+mix30m640@attn6')
          == [('mlp', 4), ('attn', 6)], str(B.composite_sites('meanrow@mlp4+mix30m640@attn6')))
    check('composite: a plain arm is not composite', B.composite_sites('mix30m640') is None)
    check('composite: whole-table if ANY part is', B.is_whole_table('meanrow@mlp4+mix30m640@attn6'))
    check('composite: not whole-table if no part is', not B.is_whole_table('map512@mlp4+nn@attn6'))
    for bad in ('meanrow@mlp99', 'meanrow@frobnicate3', 'meanrow@mlp4+nosite'):
        try:
            B.composite_sites(bad)
            check(f'composite: rejects {bad}', False, 'parsed without error')
        except ValueError:
            check(f'composite: rejects {bad}', True)


def test_whole_table_arms_are_not_fallback_variants():
    """S1983: `meanrow` replaces EVERY row, covered ones included, so it is not a fallback variant and
    the covered-input inertness guarantee does not hold between it and one. Pairing them as same-spec
    made the control assert something false -- and the control caught it."""
    inert, differ = B.inertness_pairs([('mix30m640', None, 'tab'), ('meanrow', None, 'mean')])
    check('whole-table: meanrow is NOT same-spec with a fallback variant',
          ('tab', 'mean') in differ, f'inert={inert} differ={differ}')
    i2, _d2 = B.inertness_pairs([('mix30m640', None, 'a'), ('map512', None, 'b')])
    check('whole-table: two real fallback variants are still same-spec', ('a', 'b') in i2, str(i2))


def test_inert_side_of_the_control_is_still_strict():
    """S1979 relaxed the DIFFERING side (a site-subset change can flip no covered-input argmax). The
    INERT side must stay exact -- S1765/S1936 guarantee it, and it is the half that catches a fallback
    leaking into covered rows."""
    src = open(os.path.join(HERE, 'bqlib.py'), errors='replace').read()
    check('control: same-spec pairs are still asserted EXACTLY inert',
          "chg[c][r][p] == 0 for c in chg for r in chg[c] for p in inert" in src)
    check('control: the differing side asks only that ONE pair moves',
          "'some_differing_pair_moves'" in src)
    check('control: a vacuous side is still a hard FAIL',
          "ctl['control_is_two_sided'] = False" in src)


def test_site_subsets_change_the_cache_key():
    """S1977: an arm may substitute only SOME of the 36 sites. The subset is part of what the rows mean,
    so two arms differing only in it must not collide in the cache -- and must not be treated as
    covered-input-inert relative to each other."""
    prog = object.__new__(B.Program)
    prog.fit_path, prog.ncov = B.FIT_5419, 5419
    prog.digest = 'x' * 32
    check('site subset: an explicit full list normalises to all36',
          B._sites_key(list(B.SITES)) == B._sites_key(None) == 'all36')
    check('site subset: order and duplicates do not matter',
          B._sites_key([('mlp', 1), ('mlp', 0)]) == B._sites_key([('mlp', 0), ('mlp', 1), ('mlp', 0)]))
    all36 = B._key(prog, 'map512', None, 'skip7000', None)
    mlps = B._key(prog, 'map512', None, 'skip7000', [('mlp', i) for i in range(18)])
    check('site subset: a different subset gives a different cache key', all36 != mlps)
    check('site subset: None means all 36 and is stable', all36 == B._key(prog, 'map512', None,
                                                                         'skip7000', None))
    fa = B._fingerprint(prog, 'map512', None, None)
    fm = B._fingerprint(prog, 'map512', None, [('mlp', i) for i in range(18)])
    check('site subset: it is in the fingerprint too', fa != fm)
    inert, differ = B.inertness_pairs([('map512', None, 'a'), ('map512', None, 'b', [('mlp', 0)])])
    check('site subset: arms differing only in the subset are NOT treated as inert',
          ('a', 'b') in differ, f'inert={inert} differ={differ}')


def test_pooled_t_weights_by_evidence():
    """S1971: pooling three roles must weight a half-sized role by its positions, not by a vote. A
    concatenation of two agreeing halves and one dissenting half-sized one must come out agreeing."""
    big_a = torch.cat([torch.full((100,), 1.0), torch.full((100,), 1.0)])
    big_b = torch.zeros(200)
    small_a, small_b = torch.zeros(50), torch.full((50,), 1.0)      # dissents, half the size
    pooled = B.paired_t(torch.cat([big_a, small_a]), torch.cat([big_b, small_b]))
    check('pooled_t: n is the total across roles', pooled['n'] == 250, str(pooled['n']))
    check('pooled_t: the majority-by-evidence wins the sign', pooled['mean'] > 0, str(pooled['mean']))
    per_role = B.paired_t(small_a, small_b)
    check('pooled_t: and it disagrees with the dissenting role alone', per_role['mean'] < 0)


def test_cost_matches_the_published_closed_form():
    """S1754's accounting, against numbers already published: nn is 224.868M and map512 267.245M
    at 5,419 coverage."""
    prog = object.__new__(B.Program)
    prog.ncov = 5419
    prog.unc = torch.zeros(50257 - 5419, dtype=torch.long)
    full_table = 36 * (5419 * B.D + B.D)
    check('cost: full tables match 36*(NCOV*D + D)',
          abs(prog.cost('map0', None) - full_table) < 1, str(prog.cost('map0', None)))
    got = prog.cost('map512', None) / 1e6
    check('cost: map512 at 5,419 is the published 267.245M', abs(got - 267.245) < 0.002, f'{got:.3f}M')
    got_nn = prog.cost('nn', None) / 1e6
    check('cost: nn at 5,419 is the published 224.868M', abs(got_nn - 224.868) < 0.002, f'{got_nn:.3f}M')
    got_r = prog.cost('map64', {'mlp': 768, 'attn': 256}) / 1e6
    check('cost: a per-site dict is summed over the ACTUAL per-site ranks', got_r < full_table / 1e6,
          f'{got_r:.3f}M')


def test_arm_names_parse_the_way_the_grammar_says():
    """nn<P> must stay byte-identical to nn<P>m64 or every cached key describes different rows."""
    prog = object.__new__(B.Program)
    prog.ncov = 5419
    prog.unc = torch.zeros(50257 - 5419, dtype=torch.long)
    check('arm grammar: nn75 and nn75m64 cost the same (same build)',
          prog.cost('nn75', None) == prog.cost('nn75m64', None))
    check('arm grammar: mix<A>m<R> prices on R, not A',
          prog.cost('mix10m512', None) == prog.cost('mix40m512', None))
    check('arm grammar: msk<P>m<R> prices on R too',
          prog.cost('msk10m512', None) == prog.cost('msk50m512', None))


# ---------------------------------------------------------------- the gate itself

GATE_FIXTURES = [
    ('except name used outside its handler is a NameError', False, '''
import json
def f():
    try:
        x = 1
    except Exception as e:
        print(e)
    return str(e)
def main():
    json.dump({'pred_a_x': True, 'pred_b_y': True, 'pred_c_z': True}, open('/dev/null', 'w'))
main()
'''),
    ('except name used INSIDE its handler is fine', True, '''
import json
def f():
    try:
        x = 1
    except Exception as e:
        return str(e)
    return x
def main():
    json.dump({'pred_a_x': True, 'pred_b_y': True, 'pred_c_z': True}, open('/dev/null', 'w'))
main()
'''),
    ('module-level name used but never bound', False, '''
import json
def main():
    json.dump({'pred_a_x': True, 'pred_b_y': True, 'pred_c_z': True, 'v': MISSING_NAME},
              open('/dev/null', 'w'))
main()
'''),
    ('a module constant assigned twice', False, '''
import json
K = (1.0, 2.0)
K = (3.0, 4.0)
def main():
    json.dump({'pred_a_x': True, 'pred_b_y': True, 'pred_c_z': K[0]}, open('/dev/null', 'w'))
main()
'''),
    ('an experiment with fewer than three predicates', False, '''
import json
def main():
    json.dump({'pred_a_x': True, 'pred_b_y': True}, open('/dev/null', 'w'))
main()
'''),
    ('a function referring to a module-level class', True, '''
import json
class Thing:
    pass
def make():
    return Thing()
def main():
    json.dump({'pred_a_x': True, 'pred_b_y': True, 'pred_c_z': bool(make())}, open('/dev/null', 'w'))
main()
'''),
    ('a B.run() declaration with real registered text', True, '''
import json
PREDS = [('pred_a_thing', 'the thing happens on at least two roles', lambda x: True),
         ('pred_b_other', 'and the other thing does not', lambda x: True),
         ('pred_c_third', 'and the third thing is bounded', lambda x: True)]
def main():
    json.dump({k: v(None) for k, _t, v in PREDS}, open('/dev/null', 'w'))
main()
'''),
    ('a B.run() predicate whose registered text is missing', False, '''
import json
PREDS = [('pred_a_thing', lambda x: True),
         ('pred_b_other', 'and the other thing does not', lambda x: True),
         ('pred_c_third', 'and the third thing is bounded', lambda x: True)]
def main():
    json.dump({k: True for k in ('pred_a_thing', 'pred_b_other', 'pred_c_third')},
              open('/dev/null', 'w'))
main()
'''),
    ('a clean minimal experiment', True, '''
import json
def main():
    json.dump({'pred_a_x': True, 'pred_b_y': True, 'pred_c_z': True, 'pred_d_controls': True},
              open('/dev/null', 'w'))
main()
'''),
]


def test_every_ref_path_exists():
    """Guards against exactly what Logan asked about: moving or deleting an artifact silently breaks a
    reproduction control in a script that still looks fine. 237 of 239 result JSONs are referenced by a
    .py or the registry, so nothing here is safely movable without this check."""
    import glob
    import re
    missing = []
    for f in glob.glob(os.path.join(HERE, '*.py')):
        src = open(f, errors='replace').read()
        # only READS -- B.ref(B.PT + '...'). The first version also matched a script's own
        # `OUT = B.PT + '...'` write path and flagged every experiment whose run had not produced its
        # artifact yet, which is not a broken reference.
        for m in re.finditer(r"B\.ref\(\s*B\.PT \+ '(ops/[A-Za-z0-9_]+_results\.json)'", src):
            if not os.path.exists(B.PT + m.group(1)):
                missing.append((os.path.basename(f), m.group(1)))
    check(f'ref paths: every B.ref() artifact referenced in ops/ still exists', not missing,
          str(missing[:3]))


def test_no_build_level_comparison_is_vote_dependent():
    """S1974's mechanism. S1965 published a claim whose per-role vote passed while the pooled evidence
    said the opposite; the audit re-reads every artifact and exits non-zero if any BUILD-level
    comparison has that shape. Wired here so it gates enqueue instead of relying on anyone re-running
    the audit."""
    out = subprocess.run([sys.executable, os.path.join(HERE, 'pooling_audit.py')],
                         capture_output=True, text=True, timeout=120)
    check('pooling audit: no build-level comparison depends on the vote', out.returncode == 0,
          out.stdout.strip()[-200:])


def test_gate_fixtures():
    """The gate protects every run; these are the shapes that have actually reached the GPU."""
    for name, want_pass, src in GATE_FIXTURES:
        with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False, dir='/tmp') as fh:
            fh.write(src.lstrip())
            path = fh.name
        try:
            out = subprocess.run([sys.executable, os.path.join(HERE, 'gate.py'), path],
                                 capture_output=True, text=True, timeout=60)
            passed = out.stdout.strip().endswith('GATE: PASS')
            check(f'gate: {name} -> {"PASS" if want_pass else "FAIL"}', passed == want_pass,
                  out.stdout.strip().splitlines()[-1] if out.stdout.strip() else out.stderr[:120])
        finally:
            os.unlink(path)


def test_gate_accepts_the_library_itself():
    """bqlib carries the LIBRARY marker and must keep passing; a marker on something that writes
    results must NOT."""
    out = subprocess.run([sys.executable, os.path.join(HERE, 'gate.py'),
                          os.path.join(HERE, 'bqlib.py')], capture_output=True, text=True, timeout=60)
    check('gate: bqlib.py passes as a LIBRARY', out.stdout.strip().endswith('GATE: PASS'),
          out.stdout.strip()[-120:])


for fn in (test_penalty_accessor, test_composite_arm_grammar, test_whole_table_arms_are_not_fallback_variants, test_inert_side_of_the_control_is_still_strict, test_site_subsets_change_the_cache_key, test_no_build_level_comparison_is_vote_dependent, test_pooled_t_weights_by_evidence, test_every_ref_path_exists, test_ref_refuses_to_guess_a_coverage, test_rk_key_is_order_independent, test_inertness_pairs_splits_by_table_rank,
           test_inertness_pairs_warns_when_a_side_is_vacuous, test_ref_reads_published_triples,
           test_paired_t_arithmetic, test_cost_matches_the_published_closed_form,
           test_arm_names_parse_the_way_the_grammar_says, test_gate_fixtures,
           test_gate_accepts_the_library_itself):
    try:
        fn()
    except Exception as exc:                                  # a crashing test is a failing test
        why = f'{type(exc).__name__}: {exc}'
        print(f'  FAIL  {fn.__name__} CRASHED   {why}')
        FAILS.append(fn.__name__)

EL = time.time() - T0
print(f'\n{"FAILED" if FAILS else "PASSED"}  {len(FAILS)} failing  ({EL:.2f}s)')
if FAILS:
    print('  ' + '\n  '.join(FAILS))
sys.exit(1 if FAILS else 0)
