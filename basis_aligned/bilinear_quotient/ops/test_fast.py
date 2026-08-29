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


def test_paired_t_arithmetic():
    """The instrument S1939 lacked and S1940 used to retract its headline."""
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    b = torch.tensor([0.0, 1.0, 2.0, 3.0])
    p = B.paired_t(a, b)
    check('paired_t: mean of a constant difference', abs(p['mean'] - 1.0) < 1e-12, str(p['mean']))
    check('paired_t: zero variance gives a huge t, not a crash', p['t'] > 1e6, str(p['t']))
    check('paired_t: counts every position and every nonzero', p['n'] == 4 and p['n_nonzero'] == 4)


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
    ('a clean minimal experiment', True, '''
import json
def main():
    json.dump({'pred_a_x': True, 'pred_b_y': True, 'pred_c_z': True, 'pred_d_controls': True},
              open('/dev/null', 'w'))
main()
'''),
]


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


for fn in (test_rk_key_is_order_independent, test_inertness_pairs_splits_by_table_rank,
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
