# WHERE SHOULD A COMPILER SPEND ITS BUDGET? -- the decision §1736-§1738 was actually about.
#
# §1738 found that in program context the one-at-a-time (OAT) importance ranking is near-INVERTED:
# Spearman -0.66 and -0.69. The front MLPs top the OAT column because ablating them wrecks everything
# downstream, but they are the most tabular sites in the model (§1662: mlp0 90%, mlp1 96%) and add
# essentially nothing over their own tables. The sites a program has to work for are late.
#
# That is a claim about where to SPEND, and it has a direct test that uses the arc's own metric
# rather than another correlation. Fix a budget of K sites that stay NATIVE; table the other 36-K.
# Then compare allocations of that budget at matched size:
#
#   ALLOC-PROG    the K sites with the highest program-context importance (§1738)
#   ALLOC-OAT     the K sites with the highest one-at-a-time removal (§1737) -- what an importance
#                 column of the kind this arc has been quoting would tell a compiler to keep
#   ALLOC-RANDOM  8 random matched-size draws, reported as median and best, so "better than OAT" can
#                 be distinguished from "better than arbitrary"
#
# Both ranked lists are READ FROM THE PRIOR RUNS' JSON, never hand-copied, so a transcription slip
# cannot decide the comparison.
#
# ROLES. Same family as §1736-§1738; both large roles are spent. DISCOVERY ONLY -- this confirms
# nothing and certifies nothing. It is the decision-relevant measurement, and it will need a clean
# role before any of it is registered.
#
# Registered predictions, TWO-SIDED per LESSONS 31:
#   pred_a PROG BEATS OAT AT MATCHED BUDGET: the PROG allocation recovers more nats than the OAT
#          allocation on BOTH roles. If FALSE, §1738's inversion does not translate into a better
#          spend and the ranking result is interesting but inert.
#   pred_b PROG BEATS ARBITRARY: the PROG allocation beats the BEST of 8 random matched-size
#          allocations on both roles. Beating the median is a low bar when the pool is 36 sites;
#          beating the best of eight is the bar that means the ranking carries real information.
#   pred_c THE STRONG FORM OF THE INVERSION: the OAT allocation is WORSE than the MEDIAN random
#          allocation on both roles. If FALSE, an OAT importance column is merely suboptimal rather
#          than actively misleading, which is a materially weaker claim than §1738 invites and
#          should be recorded as the weaker one.
#   pred_d CONTROLS: the all-36-tabled program CE reproduces §1738's 7.35114 within 0.005, the live
#          CE reproduces 3.29205 within 1e-3, and fit coverage is 5419 of 50257 token ids.
import json, time, sys, os, random, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256; BUDGET = 6; NRAND = 8; SEED = 1738
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/program_budget_allocation_results.json'
PROG_JSON = PT + 'ops/importance_in_program_context_results.json'
OAT_JSON = PT + 'ops/loo_vs_oat_importance_results.json'
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711)]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
S1738_PROGRAM_CE = 7.35114
COV = {}
STATE = {}


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


def table_hook(tbl, seen):
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        sub = tbl[STATE['idx'].reshape(-1)].reshape(y.shape).to(y.dtype)
        sub = torch.where(seen[STATE['idx']].unsqueeze(-1), sub, y)
        return (sub,) + tuple(out[1:]) if isinstance(out, tuple) else sub
    return hook


def mod_of(kind, L):
    return H[L].mlp if kind == 'mlp' else H[L].attn


def name_to_site(n):
    return ('mlp', int(n[3:])) if n.startswith('mlp') else ('attn', int(n[4:]))


@torch.no_grad()
def sweep(rows, hooks=(), score=None):
    hs = list(hooks)
    try:
        for i in range(0, rows.shape[0], 8):
            bb = rows[i:i + 8]
            idx = bb[:, :-1].to(DEV).contiguous()
            STATE['idx'] = idx
            x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
            for blk in H:
                x, v1 = blk(x, v1, x0)
            if score is not None:
                lg = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
                score(lg, bb[:, 1:].to(DEV), idx)
    finally:
        for h in hs:
            h.remove()


@torch.no_grad()
def ce(rows, hooks=()):
    acc = {'t': 0.0, 'n': 0}

    def score(lg, tg, idx):
        e = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                            reduction='none').reshape(tg.shape)[:, 64:].double()
        cov = COV['seen'][idx[:, 64:]]
        acc['t'] += float(e[cov].sum()); acc['n'] += int(cov.sum())
    sweep(rows, hooks=hooks, score=score)
    return acc['t'] / acc['n']


@torch.no_grad()
def fit_tables(rows, sites):
    s = {st: torch.zeros(50257, D, device=DEV) for st in sites}
    c = torch.zeros(50257, device=DEV)
    fired = {'n': 0}

    def mk(st, first):
        def hook(mod, args, out):
            y = (out[0] if isinstance(out, tuple) else out).float().reshape(-1, D)
            t = STATE['idx'].reshape(-1)
            s[st].index_add_(0, t, y)
            if first:
                c.index_add_(0, t, torch.ones_like(t, dtype=torch.float32))
                fired['n'] += 1
            return None
        return hook
    hooks = [mod_of(*st).register_forward_hook(mk(st, j == 0)) for j, st in enumerate(sites)]
    sweep(rows, hooks=hooks)
    assert fired['n'] > 0, 'table fit never fired'
    seen = c > 0
    out = {}
    for st in sites:
        mean = s[st].sum(0) / c.sum()
        tbl = mean.unsqueeze(0).repeat(50257, 1)
        tbl[seen] = s[st][seen] / c[seen].unsqueeze(1)
        out[st] = tbl
    return out, seen


@torch.no_grad()
def main():
    t0 = time.time()
    fit = load(FIT_ROWS)
    sites = [(k, L) for k in ('mlp', 'attn') for L in range(18)]
    names = [f'{k}{L}' for k, L in sites]

    # both rankings read from the prior runs, never hand-copied
    pj = json.load(open(PROG_JSON))['results']
    oj = json.load(open(OAT_JSON))['results']
    prog_rank = sorted(names, key=lambda n: -min(pj['skip7000']['sites'][n]['prog'],
                                                 pj['skip11000']['sites'][n]['prog']))
    oat_rank = sorted(names, key=lambda n: -min(oj['skip7000']['sites'][n]['oat'],
                                                oj['skip11000']['sites'][n]['oat']))
    alloc = {'PROG': prog_rank[:BUDGET], 'OAT': oat_rank[:BUDGET]}
    rng = random.Random(SEED)
    rand = [sorted(rng.sample(names, BUDGET)) for _ in range(NRAND)]
    assert len(set(map(tuple, rand))) == NRAND, 'random allocations collided'
    print(f'PROGRAM BUDGET ALLOCATION | {BUDGET} of 36 sites stay NATIVE, the rest are tabled | '
          f'DISCOVERY ONLY, certifies nothing', flush=True)
    print(f'  ALLOC-PROG {alloc["PROG"]}', flush=True)
    print(f'  ALLOC-OAT  {alloc["OAT"]}', flush=True)

    COV['seen'] = torch.zeros(50257, dtype=torch.bool, device=DEV)
    tables, seen = fit_tables(fit, sites)
    COV['seen'] = seen
    ncov = int(seen.sum())
    print(f'  fit coverage {ncov} of 50257 token ids', flush=True)

    out = {}
    for ename, epath, ce_ref in EVAL_SETS:
        ev = load(epath)
        cl = ce(ev)
        assert abs(cl - ce_ref) <= 1e-2, f'{ename} live CE {cl:.5f} != {ce_ref}'

        def run(keep):
            ks = {name_to_site(n) for n in keep}
            return ce(ev, [mod_of(*st).register_forward_hook(table_hook(tables[st], seen))
                           for st in sites if st not in ks])
        full = run([])
        stake = full - cl
        res = {k: round(full - run(v), 5) for k, v in alloc.items()}
        rv = [full - run(v) for v in rand]
        srv = sorted(rv)
        res['RANDOM_median'] = round(0.5 * (srv[3] + srv[4]), 5)
        res['RANDOM_best'] = round(srv[-1], 5)
        res['RANDOM_worst'] = round(srv[0], 5)
        print(f'\n  {ename}: live CE {cl:.5f} | all-36-tabled CE {full:.5f} (stake {stake:.4f} nats)',
              flush=True)
        for k in ('PROG', 'OAT', 'RANDOM_best', 'RANDOM_median', 'RANDOM_worst'):
            print(f'    {k:14s} recovers {res[k]:7.4f} nats  ({res[k] / stake:6.2%} of the stake)',
                  flush=True)
        out[ename] = {'live_ce': round(cl, 5), 'all_tabled_ce': round(full, 5),
                      'stake_nats': round(stake, 5), 'recovered': res,
                      'random_draws': [round(x, 5) for x in rv]}
        del ev
        torch.cuda.empty_cache()

    pa = all(out[e]['recovered']['PROG'] > out[e]['recovered']['OAT'] for e in out)
    pb = all(out[e]['recovered']['PROG'] > out[e]['recovered']['RANDOM_best'] for e in out)
    pc = all(out[e]['recovered']['OAT'] < out[e]['recovered']['RANDOM_median'] for e in out)
    pd = (abs(out['skip7000']['all_tabled_ce'] - S1738_PROGRAM_CE) <= 0.005
          and abs(out['skip7000']['live_ce'] - 3.29205) <= 1e-3 and ncov == 5419)

    print(f'\n  PROG beats OAT at matched budget on both roles -> {pa}', flush=True)
    print(f'  PROG beats the BEST of {NRAND} random allocations -> {pb}', flush=True)
    print(f'  OAT is worse than the MEDIAN random allocation -> actively misleading {pc}', flush=True)
    print(f'  all-tabled CE {out["skip7000"]["all_tabled_ce"]:.5f} vs §1738 {S1738_PROGRAM_CE} + '
          f'coverage {ncov} -> control {pd}', flush=True)

    r = {'config': {'budget_native_sites': BUDGET, 'random_draws': NRAND, 'seed': SEED,
                    'setup': f'{BUDGET} of 36 sites stay NATIVE, the other {36 - BUDGET} are replaced '
                             'by per-token tables (hybrid hook, §1661). Recovered = CE(all tabled) '
                             'minus CE(this allocation).',
                    'rankings': 'read from the prior runs JSON, never hand-copied; each site ranked '
                                'by the WORSE of its two roles so a single-role fluke cannot enter',
                    'ROLE_NOTE': 'DISCOVERY ONLY. Same family as §1736-§1738; both large roles spent.'},
         'allocations': {**alloc, 'RANDOM': rand}, 'results': out,
         'predictions': {'pred_a_prog_beats_oat': bool(pa),
                         'pred_b_prog_beats_best_random': bool(pb),
                         'pred_c_oat_worse_than_median_random': bool(pc),
                         'pred_d_controls': bool(pd)},
         'runtime_s': round(time.time() - t0, 1)}
    json.dump(r, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
