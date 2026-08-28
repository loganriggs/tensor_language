# PROGRAM BUDGET CURVE -- how far does the program-context ranking's advantage carry, and is there
# a knee?
#
# §1739 measured one budget, K=6 native sites of 36: the program-context ranking recovered 25.04% and
# 24.37% of the table-program stake against the one-at-a-time ranking's 12.79% and 12.22%, which is
# the random median (12.48%, 12.10%). One point is not a curve. Two things it cannot say:
#
#   - whether the advantage is a small-K artifact that closes as the budget grows. At K=36 both
#     allocations ARE the whole model and recover 100% by construction, so it must close eventually;
#     the question is how fast over the range a compiler would actually use.
#   - whether there is a KNEE: a budget after which each extra native site buys much less, which is
#     what a compiler plans around, and what §1718's marginal-cost sweep looked for on the other axis
#     and did not find.
#
# Budgets 2, 3, 6, 9, 12. Both ranked lists read from the prior runs' JSON, never hand-copied, each
# site ranked by the WORSE of its two roles so a single-role fluke cannot enter an allocation.
#
# ROLES. Same family as §1736-§1739; both large roles are spent. DISCOVERY ONLY -- confirms nothing,
# certifies nothing, and the frozen lists stay frozen for whichever clean role appears next.
#
# Registered predictions, TWO-SIDED per LESSONS 31:
#   pred_a PROG BEATS OAT AT EVERY BUDGET on both roles. If it fails at some K, §1739's single point
#          was budget-specific and the ranking claim needs a budget attached whenever it is stated.
#   pred_b THE ADVANTAGE CONVERGES: the PROG/OAT recovery ratio is non-increasing across the five
#          budgets on both roles. It must reach 1.0 at K=36; if the ratio instead GROWS with K, the
#          two rankings are diverging over exactly the range that matters and the convergence
#          intuition is wrong.
#   pred_c NO KNEE: PROG's marginal recovery per added native site is strictly decreasing across the
#          sweep. A knee -- a budget where the marginal jumps back up -- would name a natural stopping
#          point for a compiler and is the more useful outcome, which is why the bar is set so that
#          finding one FAILS this prediction.
#   pred_d CONTROLS: the all-36-tabled CE reproduces §1738/§1739's 7.35114 within 0.005, live CE
#          reproduces 3.29205, fit coverage is 5419 of 50257, and the K=6 column reproduces §1739's
#          PROG 1.0165 / 1.0383 and OAT 0.5191 / 0.5208 within 0.001 -- a same-quantity,
#          different-script check, the kind §1733 said this arc was short of.
import json, time, sys, os, random, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256; BUDGETS = (2, 3, 6, 9, 12); NRAND = 6; SEED = 1739
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/program_budget_curve_results.json'
PROG_JSON = PT + 'ops/importance_in_program_context_results.json'
OAT_JSON = PT + 'ops/loo_vs_oat_importance_results.json'
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711)]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
S1738_PROGRAM_CE = 7.35114
S1739_K6 = {'skip7000': {'PROG': 1.0165, 'OAT': 0.5191},
            'skip11000': {'PROG': 1.0383, 'OAT': 0.5208}}
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
    rng = random.Random(SEED)
    alloc = {K: {'PROG': prog_rank[:K], 'OAT': oat_rank[:K]} for K in BUDGETS}
    rand = {K: [sorted(rng.sample(names, K)) for _ in range(NRAND)] for K in BUDGETS}
    for K in BUDGETS:
        assert len(set(map(tuple, rand[K]))) == NRAND, f'random allocations collided at K={K}'
    print(f'PROGRAM BUDGET CURVE | budgets {BUDGETS} of 36 native sites | DISCOVERY ONLY, '
          f'certifies nothing', flush=True)

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
        print(f'\n  {ename}: live CE {cl:.5f} | all-36-tabled CE {full:.5f} '
              f'(stake {stake:.4f} nats)', flush=True)
        print(f'    {"K":>3s} {"PROG":>9s} {"OAT":>9s} {"rand med":>9s} {"rand best":>9s}   '
              f'PROG/OAT   PROG marginal/site', flush=True)
        per_k = {}
        prev, prevK = 0.0, 0
        for K in BUDGETS:
            g = {k: full - run(v) for k, v in alloc[K].items()}
            rv = sorted(full - run(v) for v in rand[K])
            med = 0.5 * (rv[NRAND // 2 - 1] + rv[NRAND // 2])
            marg = (g['PROG'] - prev) / (K - prevK)
            prev, prevK = g['PROG'], K
            per_k[K] = {'PROG': round(g['PROG'], 5), 'OAT': round(g['OAT'], 5),
                        'RANDOM_median': round(med, 5), 'RANDOM_best': round(rv[-1], 5),
                        'PROG_frac': round(g['PROG'] / stake, 5),
                        'OAT_frac': round(g['OAT'] / stake, 5),
                        'prog_over_oat': round(g['PROG'] / max(g['OAT'], 1e-9), 4),
                        'prog_marginal_per_site': round(marg, 5)}
            r = per_k[K]
            print(f'    {K:3d} {r["PROG"]:9.4f} {r["OAT"]:9.4f} {r["RANDOM_median"]:9.4f} '
                  f'{r["RANDOM_best"]:9.4f}   {r["prog_over_oat"]:6.2f}x   {marg:8.5f}', flush=True)
        out[ename] = {'live_ce': round(cl, 5), 'all_tabled_ce': round(full, 5),
                      'stake_nats': round(stake, 5), 'by_budget': per_k}
        del ev
        torch.cuda.empty_cache()

    pa = all(out[e]['by_budget'][K]['PROG'] > out[e]['by_budget'][K]['OAT']
             for e in out for K in BUDGETS)
    ratios = {e: [out[e]['by_budget'][K]['prog_over_oat'] for K in BUDGETS] for e in out}
    pb = all(all(r[i] >= r[i + 1] - 1e-9 for i in range(len(r) - 1)) for r in ratios.values())
    marg = {e: [out[e]['by_budget'][K]['prog_marginal_per_site'] for K in BUDGETS] for e in out}
    pc = all(all(mm[i] > mm[i + 1] for i in range(len(mm) - 1)) for mm in marg.values())
    pd = (abs(out['skip7000']['all_tabled_ce'] - S1738_PROGRAM_CE) <= 0.005
          and abs(out['skip7000']['live_ce'] - 3.29205) <= 1e-3 and ncov == 5419
          and all(abs(out[e]['by_budget'][6][k] - v) <= 0.001
                  for e, kv in S1739_K6.items() for k, v in kv.items()))

    print(f'\n  PROG beats OAT at EVERY budget on both roles -> {pa}', flush=True)
    print(f'  PROG/OAT ratio non-increasing in K {ratios} -> advantage converges {pb}', flush=True)
    print(f'  PROG marginal per added site strictly decreasing {marg} -> no knee {pc}', flush=True)
    print(f'  all-tabled CE + §1739 K=6 reproduction + coverage {ncov} -> control {pd}', flush=True)

    r = {'config': {'budgets': list(BUDGETS), 'random_draws': NRAND, 'seed': SEED,
                    'setup': 'K of 36 sites stay NATIVE, the other 36-K are replaced by per-token '
                             'tables (hybrid hook, §1661). Recovered = CE(all tabled) minus '
                             'CE(this allocation).',
                    'rankings': 'read from the prior runs JSON, never hand-copied; each site ranked '
                                'by the WORSE of its two roles so a single-role fluke cannot enter',
                    'ROLE_NOTE': 'DISCOVERY ONLY. Same family as §1736-§1738; both large roles spent.'},
         'allocations': {str(K): alloc[K] for K in BUDGETS},
         'random_allocations': {str(K): rand[K] for K in BUDGETS}, 'results': out,
         'predictions': {'pred_a_prog_beats_oat_at_every_budget': bool(pa),
                         'pred_b_advantage_converges': bool(pb),
                         'pred_c_no_knee_in_marginal': bool(pc),
                         'pred_d_controls': bool(pd)},
         'runtime_s': round(time.time() - t0, 1)}
    json.dump(r, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
