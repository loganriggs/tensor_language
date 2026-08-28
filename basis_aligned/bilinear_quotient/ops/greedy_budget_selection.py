# GREEDY BUDGET SELECTION -- does the compiler's actual decision procedure beat a ranking?
#
# §1739/§1740 allocated a budget by taking the top K sites of a RANKING of individual program-context
# scores. But §1736 established that these sites interact strongly -- the sum of 18 one-at-a-time MLP
# removals is 2.36x the joint, attention 0.40x -- so a ranking of individual scores is exactly the
# object that cannot see interactions. The procedure a compiler would actually run is GREEDY: at each
# step add the site that buys the most GIVEN what is already native.
#
# Greedy can also OVERFIT the rows it selects on, and a ranking cannot in the same way, so a fair
# comparison needs a transfer test. Selection runs on skip7000 alone; skip11000 is never used to
# choose anything and measures whether the gain survives.
#
# ROLES. Both large roles remain spent for CERTIFICATION -- this stays DISCOVERY ONLY, same family as
# §1736-§1740. The selection/transfer split here guards against selection overfit, which is a
# different failure from role burn and needs its own guard regardless.
#
# Registered predictions, TWO-SIDED per LESSONS 31:
#   pred_a GREEDY BEATS THE RANKING ON ITS OWN ROLE: greedy recovers more than the individual-score
#          top-6 on skip7000. If FALSE the individual scores already capture whatever interaction
#          matters at this budget, and §1739's ranking is not just usable but near-optimal.
#   pred_b THE GAIN TRANSFERS: greedy also beats the top-6 on skip11000, which chose nothing. If
#          FALSE, greedy's advantage is selection overfit -- it found six sites good for 192
#          particular rows -- and the ranking is the better recommendation despite losing on its own
#          selection role. This is the prediction that decides which procedure to recommend.
#   pred_c THE SETS ACTUALLY DIFFER by at least two sites. If they barely differ, pred_a and pred_b
#          are measuring noise between near-identical allocations and neither is informative.
#   pred_d CONTROLS: the all-36-tabled CE reproduces §1738-§1740's 7.35114 within 0.005, fit coverage
#          is 5419 of 50257, and the top-6 column reproduces §1739's 1.0165 and 1.0383 within 0.001 --
#          a same-quantity, different-script check.
import json, time, sys, os, random, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256; BUDGET = 6; SEED = 1740
SELECT_ROLE = 'skip7000'
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/greedy_budget_selection_results.json'
PROG_JSON = PT + 'ops/importance_in_program_context_results.json'
OAT_JSON = PT + 'ops/loo_vs_oat_importance_results.json'
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711)]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
S1738_PROGRAM_CE = 7.35114
S1739_TOP6 = {'skip7000': 1.0165, 'skip11000': 1.0383}
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
    pj = json.load(open(PROG_JSON))['results']
    prog_rank = sorted(names, key=lambda n: -min(pj['skip7000']['sites'][n]['prog'],
                                                 pj['skip11000']['sites'][n]['prog']))
    top6 = prog_rank[:BUDGET]
    print(f'GREEDY BUDGET SELECTION | {BUDGET} of 36 sites stay NATIVE | selection on '
          f'{SELECT_ROLE}, TRANSFER measured on the other role | DISCOVERY ONLY', flush=True)
    print(f'  PROG top-{BUDGET} (§1739, ranked by individual score): {top6}', flush=True)

    COV['seen'] = torch.zeros(50257, dtype=torch.bool, device=DEV)
    tables, seen = fit_tables(fit, sites)
    COV['seen'] = seen
    ncov = int(seen.sum())
    print(f'  fit coverage {ncov} of 50257 token ids', flush=True)

    ev = {}
    for ename, epath, ce_ref in EVAL_SETS:
        e = load(epath)
        ev[ename] = e

    def run(rows, keep):
        ks = {name_to_site(n) for n in keep}
        return ce(rows, [mod_of(*st).register_forward_hook(table_hook(tables[st], seen))
                         for st in sites if st not in ks])

    base = {}
    for ename, _, ce_ref in EVAL_SETS:
        cl = ce(ev[ename])
        assert abs(cl - ce_ref) <= 1e-2, f'{ename} live CE {cl:.5f} != {ce_ref}'
        base[ename] = {'live': cl, 'full': run(ev[ename], [])}
        print(f'  {ename}: live CE {cl:.5f} | all-36-tabled CE {base[ename]["full"]:.5f} '
              f'(stake {base[ename]["full"] - cl:.4f})', flush=True)

    # GREEDY on the selection role only: at each step add the site that buys the most GIVEN what is
    # already native. This is the compiler's actual decision procedure, and it can see the
    # interactions that a ranking of individual scores cannot (§1736).
    sel, trace, evals = [], [], 0
    full_s = base[SELECT_ROLE]['full']
    for step in range(BUDGET):
        best, bestv = None, None
        for n in names:
            if n in sel:
                continue
            v = full_s - run(ev[SELECT_ROLE], sel + [n])
            evals += 1
            if bestv is None or v > bestv:
                best, bestv = n, v
        sel.append(best)
        trace.append({'step': step + 1, 'added': best, 'recovered': round(bestv, 5)})
        print(f'    greedy step {step + 1}: + {best:7s} -> {bestv:7.4f} nats', flush=True)
    print(f'  greedy set: {sel}   ({evals} candidate evaluations)', flush=True)

    rng = random.Random(SEED)
    rand = [sorted(rng.sample(names, BUDGET)) for _ in range(6)]
    out = {}
    for ename, _, _ in EVAL_SETS:
        f = base[ename]['full']; st = f - base[ename]['live']
        g = f - run(ev[ename], sel)
        p = f - run(ev[ename], top6)
        rv = sorted(f - run(ev[ename], r) for r in rand)
        out[ename] = {'live_ce': round(base[ename]['live'], 5), 'all_tabled_ce': round(f, 5),
                      'stake_nats': round(st, 5), 'greedy': round(g, 5), 'prog_top6': round(p, 5),
                      'random_best': round(rv[-1], 5), 'random_median': round(0.5 * (rv[2] + rv[3]), 5),
                      'greedy_frac': round(g / st, 5), 'prog_top6_frac': round(p / st, 5),
                      'role': 'SELECTION' if ename == SELECT_ROLE else 'TRANSFER'}
        r = out[ename]
        print(f'\n  {ename} [{r["role"]}]: greedy {g:.4f} ({r["greedy_frac"]:.2%})  '
              f'PROG top-{BUDGET} {p:.4f} ({r["prog_top6_frac"]:.2%})  '
              f'random best {rv[-1]:.4f}  median {r["random_median"]:.4f}', flush=True)

    other = [e for e, _, _ in EVAL_SETS if e != SELECT_ROLE][0]
    pa = out[SELECT_ROLE]['greedy'] > out[SELECT_ROLE]['prog_top6']
    pb = out[other]['greedy'] > out[other]['prog_top6']
    pc = len(set(sel) - set(top6)) >= 2
    pd = (abs(out['skip7000']['all_tabled_ce'] - S1738_PROGRAM_CE) <= 0.005 and ncov == 5419
          and all(abs(out[e]['prog_top6'] - v) <= 0.001 for e, v in S1739_TOP6.items()))
    print(f'\n  greedy beats the individual-score top-{BUDGET} on the SELECTION role -> {pa}',
          flush=True)
    print(f'  and on the TRANSFER role {other} -> the gain is not selection overfit {pb}', flush=True)
    print(f'  greedy set differs from top-{BUDGET} by {len(set(sel) - set(top6))} sites -> '
          f'interactions matter {pc}', flush=True)
    print(f'  all-tabled CE + §1739 top-6 reproduction + coverage {ncov} -> control {pd}', flush=True)

    r = {'config': {'budget_native_sites': BUDGET, 'select_role': SELECT_ROLE,
                    'transfer_role': other, 'seed': SEED,
                    'greedy': 'at each step add the site that buys the most GIVEN what is already '
                              'native. Selection uses ONE role; the other role is never used to '
                              'choose anything and measures transfer.',
                    'ROLE_NOTE': 'DISCOVERY ONLY. Same family as §1736-§1740; both large roles are '
                                 'spent for certification purposes. The selection/transfer split '
                                 'here guards against selection overfit, not against role burn.'},
         'greedy_set': sel, 'greedy_trace': trace, 'prog_top6': top6, 'random_sets': rand,
         'results': out,
         'predictions': {'pred_a_greedy_beats_topk_on_selection': bool(pa),
                         'pred_b_gain_transfers': bool(pb),
                         'pred_c_sets_differ': bool(pc),
                         'pred_d_controls': bool(pd)},
         'runtime_s': round(time.time() - t0, 1)}
    json.dump(r, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
