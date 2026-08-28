# ALLOCATION LOCAL SEARCH -- how far from good is greedy, now that we know it has no guarantee?
#
# S1742 proved, from the greedy trace alone, that this objective is NOT submodular: the marginal gain
# rises four times, and at step 9 `attn7` delivered 0.0723 after having been worth at most 0.0713 at
# step 8, because `attn9` had become native in between. The standard `1 - 1/e` greedy bound requires
# submodularity, so greedy here carries no approximation guarantee and S1739-S1742's frontier is a
# lower bound rather than the best allocation at each budget. That was stated as a caveat. This
# measures it.
#
# Steepest-improvement SWAP search at K=6: try every (native site out, tabled site in) pair, take the
# best strict improvement, repeat. Two starts -- the greedy set and a random set -- so that "greedy is
# locally optimal" and "the landscape has a single basin" are separate answerable questions instead of
# one conflated one.
#
# ROLES. Selection on skip7000; skip11000 chooses nothing and carries every comparison, because a
# swap search has strictly more chances to overfit its selection rows than greedy did. Both large
# roles remain spent for CERTIFICATION -- DISCOVERY ONLY, same family as S1736-S1742.
#
# Registered predictions, TWO-SIDED per LESSONS 31:
#   pred_a LOCAL SEARCH IMPROVES ON GREEDY on the selection role. If FALSE -- no improving swap exists
#          -- greedy is locally optimal at K=6, which is a genuine positive result for greedy and
#          bounds how much S1742's non-submodularity actually costs.
#   pred_b THE IMPROVEMENT TRANSFERS: whatever local search gains on skip7000 it also gains on
#          skip11000, which chose nothing. If FALSE the gain is selection overfit and greedy is the
#          better recommendation despite losing on its own rows -- the same trap pred_b guarded in
#          S1741, and a swap search is more exposed to it.
#   pred_c BOTH STARTS CONVERGE to the same set. If FALSE the landscape has multiple local optima
#          reachable from reasonable starting points, which means site selection here is genuinely
#          hard and no single-pass procedure should be trusted to find a good allocation.
#   pred_d CONTROLS: the greedy start reproduces S1741's 1.2037 (selection) and 1.2414 (transfer)
#          within 0.001, the all-36-tabled CE reproduces 7.35114 within 0.005, and fit coverage is
#          5419 of 50257.
import json, time, sys, os, random, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256; BUDGET = 6; SEED = 1742; MAX_SWEEPS = 2
SELECT_ROLE = 'skip7000'
# corrected per-module prices, §1718: reals to store one native module
PRICE_M = {'mlp': 15.926, 'attn': 7.963}
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/allocation_local_search_results.json'
PROG_JSON = PT + 'ops/importance_in_program_context_results.json'
OAT_JSON = PT + 'ops/loo_vs_oat_importance_results.json'
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711)]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
S1738_PROGRAM_CE = 7.35114
S1742_GREEDY6 = ['mlp17', 'attn16', 'attn14', 'attn11', 'attn17', 'attn13']
S1741_K6 = {'skip7000': 1.2037, 'skip11000': 1.2414}
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


def price(keep):
    return round(sum(PRICE_M[name_to_site(n)[0]] for n in keep), 3)


@torch.no_grad()
def main():
    t0 = time.time()
    fit = load(FIT_ROWS)
    sites = [(k, L) for k in ('mlp', 'attn') for L in range(18)]
    names = [f'{k}{L}' for k, L in sites]
    print(f'ALLOCATION LOCAL SEARCH | K={BUDGET} | steepest-improvement swaps from two starts | '
          f'selection on {SELECT_ROLE}, TRANSFER on the other | DISCOVERY ONLY', flush=True)

    COV['seen'] = torch.zeros(50257, dtype=torch.bool, device=DEV)
    tables, seen = fit_tables(fit, sites)
    COV['seen'] = seen
    ncov = int(seen.sum())
    print(f'  fit coverage {ncov} of 50257 token ids', flush=True)

    ev = {n: load(p) for n, p, _ in EVAL_SETS}

    def run(rows, keep):
        ks = {name_to_site(n) for n in keep}
        return ce(rows, [mod_of(*st).register_forward_hook(table_hook(tables[st], seen))
                         for st in sites if st not in ks])

    base = {}
    for ename, _, ce_ref in EVAL_SETS:
        cl = ce(ev[ename])
        assert abs(cl - ce_ref) <= 1e-2, f'{ename} live CE {cl:.5f} != {ce_ref}'
        base[ename] = {'live': cl, 'full': run(ev[ename], [])}
    other = [e for e, _, _ in EVAL_SETS if e != SELECT_ROLE][0]
    full_s = base[SELECT_ROLE]['full']

    def val(keep, role=SELECT_ROLE):
        return base[role]['full'] - run(ev[role], keep)

    rng = random.Random(SEED)
    starts = {'greedy': list(S1742_GREEDY6), 'random': sorted(rng.sample(names, BUDGET))}
    print(f'  starts: greedy {starts["greedy"]}', flush=True)
    print(f'          random {starts["random"]}', flush=True)

    results, evals = {}, 0
    for tag, start in starts.items():
        cur, curv = list(start), val(start)
        trace = [{'sweep': 0, 'set': list(cur), 'value': round(curv, 5)}]
        print(f'\n  [{tag}] start {curv:.4f} nats', flush=True)
        for sweep in range(MAX_SWEEPS):
            best, bestv = None, curv
            for out_n in list(cur):
                for in_n in names:
                    if in_n in cur:
                        continue
                    cand = [x for x in cur if x != out_n] + [in_n]
                    v = val(cand)
                    evals += 1
                    if v > bestv + 1e-9:
                        best, bestv = (out_n, in_n, cand), v
            if best is None:
                print(f'    sweep {sweep + 1}: no improving swap -- locally optimal', flush=True)
                break
            o, i, cand = best
            print(f'    sweep {sweep + 1}: swap out {o:7s} in {i:7s} -> {bestv:.4f} nats '
                  f'(+{bestv - curv:.4f})', flush=True)
            cur, curv = sorted(cand), bestv
            trace.append({'sweep': sweep + 1, 'swap_out': o, 'swap_in': i,
                          'set': list(cur), 'value': round(curv, 5)})
        results[tag] = {'start': list(start), 'start_value': round(val(start), 5),
                        'start_transfer': round(val(start, other), 5),
                        'final': list(cur), 'final_value': round(curv, 5),
                        'final_transfer': round(val(cur, other), 5),
                        'final_cost_M': price(cur), 'start_cost_M': price(start),
                        'trace': trace, 'locally_optimal': len(trace) - 1 < MAX_SWEEPS}
        r = results[tag]
        print(f'    final {r["final"]}  sel {r["final_value"]:.4f}  transfer '
              f'{r["final_transfer"]:.4f}  cost {r["final_cost_M"]:.3f}M', flush=True)

    g = results['greedy']
    pa = g['final_value'] > g['start_value'] + 1e-9
    pb = g['final_transfer'] > g['start_transfer'] + 1e-9
    pc = set(results['greedy']['final']) == set(results['random']['final'])
    pd = (abs(base['skip7000']['full'] - S1738_PROGRAM_CE) <= 0.005 and ncov == 5419
          and abs(g['start_value'] - S1741_K6['skip7000']) <= 0.001
          and abs(g['start_transfer'] - S1741_K6['skip11000']) <= 0.001)

    print(f'\n  local search improves on greedy (selection role) -> {pa}', flush=True)
    print(f'  the improvement TRANSFERS to {other} -> not selection overfit {pb}', flush=True)
    print(f'  both starts converge to the same set -> single basin {pc}', flush=True)
    print(f'  greedy start reproduces §1741 + coverage {ncov} -> control {pd}', flush=True)
    print(f'  ({evals} swap evaluations)', flush=True)

    r = {'config': {'K': BUDGET, 'max_sweeps': MAX_SWEEPS, 'seed': SEED,
                    'select_role': SELECT_ROLE, 'transfer_role': other,
                    'search': 'steepest-improvement swap: at each sweep try every (native out, '
                              'tabled in) pair and take the best strict improvement. Two starts, so '
                              '"greedy is locally optimal" and "the landscape has one basin" are '
                              'separate answerable questions.',
                    'WHY': 'S1742 proved from the greedy trace that the objective is NOT submodular '
                           '(the marginal gain rises four times), so greedy has no approximation '
                           'guarantee. This measures how far from good it actually is.',
                    'ROLE_NOTE': 'DISCOVERY ONLY. Both large roles spent for this family '
                                 '(S1736-S1742).'},
         'results': results, 'swap_evaluations': evals,
         'predictions': {'pred_a_local_search_improves': bool(pa),
                         'pred_b_improvement_transfers': bool(pb),
                         'pred_c_single_basin': bool(pc),
                         'pred_d_controls': bool(pd)},
         'runtime_s': round(time.time() - t0, 1)}
    json.dump(r, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
