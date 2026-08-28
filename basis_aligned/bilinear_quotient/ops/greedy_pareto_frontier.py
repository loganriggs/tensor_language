# GREEDY PARETO FRONTIER -- the deliverable the whole thread has been circling: recovered fidelity
# against native-module cost, at every budget, chosen by the procedure that actually works.
#
# The arc's stated metric is reconstruction AGAINST simplicity, not either alone. §1741 gave one
# point: greedy at six native sites recovers 29.13% on a transfer role against the ranking's 24.37%,
# and because five of its six picks are attention sites -- which cost half an MLP -- it does so at
# 55.741M reals against the ranking's 71.667M. Better AND cheaper, with nothing in the greedy
# objective knowing about cost.
#
# Greedy prefixes are NESTED, so one run to K=14 yields the whole curve for free: fourteen points in
# (native cost, recovered nats), each with the ranking's matched-K allocation beside it and each
# evaluated on the role that selected nothing.
#
# Prices are the corrected §1718 figures: 15.926M reals for an MLP module, 7.963M for an attention
# site. They enter only in reporting; the greedy objective is pure fidelity, so any cost advantage is
# a consequence rather than a target.
#
# ROLES. Both large roles remain spent for CERTIFICATION -- DISCOVERY ONLY, same family as
# §1736-§1741. Selection runs on skip7000; skip11000 chooses nothing and carries every comparison, so
# greedy's own overfitting risk is scored rather than assumed away.
#
# Registered predictions, TWO-SIDED per LESSONS 31:
#   pred_a GREEDY DOMINATES ON BOTH AXES AT EVERY BUDGET: at every K from 1 to 14, on the TRANSFER
#          role, greedy recovers more than the individual-score ranking AND costs no more in native
#          modules. A single K where the ranking is cheaper-or-better breaks it, and would mean the
#          §1741 point was a budget where the two happened to separate.
#   pred_b ATTENTION IS THE MAJORITY at K=14. §1736 says the attention stack's joint contribution is
#          2.5x the sum of its individual sites, so a procedure that scores GIVEN what is already
#          native should keep finding attention worth keeping. If MLPs dominate at larger budgets,
#          §1741's five-of-six was a small-budget effect.
#   pred_c NO KNEE: greedy's marginal nats per million reals is strictly decreasing across all
#          fourteen steps. A knee -- a budget where the marginal rises again -- names a natural
#          stopping point and is the more useful outcome, which is why finding one FAILS this.
#   pred_d CONTROLS: the all-36-tabled CE reproduces 7.35114 within 0.005, fit coverage is 5419 of
#          50257, and the K=6 prefix reproduces §1741's greedy SET exactly -- mlp17, attn16, attn14,
#          attn11, attn17, attn13 -- and its 1.2037 / 1.2414 within 0.001. Greedy is deterministic,
#          so an exact set reproduction is available as a control and is used as one.
import json, time, sys, os, random, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256; BUDGET = 14; SEED = 1741
SELECT_ROLE = 'skip7000'
# corrected per-module prices, §1718: reals to store one native module
PRICE_M = {'mlp': 15.926, 'attn': 7.963}
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/greedy_pareto_frontier_results.json'
PROG_JSON = PT + 'ops/importance_in_program_context_results.json'
OAT_JSON = PT + 'ops/loo_vs_oat_importance_results.json'
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711)]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
S1738_PROGRAM_CE = 7.35114
S1741_GREEDY6 = ['mlp17', 'attn16', 'attn14', 'attn11', 'attn17', 'attn13']
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
    pj = json.load(open(PROG_JSON))['results']
    prog_rank = sorted(names, key=lambda n: -min(pj['skip7000']['sites'][n]['prog'],
                                                 pj['skip11000']['sites'][n]['prog']))
    print(f'GREEDY PARETO FRONTIER | greedy to K={BUDGET} of 36 native sites, nested prefixes | '
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

    sel, evals = [], 0
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
        print(f'    greedy step {step + 1:2d}: + {best:7s} -> {bestv:7.4f} nats  '
              f'(native cost {price(sel):7.3f}M)', flush=True)
    print(f'  greedy order: {sel}   ({evals} candidate evaluations)', flush=True)

    frontier, prev, prevp = [], 0.0, 0.0
    print(f'\n  {"K":>3s} {"native cost":>11s} {"greedy sel":>10s} {"greedy TRF":>10s} '
          f'{"rank sel":>9s} {"rank cost":>9s}   nats per M', flush=True)
    for K in range(1, BUDGET + 1):
        g = {e: base[e]['full'] - run(ev[e], sel[:K]) for e in ev}
        rk = prog_rank[:K]
        rv = {e: base[e]['full'] - run(ev[e], rk) for e in ev}
        p = price(sel[:K])
        npm = (g[SELECT_ROLE] - prev) / max(p - prevp, 1e-9)
        prev, prevp = g[SELECT_ROLE], p
        row = {'K': K, 'greedy_set': list(sel[:K]), 'native_cost_M': p,
               'greedy': {e: round(g[e], 5) for e in g},
               'greedy_frac': {e: round(g[e] / (base[e]['full'] - base[e]['live']), 5) for e in g},
               'rank_set': rk, 'rank_native_cost_M': price(rk),
               'rank': {e: round(rv[e], 5) for e in rv},
               'greedy_marginal_nats_per_Mreal': round(npm, 6),
               'n_attn': sum(1 for n in sel[:K] if n.startswith('attn'))}
        frontier.append(row)
        print(f'  {K:3d} {p:11.3f} {g[SELECT_ROLE]:10.4f} {g[other]:10.4f} '
              f'{rv[SELECT_ROLE]:9.4f} {price(rk):9.3f}   {npm:9.5f}', flush=True)

    pa = all(r['greedy'][other] > r['rank'][other] and r['native_cost_M'] <= r['rank_native_cost_M']
             for r in frontier)
    pb = frontier[-1]['n_attn'] > BUDGET - frontier[-1]['n_attn']
    npm = [r['greedy_marginal_nats_per_Mreal'] for r in frontier]
    pc = all(npm[i] > npm[i + 1] for i in range(len(npm) - 1))
    k6 = frontier[5]
    pd = (abs(base['skip7000']['full'] - S1738_PROGRAM_CE) <= 0.005 and ncov == 5419
          and k6['greedy_set'] == S1741_GREEDY6
          and all(abs(k6['greedy'][e] - v) <= 0.001 for e, v in S1741_K6.items()))

    print(f'\n  greedy dominates the ranking on BOTH axes at every K (transfer role) -> {pa}',
          flush=True)
    print(f'  attention is the majority of the K={BUDGET} greedy set '
          f'({frontier[-1]["n_attn"]}/{BUDGET}) -> {pb}', flush=True)
    print(f'  nats per M real strictly decreasing {[round(x, 4) for x in npm]} -> no knee {pc}',
          flush=True)
    print(f'  K=6 prefix reproduces §1741 set and values + coverage {ncov} -> control {pd}',
          flush=True)

    r = {'config': {'budget_max': BUDGET, 'select_role': SELECT_ROLE, 'transfer_role': other,
                    'prices_M_reals': PRICE_M,
                    'frontier': 'each K is a point in (native module cost, recovered nats). Greedy '
                                'prefixes are NESTED, so one greedy run to K gives the whole curve.',
                    'ROLE_NOTE': 'DISCOVERY ONLY. Both large roles remain spent for certification '
                                 '(§1736-§1741); the selection/transfer split guards selection '
                                 'overfit, which is a different failure from role burn.'},
         'greedy_order': sel, 'frontier': frontier,
         'predictions': {'pred_a_greedy_dominates_ranking_both_axes': bool(pa),
                         'pred_b_attention_majority': bool(pb),
                         'pred_c_no_knee_in_nats_per_Mreal': bool(pc),
                         'pred_d_controls': bool(pd)},
         'runtime_s': round(time.time() - t0, 1)}
    json.dump(r, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
