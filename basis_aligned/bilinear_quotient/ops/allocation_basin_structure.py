# ALLOCATION BASIN STRUCTURE -- the question S1743 asked and could not answer.
#
# S1743 found greedy's K=6 set locally optimal under all 180 single swaps, which is a real result.
# Its second arm asked whether a random start converges to the same set, and could not answer:
# `MAX_SWEEPS = 2` cut the random start off while it was still improving by +0.21 nats per sweep.
# An under-budgeted arm cannot answer its own question, which is now recorded as an addendum to
# LESSONS 31 alongside the nested arm in the same run.
#
# This is the arm re-run with enough budget: three random starts, steepest-improvement swaps, run to
# actual CONVERGENCE (no improving swap exists) with an 8-sweep ceiling, and every set's value cached
# by frozenset since it cannot depend on the path that reached it.
#
# What turns on it. If reasonable starts land in different local optima, site selection at this
# budget is genuinely hard, no single-pass procedure should be trusted to find a good allocation, and
# S1739-S1742's frontier is a lower bound with a landscape reason behind it rather than only a
# missing-guarantee reason. If they all reach the same place, the landscape is benign and greedy's
# local optimality is much stronger evidence than it currently is.
#
# ROLES. Selection on skip7000; skip11000 chooses nothing and reports transfer. DISCOVERY ONLY, same
# family as S1736-S1743, both large roles spent for certification.
#
# Registered predictions, TWO-SIDED per LESSONS 31 and checked against each other so that no arm is
# decided by another's outcome:
#   pred_a ALL THREE STARTS CONVERGE within 8 sweeps. If FALSE the ceiling is still too low and this
#          run inherits S1743's defect -- which would be reported as such, not as ruggedness.
#   pred_b EVERY CONVERGED OPTIMUM IS STRICTLY WORSE than greedy's 1.2037. If FALSE, some random
#          start reaches greedy's value or better and greedy's local optimality is unremarkable.
#          Scored over whichever starts converge, so it is not decided by pred_a.
#   pred_c AT LEAST TWO DISTINCT converged optima -- multiple basins. If FALSE and all converge to
#          one set, the landscape is benign. Also scored over whichever starts converge.
#   pred_d CONTROLS: the greedy reference set reproduces S1741's 1.2037 within 0.001, the
#          all-36-tabled CE reproduces 7.35114 within 0.005, and fit coverage is 5419 of 50257.
import json, time, sys, os, random, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256; BUDGET = 6; SEED = 1743; MAX_SWEEPS = 8; NSTART = 3
SELECT_ROLE = 'skip7000'
# corrected per-module prices, §1718: reals to store one native module
PRICE_M = {'mlp': 15.926, 'attn': 7.963}
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/allocation_basin_structure_results.json'
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
    print(f'ALLOCATION BASIN STRUCTURE | K={BUDGET} | {NSTART} random starts run to CONVERGENCE '
          f'(cap {MAX_SWEEPS} sweeps) | selection on {SELECT_ROLE} | DISCOVERY ONLY', flush=True)
    print(f'  §1743 left this unanswered: its random arm was still improving by +0.21 nats when a '
          f'2-sweep cap hit, so it could not say whether starts converge to one basin.', flush=True)

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

    # a set's value does not depend on how the search reached it, so cache across starts and sweeps
    cache = {}
    stats = {'evals': 0, 'hits': 0}

    def val(keep, role=SELECT_ROLE):
        k = (role, frozenset(keep))
        if k in cache:
            stats['hits'] += 1
            return cache[k]
        v = base[role]['full'] - run(ev[role], keep)
        cache[k] = v
        stats['evals'] += 1
        return v

    rng = random.Random(SEED)
    starts = [sorted(rng.sample(names, BUDGET)) for _ in range(NSTART)]
    assert len({tuple(s) for s in starts}) == NSTART, 'random starts collided'

    results = []
    for si, start in enumerate(starts):
        cur, curv = list(start), val(start)
        conv, sweeps = False, 0
        print(f'\n  [start {si + 1}] {start}  ->  {curv:.4f} nats', flush=True)
        for sweep in range(MAX_SWEEPS):
            sweeps = sweep + 1
            best, bestv = None, curv
            for out_n in list(cur):
                for in_n in names:
                    if in_n in cur:
                        continue
                    cand = [x for x in cur if x != out_n] + [in_n]
                    v = val(cand)
                    if v > bestv + 1e-9:
                        best, bestv = (out_n, in_n, cand), v
            if best is None:
                conv = True
                print(f'    sweep {sweeps}: no improving swap -- CONVERGED at {curv:.4f}', flush=True)
                break
            o, i, cand = best
            print(f'    sweep {sweeps}: out {o:7s} in {i:7s} -> {bestv:.4f} (+{bestv - curv:.4f})',
                  flush=True)
            cur, curv = sorted(cand), bestv
        results.append({'start': start, 'start_value': round(val(start), 5),
                        'final': list(cur), 'final_value': round(curv, 5),
                        'final_transfer': round(val(cur, other), 5),
                        'final_cost_M': price(cur), 'converged': conv, 'sweeps': sweeps})

    gv = val(list(S1742_GREEDY6))
    conv = [r for r in results if r['converged']]
    pa = len(conv) == NSTART
    pb = all(r['final_value'] < gv - 1e-9 for r in conv) if conv else False
    pc = len({tuple(sorted(r['final'])) for r in conv}) >= 2 if len(conv) >= 2 else False
    pd = (abs(base['skip7000']['full'] - S1738_PROGRAM_CE) <= 0.005 and ncov == 5419
          and abs(gv - S1741_K6['skip7000']) <= 0.001)

    print(f'\n  greedy reference {gv:.4f} nats  ({S1742_GREEDY6})', flush=True)
    for r in results:
        print(f'    start {r["start_value"]:7.4f} -> final {r["final_value"]:7.4f} '
              f'(transfer {r["final_transfer"]:7.4f}, cost {r["final_cost_M"]:7.3f}M, '
              f'{r["sweeps"]} sweeps, converged {r["converged"]}) {r["final"]}', flush=True)
    print(f'\n  all {NSTART} starts converged within {MAX_SWEEPS} sweeps -> {pa}', flush=True)
    print(f'  every converged optimum is strictly worse than greedy -> {pb}', flush=True)
    print(f'  at least two DISTINCT converged optima -> multiple basins {pc}', flush=True)
    print(f'  greedy reference + all-tabled CE + coverage {ncov} -> control {pd}', flush=True)
    print(f'  ({stats["evals"]} evaluations, {stats["hits"]} cache hits)', flush=True)

    r = {'config': {'K': BUDGET, 'n_starts': NSTART, 'max_sweeps': MAX_SWEEPS, 'seed': SEED,
                    'select_role': SELECT_ROLE, 'transfer_role': other,
                    'search': 'steepest-improvement swap to convergence; a set value is cached by '
                              'frozenset since it cannot depend on the path that reached it',
                    'WHY': 'S1743 found greedy locally optimal but left the basin question open -- '
                           'its random arm was still improving when a 2-sweep cap hit. An '
                           'under-budgeted arm cannot answer its own question (LESSONS 31 addendum).',
                    'ROLE_NOTE': 'DISCOVERY ONLY. Both large roles spent for this family.'},
         'greedy_reference': {'set': S1742_GREEDY6, 'value': round(gv, 5)},
         'results': results, 'evaluations': stats['evals'], 'cache_hits': stats['hits'],
         'predictions': {'pred_a_all_starts_converge': bool(pa),
                         'pred_b_all_optima_worse_than_greedy': bool(pb),
                         'pred_c_multiple_basins': bool(pc),
                         'pred_d_controls': bool(pd)},
         'runtime_s': round(time.time() - t0, 1)}
    json.dump(r, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
