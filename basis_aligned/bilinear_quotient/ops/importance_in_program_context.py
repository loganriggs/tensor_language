# SITE IMPORTANCE MEASURED IN PROGRAM CONTEXT -- a third definition, because the first two disagree
# completely and both have a known defect.
#
# §1736: one-at-a-time (OAT) constant ablation overstates MLP sites 2.4x and understates attention
# sites 2.5x relative to the joint stack. §1737: OAT and leave-one-out (LOO) rank the 36 sites with
# Spearman 0.026 -- no relationship at all -- and LOO has a defect of its own. Leaving mlp2 LIVE
# while the other seventeen MLPs are constants costs 1.72 nats MORE than ablating mlp2 too, because
# mlp2 then reads a residual stream that no longer resembles anything it was trained on. So:
#
#   OAT  ignores redundancy: seventeen other sites absorb the loss, and the number reflects that.
#   LOO  puts the surviving site OFF-DISTRIBUTION: the number is partly about how badly it copes
#        with inputs it never sees, which is not what "how important is this site" was asking.
#
# A third context avoids both. Replace the other 35 sites with their PER-TOKEN TABLES -- the simplest
# non-trivial program, and the one `ops/circuit_audit` already uses -- instead of with constants. A
# table neighbour still carries token-dependent structure, so the surviving site sees something much
# closer to its training distribution than a constant, while the measurement still asks what that one
# site adds over a program that could stand in for it.
#
#   PROG_i = CE(all 36 tabled) - CE(35 tabled, site i LIVE)
#            what site i adds over the best per-token stand-in for itself, in a context where every
#            other site also has a stand-in rather than a constant.
#
# ROLES. Same family as §1736/§1737, whose confirmation role was skip11000, so both large roles are
# spent. DISCOVERY ONLY: this confirms nothing and certifies nothing. Its product is a third ranking
# and a frozen list, for whichever clean role appears next.
#
# Registered predictions, TWO-SIDED per LESSONS 31:
#   pred_a PROGRAM CONTEXT SITS CLOSER TO LOO THAN TO OAT: Spearman(PROG, LOO) > Spearman(PROG, OAT)
#          on both roles. Both alternatives are informative -- if PROG tracks OAT instead, then the
#          OAT ranking was defensible all along and §1737 overstated the damage to it.
#   pred_b THE TABLE CONTEXT REMOVES THE NEGATIVE-IMPORTANCE PATHOLOGY: no site has negative PROG on
#          either role, against LOO's three sites on skip7000 and five on skip11000. If FALSE, an
#          off-distribution penalty survives even when neighbours keep token-dependent structure,
#          which is a stronger and more troubling statement about ablation-based importance in
#          general.
#   pred_c CONTROLS: the live CE reproduces 3.29205 (§1695) within 1e-3; the all-36-table program CE
#          reproduces 7.3515 within 0.02, the value implied by `circuit_audit`'s independently
#          computed 5.5684-nat stake and 27.10% table extraction for `_whole_model_program`; and the
#          fit coverage is exactly 5419 of 50257 token ids, as every run in this thread has had.
#   pred_d THE FRONT BAND DOMINATES IN PROGRAM CONTEXT: at least two of mlp0-mlp3 are in the top six
#          by PROG on both roles. §1662 found the front MLPs the most tabular sites (mlp0 90%, mlp1
#          96%) -- if a site's own table is nearly as good as the site, it should add LITTLE over its
#          table, so a pass here would mean tabularity and program-context importance are measuring
#          different things and a fail would mean they agree.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/importance_in_program_context_results.json'
PRIOR = PT + 'ops/loo_vs_oat_importance_results.json'
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711)]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
S_WHOLE_PROGRAM_CE = 7.3515   # implied by circuit_audit: 3.29205 + 5.5684 - 0.2710 * 5.5684
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


def spearman(a, b):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((x - mb) ** 2 for x in rb) ** 0.5
    return num / (da * db) if da * db else float('nan')


@torch.no_grad()
def main():
    t0 = time.time()
    fit = load(FIT_ROWS)
    sites = [(k, L) for k in ('mlp', 'attn') for L in range(18)]
    prior = json.load(open(PRIOR))['results']
    print('SITE IMPORTANCE IN PROGRAM CONTEXT | 36 per-token tables | DISCOVERY ONLY -- both large '
          'roles are spent for this family (§1736/§1737). Confirms nothing.', flush=True)

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
        prog = ce(ev, [mod_of(*st).register_forward_hook(table_hook(tables[st], seen))
                       for st in sites])
        rows = {}
        for st in sites:
            others = [s for s in sites if s != st]
            c1 = ce(ev, [mod_of(*s).register_forward_hook(table_hook(tables[s], seen))
                         for s in others])
            rows[f'{st[0]}{st[1]}'] = {'prog': round(prog - c1, 5), 'ce_site_live': round(c1, 5),
                                       'kind': st[0]}
        names = list(rows)
        pv = [rows[n]['prog'] for n in names]
        oat = [prior[ename]['sites'][n]['oat'] for n in names]
        loo = [prior[ename]['sites'][n]['loo'] for n in names]
        r_loo, r_oat = spearman(pv, loo), spearman(pv, oat)
        neg = sorted([(n, rows[n]['prog']) for n in names if rows[n]['prog'] < 0],
                     key=lambda x: x[1])
        top = sorted(names, key=lambda n: -rows[n]['prog'])[:6]
        print(f'\n  {ename}: live CE {cl:.5f} | 36-table program CE {prog:.5f} '
              f'(+{prog - cl:.4f} over live)', flush=True)
        print(f'    Spearman(PROG, LOO) {r_loo:+.4f}   Spearman(PROG, OAT) {r_oat:+.4f}', flush=True)
        print(f'    top 6 by PROG: {[(n, rows[n]["prog"]) for n in top]}', flush=True)
        print(f'    negative-PROG sites: {len(neg)} {neg[:5]}', flush=True)
        out[ename] = {'live_ce': round(cl, 5), 'program_ce': round(prog, 5), 'sites': rows,
                      'spearman_prog_loo': round(r_loo, 5), 'spearman_prog_oat': round(r_oat, 5),
                      'top6_prog': top, 'negative_prog': neg}
        del ev
        torch.cuda.empty_cache()

    pa = all(out[e]['spearman_prog_loo'] > out[e]['spearman_prog_oat'] for e in out)
    pb = all(len(out[e]['negative_prog']) == 0 for e in out)
    pc = (abs(out['skip7000']['live_ce'] - 3.29205) <= 1e-3
          and abs(out['skip7000']['program_ce'] - S_WHOLE_PROGRAM_CE) <= 0.02
          and ncov == 5419)
    front = {'mlp0', 'mlp1', 'mlp2', 'mlp3'}
    pd = all(len(front & set(out[e]['top6_prog'])) >= 2 for e in out)

    frozen = sorted(out['skip7000']['sites'],
                    key=lambda n: -min(out['skip7000']['sites'][n]['prog'],
                                       out['skip11000']['sites'][n]['prog']))[:6]
    print(f'\n  PROG closer to LOO than to OAT on both roles -> {pa}', flush=True)
    print(f'  no negative-importance site in table context -> pathology removed {pb}', flush=True)
    print(f'  live CE + 36-table program CE {out["skip7000"]["program_ce"]:.5f} vs '
          f'{S_WHOLE_PROGRAM_CE} + coverage {ncov} -> control {pc}', flush=True)
    print(f'  front band holds >=2 of the top six on both roles -> {pd}', flush=True)
    print(f'\n  FROZEN: top 6 by the WORSE of the two roles\' PROG: {frozen}', flush=True)

    r = {'config': {'eval_sets': [e[0] for e in EVAL_SETS],
                    'PROG': 'CE(all 36 sites tabled) - CE(35 tabled, site i live). What a site adds '
                            'over the best per-token stand-in for itself, in a context where every '
                            'other site also has a stand-in rather than a constant.',
                    'ROLE_NOTE': 'DISCOVERY ONLY. Same family as §1736/§1737, whose confirmation role '
                                 'was skip11000, so both large roles are spent. Certifies nothing.',
                    'tables': 'per-token mean output fitted on fineweb_n96_skip80, hybrid hook '
                              '(§1661): the table applies only where the token was seen at fit, the '
                              'module runs live elsewhere'},
         'results': out, 'frozen_for_next_clean_role': frozen,
         'predictions': {'pred_a_prog_tracks_loo_not_oat': bool(pa),
                         'pred_b_no_negative_importance': bool(pb),
                         'pred_c_controls': bool(pc),
                         'pred_d_front_band_dominates': bool(pd)},
         'runtime_s': round(time.time() - t0, 1)}
    json.dump(r, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
