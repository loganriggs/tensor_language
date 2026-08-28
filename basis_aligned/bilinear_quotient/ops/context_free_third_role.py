# SECOND-CLASS CONFIRMATION OF THE CONTEXT-FREE FRONTIER ON A THIRD ROLE
#
# `_CONTEXT_FREE_TABLE_FRONTIER` was certified from §1769-§1771 on skip7000 and skip11000 only. It is
# the headline of this arc -- it dominates every program in §1748-§1758 by 13x on cost -- and it has
# never been scored on a third set of rows. House pattern (S1595, S1598, S1603): confirm on a second
# class of rows before leaning on it.
#
# `fineweb_n96_skip1200` is a pinned role, half the row count of the others, and has never been scored
# for any program-frontier quantity. It WAS used as the confirmation role for the token-class family
# in §1734, which is a different hypothesis; that is stated rather than glossed, and it means this is
# a second-class confirmation and not a virgin role.
#
# The fit-mean all-tabled baseline is not published for skip1200, so it is computed in-run from the
# same fit rows, and every recovery on every role is measured against that role's own baseline.
#
# ROLES. skip7000 and skip11000 replicate §1770/§1771 as controls; skip1200 carries the confirmation.
# DISCOVERY plus SECOND-CLASS CONFIRMATION.
#
# Registered predictions, TWO-SIDED per LESSONS 31, with MARGINS per LESSON 40, each read back against
# its own sentence per LESSON 39:
#   pred_a THE DESIGN POINT REPLICATES: on skip1200 the cost-efficiency optimum over the tested ranks
#          is rank 4, as it is on skip11000 (§1771). If FALSE the design point is role-specific and
#          the frontier entry needs a per-role caveat.
#   pred_b THE FLOOR REPLICATES: rank 1 recovers less than zero on skip1200, as on both other roles.
#          Scored independently of pred_a.
#   pred_c THE LEVELS TRANSFER: at every tested rank, skip1200's recovery is within 15% RELATIVE of
#          skip11000's. If FALSE the frontier's shape survives but its magnitudes do not, and the
#          certified numbers need role-conditional wording.
#   pred_d CONTROLS: skip11000 reproduces §1770/§1771's +1.37925 (full), +1.14673 (64), +0.80353 (8),
#          +0.63791 (4), +0.23992 (2), -0.34772 (1) within 0.002, and coverage is 5419 of 50257.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256; V = 50257
TABLE_RANKS = (None, 64, 8, 4, 2, 1)
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/context_free_third_role_results.json'
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt'),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt'),
             ('skip1200', PT + '.rowcache/fineweb_n96_skip1200.pt')]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
NCOV = 5419
S1771_SKIP11000 = {'full': 1.37925, '64': 1.14673, '8': 0.80353,
                   '4': 0.63791, '2': 0.23992, '1': -0.34772}
STATE = {}
COV = {}


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


def mod_of(kind, L):
    return H[L].mlp if kind == 'mlp' else H[L].attn


def table_hook(tbl, seen):
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        sub = tbl[STATE['idx'].reshape(-1)].reshape(y.shape).to(y.dtype)
        sub = torch.where(seen[STATE['idx']].unsqueeze(-1), sub, y)
        return (sub,) + tuple(out[1:]) if isinstance(out, tuple) else sub
    return hook


@torch.no_grad()
def forward_logits(idx, hooks=()):
    hs = [mod_of(*st).register_forward_hook(h) for st, h in hooks]
    STATE['idx'] = idx
    try:
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in H:
            x, v1 = blk(x, v1, x0)
        return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
    finally:
        for h in hs:
            h.remove()


@torch.no_grad()
def ce(rows, hooks=()):
    tot, cnt = 0.0, 0
    for i in range(0, rows.shape[0], 8):
        bb = rows[i:i + 8]
        idx = bb[:, :-1].to(DEV).contiguous()
        lg = forward_logits(idx, hooks)
        tg = bb[:, 1:].to(DEV)
        e = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                            reduction='none').reshape(tg.shape)[:, 64:].double()
        c = COV['seen'][idx[:, 64:]]
        tot += float(e[c].sum()); cnt += int(c.sum())
    return tot / cnt


@torch.no_grad()
def build_tables(fit, sites, seen, toks, context_free):
    """context_free=True: each site's output on a LENGTH-1 sequence per covered token (§1769).
    context_free=False: the per-token MEAN over the fit rows -- the §1747-§1758 construction, needed
    here only to produce each role's own all-tabled baseline."""
    tables = {st: torch.zeros(V, D, device=DEV) for st in sites}
    if context_free:
        cap = {}

        def mk(st):
            def hook(mod, args, out):
                cap[st] = (out[0] if isinstance(out, tuple) else out)[:, 0].float()
                return None
            return hook
        for i in range(0, toks.numel(), 256):
            t = toks[i:i + 256].to(DEV).unsqueeze(1)
            forward_logits(t, [(st, mk(st)) for st in sites])
            for st in sites:
                tables[st][t.squeeze(1)] = cap[st]
    else:
        c = torch.zeros(V, device=DEV)
        acc = {st: torch.zeros(V, D, device=DEV) for st in sites}

        def mk2(st, first):
            def hook(mod, args, out):
                y = (out[0] if isinstance(out, tuple) else out).float().reshape(-1, D)
                tk = STATE['idx'].reshape(-1)
                acc[st].index_add_(0, tk, y)
                if first:
                    c.index_add_(0, tk, torch.ones_like(tk, dtype=torch.float32))
                return None
            return hook
        for i in range(0, fit.shape[0], 8):
            forward_logits(fit[i:i + 8, :-1].to(DEV).contiguous(),
                           [(st, mk2(st, j == 0)) for j, st in enumerate(sites)])
        sn = c > 0
        for st in sites:
            tables[st][sn] = acc[st][sn] / c[sn].unsqueeze(1)
    for st in sites:
        mu = tables[st][toks.to(DEV)].mean(0)
        tables[st][~seen] = mu
    return tables


@torch.no_grad()
def truncate(tables, toks, r):
    if r is None:
        return tables, 36 * (NCOV * D + D)
    out = {}
    idx = toks.to(DEV)
    for st, tbl in tables.items():
        blk = tbl[idx].double()
        mu = blk.mean(0, keepdim=True)
        U, S, Vh = torch.linalg.svd(blk - mu, full_matrices=False)
        t2 = tbl.clone()
        t2[idx] = (mu + (U[:, :r] * S[:r]) @ Vh[:r]).float()
        out[st] = t2
    return out, 36 * (r * (NCOV + D) + 2 * D)


@torch.no_grad()
def main():
    t0 = time.time()
    fit = load(FIT_ROWS)
    seen_cpu = torch.zeros(V, dtype=torch.bool)
    seen_cpu[fit[:, :T].reshape(-1).long()] = True
    ncov = int(seen_cpu.sum())
    assert ncov == NCOV, f'coverage {ncov} != {NCOV}'
    seen = seen_cpu.to(DEV)
    COV['seen'] = seen
    toks = seen_cpu.nonzero(as_tuple=True)[0]
    sites = [(k, L) for k in ('mlp', 'attn') for L in range(18)]
    print(f'CONTEXT-FREE FRONTIER, THIRD ROLE | ranks {TABLE_RANKS} | skip1200 confirms | '
          f'SECOND-CLASS CONFIRMATION', flush=True)

    fm = build_tables(fit, sites, seen, toks, context_free=False)
    fm_hooks = [(st, table_hook(fm[st], seen)) for st in sites]
    ev, base = {}, {}
    for ename, epath in EVAL_SETS:
        e = load(epath)
        ev[ename] = e
        base[ename] = {'live': ce(e), 'all_tabled_fitmean': ce(e, fm_hooks)}
        b = base[ename]
        print(f'  {ename}: live {b["live"]:.5f} | fit-mean all-tabled {b["all_tabled_fitmean"]:.5f} '
              f'| stake {b["all_tabled_fitmean"] - b["live"]:.5f}', flush=True)
    del fm, fm_hooks
    torch.cuda.empty_cache()

    cf = build_tables(fit, sites, seen, toks, context_free=True)
    print(f'  built context-free tables ({time.time() - t0:.0f}s)', flush=True)

    out = {}
    for r in TABLE_RANKS:
        tr, cost = truncate(cf, toks, r)
        hooks = [(st, table_hook(tr[st], seen)) for st in sites]
        key = 'full' if r is None else str(r)
        row = {'cost_M': round(cost / 1e6, 4)}
        for ename in ev:
            c1 = ce(ev[ename], hooks)
            rec = base[ename]['all_tabled_fitmean'] - c1
            row[ename] = {'ce': round(c1, 5), 'recovered': round(rec, 5),
                          'nats_per_M': round(rec / (cost / 1e6), 6)}
        out[key] = row
        print(f'  rank {key:5s} {row["cost_M"]:8.3f}M | ' + '  '.join(
            f'{e} {row[e]["recovered"]:+.5f} ({row[e]["nats_per_M"]:.4f}/M)' for e in ev)
            + f'   [{time.time() - t0:.0f}s]', flush=True)
        if r is not None:
            del tr
            torch.cuda.empty_cache()

    keys = ['full' if r is None else str(r) for r in TABLE_RANKS]
    def best_eff(ename):
        return max(keys, key=lambda k: out[k][ename]['nats_per_M'])
    pa = best_eff('skip1200') == '4'
    pb = out['1']['skip1200']['recovered'] < 0.0
    rel = {k: abs(out[k]['skip1200']['recovered'] - out[k]['skip11000']['recovered'])
              / max(abs(out[k]['skip11000']['recovered']), 1e-9) for k in keys}
    pc = all(v <= 0.15 for v in rel.values())
    pd = (all(abs(out[k]['skip11000']['recovered'] - v) <= 0.002
              for k, v in S1771_SKIP11000.items()) and ncov == NCOV)

    print(f'\n  efficiency optimum on skip1200 is rank {best_eff("skip1200")} '
          f'(skip11000: {best_eff("skip11000")}) -> design point replicates {pa}', flush=True)
    print(f'  rank 1 negative on skip1200 ({out["1"]["skip1200"]["recovered"]:+.5f}) -> {pb}',
          flush=True)
    print(f'  relative level gaps vs skip11000 {[f"{k}:{rel[k]:.1%}" for k in keys]} '
          f'-> all within 15% {pc}', flush=True)
    print(f'  skip11000 reproduces §1770/§1771 + coverage {ncov} -> control {pd}', flush=True)

    r2 = {'config': {'ranks': [str(x) for x in TABLE_RANKS],
                     'third_role': 'fineweb_n96_skip1200 -- a pinned role, half the row count, never '
                                   'scored for a program-frontier quantity. It WAS the confirmation '
                                   'role for the token-class family in S1734, a different hypothesis, '
                                   'so this is a SECOND-CLASS confirmation and not a virgin role.',
                     'baseline': "each role's own fit-mean all-tabled CE, computed in-run because "
                                 'skip1200 has no published baseline',
                     'ROLE_NOTE': 'DISCOVERY plus SECOND-CLASS CONFIRMATION.'},
          'baseline': {e: {k: round(v, 5) for k, v in base[e].items()} for e in base},
          'by_table_rank': out,
          'predictions': {'pred_a_design_point_replicates': bool(pa),
                          'pred_b_floor_replicates': bool(pb),
                          'pred_c_levels_transfer': bool(pc),
                          'pred_d_controls': bool(pd)},
          'runtime_s': round(time.time() - t0, 1)}
    json.dump(r2, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r2["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
