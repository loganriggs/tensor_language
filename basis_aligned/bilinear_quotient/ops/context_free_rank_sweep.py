# DO CONTEXT-FREE TABLES COMPRESS? -- and does §1755's explanation survive?
#
# §1769: rebuilding each site's table from a length-1 forward -- the context-FREE value rather than a
# per-token mean over fit contexts -- closes the entire 0.594-nat gap to the position-wise ceiling,
# to -0.00002 nats, and beats the best fit-mean program WITHOUT any linear correction at all. At full
# rank it costs 224.737M reals for +1.37925 of the 4.2611 stake, against §1758's 25.839M for +0.78536.
# Better and much more expensive, so the two are not yet comparable on the frontier.
#
# §1755 found that truncating fit-mean tables to rank 64 IMPROVED fidelity, and attributed that to the
# tables being overfitted to 96 fit rows -- a per-token mean over few contexts is noisy at the rare
# end. A context-free table is not estimated from data at all: it is an exact model output for that
# token. **If §1755's explanation is right, truncation here should be pure loss with no denoising to
# gain.** That makes this a test of the explanation and not only a sweep.
#
# ROLES. Both eval roles, covered positions from 64, hybrid coverage rule, recoveries measured against
# the same FIT-MEAN all-tabled baseline as §1747-§1758 so every number is directly comparable.
# DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, each read back against its own sentence per
# LESSON 39:
#   pred_a COMPRESSION IS PURE LOSS: covered CE increases monotonically as the rank falls through
#          full, 256, 64, 16, 8 (higher CE = worse). If FALSE -- if truncation helps here too --
#          §1755's overfitting explanation is wrong and something else makes low-rank tables better.
#   pred_b THE EFFICIENCY OPTIMUM IS INTERIOR: nats per million reals peaks at a rank strictly inside
#          the sweep. Scored independently of pred_a, since fidelity can fall monotonically while
#          efficiency still peaks in the middle -- that is the normal shape and its absence would say
#          the sweep is under-budgeted at one end.
#   pred_c IT BEATS §1758 AT MATCHED COST: among arms costing at most 25.839M reals, the best
#          recovers more than §1758's +0.78536 held out. If FALSE the context-free table's advantage
#          is real but only affordable at full rank, and the §1758 frontier point survives on cost.
#   pred_d CONTROLS: the full-rank arm reproduces §1769's 6.03465 and 5.97900 within 0.001; live CE
#          reproduces 3.29205 and 3.09711; coverage is exactly 5419 of 50257.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256; V = 50257; W = 50304
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/context_free_rank_sweep_results.json'
TABLE_RANKS = (None, 256, 64, 16, 8)
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711)]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
NCOV = 5419
S1769_FULL = {'skip7000': 6.03465, 'skip11000': 5.97900}
S1758_BEST = {'cost_M': 25.839, 'recovered': {'skip7000': 0.77602, 'skip11000': 0.78536}}
BEST_FITMEAN_PROGRAM = {'skip7000': 6.57512, 'skip11000': 6.57289}
ALL_TABLED = {'skip7000': 7.35114, 'skip11000': 7.35825}
STATE = {}
COV = {}


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


def mod_of(kind, L):
    return H[L].mlp if kind == 'mlp' else H[L].attn


def table_hook(tbl, seen):
    """The hybrid rule of §1661, unchanged: table where covered, live module elsewhere."""
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
def truncate(tables, seen, toks, r):
    """Rank-r truncate the COVERED block of each context-free table. r=None returns it unchanged.

    Cost per site: full covered block is 5419 x 1152; rank r is r*(5419+1152) plus a mean row and an
    uncovered fallback row -- the identical accounting §1755/§1756 used for fit-mean tables.
    """
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
    print(f'CONTEXT-FREE TABLE RANK SWEEP | ranks {TABLE_RANKS} | no linear correction | '
          f'DISCOVERY ONLY', flush=True)

    tables = {st: torch.zeros(V, D, device=DEV) for st in sites}
    cap = {}

    def mk(st):
        def hook(mod, args, out):
            cap[st] = (out[0] if isinstance(out, tuple) else out)[:, 0].float()
            return None
        return hook

    for i in range(0, ncov, 256):
        t = toks[i:i + 256].to(DEV).unsqueeze(1)
        forward_logits(t, [(st, mk(st)) for st in sites])
        for st in sites:
            tables[st][t.squeeze(1)] = cap[st]
    for st in sites:
        mu = tables[st][toks.to(DEV)].mean(0)
        tables[st][~seen] = mu
    print(f'  built 36 context-free tables ({time.time() - t0:.0f}s)', flush=True)

    ev, base = {}, {}
    for ename, epath, ref in EVAL_SETS:
        e = load(epath)
        ev[ename] = e
        lv = ce(e)
        assert abs(lv - ref) <= 1e-3, f'{ename} live CE {lv:.5f} != {ref}'
        base[ename] = {'live': lv, 'all_tabled_fitmean': ALL_TABLED[ename]}

    out = {}
    for r in TABLE_RANKS:
        tr, cost = truncate(tables, seen, toks, r)
        hooks = [(st, table_hook(tr[st], seen)) for st in sites]
        key = 'full' if r is None else str(r)
        res = {}
        for ename in ev:
            c1 = ce(ev[ename], hooks)
            res[ename] = {'ce': round(c1, 5),
                          'recovered_vs_fitmean_base': round(ALL_TABLED[ename] - c1, 5)}
        out[key] = {'table_rank': r, 'cost_M': round(cost / 1e6, 4), **res,
                    'nats_per_Mreal': round(res['skip11000']['recovered_vs_fitmean_base']
                                            / (cost / 1e6), 6)}
        o = out[key]
        print(f'  rank {key:5s}: cost {o["cost_M"]:9.3f}M | ' + '  '.join(
            f'{e} CE {res[e]["ce"]:.5f} rec {res[e]["recovered_vs_fitmean_base"]:+.5f}'
            for e in res) + f' | {o["nats_per_Mreal"]:.6f} nats/M   [{time.time() - t0:.0f}s]',
            flush=True)
        if r is not None:
            del tr
            torch.cuda.empty_cache()

    ho = 'skip11000'
    keys = ['full' if r is None else str(r) for r in TABLE_RANKS]
    ces = [out[k][ho]['ce'] for k in keys]
    pa = all(ces[i] < ces[i + 1] for i in range(len(ces) - 1))
    eff = [out[k]['nats_per_Mreal'] for k in keys]
    best_eff = max(range(len(eff)), key=lambda i: eff[i])
    pb = 0 < best_eff < len(eff) - 1
    cand = [k for k in keys if out[k]['cost_M'] <= S1758_BEST['cost_M']]
    best_cheap = max(cand, key=lambda k: out[k][ho]['recovered_vs_fitmean_base']) if cand else None
    pc = (best_cheap is not None
          and out[best_cheap][ho]['recovered_vs_fitmean_base'] > S1758_BEST['recovered'][ho])
    pd = (all(abs(out['full'][e]['ce'] - v) <= 0.001 for e, v in S1769_FULL.items())
          and abs(base['skip7000']['live'] - 3.29205) <= 1e-3
          and abs(base[ho]['live'] - 3.09711) <= 1e-3 and ncov == NCOV)

    print(f'\n  CE degrades monotonically as rank falls {ces} -> compression is pure loss {pa}',
          flush=True)
    print(f'  efficiency {[round(x, 4) for x in eff]} peaks at rank {keys[best_eff]} -> interior '
          f'optimum {pb}', flush=True)
    print(f'  at <= {S1758_BEST["cost_M"]}M the best context-free arm is {best_cheap} recovering '
          f'{out[best_cheap][ho]["recovered_vs_fitmean_base"]:+.5f} vs §1758 '
          f'{S1758_BEST["recovered"][ho]:+.5f} -> {pc}', flush=True)
    print(f'  full rank reproduces §1769 + live CEs + coverage {ncov} -> control {pd}', flush=True)

    r2 = {'config': {'tables': 'CONTEXT-FREE (length-1 forward per covered token), then rank-r '
                               'truncated; no linear correction anywhere',
                     'costing': 'identical accounting to §1755/§1756: r*(5419+1152) + 2*1152 per site',
                     'recovered_against': 'the FIT-MEAN all-tabled baseline (7.35114 / 7.35825), so '
                                          'these recoveries are directly comparable to §1747-§1758',
                     'WHY': '§1769 attributed the whole 0.594-nat gap to the tables being context '
                            'averages. §1755 found fit-mean tables IMPROVED under truncation because '
                            'they were overfitted to 96 fit rows; a context-free table is an exact '
                            'model output and is not estimated from data, so truncation should be '
                            'pure loss. That is a test of §1755\'s explanation, not just a sweep.',
                     'ROLE_NOTE': 'DISCOVERY ONLY.'},
          'baseline': {e: {k: round(v, 5) for k, v in base[e].items()} for e in base},
          'by_table_rank': out,
          'predictions': {'pred_a_compression_is_pure_loss': bool(pa),
                          'pred_b_efficiency_interior_optimum': bool(pb),
                          'pred_c_beats_S1758_at_matched_cost': bool(pc),
                          'pred_d_controls': bool(pd)},
          'runtime_s': round(time.time() - t0, 1)}
    json.dump(r2, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r2["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
