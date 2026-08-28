# IS IT THE EMBEDDING CHANNEL? -- testing §1782's mechanism instead of asserting it.
#
# §1782: the composed 36-site program beats a direct length-1 logit lookup on all positions by
# 0.127-0.151 nats, and the two are BIT-IDENTICAL on covered positions. The whole difference is at the
# uncovered quarter, and §1782 proposed a mechanism from the construction rather than from a
# measurement: the table hook replaces every site OUTPUT, but the residual stream still starts at
# `rms_norm(wte(token_j))` -- the TRUE token's embedding. So at an uncovered position the composed
# program is the true token's embedding plus the neighbour's 36 site writes, while the direct lookup
# is the neighbour and nothing else. Composition would then be keeping one channel of the real token
# that a pure neighbour lookup throws away.
#
# That is a hypothesis. The check is one flag: also present the NEIGHBOUR's token id to the embedding
# at uncovered positions. The table rows are unchanged -- rowmap[neighbour] is the neighbour's own
# row -- so exactly one channel moves. If the mechanism is right, the swapped program should collapse
# onto the direct lookup.
#
# ROLES. skip7000, skip11000, skip1200; covered and all-position CE. DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, absolute nats with margins per LESSON 40, read
# back per LESSON 39:
#   pred_a THE EMBEDDING CHANNEL EXPLAINS THE GAP: with the embedding swapped too, all-position CE
#          lands within 0.02 nats of the direct lookup at every role. If FALSE the mechanism is
#          wrong or incomplete -- something other than the residual's input channel is carrying
#          §1782's advantage -- and §1782's paragraph needs withdrawing rather than confirming.
#   pred_b THE SWAP IS EXPENSIVE: it costs at least 0.10 nats against the true-embedding program at
#          every role. Scored independently of pred_a, since the swap could move a lot and still not
#          land on the direct lookup.
#   pred_c COVERED CE IS UNTOUCHED by the swap, to 1e-9, at every role. Covered tokens are their own
#          neighbour, so this must hold exactly; if it does not, the substitution map is wrong.
#   pred_d CONTROLS: the composed arms reproduce §1782's 6.01897/6.03465, 6.00091/5.97900 and
#          6.00733/5.96423 within 0.002, the direct arm its 6.14589, 6.15184 and 6.15373, and
#          coverage is exactly 5419 of 50257.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256; V = 50257; W = 50304
RANKS = (None, 64)
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/embedding_channel_check_results.json'
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711),
             ('skip1200', PT + '.rowcache/fineweb_n96_skip1200.pt', None)]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
NCOV = 5419
S1782_COMPOSED = {'skip7000': {'all': 6.01897, 'cov': 6.03465},
                  'skip11000': {'all': 6.00091, 'cov': 5.97900},
                  'skip1200': {'all': 6.00733, 'cov': 5.96423}}
S1782_DIRECT = {'skip7000': 6.14589, 'skip11000': 6.15184, 'skip1200': 6.15373}
STATE = {}
COV = {}


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


def mod_of(kind, L):
    return H[L].mlp if kind == 'mlp' else H[L].attn


def table_hook(tbl):
    """STANDALONE: every position takes a table row, chosen by COV['rowmap'] -- its own row if the
    token was covered at fit, else its output-NN neighbour's. No native output is ever used."""
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        sub = tbl[COV['rowmap'][STATE['idx']].reshape(-1)].reshape(y.shape).to(y.dtype)
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
def ce_both(rows, hooks=(), swap_embedding=False):
    """`swap_embedding=True` replaces the uncovered token ID ITSELF by its neighbour before the
    forward, so the residual stream starts from the NEIGHBOUR's embedding rather than the true
    token's. The table rows are unchanged -- rowmap[neighbour] is the neighbour's own row -- so this
    isolates exactly one channel: what the residual stream carries in from position j's own token."""
    acc = {'cov': [0.0, 0], 'all': [0.0, 0]}
    for i in range(0, rows.shape[0], 8):
        bb = rows[i:i + 8]
        idx0 = bb[:, :-1].to(DEV).contiguous()
        idx = COV['subst'][idx0] if swap_embedding else idx0
        lg = forward_logits(idx, hooks)
        tg = bb[:, 1:].to(DEV)
        e = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                            reduction='none').reshape(tg.shape)[:, 64:].double()
        c = COV['seen'][idx0[:, 64:]]
        acc['cov'][0] += float(e[c].sum()); acc['cov'][1] += int(c.sum())
        acc['all'][0] += float(e.sum()); acc['all'][1] += int(e.numel())
    return {k: acc[k][0] / acc[k][1] for k in acc}


@torch.no_grad()
def ce_direct(rows, lp_cov):
    """No forward at all: the prediction IS the length-1 logit row of the token (or its neighbour)."""
    acc = {'cov': [0.0, 0], 'all': [0.0, 0]}
    for i in range(0, rows.shape[0], 8):
        bb = rows[i:i + 8]
        idx = bb[:, :-1].to(DEV)[:, 64:]
        tg = bb[:, 1:].to(DEV)[:, 64:]
        r = COV['rowmap'][idx]
        v = -lp_cov[r].gather(-1, tg.unsqueeze(-1)).squeeze(-1).double()
        c = COV['seen'][idx]
        acc['cov'][0] += float(v[c].sum()); acc['cov'][1] += int(c.sum())
        acc['all'][0] += float(v.sum()); acc['all'][1] += int(v.numel())
    return {k: acc[k][0] / acc[k][1] for k in acc}


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
    tk = toks.to(DEV)
    sites = [(k, L) for k in ('mlp', 'attn') for L in range(18)]
    print(f'DIRECT LOOKUP vs COMPOSED PROGRAM | ranks {RANKS} | output-NN fallback (§1780/§1781) | '
          f'DISCOVERY ONLY', flush=True)

    # length-1 log-probs for the covered tokens: the direct arm's whole program
    lp_cov = torch.zeros(ncov, W, device=DEV)
    for i in range(0, ncov, 256):
        t = tk[i:i + 256].unsqueeze(1)
        lp_cov[i:i + t.shape[0]] = torch.log_softmax(forward_logits(t)[:, 0].float(), -1)
    pc = torch.softmax(lp_cov, -1)
    pc = (pc / pc.norm(dim=-1, keepdim=True).clamp_min(1e-9)).half()

    # the settled fallback: uncovered -> covered token with the nearest length-1 OUTPUT distribution
    rowmap = torch.zeros(V, dtype=torch.long, device=DEV)
    rowmap[tk] = torch.arange(ncov, device=DEV)
    unc = (~seen).nonzero(as_tuple=True)[0]
    for s0 in range(0, unc.numel(), 512):
        u = unc[s0:s0 + 512]
        lg = forward_logits(u.unsqueeze(1))[:, 0].float()
        p = torch.softmax(lg, -1)
        p = p / p.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        rowmap[u] = (p.half() @ pc.T).float().argmax(-1)
    COV['rowmap'] = rowmap
    # the token each position PRESENTS to the embedding under arm B: itself if covered, else its
    # neighbour's id
    subst = torch.arange(V, device=DEV)
    subst[unc] = tk[rowmap[unc]]
    COV['subst'] = subst
    del pc
    torch.cuda.empty_cache()
    print(f'  built the length-1 lookup and the output-NN map for {unc.numel()} uncovered ids '
          f'({time.time() - t0:.0f}s)', flush=True)

    # the composed program's 36 context-free site tables, compact [ncov, D]
    tables = {st: torch.zeros(ncov, D, device=DEV) for st in sites}
    cap = {}

    def mk(st):
        def hook(mod, args, out):
            cap[st] = (out[0] if isinstance(out, tuple) else out)[:, 0].float()
            return None
        return hook
    for i in range(0, ncov, 256):
        t = tk[i:i + 256].unsqueeze(1)
        forward_logits(t, [(st, mk(st)) for st in sites])
        for st in sites:
            tables[st][i:i + t.shape[0]] = cap[st]
    print(f'  built 36 context-free tables ({time.time() - t0:.0f}s)', flush=True)

    def truncate(r):
        if r is None:
            return tables, 36 * (NCOV * D + D)
        out = {}
        for st, tbl in tables.items():
            b = tbl.double()
            mu = b.mean(0, keepdim=True)
            U, S, Vh = torch.linalg.svd(b - mu, full_matrices=False)
            out[st] = (mu + (U[:, :r] * S[:r]) @ Vh[:r]).float()
        return out, 36 * (r * (NCOV + D) + 2 * D)

    res = {}
    for ename, epath, ref in EVAL_SETS:
        ev = load(epath)
        live = ce_both(ev)
        if ref is not None:
            assert abs(live['cov'] - ref) <= 1e-3, f'{ename} live cov {live["cov"]:.5f} != {ref}'
        d = ce_direct(ev, lp_cov)
        row = {'live': {k: round(v, 5) for k, v in live.items()},
               'direct': {k: round(v, 5) for k, v in d.items()},
               'direct_cost_M': round(NCOV * W / 1e6, 3)}
        for r in RANKS:
            tr, cost = truncate(r)
            hk = [(st, table_hook(tr[st])) for st in sites]
            key = 'full' if r is None else str(r)
            row[f'composed_{key}'] = {**{k: round(v, 5) for k, v in ce_both(ev, hk).items()},
                                      'cost_M': round(cost / 1e6, 4)}
            row[f'swapped_{key}'] = {k: round(v, 5)
                                     for k, v in ce_both(ev, hk, swap_embedding=True).items()}
            if r is not None:
                del tr
                torch.cuda.empty_cache()
        res[ename] = row
        print(f'\n  {ename}: live cov {live["cov"]:.5f} all {live["all"]:.5f}', flush=True)
        print(f'    DIRECT lookup      cov {d["cov"]:.5f}  all {d["all"]:.5f}  '
              f'({row["direct_cost_M"]:.1f}M reals)', flush=True)
        for r in RANKS:
            key = 'full' if r is None else str(r)
            c, sw = row[f'composed_{key}'], row[f'swapped_{key}']
            print(f'    COMPOSED rank {key:5s} cov {c["cov"]:.5f}  all {c["all"]:.5f}   | '
                  f'EMBEDDING SWAPPED cov {sw["cov"]:.5f}  all {sw["all"]:.5f}', flush=True)
        del ev
        torch.cuda.empty_cache()

    roles = [e for e, _, _ in EVAL_SETS]
    pa = all(abs(res[e]['swapped_full']['all'] - res[e]['direct']['all']) <= 0.02 for e in roles)
    pb = all(res[e]['swapped_full']['all'] - res[e]['composed_full']['all'] >= 0.10 for e in roles)
    pc = all(abs(res[e]['swapped_full']['cov'] - res[e]['composed_full']['cov']) <= 1e-9
             for e in roles)
    pd = (all(abs(res[e]['composed_full'][k] - v) <= 0.002
              for e, kv in S1782_COMPOSED.items() for k, v in kv.items())
          and all(abs(res[e]['direct']['all'] - v) <= 0.002 for e, v in S1782_DIRECT.items())
          and ncov == NCOV)

    print(f'\n  swapping the embedding lands within 0.02 of the direct lookup -> the embedding '
          f'channel explains the gap {pa}', flush=True)
    print(f'    ' + '  '.join(
        f'{e} swapped {res[e]["swapped_full"]["all"]:.5f} vs direct {res[e]["direct"]["all"]:.5f}'
        for e in roles), flush=True)
    print(f'  swapping costs >=0.10 against the true-embedding program -> {pb}  ' + '  '.join(
        f'{e} {res[e]["swapped_full"]["all"] - res[e]["composed_full"]["all"]:+.5f}' for e in roles),
        flush=True)
    print(f'  covered CE untouched by the swap -> {pc}', flush=True)
    print(f'  composed and direct arms reproduce §1782 + coverage {ncov} -> control {pd}',
          flush=True)

    r2 = {'config': {'ranks': [str(x) for x in RANKS],
                     'fallback': 'output-NN: the uncovered token takes the covered token whose '
                                 'length-1 next-token distribution is most similar, one neighbour '
                                 '(§1780/§1781)',
                     'direct': "the prediction IS the token's (or its neighbour's) length-1 logit "
                               'row; no forward through the substituted model at all',
                     'composed': 'the 36 context-free site tables installed as hooks, standalone',
                     'cost': f'direct stores {NCOV} x {W} = {round(NCOV * W / 1e6, 1)}M reals; '
                             'composed stores 224.778M at full rank and 15.223M at rank 64. This is '
                             'a FIDELITY comparison; the direct arm is not a frontier point.',
                     'WHY': "§1781 compared a covered-position reference against an all-position "
                            'program and was amended. This builds the all-position reference.',
                     'ROLE_NOTE': 'DISCOVERY ONLY.'},
          'results': res,
          'predictions': {'pred_a_embedding_channel_explains_the_gap': bool(pa),
                          'pred_b_swap_costs_at_least_0p10': bool(pb),
                          'pred_c_covered_untouched': bool(pc),
                          'pred_d_controls': bool(pd)},
          'runtime_s': round(time.time() - t0, 1)}
    json.dump(r2, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r2["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
