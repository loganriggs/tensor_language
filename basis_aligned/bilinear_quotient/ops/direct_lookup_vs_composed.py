# THE HONEST ALL-POSITION REFERENCE: a direct logit lookup against the composed 36-site program
#
# §1781 originally claimed the full-rank standalone program is "within 0.021 nats of the position-wise
# reference on all positions". That compared DIFFERENT POPULATIONS -- 5.97902 is a covered-position
# number and 6.00048 is an all-position one -- and it was withdrawn and amended on re-reading. The
# honest all-position reference has to be BUILT, and building it is a real experiment rather than a
# lookup, for a reason that only shows up at the uncovered quarter:
#
#   at a COVERED position the two are the same object. §1769 measured the composed 36-site
#   context-free program against the length-1 model at -0.00002 nats, so a direct lookup of that
#   token's length-1 logits must agree with the program there.
#   at an UNCOVERED position they diverge. The composed program takes the NEIGHBOUR's 36 SITE ROWS and
#   pushes them through eighteen blocks; a direct lookup takes the NEIGHBOUR's LOGITS. Composing the
#   neighbour's rows is not the neighbour's output, and nothing says which is the better stand-in.
#
# Both arms are position-wise, standalone, and use the settled fallback (§1780/§1781): the uncovered
# token is sent to the covered token whose LENGTH-1 NEXT-TOKEN DISTRIBUTION is most similar, exactly
# one neighbour.
#
# COST, stated because it decides how to read the result. DIRECT stores 5419 x 50304 = 272.6M reals;
# COMPOSED at full rank stores 224.8M and at rank 64 stores 15.2M. So this is a FIDELITY comparison
# and the direct arm is not a frontier point.
#
# ROLES. skip7000, skip11000, skip1200; covered and all-position CE in one pass. DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, absolute nats with margins per LESSON 40, each
# read back against its own sentence per LESSON 39:
#   pred_a THE DIRECT LOOKUP WINS ON ALL POSITIONS at every role, by more than 0.005 nats (lower CE
#          is better). It has one fewer approximation step at the uncovered quarter. If FALSE, then
#          composing the neighbour's site rows through eighteen blocks is BETTER than taking the
#          neighbour's logits -- the composition repairs something about the substitution -- and that
#          is the more interesting outcome of the two.
#   pred_b THEY AGREE ON COVERED POSITIONS to within 0.001 nats at every role, extending §1769's
#          identity. This is a wiring check: if it fails, the two arms are not the same object where
#          they must be, and neither pred_a nor pred_c means anything.
#   pred_c §1781'S WITHDRAWN SENTENCE UNDERSTATED THE GAP: the direct all-position reference is BELOW
#          6.00048, the composed full-rank program's all-position CE. Scored independently of pred_a,
#          which is about every role and a margin; this is the single comparison the withdrawn
#          sentence actually got wrong. If FALSE the composed program is at or better than the honest
#          reference, and that sentence was -- by luck rather than by care -- not an overclaim.
#   pred_d CONTROLS: the composed full-rank arm reproduces §1780's all-position 6.00048 and
#          covered 5.97900 on skip11000 within 0.002, and coverage is exactly 5419 of 50257.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256; V = 50257; W = 50304
RANKS = (None, 64)
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/direct_lookup_vs_composed_results.json'
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711),
             ('skip1200', PT + '.rowcache/fineweb_n96_skip1200.pt', None)]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
NCOV = 5419
S1780_COMPOSED_FULL = {'all': 6.00048, 'cov': 5.97900}   # skip11000, output-NN fallback
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
def ce_both(rows, hooks=()):
    acc = {'cov': [0.0, 0], 'all': [0.0, 0]}
    for i in range(0, rows.shape[0], 8):
        bb = rows[i:i + 8]
        idx = bb[:, :-1].to(DEV).contiguous()
        lg = forward_logits(idx, hooks)
        tg = bb[:, 1:].to(DEV)
        e = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                            reduction='none').reshape(tg.shape)[:, 64:].double()
        c = COV['seen'][idx[:, 64:]]
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
            c1 = ce_both(ev, [(st, table_hook(tr[st])) for st in sites])
            key = 'full' if r is None else str(r)
            row[f'composed_{key}'] = {**{k: round(v, 5) for k, v in c1.items()},
                                      'cost_M': round(cost / 1e6, 4)}
            if r is not None:
                del tr
                torch.cuda.empty_cache()
        res[ename] = row
        print(f'\n  {ename}: live cov {live["cov"]:.5f} all {live["all"]:.5f}', flush=True)
        print(f'    DIRECT lookup      cov {d["cov"]:.5f}  all {d["all"]:.5f}  '
              f'({row["direct_cost_M"]:.1f}M reals)', flush=True)
        for r in RANKS:
            key = 'full' if r is None else str(r)
            c = row[f'composed_{key}']
            print(f'    COMPOSED rank {key:5s} cov {c["cov"]:.5f}  all {c["all"]:.5f}  '
                  f'({c["cost_M"]:.3f}M reals)', flush=True)
        del ev
        torch.cuda.empty_cache()

    roles = [e for e, _, _ in EVAL_SETS]
    pa = all(res[e]['direct']['all'] < res[e]['composed_full']['all'] - 0.005 for e in roles)
    pb = all(abs(res[e]['direct']['cov'] - res[e]['composed_full']['cov']) <= 0.001 for e in roles)
    pc = res['skip11000']['direct']['all'] < S1780_COMPOSED_FULL['all']
    pd = (abs(res['skip11000']['composed_full']['all'] - S1780_COMPOSED_FULL['all']) <= 0.002
          and abs(res['skip11000']['composed_full']['cov'] - S1780_COMPOSED_FULL['cov']) <= 0.002
          and ncov == NCOV)

    print(f'\n  DIRECT beats COMPOSED on all positions by >0.005 at every role -> {pa}', flush=True)
    print(f'    margins ' + '  '.join(
        f'{e} {res[e]["composed_full"]["all"] - res[e]["direct"]["all"]:+.5f}' for e in roles),
        flush=True)
    print(f'  they agree on COVERED positions within 0.001 -> {pb}  ' + '  '.join(
        f'{e} {abs(res[e]["direct"]["cov"] - res[e]["composed_full"]["cov"]):.2e}' for e in roles),
        flush=True)
    print(f'  the honest all-position reference {res["skip11000"]["direct"]["all"]:.5f} is below '
          f'§1780\'s composed {S1780_COMPOSED_FULL["all"]} -> {pc}', flush=True)
    print(f'  composed full-rank reproduces §1780 + coverage {ncov} -> control {pd}', flush=True)

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
          'predictions': {'pred_a_direct_wins_all_positions': bool(pa),
                          'pred_b_agree_on_covered': bool(pb),
                          'pred_c_reference_below_composed': bool(pc),
                          'pred_d_controls': bool(pd)},
          'runtime_s': round(time.time() - t0, 1)}
    json.dump(r2, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r2["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
