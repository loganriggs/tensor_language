# PREDICT THE UNCOVERED TOKEN'S ROWS INSTEAD OF COPYING A NEIGHBOUR'S
#
# §1783 decomposed the standalone position-wise program exactly. At a COVERED position it is the
# length-1 model, bit-identical. At an UNCOVERED position it is the true token's embedding pushed
# through the OUTPUT-NN neighbour's 36 site writes -- worth 0.15 nats over the neighbour's own output,
# because the embedding channel survives.
#
# But at an uncovered position we know more about the token than which covered token it resembles: we
# know its embedding exactly. Copying a neighbour's row discards that. A LEARNED MAP from embedding to
# site row -- fitted on the 5,419 covered tokens, where both sides are known -- uses it, generalises
# rather than copies, and is still a function of the current token alone, so the program stays
# position-wise.
#
# Rank-64 maps, one per site, ridge-fitted on the covered tokens: 36 x 64 x (1152 + 1152) = 5.308M
# reals on top of the tables. Applied ONLY at uncovered positions; covered tokens keep their exact
# rows, so the covered half of the program is untouched by construction.
#
# ROLES. skip7000, skip11000, skip1200; covered and all-position CE. DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, absolute nats with margins per LESSON 40, each
# read back against its own sentence per LESSON 39:
#   pred_a THE LEARNED MAP BEATS THE NEIGHBOUR by more than 0.005 nats on all-position CE, at every
#          rank and role. If FALSE, copying the nearest covered token's row is better than predicting
#          the row from the embedding -- which would say the site-row manifold is not linearly
#          reachable from the embedding, and would make the neighbour copy the right design after all.
#   pred_b IT DOES NOT REACH THE COVERED-POSITION LEVEL: even with the learned map the all-position CE
#          stays above the covered-position CE at every role. An uncovered token is still a token the
#          tables never saw; if this FAILS the fallback is fully solved and the covered/uncovered
#          distinction stops mattering for this program class.
#   pred_c COVERED CE IS UNTOUCHED to 1e-9 at every role and rank -- the map is applied only where the
#          token was uncovered. A wiring check.
#   pred_d CONTROLS: the neighbour arm reproduces §1782/§1783's all-position 6.01897 / 6.00091 /
#          6.00733 at full rank within 0.002, its covered 6.03465 / 5.97900 / 5.96423, and coverage is
#          exactly 5419 of 50257.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256; V = 50257; W = 50304
RANKS = (None, 64)
MAP_RANK = 64
RIDGE = 1e-2
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/learned_row_from_embedding_results.json'
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711),
             ('skip1200', PT + '.rowcache/fineweb_n96_skip1200.pt', None)]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
NCOV = 5419
S1783 = {'skip7000': {'all': 6.01897, 'cov': 6.03465},
         'skip11000': {'all': 6.00091, 'cov': 5.97900},
         'skip1200': {'all': 6.00733, 'cov': 5.96423}}
STATE = {}
COV = {}


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


def mod_of(kind, L):
    return H[L].mlp if kind == 'mlp' else H[L].attn


def row_hook(full_rows):
    """`full_rows` is [V, D]: every token id's site row, already resolved by whichever fallback the
    arm uses. Standalone -- no native output is ever consulted."""
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        sub = full_rows[STATE['idx'].reshape(-1)].reshape(y.shape).to(y.dtype)
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
    unc = (~seen).nonzero(as_tuple=True)[0]
    sites = [(k, L) for k in ('mlp', 'attn') for L in range(18)]
    print(f'LEARNED ROW FROM EMBEDDING | map rank {MAP_RANK} | table ranks {RANKS} | '
          f'DISCOVERY ONLY', flush=True)

    # the settled output-NN map (§1780/§1781), for the baseline arm
    lp = torch.zeros(ncov, W, device=DEV)
    for i in range(0, ncov, 256):
        t = tk[i:i + 256].unsqueeze(1)
        lp[i:i + t.shape[0]] = torch.log_softmax(forward_logits(t)[:, 0].float(), -1)
    pcn = torch.softmax(lp, -1)
    pcn = (pcn / pcn.norm(dim=-1, keepdim=True).clamp_min(1e-9)).half()
    del lp
    nnrow = torch.zeros(V, dtype=torch.long, device=DEV)
    nnrow[tk] = torch.arange(ncov, device=DEV)
    for s0 in range(0, unc.numel(), 512):
        u = unc[s0:s0 + 512]
        p = torch.softmax(forward_logits(u.unsqueeze(1))[:, 0].float(), -1)
        p = p / p.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        nnrow[u] = (p.half() @ pcn.T).float().argmax(-1)
    del pcn
    torch.cuda.empty_cache()

    # the 36 context-free site tables on the covered tokens
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
    print(f'  built the output-NN map and 36 tables ({time.time() - t0:.0f}s)', flush=True)

    # ridge fit: embedding -> site row, on the COVERED tokens only, then rank-truncated
    Ecov = m.transformer.wte.weight.detach()[tk].float().double()
    A = Ecov.T @ Ecov + RIDGE * torch.eye(D, device=DEV, dtype=torch.float64) * (ncov / D)
    Eunc = m.transformer.wte.weight.detach()[unc].float().double()
    maps = {}
    for st in sites:
        Ws = torch.linalg.solve(A, Ecov.T @ tables[st].double())
        U, S, Vh = torch.linalg.svd(Ws, full_matrices=False)
        maps[st] = ((U[:, :MAP_RANK] * S[:MAP_RANK]) @ Vh[:MAP_RANK])
    map_cost = 36 * MAP_RANK * 2 * D
    print(f'  fitted 36 rank-{MAP_RANK} embedding->row maps ({map_cost / 1e6:.3f}M reals, '
          f'{time.time() - t0:.0f}s)', flush=True)

    def build_full(tbl_c, mode):
        """[V, D] rows: covered tokens keep their exact row; uncovered get the neighbour's row or the
        learned prediction from their own embedding."""
        out = {}
        for st in sites:
            fr = torch.zeros(V, D, device=DEV)
            fr[tk] = tbl_c[st]
            if mode == 'neighbour':
                fr[unc] = tbl_c[st][nnrow[unc]]
            else:
                fr[unc] = (Eunc @ maps[st]).float()
            out[st] = fr
        return out

    def truncate(r):
        if r is None:
            return tables, 36 * (NCOV * D + D)
        o = {}
        for st, tbl in tables.items():
            b = tbl.double()
            mu = b.mean(0, keepdim=True)
            U, S, Vh = torch.linalg.svd(b - mu, full_matrices=False)
            o[st] = (mu + (U[:, :r] * S[:r]) @ Vh[:r]).float()
        return o, 36 * (r * (NCOV + D) + 2 * D)

    res = {}
    for ename, epath, ref in EVAL_SETS:
        ev = load(epath)
        live = ce_both(ev)
        if ref is not None:
            assert abs(live['cov'] - ref) <= 1e-3, f'{ename} live cov {live["cov"]:.5f} != {ref}'
        row = {'live': {k: round(v, 5) for k, v in live.items()}}
        for r in RANKS:
            tc, cost = truncate(r)
            key = 'full' if r is None else str(r)
            for mode in ('neighbour', 'learned'):
                fr = build_full(tc, mode)
                c1 = ce_both(ev, [(st, row_hook(fr[st])) for st in sites])
                row[f'{mode}_{key}'] = {**{k: round(v, 5) for k, v in c1.items()},
                                        'cost_M': round((cost + (map_cost if mode == 'learned'
                                                                else 0)) / 1e6, 4)}
                del fr
                torch.cuda.empty_cache()
            if r is not None:
                del tc
        res[ename] = row
        print(f'\n  {ename}: live cov {live["cov"]:.5f} all {live["all"]:.5f}', flush=True)
        for r in RANKS:
            key = 'full' if r is None else str(r)
            n, l = row[f'neighbour_{key}'], row[f'learned_{key}']
            print(f'    rank {key:5s}  neighbour cov {n["cov"]:.5f} all {n["all"]:.5f} '
                  f'({n["cost_M"]:.3f}M) | learned cov {l["cov"]:.5f} all {l["all"]:.5f} '
                  f'({l["cost_M"]:.3f}M)', flush=True)
        del ev
        torch.cuda.empty_cache()

    roles = [e for e, _, _ in EVAL_SETS]
    keys = ['full' if r is None else str(r) for r in RANKS]
    pa = all(res[e][f'learned_{k}']['all'] < res[e][f'neighbour_{k}']['all'] - 0.005
             for e in roles for k in keys)
    pb = all(res[e][f'learned_{k}']['all'] > res[e][f'learned_{k}']['cov'] for e in roles
             for k in keys)
    pc = all(abs(res[e][f'learned_{k}']['cov'] - res[e][f'neighbour_{k}']['cov']) <= 1e-9
             for e in roles for k in keys)
    pd = (all(abs(res[e]['neighbour_full'][k] - v) <= 0.002
              for e, kv in S1783.items() for k, v in kv.items()) and ncov == NCOV)

    print(f'\n  the learned map beats the neighbour by >0.005 everywhere -> {pa}', flush=True)
    print(f'    margins ' + '  '.join(
        f'{e}/{k} {res[e][f"neighbour_{k}"]["all"] - res[e][f"learned_{k}"]["all"]:+.5f}'
        for e in roles for k in keys), flush=True)
    print(f'  all-position stays above covered -> the fallback is not solved {pb}', flush=True)
    print(f'  covered CE untouched by the fallback choice -> {pc}', flush=True)
    print(f'  the neighbour arm reproduces §1782/§1783 + coverage {ncov} -> control {pd}', flush=True)

    r2 = {'config': {'map_rank': MAP_RANK, 'ridge': RIDGE, 'table_ranks': keys,
                     'learned': 'ridge map from the token embedding to the site row, fitted on the '
                                '5419 COVERED tokens and applied only at uncovered ones; still a '
                                'function of the current token alone',
                     'map_cost_M': round(map_cost / 1e6, 4),
                     'neighbour': 'the settled output-NN fallback (§1780/§1781)',
                     'ROLE_NOTE': 'DISCOVERY ONLY.'},
          'results': res,
          'predictions': {'pred_a_learned_beats_neighbour': bool(pa),
                          'pred_b_fallback_not_solved': bool(pb),
                          'pred_c_covered_untouched': bool(pc),
                          'pred_d_controls': bool(pd)},
          'runtime_s': round(time.time() - t0, 1)}
    json.dump(r2, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r2["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
