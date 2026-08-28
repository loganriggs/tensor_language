# THE RANK-CONSISTENT REPAIR OF §1784
#
# §1784 fitted the embedding->row maps once against the FULL-rank context-free tables and then reused
# them unchanged when the covered rows were truncated to rank 64. That arm therefore had TRUNCATED
# covered rows and UNTRUNCATED predicted rows -- not a coherent rank-64 program, and its cost figure
# did not describe one. The tell was in the output: the learned arm's uncovered-only CE was identical
# between table ranks to five decimals, because that half never saw the truncation.
#
# The repair is one line of structure: refit the map inside each truncated basis, so the predicted
# rows live in the same rank-r space as the copied ones.
#
# §1784's FULL-rank arm was already consistent, so it is the control: refitting per rank must leave it
# exactly where it was.
#
# ROLES. skip7000, skip11000, skip1200; covered, all-position, and the backed-out uncovered-only CE.
# DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, absolute nats with margins per LESSON 40, read
# back per LESSON 39:
#   pred_a THE REPAIRED ARM CLEARS THE BAR §1784 FAILED: the learned map beats the neighbour by more
#          than 0.005 nats at every rank and role. §1784 missed on skip1200 at rank 64 by -0.00091,
#          in the arm that was not coherent. If it FAILS again, the advantage is genuinely marginal
#          and role-dependent and the neighbour copy stands as the design.
#   pred_b THE ARM IS ACTUALLY RANK-CONSISTENT NOW: the rank-64 learned arm's uncovered-only CE
#          differs from the full-rank one by more than 0.005 at every role. In §1784 that difference
#          was under 3e-5, which is what exposed the defect. If this FAILS the refit did not take and
#          the repair is not in force -- a wiring check with teeth rather than a formality.
#   pred_c COVERED CE IS UNTOUCHED by the fallback choice, to 1e-9, at every rank and role.
#   pred_d CONTROLS: the neighbour arms reproduce §1782/§1783's three all-position and three covered
#          numbers within 0.002, the FULL-rank learned arm reproduces §1784's 6.01167 / 5.98477 /
#          6.00165 within 0.002 -- it was already coherent and must not move -- and coverage is 5419.
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
OUT = PT + 'ops/learned_row_rank_consistent_results.json'
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711),
             ('skip1200', PT + '.rowcache/fineweb_n96_skip1200.pt', None)]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
NCOV = 5419
S1783 = {'skip7000': {'all': 6.01897, 'cov': 6.03465},
         'skip11000': {'all': 6.00091, 'cov': 5.97900},
         'skip1200': {'all': 6.00733, 'cov': 5.96423}}
# §1784's FULL-rank learned arm was already rank-consistent, so refitting per rank must not move it
S1784_LEARNED_FULL = {'skip7000': 6.01167, 'skip11000': 5.98477, 'skip1200': 6.00165}
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
    map_cost = 36 * MAP_RANK * 2 * D

    def fit_maps(tbl_c):
        """Fit the embedding->row map against THESE rows. §1784 fitted once against the FULL-rank
        tables and reused the maps at rank 64, so its rank-64 arm had truncated covered rows and
        untruncated predicted ones -- not a coherent program. Refitting inside each basis is the
        repair."""
        mp = {}
        for st in sites:
            Ws = torch.linalg.solve(A, Ecov.T @ tbl_c[st].double())
            U, S, Vh = torch.linalg.svd(Ws, full_matrices=False)
            mp[st] = ((U[:, :MAP_RANK] * S[:MAP_RANK]) @ Vh[:MAP_RANK])
        return mp

    def build_full(tbl_c, mode, maps):
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
            maps = fit_maps(tc)
            for mode in ('neighbour', 'learned'):
                fr = build_full(tc, mode, maps)
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
    N = {'skip7000': (27974, 36864), 'skip11000': (27497, 36864), 'skip1200': (13973, 18432)}

    def unc_ce(e, arm):
        nc, na = N[e]
        a, c = res[e][arm]['all'], res[e][arm]['cov']
        return (na * a - nc * c) / (na - nc)

    for e in roles:
        for k in keys:
            for mode in ('neighbour', 'learned'):
                res[e][f'{mode}_{k}']['uncovered_ce'] = round(unc_ce(e, f'{mode}_{k}'), 5)
    pa = all(res[e][f'learned_{k}']['all'] < res[e][f'neighbour_{k}']['all'] - 0.005
             for e in roles for k in keys)
    pb = all(abs(unc_ce(e, 'learned_64') - unc_ce(e, 'learned_full')) > 0.005 for e in roles)
    pc = all(abs(res[e][f'learned_{k}']['cov'] - res[e][f'neighbour_{k}']['cov']) <= 1e-9
             for e in roles for k in keys)
    pd = (all(abs(res[e]['neighbour_full'][k] - v) <= 0.002
              for e, kv in S1783.items() for k, v in kv.items())
          and all(abs(res[e]['learned_full']['all'] - v) <= 0.002
                  for e, v in S1784_LEARNED_FULL.items()) and ncov == NCOV)

    print(f'\n  the learned map beats the neighbour by >0.005 everywhere -> {pa}', flush=True)
    print(f'    margins ' + '  '.join(
        f'{e}/{k} {res[e][f"neighbour_{k}"]["all"] - res[e][f"learned_{k}"]["all"]:+.5f}'
        for e in roles for k in keys), flush=True)
    print(f'  the rank-64 learned arm now DIFFERS from full rank on uncovered-only CE -> '
          f'rank-consistent {pb}  ' + '  '.join(
              f'{e} {abs(unc_ce(e, "learned_64") - unc_ce(e, "learned_full")):.5f}' for e in roles),
          flush=True)
    print(f'  uncovered-only CE: ' + ' | '.join(
        f'{e} nb {unc_ce(e, "neighbour_full"):.5f} lr {unc_ce(e, "learned_full"):.5f}'
        for e in roles), flush=True)
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
          'predictions': {'pred_a_learned_beats_neighbour_everywhere': bool(pa),
                          'pred_b_rank64_arm_is_now_rank_consistent': bool(pb),
                          'pred_c_covered_untouched': bool(pc),
                          'pred_d_controls': bool(pd)},
          'runtime_s': round(time.time() - t0, 1)}
    json.dump(r2, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r2["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
