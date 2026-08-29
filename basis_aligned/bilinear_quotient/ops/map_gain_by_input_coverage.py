# WHERE DOES THE MAP ACT? -- cross-tabulating §1935's gain by the INPUT token's coverage.
#
# §1935 found the fallback map's 64 -> 512 gain is BROAD: all four rarer target buckets gain on 3/3 roles
# (12/12 cells), only the 125+ bucket loses. That left an open question I flagged in the section itself --
# the 1-4 through 25-124 buckets are mostly COVERED targets, so the map appears to be doing something
# beyond serving the uncovered arm.
#
# It cannot be. The map supplies rows only for tokens with no table entry, and §1765 makes the compiled
# program a pure function of the CURRENT token, so at any position whose INPUT token is one of the 5,419
# covered types the map is never consulted and changing its rank cannot move the prediction AT ALL. The
# bucket axis is the TARGET's frequency, not the input's -- so a frequent target reached from an uncovered
# input sits in the 125+ bucket and is fully exposed to the map. That is the resolution, if it is right,
# and nothing in the record has ever partitioned this thread's scoring by INPUT coverage.
#
# This measures it: two arms (map rank 64 = §1789's deployed design, and 512), full table rank, 5,419
# coverage, every scored position cross-tabulated by (input token covered vs uncovered) x (target bucket).
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY, 5,419 coverage. Rung 3 -- §1935's open question.
#
# Registered predictions, SIGNED and one-sided per LESSON 72; pred_a is an EXACT-ZERO bar, not a tolerance.
#   pred_a THE MAP IS INERT AT COVERED INPUTS: summed over all three roles, the number of scored positions
#          whose input token is one of the 5,419 covered types and whose top-1 DIFFERS between the rank-64
#          and rank-512 arms is EXACTLY 0. Not "small" -- zero. If FALSE, either §1765 does not hold where
#          I have been assuming it or the map is leaking into covered rows, and every map result from
#          §1870 onward needs re-reading. This is the mechanism claim and it is the point of the run.
#   pred_b SO THE WHOLE GAIN LIVES ON UNCOVERED INPUTS: restricted to uncovered-input positions, the
#          64 -> 512 kept-fraction change is POSITIVE in all four rarer buckets, on at least 2 of 3 roles.
#          If FALSE the breadth §1935 found is not simply the uncovered arm seen through a target-frequency
#          lens and §1935's post-hoc paragraph is wrong about what it was looking at.
#   pred_c AND THE COMMON-TARGET DAMAGE IS SEVERE WHERE IT ACTS: on uncovered-input positions the 125+
#          kept-fraction FALLS by at least 1.5pp, on at least 2 of 3 roles. Pooled, §1935 measured only
#          -0.4 / -0.8 / -0.6pp; if the loss is confined to ~a quarter of positions it must be several
#          times larger there. Deployment cares: a build could keep rank-64 rows for uncovered inputs and
#          spend the rank only where it pays. If FALSE the pooled figure is already the true size and the
#          loss is spread more thinly than pred_a would imply.
#   pred_d CONTROLS: coverage is exactly 5,419; the covered and uncovered input counts sum to the scored
#          total in every bucket; recombining the two input classes reproduces §1935's PUBLISHED pooled
#          kept-fractions -- 125+ 63.5/62.9/63.4 -> 63.1/62.1/62.8 and unseen 2.7/6.2/3.6 -> 4.0/6.7/4.0 --
#          within 0.1pp; and the LIVE per-cell accuracy is identical across both arms.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256; V = 50257; W = 50304
RANKS = ('map64', 'map512')
MAPRANK_OF = {'map64': 64, 'map512': 512}
# UNIFORM ranks only -- the question is about the table axis, not the allocation.
ALLOC = {k9: None for k9 in ('map64', 'map512')}   # FULL table rank throughout
S1932_TOP = {'skip7000': 0.635, 'skip11000': 0.629, 'skip1200': 0.634}   # §1932 deployed @5,419
S1932_BOT = {'skip7000': 0.027, 'skip11000': 0.062, 'skip1200': 0.036}   # §1932 deployed @5,419
S1935_TOP512 = {'skip7000': 0.631, 'skip11000': 0.621, 'skip1200': 0.628}   # §1935 pooled, map512
S1935_BOT512 = {'skip7000': 0.040, 'skip11000': 0.067, 'skip1200': 0.040}   # §1935 pooled, map512

RIDGE = 1e-2
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/map_gain_by_input_coverage_results.json'
BUCKETS = ((0, 0), (1, 4), (5, 24), (25, 124), (125, 10 ** 9))
# live COVERED-CE refs set to None. This lineage runs at 5,419, where 3.29205 / 3.09711 / 3.40277
# WOULD be the right constants -- but they are left None anyway so that a fork to another coverage
# cannot inherit the population-dependence trap that cost §1930 a launch and failed §1905's pred_d.
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', None),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', None),
             ('skip1200', PT + '.rowcache/fineweb_n96_skip1200.pt', None)]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'   # 5,419 types at T=256 -- DEPLOYED coverage
H = m.transformer.h
NCOV = 5419       # §1834's deployed coverage
S1789_COV = 5419  # below are AT 5419 and are printed for context, never used as bars (§1882's trap)
S1788_ACC = {'skip7000': {'prog': 0.1355, 'live': 0.3932},
             'skip11000': {'prog': 0.1425, 'live': 0.4235},
             'skip1200': {'prog': 0.1364, 'live': 0.3888}}
STATE = {}
COV = {}


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


def mod_of(kind, L):
    return H[L].mlp if kind == 'mlp' else H[L].attn


def row_hook(full_rows):
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
def collect(rows, hooks):
    """Flat per-position top-1 for one arm (or the live model when hooks is empty).

    Returns one int32 vector over every scored position, in a fixed order, so two arms can be
    compared POSITION BY POSITION rather than only through their aggregates -- which is what an
    exact-zero bar on 'the map never moves a covered-input prediction' requires."""
    out = []
    for i in range(0, rows.shape[0], 8):
        bb = rows[i:i + 8]
        idx = bb[:, :-1].to(DEV).contiguous()
        out.append(forward_logits(idx, hooks)[:, 64:].argmax(-1).reshape(-1).to(torch.int32).cpu())
    return torch.cat(out)


@torch.no_grad()
def axes(rows):
    """The scored positions' fixed metadata, in the SAME order collect() emits.

    tgt   -- the true next token
    icov  -- True where the INPUT token is one of the covered types (the map is inert there)
    """
    tg, ic = [], []
    for i in range(0, rows.shape[0], 8):
        bb = rows[i:i + 8]
        idx = bb[:, :-1].to(DEV).contiguous()
        tg.append(bb[:, 1:].to(DEV)[:, 64:].reshape(-1).cpu())
        ic.append(COV['seen'][idx[:, 64:]].reshape(-1).cpu())
    return torch.cat(tg), torch.cat(ic)


def cells(tgt, icov, live, arm):
    """kept-fraction per (input-coverage class) x (target bucket), plus the pooled class."""
    freq = COV['freq'].cpu()[tgt.long()]
    o = {}
    for cname, cm in (('covered_input', icov), ('uncovered_input', ~icov),
                      ('pooled', torch.ones_like(icov))):
        o[cname] = {}
        for b in BUCKETS:
            msk = cm & (freq >= b[0]) & (freq <= b[1])
            n = int(msk.sum())
            al = float((live[msk] == tgt[msk]).float().mean()) if n else 0.0
            ap = float((arm[msk] == tgt[msk]).float().mean()) if n else 0.0
            o[cname][f'{b[0]}-{b[1]}'] = {'n': n, 'top1_acc_live': al, 'top1_acc_prog': ap,
                                          'kept_fraction': ap / max(al, 1e-9)}
        n = int(cm.sum())
        o[cname]['overall'] = {'n': n,
                               'top1_acc_live': float((live[cm] == tgt[cm]).float().mean()),
                               'top1_acc_prog': float((arm[cm] == tgt[cm]).float().mean())}
    return o


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
    COV['freq'] = torch.bincount(fit[:, 1:T + 1].reshape(-1).long(),
                                 minlength=V).to(DEV)
    print(f'MAP GAIN BY INPUT COVERAGE at {NCOV} | (input covered/uncovered) x (target bucket '
          f'{BUCKETS}) | map rank 64 vs 512, FULL table rank | DISCOVERY ONLY', flush=True)

    # the settled fallback: output-NN neighbour (§1780/§1781)
    lpc = torch.zeros(ncov, W, device=DEV)
    for i in range(0, ncov, 256):
        t = tk[i:i + 256].unsqueeze(1)
        lpc[i:i + t.shape[0]] = torch.log_softmax(forward_logits(t)[:, 0].float(), -1)
    pcn = torch.softmax(lpc, -1)
    pcn = (pcn / pcn.norm(dim=-1, keepdim=True).clamp_min(1e-9)).half()
    del lpc
    nnrow = torch.zeros(V, dtype=torch.long, device=DEV)
    nnrow[tk] = torch.arange(ncov, device=DEV)
    for s0 in range(0, unc.numel(), 512):
        u = unc[s0:s0 + 512]
        p = torch.softmax(forward_logits(u.unsqueeze(1))[:, 0].float(), -1)
        p = p / p.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        nnrow[u] = (p.half() @ pcn.T).float().argmax(-1)
    del pcn
    torch.cuda.empty_cache()

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
    Ecov = m.transformer.wte.weight.detach()[tk].float().double()
    A = Ecov.T @ Ecov + RIDGE * torch.eye(D, device=DEV, dtype=torch.float64) * (ncov / D)
    Eunc = m.transformer.wte.weight.detach()[unc].float().double()
    print(f'  built the settled fallback and 36 tables ({time.time() - t0:.0f}s)', flush=True)

    def program_rows(r):
        a = ALLOC[r]
        if a is None:
            tc = tables
        else:
            tc = {}
            for st, tbl in tables.items():
                rk = a[st[0]]
                b = tbl.double()
                mu = b.mean(0, keepdim=True)
                U, S, Vh = torch.linalg.svd(b - mu, full_matrices=False)
                tc[st] = (mu + (U[:, :rk] * S[:rk]) @ Vh[:rk]).float()
        out = {}
        for st in sites:
            Ws = torch.linalg.solve(A, Ecov.T @ tc[st].double())
            U, S, Vh = torch.linalg.svd(Ws, full_matrices=False)
            mr = MAPRANK_OF[r]
            mp = (U[:, :mr] * S[:mr]) @ Vh[:mr]
            fr = torch.zeros(V, D, device=DEV)
            fr[tk] = tc[st]
            fr[unc] = (Eunc @ mp).float()
            out[st] = fr
        return out

    roles = [e for e, _, _ in EVAL_SETS]
    top, bot = f'{BUCKETS[-1][0]}-{BUCKETS[-1][1]}', f'{BUCKETS[0][0]}-{BUCKETS[0][1]}'
    RARE = [f'{x}-{y}' for x, y in BUCKETS[:-1]]     # the four buckets BELOW 125+

    # per-position predictions, one pass per (arm, role); live collected once.
    P, AX, LV = {}, {}, {}
    for ename, epath, _ce in EVAL_SETS:
        ev = load(epath)
        AX[ename] = axes(ev)
        LV[ename] = collect(ev, ())
        ev = None
        torch.cuda.empty_cache()
    for r in RANKS:
        fr = program_rows(r)
        hooks = [(st, row_hook(fr[st])) for st in sites]
        for ename, epath, _ce in EVAL_SETS:
            ev = load(epath)
            P[(r, ename)] = collect(ev, hooks)
            ev = None
            torch.cuda.empty_cache()
        fr, hooks = None, None
        torch.cuda.empty_cache()
        print(f'  scored map rank {MAPRANK_OF[r]} on {len(roles)} roles '
              f'({time.time() - t0:.0f}s)', flush=True)

    res = {}
    for ename in roles:
        tgt, icov = AX[ename]
        res[ename] = {r: cells(tgt, icov, LV[ename], P[(r, ename)]) for r in RANKS}

    # pred_a -- EXACT zero changed predictions at covered inputs, summed over roles.
    chg_cov = sum(int(((P[('map64', e)] != P[('map512', e)]) & AX[e][1]).sum()) for e in roles)
    chg_unc = sum(int(((P[('map64', e)] != P[('map512', e)]) & ~AX[e][1]).sum()) for e in roles)
    pa = (chg_cov == 0)

    def kf(e, cls, arm, b):
        return res[e][arm][cls][b]['kept_fraction']

    def gain(e, cls, b):
        return kf(e, cls, 'map512', b) - kf(e, cls, 'map64', b)

    # pred_b -- on uncovered inputs, all four rarer buckets gain.
    bpos = sum(1 for e in roles if all(gain(e, 'uncovered_input', b) > 0 for b in RARE))
    pb = bpos >= 2
    # pred_c -- on uncovered inputs the 125+ bucket falls by >= 1.5pp.
    cbad = sum(1 for e in roles if gain(e, 'uncovered_input', top) <= -0.015)
    pc = cbad >= 2

    # pred_d -- partition, pooled reproduction of §1935, live identical across arms.
    partition = all(res[e][r]['covered_input'][b]['n'] + res[e][r]['uncovered_input'][b]['n']
                    == res[e][r]['pooled'][b]['n']
                    for e in roles for r in RANKS for b in RARE + [top])
    livespread = max(abs(res[e][r][c][b]['top1_acc_live'] - res[e]['map64'][c][b]['top1_acc_live'])
                     for e in roles for r in RANKS
                     for c in ('covered_input', 'uncovered_input', 'pooled')
                     for b in RARE + [top])
    repro = max(max(abs(kf(e, 'pooled', 'map64', top) - S1932_TOP[e]),
                    abs(kf(e, 'pooled', 'map64', bot) - S1932_BOT[e]),
                    abs(kf(e, 'pooled', 'map512', top) - S1935_TOP512[e]),
                    abs(kf(e, 'pooled', 'map512', bot) - S1935_BOT512[e])) for e in roles)
    pd = (ncov == NCOV and partition and livespread <= 1e-9 and repro <= 0.001)

    print(f'\n  CHANGED top-1 between the rank-64 and rank-512 arms, summed over {len(roles)} roles:',
          flush=True)
    print(f'    covered   inputs (map inert): {chg_cov}', flush=True)
    print(f'    uncovered inputs (map acts) : {chg_unc}', flush=True)
    for e in roles:
        nu = res[e]['map64']['uncovered_input']['overall']['n']
        npo = res[e]['map64']['pooled']['overall']['n']
        print(f'\n  {e}: {nu}/{npo} = {nu / npo:.1%} of scored positions have an UNCOVERED input',
              flush=True)
        for cls in ('pooled', 'uncovered_input', 'covered_input'):
            print(f'    {cls:16s} ' + '  '.join(
                f'{b:>7s} {kf(e, cls, "map64", b):5.1%}->{kf(e, cls, "map512", b):5.1%} '
                f'({gain(e, cls, b) * 100:+.2f})' for b in RARE + [top]), flush=True)
            print(f'    {"":16s} ' + '  '.join(
                f'{b:>7s} n={res[e]["map64"][cls][b]["n"]:6d}' for b in RARE + [top]), flush=True)

    print(f'\n  the map is INERT at covered inputs: exactly 0 changed -> {pa}  ({chg_cov} changed)',
          flush=True)
    print(f'  the whole gain lives on UNCOVERED inputs, all 4 rare buckets (>=2 roles) -> {pb}  '
          f'{bpos}/3', flush=True)
    print(f'  and the 125+ loss is >=1.5pp THERE (>=2 roles) -> {pc}  {cbad}/3', flush=True)
    print(f'  coverage {ncov}, partitions, LIVE identical, pooled reproduces §1935 (max dev '
          f'{repro * 100:.2f}pp) -> control {pd}', flush=True)

    r2 = {'config': {'arms': list(RANKS), 'map_rank_of_arm': {str(k): v for k, v in MAPRANK_OF.items()},
                     'table_rank': 'FULL in both arms -- only the fallback map differs',
                     'coverage': ncov,
                     'axis': 'every scored position cross-tabulated by (INPUT token covered vs '
                             'uncovered) x (TRUE TARGET fit-row bucket). The input axis is new to '
                             'this thread; §1789 onward bucketed only on the target.',
                     'ROLE_NOTE': 'DISCOVERY ONLY, rung 3 -- §1935 open question.'},
          'results': {e: {r: {c: {b: {k: (round(v, 6) if isinstance(v, float) else v)
                                      for k, v in res[e][r][c][b].items()}
                                  for b in res[e][r][c]} for c in res[e][r]} for r in RANKS}
                      for e in roles},
          'changed_top1': {'covered_input': chg_cov, 'uncovered_input': chg_unc},
          'predictions': {'pred_a_map_inert_at_covered_inputs': bool(pa),
                          'pred_b_gain_on_uncovered_inputs': bool(pb),
                          'pred_c_common_loss_severe_there': bool(pc),
                          'pred_d_controls': bool(pd)},
          'runtime_s': round(time.time() - t0, 1)}
    json.dump(r2, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({time.time() - t0:.1f}s)', flush=True)


main()
