# THE FIRST BUILDABLE HYBRID -- route per INPUT TOKEN on the neighbour's own cosine.
#
# §1938 established that CE and top-1 rank the three fallback forms in EXACTLY REVERSED order, and located
# the cause precisely: on uncovered inputs with an UNSEEN target the neighbour's CE is +1.124/+1.162/
# +1.059 nats worse than the map's, because it emits a real covered token's distribution and that puts
# near-zero mass on a token no fit row contains. Everywhere else the neighbour wins -- 4 of 5 buckets,
# 12/12 cells, +2.54/+1.62/+2.47pp of top-1 on the uncovered arm (§1937).
#
# LESSON 74 forbids the obvious fix: the bucket is a property of the TARGET and the row must be chosen per
# INPUT token. But the neighbour's failure has an INPUT-SIDE signature. A token whose nearest covered
# neighbour is FAR in output-distribution space is exactly the token whose neighbour row will be wrong --
# and that cosine is already computed when the neighbour index is built, and thrown away (LESSON 75, a
# second time in the same lineage).
#
# ARMS. nn (100% neighbour), nn75 / nn50 / nn25 (the top 75 / 50 / 25% of uncovered types BY COSINE take
# the neighbour row, the rest take the rank-64 map row), map64 (0% neighbour = §1789 DEPLOYED), map512.
# The routed arms cost ~5.40M -- the deployed map plus one index -- a 1.7% premium on §1789.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY, 5,419 coverage. Rung 3 -- §1938's open question.
#
# Registered predictions, SIGNED and one-sided per LESSON 72. LESSON 77 applied: pred_b is a SHARE of the
# known deficit, not an absolute floor, because the mechanism predicts CONCENTRATION.
#   pred_a THE HYBRID BEATS BOTH ENDPOINTS ON TOP-1: at least one routed arm's uncovered-arm top-1 exceeds
#          BOTH nn's and map64's, on at least 2 of 3 roles. If FALSE, routing on cosine does not separate
#          the cases and the neighbour's advantage is not concentrated in its close tokens.
#   pred_b AND IT CLOSES MOST OF THE CE HOLE: that same routed arm recovers at least 50% of the neighbour's
#          unseen-bucket CE deficit against map512 (i.e. its uncovered unseen-bucket CE is at least halfway
#          from nn's 9.81/9.59/9.47 down to map512's 8.68/8.42/8.41), on at least 2 of 3 roles. A share of
#          the measured deficit, not a floor.
#   pred_c AND IT STRICTLY DOMINATES THE DEPLOYED DESIGN: at least one routed arm has pooled top-1 ABOVE
#          map64's AND pooled CE BELOW map64's, on at least 2 of 3 roles. This is the deployment claim --
#          better on BOTH instruments at once, for +0.09M on a 230.087M build. If FALSE, §1938's reversal
#          is a genuine fork and no cheap hybrid escapes it, which is the more likely outcome and the one
#          worth stating plainly rather than hunting for.
#   pred_d CONTROLS: coverage exactly 5,419; the nn and map64 arms reproduce §1938's PUBLISHED uncovered
#          top-1 (14.68/14.25/13.90 and 12.14/12.63/11.44%) within 0.05pp and pooled CE (6.01897/6.00091/
#          6.00733 and 6.01167/5.98477/6.00165) within 0.0005 nats; the routed arms send the intended
#          FRACTION of uncovered types to the neighbour (within 1%); every arm is inert at covered inputs;
#          counts partition; live per-cell top-1 and CE identical across all six arms.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256; V = 50257; W = 50304
RANKS = ('nn', 'nn75', 'nn50', 'nn25', 'map64', 'map512')
ROUTED = {'nn75': 0.75, 'nn50': 0.50, 'nn25': 0.25}   # fraction of uncovered types on nn
MAPRANK_OF = {'nn': 0, 'nn75': 64, 'nn50': 64, 'nn25': 64, 'map64': 64, 'map512': 512}
# UNIFORM ranks only -- the question is about the table axis, not the allocation.
ALLOC = {k9: None for k9 in ('nn', 'nn75', 'nn50', 'nn25', 'map64', 'map512')}   # FULL table rank throughout
# §1937's PUBLISHED uncovered-input overall top-1, per arm -- the reproduction anchor.
S1937_UOV = {'nn': {'skip7000': 0.1468, 'skip11000': 0.1425, 'skip1200': 0.1390},
             'map64': {'skip7000': 0.1214, 'skip11000': 0.1263, 'skip1200': 0.1144}}
S1938_PCE = {'nn': {'skip7000': 6.01897, 'skip11000': 6.00091, 'skip1200': 6.00733},
             'map64': {'skip7000': 6.01167, 'skip11000': 5.98477, 'skip1200': 6.00165}}
# §1938's uncovered UNSEEN-bucket CE, the deficit pred_b takes a share of.
S1938_UBOT = {'nn': {'skip7000': 9.80856, 'skip11000': 9.58640, 'skip1200': 9.47057},
              'map512': {'skip7000': 8.68423, 'skip11000': 8.42422, 'skip1200': 8.41154}}

RIDGE = 1e-2
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/cosine_routed_fallback_results.json'
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
STATE = {}
COV = {}
ROUTEFRAC = {}


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
    """Flat per-position top-1 AND per-position CE for one arm (or live when hooks is empty).

    Returns one int32 vector over every scored position, in a fixed order, so two arms can be
    compared POSITION BY POSITION rather than only through their aggregates -- which is what an
    exact-zero bar on 'the map never moves a covered-input prediction' requires."""
    am, nl = [], []
    for i in range(0, rows.shape[0], 8):
        bb = rows[i:i + 8]
        idx = bb[:, :-1].to(DEV).contiguous()
        tg = bb[:, 1:].to(DEV)[:, 64:]
        lg = forward_logits(idx, hooks)[:, 64:]
        am.append(lg.argmax(-1).reshape(-1).to(torch.int32).cpu())
        lp = torch.log_softmax(lg.float(), -1)
        nl.append((-lp.gather(-1, tg.unsqueeze(-1).long()).squeeze(-1)).reshape(-1).float().cpu())
        del lg, lp
    return torch.cat(am), torch.cat(nl)


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
    """kept-fraction AND mean CE per (input-coverage class) x (target bucket), plus pooled.

    live and arm are each an (argmax, nll) pair from collect()."""
    freq = COV['freq'].cpu()[tgt.long()]
    (lam, lnl), (aam, anl) = live, arm
    o = {}
    for cname, cm in (('covered_input', icov), ('uncovered_input', ~icov),
                      ('pooled', torch.ones_like(icov))):
        o[cname] = {}
        for b in BUCKETS:
            msk = cm & (freq >= b[0]) & (freq <= b[1])
            n = int(msk.sum())
            al = float((lam[msk] == tgt[msk]).float().mean()) if n else 0.0
            ap = float((aam[msk] == tgt[msk]).float().mean()) if n else 0.0
            o[cname][f'{b[0]}-{b[1]}'] = {'n': n, 'top1_acc_live': al, 'top1_acc_prog': ap,
                                          'kept_fraction': ap / max(al, 1e-9),
                                          'ce_live': float(lnl[msk].mean()) if n else 0.0,
                                          'ce_prog': float(anl[msk].mean()) if n else 0.0}
        n = int(cm.sum())
        o[cname]['overall'] = {'n': n,
                               'top1_acc_live': float((lam[cm] == tgt[cm]).float().mean()),
                               'top1_acc_prog': float((aam[cm] == tgt[cm]).float().mean()),
                               'ce_live': float(lnl[cm].mean()),
                               'ce_prog': float(anl[cm].mean())}
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
    print(f'COSINE-ROUTED FALLBACK at {NCOV} | arms {RANKS} | routed per INPUT TOKEN on the '
          f'neighbour cosine | top-1 AND CE | FULL table rank | DISCOVERY ONLY', flush=True)

    # nnrow: the output-NN neighbour index (§1780/§1781). NOT used by program_rows below --
    # the settled design fills uncovered rows from the MAP alone (§1870). Kept because the
    # next experiment needs it as an arm; the misleading banner is LESSON 75.
    lpc = torch.zeros(ncov, W, device=DEV)
    for i in range(0, ncov, 256):
        t = tk[i:i + 256].unsqueeze(1)
        lpc[i:i + t.shape[0]] = torch.log_softmax(forward_logits(t)[:, 0].float(), -1)
    pcn = torch.softmax(lpc, -1)
    pcn = (pcn / pcn.norm(dim=-1, keepdim=True).clamp_min(1e-9)).half()
    del lpc
    nnrow = torch.zeros(V, dtype=torch.long, device=DEV)
    nnrow[tk] = torch.arange(ncov, device=DEV)
    nnsim = torch.zeros(V, device=DEV)
    for s0 in range(0, unc.numel(), 512):
        u = unc[s0:s0 + 512]
        p = torch.softmax(forward_logits(u.unsqueeze(1))[:, 0].float(), -1)
        p = p / p.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        sim, arg = (p.half() @ pcn.T).float().max(-1)
        nnrow[u] = arg
        nnsim[u] = sim          # LESSON 75: this was computed and thrown away in every prior script
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
        if r in ROUTED:
            # the top ROUTED[r] fraction of uncovered types BY COSINE take the neighbour row.
            su = nnsim[unc]
            tau = torch.quantile(su.double(), 1.0 - ROUTED[r]).float()
            usenn = (su >= tau)
            for st in sites:
                Ws = torch.linalg.solve(A, Ecov.T @ tc[st].double())
                U, S, Vh = torch.linalg.svd(Ws, full_matrices=False)
                mr = MAPRANK_OF[r]
                mp = (U[:, :mr] * S[:mr]) @ Vh[:mr]
                fr = torch.zeros(V, D, device=DEV)
                fr[tk] = tc[st]
                fr[unc] = torch.where(usenn.unsqueeze(1), tc[st][nnrow[unc]],
                                      (Eunc @ mp).float())
                out[st] = fr
            ROUTEFRAC[r] = float(usenn.float().mean())
            return out
        if r == 'nn':
            # §1780/§1781: an uncovered token gets the ROW OF ITS NEAREST COVERED TOKEN. No map.
            for st in sites:
                fr = torch.zeros(V, D, device=DEV)
                fr[tk] = tc[st]
                fr[unc] = tc[st][nnrow[unc]]
                out[st] = fr
            return out
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

    # every arm must be inert at covered inputs.
    PAIRS = [(a, 'map64') for a in RANKS if a != 'map64']
    chg_cov = {f'{a}_vs_{b}': sum(int(((P[(a, e)][0] != P[(b, e)][0]) & AX[e][1]).sum()) for e in roles)
               for a, b in PAIRS}
    chg_unc = {f'{a}_vs_{b}': sum(int(((P[(a, e)][0] != P[(b, e)][0]) & ~AX[e][1]).sum()) for e in roles)
               for a, b in PAIRS}

    def kf(e, cls, arm, b):
        return res[e][arm][cls][b]['kept_fraction']

    def ov(e, cls, arm):
        return res[e][arm][cls]['overall']['top1_acc_prog']

    def ce(e, cls, arm, b='overall'):
        return res[e][arm][cls][b]['ce_prog']

    ARMS_R = tuple(ROUTED)
    # pred_a -- a routed arm beats BOTH endpoints on the uncovered arm's top-1.
    abeats = {r: sum(1 for e in roles
                     if ov(e, 'uncovered_input', r) > ov(e, 'uncovered_input', 'nn')
                     and ov(e, 'uncovered_input', r) > ov(e, 'uncovered_input', 'map64'))
              for r in ARMS_R}
    pa_arm = max(ARMS_R, key=lambda r: abeats[r])
    pa = abeats[pa_arm] >= 2

    # pred_b -- that arm recovers >=50% of nn's unseen-bucket CE deficit against map512.
    def recov(e, r):
        hi, lo = S1938_UBOT['nn'][e], S1938_UBOT['map512'][e]
        return (hi - ce(e, 'uncovered_input', r, bot)) / (hi - lo)
    brec = sum(1 for e in roles if recov(e, pa_arm) >= 0.50)
    pb = brec >= 2

    # pred_c -- a routed arm dominates the DEPLOYED design on BOTH instruments at once.
    cdom = {r: sum(1 for e in roles
                   if ov(e, 'pooled', r) > ov(e, 'pooled', 'map64')
                   and ce(e, 'pooled', r) < ce(e, 'pooled', 'map64')) for r in ARMS_R}
    pc_arm = max(ARMS_R, key=lambda r: cdom[r])
    pc = cdom[pc_arm] >= 2

    partition = all(res[e][r]['covered_input'][b]['n'] + res[e][r]['uncovered_input'][b]['n']
                    == res[e][r]['pooled'][b]['n']
                    for e in roles for r in RANKS for b in RARE + [top])
    livet1 = max(abs(res[e][r][c][b]['top1_acc_live'] - res[e]['map64'][c][b]['top1_acc_live'])
                 for e in roles for r in RANKS
                 for c in ('covered_input', 'uncovered_input', 'pooled') for b in RARE + [top])
    livece = max(abs(res[e][r][c][b]['ce_live'] - res[e]['map64'][c][b]['ce_live'])
                 for e in roles for r in RANKS
                 for c in ('covered_input', 'uncovered_input', 'pooled') for b in RARE + [top])
    reprot1 = max(abs(ov(e, 'uncovered_input', r) - S1937_UOV[r][e])
                  for e in roles for r in S1937_UOV)
    reproce = max(abs(ce(e, 'pooled', r) - S1938_PCE[r][e]) for e in roles for r in S1938_PCE)
    fracok = max(abs(ROUTEFRAC[r] - ROUTED[r]) for r in ARMS_R)
    pd = (ncov == NCOV and partition and livet1 <= 1e-9 and livece <= 1e-9
          and reprot1 <= 0.0005 and reproce <= 0.0005 and fracok <= 0.01
          and all(v == 0 for v in chg_cov.values()))

    print(f'\n  routed fraction actually sent to the neighbour: ' + '  '.join(
        f'{r} {ROUTEFRAC[r]:.3f} (target {ROUTED[r]:.2f})' for r in ARMS_R), flush=True)
    print(f'  CHANGED top-1 between arms, summed over {len(roles)} roles:', flush=True)
    for k in chg_cov:
        print(f'    {k:18s} covered inputs {chg_cov[k]:6d}   uncovered inputs {chg_unc[k]:6d}',
              flush=True)
    for e in roles:
        u = res[e]['map64']['uncovered_input']['overall']
        print(f'\n  {e}: uncovered n={u["n"]} | live there top1 {u["top1_acc_live"]:.2%} '
              f'CE {u["ce_live"]:.5f}', flush=True)
        for r in RANKS:
            dt = (ov(e, 'pooled', r) - ov(e, 'pooled', 'map64')) * 100
            dc = ce(e, 'pooled', r) - ce(e, 'pooled', 'map64')
            mark = '  <== DOMINATES DEPLOYED' if (dt > 0 and dc < 0) else ''
            print(f'      {r:7s} UNCOV top1 {ov(e, "uncovered_input", r):6.2%} CE '
                  f'{ce(e, "uncovered_input", r):7.5f} unseen-CE {ce(e, "uncovered_input", r, bot):7.4f}'
                  f' | POOLED top1 {ov(e, "pooled", r):6.2%} ({dt:+.2f}pp) CE '
                  f'{ce(e, "pooled", r):7.5f} ({dc:+.5f}){mark}', flush=True)
        print(f'      unseen-bucket CE deficit recovered vs nn->map512: ' + '  '.join(
            f'{r} {recov(e, r):.1%}' for r in ARMS_R), flush=True)

    print(f'\n  a routed arm beats BOTH endpoints on uncovered top-1 (>=2 roles) -> {pa}  '
          f'best {pa_arm} {abeats[pa_arm]}/3  (all: ' +
          ' '.join(f'{r}:{abeats[r]}' for r in ARMS_R) + ')', flush=True)
    print(f'  and that arm recovers >=50% of the unseen-bucket CE deficit (>=2 roles) -> {pb}  '
          f'{brec}/3', flush=True)
    print(f'  and a routed arm DOMINATES the deployed design on both instruments (>=2 roles) -> {pc}  '
          f'best {pc_arm} {cdom[pc_arm]}/3  (all: ' +
          ' '.join(f'{r}:{cdom[r]}' for r in ARMS_R) + ')', flush=True)
    print(f'  coverage {ncov}, arms inert at covered inputs, partitions, LIVE identical, nn/map64 '
          f'reproduce §1937/§1938 (top1 {reprot1 * 100:.3f}pp, CE {reproce:.5f}), route fracs within '
          f'{fracok:.4f} -> control {pd}', flush=True)

    r2 = {'config': {'arms': list(RANKS), 'routed_fraction_target': ROUTED,
                     'routed_fraction_actual': {r: round(ROUTEFRAC[r], 5) for r in ARMS_R},
                     'routing': 'per INPUT TOKEN on the neighbour cosine: an uncovered type takes the '
                                'output-NN neighbour row when its cosine to that neighbour is in the '
                                'top ROUTED[arm] fraction, else the rank-64 map row. Cost ~5.40M = the '
                                'deployed map plus one index, a 1.7% premium on §1789.',
                     'coverage': ncov,
                     'instruments': 'per-position top-1 AND per-position CE, same build (§1938).',
                     'ROLE_NOTE': 'DISCOVERY ONLY, rung 3 -- §1938 open question.'},
          'results': {e: {r: {c: {b: {k: (round(v, 6) if isinstance(v, float) else v)
                                      for k, v in res[e][r][c][b].items()}
                                  for b in res[e][r][c]} for c in res[e][r]} for r in RANKS}
                      for e in roles},
          'changed_top1_covered': chg_cov, 'changed_top1_uncovered': chg_unc,
          'unseen_ce_deficit_recovered': {e: {r: round(recov(e, r), 5) for r in ARMS_R}
                                          for e in roles},
          'dominates_deployed_roles': {r: cdom[r] for r in ARMS_R},
          'beats_both_endpoints_roles': {r: abeats[r] for r in ARMS_R},
          'predictions': {'pred_a_hybrid_beats_both_endpoints': bool(pa),
                          'pred_b_closes_ce_hole': bool(pb),
                          'pred_c_dominates_deployed': bool(pc),
                          'pred_d_controls': bool(pd)},
          'runtime_s': round(time.time() - t0, 1)}
    json.dump(r2, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({time.time() - t0:.1f}s)', flush=True)


main()
