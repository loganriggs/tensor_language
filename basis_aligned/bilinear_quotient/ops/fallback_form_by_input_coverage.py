# WHICH FALLBACK FORM? -- the neighbour against the map, on the axis where fallbacks actually act.
#
# §1936 established that the fallback touches ONLY the ~24-25% of positions whose current token has no
# table entry (0 changed top-1 at 69,444 covered-input positions), and that on that arm the compiled
# program runs at 28.7% kept-fraction against the covered arm's 37.5% -- while the LIVE model scores
# HIGHER there (45.4/49.3/46.1%) than at covered inputs (37.4/40.0/36.6%). A quarter of positions served
# at three-quarters of the covered arm's efficiency, on easier material. That gap is the open cost lever
# and §1936 named it: every measurement since §1870 has varied the map's RANK and never its FORM.
#
# There is exactly one alternative form this thread has measured -- §1780/§1781's output-NN neighbour,
# which gives an uncovered token the ROW OF ITS NEAREST COVERED TOKEN in output-distribution space. §1870
# priced it in CE and the map won (+0.0073 / +0.0161 / +0.0057 nats). It has never been scored on top-1,
# never on buckets, and never on the input-coverage axis. It is also FREE: it stores one index per
# uncovered type (~0.09M against the rank-64 map's 5.308M and the rank-512 map's 42.467M).
#
# The mechanism reason to expect a split: the neighbour copies a REAL covered token's row, so it should
# behave like a real token -- and real tokens are exactly what the 125+ bucket rewards. The map's low-rank
# least-squares reconstruction has no such guarantee, and §1935/§1936 found it pays a 125+ toll
# (-1.20/-2.48/-2.16pp on the uncovered arm) that no map rank avoids.
#
# ARMS. nn (neighbour, ~0.09M), map64 (§1789 deployed, 5.308M), map512 (42.467M). Full table rank, 5,419
# coverage, scored on (input covered/uncovered) x (target bucket).
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §1936's open question.
#
# Registered predictions, SIGNED and one-sided per LESSON 72.
#   pred_a THE MAP STILL WINS OVERALL ON ITS OWN ARM: restricted to uncovered inputs, the neighbour's
#          overall top-1 is BELOW map64's, on at least 2 of 3 roles. §1870 established this in CE over all
#          positions; this asks whether it survives on top-1 and on the ~24% where the choice acts. If
#          FALSE, §1870's ranking is an artefact of CE or of pooling and the deployed design is wrong.
#   pred_b BUT THE NEIGHBOUR WINS THE COMMON BUCKET: restricted to uncovered inputs, the neighbour's 125+
#          kept-fraction is ABOVE map512's, on at least 2 of 3 roles. This is the mechanism claim -- the
#          125+ toll is a property of the MAP's form, not of fallbacks in general. If FALSE the toll is
#          intrinsic to substituting a row at all and no hybrid can avoid it.
#   pred_c AND A PER-BUCKET HYBRID WOULD BEAT EITHER: on uncovered inputs, taking the better of {nn,
#          map512} in each of the five buckets beats map512's own five kept-fractions by at least 1.0pp
#          SUMMED, on at least 2 of 3 roles. This is an ORACLE CEILING on a hybrid, not a build -- a
#          per-bucket choice needs the target, which no deployed program has. Registered and labelled as
#          a bound. If FALSE the two forms do not differ enough by bucket for any mixture to help.
#   pred_d CONTROLS: coverage exactly 5,419; ALL THREE arms change the top-1 at EXACTLY 0 covered-input
#          positions (§1936's instrument extended to the neighbour, which must also be inert there);
#          counts partition; the map64 arm reproduces §1936's PUBLISHED uncovered-input kept-fractions --
#          125+ 48.7/44.9/48.1% and 0-0 6.5/13.0/6.7% -- within 0.2pp; live per-cell accuracy identical
#          across all three arms.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256; V = 50257; W = 50304
RANKS = ('nn', 'map64', 'map512')
MAPRANK_OF = {'nn': 0, 'map64': 64, 'map512': 512}   # rank 0 == the neighbour, no map at all
# UNIFORM ranks only -- the question is about the table axis, not the allocation.
ALLOC = {k9: None for k9 in ('nn', 'map64', 'map512')}   # FULL table rank throughout
S1936_UTOP64 = {'skip7000': 0.487, 'skip11000': 0.449, 'skip1200': 0.481}  # §1936 UNCOVERED-input, map64
S1936_UBOT64 = {'skip7000': 0.065, 'skip11000': 0.130, 'skip1200': 0.067}  # §1936 UNCOVERED-input, map64

RIDGE = 1e-2
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/fallback_form_by_input_coverage_results.json'
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
    print(f'FALLBACK FORM BY INPUT COVERAGE at {NCOV} | arms {RANKS} | (input covered/uncovered) x '
          f'(target bucket {BUCKETS}) | FULL table rank | DISCOVERY ONLY', flush=True)

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

    # every arm must be inert at covered inputs -- §1936's instrument, extended to all three pairs.
    PAIRS = [('nn', 'map64'), ('nn', 'map512'), ('map64', 'map512')]
    chg_cov = {f'{a}_vs_{b}': sum(int(((P[(a, e)] != P[(b, e)]) & AX[e][1]).sum()) for e in roles)
               for a, b in PAIRS}
    chg_unc = {f'{a}_vs_{b}': sum(int(((P[(a, e)] != P[(b, e)]) & ~AX[e][1]).sum()) for e in roles)
               for a, b in PAIRS}

    def kf(e, cls, arm, b):
        return res[e][arm][cls][b]['kept_fraction']

    def ov(e, cls, arm):
        return res[e][arm][cls]['overall']['top1_acc_prog']

    # pred_a -- the neighbour is BELOW map64 on the uncovered arm's overall top-1.
    abelow = sum(1 for e in roles
                 if ov(e, 'uncovered_input', 'nn') < ov(e, 'uncovered_input', 'map64'))
    pa = abelow >= 2
    # pred_b -- but the neighbour is ABOVE map512 on the uncovered 125+ bucket.
    babove = sum(1 for e in roles
                 if kf(e, 'uncovered_input', 'nn', top) > kf(e, 'uncovered_input', 'map512', top))
    pb = babove >= 2
    # pred_c -- ORACLE CEILING: per-bucket better-of {nn, map512}, summed advantage over map512.
    hyb = {e: sum(max(0.0, kf(e, 'uncovered_input', 'nn', b) - kf(e, 'uncovered_input', 'map512', b))
                  for b in RARE + [top]) for e in roles}
    cwin = sum(1 for e in roles if hyb[e] >= 0.010)
    pc = cwin >= 2

    partition = all(res[e][r]['covered_input'][b]['n'] + res[e][r]['uncovered_input'][b]['n']
                    == res[e][r]['pooled'][b]['n']
                    for e in roles for r in RANKS for b in RARE + [top])
    livespread = max(abs(res[e][r][c][b]['top1_acc_live'] - res[e]['map64'][c][b]['top1_acc_live'])
                     for e in roles for r in RANKS
                     for c in ('covered_input', 'uncovered_input', 'pooled')
                     for b in RARE + [top])
    repro = max(max(abs(kf(e, 'uncovered_input', 'map64', top) - S1936_UTOP64[e]),
                    abs(kf(e, 'uncovered_input', 'map64', bot) - S1936_UBOT64[e])) for e in roles)
    pd = (ncov == NCOV and partition and livespread <= 1e-9 and repro <= 0.002
          and all(v == 0 for v in chg_cov.values()))

    print(f'\n  CHANGED top-1 between arms, summed over {len(roles)} roles:', flush=True)
    for k in chg_cov:
        print(f'    {k:18s} covered inputs {chg_cov[k]:6d}   uncovered inputs {chg_unc[k]:6d}',
              flush=True)
    for e in roles:
        nu = res[e]['map64']['uncovered_input']['overall']['n']
        npo = res[e]['map64']['pooled']['overall']['n']
        print(f'\n  {e}: {nu}/{npo} = {nu / npo:.1%} of scored positions have an UNCOVERED input; '
              f'live there {res[e]["map64"]["uncovered_input"]["overall"]["top1_acc_live"]:.2%}',
              flush=True)
        print(f'    UNCOVERED-input kept-fraction by arm', flush=True)
        for r in RANKS:
            print(f'      {r:7s} overall top1 {ov(e, "uncovered_input", r):6.2%} | ' + '  '.join(
                f'{b:>7s} {kf(e, "uncovered_input", r, b):5.1%}' for b in RARE + [top]), flush=True)
        print(f'      oracle per-bucket better-of(nn, map512) beats map512 by '
              f'{hyb[e] * 100:.2f}pp summed', flush=True)

    print(f'\n  the MAP still wins overall on the uncovered arm (>=2 roles) -> {pa}  {abelow}/3',
          flush=True)
    print(f'  but the NEIGHBOUR wins the 125+ bucket there (>=2 roles) -> {pb}  {babove}/3', flush=True)
    print(f'  and a per-bucket hybrid CEILING beats map512 by >=1.0pp summed (>=2 roles) -> {pc}  '
          f'{cwin}/3', flush=True)
    print(f'  coverage {ncov}, ALL arms inert at covered inputs, partitions, LIVE identical, map64 '
          f'reproduces §1936 (max dev {repro * 100:.2f}pp) -> control {pd}', flush=True)

    r2 = {'config': {'arms': list(RANKS),
                     'nn': 'output-NN neighbour (§1780/§1781): an uncovered token takes the ROW of its '
                           'nearest covered token in output-distribution space. ~0.09M (one index per '
                           'uncovered type).',
                     'map64': '§1789 deployed, rank-64 embedding->row map, 5.308M',
                     'map512': 'rank-512 map, 42.467M',
                     'table_rank': 'FULL in all arms -- only the fallback FORM/rank differs',
                     'coverage': ncov,
                     'axis': '(INPUT token covered vs uncovered) x (TRUE TARGET fit-row bucket), the '
                             'axis introduced in §1936.',
                     'ROLE_NOTE': 'DISCOVERY ONLY, rung 3 -- §1936 open question.'},
          'results': {e: {r: {c: {b: {k: (round(v, 6) if isinstance(v, float) else v)
                                      for k, v in res[e][r][c][b].items()}
                                  for b in res[e][r][c]} for c in res[e][r]} for r in RANKS}
                      for e in roles},
          'changed_top1_covered': chg_cov,
          'changed_top1_uncovered': chg_unc,
          'oracle_hybrid_ceiling_pp': {e: round(hyb[e] * 100, 3) for e in roles},
          'predictions': {'pred_a_map_wins_overall': bool(pa),
                          'pred_b_neighbour_wins_common': bool(pb),
                          'pred_c_hybrid_ceiling': bool(pc),
                          'pred_d_controls': bool(pd)},
          'runtime_s': round(time.time() - t0, 1)}
    json.dump(r2, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({time.time() - t0:.1f}s)', flush=True)


main()
