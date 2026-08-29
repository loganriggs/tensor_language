# DO CE AND TOP-1 ACTUALLY DISAGREE? -- both instruments, same build, same positions.
#
# §1937 found that §1780/§1781's output-NN neighbour beats §1870's map on TOP-1 by +2.54/+1.62/+2.47pp on
# the uncovered arm and +0.61/+0.41/+0.60pp pooled, at ~0.09M against 5.308M. §1870 had selected the map
# on CE, by +0.0073/+0.0161/+0.0057 nats. So the two instruments appear to rank the two fallback FORMS in
# opposite directions -- but §1870's CE and §1937's top-1 come from DIFFERENT SCRIPTS on different builds,
# and a cross-run comparison of two instruments is exactly the shape LESSON 71 warns about. Nothing has
# measured both, in one build, on one set of positions.
#
# This does. Three arms (nn, map64, map512), full table rank, 5,419 coverage, and BOTH per-position
# top-1 and per-position CE, cross-tabulated on §1936's (input covered/uncovered) x (target bucket) axis.
# Rung 2: a second-class confirmation of §1937 with a DIFFERENT instrument, not a replication of the same
# one.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY.
#
# Registered predictions, SIGNED and one-sided per LESSON 72.
#   pred_a THE INVERSION IS REAL WITHIN ONE BUILD: on uncovered-input positions the neighbour's top-1 is
#          ABOVE map64's on 3 of 3 roles (replicating §1937) AND its CE is WORSE -- strictly higher -- than
#          map64's on at least 2 of 3. Both halves must hold; that conjunction IS the disagreement. If the
#          CE half fails, the two instruments agree after all, §1870's ranking was a cross-run artefact,
#          and the neighbour is simply better -- a cleaner and more consequential outcome than the one I
#          am predicting.
#   pred_b AND THE CE SIDE REPRODUCES §1870: pooled over all scored positions, the neighbour's CE exceeds
#          map64's by between 0.004 and 0.020 nats on at least 2 of 3 roles -- the band §1870 published
#          (+0.0073 / +0.0161 / +0.0057). A two-sided band on a quantity whose SIGN pred_a already tests,
#          so this is a reproduction check, not a direction test (LESSON 72).
#   pred_c AND THE CE PENALTY IS THE UNSEEN BUCKET: restricted to uncovered inputs whose true target is in
#          the 0-0 bucket, the neighbour's CE exceeds map512's by at least 0.05 nats on at least 2 of 3
#          roles. §1937's mechanism says the neighbour cannot reach a target no fit row contains, so its
#          whole CE deficit should live there. If FALSE the CE penalty is spread and the two instruments
#          are measuring something other than the unseen case.
#   pred_d CONTROLS: coverage exactly 5,419; all three arms change the top-1 at EXACTLY 0 covered-input
#          positions; the top-1 figures reproduce §1937's PUBLISHED uncovered-input overalls -- nn 14.68/
#          14.25/13.90, map64 12.14/12.63/11.44, map512 13.05/13.10/11.75% -- within 0.05pp; counts
#          partition; and the LIVE per-cell top-1 AND CE are identical across all three arms.
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
# §1937's PUBLISHED uncovered-input overall top-1, per arm -- the reproduction anchor.
S1937_UOV = {'nn': {'skip7000': 0.1468, 'skip11000': 0.1425, 'skip1200': 0.1390},
             'map64': {'skip7000': 0.1214, 'skip11000': 0.1263, 'skip1200': 0.1144},
             'map512': {'skip7000': 0.1305, 'skip11000': 0.1310, 'skip1200': 0.1175}}
S1870_CE_BAND = (0.004, 0.020)   # §1870 published +0.0073 / +0.0161 / +0.0057 nats (nn worse than map)

RIDGE = 1e-2
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/both_instruments_same_build_results.json'
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
    print(f'BOTH INSTRUMENTS, SAME BUILD at {NCOV} | arms {RANKS} | top-1 AND CE on (input '
          f'covered/uncovered) x (target bucket {BUCKETS}) | FULL table rank | DISCOVERY ONLY', flush=True)

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

    # pred_a -- the CONJUNCTION is the disagreement: nn wins top-1, nn loses CE, same build.
    t1win = sum(1 for e in roles if ov(e, 'uncovered_input', 'nn') > ov(e, 'uncovered_input', 'map64'))
    celose = sum(1 for e in roles if ce(e, 'uncovered_input', 'nn') > ce(e, 'uncovered_input', 'map64'))
    pa = (t1win == 3 and celose >= 2)
    # pred_b -- pooled CE gap reproduces §1870's published band.
    gap = {e: ce(e, 'pooled', 'nn') - ce(e, 'pooled', 'map64') for e in roles}
    inband = sum(1 for e in roles if S1870_CE_BAND[0] <= gap[e] <= S1870_CE_BAND[1])
    pb = inband >= 2
    # pred_c -- the CE penalty lives in the unseen bucket, on the uncovered arm.
    ubot = {e: ce(e, 'uncovered_input', 'nn', bot) - ce(e, 'uncovered_input', 'map512', bot)
            for e in roles}
    cbot = sum(1 for e in roles if ubot[e] >= 0.05)
    pc = cbot >= 2

    partition = all(res[e][r]['covered_input'][b]['n'] + res[e][r]['uncovered_input'][b]['n']
                    == res[e][r]['pooled'][b]['n']
                    for e in roles for r in RANKS for b in RARE + [top])
    livet1 = max(abs(res[e][r][c][b]['top1_acc_live'] - res[e]['map64'][c][b]['top1_acc_live'])
                 for e in roles for r in RANKS
                 for c in ('covered_input', 'uncovered_input', 'pooled') for b in RARE + [top])
    livece = max(abs(res[e][r][c][b]['ce_live'] - res[e]['map64'][c][b]['ce_live'])
                 for e in roles for r in RANKS
                 for c in ('covered_input', 'uncovered_input', 'pooled') for b in RARE + [top])
    repro = max(abs(ov(e, 'uncovered_input', r) - S1937_UOV[r][e]) for e in roles for r in RANKS)
    pd = (ncov == NCOV and partition and livet1 <= 1e-9 and livece <= 1e-9 and repro <= 0.0005
          and all(v == 0 for v in chg_cov.values()))

    print(f'\n  CHANGED top-1 between arms, summed over {len(roles)} roles:', flush=True)
    for k in chg_cov:
        print(f'    {k:18s} covered inputs {chg_cov[k]:6d}   uncovered inputs {chg_unc[k]:6d}',
              flush=True)
    for e in roles:
        u = res[e]['map64']['uncovered_input']['overall']
        print(f'\n  {e}: uncovered n={u["n"]} ({u["n"] / res[e]["map64"]["pooled"]["overall"]["n"]:.1%}) '
              f'| live there top1 {u["top1_acc_live"]:.2%} CE {u["ce_live"]:.5f}', flush=True)
        for r in RANKS:
            print(f'      {r:7s} UNCOV top1 {ov(e, "uncovered_input", r):6.2%} CE '
                  f'{ce(e, "uncovered_input", r):7.5f} | POOLED top1 '
                  f'{ov(e, "pooled", r):6.2%} CE {ce(e, "pooled", r):7.5f} | unseen-bucket CE '
                  f'{ce(e, "uncovered_input", r, bot):7.5f}', flush=True)
        print(f'      pooled CE gap nn - map64 = {gap[e]:+.5f} nats (§1870 band '
              f'{S1870_CE_BAND[0]}-{S1870_CE_BAND[1]}) | uncovered unseen-bucket CE gap '
              f'nn - map512 = {ubot[e]:+.5f}', flush=True)

    print(f'\n  the INVERSION is real in one build: nn wins top-1 3/3 AND loses CE (>=2 roles) -> {pa}  '
          f'top1 {t1win}/3, CE-worse {celose}/3', flush=True)
    print(f'  and the pooled CE gap reproduces §1870\'s band (>=2 roles) -> {pb}  {inband}/3', flush=True)
    print(f'  and the CE penalty is the UNSEEN bucket, >=0.05 nats (>=2 roles) -> {pc}  {cbot}/3',
          flush=True)
    print(f'  coverage {ncov}, all arms inert at covered inputs, partitions, LIVE top1 AND CE '
          f'identical, top-1 reproduces §1937 (max dev {repro * 100:.3f}pp) -> control {pd}', flush=True)

    r2 = {'config': {'arms': list(RANKS),
                     'nn': 'output-NN neighbour (§1780/§1781), ~0.09M',
                     'map64': '§1789 deployed, rank-64 map, 5.308M',
                     'map512': 'rank-512 map, 42.467M',
                     'table_rank': 'FULL in all arms -- only the fallback FORM/rank differs',
                     'coverage': ncov,
                     'instruments': 'BOTH per-position top-1 and per-position CE, in the same build on '
                                    'the same positions -- the point of the run.',
                     'axis': '(INPUT token covered vs uncovered) x (TRUE TARGET fit-row bucket), §1936.',
                     'ROLE_NOTE': 'DISCOVERY ONLY, rung 2 -- second-class confirmation of §1937 with a '
                                  'DIFFERENT instrument.'},
          'results': {e: {r: {c: {b: {k: (round(v, 6) if isinstance(v, float) else v)
                                      for k, v in res[e][r][c][b].items()}
                                  for b in res[e][r][c]} for c in res[e][r]} for r in RANKS}
                      for e in roles},
          'changed_top1_covered': chg_cov,
          'changed_top1_uncovered': chg_unc,
          'pooled_ce_gap_nn_minus_map64': {e: round(gap[e], 6) for e in roles},
          'uncovered_unseen_ce_gap_nn_minus_map512': {e: round(ubot[e], 6) for e in roles},
          'predictions': {'pred_a_inversion_is_real': bool(pa),
                          'pred_b_ce_reproduces_s1870': bool(pb),
                          'pred_c_penalty_is_unseen_bucket': bool(pc),
                          'pred_d_controls': bool(pd)},
          'runtime_s': round(time.time() - t0, 1)}
    json.dump(r2, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({time.time() - t0:.1f}s)', flush=True)


main()
