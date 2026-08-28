# front_mlp_site_shortfall: WHICH OF THE FOUR FRONT MLPs COSTS THE 7.53 POINTS?
#
# §1703 normalised the joint program's band shortfall by site count and found the reading the
# totals hide: the FRONT MLPs are the worst-modelled sites in bilin18 at 1.883 points each,
# nearly double either middle band, while the front attention writes are the best at 0.150.
# Four sites is the only band small enough for site-level exempt-one at tractable cost (four
# recompiles at ~27 minutes against twelve for a middle band), and mlp1 is the model's largest
# single module.
#
# The front band already uses the best family available (§1672: token tables at mlp0-2, linear
# at mlp3), so its 7.53 points are not a family-choice failure -- they are what the best known
# program still cannot do at those four sites. Locating them is what says where to spend next.
#
# WHAT IS ALREADY KNOWN ABOUT THESE FOUR, all measured in the MLP-ONLY condition, which
# §1696/§1697 showed does not transfer at face value -- which is why this run exists:
#   per-site table ceilings (§1662):  mlp0 90.27%  mlp1 96.01%  mlp2 76.98%  mlp3 67.55%
#   per-site family verdicts (§1672): mlp0 and mlp1 want a table, mlp2 is indifferent, mlp3 does not
#   stakes (§1662): mlp0 0.855  mlp1 7.005  mlp2 0.772  mlp3 0.620 nats -- mlp1 dominates, though
#     §1666 flagged that stake as SATURATED (CE 10.28 against a 10.82 uniform ceiling) and
#     therefore ordinal at best
#
# Method and controls are §1703's: exempt one site, RECOMPILE the whole 36-site program, read the
# gain, with a 95% row-level cluster bootstrap. The no-exemption arm must reproduce §1696's
# 55.04% and the four site gains must be positive.
#
# Registered predictions:
#   pred_a mlp1 DOMINATES THE FRONT BAND: its exemption gain exceeds the other three front sites'
#          gains summed. It is the largest module in the model by stake and the front band's cost
#          is the thing being decomposed.
#   pred_b THE SITES ARE ROUGHLY ADDITIVE, as §1697 found for half-upgrades: the four site gains
#          sum to within 25% of the band's own +7.53. Strong sub-additivity would mean the sites
#          are covering for each other; strong super-additivity would mean the recompile is not
#          converging.
#   pred_c CONTROLS: the no-exemption arm reproduces §1696's 55.04% within 0.5 points, the
#          baseline CE reproduces 3.29205 (§1695), and all four site gains are positive.
#   pred_d SPREAD: the largest site gain's 95% interval excludes the smallest's, so the ordering
#          is resolved rather than asserted -- §1691 failed at a grid boundary for want of this.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
ALL18 = list(range(0, 18))
RIDGE = 1e-3
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'front_mlp_site_shortfall_results.json'
FIT_ROWS = PT + '.rowcache/fineweb_n480_skip80.pt'
MASK_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
EVAL_ROWS = PT + '.rowcache/fineweb_n192_skip7000.pt'
CONSTS = PT + 'opt_ablation_consts_all.pt'
H = m.transformer.h
S1694_SIMPLE = 0.5094
S1696_BOTH = 0.5504
S1698_RANK8 = 0.5475
BANDS = {f'mlp{L}': ('mlp', range(L, L + 1)) for L in range(0, 4)}
S1703_FRONT_BAND_GAIN = 0.0753
S1662_PER_SITE = {'mlp0': 0.9027, 'mlp1': 0.9601, 'mlp2': 0.7698, 'mlp3': 0.6755}
EXEMPT = {'set': frozenset()}
S1694_JOINT_STAKE = 5.5684
S1687_ATTN_BEST = 0.6805
S1672_MLP_TABLE_SITES = (0, 1, 2)
ATTN_LAGS = (1, 2, 4, 8)
S1683_CE_LIVE = 3.29205
CFG = {'lags': ATTN_LAGS, 'tables': S1672_MLP_TABLE_SITES, 'v1': None}
V1P = {}
STATE = {}
SEENREF = {}


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


def lagged(x):
    """[x_t, x_{t-l} for l in CFG['lags']], zero-padded at the start (§1685/§1686)."""
    lags = CFG['lags']
    parts = [x] + [torch.cat([torch.zeros_like(x[:, :l]), x[:, :-l]], dim=1) for l in lags]
    return torch.cat(parts, dim=-1).reshape(-1, D * (1 + len(lags)))


def table_hook(tbl):
    def hook(mod, args, out):
        sub = tbl[STATE['idx'].reshape(-1)].reshape(out.shape).to(out.dtype)
        return torch.where(SEENREF['m'][STATE['idx']].unsqueeze(-1), sub, out)
    return hook


def mlp_const_hook(c):
    def hook(mod, args, out):
        return c.to(out.dtype).expand_as(out)
    return hook


def attn_const_hook(c):
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        sub = c.to(y.dtype).expand_as(y)
        return (sub,) + tuple(out[1:]) if isinstance(out, tuple) else sub
    return hook


def mlp_prog_hook(W):
    def hook(mod, args, out):
        sub = (args[0].reshape(-1, D) @ W).reshape(out.shape).to(out.dtype)
        return torch.where(SEENREF['m'][STATE['idx']].unsqueeze(-1), sub, out)
    return hook


def sub_v1(v):
    """Replace the v1 tensor block 0 exports, per CFG['v1']."""
    mode = CFG['v1']
    if mode is None or 'W' not in V1P:
        return v
    # v1 is the HEAD-SPLIT view (B, T, n_head, head_dim) -- last dim is 128, not D.
    # n_head * head_dim == D and the two dims are adjacent, so reshape(-1, D) flattens
    # them correctly. The first build indexed v.shape[-1] and died on 1152 vs 128.
    flat = v.reshape(-1, D)
    if mode == 'table':
        new = V1P['W'][STATE['idx'].reshape(-1)]
    elif mode == 'const':
        new = V1P['W'].unsqueeze(0).expand(flat.shape[0], -1)
    else:
        new = V1P['x'].reshape(-1, D) @ V1P['W']
    new = torch.where(SEENREF['m'][STATE['idx']].reshape(-1).unsqueeze(-1), new, flat)
    return new.reshape(v.shape).to(v.dtype)


def attn_prog_hook(W, L=None):
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        sub = (lagged(args[0]) @ W).reshape(y.shape).to(y.dtype)
        sub = torch.where(SEENREF['m'][STATE['idx']].unsqueeze(-1), sub, y)
        if not isinstance(out, tuple):
            return sub
        rest = list(out[1:])
        if L == 0 and rest and torch.is_tensor(rest[0]):
            V1P['x'] = args[0]
            rest[0] = sub_v1(rest[0])
        return (sub,) + tuple(rest)
    return hook


def install(prog):
    hs = []
    for (kind, L), W in prog.items():
        if kind == 'mlp':
            hs.append(H[L].mlp.register_forward_hook(
                table_hook(W) if L in CFG['tables'] else mlp_prog_hook(W)))
        else:
            hs.append(H[L].attn.register_forward_hook(attn_prog_hook(W, L)))
    return hs


@torch.no_grad()
def sweep(rows, hooks=(), score=None):
    hs = list(hooks)
    try:
        for i in range(0, rows.shape[0], 8):
            bb = rows[i:i + 8]
            idx = bb[:, :-1].to(DEV).contiguous()
            STATE['idx'] = idx
            x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
            for blk in H:
                x, v1 = blk(x, v1, x0)
            if score is not None:
                lg = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
                score(lg, bb[:, 1:].to(DEV), idx)
    finally:
        for h in hs:
            h.remove()


@torch.no_grad()
def fit_table(rows, L, prog):
    """Per-token mean of mlp_L's output with the stack below substituted (§1661 hybrid)."""
    s = torch.zeros(50257, D, device=DEV)
    c = torch.zeros(50257, device=DEV)

    def collect(mod, args, out):
        t = STATE['idx'].reshape(-1)
        s.index_add_(0, t, out.float().reshape(-1, D))
        c.index_add_(0, t, torch.ones_like(t, dtype=torch.float32))
        return None
    sweep(rows, hooks=install(prog) + [H[L].mlp.register_forward_hook(collect)])
    assert float(c.sum()) > 0, f'mlp{L}: no token counts'
    sn = c > 0
    tbl = (s.sum(0) / c.sum()).unsqueeze(0).repeat(50257, 1)
    tbl[sn] = s[sn] / c[sn].unsqueeze(1)
    return tbl


@torch.no_grad()
def fit_site(rows, kind, L, prog):
    if kind == 'mlp' and L in CFG['tables']:
        return fit_table(rows, L, prog)
    din = D if kind == 'mlp' else D * (1 + len(CFG['lags']))
    A = torch.zeros(din, din, device=DEV, dtype=torch.float64)
    B = torch.zeros(din, D, device=DEV, dtype=torch.float64)
    n = {'v': 0}

    def collect(mod, args, out):
        x = (args[0].reshape(-1, D) if kind == 'mlp' else lagged(args[0])).double()
        y = (out if kind == 'mlp' else (out[0] if isinstance(out, tuple) else out))
        A.add_(x.T @ x); B.add_(x.T @ y.reshape(-1, D).double()); n['v'] += x.shape[0]
        return None
    tgt = H[L].mlp if kind == 'mlp' else H[L].attn
    sweep(rows, hooks=install(prog) + [tgt.register_forward_hook(collect)])
    assert n['v'] > 0, f'{kind}{L}: no fit positions accumulated'
    a = A / n['v']
    reg = RIDGE * torch.diag(a).mean() * torch.eye(din, device=DEV, dtype=torch.float64)
    return torch.linalg.solve(a + reg, B / n['v']).float()


@torch.no_grad()
def compile_stack(rows, kinds):
    """Interleaved bottom-up: within block L, attn_L then mlp_L (§1669). Sites in
    EXEMPT['set'] are skipped entirely and therefore run LIVE."""
    prog = {}
    for L in ALL18:
        for kind in ('attn', 'mlp'):
            if (kind, L) in EXEMPT['set']:
                continue
            if kind in kinds:
                prog[(kind, L)] = fit_site(rows, kind, L, prog)
    return prog


@torch.no_grad()
def seen_mask(rows):
    c = torch.zeros(50257, device=DEV)
    for i in range(0, rows.shape[0], 8):
        t = rows[i:i + 8, :-1].to(DEV).reshape(-1)
        c.index_add_(0, t, torch.ones_like(t, dtype=torch.float32))
    return c > 0


@torch.no_grad()
def ce_rows(rows, seen, hooks=()):
    """Per-row covered loss sums and counts, plus the pooled scalar. The scalar is recomputed
    from the per-row arrays and cross-checked against direct accumulation (known answer)."""
    S, N = [], []
    acc = {'t': 0.0, 'n': 0}

    def score(lg, tg, idx):
        e = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                            reduction='none').reshape(tg.shape)[:, 64:]
        cov = seen[idx[:, 64:]]
        masked = e * cov
        S.append(masked.sum(dim=1).double().cpu())
        N.append(cov.sum(dim=1).double().cpu())
        acc['t'] += float(e[cov].sum()); acc['n'] += int(cov.sum())
    sweep(rows, hooks=hooks, score=score)
    s_all = torch.cat(S); n_all = torch.cat(N)
    pooled = float(s_all.sum() / n_all.sum())
    # PRE-FLIGHT E: scale by the precision the data was COMPUTED in, never a fixed absolute
    # bar. Both paths accumulate float32 losses in different ORDERS before the double sum, so
    # they can differ by a few float32 ulps of the CE magnitude. My first bar was 1e-9
    # absolute, which is 0.0006 of the achievable floor and fired on a correct run at 1.41e-08
    # -- the same error I had flagged in a peer's 1e-12 gate three hours earlier.
    direct = acc['t'] / max(acc['n'], 1)
    tol = 4 * 1.1920929e-07 * max(abs(direct), 1.0)
    assert abs(pooled - direct) <= tol, (
        f'per-row reconstruction {pooled} != direct accumulation {direct} '
        f'(|diff| {abs(pooled - direct):.3e} > tol {tol:.3e})')
    return pooled, s_all, n_all


@torch.no_grad()
def ce(rows, seen, hooks=()):
    return ce_rows(rows, seen, hooks)[0]


@torch.no_grad()
def fit_v1(rows, prog, mode, rank):
    """Fit v1's program from block 0's attention input, with `prog` installed."""
    cap = {}

    def grab(mod, args, out):
        cap['x'] = args[0].reshape(-1, D).detach()
        cap['v'] = (out[1] if isinstance(out, tuple) else out).reshape(-1, D).detach()
        return None
    if mode == 'const':
        acc = {'s': torch.zeros(D, device=DEV, dtype=torch.float64), 'n': 0}

        def collect_c(mod, args, out):
            grab(mod, args, out)
            acc['s'] += cap['v'].double().sum(0); acc['n'] += cap['v'].shape[0]
            return None
        sweep(rows, hooks=install(prog) + [H[0].attn.register_forward_hook(collect_c)])
        assert acc['n'] > 0, 'v1: no positions for the constant'
        return (acc['s'] / acc['n']).float()
    if mode == 'table':
        s = torch.zeros(50257, D, device=DEV)
        c = torch.zeros(50257, device=DEV)

        def collect(mod, args, out):
            grab(mod, args, out)
            t = STATE['idx'].reshape(-1)
            s.index_add_(0, t, cap['v'].float())
            c.index_add_(0, t, torch.ones_like(t, dtype=torch.float32))
            return None
        sweep(rows, hooks=install(prog) + [H[0].attn.register_forward_hook(collect)])
        assert float(c.sum()) > 0, 'v1: no token counts'
        sn = c > 0
        tbl = (s.sum(0) / c.sum()).unsqueeze(0).repeat(50257, 1)
        tbl[sn] = s[sn] / c[sn].unsqueeze(1)
        return tbl
    A = torch.zeros(D, D, device=DEV, dtype=torch.float64)
    B = torch.zeros(D, D, device=DEV, dtype=torch.float64)
    n = {'v': 0}

    def collect_l(mod, args, out):
        grab(mod, args, out)
        x = cap['x'].double()
        A.add_(x.T @ x); B.add_(x.T @ cap['v'].double()); n['v'] += x.shape[0]
        return None
    sweep(rows, hooks=install(prog) + [H[0].attn.register_forward_hook(collect_l)])
    assert n['v'] > 0, 'v1: no fit positions'
    a = A / n['v']
    reg = RIDGE * torch.diag(a).mean() * torch.eye(D, device=DEV, dtype=torch.float64)
    W = torch.linalg.solve(a + reg, B / n['v']).float()
    if rank < D:
        U, S, Vh = torch.linalg.svd(W.double(), full_matrices=False)
        W = ((U[:, :rank] * S[:rank]) @ Vh[:rank]).float()
    return W


def boot_gains(SL, NL, SC, NC, arms_rows, base_key, draws=2000, seed=20260828):
    """Paired row-level cluster bootstrap of each arm's gain over the no-exemption arm."""
    g = torch.Generator().manual_seed(seed)
    R = NL.numel()
    out = {k: [] for k in arms_rows if k != base_key}
    for _ in range(draws):
        pick = torch.randint(0, R, (R,), generator=g)
        cl = SL[pick].sum() / NL[pick].sum()
        cc = SC[pick].sum() / NC[pick].sum()
        st = cc - cl
        if abs(float(st)) < 1e-12:
            continue
        Sb, Nb = arms_rows[base_key]
        base = float((cc - Sb[pick].sum() / Nb[pick].sum()) / st)
        for k, (S, N) in arms_rows.items():
            if k == base_key:
                continue
            out[k].append(float((cc - S[pick].sum() / N[pick].sum()) / st) - base)

    def ci(v):
        t = torch.tensor(v, dtype=torch.float64).sort().values
        return [round(float(t[max(0, int(0.025 * t.numel()))]), 5),
                round(float(t[min(t.numel() - 1, int(0.975 * t.numel()))]), 5)]
    return {k: ci(v) for k, v in out.items()}


@torch.no_grad()
def main():
    t0 = time.time()
    K = torch.load(CONSTS, map_location='cpu')
    ev = load(EVAL_ROWS)
    mask_rows = load(MASK_ROWS)
    seen = seen_mask(mask_rows)
    SEENREF['m'] = seen
    del mask_rows
    torch.cuda.empty_cache()
    fit = load(FIT_ROWS)

    CFG['v1'] = None
    V1P.pop('W', None)
    EXEMPT['set'] = frozenset()
    cl, SL, NL = ce_rows(ev, seen)
    assert abs(cl - S1683_CE_LIVE) <= 1e-3, (
        f'baseline CE {cl:.5f} disagrees with {S1683_CE_LIVE:.5f} (§1695)')
    hs = [H[L].mlp.register_forward_hook(mlp_const_hook(K[f'mlp{L}'].to(DEV).float()))
          for L in ALL18]
    hs += [H[L].attn.register_forward_hook(attn_const_hook(K[f'attn{L}'].to(DEV).float()))
           for L in ALL18]
    cc, SC, NC = ce_rows(ev, seen, hooks=hs)
    st = cc - cl
    assert st > 0, f'non-positive stake {st}'
    print(f'FRONT MLP SITE SHORTFALL | exempt-one at SITE grain over mlp0-3, RECOMPILED | '
          f'CE live {cl:.5f} | stake {st:.4f} | {NL.numel()} eval rows', flush=True)

    arms, arms_rows = {}, {}
    todo = [('none', frozenset())] + [
        (name, frozenset((kind, L) for L in sites)) for name, (kind, sites) in BANDS.items()]
    for name, ex in todo:
        EXEMPT['set'] = ex
        prog = compile_stack(fit, ('mlp', 'attn'))
        ct, S, N = ce_rows(ev, seen, hooks=install(prog))
        arms_rows[name] = (S, N)
        arms[name] = {'exempt_sites': len(ex), 'ce': round(ct, 5),
                      'ceiling': round((cc - ct) / st, 5), 'ceiling_exact': (cc - ct) / st}
        print(f'  exempt {name:11s} ({len(ex):2d} sites live) CEILING '
              f'{arms[name]["ceiling"]:8.2%}', flush=True)
        del prog
        torch.cuda.empty_cache()
    EXEMPT['set'] = frozenset()

    base = arms['none']['ceiling_exact']
    gains = {k: arms[k]['ceiling_exact'] - base for k in BANDS}
    vals = [arms[k]['ceiling_exact'] for k in arms]
    assert len(set(round(v, 9) for v in vals)) > 1, 'all arms identical -- exemption is a no-op'

    print('  bootstrapping band gains (2000 draws, ROW-level clusters)...', flush=True)
    ci_g = boot_gains(SL, NL, SC, NC, arms_rows, 'none')

    others = sum(gains[k] for k in ('mlp0', 'mlp2', 'mlp3'))
    total = sum(gains.values())
    top = max(gains, key=lambda k: gains[k]); bot = min(gains, key=lambda k: gains[k])

    pa = gains['mlp1'] > others
    pb = abs(total - S1703_FRONT_BAND_GAIN) <= 0.25 * S1703_FRONT_BAND_GAIN
    pc = (abs(base - S1696_BOTH) <= 0.005 and abs(cl - S1683_CE_LIVE) <= 1e-3
          and all(v > 0 for v in gains.values()))
    pd = ci_g[top][0] > ci_g[bot][1]

    print(f'\n  band gains (exempting the band, i.e. what leaving it REAL buys back):', flush=True)
    for k in sorted(gains, key=lambda k: -gains[k]):
        print(f'    {k:11s} {gains[k]:+7.2%}  95% CI [{ci_g[k][0]:+.2%}, {ci_g[k][1]:+.2%}]',
              flush=True)
    print(f'  mlp1 {gains["mlp1"]:+.2%} vs mlp0+mlp2+mlp3 {others:+.2%} -> mlp1 dominates {pa}',
          flush=True)
    print(f'  four site gains sum {total:+.2%} vs the band\'s own {S1703_FRONT_BAND_GAIN:+.2%} '
          f'-> roughly additive {pb}', flush=True)
    print(f'  (§1662 MLP-only table ceilings: ' +
          '  '.join(f'{k} {v:.0%}' for k, v in S1662_PER_SITE.items()) + ')', flush=True)
    print(f'  ordering resolved: {top} CI excludes {bot} CI -> {pd}', flush=True)
    print(f'  CONTROLS base {base:.2%} vs §1696 {S1696_BOTH:.2%}, all gains positive -> {pc}',
          flush=True)

    res = {'config': {'sites': ALL18, 'ridge': RIDGE,
                      'program': 'the §1696 best-families 36-site program (tables mlp0-2, lags 1,2,4,8)',
                      'method': 'exempt-one at band grain, RECOMPILED per exemption (§1671 pattern; '
                                'un-substituting without refitting measures LESSONS 28 compounding)',
                      'sites_probed': {k: [v[0], list(v[1])] for k, v in BANDS.items()},
                      'why_front_only': 'four sites is the only band small enough for site-level '
                                        'exempt-one at tractable cost (4 recompiles vs 12 for a middle band)',
                      's1703_front_band_gain': S1703_FRONT_BAND_GAIN,
                      's1662_per_site_mlp_only': S1662_PER_SITE,
                      'intervals': '95% ROW-level cluster bootstrap, 2000 paired draws (§1700, §1701); '
                                   'NOT document-clustered',
                      'compilation': 'INTERLEAVED bottom-up (§1669)',
                      'coverage': 'mask pinned to n96_skip80', 'scoring': 'covered positions only',
                      'fit_rows': 'fineweb_n480_skip80.pt', 'eval_rows': 'fineweb_n192_skip7000.pt',
                      's1696_both': S1696_BOTH},
           'ce_live': round(cl, 5), 'joint_stake': round(st, 5),
           'arms': {k: {kk: vv for kk, vv in v.items() if kk != 'ceiling_exact'}
                    for k, v in arms.items()},
           'band_gains': {k: round(v, 5) for k, v in gains.items()},
           'band_gain_ci95_rowlevel': ci_g,
           'mlp1_vs_others': [round(gains['mlp1'], 5), round(others, 5)],
           'sum_of_site_gains': round(total, 5),
           'predictions': {'pred_a_mlp1_dominates_front_band': bool(pa),
                           'pred_b_sites_roughly_additive': bool(pb),
                           'pred_c_controls_hold': bool(pc),
                           'pred_d_ordering_resolved': bool(pd)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
