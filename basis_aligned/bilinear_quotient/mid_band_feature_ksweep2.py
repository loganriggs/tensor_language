# mid_band_feature_ksweep2: FINDING THE SHAPE, WITH A SEPARATE EXACT KNOWN-ANSWER ARM
#
# §1710 took the middle-band feature augmentation to k=512 and the curve was still steep: +0.77 at
# 64, +2.34 at 256, +3.67 at 512, giving a best whole-model program of 58.71%. Its pred_c --
# "returns are diminishing by now" -- FAILED, and the docstring stated the consequence in advance:
# the family is still in its steep region and the sweep has not found the shape.
#
# CORRECTION BEFORE THE VALID ATTEMPT. The first committed draft incorrectly called the ridge-fitted
# k=4608 arm an identity: [x,h]W has no intercept and RIDGE shrinks even a representable solution,
# so that arm is empirical. The actual known-answer arm below is separate: at every middle MLP it
# constructs the exact native map Down(Left(x)*Right(x)) + Down_bias inside the SAME
# interleaved compiler. It must equal leaving mlp4-15 real (§1703's 67.55%) up to run-to-run bars.
# A miss invalidates hook/feature/compiler construction; it does not retroactively make a ridge
# approximation at lower k algebraically false.
#
# k in {512, 1024, 2048, 4608}. 512 is carried over as a direct replication of §1710's +3.67 in the
# same run, so the sweep is anchored at the bottom as well as pinned at the top.
#
# COST NOTE, stated because it is the largest run of this arc: at k=4608 the augmented input is
# 5760-dimensional and the normal equations are 5760x5760 in float64 per site, ~265MB, over twelve
# sites and four arms. This is expected to be slow.
#
# Registered predictions, TWO-SIDED per LESSONS 31:
#   pred_a IDENTITY CHECK: the separately CONSTRUCTED exact arm lands within 1.0 point of §1703's
#          67.55% band-exempt ceiling. This is derivable; the ridge k=4608 arm is not.
#   pred_b THE SHAPE FINALLY BENDS: the 512->4608 step is smaller than 9x the 512-point gain would
#          imply under continued log-linearity -- concretely, the gain from 512 to 4608 is less
#          than 2x the gain from 64 to 512 (+2.91). Failure means the ridge curve is still steep
#          at the top of the feature basis; the separate exact arm does not force this shape.
#   pred_c 512 REPLICATES: its gain lands within 0.2 points of §1710's +3.67.
#   pred_d CONTROLS: the k=0 arm reproduces §1696's 55.04% within 0.5 points and the baseline CE
#          reproduces 3.29205 (§1695).
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
ALL18 = list(range(0, 18))
RIDGE = 1e-3
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mid_band_feature_ksweep2_results.json'
FIT_ROWS = PT + '.rowcache/fineweb_n480_skip80.pt'
MASK_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
EVAL_ROWS = PT + '.rowcache/fineweb_n192_skip7000.pt'
CONSTS = PT + 'opt_ablation_consts_all.pt'
H = m.transformer.h
S1694_SIMPLE = 0.5094
S1696_BOTH = 0.5504
S1698_RANK8 = 0.5475
MID = list(range(4, 16))
DH = 4608
KS = [0, 512, 1024, 2048, 4608]
S1703_MID_BAND_GAIN = 0.12515    # exempting mlp4-15 entirely
S1703_BAND_EXEMPT_CEILING = 0.67553   # leaving mlp4-15 REAL; the k=4608 known answer
S1710_K512_GAIN = 0.03672
S1710_K64_TO_K512 = 0.02905
FEAT = {'k': 0, 'sel': {}, 'mode': 'ridge'}
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


def feats(mlp, x, keep):
    """The module's own bilinear features, restricted to the kept indices."""
    return (mlp.Left(x) * mlp.Right(x))[:, keep]


def mlp_prog_hook(W, L=None):
    def hook(mod, args, out):
        xin = args[0].reshape(-1, D)
        if L is not None and L in FEAT['sel'] and FEAT['k'] > 0:
            xin = torch.cat([xin, feats(H[L].mlp, xin, FEAT['sel'][L])], dim=-1)
        sub = (xin @ W).reshape(out.shape).to(out.dtype)
        return torch.where(SEENREF['m'][STATE['idx']].unsqueeze(-1), sub, out)
    return hook


def exact_mlp_hook(L):
    """Construct the module's exact bilinear map; independent of its prior hook output."""
    def hook(mod, args, out):
        xin = args[0].reshape(-1, D)
        hidden = mod.Left(xin) * mod.Right(xin)
        exact = (mod.Down(hidden) + mod.Down_bias).reshape(out.shape).to(out.dtype)
        return torch.where(SEENREF['m'][STATE['idx']].unsqueeze(-1), exact, out)
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
            if isinstance(W, tuple) and W == ('exact_mlp', L):
                hook = exact_mlp_hook(L)
            else:
                hook = table_hook(W) if L in CFG['tables'] else mlp_prog_hook(W, L)
            hs.append(H[L].mlp.register_forward_hook(hook))
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
def select_feats(rows, L, prog):
    """Top-k features by std(h_j)*||Down[:, j]||, with the stack below substituted (§1678)."""
    mlp = H[L].mlp
    s1 = torch.zeros(DH, device=DEV, dtype=torch.float64)
    s2_ = torch.zeros(DH, device=DEV, dtype=torch.float64)
    n = {'v': 0}

    def collect(mod, args, out):
        h = (mlp.Left(args[0].reshape(-1, D)) * mlp.Right(args[0].reshape(-1, D))).double()
        s1.add_(h.sum(0)); s2_.add_((h * h).sum(0)); n['v'] += h.shape[0]
        return None
    sweep(rows, hooks=install(prog) + [mlp.register_forward_hook(collect)])
    assert n['v'] > 0, f'mlp{L}: no positions for feature selection'
    mean = s1 / n['v']
    var = (s2_ / n['v'] - mean * mean).clamp_min(0)
    score = var.sqrt() * mlp.Down.weight.double().norm(dim=0)
    return torch.topk(score, min(FEAT['k'], DH)).indices.sort().values


@torch.no_grad()
def fit_site(rows, kind, L, prog):
    if kind == 'mlp' and L in CFG['tables']:
        return fit_table(rows, L, prog)
    if kind == 'mlp' and L in MID and FEAT['mode'] == 'exact':
        return ('exact_mlp', L)
    aug = (kind == 'mlp' and FEAT['k'] > 0 and L in MID)
    if aug:
        FEAT['sel'][L] = select_feats(rows, L, prog)
    din = (D + FEAT['k']) if aug else (D if kind == 'mlp' else D * (1 + len(CFG['lags'])))
    A = torch.zeros(din, din, device=DEV, dtype=torch.float64)
    B = torch.zeros(din, D, device=DEV, dtype=torch.float64)
    n = {'v': 0}

    def collect(mod, args, out):
        xr = args[0].reshape(-1, D)
        if aug:
            x = torch.cat([xr, feats(H[L].mlp, xr, FEAT['sel'][L])], dim=-1).double()
        else:
            x = (xr if kind == 'mlp' else lagged(args[0])).double()
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
    FEAT['k'] = 0; FEAT['sel'] = {}; FEAT['mode'] = 'ridge'
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
    print(f'MID BAND FEATURE K-SWEEP 2 | linear + k own features + separate exact identity | '
          f'mlp{MID[0]}-{MID[-1]} | k in {KS} | CE live {cl:.5f} | stake {st:.4f}', flush=True)
    print(f'  §1703: exempting this band entirely buys +{S1703_MID_BAND_GAIN:.2%}', flush=True)

    arms, arms_rows = {}, {}
    for k in KS:
        FEAT['k'] = k; FEAT['sel'] = {}; FEAT['mode'] = 'ridge'
        prog = compile_stack(fit, ('mlp', 'attn'))
        ct, S, N = ce_rows(ev, seen, hooks=install(prog))
        name = f'k{k}'
        arms_rows[name] = (S, N)
        arms[name] = {'k': k, 'ce': round(ct, 5), 'ceiling': round((cc - ct) / st, 5),
                      'ceiling_exact': (cc - ct) / st,
                      'extra_reals_per_site': k * D}
        print(f'  k {k:4d}: CEILING {arms[name]["ceiling"]:8.2%}', flush=True)
        del prog
        torch.cuda.empty_cache()

    # Algebraic known answer, intentionally distinct from ridge-fitted k=4608.
    FEAT['k'] = DH; FEAT['sel'] = {}; FEAT['mode'] = 'exact'
    exact_prog = compile_stack(fit, ('mlp', 'attn'))
    exact_ce, exact_S, exact_N = ce_rows(ev, seen, hooks=install(exact_prog))
    arms_rows['exact4608'] = (exact_S, exact_N)
    arms['exact4608'] = {
        'k': DH, 'ce': round(exact_ce, 5),
        'ceiling': round((cc - exact_ce) / st, 5),
        'ceiling_exact': (cc - exact_ce) / st,
        'construction': 'exact Down(Left(x)*Right(x)) + Down_bias',
    }
    print(f'  exact constructed: CEILING {arms["exact4608"]["ceiling"]:8.2%}', flush=True)
    del exact_prog
    torch.cuda.empty_cache()
    FEAT['k'] = 0; FEAT['sel'] = {}; FEAT['mode'] = 'ridge'

    base = arms['k0']['ceiling_exact']
    g512 = arms['k512']['ceiling_exact'] - base
    full_ridge = arms['k4608']['ceiling_exact']
    exact = arms['exact4608']['ceiling_exact']
    vals = [a['ceiling_exact'] for a in arms.values()]
    assert len(set(round(v, 9) for v in vals)) > 1, 'all arms identical -- augmentation is a no-op'

    print('  bootstrapping gains (2000 draws, ROW-level clusters)...', flush=True)
    ci_g = boot_gains(SL, NL, SC, NC, arms_rows, 'k0')

    g4608 = full_ridge - base
    pa = abs(exact - S1703_BAND_EXEMPT_CEILING) <= 0.010
    pb = (g4608 - g512) < 2.0 * S1710_K64_TO_K512
    pc = abs(g512 - S1710_K512_GAIN) <= 0.002
    pd = (abs(base - S1696_BOTH) <= 0.005 and abs(cl - S1683_CE_LIVE) <= 1e-3)

    for k in KS[1:]:
        g = arms[f'k{k}']['ceiling_exact'] - base
        print(f'  k={k:5d} gain {g:+.2%}  95% CI [{ci_g[f"k{k}"][0]:+.2%}, '
              f'{ci_g[f"k{k}"][1]:+.2%}]  ceiling {arms[f"k{k}"]["ceiling"]:.2%}  '
              f'{k * D * len(MID) / 1e6:.2f}M extra reals total', flush=True)
    print(f'\n  IDENTITY CHECK exact constructed: {exact:.2%} vs the derivable '
          f'{S1703_BAND_EXEMPT_CEILING:.2%} '
          f'(band left REAL) -> {pa}', flush=True)
    print(f'  512 replicates §1710 (+{S1710_K512_GAIN:.2%}) -> {pc}', flush=True)
    print(f'  512->4608 step {g4608 - g512:+.2%} vs 2x the 64->512 step '
          f'({2 * S1710_K64_TO_K512:+.2%}) -> shape bends {pb}', flush=True)
    print(f'  fraction of the band gain +{S1703_MID_BAND_GAIN:.2%} recovered: ' +
          '  '.join(f'k{k} {(arms[f"k{k}"]["ceiling_exact"] - base) / S1703_MID_BAND_GAIN:.0%}'
                    for k in KS[1:]), flush=True)
    print(f'  CONTROL k0 {base:.2%} vs §1696 {S1696_BOTH:.2%} -> {pd}', flush=True)

    res = {'config': {'mid_sites': MID, 'ks': KS, 'hidden_dim': DH, 'ridge': RIDGE,
                      'family': 'y ~ [x, h_k] W, where h_k are the k features of the module OWN '
                                'bilinear form with the largest std(h_j)*||Down[:,j]|| -- a linear '
                                'map plus k quadratic corrections in the module basis',
                      'applied_at': 'mlp4-15 only, inside the §1696 best-families 36-site program',
                      'why': '§1703 puts the largest total shortfall here and §1668 puts 37.7% of the '
                             'band beyond any linear map; §1667/§1677 show a token term does not reach it',
                      'caution_S1679': 'keeping k features and PINNING the rest replaced the module and '
                                       'failed at -49.93%; here the linear map carries the bulk and the '
                                       'features add a correction, so the cancellation structure is not '
                                       'broken the same way -- an argument, not a result',
                      'compilation': 'INTERLEAVED bottom-up (§1669)',
                      'coverage': 'hybrid, mask pinned to n96_skip80', 'scoring': 'covered positions only',
                      'fit_rows': 'fineweb_n480_skip80.pt', 'eval_rows': 'fineweb_n192_skip7000.pt',
                      's1703_mid_band_gain': S1703_MID_BAND_GAIN, 's1696_both': S1696_BOTH},
           'ce_live': round(cl, 5), 'joint_stake': round(st, 5),
           'arms': {k: {kk: vv for kk, vv in v.items() if kk != 'ceiling_exact'}
                    for k, v in arms.items()},
           'gains': {f'k{k}': round(arms[f'k{k}']['ceiling_exact'] - base, 5) for k in KS[1:]},
           'gain_ci95_rowlevel': ci_g,
           'ridge_k4608': round(full_ridge, 5),
           'identity_exact_constructed': round(exact, 5),
           'identity_target_s1703': S1703_BAND_EXEMPT_CEILING,
           'fraction_of_band_recovered': {f'k{k}': round((arms[f'k{k}']['ceiling_exact'] - base)
                                                         / S1703_MID_BAND_GAIN, 4) for k in KS[1:]},
           'predictions': {'pred_a_identity_check_k4608': bool(pa),
                           'pred_b_shape_bends': bool(pb),
                           'pred_c_k512_replicates_s1710': bool(pc),
                           'pred_d_controls_hold': bool(pd)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
