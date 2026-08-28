# mid_band_feature_heldout: DOES THE PRICE CURVE HOLD ON DOCUMENTS IT HAS NEVER SEEN?
#
# §1714/§1716 established the middle-band feature price curve and validated its construction
# bit-exactly. Every figure in it was scored on ONE eval set, fineweb_n192_skip7000, and no
# augmented arm has ever been replicated.
#
# This matters more than it usually would, because the price curve is a claim about GAINS
# (+3.675 at k=512, and the proportionality that follows from it), not about levels. The three
# prior replications in this arc all found the same shape:
#
#   §1683  four MLP-program arms      levels moved -0.14 to -0.91 points
#   §1693  four compressibility paths levels moved -0.51 to +0.74, ordering identical
#   §1701  four whole-model arms      levels moved -1.23 to -1.34, GAINS moved <= 0.12
#
# So the prediction is that the level drops by something like a point while the gain survives. If
# the GAIN moves materially, the price curve is a property of those 192 documents rather than of
# bilin18, and §1714's proportionality reading goes with it.
#
# Two arms, k=0 and k=512, compiled ONCE on n480_skip80 with the coverage mask pinned to
# n96_skip80 (§1676), then scored on both eval sets. k=512 is chosen over the larger k because it
# is the practical point on the curve -- 29.4% of the band shortfall for 0.3x the base program --
# and because it is the arm §1710 and §1714 both report, so a drift would be unambiguous.
#
# SCOPE, per §1700: skip11000 was exposed to component-level experiments earlier in this arc. It
# is unseen by THESE programs, which is what the claim needs, but this is prospective conditional
# replication and not fresh out-of-distribution evidence.
#
# Registered predictions, TWO-SIDED per LESSONS 31:
#   pred_a THE GAIN REPLICATES, which is the load-bearing one: the k=512 gain on skip11000 lands
#          within 1.0 point of skip7000's +3.675. This can fail from either side.
#   pred_b THE LEVEL MOVES LIKE ITS PREDECESSORS: the k=512 ceiling drops by between 0.3 and 3.0
#          points. A drop far outside that band means this arm does not behave like the four in
#          §1701, and the comparison to them would not be available.
#   pred_c CONTROLS: on skip7000 the k=0 arm reproduces §1714's 55.038% and the k=512 arm its
#          58.713%, both within 0.5 points, and the baseline CE reproduces 3.29205 (§1695).
#   pred_d SPREAD: the held-out gain's 95% interval excludes zero.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
ALL18 = list(range(0, 18))
RIDGE = 1e-3
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mid_band_feature_heldout_results.json'
FIT_ROWS = PT + '.rowcache/fineweb_n480_skip80.pt'
MASK_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt'),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt')]
EVAL_ROWS = EVAL_SETS[0][1]
CONSTS = PT + 'opt_ablation_consts_all.pt'
H = m.transformer.h
S1694_SIMPLE = 0.5094
S1696_BOTH = 0.5504
S1698_RANK8 = 0.5475
MID = list(range(4, 16))
DH = 4608
KS = [0, 512]
S1714_K0 = 0.55038
S1714_K512 = 0.58713
S1714_K512_GAIN = 0.03675
FEAT = {'k': 0, 'sel': {}}
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
                table_hook(W) if L in CFG['tables'] else mlp_prog_hook(W, L)))
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
    """Paired row-level cluster bootstrap of each arm's gain over the base arm."""
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
    mask_rows = load(MASK_ROWS)
    seen = seen_mask(mask_rows)
    SEENREF['m'] = seen
    del mask_rows
    torch.cuda.empty_cache()
    fit = load(FIT_ROWS)
    print(f'MID BAND FEATURE HELD-OUT | k in {KS} at mlp{MID[0]}-{MID[-1]} | compile ONCE, score '
          f'on {[n for n, _ in EVAL_SETS]}', flush=True)
    print(f'  §1714 reference on skip7000: k0 {S1714_K0:.3%}  k512 {S1714_K512:.3%}  '
          f'gain +{S1714_K512_GAIN:.3%}', flush=True)

    # The feature-selection indices live in the GLOBAL FEAT['sel'], not in the returned
    # program. The first version of this script cleared FEAT['sel'] between compiling and
    # scoring, on a comment asserting the indices "live in the stored program" -- they do not,
    # and the k=512 arm crashed with 2048x1152 @ 1664x1152. Store the selection alongside the
    # program and restore both before every scoring pass.
    progs = {}
    for k in KS:
        FEAT['k'] = k; FEAT['sel'] = {}
        prog = compile_stack(fit, ('mlp', 'attn'))
        progs[k] = (prog, dict(FEAT['sel']))
        print(f'  compiled k={k} (selection on {len(FEAT["sel"])} sites)', flush=True)
    FEAT['k'] = 0; FEAT['sel'] = {}
    del fit
    torch.cuda.empty_cache()

    out = {}
    for ename, epath in EVAL_SETS:
        ev = load(epath)
        FEAT['k'] = 0; FEAT['sel'] = {}
        cl, SL, NL = ce_rows(ev, seen)
        if ename == 'skip7000':
            assert abs(cl - S1683_CE_LIVE) <= 1e-3, (
                f'baseline CE {cl:.5f} disagrees with {S1683_CE_LIVE:.5f} (§1695)')
        hs = [H[L].mlp.register_forward_hook(mlp_const_hook(K[f'mlp{L}'].to(DEV).float()))
              for L in ALL18]
        hs += [H[L].attn.register_forward_hook(attn_const_hook(K[f'attn{L}'].to(DEV).float()))
               for L in ALL18]
        cc, SC, NC = ce_rows(ev, seen, hooks=hs)
        st = cc - cl
        assert st > 0, f'{ename}: non-positive stake {st}'
        row = {'ce_live': round(cl, 5), 'stake': round(st, 5)}
        arms_rows = {}
        for k in KS:
            prog, sel = progs[k]
            FEAT['k'] = k; FEAT['sel'] = sel
            assert (k == 0) or len(sel) == len(MID), (
                f'k={k}: selection restored for {len(sel)} sites, expected {len(MID)}')
            ct, S, N = ce_rows(ev, seen, hooks=install(prog))
            arms_rows[f'k{k}'] = (S, N)
            row[f'k{k}'] = round((cc - ct) / st, 5)
        FEAT['k'] = 0; FEAT['sel'] = {}
        row['gain_k512'] = round(row['k512'] - row['k0'], 5)
        row['gain_ci95'] = boot_gains(SL, NL, SC, NC, arms_rows, 'k0')['k512']
        print(f'  {ename:10s} CE live {cl:.5f} | stake {st:.4f} | k0 {row["k0"]:.3%} | '
              f'k512 {row["k512"]:.3%} | gain {row["gain_k512"]:+.3%} '
              f'95% CI [{row["gain_ci95"][0]:+.3%}, {row["gain_ci95"][1]:+.3%}]', flush=True)
        out[ename] = row
        del ev
        torch.cuda.empty_cache()

    ref, held = out['skip7000'], out['skip11000']
    assert abs(ref['k512'] - held['k512']) > 1e-9, 'eval sets identical -- scoring is a no-op'
    dgain = held['gain_k512'] - ref['gain_k512']
    dlevel = ref['k512'] - held['k512']
    lo, hi = held['gain_ci95']

    pa = abs(dgain) <= 0.010
    pb = 0.003 <= dlevel <= 0.030
    pc = (abs(ref['k0'] - S1714_K0) <= 0.005 and abs(ref['k512'] - S1714_K512) <= 0.005
          and abs(ref['ce_live'] - S1683_CE_LIVE) <= 1e-3)
    pd = (lo > 0) or (hi < 0)

    print(f'\n  GAIN  skip7000 {ref["gain_k512"]:+.3%} -> skip11000 {held["gain_k512"]:+.3%}  '
          f'delta {dgain:+.3%} -> replicates {pa}', flush=True)
    print(f'  LEVEL skip7000 {ref["k512"]:.3%} -> skip11000 {held["k512"]:.3%}  '
          f'drop {dlevel:+.3%} -> moves like §1701 {pb}', flush=True)
    print(f'  held-out gain CI excludes zero {pd} | controls {pc}', flush=True)

    res = {'config': {'mid_sites': MID, 'ks': KS, 'ridge': RIDGE,
                      'eval_sets': [n for n, _ in EVAL_SETS],
                      'held_out': 'fineweb_n192_skip11000 -- unseen by THESE programs; exposed to '
                                  'component-level experiments earlier in the arc, so this is '
                                  'prospective conditional replication, not fresh OOD (§1700)',
                      'compilation': 'INTERLEAVED bottom-up (§1669); compiled ONCE, scored on both evals',
                      'coverage': 'mask pinned to n96_skip80 (§1676)',
                      'fit_rows': 'fineweb_n480_skip80.pt',
                      'prior_replications': {'S1683': 'levels -0.14 to -0.91',
                                             'S1693': 'levels -0.51 to +0.74, ordering identical',
                                             'S1701': 'levels -1.23 to -1.34, gains <= 0.12'},
                      's1714_reference': {'k0': S1714_K0, 'k512': S1714_K512,
                                          'gain': S1714_K512_GAIN}},
           'evals': out, 'gain_delta': round(dgain, 5), 'level_drop': round(dlevel, 5),
           'predictions': {'pred_a_gain_replicates': bool(pa),
                           'pred_b_level_moves_like_s1701': bool(pb),
                           'pred_c_controls_hold': bool(pc),
                           'pred_d_gain_ci_excludes_zero': bool(pd)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
