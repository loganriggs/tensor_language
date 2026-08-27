# whole_model_v1_floor: THE RANK-0 ARM §1699 OWES — bounding the whole v1 path
#
# §1698 found rank-8 v1 costing 0.29 points inside the 36-site program and I wrote that v1 is
# "essentially eight-dimensional as far as the compiled program is concerned". §1699 withdrew
# that on Codex's review: without a rank-0 arm, "eight dimensions suffice" cannot be
# distinguished from "the whole v1 path contributes almost nothing inside a program that has
# already replaced the write". Both predict 0.29.
#
# The distinguishing arm is a CONSTANT v1 -- its position-weighted mean, carrying no per-token
# information at all. That bounds the entire path:
#   if v1_const costs about the same 0.29 as rank-8, the path contributes little here and no
#     dimensionality claim is licensed;
#   if v1_const costs much more, then rank 8 really was capturing most of a path that matters,
#     and the withdrawn phrase was closer to right than the review allowed.
#
# §1684 measured the v1 path at 0.7066 nats against a LIVE write path. This measures it against
# a SUBSTITUTED one, which is the condition the program claim is about, and the two are not the
# same quantity -- S1684's own finding was that v1 is nested inside the write.
#
# Registered predictions:
#   pred_a THE PATH, NOT THE RANK, EXPLAINS §1698: constant v1 costs less than 1.0 point against
#          v1_real. Combined with rank-8's 0.29 that would leave no room for a dimensionality
#          claim and confirm §1699's withdrawal.
#   pred_b IT IS NOT LITERALLY FREE: constant v1 costs >= 0.1 points, so the path is doing
#          something and the arm is not vacuous.
#   pred_c CONTROLS: v1_real reproduces §1696's 55.04% and v1_rank8 reproduces §1698's 54.75%,
#          both within 0.5 points, and the baseline CE reproduces 3.29205 (§1695).
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
ALL18 = list(range(0, 18))
RIDGE = 1e-3
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'whole_model_v1_floor_results.json'
FIT_ROWS = PT + '.rowcache/fineweb_n480_skip80.pt'
MASK_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
EVAL_ROWS = PT + '.rowcache/fineweb_n192_skip7000.pt'
CONSTS = PT + 'opt_ablation_consts_all.pt'
H = m.transformer.h
S1694_SIMPLE = 0.5094
S1696_BOTH = 0.5504
S1698_RANK8 = 0.5475
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
    """Interleaved bottom-up: within block L, attn_L then mlp_L (§1669)."""
    prog = {}
    for L in ALL18:
        for kind in ('attn', 'mlp'):
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
def ce(rows, seen, hooks=()):
    acc = {'t': 0.0, 'n': 0}

    def score(lg, tg, idx):
        e = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                            reduction='none').reshape(tg.shape)[:, 64:]
        cov = seen[idx[:, 64:]]
        acc['t'] += float(e[cov].sum()); acc['n'] += int(cov.sum())
    sweep(rows, hooks=hooks, score=score)
    return acc['t'] / max(acc['n'], 1)


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
    cl = ce(ev, seen)
    assert abs(cl - S1683_CE_LIVE) <= 1e-3, (
        f'baseline CE {cl:.5f} disagrees with {S1683_CE_LIVE:.5f} (§1695)')
    hs = [H[L].mlp.register_forward_hook(mlp_const_hook(K[f'mlp{L}'].to(DEV).float()))
          for L in ALL18]
    hs += [H[L].attn.register_forward_hook(attn_const_hook(K[f'attn{L}'].to(DEV).float()))
           for L in ALL18]
    cc = ce(ev, seen, hooks=hs)
    st = cc - cl
    print(f'WHOLE MODEL v1 FLOOR | the rank-0 arm §1699 owes | CE live {cl:.5f} | stake '
          f'{st:.4f} | targets: real {S1696_BOTH:.2%}, rank8 {S1698_RANK8:.2%}', flush=True)

    prog = compile_stack(fit, ('mlp', 'attn'))
    arms = {}
    for name, mode, rank in (('v1_real', None, D), ('v1_rank8', 'linear', 8),
                             ('v1_const', 'const', 0)):
        CFG['v1'] = None
        V1P.pop('W', None)
        if mode is not None:
            V1P['W'] = fit_v1(fit, prog, mode, rank)
        CFG['v1'] = mode
        ct = ce(ev, seen, hooks=install(prog))
        arms[name] = {'mode': mode, 'rank': rank, 'ce': round(ct, 5),
                      'ceiling': round((cc - ct) / st if st > 1e-6 else float('nan'), 5)}
        print(f'  {name:9s} mode {str(mode):7s}: CEILING {arms[name]["ceiling"]:8.2%}', flush=True)
    CFG['v1'] = None
    V1P.pop('W', None)

    real = arms['v1_real']['ceiling']
    r8 = arms['v1_rank8']['ceiling']
    const = arms['v1_const']['ceiling']
    assert abs(const - real) > 1e-9, 'constant v1 identical to real -- the arm is a no-op'

    cost_const = real - const
    cost_r8 = real - r8
    pa = cost_const < 0.01
    pb = cost_const >= 0.001
    pc = (abs(real - S1696_BOTH) <= 0.005 and abs(r8 - S1698_RANK8) <= 0.005
          and abs(cl - S1683_CE_LIVE) <= 1e-3)

    print(f'\n  cost of removing v1 entirely (constant): {cost_const:.2%} | cost at rank 8: '
          f'{cost_r8:.2%}', flush=True)
    print(f'  whole path under 1 point -> no dimensionality claim licensed {pa}', flush=True)
    print(f'  not literally free {pb} | CONTROLS {pc}', flush=True)
    print(f'  rank-8 recovers {1 - cost_r8 / cost_const:.1%} of the path\'s contribution'
          if cost_const > 1e-9 else '  path contributes nothing measurable', flush=True)

    res = {'config': {'sites': ALL18, 'ridge': RIDGE,
                      'program': 'the §1696 best-families 36-site program',
                      'question': "S1699 -- does rank-8 v1's 0.29-point cost mean eight dimensions "
                                  "suffice, or that the whole v1 path contributes little inside a "
                                  "program that already replaces the write? A constant-v1 arm bounds it.",
                      'v1_hybrid_scope': 'covered positions only; uncovered occurrences retain native v1 (§1699)',
                      'not_the_same_as_S1684': "S1684's 0.7066 nats was v1 against a LIVE write path; "
                                               "this is v1 against a SUBSTITUTED one",
                      'compilation': 'INTERLEAVED bottom-up (§1669)',
                      'coverage': 'mask pinned to n96_skip80', 'scoring': 'covered positions only',
                      'fit_rows': 'fineweb_n480_skip80.pt', 'eval_rows': 'fineweb_n192_skip7000.pt',
                      's1696_both': S1696_BOTH, 's1698_rank8': S1698_RANK8},
           'ce_live': round(cl, 5), 'joint_stake': round(st, 5), 'arms': arms,
           'cost_of_constant_v1': round(cost_const, 5), 'cost_of_rank8_v1': round(cost_r8, 5),
           'predictions': {'pred_a_whole_path_under_1pt': bool(pa),
                           'pred_b_not_literally_free': bool(pb),
                           'pred_c_controls_hold': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
