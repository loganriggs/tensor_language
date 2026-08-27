# whole_model_heldout: THE WHOLE-MODEL PROGRAM ON DOCUMENTS IT HAS NEVER SEEN
#
# The arc's headline is now a single number -- a 36-site compiled program reproducing 55.04% of
# bilin18's 5.5684-nat joint stake (§1696), with v1 included for free (§1698) and the shortfall
# attributed to cross-half compounding rather than redundancy (§1697). Every one of those
# figures was scored on fineweb_n192_skip7000.
#
# The two halves have each been confirmed held-out separately -- §1683 for the MLP program arms,
# §1693 for the compressibility ordering -- but the JOINT program never has, and it is the
# object all the later conclusions are stated about. §1697's damping factors and §1696's
# transfer discount are both differences between joint-condition ceilings; if those ceilings
# move under a document resample, the discount numbers move with them.
#
# Rung 2, house second-class-confirmation pattern (§1595, §1598, §1603). Programs compiled ONCE
# on n480_skip80 with the mask pinned, then scored on both eval sets. Only the scoring documents
# change.
#
# ARMS, the four the arc's conclusions rest on, all with v1 tabled per §1698:
#   simple            linear everywhere, lag-1        §1694/§1697 baseline
#   attn_upgraded     lags (1,2,4,8)                  §1697's +2.66 arm
#   mlp_upgraded      tables at mlp0-2                §1697's +1.31 arm
#   both              the §1696 program               the headline, 55.04%
#
# The stake is recomputed per eval set because it is a property of the eval documents (§1683
# found the MLP stake moving 4.3301 -> 4.5173 between these two); the ceiling is a ratio within
# its own set, which is what makes them comparable.
#
# Registered predictions:
#   pred_a THE HEADLINE HOLDS: the both arm on skip11000 is within 3 points of 55.04%.
#   pred_b THE GAIN STRUCTURE HOLDS, which is what §1696 and §1697 actually use: on the held-out
#          set the attention gain still exceeds the MLP gain, and the two singles still sum to
#          within 1 point of the joint gain (§1697's additivity, which was 3.97 vs 4.10).
#   pred_c CONTROLS: the skip7000 arm reproduces §1696's 55.04% and §1697's simple 50.94%, both
#          within 0.5 points, and the baseline CE reproduces 3.29205 (§1695).
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
ALL18 = list(range(0, 18))
RIDGE = 1e-3
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'whole_model_heldout_results.json'
FIT_ROWS = PT + '.rowcache/fineweb_n480_skip80.pt'
MASK_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt'),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt')]
EVAL_ROWS = EVAL_SETS[0][1]
CONSTS = PT + 'opt_ablation_consts_all.pt'
H = m.transformer.h
S1694_SIMPLE = 0.5094
S1696_BOTH = 0.5504
S1697_GAINS = {'attn': 0.0266, 'mlp': 0.0131, 'both': 0.0410}
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
    mask_rows = load(MASK_ROWS)
    seen = seen_mask(mask_rows)
    SEENREF['m'] = seen
    del mask_rows
    torch.cuda.empty_cache()
    fit = load(FIT_ROWS)
    print(f'WHOLE MODEL HELD-OUT | compile ONCE, score on {[n for n, _ in EVAL_SETS]} | '
          f'v1 tabled (§1698) | §1696 headline {S1696_BOTH:.2%}', flush=True)

    SPEC = (('simple', (1,), ()), ('attn_upgraded', ATTN_LAGS, ()),
            ('mlp_upgraded', (1,), S1672_MLP_TABLE_SITES),
            ('both', ATTN_LAGS, S1672_MLP_TABLE_SITES))
    progs, v1s = {}, {}
    for name, lags, tables in SPEC:
        CFG['lags'], CFG['tables'], CFG['v1'] = lags, tables, None
        V1P.pop('W', None)
        progs[name] = compile_stack(fit, ('mlp', 'attn'))
        v1s[name] = fit_v1(fit, progs[name], 'table', D)
        print(f'  compiled {name}', flush=True)
    del fit
    torch.cuda.empty_cache()

    out = {}
    for ename, epath in EVAL_SETS:
        ev = load(epath)
        CFG['lags'], CFG['tables'], CFG['v1'] = (1,), (), None
        V1P.pop('W', None)
        cl = ce(ev, seen)
        if ename == 'skip7000':
            assert abs(cl - S1683_CE_LIVE) <= 1e-3, (
                f'baseline CE {cl:.5f} disagrees with {S1683_CE_LIVE:.5f} (§1695)')
        hs = [H[L].mlp.register_forward_hook(mlp_const_hook(K[f'mlp{L}'].to(DEV).float()))
              for L in ALL18]
        hs += [H[L].attn.register_forward_hook(attn_const_hook(K[f'attn{L}'].to(DEV).float()))
               for L in ALL18]
        cc = ce(ev, seen, hooks=hs)
        st = cc - cl
        row = {'ce_live': round(cl, 5), 'stake': round(st, 5)}
        print(f'  {ename:10s} CE live {cl:.5f} | joint stake {st:.4f} nats', flush=True)
        for name, lags, tables in SPEC:
            CFG['lags'], CFG['tables'] = lags, tables
            V1P['W'] = v1s[name]
            CFG['v1'] = 'table'
            ct = ce(ev, seen, hooks=install(progs[name]))
            row[name] = round((cc - ct) / st if st > 1e-6 else float('nan'), 5)
            print(f'      {name:14s} CEILING {row[name]:8.2%}', flush=True)
        CFG['v1'] = None
        V1P.pop('W', None)
        out[ename] = row
        del ev
        torch.cuda.empty_cache()

    ref, held = out['skip7000'], out['skip11000']
    names = [n for n, _, _ in SPEC]
    assert len(set(held[n] for n in names)) > 1, 'all held-out arms identical -- switch is a no-op'

    def gains(r):
        return {'attn': r['attn_upgraded'] - r['simple'],
                'mlp': r['mlp_upgraded'] - r['simple'],
                'both': r['both'] - r['simple']}
    gr, gh = gains(ref), gains(held)

    pa = abs(held['both'] - S1696_BOTH) <= 0.03
    pb = (gh['attn'] > gh['mlp']) and (abs((gh['attn'] + gh['mlp']) - gh['both']) <= 0.01)
    pc = (abs(ref['both'] - S1696_BOTH) <= 0.005 and abs(ref['simple'] - 0.5094) <= 0.005
          and abs(ref['ce_live'] - S1683_CE_LIVE) <= 1e-3)

    print(f'\n  ARM-BY-ARM reference -> held out:', flush=True)
    for n in names:
        print(f'    {n:14s} {ref[n]:7.2%} -> {held[n]:7.2%}   {held[n] - ref[n]:+.2%}', flush=True)
    print(f'  gains held out: attn {gh["attn"]:+.2%} | mlp {gh["mlp"]:+.2%} | both '
          f'{gh["both"]:+.2%} | singles sum {gh["attn"] + gh["mlp"]:+.2%}', flush=True)
    print(f'  (§1697 on skip7000: attn {S1697_GAINS["attn"]:+.2%} | mlp '
          f'{S1697_GAINS["mlp"]:+.2%} | both {S1697_GAINS["both"]:+.2%})', flush=True)
    print(f'  headline holds {pa} | gain structure holds {pb} | controls {pc}', flush=True)

    res = {'config': {'sites': ALL18, 'ridge': RIDGE,
                      'eval_sets': [n for n, _ in EVAL_SETS],
                      'held_out': 'fineweb_n192_skip11000 -- the JOINT program has never been scored on it',
                      'arms': [{'name': n, 'lags': list(l), 'tables': list(t)} for n, l, t in SPEC],
                      'v1': 'per-token table on every arm (§1698)',
                      'compilation': 'INTERLEAVED bottom-up (§1669); compiled ONCE, scored on both evals',
                      'stake': 'recomputed per eval set (§1683 found it moving between these two)',
                      'pattern': 'house second-class confirmation (§1595, §1598, §1603)',
                      'coverage': 'mask pinned to n96_skip80', 'fit_rows': 'fineweb_n480_skip80.pt',
                      's1696_both': S1696_BOTH, 's1697_gains': S1697_GAINS},
           'evals': out,
           'gains': {'reference': {k: round(v, 5) for k, v in gr.items()},
                     'held_out': {k: round(v, 5) for k, v in gh.items()}},
           'deltas': {n: round(held[n] - ref[n], 5) for n in names},
           'predictions': {'pred_a_headline_within_3pts': bool(pa),
                           'pred_b_gain_structure_holds': bool(pb),
                           'pred_c_controls_hold': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
