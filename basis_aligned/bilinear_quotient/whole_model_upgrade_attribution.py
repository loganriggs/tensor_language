# whole_model_upgrade_attribution: COMPOUNDING OR REDUNDANCY? — separating §1696's two mechanisms
#
# §1696 found that upgrading both halves to their best families carries only 4.10 points into the
# whole-model program (50.94% -> 55.04%), where the attention half alone gains 11.79 under the
# same lag set. Roughly two thirds of the half-level improvement does not survive. I named two
# mechanisms there and said I could not separate them:
#
#   COMPOUNDING -- the improved attention program's residual error still propagates through
#                  eighteen MLP programs, so a better attention half cannot be fully exploited.
#   REDUNDANCY  -- what the wider lag set recovers is partly information the MLP programs were
#                  already reconstructing from the stream, so recovering it twice buys less.
#
# They make different predictions about UPGRADING ONE HALF AT A TIME inside the joint condition:
#
#   if COMPOUNDING dominates, each half's upgrade is independently damped by the other half's
#     error, so the two single upgrades should each be small and their gains should roughly ADD
#     to the both-upgraded gain;
#   if REDUNDANCY dominates, either upgrade alone captures much of the shared information, so
#     each single upgrade should be a LARGE fraction of the joint gain and the two should be
#     strongly SUB-ADDITIVE.
#
# ARMS, all 36 sites, interleaved bottom-up, identical except which half is upgraded:
#   simple        linear everywhere, lag-1                    CONTROL, must reproduce §1694 50.94%
#   attn_only     lag-(1,2,4,8), MLPs still plain linear
#   mlp_only      tables at mlp0-2, attention still lag-1
#   both          the §1696 program                           CONTROL, must reproduce 55.04%
#
# §1657's caution applies and is honoured: these are gains at a single total effect size, compared
# with each other, not ratios compared across conditions of different size.
#
# Registered predictions:
#   pred_a REDUNDANCY IS THE LARGER EFFECT: the two single-upgrade gains SUM to more than the
#          both-upgraded gain (sum > joint gain, i.e. sub-additive). Under pure compounding they
#          would add to roughly the joint gain instead.
#   pred_b THE ATTENTION UPGRADE IS THE BIGGER SINGLE CONTRIBUTOR: its gain exceeds the MLP
#          upgrade's. The attention half gained 11.79 alone against the MLP half's <= 3.5.
#   pred_c CONTROLS: baseline CE within 1e-3 of 3.29205, the simple arm within 1 point of 50.94%,
#          and the both arm within 1 point of 55.04%.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
ALL18 = list(range(0, 18))
RIDGE = 1e-3
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'whole_model_upgrade_attribution_results.json'
FIT_ROWS = PT + '.rowcache/fineweb_n480_skip80.pt'
MASK_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
EVAL_ROWS = PT + '.rowcache/fineweb_n192_skip7000.pt'
CONSTS = PT + 'opt_ablation_consts_all.pt'
H = m.transformer.h
S1694_SIMPLE = 0.5094
S1696_BOTH = 0.5504
S1694_JOINT_STAKE = 5.5684
S1687_ATTN_BEST = 0.6805
S1672_MLP_TABLE_SITES = (0, 1, 2)
ATTN_LAGS = (1, 2, 4, 8)
S1683_CE_LIVE = 3.29205
CFG = {'lags': (1,), 'tables': ()}
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


def attn_prog_hook(W):
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        sub = (lagged(args[0]) @ W).reshape(y.shape).to(y.dtype)
        sub = torch.where(SEENREF['m'][STATE['idx']].unsqueeze(-1), sub, y)
        return (sub,) + tuple(out[1:]) if isinstance(out, tuple) else sub
    return hook


def install(prog):
    hs = []
    for (kind, L), W in prog.items():
        if kind == 'mlp':
            hs.append(H[L].mlp.register_forward_hook(
                table_hook(W) if L in CFG['tables'] else mlp_prog_hook(W)))
        else:
            hs.append(H[L].attn.register_forward_hook(attn_prog_hook(W)))
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

    cl = ce(ev, seen)
    assert abs(cl - S1683_CE_LIVE) <= 1e-3, (
        f'baseline CE {cl:.5f} disagrees with the known live CE {S1683_CE_LIVE:.5f} (§1695)')
    hs = [H[L].mlp.register_forward_hook(mlp_const_hook(K[f'mlp{L}'].to(DEV).float()))
          for L in ALL18]
    hs += [H[L].attn.register_forward_hook(attn_const_hook(K[f'attn{L}'].to(DEV).float()))
           for L in ALL18]
    cc = ce(ev, seen, hooks=hs)
    st = cc - cl
    print(f'WHOLE MODEL UPGRADE ATTRIBUTION | compounding or redundancy? | CE live {cl:.5f} | '
          f'joint stake {st:.4f} nats', flush=True)
    print(f'  controls: simple must reproduce §1694 {S1694_SIMPLE:.2%}, both must reproduce '
          f'§1696 {S1696_BOTH:.2%}', flush=True)

    ARMS = (('simple', (1,), ()),
            ('attn_upgraded', ATTN_LAGS, ()),
            ('mlp_upgraded', (1,), S1672_MLP_TABLE_SITES),
            ('both', ATTN_LAGS, S1672_MLP_TABLE_SITES))
    arms = {}
    for name, lags, tables in ARMS:
        CFG['lags'], CFG['tables'] = lags, tables
        prog = compile_stack(fit, ('mlp', 'attn'))
        ct = ce(ev, seen, hooks=install(prog))
        arms[name] = {'lags': list(lags), 'table_sites': list(tables),
                      'ce': round(ct, 5),
                      'ceiling': round((cc - ct) / st if st > 1e-6 else float('nan'), 5)}
        print(f'  {name:14s} lags {str(list(lags)):12s} tables {str(list(tables)):10s} '
              f'CEILING {arms[name]["ceiling"]:8.2%}', flush=True)
        del prog
        torch.cuda.empty_cache()

    vals = [a['ceiling'] for a in arms.values()]
    assert len(set(vals)) > 1, f'all arms identical -- the upgrade switch is a no-op: {vals}'

    base = arms['simple']['ceiling']
    g_att = arms['attn_upgraded']['ceiling'] - base
    g_mlp = arms['mlp_upgraded']['ceiling'] - base
    g_both = arms['both']['ceiling'] - base

    pa = (g_att + g_mlp) > g_both
    pb = g_att > g_mlp
    pc = (abs(base - S1694_SIMPLE) <= 0.01) and (abs(arms['both']['ceiling'] - S1696_BOTH) <= 0.01)

    print(f'\n  gains over the simple program: attention {g_att:+.2%} | MLP {g_mlp:+.2%} | '
          f'both {g_both:+.2%}', flush=True)
    print(f'  sum of singles {g_att + g_mlp:+.2%} vs joint {g_both:+.2%} -> sub-additive '
          f'(redundancy) {pa}', flush=True)
    print(f'  attention is the bigger contributor {pb}', flush=True)
    print(f'  CONTROLS simple {base:.2%} vs {S1694_SIMPLE:.2%} | both '
          f'{arms["both"]["ceiling"]:.2%} vs {S1696_BOTH:.2%} -> {pc}', flush=True)

    res = {'config': {'sites': ALL18, 'ridge': RIDGE,
                      'question': "S1696 left two mechanisms for why half-upgrades do not transfer: "
                                  "COMPOUNDING (each half damped by the other's error, gains roughly "
                                  "ADD) vs REDUNDANCY (both halves recovering shared information, "
                                  "gains strongly SUB-ADDITIVE)",
                      'arms': [{'name': n, 'lags': list(l), 'tables': list(t)} for n, l, t in ARMS],
                      'compilation': 'INTERLEAVED bottom-up (§1669)',
                      'controls': ['baseline 3.29205 (§1695)', 'simple arm = §1694 50.94%',
                                   'both arm = §1696 55.04%'],
                      's1657_caution': 'gains are compared at a single total effect size, not as '
                                       'ratios across conditions of different size',
                      'coverage': 'hybrid, mask pinned to n96_skip80', 'scoring': 'covered positions only',
                      'v1_scope': 'attention v1 passed through unchanged',
                      'fit_rows': 'fineweb_n480_skip80.pt', 'eval_rows': 'fineweb_n192_skip7000.pt'},
           'ce_live': round(cl, 5), 'joint_stake': round(st, 5), 'arms': arms,
           'gains': {'attn': round(g_att, 5), 'mlp': round(g_mlp, 5), 'both': round(g_both, 5),
                     'sum_of_singles': round(g_att + g_mlp, 5)},
           'predictions': {'pred_a_subadditive_redundancy': bool(pa),
                           'pred_b_attention_bigger': bool(pb),
                           'pred_c_controls_hold': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
