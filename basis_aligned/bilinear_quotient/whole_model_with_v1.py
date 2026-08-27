# whole_model_with_v1: CLOSING THE v1 SCOPE CAVEAT ON EVERY ATTENTION RESULT IN THIS ARC
#
# Every attention number from §1682 onward carries the same caveat: attention returns (y, v1)
# and only y was ever substituted, so the results are about output PATHS, not modules. §1684
# measured the v1 path at 0.7066 nats and found it NESTED inside the write rather than additive.
# The whole-model program (§1696, 55.04%) therefore still runs one real piece of the model.
#
# VERIFIED BEFORE BUILDING, from the source rather than memory. Block.forward is
#     x = lambdas[0]*x + lambdas[1]*x0;  x1, v1 = attn(rms_norm(x), v1);  ...  return x, v1
# and CausalBilinearSelfAttention sets `if v1 is None: v1 = v` and returns v1 otherwise
# unchanged. So v1 is ONE object, established at block 0 and threaded untouched through all
# eighteen blocks -- not eighteen objects.
#
# And at block 0, x == x0 == rms_norm(wte(idx)), so the attention input is (lambdas[0]+
# lambdas[1])*x0 renormalised: a pure function of the CURRENT TOKEN. c_v carries no rotary.
# Therefore **v1 is exactly a per-token lookup**, and a covered token table for it must be
# exact -- a known answer derivable before the run, in the §1661 style.
#
# The stake is unchanged from §1694/§1696: §1684 showed that with every attention write pinned
# to a constant, v1 has no causal route to the logits, so ablating it additionally changes
# nothing. That is why these ceilings compare directly with 55.04%.
#
# ARMS, all on top of the §1696 best-families program over 36 sites:
#   v1_real        v1 left untouched                CONTROL, must reproduce §1696's 55.04%
#   v1_table       per-token table for v1           the derivable known answer
#   v1_rank8       rank-8 linear map of block 0's attention input
#   v1_full        full-rank linear map             also exact by construction (c_v is linear)
#
# Registered predictions:
#   pred_a CONTROL: the v1_real arm reproduces §1696's 55.04% within 0.5 points, and the
#          baseline CE reproduces 3.29205 (§1695).
#   pred_b THE CAVEAT CLOSES FOR FREE: v1_table lands within 0.5 points of v1_real. If v1 is
#          genuinely a current-token function, replacing it with a token lookup costs nothing on
#          covered positions, and the arc's attention results become statements about modules
#          rather than output paths. If it does NOT, my reading of the source is wrong and every
#          v1 scope note in this ledger needs revisiting.
#   pred_c MANIPULATION -- the arm is not vacuous: v1_rank8 costs >= 1 point against v1_real. A
#          v1 that can be crushed to eight dimensions for free would mean the path carries almost
#          nothing and §1684's 0.7066 nats needs re-reading.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
ALL18 = list(range(0, 18))
RIDGE = 1e-3
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'whole_model_with_v1_results.json'
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
    flat = v.reshape(-1, v.shape[-1])
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
        f'baseline CE {cl:.5f} disagrees with the known live CE {S1683_CE_LIVE:.5f} (§1695)')
    hs = [H[L].mlp.register_forward_hook(mlp_const_hook(K[f'mlp{L}'].to(DEV).float()))
          for L in ALL18]
    hs += [H[L].attn.register_forward_hook(attn_const_hook(K[f'attn{L}'].to(DEV).float()))
           for L in ALL18]
    cc = ce(ev, seen, hooks=hs)
    st = cc - cl
    print(f'WHOLE MODEL WITH v1 | closing the scope caveat | CE live {cl:.5f} | joint stake '
          f'{st:.4f} nats (§1696 {S1694_JOINT_STAKE:.4f}) | control target {S1696_BOTH:.2%}',
          flush=True)

    prog = compile_stack(fit, ('mlp', 'attn'))
    arms = {}
    for name, mode, rank in (('v1_real', None, D), ('v1_table', 'table', D),
                             ('v1_full', 'linear', D), ('v1_rank8', 'linear', 8)):
        if mode is None:
            V1P.pop('W', None)
        else:
            CFG['v1'] = None
            V1P.pop('W', None)
            V1P['W'] = fit_v1(fit, prog, mode, rank)
        CFG['v1'] = mode
        ct = ce(ev, seen, hooks=install(prog))
        arms[name] = {'mode': mode, 'rank': rank, 'ce': round(ct, 5),
                      'ceiling': round((cc - ct) / st if st > 1e-6 else float('nan'), 5)}
        print(f'  {name:9s} mode {str(mode):7s} rank {rank:5d}: CEILING '
              f'{arms[name]["ceiling"]:8.2%}', flush=True)
    CFG['v1'] = None
    V1P.pop('W', None)

    real = arms['v1_real']['ceiling']
    tab = arms['v1_table']['ceiling']
    r8 = arms['v1_rank8']['ceiling']
    assert abs(tab - r8) > 1e-6, 'v1 arms identical -- the v1 substitution is a no-op'

    pa = (abs(real - S1696_BOTH) <= 0.005) and (abs(cl - S1683_CE_LIVE) <= 1e-3)
    pb = abs(tab - real) <= 0.005
    pc = (real - r8) >= 0.01

    print(f'\n  v1 as a TOKEN TABLE {tab:.2%} vs v1 real {real:.2%} ({tab - real:+.2%}) '
          f'-> caveat closes for free {pb}', flush=True)
    print(f'  v1 full-rank linear {arms["v1_full"]["ceiling"]:.2%} (exact by construction)',
          flush=True)
    print(f'  v1 rank-8 {r8:.2%} ({r8 - real:+.2%}) -> arm not vacuous {pc}', flush=True)
    print(f'  CONTROL v1_real {real:.2%} vs §1696 {S1696_BOTH:.2%} | baseline {cl:.5f} -> {pa}',
          flush=True)

    res = {'config': {'sites': ALL18, 'ridge': RIDGE,
                      'program': 'the §1696 best-families program (tables mlp0-2, lags 1,2,4,8) '
                                 'over 36 sites, plus a program for v1',
                      'v1_structure': 'VERIFIED FROM SOURCE: v1 is ONE object, set at block 0 and '
                                      'threaded unchanged through all 18 blocks. Block 0 attention '
                                      'input is a pure function of the current token (x==x0 there '
                                      'and c_v carries no rotary), so v1 is exactly a per-token lookup.',
                      'stake_unchanged': '§1684 -- with every attention write constant, v1 has no '
                                         'causal route to the logits, so ablating it changes nothing '
                                         'and these ceilings compare directly with 55.04%',
                      'compilation': 'INTERLEAVED bottom-up (§1669)',
                      'coverage': 'hybrid, mask pinned to n96_skip80', 'scoring': 'covered positions only',
                      'fit_rows': 'fineweb_n480_skip80.pt', 'eval_rows': 'fineweb_n192_skip7000.pt',
                      's1696_both': S1696_BOTH, 's1683_ce_live': S1683_CE_LIVE},
           'ce_live': round(cl, 5), 'joint_stake': round(st, 5), 'arms': arms,
           'table_minus_real': round(tab - real, 5), 'rank8_minus_real': round(r8 - real, 5),
           'predictions': {'pred_a_controls_hold': bool(pa),
                           'pred_b_v1_is_a_token_lookup': bool(pb),
                           'pred_c_arm_not_vacuous': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
