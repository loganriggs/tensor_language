# mlp_path_rank_curve: THE SAME MATCHED-BUDGET SPLIT ON THE MLP SIDE
#
# §1690/§1691 split attention into a cheap low-rank SELECTION and an expensive nearly-full-rank
# PAYLOAD, using matched rank budgets on the module's own linear projections: at 5.6% of
# dimensions routing keeps 62.82% of the write and values keep 2.37%.
#
# The MLP has the same kind of internal split and it has never been priced this way. Each is
#     y = Down( (Left x) * (Right x) ) + b,     Left, Right: 1152 -> 4608,  Down: 4608 -> 1152
# so `Left`/`Right` FORM the bilinear features and `Down` READS them out. Truncating each to
# rank r is the same intervention, same fit, same compilation, same currency as §1691.
#
# §1679 is the reason to expect an asymmetry: each MLP's output lies within 512 of 1152
# principal directions to 94.87%, so the readout has room to be low-rank. Nothing about the
# feature-forming side is constrained that way -- §1679 also found the 4608 features are not
# individually removable at all, because the readout sums large cancelling contributions.
#
# ARMS: rank r applied to Down alone, and to Left+Right together, r in {8, 32, 64, 256, full}.
# Full rank on either side is the IDENTITY (each is linear in its input, so the fitted
# full-rank map recovers the projection itself) and must return ~100%. §1681 is the reason
# both arms carry one.
#
# Registered predictions:
#   pred_a THE READOUT IS THE COMPRESSIBLE SIDE: at rank 64, Down beats Left+Right by >= 20
#          percentage points. §1679 says the output lives in a few hundred directions; nothing
#          says the feature-forming side does.
#   pred_b THE MLP IS CHEAPER THAN ATTENTION'S PAYLOAD: at rank 64 BOTH MLP arms exceed 50%,
#          against the 2.37% attention's value path manages at that rank. If an MLP path also
#          collapses at rank 64, "attention's payload is unusually expensive" is the wrong
#          reading of §1691.
#   pred_c IDENTITY CHECKS: both full-rank arms return >= 0.99 and both curves are monotone.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
ALL18 = list(range(0, 18))
RANKS = [8, 32, 64, 256, 1152]
DH = 4608
ARMS = {'Down': ('Down',), 'LeftRight': ('Left', 'Right')}
S1691_ATTN = {'routing_r64': 0.6282, 'values_r64': 0.0237}
S1679_OUTPUT_512 = 0.9487
ARMSTATE = {'a': 'Down'}
RIDGE = 1e-3
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp_path_rank_curve_results.json'
FIT_ROWS = PT + '.rowcache/fineweb_n480_skip80.pt'
MASK_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
EVAL_ROWS = PT + '.rowcache/fineweb_n192_skip7000.pt'
CONSTS = PT + 'opt_ablation_consts_all.pt'
H = m.transformer.h
S1687_BEST_POSITIONAL = 0.7008
S1685_LAG1 = 0.5626
S1682_LAG0 = 0.1638
STATE = {}
RANKSTATE = {'r': D}


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


def const_hook(const):
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        sub = const.to(y.dtype).expand_as(y)
        return (sub,) + tuple(out[1:]) if isinstance(out, tuple) else sub
    return hook


def value_hook(W):
    """Replace a projection's output with (its own input) @ W."""
    def hook(mod, args, out):
        din = W.shape[0]
        return (args[0].reshape(-1, din) @ W).reshape(out.shape).to(out.dtype)
    return hook


def install(prog):
    hs = []
    for L, Ws in prog.items():
        for nm, W in Ws.items():
            hs.append(getattr(H[L].mlp, nm).register_forward_hook(value_hook(W)))
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
def fit_site(rows, L, prog):
    """Rank-r maps for the projections of the current arm, from one pass."""
    names = ARMS[ARMSTATE['a']]
    din = DH if names[0] == 'Down' else D
    dout = D if names[0] == 'Down' else DH
    A = torch.zeros(din, din, device=DEV, dtype=torch.float64)
    B = {nm: torch.zeros(din, dout, device=DEV, dtype=torch.float64) for nm in names}
    n = {'v': 0}
    hs = []

    def mk(nm, first):
        def hook(mod, args, out):
            x = args[0].reshape(-1, din).double()
            B[nm].add_(x.T @ out.reshape(-1, dout).double())
            if first:
                A.add_(x.T @ x); n['v'] += x.shape[0]
            return None
        return hook
    for j, nm in enumerate(names):
        hs.append(getattr(H[L].mlp, nm).register_forward_hook(mk(nm, j == 0)))
    sweep(rows, hooks=install(prog) + hs)
    assert n['v'] > 0, f'mlp{L}: no fit positions accumulated'
    a = A / n['v']
    reg = RIDGE * torch.diag(a).mean() * torch.eye(din, device=DEV, dtype=torch.float64)
    r = RANKSTATE['r']
    out = {}
    for nm in names:
        W = torch.linalg.solve(a + reg, B[nm] / n['v']).float()
        if r < min(din, dout):
            U, S, Vh = torch.linalg.svd(W.double(), full_matrices=False)
            W = ((U[:, :r] * S[:r]) @ Vh[:r]).float()
        out[nm] = W
    return out


@torch.no_grad()
def compile_stack(rows):
    prog = {}
    for L in ALL18:
        prog[L] = fit_site(rows, L, prog)
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
    ev = load(EVAL_ROWS)
    K = torch.load(CONSTS, map_location='cpu')
    mask_rows = load(MASK_ROWS)
    seen = seen_mask(mask_rows)
    del mask_rows
    torch.cuda.empty_cache()
    fit = load(FIT_ROWS)

    cl = ce(ev, seen)
    cc = ce(ev, seen, hooks=[H[L].mlp.register_forward_hook(
        (lambda cst: (lambda mo, a, o: cst.to(o.dtype).expand_as(o)))(
            K[f'mlp{L}'].to(DEV).float())) for L in ALL18])
    st = cc - cl
    print(f'MLP PATH RANK CURVE | y = Down((Left x)*(Right x)) + b | arms {list(ARMS)} | '
          f'ranks {RANKS} | stake {st:.4f} nats', flush=True)
    print(f'  §1691 attention at rank 64: routing {S1691_ATTN["routing_r64"]:.2%} | '
          f'values {S1691_ATTN["values_r64"]:.2%}', flush=True)

    curves = {}
    for arm in ARMS:
        ARMSTATE['a'] = arm
        curves[arm] = {}
        for r in RANKS:
            RANKSTATE['r'] = r
            prog = compile_stack(fit)
            ct = ce(ev, seen, hooks=install(prog))
            curves[arm][r] = round((cc - ct) / st if st > 1e-6 else float('nan'), 5)
            print(f'    {arm:10s} rank {r:5d}: CEILING {curves[arm][r]:8.2%}', flush=True)
            del prog
            torch.cuda.empty_cache()

    for arm in ARMS:
        v = list(curves[arm].values())
        assert len(set(v)) > 1, f'{arm}: every rank identical -- truncation is a no-op: {v}'

    dn, lr = curves['Down'], curves['LeftRight']
    full_ok = (dn[1152] >= 0.99) and (lr[1152] >= 0.99)
    mono = all(c[RANKS[i + 1]] >= c[RANKS[i]] - 0.005
               for c in (dn, lr) for i in range(len(RANKS) - 1))

    pa = (dn[64] - lr[64]) >= 0.20
    pb = (dn[64] > 0.50) and (lr[64] > 0.50)
    pc = full_ok and mono

    print(f'\n  rank 64: Down {dn[64]:.2%} | Left+Right {lr[64]:.2%} '
          f'({dn[64] - lr[64]:+.2%}) -> readout is the compressible side {pa}', flush=True)
    print(f'  both MLP arms at rank 64 vs attention values {S1691_ATTN["values_r64"]:.2%} '
          f'-> MLP cheaper than attention payload {pb}', flush=True)
    print(f'  IDENTITY Down {dn[1152]:.2%} | LeftRight {lr[1152]:.2%} | monotone {mono} '
          f'-> {pc}', flush=True)

    res = {'config': {'sites': ALL18, 'ranks': RANKS, 'ridge': RIDGE, 'arms': {k: list(v) for k, v in ARMS.items()},
                      'module': 'y = Down((Left x) * (Right x)) + Down_bias; Left/Right FORM the '
                                'bilinear features (1152->4608), Down READS them out (4608->1152)',
                      'intervention': 'rank-r least-squares map at the arm projections; the rest of the '
                                      'module runs exactly as trained',
                      'matched_with': '§1690/§1691 -- same fit, same compilation, same currency',
                      'identity_check': 'each projection is LINEAR in its input so the full-rank arm '
                                        'must return ~100%',
                      'compilation': 'bottom-up (§1669)', 'coverage': 'mask pinned to n96_skip80',
                      'fit_rows': 'fineweb_n480_skip80.pt', 'eval_rows': 'fineweb_n192_skip7000.pt',
                      's1691_attn_rank64': S1691_ATTN, 's1679_output_512_directions': S1679_OUTPUT_512},
           'stake': round(st, 5), 'curves': curves, 'monotone': bool(mono),
           'down_minus_leftright_at_64': round(dn[64] - lr[64], 5),
           'predictions': {'pred_a_readout_compressible_ge_20pts': bool(pa),
                           'pred_b_both_mlp_arms_gt_50_at_64': bool(pb),
                           'pred_c_identity_and_monotone': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
