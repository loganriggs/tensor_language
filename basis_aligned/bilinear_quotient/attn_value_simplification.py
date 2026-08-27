# attn_value_simplification: IS ATTENTION'S IRREDUCIBLE PART THE ROUTING OR THE VALUES?
#
# §1687 left 29.92% of attention's output write outside ANY fixed-position linear
# description, and showed it is not long-range: widening a geometric window out to lag 64
# buys 2.03 points and the last three doublings buy 0.82 between them. So the residue is
# content-dependent ROUTING -- which positions get read depends on what is in them, and no
# fixed lag set can express that.
#
# That was inferred from what the positional families CANNOT do. This tests it directly from
# the other side: hold the model's real routing fixed and simplify the VALUES instead.
#
# `CausalBilinearSelfAttention` computes v = c_v(x), mixes it with the inherited v1 by lamb,
# routes with two rotary q/k pairs under squared attention, and projects with c_proj. c_v is
# a plain Linear, so its output can be replaced by a rank-r least-squares map of x while the
# entire routing path -- q, k, q2, k2, rotary, the squared-attention mixing, lamb, c_proj --
# runs exactly as the model does it.
#
#   fixed-position families (§1685-§1687): simple values, NO routing        -> caps at 70.08%
#   this run:                              REAL routing, simplified values  -> ?
# If real routing plus even a crude value map clears 70.08%, routing is what the positional
# families were missing, and §1687's inference is confirmed from the opposite direction.
#
# EXACT IDENTITY CHECK, and it is free: c_v is LINEAR in x, so the full-rank least-squares
# map from x to c_v(x) is c_v itself up to ridge. The full-rank arm must return ~100%. Every
# result in this arc that lacked such a check was wrong (§1659, §1668, §1675, §1681).
#
# The rank-r map is fitted from the data (normal equations, compiled bottom-up) rather than
# taken as the plain SVD of c_v.weight, because the useful rank-r approximation is the one
# that minimises error under the INPUT COVARIANCE the module actually sees, not under an
# unweighted norm.
#
# SCOPE: hooking c_v also changes the v1 that block 0 exports, since v1 is that same
# pre-mix v. That is intended -- v1 is part of the value path being simplified -- but it
# means this is "attention with simplified values", not "attention with simplified values and
# an untouched v1". Stated because §1684 showed how easily the v1 seam is mis-stated.
#
# Registered predictions:
#   pred_a ROUTING IS WHAT THE POSITIONAL FAMILIES WERE MISSING: real routing with rank-64
#          values exceeds §1687's best fixed-position ceiling of 70.08% by >= 10 points.
#   pred_b ROUTING DOMINATES THE VALUES: even rank-8 values with real routing clear 70.08%.
#          Eight dimensions of value is a far cruder description than seven full positions.
#   pred_c IDENTITY CHECK: the full-rank arm returns >= 0.99, and the curve is monotone
#          non-decreasing in r. Either failing means the harness, not the model.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
ALL18 = list(range(0, 18))
RANKS = [8, 64, 256, 1152]
RIDGE = 1e-3
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'attn_value_simplification_results.json'
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
    """Replace c_v's output with x @ W. Routing downstream of it is untouched."""
    def hook(mod, args, out):
        return (args[0].reshape(-1, D) @ W).reshape(out.shape).to(out.dtype)
    return hook


def install(prog):
    return [H[L].attn.c_v.register_forward_hook(value_hook(W)) for L, W in prog.items()]


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
    A = torch.zeros(D, D, device=DEV, dtype=torch.float64)
    B = torch.zeros(D, D, device=DEV, dtype=torch.float64)
    n = {'v': 0}

    def collect(mod, args, out):
        x = args[0].reshape(-1, D).double()
        y = out.reshape(-1, D).double()
        A.add_(x.T @ x); B.add_(x.T @ y); n['v'] += x.shape[0]
        return None
    sweep(rows, hooks=install(prog) + [H[L].attn.c_v.register_forward_hook(collect)])
    assert n['v'] > 0, f'attn{L}.c_v: no fit positions accumulated'
    a = A / n['v']
    reg = RIDGE * torch.diag(a).mean() * torch.eye(D, device=DEV, dtype=torch.float64)
    W = torch.linalg.solve(a + reg, B / n['v']).float()
    r = RANKSTATE['r']
    if r < D:
        U, S, Vh = torch.linalg.svd(W.double(), full_matrices=False)
        W = ((U[:, :r] * S[:r]) @ Vh[:r]).float()
    return W


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
    cc = ce(ev, seen, hooks=[H[L].attn.register_forward_hook(
        const_hook(K[f'attn{L}'].to(DEV).float())) for L in ALL18])
    st = cc - cl
    print(f'ATTN VALUE SIMPLIFICATION | real routing, rank-r values at c_v | ranks {RANKS} | '
          f'stake {st:.4f} nats', flush=True)
    print(f'  fixed-position comparators: lag0 {S1682_LAG0:.2%} | lag1 {S1685_LAG1:.2%} | '
          f'best positional (7 slots) {S1687_BEST_POSITIONAL:.2%}', flush=True)

    curve = {}
    for r in RANKS:
        RANKSTATE['r'] = r
        prog = compile_stack(fit)
        ct = ce(ev, seen, hooks=install(prog))
        curve[r] = round((cc - ct) / st if st > 1e-6 else float('nan'), 5)
        print(f'    values rank {r:5d}: CEILING {curve[r]:8.2%}', flush=True)
        del prog
        torch.cuda.empty_cache()

    vals = [curve[r] for r in RANKS]
    assert len(set(vals)) > 1, f'every rank identical -- the truncation is a no-op: {vals}'

    full = curve[D]
    mono = all(curve[RANKS[i + 1]] >= curve[RANKS[i]] - 0.005 for i in range(len(RANKS) - 1))

    pa = (curve[64] - S1687_BEST_POSITIONAL) >= 0.10
    pb = curve[8] > S1687_BEST_POSITIONAL
    pc = (full >= 0.99) and mono

    print(f'\n  rank-64 values with REAL routing {curve[64]:.2%} vs best fixed-position '
          f'{S1687_BEST_POSITIONAL:.2%} ({curve[64] - S1687_BEST_POSITIONAL:+.2%}) '
          f'-> routing is what was missing {pa}', flush=True)
    print(f'  rank-8 values {curve[8]:.2%} -> routing dominates values {pb}', flush=True)
    print(f'  IDENTITY CHECK full rank {full:.2%} (c_v is linear in x, so this must be ~100%) '
          f'| monotone {mono} -> {pc}', flush=True)

    res = {'config': {'sites': ALL18, 'ranks': RANKS, 'ridge': RIDGE,
                      'intervention': 'replace c_v output with a rank-r least-squares map of x; '
                                      'q, k, q2, k2, rotary, squared-attention mixing, lamb and '
                                      'c_proj all run exactly as the model does',
                      'identity_check': 'c_v is LINEAR in x, so the full-rank map is c_v itself up '
                                        'to ridge and the full-rank arm must return ~100%',
                      'why_data_weighted': 'the useful rank-r approximation minimises error under the '
                                           'input covariance the module actually sees, not an '
                                           'unweighted norm on c_v.weight',
                      'scope': 'hooking c_v also changes the v1 block 0 exports, since v1 is that same '
                               'pre-mix v -- this is attention with simplified values INCLUDING v1',
                      'compilation': 'bottom-up (§1669)', 'coverage': 'mask pinned to n96_skip80',
                      'fit_rows': 'fineweb_n480_skip80.pt', 'eval_rows': 'fineweb_n192_skip7000.pt',
                      's1687_best_positional': S1687_BEST_POSITIONAL,
                      's1685_lag1': S1685_LAG1, 's1682_lag0': S1682_LAG0},
           'stake': round(st, 5), 'curve': curve, 'monotone': bool(mono),
           'rank64_minus_best_positional': round(curve[64] - S1687_BEST_POSITIONAL, 5),
           'predictions': {'pred_a_routing_was_missing_ge_10pts': bool(pa),
                           'pred_b_rank8_clears_positional': bool(pb),
                           'pred_c_identity_and_monotone': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
