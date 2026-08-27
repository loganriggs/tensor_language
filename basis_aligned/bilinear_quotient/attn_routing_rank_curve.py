# attn_routing_rank_curve: HOW MANY DIMENSIONS DOES ATTENTION'S ROUTING NEED?
#
# §1689 priced the value path but flagged that its "routing versus values" comparison was
# invalid: the positional families restrict WHICH POSITIONS may be read while leaving the read
# full-rank, and the value arms restrict the RANK of the read while leaving routing exact.
# Different budgets, no inference.
#
# This fixes it with the same machinery. `c_q`, `c_k`, `c_q2`, `c_k2` are plain Linears just
# like `c_v`, so a rank-r least-squares map from x to each of them is the matched intervention:
# the same kind of budget, on the same 1152-dimensional reads, inside the same module, fitted
# the same data-weighted way and compiled bottom-up. Routing rank then compares directly with
# §1690's value rank.
#
# All four routing projections are truncated together; c_v and c_proj run exactly as trained,
# so this is "the model's real values routed by a rank-r pattern".
#
# REFERENCE CURVE (§1690, value path, same protocol):
#     rank 256 -> 67.05% | 384 -> 94.97% | 512 -> 98.32% | 768 -> 99.71% | 1152 -> 100.01%
#
# IDENTITY CHECK: at full rank the fitted maps recover c_q/c_k/c_q2/c_k2 themselves (each is
# linear in x), so the arm must return ~100%. §1681 showed a no-op satisfies every
# relationship-between-arms prediction, so a pinned value is the only thing that catches one.
#
# Registered predictions:
#   pred_a ROUTING IS LOWER-RANK THAN VALUES: the smallest tested rank reaching 95% of the
#          full-rank ceiling is BELOW 384, the value path's figure. Attention selects with a
#          coarser object than it carries.
#   pred_b ROUTING IS CHEAP IN ABSOLUTE TERMS: rank 64 routing exceeds 50%. The value path at
#          rank 64 reached 2.37%, so if routing also collapses there, "coarser" is the wrong
#          reading and both paths are simply intolerant of compression.
#   pred_c IDENTITY CHECK: full rank returns >= 0.99 and the curve is monotone.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
ALL18 = list(range(0, 18))
RANKS = [64, 128, 256, 384, 512, 1152]
S1690_VALUE = {64: 0.0237, 256: 0.6705, 384: 0.9497, 512: 0.9832, 1152: 1.0001}
PROJ = ('c_q', 'c_k', 'c_q2', 'c_k2')
RIDGE = 1e-3
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'attn_routing_rank_curve_results.json'
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
    hs = []
    for L, Ws in prog.items():
        for nm in PROJ:
            hs.append(getattr(H[L].attn, nm).register_forward_hook(value_hook(Ws[nm])))
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
    """All four routing projections at site L, from one pass."""
    A = torch.zeros(D, D, device=DEV, dtype=torch.float64)
    B = {nm: torch.zeros(D, D, device=DEV, dtype=torch.float64) for nm in PROJ}
    n = {'v': 0}
    hs = []

    def mk(nm, first):
        def hook(mod, args, out):
            x = args[0].reshape(-1, D).double()
            B[nm].add_(x.T @ out.reshape(-1, D).double())
            if first:
                A.add_(x.T @ x); n['v'] += x.shape[0]
            return None
        return hook
    for j, nm in enumerate(PROJ):
        hs.append(getattr(H[L].attn, nm).register_forward_hook(mk(nm, j == 0)))
    sweep(rows, hooks=install(prog) + hs)
    assert n['v'] > 0, f'attn{L}: no fit positions accumulated'
    a = A / n['v']
    reg = RIDGE * torch.diag(a).mean() * torch.eye(D, device=DEV, dtype=torch.float64)
    r = RANKSTATE['r']
    out = {}
    for nm in PROJ:
        W = torch.linalg.solve(a + reg, B[nm] / n['v']).float()
        if r < D:
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
    cc = ce(ev, seen, hooks=[H[L].attn.register_forward_hook(
        const_hook(K[f'attn{L}'].to(DEV).float())) for L in ALL18])
    st = cc - cl
    print(f'ATTN ROUTING RANK CURVE | rank-r maps at {PROJ}, values and c_proj exact | '
          f'ranks {RANKS} | stake {st:.4f} nats', flush=True)
    print(f'  §1690 VALUE path for comparison: ' +
          '  '.join(f'r{k} {v:.2%}' for k, v in sorted(S1690_VALUE.items())), flush=True)

    curve = {}
    for r in RANKS:
        RANKSTATE['r'] = r
        prog = compile_stack(fit)
        ct = ce(ev, seen, hooks=install(prog))
        curve[r] = round((cc - ct) / st if st > 1e-6 else float('nan'), 5)
        vc = S1690_VALUE.get(r)
        print(f'    routing rank {r:5d} ({r / D:5.1%}): CEILING {curve[r]:8.2%}' +
              (f'   (value path at this rank: {vc:.2%})' if vc is not None else ''), flush=True)
        del prog
        torch.cuda.empty_cache()

    vals = [curve[r] for r in RANKS]
    assert len(set(vals)) > 1, f'every rank identical -- the truncation is a no-op: {vals}'

    full = curve[D]
    mono = all(curve[RANKS[i + 1]] >= curve[RANKS[i]] - 0.005 for i in range(len(RANKS) - 1))
    r95 = next((r for r in RANKS if curve[r] >= 0.95 * full), None)

    pa = (r95 is not None) and (r95 < 384)
    pb = curve[64] > 0.50
    pc = (full >= 0.99) and mono

    print(f'\n  routing reaches 95% of full at rank {r95} (value path: 384) -> routing is '
          f'lower-rank {pa}', flush=True)
    print(f'  rank-64 routing {curve[64]:.2%} vs rank-64 values {S1690_VALUE[64]:.2%} '
          f'-> routing cheap in absolute terms {pb}', flush=True)
    print(f'  IDENTITY full rank {full:.2%} | monotone {mono} -> {pc}', flush=True)

    res = {'config': {'sites': ALL18, 'ranks': RANKS, 'ridge': RIDGE, 'projections': list(PROJ),
                      'intervention': 'rank-r least-squares maps at all four routing projections; '
                                      'c_v and c_proj run exactly as trained',
                      'why_matched': 'same budget kind, same 1152-dim reads, same module, same '
                                     'data-weighted fit and bottom-up compilation as §1690, so '
                                     'routing rank compares directly with value rank (§1689 could not)',
                      'identity_check': 'each projection is LINEAR in x so the full-rank arm must return ~100%',
                      'compilation': 'bottom-up (§1669)', 'coverage': 'mask pinned to n96_skip80',
                      'fit_rows': 'fineweb_n480_skip80.pt', 'eval_rows': 'fineweb_n192_skip7000.pt',
                      's1690_value_curve': S1690_VALUE},
           'stake': round(st, 5), 'curve': curve, 'monotone': bool(mono),
           'rank_reaching_95pct': r95, 'value_path_rank_95pct': 384,
           'predictions': {'pred_a_routing_lower_rank_than_values': bool(pa),
                           'pred_b_rank64_routing_gt_50': bool(pb),
                           'pred_c_identity_and_monotone': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
