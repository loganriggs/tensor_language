# attn_value_rank_curve: WHERE DOES ATTENTION'S VALUE PATH SATURATE?
#
# §1689 established that attention's value path is high-rank: with the model's real routing
# held exactly fixed, compressing c_v to rank 256 of 1152 gives 67.05%, rank 64 gives 2.37%,
# and rank 8 gives -26.10% -- worse than replacing attention with a constant. The identity arm
# returned 100.01% with a monotone curve, so the harness is sound.
#
# That leaves the curve unresolved exactly where it matters. Between rank 256 (67.05%) and
# full rank (100.01%) there is a 33-point climb over 896 dimensions and no measurement in it.
# The number this run is for is the analogue of §1664's rank-64-of-1152 result for the MLP
# token tables: how many value dimensions does attention actually need?
#
# ARMS: ranks 256 (CONTROL, must reproduce 67.05%), 384, 512, 768, 1024, and 1152 (IDENTITY,
# must return ~100%). Two known-answer arms rather than one, at both ends of the sweep, since
# §1681 showed a no-op satisfies every relationship-between-arms prediction and only a pinned
# value catches it.
#
# Registered predictions:
#   pred_a THE VALUE PATH IS NOT MERELY HIGH-RANK BUT NEARLY FULL-RANK: rank 512 (44% of the
#          dimensions) still falls below 90%. If half the dimensions suffice, "high-rank" is
#          too strong a reading of §1689 and the curve's steep region is narrow.
#   pred_b THERE IS NO KNEE BELOW 1024: no rank in {384, 512, 768} reaches 95% of the
#          full-rank ceiling. A saturation point below 1024 would be the compact description
#          §1689 failed to find.
#   pred_c CONTROLS AT BOTH ENDS: rank 256 lands within 2 points of §1689's 67.05% and rank
#          1152 returns >= 0.99, and the curve is monotone.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
ALL18 = list(range(0, 18))
RANKS = [256, 384, 512, 768, 1024, 1152]
S1689_RANK256 = 0.6705
RIDGE = 1e-3
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'attn_value_rank_curve_results.json'
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
    print(f'ATTN VALUE RANK CURVE | real routing, rank-r values at c_v | ranks {RANKS} | '
          f'stake {st:.4f} nats', flush=True)
    print(f'  controls: rank 256 must reproduce §1689 {S1689_RANK256:.2%}; rank {D} is the '
          f'identity and must return ~100%', flush=True)

    curve = {}
    for r in RANKS:
        RANKSTATE['r'] = r
        prog = compile_stack(fit)
        ct = ce(ev, seen, hooks=install(prog))
        curve[r] = round((cc - ct) / st if st > 1e-6 else float('nan'), 5)
        print(f'    values rank {r:5d} ({r / D:5.1%} of dims): CEILING {curve[r]:8.2%}',
              flush=True)
        del prog
        torch.cuda.empty_cache()

    vals = [curve[r] for r in RANKS]
    assert len(set(vals)) > 1, f'every rank identical -- the truncation is a no-op: {vals}'

    full = curve[D]
    mono = all(curve[RANKS[i + 1]] >= curve[RANKS[i]] - 0.005 for i in range(len(RANKS) - 1))
    knees = [r for r in (384, 512, 768) if curve[r] >= 0.95 * full]

    pa = curve[512] < 0.90
    pb = len(knees) == 0
    pc = (abs(curve[256] - S1689_RANK256) <= 0.02) and (full >= 0.99) and mono

    print(f'\n  rank 512 ({512 / D:.1%} of dims) at {curve[512]:.2%} -> nearly full-rank {pa}',
          flush=True)
    print(f'  ranks reaching 95% of full ({0.95 * full:.2%}): '
          f'{knees if knees else "none below 1024"} -> no early knee {pb}', flush=True)
    print(f'  CONTROLS rank256 {curve[256]:.2%} vs §1689 {S1689_RANK256:.2%} | identity '
          f'{full:.2%} | monotone {mono} -> {pc}', flush=True)
    print(f'  dims needed for 90% of full: ' +
          str(next((r for r in RANKS if curve[r] >= 0.90 * full), 'not reached')), flush=True)

    res = {'config': {'sites': ALL18, 'ranks': RANKS, 'ridge': RIDGE,
                      'intervention': 'replace c_v output with a rank-r least-squares map of x; the '
                                      'entire routing path runs exactly as the model does',
                      'controls': ['rank 256 must reproduce §1689 67.05%',
                                   'rank 1152 is the identity and must return ~100%'],
                      'why_two_controls': '§1681 showed a no-op satisfies every '
                                          'relationship-between-arms prediction; only pinned values catch it',
                      'compilation': 'bottom-up (§1669)', 'coverage': 'mask pinned to n96_skip80',
                      'fit_rows': 'fineweb_n480_skip80.pt', 'eval_rows': 'fineweb_n192_skip7000.pt',
                      's1689_rank256': S1689_RANK256},
           'stake': round(st, 5), 'curve': curve, 'monotone': bool(mono),
           'ranks_reaching_95pct': knees,
           'predictions': {'pred_a_rank512_below_90': bool(pa),
                           'pred_b_no_knee_below_1024': bool(pb),
                           'pred_c_controls_both_ends': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
