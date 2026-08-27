# attn_program_family: HOW MUCH OF ATTENTION IS POSITION-LOCAL? — the same ladder, other half
#
# §1664-§1681 priced bilin18's eighteen MLPs: a compiled program of eighteen linear maps of
# the residual stream reproduces 60.81% of their 4.33-nat stake, and the front four are
# better described by token tables than by any computation. The other half of the model has
# not been touched by this arc.
#
# Attention is the only thing in a transformer that moves information between positions, so
# a POSITION-WISE linear map `y = xW` is exactly the wrong family for it -- and that is what
# makes it worth fitting. Whatever such a map recovers is the part of each attention
# module's output that is a function of the CURRENT position's residual stream alone. The
# complement is the part that genuinely required reading other positions. That is a
# decomposition of attention into local and non-local work, priced in nats, using machinery
# already validated on the MLP side.
#
# Everything is inherited from the MLP arc rather than reinvented: bottom-up compilation
# (§1669, without which an 18-site substitution goes negative), the coverage mask pinned to
# the fit set (§1676), covered-position scoring, optimal constants from
# opt_ablation_consts_all.pt for the stake, and an identity check.
#
# ONE THING IS GENUINELY DIFFERENT AND IS HANDLED EXPLICITLY. Each attention module returns
# a TUPLE (y, v1) -- v1 is the value-embedding threaded to the blocks above. The hook
# substitutes y and PASSES v1 THROUGH UNCHANGED. So this prices the attention OUTPUT path
# only, and any information the module contributes via v1 is left intact and is not part of
# what the program has to reproduce. Stated up front because it bounds the claim: this is
# not "attention replaced", it is "attention's output write replaced".
#
# Registered predictions:
#   pred_a ATTENTION IS MUCH LESS POSITION-LOCAL THAN THE MLPs: the compiled all-attention
#          linear program's ceiling falls at least 20 points below the MLP program's 60.81%.
#          If a position-wise map does as well on attention as on the MLPs, then either
#          attention is doing far less cross-position work than assumed, or the ceiling is
#          not measuring what I think it is.
#   pred_b BUT ATTENTION IS NOT PURELY NON-LOCAL: the ceiling exceeds 10%. A position-wise
#          map recovering nothing would say the output write carries no local component at
#          all.
#   pred_c MANIPULATION CHECK -- the stake is real: constant-ablating all eighteen attention
#          output writes costs >= 1 nat. Below that, the ceiling is a ratio against noise.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
ALL18 = list(range(0, 18))
RANKS = [8, 32, 128, 512, 1152]
RIDGE = 1e-3
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'attn_program_family_results.json'
FIT_ROWS = PT + '.rowcache/fineweb_n480_skip80.pt'
MASK_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
EVAL_ROWS = PT + '.rowcache/fineweb_n192_skip7000.pt'
CONSTS = PT + 'opt_ablation_consts_all.pt'
H = m.transformer.h
MLP_PROGRAM = {'linear_full_rank': 0.6081, 'rank128': 0.5412, 'stake': 4.3301}
STATE = {}
SEENREF = {}
RANKSTATE = {}


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


def out_y(out):
    return out[0] if isinstance(out, tuple) else out


def repack(out, y):
    """Substitute the output write, pass v1 and anything else through UNCHANGED."""
    return (y,) + tuple(out[1:]) if isinstance(out, tuple) else y


def const_hook(const):
    def hook(mod, args, out):
        y = out_y(out)
        return repack(out, const.to(y.dtype).expand_as(y))
    return hook


def linear_hook(W, seen):
    def hook(mod, args, out):
        y = out_y(out)
        sub = (args[0].reshape(-1, D) @ W).reshape(y.shape).to(y.dtype)
        return repack(out, torch.where(seen[STATE['idx']].unsqueeze(-1), sub, y))
    return hook


def install(prog):
    return [H[L].attn.register_forward_hook(linear_hook(W, SEENREF['m']))
            for L, W in prog.items()]


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
        y = out_y(out).reshape(-1, D).double()
        A.add_(x.T @ x); B.add_(x.T @ y); n['v'] += x.shape[0]
        return None
    sweep(rows, hooks=install(prog) + [H[L].attn.register_forward_hook(collect)])
    assert n['v'] > 0, f'attn{L}: no fit positions accumulated'
    a = A / n['v']
    reg = RIDGE * torch.diag(a).mean() * torch.eye(D, device=DEV, dtype=torch.float64)
    W = torch.linalg.solve(a + reg, B / n['v']).float()
    r = RANKSTATE.get('r', D)
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
    SEENREF['m'] = seen
    del mask_rows
    torch.cuda.empty_cache()
    fit = load(FIT_ROWS)

    cl = ce(ev, seen)
    cc = ce(ev, seen, hooks=[H[L].attn.register_forward_hook(
        const_hook(K[f'attn{L}'].to(DEV).float())) for L in ALL18])
    st = cc - cl
    print(f'ATTN PROGRAM FAMILY | 18 attention OUTPUT WRITES, position-wise linear maps, '
          f'compiled bottom-up | fit n480_skip80, mask n96_skip80 | stake {st:.4f} nats',
          flush=True)
    print(f'  (v1 is passed through unchanged -- this prices the output path only)', flush=True)
    print(f'  MLP comparator: linear program {MLP_PROGRAM["linear_full_rank"]:.2%} of a '
          f'{MLP_PROGRAM["stake"]:.4f}-nat stake', flush=True)

    curve = {}
    for r in RANKS:
        RANKSTATE['r'] = r
        prog = compile_stack(fit)
        ct = ce(ev, seen, hooks=install(prog))
        curve[r] = round((cc - ct) / st if st > 1e-6 else float('nan'), 5)
        print(f'    rank {r:5d}: CEILING {curve[r]:8.2%}', flush=True)
        del prog
        torch.cuda.empty_cache()

    vals = [curve[r] for r in RANKS]
    assert len(set(vals)) > 1, f'every rank identical -- truncation is a no-op: {vals}'
    full = curve[D]
    mono = all(curve[RANKS[i + 1]] >= curve[RANKS[i]] - 0.005 for i in range(len(RANKS) - 1))

    pa = (MLP_PROGRAM['linear_full_rank'] - full) >= 0.20
    pb = full > 0.10
    pc = st >= 1.0

    print(f'\n  ATTENTION output write, position-wise linear: {full:.2%}', flush=True)
    print(f'  vs the MLP stack {MLP_PROGRAM["linear_full_rank"]:.2%} '
          f'({full - MLP_PROGRAM["linear_full_rank"]:+.2%}) -> much less position-local {pa}',
          flush=True)
    print(f'  not purely non-local (>10%) {pb} | stake {st:.4f} >= 1 nat {pc} | '
          f'monotone {mono}', flush=True)
    print(f'  NON-LOCAL SHARE of the attention output write: {1.0 - full:.2%}', flush=True)

    res = {'config': {'sites': ALL18, 'ranks': RANKS, 'ridge': RIDGE,
                      'target': 'attention OUTPUT WRITE only; v1 passed through unchanged',
                      'family': 'position-wise linear map y = xW of the module input',
                      'compilation': 'bottom-up (§1669)',
                      'coverage': 'hybrid, mask pinned to n96_skip80 (§1676)',
                      'fit_rows': 'fineweb_n480_skip80.pt', 'eval_rows': 'fineweb_n192_skip7000.pt',
                      'interpretation': 'what a position-wise map recovers is the part of the output '
                                        'that is a function of the CURRENT position alone; the '
                                        'complement required reading other positions',
                      'mlp_comparator': MLP_PROGRAM},
           'stake': round(st, 5), 'curve': curve, 'monotone': bool(mono),
           'full_rank_ceiling': full, 'non_local_share': round(1.0 - full, 5),
           'predictions': {'pred_a_less_position_local_than_mlps_ge_20pts': bool(pa),
                           'pred_b_not_purely_nonlocal_gt_10pct': bool(pb),
                           'pred_c_stake_ge_1_nat': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
