# attn_v1_path: ATTENTION'S SECOND PATH — how much flows through v1?
#
# §1682 priced attention's OUTPUT WRITE: a position-wise linear map recovers 16.38% of a
# 3.5570-nat stake, so the write is 83.6% non-local. That measurement passed `v1` -- the
# value embedding each attention module returns and the blocks above consume -- through
# UNCHANGED, and I flagged in §1682 and on the board that this makes 83.6% a FLOOR rather
# than a statement about the module. §1683 named closing that gap as the next question.
#
# Each block runs `x, v1 = blk(x, v1, x0)`, so attention contributes along two paths: the
# residual write y, and v1 threaded upward. Nothing in this ledger has ever measured the
# second one. This decomposes attention's contribution into the two.
#
# ABLATION CONSTANT, and why it is not the one §1682 used. `opt_ablation_consts_all.pt` has
# optimal constants for attn outputs but nothing for v1, and fitting an optimal constant for
# v1 is a separate optimisation. So BOTH paths are ablated here at their position-weighted
# MEAN, which is matched across the two arms and is what makes the decomposition internally
# comparable. A mean constant is a WEAKER stand-in than an optimal one, so every stake here
# is larger than its optimal-constant counterpart -- §1682's 3.5570 is the optimal-constant
# figure for the write and is reported alongside as the cross-reference, NOT as a comparator.
# Comparing a mean-ablation stake against an optimal-constant one is exactly the
# cross-protocol error §1656 cost me, and it is not made here.
#
# ARMS: ablate y alone | ablate v1 alone | ablate both. Joint against the sum of the singles
# tests whether the two paths carry the same information -- if attention writes overlapping
# content down both, each will look small alone and large together, the §1663/§1665
# redundancy signature at a new grain.
#
# Registered predictions:
#   pred_a THE v1 PATH IS NOT NEGLIGIBLE: mean-ablating v1 alone across all eighteen modules
#          costs >= 0.5 nats. If it is near zero, §1682's 83.6% is not a floor but the whole
#          answer, and that is worth knowing just as much.
#   pred_b THE TWO PATHS ARE NOT ADDITIVE: the joint stake differs from the sum of the two
#          single-path stakes by >= 10% of the sum, in either direction.
#   pred_c MANIPULATION CHECK: both single-path stakes are >= 0.1 nats and the joint stake is
#          at least as large as each single. A path worth nothing alone makes the
#          decomposition vacuous, and a joint below a single would mean the arms are not
#          nested as constructed.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
ALL18 = list(range(0, 18))
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'attn_v1_path_results.json'
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
EVAL_ROWS = PT + '.rowcache/fineweb_n192_skip7000.pt'
H = m.transformer.h
S1682_WRITE_OPTIMAL = {'stake': 3.5570, 'linear_ceiling': 0.1638,
                       'note': 'OPTIMAL-constant ablation; cross-reference only, not a comparator '
                               'for the mean-ablation stakes here'}
STATE = {}


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


def path_hook(const_y, const_v1, kill_y, kill_v1):
    """Replace the residual write y and/or the threaded v1 with their fit-set means."""
    def hook(mod, args, out):
        if not isinstance(out, tuple):
            return const_y.to(out.dtype).expand_as(out) if kill_y else None
        y, rest = out[0], list(out[1:])
        if kill_y:
            y = const_y.to(y.dtype).expand_as(y)
        if kill_v1 and rest and torch.is_tensor(rest[0]):
            rest[0] = const_v1.to(rest[0].dtype).expand_as(rest[0])
        return (y,) + tuple(rest)
    return hook


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
def fit_means(rows):
    """Position-weighted mean of each attention module's y and v1 on the live model."""
    acc = {L: {'y': None, 'v1': None, 'n': 0} for L in ALL18}

    def mk(L):
        def hook(mod, args, out):
            y = (out[0] if isinstance(out, tuple) else out).float().reshape(-1, D)
            a = acc[L]
            a['y'] = y.sum(0).double() if a['y'] is None else a['y'] + y.sum(0).double()
            a['n'] += y.shape[0]
            if isinstance(out, tuple) and len(out) > 1 and torch.is_tensor(out[1]):
                v = out[1].float().reshape(-1, out[1].shape[-1])
                a['v1'] = v.sum(0).double() if a['v1'] is None else a['v1'] + v.sum(0).double()
            return None
        return hook
    sweep(rows, hooks=[H[L].attn.register_forward_hook(mk(L)) for L in ALL18])
    means = {}
    n_v1 = 0
    for L in ALL18:
        a = acc[L]
        assert a['n'] > 0, f'attn{L}: no fit positions'
        means[L] = ((a['y'] / a['n']).float(),
                    (a['v1'] / a['n']).float() if a['v1'] is not None else None)
        n_v1 += int(a['v1'] is not None)
    return means, n_v1


@torch.no_grad()
def seen_mask(rows):
    c = torch.zeros(50257, device=DEV)
    for i in range(0, rows.shape[0], 8):
        t = rows[i:i + 8, :-1].to(DEV).reshape(-1)
        c.index_add_(0, t, torch.ones_like(t, dtype=torch.float32))
    return c > 0


@torch.no_grad()
def ce(rows, seen, means=None, kill_y=False, kill_v1=False):
    hooks = []
    if means is not None and (kill_y or kill_v1):
        for L in ALL18:
            cy, cv = means[L]
            hooks.append(H[L].attn.register_forward_hook(
                path_hook(cy, cv, kill_y, kill_v1 and cv is not None)))
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
    fit = load(FIT_ROWS); ev = load(EVAL_ROWS)
    seen = seen_mask(fit)
    means, n_v1 = fit_means(fit)
    print(f'ATTN v1 PATH | decomposing attention into its residual write and its threaded v1 | '
          f'position-weighted MEAN ablation, matched across arms | {n_v1}/18 modules return a '
          f'v1 tensor', flush=True)
    assert n_v1 > 0, 'no attention module returned a v1 tensor -- the decomposition is vacuous'

    cl = ce(ev, seen)
    arms = {}
    for name, ky, kv in (('write_only', True, False), ('v1_only', False, True),
                         ('both', True, True)):
        c = ce(ev, seen, means, kill_y=ky, kill_v1=kv)
        arms[name] = {'ce': round(c, 5), 'stake': round(c - cl, 5)}
        print(f'  ablate {name:11s} CE {c:.5f} | stake {c - cl:7.4f} nats', flush=True)

    sy = arms['write_only']['stake']
    sv = arms['v1_only']['stake']
    sj = arms['both']['stake']
    ssum = sy + sv
    ratio = sj / ssum if ssum > 1e-9 else float('nan')

    pa = sv >= 0.5
    pb = abs(sj - ssum) >= 0.10 * ssum
    pc = (sy >= 0.1) and (sv >= 0.1) and (sj >= max(sy, sv) - 1e-6)

    print(f'\n  write {sy:.4f} + v1 {sv:.4f} = {ssum:.4f}  |  joint {sj:.4f}  '
          f'-> joint/sum {ratio:.3f}', flush=True)
    print(f'  v1 path carries >= .5 nats {pa} | non-additive by >= 10% {pb} | '
          f'manipulation {pc}', flush=True)
    print(f'  v1 share of the joint: {sv / sj:.1%} | write share: {sy / sj:.1%}', flush=True)
    print(f'  (§1682 priced the write at {S1682_WRITE_OPTIMAL["stake"]:.4f} nats under '
          f'OPTIMAL-constant ablation -- cross-reference, not a comparator: a mean constant '
          f'is weaker so every stake here is larger)', flush=True)

    res = {'config': {'sites': ALL18, 'fit_rows': 'fineweb_n96_skip80.pt',
                      'eval_rows': 'fineweb_n192_skip7000.pt',
                      'ablation': 'position-weighted MEAN, matched across all three arms',
                      'why_not_optimal': 'opt_ablation_consts_all.pt has no v1 constant; fitting one '
                                         'is a separate optimisation. A mean constant is WEAKER than '
                                         'an optimal one so every stake here exceeds its '
                                         'optimal-constant counterpart -- S1682 is a cross-reference, '
                                         'not a comparator (the S1656 cross-protocol error)',
                      'scoring': 'covered positions only',
                      's1682_write_optimal': S1682_WRITE_OPTIMAL,
                      'modules_returning_v1': n_v1},
           'ce_live': round(cl, 5), 'arms': arms,
           'sum_of_singles': round(ssum, 5), 'joint_over_sum': round(ratio, 4),
           'v1_share_of_joint': round(sv / sj, 4) if sj > 1e-9 else None,
           'predictions': {'pred_a_v1_path_ge_05_nats': bool(pa),
                           'pred_b_paths_non_additive_ge_10pct': bool(pb),
                           'pred_c_manipulation': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
