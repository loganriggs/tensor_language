# all18_sequential_linear: CAN SEQUENTIAL REFITTING RESCUE A WHOLE-STACK SUBSTITUTION?
#
# §1668 tried to substitute all eighteen MLPs at once with least-squares linear maps and
# the arm BROKE: the joint ceiling came out at -42.99%, i.e. the substituted model was
# worse than one with every MLP replaced by a constant. The middle band alone reached
# 62.33% and replicated exactly, so the code is right and the failure is real.
#
# The cause is off-distribution application. Each map is fitted against the REAL model's
# inputs and then applied in a model where every site below it has ALREADY been replaced,
# so its input distribution is not the one it was fitted on. With twelve sites that
# compounding was tolerable; with eighteen it is catastrophic.
#
# THE OBVIOUS REMEDY IS ALSO A KNOWN TRAP. §546 tried refitting a block-1 table against a
# model with block 0 already substituted and it made things WORSE -- +1.0647 against
# +0.6654 for the naive fit. That was tables at n=2. This is linear maps at n=18, where
# the naive arm has already failed outright, so the comparison is worth making rather than
# assuming §546 settles it.
#
# METHOD: compile the stack bottom-up. Fit mlp0's map against the real model; install it;
# fit mlp1's map with mlp0 ALREADY SUBSTITUTED; install; and so on to mlp17. Each map
# therefore sees the input distribution it will actually be applied to. Then measure the
# all-eighteen joint ceiling against the same optimal-constant stake §1666 used.
#
# CONTROL, and it is the load-bearing arm: the same sequential procedure restricted to the
# middle band, where the naive fit works and reads 62.33%. If sequential and naive agree
# there, any difference at eighteen sites is attributable to compounding rather than to
# the procedure. If they disagree there too, this run measures the procedure, not the
# stack, and I report it that way.
#
# Registered predictions:
#   pred_a SEQUENTIAL REFITTING RESCUES THE WHOLE-STACK ARM: the all-eighteen joint ceiling
#          is positive AND at least 30 points above the naive -42.99%.
#   pred_b BUT COMPOUNDING IS NOT FULLY CURABLE: the rescued all-eighteen ceiling still
#          falls below the middle band's own 62.33%.
#   pred_c CONTROL -- the procedure is not what is being measured: on the middle band alone,
#          sequential lands within 10 points of the naive 62.33%.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
ALL18 = list(range(0, 18))
MIDDLE = list(range(4, 16))
RIDGE = 1e-3
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'all18_sequential_linear_results.json'
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip1200.pt'
EVAL_ROWS = PT + '.rowcache/fineweb_n192_skip7000.pt'
CONSTS = PT + 'opt_ablation_consts_all.pt'
H = m.transformer.h
S1668_NAIVE = {'all18': -0.4299, 'middle': 0.6233, 'front': 0.6868, 'late': 0.8360}
S1666_STAKES = {'all18': 4.3196, 'middle': 2.6453}
STATE = {}


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


def linear_hook(W):
    def hook(mod, args, out):
        return (args[0].reshape(-1, D) @ W).reshape(out.shape).to(out.dtype)
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
def fit_one(rows, L, installed):
    """Least squares for site L with `installed` (a dict site->W) already substituted."""
    A = torch.zeros(D, D, device=DEV, dtype=torch.float64)
    B = torch.zeros(D, D, device=DEV, dtype=torch.float64)
    n = {'v': 0}

    def collect(mod, args, out):
        x = args[0].reshape(-1, D).double()
        y = out.reshape(-1, D).double()
        A.add_(x.T @ x); B.add_(x.T @ y); n['v'] += x.shape[0]
        return None
    hooks = [H[k].mlp.register_forward_hook(linear_hook(W)) for k, W in installed.items()]
    hooks.append(H[L].mlp.register_forward_hook(collect))
    sweep(rows, hooks=hooks)
    assert n['v'] > 0, f'site {L}: no fit positions accumulated'
    a = A / n['v']
    reg = RIDGE * torch.diag(a).mean() * torch.eye(D, device=DEV, dtype=torch.float64)
    return torch.linalg.solve(a + reg, B / n['v']).float()


@torch.no_grad()
def compile_stack(rows, sites):
    installed = {}
    for L in sites:
        installed[L] = fit_one(rows, L, installed)
    return installed


@torch.no_grad()
def seen_mask(rows):
    c = torch.zeros(50257, device=DEV)
    for i in range(0, rows.shape[0], 8):
        t = rows[i:i + 8, :-1].to(DEV).reshape(-1)
        c.index_add_(0, t, torch.ones_like(t, dtype=torch.float32))
    return c > 0


@torch.no_grad()
def ce(rows, K, sites, mode, Ws=None, seen=None):
    hooks = []
    for L in sites:
        if mode == 'const':
            hooks.append(H[L].mlp.register_forward_hook(
                (lambda cst: (lambda mo, a, o: cst.to(o.dtype).expand_as(o)))(
                    K[f'mlp{L}'].to(DEV).float())))
        elif mode == 'linear':
            hooks.append(H[L].mlp.register_forward_hook(linear_hook(Ws[L])))
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
    K = torch.load(CONSTS, map_location='cpu')
    seen = seen_mask(fit)
    print(f'ALL18 SEQUENTIAL LINEAR | bottom-up compilation: each site fitted with every '
          f'site below it ALREADY substituted | ridge {RIDGE} | fit skip1200, eval skip7000',
          flush=True)

    out = {}
    for name, sites in (('all18', ALL18), ('middle', MIDDLE)):
        Ws = compile_stack(fit, sites)
        cl = ce(ev, K, [], 'live', seen=seen)
        cc = ce(ev, K, sites, 'const', seen=seen)
        ct = ce(ev, K, sites, 'linear', Ws=Ws, seen=seen)
        st = cc - cl
        ceil = (cc - ct) / st if st > 1e-6 else float('nan')
        naive = S1668_NAIVE[name]
        out[name] = {'sites': sites, 'stake': round(st, 5), 'ce_live': round(cl, 5),
                     'ce_const': round(cc, 5), 'ce_linear': round(ct, 5),
                     'sequential_ceiling': round(ceil, 5), 'naive_ceiling_s1668': naive,
                     'gain_over_naive': round(ceil - naive, 5)}
        print(f'  {name:7s} stake {st:7.4f} | SEQUENTIAL {ceil:8.2%} | naive §1668 '
              f'{naive:8.2%} | gain {ceil - naive:+8.2%}', flush=True)
        del Ws
        torch.cuda.empty_cache()

    a = out['all18']['sequential_ceiling']
    mid_seq = out['middle']['sequential_ceiling']

    pa = (a > 0) and ((a - S1668_NAIVE['all18']) >= 0.30)
    pb = a < S1668_NAIVE['middle']
    pc = abs(mid_seq - S1668_NAIVE['middle']) <= 0.10

    print(f'\n  all-18 rescued from {S1668_NAIVE["all18"]:.2%} to {a:.2%} -> {pa}', flush=True)
    print(f'  still short of the middle band alone ({S1668_NAIVE["middle"]:.2%}) -> {pb}',
          flush=True)
    print(f'  CONTROL middle sequential {mid_seq:.2%} vs naive {S1668_NAIVE["middle"]:.2%} '
          f'-> procedure is not the variable {pc}', flush=True)
    print(f'  (§546 found sequential refitting made a 2-block TABLE substitution worse; '
          f'this is linear maps at 18 sites)', flush=True)

    res = {'config': {'ridge': RIDGE, 'fit_rows': 'fineweb_n96_skip1200.pt',
                      'eval_rows': 'fineweb_n192_skip7000.pt',
                      'procedure': 'bottom-up compilation -- site L fitted with sites below already substituted',
                      'scoring': 'covered positions only, joint substitution',
                      'naive_comparators_s1668': S1668_NAIVE,
                      's1666_stakes': S1666_STAKES,
                      's546_prior': 'sequential refitting made a 2-block table substitution WORSE'},
           'arms': out,
           'predictions': {'pred_a_sequential_rescues_all18': bool(pa),
                           'pred_b_compounding_not_fully_curable': bool(pb),
                           'pred_c_control_middle_agrees': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
