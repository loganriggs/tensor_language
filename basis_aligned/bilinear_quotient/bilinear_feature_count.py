# bilinear_feature_count: HOW MANY OF THE 4608 BILINEAR FEATURES DOES EACH MLP ACTUALLY NEED?
#
# The family ladder so far prices bilin18's MLP stack by what KIND of program stands in for
# it: constant 0%, per-token table ~57%, linear map of the residual stream 58-61%. Each of
# those is an outside description. The module itself is
#
#     y = Down( (Left x) * (Right x) ) + b,     Left, Right: 1152 -> 4608
#
# so it computes exactly 4608 bilinear features of its input and reads them out linearly.
# The natural next rung, and the first one stated in the model's own terms, is: keep k of
# those 4608 features live and pin the rest to their mean. "This MLP computes k bilinear
# features" is a claim about the model; "this MLP is 62% linear" is a claim about my basis.
#
# METHOD. Per site, compiled bottom-up (§1669 -- the selection and the means both depend on
# the input distribution, which the stack below changes): run the fit set, collect each
# hidden unit's mean and standard deviation, and rank units by std(h_j) * ||Down[:, j]||,
# which is that unit's contribution to the output in the readout's own metric. Keep the top
# k, pin the rest to their mean so the constant part of their contribution is retained, and
# substitute y = Down(h_kept) + b. Whole-stack ceiling against the same optimal-constant
# stake, covered-position scoring, mask pinned to the fit set as §1676 requires.
#
# INSTRUMENT CHECK WITH A KNOWN ANSWER, and it is exact rather than approximate: at k = 4608
# nothing is pinned, the substitution is the identity, and the ceiling MUST be 1.0. Every
# result in this arc that lacked such a check turned out to be wrong (§1659, §1668, §1675),
# and this one is free.
#
# Registered predictions:
#   pred_a A SMALL FRACTION OF THE FEATURES CARRIES THE STACK: k = 512 of 4608 (11%) reaches
#          a whole-stack ceiling >= 80%.
#   pred_b AND IT BEATS EVERY OUTSIDE-DESCRIPTION FAMILY: k = 512 exceeds the linear map's
#          58.17% at this fit set. If a ninth of the model's own features cannot beat a
#          linear approximation, the feature basis is not the useful decomposition.
#   pred_c INSTRUMENT CHECK: k = 4608 gives a ceiling >= 0.99, and the curve is monotone
#          non-decreasing in k. Failure of either means the harness, not the model.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256; DH = 4608
ALL18 = list(range(0, 18))
KS = [8, 32, 128, 512, 2048, 4608]
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'bilinear_feature_count_results.json'
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
EVAL_ROWS = PT + '.rowcache/fineweb_n192_skip7000.pt'
CONSTS = PT + 'opt_ablation_consts_all.pt'
H = m.transformer.h
S1676_SMALL = {'linear': 0.5817, 'table_mlp0_2': 0.5695, 'additive': 0.5379}
S1677_LOWRANK_ADDITIVE = 0.5726
STATE = {}


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


def hidden(mlp, x):
    """The module's own bilinear features, before the linear readout."""
    return mlp.Left(x) * mlp.Right(x)


def feature_hook(mlp, keep, hmean):
    """Keep the selected features live; pin the rest to their fit-set mean."""
    def hook(mod, args, out):
        x = args[0].reshape(-1, D)
        h = hidden(mlp, x)
        h = torch.where(keep.unsqueeze(0), h, hmean.unsqueeze(0).to(h.dtype))
        return (mlp.Down(h) + mlp.Down_bias).reshape(out.shape).to(out.dtype)
    return hook


def install(prog):
    return [H[L].mlp.register_forward_hook(feature_hook(H[L].mlp, k, hm))
            for L, (k, hm) in prog.items()]


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
def select(rows, L, prog, k):
    """Rank units by std(h_j) * ||Down[:, j]|| with the stack below already substituted."""
    s = torch.zeros(DH, device=DEV, dtype=torch.float64)
    sq = torch.zeros(DH, device=DEV, dtype=torch.float64)
    n = {'v': 0}
    mlp = H[L].mlp

    def collect(mod, args, out):
        h = hidden(mlp, args[0].reshape(-1, D)).double()
        s.add_(h.sum(0)); sq.add_((h * h).sum(0)); n['v'] += h.shape[0]
        return None
    sweep(rows, hooks=install(prog) + [H[L].mlp.register_forward_hook(collect)])
    assert n['v'] > 0, f'site {L}: no fit positions'
    mean = (s / n['v'])
    var = (sq / n['v'] - mean * mean).clamp_min(0)
    score = var.sqrt() * mlp.Down.weight.double().norm(dim=0)
    keep = torch.zeros(DH, dtype=torch.bool, device=DEV)
    keep[torch.topk(score, min(k, DH)).indices] = True
    return keep, mean.float()


@torch.no_grad()
def compile_stack(rows, k):
    prog = {}
    for L in ALL18:
        prog[L] = select(rows, L, prog, k)
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
    fit = load(FIT_ROWS); ev = load(EVAL_ROWS)
    K = torch.load(CONSTS, map_location='cpu')
    seen = seen_mask(fit)
    cl = ce(ev, seen)
    cc = ce(ev, seen, hooks=[H[L].mlp.register_forward_hook(
        (lambda cst: (lambda mo, a, o: cst.to(o.dtype).expand_as(o)))(
            K[f'mlp{L}'].to(DEV).float())) for L in ALL18])
    st = cc - cl
    print(f'BILINEAR FEATURE COUNT | y = Down((Left x)*(Right x)) + b, {DH} features per MLP | '
          f'k in {KS} | fit n96_skip80 ({int(seen.sum())} tokens) | stake {st:.4f}', flush=True)
    print(f'  outside-description comparators at this fit set: linear '
          f'{S1676_SMALL["linear"]:.2%} | table@mlp0-2 {S1676_SMALL["table_mlp0_2"]:.2%} | '
          f'low-rank additive {S1677_LOWRANK_ADDITIVE:.2%}', flush=True)

    curve = {}
    for k in KS:
        prog = compile_stack(fit, k)
        ct = ce(ev, seen, hooks=install(prog))
        curve[k] = round((cc - ct) / st if st > 1e-6 else float('nan'), 5)
        print(f'    k {k:5d} ({k / DH:5.1%} of features): CEILING {curve[k]:7.2%}', flush=True)
        del prog
        torch.cuda.empty_cache()

    k512 = curve[512]
    full = curve[DH]
    mono = all(curve[KS[i + 1]] >= curve[KS[i]] - 0.005 for i in range(len(KS) - 1))

    pa = k512 >= 0.80
    pb = k512 > S1676_SMALL['linear']
    pc = (full >= 0.99) and mono

    print(f'\n  k=512 ({512 / DH:.1%} of features): {k512:.2%} -> small fraction carries the '
          f'stack {pa}', flush=True)
    print(f'    vs linear {S1676_SMALL["linear"]:.2%} ({k512 - S1676_SMALL["linear"]:+.2%}) '
          f'-> beats outside descriptions {pb}', flush=True)
    print(f'  INSTRUMENT CHECK k={DH} (identity substitution, known answer 1.0): {full:.2%} | '
          f'monotone {mono} -> {pc}', flush=True)

    res = {'config': {'sites': ALL18, 'hidden_dim': DH, 'ks': KS,
                      'module': 'y = Down((Left x) * (Right x)) + Down_bias',
                      'selection': 'top-k hidden units by std(h_j) * ||Down[:, j]||, the unit\'s '
                                   'contribution in the readout metric; the rest pinned to their '
                                   'fit-set mean so their constant contribution is retained',
                      'compilation': 'bottom-up (§1669)',
                      'fit_rows': 'fineweb_n96_skip80.pt', 'eval_rows': 'fineweb_n192_skip7000.pt',
                      'coverage': 'mask pinned to the fit set (§1676)',
                      'instrument_check': 'k=4608 is the identity substitution; ceiling must be 1.0',
                      's1676_small': S1676_SMALL, 's1677_lowrank_additive': S1677_LOWRANK_ADDITIVE},
           'stake': round(st, 5), 'curve': curve, 'monotone': bool(mono),
           'predictions': {'pred_a_k512_ge_80pct': bool(pa),
                           'pred_b_beats_linear': bool(pb),
                           'pred_c_instrument_check_and_monotone': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
