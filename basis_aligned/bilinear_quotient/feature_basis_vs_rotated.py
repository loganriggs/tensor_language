# feature_basis_vs_rotated: IS THE MODEL'S OWN FEATURE BASIS SPARSE-SELECTABLE?
#
# §1678 tried to keep k of each MLP's 4608 bilinear features and pin the rest to their mean.
# The identity check at k=4608 was exact (100.00%), so the harness is sound -- but every k
# below 2048 came out NEGATIVE and the curve was not monotone (-34.68, -29.95, -49.62,
# -49.93, +61.57). By LESSONS 28's own rule that interior is unreadable, and I reported no
# feature count.
#
# The diagnosis there: pinning feature j to its mean leaves an output error
# sum_{pinned j} Down[:, j] * (h_j - mean_j), and my selection ranked units by MARGINAL
# contribution while ignoring the covariance between them. In a bilinear readout large
# contributions cancel; keeping the top 512 by marginal size keeps one side of many
# cancelling pairs and breaks the balance. That is why keeping 512 features is worse than
# keeping 8.
#
# If that is right, the model's own feature basis is not sparse-selectable at ANY k, and
# only a ROTATED basis -- directions decorrelated by construction -- gives a monotone curve.
# The distinction matters for what a faithful account can even claim: "this MLP computes k
# features" versus "this MLP's output lies near a k-dimensional subspace" are different
# statements, and only one of them may be available.
#
# THREE ARMS at matched k, every one compiled bottom-up:
#   topk     -- §1678's criterion, reproduced as a control
#   random   -- k features chosen by a fixed index stride, no criterion at all. If topk is
#               no better than this, the criterion is worthless; if BOTH fail, selection in
#               this basis is the problem rather than my choice of ranking.
#   rotated  -- project the module output onto the top k principal directions of its own
#               fit-set covariance, adding the mean back. Same k, decorrelated basis.
# The identity arms (k=4608 for features, k=1152 for rotated) are exact known answers.
#
# Registered predictions:
#   pred_a THE ROTATED BASIS WORKS AND THE FEATURE BASIS DOES NOT: at k=128 the rotated arm
#          exceeds both feature arms by >= 50 percentage points.
#   pred_b THE ROTATED CURVE IS MONOTONE while at least one feature curve is not -- the
#          diagnostic signature from LESSONS 28, present in one family and absent in the other.
#   pred_c CONTROLS -- both identity arms return >= 0.99, and the topk arm reproduces §1678
#          at k=128 within 5 points.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256; DH = 4608
ALL18 = list(range(0, 18))
KS = [8, 32, 128, 512]
S1678_TOPK = {8: -0.3468, 32: -0.2995, 128: -0.4962, 512: -0.4993, 2048: 0.6157, 4608: 1.0}
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'feature_basis_vs_rotated_results.json'
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


def rotated_hook(mlp, basis, mean):
    """Project the module output onto k principal directions of its own covariance."""
    def hook(mod, args, out):
        y = (mlp.Down(hidden(mlp, args[0].reshape(-1, D))) + mlp.Down_bias).float()
        c = y - mean.unsqueeze(0)
        return (mean.unsqueeze(0) + (c @ basis) @ basis.T).reshape(out.shape).to(out.dtype)
    return hook


def feature_hook(mlp, keep, hmean):
    """Keep the selected features live; pin the rest to their fit-set mean."""
    def hook(mod, args, out):
        x = args[0].reshape(-1, D)
        h = hidden(mlp, x)
        h = torch.where(keep.unsqueeze(0), h, hmean.unsqueeze(0).to(h.dtype))
        return (mlp.Down(h) + mlp.Down_bias).reshape(out.shape).to(out.dtype)
    return hook


def install(prog):
    hs = []
    for L, p in prog.items():
        mlp = H[L].mlp
        hs.append(mlp.register_forward_hook(
            rotated_hook(mlp, p[1], p[2]) if p[0] == 'rotated'
            else feature_hook(mlp, p[1], p[2])))
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
def select_rotated(rows, L, prog, k):
    """Top-k principal directions of the module output on the fit set."""
    mlp = H[L].mlp
    s = torch.zeros(D, device=DEV, dtype=torch.float64)
    C = torch.zeros(D, D, device=DEV, dtype=torch.float64)
    n = {'v': 0}

    def collect(mod, args, out):
        y = (mlp.Down(hidden(mlp, args[0].reshape(-1, D))) + mlp.Down_bias).double()
        s.add_(y.sum(0)); C.add_(y.T @ y); n['v'] += y.shape[0]
        return None
    sweep(rows, hooks=install(prog) + [mlp.register_forward_hook(collect)])
    assert n['v'] > 0, f'site {L}: no fit positions'
    mean = s / n['v']
    cov = C / n['v'] - torch.outer(mean, mean)
    ev = torch.linalg.eigh(cov).eigenvectors
    return ('rotated', ev[:, -min(k, D):].float().contiguous(), mean.float())


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
    return ('feature', keep, mean.float())


@torch.no_grad()
def select_stride(rows, L, prog, k):
    """k features by fixed index stride -- no criterion at all. Needs the means, which
    still depend on the compiled stack below."""
    _, _, mean = select(rows, L, prog, DH)
    keep = torch.zeros(DH, dtype=torch.bool, device=DEV)
    step = max(DH // max(k, 1), 1)
    keep[torch.arange(0, DH, step, device=DEV)[:k]] = True
    return ('feature', keep, mean)


@torch.no_grad()
def compile_stack(rows, k, arm):
    fn = {'topk': select, 'random': select_stride, 'rotated': select_rotated}[arm]
    prog = {}
    for L in ALL18:
        prog[L] = fn(rows, L, prog, k)
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
    print(f'FEATURE BASIS vs ROTATED | is the model own feature basis sparse-selectable? | '
          f'k in {KS} | fit n96_skip80 | stake {st:.4f}', flush=True)

    curves = {}
    for arm in ('topk', 'random', 'rotated'):
        curves[arm] = {}
        for k in KS:
            prog = compile_stack(fit, k, arm)
            ct = ce(ev, seen, hooks=install(prog))
            curves[arm][k] = round((cc - ct) / st if st > 1e-6 else float('nan'), 5)
            print(f'    {arm:8s} k {k:5d}: CEILING {curves[arm][k]:8.2%}', flush=True)
            del prog
            torch.cuda.empty_cache()

    ident = {}
    for arm, kk in (('topk', DH), ('rotated', D)):
        prog = compile_stack(fit, kk, arm)
        ct = ce(ev, seen, hooks=install(prog))
        ident[arm] = round((cc - ct) / st if st > 1e-6 else float('nan'), 5)
        print(f'    IDENTITY {arm:8s} k {kk:5d}: {ident[arm]:8.2%}  (known answer 1.0)',
              flush=True)
        del prog
        torch.cuda.empty_cache()

    def mono(c):
        return all(c[KS[i + 1]] >= c[KS[i]] - 0.005 for i in range(len(KS) - 1))

    rot128 = curves['rotated'][128]
    gap = rot128 - max(curves['topk'][128], curves['random'][128])
    mo = {a: mono(c) for a, c in curves.items()}

    pa = gap >= 0.50
    pb = mo['rotated'] and not all(mo[a] for a in ('topk', 'random'))
    pc = (all(v >= 0.99 for v in ident.values())
          and abs(curves['topk'][128] - S1678_TOPK[128]) <= 0.05)

    print(f'\n  at k=128: rotated {rot128:.2%} | topk {curves["topk"][128]:.2%} | '
          f'random {curves["random"][128]:.2%}  -> rotated basis works, feature basis does '
          f'not {pa}', flush=True)
    print(f'  monotone: ' + '  '.join(f'{a} {mo[a]}' for a in curves) + f'  -> {pb}', flush=True)
    print(f'  identity checks {ident} | topk k=128 vs §1678 {S1678_TOPK[128]:.2%} -> {pc}',
          flush=True)

    res = {'config': {'sites': ALL18, 'ks': KS, 'hidden_dim': DH, 'output_dim': D,
                      'arms': {'topk': 'top-k features by std(h_j)*||Down[:,j]||, rest pinned to mean',
                               'random': 'k features by fixed index stride, no criterion',
                               'rotated': 'top-k principal directions of the module output covariance'},
                      'compilation': 'bottom-up (§1669)', 'coverage': 'mask pinned to the fit set',
                      'fit_rows': 'fineweb_n96_skip80.pt', 'eval_rows': 'fineweb_n192_skip7000.pt',
                      'identity_checks': 'topk k=4608 and rotated k=1152 are both the identity',
                      's1678_topk': S1678_TOPK},
           'stake': round(st, 5), 'curves': curves, 'identity': ident, 'monotone': mo,
           'rotated_minus_features_at_128': round(gap, 5),
           'predictions': {'pred_a_rotated_beats_features_ge_50pts': bool(pa),
                           'pred_b_monotone_only_in_rotated': bool(pb),
                           'pred_c_controls_hold': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
