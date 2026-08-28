# CAN AN UNCOVERED POSITION AFFECT A LATER COVERED ONE AT ALL?
#
# §1762 measured the hybrid and standalone programs as differing by 0.9 nats on all-position scoring
# and by **exactly 0.00e+00** on covered-position scoring -- bit-identical float64 sums over 27,497
# positions, in the same forward pass, with hooks demonstrably active. Attention is causal, a
# substituted output at an uncovered position p enters the residual stream, and a covered position
# j > p attends to it, so a small non-zero propagation term is what I expected. I have no mechanism
# for exact zero and §1762 recorded it as an anomaly rather than a locality law.
#
# This is the direct check, and it is deliberately crude: inject a LARGE perturbation into ONE site's
# output at ONE position and look at the per-position losses afterwards. If a big perturbation at an
# uncovered position moves nothing downstream, the exact zero is structural and needs an explanation;
# if it moves something, §1762's zero is an artifact of my measurement and needs a different one.
#
# A positive control is included because a null here is only interesting if the instrument works: the
# same perturbation applied at a COVERED position must move later positions.
#
# ROLES. skip7000 only, one batch. This is an instrument check, not a scientific measurement, and it
# opens no role.
#
# Registered predictions, TWO-SIDED per LESSONS 31, checked against each other:
#   pred_a AN UNCOVERED POSITION REACHES LATER COVERED ONES: perturbing attn0's output at an uncovered
#          position changes some later COVERED position's loss by more than 1e-6 nats. If FALSE, the
#          exact zero in §1762 is structural and something about this model or my harness prevents the
#          influence entirely -- which is the more important outcome and would need naming before any
#          covered-position figure is trusted again.
#   pred_b THE SAME HOLDS AT AN MLP SITE: perturbing mlp0 at the same uncovered position also moves a
#          later covered position. Scored independently, since attention and MLP reach downstream by
#          different routes -- an MLP only via the residual stream that attention above then reads.
#   pred_c POSITIVE CONTROL: the same perturbation at a COVERED position moves later positions. If
#          FALSE the instrument is broken and neither pred_a nor pred_b means anything.
#   pred_d CONTROLS: the perturbed position's OWN loss changes (so the injection landed), the live CE
#          reproduces 3.29205 covered and 3.13704 all-position on skip7000, and coverage is 5419.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/uncovered_position_propagation_results.json'
EVAL = PT + '.rowcache/fineweb_n192_skip7000.pt'
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
NCOV = 5419
MAG = 10.0
STATE = {}


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


def mod_of(kind, L):
    return H[L].mlp if kind == 'mlp' else H[L].attn


def poke_hook(pos, mag):
    """Add a constant `mag` to every channel of this site's output at ONE position."""
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        y2 = y.clone()
        y2[:, pos, :] = y2[:, pos, :] + mag
        return (y2,) + tuple(out[1:]) if isinstance(out, tuple) else y2
    return hook


@torch.no_grad()
def losses(idx, tg, hooks=()):
    hs = [mod_of(*st).register_forward_hook(h) for st, h in hooks]
    try:
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in H:
            x, v1 = blk(x, v1, x0)
        lg = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
        return F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                               reduction='none').reshape(tg.shape).double()
    finally:
        for h in hs:
            h.remove()


@torch.no_grad()
def main():
    t0 = time.time()
    fit = load(FIT_ROWS)
    seen = torch.zeros(50257, dtype=torch.bool)
    seen[fit[:, :T].reshape(-1).long()] = True
    seen = seen.to(DEV)
    ncov = int(seen.sum())
    assert ncov == NCOV, f'coverage {ncov} != {NCOV}'
    ev = load(EVAL)
    print(f'UNCOVERED-POSITION PROPAGATION | one batch, one poke of {MAG} per channel | '
          f'coverage {ncov} of 50257 | INSTRUMENT CHECK', flush=True)

    # whole-set live CE, both populations, as the pred_d known answers
    tot = {'cov': [0.0, 0], 'all': [0.0, 0]}
    for i in range(0, ev.shape[0], 8):
        bb = ev[i:i + 8]
        idx = bb[:, :-1].to(DEV).contiguous()
        e = losses(idx, bb[:, 1:].to(DEV))[:, 64:]
        c = seen[idx[:, 64:]]
        tot['cov'][0] += float(e[c].sum()); tot['cov'][1] += int(c.sum())
        tot['all'][0] += float(e.sum()); tot['all'][1] += int(e.numel())
    live = {k: tot[k][0] / tot[k][1] for k in tot}
    print(f'  live CE: covered {live["cov"]:.5f} (ref 3.29205)  all {live["all"]:.5f} (ref 3.13704)',
          flush=True)

    bb = ev[:8]
    idx = bb[:, :-1].to(DEV).contiguous()
    tg = bb[:, 1:].to(DEV)
    cov = seen[idx]                                   # [B, T] coverage of the INPUT token
    base = losses(idx, tg)

    # pick, in row 0, an uncovered position with covered positions after it, and a covered one
    row = 0
    unc = [p for p in range(64, T - 8) if not bool(cov[row, p])]
    cvd = [p for p in range(64, T - 8) if bool(cov[row, p])]
    assert unc and cvd, 'need both an uncovered and a covered position in row 0'
    p_unc, p_cov = unc[0], cvd[0]
    print(f'  row 0: {int(cov[row].sum())}/{T} positions covered; poking uncovered p={p_unc} and '
          f'covered p={p_cov}', flush=True)

    res = {}
    for tag, site, pos in (('attn0_uncovered', ('attn', 0), p_unc),
                           ('mlp0_uncovered', ('mlp', 0), p_unc),
                           ('attn0_covered_control', ('attn', 0), p_cov)):
        pert = losses(idx, tg, [(site, poke_hook(pos, MAG))])
        d = (pert - base).abs()
        own = float(d[row, pos])
        after = torch.arange(T, device=DEV) > pos
        later_cov = after.unsqueeze(0) & cov
        later_unc = after.unsqueeze(0) & ~cov
        res[tag] = {'site': f'{site[0]}{site[1]}', 'position': pos,
                    'own_position_delta': own,
                    'max_delta_later_covered': float(d[later_cov].max()) if later_cov.any() else None,
                    'max_delta_later_uncovered': float(d[later_unc].max()) if later_unc.any() else None,
                    'n_later_covered': int(later_cov.sum())}
        r = res[tag]
        print(f'    {tag:24s} own {own:.3e} | max later COVERED {r["max_delta_later_covered"]:.3e} '
              f'over {r["n_later_covered"]} positions | max later uncovered '
              f'{r["max_delta_later_uncovered"]:.3e}', flush=True)

    pa = res['attn0_uncovered']['max_delta_later_covered'] > 1e-6
    pb = res['mlp0_uncovered']['max_delta_later_covered'] > 1e-6
    pc = res['attn0_covered_control']['max_delta_later_covered'] > 1e-6
    pd = (all(res[t]['own_position_delta'] > 1e-6 for t in res)
          and abs(live['cov'] - 3.29205) <= 1e-3 and abs(live['all'] - 3.13704) <= 1e-3
          and ncov == NCOV)

    print(f'\n  an UNCOVERED position reaches a later covered one via attn0 -> {pa}', flush=True)
    print(f'  and via mlp0 -> {pb}', flush=True)
    print(f'  positive control: a COVERED position reaches later ones -> {pc}', flush=True)
    print(f'  injection landed at every poked position + live CEs + coverage -> control {pd}',
          flush=True)

    r2 = {'config': {'eval': 'skip7000, first batch of 8 rows', 'magnitude': MAG,
                     'what': 'add MAG to every channel of one site output at one position, then '
                             'compare per-position losses against an unhooked forward',
                     'WHY': '§1762 found the covered-position difference between the hybrid and '
                            'standalone programs to be exactly 0.00e+00 while the all-position '
                            'difference was 0.9 nats. No mechanism was available for exact zero.',
                     'ROLE_NOTE': 'INSTRUMENT CHECK. Opens no role.'},
          'live_ce': {k: round(v, 5) for k, v in live.items()}, 'pokes': res,
          'predictions': {'pred_a_uncovered_reaches_covered_via_attn': bool(pa),
                          'pred_b_uncovered_reaches_covered_via_mlp': bool(pb),
                          'pred_c_positive_control': bool(pc),
                          'pred_d_controls': bool(pd)},
          'runtime_s': round(time.time() - t0, 1)}
    json.dump(r2, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r2["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
