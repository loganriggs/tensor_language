# SITE ADDITIVITY -- how much of each stack's cost is recoverable when you remove one site at a time?
#
# §1735 measured, as a by-product of a control I mis-specified, that the SUM of the 36 individual
# constant-ablation removals does not resemble the JOINT stack removals at all, and that the two
# stacks miss in OPPOSITE directions on skip7000:
#
#     18 MLPs        sum of individual 10.298 nats   vs joint 4.330   -> ratio 2.38, REDUNDANT
#     18 attention   sum of individual  1.421 nats   vs joint 3.557   -> ratio 0.40, COOPERATIVE
#
# Read plainly: knock out one MLP and the other seventeen absorb most of it, so one-at-a-time
# ablation OVERSTATES the stack by more than a factor of two. Knock out one attention site and
# almost nothing happens, because most of what the attention stack contributes only exists when the
# sites act together -- one-at-a-time ablation UNDERSTATES it by a similar factor. If that holds it
# is a statement about how the two kinds compose, and it bears directly on §1669 (independently
# fitted programs installed jointly gave -42.99%) and on every single-site importance number in the
# arc, mine and Codex's alike.
#
# It has never been measured on its own terms. It fell out of a broken control on ONE eval set.
#
# ROLES, DECLARED BEFORE THE RUN. This is a NEW hypothesis, so the class-hypothesis role burn does
# not apply -- but skip7000 does, because §1735 printed its two sums and I read them.
#   skip7000    DISCOVERY. Already observed. Reported, never used to confirm.
#   skip11000   CONFIRMATION. Its sums were computed inside §1735 but never printed, stored, or
#               seen; this run is the first time any additivity number from it is looked at.
#
# Registered predictions, TWO-SIDED per LESSONS 31:
#   pred_a CONFIRMED ON skip11000, BOTH DIRECTIONS: the MLP ratio (sum of individual over joint) has
#          a 95% interval entirely ABOVE 1.5, and the attention ratio a 95% interval entirely BELOW
#          0.7. Either half failing fails the prediction. If the intervals straddle those bars the
#          asymmetry is not resolved at 192 rows and nothing is certified.
#   pred_b THE SIGN OF THE ASYMMETRY IS STABLE: MLP ratio above 1 and attention ratio below 1 on
#          BOTH roles. A kind whose sign flips between roles is not a compositional property.
#   pred_c CONTROLS: the JOINT stack removals reproduce §1662/§1682's 4.3301 and 3.5570 within 0.01
#          on skip7000 -- the control §1735 got wrong by comparing a sum to a joint -- and both
#          baseline CEs reproduce 3.29205 and 3.09711.
#   pred_d IT IS NOT A COVERAGE ARTIFACT: the same asymmetry (MLP ratio > 1, attention ratio < 1)
#          holds when every position is scored instead of only fit-covered ones. If FALSE the effect
#          is about which tokens are scored rather than about the modules, and the whole reading
#          above is wrong.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256; NB = 2000
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/site_additivity_results.json'
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205, 'discovery'),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711, 'CONFIRMATION')]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
CONSTS = PT + 'opt_ablation_consts_all.pt'
H = m.transformer.h
S1662_JOINT = {'mlp': 4.3301, 'attn': 3.5570}
COV = {}


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


def const_hook(c):
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        sub = c.to(y.dtype).expand_as(y)
        return (sub,) + tuple(out[1:]) if isinstance(out, tuple) else sub
    return hook


def mod_of(kind, L):
    return H[L].mlp if kind == 'mlp' else H[L].attn


@torch.no_grad()
def per_row(rows, hooks=()):
    """Per-ROW loss sums and counts, under BOTH scoring populations in one pass: `cov` restricts to
    positions whose input token appeared in the fit rows (the arc's standard), `all` scores every
    position from 64 on. pred_d needs both and they cost one forward, not two."""
    n = rows.shape[0]
    # float64 on the CPU side. Removal is a DIFFERENCE OF LARGE CE SUMS -- the attention stack's
    # 1.42 nats sits on top of ~95,000 nats of total loss -- so float32 accumulation across 18 sites
    # leaves ~3e-5 of noise, which is 20x the difference being asserted. Same failure shape as the
    # 1e-9 tolerance that fired at 1.41e-08 earlier in this arc; the fix is the accumulator, not the
    # tolerance.
    acc = {k: {'s': torch.zeros(n, dtype=torch.float64),
               'k': torch.zeros(n, dtype=torch.float64)} for k in ('cov', 'all')}
    hs = list(hooks)
    try:
        for i in range(0, n, 8):
            bb = rows[i:i + 8]
            idx = bb[:, :-1].to(DEV).contiguous()
            x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
            for blk in H:
                x, v1 = blk(x, v1, x0)
            lg = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
            tg = bb[:, 1:].to(DEV)
            e = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                                reduction='none').reshape(tg.shape)[:, 64:]
            msk = {'cov': COV['seen'][idx[:, 64:]], 'all': torch.ones_like(e, dtype=torch.bool)}
            for k in acc:
                acc[k]['s'][i:i + bb.shape[0]] = (e.double() * msk[k]).sum(1).cpu()
                acc[k]['k'][i:i + bb.shape[0]] = msk[k].sum(1).double().cpu()
    finally:
        for h in hs:
            h.remove()
    return acc


def rm(a, l, pop, sel=None):
    """Removal in nats: constant-ablated CE minus live CE, over the selected rows."""
    def ce(d):
        s = d[pop]['s'] if sel is None else d[pop]['s'][sel]
        k = d[pop]['k'] if sel is None else d[pop]['k'][sel]
        return float(s.sum()) / float(k.sum())
    return ce(a) - ce(l)


def agg(inds, live, pop):
    """Per-row sum of the 18 individual removals, as one array.

    The position count K is identical across sites -- the mask does not depend on which module is
    hooked -- so sum_i (S_i/K - L/K) collapses to (sum_i S_i - 18 L) / K. Precomputing sum_i S_i
    turns each bootstrap draw from 72 index-selects into 2. It is an algebraic identity, not an
    approximation, and the run asserts it against the direct sum before using it.
    """
    tot = torch.zeros_like(live[pop]['s'])
    for d in inds:
        tot = tot + d[pop]['s']
    return tot - len(inds) * live[pop]['s']


@torch.no_grad()
def main():
    t0 = time.time()
    K = torch.load(CONSTS, map_location='cpu')
    fit = load(FIT_ROWS)
    seen = torch.zeros(50257, dtype=torch.bool)
    seen[fit[:, :T].reshape(-1).long()] = True
    COV['seen'] = seen.to(DEV)
    sites = [(k, L) for k in ('mlp', 'attn') for L in range(18)]
    stacks = {'mlp': [s for s in sites if s[0] == 'mlp'],
              'attn': [s for s in sites if s[0] == 'attn']}
    out = {}
    print(f'SITE ADDITIVITY | sum of 36 individual removals vs the two joint stack removals | '
          f'{NB} row-level draws | covered and all-position scoring in one pass', flush=True)

    for ename, epath, ce_ref, role in EVAL_SETS:
        ev = load(epath)
        live = per_row(ev)
        ncov = int(live['cov']['k'].sum()); nall = int(live['all']['k'].sum())
        base = float(live['cov']['s'].sum()) / ncov
        assert abs(base - ce_ref) <= 1e-2, f'{ename} baseline CE {base:.5f} != {ce_ref}'
        assert nall > ncov > 0, f'coverage mask is vacuous: cov {ncov} all {nall}'
        print(f'\n  {ename} [{role}]: baseline CE {base:.5f} (ref {ce_ref}) | covered {ncov} of '
              f'{nall} positions', flush=True)

        ind = {}
        for st in sites:
            ind[f'{st[0]}{st[1]}'] = per_row(ev, hooks=[mod_of(*st).register_forward_hook(
                const_hook(K[f'{st[0]}{st[1]}'].to(DEV).float()))])
        jnt = {kind: per_row(ev, hooks=[mod_of(*s).register_forward_hook(
            const_hook(K[f'{s[0]}{s[1]}'].to(DEV).float())) for s in sl])
            for kind, sl in stacks.items()}

        g = torch.Generator().manual_seed(1735)
        nrow = ev.shape[0]
        sels = [torch.randint(0, nrow, (nrow,), generator=g) for _ in range(NB)]
        res = {}
        for pop in ('cov', 'all'):
            r = {}
            for kind, sl in stacks.items():
                s_ind = sum(rm(ind[f'{s[0]}{s[1]}'], live, pop) for s in sl)
                s_jnt = rm(jnt[kind], live, pop)
                ag = agg([ind[f'{s[0]}{s[1]}'] for s in sl], live, pop)
                kk = live[pop]['k']
                # scale the tolerance to the CE magnitude being differenced, not to the small
                # difference itself: the identity is exact in real arithmetic, so what is allowed is
                # float64 rounding on ~3.3 nats/token across 18 sites.
                itol = max(1e-9 * 18 * (float(live[pop]['s'].sum()) / float(kk.sum())), 1e-9)
                idev = abs(float(ag.sum()) / float(kk.sum()) - s_ind)
                assert idev <= itol, (
                    f'aggregate identity off by {idev:.3e} against tolerance {itol:.3e}')
                dr = sorted((float(ag[sl2].sum()) / float(kk[sl2].sum())
                             / max(rm(jnt[kind], live, pop, sl2), 1e-9)) for sl2 in sels)
                r[kind] = {'sum_individual': round(s_ind, 5), 'joint': round(s_jnt, 5),
                           'ratio': round(s_ind / s_jnt, 5),
                           'ratio_ci95': (round(dr[int(0.025 * NB)], 4),
                                          round(dr[int(0.975 * NB)], 4))}
                print(f'    [{pop:3s}] {kind:4s}  sum of 18 individual {s_ind:8.4f}  joint '
                      f'{s_jnt:8.4f}  ratio {r[kind]["ratio"]:6.3f}  95% CI {r[kind]["ratio_ci95"]}',
                      flush=True)
            res[pop] = r
        out[ename] = {'role': role, 'baseline_ce': round(base, 5), 'covered': ncov, 'all': nall,
                      'by_population': res,
                      'per_site_removal_cov': {n: round(rm(ind[n], live, 'cov'), 5) for n in ind}}
        del ev, ind, jnt
        torch.cuda.empty_cache()

    cf = out['skip11000']['by_population']['cov']
    pa = cf['mlp']['ratio_ci95'][0] > 1.5 and cf['attn']['ratio_ci95'][1] < 0.7
    pb = all(out[e]['by_population']['cov']['mlp']['ratio'] > 1.0
             and out[e]['by_population']['cov']['attn']['ratio'] < 1.0 for e in out)
    pc = all(abs(out['skip7000']['by_population']['cov'][k]['joint'] - v) <= 0.01
             for k, v in S1662_JOINT.items())
    pd = all(out[e]['by_population']['all']['mlp']['ratio'] > 1.0
             and out[e]['by_population']['all']['attn']['ratio'] < 1.0 for e in out)

    print(f'\n  CONFIRMATION skip11000: mlp ratio CI {cf["mlp"]["ratio_ci95"]} vs bar >1.5, attn '
          f'CI {cf["attn"]["ratio_ci95"]} vs bar <0.7 -> {pa}', flush=True)
    print(f'  sign of the asymmetry stable across both roles -> {pb}', flush=True)
    print(f'  JOINT stakes reproduce §1662/§1682 (the control §1735 got wrong) -> {pc}', flush=True)
    print(f'  asymmetry survives all-position scoring -> not a coverage artifact {pd}', flush=True)

    r = {'config': {'eval_sets': [e[0] for e in EVAL_SETS], 'bootstrap_draws': NB,
                    'bootstrap': 'ROW-level clusters; NOT document-clustered (§1701)',
                    'measure': 'sum of the 18 one-at-a-time constant-ablation removals divided by '
                               'the removal when all 18 are ablated together. Above 1 = the sites '
                               'are REDUNDANT (others absorb the loss). Below 1 = COOPERATIVE (the '
                               'contribution only exists jointly).',
                    'roles': 'skip7000 DISCOVERY (its sums were printed in §1735); skip11000 '
                             'CONFIRMATION (computed inside §1735 but never printed, stored or seen)'},
         'results': out,
         'predictions': {'pred_a_confirmed_both_directions': bool(pa),
                         'pred_b_sign_stable': bool(pb),
                         'pred_c_joint_stakes_reproduce': bool(pc),
                         'pred_d_not_a_coverage_artifact': bool(pd)},
         'runtime_s': round(time.time() - t0, 1)}
    json.dump(r, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
