"""E14 SLOT SATURATION (Logan 2026-08-06: is the partition cost caused by
modules running at max slot capacity? fresh single-epoch batch-16, recipe
conventions, paired vs E9a).

E14a -- SLOT-UTILIZATION CENSUS (probe only) on the readable-recipe
  checkpoint qk_e9_a.pt AND the slots-only checkpoint qk_e5_slots264.pt.
  Per module: (i) singular spectrum of the write projection restricted to its
  slot rows (attention: c_proj slot rows; mlp: Down slot rows), (ii)
  effective rank (participation ratio) of the module's write covariance over
  held data (streamed 11x11 covariances), (iii) utilization = effective rank
  / slot dim (11), flagged saturated (> 0.8) / moderate / slack (< 0.4),
  plus the write-norm distribution. Discriminates Logan's three hypotheses:
  SATURATION (many saturated), RIGIDITY (bimodal saturated+slack), NEITHER
  (all mid); a mechanical verdict is recorded with the counts.

E14b -- VARIABLE SLOT ALLOCATION (train): width 264, total write budget
  unchanged (the same 264 slot dims split across the 24 modules), slot sizes
  proportional to E14a's measured utilization on the readable recipe
  (rounded, min 4 dims; the exact allocation is documented in the JSON).
  Full readable recipe (per-slot norm over the VARIABLE slots, Muon, in-loss
  lasso 3e-5), paired vs E9a -- the equal-slots twin. REGISTERED PREDICTION:
  if rigidity is real this claws back CE at zero parameter cost. If the
  census happens to imply the uniform allocation, the arm is skipped as
  identical to E9a (noted).

E14c -- COMMONS ARM: equal slots shrunk to 9 dims (216 total) + one shared
  48-dim unpartitioned subspace all modules may write into; the lasso treats
  the commons as a 25th read column group (unweighted, harness convention --
  the sqrt-size weighting was an E10-specific exception, noted). Tests the
  forbidden-sharing hypothesis: if duplication is the cost, a small commons
  recovers disproportionate CE at a known readability price on commons
  content. Wiring probe both ways: the standard per-module weight-vs-causal
  Spearman AND the commons ledger (per-module commons write norms +
  per-consumer commons read-group norms -- the unattributable channel).

Identity controls: the variable-slot class with the uniform 11-dim allocation
and no commons reproduces the E9a architecture forward exactly at init; the
segment penalty matches a naive loop; commons masks are structurally checked.
Curves in JSON, idempotent, non-blocking guards. Results -> qk_e14.json."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import math

import numpy as np

import qk_e_common as E
from qk_e_common import Q, V8T, C, R2, DEPTH, F, nn, torch
import qk_e1_slotnorm_run as E1R
import qk_e7_evenout_run as E7R

JP = E.jpath('qk_e14.json')
GC14 = 3e-5
MIN_SLOT = 1 if E.SMOKE else 4
N_CENSUS = 8 if E.SMOKE else 64

if not hasattr(V8T, '_e11_penalty_patch'):       # custom-penalty dispatch
    V8T._e11_penalty_patch = True
    _prev_pen14 = V8T.group_penalty

    def _e14_dispatch(model):
        if hasattr(model, 'custom_group_penalty'):
            return model.custom_group_penalty()
        return _prev_pen14(model)
    V8T.group_penalty = _e14_dispatch


# ---------------- E14a: utilization census ----------------
@torch.no_grad()
def census(model, label):
    """Uniform-slot census (both existing checkpoints have 11-dim slots)."""
    S = Q.D // E.NGROUP
    covs = [torch.zeros(S, S, dtype=torch.float64, device=E.DEV)
            for _ in range(E.NGROUP)]
    means = [torch.zeros(S, dtype=torch.float64, device=E.DEV)
             for _ in range(E.NGROUP)]
    norm_sum = torch.zeros(E.NGROUP, dtype=torch.float64, device=E.DEV)
    n = 0
    for i in range(0, N_CENSUS, 8):
        b = Q.HELD[i:i + 8, :Q.T]
        col = {'entry_norm': [], 'attn_write': [], 'mlp_write': []}
        model(b, collect=col)
        for l in range(DEPTH):
            for j, w in ((2 * l, col['attn_write'][l]),
                         (2 * l + 1, col['mlp_write'][l])):
                x = w[..., S * j:S * (j + 1)].double().reshape(-1, S)
                covs[j] += x.t() @ x
                means[j] += x.sum(0)
                norm_sum[j] += x.norm(dim=-1).sum()
        n += b.shape[0] * Q.T
    rows = []
    for j in range(E.NGROUP):
        l, kind = j // 2, ('attn' if j % 2 == 0 else 'mlp')
        mu = means[j] / n
        cov = covs[j] / n - torch.outer(mu, mu)
        ev = torch.linalg.eigvalsh(cov).clamp_min(0)
        pr = float(ev.sum() ** 2 / ev.pow(2).sum().clamp_min(1e-30))
        util = pr / S
        blk = model.h[l]
        Wp = (blk.c_proj.weight if kind == 'attn' else blk.Down.weight)
        sv = torch.linalg.svdvals(
            Wp[S * j:S * (j + 1), :].detach().float()).cpu()
        rows.append({
            'module': f'{kind}{l}',
            'proj_slot_singular_values': [round(float(x), 4) for x in sv],
            'write_cov_effective_rank': round(pr, 3),
            'utilization': round(util, 3),
            'flag': ('saturated' if util > 0.8 else
                     'slack' if util < 0.4 else 'moderate'),
            'mean_write_norm': round(float(norm_sum[j] / n), 4)})
    n_sat = sum(1 for r in rows if r['flag'] == 'saturated')
    n_slk = sum(1 for r in rows if r['flag'] == 'slack')
    n_mod = E.NGROUP - n_sat - n_slk
    if n_sat >= E.NGROUP // 3:
        verdict = 'SATURATION (many modules at capacity)'
    elif n_sat + n_slk > n_mod:
        verdict = 'RIGIDITY (bimodal: saturated + slack coexist)'
    else:
        verdict = 'NEITHER (utilization mostly mid-range)'
    print(f"census {label}: saturated {n_sat} / moderate {n_mod} / slack "
          f"{n_slk} -> {verdict}", flush=True)
    return {'modules': rows, 'n_saturated': n_sat, 'n_moderate': n_mod,
            'n_slack': n_slk, 'verdict': verdict,
            'slot_dim': S, 'n_held_seqs': N_CENSUS}


# ---------------- variable-slot architecture ----------------
class VarSlotRoute(E1R.E1Route):
    """E1Route with VARIABLE write-slot sizes (and an optional shared commons
    segment all modules write into). Norm segments = the slots (+ commons);
    each segment is normalized with its own F.rms_norm slice, so the uniform
    11-dim allocation reduces bit-for-bit to E1Route's grouped view."""

    def __init__(self, variant, depth, sizes, commons=0):
        super().__init__(variant, depth)
        Dm = self.wte.weight.shape[1]
        assert sum(sizes) + commons == Dm and len(sizes) == 2 * depth
        self.seg_sizes = list(sizes) + ([commons] if commons > 0 else [])
        self.commons = commons
        bounds, off = [], 0
        for s in self.seg_sizes:
            bounds.append((off, off + s))
            off += s
        self.seg_bounds = bounds
        wm = torch.zeros(2 * depth, Dm)
        for k in range(2 * depth):
            a, b = bounds[k]
            wm[k, a:b] = 1.0
            if commons > 0:
                wm[k, Dm - commons:] = 1.0        # everyone writes the commons
        with torch.no_grad():
            self.wmask.copy_(wm)
        ind = torch.zeros(Dm, len(self.seg_sizes))
        for g, (a, b) in enumerate(bounds):
            ind[a:b, g] = 1.0
        self.register_buffer('seg_ind', ind)

    def slot_norm(self, x):
        out = torch.empty_like(x)
        for (a, b) in self.seg_bounds:
            if b > a:
                out[..., a:b] = F.rms_norm(x[..., a:b], (b - a,))
        return out

    def custom_group_penalty(self):
        tot = None
        for blk in self.h:
            for nm in E.READ_NAMES:
                M = getattr(blk, nm).weight
                colsq = M.pow(2).sum(0)
                g = (colsq @ self.seg_ind + 1e-12).sqrt().sum()
                tot = g if tot is None else tot + g
        return tot


def alloc_from_census(cen):
    """Slot sizes proportional to measured effective rank, min MIN_SLOT,
    summing exactly to the width."""
    er = np.array([r['write_cov_effective_rank'] for r in cen['modules']])
    D = Q.D
    raw = er / er.sum() * D
    sizes = np.maximum(MIN_SLOT, np.round(raw)).astype(int)
    while sizes.sum() > D:                        # trim the largest
        k = int(np.argmax(sizes - raw))
        if sizes[k] > MIN_SLOT:
            sizes[k] -= 1
        else:
            sizes[int(np.argmax(sizes))] -= 1
    while sizes.sum() < D:                        # grow the most-shortchanged
        sizes[int(np.argmin(sizes - raw))] += 1
    assert sizes.sum() == D and (sizes >= MIN_SLOT).all()
    return sizes.tolist()


def make_e14b(sizes):
    C.register('E14b')
    torch.manual_seed(Q.SEED)
    m = VarSlotRoute('E14b', DEPTH, sizes).to(E.DEV)
    return m


def make_e14c():
    C.register('E14c')
    torch.manual_seed(Q.SEED)
    if E.SMOKE:
        sizes, commons = [1] * 20 + [0] * 4, 4    # 0-size slots: commons-only
    else:
        sizes, commons = [9] * 24, 48
    return VarSlotRoute('E14c', DEPTH, sizes, commons=commons).to(E.DEV)


# ---------------- controls ----------------
@torch.no_grad()
def controls():
    idx = Q.HELD[:2, :Q.T]
    S = Q.D // E.NGROUP
    C.register('E14ctl')
    C.register('E14ctl2')
    ref = E7R.make_e7m1().eval().float()
    torch.manual_seed(Q.SEED)
    m = VarSlotRoute('E14ctl', DEPTH, [S] * E.NGROUP).to(E.DEV).eval().float()
    m.norm_groups = E.NGROUP
    d = (m(idx) - ref(idx)).abs().max().item()
    print(f"control VarSlot(uniform {S}-dim)==E9a-arch at init: "
          f"max |logit diff| {d:.2e}", flush=True)
    assert (d == 0.0) if E.SMOKE else (d < 1e-6)
    # segment penalty vs naive loop (on a genuinely variable allocation)
    torch.manual_seed(Q.SEED)
    sizes = ([2] * 8 + [1] * 8 + [0] * 8 if E.SMOKE
             else [15] * 8 + [11] * 8 + [7] * 8)
    mv = VarSlotRoute('E14ctl2', DEPTH, sizes,
                      commons=Q.D - sum(sizes)).to(E.DEV).eval().float()
    p_fast = float(mv.custom_group_penalty())
    p_naive = 0.0
    for blk in mv.h:
        for nm in E.READ_NAMES:
            M = getattr(blk, nm).weight
            for (a, b) in mv.seg_bounds:
                if b > a:
                    p_naive += float(M[:, a:b].pow(2).sum()) ** 0.5
    rel = abs(p_fast - p_naive) / p_naive
    print(f"control E14 segment penalty fast {p_fast:.4f} vs naive "
          f"{p_naive:.4f} rel {rel:.2e}", flush=True)
    assert rel < 1e-6
    del m, mv, ref
    torch.cuda.empty_cache()
    # commons structural checks
    mc = make_e14c().eval().float()
    cdim = mc.commons
    assert cdim > 0
    assert bool((mc.wmask[:, Q.D - cdim:] == 1).all())
    assert sum(mc.seg_sizes) == Q.D
    out = mc(idx)
    assert torch.isfinite(out).all()
    print(f"control E14c structural: commons {cdim} dims shared by all 24 "
          f"write masks, segments sum {sum(mc.seg_sizes)}, forward finite",
          flush=True)
    del mc
    torch.cuda.empty_cache()


# ---------------- variable-slot wiring probe ----------------
def var_light_probe(m):
    """R2 consumption graph + weight support on the VARIABLE column groups
    (commons excluded from per-module support; reported in its own ledger)."""
    m.eval().float()
    base, means = R2.base_and_means(m)
    dce = R2.consumption_graph(m, base, means)
    sup = {}
    for li in range(DEPTH + 1):
        row = {}
        for k in range(E.NGROUP):
            si = k + 1
            if si not in m.vis[li] or si > 2 * min(li, DEPTH):
                continue
            a, b = m.seg_bounds[k]
            mats = ([m.wte.weight] if li == DEPTH else
                    [getattr(m.h[li], nm).weight for nm in E.READ_NAMES])
            row[si] = float(sum(M[:, a:b].float().pow(2).sum().item()
                                for M in mats)) ** 0.5 if b > a else 0.0
        sup[li] = row
    wp = [(li, si) for li in sup for si in sup[li]]
    sv = [sup[li][si] for li, si in wp]
    cv = [dce[li][si] for li, si in wp]
    eff = [k for k in range(len(wp)) if cv[k] > C.EFFECTUAL]
    top_s = set(np.argsort(sv)[::-1][:10].tolist())
    top_c = set(np.argsort(cv)[::-1][:10].tolist())
    r = {'base_ce_fp32_abl_oldheld': round(base, 5),
         'wiring_n_pairs': len(wp),
         'wiring_spearman_all': round(R2.spearman(sv, cv), 4),
         'wiring_n_effectual': len(eff),
         'wiring_spearman_effectual': round(R2.spearman(
             [sv[k] for k in eff], [cv[k] for k in eff]), 4),
         'wiring_top10_precision': len(top_s & top_c) / 10.0,
         'consumption_matrix': {str(li): {str(si): round(v, 6)
                                          for si, v in dce[li].items()}
                                for li in dce}}
    print(f"var wiring: Spearman all {r['wiring_spearman_all']} effectual"
          f"({len(eff)}) {r['wiring_spearman_effectual']} top10 "
          f"{r['wiring_top10_precision']}", flush=True)
    return r


@torch.no_grad()
def commons_ledger(m):
    """The readability price: per-module commons WRITE norms (held data) and
    per-consumer commons READ-group norms (weights)."""
    cdim = m.commons
    a = Q.D - cdim
    wsum = torch.zeros(E.NGROUP, dtype=torch.float64, device=E.DEV)
    n = 0
    for i in range(0, N_CENSUS, 8):
        b = Q.HELD[i:i + 8, :Q.T]
        col = {'entry_norm': [], 'attn_write': [], 'mlp_write': []}
        m(b, collect=col)
        for l in range(DEPTH):
            for j, w in ((2 * l, col['attn_write'][l]),
                         (2 * l + 1, col['mlp_write'][l])):
                wsum[j] += w[..., a:].double().norm(dim=-1).sum()
        n += b.shape[0] * Q.T
    write_norms = [round(float(x / n), 4) for x in wsum]
    reads = {}
    for li, blk in enumerate(m.h):
        reads[f'block{li}'] = round(float(sum(
            getattr(blk, nm).weight[:, a:].float().pow(2).sum()
            for nm in E.READ_NAMES)) ** 0.5, 3)
    reads['readout'] = round(float(
        m.wte.weight[:, a:].float().pow(2).sum()) ** 0.5, 3)
    def sname(k):
        return ('attn' if k % 2 == 0 else 'mlp') + str(k // 2)
    return {'commons_dims': cdim,
            'per_module_commons_write_norm':
                {sname(j): write_norms[j] for j in range(E.NGROUP)},
            'per_consumer_commons_read_norm': reads,
            'note': 'commons content is per-consumer readable in strength '
                    'but unattributable to a single writer'}


def pair_extra(stem, key, others):
    if E.SMOKE:
        return
    for ctl, label in others:
        f_arm, f_ctl = f'{stem}_heldloss.npy', f'{ctl}_heldloss.npy'
        if os.path.exists(f'{E.QK}/{f_ctl}') and os.path.exists(f'{E.QK}/{f_arm}'):
            E.merge(JP, f'{key}_minus_{label}_fresh',
                    E.paired(f_arm, f_ctl, len(Q.HELD), label))


def train_var_arm(stem, key, factory, design):
    E.train_arm(stem, JP, key, factory, GC14, lr=E7R.muon_lr(),
                trainer=lambda lr, gc, steps, **kw: E.train_muon(
                    lr, gc, steps, lr_adamw=E.get_lr(), **kw),
                extra={'optimizer': 'muon', 'design': design})
    E.oldheld_record(stem, factory, JP, f'{key}_oldheld')
    E.paired_fresh(stem, JP, key)
    pair_extra(stem, key, (('qk_e9_a', 'e9a'),))
    if not E.SMOKE and os.path.exists(E.ckpath(stem)) \
            and f'light_probe_{key}' not in E.loadj(JP):
        m, _ = E.load_arm(stem, factory)
        E.merge(JP, f'light_probe_{key}', var_light_probe(m))
        del m
        torch.cuda.empty_cache()


if __name__ == '__main__':
    E.setup()
    controls()

    # ---- E14a: census on both checkpoints ----
    for stem, factory, key in (('qk_e9_a', E7R.make_e7m1, 'census_e9a'),
                               ('qk_e5_slots264', E.make_e0b,
                                'census_e5slots')):
        if key in E.loadj(JP):
            continue
        if not os.path.exists(E.ckpath(stem)):
            if E.SMOKE:
                mm = factory()
                torch.save({'state_dict': mm.state_dict(), 'config': {},
                            'log': {}}, E.ckpath(stem))
                del mm
            else:
                print(f"E14a: {stem} missing -- skipped", flush=True)
                continue
        m, _ = E.load_arm(stem, factory)
        E.merge(JP, key, census(m, key))
        del m
        torch.cuda.empty_cache()

    # ---- E14b: utilization-proportional allocation ----
    cen = E.loadj(JP).get('census_e9a')
    if cen is None:
        print("E14b: no census -- cannot allocate; skipped", flush=True)
    else:
        sizes = alloc_from_census(cen)
        E.merge(JP, 'E14b_allocation', {
            'sizes': sizes, 'min_slot': MIN_SLOT, 'total': sum(sizes),
            'rule': 'proportional to write-covariance effective rank on the '
                    'readable recipe, rounded, min 4, adjusted to sum 264',
            'prediction': 'if rigidity is real, variable allocation claws '
                          'back CE vs E9a at zero parameter cost'})
        S = Q.D // E.NGROUP
        if sizes == [S] * E.NGROUP:
            E.merge(JP, 'E14b_skipped',
                    'census implies the uniform allocation; the arm would be '
                    'identical to E9a')
        else:
            train_var_arm('qk_e14_b', 'E14b', lambda: make_e14b(sizes),
                          f'variable slots {min(sizes)}..{max(sizes)} dims, '
                          f'utilization-proportional, budget 264 unchanged')

    # ---- E14c: commons arm ----
    train_var_arm('qk_e14_c', 'E14c', make_e14c,
                  'equal 9-dim slots (216) + shared 48-dim commons all '
                  'modules write; lasso treats commons as the 25th read '
                  'group (unweighted, harness convention)')
    if not E.SMOKE and os.path.exists(E.ckpath('qk_e14_c')) \
            and 'E14c_commons_ledger' not in E.loadj(JP):
        m, _ = E.load_arm('qk_e14_c', make_e14c)
        E.merge(JP, 'E14c_commons_ledger', commons_ledger(m))
        del m
        torch.cuda.empty_cache()
    print('e14 slotcap run done', flush=True)
