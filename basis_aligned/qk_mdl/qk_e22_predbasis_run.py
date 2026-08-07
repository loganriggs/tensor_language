"""E22 PREDICATE-BASIS ATTENTION -- localization probe on the frontier arm
(fresh single-epoch batch-16, recipe conventions; results -> qk_e22.json).

CONTEXT (BRAINSTORM_STATE "E21 census RESOLVED"): the slotted models' heads
are ~all positional; a MATCH_prev family EXISTS but is weak and DISTRIBUTED
(E19a: 5 programmatic heads at ~7-9% pattern mass, z ~ 2500-4400, causally
confirmed). The predicate-basis arm asks the localization question: if the
architecture provides an EXPLICIT named match term per head, does the
distributed component concentrate into it? This is a PROBE, not a recipe
candidate (per the census branch resolution).

BASE: the E19a frontier arm verbatim -- qk_e15_reinvest_run.make_e15c(s=15)
(true-small decoders, 24 x 15-dim slots, stream 360, compute 264, hidden
1056) + in-loss group-lasso 1e-4, Muon 0.02 / AdamW 0.004, same seed and
data order. Parent numbers: CE 4.9742, plain Spearman 0.7911, cov-composed
0.8259 (qk_e19.json).

CHANGE (E22a): each head's pattern becomes
    P_h[offset]  (learned per-head signed positional profile over offsets
                  0..T-1, init 0; the brief's a_h * P_h(offset) with the
                  redundant amplitude a_h absorbed into P_h -- a product of
                  two trainables both init 0 is the dead-gradient trap)
  + b_h * K_MATCHprev + c_h * K_MATCHsame
                 (binary kernels EXACTLY as in qk_e21_census_run.build_feats:
                  MATCH_prev = 1[tok_{j-1} == tok_i] with prevtok[0] = -1,
                  MATCH_same = 1[tok_j == tok_i] diagonal included, causal
                  masked; signed scalar mixture weights, init 0)
  + the standard bilinear residual pattern (s1 * s2, unchanged).
Patterns are signed (no softmax) so the library is signed by construction.
Positional profiles are applied to the final pattern (the model has RoPE
inside the bilinear path; the profile term is additive post-product).
Mixture weights + profiles train on AdamW (family lr, no weight decay) via
the muon_exclude mechanism -- NOT Muon (component customizations).

POSITIVE CONTROLS (before training, hard asserts):
  1. PRED-ZERO: with the predicate terms disabled (and, informationally, at
     init even when enabled -- all predicate parameters are exact zeros) the
     E22 model reproduces the E19a parent BIT-EXACTLY: shared-parameter
     identity, forward max |logit diff| == 0, and a 3-step run of this
     runner's trainer matches the parent trainer's 3-step trajectory exactly
     (same seed, same epoch_order(0) data).
  2. KERNELS: this runner's MATCH_prev / MATCH_same kernels match
     qk_e21_census_run.build_feats features 1 / 0 on one batch EXACTLY
     (max abs diff == 0).

REGISTERED PREDICTIONS (merged BEFORE training):
  (i)   concentration: the top-3 |b_h| over the 72 heads carry >= 60% of the
        total |b| mass;
  (ii)  absorption: the E21 census RE-RUN on the residual (bilinear-only)
        patterns shows MATCH_prev mass DROPS vs the parent's stored census
        (total eval-half cos^2 over all 72 heads, and the mean over the
        parent's 5 programmatic MATCH_prev heads);
  (iii) CE within +0.03 of the E19a parent (paired).

MEASURES: paired CE vs E19a / E9a / E0a+E0b; the full mixture-weight tables
(b_h, c_h, profile norms + top offsets, per-head pattern-mass shares of each
named term); the E21 census re-run on residual patterns (qk_e21_census_run
census_model + null_zscores reused, forward = this model with the census
callback reporting the pre-predicate bilinear pattern) compared per-head to
the parent's stored census; the standard wiring probes (generalized
variable-slot-dim light probe + covariance-composed, qk_e18 gated
machinery). Wiring-trajectory logging (qk_e_common standing upgrade) ->
qk_e22_a_traj.npz. Idempotent on qk_e22.json keys and the checkpoint."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json
import math
import time

import numpy as np

import qk_e_common as E
from qk_e_common import Q, V8T, C, DEPTH, F, nn, torch
import qk_e7_evenout_run as E7R
import qk_e15_reinvest_run as E15R        # E15cRoute + make_e15c + solve_slot_c
import qk_e18_probe_upgrades as E18U      # generalized probe + cov-composed
import qk_e17_composed_wiring as E17      # agreement
import qk_deeproute_train_2 as R2

# The E17 import (via E18U) sets E.DEV='cpu' for its own weight-only use --
# restore the runner convention.
E.DEV = 'cpu' if E.SMOKE else 'cuda'

JP = E.jpath('qk_e22.json')
GC22 = 1e-4                               # the E19a lasso, unchanged
GATE_TOL = 1e-3
NG = E.NGROUP

_E21 = None


def get_e21():
    """Lazy import of qk_e21_census_run (its module body polls the GPU and
    sets global state); E.DEV and the tf32 flags are snapshotted/restored,
    its DEV is pinned to ours, and KCLS (token classes) is built once."""
    global _E21
    if _E21 is None:
        dev_saved = E.DEV
        tf32_m = torch.backends.cuda.matmul.allow_tf32
        tf32_c = torch.backends.cudnn.allow_tf32
        import qk_e21_census_run as E21
        E.DEV = dev_saved
        E21.DEV = dev_saved
        torch.backends.cuda.matmul.allow_tf32 = tf32_m
        torch.backends.cudnn.allow_tf32 = tf32_c
        if E21.KCLS is None:
            E21.KCLS = E21.token_classes()
        _E21 = E21
    return _E21


# ---------------- kernels (E21 semantics, verified by control 2) ----------------
def match_kernels(idx, maskf):
    """MATCH_prev = 1[tok_{j-1} == tok_i] (prevtok[0] = -1), MATCH_same =
    1[tok_j == tok_i] (diagonal included), causal-masked -- exactly
    qk_e21_census_run.build_feats features 1 and 0."""
    prevtok = torch.roll(idx, 1, 1)
    prevtok[:, 0] = -1
    qtok = idx.unsqueeze(2)                       # (B, T, 1): query along dim 1
    kprev = (prevtok.unsqueeze(1) == qtok).float() * maskf
    ksame = (idx.unsqueeze(1) == qtok).float() * maskf
    return kprev, ksame


# ---------------- the E22 architecture ----------------
class E22Route(E15R.E15cRoute):
    """E15c + per-head named pattern terms: signed positional profile +
    b_h * MATCH_prev + c_h * MATCH_same, all init 0 (bit-exact parent at
    init). Predicate parameters train on AdamW (muon_exclude)."""

    def __init__(self, variant, depth, s, Dc, NH, HD, hidden):
        super().__init__(variant, depth, s, Dc, NH, HD, hidden)
        self.pred_on = True
        self.muon_exclude = ('pred_',)
        # all zeros: no RNG consumed -> shared params bit-identical to E15c
        self.pred_prof = nn.Parameter(torch.zeros(depth, NH, Q.T))
        self.pred_b = nn.Parameter(torch.zeros(depth, NH))
        self.pred_c = nn.Parameter(torch.zeros(depth, NH))
        ar = torch.arange(Q.T)
        self.register_buffer('offmat',
                             (ar[:, None] - ar[None, :]).clamp(min=0))

    def pred_terms(self, l, Kprev, Ksame, maskf, Tq):
        """(B, NH, Tq, Tq) named pattern terms for block l."""
        prof = self.pred_prof[l][:, self.offmat[:Tq, :Tq]] * maskf
        return (prof[None]
                + self.pred_b[l].view(1, -1, 1, 1) * Kprev[:, None]
                + self.pred_c[l].view(1, -1, 1, 1) * Ksame[:, None])

    def forward(self, idx, collect=None, sub_entry=None, census_cb=None,
                census_full_cb=None, pat_hook=None):
        B, Tq = idx.shape
        Ws, s = self.Ws, self.s
        cos = self.cos[None, :Tq, None, :]
        sin = self.sin[None, :Tq, None, :]
        mask = self.mask[:Tq, :Tq]
        maskf = mask.float()
        if self.pred_on:
            Kprev, Ksame = match_kernels(idx, maskf)
        e = F.rms_norm(self.wte(idx), (Ws,))
        streams = [e]

        def entry(li):
            sub = sub_entry.get(li) if sub_entry is not None else None
            tot = None
            for i in self.vis[li]:
                v = sub[i] if (sub is not None and i in sub) else streams[i]
                tot = v if tot is None else tot + v
            return tot

        for l, blk in enumerate(self.h):
            x = entry(l)
            if collect is not None:
                collect.setdefault('entry_norm', []).append(
                    x.detach().float().norm(dim=-1).mean().item())
            hn = self.slot_norm(x)

            def qk(lin):
                z = lin(hn).view(B, Tq, self.NH, self.HD)
                return Q.apply_rot(F.rms_norm(z, (self.HD,)), cos, sin)

            q, k = qk(blk.c_q), qk(blk.c_k)
            q2, k2 = qk(blk.c_q2), qk(blk.c_k2)
            v = blk.c_v(hn).view(B, Tq, self.NH, self.HD)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / self.HD
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / self.HD
            pat = (s1 * s2).masked_fill(~mask, 0.0)
            if census_cb is not None:
                census_cb(l, pat)                 # RESIDUAL bilinear pattern
            if self.pred_on:
                pat = pat + self.pred_terms(l, Kprev, Ksame, maskf,
                                            Tq).to(pat.dtype)
            if pat_hook is not None:
                pat = pat_hook(l, pat)
            if census_full_cb is not None:
                census_full_cb(l, pat)            # full pattern actually used
            y = torch.einsum('bhqk,bkhd->bqhd', pat, v)
            y = y.reshape(B, Tq, self.Dc)
            aw = torch.zeros(B, Tq, Ws, device=y.device, dtype=y.dtype)
            aw[..., s * 2 * l:s * (2 * l + 1)] = blk.c_proj(y)
            x = x + aw
            xn = self.slot_norm(x)
            mw_s = blk.Down(blk.Left(xn) * blk.Right(xn)) + blk.Down_bias
            mw = torch.zeros(B, Tq, Ws, device=y.device, dtype=y.dtype)
            mw[..., s * (2 * l + 1):s * (2 * l + 2)] = mw_s
            if collect is not None:
                collect.setdefault('attn_write', []).append(aw.detach())
                collect.setdefault('mlp_write', []).append(mw.detach())
            streams.append(aw)
            streams.append(mw)
        x = F.rms_norm(entry(self.depth), (Ws,))
        logits = x @ self.wte.weight.t()
        return 30 * torch.tanh(logits / 30)


def make_e22(s=None, pred_on=True):
    hidden = 4 * Q.D
    if s is None:
        s, _ = E15R.solve_slot_c(hidden)
    torch.manual_seed(Q.SEED)
    m = E22Route('E22a', DEPTH, s, Q.D, Q.NH, Q.HD, hidden).to(E.DEV)
    m.pred_on = pred_on
    return m


def trainer(lr, gc, steps, **kw):
    """EXACTLY the E19a trainer expression."""
    return E.train_muon(lr, gc, steps, lr_adamw=E.get_lr(), **kw)


def three_step(factory, gc, steps=3):
    log = E.train_muon(E7R.muon_lr(), gc, steps, log_every=1, save=False,
                       factory=factory, lr_adamw=E.get_lr())
    return {'per_step_ce': [x[1] for x in log['train_loss']],
            'held100_ce': log['final_held_ce']}


# ---------------- controls ----------------
def control_e18_gates():
    if E.SMOKE:
        return
    e18 = E.loadj(E.jpath('qk_e18.json'))
    g1 = e18.get('gate1_uniform11_weight_support', {}).get('pass')
    g2 = e18.get('gate2_cov_composed_E9a', {}).get('pass')
    assert g1 and g2, ('qk_e18.json gates not passed -- the reused probe '
                       'functions are not validated', g1, g2)
    print("precondition: qk_e18.json gates 1+2 passed", flush=True)


def control_pred_zero(s_c):
    key = 'control1_pred_zero'
    done = E.loadj(JP).get(key)
    if done and done.get('pass'):
        print(f"{key}: already passed -- skip", flush=True)
        return
    m19 = E15R.make_e15c(s=s_c).eval().float()
    m22 = make_e22(s=s_c, pred_on=False).eval().float()
    p19 = dict(m19.named_parameters())
    pdiff = max(float((p - p19[nm]).abs().max())
                for nm, p in m22.named_parameters() if nm in p19)
    idx = E.OLD_HELD[:2, :Q.T]
    with torch.no_grad():
        d_off = float((m22(idx) - m19(idx)).abs().max())
        m22.pred_on = True                        # params are exact zeros
        d_on = float((m22(idx) - m19(idx)).abs().max())
        m22.pred_on = False
    del m19, m22
    torch.cuda.empty_cache()
    parent = three_step(lambda: E15R.make_e15c(s=s_c), GC22)
    mine = three_step(lambda: make_e22(s=s_c, pred_on=False), GC22)
    step_diff = max(abs(a - b) for a, b in
                    zip(parent['per_step_ce'], mine['per_step_ce']))
    held_diff = abs(parent['held100_ce'] - mine['held100_ce'])
    rec = {'shared_param_identity_max_abs_diff': pdiff,
           'forward_pred_off_max_logit_diff': d_off,
           'forward_pred_on_at_init_max_logit_diff': d_on,
           'train3_parent_per_step_ce': parent['per_step_ce'],
           'train3_predzero_per_step_ce': mine['per_step_ce'],
           'train3_max_per_step_abs_diff': step_diff,
           'train3_held100_abs_diff': held_diff,
           'note': 'same seed (factory reseeds; predicate params are zeros, '
                   'no RNG consumed), same epoch_order(0) data; proves the '
                   'only E22 change vs the parent is the predicate terms',
           'pass': bool(pdiff == 0.0 and d_off == 0.0 and d_on == 0.0
                        and step_diff == 0.0 and held_diff < 1e-6)}
    E.merge(JP, key, rec)
    print(f"{key}: params {pdiff:.1e}, fwd off/on {d_off:.1e}/{d_on:.1e}, "
          f"3-step {step_diff:.1e}/{held_diff:.1e} -> "
          f"{'PASS' if rec['pass'] else 'FAIL'}", flush=True)
    assert rec['pass'], f'{key} FAILED'


def control_kernels():
    key = 'control2_kernels_vs_e21'
    done = E.loadj(JP).get(key)
    if done and done.get('pass'):
        print(f"{key}: already passed -- skip", flush=True)
        return
    E21 = get_e21()
    idx = Q.HELD[:4, :Q.T]
    maskf = torch.tril(torch.ones(Q.T, Q.T, dtype=torch.bool,
                                  device=idx.device)).float()
    Fs = E21.build_feats(idx, maskf)
    kp, ks = match_kernels(idx, maskf)
    d_prev = float((kp - Fs[:, 1]).abs().max())
    d_same = float((ks - Fs[:, 0]).abs().max())
    rec = {'batch': int(idx.shape[0]), 'T': Q.T,
           'match_prev_max_abs_diff': d_prev,
           'match_same_max_abs_diff': d_same,
           'note': 'this runner\'s kernels vs qk_e21_census_run.build_feats '
                   'features 1 (MATCH_prev) and 0 (MATCH_same)',
           'pass': bool(d_prev == 0.0 and d_same == 0.0)}
    E.merge(JP, key, rec)
    print(f"{key}: prev {d_prev:.1e}, same {d_same:.1e} -> "
          f"{'PASS' if rec['pass'] else 'FAIL'}", flush=True)
    assert rec['pass'], f'{key} FAILED'


# ---------------- mixture-weight tables ----------------
def parent_census_matchprev():
    """Stored parent (E19a) census MATCH_prev numbers for the comparison."""
    c = E.loadj(E.jpath('qk_e21_census.json'))
    t = c['models']['E19a_bandwidth_frontier']['table']
    per_head = {(r['layer'], r['head']): r['cos2_eval']['MATCH_prev']
                for r in t}
    prog = [(r['layer'], r['head']) for r in t
            if r['predicate_gain'] >= 0.05
            and r['predicate_gain_top'] == 'MATCH_prev']
    return per_head, prog


def mixture_tables(s_c):
    key = 'mixture_weights_E22a'
    stem = 'qk_e22_a'
    if E.SMOKE or key in E.loadj(JP) or not os.path.exists(E.ckpath(stem)):
        return
    m, _ = E.load_arm(stem, lambda: make_e22(s=s_c))
    b = m.pred_b.detach().float().cpu()           # (12, 6)
    c = m.pred_c.detach().float().cpu()
    prof = m.pred_prof.detach().float().cpu()     # (12, 6, T)
    # per-head pattern-mass shares on 4 held seqs + decomposition check
    idx = Q.HELD[:4, :Q.T]
    maskf = torch.tril(torch.ones(Q.T, Q.T, dtype=torch.bool,
                                  device=idx.device)).float()
    resid, full = {}, {}
    with torch.no_grad():
        m(idx, census_cb=lambda l, p: resid.__setitem__(l, p.clone()),
          census_full_cb=lambda l, p: full.__setitem__(l, p.clone()))
        Kprev, Ksame = match_kernels(idx, maskf)
        dev = idx.device
        decomp_max = 0.0
        shares = {}
        for l in range(DEPTH):
            terms = {
                'profile': (m.pred_prof[l].to(dev)[:, m.offmat] * maskf
                            ).unsqueeze(0).expand(idx.shape[0], -1, -1, -1),
                'match_prev': m.pred_b[l].to(dev).view(1, -1, 1, 1)
                * Kprev[:, None],
                'match_same': m.pred_c[l].to(dev).view(1, -1, 1, 1)
                * Ksame[:, None]}
            rec_full = resid[l] + sum(terms.values())
            decomp_max = max(decomp_max,
                             float((rec_full - full[l]).abs().max()))
            tot = full[l].float().pow(2).sum((0, 2, 3)).clamp_min(1e-12)
            shares[l] = {nm: [round(float(x), 5) for x in
                              (t.float().pow(2).sum((0, 2, 3)) / tot)]
                         for nm, t in terms.items()}
            shares[l]['residual'] = [round(float(x), 5) for x in
                                     (resid[l].float().pow(2).sum((0, 2, 3))
                                      / tot)]
    del m, resid, full
    torch.cuda.empty_cache()
    absb = b.abs().flatten()
    top3_mass = float(absb.sort(descending=True).values[:3].sum()
                      / absb.sum().clamp_min(1e-12))
    order = torch.argsort(absb, descending=True)[:10]
    topb = [{'layer': int(i) // Q.NH, 'head': int(i) % Q.NH,
             'b': round(float(b.flatten()[i]), 5)} for i in order]
    table = []
    for l in range(DEPTH):
        for h in range(Q.NH):
            pl = prof[l, h]
            top_off = torch.argsort(pl.abs(), descending=True)[:3]
            table.append({
                'layer': l, 'head': h,
                'b_match_prev': round(float(b[l, h]), 5),
                'c_match_same': round(float(c[l, h]), 5),
                'profile_l2': round(float(pl.norm()), 5),
                'profile_top_offsets': [[int(o), round(float(pl[o]), 5)]
                                        for o in top_off],
                'mass_share_profile': shares[l]['profile'][h],
                'mass_share_match_prev': shares[l]['match_prev'][h],
                'mass_share_match_same': shares[l]['match_same'][h],
                'mass_share_residual': shares[l]['residual'][h]})
    rec = {'checkpoint': f'{stem}.pt',
           'decomposition_check_max_abs_diff': decomp_max,
           'total_abs_b_mass': round(float(absb.sum()), 5),
           'top3_abs_b_mass_share': round(top3_mass, 4),
           'top10_by_abs_b': topb,
           'total_abs_c_mass': round(float(c.abs().sum()), 5),
           'mass_shares_rows': 'fresh held [33000:33004], fp32; shares = '
                               'sum-of-squares of each named term over the '
                               'full pattern, per head',
           'table': table}
    assert decomp_max < 1e-4, ('pattern decomposition off', decomp_max)
    E.merge(JP, key, rec)
    print(f"mixture tables: top-3 |b| mass share {top3_mass:.3f}, "
          f"decomp check {decomp_max:.1e}", flush=True)


# ---------------- census re-run on residual patterns ----------------
def census_residual(s_c):
    key = 'census_residual_E22a'
    stem = 'qk_e22_a'
    if E.SMOKE or key in E.loadj(JP) or not os.path.exists(E.ckpath(stem)):
        return
    E21 = get_e21()
    m, _ = E.load_arm(stem, lambda: make_e22(s=s_c))
    held = torch.from_numpy(
        np.load(E.HELD_PATH)[33000:33200].astype(np.int64)).to(E.DEV)

    def fwd(mm, idx, layer_cb=None, pat_hook=None):
        return mm(idx, census_cb=layer_cb, pat_hook=pat_hook)

    print('E22a residual-pattern census (E21 machinery reused) ...',
          flush=True)
    t0 = time.time()
    table, sc, tpl = E21.census_model('E22a_residual', m, fwd, held)
    c2r, zmat, mu_n, sd_n = E21.null_zscores(m, fwd, held)
    for r in table:
        l, h = r['layer'], r['head']
        r['cos2_eval'] = {E21.FEATN[f]: round(float(c2r[l, h, f]), 4)
                          for f in E21.NULL_FEATS}
        r['null_z'] = {E21.FEATN[f]: round(float(zmat[l, h, f]), 1)
                       for f in E21.NULL_FEATS}
    summ = E21.summarize(table)
    parent, prog = parent_census_matchprev()
    ours = {(r['layer'], r['head']): r['cos2_eval']['MATCH_prev']
            for r in table}
    total_parent = round(sum(parent.values()), 4)
    total_ours = round(sum(ours.values()), 4)
    prog_rows = [{'layer': l, 'head': h,
                  'parent_cos2': round(parent[(l, h)], 4),
                  'residual_cos2': round(ours[(l, h)], 4)}
                 for (l, h) in prog]
    mean_parent = round(sum(parent[p] for p in prog) / max(len(prog), 1), 4)
    mean_ours = round(sum(ours[p] for p in prog) / max(len(prog), 1), 4)
    verdict = bool(total_ours < total_parent and mean_ours < mean_parent)
    rec = {'checkpoint': f'{stem}.pt',
           'held_rows': 'fresh34k[33000:33200] (the E21 census rows)',
           'note': 'census scored on the RESIDUAL (bilinear-only) patterns; '
                   'the model forward includes the predicate terms (the '
                   'census callback fires pre-addition); machinery = '
                   'qk_e21_census_run.census_model + null_zscores verbatim',
           'summary': summ,
           'match_prev_total_cos2_residual': total_ours,
           'match_prev_total_cos2_parent_stored': total_parent,
           'parent_programmatic_match_prev_heads': prog_rows,
           'match_prev_mean_over_parent_prog_heads_residual': mean_ours,
           'match_prev_mean_over_parent_prog_heads_parent': mean_parent,
           'prediction_ii_drop_confirmed': verdict,
           'runtime_s': round(time.time() - t0, 1),
           'table': table}
    E.merge(JP, key, rec)
    print(f"census residual: MATCH_prev total cos2 {total_ours} vs parent "
          f"{total_parent}; prog-head mean {mean_ours} vs {mean_parent} -> "
          f"{'DROP CONFIRMED' if verdict else 'NO DROP'}", flush=True)
    del m, tpl
    torch.cuda.empty_cache()


# ---------------- standard wiring probes ----------------
def probe_e22a(s_c):
    stem = 'qk_e22_a'
    if E.SMOKE or not os.path.exists(E.ckpath(stem)):
        return
    dims = [s_c] * NG
    j = E.loadj(JP)
    if 'light_probe_E22a_var_dims' not in j:
        print('E22a light probe (variable slot dims) ...', flush=True)
        m, _ = E.load_arm(stem, lambda: make_e22(s=s_c))
        Ws = m.wte.weight.shape[1]
        assert Ws == 2 * DEPTH * s_c
        base, dce = E18U.gen_consumption(m, Ws)
        wp = E18U.wpairs(m, dims)
        G = E18U.gen_gram_table(m, dims)
        sup = E18U.score(G, wp)
        cau = [dce[li][si] for li, si in wp]
        eff = [k for k in range(len(wp)) if cau[k] > C.EFFECTUAL]
        agr = E17.agreement(sup, cau, eff)
        totals = {}
        for jj in range(DEPTH):
            totals[str(jj)] = round(sum(
                dce[li].get(si, 0.0) for li in dce
                for si in (1 + 2 * jj, 2 + 2 * jj) if si in dce[li]), 5)
        pairs_sorted = sorted([(li, si, dce[li][si]) for li in dce
                               for si in dce[li]], key=lambda p: -p[2])
        rec = {'checkpoint': f'{stem}.pt', 'stream_width': Ws,
               'slot_dims': dims, 'compute_width': Q.D,
               'base_ce_fp32_abl_oldheld': round(base, 5),
               'wiring_n_pairs': len(wp),
               'wiring_spearman_all': agr['spearman_all'],
               'wiring_n_effectual': len(eff),
               'wiring_spearman_effectual': agr['spearman_effectual'],
               'wiring_top10_precision': agr['top10_precision'],
               'per_block_total_consumption': totals,
               'dead_blocks_below_0.001':
                   [bk for bk in totals if totals[bk] < 0.001],
               'consumption_top20': [
                   {'consumer': ('readout' if li == DEPTH else f'block{li}'),
                    'source': R2.stream_name(si), 'dce': round(v, 5)}
                   for li, si, v in pairs_sorted[:20]],
               'consumption_matrix': {str(li): {str(si): round(v, 6)
                                                for si, v in dce[li].items()}
                                      for li in dce},
               'weight_support_matrix': {
                   str(li): {str(si): round(sup[i], 3)
                             for i, (l2, si) in enumerate(wp) if l2 == li}
                   for li in range(DEPTH + 1)},
               'note': 'generalized variable-slot-dim probe (qk_e18 gate 1 '
                       'validated); weight support covers the bilinear read '
                       'matrices only (the predicate terms read tokens, not '
                       'the stream); causal = mean-ablation on old cooc '
                       'held [:96], fp32'}
        E.merge(JP, 'light_probe_E22a_var_dims', rec)
        print(f"E22a wiring Spearman all {agr['spearman_all']} "
              f"effectual({len(eff)}) {agr['spearman_effectual']} "
              f"top10 {agr['top10_precision']}", flush=True)
        del m
        torch.cuda.empty_cache()
    j = E.loadj(JP)
    if 'composed_wiring_E22a' not in j:
        lp = j['light_probe_E22a_var_dims']
        print('E22a covariance-composed re-scoring ...', flush=True)
        m, _ = E.load_arm(stem, lambda: make_e22(s=s_c))
        wp = E18U.wpairs(m, dims)
        cau = E18U.stored_cau(lp, wp)
        tables, meta, _ = E18U.composed_tables(m, dims, cau, wp, E.DEV,
                                               remnant=False)
        chk = abs(tables['plain']['spearman_all']
                  - lp['wiring_spearman_all'])
        assert chk <= GATE_TOL, \
            f'E22a plain does not reproduce the light probe ({chk})'
        rec = {'checkpoint': f'{stem}.pt', 'slot_dims_uniform': s_c,
               'plain_reproduction_abs_diff': round(chk, 6),
               'tables': tables}
        rec.update(meta)
        E.merge(JP, 'composed_wiring_E22a', rec)
        print(f"E22a plain {tables['plain']['spearman_all']} -> cov "
              f"{tables['cov_composed']['spearman_all']}", flush=True)
        del m
        torch.cuda.empty_cache()


def pair_extra(stem, key, others):
    if E.SMOKE:
        return
    for ctl, label in others:
        f_arm, f_ctl = f'{stem}_heldloss.npy', f'{ctl}_heldloss.npy'
        if os.path.exists(f'{E.QK}/{f_ctl}') \
                and os.path.exists(f'{E.QK}/{f_arm}'):
            E.merge(JP, f'{key}_minus_{label}_fresh',
                    E.paired(f_arm, f_ctl, len(Q.HELD), label))


def save_seq_heldloss(stem):
    if E.SMOKE:
        return
    p = f'{E.QK}/{stem}_heldloss.npy'
    q = f'{E.QK}/{stem}_heldloss_seq.npy'
    if os.path.exists(p) and not os.path.exists(q):
        pt = np.load(p)
        np.save(q, pt.reshape(len(Q.HELD), Q.T).mean(1))
        print(f"saved {stem}_heldloss_seq.npy ({len(Q.HELD)} seq means)",
              flush=True)


def summarize():
    j = E.loadj(JP)
    summary = {'parents': {
        'E19a_gc1e-4': {'ce': 4.9742, 'cov_composed': 0.8259,
                        'plain': 0.7911},
        'E9a_recipe': {'ce': 5.0547, 'cov_composed': 0.8575,
                       'plain': 0.7711}}}
    if 'E22a' in j:
        row = {'final_held_ce_fresh_bf16':
                   j['E22a']['final_held_ce_fresh_bf16'],
               'diverged': j['E22a']['diverged']}
        if 'E22a_minus_e19a_fresh' in j:
            row['minus_e19a_paired'] = round(
                j['E22a_minus_e19a_fresh']['minus_e19a'], 4)
            row['minus_e19a_se_seq'] = round(
                j['E22a_minus_e19a_fresh']['minus_e19a_se_seq'], 5)
        if 'composed_wiring_E22a' in j:
            t = j['composed_wiring_E22a']['tables']
            row.update({
                'plain': t['plain']['spearman_all'],
                'cov_composed': t['cov_composed']['spearman_all'],
                'plain_top10': t['plain']['top10_precision'],
                'cov_top10': t['cov_composed']['top10_precision']})
        summary['E22a'] = row
        if 'mixture_weights_E22a' in j:
            m3 = j['mixture_weights_E22a']['top3_abs_b_mass_share']
            summary['prediction_i_verdict'] = (
                f"{'CONFIRMED' if m3 >= 0.60 else 'REFUTED'}: top-3 |b| "
                f"mass share {m3} vs the >= 0.60 registered threshold")
        if 'census_residual_E22a' in j:
            cr = j['census_residual_E22a']
            summary['prediction_ii_verdict'] = (
                ('CONFIRMED' if cr['prediction_ii_drop_confirmed']
                 else 'REFUTED')
                + f": residual MATCH_prev total cos2 "
                  f"{cr['match_prev_total_cos2_residual']} vs parent "
                  f"{cr['match_prev_total_cos2_parent_stored']}; "
                  f"prog-head mean "
                  f"{cr['match_prev_mean_over_parent_prog_heads_residual']} "
                  f"vs "
                  f"{cr['match_prev_mean_over_parent_prog_heads_parent']}")
        dce = row.get('minus_e19a_paired')
        if dce is not None:
            summary['prediction_iii_verdict'] = (
                f"{'CONFIRMED' if dce <= 0.03 else 'REFUTED'}: paired CE "
                f"vs E19a {dce} vs the <= +0.03 registered threshold")
    E.merge(JP, 'summary_E22', summary)
    print(json.dumps({'summary_E22': summary}, indent=2), flush=True)


if __name__ == '__main__':
    E.setup()
    control_e18_gates()

    s_c, tgt = E15R.solve_slot_c(4 * Q.D)
    if not E.SMOKE:
        assert s_c == 15, s_c                     # the E15c/E19a slot size

    # ---- positive controls (before training) ----
    control_pred_zero(s_c)
    control_kernels()

    # ---- registered predictions (before training) ----
    if 'E22_prediction' not in E.loadj(JP):
        pred = {
            'registered_before_training': True,
            'design': 'predicate-basis attention on the E19a frontier arm: '
                      'per-head pattern = signed positional profile (init '
                      '0) + b_h * MATCH_prev + c_h * MATCH_same (E21 '
                      'kernels, init 0) + the standard bilinear residual; '
                      'predicate params on AdamW; LOCALIZATION PROBE per '
                      'the E21 census branch resolution, not a recipe '
                      'candidate',
            'i_concentration': 'top-3 |b_h| over the 72 heads carry >= 60% '
                               'of total |b| mass',
            'ii_absorption': 'E21 census re-run on the RESIDUAL patterns: '
                             'MATCH_prev total eval-half cos^2 over all '
                             'heads AND the mean over the parent\'s '
                             'programmatic MATCH_prev heads both DROP vs '
                             'the parent\'s stored census',
            'iii_ce': 'paired fresh held CE vs the E19a parent <= +0.03',
            'parent_reference': {'E19a': {'ce': 4.9742, 'cov': 0.8259,
                                          'plain': 0.7911}}}
        try:
            parent, prog = parent_census_matchprev()
            pred['parent_stored_match_prev'] = {
                'total_cos2': round(sum(parent.values()), 4),
                'programmatic_heads': [list(p) for p in prog],
                'prog_mean_cos2': round(
                    sum(parent[p] for p in prog) / max(len(prog), 1), 4)}
        except Exception as ex:
            pred['parent_stored_match_prev'] = {'unavailable': str(ex)[:100]}
        E.merge(JP, 'E22_prediction', pred)

    if 'E22_config' not in E.loadj(JP):
        E.merge(JP, 'E22_config', {
            'group_coeff': GC22, 'muon_lr': E7R.muon_lr(),
            'adamw_lr': E.get_lr(), 'slot': s_c,
            'base': 'qk_e15_reinvest_run.make_e15c(s=15) verbatim + '
                    'group-lasso 1e-4 (the E19a arm)',
            'predicate_params': 'pred_prof (12 x 6 x T, signed per-offset '
                                'profile; the brief\'s a_h absorbed -- '
                                'a_h * P_h with both init 0 is the '
                                'dead-gradient trap), pred_b / pred_c '
                                '(12 x 6 signed mixture weights); all on '
                                'AdamW no-decay via muon_exclude',
            'kernels': 'MATCH_prev / MATCH_same exactly as '
                       'qk_e21_census_run.build_feats (control 2)',
            'traj_logging': 'qk_e22_a_traj.npz (qk_e_common standing '
                            'wiring-trajectory upgrade, every 200 steps)'})

    # ---- train ----
    E.train_arm('qk_e22_a', JP, 'E22a', lambda: make_e22(s=s_c), GC22,
                lr=E7R.muon_lr(), trainer=trainer,
                extra={'optimizer': 'muon', 'slot': s_c,
                       'parent': 'E19a (qk_e19_a, E15c arch at gc 1e-4)',
                       'design': 'predicate-basis attention: named signed '
                                 'positional profile + MATCH_prev + '
                                 'MATCH_same terms per head, init 0, on '
                                 'AdamW; bilinear residual unchanged'})
    E.oldheld_record('qk_e22_a', lambda: make_e22(s=s_c), JP, 'E22a_oldheld')
    E.paired_fresh('qk_e22_a', JP, 'E22a')
    pair_extra('qk_e22_a', 'E22a', (('qk_e19_a', 'e19a'),
                                    ('qk_e9_a', 'e9a')))
    save_seq_heldloss('qk_e22_a')

    # ---- measurement ----
    mixture_tables(s_c)
    census_residual(s_c)
    probe_e22a(s_c)

    summarize()
    print('e22 predbasis run done', flush=True)
