"""E23 IDENTIFIABLE WIRING -- lambda-per-edge reparameterization of the E19a
frontier arm (fresh single-epoch batch-16; results -> qk_e23.json).

IDEA (BRAINSTORM_STATE queue #4, Logan's read-scalar idea sharpened): every
read group's columns are unit-Frobenius-normalized and multiplied by an
EXPLICIT scalar lambda, so the wiring table becomes literal parameters, the
lasso acts on the lambdas, and a certified zero is lambda == 0. Pure
reparameterization + optimizer-geometry change; may improve wiring Spearman
by removing the gauge ambiguity of derived group norms.

PRODUCT-DEGENERACY FIX (reviewer-2 R6 + component notes): the pattern path
is a product of reads, so per-matrix lambdas are only identified through
products. Lambda granularity here:
  lam_pat (12 x 6 x 24): ONE lambda per (head, writer) SHARED across the
    four pattern matrices c_q / c_k / c_q2 / c_k2 -- each matrix's
    (head-row-block x writer-column-group) sub-block (44 x 15) is
    unit-Frobenius, scaled by lam_pat * rho_m;
  lam_v  (12 x 6 x 24): separate lambda per (head, writer) for c_v;
  lam_mlp (12 x 24): ONE lambda per (mlp, writer) SHARED across Left/Right
    (bilinear product -- same degeneracy), groups = full column groups
    (1056 x 15).
INIT ANISOTROPY CONSTANTS rho (fixed buffers, NOT trainable): a shared
lambda cannot bit-exactly reproduce the parent init when its constituent
sub-blocks have different norms, so lam is initialized to the GEOMETRIC
MEAN of the parent-init sub-block norms and rho_m = n_m / lam carries the
per-matrix remainder (product of the rho's across a shared group == 1 by
construction: the trainable lambda is exactly the product-path scale, the
rho's are a frozen gauge choice). For c_v (unshared) lam_v == the parent
init group norm exactly and rho == 1.

PENALTY: in-loss L1 on |lambda| (all 3744 lambdas) with the coefficient
chosen so the total penalty AT INIT equals the parent's group-lasso penalty
at init: gc_lambda = 1e-4 * P_parent_init / sum|lambda_init| (computed from
live models, documented, asserted -- control 2). Lambdas train on AdamW
no-decay (muon_exclude); the raw unit matrices stay on Muon.

UNIT-NORM MAINTENANCE: re-projection (renormalize every constrained
sub-block to unit Frobenius) after EVERY optimizer step via the new
qk_e_common.train_muon post_step hook -- Muon does not preserve the
constraint (silent-drift bug warned in BRAINSTORM_STATE).

CONTROLS:
  1. INIT REPRODUCTION: lambdas initialized from the parent init group
     norms with groups normalized reproduce the E19a init forward
     (max |logit diff| < 1e-3, tf32 off, measured value recorded -- exact
     algebra, float rounding from the decompose/recompose); applying the
     projection at init is a no-op to float tolerance (forward diff < 1e-5,
     max |group norm - 1| < 1e-6 -- raw norms are unit only to fp32
     precision); a 3-step training run with projection and one with projection
     disabled start from the identical step-0 CE (same seed/data), and the
     projection-off run's recorded group-norm drift demonstrates the bug
     the projection fixes.
  2. PENALTY EQUALITY AT INIT: gc_lambda * sum|lambda_init| ==
     1e-4 * parent group-lasso penalty at init (rel < 1e-6, by
     construction; asserted).
  3. DRIFT: after 50 training steps with projection, every constrained
     group norm is 1.0 to 1e-5 (asserted).

REGISTERED PREDICTIONS (merged BEFORE training):
  (i)  CE within +-0.02 of the E19a parent (paired) -- pure
       reparameterization + optimizer-geometry change only;
  (ii) the LITERAL lambda table's causal Spearman (all 156 edges, standard
       conventions) >= the parent's derived-norm Spearman 0.7911 (gauge
       ambiguity removed).

MEASURES: paired CE vs E19a / E9a / E0a+E0b; the literal lambda wiring
table vs causal consumption (qk_e18 generalized probe machinery on the
FOLDED effective model -- folding lambda into the read matrices reproduces
the E23 forward exactly and makes every stored probe reusable); the derived
table from the folded matrices (sanity: equals lambda x rho by unit-norm
algebra) and from the RAW normalized matrices (sanity: ~constant sqrt(32)
on block consumers); covariance-composed as usual. Wiring-trajectory
logging (effective norms via the model's traj_group_norms override +
lambda snapshots via traj_extra) -> qk_e23_a_traj.npz. Idempotent."""
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

# The E17 import (via E18U) sets E.DEV='cpu' -- restore the runner convention.
E.DEV = 'cpu' if E.SMOKE else 'cuda'

JP = E.jpath('qk_e23.json')
GC_PARENT_LASSO = 1e-4                    # the E19a group-lasso the L1 matches
GATE_TOL = 1e-3
NG = E.NGROUP
PATQ = ('c_q', 'c_k', 'c_q2', 'c_k2')


class E23Route(E15R.E15cRoute):
    """E15c with every read group unit-Frobenius-normalized and scaled by
    explicit lambdas (see module docstring for granularity and rho)."""

    def __init__(self, variant, depth, s, Dc, NH, HD, hidden):
        super().__init__(variant, depth, s, Dc, NH, HD, hidden)
        self.muon_exclude = ('lam_',)
        with torch.no_grad():
            n_pat = torch.zeros(depth, 4, NH, NG, dtype=torch.float64)
            n_v = torch.zeros(depth, NH, NG, dtype=torch.float64)
            n_lr = torch.zeros(depth, 2, NG, dtype=torch.float64)
            for l, blk in enumerate(self.h):
                for mi, nm in enumerate(PATQ):
                    W = getattr(blk, nm).weight
                    Vw = W.view(NH, HD, NG, s)
                    n = Vw.double().pow(2).sum((1, 3)).sqrt()      # (NH, NG)
                    n_pat[l, mi] = n
                    Vw.div_(n.float().clamp_min(1e-12)[:, None, :, None])
                W = blk.c_v.weight
                Vw = W.view(NH, HD, NG, s)
                n = Vw.double().pow(2).sum((1, 3)).sqrt()
                n_v[l] = n
                Vw.div_(n.float().clamp_min(1e-12)[:, None, :, None])
                for ri, nm in enumerate(('Left', 'Right')):
                    W = getattr(blk, nm).weight
                    Vw = W.view(W.shape[0], NG, s)
                    n = Vw.double().pow(2).sum((0, 2)).sqrt()      # (NG,)
                    n_lr[l, ri] = n
                    Vw.div_(n.float().clamp_min(1e-12)[None, :, None])
            lam_pat = n_pat.clamp_min(1e-12).log().mean(1).exp()   # geomean
            rho_pat = n_pat / lam_pat[:, None]
            lam_mlp = n_lr.clamp_min(1e-12).log().mean(1).exp()
            rho_mlp = n_lr / lam_mlp[:, None]
        self.lam_pat = nn.Parameter(lam_pat.float())               # (D,NH,NG)
        self.lam_v = nn.Parameter(n_v.float())                     # (D,NH,NG)
        self.lam_mlp = nn.Parameter(lam_mlp.float())               # (D,NG)
        self.register_buffer('rho_pat', rho_pat.float())           # (D,4,NH,NG)
        self.register_buffer('rho_mlp', rho_mlp.float())           # (D,2,NG)

    # ---- effective read matrices ----
    def _expand_hw(self, mat2):
        """(NH, NG) -> (Dc, Ws) block-constant scale."""
        return mat2.repeat_interleave(self.HD, 0).repeat_interleave(self.s, 1)

    def eff_read(self, l, nm):
        blk = self.h[l]
        W = getattr(blk, nm).weight
        if nm in PATQ:
            sc = self._expand_hw(self.lam_pat[l] * self.rho_pat[l,
                                                               PATQ.index(nm)])
        elif nm == 'c_v':
            sc = self._expand_hw(self.lam_v[l])
        else:                                     # Left / Right
            ri = 0 if nm == 'Left' else 1
            sc = (self.lam_mlp[l] * self.rho_mlp[l, ri]
                  ).repeat_interleave(self.s)[None, :]
        return W * sc

    def custom_group_penalty(self):
        """In-loss penalty = L1 on the lambdas (coefficient gc_lambda passed
        as the trainer's group coefficient)."""
        return (self.lam_pat.abs().sum() + self.lam_v.abs().sum()
                + self.lam_mlp.abs().sum())

    @torch.no_grad()
    def project_unit_groups(self):
        """Renormalize every constrained sub-block to unit Frobenius (the
        post-every-optimizer-step re-projection; Muon does not preserve
        the constraint)."""
        for blk in self.h:
            for nm in PATQ + ('c_v',):
                Vw = getattr(blk, nm).weight.view(self.NH, self.HD, NG,
                                                  self.s)
                n = Vw.float().pow(2).sum((1, 3), keepdim=True).sqrt()
                Vw.div_(n.clamp_min(1e-12).to(Vw.dtype))
            for nm in ('Left', 'Right'):
                W = getattr(blk, nm).weight
                Vw = W.view(W.shape[0], NG, self.s)
                n = Vw.float().pow(2).sum((0, 2), keepdim=True).sqrt()
                Vw.div_(n.clamp_min(1e-12).to(Vw.dtype))

    def max_group_norm_drift(self):
        """max |constrained group norm - 1| over all sub-blocks."""
        d = 0.0
        for blk in self.h:
            for nm in PATQ + ('c_v',):
                Vw = getattr(blk, nm).weight.detach().view(self.NH, self.HD,
                                                           NG, self.s)
                n = Vw.float().pow(2).sum((1, 3)).sqrt()
                d = max(d, float((n - 1).abs().max()))
            for nm in ('Left', 'Right'):
                W = getattr(blk, nm).weight.detach()
                n = W.view(W.shape[0], NG, self.s).float().pow(2) \
                     .sum((0, 2)).sqrt()
                d = max(d, float((n - 1).abs().max()))
        return d

    # ---- wiring-trajectory hooks (qk_e_common standing upgrade) ----
    def traj_group_norms(self):
        rows = []
        for l in range(self.depth):
            for nm in E.READ_NAMES:
                W = self.eff_read(l, nm).detach().float()
                S = W.shape[1] // NG
                rows.append(W.pow(2).view(W.shape[0], NG, S)
                            .sum((0, 2)).sqrt())
        return torch.stack(rows)

    def traj_extra(self):
        return {'lam_pat': self.lam_pat.detach().float().cpu().numpy(),
                'lam_v': self.lam_v.detach().float().cpu().numpy(),
                'lam_mlp': self.lam_mlp.detach().float().cpu().numpy()}

    def forward(self, idx, collect=None, sub_entry=None):
        B, Tq = idx.shape
        Ws, s = self.Ws, self.s
        cos = self.cos[None, :Tq, None, :]
        sin = self.sin[None, :Tq, None, :]
        mask = self.mask[:Tq, :Tq]
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

            def qk(Wm):
                z = F.linear(hn, Wm).view(B, Tq, self.NH, self.HD)
                return Q.apply_rot(F.rms_norm(z, (self.HD,)), cos, sin)

            q, k = qk(self.eff_read(l, 'c_q')), qk(self.eff_read(l, 'c_k'))
            q2 = qk(self.eff_read(l, 'c_q2'))
            k2 = qk(self.eff_read(l, 'c_k2'))
            v = F.linear(hn, self.eff_read(l, 'c_v')).view(B, Tq, self.NH,
                                                           self.HD)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / self.HD
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / self.HD
            pat = (s1 * s2).masked_fill(~mask, 0.0)
            y = torch.einsum('bhqk,bkhd->bqhd', pat, v)
            y = y.reshape(B, Tq, self.Dc)
            aw = torch.zeros(B, Tq, Ws, device=y.device, dtype=y.dtype)
            aw[..., s * 2 * l:s * (2 * l + 1)] = blk.c_proj(y)
            x = x + aw
            xn = self.slot_norm(x)
            mw_s = blk.Down(F.linear(xn, self.eff_read(l, 'Left'))
                            * F.linear(xn, self.eff_read(l, 'Right'))) \
                + blk.Down_bias
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


def make_e23(s=None):
    hidden = 4 * Q.D
    if s is None:
        s, _ = E15R.solve_slot_c(hidden)
    torch.manual_seed(Q.SEED)
    return E23Route('E23a', DEPTH, s, Q.D, Q.NH, Q.HD, hidden).to(E.DEV)


def gc_lambda_init(s_c):
    """L1 coefficient s.t. the total penalty at init equals the parent's
    group-lasso penalty at init (both from live models, deterministic)."""
    with torch.no_grad():
        mp = E15R.make_e15c(s=s_c)
        p_parent = float(mp.custom_group_penalty())
        del mp
        m23 = make_e23(s=s_c)
        s_lam = float(m23.custom_group_penalty())
        del m23
    torch.cuda.empty_cache()
    return GC_PARENT_LASSO * p_parent / s_lam, p_parent, s_lam


def trainer_factory(gcl):
    def trainer(lr, gc, steps, **kw):
        kw.setdefault('post_step', lambda st, m: m.project_unit_groups())
        return E.train_muon(lr, gc, steps, lr_adamw=E.get_lr(), **kw)
    return trainer


# ---------------- controls ----------------
def control_e18_gates():
    if E.SMOKE:
        return
    e18 = E.loadj(E.jpath('qk_e18.json'))
    g1 = e18.get('gate1_uniform11_weight_support', {}).get('pass')
    g2 = e18.get('gate2_cov_composed_E9a', {}).get('pass')
    assert g1 and g2, ('qk_e18.json gates not passed', g1, g2)
    print("precondition: qk_e18.json gates 1+2 passed", flush=True)


def control_init(s_c, gcl):
    key = 'control1_init_reproduction'
    done = E.loadj(JP).get(key)
    if done and done.get('pass'):
        print(f"{key}: already passed -- skip", flush=True)
        return
    m19 = E15R.make_e15c(s=s_c).eval().float()
    m23 = make_e23(s=s_c).eval().float()
    idx = E.OLD_HELD[:2, :Q.T]
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.no_grad():
            out19 = m19(idx)
            d_init = float((m23(idx) - out19).abs().max())
            drift0 = m23.max_group_norm_drift()
            out_pre = m23(idx)
            m23.project_unit_groups()
            d_proj = float((m23(idx) - out_pre).abs().max())
            drift1 = m23.max_group_norm_drift()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    del m19, m23, out19, out_pre
    torch.cuda.empty_cache()
    # 3-step runs: with projection vs projection disabled (same seed/data);
    # step-0 CE must be identical; the no-projection drift documents the bug.
    log_p = E.train_muon(E7R.muon_lr(), gcl, 3, log_every=1, save=False,
                         factory=lambda: make_e23(s=s_c),
                         lr_adamw=E.get_lr(), return_model=True,
                         post_step=lambda st, m: m.project_unit_groups())
    log_p, mp_ = log_p
    drift_with = mp_.max_group_norm_drift()
    del mp_
    log_n = E.train_muon(E7R.muon_lr(), gcl, 3, log_every=1, save=False,
                         factory=lambda: make_e23(s=s_c),
                         lr_adamw=E.get_lr(), return_model=True)
    log_n, mn_ = log_n
    drift_without = mn_.max_group_norm_drift()
    del mn_
    torch.cuda.empty_cache()
    ce_p = [x[1] for x in log_p['train_loss']]
    ce_n = [x[1] for x in log_n['train_loss']]
    rec = {'init_forward_max_logit_diff_vs_e19a': d_init,
           'init_max_group_norm_drift': drift0,
           'projection_at_init_forward_max_diff': d_proj,
           'post_projection_max_group_norm_drift': drift1,
           'train3_ce_with_projection': ce_p,
           'train3_ce_without_projection': ce_n,
           'step0_ce_identical': bool(ce_p[0] == ce_n[0]),
           'group_norm_drift_after3_with_projection': drift_with,
           'group_norm_drift_after3_without_projection': drift_without,
           'note': 'init lambdas = parent init group norms (shared lambdas '
                   '= geometric mean, fixed unit-product rho anisotropy '
                   'buffers -- forced by sharing + unit-norm + exact-init '
                   'simultaneously); reproduction exact up to fp32 '
                   'decompose/recompose rounding; the projection-off drift '
                   'documents the silent-drift bug the projection fixes; '
                   'projection-at-init tolerance is float (raw group norms '
                   'are unit only to fp32 precision, so re-projection moves '
                   'weights by ~1e-7)',
           'pass': bool(d_init < 1e-3 and drift0 < 1e-6 and d_proj < 1e-5
                        and drift1 < 1e-6 and ce_p[0] == ce_n[0]
                        and drift_with < 1e-5)}
    E.merge(JP, key, rec)
    print(f"{key}: init fwd {d_init:.2e}, proj no-op {d_proj:.1e} (drift "
          f"{drift0:.1e}->{drift1:.1e}), 3-step drift with/without proj "
          f"{drift_with:.2e}/{drift_without:.2e} -> "
          f"{'PASS' if rec['pass'] else 'FAIL'}", flush=True)
    assert rec['pass'], f'{key} FAILED'


def control_penalty(s_c, gcl, p_parent, s_lam):
    key = 'control2_penalty_init_equality'
    done = E.loadj(JP).get(key)
    if done and done.get('pass'):
        print(f"{key}: already passed -- skip", flush=True)
        return
    lhs = gcl * s_lam
    rhs = GC_PARENT_LASSO * p_parent
    rel = abs(lhs - rhs) / rhs
    rec = {'parent_group_lasso_penalty_at_init': p_parent,
           'parent_coeff': GC_PARENT_LASSO,
           'sum_abs_lambda_at_init': s_lam,
           'gc_lambda': gcl,
           'penalty_at_init_lambda_arm': lhs,
           'penalty_at_init_parent': rhs,
           'rel_diff': rel,
           'pass': bool(rel < 1e-6)}
    E.merge(JP, key, rec)
    print(f"{key}: gc_lambda {gcl:.6e} (parent penalty {rhs:.6e} vs lambda "
          f"{lhs:.6e}, rel {rel:.1e}) -> "
          f"{'PASS' if rec['pass'] else 'FAIL'}", flush=True)
    assert rec['pass'], f'{key} FAILED'


def control_drift50(s_c, gcl):
    key = 'control3_drift_50steps'
    done = E.loadj(JP).get(key)
    if done and done.get('pass'):
        print(f"{key}: already passed -- skip", flush=True)
        return
    steps = 5 if E.SMOKE else 50
    log, m = E.train_muon(E7R.muon_lr(), gcl, steps, save=False,
                          factory=lambda: make_e23(s=s_c),
                          lr_adamw=E.get_lr(), return_model=True,
                          post_step=lambda st, mm: mm.project_unit_groups())
    drift = m.max_group_norm_drift()
    del m
    torch.cuda.empty_cache()
    rec = {'steps': steps, 'max_group_norm_drift': drift,
           'pass': bool(drift <= 1e-5)}
    E.merge(JP, key, rec)
    print(f"{key}: drift after {steps} steps {drift:.2e} -> "
          f"{'PASS' if rec['pass'] else 'FAIL'}", flush=True)
    assert rec['pass'], f'{key} FAILED'


# ---------------- folding + probes ----------------
def fold_effective(m):
    """Materialize lambda x raw into a plain E15cRoute (reproduces the E23
    forward exactly; makes every gated probe reusable)."""
    mf = E15R.make_e15c(s=m.s).eval().float()
    with torch.no_grad():
        mf.wte.weight.copy_(m.wte.weight)
        for l in range(m.depth):
            for nm in E.READ_NAMES:
                getattr(mf.h[l], nm).weight.copy_(m.eff_read(l, nm))
            mf.h[l].c_proj.weight.copy_(m.h[l].c_proj.weight)
            mf.h[l].Down.weight.copy_(m.h[l].Down.weight)
            mf.h[l].Down_bias.copy_(m.h[l].Down_bias)
    return mf


def lambda_edge_scores(m, wp):
    """The LITERAL lambda wiring table aggregated to the probe's (consumer,
    writer) edges: sqrt(4 sum_h lam_pat^2 + sum_h lam_v^2 + 2 lam_mlp^2).
    Readout rows (li == DEPTH) have no lambdas (tied embedding) -- they use
    the derived wte slot norms, exactly as in the parent table (documented).
    Also returns the effective (lambda x rho) version, which equals the
    derived folded norms by unit-norm algebra."""
    lam_pat = m.lam_pat.detach().float()
    lam_v = m.lam_v.detach().float()
    lam_mlp = m.lam_mlp.detach().float()
    rho_pat = m.rho_pat.detach().float()
    rho_mlp = m.rho_mlp.detach().float()
    pure, eff = [], []
    for li, si in wp:
        w = si - 1
        if li == DEPTH:
            sl = m.wte.weight[:, m.s * w:m.s * (w + 1)].detach().float()
            v = float(sl.pow(2).sum()) ** 0.5
            pure.append(v)
            eff.append(v)
            continue
        p2 = float((lam_pat[li, :, w] ** 2).sum())
        v2 = float((lam_v[li, :, w] ** 2).sum())
        m2 = float(lam_mlp[li, w] ** 2)
        pure.append(math.sqrt(4 * p2 + v2 + 2 * m2))
        pe = float(((lam_pat[li, :, w][None] * rho_pat[li, :, :, w]) ** 2)
                   .sum())
        me = float(((lam_mlp[li, w] * rho_mlp[li, :, w]) ** 2).sum())
        eff.append(math.sqrt(pe + v2 + me))
    return pure, eff


def probe_e23a(s_c):
    stem = 'qk_e23_a'
    if E.SMOKE or not os.path.exists(E.ckpath(stem)):
        return
    dims = [s_c] * NG
    j = E.loadj(JP)
    need_lp = 'light_probe_E23a_var_dims' not in j
    need_lam = 'lambda_wiring_E23a' not in j
    need_cov = 'composed_wiring_E23a' not in j
    if not (need_lp or need_lam or need_cov):
        return
    m, _ = E.load_arm(stem, lambda: make_e23(s=s_c))
    mf = fold_effective(m)
    idx = E.OLD_HELD[:2, :Q.T]
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.no_grad():
            d_fold = float((mf(idx) - m(idx)).abs().max())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    print(f"fold identity: max |logit diff| {d_fold:.2e}", flush=True)
    assert d_fold < 1e-5, d_fold
    Ws = mf.wte.weight.shape[1]
    if need_lp:
        print('E23a light probe (folded model, variable slot dims) ...',
              flush=True)
        base, dce = E18U.gen_consumption(mf, Ws)
        wp = E18U.wpairs(mf, dims)
        G = E18U.gen_gram_table(mf, dims)
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
               'fold_identity_max_logit_diff': d_fold,
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
               'note': 'probes run on the FOLDED effective model (identity '
                       'asserted above); derived weight support here = '
                       'lambda x rho by unit-norm algebra'}
        E.merge(JP, 'light_probe_E23a_var_dims', rec)
        print(f"E23a folded wiring Spearman all {agr['spearman_all']} "
              f"effectual({len(eff)}) {agr['spearman_effectual']} "
              f"top10 {agr['top10_precision']}", flush=True)
    j = E.loadj(JP)
    if 'lambda_wiring_E23a' not in j:
        lp = j['light_probe_E23a_var_dims']
        wp = E18U.wpairs(mf, dims)
        cau = E18U.stored_cau(lp, wp)
        eff_idx = [k for k in range(len(wp)) if cau[k] > C.EFFECTUAL]
        pure, effv = lambda_edge_scores(m, wp)
        G_raw = E18U.gen_gram_table(m, dims)      # RAW unit-norm matrices
        raw_scores = E18U.score(G_raw, wp)
        blk_raw = [raw_scores[k] for k in range(len(wp))
                   if wp[k][0] != DEPTH]
        agr_pure = E17.agreement(pure, cau, eff_idx)
        agr_eff = E17.agreement(effv, cau, eff_idx)
        derived = [lp['weight_support_matrix'][str(li)][str(si)]
                   for li, si in wp]
        agr_derived = E17.agreement(derived, cau, eff_idx)
        rec = {'checkpoint': f'{stem}.pt', 'n_pairs': len(wp),
               'lambda_table_literal': {
                   str(li): {str(si): round(pure[i], 4)
                             for i, (l2, si) in enumerate(wp) if l2 == li}
                   for li in range(DEPTH + 1)},
               'aggregation': 'edge (consumer, writer) = sqrt(4 sum_h '
                              'lam_pat^2 + sum_h lam_v^2 + 2 lam_mlp^2); '
                              'readout rows: derived wte slot norms (no '
                              'lambdas on the tied embedding -- documented)',
               'agreement_lambda_literal': agr_pure,
               'agreement_lambda_times_rho_effective': agr_eff,
               'agreement_derived_folded': agr_derived,
               'raw_normalized_derived_sanity': {
                   'block_rows_min': round(min(blk_raw), 4),
                   'block_rows_max': round(max(blk_raw), 4),
                   'expected_constant': round(math.sqrt(
                       4 * Q.NH + Q.NH + 2), 4),
                   'note': 'derived table from the RAW unit-norm matrices '
                           'is ~constant sqrt(5*NH+2) on block consumers '
                           '-- all wiring information lives in the '
                           'lambdas'},
               'parent_derived_spearman_all_stored': 0.7911}
        E.merge(JP, 'lambda_wiring_E23a', rec)
        print(f"E23a lambda table Spearman all {agr_pure['spearman_all']} "
              f"(effective {agr_eff['spearman_all']}, folded-derived "
              f"{agr_derived['spearman_all']}) vs parent derived 0.7911",
              flush=True)
    j = E.loadj(JP)
    if 'composed_wiring_E23a' not in j:
        lp = j['light_probe_E23a_var_dims']
        print('E23a covariance-composed re-scoring (folded model) ...',
              flush=True)
        wp = E18U.wpairs(mf, dims)
        cau = E18U.stored_cau(lp, wp)
        tables, meta, _ = E18U.composed_tables(mf, dims, cau, wp, E.DEV,
                                               remnant=False)
        chk = abs(tables['plain']['spearman_all']
                  - lp['wiring_spearman_all'])
        assert chk <= GATE_TOL, \
            f'E23a plain does not reproduce the light probe ({chk})'
        rec = {'checkpoint': f'{stem}.pt', 'slot_dims_uniform': s_c,
               'plain_reproduction_abs_diff': round(chk, 6),
               'tables': tables}
        rec.update(meta)
        E.merge(JP, 'composed_wiring_E23a', rec)
        print(f"E23a plain {tables['plain']['spearman_all']} -> cov "
              f"{tables['cov_composed']['spearman_all']}", flush=True)
    del m, mf
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
    if 'E23a' in j:
        row = {'final_held_ce_fresh_bf16':
                   j['E23a']['final_held_ce_fresh_bf16'],
               'diverged': j['E23a']['diverged']}
        if 'E23a_minus_e19a_fresh' in j:
            row['minus_e19a_paired'] = round(
                j['E23a_minus_e19a_fresh']['minus_e19a'], 4)
            row['minus_e19a_se_seq'] = round(
                j['E23a_minus_e19a_fresh']['minus_e19a_se_seq'], 5)
        if 'composed_wiring_E23a' in j:
            t = j['composed_wiring_E23a']['tables']
            row.update({
                'plain': t['plain']['spearman_all'],
                'cov_composed': t['cov_composed']['spearman_all'],
                'plain_top10': t['plain']['top10_precision'],
                'cov_top10': t['cov_composed']['top10_precision']})
        if 'lambda_wiring_E23a' in j:
            lw = j['lambda_wiring_E23a']
            row['lambda_literal_spearman_all'] = \
                lw['agreement_lambda_literal']['spearman_all']
            row['lambda_effective_spearman_all'] = \
                lw['agreement_lambda_times_rho_effective']['spearman_all']
        summary['E23a'] = row
        dce = row.get('minus_e19a_paired')
        if dce is not None:
            summary['prediction_i_verdict'] = (
                f"{'CONFIRMED' if abs(dce) <= 0.02 else 'REFUTED'}: paired "
                f"CE vs E19a {dce} vs the +-0.02 registered band")
        if 'lambda_literal_spearman_all' in row:
            sp = row['lambda_literal_spearman_all']
            summary['prediction_ii_verdict'] = (
                f"{'CONFIRMED' if sp >= 0.7911 else 'REFUTED'}: literal "
                f"lambda-table Spearman {sp} vs the parent's derived-norm "
                f"0.7911")
    E.merge(JP, 'summary_E23', summary)
    print(json.dumps({'summary_E23': summary}, indent=2), flush=True)


if __name__ == '__main__':
    E.setup()
    control_e18_gates()

    s_c, tgt = E15R.solve_slot_c(4 * Q.D)
    if not E.SMOKE:
        assert s_c == 15, s_c                     # the E15c/E19a slot size

    gcl, p_parent, s_lam = gc_lambda_init(s_c)

    # ---- positive controls (before training) ----
    control_penalty(s_c, gcl, p_parent, s_lam)
    control_init(s_c, gcl)
    control_drift50(s_c, gcl)

    # ---- registered predictions (before training) ----
    if 'E23_prediction' not in E.loadj(JP):
        E.merge(JP, 'E23_prediction', {
            'registered_before_training': True,
            'design': 'identifiable wiring: every read group unit-'
                      'Frobenius, explicit lambdas (one per (head, writer) '
                      'shared across the four pattern matrices; one per '
                      '(mlp, writer) shared across Left/Right; separate '
                      'per (head, writer) for c_v -- reviewer-2 R6 product-'
                      'degeneracy fix); L1 on |lambda| at penalty-matched '
                      'coefficient; re-projection after every step',
            'i_ce': 'paired fresh held CE within +-0.02 of the E19a parent '
                    '(pure reparameterization + optimizer-geometry change)',
            'ii_spearman': 'the literal lambda table\'s causal Spearman '
                           '(all 156 edges, standard probe conventions, '
                           'readout rows derived from wte) >= the parent\'s '
                           'derived-norm Spearman 0.7911',
            'parent_reference': {'E19a': {'ce': 4.9742, 'plain': 0.7911,
                                          'cov': 0.8259}}})

    if 'E23_config' not in E.loadj(JP):
        E.merge(JP, 'E23_config', {
            'gc_lambda': gcl,
            'parent_group_lasso_penalty_at_init': p_parent,
            'sum_abs_lambda_at_init': s_lam,
            'coefficient_rule': 'gc_lambda = 1e-4 * P_parent_init / '
                                'sum|lambda_init| (penalty-at-init '
                                'equality, control 2)',
            'n_lambdas': {'lam_pat': DEPTH * Q.NH * NG,
                          'lam_v': DEPTH * Q.NH * NG,
                          'lam_mlp': DEPTH * NG},
            'muon_lr': E7R.muon_lr(), 'adamw_lr': E.get_lr(), 'slot': s_c,
            'init': 'lambdas = parent init group norms; shared lambdas = '
                    'geometric mean of constituent sub-block norms with '
                    'fixed unit-product rho anisotropy buffers (forced by '
                    'sharing + unit-norm + exact init reproduction)',
            'projection': 'renormalize every constrained sub-block to unit '
                          'Frobenius after EVERY optimizer step '
                          '(train_muon post_step hook)',
            'traj_logging': 'qk_e23_a_traj.npz: effective read-group norms '
                            '(traj_group_norms override) + lambda '
                            'snapshots (traj_extra)'})

    # ---- train ----
    E.train_arm('qk_e23_a', JP, 'E23a', lambda: make_e23(s=s_c), gcl,
                lr=E7R.muon_lr(), trainer=trainer_factory(gcl),
                extra={'optimizer': 'muon', 'slot': s_c,
                       'parent': 'E19a (qk_e19_a, E15c arch at gc 1e-4)',
                       'penalty': 'L1 on |lambda|, coefficient gc_lambda '
                                  '(group_coeff field above); '
                                  'final_penalty = sum|lambda|',
                       'design': 'identifiable wiring reparameterization '
                                 '(unit-norm read groups x explicit '
                                 'lambdas, product-degeneracy-fixed '
                                 'sharing, per-step re-projection)'})
    E.oldheld_record('qk_e23_a', lambda: make_e23(s=s_c), JP, 'E23a_oldheld')
    E.paired_fresh('qk_e23_a', JP, 'E23a')
    pair_extra('qk_e23_a', 'E23a', (('qk_e19_a', 'e19a'),
                                    ('qk_e9_a', 'e9a')))
    save_seq_heldloss('qk_e23_a')

    # ---- post-training drift audit (the trained checkpoint itself) ----
    if not E.SMOKE and os.path.exists(E.ckpath('qk_e23_a')) \
            and 'final_drift_audit' not in E.loadj(JP):
        m, _ = E.load_arm('qk_e23_a', lambda: make_e23(s=s_c))
        E.merge(JP, 'final_drift_audit',
                {'max_group_norm_drift': m.max_group_norm_drift()})
        del m
        torch.cuda.empty_cache()

    # ---- measurement ----
    probe_e23a(s_c)

    summarize()
    print('e23 idwiring run done', flush=True)
