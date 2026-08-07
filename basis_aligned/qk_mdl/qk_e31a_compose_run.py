"""E31a COMPOSITION ARM -- predicate-basis attention + variable-k codebook
slots on one model (fresh single-epoch batch-16, recipe conventions; results
-> qk_e31a.json).

MOTIVATION: the program's two best interpretability assets were built on the
same E19a parent but never combined:
  * PREDICATE-BASIS ATTENTION (E22a, qk_e22.json): per-head pattern = signed
    positional profile + b_h * MATCH_prev + c_h * MATCH_same + the bilinear
    residual. Best CE of the leading arms (4.8957 single seed; 4.9000 +-
    0.0068 over three seeds, qk_e29.json) and the best leave-one-in-context
    readability.
  * VARIABLE-K CODEBOOK SLOTS (E20b, qk_e20b.json): every written slot's
    consumed content is a k-code matching-pursuit quantization against an
    n=256 unit-norm codebook per slot, k=4 on attention slots, k=2 on MLP
    slots -- the content side becomes an enumerable dictionary (E20's code
    dictionaries passed the token-class legibility gate).
The composition asks whether the two costs are sub-additive (they price
different things: selection structure vs content granularity) and whether
each mechanism survives the other.

ARCHITECTURE (verbatim reuse, no new mechanism): E31Route inherits
qk_e20b_vark_run.E20bRoute (codebooks, EMA, dead-reinit, per-slot pursuit
depth, commitment aux loss, the qz_full interface) and
qk_e22_predbasis_run.E22Route (pred_prof / pred_b / pred_c, pred_terms, the
E21 match kernels, muon_exclude routing to AdamW). Cooperative __init__ makes
the shared parameters bit-identical to the E19a parent (predicate params are
exact zeros, codebooks come from a separate fixed-seed generator). The single
new line of logic is the merged forward: quantize the consumed slot content
(E20b) and add the named terms to the pattern (E22).

CONTROLS BEFORE TRAINING (hard gates):
  1. BOTH-OFF BYPASS: predicate terms disabled AND quantization disabled
     reproduces qk_e15_reinvest_run.make_e15c bit-exactly -- shared-parameter
     identity, forward identity, and a 3-step run of this runner's trainer
     matching the parent trainer's 3-step trajectory exactly (same seed, same
     epoch_order(0) data). Plus the informational at-init identity with the
     predicate terms ENABLED (their parameters are exact zeros).
  2. K-WIRING (E20b gate 4): a collected forward yields 4 code columns on
     every attention slot and 2 on every MLP slot.
  3. CAPACITY (E20b gate 2): n >= #distinct vectors at k=15 pursuit -> ~0
     reconstruction error through the exact mp_quantize.
  4. PLANTED TOY (E20b gate 3, reused verbatim): the EMA + dead-reinit +
     commitment machinery recovers 10 planted centers at k=4.
  5. COV PIPELINE (E20b gate 5): with quantization off, the quantized-content
     covariance pass reproduces qk_e18_probe_upgrades.gen_slot_covariances.
  6. KERNELS: inherited from qk_e22.json control2 (this runner imports the
     same match_kernels function object; the stored pass is asserted).
  7. qk_e18.json probe gates 1+2 (precondition for the reused probes).

REGISTERED PREDICTIONS (merged BEFORE training):
  (i)   SUB-ADDITIVITY: final fresh held CE lands between the predicate arm
        (4.9000) and the variable-k codebook arm (5.0626). Secondary
        reference: the additive-cost prediction is 4.9841 (= E19a 4.9742 -
        0.0785 + 0.0884, the two parents' paired deltas).
  (ii)  the named match terms SURVIVE quantization: total |b_h| mass within
        30% of the predicate arm's 52.734 (i.e. in [36.91, 68.55]).
  (iii) the code dictionaries stay token-class legible on the same
        representative slots as qk_e20_code_dictionaries.json (0, 1, 3, 15,
        22): the fraction of the top-20 codes per slot whose top-10 firing
        contexts share a majority token stays >= 0.6 of the E20a value.

MEASURES: paired fresh CE vs E19a / E22a / E20b / E20a / E9a / E0a / E0b;
old-cooc held; per-seq heldloss; wiring-trajectory npz (standing harness);
codebook snapshots + dead-code event log; variable-k code statistics
(per-slot usage, dead fraction, per-pursuit-step residual ladders, content
bits/token); the full named-term mixture tables (b_h, c_h, profile norms,
per-head pattern-mass shares); code dictionaries on the fixed audit slice;
generalized light probe + covariance-composed wiring on the QUANTIZED
forward. Idempotent on JSON keys and the checkpoint."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json
import time

import numpy as np

import qk_e_common as E
from qk_e_common import Q, C, DEPTH, F, torch
import qk_e7_evenout_run as E7R
import qk_e15_reinvest_run as E15R
import qk_e18_probe_upgrades as E18U
import qk_e17_composed_wiring as E17
import qk_e20_codebook_run as E20R
import qk_e20b_vark_run as E20bR
import qk_e22_predbasis_run as E22R
import qk_deeproute_train_2 as R2

E.DEV = 'cpu' if E.SMOKE else 'cuda'

JP = E.jpath('qk_e31a.json')
GC31 = 1e-4                               # the E19a lasso, unchanged
GATE_TOL = 1e-3
NG = E.NGROUP
QZ_N = E20R.QZ_N
K_ATTN, K_MLP = E20bR.K_ATTN, E20bR.K_MLP
STEM = 'qk_e31a_a'

# parent reference numbers (stored in their JSONs)
CE_PRED = 4.9000                          # E22a 3-seed mean (qk_e29.json)
CE_PRED_SEED0 = 4.8957                    # E22a seed 0 (qk_e22.json)
CE_VARK = 5.0626                          # E20b (qk_e20b.json)
CE_PARENT = 4.9742                        # E19a (qk_e19.json)
D_PRED = -0.0785                          # E22a paired minus E19a
D_VARK = 0.0884                           # E20b paired minus E19a
B_MASS_PRED = 52.734                      # E22a total |b| mass
REP_SLOTS = E20R.REP_SLOTS                # (0, 1, 3, 15, 22)


# ---------------- the composed architecture ----------------
class E31Route(E20bR.E20bRoute, E22R.E22Route):
    """Variable-k codebook slots (E20b) + predicate-basis attention (E22).

    Cooperative MRO: E31Route -> E20bRoute -> E20Route -> E22Route ->
    E15cRoute, so __init__ installs the codebook buffers AND the predicate
    parameters with no RNG consumed beyond the parent's stream (predicate
    parameters are zeros; codebooks use their own fixed-seed generator) --
    the shared parameters stay bit-identical to make_e15c (control 1)."""

    def forward(self, idx, collect=None, sub_entry=None, census_cb=None,
                census_full_cb=None, pat_hook=None):
        B, Tq = idx.shape
        Ws, s = self.Ws, self.s
        cos = self.cos[None, :Tq, None, :]
        sin = self.sin[None, :Tq, None, :]
        mask = self.mask[:Tq, :Tq]
        maskf = mask.float()
        if self.pred_on:
            Kprev, Ksame = E22R.match_kernels(idx, maskf)
        e = F.rms_norm(self.wte(idx), (Ws,))
        streams = [e]
        if self.training and torch.is_grad_enabled() and self.qz_on:
            self.qz_step += 1
        self._commit_sum, self._commit_n = None, 0
        # per-forward quantization cache is invalid under stream substitution
        cache = {} if sub_entry is None else None

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
            hn = self._qz_full(self.slot_norm(x), 2 * l, cache, collect)

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
                census_full_cb(l, pat)
            y = torch.einsum('bhqk,bkhd->bqhd', pat, v)
            y = y.reshape(B, Tq, self.Dc)
            aw = torch.zeros(B, Tq, Ws, device=y.device, dtype=y.dtype)
            aw[..., s * 2 * l:s * (2 * l + 1)] = blk.c_proj(y)
            x = x + aw
            xn = self._qz_full(self.slot_norm(x), 2 * l + 1, cache, collect)
            mw_s = blk.Down(blk.Left(xn) * blk.Right(xn)) + blk.Down_bias
            mw = torch.zeros(B, Tq, Ws, device=y.device, dtype=y.dtype)
            mw[..., s * (2 * l + 1):s * (2 * l + 2)] = mw_s
            if collect is not None:
                collect.setdefault('attn_write', []).append(aw.detach())
                collect.setdefault('mlp_write', []).append(mw.detach())
            streams.append(aw)
            streams.append(mw)
        # readout: continuous stream through the GLOBAL norm (E20 exemption)
        x = F.rms_norm(entry(self.depth), (Ws,))
        logits = x @ self.wte.weight.t()
        if self._commit_sum is not None:
            self._aux = E20R.QZ_BETA * self._commit_sum / self._commit_n
        return 30 * torch.tanh(logits / 30)


def make_e31(s=None, pred_on=True, qz_on=True):
    hidden = 4 * Q.D
    if s is None:
        s, _ = E15R.solve_slot_c(hidden)
    torch.manual_seed(Q.SEED)
    m = E31Route('E31a', DEPTH, s, Q.D, Q.NH, Q.HD, hidden).to(E.DEV)
    m.pred_on = pred_on
    m.qz_on = qz_on
    return m


# ---------------- training-time logging ----------------
_SNAPSHOTS = {}


def flush_dead_events(model, phase='train'):
    ev = getattr(model, 'qz_dead_events', None)
    if ev:
        with open(E.jpath('qk_e31a_deadcode_events.jsonl'), 'a') as f:
            for e_ in ev:
                e_['phase'] = phase
                f.write(json.dumps(e_) + '\n')
        model.qz_dead_events = []


def qz_step_cb(step, model):
    if not getattr(model, 'qz_on', False):
        return
    flush_dead_events(model)
    if step % 1000 == 0:
        _SNAPSHOTS[f'step{step:05d}'] = \
            model.qz_codebook.detach().cpu().numpy()
        np.savez(E.jpath('qk_e31a_codebook_snapshots.npz'), **_SNAPSHOTS)


qz_step_cb.every = 250


def trainer(lr, gc, steps, **kw):
    kw.setdefault('step_cb', qz_step_cb)
    return E.train_muon(lr, gc, steps, lr_adamw=E.get_lr(), **kw)


def three_step(factory, gc, steps=3):
    log = E.train_muon(E7R.muon_lr(), gc, steps, log_every=1, save=False,
                       factory=factory, lr_adamw=E.get_lr())
    return {'per_step_ce': [x[1] for x in log['train_loss']],
            'held100_ce': log['final_held_ce']}


# ---------------- controls ----------------
def control_preconditions():
    key = 'control0_inherited_gates'
    if E.SMOKE or (E.loadj(JP).get(key) or {}).get('pass'):
        return
    e18 = E.loadj(E.jpath('qk_e18.json'))
    g1 = e18.get('gate1_uniform11_weight_support', {}).get('pass')
    g2 = e18.get('gate2_cov_composed_E9a', {}).get('pass')
    e22 = E.loadj(E.jpath('qk_e22.json')).get('control2_kernels_vs_e21', {})
    e20b = E.loadj(E.jpath('qk_e20b.json'))
    toy = e20b.get('control3_toy_ema_k4', {})
    assert g1 and g2, ('qk_e18.json probe gates not passed', g1, g2)
    assert e22.get('pass'), 'qk_e22.json control2 (match kernels) not passed'
    rec = {'qk_e18_gate1_weight_support': bool(g1),
           'qk_e18_gate2_cov_composed': bool(g2),
           'qk_e22_control2_kernels_vs_e21': e22,
           'qk_e20b_control3_planted_toy_k4': {
               k: toy.get(k) for k in ('center_max_cos_min', 'final_rel_mse',
                                       'pass')},
           'note': 'this runner imports the SAME function objects '
                   '(qk_e22_predbasis_run.match_kernels, '
                   'qk_e20_codebook_run.mp_quantize / ema_update), so the '
                   'parents\' stored gate results apply verbatim',
           'pass': True}
    E.merge(JP, key, rec)
    print(f"{key}: inherited gates PASS", flush=True)


def control_bypass(s_c):
    """Gate 1: predicate terms off + quantization off == the E19a parent."""
    key = 'control1_bypass_both_off'
    done = E.loadj(JP).get(key)
    if done and done.get('pass'):
        print(f"{key}: already passed -- skip", flush=True)
        return
    m19 = E15R.make_e15c(s=s_c).eval().float()
    m31 = make_e31(s=s_c, pred_on=False, qz_on=False).eval().float()
    p19 = dict(m19.named_parameters())
    shared = {nm: p for nm, p in m31.named_parameters() if nm in p19}
    pdiff = max(float((p - p19[nm]).abs().max()) for nm, p in shared.items())
    extra = {nm: float(p.abs().max()) for nm, p in m31.named_parameters()
             if nm not in p19}
    idx = E.OLD_HELD[:2, :Q.T]
    with torch.no_grad():
        d_off = float((m31(idx) - m19(idx)).abs().max())
        m31.pred_on = True                        # predicate params are zeros
        d_pred_on = float((m31(idx) - m19(idx)).abs().max())
        m31.qz_on = True                          # informational: real shift
        d_both_on = float((m31(idx) - m19(idx)).abs().max())
        m31.pred_on, m31.qz_on = False, False
    del m19, m31
    if not E.SMOKE:
        torch.cuda.empty_cache()
    parent = three_step(lambda: E15R.make_e15c(s=s_c), GC31)
    mine = three_step(lambda: make_e31(s=s_c, pred_on=False, qz_on=False),
                      GC31)
    step_diff = max(abs(a - b) for a, b in
                    zip(parent['per_step_ce'], mine['per_step_ce']))
    held_diff = abs(parent['held100_ce'] - mine['held100_ce'])
    rec = {'shared_param_identity_max_abs_diff': pdiff,
           'extra_param_max_abs_at_init': extra,
           'forward_both_off_max_logit_diff': d_off,
           'forward_pred_on_at_init_max_logit_diff': d_pred_on,
           'forward_both_on_at_init_max_logit_diff_informational': d_both_on,
           'train3_parent_per_step_ce': parent['per_step_ce'],
           'train3_bypass_per_step_ce': mine['per_step_ce'],
           'train3_max_per_step_abs_diff': step_diff,
           'train3_held100_abs_diff': held_diff,
           'note': 'the 3-step training identity proves the only changes vs '
                   'the E19a parent are the predicate terms and the '
                   'quantization -- both individually reversible to zero',
           'pass': bool(pdiff == 0.0 and d_off == 0.0 and d_pred_on == 0.0
                        and max(extra.values(), default=0.0) == 0.0
                        and step_diff == 0.0 and held_diff < 1e-6)}
    E.merge(JP, key, rec)
    print(f"{key}: params {pdiff:.1e}, fwd off/pred-on {d_off:.1e}/"
          f"{d_pred_on:.1e} (both-on shift {d_both_on:.3f}), 3-step "
          f"{step_diff:.1e}/{held_diff:.1e} -> "
          f"{'PASS' if rec['pass'] else 'FAIL'}", flush=True)
    assert rec['pass'], f'{key} FAILED'


def control_k_wiring(s_c):
    key = 'control2_k_wiring'
    done = E.loadj(JP).get(key)
    if done and done.get('pass'):
        print(f"{key}: already passed -- skip", flush=True)
        return
    m = make_e31(s=s_c).eval().float()
    with torch.no_grad():
        col = {'codes': {}}
        m(E.OLD_HELD[:2, :Q.T], collect=col)
    got = {k: int(v[0].shape[1]) for k, v in col['codes'].items()}
    ok = all(got[k] == (K_ATTN if k % 2 == 0 else K_MLP) for k in got)
    rec = {'code_columns_per_slot': {str(k): got[k] for k in sorted(got)},
           'expected': f'{K_ATTN} on attention (even) slots, {K_MLP} on MLP '
                       '(odd) slots', 'pass': bool(ok and len(got) > 0)}
    E.merge(JP, key, rec)
    print(f"{key}: {len(got)} quantized slots -> "
          f"{'PASS' if rec['pass'] else 'FAIL'}", flush=True)
    assert rec['pass'], f'{key} FAILED'
    del m
    if not E.SMOKE:
        torch.cuda.empty_cache()


def control_capacity(s_c):
    key = 'control3_capacity'
    done = E.loadj(JP).get(key)
    if done and done.get('pass'):
        print(f"{key}: already passed -- skip", flush=True)
        return
    m = make_e31(s=s_c).eval().float()
    b = E.OLD_HELD[:2, :Q.T]
    with torch.no_grad():
        e = F.rms_norm(m.wte(b), (m.Ws,))
        v = F.rms_norm(e[..., :m.s], (m.s,)).reshape(-1, m.s).float()[:128]
        uniq = torch.unique(v, dim=0)
        Cb = uniq / uniq.norm(dim=1, keepdim=True).clamp_min(1e-8)
        recon, _, _, _ = E20R.mp_quantize(v, Cb, 15)
        rel = float((v - recon).norm() / v.norm().clamp_min(1e-12))
    del m
    if not E.SMOKE:
        torch.cuda.empty_cache()
    rec = {'n_distinct': int(uniq.shape[0]), 'k_steps': 15,
           'rel_error': rel, 'pass': bool(rel < 1e-4)}
    E.merge(JP, key, rec)
    print(f"{key}: {rec['n_distinct']} distinct, rel error {rel:.2e} -> "
          f"{'PASS' if rec['pass'] else 'FAIL'}", flush=True)
    assert rec['pass'], f'{key} FAILED'


def control_toy():
    """Gate 4: the planted-toy EMA recovery at k=4, run through
    qk_e20b_vark_run.control_toy_k4 verbatim (it records into qk_e20b.json
    and is idempotent there); the result is mirrored here."""
    key = 'control4_planted_toy_k4'
    if (E.loadj(JP).get(key) or {}).get('pass'):
        return
    E20bR.control_toy_k4()
    rec = E.loadj(E.jpath('qk_e20b.json')).get('control3_toy_ema_k4', {})
    assert rec.get('pass'), 'planted-toy control did not pass'
    E.merge(JP, key, {'source': 'qk_e20b.json::control3_toy_ema_k4 '
                                '(qk_e20b_vark_run.control_toy_k4 verbatim)',
                      'result': rec, 'pass': True})


def control_cov_pipeline(s_c):
    key = 'control5_cov_pipeline'
    done = E.loadj(JP).get(key)
    if done and done.get('pass'):
        print(f"{key}: already passed -- skip", flush=True)
        return
    dims = [s_c] * NG
    m = make_e31(s=s_c, qz_on=False)
    held = np.load(E.HELD_PATH)[33000:33032].astype(np.int64)
    rows = torch.from_numpy(held)
    Cb_g, _, Cr_g, n_g = E18U.gen_slot_covariances(m, rows, E.DEV, dims,
                                                   remnant=False)
    Cb_e, Cr_e, n_e = E20R.e20_slot_covariances(m, rows, E.DEV, dims)
    d_b = max(float((a - b).abs().max()) for a, b in zip(Cb_g, Cb_e))
    d_r = max(float((a - b).abs().max()) for a, b in zip(Cr_g, Cr_e))
    rec = {'max_abs_diff_content_cov': d_b, 'max_abs_diff_readout_cov': d_r,
           'pass': bool(n_g == n_e and d_b < 1e-6 and d_r < 1e-6)}
    E.merge(JP, key, rec)
    print(f"{key}: content {d_b:.2e}, readout {d_r:.2e} -> "
          f"{'PASS' if rec['pass'] else 'FAIL'}", flush=True)
    assert rec['pass'], f'{key} FAILED'
    del m
    if not E.SMOKE:
        torch.cuda.empty_cache()


# ---------------- named-term mixture tables ----------------
def mixture_tables(s_c):
    key = 'mixture_weights_E31a'
    if key in E.loadj(JP):
        return
    if E.SMOKE:
        m = make_e31(s=s_c).eval().float()        # exercise the code path
    else:
        if not os.path.exists(E.ckpath(STEM)):
            return
        m, _ = E.load_arm(STEM, lambda: make_e31(s=s_c))
    b = m.pred_b.detach().float().cpu()
    c = m.pred_c.detach().float().cpu()
    prof = m.pred_prof.detach().float().cpu()
    idx = Q.HELD[:4, :Q.T]
    maskf = torch.tril(torch.ones(Q.T, Q.T, dtype=torch.bool,
                                  device=idx.device)).float()
    resid, full = {}, {}
    with torch.no_grad():
        m(idx, census_cb=lambda l, p: resid.__setitem__(l, p.clone()),
          census_full_cb=lambda l, p: full.__setitem__(l, p.clone()))
        Kprev, Ksame = E22R.match_kernels(idx, maskf)
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
    total_b = float(absb.sum())
    top3 = float(absb.sort(descending=True).values[:3].sum()
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
    rel = abs(total_b - B_MASS_PRED) / B_MASS_PRED
    rec = {'checkpoint': f'{STEM}.pt',
           'decomposition_check_max_abs_diff': decomp_max,
           'total_abs_b_mass': round(total_b, 5),
           'total_abs_b_mass_predicate_arm_E22a': B_MASS_PRED,
           'relative_change_vs_E22a': round(rel, 4),
           'top3_abs_b_mass_share': round(top3, 4),
           'top10_by_abs_b': topb,
           'total_abs_c_mass': round(float(c.abs().sum()), 5),
           'mass_shares_rows': 'fresh held [33000:33004], fp32',
           'table': table}
    assert decomp_max < 1e-4, ('pattern decomposition off', decomp_max)
    E.merge(JP, key, rec)
    print(f"mixture tables: total |b| {total_b:.3f} vs E22a {B_MASS_PRED} "
          f"({rel * 100:.1f}% change), top-3 share {top3:.3f}", flush=True)


# ---------------- variable-k code statistics ----------------
def code_stats(s_c):
    key = 'code_stats_E31a'
    if key in E.loadj(JP):
        print(f"{key}: already done -- skip", flush=True)
        return
    if E.SMOKE:
        m = make_e31(s=s_c)
    else:
        if not os.path.exists(E.ckpath(STEM)):
            return
        m, _ = E.load_arm(STEM, lambda: make_e31(s=s_c))
    m.eval().float()
    rows = Q.HELD
    per_slot = {k: [] for k in range(NG)}
    resid_acc = {k: np.zeros(5) for k in range(NG)}
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(rows), 8):
            b = rows[i:i + 8]
            col = {'codes': {}, 'resid_stats': {}}
            m(b[:, :Q.T], collect=col)
            for k, chunks in col['codes'].items():
                per_slot[k].append(torch.cat(chunks, 0))
            for k, sums in col['resid_stats'].items():
                for s5 in sums:
                    resid_acc[k] += np.asarray(s5)
            if (i // 8) % 40 == 0:
                print(f"  code stats: {i + b.shape[0]}/{len(rows)} seqs "
                      f"({time.time() - t0:.0f}s)", flush=True)
    quantized = sorted(k for k in per_slot if per_slot[k])
    slots_rec, bits_total = {}, 0.0
    dead_num = dead_den = 0
    attn_fracs, mlp_fracs = [], []
    for k in quantized:
        codes = torch.cat(per_slot[k], 0).long().numpy()
        ntok, ks = codes.shape
        step_h = [round(E20bR.entropy_bits(
            np.bincount(codes[:, jj], minlength=QZ_N)), 3)
            for jj in range(ks)]
        hj = E20bR.tuple_entropy_bits(codes)
        used = np.zeros(QZ_N, dtype=bool)
        for jj in range(ks):
            used |= np.bincount(codes[:, jj], minlength=QZ_N) > 0
        bits_total += hj
        dead_num += int(QZ_N - used.sum())
        dead_den += QZ_N
        sx, sr1, sr2, srf, ncnt = resid_acc[k]
        rms_c = float(np.sqrt(sx / ncnt))
        rms_f = float(np.sqrt(srf / ncnt))
        frac = rms_f / max(rms_c, 1e-12)
        (attn_fracs if k % 2 == 0 else mlp_fracs).append(frac)
        slots_rec[str(k)] = {
            'module': R2.stream_name(k + 1), 'k_steps': ks,
            'n_tokens': int(ntok),
            'distinct_codes_used': int(used.sum()),
            'dead_fraction': round(float(1.0 - used.sum() / QZ_N), 4),
            'entropy_per_step_bits': step_h,
            'entropy_joint_tuple_bits': round(hj, 3),
            'pursuit_residual_rms': {
                'content': round(rms_c, 4),
                'after_code1': round(float(np.sqrt(sr1 / ncnt)), 4),
                'after_code2': round(float(np.sqrt(sr2 / ncnt)), 4),
                'final': round(rms_f, 4)},
            'final_residual_rms_fraction': round(frac, 4),
            'energy_captured_total': round(1.0 - srf / sx, 4)}
    rec = {'rows': f'fresh held ({len(rows)} seqs x {Q.T} tokens), fp32',
           'quantized_slots': quantized,
           'codebook_n': QZ_N,
           'k_select': {'attention_slots': K_ATTN, 'mlp_slots': K_MLP},
           'dead_code_fraction_overall': round(dead_num / max(dead_den, 1), 4),
           'content_bits_per_token_sum_joint_entropy': round(bits_total, 3),
           'e20b_reference_bits_per_token': 396.831,
           'mean_final_residual_rms_fraction_attention':
               round(float(np.mean(attn_fracs)), 4) if attn_fracs else None,
           'mean_final_residual_rms_fraction_mlp':
               round(float(np.mean(mlp_fracs)), 4) if mlp_fracs else None,
           'e20b_reference_attn_resid_fraction': 0.2519,
           'per_slot': slots_rec}
    E.merge(JP, key, rec)
    print(f"code stats: dead {rec['dead_code_fraction_overall']}, bits/token "
          f"{bits_total:.1f}, attention resid fraction "
          f"{rec['mean_final_residual_rms_fraction_attention']}", flush=True)
    del m
    if not E.SMOKE:
        torch.cuda.empty_cache()


# ---------------- code dictionaries (fixed audit slice) ----------------
def majority_frac(entries, topn=20):
    """Fraction of the top-n codes whose top-10 firing contexts share a
    majority token -- the automated legibility proxy."""
    from collections import Counter
    ok = 0
    use = entries[:topn]
    for e_ in use:
        toks = [ex['ctx'].split('==>')[-1] for ex in e_['examples']]
        if not toks:
            continue
        _, cnt = Counter(toks).most_common(1)[0]
        if cnt >= max(2, (len(toks) + 1) // 2):
            ok += 1
    return round(ok / max(len(use), 1), 4)


def code_dictionaries(s_c):
    key = 'code_dictionaries_E31a'
    if key in E.loadj(JP):
        return
    if E.SMOKE:
        m = make_e31(s=s_c)                       # exercise the code path
    else:
        if not os.path.exists(E.ckpath(STEM)):
            return
        m, _ = E.load_arm(STEM, lambda: make_e31(s=s_c))
    m.eval().float()
    lo, hi = (33000, 33016) if E.SMOKE else E20R.AUDIT_ROWS
    held = np.load(E.HELD_PATH)[lo:hi].astype(np.int64)
    rows = torch.from_numpy(held).to(E.DEV)
    codes_all = {k: [] for k in REP_SLOTS}
    scales_all = {k: [] for k in REP_SLOTS}
    with torch.no_grad():
        for i in range(0, len(rows), 8):
            b = rows[i:i + 8]
            col = {'codes': {}, 'scales': {}}
            m(b[:, :Q.T], collect=col)
            for k in REP_SLOTS:
                codes_all[k].append(torch.cat(col['codes'][k], 0))
                scales_all[k].append(torch.cat(col['scales'][k], 0))
    del m
    torch.cuda.empty_cache()
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained('gpt2')

        def dec(ids):
            return tok.decode(list(ids))
    except Exception as ex:
        print(f"dictionaries: tokenizer unavailable ({ex})", flush=True)

        def dec(ids):
            return ' '.join(str(int(t)) for t in ids)
    full, legib = {}, {}
    for k in REP_SLOTS:
        codes = torch.cat(codes_all[k], 0).long().numpy()
        scales = torch.cat(scales_all[k], 0).numpy()
        c1 = np.bincount(codes[:, 0], minlength=QZ_N)
        top50 = np.argsort(c1)[::-1][:50]
        entries = []
        for c in top50:
            if c1[c] == 0:
                break
            sel = np.nonzero(codes[:, 0] == c)[0]
            order = sel[np.argsort(-np.abs(scales[sel, 0]))][:10]
            exs = []
            for f in order:
                si, pos = int(f // Q.T), int(f % Q.T)
                st = max(0, pos - 7)
                ids = held[si, st:pos + 1]
                exs.append({'seq': lo + si, 'pos': pos,
                            'scale1': round(float(scales[f, 0]), 3),
                            'ctx': dec(ids[:-1]) + ' ==>' + dec(ids[-1:])})
            entries.append({'code': int(c), 'count_step1': int(c1[c]),
                            'mean_abs_scale1': round(float(
                                np.abs(scales[sel, 0]).mean()), 3),
                            'examples': exs})
        full[str(k)] = {'module': R2.stream_name(k + 1), 'codes': entries}
        legib[str(k)] = majority_frac(entries)
    with open(E.jpath('qk_e31a_code_dictionaries.json'), 'w') as f:
        json.dump({'audit_slice': f'fresh34k[{lo}:{hi}] (the FIXED audit '
                                  'slice)', 'rep_slots': list(REP_SLOTS),
                   'per_slot': full}, f, indent=1)
    ref = {}
    p20 = E.jpath('qk_e20_code_dictionaries.json')
    if os.path.exists(p20):
        d20 = json.load(open(p20))
        for k in REP_SLOTS:
            ref[str(k)] = majority_frac(d20['per_slot'][str(k)]['codes'])
    sample = {str(k): {'module': full[str(k)]['module'],
                       'top5_codes': [
                           {'code': e_['code'],
                            'count_step1': e_['count_step1'],
                            'examples': [x['ctx'] for x in e_['examples'][:3]]}
                           for e_ in full[str(k)]['codes'][:5]]}
              for k in REP_SLOTS}
    mine = float(np.mean([legib[str(k)] for k in REP_SLOTS]))
    theirs = float(np.mean([ref[str(k)] for k in REP_SLOTS])) if ref else None
    E.merge(JP, key, {
        'file': 'qk_e31a_code_dictionaries.json',
        'audit_slice': f'fresh34k[{lo}:{hi}]',
        'rep_slots': list(REP_SLOTS),
        'legibility_metric': 'fraction of the top-20 codes per slot whose '
                             'top-10 |scale1| contexts share a majority '
                             'firing token',
        'legibility_per_slot_E31a': legib,
        'legibility_per_slot_E20a_reference': ref,
        'legibility_mean_E31a': round(mine, 4),
        'legibility_mean_E20a': round(theirs, 4) if theirs else None,
        'sample_top5_per_slot': sample})
    print(f"code dictionaries: legibility {mine:.3f} (E20a {theirs})",
          flush=True)


# ---------------- probes ----------------
def probes(s_c, stem=None, jp=None, tag='E31a'):
    """The generalized variable-slot-dim light probe + the covariance-composed
    re-scoring on the QUANTIZED + predicate forward. stem/jp/tag default to
    this arm's; qk_e33_compose_seeds_run passes its per-seed checkpoints so the
    seed replicates go through this exact path."""
    STEM, JP = (stem or globals()['STEM']), (jp or globals()['JP'])
    if E.SMOKE or not os.path.exists(E.ckpath(STEM)):
        return
    dims = [s_c] * NG
    j = E.loadj(JP)
    if f'light_probe_{tag}_var_dims' not in j:
        print(f'{tag} light probe (quantized + predicate forward) ...',
              flush=True)
        m, _ = E.load_arm(STEM, lambda: make_e31(s=s_c))
        Ws = m.wte.weight.shape[1]
        assert Ws == 2 * DEPTH * s_c
        base, dce = E18U.gen_consumption(m, Ws)
        wp = E18U.wpairs(m, dims)
        G = E18U.gen_gram_table(m, dims)
        sup = E18U.score(G, wp)
        cau = [dce[li][si] for li, si in wp]
        eff = [k for k in range(len(wp)) if cau[k] > C.EFFECTUAL]
        agr = E17.agreement(sup, cau, eff)
        pairs_sorted = sorted([(li, si, dce[li][si]) for li in dce
                               for si in dce[li]], key=lambda p: -p[2])
        rec = {'checkpoint': f'{STEM}.pt', 'stream_width': Ws,
               'slot_dims': dims, 'compute_width': Q.D,
               'base_ce_fp32_abl_oldheld': round(base, 5),
               'wiring_n_pairs': len(wp),
               'wiring_spearman_all': agr['spearman_all'],
               'wiring_n_effectual': len(eff),
               'wiring_spearman_effectual': agr['spearman_effectual'],
               'wiring_top10_precision': agr['top10_precision'],
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
               'note': 'generalized variable-slot-dim probe (qk_e18 gate 1); '
                       'the forward runs with quantization AND the named '
                       'predicate terms active (they are the architecture); '
                       'weight support covers the bilinear read matrices '
                       'only -- the predicate terms read tokens, not the '
                       'stream'}
        E.merge(JP, f'light_probe_{tag}_var_dims', rec)
        print(f"{tag} wiring Spearman all {agr['spearman_all']} "
              f"effectual({len(eff)}) {agr['spearman_effectual']}", flush=True)
        del m
        torch.cuda.empty_cache()
    j = E.loadj(JP)
    if f'composed_wiring_{tag}' not in j:
        lp = j[f'light_probe_{tag}_var_dims']
        print(f'{tag} covariance-composed re-scoring (quantized content) ...',
              flush=True)
        m, _ = E.load_arm(STEM, lambda: make_e31(s=s_c))
        wp = E18U.wpairs(m, dims)
        cau = E18U.stored_cau(lp, wp)
        G = E18U.gen_gram_table(m, dims)
        plain = E18U.score(G, wp)
        held = np.load(E.HELD_PATH)[33000:33000 + E18U.N_COV].astype(np.int64)
        Cb, Cr, n_samp = E20R.e20_slot_covariances(
            m, torch.from_numpy(held), E.DEV, dims)
        cov = E18U.score(G, wp, lambda li, si: Cb[si - 1])
        cov_ro = E18U.score(G, wp, lambda li, si:
                            Cr[si - 1] if li == DEPTH else Cb[si - 1])
        eff = [k for k in range(len(wp)) if cau[k] > C.EFFECTUAL]
        tables = {'plain': E17.agreement(plain, cau, eff),
                  'cov_composed': E17.agreement(cov, cau, eff),
                  'cov_composed_readout_globalnorm':
                      E17.agreement(cov_ro, cau, eff)}
        chk = abs(tables['plain']['spearman_all'] - lp['wiring_spearman_all'])
        assert chk <= GATE_TOL, \
            f'{tag} plain does not reproduce the light probe ({chk})'
        E.merge(JP, f'composed_wiring_{tag}', {
            'checkpoint': f'{STEM}.pt', 'slot_dims_uniform': s_c,
            'plain_reproduction_abs_diff': round(chk, 6),
            'n_samples': n_samp, 'tables': tables})
        print(f"{tag} plain {tables['plain']['spearman_all']} -> cov "
              f"{tables['cov_composed']['spearman_all']}", flush=True)
        del m
        torch.cuda.empty_cache()


def pair_extra(others):
    if E.SMOKE:
        return
    for ctl, label in others:
        f_arm, f_ctl = f'{STEM}_heldloss.npy', f'{ctl}_heldloss.npy'
        if os.path.exists(f'{E.QK}/{f_ctl}') \
                and os.path.exists(f'{E.QK}/{f_arm}'):
            E.merge(JP, f'E31a_minus_{label}_fresh',
                    E.paired(f_arm, f_ctl, len(Q.HELD), label))


def summarize():
    j = E.loadj(JP)
    summary = {'parents': {
        'E19a_bandwidth_frontier': {'ce': CE_PARENT},
        'E22a_predicate_basis': {'ce_seed0': CE_PRED_SEED0,
                                 'ce_3seed_mean': CE_PRED,
                                 'paired_vs_e19a': D_PRED,
                                 'total_abs_b_mass': B_MASS_PRED},
        'E20b_variable_k_codebook': {'ce': CE_VARK,
                                     'paired_vs_e19a': D_VARK}},
        'additive_cost_reference_ce': round(CE_PARENT + D_PRED + D_VARK, 4)}
    if 'E31a' in j:
        ce = j['E31a']['final_held_ce_fresh_bf16']
        row = {'final_held_ce_fresh_bf16': ce, 'diverged': j['E31a']['diverged']}
        for lab in ('e19a', 'e22a', 'e20b', 'e20a', 'e9a'):
            k = f'E31a_minus_{lab}_fresh'
            if k in j:
                row[f'minus_{lab}_paired'] = round(j[k][f'minus_{lab}'], 4)
                row[f'minus_{lab}_se_seq'] = round(
                    j[k][f'minus_{lab}_se_seq'], 5)
        if 'composed_wiring_E31a' in j:
            t = j['composed_wiring_E31a']['tables']
            row['plain'] = t['plain']['spearman_all']
            row['cov_composed_quantized'] = t['cov_composed']['spearman_all']
        cs = j.get('code_stats_E31a', {})
        if cs:
            row['content_bits_per_token'] = \
                cs['content_bits_per_token_sum_joint_entropy']
            row['dead_code_fraction'] = cs['dead_code_fraction_overall']
            row['attn_final_resid_rms_fraction_mean'] = \
                cs['mean_final_residual_rms_fraction_attention']
        mw = j.get('mixture_weights_E31a', {})
        if mw:
            row['total_abs_b_mass'] = mw['total_abs_b_mass']
        summary['E31a'] = row
        if ce is not None:
            inside = CE_PRED <= ce <= CE_VARK
            summary['prediction_i_verdict'] = (
                f"{'CONFIRMED' if inside else 'REFUTED'}: CE {round(ce, 4)} "
                f"vs the registered window [{CE_PRED}, {CE_VARK}]; the "
                f"additive-cost reference was "
                f"{round(CE_PARENT + D_PRED + D_VARK, 4)} "
                f"({'sub' if ce < CE_PARENT + D_PRED + D_VARK else 'super'}"
                f"-additive)")
        if mw:
            rel = mw['relative_change_vs_E22a']
            summary['prediction_ii_verdict'] = (
                f"{'CONFIRMED' if rel <= 0.30 else 'REFUTED'}: total |b| "
                f"mass {mw['total_abs_b_mass']} vs the predicate arm's "
                f"{B_MASS_PRED} ({rel * 100:.1f}% change, registered <= 30%)")
        cd = j.get('code_dictionaries_E31a', {})
        if cd and cd.get('legibility_mean_E20a'):
            r = cd['legibility_mean_E31a'] / cd['legibility_mean_E20a']
            summary['prediction_iii_verdict'] = (
                f"{'CONFIRMED' if r >= 0.6 else 'REFUTED'}: dictionary "
                f"legibility {cd['legibility_mean_E31a']} vs E20a "
                f"{cd['legibility_mean_E20a']} (ratio {round(r, 3)}, "
                f"registered >= 0.6)")
    E.merge(JP, 'summary_E31a', summary)
    print(json.dumps({'summary_E31a': summary}, indent=2), flush=True)


if __name__ == '__main__':
    E.setup()
    s_c, _ = E15R.solve_slot_c(4 * Q.D)
    if not E.SMOKE:
        assert s_c == 15, s_c

    # ---- controls (before training) ----
    control_preconditions()
    control_bypass(s_c)
    control_k_wiring(s_c)
    control_capacity(s_c)
    control_toy()
    if not E.SMOKE:
        control_cov_pipeline(s_c)

    # ---- registered predictions (before training) ----
    if 'E31a_prediction' not in E.loadj(JP):
        E.merge(JP, 'E31a_prediction', {
            'registered_before_training': True,
            'design': 'predicate-basis attention (E22a: signed positional '
                      'profile + b_h*MATCH_prev + c_h*MATCH_same per head, '
                      'AdamW, init 0) composed with variable-k codebook '
                      'slots (E20b: n=256 unit-norm codes per slot, k=4 on '
                      'attention slots, k=2 on MLP slots, EMA 0.99, '
                      'commitment 0.25, dead-reinit 200, straight-through) '
                      'on the E19a base (E15c 24x15 bandwidth, lasso 1e-4, '
                      'Muon 0.02 / AdamW family lr)',
            'i_sub_additive_ce': f'final fresh held CE lands between the '
                                 f'predicate arm ({CE_PRED}) and the '
                                 f'variable-k codebook arm ({CE_VARK}); the '
                                 f'additive-cost reference is '
                                 f'{round(CE_PARENT + D_PRED + D_VARK, 4)}',
            'ii_named_terms_survive_quantization':
                f'total |b_h| mixture mass within 30% of the predicate '
                f'arm\'s {B_MASS_PRED} (i.e. in '
                f'[{round(B_MASS_PRED * 0.7, 2)}, '
                f'{round(B_MASS_PRED * 1.3, 2)}])',
            'iii_dictionaries_legible':
                'on the same representative slots as '
                'qk_e20_code_dictionaries.json (0, 1, 3, 15, 22), the '
                'fraction of the top-20 codes whose top-10 firing contexts '
                'share a majority token stays >= 0.6 of the E20a value',
            'parents': {'E22a': CE_PRED, 'E20b': CE_VARK, 'E19a': CE_PARENT}})

    if 'E31a_config' not in E.loadj(JP):
        E.merge(JP, 'E31a_config', {
            'group_coeff': GC31, 'muon_lr': E7R.muon_lr(),
            'adamw_lr': E.get_lr(), 'slot': s_c,
            'codebook': {'n': QZ_N, 'k_attention': K_ATTN, 'k_mlp': K_MLP,
                         'decay': E20R.QZ_DECAY, 'beta': E20R.QZ_BETA,
                         'dead_steps': E20R.QZ_DEAD},
            'predicate_params': 'pred_prof / pred_b / pred_c on AdamW '
                                'no-decay via muon_exclude (E22 convention)',
            'logging': ['qk_e31a_a_traj.npz',
                        'qk_e31a_codebook_snapshots.npz',
                        'qk_e31a_deadcode_events.jsonl']})

    # ---- train ----
    E.train_arm(STEM, JP, 'E31a', lambda: make_e31(s=s_c), GC31,
                lr=E7R.muon_lr(), trainer=trainer,
                extra={'optimizer': 'muon', 'slot': s_c,
                       'parents': 'E22a (qk_e22_a) + E20b (qk_e20b_a), both '
                                  'on the E19a base (qk_e19_a)',
                       'design': 'predicate-basis attention + variable-k '
                                 'codebook slots composed'})
    E.oldheld_record(STEM, lambda: make_e31(s=s_c), JP, 'E31a_oldheld')
    E.paired_fresh(STEM, JP, 'E31a')
    pair_extra((('qk_e19_a', 'e19a'), ('qk_e22_a', 'e22a'),
                ('qk_e20b_a', 'e20b'), ('qk_e20_a', 'e20a'),
                ('qk_e9_a', 'e9a')))
    E20R.save_seq_heldloss(STEM)

    # ---- measurement ----
    mixture_tables(s_c)
    code_stats(s_c)
    code_dictionaries(s_c)
    probes(s_c)
    summarize()
    print('e31a compose run done', flush=True)
