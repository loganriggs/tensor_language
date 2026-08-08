"""Predicate-basis attention at w1152 -- local's new top-priority branch point.

At w264 the predicate-basis arm is the program's leader on both axes at once
(qk_e22/e29/e31/e32): CE 4.9000 +/- 0.0068 over 3 seeds against the frontier
4.9858 and the recipe 5.0454 -- gaps of 8 to 22 pooled sd, and only +0.049
over UNCONSTRAINED vanilla -- while simultaneously being the best
interpretability asset in the program, because the named terms ABSORB match
structure out of the learned bilinear pattern (residual MATCH_prev cos^2
0.5036 -> 0.0951, zero programmatic heads left in the residual against 42 in
the full model), carry 77% of induction causally, and account for 83-86% of
the selection cost.

THE CHANGE, and only this change: every head gets three named pattern terms
added to its learned bilinear pattern --

    pattern_h  <-  s1*s2  +  profile_h[offset]  +  b_h * MATCH_prev
                                               +  c_h * MATCH_same

where MATCH_prev[i,j] = 1[token_{j-1} == token_i] and MATCH_same[i,j] =
1[token_j == token_i], both causal-masked, exactly qk_e21_census_run's
features 1 and 0; the positional profile is indexed by query-key offset and
is SIGNED (this family has no softmax, so predicates must be signed). All
three parameters init to EXACTLY zero, so the model is bit-identical to its
parent at init and consumes no RNG -- gate 1 asserts that rather than
assuming it.

PARENT: combo3e5loss, the readable recipe at w1152 (E1Route: partitioned
write slots with masked full decoders, per-slot RMSNorm, Muon 0.02, in-loss
group-lasso 3e-5). This follows local's instruction to put the named terms
on the RECIPE rather than on the bandwidth arm, so the comparison is a clean
single change. Note that the w264 predicate arm sits on the BANDWIDTH
architecture instead, so this run additionally tells us whether naming pays
on a slot geometry it has never been tried on -- stated here so the result
is not over-read as a straight transfer.

OPTIMIZER ROUTING (the one thing that would silently corrupt this arm):
pred_prof is 3D, so the scale trainer's dim>=2 rule would hand it to Muon,
whose orthogonalization has no meaning for a table of per-offset pattern
coefficients. The model therefore declares muon_exclude = ('pred_',) and
qk_s_muon_run.muon_params_split now honours that prefix list exactly as
qk_e_common's has all along -- predicate parameters train on AdamW-no-decay.

GATES, all before training:
  1. PREDICATE-ZERO BIT-EXACT REDUCTION: with the named terms off, the model
     must be parameter-identical to a recipe model, reproduce its forward
     exactly, AND reproduce 3 steps of its training exactly through the
     SCALE trainer (which also proves the muon_exclude routing change is
     inert when there is nothing to exclude).
  2. KERNEL IDENTITY vs qk_e21_census_run.build_feats: our MATCH_prev and
     MATCH_same must equal the census kernels elementwise on real token
     rows. Every predicate claim downstream is stated in the census's
     vocabulary, so the two must be the same object.
  3. TERMS ARE LIVE: with a nonzero b_h the forward must actually move, and
     the analytic derivative of the pattern w.r.t. b_h must equal the
     MATCH_prev kernel exactly (the named terms are additive on the
     bilinear product, so d(pattern)/d(b_h) = K exactly -- E28's rule).

Outputs qk_s_w1152_pred3e5.{json,pt} + heldloss/f34kloss npys. Idempotent.
"""
import os
import sys

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import json
import math

import qk_tokenline_train as Q

Q.gpu_guard = lambda *a, **k: None

import qk_s_gate_run as G

G.MEM_BUDGET_MIB = 26000

import qk_w1152_train as W2
import qk_deeproute_train as R
import qk_v8_train as V8T
import qk_e_common as E
from qk_e_common import DEPTH, F, nn, torch
import qk_e1_slotnorm_run as E1R
import qk_s_muon_run as M

ARM = sys.argv[1] if len(sys.argv) > 1 else 'pred3e5'
COEFF = 3e-5                                  # the recipe's lasso, unchanged
STEM = f'qk_s_w1152_{ARM}'
JP = os.path.join(G.OUT_DIR, f'{STEM}.json')
CONTROL_STEM = 'qk_s_w1152_combo3e5loss'


def match_kernels(idx, maskf):
    """MATCH_prev = 1[tok_{j-1} == tok_i], MATCH_same = 1[tok_j == tok_i],
    causal-masked. Verbatim qk_e22_predbasis_run.match_kernels, which gate 2
    checks against qk_e21_census_run.build_feats features 1 and 0."""
    prevtok = torch.roll(idx, 1, 1)
    prevtok[:, 0] = -1
    qtok = idx.unsqueeze(2)
    kprev = (prevtok.unsqueeze(1) == qtok).float() * maskf
    ksame = (idx.unsqueeze(1) == qtok).float() * maskf
    return kprev, ksame


class PredRecipeRoute(E1R.E1Route):
    """The w1152 readable recipe + per-head named pattern terms.

    Same three terms and the same insertion point as qk_e22's E22Route, but
    carried on E1Route (masked full decoders, global embedding norm,
    per-slot RMSNorm at the two module inputs) instead of E15cRoute."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.pred_on = True
        self.muon_exclude = ('pred_',)
        NH = Q.NH
        # all zeros -> no RNG consumed -> shared params bit-identical to the
        # recipe at init (gate 1)
        self.pred_prof = nn.Parameter(torch.zeros(DEPTH, NH, Q.T))
        self.pred_b = nn.Parameter(torch.zeros(DEPTH, NH))
        self.pred_c = nn.Parameter(torch.zeros(DEPTH, NH))
        ar = torch.arange(Q.T)
        self.register_buffer('offmat',
                             (ar[:, None] - ar[None, :]).clamp(min=0))

    def pred_terms(self, l, Kprev, Ksame, maskf, Tq):
        prof = self.pred_prof[l][:, self.offmat[:Tq, :Tq]] * maskf
        return (prof[None]
                + self.pred_b[l].view(1, -1, 1, 1) * Kprev[:, None]
                + self.pred_c[l].view(1, -1, 1, 1) * Ksame[:, None])

    def forward(self, idx, collect=None, sub_entry=None, entry_override=None,
                mlp_sub=None, coef_out=None, attn_sub=None,
                census_cb=None, census_full_cb=None, pat_hook=None):
        B, Tq = idx.shape
        Dm = self.wte.weight.shape[1]
        NHm, HDm = Q.NH, Q.HD
        e = F.rms_norm(self.wte(idx), (Dm,))
        streams = [e]
        cos = self.cos[None, :Tq, None, :]
        sin = self.sin[None, :Tq, None, :]
        mask = self.mask[:Tq, :Tq]
        maskf = mask.float()
        if self.pred_on:
            Kprev, Ksame = match_kernels(idx, maskf)

        def entry(li):
            if entry_override is not None and li in entry_override:
                return entry_override[li]
            sub = sub_entry.get(li) if sub_entry is not None else None
            return self.assemble(li, streams, sub, coef_out)

        for l, blk in enumerate(self.h):
            x = entry(l)
            if collect is not None:
                collect['entry_norm'].append(
                    x.detach().float().norm(dim=-1).mean().item())
                if 'entry' in collect:
                    collect['entry'].append(x.detach())
            hn = self.slot_norm(x)

            def qk(lin):
                z = lin(hn).view(B, Tq, NHm, HDm)
                return Q.apply_rot(F.rms_norm(z, (HDm,)), cos, sin)

            q, k = qk(blk.c_q), qk(blk.c_k)
            q2, k2 = qk(blk.c_q2), qk(blk.c_k2)
            v = blk.c_v(hn).view(B, Tq, NHm, HDm)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HDm
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HDm
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
            y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, Tq, Dm)
            aw = blk.c_proj(y)
            if self.proj:
                aw = aw * self.wmask[2 * l].to(aw.dtype)
            if attn_sub is not None and l in attn_sub:
                aw = attn_sub[l]
            x = x + aw
            if mlp_sub is not None and l in mlp_sub:
                mw = mlp_sub[l]
            else:
                xn = self.slot_norm(x)
                mw = blk.Down(blk.Left(xn) * blk.Right(xn)) + blk.Down_bias
                if self.proj:
                    mw = mw * self.wmask[2 * l + 1].to(mw.dtype)
            if collect is not None:
                collect['attn_write'].append(aw.detach())
                collect['mlp_write'].append(mw.detach())
            streams.append(aw)
            streams.append(mw)
        x = entry(self.depth)
        if collect is not None and 'entry' in collect:
            collect['entry'].append(x.detach())
        x = F.rms_norm(x, (Dm,))
        logits = x @ self.wte.weight.t()
        return 30 * torch.tanh(logits / 30)


def make_pred(pred_on=True):
    """Mirrors E1R.make_e1's construction discipline exactly."""
    from qk_e_common import C
    C.register('PRED1152')
    torch.manual_seed(Q.SEED)
    m = PredRecipeRoute('PRED1152', DEPTH).to(E.DEV)
    m.norm_groups = E.NGROUP
    m.pred_on = pred_on
    return m


def factory():
    return make_pred(pred_on=True)


def three_step_scale(fac, micro, lr_adamw, steps=3):
    prev = M.factory
    M.factory = fac
    try:
        log = M.train_muon_run(0.02, lr_adamw, steps, micro, save_stem=None,
                               log_every=1)
    finally:
        M.factory = prev
    return {'per_step_ce': [x[1] for x in log['train_loss']],
            'held100_ce': log['final_held_ce']}


# ---------------- gates ----------------
def gate_pred_zero(micro, lr_adamw):
    key = 'gate1_pred_zero_reduction'
    out = G.loadj(JP)
    if out.get(key, {}).get('pass'):
        print(f"{key}: already passed -- skip", flush=True)
        return
    ref = E1R.make_e1().eval().float()
    m = make_pred(pred_on=False).eval().float()
    pr = dict(ref.named_parameters())
    pdiff = max(float((p - pr[nm]).abs().max())
                for nm, p in m.named_parameters() if not nm.startswith('pred_'))
    pred_max = max(float(p.abs().max()) for nm, p in m.named_parameters()
                   if nm.startswith('pred_'))
    idx = E.OLD_HELD[:2, :Q.T]
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.no_grad():
            o_ref = ref(idx)
            d_off = float((m(idx) - o_ref).abs().max())
            m.pred_on = True
            # terms are all zero, so ON must ALSO reduce exactly at init
            d_on_zero = float((m(idx) - o_ref).abs().max())
            m.pred_on = False
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    del ref, o_ref
    torch.cuda.empty_cache()
    parent = three_step_scale(E1R.make_e1, micro, lr_adamw)
    mine = three_step_scale(lambda: make_pred(pred_on=False), micro, lr_adamw)
    sd = max(abs(a - b) for a, b in zip(parent['per_step_ce'],
                                        mine['per_step_ce']))
    hd = abs(parent['held100_ce'] - mine['held100_ce'])
    rec = {'shared_param_identity_max_abs_diff': pdiff,
           'predicate_param_max_abs_at_init': pred_max,
           'forward_pred_off_max_logit_diff': d_off,
           'forward_pred_on_but_zero_max_logit_diff': d_on_zero,
           'train3_parent_per_step_ce': parent['per_step_ce'],
           'train3_pred_off_per_step_ce': mine['per_step_ce'],
           'train3_max_per_step_abs_diff': sd,
           'train3_held100_abs_diff': hd,
           'note': '3-step identity runs through the SCALE trainer, so it '
                   'also proves the muon_exclude routing addition to '
                   'qk_s_muon_run.muon_params_split is inert when a model '
                   'declares nothing to exclude',
           'pass': bool(pdiff == 0.0 and pred_max == 0.0 and d_off == 0.0
                        and d_on_zero == 0.0 and sd == 0.0 and hd < 1e-6)}
    G.savej(JP, {**G.loadj(JP), key: rec})
    print(f"{key}: shared params {pdiff:.1e}, predicate params {pred_max:.1e}, "
          f"forward off {d_off:.1e} / on-but-zero {d_on_zero:.1e}, 3-step "
          f"{sd:.1e}/{hd:.1e} -> {'PASS' if rec['pass'] else 'FAIL'}",
          flush=True)
    assert rec['pass'], f'{key} FAILED'
    del m
    torch.cuda.empty_cache()


def gate_kernels():
    """Our MATCH kernels must equal qk_e21_census_run.build_feats' features."""
    key = 'gate2_kernels_vs_e21'
    out = G.loadj(JP)
    if out.get(key, {}).get('pass'):
        print(f"{key}: already passed -- skip", flush=True)
        return
    dev_saved, tf32 = E.DEV, torch.backends.cuda.matmul.allow_tf32
    import qk_e21_census_run as E21
    E.DEV = dev_saved
    E21.DEV = dev_saved
    torch.backends.cuda.matmul.allow_tf32 = tf32
    if getattr(E21, 'KCLS', None) is None:
        E21.KCLS = E21.token_classes()
    idx = E.OLD_HELD[:4, :Q.T].to(E.DEV)
    maskf = torch.tril(torch.ones(Q.T, Q.T, device=E.DEV))
    kprev, ksame = match_kernels(idx, maskf)
    feats = E21.build_feats(idx, maskf)
    # build_feats returns (B, NF, T, T): the FEATURE axis is 1, not 0.
    # feats[0] would be batch row 0 (shape (NF, T, T)) -- the mismatch that
    # crashed the first launch.
    assert feats.shape[0] == idx.shape[0] and feats.ndim == 4, feats.shape
    d_same = float((ksame - feats[:, 0]).abs().max())
    d_prev = float((kprev - feats[:, 1]).abs().max())
    rec = {'match_same_vs_e21_feat0_max_abs_diff': d_same,
           'match_prev_vs_e21_feat1_max_abs_diff': d_prev,
           'n_rows': int(idx.shape[0]),
           'pass': bool(d_same == 0.0 and d_prev == 0.0)}
    G.savej(JP, {**G.loadj(JP), key: rec})
    print(f"{key}: MATCH_same {d_same:.1e}, MATCH_prev {d_prev:.1e} -> "
          f"{'PASS' if rec['pass'] else 'FAIL'}", flush=True)
    assert rec['pass'], f'{key} FAILED'


def gate_terms_live():
    """A nonzero b_h must move the forward, and d(pattern)/d(b_h) must equal
    the MATCH_prev kernel EXACTLY (the named terms are additive on the
    bilinear product -- E28's rule)."""
    key = 'gate3_terms_live'
    out = G.loadj(JP)
    if out.get(key, {}).get('pass'):
        print(f"{key}: already passed -- skip", flush=True)
        return
    m = make_pred(pred_on=True).eval().float()
    idx = E.OLD_HELD[:2, :Q.T]
    with torch.no_grad():
        base = m(idx)
        m.pred_b[3, 2] = 0.05
        moved = float((m(idx) - base).abs().max())
        m.pred_b[3, 2] = 0.0
    # analytic derivative check on block 3, head 2
    maskf = torch.tril(torch.ones(Q.T, Q.T, device=E.DEV))
    kprev, ksame = match_kernels(idx, maskf)
    with torch.no_grad():
        t0 = m.pred_terms(3, kprev, ksame, maskf, Q.T)[:, 2]
        m.pred_b[3, 2] = 1.0
        t1 = m.pred_terms(3, kprev, ksame, maskf, Q.T)[:, 2]
        m.pred_b[3, 2] = 0.0
    d_deriv = float(((t1 - t0) - kprev).abs().max())
    rec = {'logit_shift_from_b_0.05': moved,
           'd_pattern_d_b_minus_matchprev_max_abs': d_deriv,
           'pass': bool(moved > 1e-4 and d_deriv == 0.0)}
    G.savej(JP, {**G.loadj(JP), key: rec})
    print(f"{key}: b=0.05 moves logits by {moved:.4f}, "
          f"d(pattern)/db - MATCH_prev = {d_deriv:.1e} -> "
          f"{'PASS' if rec['pass'] else 'FAIL'}", flush=True)
    assert rec['pass'], f'{key} FAILED'
    del m
    torch.cuda.empty_cache()


def register_predictions():
    out = G.loadj(JP)
    if 'registered_predictions' in out:
        return
    out['registered_predictions'] = {
        'registered': 'before training',
        'parent': 'combo3e5loss', 'parent_scale_held_ce': 4.105955,
        'parent_f34k': 4.155002, 'muonvanilla_scale': 3.964506,
        'w264_basis': {
            'predicate_basis_ce': 4.9000, 'predicate_basis_sd_n3': 0.0068,
            'its_parent_frontier': 4.9858, 'recipe': 5.0454,
            'note': 'the w264 predicate arm sits on the BANDWIDTH '
                    'architecture; this one sits on the RECIPE, per local'},
        'primary': (
            'HOLDS if this beats combo3e5loss on scale held CE by more than '
            'the seed noise floor (0.0127) plus the paired SE. Full transfer '
            'of the w264 predicate-minus-its-parent gap (-0.0858) on top of '
            'the recipe would land near 4.020; even partial transfer should '
            'clear 4.09.'),
        'cross_check': (
            'our own bw3e5 measured -0.05478 vs the same parent. If naming '
            'beats that, naming outranks bandwidth at scale and the retrain '
            'recipe should carry named terms; if it lands between 4.051 and '
            '4.106, both help and the composition question becomes the next '
            'experiment; if it does not beat the recipe at all, naming is a '
            'w264/bandwidth-architecture effect and does not transfer.'),
        'absorption': (
            'the census follow-up asks whether naming still ABSORBS at '
            '48-dim slots: predict residual MATCH_prev cos^2 collapses as it '
            'did at w264 (0.5036 -> 0.0951) and the full-pattern programmatic '
            'head count rises well above the parent census baseline'),
        'measurement_note': (
            'per local E29/E30 this arm will be ranked on CE and causal '
            'mechanism tests, NOT on wiring Spearman: readability differences '
            'between leading arms do not survive seeds (recipe sd 0.076) or '
            'interaction-aware causal targets (0.858 -> 0.343)')}
    G.savej(JP, out)
    print("registered predictions written before training", flush=True)


def pair_results():
    import numpy as np
    out = G.loadj(JP)
    pairs = {}
    for ctl, label in ((CONTROL_STEM, 'combo3e5loss'),
                       ('qk_s_w1152_bw3e5', 'bw3e5'),
                       ('qk_s_w1152_muonvanilla', 'muonvanilla')):
        for which in ('heldloss', 'f34kloss'):
            fa = os.path.join(G.OUT_DIR, f'{STEM}_{which}.npy')
            fb = os.path.join(G.OUT_DIR, f'{ctl}_{which}.npy')
            if not (os.path.exists(fa) and os.path.exists(fb)):
                continue
            la, lb = np.load(fa), np.load(fb)
            if la.shape != lb.shape:
                continue
            dd = la - lb
            ds = dd.reshape(G.HELD_N, -1).mean(1)
            pairs[f'{which}_minus_{label}'] = {
                'arm_ce': float(la.mean()), 'control_ce': float(lb.mean()),
                'delta': float(dd.mean()),
                'se_token': float(dd.std(ddof=1) / math.sqrt(len(dd))),
                'se_seq': float(ds.std(ddof=1) / math.sqrt(len(ds)))}
    out['paired'] = pairs
    G.savej(JP, out)
    for k, v in pairs.items():
        print(f"  {k}: {v['delta']:+.5f} +/- {v['se_seq']:.5f} (seq SE)",
              flush=True)


def main():
    W2.patch_width(G.WIDTH)
    M.ARM = ARM
    M.CFG = dict(stem=STEM, coeff=COEFF, prox=None, sweep=False)
    M.COEFF = COEFF
    M.PROX = None
    M.STEM = STEM
    M.JP = JP
    M.factory = factory

    total_steps, spec, f34k_held = G.setup_data()
    lr_adamw, lr_src = M.resolve_lr_adamw()
    out = G.loadj(JP)
    out['data'] = spec
    out['env'] = {'gpu': torch.cuda.get_device_name(0),
                  'torch': torch.__version__, 'cooc_substitute': True}
    out['arm_description'] = {
        'name': 'predicate_basis_on_recipe_w1152',
        'architecture': 'combo3e5loss recipe (E1Route: masked full decoders, '
                        'per-slot RMSNorm, 24 x 48-dim slots) + per-head '
                        'named pattern terms (signed positional profile + '
                        'b*MATCH_prev + c*MATCH_same), predicate params on '
                        'AdamW-no-decay via muon_exclude',
        'group_coeff': COEFF, 'matched_control': 'combo3e5loss',
        'seed': Q.SEED, 'data_seed': Q.DATA_SEED,
        'data_order': 'epoch_order(0)'}
    G.savej(JP, out)

    micro = M.preflight(G.loadj(JP), lr_adamw)
    print(f"micro {micro} (accum {G.EFF_BATCH // micro}), lr_adamw "
          f"{lr_adamw} from {lr_src}", flush=True)

    gate_pred_zero(micro, lr_adamw)
    gate_kernels()
    gate_terms_live()
    register_predictions()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    if os.path.exists(os.path.join(G.OUT_DIR, f'{STEM}.pt')) \
            and 'run' in G.loadj(JP):
        print(f"{STEM}.pt exists -- done", flush=True)
        pair_results()
        return
    print(f"==== training {STEM} ({total_steps} steps) ====", flush=True)
    log = M.train_muon_run(0.02, lr_adamw, total_steps, micro,
                           save_stem=STEM, f34k_held=f34k_held)
    out = G.loadj(JP)
    out['run'] = {'lr_muon': 0.02, 'lr_adamw': lr_adamw,
                  'held_ce_scale_bf16': log.get('final_held_ce'),
                  'held_ce_f34k_bf16': log.get('final_f34k_ce'),
                  'spikes': log['spikes'], 'diverged': log['diverged'],
                  'peak_mem_mib': log.get('peak_mem_mib'),
                  'sec_per_step': log.get('sec_per_step_measured'),
                  'train_curve_every200': log['train_loss'],
                  'held100_scale_curve': log['held_ce']}
    G.savej(JP, out)
    pair_results()


if __name__ == '__main__':
    main()
