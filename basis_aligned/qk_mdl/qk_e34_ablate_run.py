"""E34 NAMED-TERM VALUE: does the predicate library help an UNCONSTRAINED model,
and what is each named ingredient worth?  (training; results -> qk_e34.json)

WHY.  The program leader is PREDICATE-BASIS attention (qk_e22_predbasis_run arm
E22a): the bandwidth architecture make_e15c(s=15) + in-loss group-lasso 1e-4 +
per-head NAMED pattern terms -- a signed positional profile P_h over offsets, a
MATCH_prev coefficient b_h and a MATCH_same coefficient c_h, all added to the
bilinear pattern, all on AdamW.  Fresh held CE 4.9000 +- 0.0068 (n = 3,
qk_e29.json) against its own base's 4.9858 +- 0.0081 (n = 4, the E19a frontier
arm).  The named terms are worth about -0.086 nats ON TOP of a slotted, lassoed,
bandwidth-reinvested model.  Two questions follow, and this run answers both.

ARM 1 (E34a) -- IS THIS A GENERAL ARCHITECTURE IMPROVEMENT?  Add the IDENTICAL
named terms to the VANILLA width-264 architecture (E.make_e0a: sum kernel, full
visibility, NO write slots, NO per-slot RMSNorm, NO lasso), trained with Muon
exactly as the vanilla Muon control E0a-muon was.  If the same three named
ingredients buy a comparable CE gain with no interpretability constraint in
sight, "give attention heads named pattern ingredients" is a general modelling
result; if the gain collapses, the named terms are an INTERPRETABILITY-TAX
REDUCER that only pays under constraint -- they hand back structure the
constraints took away.  Paired against BOTH vanilla controls: E0a-muon
(qk_e0a_muon264, CE 4.7570) is the matched-optimizer parent and the primary
comparison, and E0a (AdamW, CE 4.8513) is reported for continuity.

ARMS 2-4 (E34b/c/d) -- WHAT IS EACH NAMED INGREDIENT WORTH?  Retrain the leader
three times, each time DROPPING one named term (everything else identical to
qk_e22_predbasis_run, same seed, same data order, same lasso, same trainer):
  E34b  profile only                      (no MATCH_prev, no MATCH_same)
  E34c  profile + MATCH_prev              (no MATCH_same)
  E34d  MATCH_prev + MATCH_same           (NO positional profile)
With the full arm E22a and the term-free base E19a already on disk, the four
marginal costs are read straight off:
  drop MATCH_same   = CE(E34c) - CE(E22a)
  drop MATCH_prev   = CE(E34b) - CE(E34c)
  drop the profile  = CE(E34d) - CE(E22a)
  drop all three    = CE(E19a) - CE(E22a)
A dropped term is implemented by REMOVING IT FROM THE FORWARD, so its parameter
never receives a gradient and stays exactly zero -- asserted after training.

INDUCTION IS MEASURED FOR EVERY ARM, not just CE.  E28 showed the match family
supplies 77% of the model's induction advantage (2.079 -> 0.485 when every b_h
is zeroed) while costing little CE, so CE alone cannot price MATCH_prev.  The
repeated-prefix probe is qk_e28_composed_sign's construction verbatim: 96 real
64-token held prefixes (fresh34k[33000:33096]) each repeated once, advantage =
CE(first copy) - CE(second copy) on IDENTICAL target tokens.  Control: the probe
reproduces the stored E28/E32 clean advantage for E22a (2.0786) to 0.02.

POSITIVE CONTROLS, BEFORE TRAINING (hard asserts, the program rule):
  1. PRED-ZERO per arm: with its predicate terms disabled, each arm reduces to
     its OWN base BIT-EXACTLY -- shared-parameter identity, forward max |logit
     diff| == 0 (terms off AND on at init, since every predicate parameter is an
     exact zero), and a 3-step run of this runner's trainer reproduces the base
     trainer's 3-step trajectory exactly.  Base = qk_e15_reinvest_run.make_e15c
     (s=15) for E34b/c/d, qk_e_common.make_e0a for E34a.
  2. KERNELS: this runner's MATCH_prev / MATCH_same kernels (imported function
     objects from qk_e22_predbasis_run) match qk_e21_census_run.build_feats
     features 1 / 0 on a batch EXACTLY (max abs diff == 0).
  3. STORED-PARENT STEP-0: the 3-step base runs reproduce the stored step-0
     training CE of E0a-muon (10.8694, arm 1) and of E19a (10.9210, arms 2-4) at
     the stored 4-dp resolution -- the arms really are trained on the parents'
     configuration and data order.
  4. AFTER TRAINING per arm: every DISABLED predicate parameter is exactly 0.0
     and every ENABLED one is not -- the ablation held and the term trained.

REGISTERED PREDICTIONS (merged before any training):
  (i)   ARM 1: the named terms help the vanilla model LESS than they help the
        constrained one.  Point prediction -0.02 nats, registered interval
        (-0.05, 0.00] vs E0a-muon; a paired gain of -0.05 or better would make
        this a general modelling result worth reporting on its own, and a gain
        at or beyond the constrained -0.086 would refute the tax-reducer story.
  (ii)  the POSITIONAL PROFILE carries most of the CE gain: dropping it (E34d)
        costs more than dropping MATCH_prev and more than dropping MATCH_same.
        (E22a mass shares: median 68% profile vs 7% MATCH_prev vs 0.5%
        MATCH_same.)
  (iii) dropping MATCH_same costs < 0.01 nats (its total |c| mass was 10.4
        against |b|'s 52.7).
  (iv)  dropping MATCH_prev costs INDUCTION massively even if CE barely moves:
        the profile-only arm E34b lands below 0.8 nats of induction advantage
        (against the full arm's 2.079) while its CE stays within +0.05.
Read every single-arm CE difference against the predicate-basis arm's
seed spread, sample sd 0.0068 over 3 seeds: differences below ~0.014 nats are
NOT resolved by one seed.

MEASURES per arm: paired fresh held CE with sequence-clustered SEs against every
relevant parent; induction advantage on the repeated-prefix probe; conditional
CE and induction with the named terms deleted / the bilinear residual deleted
(E32's decomposition, recomputed for each ablated library); mixture-weight
tables (b_h, c_h, profile norms, per-head pattern-mass shares of each named
term); the standard wiring probes (REPORTED, NOT RANKED ON -- per
BRAINSTORM_STATE's foundations correction the readability axis needs 3+ seeds);
per-sequence heldloss npy; wiring-trajectory npz; old-cooc held record.
The vanilla arm gets the CAUSAL consumption half of the probe only: its writes
are not partitioned, so weight support on an arbitrary column partition -- and
therefore the wiring Spearman -- is undefined for it.

Idempotent on qk_e34.json keys and on checkpoints; smoke-gated via QK_SMOKE=1.
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json
import math
import time

import numpy as np

import qk_e_common as E
from qk_e_common import Q, V8T, C, DEPTH, F, nn, torch
import qk_deeproute_train as R
import qk_deeproute_train_2 as R2
import qk_e7_evenout_run as E7R
import qk_e15_reinvest_run as E15R           # E15cRoute + make_e15c + solve_slot_c
import qk_e22_predbasis_run as E22R          # E22Route + match_kernels + trainer
import qk_e18_probe_upgrades as E18U         # generalized probe + cov-composed
import qk_e17_composed_wiring as E17         # agreement

# qk_e17 (via qk_e18) sets E.DEV='cpu' for its weight-only use -- restore.
E.DEV = 'cpu' if E.SMOKE else 'cuda'

JP = E.jpath('qk_e34.json')
GC_PRED = E22R.GC22                          # 1e-4, the E19a/E22a lasso
GC_VAN = 0.0                                 # vanilla carries no lasso
GATE_TOL = 1e-3
NG = E.NGROUP
ALL_TERMS = ('profile', 'match_prev', 'match_same')
ROWS0 = 33000                                # the fixed E21/E28/E31 audit slice
P_IND = 8 if E.SMOKE else 64                 # induction prefix length
N_IND = 8 if E.SMOKE else 96                 # induction probe rows
N_AUDIT = 16 if E.SMOKE else 200             # conditional-CE audit rows

# stored references (all read back from the JSONs at runtime; these are the
# values the registered predictions were written against)
E28_CLEAN_ADV = 2.0786                       # qk_e28.json clean adv_matched
E28_ALL_B_ZERO_ADV = 0.4853                  # every b_h zeroed
PRED_SEED_SD = 0.0068                        # predicate-basis CE sd, n=3 (E29)
CONSTRAINED_GAIN = -0.086                    # E22a mean - E19a mean (E29)

_E21 = None


def get_e21():
    """Lazy import of qk_e21_census_run (module body polls the GPU and sets
    global state); E.DEV and the tf32 flags are snapshotted and restored."""
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
        _E21 = E21
    return _E21


# ============================ architectures =================================
def _named_terms(model, l, Kprev, Ksame, maskf, Tq):
    """The enabled named pattern terms for block l, summed.  Identical algebra
    to qk_e22_predbasis_run.E22Route.pred_terms with the disabled terms simply
    ABSENT from the graph (so their parameters never get a gradient)."""
    out = None
    if 'profile' in model.terms:
        out = (model.pred_prof[l][:, model.offmat[:Tq, :Tq]] * maskf)[None]
    if 'match_prev' in model.terms:
        t = model.pred_b[l].view(1, -1, 1, 1) * Kprev[:, None]
        out = t if out is None else out + t
    if 'match_same' in model.terms:
        t = model.pred_c[l].view(1, -1, 1, 1) * Ksame[:, None]
        out = t if out is None else out + t
    if out is None:                          # no named terms at all
        out = torch.zeros(1, model.NH_pred, Tq, Tq, device=maskf.device,
                          dtype=maskf.dtype)
    return out


class E34PredRoute(E22R.E22Route):
    """The predicate-basis architecture with a SUBSET of the named terms.
    Everything else -- parameters, init, forward, trainer -- is E22Route."""

    def __init__(self, variant, depth, s, Dc, NH, HD, hidden, terms):
        super().__init__(variant, depth, s, Dc, NH, HD, hidden)
        self.terms = tuple(terms)
        self.NH_pred = NH

    def pred_terms(self, l, Kprev, Ksame, maskf, Tq):
        return _named_terms(self, l, Kprev, Ksame, maskf, Tq)


class E34VanRoute(V8T.V8Route):
    """The VANILLA width-264 architecture (E.make_e0a: sum kernel, full
    visibility, no write slots, no per-slot norm) plus the identical per-head
    named pattern terms, all init 0 so the model is bit-exactly vanilla at
    init.  Predicate parameters go to AdamW via muon_exclude, as in E22."""

    def __init__(self, variant, depth, terms):
        super().__init__(variant, depth)
        self.terms = tuple(terms)
        self.NH_pred = V8T.NH
        self.pred_on = True
        self.muon_exclude = ('pred_',)
        # zeros consume no RNG -> the shared parameters stay bit-identical
        self.pred_prof = nn.Parameter(torch.zeros(depth, V8T.NH, Q.T))
        self.pred_b = nn.Parameter(torch.zeros(depth, V8T.NH))
        self.pred_c = nn.Parameter(torch.zeros(depth, V8T.NH))
        ar = torch.arange(Q.T)
        self.register_buffer('offmat',
                             (ar[:, None] - ar[None, :]).clamp(min=0))

    def pred_terms(self, l, Kprev, Ksame, maskf, Tq):
        return _named_terms(self, l, Kprev, Ksame, maskf, Tq)

    def forward(self, idx, collect=None, sub_entry=None, entry_override=None,
                mlp_sub=None, coef_out=None, attn_sub=None, census_cb=None,
                pat_hook=None):
        """qk_v8_train.V8Route.forward verbatim + the named pattern terms."""
        D, NH, HD = V8T.D, V8T.NH, V8T.HD
        B, Tq = idx.shape
        e = F.rms_norm(self.wte(idx), (D,))
        streams = [e]
        cos = self.cos[None, :Tq, None, :]
        sin = self.sin[None, :Tq, None, :]
        mask = self.mask[:Tq, :Tq]
        maskf = mask.float()
        if self.pred_on:
            Kprev, Ksame = E22R.match_kernels(idx, maskf)

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
            hn = F.rms_norm(x, (D,))

            def qk(lin):
                z = lin(hn).view(B, Tq, NH, HD)
                return Q.apply_rot(F.rms_norm(z, (HD,)), cos, sin)

            q, k = qk(blk.c_q), qk(blk.c_k)
            q2, k2 = qk(blk.c_q2), qk(blk.c_k2)
            v = blk.c_v(hn).view(B, Tq, NH, HD)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
            pat = (s1 * s2).masked_fill(~mask, 0.0)
            if census_cb is not None:
                census_cb(l, pat)                 # RESIDUAL bilinear pattern
            if self.pred_on:
                pat = pat + self.pred_terms(l, Kprev, Ksame, maskf,
                                            Tq).to(pat.dtype)
            if pat_hook is not None:
                pat = pat_hook(l, pat)
            y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, Tq, D)
            aw = blk.c_proj(y)
            if self.proj:
                aw = aw * self.wmask[2 * l].to(aw.dtype)
            if attn_sub is not None and l in attn_sub:
                aw = attn_sub[l]
            x = x + aw
            if mlp_sub is not None and l in mlp_sub:
                mw = mlp_sub[l]
            else:
                xn = F.rms_norm(x, (D,))
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
        x = F.rms_norm(x, (D,))
        logits = x @ self.wte.weight.t()
        return 30 * torch.tanh(logits / 30)


R.KERNEL['E34a'] = 'sum'                     # same config as E0a (control 1)


def make_pred(terms, s, pred_on=True, variant='E34p'):
    hidden = 4 * Q.D
    torch.manual_seed(Q.SEED)
    m = E34PredRoute(variant, DEPTH, s, Q.D, Q.NH, Q.HD, hidden,
                     terms).to(E.DEV)
    m.pred_on = pred_on
    return m


def make_van(terms, pred_on=True):
    torch.manual_seed(Q.SEED)
    m = E34VanRoute('E34a', DEPTH, terms).to(E.DEV)
    m.pred_on = pred_on
    return m


def trainer(lr, gc, steps, **kw):
    """EXACTLY the E19a/E22a trainer expression."""
    return E.train_muon(lr, gc, steps, lr_adamw=E.get_lr(), **kw)


# ============================ arm table =====================================
def arms(s_c):
    return [
        dict(tag='E34a', stem='qk_e34_a_vanilla_named', kind='vanilla',
             terms=ALL_TERMS, gc=GC_VAN, dims=None, slot=None,
             factory=(lambda: make_van(ALL_TERMS)),
             base_factory=E.make_e0a, base_name='vanilla (E.make_e0a)',
             parent_stem='qk_e0a_muon264', parent_label='e0a_muon',
             parent_json='qk_e0m.json', parent_key='E0a_muon',
             extra_parents=[('qk_e0a_vanilla264', 'e0a_adamw')],
             design='VANILLA width-264 architecture (sum kernel, full '
                    'visibility, no write slots, no per-slot RMSNorm, no '
                    'lasso) + the identical per-head named pattern terms '
                    '(signed positional profile + b_h MATCH_prev + c_h '
                    'MATCH_same, init 0, AdamW via muon_exclude), Muon 0.02'),
        dict(tag='E34b', stem='qk_e34_b_profile_only', kind='pred',
             terms=('profile',), gc=GC_PRED, dims=[s_c] * NG, slot=s_c,
             factory=(lambda: make_pred(('profile',), s_c, variant='E34b')),
             base_factory=(lambda: E15R.make_e15c(s=s_c)),
             base_name='E19a frontier base (make_e15c s=15)',
             parent_stem='qk_e22_a', parent_label='e22a',
             parent_json='qk_e19.json', parent_key='E19a',
             extra_parents=[('qk_e19_a', 'e19a')],
             design='predicate-basis attention with the POSITIONAL PROFILE '
                    'ONLY (no MATCH_prev, no MATCH_same); everything else '
                    'verbatim qk_e22_predbasis_run'),
        dict(tag='E34c', stem='qk_e34_c_profile_prev', kind='pred',
             terms=('profile', 'match_prev'), gc=GC_PRED, dims=[s_c] * NG,
             slot=s_c,
             factory=(lambda: make_pred(('profile', 'match_prev'), s_c,
                                        variant='E34c')),
             base_factory=(lambda: E15R.make_e15c(s=s_c)),
             base_name='E19a frontier base (make_e15c s=15)',
             parent_stem='qk_e22_a', parent_label='e22a',
             parent_json='qk_e19.json', parent_key='E19a',
             extra_parents=[('qk_e19_a', 'e19a')],
             design='predicate-basis attention with the positional profile + '
                    'MATCH_prev (no MATCH_same); everything else verbatim '
                    'qk_e22_predbasis_run'),
        dict(tag='E34d', stem='qk_e34_d_prev_same', kind='pred',
             terms=('match_prev', 'match_same'), gc=GC_PRED, dims=[s_c] * NG,
             slot=s_c,
             factory=(lambda: make_pred(('match_prev', 'match_same'), s_c,
                                        variant='E34d')),
             base_factory=(lambda: E15R.make_e15c(s=s_c)),
             base_name='E19a frontier base (make_e15c s=15)',
             parent_stem='qk_e22_a', parent_label='e22a',
             parent_json='qk_e19.json', parent_key='E19a',
             extra_parents=[('qk_e19_a', 'e19a')],
             design='predicate-basis attention with MATCH_prev + MATCH_same '
                    'and NO positional profile; everything else verbatim '
                    'qk_e22_predbasis_run'),
    ]


# ============================ controls ======================================
def control_inherited_gates():
    key = 'control0_inherited_gates'
    if E.SMOKE or (E.loadj(JP).get(key) or {}).get('pass'):
        return
    e18 = E.loadj(E.jpath('qk_e18.json'))
    g1 = e18.get('gate1_uniform11_weight_support', {}).get('pass')
    g2 = e18.get('gate2_cov_composed_E9a', {}).get('pass')
    assert g1 and g2, ('qk_e18.json probe gates 1+2 not passed -- the reused '
                       'probe functions are not validated', g1, g2)
    E.merge(JP, key, {'qk_e18_gate1_weight_support': bool(g1),
                      'qk_e18_gate2_cov_composed': bool(g2),
                      'note': 'this runner calls the SAME probe function '
                              'objects (qk_e18_probe_upgrades), so the stored '
                              'gate results apply verbatim',
                      'pass': True})
    print(f'{key}: inherited probe gates PASS', flush=True)


def control_kernels():
    key = 'control2_kernels_vs_e21'
    if (E.loadj(JP).get(key) or {}).get('pass'):
        print(f'{key}: already passed -- skip', flush=True)
        return
    E21 = get_e21()
    if E21.KCLS is None:
        E21.KCLS = E21.token_classes()
    idx = Q.HELD[:4, :Q.T]
    maskf = torch.tril(torch.ones(Q.T, Q.T, dtype=torch.bool,
                                  device=idx.device)).float()
    Fs = E21.build_feats(idx, maskf)
    kp, ks = E22R.match_kernels(idx, maskf)
    d_prev = float((kp - Fs[:, 1]).abs().max())
    d_same = float((ks - Fs[:, 0]).abs().max())
    rec = {'batch': int(idx.shape[0]), 'T': Q.T,
           'match_prev_max_abs_diff': d_prev,
           'match_same_max_abs_diff': d_same,
           'note': 'the kernels used here are the IMPORTED function object '
                   'qk_e22_predbasis_run.match_kernels, checked against '
                   'qk_e21_census_run.build_feats features 1 (MATCH_prev) '
                   'and 0 (MATCH_same)',
           'pass': bool(d_prev == 0.0 and d_same == 0.0)}
    E.merge(JP, key, rec)
    print(f'{key}: prev {d_prev:.1e}, same {d_same:.1e} -> '
          f"{'PASS' if rec['pass'] else 'FAIL'}", flush=True)
    assert rec['pass'], f'{key} FAILED'


def three_step(factory, gc, steps=3):
    log = trainer(E7R.muon_lr(), gc, steps, log_every=1, save=False,
                  factory=factory)
    return {'per_step_ce': [x[1] for x in log['train_loss']],
            'held100_ce': log['final_held_ce']}


def control_pred_zero(a, stored_step0):
    """Control 1 + control 3 for one arm: with the predicate terms disabled the
    arm reduces to its own base bit-exactly (parameters, forward, and a 3-step
    training trajectory), and that base reproduces the stored parent step-0 CE.
    """
    key = f"control1_pred_zero_{a['tag']}"
    done = E.loadj(JP).get(key)
    if done and done.get('pass'):
        print(f'{key}: already passed -- skip', flush=True)
        return
    terms = a['terms']
    off = (make_van(terms, pred_on=False) if a['kind'] == 'vanilla'
           else make_pred(terms, a['slot'], pred_on=False,
                          variant=a['tag'])).eval().float()
    base = a['base_factory']().eval().float()
    pb = dict(base.named_parameters())
    pdiff = max(float((p.detach() - pb[nm].detach()).abs().max())
                for nm, p in off.named_parameters() if nm in pb)
    n_shared = sum(1 for nm, _ in off.named_parameters() if nm in pb)
    idx = E.OLD_HELD[:2, :Q.T]
    with torch.no_grad():
        d_off = float((off(idx) - base(idx)).abs().max())
        off.pred_on = True                        # every predicate param is 0
        d_on = float((off(idx) - base(idx)).abs().max())
        off.pred_on = False
    del off, base
    if not E.SMOKE:
        torch.cuda.empty_cache()
    fac_off = ((lambda: make_van(terms, pred_on=False)) if a['kind'] == 'vanilla'
               else (lambda: make_pred(terms, a['slot'], pred_on=False,
                                       variant=a['tag'])))
    parent = three_step(a['base_factory'], a['gc'])
    mine = three_step(fac_off, a['gc'])
    step_diff = max(abs(x - y) for x, y in
                    zip(parent['per_step_ce'], mine['per_step_ce']))
    held_diff = abs(parent['held100_ce'] - mine['held100_ce'])
    d_stored = (None if stored_step0 is None else
                abs(round(parent['per_step_ce'][0], 4) - stored_step0))
    rec = {'arm': a['tag'], 'base': a['base_name'], 'terms_enabled': list(terms),
           'n_shared_parameters': n_shared,
           'shared_param_identity_max_abs_diff': pdiff,
           'forward_pred_off_max_logit_diff': d_off,
           'forward_pred_on_at_init_max_logit_diff': d_on,
           'train3_base_per_step_ce': parent['per_step_ce'],
           'train3_predzero_per_step_ce': mine['per_step_ce'],
           'train3_max_per_step_abs_diff': step_diff,
           'train3_held100_abs_diff': held_diff,
           'stored_parent_step0_ce': stored_step0,
           'stored_parent_step0_abs_diff': d_stored,
           'note': 'same seed (the factory reseeds; predicate parameters are '
                   'zeros and consume no RNG), same epoch_order(0) data; '
                   'proves the ONLY difference from the base is the named '
                   'terms.  The stored-step-0 check (control 3) ties this '
                   'runner\'s training call to the parent record on disk.',
           'pass': bool(pdiff == 0.0 and d_off == 0.0 and d_on == 0.0
                        and step_diff == 0.0 and held_diff < 1e-6
                        and (d_stored is None or d_stored <= 1e-4))}
    E.merge(JP, key, rec)
    print(f'{key}: params {pdiff:.1e}, fwd off/on {d_off:.1e}/{d_on:.1e}, '
          f'3-step {step_diff:.1e}/{held_diff:.1e}, stored step0 '
          f'{d_stored} -> ' + ('PASS' if rec['pass'] else 'FAIL'), flush=True)
    assert rec['pass'], f'{key} FAILED'


def control_terms_zero(a):
    """Control 4 (after training): the DISABLED predicate parameters are still
    exactly zero and the ENABLED ones moved."""
    key = f"control4_disabled_terms_zero_{a['tag']}"
    if E.SMOKE or key in E.loadj(JP) or not os.path.exists(E.ckpath(a['stem'])):
        return
    sd = torch.load(E.ckpath(a['stem']), map_location='cpu',
                    weights_only=False)['state_dict']
    absmax = {'profile': float(sd['pred_prof'].abs().max()),
              'match_prev': float(sd['pred_b'].abs().max()),
              'match_same': float(sd['pred_c'].abs().max())}
    dis = [t for t in ALL_TERMS if t not in a['terms']]
    ok = (all(absmax[t] == 0.0 for t in dis)
          and all(absmax[t] > 0.0 for t in a['terms']))
    rec = {'arm': a['tag'], 'terms_enabled': list(a['terms']),
           'terms_disabled': dis, 'max_abs_parameter': absmax,
           'note': 'a disabled term is absent from the forward, so its '
                   'parameter receives no gradient and AdamW skips it; this '
                   'asserts the ablation actually held and that every enabled '
                   'term trained away from its zero init',
           'pass': bool(ok)}
    E.merge(JP, key, rec)
    print(f"{key}: enabled {list(a['terms'])} disabled {dis} max|param| "
          f"{absmax} -> " + ('PASS' if ok else 'FAIL'), flush=True)
    assert ok, f'{key} FAILED'


# ============================ induction probe ===============================
def build_probe():
    """qk_e28_composed_sign.build_probe verbatim: real held prefixes, each
    repeated once; matched windows so the two copies predict IDENTICAL tokens."""
    rows = np.asarray(np.load(E.HELD_PATH, mmap_mode='r')
                      [ROWS0:ROWS0 + N_IND]).astype(np.int64)
    pref = torch.from_numpy(rows[:, 1:1 + P_IND]).to(E.DEV)
    ev = torch.cat([pref, pref], 1)
    idx, tgt = ev[:, :-1], ev[:, 1:]
    fir = torch.arange(1, P_IND - 1, device=E.DEV)
    sec = torch.arange(P_IND + 1, 2 * P_IND - 1, device=E.DEV)
    assert torch.equal(tgt[:, fir], tgt[:, sec])
    return idx, tgt, fir, sec


def make_hook(m, x, mode):
    """pat_hook over the FULL pattern (qk_e32_residual_mine_run semantics).
    'named_only'    -> pattern = the named terms alone (residual deleted)
    'residual_only' -> pattern = the bilinear residual alone (names deleted)"""
    if mode == 'full':
        return None
    Tq = x.shape[1]
    maskf = torch.tril(torch.ones(Tq, Tq, dtype=torch.bool,
                                  device=x.device)).float()
    Kp, Ks = E22R.match_kernels(x, maskf)

    def hook(l, pat):
        terms = m.pred_terms(l, Kp, Ks, maskf, Tq).to(pat.dtype)
        if mode == 'named_only':
            return terms.expand_as(pat).clone()
        if mode == 'residual_only':
            return pat - terms
        raise ValueError(mode)
    return hook


@torch.no_grad()
def induction(m, probe, mode='full', bs=8):
    idx, tgt, fir, sec = probe
    ces = []
    for i in range(0, idx.shape[0], bs):
        x, t = idx[i:i + bs], tgt[i:i + bs]
        hk = make_hook(m, x, mode)
        lg = (m(x) if hk is None else m(x, pat_hook=hk)).float()
        lsm = torch.log_softmax(lg, -1)
        ces.append(-lsm.gather(-1, t[..., None])[..., 0])
    ce = torch.cat(ces)
    per_seq = ce[:, fir].mean(1) - ce[:, sec].mean(1)
    return {'ce_first': round(float(ce[:, fir].mean()), 4),
            'ce_second': round(float(ce[:, sec].mean()), 4),
            'induction_advantage': round(float(per_seq.mean()), 4),
            'se_seq': round(float(per_seq.std(unbiased=True)
                                  / math.sqrt(len(per_seq))), 4)}, per_seq


@torch.no_grad()
def audit_ce(m, mode='full', bs=4):
    """Per-sequence held CE on the fixed audit slice, optionally with the named
    terms or the bilinear residual deleted."""
    rows = np.asarray(np.load(E.HELD_PATH, mmap_mode='r')
                      [ROWS0:ROWS0 + N_AUDIT]).astype(np.int64)
    held = torch.from_numpy(rows).to(E.DEV)
    out = []
    for i in range(0, held.shape[0], bs):
        b = held[i:i + bs]
        x = b[:, :Q.T]
        hk = make_hook(m, x, mode)
        lg = (m(x) if hk is None else m(x, pat_hook=hk)).float()
        ce = F.cross_entropy(lg.reshape(-1, Q.V), b[:, 1:Q.T + 1].reshape(-1),
                             reduction='none')
        out.append(ce.view(x.shape[0], Q.T).mean(1).cpu())
    return torch.cat(out)


def _paired(a, b):
    d = a - b
    return (round(float(d.mean()), 5),
            round(float(d.std(unbiased=True) / math.sqrt(len(d))), 5))


def induction_record(key, stem, factory, probe, named=True, ref=None):
    """Clean induction advantage + (for predicate models) the named/residual
    decomposition of CE and induction.  In SMOKE the model is built untrained
    so the code path is exercised without a checkpoint."""
    if key in E.loadj(JP):
        print(f'{key}: cached', flush=True)
        return E.loadj(JP)[key]
    if E.SMOKE:
        m = factory().eval().float()
    else:
        if not os.path.exists(E.ckpath(stem)):
            return None
        m, _ = E.load_arm(stem, factory)
    t0 = time.time()
    modes = ('full', 'named_only', 'residual_only') if named else ('full',)
    ind, inds, ce = {}, {}, {}
    for mode in modes:
        ind[mode], inds[mode] = induction(m, probe, mode)
        ce[mode] = audit_ce(m, mode)
    rec = {'checkpoint': f'{stem}.pt',
           'probe': f'{N_IND} held prefixes of {P_IND} tokens from '
                    f'fresh34k[{ROWS0}:{ROWS0 + N_IND}], each repeated once; '
                    'advantage = CE(first copy) - CE(second copy) on '
                    'IDENTICAL targets (qk_e28_composed_sign construction)',
           'audit_slice': f'fresh34k[{ROWS0}:{ROWS0 + N_AUDIT}], T={Q.T}',
           'induction': ind,
           'audit_held_ce': {k: round(float(v.mean()), 4) for k, v in ce.items()},
           'runtime_s': round(time.time() - t0, 1)}
    if named and len(modes) == 3:
        d_res, se_res = _paired(ce['named_only'], ce['full'])
        d_nam, se_nam = _paired(ce['residual_only'], ce['full'])
        di_res, sei_res = _paired(inds['named_only'], inds['full'])
        di_nam, sei_nam = _paired(inds['residual_only'], inds['full'])
        rec.update({
            'delta_ce_residual_zeroed': d_res,
            'delta_ce_residual_zeroed_se': se_res,
            'delta_ce_named_zeroed': d_nam,
            'delta_ce_named_zeroed_se': se_nam,
            'delta_induction_residual_zeroed': di_res,
            'delta_induction_residual_zeroed_se': sei_res,
            'delta_induction_named_zeroed': di_nam,
            'delta_induction_named_zeroed_se': sei_nam,
            'named_share_of_ce_cost': round(d_nam / max(d_nam + d_res, 1e-9), 4),
            'note': 'residual zeroed = every head\'s pattern is exactly its '
                    'ENABLED named terms; named zeroed = every head\'s '
                    'pattern is exactly the learned bilinear residual'})
    if ref is not None:
        rec['reference_stored_full_advantage'] = ref
        rec['abs_diff_vs_stored'] = round(
            abs(ind['full']['induction_advantage'] - ref), 4)
    E.merge(JP, key, rec)
    print(f"{key}: induction advantage "
          f"{ind['full']['induction_advantage']} "
          f"(se {ind['full']['se_seq']}), audit CE "
          f"{rec['audit_held_ce']['full']} ({rec['runtime_s']}s)", flush=True)
    del m
    if not E.SMOKE:
        torch.cuda.empty_cache()
    return rec


def control_probe_reproduces_e28():
    """The probe must reproduce the stored E28/E32 clean induction advantage
    for the full predicate-basis arm E22a."""
    key = 'control5_probe_reproduces_e28'
    if E.SMOKE or (E.loadj(JP).get(key) or {}).get('pass'):
        return
    rec = E.loadj(JP).get('induction_E22a_full')
    if rec is None:
        return
    e32 = E.loadj(E.jpath('qk_e32.json')).get('causal_weight_s0', {})
    d28 = rec['abs_diff_vs_stored']
    out = {'e22a_full_advantage_here': rec['induction']['full'][
               'induction_advantage'],
           'stored_e28_clean_adv_matched': E28_CLEAN_ADV,
           'abs_diff_vs_e28': d28,
           'stored_e32_causal_weight_s0_induction': e32.get('induction'),
           'stored_e32_delta_ce_named_zeroed': e32.get('delta_ce_named_zeroed'),
           'this_run_delta_ce_named_zeroed': rec.get('delta_ce_named_zeroed'),
           'tolerance': 0.02,
           'note': 'E28 ran the probe on CPU fp32 through a per-head '
                   'decomposed forward; this runs the model forward on GPU. '
                   'Agreement to 0.02 nats validates the probe construction '
                   'and the named-term hooks reused for every arm.',
           'pass': bool(d28 <= 0.02)}
    E.merge(JP, key, out)
    print(f'{key}: {out["e22a_full_advantage_here"]} vs stored '
          f'{E28_CLEAN_ADV} (diff {d28}) -> '
          + ('PASS' if out['pass'] else 'FAIL'), flush=True)
    assert out['pass'], f'{key} FAILED'


# ============================ mixture tables ================================
def mixture_tables(a):
    key = f"mixture_weights_{a['tag']}"
    if key in E.loadj(JP):
        return
    if E.SMOKE:
        m = a['factory']().eval().float()
    else:
        if not os.path.exists(E.ckpath(a['stem'])):
            return
        m, _ = E.load_arm(a['stem'], a['factory'])
    b = m.pred_b.detach().float().cpu()
    c = m.pred_c.detach().float().cpu()
    prof = m.pred_prof.detach().float().cpu()
    NHp = b.shape[1]
    idx = Q.HELD[:4, :Q.T]
    maskf = torch.tril(torch.ones(Q.T, Q.T, dtype=torch.bool,
                                  device=idx.device)).float()
    resid, full = {}, {}
    with torch.no_grad():
        m(idx, census_cb=lambda l, p: resid.__setitem__(l, p.detach().clone()),
          pat_hook=lambda l, p: (full.__setitem__(l, p.detach().clone()) or p))
        Kprev, Ksame = E22R.match_kernels(idx, maskf)
        dev = idx.device
        decomp_max, shares = 0.0, {}
        for l in range(DEPTH):
            terms = {}
            if 'profile' in m.terms:
                terms['profile'] = (
                    m.pred_prof[l].to(dev)[:, m.offmat] * maskf
                ).unsqueeze(0).expand(idx.shape[0], -1, -1, -1)
            if 'match_prev' in m.terms:
                terms['match_prev'] = (m.pred_b[l].to(dev).view(1, -1, 1, 1)
                                       * Kprev[:, None])
            if 'match_same' in m.terms:
                terms['match_same'] = (m.pred_c[l].to(dev).view(1, -1, 1, 1)
                                       * Ksame[:, None])
            rec_full = resid[l] + (sum(terms.values()) if terms else 0.0)
            decomp_max = max(decomp_max,
                             float((rec_full - full[l]).abs().max()))
            tot = full[l].float().pow(2).sum((0, 2, 3)).clamp_min(1e-12)
            shares[l] = {nm: [round(float(x), 5) for x in
                              (t.float().pow(2).sum((0, 2, 3)) / tot)]
                         for nm, t in terms.items()}
            for nm in ALL_TERMS:
                shares[l].setdefault(nm, [0.0] * NHp)
            shares[l]['residual'] = [round(float(x), 5) for x in
                                     (resid[l].float().pow(2).sum((0, 2, 3))
                                      / tot)]
    del m, resid, full
    if not E.SMOKE:
        torch.cuda.empty_cache()
    table = []
    for l in range(DEPTH):
        for h in range(NHp):
            pl = prof[l, h]
            top = torch.argsort(pl.abs(), descending=True)[:3]
            table.append({
                'layer': l, 'head': h,
                'b_match_prev': round(float(b[l, h]), 5),
                'c_match_same': round(float(c[l, h]), 5),
                'profile_l2': round(float(pl.norm()), 5),
                'profile_top_offsets': [[int(o), round(float(pl[o]), 5)]
                                        for o in top],
                'mass_share_profile': shares[l]['profile'][h],
                'mass_share_match_prev': shares[l]['match_prev'][h],
                'mass_share_match_same': shares[l]['match_same'][h],
                'mass_share_residual': shares[l]['residual'][h]})
    med = {f'median_mass_share_{nm}':
           round(float(np.median([r[f'mass_share_{nm}'] for r in table])), 5)
           for nm in ('profile', 'match_prev', 'match_same', 'residual')}
    rec = {'checkpoint': f"{a['stem']}.pt", 'terms_enabled': list(a['terms']),
           'decomposition_check_max_abs_diff': decomp_max,
           'total_abs_b_mass': round(float(b.abs().sum()), 5),
           'total_abs_c_mass': round(float(c.abs().sum()), 5),
           'profile_l2_norm': round(float(prof.norm()), 5),
           'mass_shares_rows': 'fresh held [33000:33004]; shares = '
                               'sum-of-squares of each named term over the '
                               'full pattern, per head',
           'reference_E22a': {'total_abs_b_mass': 52.73384,
                              'total_abs_c_mass': 10.38871,
                              'median_mass_share_profile': 0.6766,
                              'median_mass_share_match_prev': 0.0721},
           'table': table}
    rec.update(med)
    assert decomp_max < 1e-4, ('pattern decomposition off', decomp_max)
    E.merge(JP, key, rec)
    print(f"{key}: |b| {rec['total_abs_b_mass']}, |c| "
          f"{rec['total_abs_c_mass']}, profile L2 {rec['profile_l2_norm']}, "
          f"median shares {med}, decomp {decomp_max:.1e}", flush=True)


# ============================ wiring probes =================================
def probes(a):
    """The generalized variable-slot-dim light probe + covariance-composed
    re-scoring (qk_e18 gates 1+2 validated), exactly the E22/E29 probe path.
    REPORTED, NOT RANKED ON: the readability axis needs 3+ seeds."""
    stem, tag, dims = a['stem'], a['tag'], a['dims']
    if E.SMOKE or not os.path.exists(E.ckpath(stem)):
        return
    j = E.loadj(JP)
    lp_key, cw_key = f'light_probe_{tag}_var_dims', f'composed_wiring_{tag}'
    if lp_key not in j:
        print(f'{tag} light probe ...', flush=True)
        m, _ = E.load_arm(stem, a['factory'])
        Ws = m.wte.weight.shape[1]
        base, dce = E18U.gen_consumption(m, Ws)
        totals = {str(jj): round(sum(dce[li].get(si, 0.0) for li in dce
                                     for si in (1 + 2 * jj, 2 + 2 * jj)
                                     if si in dce[li]), 5)
                  for jj in range(DEPTH)}
        srt = sorted([(li, si, dce[li][si]) for li in dce for si in dce[li]],
                     key=lambda p: -p[2])
        rec = {'checkpoint': f'{stem}.pt', 'stream_width': Ws,
               'compute_width': Q.D,
               'base_ce_fp32_abl_oldheld': round(base, 5),
               'per_block_total_consumption': totals,
               'consumption_top20': [
                   {'consumer': ('readout' if li == DEPTH else f'block{li}'),
                    'source': R2.stream_name(si), 'dce': round(v, 5)}
                   for li, si, v in srt[:20]],
               'consumption_matrix': {str(li): {str(si): round(v, 6)
                                                for si, v in dce[li].items()}
                                      for li in dce}}
        if dims is None:
            rec['slot_dims'] = None
            rec['wiring_spearman_all'] = None
            rec['note'] = ('CAUSAL half only.  This arm is the VANILLA '
                           'architecture: module writes are not partitioned '
                           'into slots, so read-weight support on any column '
                           'partition does not correspond to a source module '
                           'and the wiring Spearman is UNDEFINED for it -- '
                           'reporting one would be a category error.')
        else:
            wp = E18U.wpairs(m, dims)
            sup = E18U.score(E18U.gen_gram_table(m, dims), wp)
            cau = [dce[li][si] for li, si in wp]
            eff = [k for k in range(len(wp)) if cau[k] > C.EFFECTUAL]
            agr = E17.agreement(sup, cau, eff)
            rec.update({
                'slot_dims': dims,
                'wiring_n_pairs': len(wp),
                'wiring_spearman_all': agr['spearman_all'],
                'wiring_n_effectual': len(eff),
                'wiring_spearman_effectual': agr['spearman_effectual'],
                'wiring_top10_precision': agr['top10_precision'],
                'weight_support_matrix': {
                    str(li): {str(si): round(sup[i], 3)
                              for i, (l2, si) in enumerate(wp) if l2 == li}
                    for li in range(DEPTH + 1)},
                'note': 'generalized variable-slot-dim probe (qk_e18 gate 1); '
                        'weight support covers the bilinear read matrices '
                        'only -- the named terms read tokens, not the stream; '
                        'causal = mean-ablation on old cooc held [:96], fp32'})
        E.merge(JP, lp_key, rec)
        print(f"{tag} wiring Spearman {rec.get('wiring_spearman_all')}",
              flush=True)
        del m
        torch.cuda.empty_cache()
    if dims is None:
        return
    j = E.loadj(JP)
    if cw_key not in j:
        lp = j[lp_key]
        print(f'{tag} covariance-composed re-scoring ...', flush=True)
        m, _ = E.load_arm(stem, a['factory'])
        wp = E18U.wpairs(m, dims)
        cau = E18U.stored_cau(lp, wp)
        tables, meta, _ = E18U.composed_tables(m, dims, cau, wp, E.DEV,
                                               remnant=False)
        chk = abs(tables['plain']['spearman_all'] - lp['wiring_spearman_all'])
        assert chk <= GATE_TOL, f'plain does not reproduce the probe ({chk})'
        rec = {'checkpoint': f'{stem}.pt', 'slot_dims_uniform': dims[0],
               'plain_reproduction_abs_diff': round(chk, 6), 'tables': tables}
        rec.update(meta)
        E.merge(JP, cw_key, rec)
        print(f"{tag} plain {tables['plain']['spearman_all']} -> cov "
              f"{tables['cov_composed']['spearman_all']}", flush=True)
        del m
        torch.cuda.empty_cache()


# ============================ pairing =======================================
def pair_all(a):
    """Paired per-token fresh held CE (sequence-clustered SE) against every
    relevant parent."""
    if E.SMOKE:
        return
    fa = f"{a['stem']}_heldloss.npy"
    if not os.path.exists(f'{E.QK}/{fa}'):
        return
    pairs = [(a['parent_stem'], a['parent_label'])] + list(a['extra_parents'])
    for pstem, lab in pairs:
        fb = f'{pstem}_heldloss.npy'
        k = f"{a['tag']}_minus_{lab}_fresh"
        if os.path.exists(f'{E.QK}/{fb}') and k not in E.loadj(JP):
            E.merge(JP, k, dict(E.paired(fa, fb, len(Q.HELD), lab),
                                parent=pstem,
                                note='paired per-token fresh held CE on '
                                     'fresh34k[33000:34500], '
                                     'sequence-clustered SE'))
    p = f"{E.QK}/{a['stem']}_heldloss.npy"
    q = f"{E.QK}/{a['stem']}_heldloss_seq.npy"
    if os.path.exists(p) and not os.path.exists(q):
        np.save(q, np.load(p).reshape(len(Q.HELD), Q.T).mean(1))
    tp = f"{E.QK}/{a['stem']}_traj.npz"
    if os.path.exists(tp) and f"{a['tag']}_traj" not in E.loadj(JP):
        z = np.load(tp)
        E.merge(JP, f"{a['tag']}_traj", {'path': os.path.basename(tp),
                                         'n_snapshots': int(len(z['step'])),
                                         'every': E.TRAJ_EVERY,
                                         'keys': sorted(z.files)})


def cross_pairs():
    """Every pairwise fresh-held delta among the term-library ladder."""
    if E.SMOKE:
        return
    ladder = [('full_E22a', 'qk_e22_a'),
              ('profile_prev_E34c', 'qk_e34_c_profile_prev'),
              ('profile_only_E34b', 'qk_e34_b_profile_only'),
              ('prev_same_E34d', 'qk_e34_d_prev_same'),
              ('none_E19a', 'qk_e19_a')]
    out = {}
    for i, (na, sa) in enumerate(ladder):
        for nb, sb in ladder[i + 1:]:
            fa, fb = f'{sa}_heldloss.npy', f'{sb}_heldloss.npy'
            if not (os.path.exists(f'{E.QK}/{fa}')
                    and os.path.exists(f'{E.QK}/{fb}')):
                continue
            out[f'{na}_minus_{nb}'] = E.paired(fa, fb, len(Q.HELD), nb)
    if out:
        E.merge(JP, 'ladder_paired_deltas', {
            'note': 'paired per-token fresh held CE differences with '
                    'sequence-clustered SEs across the whole named-term '
                    'ladder; marginal cost of a term = the delta between the '
                    'two arms that differ only in that term',
            'predicate_basis_seed_sd_n3': PRED_SEED_SD,
            'resolution_note': f'the predicate-basis arm\'s CE sample sd over '
                               f'3 seeds is {PRED_SEED_SD}, so single-seed '
                               f'differences below ~{2 * PRED_SEED_SD:.3f} '
                               f'nats are not resolved',
            'deltas': out})
    return out


# ============================ summary =======================================
def summarize(A):
    j = E.loadj(JP)

    def ce(tag):
        return (j.get(tag) or {}).get('final_held_ce_fresh_bf16')

    def pd(key, lab):
        r = j.get(key) or {}
        return (r.get(f'minus_{lab}'), r.get(f'minus_{lab}_se_seq'))

    def ind(key):
        r = (j.get(key) or {}).get('induction', {}).get('full', {})
        return r.get('induction_advantage'), r.get('se_seq')

    rows = {}
    for a in A:
        t = a['tag']
        d, se = pd(f"{t}_minus_{a['parent_label']}_fresh", a['parent_label'])
        adv, adv_se = ind(f'induction_{t}')
        rows[t] = {
            'stem': a['stem'], 'terms_enabled': list(a['terms']),
            'design': a['design'],
            'final_held_ce_fresh_bf16': ce(t),
            'diverged': (j.get(t) or {}).get('diverged'),
            f"paired_minus_{a['parent_label']}": d,
            f"paired_minus_{a['parent_label']}_se_seq": se,
            'induction_advantage': adv, 'induction_se_seq': adv_se,
            'wiring_spearman_plain':
                (j.get(f'light_probe_{t}_var_dims') or {}).get(
                    'wiring_spearman_all'),
            'wiring_spearman_cov_composed':
                ((j.get(f'composed_wiring_{t}') or {}).get('tables', {})
                 .get('cov_composed', {}).get('spearman_all')),
            'median_mass_share_profile':
                (j.get(f'mixture_weights_{t}') or {}).get(
                    'median_mass_share_profile'),
            'median_mass_share_match_prev':
                (j.get(f'mixture_weights_{t}') or {}).get(
                    'median_mass_share_match_prev')}
        for pstem, lab in a['extra_parents']:
            d2, se2 = pd(f'{t}_minus_{lab}_fresh', lab)
            rows[t][f'paired_minus_{lab}'] = d2
            rows[t][f'paired_minus_{lab}_se_seq'] = se2

    refs = {}
    for lab, key in (('E22a_full_predicate_basis', 'induction_E22a_full'),
                     ('E19a_no_named_terms', 'induction_E19a_base'),
                     ('E0a_muon_vanilla', 'induction_E0a_muon')):
        adv, se = ind(key)
        refs[lab] = {'induction_advantage': adv, 'induction_se_seq': se,
                     'audit_held_ce': (j.get(key) or {}).get(
                         'audit_held_ce', {}).get('full')}
    refs['stored_E29_ce'] = {
        'predicate_basis_mean_n3': 4.89998, 'predicate_basis_sd': PRED_SEED_SD,
        'frontier_base_mean_n4': 4.98575, 'frontier_base_sd': 0.00805,
        'constrained_named_term_gain': CONSTRAINED_GAIN}
    refs['stored_vanilla_ce'] = {'E0a_muon': 4.75698, 'E0a_adamw': 4.85127}

    lad = (j.get('ladder_paired_deltas') or {}).get('deltas', {})

    def delta(x, y):
        """CE(x) - CE(y) from the stored ladder, whichever direction it was
        stored in."""
        r = lad.get(f'{x}_minus_{y}')
        if r is not None and r.get(f'minus_{y}') is not None:
            return round(r[f'minus_{y}'], 5)
        r = lad.get(f'{y}_minus_{x}')
        if r is not None and r.get(f'minus_{x}') is not None:
            return round(-r[f'minus_{x}'], 5)
        return None

    marg = {
        'drop_match_same_cost_nats': delta('profile_prev_E34c', 'full_E22a'),
        'drop_match_prev_cost_nats': delta('profile_only_E34b',
                                           'profile_prev_E34c'),
        'drop_profile_cost_nats': delta('prev_same_E34d', 'full_E22a'),
        'drop_all_named_cost_nats': delta('none_E19a', 'full_E22a'),
        'sign_convention': 'positive = the ablated arm is WORSE (higher CE) '
                           'than the arm that keeps the term; each number is '
                           'a paired per-token delta on the same held rows'}

    verdicts = {}
    a1 = rows.get('E34a', {})
    g = a1.get('paired_minus_e0a_muon')
    if g is not None:
        verdicts['i_arm1_named_terms_on_vanilla'] = {
            'paired_gain_vs_E0a_muon_nats': g,
            'se_seq': a1.get('paired_minus_e0a_muon_se_seq'),
            'constrained_gain_reference': CONSTRAINED_GAIN,
            'registered_interval': '(-0.05, 0.00]',
            'prediction_held': bool(-0.05 < g <= 0.0),
            'general_modelling_result': bool(g <= -0.05),
            'refutes_tax_reducer_story': bool(g <= CONSTRAINED_GAIN),
            'reading': ('the named terms are an interpretability-tax reducer: '
                        'they pay under constraint and not without it'
                        if g > -0.05 else
                        'the named terms are a GENERAL architecture '
                        'improvement: they pay on the unconstrained model too')}
    dp, dpv, ds = (marg['drop_profile_cost_nats'],
                   marg['drop_match_prev_cost_nats'],
                   marg['drop_match_same_cost_nats'])
    if None not in (dp, dpv, ds):
        verdicts['ii_profile_carries_most_of_the_ce_gain'] = {
            'drop_profile': dp, 'drop_match_prev': dpv,
            'drop_match_same': ds,
            'verdict': bool(dp > dpv and dp > ds)}
    if ds is not None:
        verdicts['iii_match_same_costs_under_0.01'] = {
            'drop_match_same': ds, 'verdict': bool(ds < 0.01)}
    advb, _ = ind('induction_E34b')
    ceb = ce('E34b')
    cef = ce('E22a') if ce('E22a') is not None else 4.895678
    if advb is not None:
        verdicts['iv_dropping_match_prev_costs_induction'] = {
            'E34b_profile_only_induction_advantage': advb,
            'full_arm_induction_advantage': refs[
                'E22a_full_predicate_basis']['induction_advantage'],
            'e28_all_b_zeroed_reference': E28_ALL_B_ZERO_ADV,
            'E34b_ce_minus_full': (None if ceb is None else
                                   round(ceb - cef, 5)),
            'verdict': bool(advb <= 0.8 and (ceb is None
                                             or ceb - cef <= 0.05))}
    out = {'arms': rows, 'references': refs, 'marginal_costs': marg,
           'ladder_deltas': lad,
           'registered_prediction_verdicts': verdicts,
           'readability_caveat': 'wiring Spearman values are REPORTED, not '
                                 'ranked on: BRAINSTORM_STATE\'s foundations '
                                 'correction (E27) showed the axis moves '
                                 '0.128 between seeds, so a single seed '
                                 'cannot order arms on readability',
           'ce_resolution_caveat': f'the predicate-basis arm\'s CE sample sd '
                                   f'over 3 seeds is {PRED_SEED_SD} nats; '
                                   f'single-seed differences below about '
                                   f'{2 * PRED_SEED_SD:.3f} nats are not '
                                   f'resolved'}
    E.merge(JP, 'summary_E34', out)
    print(json.dumps({'marginal_costs': marg,
                      'registered_prediction_verdicts': verdicts}, indent=2,
                     default=str), flush=True)


# ============================ main ==========================================
if __name__ == '__main__':
    E.setup()
    E.DEV = 'cpu' if E.SMOKE else 'cuda'
    mlr = E7R.muon_lr()
    s_c, _ = E15R.solve_slot_c(4 * Q.D)
    if not E.SMOKE:
        assert s_c == 15, s_c
        for st in ('qk_e22_a', 'qk_e19_a', 'qk_e0a_muon264',
                   'qk_e0a_vanilla264'):
            assert os.path.exists(E.ckpath(st)), f'{st}.pt missing'
    A = arms(s_c)

    # ---- registered predictions, BEFORE any training ----
    if 'E34_prediction' not in E.loadj(JP):
        E.merge(JP, 'E34_prediction', {
            'registered_before_training': True,
            'arms': {a['tag']: {'terms_enabled': list(a['terms']),
                                'design': a['design']} for a in A},
            'i_arm1_vanilla': 'the named terms help the VANILLA model LESS '
                              'than the constrained one: paired gain vs '
                              'E0a-muon in (-0.05, 0.00], point prediction '
                              '-0.02 nats, against the constrained arm\'s '
                              f'{CONSTRAINED_GAIN}. A gain of -0.05 or better '
                              'makes this a general modelling result; a gain '
                              f'at or beyond {CONSTRAINED_GAIN} refutes the '
                              'interpretability-tax-reducer reading',
            'ii_profile_dominant': 'dropping the positional profile costs '
                                   'more CE than dropping MATCH_prev and more '
                                   'than dropping MATCH_same (E22a mass '
                                   'shares: median 68% profile vs 7% '
                                   'MATCH_prev vs 0.5% MATCH_same)',
            'iii_match_same_cheap': 'dropping MATCH_same costs < 0.01 nats '
                                    '(total |c| mass 10.4 vs |b| 52.7)',
            'iv_match_prev_is_induction': 'the profile-only arm E34b falls '
                                          'below 0.8 nats of induction '
                                          f'advantage (full arm '
                                          f'{E28_CLEAN_ADV}, E28 all-b-zeroed '
                                          f'{E28_ALL_B_ZERO_ADV}) while its '
                                          'CE stays within +0.05 of the full '
                                          'arm -- CE alone cannot price '
                                          'MATCH_prev',
            'resolution': f'predicate-basis CE sample sd over 3 seeds is '
                          f'{PRED_SEED_SD}; single-seed differences below '
                          f'~{2 * PRED_SEED_SD:.3f} nats are not resolved',
            'design': 'one seed (Q.SEED = 0) per arm, data order fixed via '
                      f'the unchanged Q.DATA_SEED = {Q.DATA_SEED}'})
    if 'E34_config' not in E.loadj(JP):
        E.merge(JP, 'E34_config', {
            'muon_lr': mlr, 'adamw_lr': E.get_lr(), 'slot': s_c,
            'group_coeffs': {a['tag']: a['gc'] for a in A},
            'stems': {a['tag']: a['stem'] for a in A},
            'data_seed': Q.DATA_SEED, 'seed': Q.SEED,
            'trainer': 'qk_e_common.train_muon with lr_adamw = the family '
                       'AdamW lr -- the E19a/E22a trainer expression verbatim',
            'predicate_params': 'pred_prof (depth x NH x T signed per-offset '
                                'profile), pred_b / pred_c (depth x NH signed '
                                'mixture weights); all init 0, all on AdamW '
                                'no-decay via muon_exclude; a DISABLED term '
                                'is removed from the forward so its parameter '
                                'never receives a gradient',
            'kernels': 'MATCH_prev / MATCH_same are the imported '
                       'qk_e22_predbasis_run.match_kernels function object '
                       '(control 2 checks it against qk_e21_census_run)'})

    # ---- controls, before any training (cheap first) ----
    control_inherited_gates()
    control_kernels()
    stored0 = {}
    for jf, k in (('qk_e0m.json', 'E0a_muon'), ('qk_e19.json', 'E19a')):
        cur = E.loadj(E.jpath(jf)).get(k, {}).get('train_curve_every200')
        stored0[k] = (cur[0][1] if cur else None)
    for a in A:
        control_pred_zero(a, None if E.SMOKE else
                          stored0['E0a_muon' if a['kind'] == 'vanilla'
                                  else 'E19a'])

    # ---- reference induction on the checkpoints already on disk (cheap) ----
    probe = build_probe()
    induction_record('induction_E22a_full', 'qk_e22_a',
                     lambda: E22R.make_e22(s=s_c), probe, named=True,
                     ref=E28_CLEAN_ADV)
    control_probe_reproduces_e28()
    induction_record('induction_E19a_base', 'qk_e19_a',
                     lambda: E15R.make_e15c(s=s_c), probe, named=False)
    induction_record('induction_E0a_muon', 'qk_e0a_muon264', E.make_e0a,
                     probe, named=False)

    # ---- train + measure, arm by arm ----
    for a in A:
        try:
            E.train_arm(a['stem'], JP, a['tag'], a['factory'], a['gc'],
                        lr=mlr, trainer=trainer,
                        extra={'optimizer': 'muon', 'seed': Q.SEED,
                               'data_seed': Q.DATA_SEED, 'slot': a['slot'],
                               'terms_enabled': list(a['terms']),
                               'terms_disabled': [t for t in ALL_TERMS
                                                  if t not in a['terms']],
                               'base': a['base_name'],
                               'parent_for_pairing': a['parent_stem'],
                               'design': a['design']})
            control_terms_zero(a)
            E.oldheld_record(a['stem'], a['factory'], JP, f"{a['tag']}_oldheld")
            E.paired_fresh(a['stem'], JP, a['tag'])
            pair_all(a)
            mixture_tables(a)
            induction_record(f"induction_{a['tag']}", a['stem'], a['factory'],
                             probe, named=True)
            probes(a)
        except Exception as ex:
            import traceback
            traceback.print_exc()
            E.merge(JP, f"{a['tag']}_FAILED",
                    {'error': f'{type(ex).__name__}: {str(ex)[:300]}'})
            if not E.SMOKE:
                torch.cuda.empty_cache()

    cross_pairs()
    summarize(A)
    print('e34 ablate run done', flush=True)
