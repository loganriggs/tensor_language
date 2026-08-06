"""E20 CODEBOOK SLOTS -- discrete content on the frontier-best architecture
(fresh single-epoch batch-16, recipe conventions; results -> qk_e20.json).

BASE (the E19a arm, current CE-x-readability frontier at width 264): E15c
bandwidth-reinvestment architecture -- true-small decoders, 24 slots x 15
dims, stream 360, compute width 264, hidden 1056 (qk_e15_reinvest_run.
make_e15c(s=15) reused verbatim as the superclass) -- with in-loss
group-lasso 1e-4 and the full readable recipe (per-slot RMSNorm, Muon 0.02 /
embedding AdamW 0.004). Parent numbers: CE 4.9742, covariance-composed
Spearman 0.8259, plain 0.7911 (qk_e19.json).

CHANGE (E20a): vector-quantize each slot's content AFTER per-slot RMSNorm at
the block boundaries. Per-slot codebook of n=256 UNIT-NORM 15-dim codes; k=2
selection by 2-step matching pursuit: pick the nearest code (for unit-norm
codes nearest-in-Euclidean == argmax signed inner product), scale = the inner
product <residual, code> (the least-squares coefficient for a unit code),
subtract the scaled code, pick nearest again; the message is the sum of the
2 scaled codes (code identities discrete, the 2 scales continuous -- the MDL
content term below counts code-identity bits only). Straight-through
gradients; EMA codebook updates (decay 0.99, codes re-normalized to unit
norm after each update, update direction = the coefficient-weighted residual
each code matched); commitment loss beta=0.25 (mean squared distance between
the continuous post-norm content and its stopped-gradient reconstruction,
averaged over quantized slots, injected into the training loss via the
qk_e_common.train_muon pop_aux_loss hook -- NOT part of final_penalty);
dead-code reinit (any code unused for 200 consecutive training steps is
reinitialized to a random current slot content vector from the batch,
unit-normalized).

WHAT IS AND IS NOT QUANTIZED (the brief requires this documented exactly):
  QUANTIZED: every module-written slot's post-per-slot-RMSNorm content at
  every block-level read -- both the attention entry (hn) and the post-attn
  MLP entry (xn). In E15c slot k's pre-norm content is (embedding slot k) +
  (module k's write): the bottom-injected embedding is NOT routed separately
  in this architecture, so the embedding component inside a written slot is
  quantized together with the write (there is no separately-routed
  embedding channel to exempt). Because each module writes exactly once and
  scattered writes are exact zeros elsewhere, a written slot's content is
  bit-identical at every consumer; it is therefore quantized ONCE per
  forward (at its first consumer) and the quantized tensor reused -- one
  EMA/usage assignment per slot per training step.
  NOT QUANTIZED: (1) not-yet-written slots -- they carry only the
  bottom-injected embedding (the embedding-carrying content path); (2) the
  readout path -- the readout reads the CONTINUOUS stream through the
  GLOBAL RMSNorm (there is no per-slot norm at the readout, and per-slot
  unit-RMS quantized content would destroy the relative slot magnitudes the
  global norm preserves). Consequence: slot 23 (mlp11's write) is read ONLY
  by the readout and never passes through a codebook; codebook 23 is
  causally inert and is excluded from the dead-code / content-bits
  accounting (reported explicitly).

CONTROLS BEFORE TRAINING (hard gates):
  1. BYPASS: with quantization disabled the E20 model reproduces the E19a
     forward bit-exactly at init (same seed discipline: the codebook is
     initialized from a separate fixed-seed generator so the shared
     parameters consume the identical global RNG stream; parameter identity
     asserted too), AND a 3-step training run of this runner's trainer on
     the bypassed model reproduces the E19a trainer's 3-step trajectory
     exactly (proves the aux-loss hook + extra buffers change nothing when
     quantization is off).
  2. CAPACITY: with n >= the number of distinct slot vectors in a tiny
     batch and k=15 pursuit steps, quantization error must be ~0 (the
     codebook can represent anything).
  3. EMA/COMMITMENT machinery on a planted toy: 10 known cluster centers,
     the codebook must recover them (usage concentrated, reconstruction ~0)
     -- run through the SAME mp_quantize/ema_update functions the model
     uses.
  Plus (4, before the composed probe): the E20 covariance pipeline with
  quantization disabled must reproduce qk_e18_probe_upgrades.
  gen_slot_covariances exactly; and the E18 gates 1+2 precondition.

REGISTERED PREDICTIONS (merged into the JSON before training):
  (a) quantization CE cost vs the E19a parent <= +0.15 is PROMISING,
      > +0.30 REFUTES this granularity (n=256, k=2);
  (b) dead-code fraction < 30% after training;
  (c) slack-census modules (low write-covariance effective rank, qk_e14
      census on the recipe) will use fewer distinct codes than saturated
      ones.

MEASURE: fresh held CE paired vs E19a parent / E15c grandparent / E9a
recipe / E0a+E0b controls; per-slot code usage histograms + dead-code
fraction + usage entropy (bits); per-slot top-20 code PAIRS by pointwise
mutual information (are combinations reused as units? -- the superposition-
enumerability question); content bits/token = sum over quantized slots of
the joint (code1, code2) usage entropy; the generalized variable-slot-dim
light probe + covariance-composed wiring where the slot content covariance
comes from the QUANTIZED content consumers actually read.

REVIEWER-2 ADDITIONS (BRAINSTORM_STATE.md 'Reviewer-2 findings'):
  (1) conditional DISTILLATION CONTROL: if the scratch quantized arm costs
      > +0.15 vs the E19a parent, initialize from the continuous parent
      checkpoint, data-init codebooks from the parent's own content,
      quantize-and-finetune 2000 steps on corpus rows [132000:164000) --
      never seen by any arm -- and compare (trainability vs granularity);
  (2) per-pursuit-step residual norms (content / after code 1 / after
      code 2) per slot on the held pass;
  (3) code DICTIONARIES on the fixed audit slice fresh34k[33000:33200]:
      top-50 codes for representative slots with top-10 token contexts
      each -> qk_e20_code_dictionaries.json;
  (4) dead-code event log (step, slot, codes, reinit source) ->
      qk_e20_deadcode_events.jsonl + codebook snapshots every 1000 steps
      -> qk_e20_codebook_snapshots.npz;
  (5) per-sequence heldloss files ({stem}_heldloss_seq.npy).
Idempotent on qk_e20.json keys and the checkpoint."""
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
import qk_e18_probe_upgrades as E18U      # generalized probe + cov machinery
import qk_e17_composed_wiring as E17      # agreement
import qk_deeproute_train_2 as R2

# The E17 import (pulled in by E18U) sets E.DEV='cpu' for its own weight-only
# use -- restore the runner convention.
E.DEV = 'cpu' if E.SMOKE else 'cuda'

JP = E.jpath('qk_e20.json')
GC20 = 1e-4                               # the E19a lasso (unchanged)
GATE_TOL = 1e-3
NG = E.NGROUP

QZ_N = 256                                # codes per slot codebook
QZ_K = 2                                  # matching-pursuit steps
QZ_DECAY = 0.99                           # EMA decay
QZ_BETA = 0.25                            # commitment coefficient
QZ_DEAD = 200                             # steps unused -> reinit
READOUT_ONLY_SLOT = NG - 1                # slot 23: mlp11 -> readout only


# ---------------- quantization machinery (shared with the controls) ----------------
def mp_quantize(v, Cb, ksteps):
    """k-step matching pursuit against a unit-norm codebook.
    v: (N, s); Cb: (n, s) with unit-norm rows. Per step: nearest code
    (== argmax signed inner product for unit codes, since
    ||r - c||^2 = ||r||^2 + 1 - 2<r, c>), scale = the inner product,
    subtract the scaled code. Returns (recon, idxs, alphas, residuals),
    residuals[j] = the residual BEFORE step j (what code j matched)."""
    r = v
    recon = torch.zeros_like(v)
    idxs, alphas, resids = [], [], []
    for _ in range(ksteps):
        ip = r @ Cb.t()
        idx = ip.argmax(1)
        a = ip.gather(1, idx[:, None]).squeeze(1)
        resids.append(r)
        idxs.append(idx)
        alphas.append(a)
        recon = recon + a[:, None] * Cb[idx]
        r = r - a[:, None] * Cb[idx]
    return recon, idxs, alphas, resids


@torch.no_grad()
def ema_update(Cb, M, last_used, usage, step_now, x, idxs, alphas, resids,
               decay=QZ_DECAY, dead_t=QZ_DEAD):
    """EMA codebook update + dead-code reinit (the exact machinery the model
    runs each training step; the planted-toy control drives this same
    function). M accumulates the coefficient-weighted residual each code
    matched (the online least-squares direction for a unit code); the code
    is the unit-normalized accumulator, so unused codes keep their
    direction. Dead codes (unused > dead_t steps) reinit to a random
    current content vector x[i], unit-normalized."""
    n = Cb.shape[0]
    acc = torch.zeros_like(M)
    counts = torch.zeros(n, dtype=torch.long, device=Cb.device)
    for idx, a, r in zip(idxs, alphas, resids):
        acc.index_add_(0, idx, a[:, None] * r)
        counts += torch.bincount(idx, minlength=n)
    M.mul_(decay).add_(acc, alpha=1 - decay)
    nm = M.norm(dim=1)
    upd = nm > 1e-8
    Cb[upd] = M[upd] / nm[upd, None]
    used = counts > 0
    last_used[used] = step_now
    usage += counts
    dead = (step_now - last_used) > dead_t
    reinit = None
    if bool(dead.any()):
        nd = int(dead.sum())
        ridx = torch.randint(0, x.shape[0], (nd,), device=x.device)
        cand = x[ridx]
        Cb[dead] = cand / cand.norm(dim=1, keepdim=True).clamp_min(1e-8)
        M[dead] = Cb[dead]
        last_used[dead] = step_now
        reinit = (dead.nonzero().squeeze(1).tolist(), ridx.tolist())
    return counts, reinit


# ---------------- the E20 architecture ----------------
class E20Route(E15R.E15cRoute):
    """E15c + per-slot codebook quantization at the per-slot-RMSNorm consumer
    interface (see the module docstring for exactly what is quantized)."""

    def __init__(self, variant, depth, s, Dc, NH, HD, hidden, n_codes=QZ_N):
        super().__init__(variant, depth, s, Dc, NH, HD, hidden)
        self.qz_on = True
        self.n_codes = n_codes
        self._aux = None
        self._commit_sum = None
        self._commit_n = 0
        # dead-code event log (python list; flushed to a jsonl by the
        # training step callback -- not persisted in the checkpoint)
        self.qz_dead_events = []
        # Codebook init from a SEPARATE fixed-seed generator: the global RNG
        # stream stays identical to make_e15c, so all shared parameters are
        # bit-identical at init (asserted by the bypass control).
        gq = torch.Generator().manual_seed(4242)
        cb = torch.randn(2 * depth, n_codes, s, generator=gq)
        cb = cb / cb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self.register_buffer('qz_codebook', cb.float())
        self.register_buffer('qz_ema', cb.float().clone())
        self.register_buffer('qz_last_used',
                             torch.zeros(2 * depth, n_codes, dtype=torch.long))
        self.register_buffer('qz_usage',
                             torch.zeros(2 * depth, n_codes, dtype=torch.long))
        self.register_buffer('qz_step', torch.zeros((), dtype=torch.long))

    def pop_aux_loss(self):
        a = self._aux
        self._aux = None
        return a

    def _qz_slot(self, k, v, collect, cache):
        """Quantize one written slot's post-norm content v (B, T, s):
        matching pursuit in fp32 (outside autocast), straight-through
        output, EMA/usage/dead-reinit + commitment stash in training."""
        upd = self.training and torch.is_grad_enabled() and cache is not None
        with torch.autocast('cpu' if v.device.type == 'cpu' else 'cuda',
                            enabled=False):
            x32 = v.detach().float().reshape(-1, self.s)
            recon, idxs, alphas, resids = mp_quantize(
                x32, self.qz_codebook[k], QZ_K)
            if upd:
                _, reinit = ema_update(
                    self.qz_codebook[k], self.qz_ema[k],
                    self.qz_last_used[k], self.qz_usage[k],
                    int(self.qz_step), x32, idxs, alphas, resids)
                if reinit is not None:
                    self.qz_dead_events.append(
                        {'step': int(self.qz_step), 'slot': k,
                         'n_reinit': len(reinit[0]),
                         'codes': reinit[0][:40],
                         'source_flat_token_idx': reinit[1][:40]})
            q = recon.view(v.shape)
        out = v + (q.to(v.dtype) - v).detach()
        if upd:
            commit = (v.float() - q.detach()).pow(2).mean()
            self._commit_sum = commit if self._commit_sum is None \
                else self._commit_sum + commit
            self._commit_n += 1
        if collect is not None:
            collect.setdefault('qcontent', {})[k] = q.detach()
            if 'codes' in collect:
                collect['codes'].setdefault(k, []).append(
                    torch.stack(idxs, 1).to(torch.int32).cpu())
            if 'scales' in collect:
                collect['scales'].setdefault(k, []).append(
                    torch.stack(alphas, 1).float().cpu())
            if 'resid_stats' in collect:
                # sums of squared L2 norms: content, after code 1, after
                # code 2, plus token count (exact weighted aggregation)
                r_fin = x32 - recon
                collect['resid_stats'].setdefault(k, []).append([
                    float(x32.pow(2).sum()),
                    float(resids[1].pow(2).sum()) if len(resids) > 1
                    else float(r_fin.pow(2).sum()),
                    float(r_fin.pow(2).sum()),
                    float(x32.shape[0])])
        if cache is not None:
            cache[k] = out
        return out

    def _qz_full(self, full, n_written, cache, collect):
        """Replace the first n_written slots' post-norm content with the
        quantized version; not-yet-written slots pass through untouched.
        With qz_on False this returns `full` unchanged (bit-exact bypass)."""
        if not self.qz_on or n_written == 0:
            return full
        s = self.s
        segs = []
        for k in range(2 * self.depth):
            v = full[..., s * k:s * (k + 1)]
            if k >= n_written:
                segs.append(v)
            elif cache is not None and k in cache:
                segs.append(cache[k])
            else:
                segs.append(self._qz_slot(k, v, collect, cache))
        return torch.cat(segs, dim=-1)

    def forward(self, idx, collect=None, sub_entry=None):
        B, Tq = idx.shape
        Ws, s = self.Ws, self.s
        cos = self.cos[None, :Tq, None, :]
        sin = self.sin[None, :Tq, None, :]
        mask = self.mask[:Tq, :Tq]
        e = F.rms_norm(self.wte(idx), (Ws,))
        streams = [e]
        if self.training and torch.is_grad_enabled() and self.qz_on:
            self.qz_step += 1
        self._commit_sum, self._commit_n = None, 0
        # The per-forward cache is only valid when no per-consumer stream
        # substitution is active (sub_entry changes one consumer's view of a
        # slot; the cached quantization of the unsubstituted content would
        # silently leak into the ablated consumer).
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
        # READOUT: continuous stream through the GLOBAL norm (unquantized --
        # documented exemption; slot 23 is consequently never quantized).
        x = F.rms_norm(entry(self.depth), (Ws,))
        logits = x @ self.wte.weight.t()
        if self._commit_sum is not None:
            self._aux = QZ_BETA * self._commit_sum / self._commit_n
        return 30 * torch.tanh(logits / 30)


def make_e20(s=None, qz_on=True):
    hidden = 4 * Q.D
    if s is None:
        s, _ = E15R.solve_slot_c(hidden)
    torch.manual_seed(Q.SEED)
    m = E20Route('E20a', DEPTH, s, Q.D, Q.NH, Q.HD, hidden).to(E.DEV)
    m.qz_on = qz_on
    return m


_SNAPSHOTS = {}


def flush_dead_events(model, phase='train'):
    ev = getattr(model, 'qz_dead_events', None)
    if ev:
        with open(E.jpath('qk_e20_deadcode_events.jsonl'), 'a') as f:
            for e_ in ev:
                e_['phase'] = phase
                f.write(json.dumps(e_) + '\n')
        model.qz_dead_events = []


def qz_step_cb(step, model):
    """Training-time logging (reviewer-2 item 4): codebook snapshot every
    1000 steps into qk_e20_codebook_snapshots.npz; dead-code events flushed
    to qk_e20_deadcode_events.jsonl every 250 steps (events in the final
    <250 steps stay unflushed -- noted in the JSON). Read-only w.r.t. the
    model; does not perturb training."""
    if not getattr(model, 'qz_on', False):
        return
    flush_dead_events(model)
    if step % 1000 == 0:
        _SNAPSHOTS[f'step{step:05d}'] = \
            model.qz_codebook.detach().cpu().numpy()
        np.savez(E.jpath('qk_e20_codebook_snapshots.npz'), **_SNAPSHOTS)


qz_step_cb.every = 250


def trainer(lr, gc, steps, **kw):
    """EXACTLY the E19a trainer expression (commitment enters via the
    pop_aux_loss hook inside train_muon, not here) + the read-only logging
    callback."""
    kw.setdefault('step_cb', qz_step_cb)
    return E.train_muon(lr, gc, steps, lr_adamw=E.get_lr(), **kw)


def three_step(factory, gc, steps=3):
    log = E.train_muon(E7R.muon_lr(), gc, steps, log_every=1, save=False,
                       factory=factory, lr_adamw=E.get_lr())
    return {'per_step_ce': [x[1] for x in log['train_loss']],
            'held100_ce': log['final_held_ce']}


# ---------------- controls (before training) ----------------
def control_e18_gates():
    if E.SMOKE:
        return
    e18 = E.loadj(E.jpath('qk_e18.json'))
    g1 = e18.get('gate1_uniform11_weight_support', {}).get('pass')
    g2 = e18.get('gate2_cov_composed_E9a', {}).get('pass')
    assert g1 and g2, ('qk_e18.json gates not passed -- the reused probe '
                       'functions are not validated', g1, g2)
    print("precondition: qk_e18.json gates 1+2 passed", flush=True)


def control_bypass(s_c):
    """Gate 1: quantization-off == E19a, at init (bit-exact forward +
    parameter identity) and over a 3-step training run."""
    key = 'control1_bypass'
    done = E.loadj(JP).get(key)
    if done and done.get('pass'):
        print(f"{key}: already passed -- skip", flush=True)
        return
    m19 = E15R.make_e15c(s=s_c).eval().float()
    m20 = make_e20(s=s_c, qz_on=False).eval().float()
    p19 = dict(m19.named_parameters())
    pdiff = max(float((p - p19[nm]).abs().max())
                for nm, p in m20.named_parameters())
    idx = E.OLD_HELD[:2, :Q.T]
    with torch.no_grad():
        d_fwd = float((m20(idx) - m19(idx)).abs().max())
        m20.qz_on = True
        d_qz = float((m20(idx) - m19(idx)).abs().max())
        m20.qz_on = False
    del m19, m20
    torch.cuda.empty_cache()
    parent = three_step(lambda: E15R.make_e15c(s=s_c), GC20)
    mine = three_step(lambda: make_e20(s=s_c, qz_on=False), GC20)
    step_diff = max(abs(a - b) for a, b in
                    zip(parent['per_step_ce'], mine['per_step_ce']))
    held_diff = abs(parent['held100_ce'] - mine['held100_ce'])
    rec = {'param_identity_max_abs_diff': pdiff,
           'forward_bypass_max_logit_diff': d_fwd,
           'forward_quantized_max_logit_diff_at_init_informational': d_qz,
           'train3_parent_per_step_ce': parent['per_step_ce'],
           'train3_bypass_per_step_ce': mine['per_step_ce'],
           'train3_max_per_step_abs_diff': step_diff,
           'train3_held100_abs_diff': held_diff,
           'note': 'same seed discipline as E19 (factory reseeds; codebook '
                   'from a separate generator); 3-step run proves the '
                   'aux-loss hook + buffers are inert with quantization off',
           'pass': bool(pdiff == 0.0 and d_fwd == 0.0 and step_diff == 0.0
                        and held_diff < 1e-6)}
    E.merge(JP, key, rec)
    print(f"{key}: params {pdiff:.1e}, forward {d_fwd:.1e}, 3-step "
          f"{step_diff:.1e}/{held_diff:.1e} (quantized-at-init logit shift "
          f"{d_qz:.3f}) -> {'PASS' if rec['pass'] else 'FAIL'}", flush=True)
    assert rec['pass'], f'{key} FAILED'


def control_capacity(s_c):
    """Gate 2: n >= #distinct vectors and k=15 -> quantization error ~0."""
    key = 'control2_capacity'
    done = E.loadj(JP).get(key)
    if done and done.get('pass'):
        print(f"{key}: already passed -- skip", flush=True)
        return
    m = make_e20(s=s_c, qz_on=True).eval().float()
    b = E.OLD_HELD[:2, :Q.T]
    with torch.no_grad():
        e = F.rms_norm(m.wte(b), (m.Ws,))
        v = F.rms_norm(e[..., :m.s], (m.s,)).reshape(-1, m.s).float()[:128]
        uniq = torch.unique(v, dim=0)
        Cb = uniq / uniq.norm(dim=1, keepdim=True).clamp_min(1e-8)
        recon, _, _, _ = mp_quantize(v, Cb, 15)
        rel = float((v - recon).norm() / v.norm().clamp_min(1e-12))
    del m
    torch.cuda.empty_cache()
    rec = {'n_vectors': int(v.shape[0]), 'n_distinct': int(uniq.shape[0]),
           'codebook_n': int(Cb.shape[0]), 'k_steps': 15,
           'rel_error': rel, 'pass': bool(rel < 1e-4)}
    E.merge(JP, key, rec)
    print(f"{key}: {rec['n_distinct']} distinct vectors, k=15, rel error "
          f"{rel:.2e} -> {'PASS' if rec['pass'] else 'FAIL'}", flush=True)
    assert rec['pass'], f'{key} FAILED'


def control_toy():
    """Gate 3: the EMA + dead-reinit + commitment machinery recovers 10
    planted cluster centers (through the exact functions the model uses)."""
    key = 'control3_toy_ema'
    done = E.loadj(JP).get(key)
    if done and done.get('pass'):
        print(f"{key}: already passed -- skip", flush=True)
        return
    dim, ncen, ncode = 15, 10, 64
    g = torch.Generator().manual_seed(7)
    cen = torch.randn(ncen, dim, generator=g)
    cen = cen / cen.norm(dim=1, keepdim=True) * math.sqrt(dim)
    Cb = torch.randn(ncode, dim, generator=g)
    Cb = Cb / Cb.norm(dim=1, keepdim=True)
    M = Cb.clone()
    last_used = torch.zeros(ncode, dtype=torch.long)
    usage = torch.zeros(ncode, dtype=torch.long)
    steps = 120 if E.SMOKE else 600
    commit0 = commit_end = None
    for t in range(1, steps + 1):
        lab = torch.randint(0, ncen, (512,), generator=g)
        x = cen[lab] + 0.01 * torch.randn(512, dim, generator=g)
        recon, idxs, alphas, resids = mp_quantize(x, Cb, QZ_K)
        cv = float((x - recon).pow(2).mean())
        if t == 1:
            commit0 = cv
        commit_end = cv
        ema_update(Cb, M, last_used, usage, t, x, idxs, alphas, resids)
    # final assessment on a fresh batch
    lab = torch.randint(0, ncen, (2048,), generator=g)
    x = cen[lab] + 0.01 * torch.randn(2048, dim, generator=g)
    recon, idxs, _, _ = mp_quantize(x, Cb, QZ_K)
    rel_mse = float((x - recon).pow(2).mean() / x.pow(2).mean())
    cosmat = (cen / cen.norm(dim=1, keepdim=True)) @ Cb.t()
    center_maxcos = cosmat.max(1).values
    samp_cos = (Cb[idxs[0]] * (cen[lab] / cen[lab].norm(dim=1, keepdim=True))
                ).sum(1)
    frac_matched = float((samp_cos > 0.99).float().mean())
    cnt = torch.bincount(idxs[0], minlength=ncode).float()
    p = (cnt / cnt.sum()).clamp_min(1e-12)
    h1 = float(-(p * p.log2()).sum())
    rec = {'n_centers': ncen, 'n_codes': ncode, 'noise': 0.01,
           'steps': steps,
           'center_max_cos_min': float(center_maxcos.min()),
           'frac_samples_step1_code_cos_gt_0.99': frac_matched,
           'final_rel_mse': rel_mse,
           'step1_usage_entropy_bits': round(h1, 3),
           'ideal_entropy_bits': round(math.log2(ncen), 3),
           'commit_first_step': commit0, 'commit_last_step': commit_end,
           'pass': bool(float(center_maxcos.min()) > 0.99
                        and frac_matched >= 0.99 and rel_mse < 1e-2
                        and h1 <= math.log2(ncen) + 1.2
                        and commit_end < 0.1 * commit0)}
    E.merge(JP, key, rec)
    print(f"{key}: center cos min {rec['center_max_cos_min']:.4f}, matched "
          f"{frac_matched:.3f}, rel MSE {rel_mse:.2e}, usage entropy "
          f"{h1:.2f} bits (ideal {math.log2(ncen):.2f}), commitment "
          f"{commit0:.3f} -> {commit_end:.5f} -> "
          f"{'PASS' if rec['pass'] else 'FAIL'}", flush=True)
    assert rec['pass'], f'{key} FAILED'


# ---------------- code-usage statistics ----------------
def entropy_bits(counts):
    tot = counts.sum()
    if tot == 0:
        return 0.0
    p = counts[counts > 0].astype(np.float64) / tot
    return float(-(p * np.log2(p)).sum())


def code_stats(s_c):
    """Per-slot usage histograms, dead-code fraction, usage entropy, PMI
    code pairs, content bits/token -- on the full fresh held set."""
    key = 'code_stats_E20a'
    if key in E.loadj(JP):
        print(f"{key}: already done -- skip", flush=True)
        return
    stem = 'qk_e20_a'
    if E.SMOKE:
        m = make_e20(s=s_c)
    else:
        if not os.path.exists(E.ckpath(stem)):
            return
        m, _ = E.load_arm(stem, lambda: make_e20(s=s_c))
    m.eval().float()
    rows = Q.HELD
    per_slot = {k: [] for k in range(NG)}
    resid_acc = {k: np.zeros(4) for k in range(NG)}
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(rows), 8):
            b = rows[i:i + 8]
            col = {'codes': {}, 'resid_stats': {}}
            m(b[:, :Q.T], collect=col)
            for k, chunks in col['codes'].items():
                per_slot[k].append(torch.cat(chunks, 0))
            for k, sums in col['resid_stats'].items():
                for s4 in sums:
                    resid_acc[k] += np.asarray(s4)
            if (i // 8) % 40 == 0:
                print(f"  code stats: {i + b.shape[0]}/{len(rows)} seqs "
                      f"({time.time() - t0:.0f}s)", flush=True)
    quantized_slots = sorted(k for k in per_slot if per_slot[k])
    ntok = None
    slots_rec = {}
    bits_total = 0.0
    dead_num = dead_den = 0
    for k in quantized_slots:
        codes = torch.cat(per_slot[k], 0).long().numpy()   # (Ntok, 2)
        ntok = codes.shape[0]
        c1 = np.bincount(codes[:, 0], minlength=QZ_N)
        c2 = np.bincount(codes[:, 1], minlength=QZ_N)
        joint = np.bincount(codes[:, 0] * QZ_N + codes[:, 1],
                            minlength=QZ_N * QZ_N)
        used = (c1 + c2) > 0
        h1, h2, hj = entropy_bits(c1), entropy_bits(c2), entropy_bits(joint)
        bits_total += hj
        dead_num += int(QZ_N - used.sum())
        dead_den += QZ_N
        # PMI over the (code1, code2) pairs chosen together on one token
        p1 = c1.astype(np.float64) / ntok
        p2 = c2.astype(np.float64) / ntok
        pj = joint.astype(np.float64) / ntok
        min_count = 20 if not E.SMOKE else 2
        cand = np.nonzero(joint >= min_count)[0]
        pmis = []
        for ji in cand:
            a, bcode = ji // QZ_N, ji % QZ_N
            pmi = math.log2(pj[ji] / (p1[a] * p2[bcode]))
            pmis.append((float(pmi), int(a), int(bcode), int(joint[ji])))
        pmis.sort(reverse=True)
        by_count = sorted(((int(joint[ji]), int(ji // QZ_N), int(ji % QZ_N))
                           for ji in np.nonzero(joint)[0]), reverse=True)
        # per-pursuit-step residual norms (reviewer-2 item 2): sums of
        # squared L2 norms accumulated exactly over all held tokens
        sx, sr1, sr2, ncnt = resid_acc[k]
        resid_rec = {
            'rms_content': round(float(np.sqrt(sx / ncnt)), 4),
            'rms_after_code1': round(float(np.sqrt(sr1 / ncnt)), 4),
            'rms_after_code2': round(float(np.sqrt(sr2 / ncnt)), 4),
            'energy_captured_code1': round(float(1 - sr1 / sx), 4),
            'energy_captured_total': round(float(1 - sr2 / sx), 4)}
        slots_rec[str(k)] = {
            'module': R2.stream_name(k + 1),
            'n_tokens': int(ntok),
            'distinct_codes_used': int(used.sum()),
            'dead_fraction': round(float(1.0 - used.sum() / QZ_N), 4),
            'entropy_step1_bits': round(h1, 3),
            'entropy_step2_bits': round(h2, 3),
            'entropy_joint_bits': round(hj, 3),
            'mutual_information_bits': round(h1 + h2 - hj, 3),
            'pursuit_residual_norms': resid_rec,
            'usage_hist_combined': (c1 + c2).tolist(),
            'top20_pairs_by_pmi_mincount%d' % min_count: [
                {'code1': a, 'code2': bb, 'count': cc,
                 'pmi_bits': round(pm, 3)}
                for pm, a, bb, cc in pmis[:20]],
            'top20_pairs_by_count': [
                {'code1': a, 'code2': bb, 'count': cc}
                for cc, a, bb in by_count[:20]]}
    dead_frac = dead_num / max(dead_den, 1)
    rec = {'rows': f'fresh held ({len(rows)} seqs x {Q.T} tokens), fp32 '
                   'eval forward',
           'quantized_slots': quantized_slots,
           'excluded_slots': {str(READOUT_ONLY_SLOT):
                              'mlp11 write: read only by the (unquantized) '
                              'readout -- codebook causally inert, excluded '
                              'from dead-code and bits accounting'},
           'codebook_n': QZ_N, 'k_select': QZ_K,
           'dead_code_fraction_overall': round(dead_frac, 4),
           'content_bits_per_token_sum_joint_entropy': round(bits_total, 3),
           'note_scales': 'the 2 pursuit scales per slot are continuous and '
                          'NOT counted in the content bits (code identity '
                          'bits only)',
           'per_slot': slots_rec}
    if not E.SMOKE:
        # training-time buffers for reference
        m_usage = m.qz_usage.cpu().numpy()
        rec['train_usage_distinct_per_slot'] = {
            str(k): int((m_usage[k] > 0).sum()) for k in range(NG)}
        rec['train_qz_steps'] = int(m.qz_step)
        # prediction (c): census slack vs saturated
        cen = E.loadj(E.jpath('qk_e14.json'))['census_e9a']['modules']
        flag = {c['module']: c['flag'] for c in cen}
        rank = {c['module']: c['write_cov_effective_rank'] for c in cen}
        by_flag = {}
        pairs_rc = []
        for k in quantized_slots:
            mod = R2.stream_name(k + 1)
            hj = slots_rec[str(k)]['entropy_joint_bits']
            by_flag.setdefault(flag[mod], []).append(hj)
            pairs_rc.append((rank[mod], hj))
        mean_by_flag = {f: round(sum(v) / len(v), 3)
                        for f, v in by_flag.items()}
        sp = R2.spearman([a for a, _ in pairs_rc], [b for _, b in pairs_rc])
        rec['prediction_c_slack_vs_saturated'] = {
            'census': 'qk_e14.json census_e9a (recipe arm; module identity '
                      'prior, slot 23 excluded)',
            'mean_joint_entropy_bits_by_flag': mean_by_flag,
            'spearman_effrank_vs_joint_entropy': round(sp, 4),
            'verdict': ('CONFIRMED' if mean_by_flag.get('slack', 99)
                        < mean_by_flag.get('saturated', -99) and sp > 0
                        else 'REFUTED')}
    E.merge(JP, key, rec)
    print(f"code stats: dead fraction {dead_frac:.3f}, content bits/token "
          f"{bits_total:.1f} over {len(quantized_slots)} quantized slots",
          flush=True)
    del m
    torch.cuda.empty_cache()


# ---------------- probes ----------------
@torch.no_grad()
def e20_slot_covariances(model, rows, dev, dims):
    """gen_slot_covariances (remnant=False branch) with the slot content
    taken from the QUANTIZED post-norm content consumers actually read
    (collect['qcontent']); slots without quantized content (slot 23, or all
    slots when qz_on is False) fall back to the continuous per-slot-norm
    content -- the fallback path is the exact gen computation (control 4).
    Returns (Cb, Cr, n): per-slot content covariance + readout-globalnorm
    covariance (the readout reads the continuous stream)."""
    model = model.to(dev).eval().float()
    offs = E18U.offsets(dims)
    n = 0
    s_b = [torch.zeros(dims[k], dtype=torch.float64, device=dev)
           for k in range(NG)]
    ss_b = [torch.zeros(dims[k], dims[k], dtype=torch.float64, device=dev)
            for k in range(NG)]
    s_r = [torch.zeros(dims[k], dtype=torch.float64, device=dev)
           for k in range(NG)]
    ss_r = [torch.zeros(dims[k], dims[k], dtype=torch.float64, device=dev)
            for k in range(NG)]
    Ws = model.wte.weight.shape[1]
    t0 = time.time()
    for i in range(0, len(rows), 4):
        b = rows[i:i + 4].to(dev)
        col = {'entry_norm': [], 'attn_write': [], 'mlp_write': []}
        model(b[:, :Q.T], collect=col)
        e = F.rms_norm(model.wte(b[:, :Q.T]), (Ws,))
        writes = []
        for j in range(DEPTH):
            writes.append(col['attn_write'][j])
            writes.append(col['mlp_write'][j])
        xf = e + sum(writes)
        gro = F.rms_norm(xf, (Ws,))
        qc = col.get('qcontent', {})
        for k in range(NG):
            sl = slice(offs[k], offs[k + 1])
            S = dims[k]
            if k in qc:
                pn = qc[k].reshape(-1, S).double()
            else:
                cont = e[..., sl] + writes[k][..., sl]
                pn = F.rms_norm(cont, (S,)).reshape(-1, S).double()
            s_b[k] += pn.sum(0)
            ss_b[k] += pn.t() @ pn
            gs = gro[..., sl].reshape(-1, S).double()
            s_r[k] += gs.sum(0)
            ss_r[k] += gs.t() @ gs
        n += b.shape[0] * Q.T
        if (i // 4) % 25 == 0:
            print(f"  e20 cov pass: {i + b.shape[0]}/{len(rows)} seqs "
                  f"({time.time() - t0:.0f}s)", flush=True)
    if dev == 'cuda':
        torch.cuda.empty_cache()

    def fin(s_, ss_):
        mu = s_ / n
        return (ss_ / n - torch.outer(mu, mu)).cpu()
    return ([fin(s_b[k], ss_b[k]) for k in range(NG)],
            [fin(s_r[k], ss_r[k]) for k in range(NG)], n)


def control_cov_pipeline(s_c):
    """Gate 4: with quantization disabled, e20_slot_covariances must equal
    gen_slot_covariances (remnant=False) exactly (fallback-path identity)."""
    key = 'control4_cov_pipeline'
    done = E.loadj(JP).get(key)
    if done and done.get('pass'):
        print(f"{key}: already passed -- skip", flush=True)
        return
    dims = [s_c] * NG
    m = make_e20(s=s_c, qz_on=False)
    held = np.load(E.HELD_PATH)[33000:33032].astype(np.int64)
    rows = torch.from_numpy(held)
    Cb_g, _, Cr_g, n_g = E18U.gen_slot_covariances(m, rows, E.DEV, dims,
                                                   remnant=False)
    Cb_e, Cr_e, n_e = e20_slot_covariances(m, rows, E.DEV, dims)
    d_b = max(float((a - b).abs().max()) for a, b in zip(Cb_g, Cb_e))
    d_r = max(float((a - b).abs().max()) for a, b in zip(Cr_g, Cr_e))
    rec = {'n_rows': 32, 'n_samples': n_e,
           'max_abs_diff_content_cov': d_b,
           'max_abs_diff_readout_cov': d_r,
           'pass': bool(n_g == n_e and d_b < 1e-6 and d_r < 1e-6)}
    E.merge(JP, key, rec)
    print(f"{key}: content cov diff {d_b:.2e}, readout cov diff {d_r:.2e} "
          f"-> {'PASS' if rec['pass'] else 'FAIL'}", flush=True)
    assert rec['pass'], f'{key} FAILED'
    del m
    torch.cuda.empty_cache()


def probe_e20a(s_c):
    """Generalized variable-slot-dim light probe + covariance-composed
    wiring with QUANTIZED slot content."""
    stem = 'qk_e20_a'
    if E.SMOKE or not os.path.exists(E.ckpath(stem)):
        return
    dims = [s_c] * NG
    j = E.loadj(JP)
    if 'light_probe_E20a_var_dims' not in j:
        print('E20a light probe (variable slot dims, quantized forward) ...',
              flush=True)
        m, _ = E.load_arm(stem, lambda: make_e20(s=s_c))
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
                   [b for b in totals if totals[b] < 0.001],
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
                       'validated); the forward runs with quantization '
                       'active (it is the architecture); mean-ablation '
                       'passes disable the per-forward quantization cache '
                       'so the substituted consumer view is re-quantized'}
        E.merge(JP, 'light_probe_E20a_var_dims', rec)
        print(f"E20a wiring Spearman all {agr['spearman_all']} "
              f"effectual({len(eff)}) {agr['spearman_effectual']} "
              f"top10 {agr['top10_precision']}", flush=True)
        del m
        torch.cuda.empty_cache()
    j = E.loadj(JP)
    if 'composed_wiring_E20a' not in j:
        lp = j['light_probe_E20a_var_dims']
        print('E20a covariance-composed re-scoring (quantized content) ...',
              flush=True)
        m, _ = E.load_arm(stem, lambda: make_e20(s=s_c))
        wp = E18U.wpairs(m, dims)
        cau = E18U.stored_cau(lp, wp)
        G = E18U.gen_gram_table(m, dims)
        plain = E18U.score(G, wp)
        held = np.load(E.HELD_PATH)[33000:33000 + E18U.N_COV].astype(np.int64)
        Cb, Cr, n_samp = e20_slot_covariances(m, torch.from_numpy(held),
                                              E.DEV, dims)
        cov = E18U.score(G, wp, lambda li, si: Cb[si - 1])
        cov_ro = E18U.score(G, wp, lambda li, si:
                            Cr[si - 1] if li == DEPTH else Cb[si - 1])
        eff = [k for k in range(len(wp)) if cau[k] > C.EFFECTUAL]
        tables = {'plain': E17.agreement(plain, cau, eff),
                  'cov_composed': E17.agreement(cov, cau, eff),
                  'cov_composed_readout_globalnorm':
                      E17.agreement(cov_ro, cau, eff)}
        chk = abs(tables['plain']['spearman_all']
                  - lp['wiring_spearman_all'])
        assert chk <= GATE_TOL, \
            f'E20a plain does not reproduce the light probe ({chk})'
        rec = {'checkpoint': f'{stem}.pt', 'slot_dims_uniform': s_c,
               'plain_reproduction_abs_diff': round(chk, 6),
               'content_source': 'slot content covariance from the '
                                 'QUANTIZED post-norm content consumers '
                                 'read (slot 23 + readout rows: continuous '
                                 '-- documented exemptions); pipeline '
                                 'identity vs gen_slot_covariances gated '
                                 'by control4',
               'n_pairs': len(wp), 'n_effectual': len(eff),
               'cov_pass': {'device': E.DEV, 'n_seq': E18U.N_COV,
                            'n_samples': n_samp,
                            'rows': f'fresh34k[33000:{33000 + E18U.N_COV}]',
                            'centered': True},
               'tables': tables}
        E.merge(JP, 'composed_wiring_E20a', rec)
        print(f"E20a plain {tables['plain']['spearman_all']} -> cov "
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
    """Standing requirement: per-SEQUENCE held losses alongside the
    per-token file (seq-mean over the 512 positions)."""
    if E.SMOKE:
        return
    p = f'{E.QK}/{stem}_heldloss.npy'
    q = f'{E.QK}/{stem}_heldloss_seq.npy'
    if os.path.exists(p) and not os.path.exists(q):
        pt = np.load(p)
        np.save(q, pt.reshape(len(Q.HELD), Q.T).mean(1))
        print(f"saved {stem}_heldloss_seq.npy ({len(Q.HELD)} seq means)",
              flush=True)


# ---------------- code dictionaries on the audit slice (reviewer-2 item 3) ----------------
AUDIT_ROWS = (33000, 33200)          # THE fixed audit slice: same 200 held
                                     # seqs forever (fresh34k[33000:33200])
REP_SLOTS = (0, 1, 3, 15, 22)        # attn0, mlp0, mlp1 (slack census),
                                     # mlp7 (saturated), attn11


def code_dictionaries(s_c):
    """For representative slots: top-50 most-used codes, each with its
    top-10 token contexts by |scale1| on the fixed audit slice. Without
    these, discrete content is just discrete spectral content. Full record
    -> qk_e20_code_dictionaries.json; a small sample + pointer -> JP."""
    key = 'code_dictionaries_E20a'
    if E.SMOKE or key in E.loadj(JP):
        return
    stem = 'qk_e20_a'
    if not os.path.exists(E.ckpath(stem)):
        return
    m, _ = E.load_arm(stem, lambda: make_e20(s=s_c))
    m.eval().float()
    held = np.load(E.HELD_PATH)[AUDIT_ROWS[0]:AUDIT_ROWS[1]].astype(np.int64)
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
        print(f"code dictionaries: tokenizer unavailable ({ex}) -- "
              "recording raw token ids", flush=True)

        def dec(ids):
            return ' '.join(str(int(t)) for t in ids)
    full = {}
    for k in REP_SLOTS:
        codes = torch.cat(codes_all[k], 0).long().numpy()      # (Ntok, 2)
        scales = torch.cat(scales_all[k], 0).numpy()           # (Ntok, 2)
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
                lo = max(0, pos - 7)
                ids = held[si, lo:pos + 1]
                exs.append({'seq': AUDIT_ROWS[0] + si, 'pos': pos,
                            'scale1': round(float(scales[f, 0]), 3),
                            'ctx': dec(ids[:-1]) + ' ==>' + dec(ids[-1:])})
            entries.append({'code': int(c), 'count_step1': int(c1[c]),
                            'mean_abs_scale1': round(float(
                                np.abs(scales[sel, 0]).mean()), 3),
                            'examples': exs})
        full[str(k)] = {'module': R2.stream_name(k + 1), 'codes': entries}
    rec_full = {
        'audit_slice': f'fresh34k[{AUDIT_ROWS[0]}:{AUDIT_ROWS[1]}] -- the '
                       'FIXED audit slice (same 200 seqs forever, standing '
                       'logging requirement)',
        'rep_slots': list(REP_SLOTS),
        'ranking': 'codes by step-1 usage; contexts by |scale1|; ctx shows '
                   'the 7 preceding tokens ==> the firing position\'s token',
        'per_slot': full}
    with open(E.jpath('qk_e20_code_dictionaries.json'), 'w') as f:
        json.dump(rec_full, f, indent=1)
    sample = {str(k): {'module': full[str(k)]['module'],
                       'top5_codes': [
                           {'code': e_['code'],
                            'count_step1': e_['count_step1'],
                            'examples': [x['ctx'] for x in
                                         e_['examples'][:3]]}
                           for e_ in full[str(k)]['codes'][:5]]}
              for k in REP_SLOTS}
    E.merge(JP, key, {'file': 'qk_e20_code_dictionaries.json',
                      'audit_slice': rec_full['audit_slice'],
                      'rep_slots': list(REP_SLOTS),
                      'sample_top5_per_slot': sample})
    print("code dictionaries recorded (full file "
          "qk_e20_code_dictionaries.json)", flush=True)


# ---------------- distillation control (reviewer-2 item 1; conditional) ----------------
def load_rows_range(a, b):
    """Rows [a:b) of the concatenated corpus_fresh shards (the training
    protocol used the [0:132000) prefix; rows beyond it are FRESH -- never
    seen by any arm; single-visit discipline preserved)."""
    rows, seen = [], 0
    for i in range(7):
        arr = np.load(f'{E.SHARD_DIR}/shard{i:02d}.npy', mmap_mode='r')
        lo, hi = max(a - seen, 0), min(b - seen, len(arr))
        if hi > lo:
            rows.append(np.asarray(arr[lo:hi]))
        seen += len(arr)
    out = np.concatenate(rows).astype(np.int64)
    assert len(out) == b - a, (len(out), a, b)
    return torch.from_numpy(out).to(E.DEV)


def distill_control(s_c):
    """If the scratch-trained quantized arm costs > +0.15 vs the E19a
    parent: initialize from the continuous parent checkpoint, data-init the
    codebooks from the parent's own slot content, quantize-and-finetune
    2000 steps on rows the protocol never used, and compare. Distilled <<
    scratch => the cost is TRAINABILITY, not expressivity; distilled ~=
    scratch => GRANULARITY (pre-registered decision tree, BRAINSTORM_STATE
    'Reviewer-2 findings')."""
    key = 'E20a_distill_control'
    j = E.loadj(JP)
    if E.SMOKE or key in j:
        return
    pair = j.get('E20a_minus_e19a_fresh')
    if pair is None or not os.path.exists(E.ckpath('qk_e19_a')):
        return
    scratch_cost = pair['minus_e19a']
    if scratch_cost <= 0.15:
        E.merge(JP, key, {
            'skipped': True,
            'scratch_cost_vs_e19a': round(scratch_cost, 4),
            'reason': 'scratch quantization cost <= +0.15 (PROMISING zone) '
                      '-- distillation control not needed per the '
                      'pre-registered decision tree'})
        print(f"distill control skipped (scratch cost "
              f"{scratch_cost:+.4f} <= +0.15)", flush=True)
        return
    steps, bsz = 2000, 16
    print(f"==== distill control: parent-init + quantize-and-finetune "
          f"{steps} steps (scratch cost {scratch_cost:+.4f}) ====",
          flush=True)
    rows = load_rows_range(132000, 132000 + steps * bsz)
    m = make_e20(s=s_c)
    ck = torch.load(E.ckpath('qk_e19_a'), map_location=E.DEV,
                    weights_only=False)
    missing, unexpected = m.load_state_dict(ck['state_dict'], strict=False)
    assert not unexpected and all(kk.startswith('qz_') for kk in missing), \
        (missing, unexpected)
    # data-init the codebooks from the PARENT's continuous slot content
    with torch.no_grad():
        m.eval().float()
        m.qz_on = False
        b = rows[:8]
        col = {'entry_norm': [], 'attn_write': [], 'mlp_write': []}
        m(b[:, :Q.T], collect=col)
        e = F.rms_norm(m.wte(b[:, :Q.T]), (m.Ws,))
        writes = []
        for jj in range(DEPTH):
            writes.append(col['attn_write'][jj])
            writes.append(col['mlp_write'][jj])
        g = torch.Generator().manual_seed(31337)
        for k in range(NG):
            sl = slice(m.s * k, m.s * (k + 1))
            cont = F.rms_norm(e[..., sl] + writes[k][..., sl],
                              (m.s,)).reshape(-1, m.s).float()
            ridx = torch.randperm(cont.shape[0], generator=g)[:QZ_N]
            cb = cont[ridx].to(m.qz_codebook.device)
            cb = cb / cb.norm(dim=1, keepdim=True).clamp_min(1e-8)
            m.qz_codebook[k] = cb
            m.qz_ema[k] = cb.clone()
        m.qz_on = True
    m.train()
    mu, dec_p, nod = E.muon_params_split(m)
    lr_mu, lr_ad, warm = 0.005, 0.001, 100
    opt_m = E.Muon(mu, lr=lr_mu)
    opt_a = torch.optim.AdamW([{'params': dec_p, 'weight_decay': Q.WD},
                               {'params': nod, 'weight_decay': 0.0}],
                              lr=lr_ad, betas=(0.9, 0.95))
    curve, spikes, run, t0 = [], 0, None, time.time()
    for st in range(steps):
        f_ = (st + 1) / warm if st < warm else 0.5 * (
            1 + math.cos(math.pi * (st - warm) / max(1, steps - warm)))
        for gpg in opt_m.param_groups:
            gpg['lr'] = lr_mu * f_
        for gpg in opt_a.param_groups:
            gpg['lr'] = lr_ad * f_
        seqs = rows[st * bsz:(st + 1) * bsz]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = m(seqs[:, :Q.T])
        ce = F.cross_entropy(logits.float().reshape(-1, Q.V),
                             seqs[:, 1:Q.T + 1].reshape(-1))
        loss = ce + GC20 * V8T.group_penalty(m)
        aux = m.pop_aux_loss()
        if aux is not None:
            loss = loss + aux
        l = ce.item()
        assert math.isfinite(l) and l < 30, ('distill diverged', st, l)
        opt_m.zero_grad(set_to_none=True)
        opt_a.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), Q.GRAD_CLIP)
        opt_m.step()
        opt_a.step()
        run = l if run is None else 0.98 * run + 0.02 * l
        if l > run + 1.0:
            spikes += 1
        if st % 200 == 0:
            curve.append([st, round(l, 4), round(run, 4)])
            print(f"  DISTILL step {st}/{steps} ce {l:.4f} (ema {run:.4f}) "
                  f"{time.time() - t0:.0f}s", flush=True)
    flush_dead_events(m, phase='distill')
    hce, pt = Q.eval_held(m, per_token=True)
    np.save(f'{E.QK}/qk_e20_distill_heldloss.npy', pt)
    np.save(f'{E.QK}/qk_e20_distill_heldloss_seq.npy',
            pt.reshape(len(Q.HELD), Q.T).mean(1))
    torch.save({'state_dict': m.state_dict(), 'variant': 'E20a_distill',
                'config': dict(steps=steps, lr_muon=lr_mu, lr_adamw=lr_ad,
                               group_coeff=GC20, warmup=warm,
                               rows='corpus_fresh shards [132000:164000)')},
               E.ckpath('qk_e20_distill'))
    usage = m.qz_usage.cpu().numpy()
    del m
    torch.cuda.empty_cache()
    pd = E.paired('qk_e20_distill_heldloss.npy', 'qk_e19_a_heldloss.npy',
                  len(Q.HELD), 'e19a')
    ps = E.paired('qk_e20_distill_heldloss.npy', 'qk_e20_a_heldloss.npy',
                  len(Q.HELD), 'e20a_scratch')
    distilled_cost = pd['minus_e19a']
    if distilled_cost < 0.5 * scratch_cost:
        verdict = (f'TRAINABILITY: distilled cost {round(distilled_cost, 4)}'
                   f' << scratch {round(scratch_cost, 4)} -- the CE cost is '
                   'trainability, not expressivity; anneal-in quantization '
                   'is the pre-registered next step')
    elif distilled_cost >= 0.8 * scratch_cost:
        verdict = (f'GRANULARITY: distilled cost {round(distilled_cost, 4)}'
                   f' ~= scratch {round(scratch_cost, 4)} -- n=256/k=2 is '
                   'genuinely too coarse; retry at n=1024 or k=4 per the '
                   'decision tree')
    else:
        verdict = (f'MIXED: distilled cost {round(distilled_cost, 4)} vs '
                   f'scratch {round(scratch_cost, 4)} (between the 0.5x and '
                   '0.8x pre-registered thresholds)')
    rec = {'held_ce_fresh_bf16': round(float(hce), 5),
           'scratch_cost_vs_e19a': round(scratch_cost, 4),
           'distilled_cost_vs_e19a': round(distilled_cost, 4),
           'paired_vs_e19a': pd, 'paired_vs_e20a_scratch': ps,
           'steps': steps, 'lr_muon': lr_mu, 'lr_adamw': lr_ad,
           'warmup': warm, 'spikes': spikes, 'train_curve_every200': curve,
           'rows': 'corpus_fresh shards [132000:164000) -- 32000 rows never '
                   'seen by any arm (single-visit discipline preserved)',
           'init': 'continuous E19a parent checkpoint; codebooks data-'
                   'initialized from the parent\'s own post-norm slot '
                   'content (random 256 vectors per slot, unit-normalized)',
           'distinct_codes_used_train_per_slot':
               {str(k): int((usage[k] > 0).sum()) for k in range(NG)},
           'verdict': verdict}
    E.merge(JP, key, rec)
    print(f"distill control: held CE {hce:.4f}; {verdict}", flush=True)


def summarize():
    j = E.loadj(JP)
    summary = {'parents': {
        'E19a_gc1e-4': {'ce': 4.9742, 'cov_composed': 0.8259,
                        'plain': 0.7911},
        'E15c_gc3e-5': {'ce': 4.9038, 'cov_composed': 0.6728,
                        'plain': 0.6294},
        'E9a_recipe': {'ce': 5.0547, 'cov_composed': 0.8575,
                       'plain': 0.7711}}}
    if 'E20a' in j:
        row = {'final_held_ce_fresh_bf16': j['E20a']['final_held_ce_fresh_bf16'],
               'diverged': j['E20a']['diverged']}
        if 'E20a_minus_e19a_fresh' in j:
            row['minus_e19a_paired'] = round(
                j['E20a_minus_e19a_fresh']['minus_e19a'], 4)
            row['minus_e19a_se_seq'] = round(
                j['E20a_minus_e19a_fresh']['minus_e19a_se_seq'], 5)
        if 'composed_wiring_E20a' in j:
            t = j['composed_wiring_E20a']['tables']
            row.update({
                'plain': t['plain']['spearman_all'],
                'cov_composed_quantized': t['cov_composed']['spearman_all'],
                'plain_effectual': t['plain']['spearman_effectual'],
                'cov_effectual': t['cov_composed']['spearman_effectual'],
                'plain_top10': t['plain']['top10_precision'],
                'cov_top10': t['cov_composed']['top10_precision']})
        if 'code_stats_E20a' in j:
            cs = j['code_stats_E20a']
            row['dead_code_fraction'] = cs['dead_code_fraction_overall']
            row['content_bits_per_token'] = \
                cs['content_bits_per_token_sum_joint_entropy']
        summary['E20a'] = row
        dce = row.get('minus_e19a_paired')
        if dce is None and row['final_held_ce_fresh_bf16'] is not None:
            dce = round(row['final_held_ce_fresh_bf16'] - 4.9742, 4)
        if dce is not None:
            if dce <= 0.15:
                va = (f'PROMISING: quantization CE cost {dce} vs E19a '
                      '<= +0.15 -- n=256/k=2 discrete content is nearly '
                      'free at this granularity')
            elif dce > 0.30:
                va = (f'REFUTED: quantization CE cost {dce} vs E19a > '
                      '+0.30 -- this granularity (n=256, k=2) is too coarse')
            else:
                va = (f'INTERMEDIATE: quantization CE cost {dce} vs E19a '
                      '-- between the +0.15 PROMISING and +0.30 REFUTED '
                      'thresholds')
            summary['prediction_a_verdict'] = va
        if 'dead_code_fraction' in row:
            df = row['dead_code_fraction']
            summary['prediction_b_verdict'] = (
                f"{'CONFIRMED' if df < 0.30 else 'REFUTED'}: dead-code "
                f"fraction {df} vs the < 0.30 registered threshold")
        if 'code_stats_E20a' in j and \
                'prediction_c_slack_vs_saturated' in j['code_stats_E20a']:
            summary['prediction_c_verdict'] = \
                j['code_stats_E20a']['prediction_c_slack_vs_saturated'][
                    'verdict']
        if 'E20a_distill_control' in j and \
                'verdict' in j['E20a_distill_control']:
            summary['distill_control_verdict'] = \
                j['E20a_distill_control']['verdict']
    E.merge(JP, 'summary_E20', summary)
    print(json.dumps({'summary_E20': summary}, indent=2), flush=True)


if __name__ == '__main__':
    E.setup()
    control_e18_gates()

    s_c, tgt = E15R.solve_slot_c(4 * Q.D)
    if not E.SMOKE:
        assert s_c == 15, s_c                     # the E15c/E19a slot size

    # ---- hard-gate controls (before training) ----
    control_bypass(s_c)
    control_capacity(s_c)
    control_toy()

    # ---- registered predictions (before training) ----
    if 'E20_prediction' not in E.loadj(JP):
        E.merge(JP, 'E20_prediction', {
            'registered_before_training': True,
            'design': 'E20a codebook slots: per-slot codebook n=256 '
                      'unit-norm 15-dim codes, k=2 matching-pursuit '
                      'selection (scales = inner products, continuous), '
                      'straight-through gradients, EMA decay 0.99, '
                      'commitment beta 0.25, dead-code reinit after 200 '
                      'unused steps -- on the E19a frontier arm (E15c '
                      'bandwidth-reinvestment architecture, group-lasso '
                      '1e-4, full readable recipe)',
            'a_ce_cost': 'fresh held CE minus the E19a parent (4.9742): '
                         '<= +0.15 PROMISING; > +0.30 REFUTES this '
                         'granularity (n=256, k=2)',
            'b_dead_codes': 'dead-code fraction (codes never used on the '
                            'full fresh held pass, codebooks 0-22; '
                            'codebook 23 causally inert and excluded) '
                            '< 30% after training',
            'c_slack_vs_saturated': 'modules flagged slack in the E14 '
                                    'census (qk_e14.json census_e9a: mlp1, '
                                    'attn2, attn10) will use fewer distinct '
                                    'codes / lower usage entropy than '
                                    'saturated modules; tested as '
                                    'mean joint-usage-entropy(slack) < '
                                    'mean(saturated) AND Spearman(census '
                                    'write_cov_effective_rank, per-slot '
                                    'joint usage entropy) > 0 over slots '
                                    '0-22',
            'parent_reference': {
                'E19a': {'ce': 4.9742, 'cov': 0.8259, 'plain': 0.7911},
                'E15c': {'ce': 4.9038, 'cov': 0.6728},
                'E9a_recipe': {'ce': 5.0547, 'cov': 0.8575}}})

    if 'E20_config' not in E.loadj(JP):
        E.merge(JP, 'E20_config', {
            'group_coeff': GC20, 'muon_lr': E7R.muon_lr(),
            'adamw_lr': E.get_lr(), 'slot': s_c,
            'quantizer': {'n_codes': QZ_N, 'k_select': QZ_K,
                          'ema_decay': QZ_DECAY, 'commit_beta': QZ_BETA,
                          'dead_reinit_steps': QZ_DEAD,
                          'codes': 'unit-norm; scales = inner products '
                                   '(continuous)'},
            'quantized': 'every module-written slot\'s post-per-slot-'
                         'RMSNorm content at every block-level read (hn '
                         'and xn); slot content includes the bottom-'
                         'injected embedding component (not separately '
                         'routed in E15c); one quantization per slot per '
                         'forward (content constant across consumers)',
            'not_quantized': 'not-yet-written slots (pure embedding '
                             'content); the readout (continuous stream '
                             'through the global RMSNorm) -- hence slot 23 '
                             '(mlp11 write, readout-only) never passes '
                             'through a codebook',
            'commitment': 'injected via qk_e_common.train_muon '
                          'pop_aux_loss hook; NOT in final_penalty',
            'factory': 'E20Route subclass of qk_e15_reinvest_run.E15cRoute '
                       '(make_e15c(s=15) config verbatim: true-small '
                       'decoders, 24 x 15-dim slots, stream 360, compute '
                       '264, hidden 1056)'})

    # ---- train ----
    E.train_arm('qk_e20_a', JP, 'E20a', lambda: make_e20(s=s_c), GC20,
                lr=E7R.muon_lr(), trainer=trainer,
                extra={'optimizer': 'muon', 'slot': s_c,
                       'parent': 'E19a (qk_e19_a, E15c arch at gc 1e-4)',
                       'design': 'E19a architecture + per-slot codebook '
                                 'quantization (n=256, k=2 matching '
                                 'pursuit, straight-through, EMA + '
                                 'commitment beta 0.25 + dead-code reinit)',
                       'logging': 'codebook snapshots every 1000 steps -> '
                                  'qk_e20_codebook_snapshots.npz; dead-code '
                                  'events -> qk_e20_deadcode_events.jsonl '
                                  '(flushed every 250 steps; final <250 '
                                  'steps unflushed)'})
    E.oldheld_record('qk_e20_a', lambda: make_e20(s=s_c), JP, 'E20a_oldheld')
    E.paired_fresh('qk_e20_a', JP, 'E20a')
    pair_extra('qk_e20_a', 'E20a', (('qk_e19_a', 'e19a'),
                                    ('qk_e15_c', 'e15c'),
                                    ('qk_e9_a', 'e9a')))
    save_seq_heldloss('qk_e20_a')

    # ---- reviewer-2 conditional distillation control ----
    distill_control(s_c)

    # ---- measurement ----
    code_stats(s_c)
    code_dictionaries(s_c)
    if not E.SMOKE:
        control_cov_pipeline(s_c)
    probe_e20a(s_c)

    summarize()
    print('e20 codebook run done', flush=True)
