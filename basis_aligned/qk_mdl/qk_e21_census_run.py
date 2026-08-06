"""E21: predicate census over the slotted width-264 models (Logan's
architecture-brainstorm question: do our slotted models already contain heads
nameable by the selection predicates the bilin18 census used? seeds the
predicate-library architecture idea and tests whether the census ports).

CHECKPOINT-ONLY DIAGNOSTIC (no training). Models:
  qk_e9_a.pt  -- the readable recipe (E7R.make_e7m1: per-slot RMSNorm, 24x11
                 slots, group-lasso 3e-5, Muon)
  qk_e19_a.pt -- bandwidth+1e-4, frontier-best (E15R.make_e15c(s=15): 24x15
                 slots Ws=360, true-small decoders)
Both 12-block x 6-head bilinear attention: pattern = (q1k1/HD)*(q2k2/HD),
causal mask, NO softmax (signed patterns).

For every head (12 x 6 x 2): pattern on 200 fresh held sequences
(fresh34k rows [33000:33200], never trained on), scored against a predicate
library (the bilin18 selection-census family):
  MATCH_same   1[tok_j == tok_i]              (same-token; diagonal included)
  MATCH_prev   1[tok_{j-1} == tok_i]          (classic induction target)
  MATCH_next   1[tok_{j+1} == tok_i], j<i     (the brief's literal reading of
                                               "positions whose NEXT token
                                               equals the query token";
                                               diagonal excluded -- it would
                                               use the future token)
  PREV_token   1[j == i-1]
  FIRST_token  1[j == 0]
  KEY_newline / KEY_punct / KEY_func / KEY_cap / KEY_digit  (key-token class)
  POSITIONAL_decay  per-offset scalar profile prof[i-j] (LS fit)
Score = held-out uncentered R^2: scalar coefficient (or offset profile) fit on
sequences 0:100, score = 1 - sum((P - pred)^2)/sum(P^2) on sequences 100:200,
over causal-masked entries. This is "fraction of the head's pattern (squared)
mass explained" and reduces to the squared normalized inner product when fit
and eval distributions agree. Additionally a 2-parameter joint fit
(predicate + positional profile) gives predicate_gain = R^2(joint) -
R^2(positional), the analogue of the bilin18 census's "additional variance
beyond the positional template" (threshold 0.05 = programmatic).

POSITIVE CONTROLS (run first, hard asserts):
  (a) forward fidelity: the reimplemented pattern-collecting forward matches
      each model's own forward to 1e-5 max |logit diff| on one batch (tf32 off);
  (b) synthetic EXACT previous-token pattern scored against the library:
      PREV_token ~1.0 and POSITIONAL_decay ~1.0 (documented overlap; offset
      profile must concentrate >99% of its explained mass at offset 1);
  (c) synthetic MATCH_same pattern: MATCH_same ~1.0 (catches query/key axis
      transposition bugs the symmetric control cannot).

CAUSAL check on the top-5 best-scoring heads per model (+ up to 2 extra top
selection-predicate heads): replace the head's pattern with the pure predicate
pattern (renormalized per query row to the head's signed row mass; also the
global-LS-coefficient version) and measure held CE change vs mean-pattern
ablation (fit-half mean pattern) of the same head. If the name is right, the
predicate substitution should be much cheaper than the ablation.

Output: qk_e21_census.json. GPU etiquette: polls up to 3 h for >=2500 MiB
free before importing the cuda-touching harness; falls back to CPU (small
models, feasible) if it never frees. Width-generic: everything is read off
the loaded model / Q dims, nothing hard-codes 264.
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json
import math
import subprocess
import time


def _gpu_free_mib():
    try:
        return int(subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free',
             '--format=csv,noheader,nounits']).decode().split('\n')[0])
    except Exception:
        return -1


# ---- polite pre-import GPU wait (harness import touches cuda) ----
USE_CUDA = True
_t0 = time.time()
while True:
    _free = _gpu_free_mib()
    if _free < 0:
        USE_CUDA = False
        break
    if _free >= 2500:
        print(f"gpu: {_free} MiB free -- using cuda", flush=True)
        break
    if time.time() - _t0 > 3 * 3600:
        print("gpu: never freed within 3 h -- running on CPU", flush=True)
        USE_CUDA = False
        break
    print(f"gpu: only {_free} MiB free -- waiting", flush=True)
    time.sleep(60)

import torch                                     # noqa: E402

if not USE_CUDA:
    # SMOKE-style cuda->cpu redirect (qk_e_common convention) so the harness
    # modules (which hard-code DEV='cuda') import cleanly on a busy card.
    def _cpu_dev(a):
        if isinstance(a, str) and a.startswith('cuda'):
            return 'cpu'
        if isinstance(a, torch.device) and a.type == 'cuda':
            return torch.device('cpu')
        return a
    _t_to = torch.Tensor.to

    def _tto(self, *args, **kw):
        args = tuple(_cpu_dev(a) for a in args)
        if 'device' in kw:
            kw['device'] = _cpu_dev(kw['device'])
        return _t_to(self, *args, **kw)
    torch.Tensor.to = _tto
    _m_to = torch.nn.Module.to

    def _mto(self, *args, **kw):
        args = tuple(_cpu_dev(a) for a in args)
        if 'device' in kw:
            kw['device'] = _cpu_dev(kw['device'])
        return _m_to(self, *args, **kw)
    torch.nn.Module.to = _mto

import numpy as np                               # noqa: E402

# Neutralize the import-time BLOCKING gpu_guard (checkpoint-only diagnostic;
# same rationale as qk_e17/qk_e18).
import qk_tokenline_train as _Q0                 # noqa: E402
_Q0.gpu_guard = lambda *a, **k: None

import qk_e_common as E                          # noqa: E402
from qk_e_common import Q, DEPTH, F              # noqa: E402
import qk_e7_evenout_run as E7R                  # noqa: E402
import qk_e15_reinvest_run as E15R               # noqa: E402
# E17 import runs the minimal width patch (W.patch_width -> 264/6/44/slot 11)
# and sets E.DEV='cpu'; DEV is set for real below.
import qk_e17_composed_wiring as E17             # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

DEV = 'cuda' if USE_CUDA else 'cpu'
E.DEV = DEV
QK = Q.QK
JOUT = f'{QK}/qk_e21_census.json'
T = Q.T                                          # 512
NH = Q.NH                                        # 6
NL = DEPTH                                       # 12
BP = 4                                           # pattern micro-batch
ROWS = (33000, 33200)                            # 200 fresh held seqs
N_FIT = 100                                      # fit half; eval = the rest

FEATN = ['MATCH_same', 'MATCH_prev', 'MATCH_next', 'PREV_token',
         'FIRST_token', 'KEY_newline', 'KEY_punct', 'KEY_func', 'KEY_cap',
         'KEY_digit']
NF = len(FEATN)
POS = 'POSITIONAL_decay'


# ---------------- token classes (gpt2 byte-level, v2-census convention) -----
def token_classes():
    from transformers import AutoTokenizer
    import string as _string
    tok = AutoTokenizer.from_pretrained('gpt2')
    P = set(_string.punctuation)
    FUNC = {'the', 'of', 'and', 'to', 'a', 'in', 'is', 'that', 'it', 'for',
            'was', 'as', 'with', 'on', 'be', 'at', 'by', 'this', 'are',
            'from', 'or', 'an', 'but', 'not', 'which'}
    V = Q.V
    KN = torch.zeros(V, dtype=torch.bool)
    KP = torch.zeros(V, dtype=torch.bool)
    KF = torch.zeros(V, dtype=torch.bool)
    KC = torch.zeros(V, dtype=torch.bool)
    KD = torch.zeros(V, dtype=torch.bool)
    for i in range(V):
        s = tok.convert_ids_to_tokens(i)
        if s is None:
            continue
        core = s.replace('Ġ', '')
        lead = s.startswith('Ġ')
        if 'Ċ' in s:
            KN[i] = True
        if len(core) and all(c in P for c in core):
            KP[i] = True
        if core.lower() in FUNC:
            KF[i] = True
        if lead and len(core) and core[0].isupper():
            KC[i] = True
        if len(core) and all(c.isdigit() for c in core):
            KD[i] = True
    return [k.to(DEV) for k in (KN, KP, KF, KC, KD)]


KCLS = None                                      # set in main


def build_feats(idx, mask):
    """(B, NF, T, T) causal-masked 0/1 predicate features. Entry [i, j]:
    i = query position (dim 2), j = key position (dim 3)."""
    B, Tq = idx.shape
    prevtok = torch.roll(idx, 1, 1)
    prevtok[:, 0] = -1
    nexttok = torch.roll(idx, -1, 1)
    nexttok[:, -1] = -1
    Fs = torch.zeros(B, NF, Tq, Tq, device=idx.device)
    qtok = idx.unsqueeze(2)                       # (B, T, 1): query along dim 1
    Fs[:, 0] = (idx.unsqueeze(1) == qtok).float()       # MATCH_same
    Fs[:, 1] = (prevtok.unsqueeze(1) == qtok).float()   # MATCH_prev
    mnext = (nexttok.unsqueeze(1) == qtok).float()      # MATCH_next
    mnext.diagonal(dim1=1, dim2=2).zero_()              # j==i would use future
    Fs[:, 2] = mnext
    Fs[:, 3] = torch.diag(torch.ones(Tq - 1, device=idx.device), -1)
    Fs[:, 4, :, 0] = 1.0                                # FIRST_token
    for fi, cls in enumerate(KCLS):
        Fs[:, 5 + fi] = cls[idx].float().unsqueeze(1).expand(B, Tq, Tq)
    return Fs * mask


# ---------------- pattern-collecting / pattern-substituting forwards --------
@torch.no_grad()
def forward_e9(m, idx, layer_cb=None, pat_hook=None):
    """Verbatim E1Route.forward (per-slot RMSNorm recipe) with a pattern
    callback/override. Returns tanh-30 logits."""
    B, Tq = idx.shape
    Dm = m.wte.weight.shape[1]
    NHm, HDm = Q.NH, Q.HD
    e = F.rms_norm(m.wte(idx), (Dm,))
    streams = [e]
    cos = m.cos[None, :Tq, None, :]
    sin = m.sin[None, :Tq, None, :]
    mask = m.mask[:Tq, :Tq]
    for l, blk in enumerate(m.h):
        x = m.assemble(l, streams, None, None)
        hn = m.slot_norm(x)

        def qk(lin):
            z = lin(hn).view(B, Tq, NHm, HDm)
            return Q.apply_rot(F.rms_norm(z, (HDm,)), cos, sin)

        q, k = qk(blk.c_q), qk(blk.c_k)
        q2, k2 = qk(blk.c_q2), qk(blk.c_k2)
        v = blk.c_v(hn).view(B, Tq, NHm, HDm)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HDm
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HDm
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        if pat_hook is not None:
            pat = pat_hook(l, pat)
        if layer_cb is not None:
            layer_cb(l, pat)
        y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, Tq, Dm)
        aw = blk.c_proj(y)
        if m.proj:
            aw = aw * m.wmask[2 * l].to(aw.dtype)
        x = x + aw
        xn = m.slot_norm(x)
        mw = blk.Down(blk.Left(xn) * blk.Right(xn)) + blk.Down_bias
        if m.proj:
            mw = mw * m.wmask[2 * l + 1].to(mw.dtype)
        streams.append(aw)
        streams.append(mw)
    x = m.assemble(m.depth, streams, None, None)
    x = F.rms_norm(x, (Dm,))
    logits = x @ m.wte.weight.t()
    return 30 * torch.tanh(logits / 30)


@torch.no_grad()
def forward_e19(m, idx, layer_cb=None, pat_hook=None):
    """Verbatim E15cRoute.forward (bandwidth reinvestment, Ws=24*s) with a
    pattern callback/override."""
    B, Tq = idx.shape
    Ws, s = m.Ws, m.s
    cos = m.cos[None, :Tq, None, :]
    sin = m.sin[None, :Tq, None, :]
    mask = m.mask[:Tq, :Tq]
    e = F.rms_norm(m.wte(idx), (Ws,))
    streams = [e]

    def entry(li):
        tot = None
        for i in m.vis[li]:
            tot = streams[i] if tot is None else tot + streams[i]
        return tot

    for l, blk in enumerate(m.h):
        x = entry(l)
        hn = m.slot_norm(x)

        def qk(lin):
            z = lin(hn).view(B, Tq, m.NH, m.HD)
            return Q.apply_rot(F.rms_norm(z, (m.HD,)), cos, sin)

        q, k = qk(blk.c_q), qk(blk.c_k)
        q2, k2 = qk(blk.c_q2), qk(blk.c_k2)
        v = blk.c_v(hn).view(B, Tq, m.NH, m.HD)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / m.HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / m.HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        if pat_hook is not None:
            pat = pat_hook(l, pat)
        if layer_cb is not None:
            layer_cb(l, pat)
        y = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        y = y.reshape(B, Tq, m.Dc)
        aw = torch.zeros(B, Tq, Ws, device=y.device, dtype=y.dtype)
        aw[..., s * 2 * l:s * (2 * l + 1)] = blk.c_proj(y)
        x = x + aw
        xn = m.slot_norm(x)
        mw_s = blk.Down(blk.Left(xn) * blk.Right(xn)) + blk.Down_bias
        mw = torch.zeros(B, Tq, Ws, device=y.device, dtype=y.dtype)
        mw[..., s * (2 * l + 1):s * (2 * l + 2)] = mw_s
        streams.append(aw)
        streams.append(mw)
    x = F.rms_norm(entry(m.depth), (Ws,))
    logits = x @ m.wte.weight.t()
    return 30 * torch.tanh(logits / 30)


# ---------------- streaming score accumulators ----------------
class Stats:
    """Per-half sufficient statistics for the predicate/positional fits."""

    def __init__(self, nl, nh):
        self.pp = torch.zeros(nl, nh, dtype=torch.float64, device=DEV)
        self.dot = torch.zeros(nl, nh, NF, dtype=torch.float64, device=DEV)
        self.bb = torch.zeros(NF, dtype=torch.float64, device=DEV)
        self.offsum = torch.zeros(nl, nh, T, dtype=torch.float64, device=DEV)
        self.boff = torch.zeros(NF, T, dtype=torch.float64, device=DEV)
        self.nseq = 0

    def offcnt(self):
        ar = torch.arange(T, device=DEV, dtype=torch.float64)
        return self.nseq * (T - ar)

    def update(self, l, pat, Fs, off_flat):
        p = pat.float()
        self.pp[l] += p.pow(2).sum((0, 2, 3)).double()
        self.dot[l] += torch.einsum('bhqk,bfqk->hf', p, Fs).double()
        src = p.sum(0).reshape(p.shape[1], -1).double()
        self.offsum[l].index_add_(1, off_flat, src)
        if l == 0:                                # batch-level, once per batch
            self.bb += Fs.pow(2).sum((0, 2, 3)).double()
            bsrc = Fs.sum(0).reshape(NF, -1).double()
            self.boff.index_add_(1, off_flat, bsrc)
            self.nseq += pat.shape[0]


def score_heads(fit, ev):
    """Held-out scores from the two halves' statistics. Returns dict of
    (NL, NH[, NF]) tensors: per-predicate score, positional score, joint
    predicate+positional score, predicate gain, plus fit coefs and profile."""
    cnt_f, cnt_e = fit.offcnt(), ev.offcnt()
    coef = fit.dot / fit.bb.clamp(min=1e-9)                  # (L,H,NF)
    prof = fit.offsum / cnt_f.clamp(min=1e-9)                # (L,H,T)
    # per-predicate held-out uncentered R^2
    res = ev.pp[..., None] - 2 * coef * ev.dot + coef ** 2 * ev.bb
    s_pred = 1 - res / ev.pp[..., None].clamp(min=1e-12)
    # positional profile held-out R^2
    dotprof_e = (prof * ev.offsum).sum(-1)                   # (L,H)
    profss_e = (prof ** 2 * cnt_e).sum(-1)
    s_pos = 1 - (ev.pp - 2 * dotprof_e + profss_e) / ev.pp.clamp(min=1e-12)
    # 2-parameter joint fit (predicate + profile) on fit half, scored on eval
    dotprof_f = (prof * fit.offsum).sum(-1)                  # (L,H)
    profss_f = (prof ** 2 * cnt_f).sum(-1)
    cross_f = torch.einsum('lht,ft->lhf', prof, fit.boff)    # sum B*prof, fit
    cross_e = torch.einsum('lht,ft->lhf', prof, ev.boff)
    det = fit.bb[None, None] * profss_f[..., None] - cross_f ** 2
    det = det.clamp(min=1e-9)
    a = (profss_f[..., None] * fit.dot
         - cross_f * dotprof_f[..., None]) / det             # predicate coef
    b = (fit.bb[None, None] * dotprof_f[..., None]
         - cross_f * fit.dot) / det                          # profile coef
    res_j = (ev.pp[..., None] - 2 * (a * ev.dot + b * dotprof_e[..., None])
             + a ** 2 * ev.bb + 2 * a * b * cross_e
             + b ** 2 * profss_e[..., None])
    s_joint = 1 - res_j / ev.pp[..., None].clamp(min=1e-12)
    return {'s_pred': s_pred, 's_pos': s_pos, 's_joint': s_joint,
            'gain': s_joint - s_pos[..., None], 'coef': coef, 'prof': prof,
            'joint_a': a, 'joint_b': b}


# ---------------- shuffled-token null (Reviewer-2 R8) ----------------
NULL_K = 20
NULL_FEATS = [0, 1, 2, 5, 6, 7, 8, 9]            # token-dependent predicates
# (PREV_token / FIRST_token are position-only: invariant under token shuffle,
# null sd exactly 0, so no z is defined for them.)


@torch.no_grad()
def null_zscores(m, fwd, held):
    """Per-head z-scores of each token-dependent predicate's explained-mass
    fraction (cos^2 on the eval half) against a null where the key-token
    assignment is shuffled WITHIN each sequence (preserves the sequence's
    token multiset, so e.g. MATCH_same keeps its natural repetition base
    rate) and the predicate features are recomputed. Patterns are the real
    model patterns throughout. K = NULL_K shuffles."""
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=DEV)).float()
    g = torch.Generator().manual_seed(12345)
    pp = torch.zeros(NL, NH, dtype=torch.float64, device=DEV)
    dot = torch.zeros(NULL_K + 1, NL, NH, NF, dtype=torch.float64, device=DEV)
    bb = torch.zeros(NULL_K + 1, NF, dtype=torch.float64, device=DEV)
    for i in range(N_FIT, held.shape[0], BP):
        idx = held[i:i + BP, :T]
        B = idx.shape[0]
        fs = [build_feats(idx, mask)]
        for k in range(NULL_K):
            perm = torch.stack([torch.randperm(T, generator=g)
                                for _ in range(B)]).to(DEV)
            fs.append(build_feats(torch.gather(idx, 1, perm), mask))
        for k in range(NULL_K + 1):
            bb[k] += fs[k].pow(2).sum((0, 2, 3)).double()

        def cb(l, pat, fs=fs):
            p = pat.float()
            pp[l] += p.pow(2).sum((0, 2, 3)).double()
            for k in range(NULL_K + 1):
                dot[k, l] += torch.einsum('bhqk,bfqk->hf', p, fs[k]).double()
        fwd(m, idx, layer_cb=cb)
    # cos^2 = fraction of pattern squared mass explained by the predicate
    c2 = dot ** 2 / (pp[None, ..., None] * bb[:, None, None, :]).clamp(
        min=1e-12)
    real, null = c2[0], c2[1:]
    mu, sd = null.mean(0), null.std(0)
    z = (real - mu) / sd.clamp(min=1e-12)
    return real, z, mu, sd


# ---------------- controls ----------------
def control_synthetic(held):
    """Score two known-answer synthetic patterns through the exact scoring
    machinery: (b) previous-token delta, (c) same-token match."""
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=DEV))
    ar = torch.arange(T, device=DEV)
    off_flat = (ar[:, None] - ar[None, :]).clamp(min=0).reshape(-1)
    prev_pat = torch.diag(torch.ones(T - 1, device=DEV), -1)
    halves = []
    for lo, hi in ((0, N_FIT), (N_FIT, held.shape[0])):
        st = Stats(1, 2)
        for i in range(lo, hi, BP):
            idx = held[i:i + BP, :T]
            Fs = build_feats(idx, mask.float())
            B = idx.shape[0]
            same_pat = (idx.unsqueeze(1) == idx.unsqueeze(2)).float() * mask
            pat = torch.stack(
                [prev_pat.expand(B, T, T), same_pat], 1)     # (B, 2, T, T)
            st.update(0, pat, Fs, off_flat)
        halves.append(st)
    sc = score_heads(*halves)
    prof_prev = sc['prof'][0, 0]
    cnt = halves[0].offcnt()
    expl = prof_prev ** 2 * cnt
    conc1 = float(expl[1] / expl.sum().clamp(min=1e-12))
    s_prev = {FEATN[f]: round(float(sc['s_pred'][0, 0, f]), 4)
              for f in range(NF)}
    s_prev[POS] = round(float(sc['s_pos'][0, 0]), 4)
    s_same = {FEATN[f]: round(float(sc['s_pred'][0, 1, f]), 4)
              for f in range(NF)}
    s_same[POS] = round(float(sc['s_pos'][0, 1]), 4)
    print(f"control(b) exact prev-token pattern: {s_prev}", flush=True)
    print(f"          profile concentration at offset 1: {conc1:.6f}",
          flush=True)
    print(f"control(c) exact same-token pattern: {s_same}", flush=True)
    assert s_prev['PREV_token'] > 0.99, s_prev
    assert s_prev[POS] > 0.99, s_prev
    assert conc1 > 0.99, conc1
    assert all(v < 0.5 for k, v in s_prev.items()
               if k not in ('PREV_token', POS)), s_prev
    assert s_same['MATCH_same'] > 0.99, s_same
    best_same = max(s_same, key=s_same.get)
    assert best_same == 'MATCH_same', s_same
    return {'prev_pattern_scores': s_prev,
            'prev_profile_concentration_offset1': round(conc1, 6),
            'same_pattern_scores': s_same,
            'note': ('PREV_token and POSITIONAL_decay overlap by construction '
                     '(offset -1 is also positional); disambiguated by the '
                     'offset profile being a delta at offset 1.')}


# ---------------- census over one model ----------------
def census_model(name, m, fwd, held):
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=DEV)).float()
    ar = torch.arange(T, device=DEV)
    off_flat = (ar[:, None] - ar[None, :]).clamp(min=0).reshape(-1)
    tpl = torch.zeros(NL, NH, T, T, device=DEV)
    halves = []
    for hidx, (lo, hi) in enumerate(((0, N_FIT), (N_FIT, held.shape[0]))):
        st = Stats(NL, NH)
        for i in range(lo, hi, BP):
            idx = held[i:i + BP, :T]
            Fs = build_feats(idx, mask)

            def cb(l, pat, Fs=Fs, st=st, hidx=hidx):
                st.update(l, pat, Fs, off_flat)
                if hidx == 0:
                    tpl[l] += pat.float().sum(0)
            fwd(m, idx, layer_cb=cb)
        halves.append(st)
        print(f"{name}: half {hidx} ({hi - lo} seqs) done", flush=True)
    tpl /= halves[0].nseq
    sc = score_heads(*halves)
    cnt_f = halves[0].offcnt()
    table = []
    for l in range(NL):
        for h in range(NH):
            expl = sc['prof'][l, h] ** 2 * cnt_f
            etot = float(expl.sum().clamp(min=1e-12))
            top_off = torch.argsort(expl, descending=True)[:3]
            prof_top = [[int(o), round(float(sc['prof'][l, h, o]), 5),
                         round(float(expl[o]) / etot, 3)] for o in top_off]
            scores = {FEATN[f]: round(float(sc['s_pred'][l, h, f]), 4)
                      for f in range(NF)}
            scores[POS] = round(float(sc['s_pos'][l, h]), 4)
            top2 = sorted(scores, key=scores.get, reverse=True)[:2]
            gains = {FEATN[f]: round(float(sc['gain'][l, h, f]), 4)
                     for f in range(NF)}
            gtop = max(gains, key=gains.get)
            table.append({
                'layer': l, 'head': h,
                'best': top2[0], 'best_score': scores[top2[0]],
                'second': top2[1], 'second_score': scores[top2[1]],
                'best_coef': (None if top2[0] == POS else
                              round(float(sc['coef'][l, h,
                                                     FEATN.index(top2[0])]),
                                    5)),
                'positional_score': scores[POS],
                'profile_top_offsets': prof_top,
                'predicate_gain_top': gtop,
                'predicate_gain': gains[gtop],
                'scores': scores})
    return table, sc, tpl


def summarize(table):
    from collections import Counter
    best = [r['best_score'] for r in table]
    sel_scores = [max(v for k, v in r['scores'].items() if k != POS)
                  for r in table]
    prog = [r for r in table if r['predicate_gain'] >= 0.05]
    return {
        'n_heads': len(table),
        'best_gt_05': sum(s > 0.5 for s in best),
        'best_gt_03': sum(s > 0.3 for s in best),
        'best_selection_gt_05': sum(s > 0.5 for s in sel_scores),
        'best_selection_gt_03': sum(s > 0.3 for s in sel_scores),
        'best_predicate_hist_score_gt_03': dict(Counter(
            r['best'] for r in table if r['best_score'] > 0.3)),
        'programmatic_gain_ge_005': len(prog),
        'programmatic_hist': dict(Counter(
            r['predicate_gain_top'] for r in prog)),
        'programmatic_layers': sorted({r['layer'] for r in prog}),
    }


# ---------------- causal check ----------------
@torch.no_grad()
def perseq_ce(m, fwd, held, pat_hook=None):
    """Per-sequence mean CE on the eval half."""
    out = []
    for i in range(N_FIT, held.shape[0], BP):
        idx = held[i:i + BP]
        lg = fwd(m, idx[:, :T], pat_hook=pat_hook)
        ce = F.cross_entropy(lg.float().reshape(-1, Q.V),
                             idx[:, 1:T + 1].reshape(-1), reduction='none')
        out.append(ce.view(idx.shape[0], T).mean(1).cpu())
    return torch.cat(out)


def causal_checks(name, m, fwd, held, table, sc, tpl):
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=DEV)).float()
    prev_pat = torch.diag(torch.ones(T - 1, device=DEV), -1)
    byscore = sorted(table, key=lambda r: -r['best_score'])
    picks = byscore[:5]
    tags = {(r['layer'], r['head']): 'top5_overall' for r in picks}
    bysel = sorted(table, key=lambda r: -max(
        v for k, v in r['scores'].items() if k != POS))
    for r in bysel:
        if len(picks) >= 7:
            break
        if r not in picks:
            picks.append(r)
            tags[(r['layer'], r['head'])] = 'top2_selection'
    # Reviewer-2 R8: 2 RANDOM mid-scoring heads as causal controls (guards
    # against selection bias in the top picks)
    mid = byscore[len(byscore) // 3: 2 * len(byscore) // 3]
    rng = np.random.RandomState(0)
    for j in rng.choice(len(mid), 2, replace=False):
        r = mid[int(j)]
        if r not in picks:
            picks.append(r)
            tags[(r['layer'], r['head'])] = 'random_mid'
    ce0 = perseq_ce(m, fwd, held)
    print(f"{name}: baseline eval-half CE {ce0.mean():.4f}", flush=True)
    results = []
    for r in picks:
        l0, h0, pred = r['layer'], r['head'], r['best']
        fidx = None if pred == POS else FEATN.index(pred)
        prof_lh = sc['prof'][l0, h0].float()
        coef_lh = None if fidx is None else float(sc['coef'][l0, h0, fidx])
        ar = torch.arange(T, device=DEV)
        profpat = prof_lh[(ar[:, None] - ar[None, :]).clamp(min=0)] * mask

        def predicate_B(idx):
            if fidx is None:
                return profpat.expand(idx.shape[0], T, T)
            if pred == 'PREV_token':
                return prev_pat.expand(idx.shape[0], T, T)
            return build_feats(idx, mask)[:, fidx]

        def hook_factory(mode):
            def hook(l, pat, mode=mode):
                if l != l0:
                    return pat
                pat = pat.clone()
                if mode == 'ablate':
                    pat[:, h0] = tpl[l0, h0]
                else:
                    Bf = predicate_B(hook.idx)
                    if mode == 'rowmass':
                        rowP = pat[:, h0].sum(-1)
                        rowB = Bf.sum(-1)
                        scale = torch.where(rowB.abs() > 1e-6, rowP / rowB,
                                            torch.zeros_like(rowB))
                        pat[:, h0] = Bf * scale.unsqueeze(-1)
                    else:                          # global LS coefficient
                        pat[:, h0] = Bf if fidx is None else coef_lh * Bf
                return pat
            return hook

        row = {'model': name, 'layer': l0, 'head': h0, 'predicate': pred,
               'predicate_score': r['best_score'],
               'pick': tags[(l0, h0)]}
        for mode, key in (('rowmass', 'dce_pred_rowmass'),
                          ('lscoef', 'dce_pred_lscoef'),
                          ('ablate', 'dce_ablate')):
            hk = hook_factory(mode)
            ces = []
            for i in range(N_FIT, held.shape[0], BP):
                idx = held[i:i + BP]
                hk.idx = idx[:, :T]
                lg = fwd(m, idx[:, :T], pat_hook=hk)
                ce = F.cross_entropy(lg.float().reshape(-1, Q.V),
                                     idx[:, 1:T + 1].reshape(-1),
                                     reduction='none')
                ces.append(ce.view(idx.shape[0], T).mean(1).cpu())
            ces = torch.cat(ces)
            d = ces - ce0
            row[key] = round(float(d.mean()), 5)
            row[key + '_se'] = round(float(d.std() / math.sqrt(len(d))), 5)
        row['ce_base'] = round(float(ce0.mean()), 4)
        row['verdict_pred_cheaper'] = bool(
            min(row['dce_pred_rowmass'], row['dce_pred_lscoef'])
            < row['dce_ablate'])
        print(f"  L{l0}H{h0} [{pred} {row['predicate_score']}]: "
              f"pred(rowmass) {row['dce_pred_rowmass']:+.4f} "
              f"pred(lscoef) {row['dce_pred_lscoef']:+.4f} "
              f"ablate {row['dce_ablate']:+.4f}", flush=True)
        results.append(row)
    return results


def causal_joint_checks(name, m, fwd, held, table, sc, tpl):
    """The match/copy question, causally: for each programmatic head
    (predicate_gain >= 0.045), substitute (i) the positional-profile-only
    pattern, (ii) the joint coded pattern a*predicate + b*profile (the
    2-parameter fit the gain came from), (iii) the fit-half mean pattern
    (ablation). If the predicate component is causally real, joint must be
    cheaper than profile-only."""
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=DEV)).float()
    ar = torch.arange(T, device=DEV)
    offmat = (ar[:, None] - ar[None, :]).clamp(min=0)
    picks = [r for r in table if r['predicate_gain'] >= 0.045]
    ce0 = perseq_ce(m, fwd, held)
    results = []
    for r in picks:
        l0, h0, pred = r['layer'], r['head'], r['predicate_gain_top']
        fidx = FEATN.index(pred)
        profpat = sc['prof'][l0, h0].float()[offmat] * mask
        a = float(sc['joint_a'][l0, h0, fidx])
        b = float(sc['joint_b'][l0, h0, fidx])

        def hook_factory(mode):
            def hook(l, pat, mode=mode):
                if l != l0:
                    return pat
                pat = pat.clone()
                if mode == 'ablate':
                    pat[:, h0] = tpl[l0, h0]
                elif mode == 'profile':
                    pat[:, h0] = profpat
                else:                              # joint a*B + b*profile
                    Bf = build_feats(hook.idx, mask)[:, fidx]
                    pat[:, h0] = a * Bf + b * profpat
                return pat
            return hook

        row = {'model': name, 'layer': l0, 'head': h0,
               'predicate': pred, 'predicate_gain': r['predicate_gain'],
               'joint_coef_predicate': round(a, 5),
               'joint_coef_profile': round(b, 5)}
        for mode, key in (('profile', 'dce_profile_only'),
                          ('joint', 'dce_joint'),
                          ('ablate', 'dce_ablate')):
            hk = hook_factory(mode)
            ces = []
            for i in range(N_FIT, held.shape[0], BP):
                idx = held[i:i + BP]
                hk.idx = idx[:, :T]
                lg = fwd(m, idx[:, :T], pat_hook=hk)
                ce = F.cross_entropy(lg.float().reshape(-1, Q.V),
                                     idx[:, 1:T + 1].reshape(-1),
                                     reduction='none')
                ces.append(ce.view(idx.shape[0], T).mean(1).cpu())
            d = torch.cat(ces) - ce0
            row[key] = round(float(d.mean()), 5)
            row[key + '_se'] = round(float(d.std() / math.sqrt(len(d))), 5)
        row['predicate_causally_real'] = bool(
            row['dce_joint'] < row['dce_profile_only']
            - 2 * (row['dce_joint_se'] + row['dce_profile_only_se']))
        print(f"  JOINT L{l0}H{h0} [{pred} gain {r['predicate_gain']}]: "
              f"profile-only {row['dce_profile_only']:+.4f} "
              f"joint {row['dce_joint']:+.4f} "
              f"ablate {row['dce_ablate']:+.4f} -> "
              f"{'REAL' if row['predicate_causally_real'] else 'not shown'}",
              flush=True)
        results.append(row)
    return results


# ---------------- main ----------------
def main():
    global KCLS
    t0 = time.time()
    KCLS = token_classes()
    held = torch.from_numpy(
        np.load(E.HELD_PATH)[ROWS[0]:ROWS[1]].astype(np.int64)).to(DEV)
    print(f"held: fresh34k rows [{ROWS[0]}:{ROWS[1]}] on {DEV}", flush=True)

    ctrl = control_synthetic(held)
    print("CONTROL (b)+(c) PASS", flush=True)

    out = {'control': ctrl, 'models': {}, 'causal': [], 'causal_joint': [],
           'config': {
               'checkpoints': {'E9a': 'qk_e9_a.pt', 'E19a': 'qk_e19_a.pt'},
               'factories': {'E9a': 'qk_e7_evenout_run.make_e7m1',
                             'E19a': 'qk_e15_reinvest_run.make_e15c(s=15)'},
               'held_rows': f'fresh34k[{ROWS[0]}:{ROWS[1]}]',
               'fit_eval_split': f'fit seqs 0:{N_FIT}, eval {N_FIT}:200',
               'T': T, 'heads': f'{NL} blocks x {NH} heads x 2 models',
               'score': ('held-out uncentered R^2 of a 1-parameter LS fit of '
                         'the causal-masked signed pattern on the predicate '
                         '(POSITIONAL_decay: per-offset profile); '
                         'predicate_gain: R^2(predicate+profile joint) - '
                         'R^2(profile), the bilin18-census programmatic '
                         'criterion (>= 0.05)'),
               'match_direction_note': (
                   'MATCH_prev = key whose PREVIOUS token equals the query '
                   'token (classic induction target, the bilin18 census '
                   'convention); MATCH_next = key whose NEXT token equals '
                   'the query token (the task brief\'s literal wording), '
                   'diagonal excluded; both are scored.'),
               'reviewer2_additions': (
                   f'(1) shuffled-token nulls, K={NULL_K} within-sequence '
                   'key-token shuffles, z-scores on eval-half cos^2 for the '
                   'token-dependent predicates (PREV/FIRST are shuffle-'
                   'invariant, no z defined); (2) 2 random mid-scoring '
                   'causal-control heads per model (pick tag random_mid); '
                   '(3) untrained-init floor: same census on a fresh-init '
                   'model of each architecture.'),
               'device': DEV}}

    for name, stem, factory, fwd in (
            ('E9a_readable_recipe', 'qk_e9_a', E7R.make_e7m1, forward_e9),
            ('E19a_bandwidth_frontier', 'qk_e19_a',
             lambda: E15R.make_e15c(s=15), forward_e19)):
        print(f"==== {name} ====", flush=True)
        m, _ = E.load_arm(stem, factory)
        # control (a): forward fidelity on one batch
        idx = held[:8]
        with torch.no_grad():
            d = float((fwd(m, idx[:, :T]) - m(idx[:, :T])).abs().max())
        print(f"control(a) {name}: reimplemented forward vs model forward "
              f"max |logit diff| {d:.2e}", flush=True)
        assert d < 1e-5, d
        table, sc, tpl = census_model(name, m, fwd, held)
        # Reviewer-2 R8: shuffled-token nulls -> z-scores per head/predicate
        c2r, zmat, mu_n, sd_n = null_zscores(m, fwd, held)
        from collections import Counter
        zcnt = Counter()
        for r in table:
            l, h = r['layer'], r['head']
            r['cos2_eval'] = {FEATN[f]: round(float(c2r[l, h, f]), 4)
                              for f in NULL_FEATS}
            r['null_z'] = {FEATN[f]: round(float(zmat[l, h, f]), 1)
                           for f in NULL_FEATS}
            for k, v in r['null_z'].items():
                if v >= 3:
                    zcnt[k] += 1
        summ = summarize(table)
        summ['heads_with_null_z_ge3'] = dict(zcnt)
        print(f"{name} summary: {json.dumps(summ)}", flush=True)
        out['models'][name] = {
            'checkpoint': f'{stem}.pt',
            'forward_fidelity_max_logit_diff': float(f'{d:.3e}'),
            'table': table, 'summary': summ}
        out['causal'] += causal_checks(name, m, fwd, held, table, sc, tpl)
        out['causal_joint'] += causal_joint_checks(name, m, fwd, held, table,
                                                   sc, tpl)
        del m, tpl
        if USE_CUDA:
            torch.cuda.empty_cache()
        # Reviewer-2: untrained-init floor (same architecture, fresh init)
        mu0 = factory().eval().float()
        t_u, sc_u, tpl_u = census_model(name + '_UNTRAINED', mu0, fwd, held)
        summ_u = summarize(t_u)
        top_u = sorted(t_u, key=lambda r: -r['best_score'])[:3]
        print(f"{name} UNTRAINED floor: {json.dumps(summ_u)}", flush=True)
        out['models'][name]['untrained_floor'] = {
            'summary': summ_u,
            'top3': [{k: r[k] for k in ('layer', 'head', 'best',
                                        'best_score', 'second',
                                        'second_score')} for r in top_u]}
        del mu0, tpl_u
        if USE_CUDA:
            torch.cuda.empty_cache()

    # bilin18 comparison (actual stored numbers, v2 census)
    try:
        v2 = json.load(open(f'{QK}/qk_selection_census_v2.json'))
        from collections import Counter
        prog = v2['programmatic']
        out['bilin18_comparison'] = {
            'source': 'qk_selection_census_v2.json (bilin18, 18 layers x 9 '
                      'heads)',
            'programmatic_heads': f"{len(prog)}/162",
            'layers_with_programmatic_heads':
                f"{len({c['layer'] for c in prog})}/18",
            'top_predicate_hist': dict(Counter(
                c['top_predicate'] for c in prog)),
            'note': ('the task brief\'s recollection was "nameable heads at '
                     '15/17 layers, all match/copy family"; the stored v2 '
                     'census says 30/162 programmatic heads across 16 of 18 '
                     'layers, dominated by KEY_newline / MATCH_prev / '
                     'MATCH_same / KEY_cap (match-copy plus key-class '
                     'families)')}
    except Exception as ex:                       # pragma: no cover
        out['bilin18_comparison'] = {'error': str(ex)}

    out['config']['runtime_s'] = round(time.time() - t0, 1)
    json.dump(out, open(JOUT, 'w'), indent=2)
    print(f"E21 CENSUS DONE -> {JOUT} ({out['config']['runtime_s']} s)",
          flush=True)


if __name__ == '__main__':
    main()
