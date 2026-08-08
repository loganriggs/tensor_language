"""Rungs 2-5 for DEPTH >= 1 cells: the depth-1 ladder, generalised to a stack.

WHAT IS REUSED VERBATIM.  Everything that does not care about depth is imported
from `tf_interp` and called unchanged: `rung2` (spectra of the layer-0 factors
and of an MLP tensor), `data_baselines`, `held_batches`, `ce_of`,
`induction_battery`, `induction_sensitivity`, `_band`, `_tok_class`.  This file
adds ONLY what a second layer genuinely requires.

WHAT DEPTH 2 GENUINELY REQUIRES.  At depth 1 the layer-0 module input is
EXACTLY the token table Ehn[t], so the whole model is a token-pair polynomial.
At depth >= 2 that is true of layer 0 only.  Layer 1 reads

    hn1 = rms(e + A0 + M0)

which is context dependent, so layer 1's queries, keys AND values are not token
tables.  That is not a nuisance, it is the composition path, and it is the thing
this file measures:

  * the ladder is split BY LAYER (layer-0 attention vs layer-1 attention, and
    the same for the two MLPs), so "attention matters" can be attributed;
  * layer 1's READ is substituted -- rms(e) instead of rms(e+A0+M0), and the
    intermediate substitutions -- which asks directly whether layer 1 reads what
    layer 0 wrote or reads the embedding it could have read at layer 0;
  * the LADDER ORDER is reversed and both orderings are reported, because the
    marginal value of a term depends on what is already in the ladder and the
    depth-1 headline ("attention is inert") was an increment measured after a
    term that had already frozen the context away.

SIGN IS A GAUGE FREEDOM (standing rule, inherited).  Nothing is named from the
sign of a factor; every sign-bearing statement is composed to the logits and
confirmed by a causal ablation.

Usage
    python tf_interp2.py --stem tf_vanilla_d2_w64_b8192_s0
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import tf_corpus
import tf_fold
import tf_interp as I1
import tf_model as M

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
_rms = I1._rms

# ---------------------------------------------------------------------------
# REGISTERED PREDICTIONS.  Written into the results JSON BEFORE any rung runs.
# ---------------------------------------------------------------------------
REGISTERED2 = {
    'written_before_any_depth2_rung_ran': True,
    'honesty_note': 'the depth-2 INDUCTION score was already measured for '
                    'w32 s0 and w64 s0/s1 before this analysis was scoped '
                    '(RESULTS.md FINDING 5: -0.007 and -0.014).  Those three '
                    'numbers are therefore NOT covered by the prediction '
                    'below; the prediction is registered for the cells whose '
                    'induction score has never been computed (d2 w128 seeds '
                    '0/1/2 and d2 w256 s0) and for every non-induction '
                    'quantity in this file, none of which has been measured '
                    'at depth 2 at all.',
    'd2_induction': 'PREDICT ABSENT.  At depth 2, widths 128 and 256, on this '
                    'corpus and this 15,000-step single-epoch budget, the '
                    'induction score (CE of a shuffled-prefix repeat minus CE '
                    'of an order-preserving repeat, scored on the second copy) '
                    'stays within +-0.05 nats of zero and is not more than 3 '
                    'standard errors above zero.  Composition being POSSIBLE '
                    'is not composition being LEARNED: the depth-1 heads are '
                    'not copy heads (FINDING 6), the composed pair table is '
                    'nearly an outer product, and depth 2 buys almost no CE at '
                    'width 64 (5.021 vs 5.044), which is not what a model that '
                    'had just discovered a new algorithm looks like.  If a '
                    'score does appear I predict it is < 0.2 nats and '
                    'concentrated in ONE layer-1 head.',
    'd2_attention_not_inert': 'PREDICT THE HEADLINE FLIPS ONLY IN WORDING.  '
                              'Removing all attention from a depth-2 model '
                              'will cost MORE KL than the same knockout at '
                              'depth 1 at the same width (predict > 1.0 nats '
                              'at width 64, vs 0.47 at depth 1), but the '
                              'increment measured the depth-1 way -- '
                              'no-attention-at-all MINUS bigram-only -- will '
                              'again be small (predict < 0.3 nats).  The '
                              'prediction is that the two framings continue to '
                              'disagree, i.e. that the depth-1 "attention is '
                              'inert" headline is an artifact of WHERE in the '
                              'ladder attention is added, not a property of '
                              'the model.',
    'd2_layer_split': 'PREDICT LAYER 0 DOMINATES.  Deleting layer-0 attention '
                      'costs more KL than deleting layer-1 attention at every '
                      'width, and the two deletions are SUB-additive (their '
                      'individual costs sum to more than the joint deletion) '
                      'because the second layer partly re-does the first.',
    'd2_composition': 'PREDICT WEAK COMPOSITION.  Making layer 1 read rms(e) '
                      '-- the embedding it could have read at layer 0 -- '
                      'instead of rms(e+A0+M0) costs LESS than half of what '
                      'deleting layer-1 attention outright costs.  I.e. most '
                      "of layer-1 attention's value is a second pass over the "
                      'token identity, not a read of layer 0 output.',
    'd2_ladder_order': 'PREDICT STRONG ORDER DEPENDENCE.  Attention added '
                       'FIRST (to the bare embedding, MLPs off) will buy '
                       'several nats; attention added LAST (after the bigram '
                       'stage) will buy under 1 nat.  Both at depth 1 and '
                       'depth 2.  This is a prediction that the ladder order '
                       'stacks the deck, registered before the test is run.',
    'd2_head_compensation': 'PREDICT PER-HEAD DROPS ARE SUB-ADDITIVE.  The sum '
                            'of the individual single-head drop costs will '
                            'exceed the cost of dropping the whole layer, '
                            'because the heads write overlapping directions.',
}


# ---------------------------------------------------------------------------
class DeepFold:
    """The folded pipeline for a vanilla cell at ANY depth.

    Layer 0 is driven by the EXACT token factor tables (`Q1..K2`, `Vv`, `OV`)
    -- a table program with no forward pass.  Layers >= 1 necessarily apply
    their read matrices to the stream, and that asymmetry IS the depth-2
    result, so it is stated rather than hidden.
    """

    def __init__(self, stem, device=DEV):
        model, cfg, ck = tf_fold.load_checkpoint(stem, device)
        assert cfg.n_slots == 1 and not cfg.predicate and not cfg.codebook \
            and not cfg.small_dec and not cfg.shrink_mode, \
            'DeepFold assumes the vanilla variant'
        self.stem, self.model, self.cfg, self.dev = stem, model, cfg, device
        self.L = cfg.depth
        self.V, self.H, self.hd = cfg.vocab, cfg.n_heads, cfg.head_dim
        self.Ws, self.Dc = model.Ws, model.Dc
        self.Ehn = model.token_input_table().to(device)          # (V, Ws)
        self.WU = model.wte.weight.detach().float()              # (V, Ws)
        self.cos, self.sin = model.cos.to(device), model.sin.to(device)
        # ---- layer 0 exact token factors ----
        f = model.fold_layer0_qk(materialize=False, device=device)
        self.Q1, self.K1, self.Q2, self.K2 = f['Q1'], f['K1'], f['Q2'], f['K2']
        self.Vv = f['Vv']
        Wp0 = model.h[0].c_proj.weight.detach().float()
        self.OV = torch.stack([self.Vv[h] @ Wp0[:, h * self.hd:(h + 1) * self.hd].t()
                               for h in range(self.H)])          # (H, V, Ws)
        # ---- per-layer weights and MLP tensors ----
        self.Wq, self.Wk, self.Wq2, self.Wk2, self.Wv, self.Wproj = \
            [], [], [], [], [], []
        self.Tl, self.bl, self.Dn, self.Lf, self.Rf = [], [], [], [], []
        for li in range(self.L):
            b = model.h[li]
            self.Wq.append(b.c_q.weight.detach().float())
            self.Wk.append(b.c_k.weight.detach().float())
            self.Wq2.append(b.c_q2.weight.detach().float())
            self.Wk2.append(b.c_k2.weight.detach().float())
            self.Wv.append(b.c_v.weight.detach().float())
            self.Wproj.append(b.c_proj.weight.detach().float())
            self.Tl.append(model.fold_mlp(li, device=device).to(device))
            self.bl.append(b.Down_bias.detach().float())
            self.Dn.append(b.Down.weight.detach().float())
            self.Lf.append(b.Left.weight.detach().float())
            self.Rf.append(b.Right.weight.detach().float())
        # aliases so tf_interp.rung2 / data_baselines run unchanged on layer 0
        self.T = self.Tl[0]
        self.bias = self.bl[0]
        # est-fitted ablation values, filled lazily
        self._est = None

    # ------------------------------------------------------------- pieces
    def _pat_layer0(self, idx):
        """Layer-0 pattern from the FOLDED token factors only."""
        B, Tq = idx.shape
        cos, sin = self.cos[None, :Tq, None, :], self.sin[None, :Tq, None, :]

        def g(fac):
            return M.apply_rot(fac.permute(1, 0, 2)[idx], cos, sin)
        s1 = torch.einsum('bqhd,bkhd->bhqk', g(self.Q1), g(self.K1)) / self.hd
        s2 = torch.einsum('bqhd,bkhd->bhqk', g(self.Q2), g(self.K2)) / self.hd
        return (s1 * s2).masked_fill(~self.model.mask[:Tq, :Tq], 0.0)

    def _pat_from(self, li, hn, rot=True):
        """Layer-li pattern from a read vector (B, T, Ws)."""
        B, Tq, _ = hn.shape
        cos, sin = self.cos[None, :Tq, None, :], self.sin[None, :Tq, None, :]

        def qk(W):
            z = F.rms_norm((hn @ W.t()).view(B, Tq, self.H, self.hd), (self.hd,))
            return M.apply_rot(z, cos, sin) if rot else z
        s1 = torch.einsum('bqhd,bkhd->bhqk', qk(self.Wq[li]), qk(self.Wk[li])) \
            / self.hd
        s2 = torch.einsum('bqhd,bkhd->bhqk', qk(self.Wq2[li]), qk(self.Wk2[li])) \
            / self.hd
        return (s1 * s2).masked_fill(~self.model.mask[:Tq, :Tq], 0.0)

    def _attn_out(self, li, pat, hn, idx=None):
        if li == 0 and idx is not None:
            ov = self.OV.permute(1, 0, 2)[idx]                   # (B,T,H,Ws)
            return torch.einsum('bhqk,bkho->bqo', pat, ov)
        B, Tq, _ = hn.shape
        v = (hn @ self.Wv[li].t()).view(B, Tq, self.H, self.hd)
        y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, Tq, self.Dc)
        return y @ self.Wproj[li].t()

    def _mlp(self, li, r, topk=None):
        """MLP(rms(r)).  The bilinear MLP is EXACTLY the rank-`hidden` symmetric
        CP decomposition sum_f Down[:,f] (l_f.x)(r_f.x), so the factored
        evaluation and the tensor evaluation are the same object; the tensor
        form is what rung 2 spectra use and the two are gated against each
        other in `self_check`."""
        xn = _rms(r, self.Ws)
        Dn, Lf, Rf = self.Dn[li], self.Lf[li], self.Rf[li]
        if topk is not None:
            keep = self._mlp_use(li)[1][:topk]
            Dn, Lf, Rf = Dn[:, keep], Lf[keep], Rf[keep]
        return ((xn @ Lf.t()) * (xn @ Rf.t())) @ Dn.t() + self.bl[li]

    def _mlp_use(self, li):
        """Hidden-unit importance for layer li, measured on the ESTIMATION
        split (never on the text the truncation is scored on)."""
        key = f'_use{li}'
        if not hasattr(self, key):
            arr = tf_corpus.load_split(self.V, 'est', 16, tok=self.cfg.tok)
            x = torch.from_numpy(arr[:, :256]).to(self.dev)
            r = self.run(x)['pre_mlp'][li]
            xn = _rms(r, self.Ws)
            u = (self.Dn[li].norm(dim=0)
                 * ((xn @ self.Lf[li].t()) * (xn @ self.Rf[li].t())
                    ).abs().mean((0, 1)))
            setattr(self, key, (u, u.argsort(descending=True)))
        return getattr(self, key)

    # ------------------------------------------------------------- the run
    def run(self, idx, attn=None, mlp=None, reads=None, mlp_reads=None,
            want=()):
        """One pass of the folded pipeline with per-layer knockouts.

        attn[li] : None/'full' | 'self' | 'zero' | 'meanpast' | ('band', d)
                   | 'pos' | 'norot' | ('drop', h) | ('keep', h) | ('freq', k)
        mlp[li]  : None/'full' | 'zero' | ('top', k)
        reads[li]: None -> the true stream; otherwise a callable(pieces)->tensor
                   giving the vector layer li's Q/K/V read instead.  Use
                   ('qk', fn) / ('v', fn) to substitute only one of them.
        """
        B, Tq = idx.shape
        attn = attn or {}
        mlp = mlp or {}
        reads = reads or {}
        e = self.Ehn[idx]
        x = e
        P = {'e': e, 'A': [], 'M': [], 'Apast': [], 'pre_mlp': [],
             'read': [], 'pat': []}
        for li in range(self.L):
            hn_true = _rms(x, self.Ws)
            sub = reads.get(li)
            hn_qk = hn_v = hn_true
            if sub is not None:
                if isinstance(sub, tuple):
                    which, fn = sub
                    v = fn(P)
                    if which == 'qk':
                        hn_qk = v
                    else:
                        hn_v = v
                else:
                    hn_qk = hn_v = sub(P)
            mode = attn.get(li, 'full')
            A = self._attn_step(li, idx, hn_qk, hn_v, mode, P,
                                use_fold0=(sub is None))
            P['A'].append(A)
            x = x + A
            P['pre_mlp'].append(x)
            mm = mlp.get(li, 'full')
            xm = x
            if mlp_reads and li in mlp_reads:
                # substitute what the MLP squares WITHOUT touching the residual
                xm = mlp_reads[li](P, x)
            if mm == 'zero':
                Mo = torch.zeros_like(x)
            elif isinstance(mm, tuple):
                Mo = self._mlp(li, xm, topk=mm[1])
            else:
                Mo = self._mlp(li, xm)
            P['M'].append(Mo)
            x = x + Mo
            P['read'].append(hn_true)
        P['r'] = x
        return P

    def _attn_step(self, li, idx, hn_qk, hn_v, mode, P, use_fold0=True):
        Tq = idx.shape[1]
        if mode == 'zero':
            return torch.zeros(idx.shape[0], Tq, self.Ws, device=self.dev)
        rot = not (isinstance(mode, str) and mode == 'norot')
        if li == 0 and use_fold0 and rot:
            # the EXACT folded token-factor path: a table program, no weights
            # applied to a stream
            pat = self._pat_layer0(idx)
        else:
            pat = self._pat_from(li, hn_qk, rot=rot)
        if isinstance(mode, tuple):
            kind, arg = mode
            if kind == 'band':
                pat = I1._band(pat, arg)
            elif kind == 'drop':
                k = torch.ones(self.H, device=self.dev)
                k[arg] = 0.0
                pat = pat * k[None, :, None, None]
            elif kind == 'keep':
                k = torch.zeros(self.H, device=self.dev)
                k[arg] = 1.0
                pat = pat * k[None, :, None, None]
            elif kind == 'freq':
                pat = self._pat_freq(li, idx, hn_qk, arg)
            elif kind == 'inject':
                # RESAMPLE ABLATION: use a write this layer actually produced,
                # on a different context.  On-distribution by construction, so
                # it separates 'the layer's information is gone' from 'the
                # downstream modules are off distribution'.
                return arg
        elif mode == 'self':
            eye = torch.eye(Tq, device=self.dev, dtype=torch.bool)
            pat = pat * eye
        elif mode == 'pos':
            pat = self.est()['prof'][li][None, :, :Tq, :Tq].expand(
                idx.shape[0], -1, -1, -1)
        A = self._attn_out(li, pat, hn_v, idx if li == 0 else None)
        if mode == 'meanpast':
            eye = torch.eye(Tq, device=self.dev, dtype=torch.bool)
            Aself = self._attn_out(li, pat * eye, hn_v, idx if li == 0 else None)
            A = Aself + self.est()['meanpast'][li][None, :Tq]
        return A

    def _pat_freq(self, li, idx, hn, k):
        """Keep the top-k ROTARY FREQUENCY pairs -- the only delta-equivariant
        way to cut a head's subspace."""
        d = self.hd // 2
        if li == 0:
            en = torch.zeros(self.H, d, device=self.dev)
            for qn, kn in (('Q1', 'K1'), ('Q2', 'K2')):
                Q, K = getattr(self, qn), getattr(self, kn)
                en = en + ((Q[:, :, :d] ** 2 + Q[:, :, d:] ** 2).mean(1)
                           * (K[:, :, :d] ** 2 + K[:, :, d:] ** 2).mean(1)).sqrt()
        else:
            B, Tq, _ = hn.shape
            en = torch.zeros(self.H, d, device=self.dev)
            for Wq_, Wk_ in ((self.Wq[li], self.Wk[li]),
                             (self.Wq2[li], self.Wk2[li])):
                zq = F.rms_norm((hn @ Wq_.t()).view(B, Tq, self.H, self.hd),
                                (self.hd,))
                zk = F.rms_norm((hn @ Wk_.t()).view(B, Tq, self.H, self.hd),
                                (self.hd,))
                eq = (zq[..., :d] ** 2 + zq[..., d:] ** 2).mean((0, 1))
                ek = (zk[..., :d] ** 2 + zk[..., d:] ** 2).mean((0, 1))
                en = en + (eq * ek).sqrt()
        keep = torch.zeros(self.H, d, device=self.dev)
        keep.scatter_(1, en.topk(k, dim=1).indices, 1.0)
        m = torch.cat([keep, keep], 1)[:, None, :]
        if li == 0:
            old = (self.Q1, self.K1, self.Q2, self.K2)
            self.Q1, self.K1, self.Q2, self.K2 = (f * m for f in old)
            try:
                return self._pat_layer0(idx)
            finally:
                self.Q1, self.K1, self.Q2, self.K2 = old
        B, Tq, _ = hn.shape
        cos, sin = self.cos[None, :Tq, None, :], self.sin[None, :Tq, None, :]
        mk = m.permute(1, 0, 2)                                  # (1,H,hd)

        def qk(W):
            z = F.rms_norm((hn @ W.t()).view(B, Tq, self.H, self.hd), (self.hd,))
            return M.apply_rot(z * mk[None], cos, sin)
        s1 = torch.einsum('bqhd,bkhd->bhqk', qk(self.Wq[li]),
                          qk(self.Wk[li])) / self.hd
        s2 = torch.einsum('bqhd,bkhd->bhqk', qk(self.Wq2[li]),
                          qk(self.Wk2[li])) / self.hd
        return (s1 * s2).masked_fill(~self.model.mask[:Tq, :Tq], 0.0)

    # ---------------------------------------- depth-1 compatibility shim
    def terms(self, idx, pat=None):
        """The depth-1 term dict, aggregated over layers, so the depth-1
        helpers (`position_profile`, anything expecting e/A/M) run unchanged.
        A = every attention write summed, M = every MLP write summed; the
        residual is still EXACTLY e + A + M because RMSNorm is a scalar gauge
        and all writes are additive."""
        P = self.run(idx)
        A0 = self._attn_step(0, idx, self.Ehn[idx], self.Ehn[idx], 'self', P)
        A = sum(P['A'])
        return {'e': P['e'], 'A': A, 'A0': A0, 'Apast': A - A0,
                'M': sum(P['M']), 'r': P['r'], 'pat': None}

    # ------------------------------------------------------------ readout
    def readout(self, r):
        return 30 * torch.tanh((_rms(r, self.Ws) @ self.WU.t()) / 30)

    # ------------------------------------------- weights-only bigram table
    @torch.no_grad()
    def bigram_table(self):
        """The model's EXACT output on a length-1 context, as a (V, Ws) row
        table computed from weights alone -- every layer self-attending at
        delta = 0, where the rotary is the identity.  This is the depth-2
        generalisation of the depth-1 r0 table and it is still a pure token
        function, because at length 1 there is nothing else to read."""
        x = self.Ehn
        for li in range(self.L):
            hn = _rms(x, self.Ws)
            if li == 0:
                s1 = (self.Q1 * self.K1).sum(-1) / self.hd
                s2 = (self.Q2 * self.K2).sum(-1) / self.hd
                p = s1 * s2                                       # (H, V)
                A = torch.einsum('hv,hvd->vd', p, self.OV)
            else:
                def qk(W):
                    return F.rms_norm((hn @ W.t()).view(self.V, self.H, self.hd),
                                      (self.hd,))
                s1 = (qk(self.Wq[li]) * qk(self.Wk[li])).sum(-1) / self.hd
                s2 = (qk(self.Wq2[li]) * qk(self.Wk2[li])).sum(-1) / self.hd
                p = s1 * s2                                       # (V, H)
                v = (hn @ self.Wv[li].t()).view(self.V, self.H, self.hd)
                A = (p[..., None] * v).reshape(self.V, self.Dc) \
                    @ self.Wproj[li].t()
            x = x + A
            x = x + self._mlp(li, x[None])[0]
        return x

    # ------------------------------------------------- est-fitted ablations
    @torch.no_grad()
    def est(self, T=256, n_seq=32):
        if self._est is not None:
            return self._est
        arr = tf_corpus.load_split(self.V, 'est', n_seq, tok=self.cfg.tok)
        x = torch.from_numpy(arr[:, :T]).to(self.dev)
        prof = [None] * self.L
        mp = [None] * self.L
        n = 0
        eye = torch.eye(T, device=self.dev, dtype=torch.bool)
        for a in range(0, x.shape[0], 8):
            b = x[a:a + 8]
            P = self.run(b)
            for li in range(self.L):
                hn = P['read'][li]
                pat = self._pat_layer0(b) if li == 0 else self._pat_from(li, hn)
                Ap = self._attn_out(li, pat * ~eye, hn, b if li == 0 else None)
                prof[li] = pat.sum(0) if prof[li] is None else prof[li] + pat.sum(0)
                mp[li] = Ap.sum(0) if mp[li] is None else mp[li] + Ap.sum(0)
            n += b.shape[0]
        self._est = {'prof': [p / n for p in prof],
                     'meanpast': [m / n for m in mp], 'n': n}
        return self._est

    # ------------------------------------------------------ POSITIVE CONTROL
    @torch.no_grad()
    def self_check(self, idx):
        out = {}
        with M.exact_math():
            P = self.run(idx)
            rec = self.readout(P['r'])
            ref = self.model(idx)
            sc = float(ref.abs().max())
            out['pipeline_rel_logit_diff'] = float((rec - ref).abs().max()) / sc
            # the model's own bigram table must reproduce a length-1 context
            one = idx[:, :1]
            ref1 = self.model(one)[:, 0]
            rec1 = self.readout(self.bigram_table()[one[:, 0]])
            out['length1_table_rel_logit_diff'] = float(
                (rec1 - ref1).abs().max()) / max(float(ref1.abs().max()), 1e-30)
            # factored MLP == folded tensor MLP (the rung-2 object)
            r = P['pre_mlp'][0]
            xn = _rms(r, self.Ws)
            mt = torch.einsum('oij,bti,btj->bto', self.Tl[0], xn, xn) + self.bl[0]
            out['mlp_tensor_vs_factored_rel'] = float(
                (mt - P['M'][0]).abs().max() / mt.abs().max())
            # layer-0 folded token factors == applying the weights to the stream
            p_fold = self._pat_layer0(idx)
            p_w = self._pat_from(0, self.Ehn[idx])
            out['layer0_folded_vs_weight_pattern_rel'] = float(
                (p_fold - p_w).abs().max() / p_fold.abs().max())
        out['pass'] = bool(out['pipeline_rel_logit_diff'] < 1e-5
                           and out['length1_table_rel_logit_diff'] < 1e-5
                           and out['mlp_tensor_vs_factored_rel'] < 1e-5
                           and out['layer0_folded_vs_weight_pattern_rel'] < 1e-5)
        return out


# ---------------------------------------------------------------- the ladder
@torch.no_grad()
def ladder2(D, n_seq=96, T=256, batch=8, split='held', extra=True):
    """The depth-1 ladder stage-for-stage, plus the depth-2 layer split."""
    L = D.L
    acc, kl, ntok = {}, {}, 0
    D.est(T)
    bigt = D.bigram_table()
    allL = list(range(L))

    def cands(x):
        c = {}
        P = D.run(x)
        c['full_exact'] = P['r']
        c['embed_only'] = P['e']
        c['plus_self_attn'] = D.run(x, attn={l: 'self' for l in allL},
                                    mlp={l: 'zero' for l in allL})['r']
        c['model_bigram'] = bigt[x]
        c['no_attention_at_all'] = D.run(x, attn={l: 'zero' for l in allL})['r']
        c['past_attn_mean_ablated'] = D.run(
            x, attn={l: 'meanpast' for l in allL})['r']
        c['no_mlp'] = D.run(x, mlp={l: 'zero' for l in allL})['r']
        c['mlp_write_only'] = sum(P['M'])
        c['attn_write_only'] = P['e'] + sum(P['A'])
        # --- routes of the past attention (depth-1 names, generalised) ---
        Pself = D.run(x, attn={l: 'self' for l in allL})
        c['past_attn_direct_route_only'] = Pself['r'] + sum(
            P['A']) - sum(Pself['A'])
        c['past_attn_mlp_route_only'] = P['r'] - (sum(P['A']) - sum(Pself['A']))
        # --- BY LAYER: the composition story ---
        for li in allL:
            c[f'no_attn_layer{li}'] = D.run(x, attn={li: 'zero'})['r']
            c[f'no_past_attn_layer{li}'] = D.run(x, attn={li: 'meanpast'})['r']
            c[f'self_attn_only_layer{li}'] = D.run(x, attn={li: 'self'})['r']
            c[f'no_mlp_layer{li}'] = D.run(x, mlp={li: 'zero'})['r']
        if extra:
            for nm, dmax in (('trunc_delta1_only', 1), ('trunc_delta_le4', 4),
                             ('trunc_delta_le16', 16), ('trunc_delta_le64', 64)):
                c[nm] = D.run(x, attn={l: ('band', dmax) for l in allL})['r']
            c['positional_only_pattern'] = D.run(
                x, attn={l: 'pos' for l in allL})['r']
            c['no_rotary_pattern'] = D.run(
                x, attn={l: 'norot' for l in allL})['r']
            for li in allL:
                for h in range(D.H):
                    c[f'drop_l{li}_head{h}'] = D.run(
                        x, attn={li: ('drop', h)})['r']
                    c[f'keep_only_l{li}_head{h}'] = D.run(
                        x, attn={li: ('keep', h)})['r']
                for k in (1, 2, 4):
                    if k < D.hd // 2:
                        c[f'attn_l{li}_top{k}_rotary_freqs'] = D.run(
                            x, attn={li: ('freq', k)})['r']
                for k in (1, 2, 4, 8, 16, 32, 64, 128, 256):
                    if k < D.cfg.hidden:
                        c[f'mlp{li}_top{k}_hidden_units'] = D.run(
                            x, mlp={li: ('top', k)})['r']
            for k in (8, 32, 64, 128, 256):
                if k < D.cfg.hidden:
                    c[f'mlp_all_top{k}_hidden_units'] = D.run(
                        x, mlp={l: ('top', k) for l in allL})['r']
            # --- COMPOSITION: what does layer 1+ actually read? ---
            for li in range(1, L):
                emb = lambda P_: P_['e']                       # noqa: E731
                eA0 = lambda P_: _rms(P_['e'] + P_['A'][0], D.Ws)   # noqa: E731
                eM0 = lambda P_: _rms(P_['e'] + P_['M'][0], D.Ws)   # noqa: E731
                c[f'l{li}_reads_embedding'] = D.run(x, reads={li: emb})['r']
                c[f'l{li}_qk_reads_embedding'] = D.run(
                    x, reads={li: ('qk', emb)})['r']
                c[f'l{li}_v_reads_embedding'] = D.run(
                    x, reads={li: ('v', emb)})['r']
                c[f'l{li}_reads_e_plus_attn0'] = D.run(x, reads={li: eA0})['r']
                c[f'l{li}_reads_e_plus_mlp0'] = D.run(x, reads={li: eM0})['r']
        return c

    for x, y in I1.held_batches(D, n_seq, T, batch, split):
        ref = D.readout(D.run(x)['r'])
        logp_ref = F.log_softmax(ref.float(), -1)
        p_ref = logp_ref.exp()
        for s, r in cands(x).items():
            lg = D.readout(r)
            lp = F.log_softmax(lg.float(), -1)
            kl[s] = kl.get(s, 0.0) + float((p_ref * (logp_ref - lp)).sum())
            cc, n = I1.ce_of(lg, y)
            a = acc.setdefault(s, [0.0, 0])
            a[0] += float(cc)
            a[1] += n
        ntok += y.numel()
    out = {s: {'ce': acc[s][0] / acc[s][1], 'kl_from_model': kl[s] / ntok}
           for s in acc}
    out['_model_ce'] = out['full_exact']['ce']
    out['_tokens'] = ntok
    out['_split'] = split
    return out


# --------------------------------------------------- ladder ORDER reversal
@torch.no_grad()
def ladder_order(D, n_seq=64, T=256, batch=8):
    """THE ADVERSARIAL TEST the README's review stage demands.

    'Attention is inert' was measured at depth 1 as the increment between the
    bigram-only stage and the no-attention-at-all stage, i.e. attention was
    added LAST, after a term that had already frozen the context away.  Here
    the same two components are added in BOTH orders and both marginal values
    are reported.  If the two orderings disagree the increment is a property of
    the ladder, not of the model, and no single number may be called
    'the value of attention'.

    Order A (the depth-1 order):  embed -> +MLPs (bigram) -> +attention
    Order B (reversed)         :  embed -> +attention    -> +MLPs
    Shapley-style average of the two marginals is also reported, which is the
    order-free thing to quote.
    """
    L = D.L
    allL = list(range(L))
    st = {}
    kl, ntok = {}, 0
    for x, y in I1.held_batches(D, n_seq, T, batch):
        ref = D.readout(D.run(x)['r'])
        logp = F.log_softmax(ref.float(), -1)
        p = logp.exp()
        cand = {
            'neither': D.run(x, attn={l: 'zero' for l in allL},
                             mlp={l: 'zero' for l in allL})['r'],
            'mlp_only': D.run(x, attn={l: 'zero' for l in allL})['r'],
            'attn_only': D.run(x, mlp={l: 'zero' for l in allL})['r'],
            'both': D.run(x)['r'],
        }
        for s, r in cand.items():
            lg = F.log_softmax(D.readout(r).float(), -1)
            kl[s] = kl.get(s, 0.0) + float((p * (logp - lg)).sum())
        ntok += y.numel()
    for k in kl:
        st[k] = kl[k] / ntok
    a_first = st['neither'] - st['attn_only']       # attention added FIRST
    a_last = st['mlp_only'] - st['both']            # attention added LAST
    m_first = st['neither'] - st['mlp_only']
    m_last = st['attn_only'] - st['both']
    return {
        'kl': st,
        'attention_marginal_first': a_first,
        'attention_marginal_last': a_last,
        'attention_marginal_shapley': 0.5 * (a_first + a_last),
        'mlp_marginal_first': m_first,
        'mlp_marginal_last': m_last,
        'mlp_marginal_shapley': 0.5 * (m_first + m_last),
        'order_dependence_ratio_attention':
            (a_first / a_last) if a_last > 0 else float('inf'),
        'interaction_nats': a_first - a_last,
        'note': 'KL from the true model, nats/token, held split.  A large '
                'ratio means the single number "what attention is worth" is '
                'not well defined and must always be quoted with its position '
                'in the ladder.  Note that neither marginal is the depth-1 '
                'headline increment: THAT one compared no-attention-at-all '
                'against the BIGRAM stage, which itself contains the '
                'self-attention term, so it is not a knockout of attention at '
                'all -- it is the gap between two different reduced models.',
    }


# ------------------------------------------------ induction, located by head
@torch.no_grad()
def induction_by_head(D, seeds=3):
    """Where does the induction score live?  The identical battery run with the
    folded pipeline as the forward, one head removed at a time.  Reported for
    every cell whether or not a score appears, so a null is a measurement and
    not an omission."""
    base = [I1.induction_battery(D, seed=s, model=lambda z: D.readout(D.run(z)['r']))
            for s in range(seeds)]
    out = {'folded_pipeline_baseline': {
        'induction_score_mean': float(np.mean([b['induction_score'] for b in base])),
        'induction_score_sd': float(np.std([b['induction_score'] for b in base],
                                           ddof=1)),
        'bag_score_mean': float(np.mean([b['bag_score'] for b in base]))}}
    per = {}
    for li in range(D.L):
        for h in range(D.H):
            f = (lambda li_, h_: lambda z: D.readout(
                D.run(z, attn={li_: ('drop', h_)})['r']))(li, h)
            r = [I1.induction_battery(D, seed=s, model=f) for s in range(seeds)]
            per[f'drop_l{li}_head{h}'] = {
                'induction_score_mean': float(np.mean([q['induction_score']
                                                       for q in r])),
                'bag_score_mean': float(np.mean([q['bag_score'] for q in r]))}
    out['per_head'] = per
    return out


@torch.no_grad()
def induction_power(D, seeds=5):
    """DETECTABLE-EFFECT FLOOR.  A null is worthless without the smallest
    effect the battery could have resolved, so the oracle-mixture sensitivity
    curve is extended DOWN until it hits the noise, and the floor is quoted in
    both nats and oracle-mixture weight."""
    eps = (0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 1e-3, 1e-2)
    sens = I1.induction_sensitivity(D, epsilons=eps, seeds=seeds)
    s0 = sens['eps_0.0']
    sd = s0['induction_score_sd']
    se = sd / math.sqrt(seeds)
    floor_nats = 3 * se
    # smallest eps whose mean clears mean(0) + 3 SE
    thr = s0['induction_score_mean'] + floor_nats
    hit = None
    for e in eps[1:]:
        if sens[f'eps_{e}']['induction_score_mean'] > thr:
            hit = e
            break
    return {'sensitivity': sens, 'null_sd_across_probe_seeds': sd,
            'null_standard_error': se,
            'detectable_effect_floor_nats_3se': floor_nats,
            'smallest_detected_oracle_mixture_weight': hit,
            'note': 'the floor is 3 standard errors of the score across probe '
                    'seeds.  Any registered "induction is absent" claim is '
                    'only a claim down to this floor.'}


# --------------------------------------------------------------- rung 2 (L1)
@torch.no_grad()
def rung2_layers(D):
    """The layer-0 rung 2 verbatim from tf_interp, plus, for every layer >= 1,
    the MLP tensor spectra and the READ-MATRIX spectra.  Layer >= 1 has no
    token factor table -- that absence IS the finding, so it is stated."""
    out = {'layer0': I1.rung2(D)}
    for li in range(1, D.L):
        T = D.Tl[li]
        Ws = D.Ws
        unf0 = torch.linalg.svdvals(T.reshape(Ws, -1).double()).cpu().numpy()
        ev = torch.linalg.eigvalsh(T.double()).cpu().numpy()
        sl = [tf_fold.eff_rank(np.sort(np.abs(a))[::-1]) for a in ev]
        d = {'no_token_factor_table': 'layer %d reads rms(e + writes), which is '
                                      'context dependent, so its queries, keys '
                                      'and values are NOT token tables; this is '
                                      'the structural difference depth buys'
                                      % li,
             'mlp': {'shape': list(T.shape),
                     'mode0_unfolding': {**tf_fold.eff_rank(unf0),
                                         'rank_bound': Ws},
                     'slice_eigen_mean_entropy_rank':
                         float(np.mean([s['entropy_rank'] for s in sl])),
                     'mean_negative_eig_share': float((ev < 0).sum() / ev.size)}}
        rd = {}
        for nm, W in (('c_q', D.Wq[li]), ('c_k', D.Wk[li]), ('c_q2', D.Wq2[li]),
                      ('c_k2', D.Wk2[li]), ('c_v', D.Wv[li]),
                      ('c_proj', D.Wproj[li])):
            sv = torch.linalg.svdvals(W.double()).cpu().numpy()
            rd[nm] = {**tf_fold.eff_rank(sv), 'shape': list(W.shape)}
        d['read_matrices'] = rd
        out[f'layer{li}'] = d
    return out


# ----------------------------------------------------- stream geometry (L)
@torch.no_grad()
def stream_geometry2(D, n_seq=32, T=256, batch=8):
    """EXACT additive attribution of the pre-tanh logit across all 2L writes."""
    keys = ['e'] + [f'A{l}' for l in range(D.L)] + [f'M{l}' for l in range(D.L)]
    tot = {f'{k}_{q}': 0.0 for k in keys for q in ('norm', 'share')}
    n = 0
    for x, y in I1.held_batches(D, n_seq, T, batch):
        P = D.run(x)
        parts = {'e': P['e']}
        for l in range(D.L):
            parts[f'A{l}'] = P['A'][l]
            parts[f'M{l}'] = P['M'][l]
        r = P['r']
        g = math.sqrt(D.Ws) / r.norm(dim=-1, keepdim=True)
        z = (r * g) @ D.WU.t()
        zc = z - z.mean(-1, keepdim=True)
        den = float((zc * zc).sum())
        for k, v in parts.items():
            zz = (v * g) @ D.WU.t()
            zz = zz - zz.mean(-1, keepdim=True)
            tot[f'{k}_norm'] += float(v.norm(dim=-1).sum())
            tot[f'{k}_share'] += float((zz * zc).sum()) / max(den, 1e-30) * \
                y.numel()
        n += y.numel()
    out = {k: (v / n) for k, v in tot.items()}
    out['note'] = ('RMSNorm is a scalar gauge so the pre-tanh logit is exactly '
                   'additive in the 2L+1 folded writes; the shares are '
                   'projections onto the centred logit and sum to 1 by '
                   'construction')
    out['share_sum'] = sum(out[f'{k}_share'] for k in keys)
    return out


# ---------------------------------------------------------------- driver
@torch.no_grad()
def composed_vs_causal(D, n_seq=32, T=256, batch=8):
    """THE STANDING SIGN RULE, made falsifiable.

    Every named quantity in rung 4 is a COMPOSED path
    `p_h(t,u,delta) * (OV_h[u] W_U^T)[v]`.  That composition is only worth
    trusting if it predicts what actually happens when the head is removed.
    For each head this measures the agreement between

      predicted delta-logit  = (A_h * gauge) @ W_U^T      (the additive term)
      actual   delta-logit   = logits(full) - logits(drop head h)

    on real held text, as a Pearson correlation, a Spearman correlation and a
    sign-agreement rate.  A head whose composed prediction does not track its
    causal effect may not be named."""
    out = {}
    from scipy.stats import spearmanr
    for li in range(D.L):
        for h in range(D.H):
            pr, ac = [], []
            for x, y in I1.held_batches(D, n_seq, T, batch):
                P = D.run(x)
                full = D.readout(P['r'])
                drop = D.readout(D.run(x, attn={li: ('drop', h)})['r'])
                # the head's own additive write, composed to the logit
                keep = D.run(x, attn={li: ('keep', h)})
                Ah = keep['A'][li]
                g = math.sqrt(D.Ws) / P['r'].norm(dim=-1, keepdim=True)
                p = (Ah * g) @ D.WU.t()
                p = p - p.mean(-1, keepdim=True)
                a = full - drop
                a = a - a.mean(-1, keepdim=True)
                pr.append(p.reshape(-1)[::37].cpu())
                ac.append(a.reshape(-1)[::37].cpu())
                break
            p = torch.cat(pr).double().numpy()
            a = torch.cat(ac).double().numpy()
            out[f'l{li}_head{h}'] = {
                'pearson': float(np.corrcoef(p, a)[0, 1]),
                'spearman': float(spearmanr(p, a).statistic),
                'sign_agreement': float((np.sign(p) == np.sign(a)).mean()),
                'n': int(p.size)}
    out['note'] = ('a composed prediction that tracks the causal effect is the '
                   'licence to name a sign; raw factor signs remain gauge')
    return out


@torch.no_grad()
def resample_ablation(D, n_seq=64, T=256, batch=8):
    """REVIEWER FIX for 'a zeroed layer pushes the model off distribution'.

    Instead of deleting a layer's attention write, replace it with the write
    that same layer produced on a DIFFERENT sequence (the batch rolled by one).
    The substituted vector is on-distribution by construction -- it is a real
    output of that module -- so the difference between the zero-ablation KL and
    the resample KL bounds how much of the knockout cost was distribution shift
    rather than lost information."""
    out, kl, ntok = {}, {}, 0
    for x, y in I1.held_batches(D, n_seq, T, batch):
        if x.shape[0] < 2:
            continue
        P = D.run(x)
        ref = D.readout(P['r'])
        lp = F.log_softmax(ref.float(), -1)
        p = lp.exp()
        cand = {}
        for li in range(D.L):
            alt = P['A'][li].roll(1, 0)
            cand[f'resample_attn_layer{li}'] = D.run(
                x, attn={li: ('inject', alt)})['r']
            cand[f'zero_attn_layer{li}'] = D.run(x, attn={li: 'zero'})['r']
        cand['resample_attn_all'] = D.run(
            x, attn={li: ('inject', P['A'][li].roll(1, 0))
                     for li in range(D.L)})['r']
        cand['zero_attn_all'] = D.run(
            x, attn={li: 'zero' for li in range(D.L)})['r']
        for s, r in cand.items():
            q = F.log_softmax(D.readout(r).float(), -1)
            kl[s] = kl.get(s, 0.0) + float((p * (lp - q)).sum())
        ntok += y.numel()
    out = {k: v / ntok for k, v in kl.items()}
    for li in list(range(D.L)) + ['all']:
        z, r = out.get(f'zero_attn_layer{li}'), out.get(f'resample_attn_layer{li}')
        if li == 'all':
            z, r = out['zero_attn_all'], out['resample_attn_all']
        if z and r:
            out[f'distribution_shift_share_layer{li}'] = max(0.0, (z - r)) / z
    out['note'] = ('resample >= zero would mean the zeroing was the GENTLER '
                   'intervention; resample < zero means part of the zeroing '
                   'cost was distribution shift, and the share is quoted')
    return out


@torch.no_grad()
def composition_budget(D, n_seq=32, T=256, batch=8):
    """WHAT DOES LAYER l READ, IN THE STREAM'S OWN UNITS?

    Layer l's read is rms(e + A_0 + M_0 + ... ).  Induction needs the
    attention->attention path: layer 1's KEYS must depend on what layer 0's
    ATTENTION wrote.  So the quantity that decides whether that path is even
    open is how much of layer 1's read is layer-0 attention.  Reported three
    ways, none of which is a sign: (a) the norm share of each write in the read
    vector, (b) the relative change in the layer-l attention PATTERN when each
    upstream write is deleted from the read only (the residual is untouched, so
    this isolates the read), (c) the same for the layer-l value."""
    out = {}
    accum = {}
    n = 0
    for x, y in I1.held_batches(D, n_seq, T, batch):
        P = D.run(x)
        for li in range(1, D.L):
            pre = P['e']
            parts = {'e': P['e']}
            for j in range(li):
                parts[f'A{j}'] = P['A'][j]
                parts[f'M{j}'] = P['M'][j]
                pre = pre + P['A'][j] + P['M'][j]
            hn = _rms(pre, D.Ws)
            pat = D._pat_from(li, hn)
            pn = pat.norm()
            for nm, v in parts.items():
                k = f'l{li}_read_norm_share_{nm}'
                accum[k] = accum.get(k, 0.0) + float(
                    (v.norm(dim=-1) / pre.norm(dim=-1)).mean())
                hn2 = _rms(pre - v, D.Ws)
                dp = (D._pat_from(li, hn2) - pat).norm() / pn
                k2 = f'l{li}_pattern_rel_change_without_{nm}'
                accum[k2] = accum.get(k2, 0.0) + float(dp)
                v1 = (hn @ D.Wv[li].t())
                v2 = (hn2 @ D.Wv[li].t())
                k3 = f'l{li}_value_rel_change_without_{nm}'
                accum[k3] = accum.get(k3, 0.0) + float(
                    (v2 - v1).norm() / v1.norm())
        n += 1
    out = {k: v / n for k, v in accum.items()}
    out['note'] = ('a layer-1 pattern that barely moves when A0 is removed '
                   'from its read means the attention->attention composition '
                   'path -- the one induction needs -- is numerically closed, '
                   'whatever the architecture permits')
    return out


@torch.no_grad()
def natural_induction(D, n_seq=512, T=256, seed=0):
    """INDUCTION ON REAL TEXT, as a causal intervention.

    The synthetic battery uses unigram-sampled random tokens, and a reviewer is
    entitled to ask whether a model could induct on natural text while failing
    on random text.  So the same question is asked on held text by BREAKING the
    induction evidence: find a position i whose token has occurred before at j,
    take the model's probability for the token that followed that earlier
    occurrence (x[j+1], the induction target), then re-run with x[j+1] replaced
    by a random token and measure how much that probability falls.

    THE CONFOUND, AND THE FIX.  Overwriting x[j+1] with a random token also
    REMOVES the target token from the prefix, and these models have a positive
    BAG effect, so that patch measures bag + order and would report 'induction'
    for a purely order-free model.  The order-specific patch is therefore a
    SWAP: exchange x[j+1] with another prefix token.  The bag is then exactly
    preserved (it is a permutation) and only the adjacency that induction reads
    is destroyed.  Both patches are reported; only the swap licenses an
    induction claim.  Each is additionally differenced against the same patch
    scored on a control token, which absorbs the generic 'the prefix changed'
    effect."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    arr = tf_corpus.load_split(D.V, 'held', n_seq, tok=D.cfg.tok)
    x = torch.from_numpy(arr[:, :T]).to(D.dev)
    B = x.shape[0]
    tgt = I1._induction_target(x, D.V)                     # (B,T)
    # the source position j+1 for each i: recompute the last-occurrence index
    last = torch.full((B, D.V), -1, dtype=torch.long, device=D.dev)
    src = torch.full((B, T), -1, dtype=torch.long, device=D.dev)
    for i in range(T):
        cur = x[:, i:i + 1]
        prev = last.gather(1, cur)[:, 0]
        src[:, i] = torch.where((prev >= 0) & (prev + 1 < T), prev + 1,
                                torch.full_like(prev, -1))
        last.scatter_(1, cur, torch.full_like(cur, i))
    # one eligible position per sequence: the LAST eligible one before T-1
    elig = (src >= 0) & (torch.arange(T, device=D.dev)[None] > T // 2) \
        & (torch.arange(T, device=D.dev)[None] < T - 1)
    idxs = torch.where(elig.any(1))[0]
    pos = elig.float().cumsum(1).argmax(1)                  # last eligible
    pos = pos[idxs]
    xs = x[idxs]
    jj = src[idxs].gather(1, pos[:, None])[:, 0]
    tt = tgt[idxs].gather(1, pos[:, None])[:, 0]
    nn = xs.shape[0]
    rnd = torch.randint(0, D.V, (nn,), generator=g).to(D.dev)
    x_rand = xs.clone()
    x_rand.scatter_(1, jj[:, None], rnd[:, None])
    # ORDER-ONLY patch: swap x[j+1] with another prefix token (bag preserved)
    u = torch.rand(nn, T, generator=g).to(D.dev)
    ar = torch.arange(T, device=D.dev)[None]
    ok = (ar < pos[:, None]) & (ar != jj[:, None]) & (ar != (jj - 1)[:, None])
    mm = (u * ok.float()).argmax(1)
    x_swap = xs.clone()
    a_ = x_swap.gather(1, jj[:, None])
    b_ = x_swap.gather(1, mm[:, None])
    x_swap.scatter_(1, jj[:, None], b_)
    x_swap.scatter_(1, mm[:, None], a_)
    ctrl = torch.randint(0, D.V, (nn,), generator=g).to(D.dev)
    acc = {k: [] for k in ('rand_t', 'rand_c', 'swap_t', 'swap_c')}
    for a in range(0, nn, 16):
        s = slice(a, a + 16)
        lp0 = F.log_softmax(D.model(xs[s]).float(), -1)
        lpr = F.log_softmax(D.model(x_rand[s]).float(), -1)
        lps = F.log_softmax(D.model(x_swap[s]).float(), -1)
        p = pos[s]
        r = torch.arange(p.shape[0], device=D.dev)
        acc['rand_t'].append((lp0[r, p, tt[s]] - lpr[r, p, tt[s]]).cpu())
        acc['rand_c'].append((lp0[r, p, ctrl[s]] - lpr[r, p, ctrl[s]]).cpu())
        acc['swap_t'].append((lp0[r, p, tt[s]] - lps[r, p, tt[s]]).cpu())
        acc['swap_c'].append((lp0[r, p, ctrl[s]] - lps[r, p, ctrl[s]]).cpu())
    v = {k: torch.cat(x_).double().numpy() for k, x_ in acc.items()}

    def stat(d):
        se = d.std(ddof=1) / math.sqrt(d.size)
        return {'mean': float(d.mean()), 'se': float(se),
                't': float(d.mean() / se),
                'detectable_effect_floor_nats_3se': float(3 * se)}
    return {
        'n_positions': int(nn),
        'bag_plus_order_patch_random_token': stat(v['rand_t'] - v['rand_c']),
        'ORDER_ONLY_patch_swap': stat(v['swap_t'] - v['swap_c']),
        'raw': {k: float(x_.mean()) for k, x_ in v.items()},
        'note': 'the SWAP row is the induction number: it preserves the prefix '
                'token multiset exactly and destroys only the adjacency that '
                'induction reads.  The RANDOM row also removes the target from '
                'the prefix and therefore contains the order-free BAG effect, '
                'which these models do have; it must not be quoted as '
                'induction.'}


@torch.no_grad()
def fit_score_disjointness(D, n_seq=96, T=256):
    """'IS THE KL MEASURED ON THE TOKENS THE TABLES WERE FIT ON?'

    Answered two ways.  (1) Structurally: the only fitted objects in the ladder
    are the token-independent distance profile, the mean past-attention
    ablation value and the MLP hidden-unit ordering, all fitted on the
    ESTIMATION split; the model's own bigram table is fitted on nothing at all.
    (2) Empirically: the entire ladder is re-scored on the estimation split
    and the two sets of KLs are compared -- if any stage were fit to the held
    text it would look better on held than on est."""
    h = ladder2(D, n_seq, T, split='held', extra=False)
    e = ladder2(D, n_seq, T, split='est', extra=False)
    rows = {k: {'held': h[k]['kl_from_model'], 'est': e[k]['kl_from_model'],
                'held_minus_est': h[k]['kl_from_model'] - e[k]['kl_from_model']}
            for k in h if not k.startswith('_')}
    worst = max(abs(r['held_minus_est']) for r in rows.values())
    return {'per_stage': rows, 'max_abs_held_minus_est': worst,
            'fitted_objects': ['distance profile (est)',
                               'mean past-attention ablation value (est)',
                               'MLP hidden-unit importance ordering (est)'],
            'unfitted_objects': ["the model's own bigram table (weights only)",
                                 'every knockout stage (weights only)'],
            'note': 'a stage that had been fit to held would show a NEGATIVE '
                    'held_minus_est well outside the sampling noise'}


@torch.no_grad()
def mlp_composed_causal(D, n_seq=8, T=256):
    """THE FIX for the composition failure `composed_vs_causal` exposes.

    The rung-4 object `p_h * (OV_h W_U^T)` is the head's DIRECT route to the
    logit.  FINDING 2 already established that the direct route is dead and
    that attention acts by moving the MLP's argument, so the direct composition
    cannot be expected to predict a head's causal effect -- and it does not.

    The correct composed path is exact here because the MLP is bilinear: with
    the RMSNorm scalar gauge,
        MLP(x - d) - MLP(x) = -2 T(x, d) + T(d, d)   (exactly)
    so the head's full contribution can be propagated analytically.  This
    measures the same correlation for the THROUGH-MLP composition, which is the
    object any rung-4 naming statement must use from now on."""
    out = {}
    for x, y in I1.held_batches(D, n_seq, T, 8):
        break
    P = D.run(x)
    full = D.readout(P['r'])
    for li in range(D.L):
        for h in range(D.H):
            drop = D.run(x, attn={li: ('drop', h)})
            # analytic propagation of the removed write through everything
            # downstream, using the folded pipeline itself (exact, not a
            # linearisation): the difference of two folded residuals
            pred_r = P['r'] - drop['r']
            g = math.sqrt(D.Ws) / P['r'].norm(dim=-1, keepdim=True)
            p = (pred_r * g) @ D.WU.t()
            p = p - p.mean(-1, keepdim=True)
            a = full - D.readout(drop['r'])
            a = a - a.mean(-1, keepdim=True)
            pv = p.reshape(-1)[::37].double().cpu().numpy()
            av = a.reshape(-1)[::37].double().cpu().numpy()
            # and the DIRECT-route object for the same head, for contrast
            dr = (D.run(x, attn={li: ('keep', h)})['A'][li] * g) @ D.WU.t()
            dr = dr - dr.mean(-1, keepdim=True)
            dv = dr.reshape(-1)[::37].double().cpu().numpy()
            out[f'l{li}_head{h}'] = {
                'through_mlp_pearson': float(np.corrcoef(pv, av)[0, 1]),
                'through_mlp_sign_agreement': float(
                    (np.sign(pv) == np.sign(av)).mean()),
                'direct_route_pearson': float(np.corrcoef(dv, av)[0, 1]),
                'direct_route_sign_agreement': float(
                    (np.sign(dv) == np.sign(av)).mean())}
    out['note'] = ('through_mlp is the residual difference composed to the '
                   'logit through the SAME gauge; it differs from the causal '
                   'delta only by the tanh and the gauge change, so a high '
                   'correlation is the expected PASS and a low one would '
                   'indicate the readout nonlinearity dominates.  The contrast '
                   'with direct_route is the point: only the through-MLP '
                   'composition licences a naming statement.')
    return out


@torch.no_grad()
def causal_copy_test(D, npairs=512, seed=0):
    """FINDING 6 ('the heads are not copy heads') RE-DERIVED CAUSALLY.

    The original statement came from the composed table
    `p_h(t,u,1) * (OV_h[u] W_U^T)`, i.e. the direct route.  Here the identical
    question is asked as an intervention on the real model: build the two-token
    context [u, t], read the logits at position 1 with the head present and
    with the head removed, and ask where the ATTENDED token u ranks among the
    tokens the head boosts.  Nothing about this is a factor sign; the whole
    path -- pattern, value, output, both MLPs, readout -- is in the number."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    tr = tf_corpus.load_split(D.V, 'train', 2000, tok=D.cfg.tok)
    cnt = np.bincount(tr.reshape(-1), minlength=D.V).astype(np.float64) + 1.0
    p = torch.from_numpy(cnt / cnt.sum()).float()
    t = torch.multinomial(p, npairs, True, generator=g)
    u = torch.multinomial(p, npairs, True, generator=g)
    ctx = torch.stack([u, t], 1).to(D.dev)                    # [u, t]
    base = D.readout(D.run(ctx)['r'])[:, 1]                   # (n, V)
    out = {}
    for li in range(D.L):
        for h in range(D.H):
            dl = base - D.readout(D.run(ctx, attn={li: ('drop', h)})['r'])[:, 1]
            dl = dl - dl.mean(-1, keepdim=True)
            uu = u.to(D.dev)
            own = dl.gather(1, uu[:, None])[:, 0]
            rank = (dl > own[:, None]).sum(1)                 # 0 = top
            out[f'l{li}_head{h}'] = {
                'median_rank_of_attended_token': float(rank.median()),
                'mean_rank_of_attended_token': float(rank.float().mean()),
                'frac_attended_token_in_top100': float((rank < 100).float().mean()),
                'mean_own_logit_push': float(own.mean()),
                'z_of_own_push_vs_random_token': float(
                    (own.mean() - dl.mean()) / dl.std()),
                'vocab': D.V}
    out['note'] = ('rank 0 would mean attending to u boosts u more than any '
                   'other token; rank V/2 means attending to u says nothing '
                   'about u.  This is the causal version of the claim; the '
                   'composed-table version is in rung4.')
    return out


def main(stem, quick=False, skip_baselines=False):
    t0 = time.time()
    jp = f'{HERE}/{stem}_interp2.json'
    rep = json.load(open(jp)) if os.path.exists(jp) else {}
    rep['stem'] = stem
    rep['registered_predictions'] = I1.REGISTERED
    rep['registered_predictions_depth2'] = REGISTERED2
    json.dump(rep, open(jp, 'w'), indent=2)          # registered BEFORE running
    D = DeepFold(stem)
    rep['depth'] = D.L
    rep['width'] = D.cfg.width
    x = torch.from_numpy(tf_corpus.load_split(D.V, 'held', 4,
                                              tok=D.cfg.tok))[:, :128].to(D.dev)
    rep['decomposition_control'] = D.self_check(x)
    print('  control:', rep['decomposition_control'], flush=True)
    assert rep['decomposition_control']['pass'], rep['decomposition_control']
    n_seq, T = (16, 128) if quick else (96, 256)
    rep['rung2'] = rung2_layers(D)
    print(f'  rung2 {time.time()-t0:.0f}s', flush=True)
    if not skip_baselines:
        rep['rung3_baselines'] = I1.data_baselines(D, n_seq, T)
        print(f'  baselines {time.time()-t0:.0f}s', flush=True)
    rep['stream_geometry'] = stream_geometry2(D, min(n_seq, 32), T)
    print(f'  geometry {time.time()-t0:.0f}s', flush=True)
    rep['rung5_ladder'] = ladder2(D, n_seq, T)
    print(f'  ladder {time.time()-t0:.0f}s', flush=True)
    rep['ladder_order'] = ladder_order(D, min(n_seq, 64), T)
    print(f'  order test {time.time()-t0:.0f}s', flush=True)
    ind = [I1.induction_battery(D, seed=s, model=D.model)
           for s in range(3 if quick else 5)]
    rep['rung3_induction'] = {
        'per_seed': ind,
        'induction_score_mean': float(np.mean([i['induction_score'] for i in ind])),
        'induction_score_sd': float(np.std([i['induction_score'] for i in ind],
                                           ddof=1)),
        'bag_score_mean': float(np.mean([i['bag_score'] for i in ind])),
        'bag_score_sd': float(np.std([i['bag_score'] for i in ind], ddof=1)),
        'design': 'repeat vs shuffled-prefix vs fresh-prefix, matched token '
                  'multisets, scored on the second copy only'}
    rep['induction_power'] = induction_power(D)
    rep['induction_by_head'] = induction_by_head(D)
    rep['resample_ablation'] = resample_ablation(D)
    rep['composition_budget'] = composition_budget(D)
    rep['natural_induction'] = natural_induction(D, n_seq=1024)
    rep['fit_score_disjointness'] = fit_score_disjointness(D, min(n_seq, 64), T)
    print(f'  induction {time.time()-t0:.0f}s', flush=True)
    rep['rung4'] = I1.rung4(D)
    rep['composed_vs_causal'] = composed_vs_causal(D, min(n_seq, 32), T)
    rep['mlp_composed_causal'] = mlp_composed_causal(D)
    rep['causal_copy_test'] = causal_copy_test(D)
    print(f'  composed-vs-causal {time.time()-t0:.0f}s', flush=True)
    rep['bits_per_byte'] = tf_corpus.bits_per_byte(
        rep['rung5_ladder']['_model_ce'], D.V, tok=D.cfg.tok) \
        if hasattr(tf_corpus, 'bits_per_byte') else None
    rep['seconds'] = round(time.time() - t0, 1)
    json.dump(rep, open(jp, 'w'), indent=2)
    print(f'== interp2 written to {jp} ({rep["seconds"]}s)', flush=True)
    return rep


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stem', required=True)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--skip-baselines', action='store_true')
    ap.add_argument('--order-only', action='store_true')
    a = ap.parse_args()
    if a.order_only:
        D = DeepFold(a.stem)
        r = ladder_order(D)
        jp = f'{HERE}/{a.stem}_order.json'
        json.dump(r, open(jp, 'w'), indent=2)
        print(json.dumps(r, indent=2))
        raise SystemExit(0)
    r = main(a.stem, a.quick, a.skip_baselines)
    print(json.dumps({'ladder': r['rung5_ladder'],
                      'order': r['ladder_order']}, indent=2))
