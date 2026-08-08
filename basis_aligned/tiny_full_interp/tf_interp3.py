"""The SIX-ARCHITECTURE COMPARISON SLICE (GRID.md phase V1): one analysis path
that runs on every variant, so a difference between variants can never be a
difference between analysis scripts.

WHY A NEW FILE AND NOT AN EDIT TO tf_interp2.  `tf_interp2.DeepFold` asserts the
vanilla variant on entry and hard-codes three things a variant breaks:

  * the module input is `rms(x)` -- but B-F use per-SLOT RMSNorm, and for
    n_slots > 1 that is a different function of the same vector;
  * a module's write occupies the whole stream -- but B-F write into ONE SLOT,
    and C/D/E's decoders are physically slot-sized so the write must be
    SCATTERED, not masked;
  * the stream start equals the layer-0 module input -- true only when
    slot_norm is the global RMSNorm, i.e. only for vanilla.  Here the two are
    separated: `E` = rms(wte) is the stream start, `Ehn` = slot_norm(E) is the
    layer-0 module input, and they coincide exactly when n_slots == 1.

Plus two mechanisms with no vanilla analogue: the predicate variant adds named
terms to the pattern (a signed positional profile, a previous-token match and a
same-token match), and the codebook variant quantises already-written slots at
every module input.

WHAT IS DELEGATED RATHER THAN RE-DERIVED.  `slot_norm`, `remnants`, `write_out`,
`_qz_full` and `pred_terms` are called on the MODEL, exactly as
`tf_model.fold_forward` does.  The fold gate certifies the FOLD OBJECTS (the
layer-0 token factors and the MLP tensors), not those helpers, and re-deriving
them here would only add a second place to be wrong.  `self_check` gates the
whole pipeline against the model's own forward anyway.

THE POSITIVE CONTROL THAT MAKES THIS COMPARABLE (`--control`).  Every stage of
this file is run on the VANILLA checkpoint and required to reproduce
`tf_interp2`'s number for the same stage.  Without that gate, "variant B's
ladder differs from vanilla's" would be unfalsifiable, because vanilla's
published ladder came out of different code.  Vanilla is therefore ALSO
re-measured here and the comparison table quotes the tf_interp3 numbers for all
six.

SIGN IS A GAUGE FREEDOM (standing rule, inherited).  Nothing is named from the
sign of a factor.

Usage
    python tf_interp3.py --control                     # gate against tf_interp2
    python tf_interp3.py --stem tf_slots_d2_w128_b8192_s0
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
import tf_interp2 as I2
import tf_model as M

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
_rms = I1._rms


# ---------------------------------------------------------------------------
class VariantFold(I2.DeepFold):
    """The folded pipeline for ANY variant at any depth.  Same public API as
    `DeepFold`, so every driver in `tf_interp` / `tf_interp2` that only touches
    `run`, `readout`, `model`, `A`/`M`/`r` runs on it unchanged."""

    def __init__(self, stem, device=DEV):
        model, cfg, ck = tf_fold.load_checkpoint(stem, device)
        self.stem, self.model, self.cfg, self.dev = stem, model, cfg, device
        self.L = cfg.depth
        self.V, self.H, self.hd = cfg.vocab, cfg.n_heads, cfg.head_dim
        self.Ws, self.Dc, self.s = model.Ws, model.Dc, model.s
        self.G = cfg.n_slots
        # stream start vs layer-0 module input -- EQUAL only when n_slots == 1
        self.E = F.rms_norm(model.wte.weight.detach().float(), (self.Ws,))
        self.Ehn = model.token_input_table().to(device)          # (V, Ws)
        self.WU = model.wte.weight.detach().float()              # (V, Ws)
        self.cos, self.sin = model.cos.to(device), model.sin.to(device)
        f = model.fold_layer0_qk(materialize=False, device=device)
        self.Q1, self.K1, self.Q2, self.K2 = f['Q1'], f['K1'], f['Q2'], f['K2']
        self.Vv = f['Vv']
        Wp0 = model.h[0].c_proj.weight.detach().float()
        # OV composed to the STREAM: value -> output -> the slot it is written
        # into.  For a small decoder Wp0 is (s, Dc) so this is a slot-sized
        # write and write_out SCATTERS it; for a masked decoder write_out zeroes
        # everything outside slot 0.  Either way OV is (H, V, Ws) and every
        # downstream composition (rung 4, stream geometry) is unchanged.
        ov = torch.stack([self.Vv[h] @ Wp0[:, h * self.hd:(h + 1) * self.hd].t()
                          for h in range(self.H)])
        self.OV = model.write_out(ov, 0, ov.shape[0], ov.shape[1])
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
        self.T = self.Tl[0]
        self.bias = self.bl[0]
        self._est = None
        self._bigt = None

    # --------------------------------------------------------- variant pieces
    def _pre(self, k, x, cache=None):
        """A module input: per-slot norm, then quantisation of the slots that
        have already been written (`k` = how many writes precede this consumer).
        Reduces to the global RMSNorm when n_slots == 1 and qz is off."""
        return self.model._qz_full(self.model.slot_norm(x), k, cache, None)

    def _match(self, idx):
        Tq = idx.shape[1]
        maskf = self.model.mask[:Tq, :Tq].float()
        return M.match_kernels(idx, maskf) + (maskf,)

    def _pred(self, li, idx):
        if not self.model.pred_on:
            return None
        Kprev, Ksame, maskf = self._match(idx)
        return self.model.pred_terms(li, Kprev, Ksame, maskf, idx.shape[1])

    # ------------------------------------------------------------- pieces
    def _pat_layer0(self, idx):
        B, Tq = idx.shape
        cos, sin = self.cos[None, :Tq, None, :], self.sin[None, :Tq, None, :]

        def g(fac):
            return M.apply_rot(fac.permute(1, 0, 2)[idx], cos, sin)
        s1 = torch.einsum('bqhd,bkhd->bhqk', g(self.Q1), g(self.K1)) / self.hd
        s2 = torch.einsum('bqhd,bkhd->bhqk', g(self.Q2), g(self.K2)) / self.hd
        pat = (s1 * s2).masked_fill(~self.model.mask[:Tq, :Tq], 0.0)
        pt = self._pred(0, idx)
        return pat if pt is None else pat + pt

    def _pat_from(self, li, hn, rot=True, idx=None):
        """The predicate variant's pattern is not a function of `hn` alone, so
        `idx` is REQUIRED there rather than silently dropped."""
        B, Tq, _ = hn.shape
        cos, sin = self.cos[None, :Tq, None, :], self.sin[None, :Tq, None, :]

        def qk(W):
            z = F.rms_norm((hn @ W.t()).view(B, Tq, self.H, self.hd), (self.hd,))
            return M.apply_rot(z, cos, sin) if rot else z
        s1 = torch.einsum('bqhd,bkhd->bhqk', qk(self.Wq[li]), qk(self.Wk[li])) \
            / self.hd
        s2 = torch.einsum('bqhd,bkhd->bhqk', qk(self.Wq2[li]), qk(self.Wk2[li])) \
            / self.hd
        pat = (s1 * s2).masked_fill(~self.model.mask[:Tq, :Tq], 0.0)
        if self.model.pred_on:
            assert idx is not None, \
                'the predicate variant needs idx to build its named terms'
            pat = pat + self._pred(li, idx)
        return pat

    def _attn_out(self, li, pat, hn, idx=None):
        if li == 0 and idx is not None:
            ov = self.OV.permute(1, 0, 2)[idx]                   # (B,T,H,Ws)
            return torch.einsum('bhqk,bkho->bqo', pat, ov)
        B, Tq, _ = hn.shape
        v = (hn @ self.Wv[li].t()).view(B, Tq, self.H, self.hd)
        y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, Tq, self.Dc)
        return self.model.write_out(y @ self.Wproj[li].t(), 2 * li, B, Tq)

    def _mlp_norm(self, li, xn, topk=None):
        """MLP applied to an ALREADY normalised (and quantised) input, with the
        write placed in its slot."""
        Dn, Lf, Rf, bias = self.Dn[li], self.Lf[li], self.Rf[li], self.bl[li]
        if topk is not None:
            keep = self._mlp_use(li)[1][:topk]
            Dn, Lf, Rf = Dn[:, keep], Lf[keep], Rf[keep]
        w = ((xn @ Lf.t()) * (xn @ Rf.t())) @ Dn.t() + bias
        return self.model.write_out(w, 2 * li + 1, xn.shape[0], xn.shape[1])

    def _mlp(self, li, r, topk=None):
        return self._mlp_norm(li, self._pre(2 * li + 1, r), topk)

    def _mlp_use(self, li):
        key = f'_use{li}'
        if not hasattr(self, key):
            arr = tf_corpus.load_split(self.V, 'est', 16, tok=self.cfg.tok)
            x = torch.from_numpy(arr[:, :256]).to(self.dev)
            r = self.run(x)['pre_mlp'][li]
            xn = self._pre(2 * li + 1, r)
            u = (self.Dn[li].norm(dim=0)
                 * ((xn @ self.Lf[li].t()) * (xn @ self.Rf[li].t())
                    ).abs().mean((0, 1)))
            setattr(self, key, (u, u.argsort(descending=True)))
        return getattr(self, key)

    # ------------------------------------------------------------- the run
    def run(self, idx, attn=None, mlp=None, reads=None, mlp_reads=None,
            want=()):
        B, Tq = idx.shape
        attn = attn or {}
        mlp = mlp or {}
        reads = reads or {}
        e = self.E[idx]
        rem = self.model.remnants(e)
        cache = {} if self.model.qz_on else None
        streams = []
        P = {'e': e, 'rem': rem, 'idx': idx, 'A': [], 'M': [], 'pre_mlp': [],
             'read': [], 'pat': []}

        def entry(li):
            tot = rem[li]
            for i in range(2 * li):
                tot = streams[i] if tot is None else tot + streams[i]
            return tot

        for li in range(self.L):
            x = entry(li)
            hn_true = self._pre(2 * li, x, cache)
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
            streams.append(A)
            x = x + A
            P['pre_mlp'].append(x)
            mm = mlp.get(li, 'full')
            xm = x
            if mlp_reads and li in mlp_reads:
                xm = mlp_reads[li](P, x)
            if mm == 'zero':
                Mo = torch.zeros_like(x)
            else:
                xn = self._pre(2 * li + 1, xm, cache)
                Mo = self._mlp_norm(li, xn, topk=mm[1] if isinstance(mm, tuple)
                                    else None)
            P['M'].append(Mo)
            streams.append(Mo)
            P['read'].append(hn_true)
        P['r'] = entry(self.L)
        return P

    def _attn_step(self, li, idx, hn_qk, hn_v, mode, P, use_fold0=True):
        Tq = idx.shape[1]
        if mode == 'zero':
            return torch.zeros(idx.shape[0], Tq, self.Ws, device=self.dev)
        rot = not (isinstance(mode, str) and mode == 'norot')
        if li == 0 and use_fold0 and rot:
            pat = self._pat_layer0(idx)
        else:
            pat = self._pat_from(li, hn_qk, rot=rot, idx=idx)
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
            elif kind == 'predoff':
                # PREDICATE-ONLY knockout: the learned bilinear branches with
                # the named terms removed (or the named terms alone).
                pat = self._pat_predsplit(li, idx, hn_qk, arg, use_fold0)
            elif kind == 'inject':
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

    def _pat_predsplit(self, li, idx, hn, which, use_fold0):
        """`which` in {'branches', 'named', 'no_prev', 'no_same', 'no_prof'}:
        the pattern with part of the predicate machinery removed.  Non-predicate
        variants have no named terms, so 'branches' is the full pattern and
        'named' is zero -- stated rather than special-cased away."""
        Tq = idx.shape[1]
        was = self.model.pred_on
        try:
            self.model.pred_on = False
            base = self._pat_layer0(idx) if (li == 0 and use_fold0) \
                else self._pat_from(li, hn, idx=idx)
        finally:
            self.model.pred_on = was
        if which == 'branches' or not was:
            return base if which != 'named' else torch.zeros_like(base)
        Kprev, Ksame, maskf = self._match(idx)
        prof = self.model.pred_prof[li][:, self.model.offmat[:Tq, :Tq]] * maskf
        b = self.model.pred_b[li].view(1, -1, 1, 1) * Kprev[:, None]
        c = self.model.pred_c[li].view(1, -1, 1, 1) * Ksame[:, None]
        named = prof[None] + b + c
        if which == 'named':
            return named
        if which == 'no_prev':
            return base + named - b
        if which == 'no_same':
            return base + named - c
        if which == 'no_prof':
            return base + named - prof[None]
        raise ValueError(which)

    def _pat_freq(self, li, idx, hn, k):
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
        mk = m.permute(1, 0, 2)

        def qk(W):
            z = F.rms_norm((hn @ W.t()).view(B, Tq, self.H, self.hd), (self.hd,))
            return M.apply_rot(z * mk[None], cos, sin)
        s1 = torch.einsum('bqhd,bkhd->bhqk', qk(self.Wq[li]),
                          qk(self.Wk[li])) / self.hd
        s2 = torch.einsum('bqhd,bkhd->bhqk', qk(self.Wq2[li]),
                          qk(self.Wk2[li])) / self.hd
        pat = (s1 * s2).masked_fill(~self.model.mask[:Tq, :Tq], 0.0)
        if self.model.pred_on:
            pat = pat + self._pred(li, idx)
        return pat

    # ---------------------------------------- depth-1 compatibility shim
    def terms(self, idx, pat=None):
        P = self.run(idx)
        A0 = self._attn_step(0, idx, self.Ehn[idx], self.Ehn[idx], 'self', P)
        A = sum(P['A'])
        return {'e': P['rem'][self.L], 'A': A, 'A0': A0, 'Apast': A - A0,
                'M': sum(P['M']), 'r': P['r'], 'pat': None}

    def readout(self, r):
        return 30 * torch.tanh((_rms(r, self.Ws) @ self.WU.t()) / 30)

    # ------------------------------------------- weights-only bigram table
    @torch.no_grad()
    def bigram_table(self, chunk=1024):
        """The model's exact output on a LENGTH-1 context, as a (V, Ws) row
        table.  Computed by pushing every vocabulary item through the folded
        pipeline as its own length-1 sequence: still a pure token function of
        the weights with no data, and -- unlike a hand-written delta=0 shortcut
        -- correct for every variant mechanism (named terms at offset 0,
        quantisation of the already-written slots, the shrink remnants) without
        a second implementation of any of them to get wrong."""
        if self._bigt is None:
            rows = []
            for a in range(0, self.V, chunk):
                ids = torch.arange(a, min(a + chunk, self.V),
                                   device=self.dev)[:, None]
                rows.append(self.run(ids)['r'][:, 0])
            self._bigt = torch.cat(rows)
        return self._bigt

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
                pat = self._pat_layer0(b) if li == 0 \
                    else self._pat_from(li, hn, idx=b)
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
            one = idx[:, :1]
            ref1 = self.model(one)[:, 0]
            rec1 = self.readout(self.bigram_table()[one[:, 0]])
            out['length1_table_rel_logit_diff'] = float(
                (rec1 - ref1).abs().max()) / max(float(ref1.abs().max()), 1e-30)
            # factored MLP == folded tensor MLP (the rung-2 object)
            xn = self._pre(1, P['pre_mlp'][0])
            mt = self.model.write_out(
                torch.einsum('oij,bti,btj->bto', self.Tl[0], xn, xn) + self.bl[0],
                1, xn.shape[0], xn.shape[1])
            out['mlp_tensor_vs_factored_rel'] = float(
                (mt - P['M'][0]).abs().max() / mt.abs().max())
            p_fold = self._pat_layer0(idx)
            p_w = self._pat_from(0, self.Ehn[idx], idx=idx)
            out['layer0_folded_vs_weight_pattern_rel'] = float(
                (p_fold - p_w).abs().max() / p_fold.abs().max())
            # the additive decomposition the whole attribution rests on
            tot = P['rem'][self.L] + sum(P['A']) + sum(P['M'])
            out['residual_additivity_rel'] = float(
                (tot - P['r']).abs().max() / P['r'].abs().max())
        out['pass'] = bool(out['pipeline_rel_logit_diff'] < 1e-5
                           and out['length1_table_rel_logit_diff'] < 1e-5
                           and out['mlp_tensor_vs_factored_rel'] < 1e-5
                           and out['layer0_folded_vs_weight_pattern_rel'] < 1e-5
                           and out['residual_additivity_rel'] < 1e-5)
        return out


# ---------------------------------------------------------------- the ladder
@torch.no_grad()
def ladder_v(D, n_seq=96, T=256, batch=8, split='held', extra=True):
    """`tf_interp2.ladder2` stage for stage, with the read substitutions written
    in a form that means the same thing in every variant.

    In vanilla the layer-0 module input IS the stream start, so ladder2 could
    write the counterfactual reads as `P['e']` and `_rms(P['e'] + A0)`.  Under
    per-slot norm those two are different functions, so here every
    counterfactual read is built the way a real module input is built:
    `slot_norm` (and, for the codebook variant, quantisation) applied to the
    counterfactual stream.  On a vanilla checkpoint the two definitions are the
    same expression and the control gate checks that they agree numerically."""
    L = D.L
    acc, kl, ntok = {}, {}, 0
    D.est(T)
    bigt = D.bigram_table()
    allL = list(range(L))

    def cands(x):
        c = {}
        P = D.run(x)
        c['full_exact'] = P['r']
        c['embed_only'] = P['rem'][L]
        c['plus_self_attn'] = D.run(x, attn={l: 'self' for l in allL},
                                    mlp={l: 'zero' for l in allL})['r']
        c['model_bigram'] = bigt[x]
        c['no_attention_at_all'] = D.run(x, attn={l: 'zero' for l in allL})['r']
        c['past_attn_mean_ablated'] = D.run(
            x, attn={l: 'meanpast' for l in allL})['r']
        c['no_mlp'] = D.run(x, mlp={l: 'zero' for l in allL})['r']
        c['mlp_write_only'] = sum(P['M'])
        c['attn_write_only'] = P['rem'][L] + sum(P['A'])
        Pself = D.run(x, attn={l: 'self' for l in allL})
        c['past_attn_direct_route_only'] = Pself['r'] + sum(
            P['A']) - sum(Pself['A'])
        c['past_attn_mlp_route_only'] = P['r'] - (sum(P['A']) - sum(Pself['A']))
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
                def mk(fn, li_=li):
                    return lambda P_: D._pre(2 * li_, fn(P_, li_), {})
                emb = mk(lambda P_, l: P_['rem'][l])
                eA0 = mk(lambda P_, l: P_['rem'][l] + P_['A'][0])
                eM0 = mk(lambda P_, l: P_['rem'][l] + P_['M'][0])
                c[f'l{li}_reads_embedding'] = D.run(x, reads={li: emb})['r']
                c[f'l{li}_qk_reads_embedding'] = D.run(
                    x, reads={li: ('qk', emb)})['r']
                c[f'l{li}_v_reads_embedding'] = D.run(
                    x, reads={li: ('v', emb)})['r']
                c[f'l{li}_reads_e_plus_attn0'] = D.run(x, reads={li: eA0})['r']
                c[f'l{li}_reads_e_plus_mlp0'] = D.run(x, reads={li: eM0})['r']
            # --- predicate split (identity on the other five variants) ---
            if D.model.pred_on:
                for w in ('branches', 'named', 'no_prev', 'no_same', 'no_prof'):
                    c[f'pred_{w}_all_layers'] = D.run(
                        x, attn={l: ('predoff', w) for l in allL})['r']
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


# ------------------------------------------------------- composition budget
@torch.no_grad()
def composition_budget_v(D, n_seq=32, T=256, batch=8):
    """IS THE ATTENTION-TO-ATTENTION PATH OPEN?  Three measures, and only the
    third is comparable across the partition boundary.

    (a) norm shares and read sensitivities, as in `tf_interp2` -- kept so the
        vanilla numbers reproduce, but DECLARED NOT COMPARABLE across variants:
        with per-slot RMSNorm, removing a write from the read zeroes a whole
        slot, so a large change is arithmetic and not learning.
    (b) the same sensitivities measured in the NORMALISED read `hn` (what the
        layer actually multiplies), which is what per-slot norm pins.
    (c) THE ARCHITECTURE-NEUTRAL ONE: the KL cost of deleting each upstream
        write from layer l's read ONLY, with the residual stream untouched and
        everything downstream recomputed.  This is a causal intervention with
        the same meaning in every variant, and it is the number the
        attention-to-attention verdict is quoted from."""
    accum, n = {}, 0
    for x, y in I1.held_batches(D, n_seq, T, batch):
        P = D.run(x)
        for li in range(1, D.L):
            pre = P['rem'][li]
            parts = {'e': P['rem'][li]}
            for j in range(li):
                parts[f'A{j}'] = P['A'][j]
                parts[f'M{j}'] = P['M'][j]
                pre = pre + P['A'][j] + P['M'][j]
            hn = D._pre(2 * li, pre)
            pat = D._pat_from(li, hn, idx=x)
            pn = pat.norm()
            for nm, v in parts.items():
                hn2 = D._pre(2 * li, pre - v)
                for k, val in (
                        (f'l{li}_read_norm_share_{nm}',
                         float((v.norm(dim=-1) / pre.norm(dim=-1)).mean())),
                        (f'l{li}_normed_read_rel_change_without_{nm}',
                         float((hn2 - hn).norm() / hn.norm())),
                        (f'l{li}_pattern_rel_change_without_{nm}',
                         float((D._pat_from(li, hn2, idx=x) - pat).norm() / pn)),
                        (f'l{li}_value_rel_change_without_{nm}',
                         float(((hn2 - hn) @ D.Wv[li].t()).norm()
                               / (hn @ D.Wv[li].t()).norm()))):
                    accum[k] = accum.get(k, 0.0) + val
        n += 1
    out = {k: v / n for k, v in accum.items()}
    # ---- (c) the causal read-deletion KL ----
    kl, ntok = {}, 0
    for x, y in I1.held_batches(D, n_seq, T, batch):
        P = D.run(x)
        lp = F.log_softmax(D.readout(P['r']).float(), -1)
        p = lp.exp()
        cand = {}
        for li in range(1, D.L):
            def mk(drop, li_=li):
                def fn(P_):
                    v = P_['rem'][li_]
                    for j in range(li_):
                        if drop != f'A{j}':
                            v = v + P_['A'][j]
                        if drop != f'M{j}':
                            v = v + P_['M'][j]
                    return D._pre(2 * li_, v, {})
                return fn
            for j in range(li):
                cand[f'l{li}_read_without_A{j}'] = D.run(
                    x, reads={li: mk(f'A{j}')})['r']
                cand[f'l{li}_read_without_M{j}'] = D.run(
                    x, reads={li: mk(f'M{j}')})['r']
                cand[f'l{li}_qk_read_without_A{j}'] = D.run(
                    x, reads={li: ('qk', mk(f'A{j}'))})['r']
        for s, r in cand.items():
            q = F.log_softmax(D.readout(r).float(), -1)
            kl[s] = kl.get(s, 0.0) + float((p * (lp - q)).sum())
        ntok += y.numel()
    out['causal_read_deletion_kl'] = {k: v / ntok for k, v in kl.items()}
    out['note'] = (
        'norm shares and pattern sensitivities are NOT comparable across the '
        'partition boundary: under per-slot RMSNorm, deleting a write from the '
        'read empties a whole slot, so a big change there is arithmetic.  The '
        'attention-to-attention verdict is quoted from '
        'causal_read_deletion_kl[l1_read_without_A0], an intervention with the '
        'same meaning in every variant.')
    return out


@torch.no_grad()
def read_ablation_causal(D, n_seq=32, T=256, batch=8):
    """THE NORMALISATION-INVARIANT COMPOSITION BUDGET.

    Norm share is NOT a normalisation-invariant statistic and is not what the
    attention-to-attention verdict may rest on.  Here each upstream write is
    removed from layer l's READ ONLY -- the residual stream is untouched and
    every downstream module is recomputed -- in BOTH the zeroing and the
    resampling flavour, and the effect is reported as a KL and a CE against the
    true model plus the relative change it induces in layer l's pattern and
    values.  All five quantities are ratios or model outputs, so none of them
    can be moved by a change of normalisation convention alone.

    RESAMPLE is the harsher and more honest of the two: the substituted write is
    one the SAME module produced on a different sequence, so it is
    on-distribution by construction and the intervention cannot be dismissed as
    'you pushed the read somewhere no read ever goes'.  Zeroing is reported
    beside it as the lower bound (the parent finding: resample beat zero at 13
    of 14 layer-cells)."""
    kl, ce, ntok, geo, ng = {}, {}, 0, {}, 0
    for x, y in I1.held_batches(D, n_seq, T, batch):
        if x.shape[0] < 2:
            continue
        P = D.run(x)
        lp = F.log_softmax(D.readout(P['r']).float(), -1)
        p = lp.exp()
        for li in range(1, D.L):
            srcs = {'e': P['rem'][li]}
            for j in range(li):
                srcs[f'A{j}'] = P['A'][j]
                srcs[f'M{j}'] = P['M'][j]
            full = sum(srcs.values())
            hn = D._pre(2 * li, full)
            pat = D._pat_from(li, hn, idx=x)
            val = hn @ D.Wv[li].t()
            for nm, v in srcs.items():
                for how, rep in (('zero', torch.zeros_like(v)),
                                 ('resample', v.roll(1, 0))):
                    alt = full - v + rep
                    hn2 = D._pre(2 * li, alt)
                    k = f'l{li}_read_{how}_{nm}'
                    # a read substitution is a CLOSURE over the counterfactual
                    # stream, so the residual is untouched by construction
                    r = D.run(x, reads={li: (lambda P_, a=alt, li_=li:
                                             D._pre(2 * li_, a, {}))})['r']
                    q = F.log_softmax(D.readout(r).float(), -1)
                    kl[k] = kl.get(k, 0.0) + float((p * (lp - q)).sum())
                    cc, n = I1.ce_of(D.readout(r), y)
                    ce[k] = ce.get(k, 0.0) + float(cc)
                    geo[k + '_pattern_rel'] = geo.get(k + '_pattern_rel', 0.0) \
                        + float((D._pat_from(li, hn2, idx=x) - pat).norm()
                                / pat.norm())
                    geo[k + '_value_rel'] = geo.get(k + '_value_rel', 0.0) \
                        + float(((hn2 @ D.Wv[li].t()) - val).norm() / val.norm())
                    geo[k + '_read_rel'] = geo.get(k + '_read_rel', 0.0) \
                        + float((hn2 - hn).norm() / hn.norm())
            ng += 1
        ntok += y.numel()
    out = {'kl_from_model': {k: v / ntok for k, v in kl.items()},
           'ce': {k: v / ntok for k, v in ce.items()},
           'relative_change': {k: v / max(ng, 1) for k, v in geo.items()}}
    out['note'] = (
        'the attention-to-attention verdict is quoted from '
        'kl_from_model[l1_read_resample_A0] with l1_read_zero_A0 beside it as '
        'the lower bound.  KL, CE and the relative pattern/value changes are '
        'all invariant to the normalisation convention, unlike a norm share.')
    return out


@torch.no_grad()
def norm_confound_control(D, G=4, n_seq=32, T=256, batch=8):
    """THE REVIEWER'S SANITY CHECK, made explicit.

    Objection: the slot variants apply per-SLOT RMSNorm, which equalises what
    each module contributes to a read, so a balanced norm share could be
    mechanical and say nothing about the computation.

    Test: impose a G-way slot norm on THIS model at analysis time -- no
    retraining, same weights -- and recompute every composition-budget statistic
    under it.  Whatever moves is a property of the normalisation convention and
    not of the model.

    Two things are expected and both are reported honestly:
      * `pre_norm_share` CANNOT move, because it is measured on the residual
        stream before any norm is applied.  So the norm-share number is not
        literally an artifact of the metric -- but it is a statistic about the
        stream's magnitudes, which is exactly what the training pressure the
        slot norm removes was shaping, so it is still not comparable across the
        partition boundary.
      * `post_norm_share` -- the share of the NORMALISED read carried by each
        slot -- is forced to 1/G by construction and is therefore worthless as
        evidence.  It is reported so nobody quotes it later.
    The pattern and value sensitivities are the interesting rows: if imposing
    the slot norm on the plain model reproduces the slot variant's
    sensitivities, the sensitivity metric is confounded too and only the causal
    KL survives."""
    Ws = D.Ws
    S = Ws // G

    def gnorm(x):
        sh = x.shape
        return F.rms_norm(x.view(*sh[:-1], G, S), (S,)).view(sh)

    acc, n = {}, 0
    for x, y in I1.held_batches(D, n_seq, T, batch):
        P = D.run(x)
        for li in range(1, D.L):
            srcs = {'e': P['rem'][li]}
            for j in range(li):
                srcs[f'A{j}'] = P['A'][j]
                srcs[f'M{j}'] = P['M'][j]
            pre = sum(srcs.values())
            for tag, nf in (('true_norm', lambda z: D._pre(2 * li, z)),
                            ('imposed_slot_norm', gnorm)):
                hn = nf(pre)
                pat = D._pat_from(li, hn, idx=x)
                val = hn @ D.Wv[li].t()
                for nm, v in srcs.items():
                    hn2 = nf(pre - v)
                    for k, z in (
                            (f'{tag}_l{li}_pattern_rel_change_without_{nm}',
                             float((D._pat_from(li, hn2, idx=x) - pat).norm()
                                   / pat.norm())),
                            (f'{tag}_l{li}_value_rel_change_without_{nm}',
                             float(((hn2 @ D.Wv[li].t()) - val).norm()
                                   / val.norm()))):
                        acc[k] = acc.get(k, 0.0) + z
                # the two share statistics
                for nm, v in srcs.items():
                    acc[f'{tag}_l{li}_pre_norm_share_{nm}'] = \
                        acc.get(f'{tag}_l{li}_pre_norm_share_{nm}', 0.0) + float(
                            (v.norm(dim=-1) / pre.norm(dim=-1)).mean())
                    hv = nf(v)
                    acc[f'{tag}_l{li}_post_norm_share_{nm}'] = \
                        acc.get(f'{tag}_l{li}_post_norm_share_{nm}', 0.0) + float(
                            (hv.norm(dim=-1) / hn.norm(dim=-1)).mean())
        n += 1
    out = {k: v / n for k, v in acc.items()}
    out['G_imposed'] = G
    out['note'] = ('true_norm rows use whatever norm this model was TRAINED '
                   'with; imposed_slot_norm rows apply a G-way slot norm to the '
                   'same weights at analysis time.  A statistic that moves '
                   'between the two blocks is a property of the normalisation, '
                   'not of the computation, and may not be used as evidence.')
    return out


@torch.no_grad()
def per_head_ablation(D, n_seq=32, T=256, batch=8):
    """EVERY HEAD'S VALUE AS A RANGE, NEVER A POINT.

    Zeroing is not the harshest ablation -- resampling beat it at 13 of 14
    layer-cells in this program -- so a per-head knockout obtained by zeroing is
    a LOWER BOUND and an ordering derived from it can invert.  Both are measured
    here for every head.  Attention is linear in the pattern and the heads are
    separable in it, so head h's own write is exactly `keep`-mode h and

        zero h      = A_layer - A_h
        resample h  = A_layer - A_h + A_h(a different sequence)

    which keeps the layer's other heads untouched and puts an on-distribution
    vector where head h's write was.  Sub-additivity is reported beside it: the
    sum of the single-head costs against the cost of dropping the whole layer."""
    kl, ntok = {}, 0
    for x, y in I1.held_batches(D, n_seq, T, batch):
        if x.shape[0] < 2:
            continue
        P = D.run(x)
        lp = F.log_softmax(D.readout(P['r']).float(), -1)
        p = lp.exp()
        cand = {}
        for li in range(D.L):
            Al = P['A'][li]
            for h in range(D.H):
                Ah = D.run(x, attn={li: ('keep', h)})['A'][li]
                cand[f'l{li}_head{h}_zero'] = D.run(
                    x, attn={li: ('inject', Al - Ah)})['r']
                cand[f'l{li}_head{h}_resample'] = D.run(
                    x, attn={li: ('inject', Al - Ah + Ah.roll(1, 0))})['r']
            cand[f'l{li}_wholelayer_zero'] = D.run(
                x, attn={li: ('inject', torch.zeros_like(Al))})['r']
            cand[f'l{li}_wholelayer_resample'] = D.run(
                x, attn={li: ('inject', Al.roll(1, 0))})['r']
        for s, r in cand.items():
            q = F.log_softmax(D.readout(r).float(), -1)
            kl[s] = kl.get(s, 0.0) + float((p * (lp - q)).sum())
        ntok += y.numel()
    out = {k: v / ntok for k, v in kl.items()}
    res = {'per_head_kl_range_zero_resample': {
        f'l{li}_head{h}': [out[f'l{li}_head{h}_zero'],
                           out[f'l{li}_head{h}_resample']]
        for li in range(D.L) for h in range(D.H)}}
    for li in range(D.L):
        for how in ('zero', 'resample'):
            s = sum(out[f'l{li}_head{h}_{how}'] for h in range(D.H))
            w = out[f'l{li}_wholelayer_{how}']
            res[f'l{li}_{how}_sum_of_heads'] = s
            res[f'l{li}_{how}_whole_layer'] = w
            res[f'l{li}_{how}_subadditivity_ratio'] = s / max(w, 1e-30)
        # does the head ORDERING survive the change of ablation?
        oz = sorted(range(D.H), key=lambda h: -out[f'l{li}_head{h}_zero'])
        orr = sorted(range(D.H), key=lambda h: -out[f'l{li}_head{h}_resample'])
        res[f'l{li}_head_order_zero'] = oz
        res[f'l{li}_head_order_resample'] = orr
        res[f'l{li}_head_order_agrees'] = (oz == orr)
        res[f'l{li}_top_head_agrees'] = (oz[0] == orr[0])
    res['resample_harsher_than_zero_fraction'] = float(np.mean(
        [out[f'l{li}_head{h}_resample'] > out[f'l{li}_head{h}_zero']
         for li in range(D.L) for h in range(D.H)]))
    res['note'] = ('subadditivity_ratio = (sum of single-head costs) / (whole-'
                   'layer cost).  Above 1 the heads write overlapping '
                   'directions and the single-head numbers over-count; BELOW 1 '
                   'they are complementary and the layer is worth more than its '
                   'parts.  Measured below 1 here (0.52-0.83 in the plain '
                   'model), which is the opposite of the registered '
                   'head-compensation prediction.  The ordering rows say '
                   'whether a head ranking obtained by zeroing survives the '
                   'harsher ablation -- in the plain model it does NOT at '
                   'either layer, though the top head is the same.')
    return res


@torch.no_grad()
def induction_route_split(D, seeds=5):
    """BY WHICH ROUTE DOES THE INDUCTION SIGNAL REACH LAYER-1 ATTENTION?

    This is the measurement that overturned the program's first induction-circuit
    claim: in the plain model at width 256, deleting layer-0 attention's write
    from layer 1's Q/K/V READ moved the induction score by 0.0000, while
    deleting it from MLP-0's INPUT reproduced the whole effect -- the signal
    travels through the feed-forward block, not the attention-to-attention path.

    Any claim that a variant 'opens the attention-to-attention path' has to be
    settled with the same instrument, because an open route that carries no
    induction is a different (and weaker) result than an open route the
    algorithm actually uses.  Four arms, all with everything downstream
    recomputed:

      read_only : A0 removed from layer 1's Q/K/V read; MLP-0 untouched
      mlp_only  : A0 removed from MLP-0's input; layer 1's read untouched
      both      : removed from both
      baseline  : the folded pipeline, unmodified
    """
    if D.L < 2:
        return None

    def read_sub(P_):
        v = P_['rem'][1] + P_['M'][0]
        return D._pre(2, v, {})

    def mlp_sub(P_, x):
        return x - P_['A'][0]

    arms = {
        'baseline': {},
        'A0_out_of_layer1_read_only': {'reads': {1: read_sub}},
        'A0_out_of_mlp0_input_only': {'mlp_reads': {0: mlp_sub}},
        'A0_out_of_both': {'reads': {1: read_sub}, 'mlp_reads': {0: mlp_sub}},
        'A0_write_deleted_entirely': {'attn': {0: 'zero'}},
    }
    out = {}
    for nm, kw in arms.items():
        fwd = (lambda kw_: lambda z: D.readout(D.run(z, **kw_)['r']))(kw)
        r = [I1.induction_battery(D, seed=s, model=fwd) for s in range(seeds)]
        out[nm] = {
            'induction_score_mean': float(np.mean([q['induction_score']
                                                   for q in r])),
            'induction_score_sd': float(np.std([q['induction_score']
                                                for q in r], ddof=1)),
            'bag_score_mean': float(np.mean([q['bag_score'] for q in r]))}
    b = out['baseline']['induction_score_mean']
    out['fraction_of_induction_removed'] = {
        k: (b - v['induction_score_mean']) / b for k, v in out.items()
        if isinstance(v, dict) and 'induction_score_mean' in v and k != 'baseline'}
    out['note'] = ('read_only large => the attention-to-attention path is not '
                   'merely open, the algorithm USES it.  mlp_only large with '
                   'read_only near zero is the plain model pattern: the signal '
                   'goes through the feed-forward block.')
    return out


@torch.no_grad()
def predicate_induction_split(D, seeds=5):
    """WHICH NAMED TERM CARRIES THE PREDICATE VARIANT'S INDUCTION?

    `MATCH_prev[i, j] = 1[tok_{j-1} == tok_i]` attends from the current token to
    the position AFTER an earlier copy of it -- a complete induction head in ONE
    layer, handed to the model as a single scalar per head.  So the registered
    question is whether zeroing that one scalar per head removes the score.
    Every arm zeroes a named parameter IN PLACE and restores it, so the model is
    unchanged afterwards; the battery is the identical one every other cell
    uses.  Non-predicate variants return None rather than a fabricated row."""
    if not D.model.pred_on:
        return None
    P = {'b': D.model.pred_b, 'c': D.model.pred_c, 'prof': D.model.pred_prof}
    saved = {k: v.detach().clone() for k, v in P.items()}

    def battery(tag):
        r = [I1.induction_battery(D, seed=s, model=D.model) for s in range(seeds)]
        return {'induction_score_mean': float(np.mean([q['induction_score']
                                                       for q in r])),
                'induction_score_sd': float(np.std([q['induction_score']
                                                    for q in r], ddof=1)),
                'bag_score_mean': float(np.mean([q['bag_score'] for q in r]))}
    out = {'all_named_terms_on': battery('full')}
    try:
        for arms, tag in ((('b',), 'zero_prev_token_match_b'),
                          (('c',), 'zero_same_token_match_c'),
                          (('prof',), 'zero_positional_profile'),
                          (('b', 'c', 'prof'), 'zero_all_named_terms')):
            for k in arms:
                P[k].zero_()
            out[tag] = battery(tag)
            for k in arms:
                P[k].copy_(saved[k])
        # per-layer, for the one that matters
        for li in range(D.L):
            P['b'][li].zero_()
            out[f'zero_prev_token_match_b_layer{li}'] = battery('bl')
            P['b'].copy_(saved['b'])
        # and per HEAD in layer 0, to see whether it is one head or many
        per = {}
        for h in range(D.H):
            P['b'][0, h] = 0.0
            per[f'l0_head{h}'] = battery('h')['induction_score_mean']
            P['b'].copy_(saved['b'])
        out['zero_prev_match_one_head_at_a_time_layer0'] = per
    finally:
        for k, v in saved.items():
            P[k].copy_(v)
    base = out['all_named_terms_on']['induction_score_mean']
    out['fraction_removed'] = {
        k: (base - v['induction_score_mean']) / base
        for k, v in out.items()
        if isinstance(v, dict) and 'induction_score_mean' in v}
    out['note'] = ('the previous-token match is one scalar per head; if zeroing '
                   'it removes the score, this variant inducts by a NAMED TERM '
                   'in a single layer and not by the two-layer composition '
                   'circuit the plain model needs width 256 to build')
    return out


# ----------------------------------------------------------- stream geometry
@torch.no_grad()
def stream_geometry_v(D, n_seq=32, T=256, batch=8):
    """Exact additive attribution of the pre-tanh logit across the 2L writes and
    the token remnant.  The 'e' row is `rem[L]`, the token channel AS THE
    READOUT SEES IT -- for the shrink variant that is a projection of the
    embedding, not the embedding, and using `e` there would not sum to 1."""
    keys = ['e'] + [f'A{l}' for l in range(D.L)] + [f'M{l}' for l in range(D.L)]
    tot = {f'{k}_{q}': 0.0 for k in keys for q in ('norm', 'share')}
    n = 0
    for x, y in I1.held_batches(D, n_seq, T, batch):
        P = D.run(x)
        parts = {'e': P['rem'][D.L]}
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
    out['share_sum'] = sum(out[f'{k}_share'] for k in keys)
    return out


# --------------------------------------------------------------- rung 2
@torch.no_grad()
def rung2_v(D):
    """Layer-0 factor and branch-table spectra (verbatim `tf_interp.rung2`) plus
    the per-layer MLP tensor and read-matrix spectra.

    ONE CORRECTION over `tf_interp2.rung2_layers`: the mode-0 unfolding is
    `T.reshape(T.shape[0], -1)`, not `T.reshape(Ws, -1)`.  For a small decoder
    the tensor is (s, Ws, Ws) with s != Ws and the hard-coded Ws silently
    reinterprets the memory as a different matrix.  Identical for vanilla."""
    out = {'layer0': I1.rung2(D)}
    for li in range(D.L):
        T = D.Tl[li]
        d = out.get(f'layer{li}', {})       # layer 0 already has the factor and
        #                                     branch-table spectra; MERGE, do not
        #                                     overwrite them
        O, Ws = T.shape[0], D.Ws
        unf0 = torch.linalg.svdvals(T.reshape(O, -1).double()).cpu().numpy()
        ev = torch.linalg.eigvalsh(T.double()).cpu().numpy()
        sl = [tf_fold.eff_rank(np.sort(np.abs(a))[::-1]) for a in ev]
        g = torch.Generator(device='cpu').manual_seed(5)
        Dn = torch.randn(O, D.cfg.hidden, generator=g).to(D.dev)
        Lr = torch.randn(D.cfg.hidden, Ws, generator=g).to(D.dev)
        Rr = torch.randn(D.cfg.hidden, Ws, generator=g).to(D.dev)
        Mn = torch.einsum('of,fi,fj->oij', Dn, Lr, Rr)
        Tn = 0.5 * (Mn + Mn.transpose(1, 2))
        svn = torch.linalg.svdvals(Tn.reshape(O, -1).double()).cpu().numpy()
        d['mlp'] = {'shape': list(T.shape),
                    'mode0_unfolding': {**tf_fold.eff_rank(unf0),
                                        'rank_bound': min(O, Ws * Ws)},
                    'random_factored_null_mode0': tf_fold.eff_rank(svn),
                    'slice_eigen_mean_entropy_rank':
                        float(np.mean([s['entropy_rank'] for s in sl])),
                    'mean_negative_eig_share': float((ev < 0).sum() / ev.size)}
        d['mlp']['content_over_null_ALL_ROWS_DO_NOT_QUOTE'] = (
            d['mlp']['mode0_unfolding']['entropy_rank']
            / max(d['mlp']['random_factored_null_mode0']['entropy_rank'], 1e-9))
        # ---- LIVE ROWS ONLY, and this is the number that may be quoted ----
        # ARITHMETIC DRESSED AS A FINDING, caught before it was reported: for a
        # MASKED decoder (variants slots and shrink) the folded tensor is the
        # full (Ws, Ws, Ws), but write_out discards every output row outside the
        # module's own slot, so 96 of 128 rows never receive a gradient and stay
        # at their small init.  Measured: row norms are 100.5 inside slot 1 and
        # 4.7 everywhere else.  A spectrum over all 128 rows therefore reports
        # `entropy rank 51 against a null of 123` -- which is just 32/128, the
        # masking, and nothing about content.  Restricting to the live rows and
        # matching the null's shape to them makes the six variants comparable:
        # small decoders are physically slot-sized so nothing changes for them.
        k = 2 * li + 1
        if D.cfg.n_slots > 1 and not D.cfg.small_dec:
            live = list(range(D.s * k, D.s * (k + 1)))
        else:
            live = list(range(O))
        Tl_ = T[live]
        OL = len(live)
        unfl = torch.linalg.svdvals(Tl_.reshape(OL, -1).double()).cpu().numpy()
        gl = torch.Generator(device='cpu').manual_seed(5)
        Dnl = torch.randn(OL, D.cfg.hidden, generator=gl).to(D.dev)
        Lrl = torch.randn(D.cfg.hidden, Ws, generator=gl).to(D.dev)
        Rrl = torch.randn(D.cfg.hidden, Ws, generator=gl).to(D.dev)
        Mnl = torch.einsum('of,fi,fj->oij', Dnl, Lrl, Rrl)
        Tnl = 0.5 * (Mnl + Mnl.transpose(1, 2))
        svnl = torch.linalg.svdvals(Tnl.reshape(OL, -1).double()).cpu().numpy()
        d['mlp']['live_output_rows'] = OL
        d['mlp']['live_row_norm_ratio_inside_over_outside'] = float(
            T[live].reshape(OL, -1).norm(dim=1).mean()
            / max(float(T.reshape(O, -1).norm(dim=1).mean()), 1e-30))
        d['mlp']['mode0_unfolding_live'] = {**tf_fold.eff_rank(unfl),
                                            'rank_bound': min(OL, Ws * Ws)}
        d['mlp']['random_factored_null_mode0_live'] = tf_fold.eff_rank(svnl)
        d['mlp']['content_over_null'] = (
            d['mlp']['mode0_unfolding_live']['entropy_rank']
            / max(d['mlp']['random_factored_null_mode0_live']['entropy_rank'],
                  1e-9))
        rd = {}
        for nm, W in (('c_q', D.Wq[li]), ('c_k', D.Wk[li]), ('c_q2', D.Wq2[li]),
                      ('c_k2', D.Wk2[li]), ('c_v', D.Wv[li]),
                      ('c_proj', D.Wproj[li])):
            sv = torch.linalg.svdvals(W.double()).cpu().numpy()
            rd[nm] = {**tf_fold.eff_rank(sv), 'shape': list(W.shape)}
        d['read_matrices'] = rd
        out[f'layer{li}'] = d
    return out


# ------------------------------------------------- what the mechanism did
@torch.no_grad()
def mechanism_report(D, n_seq=32, T=256):
    """Per-variant: what did the interpretability mechanism actually do?

    slot occupancy of the reads (the group lasso's own objective), the named
    predicate coefficients, the codebook's realised usage, the shrink remnant
    schedule.  A mechanism that is present but unused is a result about the
    architecture, so 'zero' is reported rather than omitted."""
    cfg = D.cfg
    out = {'variant': cfg.variant, 'n_slots': cfg.n_slots, 'slot': cfg.slot,
           'stream_width': D.Ws, 'compute_width': D.Dc,
           'group_coeff': cfg.group_coeff}
    # ---- slot occupancy of every read matrix (the lasso's own objective) ----
    occ = {}
    G, s = cfg.n_slots, D.s
    for li in range(D.L):
        blk = D.model.h[li]
        for nm in M.READ_NAMES:
            W = getattr(blk, nm).weight.detach().float()
            gnorm = W.pow(2).view(W.shape[0], G, s).sum(dim=(0, 2)).sqrt()
            tot = float(gnorm.sum()) + 1e-30
            occ[f'l{li}_{nm}'] = {
                'group_norms': [float(v) for v in gnorm],
                'share': [float(v / tot) for v in gnorm],
                'n_groups_over_1pct': int((gnorm / tot > 0.01).sum())}
    out['read_slot_occupancy'] = occ
    frac = []
    for k, v in occ.items():
        frac.append(v['n_groups_over_1pct'])
    out['mean_live_slots_per_read'] = float(np.mean(frac))
    out['slot_semantics'] = ('slot 2l holds block l attention, slot 2l+1 holds '
                             'block l MLP; a read whose mass sits on slot 0 is '
                             'reading block-0 attention')
    # ---- predicate ----
    if D.model.pred_on:
        pr = {}
        for li in range(D.L):
            prof = D.model.pred_prof[li].detach().float().cpu()
            pr[f'layer{li}'] = {
                'prev_token_match_b': [float(v) for v in
                                       D.model.pred_b[li].detach().cpu()],
                'same_token_match_c': [float(v) for v in
                                       D.model.pred_c[li].detach().cpu()],
                'positional_profile_absmax_per_head':
                    [float(v) for v in prof.abs().max(-1).values],
                'positional_profile_argmax_distance_per_head':
                    [int(v) for v in prof.abs().argmax(-1)],
                'profile_first8_head0': [float(v) for v in prof[0, :8]]}
        out['predicate'] = pr
    # ---- codebook usage on held text ----
    if D.model.qz_on:
        arr = tf_corpus.load_split(D.V, 'held', n_seq, tok=D.cfg.tok)
        x = torch.from_numpy(arr[:, :T]).to(D.dev)
        collect = {}
        for a in range(0, x.shape[0], 8):
            D.model(x[a:a + 8], collect=collect)
        cbu = {}
        for k, chunks in collect.get('codes', {}).items():
            ids = torch.cat(chunks).reshape(-1)
            cnt = torch.bincount(ids.long(), minlength=D.cfg.cb_n).float()
            p = cnt / cnt.sum()
            nz = p[p > 0]
            srt = torch.sort(p, descending=True).values.cumsum(0)
            cbu[f'module{k}'] = {
                'k_steps': D.model.qz_ksteps[k],
                'atoms_used': int((cnt > 0).sum()),
                'atoms_total': D.cfg.cb_n,
                'usage_entropy_nats': float(-(nz * nz.log()).sum()),
                'perplexity': float(torch.exp(-(nz * nz.log()).sum())),
                'atoms_for_90pct': int((srt < 0.9).sum()) + 1}
            cbu[f'module{k}']['ema_usage_counts_nonzero'] = int(
                (D.model.qz_usage[k] > 0).sum())
        out['codebook'] = cbu
        # quantisation error actually incurred, as a relative residual
        errs = {}
        for a in range(0, min(8, x.shape[0]), 8):
            b = x[a:a + 8]
            e = F.rms_norm(D.model.wte(b), (D.Ws,))
            P = D.run(b)
            for li in range(D.L):
                pre = P['pre_mlp'][li]
                exact = D.model.slot_norm(pre)
                q = D._pre(2 * li + 1, pre)
                errs[f'mlp_input_l{li}_rel_quant_error'] = float(
                    (q - exact).norm() / exact.norm())
        out['codebook_relative_quantisation_error'] = errs
    # ---- shrink ----
    if D.model.W_rem is not None:
        out['shrink'] = {'remnant_dims_per_consumer': list(D.model.rem_dims),
                         'note': 'consumer 0 is block 0, consumer L is the '
                                 'readout; the token channel is projected down '
                                 'at each step with a floor'}
        sh = {}
        for li, lin in D.model.W_rem.items():
            W = lin.weight.detach().float()
            sv = torch.linalg.svdvals(W.double()).cpu().numpy()
            sh[f'consumer{li}'] = {**tf_fold.eff_rank(sv), 'shape': list(W.shape)}
        out['shrink']['remnant_projection_spectra'] = sh
    return out


# ------------------------------------------------------------- disjointness
@torch.no_grad()
def fit_score_disjointness_v(D, n_seq=64, T=256):
    h = ladder_v(D, n_seq, T, split='held', extra=False)
    e = ladder_v(D, n_seq, T, split='est', extra=False)
    rows = {k: {'held': h[k]['kl_from_model'], 'est': e[k]['kl_from_model'],
                'held_minus_est': h[k]['kl_from_model'] - e[k]['kl_from_model']}
            for k in h if not k.startswith('_')}
    return {'per_stage': rows,
            'max_abs_held_minus_est': max(abs(r['held_minus_est'])
                                          for r in rows.values())}


# ---------------------------------------------------------------- driver
def analyse(stem, quick=False, skip_baselines=True):
    t0 = time.time()
    jp = f'{HERE}/{stem}_interp3.json'
    rep = json.load(open(jp)) if os.path.exists(jp) else {}
    rep['stem'] = stem
    rep['registered_predictions_variant_slice'] = json.load(
        open(f'{HERE}/tf_variant_predictions.json'))
    json.dump(rep, open(jp, 'w'), indent=2)
    D = VariantFold(stem)
    rep['variant'] = D.cfg.variant
    rep['depth'], rep['width'] = D.L, D.cfg.width
    rep['config'] = {k: v for k, v in vars(D.cfg).items()}
    rep['params'] = {'total': D.model.n_params(),
                     'body': D.model.body_params(),
                     'embedding': D.model.wte.weight.numel(),
                     'stream_width': D.Ws,
                     'codebook_buffer_floats':
                         int(D.model.qz_codebook.numel()) if D.model.qz_on else 0}
    rep['params']['effective_total'] = (rep['params']['total']
                                        + rep['params']['codebook_buffer_floats'])
    ck = torch.load(f'{HERE}/{stem}.pt', map_location='cpu', weights_only=False)
    rep['train'] = {k: ck['log'].get(k) for k in
                    ('final_held_ce', 'lr_muon', 'steps', 'batch', 'spikes',
                     'diverged', 'wall_seconds')}
    rep['train']['bits_per_byte'] = tf_corpus.bits_per_byte(
        ck['log']['final_held_ce'], D.V, tok=D.cfg.tok)
    # ---- HARD GATES first ----
    xg = torch.from_numpy(tf_corpus.load_split(D.V, 'held', 4,
                                               tok=D.cfg.tok))[:, :128].to(D.dev)
    rep['fold_gate'] = M.check_fold_identities(D.model, xg, verbose=False)
    rep['decomposition_control'] = D.self_check(xg)
    print('  fold gate:', rep['fold_gate']['pass'],
          ' pipeline:', rep['decomposition_control'], flush=True)
    json.dump(rep, open(jp, 'w'), indent=2)
    assert rep['decomposition_control']['pass'], rep['decomposition_control']
    n_seq, T = (16, 128) if quick else (96, 256)
    rep['rung2'] = rung2_v(D)
    print(f'  rung2 {time.time()-t0:.0f}s', flush=True)
    rep['mechanism'] = mechanism_report(D)
    rep['stream_geometry'] = stream_geometry_v(D, min(n_seq, 32), T)
    rep['rung5_ladder'] = ladder_v(D, n_seq, T)
    print(f'  ladder {time.time()-t0:.0f}s', flush=True)
    rep['ladder_order'] = I2.ladder_order(D, min(n_seq, 64), T)
    ind = [I1.induction_battery(D, seed=s, model=D.model)
           for s in range(3 if quick else 5)]
    rep['rung3_induction'] = {
        'per_seed': ind,
        'induction_score_mean': float(np.mean([i['induction_score'] for i in ind])),
        'induction_score_sd': float(np.std([i['induction_score'] for i in ind],
                                           ddof=1)),
        'bag_score_mean': float(np.mean([i['bag_score'] for i in ind])),
        'bag_score_sd': float(np.std([i['bag_score'] for i in ind], ddof=1))}
    rep['induction_power'] = I2.induction_power(D)
    rep['induction_by_head'] = I2.induction_by_head(D)
    rs = induction_route_split(D)
    if rs is not None:
        rep['induction_route_split'] = rs
    ps = predicate_induction_split(D)
    if ps is not None:
        rep['predicate_induction_split'] = ps
    rep['natural_induction'] = I2.natural_induction(D, n_seq=1024)
    print(f'  induction {time.time()-t0:.0f}s', flush=True)
    rep['resample_ablation'] = I2.resample_ablation(D)
    rep['per_head_ablation'] = per_head_ablation(D, min(n_seq, 32), T)
    rep['composition_budget'] = composition_budget_v(D, min(n_seq, 32), T)
    rep['read_ablation_causal'] = read_ablation_causal(D, min(n_seq, 32), T)
    rep['norm_confound_control'] = norm_confound_control(D, 4, min(n_seq, 32), T)
    rep['fit_score_disjointness'] = fit_score_disjointness_v(D, min(n_seq, 64), T)
    print(f'  ablations {time.time()-t0:.0f}s', flush=True)
    rep['rung4'] = I1.rung4(D)
    rep['composed_vs_causal'] = I2.composed_vs_causal(D, min(n_seq, 32), T)
    rep['mlp_composed_causal'] = I2.mlp_composed_causal(D)
    rep['causal_copy_test'] = I2.causal_copy_test(D)
    if not skip_baselines:
        rep['rung3_baselines'] = I1.data_baselines(D, n_seq, T)
    rep['bits_per_byte_ladder'] = tf_corpus.bits_per_byte(
        rep['rung5_ladder']['_model_ce'], D.V, tok=D.cfg.tok)
    rep['seconds'] = round(time.time() - t0, 1)
    json.dump(rep, open(jp, 'w'), indent=2)
    print(f'== interp3 written to {jp} ({rep["seconds"]}s)', flush=True)
    return rep


# --------------------------------------------- POSITIVE CONTROL vs tf_interp2
def control(stem='tf_vanilla_d2_w128_b8192_s0', n_seq=48, T=256):
    """`VariantFold` must reproduce `DeepFold` on a vanilla checkpoint, stage by
    stage.  Without this the whole comparison is uninterpretable: a variant
    difference could be a difference of analysis code."""
    out = {'stem': stem}
    A = I2.DeepFold(stem)
    B = VariantFold(stem)
    x = torch.from_numpy(tf_corpus.load_split(A.V, 'held', 4,
                                              tok=A.cfg.tok))[:, :128].to(A.dev)
    with M.exact_math():
        ra, rb = A.run(x)['r'], B.run(x)['r']
        out['residual_rel_maxdiff'] = float(
            (ra - rb).abs().max() / ra.abs().max())
        la, lb = A.readout(ra), B.readout(rb)
        out['logit_rel_maxdiff'] = float((la - lb).abs().max() / la.abs().max())
        ba, bb = A.bigram_table(), B.bigram_table()
        out['bigram_table_rel_maxdiff'] = float(
            (ba - bb).abs().max() / ba.abs().max())
        out['OV_rel_maxdiff'] = float(
            (A.OV - B.OV).abs().max() / A.OV.abs().max())
        # the counterfactual reads: ladder2's `P['e']` form vs ladder_v's
        # slot_norm(rem) form.  Equal expressions only when n_slots == 1.
        pa = A.run(x, reads={1: lambda P_: _rms(P_['e'] + P_['M'][0], A.Ws)})['r']
        pb = B.run(x, reads={1: lambda P_: B._pre(2, P_['rem'][1] + P_['M'][0],
                                                  {})})['r']
        out['counterfactual_read_rel_maxdiff'] = float(
            (pa - pb).abs().max() / pa.abs().max())
    la_ = I2.ladder2(A, n_seq, T, extra=False)
    lb_ = ladder_v(B, n_seq, T, extra=False)
    rows = {}
    for k in la_:
        if k.startswith('_'):
            continue
        rows[k] = {'interp2': la_[k]['kl_from_model'],
                   'interp3': lb_[k]['kl_from_model'],
                   'diff': lb_[k]['kl_from_model'] - la_[k]['kl_from_model']}
    out['ladder_stage_agreement'] = rows
    out['ladder_max_abs_diff'] = max(abs(r['diff']) for r in rows.values())
    ca = I2.composition_budget(A, 32, T)
    cb = composition_budget_v(B, 32, T)
    out['composition_budget_agreement'] = {
        k: {'interp2': ca[k], 'interp3': cb[k], 'diff': cb[k] - ca[k]}
        for k in ca if k != 'note'}
    out['composition_budget_max_abs_diff'] = max(
        abs(v['diff']) for v in out['composition_budget_agreement'].values())
    ga = I2.stream_geometry2(A, 32, T)
    gb = stream_geometry_v(B, 32, T)
    # RELATIVE, because the `_norm` rows are O(1e3) (the stream is dominated by
    # the MLP writes) and an absolute 1e-4 on a norm of 7000 is 1e-8 relative --
    # a criterion no fp32 pipeline can meet and none should have to.  The
    # `_share` rows are O(1) so relative and absolute coincide there.
    out['stream_geometry_agreement'] = {
        k: {'interp2': ga[k], 'interp3': gb[k],
            'rel': abs(gb[k] - ga[k]) / max(abs(ga[k]), 1e-12)}
        for k in gb if k in ga and isinstance(gb[k], float)}
    out['stream_geometry_max_rel_diff'] = max(
        v['rel'] for v in out['stream_geometry_agreement'].values())
    out['pass'] = bool(out['logit_rel_maxdiff'] < 1e-5
                       and out['bigram_table_rel_maxdiff'] < 1e-5
                       and out['counterfactual_read_rel_maxdiff'] < 1e-5
                       and out['ladder_max_abs_diff'] < 1e-4
                       and out['composition_budget_max_abs_diff'] < 1e-4
                       and out['stream_geometry_max_rel_diff'] < 1e-4)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stem')
    ap.add_argument('--control', action='store_true')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--baselines', action='store_true')
    a = ap.parse_args()
    if a.control:
        r = control()
        json.dump(r, open(f'{HERE}/tf_interp3_control.json', 'w'), indent=2)
        print(json.dumps({k: v for k, v in r.items()
                          if not isinstance(v, dict)}, indent=2))
        raise SystemExit(0 if r['pass'] else 1)
    analyse(a.stem, a.quick, skip_baselines=not a.baselines)
