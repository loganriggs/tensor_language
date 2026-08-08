"""Rungs 2-5 for DEPTH-1 cells: the full interpretation, built on the fold.

WHY DEPTH 1 IS SPECIAL.  With n_slots == 1 the stream norm is a plain RMSNorm
and it is idempotent, so the layer-0 module input is EXACTLY the folded token
table Ehn[t].  The whole model is then

    e_i      = Ehn[t_i]                                   (current token only)
    p_h[i,j] = s1_h(t_i, t_j, i-j) * s2_h(t_i, t_j, i-j)   (token-pair table)
    A_i      = sum_h sum_{j<=i} p_h[i,j] * OV_h[t_j]       OV_h = Vv_h W_proj_h
    M_i      = T(rms(e_i + A_i), rms(e_i + A_i)) + b
    logits_i = 30 tanh( rms(e_i + A_i + M_i) W_U^T / 30 )

Two facts make the accounting clean:
  * p_h[i,i] has delta = 0, where the rotary is the identity, so the SELF term
    A0(t) = sum_h p_h(t,t,0) OV_h[t] is a pure function of the current token.
    Everything current-token-only therefore collapses into ONE V x Ws table,
    and the model's exact output on a length-1 context is a V x V table
    computed from weights alone with no data.  That is the model's OWN bigram.
  * A_i is exactly a sum over (query token, key token, distance) triples.  A
    depth-1 model is a SKIP-GRAM machine; it has no route by which a key can
    depend on anything but its own token, which is the structural reason it
    cannot do induction.

SIGN IS A GAUGE FREEDOM (standing rule).  Nothing here is positivity
constrained, so the sign of s1, of s2, of a value factor or of an unembedding
row can be flipped and compensated elsewhere with no functional change.  Only
COMPLETE PATHS to an observable have invariant signs.  Every rung-4 quantity in
this file is therefore composed all the way to logits before it is named:
the object we decode is  C_h(t, u, delta)[v] = p_h(t,u,delta) * (OV_h[u] W_U^T)[v],
never p or OV alone.  A negative p is NOT suppression: the contribution of
attending j is p times the unembedded value of x_j, so negative-times-negative
is a positive push and the effect of a negative pattern entry lands on the
positions that were NOT attended.

Usage
    python tf_interp.py --stem tf_vanilla_d1_w32_b8192_s0 [--quick]
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
import tf_model as M

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

REGISTERED = {
    'written_before_any_rung_ran': True,
    'rung2_low_rank': 'the four V x head_dim query/key factors are FULL rank '
                      '(16) -- rank <= 16 is arithmetic, not a finding -- but '
                      'their SPECTRA are steep, so the entropy-effective rank '
                      'of the realized delta=0 score table is well below 16 '
                      '(predict < 8 for a majority of head-branches).  The '
                      'MLP tensor mode-0 unfolding is predicted to have '
                      'entropy rank well below its architectural bound '
                      'min(Ws, hidden).',
    'rung3_unigram': 'the model beats the unigram floor by > 1.5 nats at every '
                     'width',
    'rung3_bigram': "the model's OWN bigram table (its exact output on a "
                    'length-1 context, weights only) carries most of its held '
                    'CE: freezing the context away costs less than 0.5 nats at '
                    'width 32 and more at larger widths',
    'rung3_skipgram': 'attention from delta >= 2 is worth less than attention '
                      'from delta = 1 (the adjacent-token term dominates)',
    'rung3_positional': 'a positional-only predictor is worse than unigram+eps; '
                        'replacing the token-dependent pattern by its '
                        'distance-only average destroys most of the attention '
                        'gain, i.e. the pattern is token-driven not positional',
    'rung3_induction': 'PREDICTION, STRUCTURAL: a depth-1 model CANNOT do '
                       'induction.  On repeated random text the induction '
                       'score (CE with an order-preserving repeat minus CE '
                       'with a SHUFFLED repeat) is ~0 within noise, while the '
                       'BAG score (shuffled repeat vs no repeat) is clearly '
                       'positive, because an order-free bag effect is '
                       'reachable at depth 1.  A depth-2 '
                       'cell run through the identical battery must show a '
                       'nonzero induction score, otherwise the METRIC is '
                       'broken rather than the model being shallow.',
    'rung4_tokens': 'the composed copy score is dominated by a small number of '
                    'token pairs (effective pair count << V^2) and the top '
                    'pairs are enriched for identical or orthographically '
                    'related tokens relative to a frequency-matched null',
    'rung5_kl': 'the weights-only model-bigram stage gets KL to the true model '
                'below 0.5 nats; adding the folded past attention at least '
                'halves what remains; the exact MLP closes it to ~0 (the fold '
                'is exact, so the last stage is a control, not a result)',
}


# ---------------------------------------------------------------- utilities
def _rms(x, d):
    return F.rms_norm(x, (d,))


class Depth1Fold:
    """Every folded object a depth-1 cell needs, on device, in fp32.

    ASSERTED, not assumed: the class refuses to build for anything but a
    depth-1 vanilla cell with n_slots == 1, because the identity
    'module input == Ehn[t]' is what the whole decomposition rests on, and it
    is checked numerically in `self_check`."""

    def __init__(self, stem, device=DEV):
        model, cfg, ck = tf_fold.load_checkpoint(stem, device)
        assert cfg.depth == 1, 'Depth1Fold is depth-1 only'
        assert cfg.n_slots == 1 and not cfg.predicate and not cfg.codebook \
            and not cfg.small_dec and not cfg.shrink_mode, \
            'Depth1Fold assumes the vanilla variant'
        self.stem, self.model, self.cfg, self.dev = stem, model, cfg, device
        self.V, self.H, self.hd = cfg.vocab, cfg.n_heads, cfg.head_dim
        self.Ws, self.Dc = model.Ws, model.Dc
        blk = model.h[0]
        f = model.fold_layer0_qk(materialize=False, device=device)
        self.Q1, self.K1, self.Q2, self.K2 = f['Q1'], f['K1'], f['Q2'], f['K2']
        self.Vv = f['Vv']                                   # (H, V, hd)
        self.Ehn = model.token_input_table().to(device)      # (V, Ws)
        self.WU = model.wte.weight.detach().float()          # (V, Ws)
        Wp = blk.c_proj.weight.detach().float()              # (Ws, Dc)
        # OV_h[u] = the STREAM vector written when head h attends to token u
        self.OV = torch.stack([self.Vv[h] @ Wp[:, h * self.hd:(h + 1) * self.hd].t()
                               for h in range(self.H)])      # (H, V, Ws)
        # OVU_h[u, v] = the LOGIT pushed on v -- the composed, sign-invariant
        # object.  (H, V, V); 268 MB per head at V=8192 in fp32, so it is built
        # lazily / in chunks by the callers that need it.
        self.T = model.fold_mlp(0, device=device).to(device)  # (Ws, Ws, Ws)
        self.bias = blk.Down_bias.detach().float()
        self.cos, self.sin = model.cos.to(device), model.sin.to(device)
        # ---- pure-token tables ----
        p0 = self.pair_scores_diag()                          # (H, V)
        self.A0 = torch.einsum('hv,hvd->vd', p0, self.OV)     # (V, Ws)
        self.p_self = p0
        self.selfterm = self.Ehn + self.A0                    # (V, Ws)
        xn0 = _rms(self.selfterm, self.Ws)
        self.M0 = torch.einsum('oij,vi,vj->vo', self.T, xn0, xn0) + self.bias
        self.r0 = self.selfterm + self.M0                     # (V, Ws)

    # ---------------- exact pieces ----------------
    def pair_scores_diag(self):
        """p_h(t, t, delta=0) for every token -- the SELF-attention weight.
        delta = 0 makes the rotary the identity, so this is a pure token
        function and it is what makes A0 a table."""
        s1 = (self.Q1 * self.K1).sum(-1) / self.hd            # (H, V)
        s2 = (self.Q2 * self.K2).sum(-1) / self.hd
        return s1 * s2

    def _pattern(self, idx):
        """(B, H, T, T) causal pattern, from the FOLDED factors only."""
        B, Tq = idx.shape
        cos = self.cos[None, :Tq, None, :]
        sin = self.sin[None, :Tq, None, :]

        def g(fac):
            return M.apply_rot(fac.permute(1, 0, 2)[idx], cos, sin)
        s1 = torch.einsum('bqhd,bkhd->bhqk', g(self.Q1), g(self.K1)) / self.hd
        s2 = torch.einsum('bqhd,bkhd->bhqk', g(self.Q2), g(self.K2)) / self.hd
        mask = self.model.mask[:Tq, :Tq]
        return (s1 * s2).masked_fill(~mask, 0.0)

    def freq_energy(self):
        """Energy per ROTARY FREQUENCY pair.  apply_rot pairs coordinate f with
        f + hd/2, and R(delta) rotates only within such a pair, so the rotary
        frequencies are the only delta-equivariant way to cut the head's
        subspace down -- an arbitrary hd-subspace truncation would not commute
        with the rotation and would not be a statement about the head."""
        d = self.hd // 2
        e = torch.zeros(self.H, d, device=self.dev)
        for qn, kn in (('Q1', 'K1'), ('Q2', 'K2')):
            Q, K = getattr(self, qn), getattr(self, kn)
            eq = (Q[:, :, :d] ** 2 + Q[:, :, d:] ** 2).mean(1)
            ek = (K[:, :, :d] ** 2 + K[:, :, d:] ** 2).mean(1)
            e = e + (eq * ek).sqrt()
        return e

    def freq_mask(self, k):
        """(H, hd) 0/1 mask keeping the top-k rotary frequency pairs per head."""
        d = self.hd // 2
        e = self.freq_energy()
        keep = torch.zeros(self.H, d, device=self.dev)
        keep.scatter_(1, e.topk(k, dim=1).indices, 1.0)
        return torch.cat([keep, keep], 1)

    def pattern(self, idx, freq_keep=None):
        if freq_keep is None:
            return self._pattern(idx)
        m = freq_keep[:, None, :]
        old = (self.Q1, self.K1, self.Q2, self.K2)
        self.Q1, self.K1, self.Q2, self.K2 = (f * m for f in old)
        try:
            return self._pattern(idx)
        finally:
            self.Q1, self.K1, self.Q2, self.K2 = old

    def mlp_topk(self, k):
        """The MLP restricted to its k most-used hidden units.  The bilinear
        MLP is EXACTLY a CP decomposition with `hidden` rank-1 symmetric terms,
        M(x) = sum_f Down[:,f] (l_f.x)(r_f.x), so dropping units is a genuine
        term-count truncation, not a projection of the tensor."""
        blk = self.model.h[0]
        Dn = blk.Down.weight.detach().float()
        L = blk.Left.weight.detach().float()
        R = blk.Right.weight.detach().float()
        xn = _rms(self.selfterm, self.Ws)
        use = (Dn.norm(dim=0) * ((xn @ L.t()) * (xn @ R.t())).abs().mean(0))
        keep = use.topk(k).indices
        return Dn[:, keep], L[keep], R[keep]

    def mlp_apply(self, r, parts=None):
        xn = _rms(r, self.Ws)
        if parts is None:
            return torch.einsum('oij,bti,btj->bto', self.T, xn, xn) + self.bias
        Dn, L, R = parts
        return ((xn @ L.t()) * (xn @ R.t())) @ Dn.t() + self.bias

    def terms(self, idx, pat=None):
        """The exact additive residual pieces for a batch of contexts."""
        B, Tq = idx.shape
        e = self.Ehn[idx]
        pat = self.pattern(idx) if pat is None else pat
        ov = self.OV.permute(1, 0, 2)[idx]                    # (B,T,H,Ws)
        A = torch.einsum('bhqk,bkho->bqo', pat, ov)
        A0 = self.A0[idx]
        xn = _rms(e + A, self.Ws)
        Mt = torch.einsum('oij,bti,btj->bto', self.T, xn, xn) + self.bias
        return {'e': e, 'A': A, 'A0': A0, 'Apast': A - A0, 'M': Mt,
                'M0': self.M0[idx], 'pat': pat}

    def readout(self, r):
        return 30 * torch.tanh((_rms(r, self.Ws) @ self.WU.t()) / 30)

    @torch.no_grad()
    def self_check(self, idx):
        """POSITIVE CONTROL for this whole file: the additive decomposition
        must reproduce the model's logits, and the pure-token r0 table must
        reproduce the model's output on a length-1 context."""
        with M.exact_math():
            t = self.terms(idx)
            rec = self.readout(t['e'] + t['A'] + t['M'])
            ref = self.model(idx)
            sc = float(ref.abs().max())
            out = {'decomposition_rel_logit_diff':
                   float((rec - ref).abs().max()) / sc}
            one = idx[:, :1]
            ref1 = self.model(one)[:, 0]
            rec1 = self.readout(self.r0[one[:, 0]])
            out['length1_table_rel_logit_diff'] = float(
                (rec1 - ref1).abs().max()) / max(float(ref1.abs().max()), 1e-30)
            # A0 is claimed to be position-independent: check it against the
            # diagonal of the pattern at several positions
            pat = t['pat']
            dg = torch.stack([pat[:, h].diagonal(dim1=-2, dim2=-1)
                              for h in range(self.H)], 1)      # (B,H,T)
            ref_d = self.p_self[:, idx].permute(1, 0, 2)
            out['self_pattern_position_independent_absmax'] = float(
                (dg - ref_d).abs().max())
        out['pass'] = bool(out['decomposition_rel_logit_diff'] < 1e-5
                           and out['length1_table_rel_logit_diff'] < 1e-5
                           and out['self_pattern_position_independent_absmax']
                           < 1e-5)
        return out


# ---------------------------------------------------------------- rung 2
@torch.no_grad()
def rung2(D):
    """Factor matrices, their spectra and effective ranks, and the MLP tensor.

    THE RANK BOUND IS ARITHMETIC.  Every V x head_dim factor has rank <= 16 and
    every branch table rank <= 16 BY CONSTRUCTION; saying so is not a finding.
    What is reported here is how far BELOW the bound the realized objects sit,
    measured by the spectral-entropy effective rank exp(H(sigma/sum sigma)) and
    by the participation ratio, both of which are bound-free."""
    out = {'note': 'rank <= head_dim and rank <= hidden are architectural '
                   'guarantees, NOT findings; only the gap below the bound is '
                   'reported as a result',
           'head_dim': D.hd, 'n_heads': D.H, 'Ws': D.Ws,
           'hidden': D.cfg.hidden}
    fac = {}
    for nm in ('Q1', 'K1', 'Q2', 'K2', 'Vv'):
        F_ = getattr(D, nm)
        per = []
        for h in range(D.H):
            sv = torch.linalg.svdvals(F_[h].double()).cpu().numpy()
            per.append({'sv': [float(x) for x in sv],
                        **tf_fold.eff_rank(sv)})
        fac[nm] = {'shape': list(F_.shape), 'rank_bound': D.hd, 'per_head': per,
                   'mean_entropy_rank': float(np.mean(
                       [p['entropy_rank'] for p in per]))}
    out['factors'] = fac
    # realized delta=0 branch tables (spectra without forming V x V)
    tab = {}
    for d in (0, 1, 8):
        R = M.rot_matrix(d, D.hd, D.dev)
        for br, (qn, kn) in (('s1', ('Q1', 'K1')), ('s2', ('Q2', 'K2'))):
            per = []
            for h in range(D.H):
                sv = tf_fold.table_spectrum(getattr(D, qn)[h],
                                            getattr(D, kn)[h], R,
                                            D.hd).cpu().numpy()
                per.append({**tf_fold.eff_rank(sv),
                            'sv': [float(x) for x in sv]})
            tab[f'{br}_d{d}'] = {'rank_bound': D.hd, 'per_head': per,
                                 'mean_entropy_rank': float(np.mean(
                                     [p['entropy_rank'] for p in per]))}
    out['branch_tables'] = tab
    # ---- NULLS.  'entropy rank 3 of 16' means nothing until we know what a
    # random factor pair of the same shape gives; a random V x 16 Gaussian has
    # an almost flat 16-value spectrum, but the PRODUCT of two of them does
    # not, so the table null is the one that matters.
    g = torch.Generator(device='cpu').manual_seed(3)
    nfac, ntab = [], []
    R0 = M.rot_matrix(0, D.hd, D.dev)
    for _ in range(8):
        a = torch.randn(D.V, D.hd, generator=g).to(D.dev)
        b = torch.randn(D.V, D.hd, generator=g).to(D.dev)
        nfac.append(tf_fold.eff_rank(
            torch.linalg.svdvals(a.double()).cpu().numpy())['entropy_rank'])
        ntab.append(tf_fold.eff_rank(
            tf_fold.table_spectrum(a, b, R0, D.hd).cpu().numpy()
        )['entropy_rank'])
    out['nulls'] = {
        'random_factor_entropy_rank_mean': float(np.mean(nfac)),
        'random_factor_entropy_rank_sd': float(np.std(nfac, ddof=1)),
        'random_branch_table_entropy_rank_mean': float(np.mean(ntab)),
        'random_branch_table_entropy_rank_sd': float(np.std(ntab, ddof=1)),
        'note': 'iid Gaussian V x head_dim factors, same shapes; the '
                'architectural rank bound is identical, so any gap between '
                'the trained numbers and these is structure and not the bound'}
    # ---- MLP symmetric tensor ----
    T = D.T
    Ws = D.Ws
    unf0 = torch.linalg.svdvals(T.reshape(Ws, -1).double()).cpu().numpy()
    unf1 = torch.linalg.svdvals(T.permute(1, 0, 2).reshape(Ws, -1).double()
                                ).cpu().numpy()
    # per-output-direction eigenstructure: T[o] is symmetric Ws x Ws
    ev = torch.linalg.eigvalsh(T.double()).cpu().numpy()      # (Ws, Ws)
    aev = np.abs(ev)
    slice_ranks = [tf_fold.eff_rank(np.sort(a)[::-1]) for a in aev]
    out['mlp'] = {
        'shape': list(T.shape), 'fro': float(T.norm()),
        'cp_rank_bound_is_hidden': D.cfg.hidden,
        'mode0_unfolding': {**tf_fold.eff_rank(unf0), 'rank_bound': Ws,
                            'sv_top8': [float(x) for x in unf0[:8]]},
        'mode1_unfolding': {**tf_fold.eff_rank(unf1), 'rank_bound': Ws,
                            'sv_top8': [float(x) for x in unf1[:8]]},
        'slice_eigen': {
            'mean_entropy_rank': float(np.mean(
                [s['entropy_rank'] for s in slice_ranks])),
            'mean_numerical_rank': float(np.mean(
                [s['numerical_rank'] for s in slice_ranks])),
            'rank_bound': Ws,
            'mean_negative_eig_share': float(
                (ev < 0).sum() / ev.size),
            'note': 'each output coordinate o carries a SYMMETRIC quadratic '
                    'form T[o]; its eigenvalues come in both signs, so no '
                    'output is a pure square'},
    }
    # a bound-free null: the same statistics for a random tensor built from
    # random Down/Left/Right of the SAME shapes, so "steep spectrum" is not
    # just what any random factored tensor looks like
    g = torch.Generator(device='cpu').manual_seed(5)
    hidden = D.cfg.hidden
    Dn = torch.randn(Ws, hidden, generator=g).to(D.dev)
    L = torch.randn(hidden, Ws, generator=g).to(D.dev)
    Rr = torch.randn(hidden, Ws, generator=g).to(D.dev)
    Mn = torch.einsum('of,fi,fj->oij', Dn, L, Rr)
    Tn = 0.5 * (Mn + Mn.transpose(1, 2))
    svn = torch.linalg.svdvals(Tn.reshape(Ws, -1).double()).cpu().numpy()
    out['mlp']['random_factored_null_mode0'] = tf_fold.eff_rank(svn)
    return out


# ---------------------------------------------------------------- data
def held_batches(D, n_seq=64, T=256, batch=8, split='held'):
    arr = tf_corpus.load_split(D.V, split, n_seq, tok=D.cfg.tok)
    x = torch.from_numpy(arr[:, :T + 1]).to(D.dev)
    for a in range(0, x.shape[0], batch):
        b = x[a:a + batch]
        yield b[:, :-1], b[:, 1:]


@torch.no_grad()
def ce_of(logits, tgt):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                           tgt.reshape(-1), reduction='sum'), tgt.numel()


# ---------------------------------------------------------------- rung 3+5
@torch.no_grad()
def ladder(D, n_seq=64, T=256, batch=8, extra=True, split='held',
           mlp_ks=(1, 2, 4, 8, 16, 32, 64, 128, 256)):
    """RUNG 5 (and the ablations rung 3 needs): a KL ladder of weights-free
    programs against the true model, on held text.

    Every stage is a table program: look up rows of Ehn / A0 / M0 / OV, look up
    branch factors by token id, apply the rotary, and read out with W_U.  No
    stage calls the network's forward.  The FINAL stage is algebraically the
    model, so its ~0 KL is a control on the ladder, not a result."""
    acc, kl, ntok = {}, {}, 0
    # the distance-only pattern profile (token-independent) is FITTED, so it is
    # fitted on the estimation split, never on the text it is scored on
    prof = _distance_profile(D, T)
    mean_past = _mean_past_attn(D, T)
    for x, y in held_batches(D, n_seq, T, batch, split):
        t = D.terms(x)
        e, A, A0, Mx = t['e'], t['A'], t['A0'], t['M']

        def mlp(r):
            xn = _rms(r, D.Ws)
            return torch.einsum('oij,bti,btj->bto', D.T, xn, xn) + D.bias
        ref = D.readout(e + A + Mx)
        logp_ref = F.log_softmax(ref.float(), -1)
        p_ref = logp_ref.exp()
        cand = {
            # --- the additive ladder ---
            'embed_only': e,
            'plus_self_attn': e + A0,
            'model_bigram': e + A0 + t['M0'],
            'full_exact': e + A + Mx,
            # --- routes of the PAST attention, separated ---
            'past_attn_direct_route_only': e + A + t['M0'],
            'past_attn_mlp_route_only': e + A0 + Mx,
            # --- component knockouts ---
            'no_mlp': e + A,
            'mlp_write_only': Mx,
            'no_attention_at_all': e + mlp(e),
            'past_attn_mean_ablated': e + A0 + mean_past
            + mlp(e + A0 + mean_past),
        }
        if extra:
            ov = D.OV.permute(1, 0, 2)[x]
            for nm, dmax in (('trunc_delta1_only', 1),
                             ('trunc_delta_le4', 4),
                             ('trunc_delta_le16', 16),
                             ('trunc_delta_le64', 64)):
                Ab = torch.einsum('bhqk,bkho->bqo', _band(t['pat'], dmax), ov)
                cand[nm] = e + Ab + mlp(e + Ab)
            patp = prof[None, :, :T, :T].expand(x.shape[0], -1, -1, -1)
            Ap = torch.einsum('bhqk,bkho->bqo', patp, ov)
            cand['positional_only_pattern'] = e + Ap + mlp(e + Ap)
            pat0 = _flat_pattern(D, x)
            Af = torch.einsum('bhqk,bkho->bqo', pat0, ov)
            cand['no_rotary_pattern'] = e + Af + mlp(e + Af)
            for h in range(D.H):
                keep = torch.ones(D.H, device=D.dev)
                keep[h] = 0.0
                Ah = torch.einsum('bhqk,h,bkho->bqo', t['pat'], keep, ov)
                cand[f'drop_head{h}'] = e + Ah + mlp(e + Ah)
            # --- COMPRESSION AXES (rung 5: how small can the program get) ---
            for k in (1, 2, 4):
                if k >= D.hd // 2:
                    continue
                Ak = torch.einsum('bhqk,bkho->bqo',
                                  D.pattern(x, D.freq_mask(k)), ov)
                cand[f'attn_top{k}_rotary_freqs'] = e + Ak + mlp(e + Ak)
            for k in mlp_ks:
                if k >= D.cfg.hidden:
                    continue
                parts = D.mlp_topk(k)
                cand[f'mlp_top{k}_hidden_units'] = e + A + D.mlp_apply(e + A,
                                                                       parts)
        for s, r in cand.items():
            lg = D.readout(r)
            lp = F.log_softmax(lg.float(), -1)
            kl[s] = kl.get(s, 0.0) + float((p_ref * (logp_ref - lp)).sum())
            c, n = ce_of(lg, y)
            a = acc.setdefault(s, [0.0, 0])
            a[0] += float(c)
            a[1] += n
        ntok += y.numel()
    out = {s: {'ce': acc[s][0] / acc[s][1], 'kl_from_model': kl[s] / ntok}
           for s in acc}
    out['_model_ce'] = out['full_exact']['ce']
    out['_tokens'] = ntok
    out['_split'] = split
    return out


@torch.no_grad()
def _mean_past_attn(D, T, n_seq=32):
    """Mean of Apast over the ESTIMATION split, per position -- the ablation
    value for 'remove the past attention without also removing its mean
    offset'.  Zeroing alone changes the residual norm and would confound the
    knockout with a gauge change."""
    arr = tf_corpus.load_split(D.V, 'est', n_seq, tok=D.cfg.tok)
    x = torch.from_numpy(arr[:, :T]).to(D.dev)
    tot, n = None, 0
    for a in range(0, x.shape[0], 8):
        t = D.terms(x[a:a + 8])
        s = (t['A'] - t['A0']).sum(0)
        tot = s if tot is None else tot + s
        n += t['A'].shape[0]
    return (tot / n)[None]


def _band(pat, dmax):
    """Keep only |i-j| <= dmax (self included)."""
    T = pat.shape[-1]
    ar = torch.arange(T, device=pat.device)
    keep = (ar[:, None] - ar[None, :]).abs() <= dmax
    return pat * keep


@torch.no_grad()
def _distance_profile(D, T, n_seq=32):
    """Token-INDEPENDENT pattern: the mean of p_h[i,j] over the estimation
    split at each (i, j).  Fitted OFF the held text it is scored on."""
    arr = tf_corpus.load_split(D.V, 'est', n_seq, tok=D.cfg.tok)
    x = torch.from_numpy(arr[:, :T]).to(D.dev)
    tot = None
    for a in range(0, x.shape[0], 8):
        p = D.pattern(x[a:a + 8]).sum(0)
        tot = p if tot is None else tot + p
    return tot / x.shape[0]


@torch.no_grad()
def _flat_pattern(D, idx):
    """The pattern with the ROTARY REMOVED: score every pair with the delta=0
    table.  Isolates how much of the attention is distance-modulated."""
    B, T = idx.shape
    s1 = torch.einsum('bqhd,bkhd->bhqk', D.Q1.permute(1, 0, 2)[idx],
                      D.K1.permute(1, 0, 2)[idx]) / D.hd
    s2 = torch.einsum('bqhd,bkhd->bhqk', D.Q2.permute(1, 0, 2)[idx],
                      D.K2.permute(1, 0, 2)[idx]) / D.hd
    mask = D.model.mask[:T, :T]
    return (s1 * s2).masked_fill(~mask, 0.0)


@torch.no_grad()
def stream_geometry(D, n_seq=32, T=256, batch=8):
    """EXACT additive attribution of the pre-tanh logit.

    Because RMSNorm is a scalar gauge, the pre-tanh logit is exactly
        z = (sqrt(Ws)/||r||) * (e + A + M) W_U
    so the three folded terms contribute ADDITIVELY to z and their shares can
    be read off without any approximation.  Signs of individual terms are
    meaningful here only because each term is composed all the way to the logit
    (term -> W_U), which is a complete path."""
    tot = {}
    n = 0
    for x, y in held_batches(D, n_seq, T, batch):
        t = D.terms(x)
        r = t['e'] + t['A'] + t['M']
        g = math.sqrt(D.Ws) / r.norm(dim=-1, keepdim=True)
        zs = {k: (t[k] * g) @ D.WU.t() for k in ('e', 'A0', 'Apast', 'M')}
        z = (r * g) @ D.WU.t()
        zc = z - z.mean(-1, keepdim=True)
        for k, v in zs.items():
            vc = v - v.mean(-1, keepdim=True)
            tot.setdefault(f'{k}_norm', 0.0)
            tot[f'{k}_norm'] += float(t[k].norm(dim=-1).sum())
            tot.setdefault(f'{k}_logit_var_share', 0.0)
            # projection share: <zc, vc>/<zc,zc> sums to 1 over terms exactly
            tot[f'{k}_logit_var_share'] += float(
                ((zc * vc).sum(-1) / (zc * zc).sum(-1).clamp_min(1e-30)).sum())
        tot['r_norm'] = tot.get('r_norm', 0.0) + float(r.norm(dim=-1).sum())
        tot['tanh_saturation'] = tot.get('tanh_saturation', 0.0) + float(
            (z.abs() > 30).float().mean(-1).sum())
        n += x.numel()
    out = {k: v / n for k, v in tot.items()}
    out['_shares_sum_to_one'] = sum(
        out[f'{k}_logit_var_share'] for k in ('e', 'A0', 'Apast', 'M'))
    out['_note'] = ('logit_var_share is the exact projection share of each '
                    'folded term on the centred logit vector; the four shares '
                    'sum to 1 by construction (the check above)')
    return out


@torch.no_grad()
def position_profile(D, logPn, uni_log, n_seq=64, T=64, batch=8):
    """FAIRNESS CONTROL for the bigram comparison.  A bigram table sees exactly
    one token of context; the model sees the whole prefix and the position.  So
    the honest head-to-head is POSITION BY POSITION: at position 0 the model
    also has one token, and it is only from position 1 on that it has more."""
    modc = np.zeros(T)
    bigc = np.zeros(T)
    unic = np.zeros(T)
    cnt = np.zeros(T)
    for x, y in held_batches(D, n_seq, T, batch):
        t = D.terms(x)
        lg = D.readout(t['e'] + t['A'] + t['M']).float().log_softmax(-1)
        m = -lg.gather(-1, y[..., None])[..., 0]
        b = -logPn[x, y]
        u = -uni_log[y]
        k = y.shape[1]
        modc[:k] += m.sum(0).cpu().numpy()
        bigc[:k] += b.sum(0).cpu().numpy()
        unic[:k] += u.sum(0).cpu().numpy()
        cnt[:k] += y.shape[0]
    return {'positions': list(range(T)),
            'model_ce': [float(v) for v in modc / cnt],
            'bigram_ce': [float(v) for v in bigc / cnt],
            'unigram_ce': [float(v) for v in unic / cnt],
            'note': 'position 0 is the only place the model and the bigram '
                    'table see the same amount of context'}


# ---------------------------------------------------------------- baselines
@torch.no_grad()
def data_baselines(D, n_seq=64, T=256, batch=8, lowrank=(8, 32, 64),
                   sparse=(262144, 1048576)):
    """Unigram, closed-form bigram, a POSITIONAL-ONLY predictor, and
    rank-matched low-rank bigrams -- the 'same parameter count' comparators the
    rung-5 program has to beat.  All fitted on train/est, scored on held."""
    V = D.V
    # counts over the WHOLE train split (chunked), so this reproduces the
    # program's own closed-form bigram rather than a weaker resample of it
    cnt = torch.zeros(V, dtype=torch.float64)
    big = torch.zeros(V, V, dtype=torch.float32, device=D.dev)
    ntr = 240000
    for a in range(0, ntr, 20000):
        tr = tf_corpus.load_split(V, 'train', a + 20000,
                                  tok=D.cfg.tok)[a:a + 20000]
        cnt.scatter_add_(0, torch.from_numpy(tr.reshape(-1)),
                         torch.ones(tr.size, dtype=torch.float64))
        src = torch.from_numpy(tr[:, :-1].reshape(-1)).to(D.dev)
        dst = torch.from_numpy(tr[:, 1:].reshape(-1)).to(D.dev)
        big.view(-1).scatter_add_(0, src * V + dst,
                                  torch.ones_like(src, dtype=torch.float32))
        del src, dst
    uni = ((cnt + 1.0) / (cnt.sum() + V)).to(D.dev).float()
    alpha = 1000.0                      # the value tf_train tuned on est
    P = (big + alpha * uni[None, :]) / (big.sum(1, keepdim=True) + alpha)
    logP = P.clamp_min(1e-30).log()
    # positional-only: p(next | position bucket), fitted on est
    est = tf_corpus.load_split(V, 'est', 4000, tok=D.cfg.tok)[:, :T + 1]
    pos = torch.zeros(T, V, device=D.dev)
    et = torch.from_numpy(est[:, 1:T + 1].astype(np.int64)).to(D.dev)
    for i in range(T):
        pos[i].scatter_add_(0, et[:, i], torch.ones(et.shape[0], device=D.dev))
    pos = ((pos + 1.0) / (pos.sum(1, keepdim=True) + V)).log()
    logPn = logP.log_softmax(-1)
    # low-rank bigram: truncated SVD of the smoothed log-bigram, 2*V*r params
    lr = {}
    mu = logP.mean()
    U, S, Vh = torch.svd_lowrank(logP - mu, q=max(lowrank) + 8, niter=4)
    for r in lowrank:
        lr[f'lowrank_bigram_r{r}'] = (
            (U[:, :r] * S[:r]) @ Vh[:, :r].t() + mu).log_softmax(-1)
    del U, S, Vh
    # SPARSE bigram: keep the m largest counts exactly, back off to unigram.
    # 2m numbers (an index and a count), and a much stronger same-budget
    # comparator than a truncated SVD of a log-probability matrix.
    for m in sparse:
        v, i = torch.topk(big.view(-1), m)
        sp = torch.zeros_like(big)
        sp.view(-1)[i] = v
        lr[f'sparse_bigram_m{m}'] = (
            (sp + alpha * uni[None, :])
            / (sp.sum(1, keepdim=True) + alpha)).clamp_min(1e-30).log()
        del sp, v, i
    accs = {k: [0.0, 0] for k in ['unigram', 'bigram', 'positional_only']
            + list(lr)}
    lu = uni.log()
    for x, y in held_batches(D, n_seq, T, batch):
        n = y.numel()
        accs['unigram'][0] += float(-lu[y].sum())
        accs['unigram'][1] += n
        accs['bigram'][0] += float(-logPn[x, y].sum())
        accs['bigram'][1] += n
        accs['positional_only'][0] += float(
            -pos[torch.arange(y.shape[1], device=D.dev)[None, :]
                 .expand_as(y), y].sum())
        accs['positional_only'][1] += n
        for k, tab in lr.items():
            accs[k][0] += float(-tab[x, y].sum())
            accs[k][1] += n
    out = {k: v[0] / v[1] for k, v in accs.items()}
    out['position_profile'] = position_profile(D, logPn, lu, n_seq, 64, batch)
    out['_params'] = {
        'unigram': V, 'bigram_dense': V * V,
        **{f'lowrank_bigram_r{r}': 2 * V * r for r in lowrank},
        **{f'sparse_bigram_m{m}': 2 * m for m in sparse},
        'model_bigram_program_tables': 2 * V * D.Ws,
        'note': 'the stage-3 program (model bigram) is fully specified by two '
                'V x Ws tables, r0 and W_U, plus the fixed readout code; the '
                'matched comparators are the ones with the same count'}
    del big, P, logP, logPn, lr
    torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------- induction
@torch.no_grad()
def _induction_target(idx, V):
    """(B, T) -- the token that followed the most recent EARLIER occurrence of
    tok_i, or -1 when there is none.  Vectorised over the batch (T steps of
    tensor ops, not B*T python steps)."""
    B, T = idx.shape
    dev = idx.device
    last = torch.full((B, V), -1, dtype=torch.long, device=dev)
    tgt = torch.full((B, T), -1, dtype=torch.long, device=dev)
    for i in range(T):
        cur = idx[:, i:i + 1]
        prev = last.gather(1, cur)[:, 0]
        ok = (prev >= 0) & (prev + 1 < T)
        if ok.any():
            nxt = idx.gather(1, (prev + 1).clamp(0, T - 1)[:, None])[:, 0]
            tgt[:, i] = torch.where(ok, nxt, tgt[:, i])
        last.scatter_(1, cur, torch.full_like(cur, i))
    return tgt


@torch.no_grad()
def induction_sensitivity(D, epsilons=(0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
                          n_seq=48, L=96, seeds=3, model=None):
    """SENSITIVITY CALIBRATION.  A null induction score means nothing until we
    know how much induction the battery could have seen, so a KNOWN amount is
    planted and the detection floor is read off.

    The planted predictor is a mixture (1-eps)*model + eps*ORACLE, where the
    oracle puts (almost) all its mass on the token that followed the most
    recent earlier occurrence of the current token -- a perfect induction head.
    In the repeat condition it is right; in the shuffled condition it is wrong,
    but a mixture can only cost -log(1-eps) there, so the score is dominated by
    the genuine induction gain and eps maps monotonically onto 'how much
    induction'."""
    fwd = model or D.model
    V = D.V
    tr = tf_corpus.load_split(V, 'train', 4000, tok=D.cfg.tok)
    cnt = np.bincount(tr.reshape(-1), minlength=V).astype(np.float64) + 1.0
    p = torch.from_numpy(cnt / cnt.sum()).float()
    scores = {f'eps_{e}': [] for e in epsilons}
    for sd in range(seeds):
        g = torch.Generator(device='cpu').manual_seed(1000 + sd)
        R = torch.multinomial(p.expand(n_seq, V), L, True, generator=g)
        perm = torch.argsort(torch.rand(n_seq, L, generator=g), dim=1)
        conds = {'repeat': torch.cat([R, R], 1),
                 'shuffled': torch.cat([torch.gather(R, 1, perm), R], 1)}
        ce = {e: {} for e in epsilons}
        for nm, s in conds.items():
            s = s.to(D.dev)
            tot = {e: 0.0 for e in epsilons}
            n = 0
            for a in range(0, n_seq, 8):
                b = s[a:a + 8]
                x, y = b[:, :-1], b[:, 1:]
                pm = F.log_softmax(fwd(x).float(), -1).exp()
                tgt = _induction_target(x, V)
                hit = tgt >= 0
                pm_t = pm.gather(-1, y[..., None])[..., 0]
                orc_t = torch.where(hit & (tgt == y),
                                    torch.ones_like(pm_t),
                                    torch.full_like(pm_t, 1.0 / V))
                orc_t = torch.where(hit, orc_t, torch.full_like(pm_t, 1.0 / V))
                for e in epsilons:
                    q = ((1 - e) * pm_t + e * orc_t)[:, L - 1:]
                    tot[e] += float(-q.clamp_min(1e-30).log().sum())
                n += y[:, L - 1:].numel()
            for e in epsilons:
                ce[e][nm] = tot[e] / n
        for e in epsilons:
            scores[f'eps_{e}'].append(ce[e]['shuffled'] - ce[e]['repeat'])
    return {k: {'induction_score_mean': float(np.mean(v)),
                'induction_score_sd': float(np.std(v, ddof=1))}
            for k, v in scores.items()}


@torch.no_grad()
def induction_for_stem(stem, seeds=5, device=DEV, sensitivity=True):
    """The identical battery for ANY depth, so a depth-2 cell can serve as the
    POSITIVE CONTROL for the depth-1 null result."""
    model, cfg, _ = tf_fold.load_checkpoint(stem, device)

    class _Shim:
        pass
    s = _Shim()
    s.V, s.cfg, s.dev, s.model = cfg.vocab, cfg, device, model
    runs = [induction_battery(s, seed=i) for i in range(seeds)]
    return {'stem': stem, 'depth': cfg.depth, 'width': cfg.width,
            'per_seed': runs,
            'induction_score_mean': float(np.mean([r['induction_score']
                                                   for r in runs])),
            'induction_score_sd': float(np.std([r['induction_score']
                                                for r in runs], ddof=1)),
            'bag_score_mean': float(np.mean([r['bag_score'] for r in runs])),
            'bag_score_sd': float(np.std([r['bag_score'] for r in runs],
                                         ddof=1)),
            'sensitivity': induction_sensitivity(s) if sensitivity else None}


@torch.no_grad()
def induction_battery(D, n_seq=48, L=96, seed=0, model=None, use_model=True):
    """RUNG 3, the structural prediction.

    Three matched synthetic conditions, identical token multisets:
      repeat   : [R] [R]              -- order-preserving repetition
      shuffled : [shuffle(R)] [R]     -- same bag in the prefix, order destroyed
      control  : [R'] [R]             -- a different random prefix

    induction score = CE(shuffled) - CE(repeat)   (needs ORDER: 2 layers)
    bag score       = CE(control)  - CE(shuffled) (needs only a BAG: 1 layer)

    NAMING DISCIPLINE: the second number is deliberately NOT called a 'copy
    score'.  It measures only that an order-free repetition of the token bag in
    the prefix lowers CE; it says nothing about the mechanism, and the rung-4
    composed measurement in this same file shows these heads push the attended
    token's own logit DOWN relative to a random pair.  Calling it 'copying'
    would be naming a mechanism from a behavioural delta.

    Reported per-condition on the SECOND half only.  Tokens are drawn from the
    train unigram so the conditions are not out-of-distribution in frequency."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    V = D.V
    tr = tf_corpus.load_split(V, 'train', 4000, tok=D.cfg.tok)
    cnt = np.bincount(tr.reshape(-1), minlength=V).astype(np.float64) + 1.0
    p = torch.from_numpy(cnt / cnt.sum()).float()
    R = torch.multinomial(p.expand(n_seq, V), L, replacement=True,
                          generator=g)
    R2 = torch.multinomial(p.expand(n_seq, V), L, replacement=True,
                           generator=g)
    perm = torch.argsort(torch.rand(n_seq, L, generator=g), dim=1)
    Rs = torch.gather(R, 1, perm)
    seqs = {'repeat': torch.cat([R, R], 1),
            'shuffled': torch.cat([Rs, R], 1),
            'control': torch.cat([R2, R], 1)}
    fwd = (model or D.model)
    out = {}
    for nm, s in seqs.items():
        s = s.to(D.dev)
        tot, n = 0.0, 0
        for a in range(0, n_seq, 8):
            b = s[a:a + 8]
            lg = fwd(b[:, :-1]) if use_model else D.readout(
                sum(D.terms(b[:, :-1])[k] for k in ('e', 'A', 'M')))
            # score the SECOND copy only
            lg = lg[:, L - 1:]
            tgt = b[:, L:]
            c, m = ce_of(lg, tgt)
            tot += float(c)
            n += m
        out[nm] = tot / n
    out['induction_score'] = out['shuffled'] - out['repeat']
    out['bag_score'] = out['control'] - out['shuffled']
    # a paired bootstrap over sequences would be better; a cheap SE comes from
    # re-running the whole battery at 4 more seeds
    return out


# ---------------------------------------------------------------- rung 4
@torch.no_grad()
def rung4(D, topk=25, nsample=1024, seed=0):
    """Composed, sign-invariant token-space analysis.

    The object decoded is the COPY SCORE
        copy_h(t, u) = p_h(t, u, delta) * (OV_h[u] . W_U[u])
    -- 'how much does attending to u from query token t push u's own logit up'.
    It is a complete path (pattern x value x output x unembedding) so its sign
    is not a gauge choice, unlike the sign of p or of OV separately.

    Also reported: the DIFFUSENESS of each head's composed contribution
    (effective number of token pairs), and a frequency-matched null for the
    orthographic-class claims."""
    V, H = D.V, D.H
    vocab = tf_corpus.load_vocab(D.V, D.cfg.tok)
    out = {'note': 'all quantities composed to logits; raw factor signs are a '
                   'gauge freedom and are never interpreted'}
    # self-unembedding of each head's value write: (H, V)
    ovu = torch.stack([(D.OV[h] * D.WU).sum(-1) for h in range(H)])
    heads = []
    g = torch.Generator(device='cpu').manual_seed(seed)
    samp = torch.randperm(V, generator=g)[:nsample].to(D.dev)
    for h in range(H):
        hd_out = {'head': h}
        for d in (0, 1, 4):
            R = M.rot_matrix(d, D.hd, D.dev)
            s1 = (D.Q1[h][samp] @ R) @ D.K1[h][samp].t() / D.hd
            s2 = (D.Q2[h][samp] @ R) @ D.K2[h][samp].t() / D.hd
            pat = s1 * s2                                   # (n, n)
            comp = pat * ovu[h][samp][None, :]              # composed copy
            a = comp.abs()
            eff = float(a.sum() ** 2 / (a ** 2).sum())
            hd_out[f'delta{d}'] = {
                'effective_pair_count': eff,
                'pairs_total': int(a.numel()),
                'effective_pair_fraction': eff / a.numel(),
                'pattern_abs_mean': float(pat.abs().mean()),
                'composed_abs_mean': float(a.mean()),
                'diag_mean_composed': float(comp.diagonal().mean()),
                'offdiag_mean_composed': float(
                    (comp.sum() - comp.diagonal().sum())
                    / (comp.numel() - comp.shape[0])),
            }
            # is the composed pair table genuinely PAIR-specific, or just an
            # outer product query_score x key_score?  sigma1 share of the SVD
            # answers it without any naming.  (delta=1 only: the fp64 SVD of a
            # 1024^2 matrix is the expensive part of this file.)
            if d == 1:
                sv = torch.linalg.svdvals(comp.double())
                hd_out[f'delta{d}']['composed_rank1_share'] = float(
                    sv[0] / sv.sum())
                hd_out[f'delta{d}']['composed_entropy_rank'] = \
                    tf_fold.eff_rank(sv.cpu().numpy())['entropy_rank']
                hd_out[f'delta{d}']['composed_rank_bound'] = min(
                    D.hd * D.hd, comp.shape[0])
            if d == 1:
                flat = comp.reshape(-1)
                hi = torch.topk(flat, topk)
                lo = torch.topk(-flat, topk)
                n = comp.shape[0]

                def lab(i):
                    return [tf_corpus.label(vocab, int(samp[int(i) // n])),
                            tf_corpus.label(vocab, int(samp[int(i) % n]))]
                hd_out['top_composed_copy_pairs'] = [
                    lab(i) + [float(v)] for i, v in
                    zip(hi.indices.cpu(), hi.values.cpu())]
                hd_out['bottom_composed_copy_pairs'] = [
                    lab(i) + [float(-v)] for i, v in
                    zip(lo.indices.cpu(), lo.values.cpu())]
                # identity-pair enrichment: is attending to u boosting u itself
                # bigger than attending to a random other token?
                dg = comp.diagonal()
                off = flat[torch.randperm(flat.numel(),
                                          generator=g)[:flat.numel() // 50]
                           .to(D.dev)]
                hd_out['identity_pair_enrichment'] = {
                    'mean_diag': float(dg.mean()),
                    'mean_random_pair': float(off.mean()),
                    'diag_over_random_abs': float(
                        dg.abs().mean() / off.abs().mean()),
                    'z_of_diag_vs_random': float(
                        (dg.mean() - off.mean())
                        / (off.std() / math.sqrt(dg.numel())))}
            del s1, s2, pat, comp
        heads.append(hd_out)
    out['heads'] = heads
    # ---- what does attending to a token DO?  composed to logits. ----
    what = []
    for h in range(H):
        nrm = (D.OV[h] @ D.WU.t()).norm(dim=-1)
        top = nrm.topk(8).indices
        ent = []
        for u in top.tolist():
            z = D.OV[h][u] @ D.WU.t()
            hi = z.topk(5)
            ent.append({'key_token': tf_corpus.label(vocab, u),
                        'boosts': [[tf_corpus.label(vocab, int(i)), float(v)]
                                   for i, v in zip(hi.indices.cpu(),
                                                   hi.values.cpu())],
                        'boosts_itself_rank': int(
                            (z > z[u]).sum()) + 1})
        what.append({'head': h, 'strongest_keys': ent})
    out['what_attending_does'] = {
        'note': 'OV_h[u] W_U^T -- the complete path value->output->unembedding, '
                'so these signs are invariant.  boosts_itself_rank = where the '
                'attended token ranks among the tokens it boosts (1 = a pure '
                'copy head).',
        'per_head': what}
    # ---- token-class null for the value-write direction ----
    out['token_classes'] = _class_enrichment(D, ovu, vocab, seed)
    return out


def _tok_class(s):
    """Cheap orthographic classes on the BPE surface string."""
    if s.startswith('<') and s.endswith('>'):
        return 'special'
    core = s[1:] if s[:1] in (' ', 'Ġ') else s
    if core == '':
        return 'space'
    if any(c.isdigit() for c in core):
        return 'digit'
    if not any(c.isalnum() for c in core):
        return 'punct'
    if core[:1].isupper():
        return 'capitalized'
    return 'lower'


@torch.no_grad()
def _class_enrichment(D, ovu, vocab, seed=0, topn=200, ndraw=400):
    """For each head, take the tokens whose COMPOSED self-write is largest and
    ask whether their orthographic classes are enriched relative to a
    FREQUENCY-MATCHED null (tokens drawn with the same train-frequency
    distribution).  Without the null, 'the top tokens are mostly whitespace-
    initial lowercase words' is just a statement about the vocabulary."""
    V = D.V
    tr = tf_corpus.load_split(V, 'train', 4000, tok=D.cfg.tok)
    freq = np.bincount(tr.reshape(-1), minlength=V).astype(np.float64) + 1.0
    freq /= freq.sum()
    dec = vocab['decoded']
    labs = [dec[i] if i < len(dec) else '' for i in range(V)]
    cls = np.array([_tok_class(s) for s in labs])
    rng = np.random.default_rng(seed)
    out = {}
    for h in range(D.H):
        score = ovu[h].abs().cpu().numpy()
        top = np.argsort(-score)[:topn]
        obs = {c: float((cls[top] == c).mean()) for c in set(cls)}
        # frequency-matched null: draw topn tokens with the train frequency of
        # the SELECTED set, i.e. match the marginal the selection could exploit
        w = freq / freq.sum()
        null = np.zeros((ndraw, len(obs)))
        keys = sorted(obs)
        for b in range(ndraw):
            d = rng.choice(V, size=topn, replace=False, p=w)
            for ki, c in enumerate(keys):
                null[b, ki] = (cls[d] == c).mean()
        out[f'head{h}'] = {
            c: {'observed': obs[c], 'null_mean': float(null[:, ki].mean()),
                'null_sd': float(null[:, ki].std()),
                'z': float((obs[c] - null[:, ki].mean())
                           / max(null[:, ki].std(), 1e-9))}
            for ki, c in enumerate(keys)}
        out[f'head{h}']['_null'] = ('frequency-matched draws of the same size '
                                    'from the train unigram')
    return out


# ---------------------------------------------------------------- driver
def main(stem, quick=False, out_json=None):
    t0 = time.time()
    jp = out_json or f'{HERE}/{stem}_interp.json'
    rep = json.load(open(jp)) if os.path.exists(jp) else {}
    rep['registered_predictions'] = REGISTERED
    rep['stem'] = stem
    json.dump(rep, open(jp, 'w'), indent=2)          # registered BEFORE running
    D = Depth1Fold(stem)
    x = torch.from_numpy(tf_corpus.load_split(D.V, 'held', 4,
                                              tok=D.cfg.tok))[:, :128].to(D.dev)
    rep['decomposition_control'] = D.self_check(x)
    assert rep['decomposition_control']['pass'], rep['decomposition_control']
    print('  decomposition control:', rep['decomposition_control'], flush=True)
    n_seq, T = (16, 128) if quick else (96, 256)
    rep['rung2'] = rung2(D)
    print(f'  rung2 done {time.time()-t0:.0f}s', flush=True)
    rep['rung3_baselines'] = data_baselines(D, n_seq, T)
    print(f'  baselines done {time.time()-t0:.0f}s', flush=True)
    rep['stream_geometry'] = stream_geometry(D, min(n_seq, 32), T)
    print(f'  geometry done {time.time()-t0:.0f}s', flush=True)
    rep['rung5_ladder'] = ladder(D, n_seq, T)
    print(f'  ladder done {time.time()-t0:.0f}s', flush=True)
    ind = [induction_battery(D, seed=s) for s in range(3 if quick else 5)]
    rep['rung3_induction'] = {
        'per_seed': ind,
        'induction_score_mean': float(np.mean([i['induction_score']
                                               for i in ind])),
        'induction_score_sd': float(np.std([i['induction_score']
                                            for i in ind], ddof=1)),
        'bag_score_mean': float(np.mean([i['bag_score'] for i in ind])),
        'bag_score_sd': float(np.std([i['bag_score'] for i in ind], ddof=1)),
        'design': 'repeat vs shuffled-prefix vs fresh-prefix, matched token '
                  'multisets, scored on the second copy only'}
    print(f'  induction done {time.time()-t0:.0f}s', flush=True)
    rep['rung4'] = rung4(D)
    rep['seconds'] = round(time.time() - t0, 1)
    json.dump(rep, open(jp, 'w'), indent=2)
    print(f'== interp written to {jp} ({rep["seconds"]}s)', flush=True)
    return rep


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stem', required=True)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--induction-only', action='store_true',
                    help='any depth: the induction battery alone (used to get '
                         'the depth-2 positive control for the depth-1 null)')
    a = ap.parse_args()
    if a.induction_only:
        r = induction_for_stem(a.stem)
        jp = f'{HERE}/{a.stem}_induction.json'
        json.dump(r, open(jp, 'w'), indent=2)
        print(json.dumps({k: v for k, v in r.items()
                          if k != 'per_seed'}, indent=2))
        raise SystemExit(0)
    r = main(a.stem, a.quick)
    print(json.dumps({'ladder': r['rung5_ladder'],
                      'induction': {k: v for k, v in r['rung3_induction'].items()
                                    if not k.startswith('per')},
                      'baselines': r['rung3_baselines']}, indent=2))
