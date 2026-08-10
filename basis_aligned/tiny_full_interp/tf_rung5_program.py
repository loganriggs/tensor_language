"""RUNG 5, FOR REAL: a genuinely weights-free PROGRAM (code + tables indexed by
discrete symbols) fitted on `est` and scored against the model on `held`, with a
STAGED ladder, an honest bit bill, and three positive controls.

WHY THIS FILE EXISTS
--------------------
The rung-5 section of `tf_interp3.ladder_v` is ~90 entries long and every entry
but one runs the model's weights: it is a component-ABLATION ladder, not a
reconstruction ladder.  The single exception is `model_bigram` (the model's own
length-1 output table), and FINDING 23 aggregated it across 52 checkpoints: KL
0.258 at depth 1 width 32 rising monotonically to 1.204 at depth 4 width 256.
That is the SIMPLEST weights-free program, so it is a LOWER bound on what a
weights-free program can do, and FINDING 23 says in as many words that a ceiling
result needs "the best weights-free program we can write".  This file writes one.

THE GRAMMAR (what a program is allowed to contain)
--------------------------------------------------
Tables indexed by TOKEN, TOKEN-PAIR, POSITION or POSITIONAL DELTA; arithmetic on
the entries of those tables; and nothing else.  No matrix that multiplies a
computed activation, and no call into the network.

That boundary needs stating precisely, because the programme has been burned by
it before (README, rung-5 restatement): an 8192x128 array called *the embedding*
and an 8192x8192 array called *the bigram table* are both just tables.  The rule
used here, stated so a reviewer can attack it:

  * EVERY parameter of the program is attached to a discrete symbol on at least
    one axis -- a token id, a position, or a distance.  `Pc[token, k]`, `W[token,
    k]`, `g[head, delta]` all qualify.
  * NOTHING is a map between two latent spaces.  The bilinear MLP's tensor
    `T[o,i,j]`, the query/key projections `W_q`, the decoder `W_proj` -- all
    indexed by latent dimensions on every axis -- are FORBIDDEN, and none of
    them appears here.
  * The decoder (this file) is free: causal masking, the fixed rotary
    frequencies, softmax, and the arithmetic are source code, not description.
    Anything with a fitted number in it is charged.

Under that rule the layer-0 attention of these models is exactly weights-free
(it folds to per-token factor tables and a rotation that depends only on the
distance), and the MLP is exactly NOT.  Control (i) below uses that fact as a
known-answer test of this file's attention machinery.

THE LADDER (one ingredient at a time, warm-started so each stage starts from the
previous stage's fitted tables)
--------------------------------------------------------------------------------
  A unigram                 U[v]
  B + current-token content z = Pc[x_t];   logits = U[v] + <z, W[v]>
  C + prefix-mean context   z += c * mean_{j<=t} Vc[x_j]
  D + distance profile      z += sum_j g[t-j] Vc[x_j]
  E + token-pair gate       z += sum_j g[t-j] (1 + <Kq[x_t], Kk[x_j]>) Vc[x_j]
  F + rotary in the gate    ... <rot_t Kq[x_t], rot_j Kk[x_j]> ...
  G + second gate branch    ... (1+s1)(1+s2) -- the exact SHAPE of the fold
  H + heads                 nh independent gates, each with its own value table
  I + induction rule        copy the token that followed the last occurrence of
                            x_t, with a per-token gain
  J + squared content       logits += <z*z, W2[v]>   (BOUNDARY CASE, labelled)

Stage J is flagged rather than hidden: `z*z` is arithmetic on table entries and
`W2` is token-indexed, so it is legal under the grammar above, but it is the
ingredient that comes closest to re-implementing the bilinear MLP and it is
reported separately from the headline.

BITS
----
Every stage's table entries are counted and charged 32 bits (tf_compress.py's
convention), and the model is charged 32 bits per parameter with its tied
embedding counted ONCE.  A stage whose bill exceeds the model's is a FAILURE and
is labelled `over_model_budget: true`.  The dense V x V bigram is charged
honestly too (67.1M entries = 2.1 Gbit), and so is its factored form (the model's
length-1 residual table plus the unembedding table, 2*V*Ws entries), which is the
only fair "comparable size" reading of the baseline.

CONTROLS (without these the numbers do not count)
-------------------------------------------------
  (i)  EXACT FOLD.  Instantiated with the model's own folded layer-0 factors
       (Q1,K1,Q2,K2 per head, the OV table per head, no truncation, no fitted
       bias term) the SAME attention code must reproduce the model's layer-0
       attention write to float precision.  If it does not, the harness is wrong
       and nothing else in the file means anything.
  (ii) SHUFFLED TABLES.  Permuting the token axis of the fitted tables must
       destroy the KL, so the gate can fail.
  (iii) SPLITS.  Estimation rows, held rows, token counts and context length are
       reported with every number, and nothing is ever fitted on held.

Usage
    python tf_rung5_program.py --smoke
    python tf_rung5_program.py --cells tf_vanilla_d1_w32_b8192_s0,...
"""
import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field, replace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import tf_corpus
import tf_fold
import tf_model as M

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
BITS_PER_ENTRY = 32


# ===========================================================================
# the program
# ===========================================================================
@dataclass
class PCfg:
    V: int
    T: int = 256
    r: int = 32                 # content rank (columns of every content table)
    a: int = 8                  # gate rank per branch
    nh: int = 1                 # number of independent gates
    bigram: bool = False        # Pc[x_t]
    ctx: str = ''               # '' | 'mean' | 'delta' | 'pair' | 'pair_rot'
                                # | 'twobranch'
    induction: bool = False
    square: bool = False
    tied: bool = False          # Pc is the SAME table as W (halves the bill)
    # --- control mode only ---
    exact: bool = False         # gate = s1*s2 (no +1, no profile), model readout


def _rope(T, a, dev):
    c, s = M.rope_tables_exact(T, a, 'cpu')
    return c.to(dev), s.to(dev)


class Program(nn.Module):
    """The weights-free program.  Every parameter is a table with a discrete
    index on its first axis (token, head, or head x distance)."""

    def __init__(self, cfg: PCfg, device=DEV):
        super().__init__()
        self.cfg = cfg
        V, T, r, a, nh = cfg.V, cfg.T, cfg.r, cfg.a, cfg.nh
        self.dev = device
        g = torch.Generator(device='cpu').manual_seed(20260810)

        def tab(*shape, std=0.02):
            return nn.Parameter(torch.randn(*shape, generator=g) * std)

        self.U = nn.Parameter(torch.zeros(V))
        if cfg.bigram or cfg.ctx:
            self.W = tab(V, r)
        if cfg.bigram and not cfg.tied:
            self.Pc = tab(V, r)
        if cfg.ctx:
            self.Vc = tab(nh, V, r)
            self.cscale = nn.Parameter(torch.ones(nh))
            if cfg.ctx != 'mean':
                self.g = nn.Parameter(torch.full((nh, T), 1.0 / T))
            if cfg.ctx in ('pair', 'pair_rot', 'twobranch'):
                self.Kq = tab(nh, V, a)
                self.Kk = tab(nh, V, a)
            if cfg.ctx == 'twobranch':
                self.Kq2 = tab(nh, V, a)
                self.Kk2 = tab(nh, V, a)
        if cfg.induction:
            self.lam0 = nn.Parameter(torch.zeros(1))
            self.lam_tok = nn.Parameter(torch.zeros(V))
        if cfg.square:
            self.W2 = tab(V, r, std=0.002)
        ar = torch.arange(T)
        self.register_buffer('offmat', (ar[:, None] - ar[None, :]).clamp(min=0))
        self.register_buffer('causal', torch.tril(torch.ones(T, T,
                                                             dtype=torch.bool)))
        self.register_buffer('invlen', 1.0 / torch.arange(1, T + 1).float())
        cos, sin = _rope(T, a, 'cpu')
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)
        self.to(device)

    # ------------------------------------------------------------------ bits
    def table_entries(self):
        d = {}
        for n, p in self.named_parameters():
            d[n] = int(p.numel())
        return d

    def bits(self, b=BITS_PER_ENTRY):
        e = self.table_entries()
        return {'entries': e, 'total_entries': int(sum(e.values())),
                'bits_per_entry': b,
                'total_bits': int(sum(e.values()) * b)}

    # -------------------------------------------------------------- pieces
    def _pair(self, Kq, Kk, x, rot):
        """(B, nh, Tq, Tk) pairwise score from two token-indexed factor tables."""
        B, Tq = x.shape
        q = Kq[:, x].permute(1, 2, 0, 3)                     # (B,Tq,nh,a)
        k = Kk[:, x].permute(1, 2, 0, 3)
        if rot:
            c = self.cos[None, :Tq, None, :]
            s = self.sin[None, :Tq, None, :]
            q = M.apply_rot(q, c, s)
            k = M.apply_rot(k, c, s)
        return torch.einsum('bqha,bkha->bhqk', q, k)

    def gate(self, x, exact_scale=None):
        """The (B, nh, Tq, Tk) causal weight table.  ONE code path, shared by the
        fitted program and by the exact-fold positive control."""
        cfg = self.cfg
        B, Tq = x.shape
        mask = self.causal[:Tq, :Tq]
        if cfg.ctx == 'mean':
            w = (self.invlen[:Tq, None] * mask.float()).expand(cfg.nh, Tq, Tq)
            return (self.cscale[:, None, None] * w)[None].expand(B, -1, -1, -1)
        w = self.g[:, self.offmat[:Tq, :Tq]][None]           # (1,nh,Tq,Tq)
        w = w.expand(B, -1, -1, -1)
        if cfg.ctx in ('pair', 'pair_rot', 'twobranch'):
            sc = 1.0 if exact_scale is None else exact_scale
            bias = 0.0 if cfg.exact else 1.0
            s1 = self._pair(self.Kq, self.Kk, x, rot=cfg.ctx != 'pair') * sc
            w = w * (bias + s1)
            if cfg.ctx == 'twobranch':
                s2 = self._pair(self.Kq2, self.Kk2, x, rot=True) * sc
                w = w * (bias + s2)
        w = w * self.cscale[None, :, None, None]
        return w.masked_fill(~mask, 0.0)

    def context(self, x, exact_scale=None):
        w = self.gate(x, exact_scale)
        vv = self.Vc[:, x]                                   # (nh,B,Tk,r)
        return torch.einsum('bhqk,hbkr->bqr', w, vv)

    # ------------------------------------------------------------- forward
    def forward(self, x):
        cfg = self.cfg
        B, Tq = x.shape
        logits = self.U[None, None, :].expand(B, Tq, -1)
        z = None
        if cfg.bigram:
            z = (self.W if cfg.tied else self.Pc)[x]
        if cfg.ctx:
            c = self.context(x)
            z = c if z is None else z + c
        if z is not None:
            logits = logits + z @ self.W.t()
        if cfg.square:
            logits = logits + (z * z) @ self.W2.t()
        if cfg.induction:
            jj = torch.arange(Tq, device=x.device)
            eq = (x[:, :, None] == x[:, None, :]) & (jj[None, :, None]
                                                     > jj[None, None, :])
            m = torch.where(eq, jj[None, None, :].expand_as(eq),
                            torch.full_like(eq, -1, dtype=torch.long))
            jstar = m.max(-1).values                          # (B,Tq)
            has = (jstar >= 0).float()
            nxt = x.gather(1, (jstar + 1).clamp(min=0, max=Tq - 1))
            lam = (self.lam0 + self.lam_tok[x]) * has
            logits = logits.clone()
            logits.scatter_add_(2, nxt[..., None], lam[..., None])
        return logits


# ===========================================================================
# fitting (est) and scoring (held)
# ===========================================================================
def load_x(V, split, n_seq, T, tok='bpe', device=DEV):
    arr = tf_corpus.load_split(V, split, n_seq, tok=tok)
    return torch.from_numpy(arr[:, :T + 1]).to(device)


@torch.no_grad()
def model_logits(model, x):
    return model(x).float()


@torch.no_grad()
def init_unigram(prog, model, xs, T, batch=8, nmax=64):
    """Initialise U at the log of the model's marginal next-token distribution
    over the ESTIMATION split.  A fit on est, so it is legal, and it removes a
    few hundred wasted optimiser steps."""
    acc = torch.zeros(prog.cfg.V, device=xs.device, dtype=torch.float64)
    n = 0
    for a in range(0, min(nmax, xs.shape[0]), batch):
        b = xs[a:a + batch, :T]
        p = F.softmax(model_logits(model, b), -1).double()
        acc += p.reshape(-1, p.shape[-1]).sum(0)
        n += b.shape[0] * b.shape[1]
    prog.U.data.copy_((acc / n).clamp_min(1e-12).log().float())


def fit(prog, model, xs, T, steps, lr, batch=8, seed=0, log=None):
    """Distil the model's next-token distribution into the tables, on est."""
    opt = torch.optim.Adam(prog.parameters(), lr=lr, betas=(0.9, 0.99))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr / 20)
    g = torch.Generator(device='cpu').manual_seed(1234 + seed)
    N = xs.shape[0]
    prog.train()
    for it in range(steps):
        sel = torch.randint(0, N, (batch,), generator=g).to(xs.device)
        b = xs[sel][:, :T]
        with torch.no_grad():
            pref = F.softmax(model_logits(model, b), -1)
        lp = F.log_softmax(prog(b).float(), -1)
        loss = -(pref * lp).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prog.parameters(), 1.0)
        opt.step()
        sch.step()
        if log is not None and (it % max(1, steps // 5) == 0 or it == steps - 1):
            log.append((it, float(loss.detach())))
    prog.eval()
    return prog


@torch.no_grad()
def score(prog, model, xs, T, batch=8):
    """KL(model || program) per token, plus both cross-entropies against the
    real next token.  `prog` may be None -> scores the model against itself."""
    kl = ce_p = ce_m = 0.0
    n = 0
    for a in range(0, xs.shape[0], batch):
        b = xs[a:a + batch]
        x, y = b[:, :T], b[:, 1:T + 1]
        ml = model_logits(model, x)
        lpm = F.log_softmax(ml, -1)
        pm = lpm.exp()
        lg = prog(x).float()
        lpp = F.log_softmax(lg, -1)
        kl += float((pm * (lpm - lpp)).sum())
        ce_p += float(F.cross_entropy(lpp.reshape(-1, lpp.shape[-1]),
                                      y.reshape(-1), reduction='sum'))
        ce_m += float(F.cross_entropy(lpm.reshape(-1, lpm.shape[-1]),
                                      y.reshape(-1), reduction='sum'))
        n += y.numel()
    return {'kl_from_model': kl / n, 'program_ce': ce_p / n,
            'model_ce': ce_m / n, 'tokens': n}


# ===========================================================================
# the reference programs (baselines) -- not fitted, derived from the weights
# ===========================================================================
class BigramTableProgram(nn.Module):
    """`model_bigram` as an explicit program: one table R0 (V, Ws) holding the
    model's length-1 residual for every token, one table WU (V, Ws), and the
    fixed arithmetic rms/tanh.  This is EXACTLY the artefact FINDING 23 measured,
    written so it can be charged in bits.  Its dense reading is a V x V logit
    table; its factored reading is 2*V*Ws entries, and the factored reading is
    the one used for 'comparable size'."""

    def __init__(self, model, device=DEV, chunk=1024):
        super().__init__()
        self.Ws = model.Ws
        self.V = model.cfg.vocab
        rows = []
        with torch.no_grad():
            for a in range(0, self.V, chunk):
                ids = torch.arange(a, min(a + chunk, self.V),
                                   device=device)[:, None]
                r = self._resid(model, ids)
                rows.append(r[:, 0])
        self.register_buffer('R0', torch.cat(rows))
        self.register_buffer('WU', model.wte.weight.detach().float().clone())

    @staticmethod
    @torch.no_grad()
    def _resid(model, ids):
        """The model's residual stream at a length-1 context.  Uses the model,
        but only ONCE, to build a table; the program below never calls it."""
        cfg = model.cfg
        Ws = model.Ws
        e = F.rms_norm(model.wte(ids), (Ws,))
        rem = model.remnants(e)
        streams = []

        def entry(li):
            tot = rem[li]
            for i in range(2 * li):
                tot = streams[i] if tot is None else tot + streams[i]
            return tot
        B, Tq = ids.shape
        cos = model.cos[None, :Tq, None, :]
        sin = model.sin[None, :Tq, None, :]
        for li, blk in enumerate(model.h):
            x = entry(li)
            hn = model.slot_norm(x)
            H, hd = cfg.n_heads, cfg.head_dim

            def qk(lin):
                z = lin(hn).view(B, Tq, H, hd)
                if cfg.qk_norm:
                    z = F.rms_norm(z, (hd,))
                return M.apply_rot(z, cos, sin)
            q, k = qk(blk.c_q), qk(blk.c_k)
            q2, k2 = qk(blk.c_q2), qk(blk.c_k2)
            v = blk.c_v(hn).view(B, Tq, H, hd)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / hd
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / hd
            pat = s1 * s2
            y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, Tq, cfg.width)
            aw = model.write_out(blk.c_proj(y), 2 * li, B, Tq)
            x = x + aw
            xn = model.slot_norm(x)
            mw = model.write_out(blk.Down(blk.Left(xn) * blk.Right(xn))
                                 + blk.Down_bias, 2 * li + 1, B, Tq)
            streams.append(aw)
            streams.append(mw)
        return entry(cfg.depth)

    def forward(self, x):
        r = F.rms_norm(self.R0[x], (self.Ws,))
        return 30 * torch.tanh((r @ self.WU.t()) / 30)

    def bits(self, b=BITS_PER_ENTRY):
        e = {'R0': int(self.R0.numel()), 'WU': int(self.WU.numel())}
        return {'entries': e, 'total_entries': int(sum(e.values())),
                'bits_per_entry': b, 'total_bits': int(sum(e.values()) * b),
                'dense_VxV_entries': int(self.V * self.V),
                'dense_VxV_bits': int(self.V * self.V * b)}


class UnigramProgram(nn.Module):
    def __init__(self, U):
        super().__init__()
        self.register_buffer('U', U)

    def forward(self, x):
        return self.U[None, None, :].expand(x.shape[0], x.shape[1], -1)


class DenseBigramProgram(nn.Module):
    """The OPTIMAL token-only program: a dense V x V table of log-probabilities
    whose row for token t is the est-average of the model's next-token
    distribution over positions carrying t.  For the forward KL that average is
    the exact minimiser, so this row is the CEILING of the whole bigram family
    -- everything left under it is irreducibly contextual.  Rare rows are backed
    off to the global average with a single strength alpha chosen on a slice of
    est that is not in the accumulation (never on held)."""

    def __init__(self, logq):
        super().__init__()
        self.register_buffer('logq', logq)

    def forward(self, x):
        return self.logq[x]

    def bits(self, b=BITS_PER_ENTRY):
        n = int(self.logq.numel())
        return {'entries': {'table_VxV': n}, 'total_entries': n,
                'bits_per_entry': b, 'total_bits': n * b}


@torch.no_grad()
def fit_dense_bigram(model, xs_est, T, batch=8, n_rows=8192,
                     alphas=(0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0),
                     n_val=128):
    V = model.cfg.vocab
    dev = xs_est.device
    S = torch.zeros(V, V, device=dev)
    cnt = torch.zeros(V, device=dev)
    rows = min(n_rows, xs_est.shape[0] - n_val)
    for a in range(0, rows, batch):
        x = xs_est[a:a + batch, :T]
        p = F.softmax(model_logits(model, x), -1).reshape(-1, V)
        f = x.reshape(-1)
        S.index_add_(0, f, p)
        cnt.index_add_(0, f, torch.ones_like(f, dtype=torch.float))
    glob = S.sum(0) / cnt.sum().clamp_min(1.0)
    xs_val = xs_est[xs_est.shape[0] - n_val:]
    best = None
    for al in alphas:
        logq = ((S + al * glob[None]) / (cnt + al)[:, None]).clamp_min(1e-12).log()
        s = score(DenseBigramProgram(logq), model, xs_val, T, batch)
        if best is None or s['kl_from_model'] < best[1]:
            best = (al, s['kl_from_model'])
        del logq
    al = best[0]
    logq = ((S + al * glob[None]) / (cnt + al)[:, None]).clamp_min(1e-12).log()
    del S
    return DenseBigramProgram(logq), {'alpha': al, 'alpha_val_kl': best[1],
                                      'est_rows_accumulated': int(rows),
                                      'est_rows_for_alpha': int(n_val),
                                      'alphas_tried': list(alphas)}


@torch.no_grad()
def quantised_copies(prog, bits_list=(16, 8)):
    """Post-hoc uniform scalar quantisation of every table (per-row scale, the
    tf_compress.q_scalar convention), so the bill can be cut 2x/4x and the KL
    cost of doing so is measured rather than assumed."""
    import copy
    import tf_compress as C
    out = {}
    for b in bits_list:
        q = copy.deepcopy(prog)
        tot = 0
        for n, p in q.named_parameters():
            W = p.data
            flat = W.reshape(W.shape[0], -1) if W.dim() > 1 else W[:, None]
            R, bt = C.q_scalar(flat, b)
            p.data.copy_(R.reshape(W.shape))
            tot += bt.total
        out[b] = (q, int(tot))
    return out


# ===========================================================================
# CONTROL (i): the exact fold must reproduce the model's layer-0 attention
# ===========================================================================
@torch.no_grad()
def control_exact_fold(model, x, device=DEV):
    """Instantiate the program's attention code with the model's OWN folded
    layer-0 tables (per-head query/key factor tables Q1,K1,Q2,K2, the per-head
    OV value table, the causal mask and the rotary) at FULL rank and with no
    fitted term, and require it to reproduce the model's layer-0 attention write
    to float precision.  This is a known-answer test of `Program.gate` /
    `Program.context` -- the same code the fitted program uses."""
    cfg = model.cfg
    H, hd, V, Ws = cfg.n_heads, cfg.head_dim, cfg.vocab, model.Ws
    T = x.shape[1]
    with M.exact_math():
        f = model.fold_layer0_qk(materialize=False, device=device)
        Wp0 = model.h[0].c_proj.weight.detach().float()
        ov = torch.stack([f['Vv'][h] @ Wp0[:, h * hd:(h + 1) * hd].t()
                          for h in range(H)])
        OV = model.write_out(ov, 0, ov.shape[0], ov.shape[1])   # (H,V,Ws)
        pc = PCfg(V=V, T=T, r=Ws, a=hd, nh=H, bigram=False, ctx='twobranch',
                  exact=True)
        p = Program(pc, device)
        with torch.no_grad():
            p.Kq.copy_(f['Q1'])
            p.Kk.copy_(f['K1'])
            p.Kq2.copy_(f['Q2'])
            p.Kk2.copy_(f['K2'])
            p.Vc.copy_(OV)
            p.g.fill_(1.0)
            p.cscale.fill_(1.0)
        got = p.context(x, exact_scale=1.0 / hd)
        col = {}
        model(x, collect=col)
        want = col['attn_write'][0].float()
        num = float((got - want).abs().max())
        den = max(float(want.abs().max()), 1e-30)
    return {'attn_write_max_abs_diff': num, 'attn_write_absmax': den,
            'attn_write_rel_diff': num / den,
            'tol_rel': 1e-5,
            'pass': bool(num / den < 1e-5),
            'what': 'the program attention code, given the exact folded '
                    'layer-0 tables at full rank, vs the model attention write'}


# ===========================================================================
# CONTROL (ii): shuffled tables must destroy the KL
# ===========================================================================
@torch.no_grad()
def control_shuffle(prog, model, xs, T, batch=8, seed=7):
    import copy
    q = copy.deepcopy(prog)
    g = torch.Generator(device='cpu').manual_seed(seed)
    V = prog.cfg.V
    perm = torch.randperm(V, generator=g).to(next(prog.parameters()).device)
    for n, p in q.named_parameters():
        if n in ('U',):
            continue
        if p.dim() == 2 and p.shape[0] == V:
            p.data.copy_(p.data[perm])
        elif p.dim() == 3 and p.shape[1] == V:
            p.data.copy_(p.data[:, perm])
        elif p.dim() == 1 and p.shape[0] == V:
            p.data.copy_(p.data[perm])
    return score(q, model, xs, T, batch)


# ===========================================================================
# the staged ladder
# ===========================================================================
def stage_specs(r, a, nh):
    """(name, PCfg-delta, description).  Cumulative: each stage is the previous
    one plus exactly one ingredient."""
    return [
        ('A_unigram', dict(), 'U[v] only'),
        ('B_bigram_rank', dict(bigram=True), '+ current-token content Pc[x_t]'),
        ('C_prefix_mean', dict(ctx='mean'), '+ prefix-mean context'),
        ('D_distance_profile', dict(ctx='delta'), '+ per-distance weights g[d]'),
        ('E_tokenpair_gate', dict(ctx='pair'), '+ rank-a token-pair gate'),
        ('F_rotary_gate', dict(ctx='pair_rot'), '+ rotary (distance) in the gate'),
        ('G_two_branch_gate', dict(ctx='twobranch'),
         '+ second gate branch: the exact shape of the folded pattern'),
        ('H_heads', dict(nh=nh), f'+ {nh} independent gates with own value tables'),
        ('I_induction', dict(induction=True), '+ induction/copy rule'),
        ('J_squared_content', dict(square=True),
         '+ squared-content term (BOUNDARY: closest thing to the MLP)'),
    ]


def warm_start(new: Program, old: Program):
    """Copy every table the previous stage had into the new one, so the stage
    starts at (or very near) the previous stage's function.  When a stage widens
    the head axis (1 -> nh) the old tables are copied into the FIRST head and the
    new heads keep their small random init, so the starting function is the
    previous stage plus a small perturbation rather than a fresh start."""
    if old is None:
        return new
    od = dict(old.named_parameters())
    with torch.no_grad():
        for n, p in new.named_parameters():
            if n not in od:
                continue
            q = od[n]
            if q.shape == p.shape:
                p.copy_(q)
            elif q.dim() == p.dim() and q.shape[1:] == p.shape[1:] \
                    and q.shape[0] < p.shape[0]:
                p[:q.shape[0]].copy_(q)


def run_cell(stem, args):
    t0 = time.time()
    model, cfg, ck = tf_fold.load_checkpoint(stem, DEV)
    V, T = cfg.vocab, args.T
    r = args.rank or {32: 16, 64: 32, 128: 64, 256: 128}.get(cfg.width, 32)
    a, nh = args.a, args.nh
    xs_est = load_x(V, 'est', args.n_est, T, cfg.tok)
    xs_held = load_x(V, 'held', args.n_held, T, cfg.tok)
    xs_held_ref = load_x(V, 'held', 96, T, cfg.tok)      # FINDING 23's setting
    xs_est_score = xs_est[:args.n_held]

    out = {'stem': stem, 'depth': cfg.depth, 'width': cfg.width,
           'vocab': V, 'tok': cfg.tok, 'Ws': model.Ws, 'n_heads': cfg.n_heads,
           'context_length': T, 'rank_r': r, 'gate_rank_a': a, 'n_gates': nh,
           'model_params_tied_once': int(model.n_params()),
           'model_bits_at_32': int(model.n_params() * BITS_PER_ENTRY),
           'splits': {
               'est_rows_fitted_on': int(xs_est.shape[0]),
               'est_tokens_available': int(xs_est.shape[0] * T),
               'held_rows_scored_on': int(xs_held.shape[0]),
               'held_tokens_scored': int(xs_held.shape[0] * T),
               'held_ref_rows': int(xs_held_ref.shape[0]),
               'held_ref_tokens': int(xs_held_ref.shape[0] * T),
               'note': 'nothing is fitted on held; est and held are disjoint '
                       'corpus splits (tf_corpus MANIFEST)'},
           'stages': {}, 'controls': {}, 'references': {}}

    # ---------------- control (i): the exact fold ----------------
    out['controls']['i_exact_fold'] = control_exact_fold(
        model, xs_held_ref[:4, :T], DEV)
    print(f'  [ctrl i] exact fold rel diff '
          f"{out['controls']['i_exact_fold']['attn_write_rel_diff']:.3e} "
          f"pass={out['controls']['i_exact_fold']['pass']}", flush=True)

    # ---------------- references ----------------
    bg = BigramTableProgram(model, DEV)
    sc = score(bg, model, xs_held, T, args.batch)
    sc_ref = score(bg, model, xs_held_ref, T, args.batch)
    out['references']['model_bigram'] = {
        **sc, 'kl_at_finding23_setting': sc_ref['kl_from_model'],
        'bits': bg.bits(),
        'what': "the model's own length-1 output table -- the baseline "
                'FINDING 23 measured'}
    out['references']['model_self'] = {'held_ce': sc['model_ce'],
                                       'held_ce_finding23_setting':
                                           sc_ref['model_ce']}
    print(f"  [ref] model_bigram KL {sc['kl_from_model']:.4f} "
          f"(FINDING-23 setting {sc_ref['kl_from_model']:.4f}), "
          f"{bg.bits()['total_entries']:,} entries", flush=True)
    bg_bits = bg.bits()
    del bg
    torch.cuda.empty_cache()
    dbg, dbg_meta = fit_dense_bigram(model, xs_est, T, args.batch,
                                     args.n_dense)
    sd = score(dbg, model, xs_held, T, args.batch)
    out['references']['dense_fitted_bigram'] = {
        **sd, **dbg_meta, 'bits': dbg.bits(),
        'what': 'the OPTIMAL token-only program (est-average of the model '
                'distribution per current token, backed off); the ceiling of '
                'the whole bigram family, and astronomically over budget'}
    print(f"  [ref] dense_fitted_bigram KL {sd['kl_from_model']:.4f} "
          f"alpha={dbg_meta['alpha']}, {dbg.bits()['total_entries']:,} entries",
          flush=True)
    del dbg
    torch.cuda.empty_cache()

    # ---------------- the ladder ----------------
    base = dict(V=V, T=T, r=r, a=a)
    acc = {'nh': 1}
    prev = None
    prev_name = None
    for name, delta, desc in stage_specs(r, a, nh):
        acc.update(delta)
        pc = PCfg(**base, **acc)
        p = Program(pc, DEV)
        if prev is None:
            init_unigram(p, model, xs_est, T, args.batch)
        else:
            warm_start(p, prev)
        steps = args.steps_first if prev is None else args.steps
        log = []
        fit(p, model, xs_est, T, steps, args.lr, args.batch, log=log)
        s_held = score(p, model, xs_held, T, args.batch)
        s_ref = score(p, model, xs_held_ref, T, args.batch)
        s_est = score(p, model, xs_est_score, T, args.batch)
        b = p.bits()
        rec = {'description': desc,
               'ingredient_added': list(delta.keys()) or ['(base)'],
               'kl_from_model_held': s_held['kl_from_model'],
               'kl_from_model_est': s_est['kl_from_model'],
               'kl_at_finding23_setting': s_ref['kl_from_model'],
               'held_minus_est_kl': s_held['kl_from_model']
               - s_est['kl_from_model'],
               'program_ce_held': s_held['program_ce'],
               'model_ce_held': s_held['model_ce'],
               'remainder_frac_of_model_ce':
                   s_held['kl_from_model'] / s_held['model_ce'],
               'bits': b,
               'bits_over_model': b['total_bits'] / (model.n_params()
                                                     * BITS_PER_ENTRY),
               'over_model_budget': bool(b['total_bits']
                                         > model.n_params() * BITS_PER_ENTRY),
               'bits_over_bigram_baseline': b['total_bits']
               / bg_bits['total_bits'],
               'beats_bigram_kl': bool(s_held['kl_from_model']
                                       < out['references']['model_bigram']
                                       ['kl_from_model']),
               'beats_bigram_at_or_below_its_size': bool(
                   s_held['kl_from_model']
                   < out['references']['model_bigram']['kl_from_model']
                   and b['total_bits'] <= bg_bits['total_bits']),
               'fit_steps': steps, 'fit_loss_curve': log}
        out['stages'][name] = rec
        print(f'  {name:<20s} KL {rec["kl_from_model_held"]:.4f} '
              f'(est {rec["kl_from_model_est"]:.4f})  '
              f'{b["total_entries"]:>10,} entries  '
              f'{rec["bits_over_model"]:.2f}x model  '
              f'{time.time() - t0:.0f}s', flush=True)
        prev, prev_name = p, name

    # ---------------- control (ii): shuffled tables ----------------
    sh = control_shuffle(prev, model, xs_held, T, args.batch)
    out['controls']['ii_shuffled_tables'] = {
        **sh, 'stage': prev_name,
        'kl_unshuffled': out['stages'][prev_name]['kl_from_model_held'],
        'ratio': sh['kl_from_model']
        / max(out['stages'][prev_name]['kl_from_model_held'], 1e-12),
        'pass': bool(sh['kl_from_model']
                     > 5 * out['stages'][prev_name]['kl_from_model_held']),
        'what': 'token axis of every fitted table permuted; the KL must blow up'}
    print(f"  [ctrl ii] shuffled KL {sh['kl_from_model']:.4f} "
          f"({out['controls']['ii_shuffled_tables']['ratio']:.1f}x)", flush=True)

    # ------------- the model under the SAME quantisation rules --------------
    # so "shorter than the model at a stated KL" is a fair comparison rather
    # than a program charged 8 bits against a model charged 32.
    import copy as _copy
    import tf_compress as _C
    out['references']['model_quantised'] = {}
    for b_ in (16, 8, 4):
        qm = _copy.deepcopy(model)
        tot = 0
        with torch.no_grad():
            for _n, _p in qm.named_parameters():
                Wf = _p.data.reshape(_p.shape[0], -1) if _p.dim() > 1 \
                    else _p.data[:, None]
                Rq, bt = _C.q_scalar(Wf, b_)
                _p.data.copy_(Rq.reshape(_p.shape))
                tot += bt.total
        s = score(qm, model, xs_held, T, args.batch)
        out['references']['model_quantised'][f'{b_}bit'] = {
            'total_bits': int(tot), 'kl_from_model_held': s['kl_from_model'],
            'bits_over_fp32_model': tot / (model.n_params() * 32)}
        print(f'  [ref] model at {b_} bits: KL {s["kl_from_model"]:.4f}, '
              f'{tot / 8 / 1e6:.2f} MB', flush=True)
        del qm
    torch.cuda.empty_cache()

    # ------------- cheaper bills for the same program: quantisation ---------
    out['quantised'] = {}
    for b_, (q, tot) in quantised_copies(prev, (16, 8, 4)).items():
        s = score(q, model, xs_held, T, args.batch)
        out['quantised'][f'{prev_name}_at_{b_}bit'] = {
            'bits_per_entry': b_, 'total_bits': tot,
            'kl_from_model_held': s['kl_from_model'],
            'bits_over_model': tot / (model.n_params() * BITS_PER_ENTRY),
            'over_model_budget': bool(tot > model.n_params() * BITS_PER_ENTRY),
            'kl_cost_vs_fp32': s['kl_from_model']
            - out['stages'][prev_name]['kl_from_model_held']}
        print(f'  [quant {b_}b] KL {s["kl_from_model"]:.4f}  '
              f'{tot / 8 / 1e6:.2f} MB', flush=True)
        del q

    # ------------- a bits-saving variant: tie the read and write tables -----
    pc = PCfg(V=V, T=T, r=r, a=a, nh=1, bigram=True, ctx='twobranch', tied=True)
    p = Program(pc, DEV)
    init_unigram(p, model, xs_est, T, args.batch)
    fit(p, model, xs_est, T, args.steps_first, args.lr, args.batch)
    s = score(p, model, xs_held, T, args.batch)
    b = p.bits()
    out['tied_variant'] = {
        'what': 'stage G with the current-token table TIED to the output table '
                '(the model ties its embedding); halves the content bill',
        'rank': r, 'kl_from_model_held': s['kl_from_model'], 'bits': b,
        'bits_over_model': b['total_bits'] / (model.n_params() * 32),
        'over_model_budget': bool(b['total_bits'] > model.n_params() * 32)}
    print(f'  [tied] KL {s["kl_from_model"]:.4f}  '
          f'{b["total_entries"]:,} entries', flush=True)
    del p

    # ---------------- rank frontier ----------------
    out['frontier'] = {}
    for tag, acc_cfg in (('bigram_only', dict(bigram=True)),
                         ('full_gate', dict(bigram=True, ctx='twobranch'))):
        for rr in args.ranks:
            pc = PCfg(V=V, T=T, r=rr, a=a, nh=1, **acc_cfg)
            p = Program(pc, DEV)
            init_unigram(p, model, xs_est, T, args.batch)
            fit(p, model, xs_est, T, args.steps_first, args.lr, args.batch)
            s = score(p, model, xs_held, T, args.batch)
            b = p.bits()
            out['frontier'][f'{tag}_r{rr}'] = {
                'rank': rr, 'family': tag,
                'kl_from_model_held': s['kl_from_model'],
                'total_entries': b['total_entries'],
                'total_bits': b['total_bits'],
                'bits_over_model': b['total_bits'] / (model.n_params() * 32),
                'over_model_budget': bool(b['total_bits']
                                          > model.n_params() * 32)}
            print(f'  frontier {tag:<12s} r={rr:<4d} KL '
                  f'{s["kl_from_model"]:.4f}  {b["total_entries"]:>10,} entries',
                  flush=True)
            del p
    out['seconds'] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', default='tf_vanilla_d1_w32_b8192_s0,'
                                       'tf_vanilla_d2_w128_b8192_s0,'
                                       'tf_vanilla_d4_w256_b8192_s0')
    ap.add_argument('--out', default='tf_rung5_program.json')
    ap.add_argument('--T', type=int, default=256)
    ap.add_argument('--n-est', type=int, default=16384)
    ap.add_argument('--n-held', type=int, default=512)
    ap.add_argument('--n-dense', type=int, default=8192)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--steps-first', type=int, default=3000)
    ap.add_argument('--lr', type=float, default=0.02)
    ap.add_argument('--rank', type=int, default=0)
    ap.add_argument('--a', type=int, default=8)
    ap.add_argument('--nh', type=int, default=4)
    ap.add_argument('--ranks', default='8,16,32,64,128')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    args.ranks = [int(v) for v in args.ranks.split(',') if v]
    if args.smoke:
        args.cells = 'tf_vanilla_d1_w32_b8192_s0'
        args.n_est, args.n_held, args.n_dense = 512, 32, 256
        args.steps, args.steps_first = 40, 60
        args.ranks = [8]
        args.out = 'tf_rung5_program_smoke.json'
    rep = {'created': time.strftime('%Y-%m-%d %H:%M:%S'),
           'what': 'rung 5: a weights-free program, staged, with bits and '
                   'controls',
           'grammar': 'tables indexed by token / token-pair / position / '
                      'distance, plus arithmetic; no latent-to-latent matrix, '
                      'no call into the model',
           'bit_convention': 'every table entry charged 32 bits; the model '
                             'charged 32 bits per parameter with the tied '
                             'embedding counted once (tf_compress.py rules)',
           'args': {k: v for k, v in vars(args).items()},
           'cells': {}}
    for stem in args.cells.split(','):
        stem = stem.strip()
        if not stem:
            continue
        print(f'=== {stem}', flush=True)
        rep['cells'][stem] = run_cell(stem, args)
        json.dump(rep, open(f'{HERE}/{args.out}', 'w'), indent=1)
    json.dump(rep, open(f'{HERE}/{args.out}', 'w'), indent=1)
    print('wrote', args.out, flush=True)


if __name__ == '__main__':
    main()
