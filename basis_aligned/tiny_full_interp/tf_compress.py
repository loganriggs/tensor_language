"""RUNG 5, COMPRESSION FRONTIER.  Description length in BITS vs KL from the
true model, for descriptions that are real evaluable objects (a fixed decoder
program plus tables), scored on HELD text with every table fitted on the
ESTIMATION split.

WHY THIS FILE EXISTS.  The rung-5 ladder in `tf_interp` produced exactly one
weights-free artifact that reproduces the model at all -- the model's own
token-pair table -- and it is 8192 x 8192 = 67.1M entries against a 1.34M
parameter model, i.e. the "explanation" is 50x LARGER than the thing explained,
at KL 0.649.  That is a failed rung.  This file replaces the argument with a
PLOT: bits on x, KL on y, the model's own description length marked, so
"does any description beat the model itself?" is a measurement.

THE ACCOUNTING RULES (non-negotiable, and stated for every point)
  * A description is a bit string that a FIXED decoder (this file) turns into a
    next-token predictor.  Everything the decoder needs that is not in this
    file's source is charged: tables, codebooks, indices, scales, all of it.
  * fp32 = 32 bits, fp16 = 16 bits, a b-bit quantised weight = b bits, an index
    into k things = ceil(log2 k) bits.  No entropy coding is assumed unless it
    is named, in which case the code table is charged too.
  * The MODEL ITSELF is a description under the same rules, so it gets a curve,
    not a point: its weights quantised to b bits is a legitimate description
    and it is the reference every structural scheme has to beat.
  * Tables are fitted on `est`, scored on `held`.  Nothing is fitted on held.

THE ARCHITECTURE THIS EXPLOITS (depth-1 vanilla, the fully-interpreted cell).
The forward is
    e     = rms(Wte[idx])                       READ role of the embedding
    pat   = (rot(rms(Wq e)).rot(rms(Wk e)))(rot(rms(Wq2 e)).rot(rms(Wk2 e)))/hd^2
    x     = e + Wproj (pat @ (Wv e))
    xn    = rms(x)
    x     = x + Down(Left(xn) * Right(xn)) + bias
    logit = 30 tanh(rms(x) @ Wte^T / 30)        WRITE role of the embedding
so the embedding appears in TWO roles and a description may compress them
separately (paying for both tables).  That separation is what turns Logan's
memorisation/structure question into a measurement: the READ role is the
token's *input signature* (how it steers the computation) and the WRITE role is
the token's *identity as an answer*.

SIGN IS A GAUGE FREEDOM (standing rule) -- nothing here is named from a sign.
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


def rms(x, d):
    return F.rms_norm(x, (d,))


# ===========================================================================
# BIT ACCOUNTING -- one place, so no scheme can quietly under-charge
# ===========================================================================
def bits_dense(n, b):
    """n numbers at b bits each."""
    return n * b


def bits_index(n, k):
    """n indices into an alphabet of k symbols, flat code."""
    return 0 if k <= 1 else n * math.ceil(math.log2(k))


class Bits:
    """An itemised bit bill.  Every scheme returns one of these so the total
    can always be broken down and audited."""

    def __init__(self, **items):
        self.items = dict(items)

    def add(self, **items):
        self.items.update(items)
        return self

    def merge(self, other, prefix=''):
        for k, v in other.items.items():
            self.items[prefix + k] = v
        return self

    @property
    def total(self):
        return int(sum(self.items.values()))

    def to_json(self):
        d = {k: int(v) for k, v in self.items.items()}
        d['TOTAL'] = self.total
        return d


# ===========================================================================
# QUANTISERS / CODERS.  Each returns (reconstruction, Bits).
# ===========================================================================
@torch.no_grad()
def q_scalar(W, b, group='row'):
    """Uniform scalar quantisation to b bits with a per-row (or per-tensor)
    fp16 scale and zero-point.  b=32 is the identity and is charged 32 bits."""
    if b >= 32:
        return W.clone(), Bits(values=bits_dense(W.numel(), 32))
    W2 = W.reshape(W.shape[0], -1) if group == 'row' else W.reshape(1, -1)
    lo = W2.min(dim=1, keepdim=True).values.half().float()
    hi = W2.max(dim=1, keepdim=True).values.half().float()
    step = ((hi - lo) / (2 ** b - 1)).clamp_min(1e-30)
    q = ((W2 - lo) / step).round().clamp(0, 2 ** b - 1)
    R = (q * step + lo).reshape(W.shape)
    return R, Bits(values=bits_dense(W.numel(), b),
                   scales=2 * W2.shape[0] * 16)


@torch.no_grad()
def q_lowrank(W, r, b=32):
    """Rank-r truncated SVD, factors stored at b bits."""
    U, S, Vh = torch.linalg.svd(W.double(), full_matrices=False)
    A = (U[:, :r] * S[:r]).float()
    B = Vh[:r].float()
    bt = Bits()
    if b >= 32:
        R = A @ B
        bt.add(factorA=bits_dense(A.numel(), 32), factorB=bits_dense(B.numel(), 32))
    else:
        Aq, ba = q_scalar(A, b)
        Bq, bb = q_scalar(B, b)
        R = Aq @ Bq
        bt.merge(ba, 'A_').merge(bb, 'B_')
    return R, bt


@torch.no_grad()
def kmeans(X, k, iters=40, seed=0):
    """Plain Lloyd k-means (k-means++ init) on rows of X.  Deterministic."""
    n, d = X.shape
    g = torch.Generator(device='cpu').manual_seed(seed)
    if k >= n:
        return X.clone(), torch.arange(n, device=X.device)
    if k <= 512:
        # k-means++
        idx0 = int(torch.randint(n, (1,), generator=g).item())
        C = X[idx0:idx0 + 1].clone()
        d2 = ((X - C[0]) ** 2).sum(1)
        for _ in range(k - 1):
            p = (d2 / d2.sum().clamp_min(1e-30)).cpu()
            j = int(torch.multinomial(p, 1, generator=g).item())
            C = torch.cat([C, X[j:j + 1]], 0)
            d2 = torch.minimum(d2, ((X - X[j]) ** 2).sum(1))
    else:  # k-means++ is O(k) python steps; at large k use distinct random rows
        perm = torch.randperm(n, generator=g)[:k].to(X.device)
        C = X[perm].clone()
    for _ in range(iters):
        # chunked assignment
        assign = torch.empty(n, dtype=torch.long, device=X.device)
        for a in range(0, n, 4096):
            xb = X[a:a + 4096]
            dist = (xb * xb).sum(1, keepdim=True) - 2 * xb @ C.t() + (C * C).sum(1)
            assign[a:a + 4096] = dist.argmin(1)
        Cn = torch.zeros_like(C)
        cnt = torch.zeros(k, device=X.device)
        Cn.index_add_(0, assign, X)
        cnt.index_add_(0, assign, torch.ones(n, device=X.device))
        empty = cnt == 0
        Cn[~empty] = Cn[~empty] / cnt[~empty][:, None]
        Cn[empty] = C[empty]
        if torch.allclose(Cn, C, atol=1e-7):
            C = Cn
            break
        C = Cn
    assign = torch.empty(n, dtype=torch.long, device=X.device)
    for a in range(0, n, 4096):
        xb = X[a:a + 4096]
        dist = (xb * xb).sum(1, keepdim=True) - 2 * xb @ C.t() + (C * C).sum(1)
        assign[a:a + 4096] = dist.argmin(1)
    return C, assign


@torch.no_grad()
def q_cluster(W, k, b=32, weights=None, seed=0):
    """Row clustering: k centroids + one index per row.  `weights` (a per-row
    importance, e.g. token frequency) is applied by REPLICATION-FREE weighted
    Lloyd -- implemented as weighted centroid updates."""
    C, a = kmeans_weighted(W, k, weights=weights, seed=seed)
    bt = Bits(index=bits_index(W.shape[0], k))
    if b >= 32:
        bt.add(centroids=bits_dense(C.numel(), 32))
        return C[a], bt
    Cq, bc = q_scalar(C, b)
    bt.merge(bc, 'cent_')
    return Cq[a], bt


@torch.no_grad()
def kmeans_weighted(X, k, weights=None, iters=40, seed=0):
    if weights is None:
        return kmeans(X, k, iters, seed)
    n, d = X.shape
    if k >= n:
        return X.clone(), torch.arange(n, device=X.device)
    w = weights.clamp_min(0).float()
    C, a = kmeans(X, k, iters=8, seed=seed)
    for _ in range(iters):
        Cn = torch.zeros_like(C)
        cnt = torch.zeros(k, device=X.device)
        Cn.index_add_(0, a, X * w[:, None])
        cnt.index_add_(0, a, w)
        empty = cnt <= 0
        Cn[~empty] = Cn[~empty] / cnt[~empty][:, None]
        Cn[empty] = C[empty]
        C = Cn
        for s in range(0, n, 4096):
            xb = X[s:s + 4096]
            dist = (xb * xb).sum(1, keepdim=True) - 2 * xb @ C.t() + (C * C).sum(1)
            a[s:s + 4096] = dist.argmin(1)
    return C, a


@torch.no_grad()
def q_pq(W, m, nbits, seed=0, iters=25):
    """Product quantisation: split the d columns into m contiguous subspaces,
    k-means each with 2**nbits codewords.  Charged: codebooks at fp32 + m
    indices of nbits per row."""
    n, d = W.shape
    assert d % m == 0
    k = 2 ** nbits
    sub = d // m
    out = torch.empty_like(W)
    cb_numel = 0
    for j in range(m):
        Xj = W[:, j * sub:(j + 1) * sub].contiguous()
        C, a = kmeans(Xj, k, iters=iters, seed=seed + j)
        out[:, j * sub:(j + 1) * sub] = C[a]
        cb_numel += C.numel()
    return out, Bits(codebooks=bits_dense(cb_numel, 32),
                     indices=n * m * nbits)


# ===========================================================================
# THE DEPTH-1 VANILLA DECODER, with every part swappable
# ===========================================================================
class D1Desc:
    """The description decoder.  `parts` is the complete list of tables it
    consumes; every one of them is charged for."""

    PART_NAMES = ('wte_read', 'wte_out', 'Wq', 'Wk', 'Wq2', 'Wk2', 'Wv',
                  'Wproj', 'Left', 'Right', 'Down', 'Down_bias')

    def __init__(self, stem, device=DEV):
        model, cfg, ck = tf_fold.load_checkpoint(stem, device)
        assert cfg.depth == 1 and cfg.n_slots == 1 and not cfg.predicate \
            and not cfg.codebook and not cfg.small_dec and not cfg.shrink_mode
        self.stem, self.model, self.cfg, self.dev = stem, model, cfg, device
        self.V, self.H, self.hd = cfg.vocab, cfg.n_heads, cfg.head_dim
        self.Ws, self.Dc = model.Ws, model.Dc
        b = model.h[0]
        self.base = {
            'wte_read': model.wte.weight.detach().float().clone(),
            'wte_out': model.wte.weight.detach().float().clone(),
            'Wq': b.c_q.weight.detach().float().clone(),
            'Wk': b.c_k.weight.detach().float().clone(),
            'Wq2': b.c_q2.weight.detach().float().clone(),
            'Wk2': b.c_k2.weight.detach().float().clone(),
            'Wv': b.c_v.weight.detach().float().clone(),
            'Wproj': b.c_proj.weight.detach().float().clone(),
            'Left': b.Left.weight.detach().float().clone(),
            'Right': b.Right.weight.detach().float().clone(),
            'Down': b.Down.weight.detach().float().clone(),
            'Down_bias': b.Down_bias.detach().float().clone(),
        }
        self.cos, self.sin = model.cos.to(device), model.sin.to(device)
        self.mask = model.mask
        # nominal model description length: the TIED table counted ONCE
        self.n_params_embed = self.base['wte_read'].numel()
        self.n_params_body = sum(self.base[k].numel() for k in self.PART_NAMES
                                 if k not in ('wte_read', 'wte_out'))
        self.n_params_model = self.n_params_embed + self.n_params_body

    # ------------------------------------------------------------- forward
    def forward(self, idx, P=None, e_override=None):
        P = P or {}
        g = lambda k: P.get(k, self.base[k])
        B, Tq = idx.shape
        Ws, H, hd = self.Ws, self.H, self.hd
        cos = self.cos[None, :Tq, None, :]
        sin = self.sin[None, :Tq, None, :]
        mask = self.mask[:Tq, :Tq]
        if e_override is not None:
            e = e_override
        else:
            e = rms(g('wte_read')[idx], Ws)

        def qk(W):
            z = (e @ W.t()).view(B, Tq, H, hd)
            return M.apply_rot(rms(z, hd), cos, sin)

        s1 = torch.einsum('bqhd,bkhd->bhqk', qk(g('Wq')), qk(g('Wk'))) / hd
        s2 = torch.einsum('bqhd,bkhd->bhqk', qk(g('Wq2')), qk(g('Wk2'))) / hd
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        v = (e @ g('Wv').t()).view(B, Tq, H, hd)
        y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, Tq, self.Dc)
        x = e + y @ g('Wproj').t()
        xn = rms(x, Ws)
        h = (xn @ g('Left').t()) * (xn @ g('Right').t())
        x = x + h @ g('Down').t() + g('Down_bias')
        lg = rms(x, Ws) @ g('wte_out').t()
        return 30 * torch.tanh(lg / 30)

    # ------------------------------------------------------------- scoring
    def batches(self, split='held', n_seq=64, T=256, batch=8):
        arr = tf_corpus.load_split(self.V, split, n_seq, tok=self.cfg.tok)
        x = torch.from_numpy(arr[:, :T + 1]).to(self.dev)
        for a in range(0, x.shape[0], batch):
            bb = x[a:a + batch]
            yield bb[:, :-1], bb[:, 1:]

    @torch.no_grad()
    def cache_ref(self, split='held', n_seq=64, T=256, batch=8):
        key = (split, n_seq, T, batch)
        if getattr(self, '_refkey', None) == key:
            return self._ref
        ref = []
        for x, y in self.batches(split, n_seq, T, batch):
            lp = F.log_softmax(self.model(x).float(), -1)
            ref.append((x, y, lp.half(), lp.exp().half()))
        self._refkey, self._ref = key, ref
        return ref

    @torch.no_grad()
    def score(self, P=None, split='held', n_seq=64, T=256, batch=8):
        """KL(model || description) and CE, nats/token, on held text."""
        kl, ce, n = 0.0, 0.0, 0
        for x, y, lpr, pr in self.cache_ref(split, n_seq, T, batch):
            lg = self.forward(x, P)
            lp = F.log_softmax(lg.float(), -1)
            kl += float((pr.float() * (lpr.float() - lp)).sum())
            ce += float(F.cross_entropy(lp.reshape(-1, self.V), y.reshape(-1),
                                        reduction='sum'))
            n += y.numel()
        return {'kl': kl / n, 'ce': ce / n, 'ntok': n}

    @torch.no_grad()
    def score_se(self, P=None, split='held', n_seq=64, T=256, batch=8):
        """Same KL, plus a SEQUENCE-CLUSTERED standard error: sequences are the
        independent unit, tokens inside one are not."""
        per = []
        for x, y, lpr, pr in self.cache_ref(split, n_seq, T, batch):
            lg = self.forward(x, P)
            lp = F.log_softmax(lg.float(), -1)
            k = (pr.float() * (lpr.float() - lp)).sum(-1).mean(-1)   # (B,)
            per.append(k)
        per = torch.cat(per)
        return {'kl': float(per.mean()),
                'kl_se': float(per.std(unbiased=True) / math.sqrt(len(per))),
                'n_seq': int(len(per))}

    # ------------------------------------------------------- positive control
    @torch.no_grad()
    def self_check(self):
        out = {}
        with M.exact_math():
            x = next(iter(self.batches('held', 8, 128, 8)))[0]
            a = self.forward(x)
            b = self.model(x)
            out['rel_logit_diff'] = float((a - b).abs().max()
                                          / b.abs().max())
        s = self.score()
        out['kl_identity'] = s['kl']
        out['ce_identity'] = s['ce']
        return out


# ===========================================================================
# TRANSFORM CODING AND ENTROPY CODING
# The program's standing lesson is that the neuron/coordinate basis is a GAUGE.
# A scalar quantiser is basis-dependent, so "is there a better basis to spend
# bits in?" is the right structural question to ask of the embedding.
# ===========================================================================
def hadamard(n, device):
    """Walsh-Hadamard matrix / sqrt(n) -- orthogonal, and a CONSTANT, so a
    description that uses it pays ZERO bits for the rotation."""
    assert n & (n - 1) == 0
    H = torch.ones(1, 1, device=device)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)


def entropy_bits(codes, nlev):
    """Cost of arithmetic-coding integer `codes` under their own histogram,
    INCLUDING the histogram (charged at fp16 per symbol).  This is an honest
    code length: a decoder given the histogram can decode the stream."""
    c = torch.bincount(codes.reshape(-1).long(), minlength=nlev).float()
    p = c / c.sum()
    H = float(-(p[p > 0] * p[p > 0].log2()).sum())
    return int(math.ceil(codes.numel() * H)) + nlev * 16


def _alloc(var, total_bits, bmax=12):
    """Reverse water-filling bit allocation over columns with variances `var`,
    integer bits, summing to <= total_bits per row."""
    d = len(var)
    lo, hi = 1e-30, float(var.max()) * 4 + 1e-9
    for _ in range(80):
        th = math.sqrt(lo * hi)
        b = torch.clamp(0.5 * torch.log2(var / th), 0, bmax).round()
        if b.sum() > total_bits:
            lo = th
        else:
            hi = th
    return torch.clamp(0.5 * torch.log2(var / hi), 0, bmax).round().long()


@torch.no_grad()
def q_transform(W, bits_per_row, rot='pca', entropy=True, wrow=None):
    """Transform code: rotate rows, allocate integer bits per column by
    reverse water-filling, uniform-quantise each column, optionally arithmetic-
    code the symbols.  `wrow` is an optional per-row importance used only to
    weight the variance estimate (never to fit on held text)."""
    V, d = W.shape
    mu = W.mean(0, keepdim=True)
    X = W - mu
    bt = Bits(mean=bits_dense(d, 32))
    if rot == 'pca':
        cov = (X.t() @ X) / V
        ev, Q = torch.linalg.eigh(cov.double())
        Q = Q.float().flip(1)
        Qq, bq = q_scalar(Q, 8)
        Qq, _ = torch.linalg.qr(Qq.double())       # re-orthogonalise the decoded
        Qq = Qq.float()
        bt.merge(bq, 'rot_')
    elif rot == 'hadamard':
        Qq = hadamard(d, W.device)
        bt.add(rot=0)
    else:
        Qq = torch.eye(d, device=W.device)
        bt.add(rot=0)
    Z = X @ Qq
    w = torch.ones(V, device=W.device) if wrow is None else wrow.clamp_min(1e-8)
    w = w / w.sum()
    var = (Z * Z * w[:, None]).sum(0)
    b = _alloc(var, bits_per_row)
    lo = Z.min(0).values.half().float()
    hi = Z.max(0).values.half().float()
    Zq = torch.zeros_like(Z)
    tot = 0
    for j in range(d):
        bj = int(b[j])
        if bj == 0:
            continue
        step = ((hi[j] - lo[j]) / (2 ** bj - 1)).clamp_min(1e-30)
        c = ((Z[:, j] - lo[j]) / step).round().clamp(0, 2 ** bj - 1)
        Zq[:, j] = c * step + lo[j]
        tot += entropy_bits(c, 2 ** bj) if entropy else V * bj
    bt.add(codes=tot, colscales=2 * d * 16, alloc=d * 4)
    return Zq @ Qq.t() + mu, bt


@torch.no_grad()
def q_scalar_entropy(W, b, group='row'):
    """Per-row-scale uniform quantisation, symbols arithmetic-coded."""
    if b >= 32:
        return W.clone(), Bits(values=bits_dense(W.numel(), 32))
    W2 = W.reshape(W.shape[0], -1) if group == 'row' else W.reshape(1, -1)
    lo = W2.min(dim=1, keepdim=True).values.half().float()
    hi = W2.max(dim=1, keepdim=True).values.half().float()
    step = ((hi - lo) / (2 ** b - 1)).clamp_min(1e-30)
    q = ((W2 - lo) / step).round().clamp(0, 2 ** b - 1)
    R = (q * step + lo).reshape(W.shape)
    return R, Bits(values=entropy_bits(q, 2 ** b), scales=2 * W2.shape[0] * 16)


@torch.no_grad()
def q_stratified(W, b_hi, b_lo, n_hi, order, entropy=True):
    """Frequency-stratified precision: the top `n_hi` rows by `order` get b_hi
    bits, the rest b_lo.  The smooth version of the anchor-row hybrid."""
    V = W.shape[0]
    hi_idx = order[:n_hi]
    mask = torch.ones(V, dtype=torch.bool, device=W.device)
    mask[hi_idx] = False
    lo_idx = torch.nonzero(mask).squeeze(1)
    R = W.clone()
    f = q_scalar_entropy if entropy else q_scalar
    A, ba = f(W[hi_idx], b_hi)
    B, bb = f(W[lo_idx], b_lo)
    R[hi_idx] = A
    R[lo_idx] = B
    bt = Bits(strata_ids=bits_index(n_hi, V)).merge(ba, 'hi_').merge(bb, 'lo_')
    return R, bt
