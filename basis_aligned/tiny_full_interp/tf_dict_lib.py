"""PORT OF THE PARENT PROGRAM'S LAYER-0 MDL METHOD (../qk_mdl, RESULTS_l0_mdl.md
sections 3/3b/3c + ov_metric_explainer.md) TO THE DEPTH-1 WIDTH-128 CELL.

WHAT IS BEING PORTED, AND WHY IT IS NOT JUST "SPARSE CODING"
-----------------------------------------------------------
The parent program compressed the EXACT LAYER-0 FOLD -- per head and branch the
(V x head_dim) query/key factor tables `Qb(t), Kb(t)` that the token-pair score
map factors through -- and reached +0.006 nats at 6.1% of the raw factor bits
with a 1024-atom, 8-active dictionary.  Three ingredients made that work and all
three are ported here:

  1. THE OBJECT is the folded per-head-branch factor table, not the embedding.
  2. THE OBJECTIVE is not reconstruction MSE.  It is the CONTEXT-EXPECTED OV
     error, eq. (dagger) of ov_metric_explainer.md:
         E||e_i||^2 = T (E_q||c||^2 - ||mu||^2) + T^2 ||mu||^2,
         c_j = (pattern error at key j) x (what attending to j writes),
     i.e. the scatter part of a pattern error accumulates as T and never
     cancels, the systematic part accumulates as T^2 and does.  Here that
     expectation is worked out IN CLOSED FORM as a per-token, per-block
     positive-semidefinite metric on the folded rows (see `ctx_metrics`), so
     the dictionary is learned and the codes are chosen in the geometry the
     model's own OV circuit imposes, with unigram exposure `q` folded in.
  3. ANCHORS: exact rows for the top-B tokens by attribution, bits charged,
     dictionary for the tail (parent section 3c: 1.8-2.9x at matched bits).

SIGN IS A GAUGE FREEDOM (standing rule): nothing here is named from the sign of
any factor; every claim is a held-out cross-entropy or a KL.

BIT ACCOUNTING is inherited from tf_compress.Bits so that no scheme can quietly
under-charge, and follows the PARENT'S convention exactly (verified against the
parent's own 455 Mbit figure for n=1024,k=8,V=50304,d=256,18 head-branches):
atoms fp32, coefficients fp32, one index of ceil(log2 n) bits per active atom.
fp16 variants are reported as a clearly-labelled secondary.
"""
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import tf_compress as CC
import tf_corpus
import tf_fold
import tf_model as M
from tf_compress import Bits, bits_dense, bits_index

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def log(*a):
    print(f'[{time.strftime("%H:%M:%S")}]', *a, flush=True)


def rms(x, d):
    return F.rms_norm(x, (d,))


# ===========================================================================
# THE OBJECT: the exact layer-0 fold, plus its OV reader
# ===========================================================================
class FoldDesc(CC.D1Desc):
    """D1Desc + the exact folded query/key factor tables and a forward that
    consumes them, so a description may replace the fold instead of the
    weights that generate it."""

    def __init__(self, stem, device=DEV, T_ctx=256):
        super().__init__(stem, device)
        self.T_ctx = T_ctx
        f = self.model.fold_layer0_qk(deltas=(0,), materialize=False,
                                      device=device)
        self.FT = {k: f[k].contiguous() for k in ('Q1', 'K1', 'Q2', 'K2')}
        self.Vv = f['Vv'].contiguous()                       # (H,V,hd)
        Wp = self.base['Wproj']                              # (Ws, Ws)
        hd, H = self.hd, self.H
        self.Wp_h = torch.stack([Wp[:, h * hd:(h + 1) * hd] for h in range(H)])
        # u[h,t] = what attending to token t at head h writes into the residual
        self.U = torch.einsum('hdc,hvc->hvd', self.Wp_h, self.Vv)   # (H,V,Ws)

    # ------------------------------------------------------------- forward
    def forward_fold(self, idx, FT=None, P=None):
        """Identical to D1Desc.forward except that the four query/key factor
        tables are LOOKED UP (the description's tables) instead of computed
        from the embedding through Wq/Wk/Wq2/Wk2."""
        FT = FT or self.FT
        P = P or {}
        g = lambda k: P.get(k, self.base[k])
        B, Tq = idx.shape
        Ws, H, hd = self.Ws, self.H, self.hd
        cos = self.cos[None, :Tq, None, :]
        sin = self.sin[None, :Tq, None, :]
        mask = self.mask[:Tq, :Tq]
        e = rms(g('wte_read')[idx], Ws)

        def look(name):
            z = FT[name][:, idx.reshape(-1)]                 # (H, B*T, hd)
            z = z.reshape(H, B, Tq, hd).permute(1, 2, 0, 3)  # (B,T,H,hd)
            return M.apply_rot(z, cos, sin)

        s1 = torch.einsum('bqhd,bkhd->bhqk', look('Q1'), look('K1')) / hd
        s2 = torch.einsum('bqhd,bkhd->bhqk', look('Q2'), look('K2')) / hd
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        v = (e @ g('Wv').t()).view(B, Tq, H, hd)
        y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, Tq, self.Dc)
        x = e + y @ g('Wproj').t()
        xn = rms(x, Ws)
        h = (xn @ g('Left').t()) * (xn @ g('Right').t())
        x = x + h @ g('Down').t() + g('Down_bias')
        return 30 * torch.tanh(rms(x, Ws) @ g('wte_out').t() / 30)

    # ------------------------------------------------------------- scoring
    @torch.no_grad()
    def eval_seqs(self, split='held', n_seq=256, T=256):
        arr = tf_corpus.load_split(self.V, split, n_seq, tok=self.cfg.tok)
        return torch.from_numpy(arr[:, :T + 1]).to(self.dev)

    @torch.no_grad()
    def score_fwd(self, fwd, split='held', n_seq=256, T=256, batch=16):
        """held CE (PRIMARY, against the DATA) and KL from the model
        (secondary), both nats/token, with sequence-clustered standard errors.
        The model reference is recomputed per batch (no big logit cache)."""
        x = self.eval_seqs(split, n_seq, T)
        ce_s, kl_s, cem_s = [], [], []
        for a in range(0, x.shape[0], batch):
            bb = x[a:a + batch]
            xi, y = bb[:, :-1], bb[:, 1:]
            lp = F.log_softmax(fwd(xi).float(), -1)
            with torch.no_grad():
                lpr = F.log_softmax(self.model(xi).float(), -1)
            pr = lpr.exp()
            ce_s.append(F.cross_entropy(lp.reshape(-1, self.V), y.reshape(-1),
                                        reduction='none').view(y.shape).mean(-1))
            cem_s.append(F.cross_entropy(lpr.reshape(-1, self.V),
                                         y.reshape(-1), reduction='none'
                                         ).view(y.shape).mean(-1))
            kl_s.append((pr * (lpr - lp)).sum(-1).mean(-1))
        ce, kl, cem = torch.cat(ce_s), torch.cat(kl_s), torch.cat(cem_s)
        n = len(ce)
        dce = ce - cem            # PAIRED vs the model, sequence by sequence
        return {'ce': float(ce.mean()),
                'ce_se': float(ce.std(unbiased=True) / math.sqrt(n)),
                'dce_vs_model': float(dce.mean()),
                'dce_se': float(dce.std(unbiased=True) / math.sqrt(n)),
                'ce_model': float(cem.mean()),
                'kl': float(kl.mean()),
                'kl_se': float(kl.std(unbiased=True) / math.sqrt(n)),
                'n_seq': n, 'ntok': int(n * T)}

    @torch.no_grad()
    def score_fold(self, FT, **kw):
        return self.score_fwd(lambda xi: self.forward_fold(xi, FT), **kw)

    @torch.no_grad()
    def score_emb(self, wte, **kw):
        P = {'wte_read': wte, 'wte_out': wte}
        return self.score_fwd(lambda xi: self.forward(xi, P), **kw)

    # ------------------------------------------------------- positive control
    @torch.no_grad()
    def gate(self, **kw):
        out = {}
        with M.exact_math():
            xi = self.eval_seqs('held', 8, 128)[:, :-1]
            a, b = self.forward_fold(xi), self.model(xi)
            out['fold_rel_logit_diff'] = float((a - b).abs().max()
                                               / b.abs().max())
        out['fold_identity'] = self.score_fold(self.FT, **kw)
        out['weights_identity'] = self.score_fwd(self.forward, **kw)
        return out


# ===========================================================================
# THE ROWS TO CODE, AND THEIR WAY BACK INTO A FORWARD PASS
# ===========================================================================
def build_X(D):
    """(V, 4H, hd) folded factor rows in the fixed block layout."""
    blocks = []
    for h in range(D.H):
        for b in (1, 2):
            blocks.append(D.FT[f'Q{b}'][h])
            blocks.append(D.FT[f'K{b}'][h])
    return torch.stack(blocks, 1).contiguous()


def X_to_FT(D, X, renorm=True):
    """Inverse of build_X.  `renorm` re-applies the hd-RMS normalisation that
    every true folded row satisfies by construction -- it costs ZERO bits (the
    decoder knows the fold is RMS-normed) and can only project error onto the
    sphere the truth lives on."""
    if renorm:
        X = rms(X, X.shape[-1])
    FT = {}
    for h in range(D.H):
        for bi, b in enumerate((1, 2)):
            FT.setdefault(f'Q{b}', []).append(X[:, 4 * h + 2 * bi])
            FT.setdefault(f'K{b}', []).append(X[:, 4 * h + 2 * bi + 1])
    return {k: torch.stack(v, 0).contiguous() for k, v in FT.items()}


def groups(mode, H):
    """Column groups that share a dictionary.  'perhb' = the parent's row
    cat(q_hat, k_hat) for one head-branch (2 blocks); 'joint' = one dictionary
    over the token's whole folded signature (4H blocks)."""
    if mode == 'joint':
        return [list(range(4 * H))]
    if mode == 'perhb':
        return [[4 * h + 2 * b, 4 * h + 2 * b + 1]
                for h in range(H) for b in (0, 1)]
    if mode == 'perblock':
        return [[i] for i in range(4 * H)]
    raise ValueError(mode)


# ===========================================================================
# UNIGRAM EXPOSURE q, ESTIMATION SPLIT ONLY
# ===========================================================================
def unigram_q(D, n_seq=8000):
    est = tf_corpus.load_split(D.V, 'est', n_seq, tok=D.cfg.tok)
    c = np.bincount(est.reshape(-1), minlength=D.V).astype(np.float64)
    q = torch.tensor(c / c.sum(), device=D.dev, dtype=torch.float32)
    return q, torch.tensor(c, device=D.dev, dtype=torch.float32)


def delta_buckets(T=256, nb=None):
    """Representative rotary offsets with pair-count weights: within a causal
    window of length T the offset delta = i - j occurs (T - delta) times.  This
    is the INCOHERENT form the parent found wins (tick 163): every band is kept
    and the quadratic functionals are averaged, rather than averaging the
    rotations themselves (which washes out 98.8% of the signal)."""
    edges = [0, 1, 2, 4, 8, 16, 32, 64, 128, T]
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        ds = np.arange(a, min(b, T))
        w = float((T - ds).sum())
        if w <= 0:
            continue
        rep = int(round(float((ds * (T - ds)).sum() / (T - ds).sum())))
        out.append((rep, w))
    tot = sum(w for _, w in out)
    out = [(d, w / tot) for d, w in out]
    return out[:nb] if nb else out


# ===========================================================================
# THE PORTED OBJECTIVE.  Closed-form context-expected OV metric (eq. dagger).
# ===========================================================================
@torch.no_grad()
def ctx_metrics(D, q, T=None, deltas=None, chunk=1024, verbose=True):
    """Per-token, per-block PSD metrics on the folded rows implied by eq. (†).

    BLOCK LAYOUT (fixed everywhere in this file): 4*H blocks, ordered
        (h=0,b=1,role=q), (h=0,b=1,role=k), (h=0,b=2,q), (h=0,b=2,k), (h=1,...)
    so head-branch group (h,b) owns the consecutive pair of blocks
    [q-role, k-role] -- exactly the parent's row `cat(q_hat[:,h], k_hat[:,h])`.

    DERIVATION (all first-order in one block's perturbation, which is EXACT for
    that block because the score is bilinear).  For head h, branch b, write the
    other branch's score s_ob(i,j) = coef(i,j), the OV write u(j) = Wproj_h
    Vv[h,j], nu_j = ||u(j)||^2, G = Wproj_h^T Wproj_h, and the rotary-rotated
    factors Kr(j) = R_delta K_b(j), Qr(i) = R_delta^T Q_b(i).

      QUERY role (perturb Q_b(i) by Delta): the perturbation multiplies EVERY
      key, so it is systematic within a context and the T^2 term is live:
        s_i   = Delta^T A_i Delta,  A_i = sum_j q_j nu_j coef(i,j)^2 Kr Kr^T/hd^2
        mu_i  = Wproj_h C_i Delta,  C_i = sum_j q_j coef(i,j) Vv[h,j] Kr^T / hd
        M_i   = q_i [ T A_i + (T^2 - T) C_i^T G C_i ]

      KEY role (perturb K_b(j) by Delta): the perturbation only fires on the
      draws that contain token j, so its systematic share is q_j:
        M_j   = q_j nu_j [ T + (T^2 - T) q_j ] sum_i q_i coef(i,j)^2 QrQr^T/hd^2

    The one approximation beyond eq. (†) itself is that error vectors from
    DIFFERENT blocks are treated as incoherent (block-diagonal); within a block
    the expectation is exact.

    Returns the TWO COMPONENTS separately, each (V, 4H, hd, hd):
        Mv (the scatter part, charged at T) and Ms (the systematic part,
        charged at T^2), so that   M(T) = T*Mv + T^2*Ms   for any T at no
        cost.  T is then a knob that interpolates the two limits the explainer
        identifies: T=1 is the norm/diagonal rung (no cancellation credit at
        all) and large T is the Gram rung (cancellation credited in full)."""
    H, hd, V = D.H, D.hd, D.V
    T = T or D.T_ctx
    deltas = deltas or delta_buckets(T)
    Mv = torch.zeros(4 * H, V, hd, hd, device=D.dev)
    Ms = torch.zeros(4 * H, V, hd, hd, device=D.dev)
    for h in range(H):
        Gh = D.Wp_h[h].t() @ D.Wp_h[h]                      # (hd,hd)
        nu = (D.U[h] ** 2).sum(-1)                          # (V,)
        Vvh = D.Vv[h]                                       # (V,hd)
        for bi, b in enumerate((1, 2)):
            ob = 2 if b == 1 else 1
            Qb, Kb = D.FT[f'Q{b}'][h], D.FT[f'K{b}'][h]
            Qo, Ko = D.FT[f'Q{ob}'][h], D.FT[f'K{ob}'][h]
            iq, ik = 4 * h + 2 * bi, 4 * h + 2 * bi + 1
            for dl, wgt in deltas:
                R = M.rot_matrix(dl, hd, D.dev)
                coef = (Qo @ R) @ Ko.t() / hd               # (V,V)
                Kr = Kb @ R.t()
                Qr = Qb @ R
                KK = (Kr[:, :, None] * Kr[:, None, :]).reshape(V, hd * hd)
                VK = (Vvh[:, :, None] * Kr[:, None, :]).reshape(V, hd * hd)
                QQ = (Qr[:, :, None] * Qr[:, None, :]).reshape(V, hd * hd)
                c2 = coef * coef
                # ---- query role
                for a in range(0, V, chunk):
                    w = c2[a:a + chunk] * (q * nu)[None, :]
                    A = (w @ KK).reshape(-1, hd, hd) / hd ** 2
                    Ci = ((coef[a:a + chunk] * q[None, :]) @ VK
                          ).reshape(-1, hd, hd) / hd
                    CGC = Ci.transpose(1, 2) @ Gh @ Ci
                    w0 = wgt * q[a:a + chunk, None, None]
                    Mv[iq, a:a + chunk] += w0 * (A - CGC)
                    Ms[iq, a:a + chunk] += w0 * CGC
                # ---- key role
                Gk = (c2.t() @ (q[:, None] * QQ)).reshape(V, hd, hd) / hd ** 2
                Mv[ik] += (wgt * q * nu * (1 - q))[:, None, None] * Gk
                Ms[ik] += (wgt * q * q * nu)[:, None, None] * Gk
                del coef, c2, KK, VK, QQ, Gk
        if verbose:
            log(f'  ctx_metrics head {h + 1}/{H}')
    out = []
    for Mb in (Mv, Ms):
        Mb = 0.5 * (Mb + Mb.transpose(-1, -2))
        ev, Ev = torch.linalg.eigh(Mb.double())     # PSD clamp (float noise)
        Mb = (Ev @ torch.diag_embed(ev.clamp_min(0))
              @ Ev.transpose(-1, -2)).float()
        out.append(Mb.permute(1, 0, 2, 3).contiguous())
    return out                                      # each (V, 4H, hd, hd)


def metric_at(Mv, Ms, T=256, blend=1.0, eps=0.0):
    """M(T) = T*Mv + T^2*Ms, trace-normalised (scale is irrelevant to the
    argmin but not to Adam), optionally BLENDED with the identity:
    blend=0 is plain MSE, blend=1 is the pure context objective.  `eps` adds a
    floor so no token's row is completely unconstrained."""
    M = T * Mv + (T * T) * Ms
    d = M.shape[-1]
    sc = M.diagonal(dim1=-2, dim2=-1).sum() / (M.shape[0] * M.shape[1] * d)
    M = M / sc.clamp_min(1e-30)
    I = torch.eye(d, device=M.device)[None, None]
    if blend >= 1.0 and eps == 0.0:
        return M
    return (1 - blend) * I + blend * M + eps * I


@torch.no_grad()
def ctx_cost_exact(D, FT, q, T=None, deltas=None, chunk=16):
    """The EXACT eq.(†) charge of a reconstructed fold -- no block-diagonal
    approximation, no linearisation: it forms the true pattern error including
    the product of the two perturbed branches, sums the OV writes over heads,
    and takes the context expectation.  Used to validate the metric and to
    report each arm's objective value.  Returns (absolute, relative to the
    same functional of the true pattern)."""
    H, hd, V = D.H, D.hd, D.V
    T = T or D.T_ctx
    deltas = deltas or delta_buckets(T)
    num = den = 0.0
    for dl, wgt in deltas:
        R = M.rot_matrix(dl, hd, D.dev)
        for a in range(0, V, chunk):
            sl = slice(a, min(a + chunk, V))
            dP = torch.empty(sl.stop - sl.start, H, V, device=D.dev)
            P0 = torch.empty_like(dP)
            for h in range(H):
                t1 = ((FT['Q1'][h][sl] @ R) @ FT['K1'][h].t()) / hd
                t2 = ((FT['Q2'][h][sl] @ R) @ FT['K2'][h].t()) / hd
                o1 = ((D.FT['Q1'][h][sl] @ R) @ D.FT['K1'][h].t()) / hd
                o2 = ((D.FT['Q2'][h][sl] @ R) @ D.FT['K2'][h].t()) / hd
                P0[:, h] = o1 * o2
                dP[:, h] = t1 * t2 - o1 * o2
            for tag, PP in (('d', dP), ('t', P0)):
                C = torch.einsum('ihj,hjd->ijd', PP, D.U)
                s = (q[None, :, None] * C * C).sum((1, 2))
                mu = torch.einsum('j,ijd->id', q, C)
                m = T * (s - (mu * mu).sum(-1)) + T * T * (mu * mu).sum(-1)
                val = float((q[sl] * m).sum()) * wgt
                if tag == 'd':
                    num += val
                else:
                    den += val
    return num, num / max(den, 1e-30)


# ===========================================================================
# WEIGHTED SPARSE CODING (OMP + least squares) UNDER A BLOCK-DIAGONAL METRIC
# ===========================================================================
def _blockmul(Mt, x):
    """Mt: (B,nb,d,d), x: (B,nb,d) -> (B,nb,d)."""
    return torch.einsum('bnde,bne->bnd', Mt, x)


@torch.no_grad()
def omp_metric(Dic, X, Mb, k, chunk=None):
    """Batched orthogonal matching pursuit with a least-squares refit at every
    step, in the metric ||.||_{M_t} (Mb=None -> plain MSE).

    Dic: (n, nb, d)   X: (V, nb, d)   Mb: (V, nb, d, d) or None
    returns idx (V,k) long, coef (V,k) float."""
    V, nb, d = X.shape
    n = Dic.shape[0]
    dev = X.device
    chunk = chunk or max(8, min(V, int(2 ** 22) // max(1, n)))
    idx = torch.zeros(V, k, dtype=torch.long, device=dev)
    coef = torch.zeros(V, k, device=dev)
    Dflat = Dic.reshape(n, nb * d)
    DD = (Dic[:, :, :, None] * Dic[:, :, None, :]).reshape(n, nb, d * d)
    for a in range(0, V, chunk):
        sl = slice(a, min(a + chunk, V))
        B = sl.stop - sl.start
        x = X[sl]
        Mt = None if Mb is None else Mb[sl]
        r = x.clone()
        sup = torch.zeros(B, 0, dtype=torch.long, device=dev)
        if Mt is None:
            nrm = (Dflat * Dflat).sum(1).clamp_min(1e-20)[None, :].expand(B, n)
        else:
            nrm = torch.einsum('nbe,tbe->tn', DD,
                               Mt.reshape(B, nb, d * d)).clamp_min(1e-20)
        for step in range(k):
            y = r if Mt is None else _blockmul(Mt, r)
            corr = y.reshape(B, nb * d) @ Dflat.t()          # (B,n)
            score = corr * corr / nrm
            if step:
                score.scatter_(1, sup, -1.0)
            pick = score.argmax(1, keepdim=True)
            sup = torch.cat([sup, pick], 1)
            Ds = Dic[sup.reshape(-1)].reshape(B, step + 1, nb, d)
            if Mt is None:
                G = torch.einsum('ipbd,iqbd->ipq', Ds, Ds)
                rhs = torch.einsum('ipbd,ibd->ip', Ds, x)
            else:
                MD = torch.einsum('ibde,ipbe->ipbd', Mt, Ds)
                G = torch.einsum('ipbd,iqbd->ipq', MD, Ds)
                rhs = torch.einsum('ipbd,ibd->ip', MD, x)
            ridge = (G.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-30)
                     * 1e-6)[:, None, None]
            G = G + ridge * torch.eye(step + 1, device=dev)[None]
            try:
                c = torch.linalg.solve(G, rhs[..., None])[..., 0]
            except Exception:
                c = torch.linalg.lstsq(G, rhs[..., None]).solution[..., 0]
            r = x - torch.einsum('ip,ipbd->ibd', c, Ds)
        idx[sl] = sup
        coef[sl] = c
    return idx, coef


def sparse_recon(Dic, idx, coef):
    """(V,k) codes -> (V, nb, d)."""
    V, k = idx.shape
    return torch.einsum('vk,vkbd->vbd', coef, Dic[idx.reshape(-1)].reshape(
        V, k, *Dic.shape[1:]))


@torch.no_grad()
def dict_learn(X, Mb, n, k, iters=8, inner=120, lr=0.02, seed=0, verbose=False):
    """Alternating minimisation: OMP/least-squares codes, then Adam on the
    atoms, both under the metric.  Init = distinct data rows (the same init the
    random-dictionary NULL uses, so the null isolates the learning)."""
    V, nb, d = X.shape
    g = torch.Generator(device='cpu').manual_seed(seed)
    Dic = X[torch.randperm(V, generator=g)[:n].to(X.device)].clone()
    Dic = Dic / Dic.reshape(n, -1).norm(dim=1).clamp_min(1e-9)[:, None, None]
    for it in range(iters):
        idx, coef = omp_metric(Dic, X, Mb, k)
        Dp = Dic.clone().requires_grad_(True)
        opt = torch.optim.Adam([Dp], lr=lr)
        for _ in range(inner):
            with torch.enable_grad():
                Rc = sparse_recon(Dp, idx, coef)
                e = X - Rc
                loss = (e * e).sum() if Mb is None else \
                    torch.einsum('vbd,vbde,vbe->', e, Mb, e)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        Dic = Dp.detach()
        nrm = Dic.reshape(n, -1).norm(dim=1).clamp_min(1e-9)
        Dic = Dic / nrm[:, None, None]
        if verbose:
            log(f'    dict iter {it + 1}/{iters} loss {float(loss):.6g}')
    idx, coef = omp_metric(Dic, X, Mb, k)
    return Dic, idx, coef


def dict_bits(n, k, nb, d, V, b_atom=32, b_coef=32):
    """PARENT CONVENTION (checked against their 455 Mbit at n=1024,k=8):
    atoms fp32, coefficients fp32, ceil(log2 n) bits per index."""
    return Bits(atoms=bits_dense(n * nb * d, b_atom),
                indices=bits_index(V * k, n),
                coefs=bits_dense(V * k, b_coef))


# ===========================================================================
# ANCHOR HYBRID: exact rows for the top-B tokens, dictionary for the tail
# ===========================================================================
@torch.no_grad()
def anchor_dict(X, Mb, n, k, B, order, iters=8, seed=0, b_atom=32, b_coef=32):
    """Exact fp32 rows for `order[:B]`; a dictionary FIT ON THE TAIL ONLY for
    the rest.  Every bit charged: anchor rows, anchor ids, atoms, indices,
    coefficients."""
    V, nb, d = X.shape
    dev = X.device
    anc = order[:B]
    keep = torch.ones(V, dtype=torch.bool, device=dev)
    keep[anc] = False
    tail = torch.nonzero(keep).squeeze(1)
    Xt = X[tail].contiguous()
    Mt = None if Mb is None else Mb[tail].contiguous()
    Dic, idx, coef = dict_learn(Xt, Mt, n, k, iters=iters, seed=seed)
    R = X.clone()
    R[tail] = sparse_recon(Dic, idx, coef)
    bt = Bits(anchor_rows=bits_dense(B * nb * d, 32),
              anchor_ids=bits_index(B, V),
              atoms=bits_dense(n * nb * d, b_atom),
              indices=bits_index(len(tail) * k, n),
              coefs=bits_dense(len(tail) * k, b_coef))
    return R, bt, {'B': int(B), 'n_tail': int(len(tail))}


# ===========================================================================
# REFERENCE FAMILY: low rank on the same object (the parent's SVD frontier)
# ===========================================================================
@torch.no_grad()
def lowrank(X, r, b=32):
    """Rank-r truncated SVD of the (V, nb*d) row matrix."""
    V, nb, d = X.shape
    W = X.reshape(V, nb * d)
    U, S, Vh = torch.linalg.svd(W.double(), full_matrices=False)
    A = (U[:, :r] * S[:r]).float()
    Bm = Vh[:r].float()
    R = (A @ Bm).reshape(V, nb, d)
    return R, Bits(factorA=bits_dense(A.numel(), b),
                   factorB=bits_dense(Bm.numel(), b))
