"""PHASE 2 — sparse dictionary over the LAYER-0 exact query/key factors (Logan 2026-07-21).

Object (Option A, weight-only): per head-branch rows cat([q_hat[:,h], k_hat[:,h]], 1) of shape
(V, 256), 18 head-branches, from tier2_folding.branch_factors. The vocab-by-vocab score map per
head-branch is exactly the product of the factor tables, so coding the factors codes the map.

Arms, ALL at explicit description length (program rule 3: compare only at matched bits):
  - SVD frontier on the same rows, rank r in R_GRID (the baseline curve to beat; the concatenated
    rows are exactly storable at rank 256; the score map itself is rank <= 128 by construction so
    the raw factor tables ARE the "rank-128" exact reference, 7417.6 Mbit).
  - Per-token top-k dictionary, linear encoder (train_dict mode='token').
  - Per-token top-k with orthogonal-matching-pursuit / least-squares coefficients on the same
    trained dictionary (Phase-0 strong arm).
  - Batch-top-k (Phase-0 pre-registered prediction: LOSES when atoms correlate — replication test).
  - Matryoshka (nested prefixes).
  - TWO-STAGE (Logan's stage1->stage2): per-head-branch merge K=2048 (stage-1 winner, ~free),
    then an OMP/LS dictionary over the 2048 centroids — coefficients paid per class, not per token.

Dictionaries are fit PER HEAD-BRANCH (18 dictionaries; stage-1 found per-head-branch structure
dominates the global partition at matched bits). Fraction-of-variance-unexplained is the search
metric; held-out delta-cross-entropy (patched layer-0 scores) is binding. Single seed 0 this pass;
convergence/seed variance is Phase 4. Writes qk_sae_dict.json (incrementally).
"""
import json
import math
import sys
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, build_eval_tokens, reference_forward
from tier2_folding import branch_factors, scores_from_factors
from mdl_accounting import dl_svd, dl_sparse_dict, dl_bits

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/qk_sae_dict.json'
NAMES = ('q1', 'k1', 'q2', 'k2')
BRANCHES = (('q1', 'k1'), ('q2', 'k2'))
R_GRID = (8, 16, 32, 64, 128)
BUDGETS = ((1024, 8), (4096, 8))        # (n_atoms, k) per head-branch
MERGE_K, MERGE_DICT = 2048, (512, 8)    # two-stage arm
STEPS = 3000

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
NHB, ROW = NH * 2, 2 * HD
ALL = build_eval_tokens(n_chunks=20, seq_len=513)
AUDIT = ALL[4:20]

TAB = {}
for br, (qn, kn) in enumerate(BRANCHES, start=1):
    qh, kh = branch_factors(m, br)
    TAB[qn], TAB[kn] = qh.float().to(DEV), kh.float().to(DEV)

HB = [(h, qn, kn) for h in range(NH) for (qn, kn) in BRANCHES]      # 18 head-branches


def rows(h, qn, kn):
    return torch.cat([TAB[qn][:, h], TAB[kn][:, h]], 1)             # (V, 256)


def unit_rms(t):
    return t * (t.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt())


def tables_from(recs, renorm=True):
    """recs: list of (V, 256) reconstructions in HB order -> full factor table dict."""
    out = {n: TAB[n].clone() for n in NAMES}
    for (h, qn, kn), rec in zip(HB, recs):
        out[qn][:, h] = rec[:, :HD]
        out[kn][:, h] = rec[:, HD:]
    return {n: unit_rms(out[n]) for n in NAMES} if renorm else out


# ------------------------------------------------------------------ audit (binding metric)

@torch.no_grad()
def audit_ce(tabs):
    tot, n = 0.0, 0
    for i in range(0, len(AUDIT), 4):
        b = AUDIT[i:i + 4].to(DEV)
        idx = b[:, :-1]

        def patch(li, s1, s2):
            if li != 0:
                return s1, s2
            n1 = scores_from_factors(tabs['q1'], tabs['k1'], idx, HD)
            n2 = scores_from_factors(tabs['q2'], tabs['k2'], idx, HD)
            return n1.to(s1.dtype), n2.to(s2.dtype)

        logits = reference_forward(m, idx, 'bf16', score_patch=None if tabs is None else patch).float()
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n


# ------------------------------------------------------------------ fits (Phase-0 recipes, verbatim)

def fvu(Xhat, X):
    return ((Xhat - X) ** 2).sum().item() / ((X - X.mean(0)) ** 2).sum().item()


@torch.no_grad()
def arm_svd(X, r):
    b = X.mean(0)
    U, S, Vh = torch.linalg.svd(X - b, full_matrices=False)
    return b + (U[:, :r] * S[:r]) @ Vh[:r]


def train_dict(X, n, k, mode='token', steps=STEPS, batch=2048, lr=3e-3, seed=0, nested=None):
    g = torch.Generator(device='cpu').manual_seed(seed)
    Dm = X[torch.randperm(len(X), generator=g)[:n].to(X.device)].clone()
    Dm = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
    We = Dm.clone()
    b = X.mean(0).clone()
    for t in (Dm, We, b):
        t.requires_grad_(True)
    opt = torch.optim.Adam([Dm, We, b], lr=lr)
    fired = torch.zeros(n, device=X.device)
    last = []
    for step in range(steps):
        x = X[torch.randint(0, len(X), (min(batch, len(X)),), device=X.device)]
        Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
        z = (x - b) @ We.T
        if nested is not None:
            loss = 0.0
            for P in nested:
                kp = max(1, int(round(k * P / n)))
                zp = z[:, :P]
                vals, idx = zp.abs().topk(min(kp, P), dim=1)
                coeff = torch.gather(zp, 1, idx)
                xhat = b + (coeff.unsqueeze(-1) * Dn[:P][idx]).sum(1)
                loss = loss + ((xhat - x) ** 2).mean()
        elif mode == 'token':
            vals, idx = z.abs().topk(k, dim=1)
            coeff = torch.gather(z, 1, idx)
            xhat = b + (coeff.unsqueeze(-1) * Dn[idx]).sum(1)
            fired.index_add_(0, idx.reshape(-1), torch.ones(idx.numel(), device=X.device))
            loss = ((xhat - x) ** 2).mean()
        else:
            flat = z.abs().reshape(-1)
            thresh = flat.topk(k * len(x)).values.min()
            zc = z * (z.abs() >= thresh)
            fired.index_add_(0, (zc != 0).nonzero()[:, 1], torch.ones((zc != 0).sum(), device=X.device))
            loss = ((b + zc @ Dn - x) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        last.append(loss.item())
        if (step + 1) % 500 == 0 and nested is None:
            dead = (fired == 0).nonzero().squeeze(1)
            if len(dead):
                with torch.no_grad():
                    Dn_ = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
                    zc_ = (X - b) @ We.T
                    v_, i_ = zc_.abs().topk(k, dim=1)
                    rec = b + (torch.gather(zc_, 1, i_).unsqueeze(-1) * Dn_[i_]).sum(1)
                    worst = ((rec - X) ** 2).sum(1).topk(len(dead)).indices
                    Dm[dead] = X[worst] / X[worst].norm(dim=1, keepdim=True).clamp(min=1e-8)
                    We[dead] = Dm[dead]
            fired.zero_()
    Dn = (Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)).detach()
    return Dn, b.detach(), We.detach(), sum(last[-50:]) / 50


@torch.no_grad()
def encode_token(X, Dn, b, We, k):
    z = (X - b) @ We.T
    vals, idx = z.abs().topk(k, dim=1)
    coeff = torch.gather(z, 1, idx)
    return b + (coeff.unsqueeze(-1) * Dn[idx]).sum(1), k * len(X)


@torch.no_grad()
def encode_batch(X, Dn, b, We, kavg):
    z = (X - b) @ We.T
    thresh = z.abs().reshape(-1).topk(kavg * len(X)).values.min()
    zc = z * (z.abs() >= thresh)
    return b + zc @ Dn, int((zc != 0).sum())


@torch.no_grad()
def encode_omp(X, Dn, b, k, chunk=8192):
    outs, nnz = [], 0
    for i in range(0, len(X), chunk):
        Y = X[i:i + chunk] - b
        nb = len(Y)
        r = Y.clone()
        sup = torch.full((nb, k), -1, device=X.device, dtype=torch.long)
        chosen = torch.zeros(nb, Dn.shape[0], dtype=torch.bool, device=X.device)
        recon = torch.zeros_like(Y)
        for s in range(k):
            corr = (r @ Dn.T).abs()
            corr[chosen] = -1
            a = corr.argmax(1)
            sup[:, s] = a
            chosen[torch.arange(nb, device=X.device), a] = True
            Ds = Dn[sup[:, :s + 1]]
            G = torch.bmm(Ds, Ds.transpose(1, 2))
            rhs = torch.bmm(Ds, Y.unsqueeze(-1)).squeeze(-1)
            c = torch.linalg.solve(G + 1e-6 * torch.eye(s + 1, device=X.device), rhs)
            recon = torch.bmm(c.unsqueeze(1), Ds).squeeze(1)
            r = Y - recon
        outs.append(b + recon)
        nnz += k * nb
    return torch.cat(outs), nnz


def kmeans(X, k, iters=12, seed=0, chunk=4096):
    g = torch.Generator(device='cpu').manual_seed(seed)
    C = X[torch.randperm(len(X), generator=g)[:k].to(X.device)].clone()
    for _ in range(iters):
        assign = torch.empty(len(X), dtype=torch.long, device=X.device)
        Cn2 = (C * C).sum(1)[None]
        for i in range(0, len(X), chunk):
            xx = X[i:i + chunk]
            assign[i:i + chunk] = ((xx * xx).sum(1, True) - 2 * xx @ C.T + Cn2).argmin(1)
        Cnew = torch.zeros_like(C)
        c2 = torch.zeros(k, device=X.device)
        Cnew.index_add_(0, assign, X)
        c2.index_add_(0, assign, torch.ones(len(X), device=X.device))
        nz = c2 > 0
        C[nz] = Cnew[nz] / c2[nz][:, None]
    return assign, C


# ------------------------------------------------------------------ run

BITS_RAW = 32 * NHB * V * ROW
res = {'object': 'layer-0 exact weight-only fold (Option A)', 'raw_Mbits': round(BITS_RAW / 1e6, 1),
       'seed': 0, 'steps': STEPS, 'arms': {}}
CE0 = audit_ce(None)
res['baseline_ce'] = round(CE0, 4)
print(f'baseline CE {CE0:.4f}', flush=True)


def report(name, recs, bits, mean_fvu, extra=None, renorm=True):
    d = audit_ce(tables_from(recs, renorm)) - CE0
    row = {'dce': round(d, 4), 'Mbits': round(bits / 1e6, 1),
           'pct_raw': round(100 * bits / BITS_RAW, 2), 'fvu': round(mean_fvu, 5), 'renorm': renorm}
    if extra:
        row.update(extra)
    res['arms'][name] = row
    print(f'{name:48s} dCE {d:+.4f}  fvu {mean_fvu:.4f}  {bits/1e6:8.1f} Mbit ({100*bits/BITS_RAW:5.2f}%)',
          flush=True)
    json.dump(res, open(OUT, 'w'), indent=2)


# --- SVD frontier (baseline curve) ---
for r in R_GRID:
    recs, fv = [], []
    for (h, qn, kn) in HB:
        X = rows(h, qn, kn)
        Xh = arm_svd(X, r)
        recs.append(Xh)
        fv.append(fvu(Xh, X))
    report(f'svd rank {r}', recs, NHB * dl_svd(r, V, ROW), sum(fv) / len(fv), extra={'r': r})

# --- dictionary arms per budget ---
for (NA, K) in BUDGETS:
    bits_d = NHB * dl_sparse_dict(NA, ROW, V * K)
    fits_tok, fits_bat, fits_mat = [], [], []
    for bi, (h, qn, kn) in enumerate(HB):
        X = rows(h, qn, kn)
        fits_tok.append(train_dict(X, NA, K, mode='token', seed=0))
        fits_bat.append(train_dict(X, NA, K, mode='batch', seed=0))
        fits_mat.append(train_dict(X, NA, K, mode='token', seed=0, nested=[NA // 8, NA // 2, NA]))
        print(f'  fitted head-branch {bi + 1}/{NHB} (n={NA}, k={K}); '
              f'tok loss {fits_tok[-1][3]:.5f}', flush=True)

    for arm, enc in (('token-linear', lambda X, f: encode_token(X, f[0], f[1], f[2], K)),
                     ('token-OMP/LS', lambda X, f: encode_omp(X, f[0], f[1], K))):
        recs, fv = [], []
        for f, (h, qn, kn) in zip(fits_tok, HB):
            X = rows(h, qn, kn)
            Xh, _ = enc(X, f)
            recs.append(Xh)
            fv.append(fvu(Xh, X))
        report(f'dict n={NA} k={K} {arm}', recs, bits_d, sum(fv) / len(fv), extra={'n': NA, 'k': K})
        if arm == 'token-OMP/LS' and (NA, K) == BUDGETS[0]:
            report(f'dict n={NA} k={K} {arm} (no renorm)', recs, bits_d, sum(fv) / len(fv),
                   extra={'n': NA, 'k': K}, renorm=False)

    recs, fv, nnz_tot = [], [], 0
    for f, (h, qn, kn) in zip(fits_bat, HB):
        X = rows(h, qn, kn)
        Xh, nnz = encode_batch(X, f[0], f[1], f[2], K)
        recs.append(Xh)
        fv.append(fvu(Xh, X))
        nnz_tot += nnz
    report(f'dict n={NA} k={K} batch-topk', recs,
           NHB * dl_bits(n_floats=NA * ROW + ROW) + nnz_tot * (32 + math.log2(NA)),
           sum(fv) / len(fv), extra={'n': NA, 'k': K, 'nnz': nnz_tot})

    recs, fv = [], []
    for f, (h, qn, kn) in zip(fits_mat, HB):
        X = rows(h, qn, kn)
        Xh, _ = encode_token(X, f[0], f[1], f[2], K)
        recs.append(Xh)
        fv.append(fvu(Xh, X))
    report(f'dict n={NA} k={K} matryoshka', recs, bits_d, sum(fv) / len(fv), extra={'n': NA, 'k': K})
    del fits_tok, fits_bat, fits_mat
    torch.cuda.empty_cache()

# --- two-stage: merge K=2048 then OMP dictionary over centroids ---
NA2, K2 = MERGE_DICT
recs, fv = [], []
bits_2s = 0.0
for bi, (h, qn, kn) in enumerate(HB):
    X = rows(h, qn, kn)
    assign, C = kmeans(X, MERGE_K, seed=bi)
    Dn, b, We, _ = train_dict(C, NA2, K2, mode='token', seed=0, steps=2000)
    Chat, nnz = encode_omp(C, Dn, b, K2)
    recs.append(Chat[assign])
    fv.append(fvu(Chat[assign], X))
    bits_2s += dl_sparse_dict(NA2, ROW, nnz) + V * math.log2(MERGE_K)
report(f'two-stage merge{MERGE_K} -> dict n={NA2} k={K2} OMP/LS', recs, bits_2s, sum(fv) / len(fv),
       extra={'merge_K': MERGE_K, 'n': NA2, 'k': K2})

json.dump(res, open(OUT, 'w'), indent=2)
print(f'\nwrote {OUT}', flush=True)
