"""PHASE 4 — robustness for the layer-0 merge + sparse-dictionary results (tick 152).

Question to settle: Phases 1-2 show several compressed arms with slightly NEGATIVE held-out
delta-cross-entropy (-0.006 to -0.020) on the 16-sequence audit (8192 predictions). Is
"compression is free-or-better" real, or audit noise?

  - WIDE AUDIT: 128 held-out sequences (65,536 predictions), disjoint from the original
    AUDIT = ALL[4:20]; wide set = ALL[20:148]. Every key arm re-audited on both.
  - SEEDS: k-means merge (K=2048 per-head-branch) at 3 seeds; dictionary (n=1024, k=8,
    per-head-branch) trained at 3 seeds, audited with the linear encoder and with
    orthogonal-matching-pursuit/least-squares; two-stage (merge 2048 -> OMP dict n=512 k=8)
    at 2 seeds. SVD (deterministic) at r=16, 32 re-audited wide.

Fit code is verbatim from qk_sae_dict.py (which has no import guard). Writes qk_sae_robust.json.
"""
import json
import math
import sys
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, build_eval_tokens, reference_forward
from tier2_folding import branch_factors, scores_from_factors
from mdl_accounting import dl_svd, dl_sparse_dict

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/qk_sae_robust.json'
NAMES = ('q1', 'k1', 'q2', 'k2')
BRANCHES = (('q1', 'k1'), ('q2', 'k2'))
SEEDS = (0, 1, 2)

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
NHB, ROW = NH * 2, 2 * HD
ALL = build_eval_tokens(n_chunks=148, seq_len=513)
AUDIT_ORIG, AUDIT_WIDE = ALL[4:20], ALL[20:148]

TAB = {}
for br, (qn, kn) in enumerate(BRANCHES, start=1):
    qh, kh = branch_factors(m, br)
    TAB[qn], TAB[kn] = qh.float().to(DEV), kh.float().to(DEV)

HB = [(h, qn, kn) for h in range(NH) for (qn, kn) in BRANCHES]


def rows(h, qn, kn):
    return torch.cat([TAB[qn][:, h], TAB[kn][:, h]], 1)


def unit_rms(t):
    return t * (t.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt())


def tables_from(recs, renorm=True):
    out = {n: TAB[n].clone() for n in NAMES}
    for (h, qn, kn), rec in zip(HB, recs):
        out[qn][:, h] = rec[:, :HD]
        out[kn][:, h] = rec[:, HD:]
    return {n: unit_rms(out[n]) for n in NAMES} if renorm else out


@torch.no_grad()
def audit_ce(tabs, tokens):
    tot, n = 0.0, 0
    for i in range(0, len(tokens), 4):
        b = tokens[i:i + 4].to(DEV)
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


# ---- fit code verbatim from qk_sae_dict.py ----

@torch.no_grad()
def arm_svd(X, r):
    b = X.mean(0)
    U, S, Vh = torch.linalg.svd(X - b, full_matrices=False)
    return b + (U[:, :r] * S[:r]) @ Vh[:r]


def train_dict(X, n, k, mode='token', steps=3000, batch=2048, lr=3e-3, seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    Dm = X[torch.randperm(len(X), generator=g)[:n].to(X.device)].clone()
    Dm = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
    We = Dm.clone()
    b = X.mean(0).clone()
    for t in (Dm, We, b):
        t.requires_grad_(True)
    opt = torch.optim.Adam([Dm, We, b], lr=lr)
    fired = torch.zeros(n, device=X.device)
    for step in range(steps):
        x = X[torch.randint(0, len(X), (min(batch, len(X)),), device=X.device)]
        Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
        z = (x - b) @ We.T
        vals, idx = z.abs().topk(k, dim=1)
        coeff = torch.gather(z, 1, idx)
        xhat = b + (coeff.unsqueeze(-1) * Dn[idx]).sum(1)
        fired.index_add_(0, idx.reshape(-1), torch.ones(idx.numel(), device=X.device))
        loss = ((xhat - x) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 500 == 0:
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
    return Dn, b.detach(), We.detach()


@torch.no_grad()
def encode_token(X, Dn, b, We, k):
    z = (X - b) @ We.T
    vals, idx = z.abs().topk(k, dim=1)
    coeff = torch.gather(z, 1, idx)
    return b + (coeff.unsqueeze(-1) * Dn[idx]).sum(1)


@torch.no_grad()
def encode_omp(X, Dn, b, k, chunk=8192):
    outs = []
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
    return torch.cat(outs)


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

res = {'n_orig_preds': int(AUDIT_ORIG.shape[0] * 512), 'n_wide_preds': int(AUDIT_WIDE.shape[0] * 512),
       'arms': {}}
CE0_O = audit_ce(None, AUDIT_ORIG)
CE0_W = audit_ce(None, AUDIT_WIDE)
res['baseline_ce_orig'] = round(CE0_O, 4)
res['baseline_ce_wide'] = round(CE0_W, 4)
print(f'baseline CE orig {CE0_O:.4f} | wide {CE0_W:.4f}', flush=True)


def report(name, recs, Mbits=None, renorm=True):
    tabs = tables_from(recs, renorm)
    do = audit_ce(tabs, AUDIT_ORIG) - CE0_O
    dw = audit_ce(tabs, AUDIT_WIDE) - CE0_W
    row = {'dce_orig': round(do, 4), 'dce_wide': round(dw, 4)}
    if Mbits is not None:
        row['Mbits'] = round(Mbits, 1)
    res['arms'][name] = row
    print(f'{name:52s} dCE orig {do:+.4f} | wide {dw:+.4f}', flush=True)
    json.dump(res, open(OUT, 'w'), indent=2)


# exact fold (gate on both audits)
report('exact fold', [rows(*hb) for hb in HB])

# svd (deterministic)
for r in (16, 32):
    recs = [arm_svd(rows(*hb), r) for hb in HB]
    report(f'svd rank {r}', recs, NHB * dl_svd(r, V, ROW) / 1e6)

# per-head-branch merge K=2048, 3 k-means seeds
for sd in SEEDS:
    recs = []
    for bi, hb in enumerate(HB):
        X = rows(*hb)
        assign, C = kmeans(X, 2048, seed=1000 * sd + bi)
        recs.append(C[assign])
    report(f'merge K=2048 per-head-branch seed {sd}', recs,
           (32 * NHB * 2048 * ROW + NHB * V * math.log2(2048)) / 1e6)

# dictionary n=1024 k=8, 3 training seeds, two encoders
for sd in SEEDS:
    fits = []
    for bi, hb in enumerate(HB):
        fits.append(train_dict(rows(*hb), 1024, 8, seed=sd))
    print(f'  dict seed {sd}: 18 head-branches fitted', flush=True)
    bits = NHB * dl_sparse_dict(1024, ROW, V * 8) / 1e6
    report(f'dict n=1024 k=8 token-linear seed {sd}',
           [encode_token(rows(*hb), *f, 8) for f, hb in zip(fits, HB)], bits)
    report(f'dict n=1024 k=8 token-OMP/LS seed {sd}',
           [encode_omp(rows(*hb), f[0], f[1], 8) for f, hb in zip(fits, HB)], bits)
    del fits
    torch.cuda.empty_cache()

# two-stage merge 2048 -> OMP dict n=512 k=8, 2 seeds
for sd in SEEDS[:2]:
    recs = []
    bits = 0.0
    for bi, hb in enumerate(HB):
        X = rows(*hb)
        assign, C = kmeans(X, 2048, seed=1000 * sd + bi)
        Dn, b, We = train_dict(C, 512, 8, seed=sd, steps=2000)
        Chat = encode_omp(C, Dn, b, 8)
        recs.append(Chat[assign])
        bits += dl_sparse_dict(512, ROW, 2048 * 8) + V * math.log2(2048)
    report(f'two-stage merge2048 -> OMP dict n=512 k=8 seed {sd}', recs, bits / 1e6)

json.dump(res, open(OUT, 'w'), indent=2)
print(f'\nwrote {OUT}', flush=True)
