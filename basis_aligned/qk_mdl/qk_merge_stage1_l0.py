"""PHASE 1 (Option A) — stage-one FREE MERGE for the LAYER-0 query/key circuit (Logan 2026-07-21).

Object: the EXACT weight-only layer-0 fold (tier2_folding.branch_factors) — per branch, unit-RMS
per-token factors q_hat(t), k_hat(t) of shape (V, 9, 128). No data enters the object; data enters
only the held-out audit (binding metric, program rule 2). The vocabulary-by-vocabulary score map per
head-branch IS the product of these factor tables via the rotary cosine/sine expansion, so merging /
coding the factors decomposes the vocab-by-vocab map losslessly (rule 8: never materialize V x V).

Stage one (Logan): "tokens that pay attention to the same things can be treated as the same token" —
a VOCABULARY reduction. Primary arm: a SINGLE GLOBAL PARTITION of the vocabulary shared by all 18
head-branches (the token->class index is paid ONCE). The per-head-branch partition (18 independent
partitions) is the more-expressive upper bound paying 18x index bits. Comparison only at matched
description length, never matched class count.

Unlike Option B (layer-1 conditional means, +0.014 floor), the uncompressed arm here is EXACT — its
held-out delta-cross-entropy doubles as a CE-level gate and must be ~0.

Writes qk_merge_stage1_l0.json.
"""
import json
import math
import sys
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, build_eval_tokens, reference_forward
from tier2_folding import branch_factors, scores_from_factors

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/qk_merge_stage1_l0.json'
NAMES = ('q1', 'k1', 'q2', 'k2')
BRANCHES = (('q1', 'k1'), ('q2', 'k2'))
K_GRID = (256, 512, 2048, 8192)
PCA_DIM = 256                       # merge decisions made in a reduced space (binding metric decides)

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
NHB = NH * 2                        # 18 head-branches
ROW = 2 * HD                        # 256
ALL = build_eval_tokens(n_chunks=20, seq_len=513)
AUDIT = ALL[4:20]                   # established split; no TRAIN pass — the object is weight-only

# ------------------------------------------------------------------ exact weight-only factors
TAB = {}
for br, (qn, kn) in enumerate(BRANCHES, start=1):
    qh, kh = branch_factors(m, br)                     # fp64 (V, NH, HD), unit-RMS rows
    TAB[qn], TAB[kn] = qh.float().to(DEV), kh.float().to(DEV)


def unit_rms(t):
    return t * (t.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt())


# ------------------------------------------------------------------ audit (binding metric)

@torch.no_grad()
def audit_ce(tabs=None, zero=False):
    tot, n = 0.0, 0
    for i in range(0, len(AUDIT), 4):
        b = AUDIT[i:i + 4].to(DEV)
        idx = b[:, :-1]

        def patch(li, s1, s2):
            if li != 0:
                return s1, s2
            if zero:
                return torch.zeros_like(s1), torch.zeros_like(s2)
            n1 = scores_from_factors(tabs['q1'], tabs['k1'], idx, HD)
            n2 = scores_from_factors(tabs['q2'], tabs['k2'], idx, HD)
            return n1.to(s1.dtype), n2.to(s2.dtype)

        logits = reference_forward(m, idx, 'bf16', score_patch=None if tabs is None and not zero else patch).float()
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n


# ------------------------------------------------------------------ merge machinery

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


def rows_all():
    """(V, 18*256) — every head-branch row concatenated, the object a GLOBAL merge acts on."""
    parts = []
    for h in range(NH):
        for (qn, kn) in BRANCHES:
            parts.append(torch.cat([TAB[qn][:, h], TAB[kn][:, h]], 1))
    return torch.cat(parts, 1)


def apply_global_assign(assign, renorm):
    """Replace each token's rows by its class centroid (per head-branch)."""
    out = {}
    K = int(assign.max()) + 1
    for n in NAMES:
        out[n] = torch.empty_like(TAB[n])
    for h in range(NH):
        for (qn, kn) in BRANCHES:
            X = torch.cat([TAB[qn][:, h], TAB[kn][:, h]], 1)
            C = torch.zeros(K, ROW, device=DEV)
            c2 = torch.zeros(K, device=DEV)
            C.index_add_(0, assign, X)
            c2.index_add_(0, assign, torch.ones(V, device=DEV))
            nz = c2 > 0
            C[nz] /= c2[nz][:, None]
            rec = C[assign]
            out[qn][:, h] = rec[:, :HD]
            out[kn][:, h] = rec[:, HD:]
    return {n: unit_rms(out[n]) for n in NAMES} if renorm else out


def apply_perhb_assign(K, renorm, seed0=0):
    """18 independent partitions (upper bound; pays 18x index bits)."""
    out = {n: TAB[n].clone() for n in NAMES}
    for h in range(NH):
        for br, (qn, kn) in enumerate(BRANCHES):
            X = torch.cat([TAB[qn][:, h], TAB[kn][:, h]], 1)
            assign, C = kmeans(X, K, seed=seed0 + h * 2 + br)
            rec = C[assign]
            out[qn][:, h] = rec[:, :HD]
            out[kn][:, h] = rec[:, HD:]
    return {n: unit_rms(out[n]) for n in NAMES} if renorm else out


# bits: 18 head-branch centroid tables + index bits (once for global, 18x for per-head-branch)
def bits_global(K):
    return 32 * (NHB * K * ROW) + V * math.log2(max(K, 2))


def bits_perhb(K):
    return 32 * (NHB * K * ROW) + NHB * V * math.log2(max(K, 2))


BITS_RAW = 32 * NHB * V * ROW

# ------------------------------------------------------------------ run

res = {'object': 'layer-0 exact weight-only fold (Option A)', 'V': V,
       'raw_Mbits': round(BITS_RAW / 1e6, 1), 'arms': {}}
CE0 = audit_ce(tabs=None)
res['baseline_ce'] = round(CE0, 4)
print(f'baseline CE {CE0:.4f}', flush=True)


def run(name, tabs, bits, extra=None):
    d = audit_ce(tabs=tabs) - CE0
    row = {'dce': round(d, 4), 'Mbits': round(bits / 1e6, 1), 'pct_raw': round(100 * bits / BITS_RAW, 2)}
    if extra:
        row.update(extra)
    res['arms'][name] = row
    print(f'{name:44s} dCE {d:+.4f}   {bits/1e6:8.1f} Mbit  ({100*bits/BITS_RAW:5.2f}% raw)', flush=True)
    json.dump(res, open(OUT, 'w'), indent=2)
    return d


# controls: zero-scores (the circuit matters) and the EXACT fold (CE-level gate, must be ~0)
dz = audit_ce(zero=True) - CE0
res['arms']['zero-scores (control)'] = {'dce': round(dz, 4)}
print(f'{"zero-scores (control)":44s} dCE {dz:+.4f}', flush=True)
dg = run('GATE: exact fold, no merge', TAB, BITS_RAW)
res['gate_dce'] = round(dg, 5)
assert abs(dg) < 5e-3, f'CE-level fold gate FAILED: {dg}'

# stage-1 GLOBAL merge (the real "these tokens are the same token" reduction)
Xall = rows_all()
mu = Xall.mean(0)
Xc = Xall - mu
cov = torch.zeros(Xc.shape[1], Xc.shape[1], dtype=torch.float64, device=DEV)
for i in range(0, V, 8192):
    xx = Xc[i:i + 8192].double()
    cov += xx.T @ xx
evals, evecs = torch.linalg.eigh(cov)
top = evecs[:, -PCA_DIM:].flip(-1)
var = float(evals[-PCA_DIM:].sum() / evals.clamp_min(0).sum())
Xp = (Xc.double() @ top).float()
print(f'global merge space: {tuple(Xp.shape)} (top-{PCA_DIM} of {Xall.shape[1]}; '
      f'{100*var:.1f}% variance)', flush=True)
del Xall, Xc, cov, evals, evecs, top
torch.cuda.empty_cache()

for K in K_GRID:
    a_s, _ = kmeans(Xp, K, seed=0)
    for renorm in (False, True):
        tag = f'stage1 GLOBAL merge K={K}' + (' +unit-RMS' if renorm else '')
        run(tag, apply_global_assign(a_s, renorm), bits_global(K), extra={'K': K, 'renorm': renorm})

# per-head-branch upper bound (18 partitions, 18x index bits)
for K in K_GRID:
    run(f'per-head-branch merge K={K} +unit-RMS', apply_perhb_assign(K, True),
        bits_perhb(K), extra={'K': K})

json.dump(res, open(OUT, 'w'), indent=2)
print(f'\nwrote {OUT}', flush=True)
