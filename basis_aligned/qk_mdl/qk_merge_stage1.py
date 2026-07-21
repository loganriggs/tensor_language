"""PHASE 1 — stage-one FREE MERGE for the layer-1 query/key circuit (Logan 2026-07-21).

Logan's stage one: "some tokens pay attention to the same things so they can be considered the same,
thus a reduction in description length." That is a VOCABULARY reduction, so the primary arm is a
SINGLE GLOBAL PARTITION of the vocabulary shared by all 18 head-branches (paying the token->class
index once). The per-head-branch partition (18 separate partitions, the existing vq_tables move) is
run as a more-expressive upper bound that pays 18x the index bits — reported so the comparison is at
matched description length, never at matched class count.

Object: the layer-1 conditional-mean folded factor tables q_bar(t), k_bar(t) (post query/key-norm,
pre-RoPE), unit-RMS renormalized — the established gauge. Rows per head-branch are
cat([q_bar(t)[h], k_bar(t)[h]]) of shape (V, 2*head_dim) = (V, 256).

FLOOR, printed beside every number: the conditional-mean tables ALONE already cost ~+0.014 held-out.
No merge can beat that; every merge number is measured on top of it.

Caches the tables to l1_qk_tables.pt so Phase 2 does not repeat the estimation pass.
Writes qk_merge_stage1.json.
"""
import json
import math
import os
import sys
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens, reference_forward
from tier2_folding import scores_from_factors

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/qk_merge_stage1.json'
CACHE = f'{QK}/l1_qk_tables.pt'
NAMES = ('q1', 'k1', 'q2', 'k2')
BRANCHES = (('q1', 'k1'), ('q2', 'k2'))
K_GRID = (512, 2048, 8192)
PCA_DIM = 256                       # merge decisions are made in a reduced space (binding metric still decides)

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
NHB = NH * 2                        # 18 head-branches
ROW = 2 * HD                        # 256
ALL = build_eval_tokens(n_chunks=20 + 1024, seq_len=513)
AUDIT, TRAIN = ALL[4:20], ALL[20:]


# ------------------------------------------------------------------ estimation pass (cached)

@torch.no_grad()
def capture(idx, acc, cnt):
    x = F.rms_norm(m.transformer.wte(idx), (D,))
    x0, v1 = x, None
    B, T = idx.shape
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    for li in (0, 1):
        blk = m.transformer.h[li]
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn
        h = F.rms_norm(x, (D,))
        if li == 1:
            flat = idx.reshape(-1)
            for name, lin in (('q1', a.c_q), ('k1', a.c_k), ('q2', a.c_q2), ('k2', a.c_k2)):
                z = F.rms_norm(lin(h).view(B, T, NH, HD), (HD,))
                acc[name].index_add_(0, flat, z.reshape(-1, NH * HD).float())
            cnt.index_add_(0, flat, torch.ones_like(flat, dtype=torch.float))
            return
        qn = lambda lin: apply_rot(F.rms_norm(lin(h).view(B, T, NH, HD), (HD,)), cos, sin)
        q, k, q2, k2 = qn(a.c_q), qn(a.c_k), qn(a.c_q2), qn(a.c_k2)
        v = a.c_v(h).view(B, T, NH, HD)
        v1 = v if v1 is None else v1
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        x = x + a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
        x = x + blk.mlp(F.rms_norm(x, (D,)))


def unit_rms(t):
    return t * (t.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt())


if os.path.exists(CACHE):
    blob = torch.load(CACHE, map_location=DEV)
    tables = {n: blob[n].to(DEV) for n in NAMES}
    seen = blob['seen'].to(DEV)
    print(f'loaded cached tables ({int(seen.sum())} vocab rows seen)', flush=True)
else:
    acc = {n: torch.zeros(V, NH * HD, device=DEV) for n in NAMES}
    cnt = torch.zeros(V, device=DEV)
    for i in range(0, len(TRAIN), 8):
        capture(TRAIN[i:i + 8, :-1].to(DEV), acc, cnt)
        if i % 256 == 0:
            print(f'  estimate {i}/{len(TRAIN)}', flush=True)
    seen = cnt > 0
    gmean = {n: (a.sum(0) / cnt.sum()) for n, a in acc.items()}
    tables = {}
    for n, a in acc.items():
        t = a / cnt.clamp_min(1)[:, None]
        t[~seen] = gmean[n]
        tables[n] = t.view(V, NH, HD).contiguous()
    del acc
    torch.cuda.empty_cache()
    torch.save({**{n: t.cpu() for n, t in tables.items()}, 'seen': seen.cpu()}, CACHE)
    print(f'estimated + cached ({int(seen.sum())} vocab rows seen)', flush=True)

TAB = {n: unit_rms(tables[n]) for n in NAMES}          # unit-RMS gauge (the established winner)
cov = seen[AUDIT[:, :-1].reshape(-1).to(DEV)].float().mean().item()
print(f'audit-token coverage {cov:.4f}', flush=True)


# ------------------------------------------------------------------ audit (binding metric)

@torch.no_grad()
def audit_ce(tabs=None, zero=False):
    tot, n = 0.0, 0
    for i in range(0, len(AUDIT), 4):
        b = AUDIT[i:i + 4].to(DEV)
        idx = b[:, :-1]

        def patch(li, s1, s2):
            if li != 1:
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


def apply_global_assign(assign_seen, sidx, renorm):
    """Replace each seen token's rows by its class centroid (per head-branch)."""
    out = {n: TAB[n].clone() for n in NAMES}
    K = int(assign_seen.max()) + 1
    for h in range(NH):
        for (qn, kn) in BRANCHES:
            X = torch.cat([TAB[qn][sidx, h], TAB[kn][sidx, h]], 1)
            C = torch.zeros(K, ROW, device=DEV)
            c2 = torch.zeros(K, device=DEV)
            C.index_add_(0, assign_seen, X)
            c2.index_add_(0, assign_seen, torch.ones(len(X), device=DEV))
            nz = c2 > 0
            C[nz] /= c2[nz][:, None]
            rec = C[assign_seen]
            out[qn][sidx, h] = rec[:, :HD]
            out[kn][sidx, h] = rec[:, HD:]
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
def bits_global(K, n_index):
    return 32 * (NHB * K * ROW) + n_index * math.log2(max(K, 2))


def bits_perhb(K, n_index):
    return 32 * (NHB * K * ROW) + NHB * n_index * math.log2(max(K, 2))


BITS_RAW = 32 * NHB * V * ROW


# ------------------------------------------------------------------ run

res = {'coverage': cov, 'n_seen': int(seen.sum()), 'raw_Mbits': round(BITS_RAW / 1e6, 1), 'arms': {}}
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


# controls + the floor
dz = audit_ce(zero=True) - CE0
res['arms']['zero-scores (control)'] = {'dce': round(dz, 4)}
print(f'{"zero-scores (control)":44s} dCE {dz:+.4f}', flush=True)
FLOOR = run('FLOOR: cond-mean unit-RMS, no merge', TAB, BITS_RAW)
res['floor_dce'] = round(FLOOR, 4)

# stage-1 GLOBAL merge (the real "these tokens are the same token" reduction)
sidx = seen.nonzero().squeeze(1)
Xall = rows_all()[sidx]
Xc = Xall - Xall.mean(0)
U, S, Vh = torch.linalg.svd(Xc.double(), full_matrices=False)
Xp = (Xc.double() @ Vh[:PCA_DIM].T).float()            # PCA-reduced merge space
print(f'global merge space: {tuple(Xp.shape)} (top-{PCA_DIM} of {Xall.shape[1]}; '
      f'{100*float(S[:PCA_DIM].pow(2).sum()/S.pow(2).sum()):.1f}% variance)', flush=True)
del Xall, Xc, U, S, Vh
torch.cuda.empty_cache()

for K in K_GRID:
    a_s, _ = kmeans(Xp, K, seed=0)
    for renorm in (False, True):
        tag = f'stage1 GLOBAL merge K={K}' + (' +unit-RMS' if renorm else '')
        run(tag, apply_global_assign(a_s, sidx, renorm), bits_global(K, len(sidx)),
            extra={'K': K, 'renorm': renorm})

# per-head-branch upper bound (18 partitions, 18x index bits)
for K in K_GRID:
    run(f'per-head-branch merge K={K} +unit-RMS', apply_perhb_assign(K, True),
        bits_perhb(K, V), extra={'K': K})

json.dump(res, open(OUT, 'w'), indent=2)
print(f'\nwrote {OUT}', flush=True)
