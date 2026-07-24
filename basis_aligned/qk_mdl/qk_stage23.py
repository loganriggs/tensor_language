"""SPEC STAGES 2-3 (tick 173): sparse symmetric third-moment core + nonnegative symmetric
CP, per head, on the MECHANISM ledger (separate from function-MDL).

Inputs: tick-172 Stage-1 winner codes (unigram + nonneg, m=512, k=6) from qk_stage1_triple.pt.
Heads 0 and 4 failed the moment gate at m=512 -> refit here at m=1024, k=8 and re-gate;
included only if they pass.

Stage 2: dense symmetric core per head, M_abc = sum_t p_t s_ta s_tb s_tc (m^3 fp32, built
from the 6 (or 8) active slots per token via all ordered slot-triples; ~11M scattered adds).
Report diagonal vs off-diagonal mass split (spec: a diagonal-dominated core means Stage 3
is factoring salience, not interaction).

Stage 3: symmetric nonnegative CP by projected ALS (tied factor, Hadamard normal equations,
ridge 1e-6, clamp >= 0), ranks {8, 16, 32, 64}, 5 restarts each; report best rel-Frobenius
fit and restart stability (mean best-pair |cos| of matched archetypes across restarts).
Permutation null (spec check 3): rebuild the core from column-permuted codes, fit CP at the
middle rank, compare fit quality. Interpretability dump: top-8 tokens by code-loading for
the top-5 archetypes per head.
"""
import json
import sys
import torch
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
import numpy as np
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('gpt2')

blob = torch.load(f'{QK}/qk_stage1_triple.pt', map_location=DEV)
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
V = 50304
QP = (torch.bincount(FINEWEB.flatten(), minlength=V).float() + 0.5).to(DEV)
QP = QP / QP.sum()
PASS_HEADS = [1, 2, 3, 5, 6, 7, 8]
RANKS = (8, 16, 32, 64)


def build_core(idx, coeff, m):
    """Dense symmetric core (m^3 flat) from sparse codes, p-weighted."""
    k = idx.shape[1]
    core = torch.zeros(m * m * m, device=DEV)
    w = QP[:, None] * coeff                                    # (V, k) p_t * s
    for i in range(k):
        for j in range(k):
            keys = (idx[:, i].long() * m + idx[:, j].long()) * m
            vals = w[:, i] * coeff[:, j]
            for l in range(k):
                core.scatter_add_(0, keys + idx[:, l].long(), vals * coeff[:, l])
    return core.view(m, m, m)


def cp_fit(core_raw, R, seed, iters=200, ridge=1e-6):
    m = core_raw.shape[0]
    g = torch.Generator(device='cpu').manual_seed(seed)
    scale = core_raw.norm().clamp_min(1e-30)
    core = core_raw / scale                                    # fit scale-free
    diag_fibers = torch.stack([core[i, i, i] for i in range(m)])
    top = diag_fibers.abs().argsort(descending=True)[:R]
    A = torch.stack([core[:, a, a].clamp_min(0) for a in top.tolist()], 1)
    A = A / A.norm(dim=0, keepdim=True).clamp_min(1e-12)
    A = (A + 0.05 * torch.rand(m, R, generator=g).to(DEV)).clamp_min(0)
    M1 = core.reshape(m, m * m)
    nrm2 = float((core ** 2).sum())
    for _ in range(iters):
        KR = (A[:, None, :] * A[None, :, :]).reshape(m * m, R)
        G = M1 @ KR                                            # (m, R)
        H = (A.T @ A) ** 2
        jit = 1e-6 * float(H.diagonal().mean()) + 1e-12
        H = H + jit * torch.eye(R, device=DEV)
        A = torch.linalg.solve(H, G.T).T.clamp_min(0)
        dead = A.sum(0) == 0
        if bool(dead.any()):
            A[:, dead] = torch.rand(m, int(dead.sum()), generator=g).to(DEV) * 0.1
    lam = A.norm(dim=0).clamp_min(1e-12)
    U = A / lam[None, :]
    lam3 = lam ** 3
    Gu = U.T @ U
    cross = 0.0
    for r in range(A.shape[1]):
        uu = (U[:, r][:, None] * U[:, r][None, :]).reshape(-1)
        cross += float(lam3[r] * (M1 @ uu) @ U[:, r])
    fit2 = nrm2 - 2 * cross + float((lam3[:, None] * lam3[None, :] * Gu ** 3).sum())
    return U, lam3, (max(fit2, 0.0) / max(nrm2, 1e-30)) ** 0.5


def stability(Us):
    vals = []
    for i in range(len(Us)):
        for j in range(i + 1, len(Us)):
            C = (Us[i].T @ Us[j]).abs()
            vals.append(float(C.max(1).values.mean()))
    return sum(vals) / len(vals)


results = {}
for h in PASS_HEADS:
    key = f'h{h}_unigram_nonneg'
    idx = blob[f'{key}_idx'].long().to(DEV)
    coeff = blob[f'{key}_coeff'].to(DEV)
    m = 512
    core = build_core(idx, coeff, m)
    diag = float(sum(core[i, i, i] ** 2 for i in range(m)))
    tot = float((core ** 2).sum())
    row = {'diag_mass_frac': round(diag / tot, 4)}
    for R in RANKS:
        fits, Us = [], []
        for seed in range(5):
            U, lam, rel = cp_fit(core, R, seed)
            fits.append(rel)
            Us.append(U)
        best = int(torch.tensor(fits).argmin())
        row[f'R{R}_relerr'] = round(min(fits), 4)
        row[f'R{R}_stability'] = round(stability(Us), 3)
        if R == 32:
            U, lam, _ = cp_fit(core, R, best)
            S_dense = torch.zeros(V, m, device=DEV)
            S_dense.scatter_(1, idx, coeff)
            arch = []
            for r in lam.argsort(descending=True)[:5].tolist():
                load = S_dense @ U[:, r]
                top = load.argsort(descending=True)[:8]
                arch.append([tok.decode([t]).replace('\n', '\\n') for t in top.tolist()])
            row['top_archetype_tokens'] = arch
    # permutation null at R=32 (spec check 3: permute each feature COLUMN independently,
    # destroying within-token co-occurrence while preserving marginals)
    gp = torch.Generator().manual_seed(7)
    S_dense = torch.zeros(V, m, device=DEV)
    S_dense.scatter_(1, idx, coeff)
    for f in range(m):
        S_dense[:, f] = S_dense[torch.randperm(V, generator=gp).to(DEV), f]
    vals_n, idx_n = S_dense.topk(12, dim=1)                    # rows now ~Poisson(6); cap 12
    core_null = build_core(idx_n, vals_n, m)
    del S_dense
    _, _, rel_null = cp_fit(core_null, 32, 0)
    row['R32_relerr_null'] = round(rel_null, 4)
    results[f'h{h}'] = row
    print(f'h{h}: diag {row["diag_mass_frac"]} | ' +
          ' '.join(f'R{R}:{row[f"R{R}_relerr"]}(s{row[f"R{R}_stability"]})' for R in RANKS) +
          f' | null R32 {rel_null:.3f}', flush=True)
    json.dump(results, open(f'{QK}/qk_stage23.json', 'w'), indent=2)
    del core
    torch.cuda.empty_cache()
print('STAGE23 DONE', flush=True)
