"""TICK 204: the compact two-circuit object, audited end to end.

Model of layer 1's pattern factors: TOKEN TABLE + RANK-r CONTEXT ADAPTER —
  factor(position) = table[token] + U_r U_r^T (factor_true(position) - table[token])
per (map in q1/k1/q2/k2, head), with U_r the top-r PCA directions of the deviations
(estimated on 64 disjoint co-occurrence documents). r = 0 is the pure static-table
port (+0.027 expected); r = 128 is exact (0). The audit runs a full manual 18-layer
forward (replicating reference_forward) with layer 1's pre-rotary factors replaced by
the truncated form, on the standard 307k-prediction set, for r in {4, 16, 64}.

This prices Logan's circuit-relative decomposition: how many context dimensions per
factor channel buy how much of the +0.027 static gap — the MDL curve of the
context adapter (bits ~ r x 128 floats per channel + the runtime projection).
"""
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
N_EST, TAU, N_CAP = 1024, 8.0, 64

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
blk0 = m.transformer.h[0]
a1 = m.transformer.h[1].attn
MAPS = (('q1', a1.c_q), ('k1', a1.c_k), ('q2', a1.c_q2), ('k2', a1.c_k2))


@torch.no_grad()
def block1_input(idx):
    dt = m.transformer.wte.weight.dtype
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    B, T = idx.shape
    x = blk0.lambdas[0] * x + blk0.lambdas[1] * x0
    a = blk0.attn
    hcur = F.rms_norm(x, (x.size(-1),))
    cos, sin = rope_tables(T, HD, idx.device, dt, 'bf16')
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]

    def qk(lin):
        z = lin(hcur).view(B, T, NH, HD)
        return apply_rot(F.rms_norm(z, (HD,)), cos, sin)

    v = a.c_v(hcur).view(B, T, NH, HD)
    mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
    q, k = qk(a.c_q), qk(a.c_k)
    q2, k2 = qk(a.c_q2), qk(a.c_k2)
    s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
    s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
    pat = (s1 * s2).masked_fill(~mask, 0.0)
    y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1)
    x = x + a.c_proj(y)
    x = x + blk0.mlp(F.rms_norm(x, (x.size(-1),)))
    blk1 = m.transformer.h[1]
    return blk1.lambdas[0] * x + blk1.lambdas[1] * x0


# ---- shrunk tables ----
print('estimating shrunk means...', flush=True)
sum_x = torch.zeros(V, D, device=DEV)
cnt = torch.zeros(V, device=DEV)
with torch.no_grad():
    for i in range(0, N_EST, 4):
        b = COOC[i:i + 4].to(DEV)
        idx = b[:, :-1]
        x = block1_input(idx).float().reshape(-1, D)
        ids = idx.reshape(-1)
        sum_x.index_add_(0, ids, x)
        cnt.index_add_(0, ids, torch.ones_like(ids, dtype=torch.float))
wte = m.transformer.wte.weight.detach().float().to(DEV)
mean_x = torch.where((cnt > 0)[:, None], sum_x / cnt[:, None].clamp_min(1), wte)
shr = (cnt / (cnt + TAU))[:, None] * mean_x + (TAU / (cnt + TAU))[:, None] * wte
del sum_x, mean_x
TABLES = {}
with torch.no_grad():
    xn = F.rms_norm(shr, (D,))
    for name, lin in MAPS:
        TABLES[name] = F.rms_norm(lin(xn).view(V, NH, HD).float(), (HD,)).contiguous()
del shr, xn
torch.cuda.empty_cache()

# ---- deviation PCA per (map, head) ----
print('capturing deviations for PCA...', flush=True)
DEVS = {name: [] for name, _ in MAPS}
with torch.no_grad():
    for i in range(0, N_CAP, 4):
        b = COOC[i:i + 4].to(DEV)
        idx = b[:, :-1]
        xin1 = block1_input(idx)
        h1n = F.rms_norm(xin1, (D,))
        ids = idx.reshape(-1)
        for name, lin in MAPS:
            fa = F.rms_norm(lin(h1n).view(*idx.shape, NH, HD).float(), (HD,))
            DEVS[name].append((fa.reshape(-1, NH, HD) - TABLES[name][ids]).cpu())
UBANK = {}
for name, _ in MAPS:
    Dv = torch.cat(DEVS[name])
    for h in range(NH):
        X = Dv[:, h].to(DEV)
        X = X - X.mean(0)
        _, _, Vh = torch.linalg.svd(X, full_matrices=False)
        UBANK[(name, h)] = Vh[:64].T.contiguous()             # (128, 64) top-64 basis
        del X
    del Dv
    torch.cuda.empty_cache()
DEVS = None
print('PCA bases ready', flush=True)

# ---- full manual forward with truncated layer-1 factors ----


@torch.no_grad()
def forward_truncated(idx, rank):
    dt = m.transformer.wte.weight.dtype
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    v1 = None
    B, T = idx.shape
    cos, sin = rope_tables(T, HD, idx.device, dt, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn
        hcur = F.rms_norm(x, (x.size(-1),))

        def factors(lin, name=None):
            z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,))
            if li == 1 and rank is not None and name is not None:
                tab = TABLES[name][idx]                        # (B,T,NH,HD)
                dv = z.float() - tab
                if rank == 0:
                    z = tab
                else:
                    zc = tab.clone()
                    for h in range(NH):
                        U = UBANK[(name, h)][:, :rank]
                        zc[:, :, h] += (dv[:, :, h] @ U) @ U.T
                    z = zc
                z = z.to(hcur.dtype)
            return apply_rot(z, cosb, sinb)

        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        q, k = factors(a.c_q, 'q1'), factors(a.c_k, 'k1')
        q2, k2 = factors(a.c_q2, 'q2'), factors(a.c_k2, 'k2')
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1)
        x = x + a.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
    x = F.rms_norm(x, (x.size(-1),))
    logits = m.lm_head(x)
    return 30 * torch.tanh(logits / 30)


@torch.no_grad()
def audit(rank, batch=4):
    tot, n = 0.0, 0
    for i in range(0, len(FINEWEB), batch):
        b = FINEWEB[i:i + batch].to(DEV)
        idx = b[:, :-1]
        logits = forward_truncated(idx, rank).float()
        ce = F.cross_entropy(logits.reshape(-1, V), b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n


out = {}
base = audit(None)
out['base_ce_manual'] = round(base, 5)
print(f'manual-forward baseline CE {base:.5f} (reference 3.07630)', flush=True)
for r in (0, 4, 16, 64):
    ce = audit(r)
    out[f'rank{r}_dce'] = round(ce - base, 5)
    print(f'tables + rank-{r} context adapter: dCE {ce - base:+.5f}', flush=True)
    json.dump(out, open(f'{QK}/qk_l1_lowrank_ctx.json', 'w'), indent=2)
print('L1 LOWRANK CTX DONE', flush=True)
