"""TICK 195: diagnose the l1-h1 pathology (moment residual > 1, diverging with
capacity — unlike any layer-0 failure).

Hypothesis: token-conditional mean rows are noise-dominated for rarely-seen tokens
(a once-seen token's "mean" is one contextual draw), the third moment cubes the noise,
and h1 — the most context-dependent head — amplifies it most.

Checks (cheap, one script):
  1. Per-head row-norm p99/median vs occurrence count buckets (are h1's rare-token rows
     outsized?).
  2. Moment gate recomputed on the m=512 SAE restricted to tokens with count >= {4, 16}
     (if h1 gates there, the failure is estimation noise, not head structure).
  3. Shrinkage tables: mean_x_shrunk = cnt/(cnt+tau) * mean + tau/(cnt+tau) * fallback
     (fallback = embedding row), tau = 8; retrain h1 at (1024, 8) and re-gate.
  4. Per-head port cost: patch ONE l1 head's pattern with token tables (others exact),
     full audit — decomposes tick-193's total +0.027 and says how much is h1's
     context-dependence.
"""
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, reference_forward, rope_tables, apply_rot
from tier2_folding import scores_from_factors

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
STEPS, BATCH, LR, GATE, N_EST, TAU = 12000, 2048, 3e-3, 0.05, 1024, 8.0

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
QP = (torch.bincount(FINEWEB.flatten(), minlength=V).float() + 0.5).to(DEV)
QP = QP / QP.sum()
QP_CPU = QP.cpu()


@torch.no_grad()
def block1_input(idx):
    dt = m.transformer.wte.weight.dtype
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    B, T = idx.shape
    blk = m.transformer.h[0]
    x = blk.lambdas[0] * x + blk.lambdas[1] * x0
    a = blk.attn
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
    x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
    blk1 = m.transformer.h[1]
    return blk1.lambdas[0] * x + blk1.lambdas[1] * x0


print('estimating means (with counts)...', flush=True)
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
shrunk = (cnt / (cnt + TAU))[:, None] * mean_x + (TAU / (cnt + TAU))[:, None] * wte
del sum_x
torch.cuda.empty_cache()

a0 = m.transformer.h[0].attn
a1 = m.transformer.h[1].attn
Vv0 = a0.c_v(F.rms_norm(wte, (D,))).view(V, NH, HD).float()


def tables_from(mx):
    with torch.no_grad():
        xn = F.rms_norm(mx, (D,))
        T1 = {}
        for name, lin in (('q1', a1.c_q), ('k1', a1.c_k), ('q2', a1.c_q2), ('k2', a1.c_k2)):
            z = lin(xn).view(V, NH, HD).float()
            T1[name] = F.rms_norm(z, (HD,)).contiguous()
        v_new = a1.c_v(xn).view(V, NH, HD).float()
        T1['v'] = ((1 - a1.lamb) * v_new + a1.lamb * Vv0.view_as(v_new)).contiguous()
    return T1


T1 = tables_from(mean_x)
T1s = tables_from(shrunk)

# ---- 1. row norms by count bucket ----
out = {}
print('=== rows: p99 norm by count bucket (head-space [k1|k2|v] rows) ===', flush=True)
buckets = [(0, 1), (1, 4), (4, 16), (16, 1e9)]
norm_diag = {}
for h in range(NH):
    Y = torch.cat([T1['k1'][:, h], T1['k2'][:, h], T1['v'][:, h]], 1)
    nrm = Y.norm(dim=1)
    row = []
    for lo, hi in buckets:
        sel = (cnt >= lo) & (cnt < hi)
        row.append(round(float(nrm[sel].quantile(0.99)), 2) if sel.any() else None)
    norm_diag[f'h{h}'] = row
    print(f'  h{h}: p99 by count [0,1)/[1,4)/[4,16)/[16+): {row}', flush=True)
out['p99_norm_by_count'] = norm_diag


def train_triple(Y, m_atoms, k_code, seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    Dm = Y[torch.randperm(len(Y), generator=g)[:m_atoms].to(DEV)].clone()
    Dm = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
    We = Dm.clone()
    b = (Y * QP[:, None]).sum(0).clone()
    for t in (Dm, We, b):
        t.requires_grad_(True)
    opt = torch.optim.Adam([Dm, We, b], lr=LR)
    fired = torch.zeros(m_atoms, device=DEV)
    for step in range(STEPS):
        kk = max(k_code, int(round(2 * k_code - k_code * min(1.0, 2 * step / STEPS))))
        bi = torch.multinomial(QP_CPU, BATCH, replacement=True, generator=g).to(DEV)
        y = Y[bi]
        Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
        z = torch.relu((y - b) @ We.T)
        vals, idx = z.topk(kk, dim=1)
        yhat = b + (vals.unsqueeze(-1) * Dn[idx]).sum(1)
        fired.index_add_(0, idx.reshape(-1), (vals > 1e-8).float().reshape(-1))
        loss = ((yhat - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 500 == 0:
            dead = (fired == 0).nonzero().squeeze(1)
            if len(dead):
                with torch.no_grad():
                    Dn_ = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
                    z_ = torch.relu((Y - b) @ We.T)
                    v_, i_ = z_.topk(k_code, dim=1)
                    rec = b + (v_.unsqueeze(-1) * Dn_[i_]).sum(1)
                    worst = ((rec - Y) ** 2).sum(1).topk(len(dead)).indices
                    Dm.data[dead] = Y[worst] / Y[worst].norm(dim=1, keepdim=True).clamp(min=1e-8)
                    We.data[dead] = Dm.data[dead]
                    del z_, rec
            fired.zero_()
    with torch.no_grad():
        Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
        z = torch.relu((Y - b) @ We.T)
        vals, idx = z.topk(k_code, dim=1)
        rec = b + (vals.unsqueeze(-1) * Dn[idx]).sum(1)
    return rec


@torch.no_grad()
def moment_residual(Y, rec, sel=None, n_probe=256, seed=3):
    g = torch.Generator(device='cpu').manual_seed(seed)
    w = QP.clone()
    if sel is not None:
        w = w * sel.float()
        w = w / w.sum()
    num = den = 0.0
    for _ in range(n_probe):
        u, v_, wv = (torch.randn(Y.shape[1], generator=g).to(DEV) for _ in range(3))
        t = (w * (Y @ u) * (Y @ v_) * (Y @ wv)).sum()
        th = (w * (rec @ u) * (rec @ v_) * (rec @ wv)).sum()
        num += float((t - th) ** 2)
        den += float(t ** 2)
    return num / max(den, 1e-30)


# ---- 2 & 3: restricted gate + shrinkage retrain for h1 ----
Y1 = torch.cat([T1['k1'][:, 1], T1['k2'][:, 1], T1['v'][:, 1]], 1)
rec1 = train_triple(Y1, 512, 6)
for thr in (4, 16):
    mres = moment_residual(Y1, rec1, sel=cnt >= thr)
    out[f'h1_gate_cnt_ge_{thr}'] = round(mres, 4)
    print(f'h1 (raw means) gate restricted to count>={thr}: {mres:.4f}', flush=True)
Y1s = torch.cat([T1s['k1'][:, 1], T1s['k2'][:, 1], T1s['v'][:, 1]], 1)
rec1s = train_triple(Y1s, 1024, 8)
mres_s = moment_residual(Y1s, rec1s)
out['h1_shrunk_gate'] = round(mres_s, 4)
print(f'h1 (shrunk tau={TAU}) m=1024 k=8 gate: {mres_s:.4f} '
      f'{"PASS" if mres_s < GATE else "FAIL"}', flush=True)
json.dump(out, open(f'{QK}/qk_l1_h1_diag.json', 'w'), indent=2)

# ---- 4. per-head port cost ----
BASE = 3.07630


@torch.no_grad()
def audit_head_tables(h, tabs, batch=4):
    tot, n = 0.0, 0
    for i in range(0, len(FINEWEB), batch):
        b = FINEWEB[i:i + batch].to(DEV)
        idx = b[:, :-1]

        def patch(li, s1, s2):
            if li != 1:
                return s1, s2
            n1 = scores_from_factors(tabs['q1'], tabs['k1'], idx, HD)
            n2 = scores_from_factors(tabs['q2'], tabs['k2'], idx, HD)
            s1 = s1.clone()
            s2 = s2.clone()
            s1[:, h] = n1[:, h].to(s1.dtype)
            s2[:, h] = n2[:, h].to(s2.dtype)
            return s1, s2

        logits = reference_forward(m, idx, 'bf16', score_patch=patch).float()
        ce = F.cross_entropy(logits.reshape(-1, V), b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n


for h in range(NH):
    d = audit_head_tables(h, T1) - BASE
    out[f'port_cost_h{h}'] = round(d, 5)
    print(f'l1 h{h} token-table port cost (alone): {d:+.5f}', flush=True)
    json.dump(out, open(f'{QK}/qk_l1_h1_diag.json', 'w'), indent=2)
print('L1 H1 DIAG DONE', flush=True)
