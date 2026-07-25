"""TICK 201 (Logan un-gated the MLP): block-0 Bilinear MLP reconnaissance.

The MLP is out = Down(Left(h) * Right(h)) with 4608 neurons — ALREADY a CP-form
third-order tensor T[d,a,b] = sum_j Down[d,j] L[j,a] R[j,b] of rank 4608. Recon:

(1) NEURON USAGE on data: activations act_j = (L_j.h)(R_j.h) over 64 held-out
    documents; usage u_j = E[|act_j|] * ||Down[:,j]|| (write-weighted). Spectrum.
(2) NEURON-COUNT FUNCTION FRONTIER: zero all but the top-n usage-ranked neurons
    (Down columns), full 307k audit, n in {2048, 1024, 512, 256, 128}. The MLP
    analog of the attention capacity frontier (dCE vs neuron count).
(3) NEXT-CIRCUIT READER MATRIX (Logan's idea, weight-space first pass): for each
    layer-1 head and each of q1/k1/q2/k2, the read map A = W_head @ Down (128 x 4608);
    per-neuron read strength = column norm. Report per (head, map): effective neuron
    count (sum s)^2 / sum s^2 and top-256 share — is the MLP->l1-QK channel sparse in
    neurons? Plus overlap of top-256 neuron sets between heads.
(4) PREVIOUS-CIRCUIT SPLIT per neuron (data): split-half token-identity R^2 of each
    neuron's activation (tokens seen >= 4) — fraction of neurons that are mostly
    token-driven vs context-driven, connecting to the tick-200 finding that the MLP
    authors layer-1's context-dependence.
"""
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, reference_forward, rope_tables, apply_rot

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
N_CAP = 64

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))

blk0 = m.transformer.h[0]
mlp = blk0.mlp
Lw = mlp.Left.weight.detach().float()          # (4608, 1152)
Rw = mlp.Right.weight.detach().float()
Dw = mlp.Down.weight.detach().float()          # (1152, 4608)
NJ = Lw.shape[0]
print(f'MLP: {NJ} neurons', flush=True)


@torch.no_grad()
def mlp_input(idx):
    """h = rms_norm(x after block-0 attention) — the exact MLP input."""
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
    return F.rms_norm(x, (x.size(-1),))


# ---- (1) usage + (4) token R^2 capture ----
print('capturing activations...', flush=True)
u_abs = torch.zeros(NJ, device=DEV)
u_sq = torch.zeros(NJ, device=DEV)
ACT_A, ACT_B, IDS_A, IDS_B = [], [], [], []
npos = 0
with torch.no_grad():
    for i in range(0, N_CAP, 4):
        b = COOC[i:i + 4].to(DEV)
        idx = b[:, :-1]
        h = mlp_input(idx).reshape(-1, D)
        act = (h @ Lw.T) * (h @ Rw.T)                     # (P, 4608)
        u_abs += act.abs().sum(0)
        u_sq += (act ** 2).sum(0)
        npos += act.shape[0]
        half = act.shape[0] // 2
        ACT_A.append(act[:half].half().cpu())
        ACT_B.append(act[half:].half().cpu())
        IDS_A.append(idx.reshape(-1)[:half].cpu())
        IDS_B.append(idx.reshape(-1)[half:].cpu())
        del h, act
u_abs /= npos
usage = u_abs * Dw.norm(dim=0).to(DEV)
order = usage.argsort(descending=True)
spec = usage.sort(descending=True).values
out = {'n_neurons': NJ,
       'usage_spectrum': {str(n): round(float(spec[:n].sum() / spec.sum()), 4)
                          for n in (128, 256, 512, 1024, 2048)}}
print('usage share of top-n:', out['usage_spectrum'], flush=True)

# ---- (4) token-identity R^2 per neuron (split-half) ----
A = torch.cat(ACT_A).float()
Bh = torch.cat(ACT_B).float()
ia = torch.cat(IDS_A)
ib = torch.cat(IDS_B)
cnt_a = torch.bincount(ia, minlength=V).float()
ok = cnt_a >= 4
sum_a = torch.zeros(V, NJ)
sum_a.index_add_(0, ia, A)
mean_a = sum_a / cnt_a[:, None].clamp_min(1)
selb = ok[ib]
pred = mean_a[ib[selb]]
resid = Bh[selb] - pred
var = Bh[selb].var(0).clamp_min(1e-12)
r2_tok = (1 - (resid ** 2).mean(0) / var).clamp(-1, 1)
out['token_r2'] = {'median': round(float(r2_tok.median()), 3),
                   'frac_gt_0.5': round(float((r2_tok > 0.5).float().mean()), 3),
                   'frac_lt_0.1': round(float((r2_tok < 0.1).float().mean()), 3)}
print(f'token-identity R2: median {out["token_r2"]["median"]}, '
      f'>0.5 {out["token_r2"]["frac_gt_0.5"]}, <0.1 {out["token_r2"]["frac_lt_0.1"]}',
      flush=True)
del A, Bh, sum_a, mean_a, pred, resid
json.dump(out, open(f'{QK}/qk_mlp_recon.json', 'w'), indent=2)

# ---- (3) next-circuit reader matrix ----
a1 = m.transformer.h[1].attn
readers = {}
tops = {}
for mapname, lin in (('q1', a1.c_q), ('k1', a1.c_k), ('q2', a1.c_q2), ('k2', a1.c_k2)):
    W = lin.weight.detach().float()
    for h in range(NH):
        Ah = W[h * HD:(h + 1) * HD] @ Dw                  # (128, 4608)
        s = Ah.norm(dim=0)
        eff = float(s.sum() ** 2 / (s ** 2).sum())
        top256 = float(s.sort(descending=True).values[:256].sum() / s.sum())
        readers[f'{mapname}_h{h}'] = {'eff_neurons': round(eff, 1),
                                      'top256_share': round(top256, 3)}
        if mapname == 'k1':
            tops[h] = set(s.argsort(descending=True)[:256].tolist())
ov = np.zeros((NH, NH))
for i in range(NH):
    for j in range(NH):
        ov[i, j] = len(tops[i] & tops[j]) / 256
out['readers'] = readers
out['k1_top256_overlap_offdiag_mean'] = round(float(
    (ov.sum() - np.trace(ov)) / (NH * NH - NH)), 3)
effs = [v['eff_neurons'] for v in readers.values()]
print(f'reader effective neurons: min {min(effs):.0f} median {np.median(effs):.0f} '
      f'max {max(effs):.0f} (of {NJ}); k1 top-256 overlap (offdiag) '
      f'{out["k1_top256_overlap_offdiag_mean"]}', flush=True)
json.dump(out, open(f'{QK}/qk_mlp_recon.json', 'w'), indent=2)

# ---- (2) neuron-count function frontier ----
BASE = 3.07630


@torch.no_grad()
def audit(batch=4):
    tot, n = 0.0, 0
    for i in range(0, len(FINEWEB), batch):
        b = FINEWEB[i:i + batch].to(DEV)
        idx = b[:, :-1]
        logits = reference_forward(m, idx, 'bf16').float()
        ce = F.cross_entropy(logits.reshape(-1, V), b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n


backup = mlp.Down.weight.data.clone()
for n_keep in (2048, 1024, 512, 256, 128):
    mask = torch.zeros(NJ, dtype=torch.bool, device=mlp.Down.weight.device)
    mask[order[:n_keep]] = True
    mlp.Down.weight.data = backup * mask[None, :].to(backup.dtype)
    ce = audit()
    out[f'prune_keep_{n_keep}_dce'] = round(ce - BASE, 5)
    print(f'keep top-{n_keep} neurons: dCE {ce - BASE:+.5f}', flush=True)
    json.dump(out, open(f'{QK}/qk_mlp_recon.json', 'w'), indent=2)
mlp.Down.weight.data = backup
print('MLP RECON DONE', flush=True)
