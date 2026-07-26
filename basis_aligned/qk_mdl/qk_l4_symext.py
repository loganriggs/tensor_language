"""TICK 255: LAYER-4 pattern with the 528-symbol dictionary (l4 fold).

Codes per position (384 numbers): 96 embedding code (rms-normed wte onto top-96
PCs, exact from token id) + 144 layer-0 archetype activations (per-head attention
outputs onto archetype value-directions, the named-basis construction) + 144
layer-1 activations (per-head attention outputs onto per-head 16-dim PCA bases).
Generator: ridge linear decoder codes -> block-2 attention input (x2), fit on the
disjoint co-occurrence corpus. Test: replace layer-2's pattern scores with scores
computed from the SYNTHETIC residual x2_hat (through block 2's own projections,
per-head rms, rotary), audit on the standard 307k set.
Gates: token tables +0.02777, layer zeroed +0.39042 (tick 247, same audit).
If 384 compositional numbers approach or beat the 50304xD token table, the fold
recurses over symbols and the dictionary-growth scaling plan is live.
"""
import json, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
wte = m.transformer.wte.weight.detach().float().to(DEV)

# archetype value-directions for layer-0 heads (named-basis construction)
mh_pt2 = torch.load(f'{QK}/qk_minimal_heads.pt', map_location=DEV)
pol0 = torch.load(f'{QK}/qk_h0_polish_g025.pt', map_location=DEV)
pol4 = torch.load(f'{QK}/qk_h04_polish.pt', map_location=DEV)
PA = {}
for hh in range(NH):
    if hh in (0, 4):
        bb = pol0 if hh == 0 else pol4
        Dv = bb[f'h{hh}_v_Dm'].to(DEV)
        Dv = Dv / Dv.norm(dim=1, keepdim=True).clamp_min(1e-8)
        Vdir = Dv.T @ bb[f'h{hh}_CJ'][:, :16].to(DEV)
    else:
        P = mh_pt2[f'h{hh}']
        Dn_ = P['Dm'].to(DEV)
        Dn_ = Dn_ / Dn_.norm(dim=1, keepdim=True).clamp_min(1e-8)
        Vdir = Dn_[:, 2 * HD:].T @ P['U'].to(DEV)[:, :16]
    if Vdir.shape[1] < 16:
        Vdir = torch.cat([Vdir, torch.zeros(HD, 16 - Vdir.shape[1], device=DEV)], 1)
    PA[hh] = (Vdir / Vdir.norm(dim=0, keepdim=True).clamp_min(1e-9)).contiguous()
EPC96 = torch.linalg.svd(F.rms_norm(wte, (D,)) - F.rms_norm(wte, (D,)).mean(0),
                         full_matrices=False).Vh[:96].T.contiguous()
EW = F.rms_norm(wte, (D,)) @ EPC96


@torch.no_grad()
def blocks01(idx):
    """Run blocks 0 and 1; return (yh0 (B,T,NH,HD), yh1, x2_input)."""
    dt = m.transformer.wte.weight.dtype
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    v1 = None
    B, T = idx.shape
    cos, sin = rope_tables(T, HD, idx.device, dt, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
    yhs = []
    for li in (0, 1, 2, 3):
        blk = m.transformer.h[li]
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn
        hcur = F.rms_norm(x, (x.size(-1),))

        def qk(lin):
            z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,))
            return apply_rot(z, cosb, sinb)

        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        q, k = qk(a.c_q), qk(a.c_k)
        q2, k2 = qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        yhs.append(yh4)
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
    blk4 = m.transformer.h[4]
    return yhs[0], yhs[1], yhs[2], blk4.lambdas[0] * x + blk4.lambdas[1] * x0


# ---- pass 1 over cooc: per-head l1 PCA bases + code means/stds + ridge stats ----
# first sub-pass: accumulate l1 per-head covariance and l0/l1 head means
cov1 = torch.zeros(NH, HD, HD, device=DEV, dtype=torch.float64)
mu1 = torch.zeros(NH, HD, device=DEV, dtype=torch.float64)
mu0 = torch.zeros(NH, HD, device=DEV, dtype=torch.float64)
cov2 = torch.zeros(NH, HD, HD, device=DEV, dtype=torch.float64)
mu2 = torch.zeros(NH, HD, device=DEV, dtype=torch.float64)
n_tot = 0
with torch.no_grad():
    for i in range(0, 256, 4):
        b = COOC[i:i + 4].to(DEV)
        y0, y1, y2, _ = blocks01(b[:, :-1])
        f1 = y1.reshape(-1, NH, HD).double()
        f2 = y2.reshape(-1, NH, HD).double()
        mu1 += f1.sum(0)
        mu2 += f2.sum(0)
        mu0 += y0.reshape(-1, NH, HD).double().sum(0)
        cov1 += torch.einsum('nhd,nhe->hde', f1, f1)
        cov2 += torch.einsum('nhd,nhe->hde', f2, f2)
        n_tot += f1.shape[0]
mu1 /= n_tot
mu2 /= n_tot
mu0 /= n_tot
cov1 = cov1 / n_tot - torch.einsum('hd,he->hde', mu1, mu1)
cov2 = cov2 / n_tot - torch.einsum('hd,he->hde', mu2, mu2)
PB, PB2 = {}, {}
for h in range(NH):
    ev, evec = torch.linalg.eigh(cov1[h])
    PB[h] = evec[:, ev.argsort(descending=True)[:16]].float().contiguous()
    ev2, evec2 = torch.linalg.eigh(cov2[h])
    PB2[h] = evec2[:, ev2.argsort(descending=True)[:16]].float().contiguous()
print('l1 bases ready', flush=True)


@torch.no_grad()
def codes_of(idx, y0, y1, y2):
    B, T = idx.shape
    ce = EW[idx.reshape(-1)]
    c0 = torch.cat([(y0[..., h, :].reshape(-1, HD) - mu0[h].float()) @ PA[h]
                    for h in range(NH)], 1)
    c1 = torch.cat([(y1[..., h, :].reshape(-1, HD) - mu1[h].float()) @ PB[h]
                    for h in range(NH)], 1)
    c2 = torch.cat([(y2[..., h, :].reshape(-1, HD) - mu2[h].float()) @ PB2[h]
                    for h in range(NH)], 1)
    return torch.cat([ce, c0, c1, c2], 1)                      # (B*T, 384)


# second sub-pass: ridge accumulation codes -> x2
KDIM = 96 + 144 + 144 + 144
AtA = torch.zeros(KDIM + 1, KDIM + 1, device=DEV, dtype=torch.float64)
AtY = torch.zeros(KDIM + 1, D, device=DEV, dtype=torch.float64)
ss_y = 0.0
n_fit = 0
with torch.no_grad():
    for i in range(0, 512, 4):
        b = COOC[i:i + 4].to(DEV)
        y0, y1, y2, x2 = blocks01(b[:, :-1])
        Cd = codes_of(b[:, :-1], y0, y1, y2).double()
        Cd = torch.cat([Cd, torch.ones(Cd.shape[0], 1, device=DEV, dtype=torch.float64)], 1)
        Y = x2.reshape(-1, D).double()
        AtA += Cd.T @ Cd
        AtY += Cd.T @ Y
        ss_y += float((Y - Y.mean(0)).pow(2).sum())
        n_fit += Y.shape[0]
W = torch.linalg.solve(AtA + 10.0 * torch.eye(KDIM + 1, device=DEV, dtype=torch.float64), AtY)
# fit R2 on a held-out cooc slice
with torch.no_grad():
    b = COOC[512:516].to(DEV)
    y0, y1, y2, x2 = blocks01(b[:, :-1])
    Cd = codes_of(b[:, :-1], y0, y1, y2).double()
    Cd = torch.cat([Cd, torch.ones(Cd.shape[0], 1, device=DEV, dtype=torch.float64)], 1)
    Y = x2.reshape(-1, D).double()
    R2 = 1 - float((Cd @ W - Y).pow(2).sum() / (Y - Y.mean(0)).pow(2).sum())
print(f'decoder held-out R2 {R2:.4f}', flush=True)
Wf = W.float()


@torch.no_grad()
def forward_symbolic(idx):
    """Full forward; layer-2 scores computed from the symbol-generated residual."""
    dt = m.transformer.wte.weight.dtype
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    v1 = None
    B, T = idx.shape
    cos, sin = rope_tables(T, HD, idx.device, dt, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
    yhs = {}
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn
        hcur = F.rms_norm(x, (x.size(-1),))

        def qk_of(lin, src):
            z = F.rms_norm(lin(src).view(B, T, NH, HD), (HD,))
            return apply_rot(z, cosb, sinb)

        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        if li == 4:
            Cd = codes_of(idx, yhs[0], yhs[1], yhs[2])
            Cd = torch.cat([Cd, torch.ones(Cd.shape[0], 1, device=DEV)], 1)
            x2h = (Cd @ Wf).view(B, T, D)
            hsyn = F.rms_norm(x2h, (D,)).to(hcur.dtype)
            q, k = qk_of(a.c_q, hsyn), qk_of(a.c_k, hsyn)
            q2, k2 = qk_of(a.c_q2, hsyn), qk_of(a.c_k2, hsyn)
        else:
            q, k = qk_of(a.c_q, hcur), qk_of(a.c_k, hcur)
            q2, k2 = qk_of(a.c_q2, hcur), qk_of(a.c_k2, hcur)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        if li in (0, 1, 2):
            yhs[li] = yh4
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
    x = F.rms_norm(x, (x.size(-1),))
    return 30 * torch.tanh(m.lm_head(x) / 30)


BASE = 3.07630
tot, n = 0.0, 0
with torch.no_grad():
    for i in range(0, len(FINEWEB), 4):
        b = FINEWEB[i:i + 4].to(DEV)
        lg = forward_symbolic(b[:, :-1]).float()
        ce = F.cross_entropy(lg.reshape(-1, V), b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
ce = tot / n
out = {'decoder_R2': round(R2, 4), 'l4_symext_dce': round(ce - BASE, 5),
       'gates': {'tables': 0.04735, 'zero': 0.34771}}
print(f'layer-4 symbol-generated pattern: dCE {ce - BASE:+.5f} '
      f'(tables +0.04735, zero +0.34771)', flush=True)
json.dump(out, open(f'{QK}/qk_l4_symext.json', 'w'), indent=2)
print('L4 SYMEXT DONE', flush=True)
