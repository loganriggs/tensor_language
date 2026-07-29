"""Compositional decomposition of the BILINEAR MLP, layers 2-6. Parallel to the attention test:
regenerate each layer L's MLP INPUT (post-attention residual it reads, rms-normed) from a symbol
basis of preceding layers + THIS layer's attention output (96 embedding PCs + layer-0 archetypes +
per-head 16-d PCA of attention layers 1..L), then replace the MLP input with the symbol-generated
version and audit dCE. Baselines: per-token TABLE and RANDOM basis. Asks: how much of what the
bilinear MLP computes is driven by nameable preceding-layer symbols vs residual content the symbols
can't express (prior MLP writes / token carriage)?
"""
import json, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
wte = m.transformer.wte.weight.detach().float().to(DEV)
BASE = 3.07630; LAYERS = [2, 3, 4, 5, 6]; MAXL = max(LAYERS)  # need attn PCA for layers 1..MAXL

mh = torch.load(f'{QK}/qk_minimal_heads.pt', map_location=DEV)
pol0 = torch.load(f'{QK}/qk_h0_polish_g025.pt', map_location=DEV); pol4 = torch.load(f'{QK}/qk_h04_polish.pt', map_location=DEV)
PA = {}
for hh in range(NH):
    if hh in (0, 4):
        bb = pol0 if hh == 0 else pol4; Dv = bb[f'h{hh}_v_Dm'].to(DEV); Dv = Dv / Dv.norm(dim=1, keepdim=True).clamp_min(1e-8)
        Vdir = Dv.T @ bb[f'h{hh}_CJ'][:, :16].to(DEV)
    else:
        Pp = mh[f'h{hh}']; Dn_ = Pp['Dm'].to(DEV); Dn_ = Dn_ / Dn_.norm(dim=1, keepdim=True).clamp_min(1e-8)
        Vdir = Dn_[:, 2*HD:].T @ Pp['U'].to(DEV)[:, :16]
    if Vdir.shape[1] < 16: Vdir = torch.cat([Vdir, torch.zeros(HD, 16-Vdir.shape[1], device=DEV)], 1)
    PA[hh] = (Vdir / Vdir.norm(dim=0, keepdim=True).clamp_min(1e-9)).contiguous()
EPC96 = torch.linalg.svd(F.rms_norm(wte, (D,)) - F.rms_norm(wte, (D,)).mean(0), full_matrices=False).Vh[:96].T.contiguous()
EW = F.rms_norm(wte, (D,)) @ EPC96


@torch.no_grad()
def run(idx, upto):
    """return yh[li] attn out, and hmlp[li] = rms_norm(post-attention residual) MLP input."""
    dt = m.transformer.wte.weight.dtype; x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B, T = idx.shape
    cos, sin = rope_tables(T, HD, idx.device, dt, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool)); yh = {}; hmlp = {}
    for li in range(upto):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0
        a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v); yh[li] = yh4
        x = x + a.c_proj(yh4.reshape(B, T, -1)); hmlp[li] = F.rms_norm(x, (D,)); x = x + blk.mlp(hmlp[li])
    return yh, hmlp

# per-head PCA bases layers 1..MAXL (need current layer L too)
cov = {l: torch.zeros(NH, HD, HD, device=DEV, dtype=torch.float64) for l in range(MAXL+1)}
mu = {l: torch.zeros(NH, HD, device=DEV, dtype=torch.float64) for l in range(MAXL+1)}
nt = 0
for i in range(0, 256, 4):
    yh, _ = run(COOC[i:i+4].to(DEV)[:, :-1], MAXL+1)
    for l in range(MAXL+1):
        f = yh[l].reshape(-1, NH, HD).double(); mu[l] += f.sum(0); cov[l] += torch.einsum('nhd,nhe->hde', f, f)
    nt += yh[0].reshape(-1, NH, HD).shape[0]
for l in range(MAXL+1):
    mu[l] /= nt; cov[l] = cov[l]/nt - torch.einsum('hd,he->hde', mu[l], mu[l])
PB = {}
for l in range(1, MAXL+1):
    PB[l] = {}
    for h in range(NH):
        ev, evec = torch.linalg.eigh(cov[l][h]); PB[l][h] = evec[:, ev.argsort(descending=True)[:16]].float().contiguous()
g = torch.Generator(device=DEV).manual_seed(1)
def rb():
    Qr, _ = torch.linalg.qr(torch.randn(HD, HD, generator=g, device=DEV)); return Qr[:, :16].contiguous()
PAr = {h: rb() for h in range(NH)}; PBr = {l: {h: rb() for h in range(NH)} for l in range(1, MAXL+1)}
print('bases ready', flush=True)


def codes_mlp(L, idx, yh, named=True):
    """symbols for layer-L MLP input: embedding + layer-0 archetypes + attn layers 1..L."""
    ce = EW[idx.reshape(-1)]; pa = PA if named else PAr; pb = PB if named else PBr
    c = [ce, torch.cat([(yh[0][..., h, :].reshape(-1, HD) - mu[0][h].float()) @ pa[h] for h in range(NH)], 1)]
    for l in range(1, L+1):
        c.append(torch.cat([(yh[l][..., h, :].reshape(-1, HD) - mu[l][h].float()) @ pb[l][h] for h in range(NH)], 1))
    return torch.cat(c, 1)

# fit generators + tables
DIM = {L: 96 + 144*(L+1) + 1 for L in LAYERS}
Wsym, Wrnd = {}, {}; R2 = {}
tabsum = {L: torch.zeros(V, D, device=DEV, dtype=torch.float64) for L in LAYERS}; tabcnt = torch.zeros(V, device=DEV, dtype=torch.float64)
for named, store in [(True, Wsym), (False, Wrnd)]:
    AtA = {L: torch.zeros(DIM[L], DIM[L], device=DEV, dtype=torch.float64) for L in LAYERS}
    AtY = {L: torch.zeros(DIM[L], D, device=DEV, dtype=torch.float64) for L in LAYERS}
    for i in range(0, 512, 4):
        b = COOC[i:i+4].to(DEV)[:, :-1]; yh, hmlp = run(b, MAXL+1)
        for L in LAYERS:
            Cd = codes_mlp(L, b, yh, named).double(); Cd = torch.cat([Cd, torch.ones(Cd.shape[0], 1, device=DEV, dtype=torch.float64)], 1)
            Y = hmlp[L].reshape(-1, D).double(); AtA[L] += Cd.T @ Cd; AtY[L] += Cd.T @ Y
            if named: tabsum[L].index_add_(0, b.reshape(-1), Y)
        if named: tabcnt.index_add_(0, b.reshape(-1), torch.ones(b.numel(), device=DEV, dtype=torch.float64))
    for L in LAYERS:
        store[L] = torch.linalg.solve(AtA[L] + 10.0*torch.eye(DIM[L], device=DEV, dtype=torch.float64), AtY[L]).float()
TAB = {L: (tabsum[L] / tabcnt.clamp_min(1).unsqueeze(1)).float() for L in LAYERS}
b = COOC[512:516].to(DEV)[:, :-1]; yh, hmlp = run(b, MAXL+1)
for L in LAYERS:
    Cd = codes_mlp(L, b, yh, True); Cd = torch.cat([Cd, torch.ones(Cd.shape[0], 1, device=DEV)], 1)
    Y = hmlp[L].reshape(-1, D); R2[L] = 1 - float((Cd @ Wsym[L] - Y).pow(2).sum() / (Y - Y.mean(0)).pow(2).sum())
print('generators fit; R2', {L: round(R2[L], 3) for L in LAYERS}, flush=True)


@torch.no_grad()
def audit(Ltgt, mode):
    tot, n = 0.0, 0
    for i in range(0, len(FINEWEB), 4):
        full = FINEWEB[i:i+4].to(DEV); idx = full[:, :-1]; B, T = idx.shape
        dt = m.transformer.wte.weight.dtype; x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        cos, sin = rope_tables(T, HD, DEV, dt, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
        mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool)); yh = {}
        for li in range(NL):
            blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
            def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
            v = a.c_v(hcur).view(B, T, NH, HD)
            if v1 is None: v1 = v
            v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
            q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
            pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v); yh[li] = yh4
            x = x + a.c_proj(yh4.reshape(B, T, -1)); hin = F.rms_norm(x, (D,))
            if li == Ltgt:
                if mode == 'tab': hin = TAB[Ltgt][idx.reshape(-1)].view(B, T, D)
                else:
                    Cd = codes_mlp(Ltgt, idx, yh, mode == 'sym'); Cd = torch.cat([Cd, torch.ones(Cd.shape[0], 1, device=DEV)], 1)
                    hin = (Cd @ (Wsym if mode == 'sym' else Wrnd)[Ltgt]).view(B, T, D)
                hin = F.rms_norm(hin, (D,)).to(x.dtype)
            x = x + blk.mlp(hin)
        lg = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30).float()
        ce = F.cross_entropy(lg.reshape(-1, V), full[:, 1:].reshape(-1))
        tot += ce.item()*full[:, 1:].numel(); n += full[:, 1:].numel()
    return tot/n - BASE

res = {}
for L in LAYERS:
    r = {'R2': round(R2[L], 4), 'sym': round(audit(L, 'sym'), 5), 'tab': round(audit(L, 'tab'), 5), 'rand': round(audit(L, 'rand'), 5)}
    res[f'layer{L}'] = r
    print(f"MLP L{L}: sym {r['sym']:+.5f} | tab {r['tab']:+.5f} | rand {r['rand']:+.5f}  (R2 {r['R2']}) "
          f"{'SYM WINS' if r['sym'] < r['tab'] and r['sym'] < r['rand'] else 'check'}", flush=True)
json.dump(res, open(f'{QK}/qk_l26_mlp.json', 'w'), indent=2)
print('QK L26 MLP DONE', flush=True)
