"""Compositional recursion, layers 2-6 (attention pattern). Decompose each layer's QK pattern
from a GROWING symbol basis built ONLY from preceding layers (no raw-token blowup):
  codes(L) = 96 embedding PCs + layer-0 named archetypes (144) + per-head 16-d PCA of each
             attention layer 1..L-1 (144 each).
Generator: ridge codes -> block-L attention input x_L, fit on the disjoint cooc corpus.
Test: replace layer-L's pattern scores with scores from the synthetic x_L; audit dCE on FineWeb.
Baselines per layer: per-token TABLE (memorization control) and RANDOM-basis (is the named
basis special). Gate: reproduce layer-2 sym +0.0176. Model: bilin18 (per-head QK RMSNorm ->
use causal source generation, not exact fold).
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
V = cfg['vocab_size']; NL = len(m.transformer.h)
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
wte = m.transformer.wte.weight.detach().float().to(DEV)
BASE = 3.07630
LAYERS = [2, 3, 4, 5, 6]
MAXL = max(LAYERS)

# layer-0 named archetype value-directions (PA), embedding PCs (EW)
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
    """Run blocks 0..upto-1; return dict yh[li] (B,T,NH,HD) and x_in[li] (residual entering block li's attn)."""
    dt = m.transformer.wte.weight.dtype
    x = m.transformer.wte(idx); x = F.rms_norm(x, (D,)); x0 = x; v1 = None
    B, T = idx.shape
    cos, sin = rope_tables(T, HD, idx.device, dt, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
    yh = {}; xin = {}
    for li in range(upto):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; xin[li] = x
        a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v); yh[li] = yh4
        x = x + a.c_proj(yh4.reshape(B, T, -1)); x = x + blk.mlp(F.rms_norm(x, (D,)))
    return yh, xin

# ---- pass 1: per-layer PCA bases (layers 1..MAXL-1) + layer means ----
cov = {l: torch.zeros(NH, HD, HD, device=DEV, dtype=torch.float64) for l in range(MAXL)}
mu = {l: torch.zeros(NH, HD, device=DEV, dtype=torch.float64) for l in range(MAXL)}
nt = 0
for i in range(0, 256, 4):
    yh, _ = run(COOC[i:i+4].to(DEV)[:, :-1], MAXL)
    for l in range(MAXL):
        f = yh[l].reshape(-1, NH, HD).double(); mu[l] += f.sum(0); cov[l] += torch.einsum('nhd,nhe->hde', f, f)
    nt += yh[0].reshape(-1, NH, HD).shape[0]
for l in range(MAXL):
    mu[l] /= nt; cov[l] = cov[l]/nt - torch.einsum('hd,he->hde', mu[l], mu[l])
PB = {}  # per-head 16-d PCA basis per layer 1..MAXL-1
for l in range(1, MAXL):
    PB[l] = {}
    for h in range(NH):
        ev, evec = torch.linalg.eigh(cov[l][h]); PB[l][h] = evec[:, ev.argsort(descending=True)[:16]].float().contiguous()
# random-null bases (same shapes)
g = torch.Generator(device=DEV).manual_seed(1)
def rand_basis():
    Qr, _ = torch.linalg.qr(torch.randn(HD, HD, generator=g, device=DEV)); return Qr[:, :16].contiguous()
PAr = {h: rand_basis() for h in range(NH)}
PBr = {l: {h: rand_basis() for h in range(NH)} for l in range(1, MAXL)}
print('bases ready', flush=True)


def codes(L, idx, yh, named=True):
    B, T = idx.shape; ce = EW[idx.reshape(-1)]
    pa = PA if named else PAr; pb = PB if named else PBr
    c = [ce, torch.cat([(yh[0][..., h, :].reshape(-1, HD) - mu[0][h].float()) @ pa[h] for h in range(NH)], 1)]
    for l in range(1, L):
        c.append(torch.cat([(yh[l][..., h, :].reshape(-1, HD) - mu[l][h].float()) @ pb[l][h] for h in range(NH)], 1))
    return torch.cat(c, 1)

# ---- pass 2: fit ridge generators (sym + random) per layer; token tables ----
Wsym, Wrnd = {}, {}; R2 = {}
tabsum = {L: torch.zeros(V, D, device=DEV, dtype=torch.float64) for L in LAYERS}; tabcnt = torch.zeros(V, device=DEV, dtype=torch.float64)
for named, store in [(True, Wsym), (False, Wrnd)]:
    AtA = {L: torch.zeros(96+144*L+1, 96+144*L+1, device=DEV, dtype=torch.float64) for L in LAYERS}
    AtY = {L: torch.zeros(96+144*L+1, D, device=DEV, dtype=torch.float64) for L in LAYERS}
    for i in range(0, 512, 4):
        b = COOC[i:i+4].to(DEV)[:, :-1]; yh, xin = run(b, MAXL+1)
        for L in LAYERS:
            Cd = codes(L, b, yh, named).double(); Cd = torch.cat([Cd, torch.ones(Cd.shape[0], 1, device=DEV, dtype=torch.float64)], 1)
            Y = xin[L].reshape(-1, D).double(); AtA[L] += Cd.T @ Cd; AtY[L] += Cd.T @ Y
            if named:
                tok = b.reshape(-1); tabsum[L].index_add_(0, tok, Y)
        if named:
            tabcnt.index_add_(0, b.reshape(-1), torch.ones(b.numel(), device=DEV, dtype=torch.float64))
    for L in LAYERS:
        store[L] = torch.linalg.solve(AtA[L] + 10.0*torch.eye(96+144*L+1, device=DEV, dtype=torch.float64), AtY[L]).float()
TAB = {L: (tabsum[L] / tabcnt.clamp_min(1).unsqueeze(1)).float() for L in LAYERS}
# held-out R2 (sym) per layer
b = COOC[512:516].to(DEV)[:, :-1]; yh, xin = run(b, MAXL+1)
for L in LAYERS:
    Cd = codes(L, b, yh, True); Cd = torch.cat([Cd, torch.ones(Cd.shape[0], 1, device=DEV)], 1)
    Y = xin[L].reshape(-1, D); R2[L] = 1 - float((Cd @ Wsym[L] - Y).pow(2).sum() / (Y - Y.mean(0)).pow(2).sum())
print('generators fit; R2', {L: round(R2[L], 3) for L in LAYERS}, flush=True)


@torch.no_grad()
def audit(Ltgt, mode):
    """mode: 'sym' | 'rand' | 'tab' — replace layer-Ltgt QK input with the generated x_L."""
    tot, n = 0.0, 0
    for i in range(0, len(FINEWEB), 4):
        idx = FINEWEB[i:i+4].to(DEV)[:, :-1]; B, T = idx.shape
        dt = m.transformer.wte.weight.dtype; x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        cos, sin = rope_tables(T, HD, DEV, dt, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
        mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool)); yh = {}
        for li in range(NL):
            blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0
            a = blk.attn; hcur = F.rms_norm(x, (D,))
            def qkf(lin, src): z = F.rms_norm(lin(src).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
            v = a.c_v(hcur).view(B, T, NH, HD)
            if v1 is None: v1 = v
            v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
            if li == Ltgt:
                if mode == 'tab': xh = TAB[Ltgt][idx.reshape(-1)].view(B, T, D)
                else:
                    Cd = codes(Ltgt, idx, yh, mode == 'sym'); Cd = torch.cat([Cd, torch.ones(Cd.shape[0], 1, device=DEV)], 1)
                    xh = (Cd @ (Wsym if mode == 'sym' else Wrnd)[Ltgt]).view(B, T, D)
                hs = F.rms_norm(xh, (D,)).to(hcur.dtype)
                q, k, q2, k2 = qkf(a.c_q, hs), qkf(a.c_k, hs), qkf(a.c_q2, hs), qkf(a.c_k2, hs)
            else:
                q, k, q2, k2 = qkf(a.c_q, hcur), qkf(a.c_k, hcur), qkf(a.c_q2, hcur), qkf(a.c_k2, hcur)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
            pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v); yh[li] = yh4
            x = x + a.c_proj(yh4.reshape(B, T, -1)); x = x + blk.mlp(F.rms_norm(x, (D,)))
        lg = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30).float()
        ce = F.cross_entropy(lg.reshape(-1, V), FINEWEB[i:i+4].to(DEV)[:, 1:].reshape(-1))
        tot += ce.item()*idx[:, 1:].numel() if False else ce.item()*FINEWEB[i:i+4][:, 1:].numel(); n += FINEWEB[i:i+4][:, 1:].numel()
    return tot/n - BASE

res = {}
for L in LAYERS:
    r = {'R2': round(R2[L], 4), 'sym': round(audit(L, 'sym'), 5), 'tab': round(audit(L, 'tab'), 5), 'rand': round(audit(L, 'rand'), 5)}
    res[f'layer{L}'] = r
    print(f"L{L}: sym {r['sym']:+.5f} | tab {r['tab']:+.5f} | rand {r['rand']:+.5f}  (R2 {r['R2']}) "
          f"{'SYM WINS' if r['sym'] < r['tab'] and r['sym'] < r['rand'] else 'check'}", flush=True)
json.dump(res, open(f'{QK}/qk_l26_symbolgen.json', 'w'), indent=2)
print('QK L26 SYMBOLGEN DONE', flush=True)
