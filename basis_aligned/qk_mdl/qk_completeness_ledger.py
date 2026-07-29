"""Completeness ledger: for EVERY interface (each layer's QK-pattern input, each layer's MLP input),
compute the mean-input FLOOR (replace the interface input with the dataset-mean residual, a constant
-> no token/context information crosses that interface). Then understood-fraction per interface
= 1 - cost(our explanation)/cost(floor). Attention L2-17 explanations = symbolgen (qk_l217). MLP 2-6
= qk_l26_mlp sym. Interfaces without a generator (MLP 1, 7-17) count as unexplained at floor value.
Audit: first 200 sequences of the standard FineWeb 600x513 audit (subset noted); dCE vs subset base.
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
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))[:200]
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))

# pass 1: per-layer mean of the QK-input residual (xin) and of the rms-normed MLP input (hmlp)
mu_x = {l: torch.zeros(D, device=DEV, dtype=torch.float64) for l in range(NL)}
mu_h = {l: torch.zeros(D, device=DEV, dtype=torch.float64) for l in range(NL)}
nt = 0
@torch.no_grad()
def collect(idx):
    global nt
    B, T = idx.shape; x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0
        mu_x[li] += x.double().sum((0, 1)); a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1)); hin = F.rms_norm(x, (D,))
        mu_h[li] += hin.double().sum((0, 1)); x = x + blk.mlp(hin)
    nt += idx.numel()
for i in range(0, 128, 4):
    collect(COOC[i:i+4].to(DEV)[:, :-1])
MUX = {l: (mu_x[l]/nt).float() for l in range(NL)}; MUH = {l: F.rms_norm((mu_h[l]/nt).float(), (D,)) for l in range(NL)}
print('means ready', flush=True)


@torch.no_grad()
def audit(kind=None, L=None):
    """kind: None base | 'qk' floor at attention L | 'mlp' floor at MLP L."""
    tot, n = 0.0, 0
    for i in range(0, len(FINEWEB), 4):
        full = FINEWEB[i:i+4].to(DEV); idx = full[:, :-1]; B, T = idx.shape
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
        mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        for li in range(NL):
            blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
            hs = F.rms_norm(MUX[li], (D,)).expand(B, T, D).to(hcur.dtype) if (kind == 'qk' and li == L) else hcur
            def qkf(lin, src): z = F.rms_norm(lin(src).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
            v = a.c_v(hcur).view(B, T, NH, HD)
            if v1 is None: v1 = v
            v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
            q, k, q2, k2 = qkf(a.c_q, hs), qkf(a.c_k, hs), qkf(a.c_q2, hs), qkf(a.c_k2, hs)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
            pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
            x = x + a.c_proj(yh4.reshape(B, T, -1)); hin = F.rms_norm(x, (D,))
            if kind == 'mlp' and li == L: hin = MUH[li].expand(B, T, D).to(x.dtype)
            x = x + blk.mlp(hin)
        lg = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30).float()
        ce = F.cross_entropy(lg.reshape(-1, V), full[:, 1:].reshape(-1))
        tot += ce.item()*full[:, 1:].numel(); n += full[:, 1:].numel()
    return tot/n

base = audit()
print(f"subset base CE {base:.5f}", flush=True)
res = {'subset_base': round(base, 5), 'attn_floor': {}, 'mlp_floor': {}}
for L in range(NL):
    f = audit('qk', L) - base; res['attn_floor'][L] = round(f, 5)
    print(f"attn L{L} floor +{f:.5f}", flush=True)
for L in range(NL):
    f = audit('mlp', L) - base; res['mlp_floor'][L] = round(f, 5)
    print(f"mlp  L{L} floor +{f:.5f}", flush=True)
json.dump(res, open(f'{QK}/qk_completeness_ledger.json', 'w'), indent=2)
print("QK COMPLETENESS LEDGER DONE", flush=True)
