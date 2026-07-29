"""Recover the FUNCTIONAL part of MLP1's tail, not the variance (Logan). Two arms:
(A) SPARSE BILINEAR ON CE: add 32 fresh quadratic features to the polished program, U init ZERO
    (start == polished program), trained PURELY on cross-entropy through the frozen model with an
    L1 penalty on feature outputs (sparse). MSE never enters -> the features can only learn function.
(B) DIAGNOSIS (what could it be): per-token-category CE breakdown of the polished program's
    remaining gap (model vs program, by subword/punct/capital/digit/funcword/newline/other) --
    localizes the functional residual by token class.
"""
import json, sys
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
tok = AutoTokenizer.from_pretrained('gpt2')
A1w, U1w = torch.load(f'{QK}/qk_mlp1_r512.pt', map_location=DEV)['table_prev_R512']
POL = torch.load(f'{QK}/qk_mlp1_ce_polish.pt', map_location=DEV); G1, AB1 = POL['g'], POL['ab']
LED = json.load(open(f'{QK}/qk_completeness_ledger.json')); SUBBASE = LED['subset_base']; FLOOR1 = 2.15118

import string as _string
_P = set(_string.punctuation)
FUNC = {'the','of','and','to','a','in','is','that','it','for','was','as','with','on','be','at','by','this','are','from','or','an','but','not','which','you','have','he','they','has'}
def masks():
    ms = {k: torch.zeros(V, dtype=torch.bool) for k in ['subword','punct','capital','digit','funcword','newline']}
    for i in range(50257):
        s = tok.convert_ids_to_tokens(i)
        if s is None: continue
        core = s.replace('Ġ', ''); lead = s.startswith('Ġ')
        if not lead and len(core) and core[0].isalpha() and core[0].islower(): ms['subword'][i] = True
        if len(core) and all(c in _P for c in core): ms['punct'][i] = True
        if lead and len(core) and core[0].isupper(): ms['capital'][i] = True
        if len(core) and all(c.isdigit() for c in core): ms['digit'][i] = True
        if core.lower() in FUNC: ms['funcword'][i] = True
        if 'Ċ' in s or '\n' in s: ms['newline'][i] = True
    return {k: v.to(DEV) for k, v in ms.items()}
MASKS = masks()


@torch.no_grad()
def pairs1(idx):
    B, T = idx.shape; x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(2):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1)); hin = F.rms_norm(x, (D,))
        if li == 1:
            prv = torch.roll(idx, 1, 1); prv[:, 0] = idx[:, 0]
            return blk.mlp(hin).reshape(-1, D), idx.reshape(-1), prv.reshape(-1)
        x = x + blk.mlp(hin)

Ys, Ts, Ps = [], [], []
for i in range(0, 400, 8):
    y, t, p = pairs1(COOC[i:i+8].to(DEV)[:, :128]); Ys.append(y); Ts.append(t); Ps.append(p)
Y = torch.cat(Ys); T = torch.cat(Ts); P = torch.cat(Ps)
def cond_table(keys, target):
    ts = torch.zeros(V, D, device=DEV); tc = torch.zeros(V, device=DEV)
    ts.index_add_(0, keys, target); tc.index_add_(0, keys, torch.ones_like(keys, dtype=torch.float32))
    lam = tc.unsqueeze(1)/(tc.unsqueeze(1)+3.0)
    return lam*(ts/tc.clamp_min(1).unsqueeze(1)) + (1-lam)*target.mean(0)
TT1 = cond_table(T, Y); PT1 = cond_table(P, Y - TT1[T])
del Ys, Ts, Ps, Y, T, P

Anew = torch.nn.Parameter(torch.randn(32, D, device=DEV) * 0.02)
Unew = torch.nn.Parameter(torch.zeros(32, D, device=DEV))     # zero init: start == polished program

def forward(idx, sub, extra):
    B, T2 = idx.shape; x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T2, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T2, T2, device=DEV, dtype=torch.bool))
    prv = torch.roll(idx, 1, 1); prv[:, 0] = idx[:, 0]
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T2, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T2, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T2, -1)); hin = F.rms_norm(x, (D,))
        if li == 1 and sub:
            flat = hin.reshape(-1, D)
            mo = AB1[0]*TT1[idx.reshape(-1)] + AB1[1]*PT1[prv.reshape(-1)] + (((flat @ A1w.T)**2) * G1) @ U1w
            if extra: mo = mo + ((flat @ Anew.T)**2) @ Unew
            x = x + mo.view(B, T2, D).to(x.dtype)
        else:
            x = x + blk.mlp(hin)
    return 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)

@torch.no_grad()
def audit(sub, extra):
    tot = 0.0; n = 0; catce = {k: [0.0, 0] for k in list(MASKS)+['other']}
    for i in range(0, 200, 4):
        b = FINEWEB[i:i+4].to(DEV)
        lg = forward(b[:, :-1], sub, extra).float(); tgt = b[:, 1:]
        ce = F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction='none').view(4, -1)
        tot += ce.sum().item(); n += ce.numel()
        oth = torch.ones_like(tgt, dtype=torch.bool)
        for k in MASKS:
            mk = MASKS[k][tgt]; oth &= ~mk
            catce[k][0] += ce[mk].sum().item(); catce[k][1] += int(mk.sum())
        catce['other'][0] += ce[oth].sum().item(); catce['other'][1] += int(oth.sum())
    return tot/n, {k: v[0]/max(v[1], 1) for k, v in catce.items()}

ce_model, cat_model = audit(False, False)
ce_prog, cat_prog = audit(True, False)
print(f"model {ce_model:.5f} | polished program dCE +{ce_prog-ce_model:.5f}", flush=True)
print("(B) residual dCE by target category (program - model):", flush=True)
gaps = {k: round(cat_prog[k]-cat_model[k], 4) for k in cat_model}
for k, v in sorted(gaps.items(), key=lambda x: -x[1]): print(f"   {k}: +{v:.4f}", flush=True)

opt = torch.optim.Adam([Anew, Unew], lr=1.5e-3)
for step in range(600):
    i = 2400 + np.random.randint(0, 2500); b = COOC[i:i+2].to(DEV)[:, :128]
    lg = forward(b[:, :-1], True, True).float()
    flat_pen = Unew.norm(dim=1).sum()
    loss = F.cross_entropy(lg.reshape(-1, V), b[:, 1:].reshape(-1)) + 1e-4*flat_pen
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 150 == 0: print(f"step {step} loss {loss.item():.4f}", flush=True)

ce_ft, cat_ft = audit(True, True)
alive = int((Unew.norm(dim=1) > 1e-3).sum())
print(f"(A) +32 CE-trained sparse features: dCE +{ce_ft-ce_model:.5f} (was +{ce_prog-ce_model:.5f}) | live features {alive}/32", flush=True)
gaps_ft = {k: round(cat_ft[k]-cat_model[k], 4) for k in cat_model}
print("post-functail residual by category:", {k: gaps_ft[k] for k in sorted(gaps_ft, key=lambda x: -gaps_ft[x])[:4]}, flush=True)
res = {'model_CE': round(ce_model, 5), 'program_dCE': round(ce_prog-ce_model, 5),
       'functail_dCE': round(ce_ft-ce_model, 5), 'understood': round(1-(ce_ft-ce_model)/FLOOR1, 3),
       'live_features': alive, 'residual_by_cat_pre': gaps, 'residual_by_cat_post': gaps_ft}
json.dump(res, open(f'{QK}/qk_mlp1_functail.json', 'w'), indent=2)
torch.save({'Anew': Anew.detach(), 'Unew': Unew.detach()}, f'{QK}/qk_mlp1_functail.pt')
print("QK MLP1 FUNCTAIL DONE", flush=True)
