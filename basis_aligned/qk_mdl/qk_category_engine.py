"""What does the shared early-MLP category engine compute? The atlas showed 5 tasks (subword/punct/
capital/digit/funcword) recruit MLP0-3. Hypothesis: MLP0-3 builds an explicit NEXT-TOKEN CATEGORY
code in the residual stream. Test: fit linear probes (ridge to one-hot, argmax) predicting the
next token's category (6-way) from the residual at increasing depth. Accuracy vs depth localizes
where the code is built. Causal: mean-ablate MLP0-3 and re-probe the layer-4 residual -- if the
category code collapses, MLP0-3 BUILDS it (not just correlated).
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
tok = AutoTokenizer.from_pretrained('gpt2')
import string as _string
_P = set(_string.punctuation)
FUNC = {'the','of','and','to','a','in','is','that','it','for','was','as','with','on','be','at','by','this','are','from','or','an','but','not','which'}

# 6-way category label per token id: 0 subword,1 punct,2 capital,3 digit,4 funcword,5 other
CAT = torch.full((V,), 5, dtype=torch.long)
for i in range(50257):
    s = tok.convert_ids_to_tokens(i)
    if s is None: continue
    core = s.replace('Ġ', ''); lead = s.startswith('Ġ')
    if len(core) and all(c in _P for c in core): CAT[i] = 1
    elif len(core) and all(c.isdigit() for c in core): CAT[i] = 3
    elif core.lower() in FUNC: CAT[i] = 4
    elif not lead and len(core) and core[0].isalpha() and core[0].islower(): CAT[i] = 0
    elif lead and len(core) and core[0].isupper(): CAT[i] = 2
CAT = CAT.to(DEV)
CATNAMES = ['subword', 'punct', 'capital', 'digit', 'funcword', 'other']

DEPTHS = ['embed', 'blk0', 'blk1', 'blk2', 'blk3', 'blk4', 'blk8', 'blk12']

@torch.no_grad()
def collect(idx, ablate=None):
    """return dict depth->residual (B*T,D). ablate: set of MLP layer indices to mean-ablate."""
    B, T = idx.shape; x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool)); out = {'embed': x.reshape(-1, D)}
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1)); mo = blk.mlp(F.rms_norm(x, (D,)))
        if ablate is not None and li in ablate: mo = MLPMEAN[li].expand_as(mo)
        x = x + mo
        if f'blk{li}' in DEPTHS: out[f'blk{li}'] = x.reshape(-1, D)
    return out

# MLP means for ablation
@torch.no_grad()
def mlp_means(idx):
    B, T = idx.shape; x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool)); mm = {}
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1)); mo = blk.mlp(F.rms_norm(x, (D,))); mm[li] = mo.mean((0, 1)); x = x + mo
    return mm
MLPMEAN = mlp_means(FINEWEB[:32, :128].to(DEV))

# build train/test residuals + next-token category labels
def gather(rng, ablate=None):
    R = {d: [] for d in DEPTHS}; Y = []
    for i in rng:
        idx = FINEWEB[i:i+4, :128].to(DEV)
        res = collect(idx[:, :-1], ablate)
        lab = CAT[idx[:, 1:].reshape(-1)]
        for d in DEPTHS: R[d].append(res[d])
        Y.append(lab)
    return {d: torch.cat(R[d]) for d in DEPTHS}, torch.cat(Y)

Rtr, Ytr = gather(range(0, 240, 4))
Rte, Yte = gather(range(300, 400, 4))
Yoh = F.one_hot(Ytr, 6).float()
maj = torch.bincount(Yte, minlength=6).max().item() / Yte.numel()
print(f"train {Ytr.numel()} test {Yte.numel()} | class dist(test) {[round(c/Yte.numel(),3) for c in torch.bincount(Yte,minlength=6).tolist()]} | majority {maj:.3f}", flush=True)

def probe_acc(Xtr, Xte):
    Xtr = torch.cat([Xtr, torch.ones(Xtr.shape[0], 1, device=DEV)], 1).double()
    Xte = torch.cat([Xte, torch.ones(Xte.shape[0], 1, device=DEV)], 1).double()
    W = torch.linalg.solve(Xtr.T @ Xtr + 50*torch.eye(Xtr.shape[1], device=DEV, dtype=torch.double), Xtr.T @ Yoh.double())
    pred = (Xte @ W).argmax(1)
    return float((pred == Yte).float().mean())

res = {'majority': round(maj, 4), 'depth_acc': {}}
for d in DEPTHS:
    a = probe_acc(Rtr[d], Rte[d]); res['depth_acc'][d] = round(a, 4)
    print(f"  probe {d}: {a:.4f}", flush=True)

# causal: ablate MLP0-3, re-probe blk4 residual
Rtr_a, Ytr_a = gather(range(0, 240, 4), ablate={0, 1, 2, 3})
Rte_a, _ = gather(range(300, 400, 4), ablate={0, 1, 2, 3})
Yoh_full = Yoh  # same labels/order
acc_abl = probe_acc(Rtr_a['blk4'], Rte_a['blk4'])
res['blk4_intact'] = res['depth_acc']['blk4']; res['blk4_ablate_mlp0_3'] = round(acc_abl, 4)
print(f"\ncausal: blk4 probe intact {res['blk4_intact']:.4f} -> MLP0-3 ablated {acc_abl:.4f}", flush=True)
json.dump(res, open(f'{QK}/qk_category_engine.json', 'w'), indent=2)
print("QK CATEGORY ENGINE DONE", flush=True)
