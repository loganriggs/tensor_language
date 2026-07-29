"""What does PC0 -- the one MLP1 output direction shared-important across all three circuits -- encode?
Recompute the MLP1 output PCA, then over a corpus project each token's MLP1 output onto PC0 and
aggregate by token id. Rank tokens by mean PC0 activation (writer side): the top/bottom tokens name
the axis. Also correlate PC0 activation with the subword-continuation and punctuation target masks
to see whether the shared direction aligns with a task axis or is a generic content signal.
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

@torch.no_grad()
def mlp1_out(idx):
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
        x = x + a.c_proj(yh4.reshape(B, T, -1)); mo = blk.mlp(F.rms_norm(x, (D,)))
        if li == 1: return mo.reshape(-1, D)
        x = x + mo

acc = []
for i in range(0, 200, 4):
    acc.append(mlp1_out(FINEWEB[i:i+4, :128].to(DEV)).double())
O = torch.cat(acc, 0); mu1 = O.mean(0)
Vh = torch.linalg.svd(O - mu1, full_matrices=False).Vh
pc0 = Vh[0].float()   # (D,)

# aggregate PC0 activation by CURRENT token id over a bigger slice
sumd = torch.zeros(V, device=DEV, dtype=torch.float64); cnt = torch.zeros(V, device=DEV, dtype=torch.float64)
allproj = []
alltok = []
for i in range(0, 400, 4):
    idx = FINEWEB[i:i+4, :128].to(DEV)
    o = mlp1_out(idx); proj = ((o.double() - mu1) @ pc0.double())  # (B*T,)
    ti = idx.reshape(-1)
    sumd.index_add_(0, ti, proj); cnt.index_add_(0, ti, torch.ones_like(proj))
    allproj.append(proj.float().cpu()); alltok.append(ti.cpu())
mean_by_tok = (sumd / cnt.clamp_min(1)).cpu()
mask_seen = (cnt.cpu() >= 5)
seen_ids = torch.nonzero(mask_seen).squeeze(1).tolist()
seen_ids.sort(key=lambda t: mean_by_tok[t].item())
def show(ids): return [tok.convert_ids_to_tokens(i) for i in ids]
low = seen_ids[:40]; high = seen_ids[-40:][::-1]
print("PC0 LOW tokens:", show(low), flush=True)
print("PC0 HIGH tokens:", show(high), flush=True)

# correlate PC0 activation with continuation/punct property of the CURRENT token
import string as _string
_P = set(_string.punctuation)
def prop(kind, tid):
    s = tok.convert_ids_to_tokens(tid)
    if s is None: return False
    if kind == 'sub': return (not s.startswith('Ġ')) and len(s) > 0 and s[0].isalpha() and s[0].islower()
    core = s.replace('Ġ', ''); return len(core) > 0 and all(c in _P for c in core)
proj_all = torch.cat(allproj); tok_all = torch.cat(alltok)
is_sub = torch.tensor([prop('sub', int(t)) for t in tok_all])
is_pun = torch.tensor([prop('pun', int(t)) for t in tok_all])
def corr(mask):
    a = proj_all.double(); b = mask.double()
    return round(float(((a-a.mean())*(b-b.mean())).mean() / (a.std()*b.std()+1e-9)), 3)
res = {'pc0_low_tokens': show(low), 'pc0_high_tokens': show(high),
       'corr_pc0_subword': corr(is_sub), 'corr_pc0_punct': corr(is_pun),
       'mean_pc0_on_subword': round(float(proj_all[is_sub].mean()), 3),
       'mean_pc0_on_punct': round(float(proj_all[is_pun].mean()), 3),
       'mean_pc0_on_other': round(float(proj_all[~(is_sub | is_pun)].mean()), 3)}
print("corr PC0 vs subword:", res['corr_pc0_subword'], "| vs punct:", res['corr_pc0_punct'], flush=True)
print("mean PC0 -- subword:", res['mean_pc0_on_subword'], "punct:", res['mean_pc0_on_punct'], "other:", res['mean_pc0_on_other'], flush=True)
json.dump(res, open(f'{QK}/qk_mlp1_pc0.json', 'w'), indent=2)
print("QK MLP1 PC0 DONE", flush=True)
