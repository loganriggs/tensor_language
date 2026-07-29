"""What does the shared hub MLP-layer-1 compute, and do the three circuits read the SAME part of it?
MLP1 is the dominant knockout in all three task circuits. Here we decompose MLP1's OUTPUT into
principal directions and ask, per task: (a) how many output directions (rank) does the task need,
(b) which specific directions, and (c) do the tasks overlap or read disjoint subspaces.
Intervention: at block 1, replace the MLP output O by mean + rank-r reconstruction in the output
PCA basis; everything else runs full. Floor = rank-0 (O = mean); ceiling = full MLP1.
Metrics: induction advantage; subword/punct CE on target positions. Per-PC importance = task drop
when that single output direction is projected out. Cross-task overlap of the important directions
tells us whether MLP1 is one shared low-rank hub or a multiplexer of task-specific pieces.
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
def vmask(kind):
    msk = torch.zeros(V, dtype=torch.bool)
    for i in range(50257):
        s = tok.convert_ids_to_tokens(i)
        if s is None: continue
        if kind == 'sub':
            if not s.startswith('Ġ') and len(s) and s[0].isalpha() and s[0].islower(): msk[i] = True
        else:
            core = s.replace('Ġ', '')
            if len(core) and all(c in _P for c in core): msk[i] = True
    return msk.to(DEV)
SUB, PUN = vmask('sub'), vmask('pun')

# MLP1 output PCA over natural text
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
U, S, Vh = torch.linalg.svd(O - mu1, full_matrices=False)
P1 = Vh[:64].T.float().contiguous(); mu1 = mu1.float()   # (D,64) output PCs
print(f"MLP1 output PCA: top-8 var frac {(S[:8]**2/ (S**2).sum()).sum():.3f}, top-32 {(S[:32]**2/(S**2).sum()).sum():.3f}", flush=True)


@torch.no_grad()
def forward(idx, mode=None, r=None, drop=None):
    """mode: None full; 'rank' keep rank-r of MLP1 output; 'drop' project out PC set `drop`."""
    B, T = idx.shape; x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
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
        if li == 1 and mode is not None:
            flat = mo.reshape(-1, D) - mu1
            if mode == 'rank':
                B1 = P1[:, :r]; rec = (flat @ B1) @ B1.T
                mo = (mu1 + rec).view(B, T, D)
            elif mode == 'drop':
                Bd = P1[:, drop]; rec = (flat @ Bd) @ Bd.T
                mo = (mu1 + flat - rec).view(B, T, D)
        x = x + mo
    return 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30).float()

# task evals
P = 64; NSEQ = 48
pref = FINEWEB[:NSEQ, 1:1+P]; EVi = torch.cat([pref, pref], 1).to(DEV)
SECi = torch.arange(P, 2*P-1, device=DEV); FIRi = torch.arange(1, P-1, device=DEV)
EVn = FINEWEB[:48, :128].to(DEV)
def score(task, mode=None, r=None, drop=None):
    if task == 'induction':
        lg = forward(EVi[:, :-1], mode, r, drop); tgt = EVi[:, 1:]
        ce = F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction='none').view(EVi.shape[0], -1)
        return ce[:, FIRi].mean().item() - ce[:, SECi].mean().item()   # higher=better
    else:
        lg = forward(EVn[:, :-1], mode, r, drop); tgt = EVn[:, 1:]
        cm = (SUB if task == 'subword' else PUN)[tgt]
        ce = F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction='none').view(EVn.shape[0], -1)
        return -ce[cm].mean().item()   # higher=better

TASKS = ['induction', 'subword', 'punct']
full = {t: score(t) for t in TASKS}
floor = {t: score(t, 'rank', 0) for t in TASKS}   # MLP1 output = mean
def ret(t, s): return (s - floor[t]) / (full[t] - floor[t])
print("MLP1 task contribution (full - floor):", {t: round(full[t]-floor[t], 3) for t in TASKS}, flush=True)

# rank sweep
RANKS = [1, 2, 4, 8, 16, 32, 64]
sweep = {t: {r: round(ret(t, score(t, 'rank', r)), 4) for r in RANKS} for t in TASKS}
for t in TASKS:
    print(f"rank sweep {t}: " + " ".join(f"r{r}={sweep[t][r]:.2f}" for r in RANKS), flush=True)

# per-PC importance (drop one PC, task drop) over top-32
imp = {t: [] for t in TASKS}
for t in TASKS:
    for k in range(32):
        s = score(t, 'drop', drop=[k]); imp[t].append(round(ret(t, s), 4))
top = {t: sorted(range(32), key=lambda k: imp[t][k])[:8] for t in TASKS}   # most important = biggest retention drop when dropped
print("top-8 important MLP1 output PCs per task:", flush=True)
for t in TASKS: print(f"  {t}: {top[t]}", flush=True)
# overlap
def jac(a, b): a, b = set(a), set(b); return round(len(a&b)/len(a|b), 3)
overlap = {'ind_sub': jac(top['induction'], top['subword']), 'ind_pun': jac(top['induction'], top['punct']),
           'sub_pun': jac(top['subword'], top['punct'])}
print("top-8 PC overlap (Jaccard):", overlap, flush=True)

res = {'var_top8': round(float((S[:8]**2/(S**2).sum()).sum()), 4),
       'mlp1_contribution': {t: round(full[t]-floor[t], 4) for t in TASKS},
       'rank_sweep': sweep, 'top8_pcs': {t: top[t] for t in TASKS}, 'overlap_jaccard': overlap,
       'per_pc_importance_top32': imp}
json.dump(res, open(f'{QK}/qk_mlp1_hub.json', 'w'), indent=2)
print("QK MLP1 HUB DONE", flush=True)
