"""FUNCTION TEST for the MLP-L1 high-rank tail: does the sub-leading band (directions
5-32) carry the hub's known INDUCTION and CATEGORY-ENGINE function, or is that carried by
the top-4?

MLP1 is the induction/category hub: MLP1 knockout inverts induction (qk_induction_minimal.py);
blocks 0-3 = category engine, block 1 = hub (qk_category_engine.py). Here we split MLP1's
output-SVD basis into the TOP-4 (already characterized in §71) and the TAIL (dirs 4-31,
i.e. "directions 5-32"), mean-ablate each band separately, and measure:
  (A) induction advantage (repeated-sequence CE gap) -- adapted from qk_induction_minimal.py
  (B) next-token category-probe accuracy at blk1/blk4 -- adapted from qk_category_engine.py

Direction basis + forward + project-out mean-ablation copied VERBATIM from
qk_mlp_superposition.py / qk_mlp1_tail.py. Induction eval + category probe adapted VERBATIM
from qk_induction_minimal.py / qk_category_engine.py.
"""
import json, sys, math, time, subprocess
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'

def gpu_guard(min_free=4500, tries=45, sleep=20):
    for _ in range(tries):
        free = int(subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits']
        ).decode().split('\n')[0].strip())
        if free >= min_free:
            print(f"GPU guard: {free} MiB free -- proceeding.", flush=True); return
        print(f"GPU guard: only {free} MiB free (<{min_free}); sleeping {sleep}s ...", flush=True)
        time.sleep(sleep)
    raise RuntimeError("GPU guard timed out waiting for free memory")
gpu_guard()

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
tok = AutoTokenizer.from_pretrained('gpt2')
LI = 1; K32 = 32
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
SEQL = 128
TRAIN = FINEWEB[0:256, :SEQL].to(DEV)
BATCH = 6

# ---- MLP L1 top-32 SVD dirs from TRAIN gram (VERBATIM) ----
gram = torch.zeros(D, D, device=DEV)
@torch.no_grad()
def fwd_gram(idx):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(LI + 1):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1)); mo = blk.mlp(F.rms_norm(x, (D,)))
        if li == LI: gram.add_(torch.einsum('btd,bte->de', mo, mo))
        x = x + mo
print("Recomputing MLP L1 SVD directions from TRAIN gram ...", flush=True)
for i in range(0, TRAIN.shape[0], BATCH): fwd_gram(TRAIN[i:i+BATCH])
_evals, _evecs = torch.linalg.eigh(gram)
DIRS = _evecs[:, -K32:].T.flip(0).contiguous()      # (K32,D) descending
del gram, _evecs
TOP4 = DIRS[:4].contiguous()
TAIL = DIRS[4:].contiguous()                        # dirs 4..31 == "5-32"
print("MLP L1 SVD dirs ready.", flush=True)

# =====================================================================================
# (A) INDUCTION -- adapted from qk_induction_minimal.py. mlp_mode:
#   None            : intact
#   ('full',)       : mean-ablate whole MLP1 output (per-batch/pos mean)
#   ('proj', Dirs)  : project out Dirs from MLP1 output, coeff replaced by batch/pos mean
# =====================================================================================
P = 64; NSEQ = 48
pref = FINEWEB[:NSEQ, 1:1+P]
EV = torch.cat([pref, pref], 1).to(DEV)
SEC = torch.arange(P, 2*P-1, device=DEV); FIR = torch.arange(1, P-1, device=DEV)

@torch.no_grad()
def ind_forward(mlp_mode=None):
    idx = EV[:, :-1]; B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
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
        if li == LI and mlp_mode is not None:
            if mlp_mode[0] == 'full':
                mo = mo.mean((0, 1), keepdim=True).expand_as(mo)
            elif mlp_mode[0] == 'proj':
                Dirs = mlp_mode[1]
                pr = torch.einsum('btd,kd->btk', mo, Dirs)          # (B,T,k)
                pm = pr.mean((0, 1), keepdim=True)                  # (1,1,k) mean-ablation
                mo = mo - torch.einsum('btk,kd->btd', pr - pm, Dirs)
        x = x + mo
    lg = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30).float(); tgt = EV[:, 1:]
    ce = F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction='none').view(B, T)
    return ce[:, FIR].mean().item() - ce[:, SEC].mean().item()

ADV_FULL = ind_forward(None)
ADV_NONE = ind_forward(('full',))                     # whole MLP1 gone
adv_top4 = ind_forward(('proj', TOP4))
adv_tail = ind_forward(('proj', TAIL))
def ret(a): return (a - ADV_NONE) / (ADV_FULL - ADV_NONE) if abs(ADV_FULL - ADV_NONE) > 1e-9 else None
induction = {
    'adv_intact': round(ADV_FULL, 4),
    'adv_ablate_all_MLP1': round(ADV_NONE, 4),
    'adv_ablate_MLP1_top4': round(adv_top4, 4), 'ret_top4': round(ret(adv_top4), 4),
    'adv_ablate_MLP1_tail_5to32': round(adv_tail, 4), 'ret_tail': round(ret(adv_tail), 4),
    'drop_from_top4': round(ADV_FULL - adv_top4, 4),
    'drop_from_tail': round(ADV_FULL - adv_tail, 4),
    'note': 'advantage = mean CE(first occ) - mean CE(second occ) on repeated sequences; higher=stronger induction. '
            'ret = fraction of (intact - allMLP1ablated) advantage retained.',
}
print(f"INDUCTION intact {ADV_FULL:+.4f}  allMLP1 {ADV_NONE:+.4f}  "
      f"top4 {adv_top4:+.4f}(ret {induction['ret_top4']})  tail {adv_tail:+.4f}(ret {induction['ret_tail']})", flush=True)

# =====================================================================================
# (B) CATEGORY ENGINE -- adapted from qk_category_engine.py. 6-way next-token category
# probe on the residual, under MLP1 band ablations.
# =====================================================================================
import string as _string
_P = set(_string.punctuation)
FUNC = {'the','of','and','to','a','in','is','that','it','for','was','as','with','on','be','at','by','this','are','from','or','an','but','not','which'}
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
DEPTHS = ['blk1', 'blk4']

@torch.no_grad()
def cat_collect(idx, mlp_mode=None):
    B, T = idx.shape; x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool)); out = {}
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
        if li == LI and mlp_mode is not None:
            if mlp_mode[0] == 'full':
                mo = MLPMEAN1.expand_as(mo)
            elif mlp_mode[0] == 'proj':
                Dirs = mlp_mode[1]
                pr = torch.einsum('btd,kd->btk', mo, Dirs)
                pm = torch.einsum('btd,kd->btk', MLPMEAN1.expand_as(mo), Dirs)
                mo = mo - torch.einsum('btk,kd->btd', pr - pm, Dirs)
        x = x + mo
        if f'blk{li}' in DEPTHS: out[f'blk{li}'] = x.reshape(-1, D)
    return out

# MLP1 mean (per-feature global mean over a fixed slice) for category-context ablation
@torch.no_grad()
def mlp1_mean(idx):
    B, T = idx.shape; x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool)); mm = None
    for li in range(LI + 1):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1)); mo = blk.mlp(F.rms_norm(x, (D,)))
        if li == LI: mm = mo.mean((0, 1))
        x = x + mo
    return mm
MLPMEAN1 = mlp1_mean(FINEWEB[:32, :128].to(DEV))     # (D,)

def gather(rng, mlp_mode=None):
    R = {d: [] for d in DEPTHS}; Y = []
    for i in rng:
        idx = FINEWEB[i:i+4, :128].to(DEV)
        res = cat_collect(idx[:, :-1], mlp_mode)
        lab = CAT[idx[:, 1:].reshape(-1)]
        for d in DEPTHS: R[d].append(res[d])
        Y.append(lab)
    return {d: torch.cat(R[d]) for d in DEPTHS}, torch.cat(Y)

Rtr, Ytr = gather(range(0, 240, 4))
Yoh = F.one_hot(Ytr, 6).float()
_, Yte0 = gather(range(300, 304, 4))   # placeholder to get label dtype
def probe_train(Xtr):
    Xtr = torch.cat([Xtr, torch.ones(Xtr.shape[0], 1, device=DEV)], 1).double()
    return torch.linalg.solve(Xtr.T @ Xtr + 50*torch.eye(Xtr.shape[1], device=DEV, dtype=torch.double), Xtr.T @ Yoh.double())
def probe_acc(W, Xte, Yte):
    Xte = torch.cat([Xte, torch.ones(Xte.shape[0], 1, device=DEV)], 1).double()
    return float(((Xte @ W).argmax(1) == Yte).float().mean())

# probes trained on INTACT residuals (fixed readout), evaluated under each ablation
Wprobe = {d: probe_train(Rtr[d]) for d in DEPTHS}
Rte, Yte = gather(range(300, 400, 4))
maj = float(torch.bincount(Yte, minlength=6).max().item()) / Yte.numel()

CAT_MODES = {'intact': None, 'ablate_all_MLP1': ('full',),
             'ablate_MLP1_top4': ('proj', TOP4), 'ablate_MLP1_tail_5to32': ('proj', TAIL)}
category = {'majority': round(maj, 4), 'probe_trained_on': 'intact residuals (fixed readout)', 'acc': {}}
for mode, mm in CAT_MODES.items():
    Rte_m, _ = gather(range(300, 400, 4), mlp_mode=mm)
    category['acc'][mode] = {d: round(probe_acc(Wprobe[d], Rte_m[d], Yte), 4) for d in DEPTHS}
    print(f"CATEGORY {mode}: " + "  ".join(f"{d} {category['acc'][mode][d]:.3f}" for d in DEPTHS), flush=True)

out = {
    'meta': {'model': 'bilin18', 'layer': LI, 'K32': K32,
             'bands': {'top4': 'dirs 0-3', 'tail_5to32': 'dirs 4-31'},
             'induction_eval': 'repeated-sequence advantage, adapted qk_induction_minimal.py',
             'category_eval': '6-way next-token category probe (ridge, fixed intact readout), adapted qk_category_engine.py'},
    'induction': induction,
    'category': category,
}
json.dump(out, open(f'{QK}/qk_mlp1_tail_func.json', 'w'), indent=2)
print("\nSaved qk_mlp1_tail_func.json", flush=True)
print("QK MLP1 TAIL FUNC DONE", flush=True)
