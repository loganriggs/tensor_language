"""SHARED CORES PART 2: (i) region-restricted sharing (section-99 regions: early 0-4,
distributed 5-11, readout 12-17) -- do blocks share cores better WITHIN a region than
globally at the same atom count?  (ii) the COMPOSED scheme at the section-92 16-fold anchor
budget: k shared atoms AND a within-atom rank restriction (input subspace Kin on both slots,
output subspace Kout per atom), budget k * Kout * Kin*(Kin+1)/2 ~= FULL/16 -- does sharing
COMPOSE with rank allocation better than rank allocation alone (+0.8032)?

Atoms are mixtures (part 1): C_i = sum_m U[m,i] T_m, so the restricted atom applied at
layer l needs no materialization:
    h'  = Hbar_l + Pin Pin^T (h_l - Hbar_l)              (shared input basis, pooled train
                                                          second moment of the normalized
                                                          inputs; Hbar_l = held per-position
                                                          input mean, uncounted table)
    c_i = sum_m U[m,i] * bil_m(h')                       (atom bilinear output)
    mo_l = bias_l + BM[l,l] + sum_i U[l,i] * Po_i Po_i^T (c_i - CM[l,i])
Po_i = top-Kout eigenvectors of atom i's train output gram (collected with the SAME h'
construction, pooled over layers with weight U[l,i]^2); CM[l,i] = held per-position mean of
c_i at layer l under the base forward (uncounted, section-92 MX/MO precedent).
GATE: k=18 atoms at full ranks reproduces the exact model (orthogonality of U collapses the
mixture to the identity).

Uses qk_sharedcore_cache.pt from part 1 (G, G_pool, U's, BM, Hmean, Sig, base).
Extends qk_sharedcore.json in place ('regional', 'composed')."""
import json, math, os, subprocess, sys, time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/qk_sharedcore.json'
CPT = f'{QK}/qk_sharedcore_cache.pt'

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
FW = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
TRAIN = FW[0:256, :128].to(DEV); HELD = FW[448:600, :128].to(DEV)
B0M = 4
S_, T_ = HELD.shape; STR = TRAIN.shape[0]
FULLBLK = D * D * (D + 1) // 2; FULL = NL * FULLBLK
REGIONS = {'early': list(range(0, 5)), 'distributed': list(range(5, 12)),
           'readout': list(range(12, 18))}

def mlp_wts(li):
    b = m.transformer.h[li].mlp
    return (b.Left.weight.detach().float(), b.Right.weight.detach().float(),
            b.Down.weight.detach().float(), b.Down_bias.detach().float())
WTS = [mlp_wts(li) for li in range(NL)]
BIAS = [WTS[li][3] for li in range(NL)]

cache = torch.load(CPT, map_location='cpu', weights_only=True)
G = cache['G'].numpy(); Gp = cache['G_pool'].numpy()
UG = cache['UG'].numpy(); UP = cache['UP'].numpy()
BM = cache['BM'].to(DEV); Hmean = cache['Hmean'].to(DEV)
base = cache['base']; Sig = cache['Sig']
res = json.load(open(OUT))
res.setdefault('regional', {}); res.setdefault('composed', {})
def dump(): json.dump(res, open(OUT, 'w'), indent=1)

def dstat(ce):
    d = (ce - base).flatten().double(); return float(d.mean()), float(d.std()/np.sqrt(d.numel()))

# pooled input basis from TRAIN second moments
wS, vS = torch.linalg.eigh(Sig.mean(0).double())
PIN_FULL = vS.flip(1).float().to(DEV)          # (D, D) descending

# ============================= shared forward skeleton ================================
@torch.no_grad()
def fwd_core(idx, hook):
    """Base skeleton; hook(li, h, B) -> mo (or None for exact block)."""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0
        a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(l): z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k_, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k_)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh.reshape(B, T, -1))
        h = F.rms_norm(x, (D,))
        mo = hook(li, h, B)
        if mo is None: mo = blk.mlp(h)
        x = x + mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    return F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1),
                           reduction='none').view(B, T-1)

def bil(mm, h):
    Lw, Rw, Dw, _ = WTS[mm]
    return ((h @ Lw.T) * (h @ Rw.T)) @ Dw.T

# ============================= (i) regional sharing probe =============================
def mix_hook(W):
    Wt = torch.from_numpy(W).float()
    def hook(li, h, B):
        mo = (BIAS[li] + BM[li, li]).unsqueeze(0).expand(B, -1, -1).clone()
        for mm in range(NL):
            wlm = float(Wt[li, mm])
            if abs(wlm) < 1e-12: continue
            mo += wlm * (bil(mm, h) - BM[li, mm])
        return mo
    return hook

def eval_hook(hook, tag, batch=B0M):
    t0 = time.time()
    ce = torch.cat([fwd_core(HELD[i:i+batch], hook).cpu() for i in range(0, S_, batch)], 0)
    mn, se = dstat(ce)
    print(f"  [{tag}] dCE {mn:+.4f} +- {se:.5f} ({time.time()-t0:.0f}s)", flush=True)
    return mn, se

def regional_W(Gm, alloc):
    W = np.zeros((NL, NL))
    per = {}
    for rn, blks in REGIONS.items():
        sub = Gm[np.ix_(blks, blks)]
        w, U = np.linalg.eigh(sub)
        U = U[:, ::-1][:, :alloc[rn]]
        W[np.ix_(blks, blks)] = U @ U.T
        wr = np.clip(w[::-1], 0, None)
        per[rn] = float(wr[:alloc[rn]].sum() / wr.sum())
    return W, per

ALLOC8 = {'early': 2, 'distributed': 3, 'readout': 3}
for vtag, Gm in [('frobenius', G), ('pooled_metric', Gp)]:
    key = f'k8_regional_{vtag}'
    if key not in res['regional']:
        W, per = regional_W(Gm, ALLOC8)
        mn, se = eval_hook(mix_hook(W), key)
        res['regional'][key] = {
            'alloc': ALLOC8, 'total_atoms': 8, 'variant': vtag,
            'dCE': round(mn, 4), 'SE': round(se, 5),
            'budget': int(8 * FULLBLK + NL * 8), 'compression_x': round(FULL/(8*FULLBLK), 3),
            'region_energy_captured': {k: round(v, 4) for k, v in per.items()},
            'global_ref': 'shared_atom k8 same variant in part 1'}
        dump()

# ============================= (ii) composed scheme at 16-fold ========================
@torch.no_grad()
def out_gram_pass(U_k, Kin, tag):
    """TRAIN pass: atom output grams (weights U[l,i]^2), with the h' input projection."""
    k = U_k.shape[1]
    Ug = torch.from_numpy(U_k).float().to(DEV)
    Gout = torch.zeros(k, D, D, device=DEV)
    Pin = PIN_FULL[:, :Kin]
    def hook(li, h, B):
        hp = Hmean[li] + ((h - Hmean[li]) @ Pin) @ Pin.T
        bl = torch.stack([bil(mm, hp) for mm in range(NL)], 0)        # (NL,B,T,D)
        c = torch.einsum('mi,mbtd->ibtd', Ug, bl)                     # (k,B,T,D)
        for i in range(k):
            Gout[i] += float(Ug[li, i]**2) * torch.einsum('btd,bte->de', c[i], c[i])
        return None                                                    # exact base forward
    t0 = time.time()
    for i in range(0, STR, B0M): fwd_core(TRAIN[i:i+B0M], hook)
    print(f"  [{tag}] output grams done ({time.time()-t0:.0f}s)", flush=True)
    return Gout

@torch.no_grad()
def cm_pass(U_k, Kin, tag):
    """HELD pass under the BASE forward: per-position means CM[l,i] of the projected atoms."""
    k = U_k.shape[1]
    Ug = torch.from_numpy(U_k).float().to(DEV)
    CM = torch.zeros(NL, k, T_, D, device=DEV)
    Pin = PIN_FULL[:, :Kin]
    def hook(li, h, B):
        hp = Hmean[li] + ((h - Hmean[li]) @ Pin) @ Pin.T
        bl = torch.stack([bil(mm, hp) for mm in range(NL)], 0)
        c = torch.einsum('mi,mbtd->ibtd', Ug, bl)
        CM[li] += c.sum(1)
        return None
    t0 = time.time()
    for i in range(0, S_, B0M): fwd_core(HELD[i:i+B0M], hook)
    print(f"  [{tag}] CM means done ({time.time()-t0:.0f}s)", flush=True)
    return CM / S_

def composed_eval(U_k, Kin, Kout, tag, gate=False):
    k = U_k.shape[1]
    Gout = out_gram_pass(U_k, Kin, tag) if not gate else None
    CM = cm_pass(U_k, Kin, tag)
    Ug = torch.from_numpy(U_k).float().to(DEV)
    Pin = PIN_FULL[:, :Kin]
    if gate:
        Po = [None]*k
    else:
        Po = []
        for i in range(k):
            w, vec = torch.linalg.eigh(Gout[i].double())
            Po.append(vec.flip(1)[:, :Kout].float().contiguous())
        del Gout
    def hook(li, h, B):
        hp = Hmean[li] + ((h - Hmean[li]) @ Pin) @ Pin.T
        bl = torch.stack([bil(mm, hp) for mm in range(NL)], 0)
        c = torch.einsum('mi,mbtd->ibtd', Ug, bl)
        mo = (BIAS[li] + BM[li, li]).unsqueeze(0).expand(B, -1, -1).clone()
        for i in range(k):
            dev = c[i] - CM[li, i]
            if Po[i] is not None:
                dev = (dev @ Po[i]) @ Po[i].T
            mo = mo + float(Ug[li, i]) * dev
        return mo
    mn, se = eval_hook(hook, tag)
    del CM; torch.cuda.empty_cache()
    return mn, se

# GATE: k=18 atoms, full ranks -> exact model
if 'composed_identity_k18' not in res['gates']:
    print("GATE: composed k=18 full-rank ...", flush=True)
    mn, se = composed_eval(UP[:, :18], D, D, 'composed k18 gate', gate=True)
    res['gates']['composed_identity_k18'] = {'dCE': mn, 'SE': se}
    dump()
    assert abs(mn) < 2e-3, "composed identity gate FAILED"

# 16-fold total budget = FULL/16 = 860,709,888 folded coefficients
CONFIGS = [
    # (k, Kin, Kout, U tag)
    (2, 950, 950, 'pooled'), (4, 754, 754, 'pooled'), (8, 598, 598, 'pooled'),
    (2, 863, 1152, 'pooled'), (4, 610, 1152, 'pooled'), (8, 431, 1152, 'pooled'),
    (4, 950, 475, 'pooled'), (8, 754, 377, 'pooled'),
    (4, 754, 754, 'frobenius'),
]
TGT16 = FULL // 16
for (k, Kin, Kout, ut) in CONFIGS:
    key = f'k{k}_Kin{Kin}_Kout{Kout}_{ut}'
    if key in res['composed']: continue
    U_k = (UP if ut == 'pooled' else UG)[:, :k]
    bud = k * min(Kout, D) * Kin * (Kin + 1) // 2
    assert bud <= TGT16 * 1.001, (key, bud, TGT16)
    mn, se = composed_eval(U_k, Kin, Kout, key)
    res['composed'][key] = {
        'k': k, 'Kin': Kin, 'Kout': int(min(Kout, D)), 'atoms': ut,
        'dCE': round(mn, 4), 'SE': round(se, 5), 'budget': int(bud),
        'budget_target_16x': int(TGT16), 'budget_frac': round(bud / TGT16, 4),
        'ref_92_16x_rank_alloc': 0.8032}
    dump()

print("QK SHAREDCORE PART 2 DONE", flush=True)
