"""SHARED CORES PART 3: paired per-position standard errors for the headline comparisons.
Re-runs the best shared-atom variant and the best matched-budget rank-allocation candidate
at each budget, keeping the per-position cross-entropy tensors, and reports the PAIRED
difference (shared minus rank-allocation) with its standard error.  Also the best composed
config against the section-92 16-fold anchor (576,288).  Appends to qk_sharedcore.json."""
import json, os, subprocess, sys, time
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
        time.sleep(sleep)
    raise RuntimeError("GPU guard timed out")
gpu_guard()

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
FW = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
HELD = FW[448:600, :128].to(DEV)
B0M = 4; B0 = 6; S_, T_ = HELD.shape

def mlp_wts(li):
    b = m.transformer.h[li].mlp
    return (b.Left.weight.detach().float(), b.Right.weight.detach().float(),
            b.Down.weight.detach().float(), b.Down_bias.detach().float())
WTS = [mlp_wts(li) for li in range(NL)]
BIAS = [WTS[li][3] for li in range(NL)]

cache = torch.load(CPT, map_location='cpu', weights_only=True)
UG = cache['UG'].numpy(); UP = cache['UP'].numpy()
BM = cache['BM'].to(DEV); base = cache['base']
cache92 = torch.load(f'{QK}/qk_rank_alloc_cache.pt', map_location='cpu', weights_only=True)
INb = [b.to(DEV) for b in cache92['INb']]; OUTb = [b.to(DEV) for b in cache92['OUTb']]
MX = [t.to(DEV) for t in cache92['MX']]; MO = [t.to(DEV) for t in cache92['MO']]
res = json.load(open(OUT)); res.setdefault('paired', {})

@torch.no_grad()
def fwd_mix(idx, W):
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
        mo = (BIAS[li] + BM[li, li]).unsqueeze(0).expand(B, -1, -1).clone()
        for mm in range(NL):
            wlm = float(W[li, mm])
            if abs(wlm) < 1e-12: continue
            Lw, Rw, Dw, _ = WTS[mm]
            mo += wlm * (((h @ Lw.T) * (h @ Rw.T)) @ Dw.T - BM[li, mm])
        x = x + mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    return F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1),
                           reduction='none').view(B, T-1)

@torch.no_grad()
def fwd_rank(idx, Kin, Kout):
    PIN = [INb[l][:, :Kin].contiguous() for l in range(NL)]
    POUT = None if Kout >= D else [OUTb[l][:, :Kout].contiguous() for l in range(NL)]
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
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh.reshape(B, T, -1))
        xr = MX[li].unsqueeze(0) + ((x - MX[li].unsqueeze(0)) @ PIN[li]) @ PIN[li].T
        mo = blk.mlp(F.rms_norm(xr, (D,)))
        if POUT is not None:
            mo = MO[li].unsqueeze(0) + ((mo - MO[li].unsqueeze(0)) @ POUT[li]) @ POUT[li].T
        x = x + mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    return F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1),
                           reduction='none').view(B, T-1)

def stats(ce_a, ce_b):
    """a = shared, b = rank-allocation; both vs base and paired a-b."""
    da = (ce_a - base).flatten().double(); db = (ce_b - base).flatten().double()
    dd = (ce_a - ce_b).flatten().double()
    return {'shared_dCE': round(float(da.mean()), 4),
            'shared_SE': round(float(da.std()/np.sqrt(da.numel())), 5),
            'rank_dCE': round(float(db.mean()), 4),
            'rank_SE': round(float(db.std()/np.sqrt(db.numel())), 5),
            'paired_diff': round(float(dd.mean()), 4),
            'paired_SE': round(float(dd.std()/np.sqrt(dd.numel())), 5),
            'z': round(float(dd.mean()/(dd.std()/np.sqrt(dd.numel()))), 1)}

# (k, best shared variant U + k, best rank candidate)
HEAD = [(16, UP, 'pooled', (1152, 1024)),
        (12, UP, 'pooled', (1006, 1006)),
        (8, UP, 'pooled', (879, 879)),
        (4, UG, 'frobenius', (696, 696)),
        (2, UG, 'frobenius', (552, 552))]
for (k, U, ut, (Kin, Kout)) in HEAD:
    key = f'k{k}'
    if key in res['paired']: continue
    W = U[:, :k] @ U[:, :k].T
    ce_a = torch.cat([fwd_mix(HELD[i:i+B0M], W).cpu() for i in range(0, S_, B0M)], 0)
    ce_b = torch.cat([fwd_rank(HELD[i:i+B0], Kin, Kout).cpu() for i in range(0, S_, B0)], 0)
    st = stats(ce_a, ce_b)
    st.update({'shared_variant': ut, 'rank_config': [Kin, Kout]})
    res['paired'][key] = st
    print(f"[k={k}] shared({ut}) {st['shared_dCE']:+.4f} vs rank{(Kin,Kout)} "
          f"{st['rank_dCE']:+.4f}  paired {st['paired_diff']:+.4f} +- {st['paired_SE']:.5f} "
          f"z={st['z']}", flush=True)
    json.dump(res, open(OUT, 'w'), indent=1)
# selection-vs-mixing probe: does the atom MIXTURE add anything beyond keeping k blocks?
# W = hard diagonal selection of the same blocks the k=8 pooled PCA effectively keeps.
if 'k8_selection_probe' not in res['paired']:
    W8 = UP[:, :8] @ UP[:, :8].T
    keep = list(np.argsort(-np.diag(W8))[:8])
    Wsel = np.zeros((NL, NL)); Wsel[keep, keep] = 1.0
    ce_sel = torch.cat([fwd_mix(HELD[i:i+B0M], Wsel).cpu() for i in range(0, S_, B0M)], 0)
    ce_pca = torch.cat([fwd_mix(HELD[i:i+B0M], W8).cpu() for i in range(0, S_, B0M)], 0)
    dsel = (ce_sel - base).flatten().double(); dpca = (ce_pca - base).flatten().double()
    dd = (ce_pca - ce_sel).flatten().double()
    res['paired']['k8_selection_probe'] = {
        'kept_blocks': [int(b) for b in sorted(keep)],
        'hard_selection_dCE': round(float(dsel.mean()), 4),
        'hard_selection_SE': round(float(dsel.std()/np.sqrt(dsel.numel())), 5),
        'pca_mixture_dCE': round(float(dpca.mean()), 4),
        'paired_pca_minus_selection': round(float(dd.mean()), 4),
        'paired_SE': round(float(dd.std()/np.sqrt(dd.numel())), 5)}
    print(f"[selection probe] keep {sorted(keep)}: hard {float(dsel.mean()):+.4f} "
          f"pca-mixture {float(dpca.mean()):+.4f} paired diff {float(dd.mean()):+.4f}", flush=True)
    json.dump(res, open(OUT, 'w'), indent=1)
print("QK SHAREDCORE PART 3 DONE", flush=True)
