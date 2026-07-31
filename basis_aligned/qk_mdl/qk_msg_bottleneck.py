"""COMPRESSED FUNCTIONAL MODEL of the mid-stack distributed region (blocks 5-11) -- script 1 of 2.

EXPERIMENT 1 -- THE JOINT-WRITE BOTTLENECK (message dimension).
The region's JOINT WRITE W(x) = sum over blocks 5..11 of the feed-forward output deviation from its
per-position mean (mean-ablation floor convention VERBATIM qk_cascade_end / qk_redteam_arcs: per-position
HELD means; deviation-keep convention VERBATIM qk_hub_hierarchy: live deviation, keep-subspace projection,
extended from one block to the seven-block region with a SHARED basis).
  (gate)   reassembly: mo_li := MEAN_li + (mo_li - MEAN_li) for all seven blocks -- must reproduce the
           model exactly (|dCE| < 0.003, the freeze-all sanity threshold of qk_redteam_arcs).
  (floor)  all seven blocks' outputs replaced by their per-position held means simultaneously -- the
           JOINT FLOOR, one number with paired standard error.
  (suff)   mo_li := MEAN_li + P_k P_k^T (mo_li - MEAN_li), P_k = top-k principal directions of the TRAIN
           gram of the summed deviation W (per-position TRAIN means for centering). By linearity the
           injected joint write is exactly the top-k projection of the live joint write. k in
           {4,16,64,144,288}; RANDOM-k control = prefix columns of one random orthogonal matrix
           (seed 2, VERBATIM qk_hub_hierarchy).
Report: k for 90/95/99% recovery of the joint floor = THE MESSAGE DIMENSION of the region.

Also collects (for script 2): full eigenbasis U + eigenvalues; held message coordinates on the top 144
directions; per-sequence mean layer-4 residual (train + held) for the topic-cluster proxy; per-position
held means of blocks 5..11. Saved to qk_msg_bottleneck_state.pt.

Model tier2_model.load_elriggs('bilin18'). TRAIN FW[0:256,:128] for all fitting (gram, centering means,
cluster features); held-back FW[448:600,:128] for all causal numbers, paired standard errors. Batch 6,
footprint <4GB, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, GPU guard. Writes qk_msg_bottleneck.json."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json, sys, time, subprocess
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/qk_msg_bottleneck.json'
STATE = f'{QK}/qk_msg_bottleneck_state.pt'

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
TRAIN = FW[0:256, :128].to(DEV); HELD = FW[448:600, :128].to(DEV); B0 = 6
STR, TTR = TRAIN.shape; S_, T_ = HELD.shape
BLK = list(range(5, 12)); BLKSET = set(BLK)
print(f"bilin18 NL={NL} D={D} region blocks {BLK} train {STR}x{TTR} held {S_}x{T_}", flush=True)

# =====================================================================================
# Unified forward (skeleton VERBATIM qk_allcore_restrict.py fwd).
#   mode: None | 'ident' | 'floor' | 'proj'   (applied to every block in BLK)
#   P: (D,k) orthonormal keep-basis for 'proj'; MEANS: dict li -> (T,D) per-position means
#   collect: dict with keys 'sums' (per-block per-position sums), 'w' (list, joint raw
#            write sum per batch, cpu), 'l4' (list, per-sequence mean layer-4 residual)
# =====================================================================================
@torch.no_grad()
def fwd(idx, mode=None, P=None, MEANS=None, collect=None):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    wsum = None
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(l): z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh.reshape(B, T, -1))
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if li in BLKSET:
            if collect is not None:
                if 'sums' in collect: collect['sums'][li] += mo.float().sum(0)
                wsum = mo.float() if wsum is None else wsum + mo.float()
            if mode == 'floor':
                mo = MEANS[li].unsqueeze(0).to(mo.dtype).expand(B, -1, -1)
            elif mode == 'ident':
                mo = MEANS[li].unsqueeze(0) + (mo - MEANS[li].unsqueeze(0))
            elif mode == 'proj':
                dev = mo - MEANS[li].unsqueeze(0)
                mo = MEANS[li].unsqueeze(0) + (dev @ P) @ P.T
        x = x + mo
        if li == 4 and collect is not None and 'l4' in collect:
            collect['l4'].append(x.mean(1).float().cpu())
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1),
                         reduction='none').view(B, T-1)
    return ce, wsum

# =====================================================================================
# PASS 1 (TRAIN): gram of the summed deviation + per-sequence layer-4 residual means.
# Deviation gram via the exact identity: G_dev = G_raw - N * sum_t Wbar_t Wbar_t^T
# where Wbar_t is the per-position TRAIN mean of the raw joint write.
# =====================================================================================
print("PASS 1 (TRAIN): joint-write gram + layer-4 features ...", flush=True)
G_raw = torch.zeros(D, D, device=DEV, dtype=torch.float64)
WS_sum = torch.zeros(TTR, D, device=DEV, dtype=torch.float64)
l4_train = []
t0 = time.time()
for i in range(0, STR, B0):
    col = {'l4': l4_train, 'sums': {li: torch.zeros(TTR, D, device=DEV) for li in BLK}}
    _, wsum = fwd(TRAIN[i:i+B0], collect=col)
    G_raw += torch.einsum('btd,bte->de', wsum.double(), wsum.double())
    WS_sum += wsum.double().sum(0)
    del col, wsum
print(f"  gram pass done ({time.time()-t0:.0f}s)", flush=True)
Wbar_tr = WS_sum / STR
G_dev = G_raw - STR * torch.einsum('td,te->de', Wbar_tr, Wbar_tr)
eigval, eigvec = torch.linalg.eigh(G_dev)
eigval = eigval.flip(0).clamp_min(0); U = eigvec.flip(1).float().contiguous()   # cols desc
tot_var = float(eigval.sum())
cum = torch.cumsum(eigval, 0) / tot_var
var_at = {k: float(cum[k-1]) for k in [4, 16, 64, 144, 288, 432, 576, 864]}
print(f"  train deviation variance captured by top-k: {var_at}", flush=True)
L4_TRAIN = torch.cat(l4_train, 0)                                # (STR, D)

# =====================================================================================
# PASS 2 (HELD, clean): base CE + per-block per-position mean sums + raw joint write
# (stored, cpu fp32) + layer-4 features.
# =====================================================================================
print("PASS 2 (HELD): base CE + per-position means + joint write ...", flush=True)
sums = {li: torch.zeros(T_, D, device=DEV) for li in BLK}
l4_held, w_held, bce = [], [], []
for i in range(0, S_, B0):
    ce, wsum = fwd(HELD[i:i+B0], collect={'sums': sums, 'l4': l4_held})
    bce.append(ce.cpu()); w_held.append(wsum.cpu())
base = torch.cat(bce, 0)
MEANS = {li: (sums[li]/S_) for li in BLK}
L4_HELD = torch.cat(l4_held, 0)
W_HELD_RAW = torch.cat(w_held, 0)                                # (S,T,D) raw joint write
WBAR_HELD = torch.stack([MEANS[li] for li in BLK]).sum(0).cpu()  # per-position mean of joint write
M_HELD = ((W_HELD_RAW - WBAR_HELD.unsqueeze(0)) @ U[:, :144].cpu())  # (S,T,144) message coords
del sums, l4_held, w_held, bce
print(f"base CE {float(base.mean()):.4f}", flush=True)

def dstat(ce):
    d = (ce - base).flatten().double(); return float(d.mean()), float(d.std()/np.sqrt(d.numel()))

def run(mode, P=None):
    ces = []
    for i in range(0, S_, B0):
        ce, _ = fwd(HELD[i:i+B0], mode=mode, P=P, MEANS=MEANS); ces.append(ce.cpu())
    return dstat(torch.cat(ces, 0))

# =====================================================================================
# GATES + FLOOR + SUFFICIENCY SWEEP
# =====================================================================================
print("GATE: reassembly (mean + own deviation, all 7 blocks) ...", flush=True)
g_mn, g_se = run('ident')
print(f"  reassembly dCE {g_mn:+.5f} +- {g_se:.6f}", flush=True)
assert abs(g_mn) < 0.003, f"reassembly gate FAILED: {g_mn}"

print("JOINT FLOOR: all 7 blocks -> per-position held means ...", flush=True)
f_mn, f_se = run('floor')
CENSUS = {int(li): rec['floor_dCE'] for li, rec in
          json.load(open(f'{QK}/qk_allterm_census.json'))['layers'].items()}
census_sum = sum(CENSUS[li] for li in BLK)
print(f"  joint floor dCE {f_mn:+.4f} +- {f_se:.5f}  (sum of per-block census floors {census_sum:.4f})", flush=True)

KS = [4, 16, 64, 144, 288, 432, 576, 864]   # spec list {4..288} extended upward: D=1152 and the
                                            # 90/95/99 crossings lie above 288 (first run)
res_svd, res_rand = {}, {}
print("SUFFICIENCY: top-k principal directions of W (train gram) ...", flush=True)
for K in KS:
    mn, se = run('proj', P=U[:, :K].to(DEV).contiguous())
    rec = 1 - mn/f_mn
    res_svd[K] = {'dCE': round(mn, 5), 'SE': round(se, 6), 'recovered_fraction': round(rec, 4)}
    print(f"  svd k={K:4d}: dCE {mn:+.5f} +- {se:.6f}  recovered {rec:.4f}", flush=True)
g = torch.Generator(device=DEV).manual_seed(2)
Q = torch.linalg.qr(torch.randn(D, D, generator=g, device=DEV))[0]
print("RANDOM-k control (prefix of one random orthogonal matrix, seed 2) ...", flush=True)
for K in KS:
    mn, se = run('proj', P=Q[:, :K].contiguous())
    rec = 1 - mn/f_mn
    res_rand[K] = {'dCE': round(mn, 5), 'SE': round(se, 6), 'recovered_fraction': round(rec, 4)}
    print(f"  rand k={K:4d}: dCE {mn:+.5f} +- {se:.6f}  recovered {rec:.4f}", flush=True)

def k_for(frac):
    for K in KS:
        if res_svd[K]['recovered_fraction'] >= frac: return K
    return None
kf = {str(int(f*100)): k_for(f) for f in (0.90, 0.95, 0.99)}
print(f"message dimension (k for 90/95/99% of joint floor): {kf}", flush=True)

exp1 = {
 'method': 'joint write W = sum over blocks 5-11 of (mlp output - per-position held mean); '
           'live-deviation keep-projection per block onto shared top-k basis of the TRAIN '
           'deviation gram (train FW[0:256,:128]); held FW[448:600,:128]; paired SE',
 'base_CE': round(float(base.mean()), 4),
 'reassembly_gate': {'dCE': round(g_mn, 6), 'SE': round(g_se, 6), 'pass': True},
 'joint_floor': {'dCE': round(f_mn, 4), 'SE': round(f_se, 5)},
 'sum_per_block_census_floors': round(census_sum, 4),
 'train_variance_captured_topk': {str(k): round(v, 4) for k, v in var_at.items()},
 'svd_keep': {str(k): v for k, v in res_svd.items()},
 'rand_keep': {str(k): v for k, v in res_rand.items()},
 'k_for_recovery': kf,
}
json.dump({'exp1_joint_write_bottleneck': exp1}, open(OUT, 'w'), indent=1)

torch.save({'U': U.cpu(), 'eigval': eigval.cpu(), 'MEANS': {li: MEANS[li].cpu() for li in BLK},
            'WBAR_HELD': WBAR_HELD, 'M_HELD': M_HELD, 'L4_TRAIN': L4_TRAIN, 'L4_HELD': L4_HELD,
            'base_ce': base, 'joint_floor': (f_mn, f_se), 'res_svd': res_svd, 'k_for': kf}, STATE)
print(f"Saved {OUT} + {STATE}", flush=True)
print("QK MSG BOTTLENECK (script 1) DONE", flush=True)
