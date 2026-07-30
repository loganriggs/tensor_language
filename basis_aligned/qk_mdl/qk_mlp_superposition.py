"""MLP SUPERPOSITION vs RANK-CUTOFF test for bilin18 (§71 follow-up).

The §71 coverage ledger found that ~73% of bilin18's feed-forward causal effect
(mean-ablation delta cross-entropy, attention intact) sits BELOW the top-72 MLP
SVD directions (top-4 per block * 18 blocks) and labeled that residual
"non-axis-aligned / superposition". This script tests whether that residual is
GENUINE high-rank distributed structure or merely a rank-cutoff artifact: it
sweeps the retained MLP-direction basis per block and measures how the captured
causal effect grows with K.

FORWARD + mean-ablation COPIED VERBATIM from qk_coverage_ledger.py /
qk_unsup_verify.py (tier2_model.reference_forward): bilin18 two-branch pattern
(q1k1/HD)(q2k2/HD) masked UNNORMALISED, per-head QK rms_norm THEN RoPE, v-lerp
via a.lamb (block-0 v cache), 30*tanh logits. Mean-ablation is per-position mean
over the held slice (NOT zero). MLP SVD directions recomputed from the TRAIN
gram exactly as the ledger does (eigh, top eigenvectors descending) but extended
to the top-64 per block. Random orthonormal control bases are QR of gaussians.

Everything is measured ATTENTION INTACT (heads never ablated) -- the same
context as the ledger's mlp_full (4.53) and mlp_top72 (1.22) reference points.
"""
import json, sys, math, time, subprocess
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'

# ---------------- GPU GUARD (verbatim from ledger) ----------------
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
KMAX = 64                                       # top-64 per block == 1152 total, near full rank
K_SWEEP = [1, 2, 4, 8, 16, 32, 64]              # K=4 reproduces §71 top-72
print(f"bilin18 NL={NL} NH={NH} HD={HD} D={D} V={V}; KMAX/block={KMAX}", flush=True)

FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
SEQL = 128
TRAIN = FINEWEB[0:256, :SEQL].to(DEV)           # discovery slice -- ONLY to recompute MLP dirs
HELD = FINEWEB[448:600, :SEQL].to(DEV)          # held-back verification slice
NHELD = HELD.shape[0]
BATCH = 6

# =====================================================================================
# MLP directions: recompute top-KMAX SVD dirs per block from the TRAIN gram (verbatim ledger).
# =====================================================================================
gram = [torch.zeros(D, D, device=DEV) for _ in range(NL)]

@torch.no_grad()
def fwd_gram(idx):
    B, T = idx.shape
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
        pat = (s1*s2).masked_fill(~mask, 0.0)
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        mo = blk.mlp(F.rms_norm(x, (D,)))
        gram[li] += torch.einsum('btd,bte->de', mo, mo)
        x = x + mo

print("Recomputing MLP SVD directions from TRAIN gram ...", flush=True)
for i in range(0, TRAIN.shape[0], BATCH):
    fwd_gram(TRAIN[i:i+BATCH])
SVD_DIRS = torch.zeros(NL, KMAX, D, device=DEV)
gram_eval = np.zeros((NL, D), np.float64)       # eigenvalues (ascending) per layer, for energy ref
for li in range(NL):
    evals, evecs = torch.linalg.eigh(gram[li])
    SVD_DIRS[li] = evecs[:, -KMAX:].T.flip(0)               # top-KMAX, descending
    gram_eval[li] = evals.cpu().numpy()
del gram
# random orthonormal control basis per block (QR of gaussian), fixed seed
g = torch.Generator(device=DEV); g.manual_seed(1234)
RAND_DIRS = torch.zeros(NL, KMAX, D, device=DEV)
for li in range(NL):
    A = torch.randn(D, KMAX, device=DEV, generator=g)
    Q, _ = torch.linalg.qr(A)                                # (D,KMAX) orthonormal columns
    RAND_DIRS[li] = Q.T
print("MLP SVD + random-orthonormal directions ready.", flush=True)

# =====================================================================================
# Core forward (VERBATIM ledger) with flexible MLP ablation.
#   spec = {'mlp_full': None|'all'|set(li),
#           'proj': None|dict{li: (Dirs (k,D), ProjMean (T,k))}}   # project out dirs -> replace by per-pos mean
#   mlp_full takes precedence for a covered layer.
# =====================================================================================
@torch.no_grad()
def forward(idx, collect=False, spec=None):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    mfull = spec.get('mlp_full') if spec else None
    proj = spec.get('proj') if spec else None
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0)
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)          # attention ALWAYS intact
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if collect:
            MO_SUM[li] += mo.sum(0)                            # (T,D)
            prS = torch.einsum('btd,kd->btk', mo, SVD_DIRS[li]); SPROJ_SUM[li] += prS.sum(0)   # (T,KMAX)
            prR = torch.einsum('btd,kd->btk', mo, RAND_DIRS[li]); RPROJ_SUM[li] += prR.sum(0)
        # ---- mlp ablation ----
        if mfull == 'all' or (mfull and li in mfull):
            mo = MOMEAN[li].unsqueeze(0).expand(B, -1, -1)
        elif proj is not None and li in proj:
            Dirs, PM = proj[li]                                # (k,D), (T,k)
            pr = torch.einsum('btd,kd->btk', mo, Dirs)         # (B,T,k)
            coeff = pr - PM.unsqueeze(0)                       # (B,T,k)
            mo = mo - torch.einsum('btk,kd->btd', coeff, Dirs)
        x = x + mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    return logits

# =====================================================================================
# PASS A: per-position means (mlp-output full, SVD proj, random proj) over held-out.
# =====================================================================================
MO_SUM = {li: torch.zeros(SEQL, D, device=DEV) for li in range(NL)}
SPROJ_SUM = {li: torch.zeros(SEQL, KMAX, device=DEV) for li in range(NL)}
RPROJ_SUM = {li: torch.zeros(SEQL, KMAX, device=DEV) for li in range(NL)}
print("PASS A: per-position means over held-out ...", flush=True)
for i in range(0, NHELD, BATCH):
    forward(HELD[i:i+BATCH], collect=True)
MOMEAN = {li: MO_SUM[li] / NHELD for li in range(NL)}
SPROJMEAN = {li: SPROJ_SUM[li] / NHELD for li in range(NL)}   # (T,KMAX)
RPROJMEAN = {li: RPROJ_SUM[li] / NHELD for li in range(NL)}
del MO_SUM, SPROJ_SUM, RPROJ_SUM
print("PASS A done.", flush=True)

# =====================================================================================
# Build ablation configs.
# =====================================================================================
def svd_topK_all(K):
    return {'proj': {li: (SVD_DIRS[li, :K].contiguous(), SPROJMEAN[li][:, :K].contiguous()) for li in range(NL)}}
def rand_topK_all(K):
    return {'proj': {li: (RAND_DIRS[li, :K].contiguous(), RPROJMEAN[li][:, :K].contiguous()) for li in range(NL)}}

CONFIGS = {'MLP_FULL': {'mlp_full': 'all'}}
for K in K_SWEEP:
    CONFIGS[f'SVD_K{K}'] = svd_topK_all(K)
    CONFIGS[f'RAND_K{K}'] = rand_topK_all(K)
# per-layer attribution: full-MLP per layer, and top-4 SVD per layer (K=4 == §71 top-72 slice)
for li in range(NL):
    CONFIGS[f'LFULL_{li}'] = {'mlp_full': {li}}
    CONFIGS[f'LTOP4_{li}'] = {'proj': {li: (SVD_DIRS[li, :4].contiguous(), SPROJMEAN[li][:, :4].contiguous())}}
print(f"{len(CONFIGS)} configs; K sweep {K_SWEEP}", flush=True)

# =====================================================================================
# PASS B: base + each config -> per-position paired delta cross-entropy.
# =====================================================================================
NPOS = NHELD * (SEQL - 1)
per_pos = {name: np.zeros(NPOS, np.float64) for name in CONFIGS}
print(f"PASS B: base + {len(CONFIGS)} configs over held-out ({NPOS} positions) ...", flush=True)
t0 = time.time(); ptr = 0
for bi, i in enumerate(range(0, NHELD, BATCH)):
    idx = HELD[i:min(i+BATCH, NHELD)]; b = idx.shape[0]
    tgt = idx
    base = forward(idx).float()
    logp = F.log_softmax(base[:, :SEQL-1], dim=-1)
    base_ce = -logp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1)   # (b,T-1)
    del base, logp
    n_here = b * (SEQL - 1)
    for name, spec in CONFIGS.items():
        abl = forward(idx, spec=spec).float()
        alogp = F.log_softmax(abl[:, :SEQL-1], dim=-1)
        abl_ce = -alogp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1)
        del abl, alogp
        per_pos[name][ptr:ptr+n_here] = (abl_ce - base_ce).reshape(-1).cpu().numpy()
    ptr += n_here
    if bi % 4 == 0:
        print(f"  batch {bi+1}/{(NHELD+BATCH-1)//BATCH}  elapsed {time.time()-t0:.0f}s", flush=True)
assert ptr == NPOS, (ptr, NPOS)
print(f"PASS B done in {time.time()-t0:.0f}s", flush=True)

# =====================================================================================
# STATS
# =====================================================================================
def mean_se(arr):
    return float(arr.mean()), float(arr.std(ddof=1)/math.sqrt(len(arr)))
def diff_se(a, b):
    d = a - b; return float(d.mean()), float(d.std(ddof=1)/math.sqrt(len(d)))

M = {name: mean_se(per_pos[name]) for name in CONFIGS}
FULL = M['MLP_FULL'][0]

# rank-sweep curves (captured dCE vs K), paired gap to full MLP
svd_curve, rand_curve = [], []
for K in K_SWEEP:
    sm, sse = M[f'SVD_K{K}']; gap, gse = diff_se(per_pos['MLP_FULL'], per_pos[f'SVD_K{K}'])
    svd_curve.append({'K_per_block': K, 'total_dirs': K*NL, 'captured_dCE': round(sm, 5), 'SE': round(sse, 5),
                      'frac_of_full': round(sm/FULL, 4), 'gap_to_full_dCE': round(gap, 5), 'gap_SE': round(gse, 5)})
    rm, rse = M[f'RAND_K{K}']
    rand_curve.append({'K_per_block': K, 'total_dirs': K*NL, 'captured_dCE': round(rm, 5), 'SE': round(rse, 5),
                       'frac_of_full': round(rm/FULL, 4)})

# effective rank: interpolate K (per block) needed for 50/80/90% of full MLP effect
def eff_rank(frac_target):
    xs = [c['K_per_block'] for c in svd_curve]; ys = [c['frac_of_full'] for c in svd_curve]
    if ys[0] >= frac_target:
        return round(xs[0] * frac_target / ys[0], 2)         # below first sampled K
    for j in range(1, len(xs)):
        if ys[j] >= frac_target:
            x0, x1, y0, y1 = xs[j-1], xs[j], ys[j-1], ys[j]
            return round(x0 + (frac_target - y0) * (x1 - x0) / (y1 - y0), 2)
    return None   # not reached even at KMAX
eff = {f'{int(p*100)}pct': eff_rank(p) for p in (0.5, 0.8, 0.9)}

# per-layer tail attribution: tail_li = LFULL_li - LTOP4_li (both attention intact, single layer)
layer_tab = []
for li in range(NL):
    lf, lfse = M[f'LFULL_{li}']; lt, ltse = M[f'LTOP4_{li}']
    tail, tse = diff_se(per_pos[f'LFULL_{li}'], per_pos[f'LTOP4_{li}'])
    layer_tab.append({'layer': li, 'full_mlp_dCE': round(lf, 5), 'top4_dCE': round(lt, 5),
                      'tail_below_top4_dCE': round(tail, 5), 'tail_SE': round(tse, 5),
                      'tail_frac_of_layer_full': round(tail/lf, 4) if lf > 1e-9 else None,
                      'gram_energy_top4_frac': round(float(gram_eval[li, -4:].sum()/gram_eval[li].sum()), 4)})
tail_total = sum(max(0.0, r['tail_below_top4_dCE']) for r in layer_tab)
for r in layer_tab:
    r['tail_share_across_layers'] = round(max(0.0, r['tail_below_top4_dCE'])/tail_total, 4) if tail_total else None

out = {
    'meta': {
        'model': 'bilin18', 'held_slice': 'FW[448:600,:128]', 'n_positions': NPOS, 'BATCH': BATCH,
        'NL': NL, 'D': D, 'KMAX_per_block': KMAX, 'K_sweep_per_block': K_SWEEP,
        'forward': 'VERBATIM qk_coverage_ledger.py / qk_unsup_verify.py (tier2_model.reference_forward), ATTENTION INTACT',
        'currency': 'mean-ablation (per-position held mean) delta cross-entropy per valid held position (nats), paired SE over positions',
        'question': 'is the sub-top-72 feed-forward causal effect genuine high-rank superposition or a rank-cutoff artifact of top-4/block?',
    },
    'reference_points': {
        'mlp_full_dCE': round(FULL, 5), 'mlp_full_SE': round(M['MLP_FULL'][1], 5),
        'mlp_top72_dCE(=SVD_K4)': round(M['SVD_K4'][0], 5), 'mlp_top72_SE': round(M['SVD_K4'][1], 5),
        'below_top72_dCE': round(diff_se(per_pos['MLP_FULL'], per_pos['SVD_K4'])[0], 5),
        'below_top72_frac_of_full': round(diff_se(per_pos['MLP_FULL'], per_pos['SVD_K4'])[0]/FULL, 4),
        'ledger_expected': {'mlp_full~4.53': 4.532147, 'mlp_top72~1.22': 1.223759},
    },
    'rank_sweep_svd': svd_curve,
    'rank_sweep_random': rand_curve,
    'effective_rank_per_block_for_pct_of_full': eff,
    'svd_vs_random_at_K': {str(K): {'svd_frac': svd_curve[j]['frac_of_full'], 'rand_frac': rand_curve[j]['frac_of_full'],
                                    'svd_over_rand': round(svd_curve[j]['captured_dCE']/rand_curve[j]['captured_dCE'], 3)
                                    if rand_curve[j]['captured_dCE'] > 1e-9 else None}
                           for j, K in enumerate(K_SWEEP)},
    'per_layer_tail': layer_tab,
}
json.dump(out, open(f'{QK}/qk_mlp_superposition.json', 'w'), indent=2)

# ---------------- console summary ----------------
print("\n===== MLP SUPERPOSITION vs RANK-CUTOFF (delta cross-entropy, nats, attention intact) =====", flush=True)
print(f"MLP_FULL {FULL:.4f}   top-72(=SVD_K4) {M['SVD_K4'][0]:.4f}   below-top72 {out['reference_points']['below_top72_dCE']:.4f} "
      f"({out['reference_points']['below_top72_frac_of_full']*100:.0f}% of full)", flush=True)
print("K/block | total | SVD captured (frac) | RAND captured (frac)", flush=True)
for j, K in enumerate(K_SWEEP):
    s, r = svd_curve[j], rand_curve[j]
    print(f"  {K:3d}   | {K*NL:4d} | {s['captured_dCE']:.4f} ({s['frac_of_full']:.3f})   | {r['captured_dCE']:.4f} ({r['frac_of_full']:.3f})", flush=True)
print(f"Effective rank per block for 50/80/90% of full MLP effect: {eff}", flush=True)
lays = sorted(layer_tab, key=lambda r: -r['tail_below_top4_dCE'])[:5]
print("Top tail-carrying layers:", [(r['layer'], round(r['tail_below_top4_dCE'],3), r['tail_share_across_layers']) for r in lays], flush=True)
print("\nSaved qk_mlp_superposition.json", flush=True)
print("QK MLP SUPERPOSITION DONE", flush=True)
