"""CROSS-ARCHITECTURE COMPLETENESS BOUNDARY (§71 + §73) on swiglu18 / bilin12.

Ports bilin18's coverage ledger (qk_coverage_ledger.py) and rank/superposition sweep
(qk_mlp_superposition.py) to a DIFFERENT architecture to test whether the completeness
boundary -- most computation super-additive (combination-only) + a high-rank but
basis-aligned feed-forward tail -- is architecture-general or bilin18-specific.

Usage: python qk_general_completeness.py <model>       (model in {swiglu18, bilin12})

FORWARD conventions (verified from jacclust/tt_model.py CausalSelfAttention +
qk_content_gate_swiglu18.py):
  swiglu18 (bilinear_attn absent, squared_attn=False)  -> SOFTMAX, single QK branch:
      sc = (einsum(q,k)/(HD**0.5)).masked_fill(~mask,-inf); pat = softmax(sc,-1)
  bilin12  (bilinear_attn absent, squared_attn=True)    -> SQUARED ROW-NORMALIZED, single branch:
      s  = einsum(q,k)/HD; pat = s.square().masked_fill(~mask,0); pat /= pat.sum(-1,keepdim).clamp_min(1e-9)
Both: token embed -> global rms_norm -> x0 skip; per block x=l0*x+l1*x0; per-head QK
rms_norm THEN RoPE; v-lerp via a.lamb (block-0 v cache); MLP = blk.mlp black box;
30*tanh logits. Everything else (per-position mean-ablation, project-out-to-mean for
MLP SVD dirs, top-KMAX SVD from the TRAIN gram, random-orthonormal control, paired
standard errors, held slice FW[448:600,:128]) is copied VERBATIM from
qk_coverage_ledger.py / qk_mlp_superposition.py / qk_unsup_verify.py.
"""
import json, sys, math, time, subprocess
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
MODEL = sys.argv[1] if len(sys.argv) > 1 else 'swiglu18'
assert MODEL in ('swiglu18', 'bilin12')

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

m, cfg = load_elriggs(MODEL)
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
ARCH = 'softmax' if not cfg.get('squared_attn') else 'sqrd_norm'   # swiglu18 vs bilin12
N_SVD = 4                                           # top-4 per block == "top" dirs (== bilin18 ledger)
KMAX = 64                                           # rank sweep upper bound per block
K_SWEEP = [1, 2, 4, 8, 16, 32, 64]                  # K=4 reproduces the "top-N_SVD" slice
print(f"{MODEL}: NL={NL} NH={NH} HD={HD} D={D} V={V}; ARCH={ARCH}; "
      f"{NL*NH} head-paths + {NL*N_SVD} top-mlp-paths", flush=True)

FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
SEQL = 128
TRAIN = FINEWEB[0:256, :SEQL].to(DEV)               # discovery slice -- ONLY to recompute MLP dirs
HELD = FINEWEB[448:600, :SEQL].to(DEV)              # held-back verification slice
NHELD = HELD.shape[0]
BATCH = 6

# =====================================================================================
# ARCH-specific attention pattern (single QK branch for both target models).
# =====================================================================================
def attn_pat(q, k, mask):
    if ARCH == 'softmax':
        sc = (torch.einsum('bqhd,bkhd->bhqk', q, k) / (HD**0.5)).masked_fill(~mask, float('-inf'))
        return F.softmax(sc, -1)
    else:  # sqrd_norm (bilin12)
        s = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        pat = s.square().masked_fill(~mask, 0.0)
        return pat / pat.sum(-1, keepdim=True).clamp_min(1e-9)

# =====================================================================================
# MLP directions: top-KMAX SVD dirs per block from the TRAIN gram (verbatim ledger).
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
        q, k = qk(a.c_q), qk(a.c_k)
        pat = attn_pat(q, k, mask)
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        mo = blk.mlp(F.rms_norm(x, (D,)))
        gram[li] += torch.einsum('btd,bte->de', mo, mo)
        x = x + mo

print("Recomputing MLP SVD directions from TRAIN gram ...", flush=True)
for i in range(0, TRAIN.shape[0], BATCH):
    fwd_gram(TRAIN[i:i+BATCH])
SVD_DIRS = torch.zeros(NL, KMAX, D, device=DEV)
gram_eval = np.zeros((NL, D), np.float64)
for li in range(NL):
    evals, evecs = torch.linalg.eigh(gram[li])
    SVD_DIRS[li] = evecs[:, -KMAX:].T.flip(0)                # top-KMAX descending
    gram_eval[li] = evals.cpu().numpy()
del gram
g = torch.Generator(device=DEV); g.manual_seed(1234)
RAND_DIRS = torch.zeros(NL, KMAX, D, device=DEV)
for li in range(NL):
    A = torch.randn(D, KMAX, device=DEV, generator=g)
    Q, _ = torch.linalg.qr(A)
    RAND_DIRS[li] = Q.T
print("MLP SVD + random-orthonormal directions ready.", flush=True)

# =====================================================================================
# Core forward (VERBATIM ledger structure) with flexible head + MLP ablation.
#   spec keys: 'heads' None|'all'|set((li,h)); 'mlp_full' None|'all'|set(li);
#              'mlp_dirs' None|'all'|set((li,kk)) [top-N_SVD project-out];
#              'proj' dict{li:(Dirs(k,D),PM(T,k))} [rank sweep project-out].
# =====================================================================================
@torch.no_grad()
def forward(idx, collect=False, spec=None):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    heads = spec.get('heads') if spec else None
    mdirs = spec.get('mlp_dirs') if spec else None
    mfull = spec.get('mlp_full') if spec else None
    proj = spec.get('proj') if spec else None
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k = qk(a.c_q), qk(a.c_k)
        pat = attn_pat(q, k, mask)
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)         # (B,T,NH,HD)
        if collect:
            YH_SUM[li] += yh4.sum(0)
        # ---- head ablation ----
        if heads == 'all':
            yh4 = YHMEAN[li].unsqueeze(0).expand(B, -1, -1, -1).clone()
        elif heads:
            hs = [h for (l, h) in heads if l == li]
            if hs:
                yh4 = yh4.clone()
                for h in hs:
                    yh4[:, :, h] = YHMEAN[li][:, h].unsqueeze(0)
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if collect:
            MO_SUM[li] += mo.sum(0)
            prS = torch.einsum('btd,kd->btk', mo, SVD_DIRS[li]); SPROJ_SUM[li] += prS.sum(0)
            prR = torch.einsum('btd,kd->btk', mo, RAND_DIRS[li]); RPROJ_SUM[li] += prR.sum(0)
        # ---- mlp ablation ----
        if mfull == 'all' or (mfull and li in mfull):
            mo = MOMEAN[li].unsqueeze(0).expand(B, -1, -1)
        elif proj is not None and li in proj:
            Dirs, PM = proj[li]
            pr = torch.einsum('btd,kd->btk', mo, Dirs)
            coeff = pr - PM.unsqueeze(0)
            mo = mo - torch.einsum('btk,kd->btd', coeff, Dirs)
        elif mdirs == 'all' or mdirs:
            ks = range(N_SVD) if mdirs == 'all' else [kk for (l, kk) in mdirs if l == li]
            for kk in ks:
                pr = torch.einsum('btd,d->bt', mo, SVD_DIRS[li, kk])
                mo = mo - (pr - SPROJMEAN[li][:, kk].unsqueeze(0)).unsqueeze(-1) * SVD_DIRS[li, kk]
        x = x + mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    return logits

# =====================================================================================
# PASS A: per-position means (yh per head, mlp full, SVD proj, random proj).
# =====================================================================================
YH_SUM = {li: torch.zeros(SEQL, NH, HD, device=DEV) for li in range(NL)}
MO_SUM = {li: torch.zeros(SEQL, D, device=DEV) for li in range(NL)}
SPROJ_SUM = {li: torch.zeros(SEQL, KMAX, device=DEV) for li in range(NL)}
RPROJ_SUM = {li: torch.zeros(SEQL, KMAX, device=DEV) for li in range(NL)}
print("PASS A: per-position means over held-out ...", flush=True)
for i in range(0, NHELD, BATCH):
    forward(HELD[i:i+BATCH], collect=True)
YHMEAN = {li: YH_SUM[li] / NHELD for li in range(NL)}
MOMEAN = {li: MO_SUM[li] / NHELD for li in range(NL)}
SPROJMEAN = {li: SPROJ_SUM[li] / NHELD for li in range(NL)}
RPROJMEAN = {li: RPROJ_SUM[li] / NHELD for li in range(NL)}
del YH_SUM, MO_SUM, SPROJ_SUM, RPROJ_SUM
print("PASS A done.", flush=True)

# =====================================================================================
# CONFIG BUILD.
#   JOINT configs (store full per-position dCE arrays, for paired-gap SE):
#     FULL_HEADROOM, JOINT_ALL, ATTN_ALL, MLP_FULL, and rank-sweep SVD_K/RAND_K,
#     per-layer LFULL_li / LTOP4_li.
#   SINGLE-PATH configs (accumulate running sums only, memory-light):
#     one per head-path (li,h) and one per top-N_SVD mlp-dir (li,kk).
# =====================================================================================
def svd_topK_all(K):
    return {'proj': {li: (SVD_DIRS[li, :K].contiguous(), SPROJMEAN[li][:, :K].contiguous()) for li in range(NL)}}
def rand_topK_all(K):
    return {'proj': {li: (RAND_DIRS[li, :K].contiguous(), RPROJMEAN[li][:, :K].contiguous()) for li in range(NL)}}

JOINT = {
    'FULL_HEADROOM': {'heads': 'all', 'mlp_full': 'all'},
    'JOINT_ALL':     {'heads': 'all', 'mlp_dirs': 'all'},          # all heads + all top-N_SVD dirs
    'ATTN_ALL':      {'heads': 'all'},
    'MLP_FULL':      {'mlp_full': 'all'},
}
for K in K_SWEEP:
    JOINT[f'SVD_K{K}'] = svd_topK_all(K)
    JOINT[f'RAND_K{K}'] = rand_topK_all(K)
for li in range(NL):
    JOINT[f'LFULL_{li}'] = {'mlp_full': {li}}
    JOINT[f'LTOP4_{li}'] = {'proj': {li: (SVD_DIRS[li, :N_SVD].contiguous(), SPROJMEAN[li][:, :N_SVD].contiguous())}}
# MLP_TOP == SVD_K{N_SVD}
MLP_TOP_KEY = f'SVD_K{N_SVD}'

SINGLE_HEADS = [(li, h) for li in range(NL) for h in range(NH)]
SINGLE_DIRS = [(li, kk) for li in range(NL) for kk in range(N_SVD)]
print(f"{len(JOINT)} joint configs; {len(SINGLE_HEADS)} single head-paths + {len(SINGLE_DIRS)} single mlp-dirs", flush=True)

# =====================================================================================
# PASS B: base + configs -> per-position paired delta cross-entropy.
#   valid positions = every query t in 0..SEQL-2 (all NHELD seqs).  Global scale.
# =====================================================================================
NPOS = NHELD * (SEQL - 1)
per_pos = {name: np.zeros(NPOS, np.float64) for name in JOINT}
# single-path running sums over valid positions (global dCE)
sh_sum = {p: 0.0 for p in SINGLE_HEADS}; sh_sq = {p: 0.0 for p in SINGLE_HEADS}; sh_n = {p: 0 for p in SINGLE_HEADS}
sd_sum = {p: 0.0 for p in SINGLE_DIRS}; sd_sq = {p: 0.0 for p in SINGLE_DIRS}; sd_n = {p: 0 for p in SINGLE_DIRS}

print(f"PASS B: base + {len(JOINT)} joint + {len(SINGLE_HEADS)+len(SINGLE_DIRS)} single over held-out ({NPOS} pos) ...", flush=True)
t0 = time.time(); ptr = 0
for bi, i in enumerate(range(0, NHELD, BATCH)):
    idx = HELD[i:min(i+BATCH, NHELD)]; b = idx.shape[0]; tgt = idx
    base = forward(idx).float()
    logp = F.log_softmax(base[:, :SEQL-1], dim=-1)
    base_ce = -logp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1)   # (b,T-1)
    del base, logp
    n_here = b * (SEQL - 1)
    # joint configs -> full per-position arrays
    for name, spec in JOINT.items():
        abl = forward(idx, spec=spec).float()
        alogp = F.log_softmax(abl[:, :SEQL-1], dim=-1)
        abl_ce = -alogp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1)
        del abl, alogp
        per_pos[name][ptr:ptr+n_here] = (abl_ce - base_ce).reshape(-1).cpu().numpy()
    # single head-paths -> running sums
    for p in SINGLE_HEADS:
        abl = forward(idx, spec={'heads': {p}}).float()
        alogp = F.log_softmax(abl[:, :SEQL-1], dim=-1)
        abl_ce = -alogp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1)
        d = (abl_ce - base_ce).reshape(-1)
        sh_sum[p] += float(d.sum()); sh_sq[p] += float((d*d).sum()); sh_n[p] += int(d.numel())
        del abl, alogp
    # single mlp-dirs -> running sums
    for p in SINGLE_DIRS:
        abl = forward(idx, spec={'mlp_dirs': {p}}).float()
        alogp = F.log_softmax(abl[:, :SEQL-1], dim=-1)
        abl_ce = -alogp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1)
        d = (abl_ce - base_ce).reshape(-1)
        sd_sum[p] += float(d.sum()); sd_sq[p] += float((d*d).sum()); sd_n[p] += int(d.numel())
        del abl, alogp
    ptr += n_here
    if bi % 2 == 0:
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
def run_stat(s, sq, n):
    if n <= 1: return 0.0, 0.0
    mean = s/n; var = max(sq/n - mean*mean, 0.0)*n/(n-1)
    return mean, math.sqrt(var/n)

M = {name: mean_se(per_pos[name]) for name in JOINT}
H = M['FULL_HEADROOM'][0]; J = M['JOINT_ALL'][0]
FULL_MLP = M['MLP_FULL'][0]

# ---- single-path sums (super-additivity) ----
def pos(x): return x if x > 0 else 0.0
head_stats = {p: run_stat(sh_sum[p], sh_sq[p], sh_n[p]) for p in SINGLE_HEADS}
dir_stats = {p: run_stat(sd_sum[p], sd_sq[p], sd_n[p]) for p in SINGLE_DIRS}
single_total = sum(pos(v[0]) for v in head_stats.values()) + sum(pos(v[0]) for v in dir_stats.values())
single_total_se = math.sqrt(sum(v[1]**2 for v in head_stats.values() if v[0] > 0)
                            + sum(v[1]**2 for v in dir_stats.values() if v[0] > 0))
super_add_ratio = J / single_total if single_total else None

# ---- coverage ledger fractions ----
def frac(x): return round(x / H, 4) if H else None
HJ_gap, HJ_gap_se = diff_se(per_pos['FULL_HEADROOM'], per_pos['JOINT_ALL'])       # non-axis residual
MLP_below_top, MLP_below_top_se = diff_se(per_pos['MLP_FULL'], per_pos[MLP_TOP_KEY])

# ---- rank sweep curves ----
svd_curve, rand_curve = [], []
for K in K_SWEEP:
    sm, sse = M[f'SVD_K{K}']; gap, gse = diff_se(per_pos['MLP_FULL'], per_pos[f'SVD_K{K}'])
    svd_curve.append({'K_per_block': K, 'total_dirs': K*NL, 'captured_dCE': round(sm, 5), 'SE': round(sse, 5),
                      'frac_of_full': round(sm/FULL_MLP, 4) if FULL_MLP else None,
                      'gap_to_full_dCE': round(gap, 5), 'gap_SE': round(gse, 5)})
    rm, rse = M[f'RAND_K{K}']
    rand_curve.append({'K_per_block': K, 'total_dirs': K*NL, 'captured_dCE': round(rm, 5), 'SE': round(rse, 5),
                       'frac_of_full': round(rm/FULL_MLP, 4) if FULL_MLP else None})
def eff_rank(frac_target):
    xs = [c['K_per_block'] for c in svd_curve]; ys = [c['frac_of_full'] for c in svd_curve]
    if ys[0] is None: return None
    if ys[0] >= frac_target:
        return round(xs[0] * frac_target / ys[0], 2)
    for j in range(1, len(xs)):
        if ys[j] is not None and ys[j] >= frac_target:
            x0, x1, y0, y1 = xs[j-1], xs[j], ys[j-1], ys[j]
            return round(x0 + (frac_target - y0) * (x1 - x0) / (y1 - y0), 2)
    return None
eff = {f'{int(p*100)}pct': eff_rank(p) for p in (0.5, 0.8, 0.9)}

# ---- per-layer tail attribution (hub identification) ----
layer_tab = []
for li in range(NL):
    lf, _ = M[f'LFULL_{li}']; lt, _ = M[f'LTOP4_{li}']
    tail, tse = diff_se(per_pos[f'LFULL_{li}'], per_pos[f'LTOP4_{li}'])
    layer_tab.append({'layer': li, 'full_mlp_dCE': round(lf, 5), 'top4_dCE': round(lt, 5),
                      'tail_below_top4_dCE': round(tail, 5), 'tail_SE': round(tse, 5),
                      'tail_frac_of_layer_full': round(tail/lf, 4) if lf > 1e-9 else None,
                      'gram_energy_top4_frac': round(float(gram_eval[li, -N_SVD:].sum()/gram_eval[li].sum()), 4)})
tail_total = sum(max(0.0, r['tail_below_top4_dCE']) for r in layer_tab)
for r in layer_tab:
    r['tail_share_across_layers'] = round(max(0.0, r['tail_below_top4_dCE'])/tail_total, 4) if tail_total else None
# hub = max-tail layer overall, and max-tail among EARLY layers (first half)
hub_overall = max(layer_tab, key=lambda r: r['tail_below_top4_dCE'])['layer']
early = [r for r in layer_tab if r['layer'] < NL // 2]
hub_early = max(early, key=lambda r: r['tail_below_top4_dCE'])['layer']

# ---- top single paths (for report) ----
head_rank = sorted(head_stats.items(), key=lambda kv: -kv[1][0])[:12]
dir_rank = sorted(dir_stats.items(), key=lambda kv: -kv[1][0])[:12]

out = {
    'meta': {
        'model': MODEL, 'arch': ARCH, 'held_slice': 'FW[448:600,:128]', 'n_positions': NPOS, 'BATCH': BATCH,
        'NL': NL, 'NH': NH, 'HD': HD, 'D': D, 'V': V, 'N_SVD_top': N_SVD, 'KMAX_per_block': KMAX, 'K_sweep': K_SWEEP,
        'n_head_paths': NL*NH, 'n_top_mlp_paths': NL*N_SVD,
        'forward': f'ARCH={ARCH} single-QK-branch; per-position mean-ablation + project-out-to-mean copied '
                   'VERBATIM from qk_coverage_ledger.py / qk_mlp_superposition.py',
        'currency': 'GLOBAL mean-ablation delta cross-entropy per valid held position (nats); paired SE over positions',
    },
    # ---- 1. COVERAGE LEDGER + WHOLE-MODEL SUPER-ADDITIVITY ----
    'coverage_ledger': {
        'full_headroom': {'dCE': round(H, 5), 'SE': round(M['FULL_HEADROOM'][1], 5),
                          'desc': 'all attention heads + all full MLP outputs mean-ablated minus full model'},
        'attn_all': {'dCE': round(M['ATTN_ALL'][0], 5), 'SE': round(M['ATTN_ALL'][1], 5)},
        'mlp_full': {'dCE': round(FULL_MLP, 5), 'SE': round(M['MLP_FULL'][1], 5)},
        'mlp_top': {'dCE': round(M[MLP_TOP_KEY][0], 5), 'SE': round(M[MLP_TOP_KEY][1], 5),
                    'desc': f'all {NL*N_SVD} top-{N_SVD}/block SVD dirs only (attn intact)'},
        'mlp_below_top': {'dCE': round(MLP_below_top, 5), 'SE': round(MLP_below_top_se, 5),
                          'frac_of_mlp_full': round(MLP_below_top/FULL_MLP, 4) if FULL_MLP else None,
                          'desc': 'mlp_full - mlp_top (paired), the sub-top feed-forward residual'},
        'joint_all': {'dCE': round(J, 5), 'SE': round(M['JOINT_ALL'][1], 5),
                      'desc': f'all {NL*NH} head-paths + all {NL*N_SVD} top-SVD mlp dirs ablated simultaneously'},
        'non_axis_aligned_residual': {'dCE': round(HJ_gap, 5), 'SE': round(HJ_gap_se, 5), 'frac_of_headroom': frac(HJ_gap),
                                      'desc': 'full_headroom - joint_all; = MLP effect below top-SVD in attn-ablated context'},
    },
    'super_additivity': {
        'joint_all_dCE': round(J, 5), 'joint_all_SE': round(M['JOINT_ALL'][1], 5),
        'sum_of_single_paths_dCE': round(single_total, 5), 'sum_of_single_paths_SE': round(single_total_se, 5),
        'joint_over_sum_ratio': round(super_add_ratio, 4) if super_add_ratio else None,
        'multipath_residual': round(J - single_total, 5),
        'interpretation': 'ratio > 1 => super-additive (most computation lives in combinations); '
                          'sum = sum over every single head-path and every top-SVD mlp-dir of max(0, its solo global dCE)',
        'top_single_head_paths': [{'path': f'h.L{p[0]}.{p[1]}', 'solo_global_dCE': round(head_stats[p][0], 5),
                                   'SE': round(head_stats[p][1], 5)} for p, _ in head_rank],
        'top_single_mlp_dirs': [{'path': f'mlp.L{p[0]}.d{p[1]}', 'solo_global_dCE': round(dir_stats[p][0], 5),
                                 'SE': round(dir_stats[p][1], 5)} for p, _ in dir_rank],
    },
    # ---- 2. FEED-FORWARD RANK / SUPERPOSITION (§73) ----
    'rank_superposition': {
        'mlp_full_dCE': round(FULL_MLP, 5), 'mlp_top_dCE': round(M[MLP_TOP_KEY][0], 5),
        'below_top_dCE': round(MLP_below_top, 5),
        'below_top_frac_of_full': round(MLP_below_top/FULL_MLP, 4) if FULL_MLP else None,
        'rank_sweep_svd': svd_curve,
        'rank_sweep_random': rand_curve,
        'effective_rank_per_block_for_pct_of_full': eff,
        'svd_vs_random_at_K': {str(K): {'svd_frac': svd_curve[j]['frac_of_full'], 'rand_frac': rand_curve[j]['frac_of_full'],
                                        'svd_over_rand': round(svd_curve[j]['captured_dCE']/rand_curve[j]['captured_dCE'], 3)
                                        if rand_curve[j]['captured_dCE'] and rand_curve[j]['captured_dCE'] > 1e-9 else None}
                               for j, K in enumerate(K_SWEEP)},
    },
    # ---- 3. HUB IDENTIFICATION (fed to qk_general_completeness_2.py) ----
    'hub': {
        'hub_layer_overall': hub_overall, 'hub_layer_early_half': hub_early,
        'per_layer_tail': layer_tab,
        'note': 'hub = layer carrying the most below-top4 MLP tail; run qk_general_completeness_2.py on it',
    },
}
json.dump(out, open(f'{QK}/qk_general_completeness_{MODEL}.json', 'w'), indent=2)

# ---------------- console summary ----------------
print(f"\n===== {MODEL} ({ARCH}) COMPLETENESS (global mean-ablation delta cross-entropy, nats) =====", flush=True)
print(f"FULL HEADROOM {H:.4f}   attn_all {M['ATTN_ALL'][0]:.4f}   mlp_full {FULL_MLP:.4f}   mlp_top {M[MLP_TOP_KEY][0]:.4f}   below_top {MLP_below_top:.4f}", flush=True)
print(f"JOINT_ALL {J:.4f} +- {M['JOINT_ALL'][1]:.4f}   SUM-of-single {single_total:.4f} +- {single_total_se:.4f}   "
      f"SUPER-ADDITIVITY joint/sum = {super_add_ratio:.3f}x", flush=True)
print(f"non-axis residual frac of headroom {frac(HJ_gap)}", flush=True)
print("K/block | total | SVD frac | RAND frac", flush=True)
for j, K in enumerate(K_SWEEP):
    s, r = svd_curve[j], rand_curve[j]
    print(f"  {K:3d}   | {K*NL:4d} | {s['frac_of_full']}   | {r['frac_of_full']}", flush=True)
print(f"effective rank per block for 50/80/90% of full MLP: {eff}", flush=True)
print(f"below-top MLP frac of full = {out['rank_superposition']['below_top_frac_of_full']}", flush=True)
lays = sorted(layer_tab, key=lambda r: -r['tail_below_top4_dCE'])[:5]
print("top tail-carrying layers:", [(r['layer'], round(r['tail_below_top4_dCE'],3), r['tail_share_across_layers']) for r in lays], flush=True)
print(f"HUB overall=L{hub_overall}  early-half=L{hub_early}", flush=True)
print(f"\nSaved qk_general_completeness_{MODEL}.json", flush=True)
print("QK GENERAL COMPLETENESS DONE", flush=True)
