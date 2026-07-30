"""CIRCUIT-COVERAGE LEDGER for bilin18.

Answers: how much of the model's causal computation have we NOT yet
found/characterized, and how does the unfound part split across easy (clean) vs
hard (low-cleanliness) circuits?  Builds a NESTED coverage ledger in delta
cross-entropy nats on the held-back slice FW[448:600,:128], paired standard
errors, positions as the paired unit.

COMMON CURRENCY: GLOBAL mean delta cross-entropy per valid held position (over
ALL 152*127 = 19304 valid next-token positions).  This is the ONLY scale on
which the full-model headroom, the joint-234 ablation and the sum of single
paths are mutually comparable (the census's per-path TRIGGER-dCE lives at each
path's OWN top-200 positions and cannot be summed onto a global headroom).  The
single-path per-path GLOBAL-dCE values are reused verbatim from
qk_census_difficulty.json.

FORWARD + mean-ablation copied VERBATIM from qk_census_difficulty.py /
qk_unsup_verify.py (tier2_model.reference_forward): bilin18 two-branch pattern
(q1k1/HD)(q2k2/HD) masked UNNORMALISED, per-head QK rms_norm THEN RoPE, v-lerp
via a.lamb (block-0 v cache), 30*tanh logits.  Mean-ablation is per-position
mean over the held slice (NOT zero), verbatim.  MLP SVD directions recomputed
from the TRAIN gram exactly as the census does.
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

# ---------------- GPU GUARD (verbatim from census) ----------------
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
N_SVD = 4
tok = AutoTokenizer.from_pretrained('gpt2')
print(f"bilin18 NL={NL} NH={NH} HD={HD} D={D} V={V}; {NL*NH} head-paths + {NL*N_SVD} mlp-paths", flush=True)

FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
SEQL = 128
TRAIN = FINEWEB[0:256, :SEQL].to(DEV)        # discovery slice -- ONLY to recompute MLP dirs (path defn)
HELD = FINEWEB[448:600, :SEQL].to(DEV)       # held-back verification slice (untouched by discovery)
NHELD = HELD.shape[0]
BATCH = 6

# =====================================================================================
# MLP directions: recompute top-4 SVD dirs per block from the TRAIN gram (verbatim census).
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
mlp_dirs = torch.zeros(NL, N_SVD, D, device=DEV)
for li in range(NL):
    evals, evecs = torch.linalg.eigh(gram[li])
    mlp_dirs[li] = evecs[:, -N_SVD:].T.flip(0)                # top-N_SVD, descending
del gram
print("MLP directions ready.", flush=True)

# =====================================================================================
# Core forward (VERBATIM census) + FLEXIBLE multi-path mean-ablation.
#   spec = {'heads': None|'all'|set((li,h)),
#           'mlp_dirs': None|'all'|set((li,kk)),   # project out top-SVD dirs (replace by mean)
#           'mlp_full': None|'all'|set(li)}        # replace whole mlp output by per-pos mean
#   mlp_full takes precedence over mlp_dirs for a covered layer.
# Means: YHMEAN (T,NH,HD), MOMEAN (T,D), PROJMEAN (T,N_SVD) per layer, per-position.
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
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0)                 # (B,NH,T,T) UNNORMALISED
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)         # (B,T,NH,HD)
        if collect:
            YH_SUM[li] += yh4.sum(0)                          # (T,NH,HD)
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
            MO_SUM[li] += mo.sum(0)                           # (T,D)
            pr = torch.einsum('btd,nd->btn', mo, mlp_dirs[li])   # (B,T,N_SVD)
            PROJ_SUM[li] += pr.sum(0)                         # (T,N_SVD)
        # ---- mlp ablation ----
        if mfull == 'all' or (mfull and li in mfull):
            mo = MOMEAN[li].unsqueeze(0).expand(B, -1, -1)
        elif mdirs == 'all' or mdirs:
            ks = range(N_SVD) if mdirs == 'all' else [kk for (l, kk) in mdirs if l == li]
            for kk in ks:
                pr = torch.einsum('btd,d->bt', mo, mlp_dirs[li, kk])
                mo = mo - (pr - PROJMEAN[li][:, kk].unsqueeze(0)).unsqueeze(-1) * mlp_dirs[li, kk]
        x = x + mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    return logits

# =====================================================================================
# PASS A: per-position means (yh per head, mlp-output full, mlp top-SVD projection).
# =====================================================================================
YH_SUM = {li: torch.zeros(SEQL, NH, HD, device=DEV) for li in range(NL)}
MO_SUM = {li: torch.zeros(SEQL, D, device=DEV) for li in range(NL)}
PROJ_SUM = {li: torch.zeros(SEQL, N_SVD, device=DEV) for li in range(NL)}
print("PASS A: per-position means over held-out ...", flush=True)
for i in range(0, NHELD, BATCH):
    forward(HELD[i:i+BATCH], collect=True)
YHMEAN = {li: YH_SUM[li] / NHELD for li in range(NL)}
MOMEAN = {li: MO_SUM[li] / NHELD for li in range(NL)}
PROJMEAN = {li: PROJ_SUM[li] / NHELD for li in range(NL)}
del YH_SUM, MO_SUM, PROJ_SUM
print("PASS A done.", flush=True)

# =====================================================================================
# NAMED SET -- the paths our toolbox has CHARACTERIZED across RESULTS_l0_mdl.md §56-§70.
# =====================================================================================
NAMED_CORE = [
    # copy family (§60/§61)
    'h.L8.3', 'h.L8.4', 'h.L13.0', 'h.L14.7', 'h.L7.3', 'h.L5.5',
    # digit heads (§65) -- h.L8.3 already listed
    'h.L8.7',
    # punctuation / delimiter router (§56/§58/§60)
    'h.L13.8',
    # positional fixed-offset (§62)
    'h.L0.3', 'h.L1.1',
    # word suppressor (§68)
    'h.L11.2',
    # capital selector (§68/§69) + suppression context (§59)
    'mlp.L17.d1',
    # class integrators (§68/§69) + suppression (§59: L16.d0)
    'mlp.L17.d2', 'mlp.L17.d3', 'mlp.L16.d2', 'mlp.L16.d0',
    # remap directions (§64) -- genuine verified
    'mlp.L15.d2', 'mlp.L16.d1',
]
# extended set adds the remaining §56/§57/§62/§64 causally-verified paths (robustness)
NAMED_EXT_EXTRA = [
    'h.L9.6', 'h.L6.0', 'h.L3.3', 'h.L8.2',      # §56 verified single-path algorithms
    'h.L6.7', 'h.L4.0',                          # §57 verified composition heads (boundary/year)
    'h.L5.7',                                     # §62 position-0 sink (load-bearing)
    'mlp.L1.d3',                                  # §64 determiner->word remap (weak but characterized)
]
NAMED_CORE = sorted(set(NAMED_CORE))
NAMED_EXT = sorted(set(NAMED_CORE) | set(NAMED_EXT_EXTRA))

def comp_to_spec(comps):
    heads, dirs = set(), set()
    for c in comps:
        p = c.split('.')
        li = int(p[1][1:])
        if c.startswith('h.'):
            heads.add((li, int(p[2])))
        else:
            dirs.add((li, int(p[2][1:])))
    return heads, dirs

named_core_heads, named_core_dirs = comp_to_spec(NAMED_CORE)
named_ext_heads, named_ext_dirs = comp_to_spec(NAMED_EXT)
print(f"NAMED_CORE {len(NAMED_CORE)} paths ({len(named_core_heads)} heads, {len(named_core_dirs)} dirs)", flush=True)
print(f"NAMED_EXT  {len(NAMED_EXT)} paths ({len(named_ext_heads)} heads, {len(named_ext_dirs)} dirs)", flush=True)

# all 162 heads / all 72 dirs
ALL_HEADS = {(li, h) for li in range(NL) for h in range(NH)}
ALL_DIRS = {(li, kk) for li in range(NL) for kk in range(N_SVD)}

# =====================================================================================
# PASS B: joint ablation configs -> per-position GLOBAL delta cross-entropy (paired).
# Positions: every query t in 0..SEQL-2 (target = token t+1) over all 152 held seqs.
# =====================================================================================
CONFIGS = {
    'FULL_HEADROOM':  {'heads': 'all', 'mlp_full': 'all'},                    # all attn + all full mlp
    'JOINT_234':      {'heads': 'all', 'mlp_dirs': 'all'},                    # all heads + all 72 top-SVD dirs
    'ATTN_ALL':       {'heads': 'all'},                                       # all attention only
    'MLP_FULL':       {'mlp_full': 'all'},                                    # all full mlp only
    'MLP_TOP72':      {'mlp_dirs': 'all'},                                    # all 72 top-SVD dirs only
    'NAMED_CORE':     {'heads': named_core_heads, 'mlp_dirs': named_core_dirs},
    'NAMED_EXT':      {'heads': named_ext_heads, 'mlp_dirs': named_ext_dirs},
}
held_np = HELD.cpu().numpy()
tgt_all = torch.from_numpy(held_np).to(DEV)
NPOS = NHELD * (SEQL - 1)
per_pos = {name: np.zeros(NPOS, np.float64) for name in CONFIGS}   # paired dCE per valid position

print(f"PASS B: {len(CONFIGS)} joint-ablation configs over held-out ({NPOS} positions) ...", flush=True)
t0 = time.time()
ptr = 0
for bi, i in enumerate(range(0, NHELD, BATCH)):
    sb = slice(i, min(i+BATCH, NHELD)); idx = HELD[sb]; b = idx.shape[0]
    tgt = tgt_all[sb]
    base = forward(idx).float()
    logp = F.log_softmax(base[:, :SEQL-1], dim=-1)
    base_ce = -logp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1)      # (b,T-1)
    del base, logp
    n_here = b * (SEQL - 1)
    for name, spec in CONFIGS.items():
        abl = forward(idx, spec=spec).float()
        alogp = F.log_softmax(abl[:, :SEQL-1], dim=-1)
        abl_ce = -alogp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1)
        del abl, alogp
        dce = (abl_ce - base_ce).reshape(-1).cpu().numpy()
        per_pos[name][ptr:ptr+n_here] = dce
    ptr += n_here
    if bi % 4 == 0:
        print(f"  batch {bi+1}/{(NHELD+BATCH-1)//BATCH}  elapsed {time.time()-t0:.0f}s", flush=True)
assert ptr == NPOS, (ptr, NPOS)
print(f"PASS B done in {time.time()-t0:.0f}s", flush=True)

def mean_se(arr):
    n = len(arr); mean = float(arr.mean())
    se = float(arr.std(ddof=1) / math.sqrt(n))
    return mean, se, n
def diff_se(a, b):
    d = a - b; return float(d.mean()), float(d.std(ddof=1)/math.sqrt(len(d)))

M = {name: mean_se(per_pos[name]) for name in CONFIGS}   # name -> (mean,se,n)
H = M['FULL_HEADROOM'][0]; J = M['JOINT_234'][0]
NCj = M['NAMED_CORE'][0]; NEj = M['NAMED_EXT'][0]

# =====================================================================================
# SINGLE-PATH accounting -- reuse census GLOBAL-dCE (same positions/scale) + TRIGGER-dCE.
# =====================================================================================
census = json.load(open(f'{QK}/qk_census_difficulty.json'))
recs = {r['comp']: r for r in census['records']}
clean_hi = census['summary']['thresholds']['clean_HIGH_pctile75_of_maggated']   # §67 axis

def pos(x): return x if x > 0 else 0.0
single_total_g = sum(pos(r['global_dCE']) for r in recs.values())
single_total_g_se = math.sqrt(sum((r['global_dCE_SE'])**2 for r in recs.values() if r['global_dCE'] > 0))
single_total_t = sum(pos(r['trigger_dCE']) for r in recs.values())

def sum_named(comps, key):
    return sum(pos(recs[c][key]) for c in comps if c in recs)
def sum_named_se(comps, key_se, key):
    return math.sqrt(sum(recs[c][key_se]**2 for c in comps if c in recs and recs[c][key] > 0))

named_core_g = sum_named(NAMED_CORE, 'global_dCE'); named_core_g_se = sum_named_se(NAMED_CORE, 'global_dCE_SE', 'global_dCE')
named_ext_g = sum_named(NAMED_EXT, 'global_dCE')
named_core_t = sum_named(NAMED_CORE, 'trigger_dCE'); named_ext_t = sum_named(NAMED_EXT, 'trigger_dCE')

# unnamed single-path importance and its EASY vs HARD split (§67 cleanliness axis)
def unnamed_split(named, key):
    lo_hard = hi_easy = 0.0
    for c, r in recs.items():
        if c in named: continue
        v = pos(r[key])
        if v == 0: continue
        if r['cleanliness'] < clean_hi: lo_hard += v
        else: hi_easy += v
    return lo_hard, hi_easy
uc_hard_g, uc_easy_g = unnamed_split(set(NAMED_CORE), 'global_dCE')
uc_hard_t, uc_easy_t = unnamed_split(set(NAMED_CORE), 'trigger_dCE')

# =====================================================================================
# ASSEMBLE THE NESTED LEDGER (fractions of full headroom H, with interaction caveat).
#   H = NAMED_joint + (J - NAMED_joint) + (H - J)
#     = [found]      + [unnamed axis-aligned] + [non-axis-aligned residual]
# =====================================================================================
def frac(x): return round(x / H, 4) if H else None
HJ_gap, HJ_gap_se = diff_se(per_pos['FULL_HEADROOM'], per_pos['JOINT_234'])       # non-axis residual
JNc_gap, JNc_gap_se = diff_se(per_pos['JOINT_234'], per_pos['NAMED_CORE'])        # unnamed axis (core)
JNe_gap, JNe_gap_se = diff_se(per_pos['JOINT_234'], per_pos['NAMED_EXT'])
MLP_below72, MLP_below72_se = diff_se(per_pos['MLP_FULL'], per_pos['MLP_TOP72'])  # mlp below top-72

ledger = {
    'currency': 'GLOBAL mean delta cross-entropy per valid held position (nats); paired standard error over positions',
    'n_positions': M['FULL_HEADROOM'][2],
    # ---- 1. FULL-MODEL HEADROOM (denominator) ----
    'full_headroom': {'dCE': round(H, 5), 'SE': round(M['FULL_HEADROOM'][1], 5),
                      'desc': 'all attention heads + all full MLP outputs mean-ablated (per-position mean) minus full model'},
    'attn_all': {'dCE': round(M['ATTN_ALL'][0], 5), 'SE': round(M['ATTN_ALL'][1], 5),
                 'desc': 'all attention only (mlp intact)'},
    'mlp_full': {'dCE': round(M['MLP_FULL'][0], 5), 'SE': round(M['MLP_FULL'][1], 5),
                 'desc': 'all full MLP outputs only (attn intact)'},
    'mlp_top72': {'dCE': round(M['MLP_TOP72'][0], 5), 'SE': round(M['MLP_TOP72'][1], 5),
                  'desc': 'all 72 top-SVD MLP directions only (attn intact)'},
    'mlp_below_top72': {'dCE': round(MLP_below72, 5), 'SE': round(MLP_below72_se, 5),
                        'desc': 'MLP effect below the top-72 SVD dirs = mlp_full - mlp_top72 (paired)',
                        'frac_of_mlp_full': round(MLP_below72/M['MLP_FULL'][0], 4) if M['MLP_FULL'][0] else None},
    # ---- 2. JOINT-ALL-234 vs SUM-OF-SINGLE-PATHS ----
    'joint_234': {'dCE': round(J, 5), 'SE': round(M['JOINT_234'][1], 5),
                  'desc': 'all 162 head-paths + all 72 top-SVD mlp dirs ablated simultaneously'},
    'single_path_sum_global': {'dCE': round(single_total_g, 5), 'SE': round(single_total_g_se, 5),
                               'desc': 'sum over 234 paths of max(0, per-path GLOBAL-dCE) [census]'},
    'joint_over_sum_ratio_global': round(J / single_total_g, 4) if single_total_g else None,
    'multipath_residual_global': {'dCE': round(J - single_total_g, 5),
                                  'desc': 'joint_234 - sum_of_single_paths; >0 super-additive (combination-only), <0 net redundancy'},
    'single_path_sum_trigger': round(single_total_t, 5),
    # ---- 3. NAMED joint ablations ----
    'named_core_joint': {'dCE': round(NCj, 5), 'SE': round(M['NAMED_CORE'][1], 5),
                         'n_paths': len(NAMED_CORE), 'paths': NAMED_CORE},
    'named_ext_joint': {'dCE': round(NEj, 5), 'SE': round(M['NAMED_EXT'][1], 5),
                        'n_paths': len(NAMED_EXT), 'paths': NAMED_EXT},
    'named_core_sum_global': round(named_core_g, 5), 'named_core_sum_global_SE': round(named_core_g_se, 5),
    'named_ext_sum_global': round(named_ext_g, 5),
    'named_core_sum_trigger': round(named_core_t, 5), 'named_ext_sum_trigger': round(named_ext_t, 5),
    # ---- THE THREE GAPS as fractions of full headroom H ----
    'coverage_fractions_of_headroom': {
        'named_core_found': {'dCE': round(NCj, 5), 'frac': frac(NCj)},
        'named_ext_found': {'dCE': round(NEj, 5), 'frac': frac(NEj)},
        'unnamed_axis_aligned_core': {'dCE': round(JNc_gap, 5), 'SE': round(JNc_gap_se, 5), 'frac': frac(JNc_gap),
                                      'desc': 'joint_234 - named_core_joint (single-path-expressible but uncharacterized)'},
        'unnamed_axis_aligned_ext': {'dCE': round(JNe_gap, 5), 'SE': round(JNe_gap_se, 5), 'frac': frac(JNe_gap)},
        'non_axis_aligned_residual': {'dCE': round(HJ_gap, 5), 'SE': round(HJ_gap_se, 5), 'frac': frac(HJ_gap),
                                      'desc': 'full_headroom - joint_234; superposition / sub-top-72 mlp; NOT captured by any single '
                                              'head-pathway or top-SVD dir. Since the 162 head-paths ablate ALL attention identically '
                                              'in H and J, this residual is EXACTLY the MLP effect below the top-72 SVD directions, '
                                              'measured in the attention-ablated context. The SAME subspace measured with attention '
                                              'INTACT (mlp_below_top72) is larger (see field) -- the two differ by mean-ablation '
                                              'interaction, not by which directions are removed.'},
    },
    'not_found_fraction_of_headroom': {
        'core': round(1 - NCj/H, 4) if H else None,
        'ext': round(1 - NEj/H, 4) if H else None,
        'desc': '1 - named_joint/full_headroom = fraction of the layers causal work NOT yet characterized '
                '(= unnamed_axis_aligned + non_axis_aligned_residual)',
    },
    # ---- item 3 gap A: named vs single-path total (axis-aligned single-path accounting) ----
    'named_fraction_of_single_path_total': {
        'core_global': round(named_core_g/single_total_g, 4) if single_total_g else None,
        'ext_global': round(named_ext_g/single_total_g, 4) if single_total_g else None,
        'core_trigger': round(named_core_t/single_total_t, 4) if single_total_t else None,
        'ext_trigger': round(named_ext_t/single_total_t, 4) if single_total_t else None,
        'desc': 'fraction of the summed positive single-path effect that is NAMED (axis-aligned single-path view)',
    },
    # ---- 4. EASY vs HARD split of the UNNAMED single-path importance (§67 cleanliness axis) ----
    'unnamed_easy_vs_hard': {
        'clean_hi_threshold': clean_hi,
        'global': {'hard_low_clean': round(uc_hard_g, 5), 'easy_high_clean': round(uc_easy_g, 5),
                   'hard_fraction': round(uc_hard_g/(uc_hard_g+uc_easy_g), 4) if (uc_hard_g+uc_easy_g) else None},
        'trigger': {'hard_low_clean': round(uc_hard_t, 5), 'easy_high_clean': round(uc_easy_t, 5),
                    'hard_fraction': round(uc_hard_t/(uc_hard_t+uc_easy_t), 4) if (uc_hard_t+uc_easy_t) else None},
        'desc': 'of the UNNAMED (not-in-NAMED_CORE) positive single-path importance, fraction sitting in the '
                'LOW-cleanliness (hard) region (cleanliness < p75-of-maggated) vs HIGH-cleanliness (easy)',
    },
}

out = {
    'meta': {
        'model': 'bilin18', 'held_slice': 'FW[448:600,:128]', 'n_head_paths': NL*NH,
        'n_mlp_paths': NL*N_SVD, 'n_paths': NL*NH + NL*N_SVD, 'BATCH': BATCH,
        'forward': 'VERBATIM from qk_census_difficulty.py / qk_unsup_verify.py (tier2_model.reference_forward)',
        'currency_note': 'GLOBAL mean delta cross-entropy per valid held position -- the only common scale for headroom, '
                         'joint-234 and sum-of-singles. Census TRIGGER-dCE (path-specific top-200 positions) is reported '
                         'as a secondary single-path view but is NOT summable onto the global headroom.',
        'interaction_caveat': 'H = named_joint + (J-named_joint) + (H-J) is exact telescoping algebra; the interpretation '
                              'of each term as an independent contribution carries the usual mean-ablation interaction caveat.',
    },
    'means_by_config': {k: {'dCE': round(v[0], 6), 'SE': round(v[1], 6), 'n': v[2]} for k, v in M.items()},
    'ledger': ledger,
}
json.dump(out, open(f'{QK}/qk_coverage_ledger.json', 'w'), indent=2)

# =====================================================================================
print("\n===== COVERAGE LEDGER (global mean delta cross-entropy, nats) =====", flush=True)
print(f"FULL HEADROOM (all attn+mlp mean-ablated):  {H:.4f} +- {M['FULL_HEADROOM'][1]:.4f}", flush=True)
print(f"  attn-all {M['ATTN_ALL'][0]:.4f}   mlp-full {M['MLP_FULL'][0]:.4f}   mlp-top72 {M['MLP_TOP72'][0]:.4f}   mlp-below-top72 {MLP_below72:.4f}", flush=True)
print(f"JOINT-234 (all heads + all 72 dirs):        {J:.4f} +- {M['JOINT_234'][1]:.4f}", flush=True)
print(f"SUM of positive single-path GLOBAL-dCE:     {single_total_g:.4f}   (joint/sum = {J/single_total_g:.3f})", flush=True)
print(f"NAMED_CORE joint ({len(NAMED_CORE)}):  {NCj:.4f} +- {M['NAMED_CORE'][1]:.4f}    NAMED_EXT joint ({len(NAMED_EXT)}): {NEj:.4f}", flush=True)
print(f"\nCOVERAGE as fraction of FULL HEADROOM:", flush=True)
print(f"  NAMED_CORE found          : {frac(NCj)}", flush=True)
print(f"  unnamed axis-aligned (core): {frac(JNc_gap)}", flush=True)
print(f"  non-axis-aligned residual : {frac(HJ_gap)}  (= mlp below top-72)", flush=True)
print(f"  NOT FOUND (core)          : {1-NCj/H:.4f}", flush=True)
print(f"\nNamed fraction of single-path total (global): core {named_core_g/single_total_g:.3f}  ext {named_ext_g/single_total_g:.3f}", flush=True)
print(f"Unnamed easy-vs-hard (global): hard_frac {uc_hard_g/(uc_hard_g+uc_easy_g):.3f}   (trigger): hard_frac {uc_hard_t/(uc_hard_t+uc_easy_t):.3f}", flush=True)
print("\nSaved qk_coverage_ledger.json", flush=True)
print("QK COVERAGE LEDGER DONE", flush=True)
