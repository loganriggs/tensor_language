"""DIFFICULTY-STRATIFIED CIRCUIT-TYPE CENSUS for bilin18.

Tests a sampling-bias concern: the unsupervised discovery loop
(qk_unsup_discover.py) ranks the 234 decomposition paths by CLEANLINESS
(trigger-purity x effect-purity) and only causally verifies the TOP ones. That
structurally favors clean SURFACE circuits and may systematically MISS circuits
that are causally important but MESSY (impure/multi-class trigger, distributed
output, deep composition). This script measures, for ALL 234 paths, TWO
INDEPENDENT axes:

  (a) CLEANLINESS  -- exactly the easy-loop ranking, read verbatim from
      qk_unsup_paths.json (produced by qk_unsup_discover.py).
  (b) CAUSAL IMPORTANCE  -- marginal mean-ablation delta cross-entropy on the
      held-back slice FW[448:600], measured INDEPENDENTLY of cleanliness:
        * TRIGGER-dCE: dCE at the path's own top-K positions, where positions are
          selected by ACTIVATION MAGNITUDE (|head-output| / |mlp-projection|),
          which is independent of trigger/effect PURITY (=cleanliness). This is
          "does the path matter WHERE it operates". Primary axis.
        * GLOBAL-dCE: dCE averaged over ALL valid held-out positions. Secondary.
      Both with paired standard errors (positions are the paired unit, matching
      qk_unsup_verify.py).

FORWARD copied VERBATIM from qk_unsup_verify.py / tier2_model.reference_forward:
bilin18 two-branch pattern (q1k1/HD)(q2k2/HD) masked UNNORMALISED, per-head QK
rms_norm THEN RoPE, v-lerp via a.lamb (block-0 v cache), 30*tanh logits.
Mean-ablation is per-position mean over the held slice (NOT zero), verbatim.
"""
import json, sys, math, os, time, subprocess
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'

# ---------------- GPU GUARD: self-serialize against concurrent jobs ----------------
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
KCAUSAL = 200                                # top-K activation positions per path (matches verify)

# special/degenerate tokens excluded from TRIGGER position selection (matches discover/verify)
_special = {tok.eos_token_id}
for _t in range(V):
    if '�' in tok.decode([_t]): _special.add(_t)
SPECIAL = np.array(sorted(_special))
print(f"{len(SPECIAL)} special token ids masked from trigger selection", flush=True)

# =====================================================================================
# MLP directions: recompute top-4 SVD dirs per block from the TRAIN gram (path definition,
# verbatim with discover: uncentred MLP-output gram over FW[0:256,:128]).
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
# Core forward (VERBATIM), optional single-path mean-ablation, optional collection.
# =====================================================================================
@torch.no_grad()
def forward(idx, collect=False, ablate=None, yhmeans=None, projmeans=None):
    """ablate: None | ('head',li,h) | ('mlp',li,kk); mean-ablated with per-position means.
    collect=True: accumulate per-position yh sums, mlp-proj sums, and return per-path
    activation magnitudes (head-output-norm for all heads; |proj| for all mlp dirs)."""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    out = {} if collect else None
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
            Wr = a.c_proj.weight.view(D, NH, HD)
            YH_SUM[li] += yh4.sum(0)                          # (T,NH,HD)
            comp = torch.einsum('bthc,ohc->btho', yh4, Wr)    # (B,T,NH,D)
            out[('hnorm', li)] = comp.norm(dim=-1).cpu().numpy()   # (B,T,NH)
        if ablate is not None and ablate[0] == 'head' and ablate[1] == li:
            yh4 = yh4.clone(); yh4[:, :, ablate[2]] = yhmeans[li][:, ablate[2]].unsqueeze(0)
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if collect:
            pr = torch.einsum('btd,nd->btn', mo, mlp_dirs[li])   # (B,T,N_SVD)
            PROJ_SUM[li] += pr.sum(0)                            # (T,N_SVD)
            out[('mproj', li)] = pr.cpu().numpy()                # (B,T,N_SVD)
        if ablate is not None and ablate[0] == 'mlp' and ablate[1] == li:
            kk = ablate[2]
            pr = torch.einsum('btd,d->bt', mo, mlp_dirs[li, kk])
            mo = mo - (pr - projmeans[li][:, kk].unsqueeze(0)).unsqueeze(-1) * mlp_dirs[li, kk]
        x = x + mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    return (logits, out) if collect else logits

# =====================================================================================
# PASS A: per-position means + per-path activation magnitudes over held-out.
# =====================================================================================
YH_SUM = {li: torch.zeros(SEQL, NH, HD, device=DEV) for li in range(NL)}
PROJ_SUM = {li: torch.zeros(SEQL, N_SVD, device=DEV) for li in range(NL)}
head_act = np.zeros((NL*NH, NHELD, SEQL), np.float32)      # ||head output vector||
mlp_act = np.zeros((NL*N_SVD, NHELD, SEQL), np.float32)    # |mlp projection|

print("PASS A: collect activation magnitudes + per-position means ...", flush=True)
for i in range(0, NHELD, BATCH):
    _, out = forward(HELD[i:i+BATCH], collect=True)
    b = HELD[i:i+BATCH].shape[0]
    for li in range(NL):
        hn = out[('hnorm', li)]                               # (b,T,NH)
        for h in range(NH):
            head_act[li*NH + h, i:i+b] = hn[:, :, h]
        pj = np.abs(out[('mproj', li)])                       # (b,T,N_SVD)
        for kk in range(N_SVD):
            mlp_act[li*N_SVD + kk, i:i+b] = pj[:, :, kk]
YHMEAN = {li: YH_SUM[li] / NHELD for li in range(NL)}
PROJMEAN = {li: PROJ_SUM[li] / NHELD for li in range(NL)}
del YH_SUM, PROJ_SUM
print("PASS A done.", flush=True)

# ---- position validity + top-K trigger selection masks (activation magnitude only) ----
held_np = HELD.cpu().numpy()
pos_t = np.tile(np.arange(SEQL), NHELD).reshape(NHELD, SEQL)
is_special = np.isin(held_np, SPECIAL)
valid_next = pos_t < (SEQL - 1)                              # position t predicts t+1
bad_trigger = (pos_t == 0) | is_special | ~valid_next       # excluded from trigger selection

# build path list matching qk_unsup_paths.json ordering conventions
PATHS = []                                                  # (comp, kind, li, idx)
for li in range(NL):
    for h in range(NH):
        PATHS.append((f"h.L{li}.{h}", 'head', li, h))
for li in range(NL):
    for kk in range(N_SVD):
        PATHS.append((f"mlp.L{li}.d{kk}", 'mlp', li, kk))

# top-KCAUSAL activation-selected trigger mask per path (shape NHELD x SEQL bool)
trig_mask = {}
for pi, (comp, kind, li, ix) in enumerate(PATHS):
    act = head_act[li*NH + ix] if kind == 'head' else mlp_act[li*N_SVD + ix]
    a = act.copy().reshape(-1); a[bad_trigger.reshape(-1)] = -1e30
    tk = np.argpartition(a, -KCAUSAL)[-KCAUSAL:]
    mk = np.zeros(NHELD*SEQL, bool); mk[tk] = True
    trig_mask[comp] = mk.reshape(NHELD, SEQL)
del head_act, mlp_act

# =====================================================================================
# PASS B: for every path, mean-ablate and accumulate GLOBAL + TRIGGER dCE (paired).
# One ablated forward per (path, batch); base computed once per batch and reused.
# =====================================================================================
valid_flat = valid_next                                     # positions with a next-token target
# accumulators per path
g_sum = {c: 0.0 for c, *_ in PATHS}; g_sq = {c: 0.0 for c, *_ in PATHS}; g_n = {c: 0 for c, *_ in PATHS}
t_sum = {c: 0.0 for c, *_ in PATHS}; t_sq = {c: 0.0 for c, *_ in PATHS}; t_n = {c: 0 for c, *_ in PATHS}
t_pos = {c: 0 for c, *_ in PATHS}                           # dCE>0 count at trigger positions

print(f"PASS B: {len(PATHS)} single-path mean-ablations over held-out ...", flush=True)
t0 = time.time()
tgt_all = torch.from_numpy(held_np).to(DEV)                 # (NHELD,SEQL) token ids
for bi, i in enumerate(range(0, NHELD, BATCH)):
    sb = slice(i, min(i+BATCH, NHELD))
    idx = HELD[sb]; b = idx.shape[0]
    # base per-position CE for query positions t (target = token at t+1)
    base = forward(idx).float()                             # (b,T,V)
    tgt = tgt_all[sb]                                       # (b,T)
    # CE at positions 0..T-2 predicting 1..T-1
    logp = F.log_softmax(base[:, :SEQL-1], dim=-1)
    base_ce = -logp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1)  # (b,T-1)
    del base, logp
    vmask = torch.from_numpy(valid_flat[sb, :SEQL-1]).to(DEV)        # (b,T-1) all True here
    for (comp, kind, li, ix) in PATHS:
        abl = forward(idx, ablate=(kind, li, ix), yhmeans=YHMEAN, projmeans=PROJMEAN).float()
        alogp = F.log_softmax(abl[:, :SEQL-1], dim=-1)
        abl_ce = -alogp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1)
        del abl, alogp
        dce = (abl_ce - base_ce)                            # (b,T-1)
        # GLOBAL: all valid positions
        dg = dce[vmask]
        g_sum[comp] += float(dg.sum()); g_sq[comp] += float((dg*dg).sum()); g_n[comp] += int(dg.numel())
        # TRIGGER: path's top-K activation positions in this batch (query positions t<SEQL-1)
        tm = torch.from_numpy(trig_mask[comp][sb, :SEQL-1]).to(DEV)
        if tm.any():
            dt = dce[tm]
            t_sum[comp] += float(dt.sum()); t_sq[comp] += float((dt*dt).sum())
            t_n[comp] += int(dt.numel()); t_pos[comp] += int((dt > 0).sum())
    if bi % 4 == 0:
        el = time.time()-t0
        print(f"  batch {bi+1}/{(NHELD+BATCH-1)//BATCH}  elapsed {el:.0f}s", flush=True)
print(f"PASS B done in {time.time()-t0:.0f}s", flush=True)

def stats(s, sq, n):
    if n <= 1: return 0.0, 0.0
    mean = s/n; var = max(sq/n - mean*mean, 0.0)*n/(n-1)
    return mean, math.sqrt(var/n)

# =====================================================================================
# assemble records, join cleanliness axis from qk_unsup_paths.json
# =====================================================================================
easy = {r['comp']: r for r in json.load(open(f'{QK}/qk_unsup_paths.json'))['ranking']}
records = []
for (comp, kind, li, ix) in PATHS:
    gm, gse = stats(g_sum[comp], g_sq[comp], g_n[comp])
    tm_, tse = stats(t_sum[comp], t_sq[comp], t_n[comp])
    e = easy[comp]
    records.append({
        'comp': comp, 'kind': kind, 'li': li, 'idx': ix,
        # axis (a): cleanliness (easy-loop ranking, verbatim from qk_unsup_paths.json)
        'cleanliness': e['cleanliness'], 'trigger_purity': e['trigger_purity'],
        'effect_purity': e['effect_purity'], 'cur_purity': e['cur_purity'],
        'src_purity': e['src_purity'], 'mag_resid_frac': e['mag_resid_frac'],
        'passes_mag_gate': e['passes_mag_gate'],
        # axis (b): causal importance (independent, this script)
        'trigger_dCE': round(tm_, 5), 'trigger_dCE_SE': round(tse, 5),
        'trigger_dCE_z': round(tm_/tse, 3) if tse > 0 else 0.0,
        'trigger_dCE_frac_pos': round(t_pos[comp]/max(1, t_n[comp]), 3),
        'global_dCE': round(gm, 6), 'global_dCE_SE': round(gse, 6),
        'global_dCE_z': round(gm/gse, 3) if gse > 0 else 0.0,
        'n_trigger': t_n[comp], 'n_global': g_n[comp],
        # descriptive (for characterization; copied from easy loop)
        'cur_classes': e.get('cur_classes'), 'src_classes': e.get('src_classes'),
        'top5_cur': e.get('top5_cur'), 'top5_src': e.get('top5_src'),
        'top8_boost': e.get('top8_boost'),
    })

# =====================================================================================
# correlations + four-quadrant cross-tab
# =====================================================================================
def pearson(a, b):
    a = np.asarray(a); b = np.asarray(b)
    if a.std() == 0 or b.std() == 0: return 0.0
    return float(np.corrcoef(a, b)[0, 1])
def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return pearson(ra, rb)

clean = np.array([r['cleanliness'] for r in records])
tdce = np.array([r['trigger_dCE'] for r in records])
gdce = np.array([r['global_dCE'] for r in records])

corr = {
    'pearson_clean_vs_trigger_dCE': round(pearson(clean, tdce), 4),
    'spearman_clean_vs_trigger_dCE': round(spearman(clean, tdce), 4),
    'pearson_clean_vs_global_dCE': round(pearson(clean, gdce), 4),
    'spearman_clean_vs_global_dCE': round(spearman(clean, gdce), 4),
}
# restricted to magnitude-gated paths (the ones the easy-loop actually ranks/verifies)
gated = [r for r in records if r['passes_mag_gate']]
gc = np.array([r['cleanliness'] for r in gated]); gt = np.array([r['trigger_dCE'] for r in gated])
corr['pearson_clean_vs_trigger_dCE_maggated'] = round(pearson(gc, gt), 4)
corr['spearman_clean_vs_trigger_dCE_maggated'] = round(spearman(gc, gt), 4)

# thresholds. Cleanliness HIGH = the region the loop actually verifies (top quartile of
# cleanliness among magnitude-gated paths). Causal HIGH = the effect is a real, non-trivial
# loss contribution: trigger_dCE >= 0.02 nats AND statistically clear (z >= 3).
clean_hi = float(np.percentile([r['cleanliness'] for r in gated], 75))
DCE_HI = 0.02; Z_HI = 3.0
def is_clean_hi(r): return r['cleanliness'] >= clean_hi
def is_causal_hi(r): return (r['trigger_dCE'] >= DCE_HI) and (r['trigger_dCE_z'] >= Z_HI)

quad = {'HH': [], 'LH': [], 'HL': [], 'LL': []}
for r in records:
    ch, uh = is_clean_hi(r), is_causal_hi(r)
    key = ('H' if ch else 'L') + ('H' if uh else 'L')
    quad[key].append(r['comp'])
quad_counts = {k: len(v) for k, v in quad.items()}

# median-split alternative (threshold-free-ish cross-tab)
clmed = float(np.median(clean)); tmed = float(np.median(tdce))
quad_med = {'HH': 0, 'LH': 0, 'HL': 0, 'LL': 0}
for r in records:
    key = ('H' if r['cleanliness'] >= clmed else 'L') + ('H' if r['trigger_dCE'] >= tmed else 'L')
    quad_med[key] += 1

# the crux: LOW-clean / HIGH-causal missed-hard paths, ranked by causal importance
missed = sorted([r for r in records if (not is_clean_hi(r)) and is_causal_hi(r)],
                key=lambda r: -r['trigger_dCE'])
# the clean winners for comparison
clean_winners = sorted([r for r in records if is_clean_hi(r) and is_causal_hi(r)],
                       key=lambda r: -r['trigger_dCE'])
# clean-but-null: clean detectors that do not matter
clean_null = sorted([r for r in records if is_clean_hi(r) and not is_causal_hi(r)],
                    key=lambda r: -r['cleanliness'])

summary = {
    'thresholds': {'clean_HIGH_pctile75_of_maggated': round(clean_hi, 4),
                   'causal_HIGH_trigger_dCE_min': DCE_HI, 'causal_HIGH_z_min': Z_HI,
                   'clean_median': round(clmed, 4), 'trigger_dCE_median': round(tmed, 5)},
    'correlations': corr,
    'quadrant_counts_threshold': quad_counts,
    'quadrant_counts_mediansplit': quad_med,
    'quadrant_legend': {'HH': 'HIGH-clean/HIGH-causal (easy wins loop finds)',
                        'LH': 'LOW-clean/HIGH-causal (hard important MISSED -- the crux)',
                        'HL': 'HIGH-clean/LOW-causal (clean detectors that do not matter)',
                        'LL': 'LOW-clean/LOW-causal (noise)'},
    'missed_hard_LH_top': [{'comp': r['comp'], 'kind': r['kind'], 'li': r['li'],
                            'cleanliness': r['cleanliness'], 'trigger_dCE': r['trigger_dCE'],
                            'trigger_dCE_SE': r['trigger_dCE_SE'], 'z': r['trigger_dCE_z'],
                            'global_dCE': r['global_dCE'], 'trigger_purity': r['trigger_purity'],
                            'effect_purity': r['effect_purity']} for r in missed[:25]],
    'clean_winners_HH': [{'comp': r['comp'], 'cleanliness': r['cleanliness'],
                          'trigger_dCE': r['trigger_dCE'], 'z': r['trigger_dCE_z']} for r in clean_winners],
    'clean_null_HL_top': [{'comp': r['comp'], 'cleanliness': r['cleanliness'],
                           'trigger_dCE': r['trigger_dCE'], 'z': r['trigger_dCE_z']} for r in clean_null[:15]],
    'dCE_comparison': {
        'missed_hard_LH_mean_trigger_dCE': round(float(np.mean([r['trigger_dCE'] for r in missed])), 5) if missed else None,
        'missed_hard_LH_max_trigger_dCE': round(float(max([r['trigger_dCE'] for r in missed])), 5) if missed else None,
        'clean_winners_HH_mean_trigger_dCE': round(float(np.mean([r['trigger_dCE'] for r in clean_winners])), 5) if clean_winners else None,
        'n_missed_hard_LH': len(missed), 'n_clean_winners_HH': len(clean_winners),
    },
}

out = {
    'meta': {
        'model': 'bilin18', 'held_slice': 'FW[448:600,:128]', 'n_paths': len(PATHS),
        'n_head_paths': NL*NH, 'n_mlp_paths': NL*N_SVD, 'KCAUSAL': KCAUSAL, 'BATCH': BATCH,
        'axis_a': 'CLEANLINESS = trigger_purity*effect_purity, verbatim from qk_unsup_paths.json (easy-loop ranking)',
        'axis_b': 'CAUSAL IMPORTANCE = marginal mean-ablation delta cross-entropy, held-back, paired SE. '
                  'TRIGGER-dCE at top-KCAUSAL ACTIVATION-selected positions (magnitude, purity-independent); '
                  'GLOBAL-dCE over all valid held positions.',
        'note': 'positions are the paired unit for the standard error (matches qk_unsup_verify.py). '
                'trigger-position selection uses activation MAGNITUDE only, independent of cleanliness/purity.',
    },
    'summary': summary,
    'records': sorted(records, key=lambda r: -r['trigger_dCE']),
}
json.dump(out, open(f'{QK}/qk_census_difficulty.json', 'w'), indent=2)

# =====================================================================================
print("\n===== CENSUS SUMMARY =====", flush=True)
print("correlations (cleanliness vs causal importance):", flush=True)
for k, v in corr.items(): print(f"  {k}: {v}", flush=True)
print(f"\nquadrant counts (threshold: clean>= p75_maggated={clean_hi:.3f}, "
      f"causal>= dCE {DCE_HI} & z {Z_HI}):", flush=True)
for k in ['HH', 'LH', 'HL', 'LL']:
    print(f"  {k} {summary['quadrant_legend'][k]}: {quad_counts[k]}", flush=True)
print(f"\nquadrant counts (median split): {quad_med}", flush=True)
print(f"\nMISSED-HARD (LOW-clean/HIGH-causal): {len(missed)} paths", flush=True)
for r in missed[:15]:
    print(f"  {r['comp']:12s} clean={r['cleanliness']:.3f} trigDCE={r['trigger_dCE']:.4f}"
          f"+-{r['trigger_dCE_SE']:.4f} z={r['trigger_dCE_z']:.1f} globalDCE={r['global_dCE']:.5f}"
          f" trigPur={r['trigger_purity']:.2f} effPur={r['effect_purity']:.2f}", flush=True)
print(f"\nCLEAN WINNERS (HIGH-clean/HIGH-causal): {len(clean_winners)}", flush=True)
for r in clean_winners[:15]:
    print(f"  {r['comp']:12s} clean={r['cleanliness']:.3f} trigDCE={r['trigger_dCE']:.4f} z={r['trigger_dCE_z']:.1f}", flush=True)
print(f"\nCLEAN-BUT-NULL (HIGH-clean/LOW-causal): {len(clean_null)}", flush=True)
for r in clean_null[:10]:
    print(f"  {r['comp']:12s} clean={r['cleanliness']:.3f} trigDCE={r['trigger_dCE']:.4f} z={r['trigger_dCE_z']:.1f}", flush=True)
print("\nSaved qk_census_difficulty.json", flush=True)
print("QK CENSUS DIFFICULTY DONE", flush=True)
