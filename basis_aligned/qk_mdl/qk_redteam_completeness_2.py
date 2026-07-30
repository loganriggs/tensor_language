"""ADVERSARIAL RED-TEAM of §71 (coverage denominator) and §73 (rank curve).

ATTACK 2 (§71): "named circuits carry ~11% of headroom; ~89% unfound."
The named set is single paths measured by their JOINT ablation (0.58 nats); the
FULL headroom (5.31) is the joint of ALL attention + ALL MLP with 2.87x whole-model
super-additivity. Is 11% a fair fraction, or an artifact of comparing a small
(near-additive) joint against a large (highly super-additive) joint? We
INDEPENDENTLY re-run the joints (FULL_HEADROOM, JOINT_234, NAMED_CORE, NAMED_EXT)
on held FW[448:600], measure the named set's OWN super-additivity (joint vs sum),
and re-express "named fraction" on every consistent denominator:
  - of FULL HEADROOM (joint of everything)          -> the §71 headline
  - of JOINT-234 (joint of all path-expressible)    -> both-joint, excl. non-axis residual
  - of SINGLE-PATH SUM (sum of 234 solos)           -> both sum-of-solos
Single-path per-path GLOBAL-dCE reused from qk_census_difficulty.json (same
positions/scale, as the ledger does).

ATTACK 3 (§73): "50% needs 8 dirs/block, 80% needs 28/block" (K=8->0.497, K=32->0.835
of full MLP). Independent recompute of the captured-effect-vs-rank curve at K=8 and
K=32, SVD vs random, attention intact.

FORWARD + mean-ablation + MLP-SVD construction copied VERBATIM from
qk_coverage_ledger.py / qk_mlp_superposition.py. Held-back FW[448:600], paired SE.
"""
import json, sys, math, time, subprocess
import numpy as np
import torch
import torch.nn.functional as F
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
KMAX = 32
print(f"bilin18 NL={NL} NH={NH} HD={HD} D={D} V={V}", flush=True)

FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
SEQL = 128
TRAIN = FINEWEB[0:256, :SEQL].to(DEV)
HELD = FINEWEB[448:600, :SEQL].to(DEV)
NHELD = HELD.shape[0]
BATCH = 6

# ---- MLP SVD dirs (top-32/block) + random-32/block, from TRAIN gram (verbatim) ----
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
        pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        mo = blk.mlp(F.rms_norm(x, (D,)))
        gram[li] += torch.einsum('btd,bte->de', mo, mo)
        x = x + mo
print("Recomputing MLP SVD dirs from TRAIN gram ...", flush=True)
for i in range(0, TRAIN.shape[0], BATCH): fwd_gram(TRAIN[i:i+BATCH])
SVD = torch.zeros(NL, KMAX, D, device=DEV)
for li in range(NL):
    ev, evec = torch.linalg.eigh(gram[li]); SVD[li] = evec[:, -KMAX:].T.flip(0)
del gram
g = torch.Generator(device=DEV); g.manual_seed(1234)
RAND = torch.zeros(NL, KMAX, D, device=DEV)
for li in range(NL):
    RAND[li] = torch.linalg.qr(torch.randn(D, KMAX, device=DEV, generator=g))[0].T
print("dirs ready.", flush=True)

# =====================================================================================
# Unified forward (verbatim). spec keys:
#   heads: None|'all'|set((li,h))
#   mlp_full: None|'all'|set(li)
#   mlp_dirs4: None|'all'|set((li,kk))    # top-4 per block project-out (coverage ledger)
#   proj_all: None|dict{li:(Dirs(k,D),PM(T,k))}   # project-out K dirs/block (rank sweep)
#   collect: bool
# =====================================================================================
@torch.no_grad()
def forward(idx, collect=False, spec=None):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    heads = spec.get('heads') if spec else None
    mfull = spec.get('mlp_full') if spec else None
    mdirs4 = spec.get('mlp_dirs4') if spec else None
    proj_all = spec.get('proj_all') if spec else None
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
        if collect: YH_SUM[li] += yh4.sum(0)
        if heads == 'all':
            yh4 = YHMEAN[li].unsqueeze(0).expand(B, -1, -1, -1).clone()
        elif heads:
            hs = [h for (l, h) in heads if l == li]
            if hs:
                yh4 = yh4.clone()
                for h in hs: yh4[:, :, h] = YHMEAN[li][:, h].unsqueeze(0)
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if collect:
            MO_SUM[li] += mo.sum(0)
            SPROJ_SUM[li] += torch.einsum('btd,kd->btk', mo, SVD[li]).sum(0)
            RPROJ_SUM[li] += torch.einsum('btd,kd->btk', mo, RAND[li]).sum(0)
        if mfull == 'all' or (mfull and li in mfull):
            mo = MOMEAN[li].unsqueeze(0).expand(B, -1, -1)
        elif mdirs4 == 'all' or mdirs4:
            ks = range(4) if mdirs4 == 'all' else [kk for (l, kk) in mdirs4 if l == li]
            for kk in ks:
                pr = torch.einsum('btd,d->bt', mo, SVD[li, kk])
                mo = mo - (pr - SPROJMEAN[li][:, kk].unsqueeze(0)).unsqueeze(-1) * SVD[li, kk]
        elif proj_all is not None and li in proj_all:
            Dirs, PM = proj_all[li]
            pr = torch.einsum('btd,kd->btk', mo, Dirs)
            mo = mo - torch.einsum('btk,kd->btd', pr - PM.unsqueeze(0), Dirs)
        x = x + mo
    return 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)

# ---- PASS A: means ----
YH_SUM = {li: torch.zeros(SEQL, NH, HD, device=DEV) for li in range(NL)}
MO_SUM = {li: torch.zeros(SEQL, D, device=DEV) for li in range(NL)}
SPROJ_SUM = {li: torch.zeros(SEQL, KMAX, device=DEV) for li in range(NL)}
RPROJ_SUM = {li: torch.zeros(SEQL, KMAX, device=DEV) for li in range(NL)}
print("PASS A: per-position means ...", flush=True)
for i in range(0, NHELD, BATCH): forward(HELD[i:i+BATCH], collect=True)
YHMEAN = {li: YH_SUM[li]/NHELD for li in range(NL)}
MOMEAN = {li: MO_SUM[li]/NHELD for li in range(NL)}
SPROJMEAN = {li: SPROJ_SUM[li]/NHELD for li in range(NL)}
RPROJMEAN = {li: RPROJ_SUM[li]/NHELD for li in range(NL)}
del YH_SUM, MO_SUM, SPROJ_SUM, RPROJ_SUM
print("PASS A done.", flush=True)

# ---- NAMED sets (VERBATIM coverage ledger) ----
NAMED_CORE = sorted(set([
    'h.L8.3', 'h.L8.4', 'h.L13.0', 'h.L14.7', 'h.L7.3', 'h.L5.5', 'h.L8.7', 'h.L13.8',
    'h.L0.3', 'h.L1.1', 'h.L11.2', 'mlp.L17.d1', 'mlp.L17.d2', 'mlp.L17.d3', 'mlp.L16.d2',
    'mlp.L16.d0', 'mlp.L15.d2', 'mlp.L16.d1']))
NAMED_EXT = sorted(set(NAMED_CORE) | set([
    'h.L9.6', 'h.L6.0', 'h.L3.3', 'h.L8.2', 'h.L6.7', 'h.L4.0', 'h.L5.7', 'mlp.L1.d3']))
def comp_to_spec(comps):
    heads, dirs = set(), set()
    for c in comps:
        p = c.split('.'); li = int(p[1][1:])
        if c.startswith('h.'): heads.add((li, int(p[2])))
        else: dirs.add((li, int(p[2][1:])))
    return heads, dirs
nc_h, nc_d = comp_to_spec(NAMED_CORE); ne_h, ne_d = comp_to_spec(NAMED_EXT)

def projK_cfg(dirs_tensor, projmean, K):
    return {'proj_all': {li: (dirs_tensor[li, :K].contiguous(), projmean[li][:, :K].contiguous()) for li in range(NL)}}

CONFIGS = {
    # attack 2 (global headroom joints)
    'FULL_HEADROOM': {'heads': 'all', 'mlp_full': 'all'},
    'JOINT_234':     {'heads': 'all', 'mlp_dirs4': 'all'},
    'NAMED_CORE':    {'heads': nc_h, 'mlp_dirs4': nc_d},
    'NAMED_EXT':     {'heads': ne_h, 'mlp_dirs4': ne_d},
    # attack 3 (mlp rank sweep, attention intact)
    'MLP_FULL':      {'mlp_full': 'all'},
    'SVD_K8':        projK_cfg(SVD, SPROJMEAN, 8),
    'SVD_K32':       projK_cfg(SVD, SPROJMEAN, 32),
    'RAND_K8':       projK_cfg(RAND, RPROJMEAN, 8),
    'RAND_K32':      projK_cfg(RAND, RPROJMEAN, 32),
}
held_np = HELD.cpu().numpy(); tgt_all = torch.from_numpy(held_np).to(DEV)
NPOS = NHELD*(SEQL-1)
per_pos = {name: np.zeros(NPOS, np.float64) for name in CONFIGS}
print(f"PASS B: {len(CONFIGS)} configs over {NPOS} positions ...", flush=True)
t0 = time.time(); ptr = 0
for bi, i in enumerate(range(0, NHELD, BATCH)):
    sb = slice(i, min(i+BATCH, NHELD)); idx = HELD[sb]; b = idx.shape[0]
    tgt = tgt_all[sb]
    base = forward(idx).float()
    blp = F.log_softmax(base[:, :SEQL-1], -1)
    bce = -blp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1); del base, blp
    n_here = b*(SEQL-1)
    for name, spec in CONFIGS.items():
        abl = forward(idx, spec=spec).float()
        alp = F.log_softmax(abl[:, :SEQL-1], -1)
        ace = -alp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1); del abl, alp
        per_pos[name][ptr:ptr+n_here] = (ace-bce).reshape(-1).cpu().numpy()
    ptr += n_here
    if bi % 6 == 0: print(f"  batch {bi+1}/{(NHELD+BATCH-1)//BATCH} elapsed {time.time()-t0:.0f}s", flush=True)
assert ptr == NPOS
print(f"PASS B done in {time.time()-t0:.0f}s", flush=True)

def mean_se(a): return float(a.mean()), float(a.std(ddof=1)/math.sqrt(len(a)))
def diff_se(a, b): d = a-b; return float(d.mean()), float(d.std(ddof=1)/math.sqrt(len(d)))
M = {k: mean_se(per_pos[k]) for k in CONFIGS}
H = M['FULL_HEADROOM'][0]; J = M['JOINT_234'][0]
Nc = M['NAMED_CORE'][0]; Ne = M['NAMED_EXT'][0]
FULLmlp = M['MLP_FULL'][0]

# ---- census single-path sums (same positions/scale) ----
census = json.load(open(f'{QK}/qk_census_difficulty.json'))
recs = {r['comp']: r for r in census['records']}
def pos(x): return x if x > 0 else 0.0
single_total = sum(pos(r['global_dCE']) for r in recs.values())
def sum_named(comps): return sum(pos(recs[c]['global_dCE']) for c in comps if c in recs)
nc_sum = sum_named(NAMED_CORE); ne_sum = sum_named(NAMED_EXT)

# ---- named-set super-additivity (its own joint vs its own single-path sum) ----
nc_superadd = Nc/nc_sum if nc_sum else None
ne_superadd = Ne/ne_sum if ne_sum else None

# ---- named fraction on EVERY consistent denominator ----
def r4(x): return round(x, 4)
attack2 = {
    'joints_reproduced': {
        'FULL_HEADROOM': {'dCE': r4(H), 'SE': r4(M['FULL_HEADROOM'][1])},
        'JOINT_234': {'dCE': r4(J), 'SE': r4(M['JOINT_234'][1])},
        'NAMED_CORE_joint': {'dCE': r4(Nc), 'SE': r4(M['NAMED_CORE'][1])},
        'NAMED_EXT_joint': {'dCE': r4(Ne), 'SE': r4(M['NAMED_EXT'][1])},
        'whole_model_superadditivity_J_over_singlesum': r4(J/single_total),
    },
    'named_set_own_additivity': {
        'named_core_joint': r4(Nc), 'named_core_singlepath_sum': r4(nc_sum),
        'named_core_joint_over_sum': r4(nc_superadd),
        'named_ext_joint': r4(Ne), 'named_ext_singlepath_sum': r4(ne_sum),
        'named_ext_joint_over_sum': r4(ne_superadd),
        'note': 'ratio ~1 => named set is near-ADDITIVE (its joint ~ sum of its solos), '
                'unlike the whole-model 2.87x. So the 11% uses a near-additive numerator.',
    },
    'named_fraction_by_denominator': {
        'core': {
            'of_full_headroom': r4(Nc/H),
            'of_joint234': r4(Nc/J),
            'of_singlepath_sum': r4(nc_sum/single_total),
        },
        'ext': {
            'of_full_headroom': r4(Ne/H),
            'of_joint234': r4(Ne/J),
            'of_singlepath_sum': r4(ne_sum/single_total),
        },
        'denominators': {'full_headroom': r4(H), 'joint234': r4(J), 'singlepath_sum': r4(single_total)},
        'desc': 'named fraction swings with denominator: headroom (joint of all) < joint234 < singlepath-sum. '
                'Report the range and fairest statement.',
    },
}

# ---- ATTACK 3 (rank curve) ----
def cap_frac(name):
    mval, mse = M[name]; return mval, mse, mval/FULLmlp
attack3 = {'mlp_full_dCE': r4(FULLmlp), 'mlp_full_SE': r4(M['MLP_FULL'][1])}
for name, claim in [('SVD_K8', 0.497), ('SVD_K32', 0.835), ('RAND_K8', None), ('RAND_K32', None)]:
    mval, mse, frac = cap_frac(name)
    attack3[name] = {'captured_dCE': r4(mval), 'SE': r4(mse), 'frac_of_full': r4(frac),
                     'claimed_frac_§73': claim,
                     'matches': (abs(frac-claim) < 0.02) if claim is not None else None}
attack3['svd_over_random'] = {
    'K8': r4(M['SVD_K8'][0]/M['RAND_K8'][0]) if M['RAND_K8'][0] > 1e-9 else None,
    'K32': r4(M['SVD_K32'][0]/M['RAND_K32'][0]) if M['RAND_K32'][0] > 1e-9 else None,
}

out = {
    'meta': {
        'model': 'bilin18', 'held_slice': 'FW[448:600,:128]', 'n_positions': NPOS, 'BATCH': BATCH,
        'currency': 'GLOBAL mean delta cross-entropy per valid held position (nats), paired SE',
        'forward': 'VERBATIM qk_coverage_ledger.py / qk_mlp_superposition.py',
        'named_core_n': len(NAMED_CORE), 'named_ext_n': len(NAMED_EXT),
    },
    'attack2_denominator': attack2,
    'attack3_rank_curve': attack3,
}
json.dump(out, open(f'{QK}/qk_redteam_completeness_2.json', 'w'), indent=2)
print("\n===== ATTACK 2 (denominator) =====", flush=True)
print(f"H={H:.4f}  J={J:.4f}  NAMED_CORE={Nc:.4f}  NAMED_EXT={Ne:.4f}  single_sum={single_total:.4f}", flush=True)
print(f"named_ext joint/sum = {ne_superadd:.3f} (near-additive if ~1)", flush=True)
print(f"NAMED_EXT fraction: of headroom {Ne/H:.3f} | of joint234 {Ne/J:.3f} | of singlepath-sum {ne_sum/single_total:.3f}", flush=True)
print("\n===== ATTACK 3 (rank curve) =====", flush=True)
print(f"MLP_FULL {FULLmlp:.4f}", flush=True)
for name in ['SVD_K8', 'SVD_K32', 'RAND_K8', 'RAND_K32']:
    print(f"  {name}: frac_of_full {attack3[name]['frac_of_full']} (claim {attack3[name]['claimed_frac_§73']})", flush=True)
print(f"  SVD/random K8 {attack3['svd_over_random']['K8']}x  K32 {attack3['svd_over_random']['K32']}x", flush=True)
print("\nSaved qk_redteam_completeness_2.json", flush=True)
print("QK REDTEAM COMPLETENESS (attacks 2+3) DONE", flush=True)
