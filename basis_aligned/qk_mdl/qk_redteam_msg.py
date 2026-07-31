"""ADVERSARIAL RED-TEAM of RESULTS section 102 (qk_msg_bottleneck.py/_2).

Four attacks:
 1. COMPARABILITY of half-deletion tests (surrogate abstract-coordinate deletion vs real
    shared-subspace deletion). Missing apples-to-apples cells:
      - real region, random half of a FIXED orthonormal basis of the residual,
        (i) same half across blocks, (ii) different random half per block;
      - real region, per-block INDEPENDENT random half-subspaces (fresh Haar per block);
      - surrogate r=1 with a random half-SUBSPACE (not coordinates) of the message.
 2. TOKEN-FREQUENCY INFLATION of current-token R^2 on top-16 message dims:
      (a) excluding positions of the 20 most frequent held tokens,
      (b) equal per-token weighting (tokens with >=10 held occurrences), with permutation
          controls; next-token category recomputed on the same eligible sets.
 3. BASIS-GRAIN attack on "message not compact": per-block top-k bases (equal split and
    pooled-eigenvalue allocation) and a shared basis fit on concatenated per-block
    deviations (gram = sum of per-block grams), at total kept dimension 144/288/576.
 4. SANITY on 3.8x super-additivity: recompute the seven per-block floors and the
    pair floor for blocks {7,9} with THIS machinery (same held means, same held slice).

Conventions VERBATIM qk_msg_bottleneck: bilin18, forward skeleton unchanged, held means
from qk_msg_bottleneck_state.pt, held FW[448:600,:128], paired SE. TRAIN FW[0:256,:128]
for attack-3 grams. Batch 4, GPU guard, expandable segments. Writes qk_redteam_msg.json."""
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
OUT = f'{QK}/qk_redteam_msg.json'
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

ST = torch.load(STATE, weights_only=False)
U = ST['U']                                   # (D, D) shared train-gram basis, cols desc
EIG = ST['eigval']
M_HELD = ST['M_HELD']                         # (S,T,144) held deviation coords in U
base = ST['base_ce']                          # (S, T-1)
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
FW = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
TRAIN = FW[0:256, :128].to(DEV); HELD = FW[448:600, :128].to(DEV); B0 = 4
STR, TTR = TRAIN.shape; S_, T_ = HELD.shape; N = S_*T_
BLK = list(range(5, 12)); BLKSET = set(BLK)
MEANS = {li: ST['MEANS'][li].to(DEV) for li in BLK}
print(f"bilin18 NL={NL} D={D} blocks {BLK} train {STR}x{TTR} held {S_}x{T_} batch {B0}", flush=True)

RES = {}
def save():
    json.dump(RES, open(OUT, 'w'), indent=1)

# =====================================================================================
# Forward: skeleton VERBATIM qk_msg_bottleneck.py / _2.py fwd.
#   floorset: set of blocks replaced by per-position held mean
#   keepdict: dict li -> (D,k) orthonormal keep basis applied to that block's deviation
#   codeP/codeG: shared message basis P (D,k) + code map G (k,k) applied per block
#   train_collect: dict li -> [G_raw (D,D) f64, S_sum (T,D) f64] accumulated in place
# =====================================================================================
@torch.no_grad()
def fwd(idx, floorset=None, keepdict=None, codeP=None, codeG=None, train_collect=None):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
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
            if train_collect is not None:
                mf = mo.double()
                train_collect[li][0] += torch.einsum('btd,bte->de', mf, mf)
                train_collect[li][1] += mf.sum(0)
                del mf
            if floorset is not None and li in floorset:
                mo = MEANS[li].unsqueeze(0).to(mo.dtype).expand(B, -1, -1)
            elif keepdict is not None:
                P = keepdict[li]
                dev = mo - MEANS[li].unsqueeze(0)
                mo = MEANS[li].unsqueeze(0) + (dev @ P) @ P.T
            elif codeP is not None:
                dev = mo - MEANS[li].unsqueeze(0)
                mm = dev @ codeP
                if codeG is not None: mm = mm @ codeG
                mo = MEANS[li].unsqueeze(0) + mm @ codeP.T
        x = x + mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1),
                         reduction='none').view(B, T-1)
    return ce

def dstat(ce):
    d = (ce - base).flatten().double(); return float(d.mean()), float(d.std()/np.sqrt(d.numel()))

def run(**kw):
    ces = []
    for i in range(0, S_, B0):
        ces.append(fwd(HELD[i:i+B0], **kw).cpu())
    return dstat(torch.cat(ces, 0))

# =====================================================================================
# GATES: reproduce section-102 anchor numbers with this script's machinery.
# =====================================================================================
prior = json.load(open(f'{QK}/qk_msg_bottleneck.json'))
print("GATES ...", flush=True)
g_id = run(floorset=set())                         # identity path (no intervention branch)
Uk576 = U[:, :576].to(DEV).contiguous()
g_svd = run(keepdict={li: Uk576 for li in BLK})
g_fl = run(floorset=BLKSET)
print(f"  identity {g_id[0]:+.5f}  svd576 {g_svd[0]:+.5f} (prior {prior['exp1_joint_write_bottleneck']['svd_keep']['576']['dCE']:+.5f})"
      f"  joint floor {g_fl[0]:+.4f} (prior {prior['exp1_joint_write_bottleneck']['joint_floor']['dCE']:+.4f})", flush=True)
assert abs(g_id[0]) < 0.003
assert abs(g_svd[0] - prior['exp1_joint_write_bottleneck']['svd_keep']['576']['dCE']) < 0.005
assert abs(g_fl[0] - prior['exp1_joint_write_bottleneck']['joint_floor']['dCE']) < 0.02
F_JOINT = g_fl[0]
RES['gates'] = {'identity_dCE': round(g_id[0], 6),
                'svd576_dCE': [round(g_svd[0], 5), round(g_svd[1], 6)],
                'joint_floor_dCE': [round(g_fl[0], 4), round(g_fl[1], 5)]}
save()

# =====================================================================================
# ATTACK 4 -- per-block floors + pair floor with THIS machinery.
# =====================================================================================
print("ATTACK 4: per-block floors (this machinery) ...", flush=True)
census = {int(li): rec['floor_dCE'] for li, rec in
          json.load(open(f'{QK}/qk_allterm_census.json'))['layers'].items()}
a4 = {'per_block': {}, }
tot = 0.0
for li in BLK:
    mn, se = run(floorset={li})
    a4['per_block'][li] = {'dCE': round(mn, 5), 'SE': round(se, 6), 'census': census[li]}
    tot += mn
    print(f"  block {li}: floor {mn:+.5f} +- {se:.6f}  (census {census[li]:+.4f})", flush=True)
a4['sum_per_block'] = round(tot, 4)
a4['sum_census'] = round(sum(census[li] for li in BLK), 4)
a4['joint_floor'] = {'dCE': round(F_JOINT, 4), 'SE': round(g_fl[1], 5)}
a4['ratio_joint_over_sum'] = round(F_JOINT/tot, 3)
print(f"  sum per-block {tot:+.4f}  joint {F_JOINT:+.4f}  ratio {F_JOINT/tot:.3f}", flush=True)
mn, se = run(floorset={7, 9})
s79 = a4['per_block'][7]['dCE'] + a4['per_block'][9]['dCE']
a4['pair_7_9'] = {'dCE': round(mn, 5), 'SE': round(se, 6), 'sum_singles': round(s79, 5),
                  'ratio': round(mn/s79, 3)}
print(f"  pair {{7,9}}: floor {mn:+.5f} +- {se:.6f}  sum singles {s79:+.5f}  ratio {mn/s79:.3f}", flush=True)
RES['attack4_superadditivity'] = a4
save()

# =====================================================================================
# ATTACK 1 -- apples-to-apples half-deletion cells.
# =====================================================================================
print("ATTACK 1: matched half-deletion cells ...", flush=True)
a1 = {'documentation': {
    'surrogate_op': 'delete random half of the r*576 ABSTRACT code coordinates; least-squares '
                    'decode; composite map G = projection in the 576-dim message space, applied '
                    'identically to every block; the 576-dim PCA tail is already discarded '
                    '(ceiling +0.111); for r>=2 the code space exceeds the residual width and for '
                    'independent Haar frames G=I after any half-deletion is a linear-algebra '
                    'certainty, not a model measurement',
    'real_op': 'keep ONE random 576-of-1152 subspace of the physical residual, the SAME subspace '
               'for all seven blocks (equivalently, applied to the summed write); touches both '
               'message and tail'}}

# (i) fixed orthonormal basis of the residual, delete random half of its columns, SAME half
#     across blocks (coordinate-matched analog of the existing shared-subspace cell)
gcpu = torch.Generator().manual_seed(300)
Qb = torch.linalg.qr(torch.randn(D, D, generator=gcpu))[0].to(DEV)
cells = []
for seed in (0, 1):
    gp = torch.Generator().manual_seed(310 + seed)
    perm = torch.randperm(D, generator=gp)[:D//2]
    P = Qb[:, perm.to(DEV)].contiguous()
    mn, se = run(keepdict={li: P for li in BLK})
    cells.append({'seed': seed, 'dCE': round(mn, 5), 'SE': round(se, 6)})
    print(f"  fixed-basis shared half seed {seed}: {mn:+.5f} +- {se:.6f}", flush=True)
a1['real_fixed_basis_shared_half'] = cells

# (ii) same fixed basis, DIFFERENT random half per block
cells = []
for seed in (0, 1, 2):
    kd = {}
    for j, li in enumerate(BLK):
        gp = torch.Generator().manual_seed(320 + 100*seed + li)
        perm = torch.randperm(D, generator=gp)[:D//2]
        kd[li] = Qb[:, perm.to(DEV)].contiguous()
    mn, se = run(keepdict=kd)
    cells.append({'seed': seed, 'dCE': round(mn, 5), 'SE': round(se, 6)})
    print(f"  fixed-basis PER-BLOCK half seed {seed}: {mn:+.5f} +- {se:.6f}", flush=True)
a1['real_fixed_basis_per_block_half'] = cells

# (iii) per-block INDEPENDENT Haar half-subspaces
cells = []
for seed in (0, 1):
    kd = {}
    for li in BLK:
        gg = torch.Generator(device=DEV).manual_seed(400 + 100*seed + li)
        kd[li] = torch.linalg.qr(torch.randn(D, D, generator=gg, device=DEV))[0][:, :D//2].contiguous()
    mn, se = run(keepdict=kd)
    cells.append({'seed': seed, 'dCE': round(mn, 5), 'SE': round(se, 6)})
    print(f"  independent-Haar PER-BLOCK half seed {seed}: {mn:+.5f} +- {se:.6f}", flush=True)
a1['real_independent_per_block_half'] = cells

# (iv) surrogate r=1 with a random half-SUBSPACE of the message (not coordinates)
cells = []
for seed in (0, 1):
    gg = torch.Generator().manual_seed(500 + seed)
    Vh = torch.linalg.qr(torch.randn(576, 576, generator=gg))[0][:, :288]
    G = (Vh @ Vh.T).float().to(DEV)
    mn, se = run(codeP=Uk576, codeG=G)
    cells.append({'seed': seed, 'dCE': round(mn, 5), 'SE': round(se, 6)})
    print(f"  surrogate r=1 half-SUBSPACE seed {seed}: {mn:+.5f} +- {se:.6f}", flush=True)
a1['surrogate_r1_half_subspace'] = cells
a1['surrogate_r1_half_coordinates_prior'] = prior['exp3_bespoke_surrogate']['half_deletion']['r1']
a1['real_shared_subspace_prior'] = prior['exp3_bespoke_surrogate']['real_region_half_deletion']
a1['analytic_note'] = ('with SHARED frames across copies and the SAME half deleted from every copy, '
                       'G has rank 288 for EVERY r (H = r * R^T M R): even r=16 costs ~+0.73 under '
                       'the shared-half attack -- the existing real-region cell is exactly this '
                       'shared-half attack, so it cannot lower-bound redundancy')
RES['attack1_comparability'] = a1
save()

# =====================================================================================
# ATTACK 3 -- basis grain: per-block grams (TRAIN), per-block bases, concatenated gram.
# =====================================================================================
print("ATTACK 3: train pass for per-block deviation grams ...", flush=True)
tc = {li: [torch.zeros(D, D, device=DEV, dtype=torch.float64),
           torch.zeros(TTR, D, device=DEV, dtype=torch.float64)] for li in BLK}
t0 = time.time()
for i in range(0, STR, B0):
    fwd(TRAIN[i:i+B0], train_collect=tc)
print(f"  train pass done ({time.time()-t0:.0f}s)", flush=True)
GB, EIGB, UB = {}, {}, {}
for li in BLK:
    Sbar = tc[li][1] / STR
    Gd = tc[li][0] - STR * torch.einsum('td,te->de', Sbar, Sbar)
    ev, evec = torch.linalg.eigh(Gd.cpu())
    EIGB[li] = ev.flip(0).clamp_min(0); UB[li] = evec.flip(1).float().contiguous()
    del Gd
Gcat = sum((tc[li][0] - STR*torch.einsum('td,te->de', tc[li][1]/STR, tc[li][1]/STR)) for li in BLK)
evc, evecc = torch.linalg.eigh(Gcat.cpu())
EIGC = evc.flip(0).clamp_min(0); UC = evecc.flip(1).float().contiguous()
del tc

a3 = {'shared_summed_gram_prior': {k: prior['exp1_joint_write_bottleneck']['svd_keep'][k]
                                   for k in ('144', '288', '576')},
      'joint_floor_ref': round(F_JOINT, 4), 'cells': {}}
def alloc_equal(total):
    b = total // 7; extra = total - 7*b
    return {li: b + (1 if j < extra else 0) for j, li in enumerate(BLK)}
def alloc_pooled(total):
    allv = torch.cat([EIGB[li] for li in BLK])
    lab = torch.cat([torch.full((D,), li) for li in BLK])
    idx = allv.argsort(descending=True)[:total]
    cnt = {li: int((lab[idx] == li).sum()) for li in BLK}
    return cnt
for total in (144, 288, 576):
    for name, alloc in (('per_block_equal', alloc_equal(total)),
                        ('per_block_pooled_eig', alloc_pooled(total))):
        kd = {li: UB[li][:, :alloc[li]].to(DEV).contiguous() for li in BLK}
        mn, se = run(keepdict=kd)
        rec = 1 - mn/F_JOINT
        a3['cells'][f'{name}_{total}'] = {'dCE': round(mn, 5), 'SE': round(se, 6),
                                          'recovered_fraction': round(rec, 4),
                                          'alloc': {int(li): int(alloc[li]) for li in BLK}}
        print(f"  {name} total {total}: {mn:+.5f} +- {se:.6f}  recovered {rec:.4f}  alloc {alloc}", flush=True)
    P = UC[:, :total].to(DEV).contiguous()
    mn, se = run(keepdict={li: P for li in BLK})
    rec = 1 - mn/F_JOINT
    a3['cells'][f'shared_concat_gram_{total}'] = {'dCE': round(mn, 5), 'SE': round(se, 6),
                                                  'recovered_fraction': round(rec, 4)}
    print(f"  shared concat-gram total {total}: {mn:+.5f} +- {se:.6f}  recovered {rec:.4f}", flush=True)
RES['attack3_basis_grain'] = a3
save()

# =====================================================================================
# ATTACK 2 -- token-frequency inflation (CPU, on stored message coords).
# =====================================================================================
print("ATTACK 2: token-frequency attack on conditioning R^2 ...", flush=True)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('gpt2')
held_np = HELD.cpu().numpy()
flat_ids = held_np.reshape(-1)
M16 = M_HELD[:, :, :16].reshape(N, 16).double()

def group_r2(X64, gids, elig, min_count):
    g = gids[elig]
    uniq, inv, counts = np.unique(g, return_inverse=True, return_counts=True)
    keep = counts >= min_count
    cov = np.zeros(len(gids), bool)
    idx_el = np.where(elig)[0]
    cov[idx_el[keep[inv]]] = True
    Xc = X64[torch.from_numpy(cov)]
    gcov = gids[cov]
    u2, inv2 = np.unique(gcov, return_inverse=True)
    G = len(u2)
    s = torch.zeros(G, X64.shape[1], dtype=torch.float64)
    s.index_add_(0, torch.from_numpy(inv2), Xc)
    n = torch.from_numpy(np.bincount(inv2).astype(np.float64))
    sq = float(Xc.pow(2).sum())
    within = sq - float((s.pow(2).sum(1)/n).sum())
    grand = Xc.sum(0)
    tot = sq - float(grand.pow(2).sum())/Xc.shape[0]
    return 1.0 - within/max(tot, 1e-12), G, cov

def shuffle_control(X64, gids, cov, seed=0):
    rng = np.random.default_rng(seed)
    g = gids.copy(); idxc = np.where(cov)[0]
    g[idxc] = g[rng.permutation(idxc)]
    r2, _, _ = group_r2(X64, g, cov, 1)
    return r2

# six-way next-token category, VERBATIM qk_msg_bottleneck_2.py
import string as _string
_P = set(_string.punctuation)
FUNC = {'the','of','and','to','a','in','is','that','it','for','was','as','with','on','be','at','by','this','are','from','or','an','but','not','which'}
Vgpt = 50257
CAT = np.full(max(50304, int(flat_ids.max())+1), 5, dtype=np.int64)
for i in range(Vgpt):
    s = tok.convert_ids_to_tokens(i)
    if s is None: continue
    core = s.replace('Ġ', ''); lead = s.startswith('Ġ')
    if len(core) and all(c in _P for c in core): CAT[i] = 1
    elif len(core) and all(c.isdigit() for c in core): CAT[i] = 3
    elif core.lower() in FUNC: CAT[i] = 4
    elif not lead and len(core) and core[0].isalpha() and core[0].islower(): CAT[i] = 0
    elif lead and len(core) and core[0].isupper(): CAT[i] = 2
nc = CAT[held_np[:, 1:].reshape(-1)]
next_cat_2d = np.full((S_, T_), -1, dtype=np.int64); next_cat_2d[:, :-1] = nc.reshape(S_, T_-1)
next_cat = next_cat_2d.reshape(-1)

a2 = {}
r2_base, Gb_, cov_base = group_r2(M16, flat_ids, np.ones(N, bool), 5)
ctl_base = shuffle_control(M16, flat_ids, cov_base)
a2['baseline_gate'] = {'R2': round(r2_base, 4), 'control': round(ctl_base, 4), 'n_groups': int(Gb_),
                       'prior_R2': 0.4721}
print(f"  gate: cur_token R2 {r2_base:.4f} (ctl {ctl_base:.4f}) vs prior 0.4721", flush=True)

# (a) exclude top-20 most frequent held tokens
cnt = np.bincount(flat_ids, minlength=50304)
top20 = cnt.argsort()[::-1][:20]
top20_strs = [tok.convert_ids_to_tokens(int(i)) for i in top20]
top20_share = float(cnt[top20].sum()) / N
elig_nf = ~np.isin(flat_ids, top20)
r2_nf, G_nf, cov_nf = group_r2(M16, flat_ids, elig_nf, 5)
ctl_nf = shuffle_control(M16, flat_ids, cov_nf)
r2_nc_nf, Gnc_, covnc_ = group_r2(M16, next_cat, elig_nf & (next_cat >= 0), 1)
ctl_nc_nf = shuffle_control(M16, next_cat, covnc_)
pd_nf = [round(group_r2(M16[:, j:j+1], flat_ids, elig_nf, 5)[0], 4) for j in range(16)]
a2['exclude_top20'] = {'top20_tokens': top20_strs, 'top20_position_share': round(top20_share, 4),
                       'cur_token': {'R2': round(r2_nf, 4), 'control': round(ctl_nf, 4),
                                     'n_groups': int(G_nf), 'covered_frac': round(float(cov_nf.mean()), 4),
                                     'per_dim': pd_nf},
                       'next_cat6_same_elig': {'R2': round(r2_nc_nf, 4), 'control': round(ctl_nc_nf, 4)}}
print(f"  EXCLUDING top-20 tokens ({top20_share:.1%} of positions): cur_token R2 {r2_nf:.4f} "
      f"(ctl {ctl_nf:.4f}, {G_nf} groups, cov {cov_nf.mean():.3f}); next_cat6 {r2_nc_nf:.4f} (ctl {ctl_nc_nf:.4f})", flush=True)
print(f"    top20: {top20_strs}", flush=True)

# (b) equal per-token weighting, tokens with >=10 held occurrences
def equal_weight_stats(gids, seed=None):
    g = gids.copy()
    if seed is not None:
        rng = np.random.default_rng(seed)
        g = g[rng.permutation(len(g))]
    uniq, counts = np.unique(g, return_counts=True)
    toks = uniq[counts >= 10]
    r2s, ns = [], []
    Xn = M16.numpy()
    mask_all = np.isin(g, toks)
    mu = Xn[mask_all].mean(0)
    for t in toks:
        idx = np.where(g == t)[0]
        Xt = Xn[idx]
        sst = ((Xt - mu)**2).sum()
        ssw = ((Xt - Xt.mean(0))**2).sum()
        r2s.append(1 - ssw/max(sst, 1e-12)); ns.append(len(idx))
    r2s = np.array(r2s)
    # count-weighted aggregate over the same tokens (for reference)
    return r2s, np.array(ns), toks

r2s, ns, toks10 = equal_weight_stats(flat_ids)
r2s_ctl, _, _ = equal_weight_stats(flat_ids, seed=0)
r2s_ctl2, _, _ = equal_weight_stats(flat_ids, seed=1)
eqw = float(r2s.mean()); eqw_ctl = float(np.mean([r2s_ctl.mean(), r2s_ctl2.mean()]))
cw = float((r2s*ns).sum()/ns.sum())
a2['equal_per_token'] = {'n_tokens_ge10': int(len(toks10)),
                         'mean_within_token_R2': round(eqw, 4),
                         'permutation_control': round(eqw_ctl, 4),
                         'median_within_token_R2': round(float(np.median(r2s)), 4),
                         'count_weighted_same_tokens': round(cw, 4),
                         'coverage_frac': round(float(np.isin(flat_ids, toks10).mean()), 4)}
print(f"  EQUAL per-token (>=10 occ, {len(toks10)} tokens): mean within-token R2 {eqw:.4f} "
      f"(perm ctl {eqw_ctl:.4f}); median {np.median(r2s):.4f}; count-weighted same tokens {cw:.4f}", flush=True)

# equal-token-mass weighted aggregate R2 for cur_token AND next_cat6 (weights 1/n_token)
def weighted_group_r2(gids_group, weights, elig):
    Xn = M16.numpy()[elig]; g = gids_group[elig]; w = weights[elig]
    u2, inv2 = np.unique(g, return_inverse=True)
    Wg = np.bincount(inv2, weights=w)
    s = np.zeros((len(u2), 16))
    for j in range(16):
        s[:, j] = np.bincount(inv2, weights=w*Xn[:, j])
    sq = float((w[:, None]*Xn**2).sum())
    within = sq - float((s**2).sum(1) @ (1.0/Wg))
    grand = (w[:, None]*Xn).sum(0)
    tot = sq - float((grand**2).sum())/w.sum()
    return 1 - within/max(tot, 1e-12)
nmap = np.zeros(50304); nmap[toks10] = 1.0/np.maximum(cnt[toks10], 1)
wts = nmap[flat_ids]
elig10 = np.isin(flat_ids, toks10)
r2w_cur = weighted_group_r2(flat_ids, wts, elig10)
r2w_nc = weighted_group_r2(next_cat, wts, elig10 & (next_cat >= 0))
a2['equal_token_mass_aggregate'] = {'cur_token': round(r2w_cur, 4), 'next_cat6': round(r2w_nc, 4)}
print(f"  equal-token-mass aggregate: cur_token {r2w_cur:.4f}  next_cat6 {r2w_nc:.4f}", flush=True)
RES['attack2_token_frequency'] = a2
save()
print(f"Saved {OUT}", flush=True)
print("QK REDTEAM MSG DONE", flush=True)
