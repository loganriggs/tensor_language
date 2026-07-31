"""ADVERSARIAL RED-TEAM of the depth-first arc series (sections 96-100), script 1 of 2.

ATTACK 1 -- FREEZE-PATCH CIRCULARITY on the cascade chain (sections 96/98/99).
The load-bearing evidence "freezing block k+1 clean removes ~84% of block k's damage" could be
generic: block k+1 might absorb ANY perturbation of matched size, not specifically block k's
content. CONTROLS, at k=2 (the MrxMr term of layer 2, section 96) and k=3 (the full mlp,
section 98), plus k=5 (mid-plateau):
  (real)  the arc's own intervention -- replicate total damage + freeze-next rescue fraction
          (gates: k2 total 0.0288 / freeze-mlp3 0.0047; k3 total 0.6163 / freeze-mlp4 0.1050;
          k5 total 0.0928 / freeze-mlp6 0.0453 -- from qk_arc_square.json / qk_arc_mlp3.json /
          qk_cascade_end.json, same held slice)
  (rand)  inject a RANDOM gaussian perturbation at the same place with per-position norm matched
          to the real deviation; measure the same freeze-next rescue fraction
  (shuf)  inject the REAL deviation from a DIFFERENT sequence (roll along the batch dimension,
          rescaled to the local deviation norm) -- an on-distribution but content-mismatched
          perturbation; same freeze-next rescue fraction
  (nonnext) freeze a single NON-next later mlp clean (k2: mlp4, mlp5; k3: mlp5, mlp6; k5: mlp7)
          -- within-source specificity
  (upstream degenerate) k5 with freeze_mlp={3}: freezing an upstream block is structurally a
          no-op; verifies the cache and that "rescue" cannot come from nowhere.
If rand/shuf freeze-next rescue ~ real's ~84%, the chain-specificity claim weakens to "the next
block is generically compensatory". If real >> rand/shuf, the claim survives.

ATTACK 2 -- VARIANCE-VS-CAUSATION on the block-0 table (section 97).
"Variance explained = 1.000000" is analytic, but "block 0 is a token lookup" needs CAUSAL
sufficiency: replace block 0's mlp output with its TOKEN-CONDITIONAL MEAN built on TRAIN data
(FW[0:448,:128] -- zero overlap with held FW[448:600]), and measure delta cross-entropy against
the 1.2341 floor. Surrogates:
  floor        per-position held mean (gate: must reproduce the census floor 1.2341)
  tok          train token-conditional mean table (fallback: global train mean)
  pair         train (previous, current)-pair-conditional mean where pair count >= 3, else tok
  tok_gauge    gauge-aware table: train mean of (mo - bias)*rho2 | token, divided by the
               position's actual rho2, plus bias -- tests "exact up to the shared gauge scalar"
  pair_gauge   same, pair-conditional with tok_gauge fallback
Recovered fraction = 1 - dCE_surrogate / dCE_floor.

Machinery VERBATIM from qk_arc_square_2.py (term construction + track accumulators + freeze-
patch fwd), qk_cascade_end.py (mlp mean-ablation + freeze sets + paired standard errors +
GPU guard), qk_arc_mlp0.py (layer-0 forward + pair_terms + rho2 convention).
Held FW[448:600,:128], batch 6, footprint <4GB. Writes qk_redteam_arcs.json."""
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
OUT = f'{QK}/qk_redteam_arcs.json'

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
HELD = FW[448:600, :128].to(DEV); B0 = 6
TRAIN = FW[0:448, :128]                       # table-building slice, zero overlap with held
S_, T_ = HELD.shape
STR, TTR = TRAIN.shape
print(f"bilin18 NL={NL} D={D} held {S_}x{T_} train {STR}x{TTR}", flush=True)

# ---- k=2 MrxMr term machinery (VERBATIM qk_arc_square_2.py) ----
ST = torch.load(f'{QK}/qk_arc_square_state.pt', weights_only=False)
GNAMES = ['E', 'Ae', 'Ar', 'Me', 'Mr']; NG = 5
PAIRS = [(i, j) for i in range(NG) for j in range(i, NG)]
PNAMES = [f'{GNAMES[i]}x{GNAMES[j]}' for (i, j) in PAIRS]
MRMR = PNAMES.index('MrxMr')
TMEAN_MRMR = ST['TMEAN'][MRMR].to(DEV)        # (T,D) per-position mean of the MrxMr term
LI2 = 2

def mlp_wts(li):
    b = m.transformer.h[li].mlp
    return (b.Left.weight.detach().float(), b.Right.weight.detach().float(),
            b.Down.weight.detach().float(), b.Down_bias.detach().float())
W2 = mlp_wts(LI2)
W0 = mlp_wts(0)

LAM0 = [float(b.lambdas[0]) for b in m.transformer.h]
CENSUS = {int(li): rec['floor_dCE'] for li, rec in
          json.load(open(f'{QK}/qk_allterm_census.json'))['layers'].items()}

# =====================================================================================
# PASS 0: per-position held-set means of mlp outputs for blocks 0, 3, 5 (VERBATIM
# qk_cascade_end.py fwd_collect_means) + held rho2 at layer 0 (mlp0 convention:
# rho2 = ||x_post_attn||^2 / D, the pair_terms gauge).
# =====================================================================================
MEAN_BLOCKS = [0, 3, 5]
@torch.no_grad()
def fwd_collect_means(idx, sums, rho2_out):
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
        if li == 0 and rho2_out is not None:
            rho2_out.append((x.pow(2).sum(-1)/D).float().cpu())
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if li in sums: sums[li] += mo.float().sum(0)
        x = x + mo

print("PASS 0: per-position mlp-output means (blocks 0,3,5) + held rho2 at layer 0 ...", flush=True)
sums = {k: torch.zeros(T_, D, device=DEV) for k in MEAN_BLOCKS}
rho2_held_l = []
for i in range(0, S_, B0):
    fwd_collect_means(HELD[i:i+B0], sums, rho2_held_l)
MO_MEAN = {k: (sums[k]/S_) for k in MEAN_BLOCKS}
RHO2_HELD0 = torch.cat(rho2_held_l, 0)        # (S,T) rho2 at layer 0 on held
del sums, rho2_held_l

# =====================================================================================
# Unified routed forward. config keys:
#   kind: None | 'term2' (MrxMr at layer 2) | 'mlpk' (full mlp of block k) | 'sub0'
#   ptype: 'real' | 'rand' | 'shuf'   (perturbation type; real = the arc's intervention)
#   k: target block for 'mlpk'; seed: rand generator seed (per batch, shared across configs)
#   freeze_attn / freeze_mlp: sets of later blocks patched to clean cached outputs
#   sub: (B,T,D) tensor replacing block 0's mlp output (attack 2)
# Skeleton + track accumulators VERBATIM qk_arc_square_2.py fwd; freeze-patch + mean-
# ablation VERBATIM qk_cascade_end.py fwd_route.
# =====================================================================================
@torch.no_grad()
def fwd_rt(idx, config=None, cache=None, fill_cache=False):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cfg_ = config or {}
    kind = cfg_.get('kind'); ptype = cfg_.get('ptype', 'real'); KI = cfg_.get('k')
    fa = cfg_.get('freeze_attn', ()); fm = cfg_.get('freeze_mlp', ())
    track = (kind == 'term2')
    if track:
        cE = torch.ones((), device=DEV)
        SA = torch.zeros_like(x); SM = torch.zeros_like(x); MR = torch.zeros_like(x)

    def perturb(mo, dev, seed):
        """real: subtract the deviation (the arc's intervention). rand: add a gaussian with
        per-position norm matched to ||dev||. shuf: add the (negated) deviation of ANOTHER
        sequence (roll along batch), rescaled to the local norm."""
        n = dev.norm(dim=-1, keepdim=True)
        if ptype == 'real':
            return mo - dev
        if ptype == 'rand':
            gen = torch.Generator(device=DEV); gen.manual_seed(seed)
            g = torch.randn(dev.shape, device=DEV, dtype=torch.float32, generator=gen).to(dev.dtype)
            return mo + g * (n / g.norm(dim=-1, keepdim=True).clamp_min(1e-9))
        if ptype == 'shuf':
            dr = torch.roll(dev, 1, dims=0)
            return mo - dr * (n / dr.norm(dim=-1, keepdim=True).clamp_min(1e-9))
        raise ValueError(ptype)

    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        if track and li <= LI2:
            cE = blk.lambdas[0]*cE + blk.lambdas[1]
            SA = blk.lambdas[0]*SA; SM = blk.lambdas[0]*SM; MR = blk.lambdas[0]*MR
        def qk(l): z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        aout = a.c_proj(yh.reshape(B, T, -1))
        if li in fa: aout = cache[li][0]
        x = x + aout
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if li in fm: mo = cache[li][1]
        if kind == 'sub0' and li == 0:
            mo = cfg_['sub'].to(x.dtype)
        if kind == 'term2' and li == LI2:
            # MrxMr term VERBATIM formula (pair_terms specialized to i=j=4)
            rho2 = x.pow(2).sum(-1, keepdim=True) / D
            term = ((MR @ W2[0].T) * (MR @ W2[1].T)) @ W2[2].T / rho2
            dev = term - TMEAN_MRMR[None]
            mo = perturb(mo, dev, cfg_.get('seed', 0))
        if kind == 'mlpk' and li == KI:
            dev = mo - MO_MEAN[KI].unsqueeze(0).to(x.dtype)
            mo = perturb(mo, dev, cfg_.get('seed', 0))
        if fill_cache: cache[li] = (aout, mo)
        x = x + mo
        if track and li < LI2:
            SA = SA + aout; SM = SM + MR; MR = mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1),
                         reduction='none').view(B, T-1)
    return ce

print("BASE ...", flush=True)
base = torch.cat([fwd_rt(HELD[i:i+B0]).cpu() for i in range(0, S_, B0)], 0)
print(f"base CE {float(base.mean()):.4f}", flush=True)

def dstat(ce):
    d = (ce - base).flatten().double(); return float(d.mean()), float(d.std()/np.sqrt(d.numel()))

# =====================================================================================
# ATTACK 1 config table
# =====================================================================================
cfgs = [('sanity_freeze_all_no_ablate',
         {'kind': None, 'freeze_attn': set(range(NL)), 'freeze_mlp': set(range(NL))})]
for pt in ('real', 'rand', 'shuf'):
    cfgs += [(f'k2_{pt}_total',        {'kind': 'term2', 'ptype': pt}),
             (f'k2_{pt}_freeze_next3', {'kind': 'term2', 'ptype': pt, 'freeze_mlp': {3}})]
    if pt != 'shuf':
        cfgs += [(f'k2_{pt}_freeze_only4', {'kind': 'term2', 'ptype': pt, 'freeze_mlp': {4}})]
cfgs += [('k2_real_freeze_only5', {'kind': 'term2', 'ptype': 'real', 'freeze_mlp': {5}})]
for pt in ('real', 'rand', 'shuf'):
    cfgs += [(f'k3_{pt}_total',        {'kind': 'mlpk', 'k': 3, 'ptype': pt}),
             (f'k3_{pt}_freeze_next4', {'kind': 'mlpk', 'k': 3, 'ptype': pt, 'freeze_mlp': {4}})]
    if pt != 'shuf':
        cfgs += [(f'k3_{pt}_freeze_only5', {'kind': 'mlpk', 'k': 3, 'ptype': pt, 'freeze_mlp': {5}})]
cfgs += [('k3_real_freeze_only6', {'kind': 'mlpk', 'k': 3, 'ptype': 'real', 'freeze_mlp': {6}})]
for pt in ('real', 'rand'):
    cfgs += [(f'k5_{pt}_total',        {'kind': 'mlpk', 'k': 5, 'ptype': pt}),
             (f'k5_{pt}_freeze_next6', {'kind': 'mlpk', 'k': 5, 'ptype': pt, 'freeze_mlp': {6}})]
cfgs += [('k5_real_freeze_only7',      {'kind': 'mlpk', 'k': 5, 'ptype': 'real', 'freeze_mlp': {7}}),
         ('k5_real_freeze_upstream3',  {'kind': 'mlpk', 'k': 5, 'ptype': 'real', 'freeze_mlp': {3}})]
print(f"ATTACK 1: {len(cfgs)} configs x {(S_+B0-1)//B0} batches ...", flush=True)

acc = {name: [] for name, _ in cfgs}
t0 = time.time()
for i in range(0, S_, B0):
    idx = HELD[i:i+B0]
    cache = {}
    fwd_rt(idx, fill_cache=True, cache=cache)
    for name, c in cfgs:
        acc[name].append(fwd_rt(idx, config=dict(c, seed=10000+i), cache=cache).cpu())
    del cache
    if (i//B0) % 5 == 0:
        print(f"  batch {i//B0+1}/{(S_+B0-1)//B0} ({time.time()-t0:.0f}s)", flush=True)
print(f"ATTACK 1 forwards done ({time.time()-t0:.0f}s)", flush=True)

raw1 = {}
for name, _ in cfgs:
    mn, se = dstat(torch.cat(acc[name], 0))
    raw1[name] = {'dCE': round(mn, 5), 'SE': round(se, 6)}
    print(f"  {name:26s} dCE {mn:+.5f} +- {se:.6f}", flush=True)
del acc

assert abs(raw1['sanity_freeze_all_no_ablate']['dCE']) < 0.003, "freeze-all sanity FAILED"
ANCHOR = {'k2': {'total': 0.0288, 'freeze_next': 0.0047},
          'k3': {'total': 0.6163, 'freeze_next': 0.1050},
          'k5': {'total': 0.0928, 'freeze_next': 0.0453}}
attack1 = {'raw': raw1, 'anchors': ANCHOR, 'summary': {}}
for tag, nxt in (('k2', 3), ('k3', 4), ('k5', 6)):
    for pt in ('real', 'rand', 'shuf'):
        tn = f'{tag}_{pt}_total'
        if tn not in raw1: continue
        tot = raw1[tn]['dCE']; frz = raw1[f'{tag}_{pt}_freeze_next{nxt}']['dCE']
        rec = {'total_dCE': tot, 'total_SE': raw1[tn]['SE'],
               'freeze_next_dCE': frz,
               'freeze_next_rescue_fraction': round((tot-frz)/tot, 4) if abs(tot) > 1e-6 else None}
        for extra in (f'{tag}_{pt}_freeze_only4', f'{tag}_{pt}_freeze_only5',
                      f'{tag}_{pt}_freeze_only6', f'{tag}_{pt}_freeze_only7',
                      f'{tag}_{pt}_freeze_upstream3'):
            if extra in raw1:
                rec[extra.split(pt+'_')[1] + '_rescue_fraction'] = \
                    round((tot - raw1[extra]['dCE'])/tot, 4) if abs(tot) > 1e-6 else None
        attack1['summary'][f'{tag}_{pt}'] = rec
# replication gates against the arcs (loose 20-percent relative tolerance: same intervention,
# fresh code path)
for tag in ('k2', 'k3', 'k5'):
    got_t = attack1['summary'][f'{tag}_real']['total_dCE']
    got_f = attack1['summary'][f'{tag}_real']['freeze_next_dCE']
    attack1['summary'][f'{tag}_real']['replication_gate'] = {
        'anchor_total': ANCHOR[tag]['total'], 'got_total': got_t,
        'anchor_freeze_next': ANCHOR[tag]['freeze_next'], 'got_freeze_next': got_f,
        'pass': bool(abs(got_t - ANCHOR[tag]['total']) < 0.2*ANCHOR[tag]['total'] + 0.002
                     and abs(got_f - ANCHOR[tag]['freeze_next']) < 0.2*ANCHOR[tag]['total'] + 0.002)}
print(json.dumps(attack1['summary'], indent=1), flush=True)

res = {'attack1_freeze_patch_circularity': attack1}
json.dump(res, open(OUT, 'w'), indent=1)

# =====================================================================================
# ATTACK 2 -- token-table causal sufficiency for block 0
# =====================================================================================
print("ATTACK 2 PASS T: collect block-0 mlp outputs on TRAIN (truncated forward) ...", flush=True)
@torch.no_grad()
def fwd_layer0(idx):
    """Truncated forward: exactly the arc forward through block 0's mlp (VERBATIM skeleton),
    returns (mo0, rho2) -- nothing after layer 0 is needed for table building."""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    blk = m.transformer.h[0]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
    def qk(l): z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
    v = a.c_v(hcur).view(B, T, NH, HD)
    q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
    s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
    pat = (s1*s2).masked_fill(~mask, 0.0); yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
    x = x + a.c_proj(yh.reshape(B, T, -1))
    rho2 = x.pow(2).sum(-1)/D
    mo = blk.mlp(F.rms_norm(x, (D,)))
    return mo.float(), rho2.float()

BIAS0 = W0[3].to(DEV)                          # Down_bias of block 0
tok_sum = torch.zeros(V, D, device=DEV); tok_cnt = torch.zeros(V, device=DEV)
num_sum = torch.zeros(V, D, device=DEV)        # gauge numerator (mo - bias)*rho2
mo_train_l, rho2_train_l = [], []
TRAIN_DEV = TRAIN.to(DEV)
for i in range(0, STR, B0):
    idx = TRAIN_DEV[i:i+B0]
    mo, rho2 = fwd_layer0(idx)
    fl = idx.reshape(-1)
    tok_sum.index_add_(0, fl, mo.reshape(-1, D))
    tok_cnt.index_add_(0, fl, torch.ones_like(fl, dtype=torch.float))
    num_sum.index_add_(0, fl, ((mo - BIAS0[None, None]) * rho2[..., None]).reshape(-1, D))
    mo_train_l.append(mo.half().cpu()); rho2_train_l.append(rho2.cpu())
MO_TRAIN = torch.cat(mo_train_l, 0).numpy()    # (STR,TTR,D) fp16
RHO2_TRAIN = torch.cat(rho2_train_l, 0).numpy()
del mo_train_l, rho2_train_l
GLOB = (tok_sum.sum(0)/tok_cnt.sum()).cpu().numpy()
GLOB_NUM = (num_sum.sum(0)/tok_cnt.sum()).cpu().numpy()
seen = tok_cnt > 0
TT = torch.where(seen[:, None], tok_sum/tok_cnt.clamp_min(1)[:, None],
                 torch.from_numpy(GLOB).to(DEV)[None]).cpu().numpy()
NT_ = torch.where(seen[:, None], num_sum/tok_cnt.clamp_min(1)[:, None],
                  torch.from_numpy(GLOB_NUM).to(DEV)[None]).cpu().numpy()
del tok_sum, num_sum
print(f"token table: {int(seen.sum())} tokens seen in train", flush=True)

# pair tables on CPU (previous, current); position 0 has no previous
print("building pair tables ...", flush=True)
tr_np = TRAIN.numpy()
cur = tr_np[:, 1:].reshape(-1); prv = tr_np[:, :-1].reshape(-1)
pair_id = prv.astype(np.int64)*V + cur
mo_flat = MO_TRAIN[:, 1:].reshape(-1, D).astype(np.float32)
num_flat = (mo_flat - W0[3].cpu().numpy()[None]) * RHO2_TRAIN[:, 1:].reshape(-1, 1)
uniq, inv, cnts = np.unique(pair_id, return_inverse=True, return_counts=True)
inv_t = torch.from_numpy(inv)
psum = torch.zeros(len(uniq), D); psum.index_add_(0, inv_t, torch.from_numpy(mo_flat))
pnum = torch.zeros(len(uniq), D); pnum.index_add_(0, inv_t, torch.from_numpy(num_flat))
psum = psum.numpy(); pnum = pnum.numpy()
PMIN = 3
pkeep = cnts >= PMIN
pair_mean = {int(u): psum[j]/cnts[j] for j, u in enumerate(uniq) if pkeep[j]}
pair_nmean = {int(u): pnum[j]/cnts[j] for j, u in enumerate(uniq) if pkeep[j]}
print(f"pair table: {int(pkeep.sum())} pairs with count >= {PMIN} "
      f"(of {len(uniq)} unique train pairs)", flush=True)
del mo_flat, num_flat, psum, pnum, MO_TRAIN, RHO2_TRAIN

# precompute held surrogates (S,T,D) fp32 CPU
held_np = HELD.cpu().numpy()
rho2h = RHO2_HELD0.numpy()                     # (S,T)
SUR = {}
SUR['tok'] = TT[held_np]                                        # (S,T,D)
SUR['tok_gauge'] = NT_[held_np]/rho2h[..., None] + W0[3].cpu().numpy()[None, None]
sur_pair = SUR['tok'].copy(); sur_pairg = SUR['tok_gauge'].copy()
cov = np.zeros((S_, T_), bool)
for s in range(S_):
    for t in range(1, T_):
        pid = int(held_np[s, t-1])*V + int(held_np[s, t])
        if pid in pair_mean:
            sur_pair[s, t] = pair_mean[pid]
            sur_pairg[s, t] = pair_nmean[pid]/rho2h[s, t] + W0[3].cpu().numpy()
            cov[s, t] = True
SUR['pair'] = sur_pair; SUR['pair_gauge'] = sur_pairg
SUR['floor'] = np.broadcast_to(MO_MEAN[0].cpu().numpy()[None], (S_, T_, D)).copy()
tok_seen_frac = float(seen.cpu().numpy()[held_np].mean())
pair_cov_frac = float(cov[:, 1:].mean())
print(f"held coverage: token-seen {tok_seen_frac:.4f}, pair-covered {pair_cov_frac:.4f}", flush=True)

print("ATTACK 2 eval: 5 surrogate substitutions ...", flush=True)
raw2 = {}
for name in ('floor', 'tok', 'pair', 'tok_gauge', 'pair_gauge'):
    SURt = torch.from_numpy(np.ascontiguousarray(SUR[name]))
    out = []
    for i in range(0, S_, B0):
        sub = SURt[i:i+B0].to(DEV)
        out.append(fwd_rt(HELD[i:i+B0], config={'kind': 'sub0', 'sub': sub}).cpu())
    mn, se = dstat(torch.cat(out, 0))
    raw2[name] = {'dCE': round(mn, 4), 'SE': round(se, 5)}
    print(f"  {name:12s} dCE {mn:+.4f} +- {se:.5f}", flush=True)

floor_d = raw2['floor']['dCE']
gate2 = abs(floor_d - CENSUS[0]) < 0.02
attack2 = {
 'method': 'replace block 0 mlp output with train-built conditional-mean tables '
           '(train FW[0:448,:128], held FW[448:600,:128], zero overlap); floor gate vs census 1.2341',
 'census_floor': CENSUS[0], 'floor_gate_pass': bool(gate2),
 'held_token_seen_in_train_fraction': round(tok_seen_frac, 4),
 'held_pair_covered_fraction': round(pair_cov_frac, 4),
 'pair_min_count': PMIN,
 'raw': raw2,
 'recovered_fraction_of_floor': {
     name: round(1 - raw2[name]['dCE']/floor_d, 4) for name in ('tok', 'pair', 'tok_gauge', 'pair_gauge')},
}
assert gate2, f"ATTACK 2 floor gate FAILED: {floor_d} vs census {CENSUS[0]}"
print(json.dumps(attack2['recovered_fraction_of_floor'], indent=1), flush=True)

res['attack2_token_table_causal'] = attack2
json.dump(res, open(OUT, 'w'), indent=1)
print(f"Saved {OUT}", flush=True)
print("QK REDTEAM ARCS (script 1) DONE", flush=True)
