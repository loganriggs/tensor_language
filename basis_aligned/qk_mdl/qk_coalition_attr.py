"""COALITION red-team of the §80/§81 "MLP1 hub is causally unattributable in PARTS"
claim -- SCRIPT 1: cheap proxy screen + candidate-coalition construction (NO extra
full-slice forwards beyond the two PASS-A/gradient passes).

§80 tested SINGLE dictionary features (0/32 load-bearing) and the FULL live set (all
1011 features jointly mean-ablated capture only ~2.01% of the 5.574-nat full-MLP1
knockout). §80's cumulative curve ablated features in activation-mass rank order and
never reached 50%. UNTESTED granularity: COALITIONS -- the §61 greedy-joint-ablation
move applied at feature granularity. Does some discovered GROUP of features carry a
substantial chunk of the hub's causal effect?

This script builds candidate coalitions cheaply from the activation cache + decoder:
  (a) FIRST-ORDER ATTRIBUTION (gradient screen): one grad-enabled pass computes
      dCE/dmo at LI; per-feature first-order ablation effect is
      -sum_{b,t} dev_i(b,t) * <dCE/dmo(b,t), decoder_dir_i>. The §61 signed screen:
      lets us pick features whose ablation *raises* CE and drop counteracting ones.
  (b) DEVIATION-ENERGY rank (mean_{b,t} dev_i^2 * ||decoder_dir_i||^2).
  (c) CO-ACTIVATION family: seed = highest-energy feature; grow by correlation of the
      per-position activation pattern (features that FIRE TOGETHER).
  (d) DECODER-DIRECTION family: seed = highest-energy; grow by |cosine| of decoder
      directions (features spanning a COMMON SUBSPACE).
  (e) SVD-SUBSPACE-aligned: top-28 SVD directions of mo carry ~80% (§73); take the
      SAE features whose decoder dir is most aligned with that 28-dim subspace (tests
      whether SAE features FACTOR the known-sufficient subspace).
  (f) RANDOM same-size controls (the §61 control).

Saves candidate coalitions (+ screen arrays) to qk_coalition_cands.npz for SCRIPT 2 to
ablate jointly. FORWARD + encode() + per-position mean convention COPIED VERBATIM from
qk_sae_moredata_2.py. Held-back canonical FW[448:600,:128].
"""
import json, sys, math, time, subprocess
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0); np.random.seed(0)
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
tok = AutoTokenizer.from_pretrained('gpt2')
LI = 1

FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
SEQL = 128
HELD = FINEWEB[448:600, :SEQL].to(DEV)
NHELD = HELD.shape[0]
BATCH = 6

_special = {tok.eos_token_id}
for _t in range(V):
    if '�' in tok.decode([_t]): _special.add(_t)
SPECIAL = np.array(sorted(_special))

# ---------------- load dictionary (10x-data best-VAL SAE) ----------------
Z = np.load(f'{QK}/qk_sae_moredata.npz')
W_dec = torch.from_numpy(Z['W_dec']).to(DEV)
W_enc = torch.from_numpy(Z['W_enc']).to(DEV)
b_enc = torch.from_numpy(Z['b_enc']).to(DEV)
b_dec = torch.from_numpy(Z['b_dec']).to(DEV)
MU = torch.from_numpy(Z['MU']).to(DEV)
SCALE = float(Z['SCALE'])
NFEAT = int(Z['NFEAT']); K = int(Z['K'])
TOP_FEATS = [int(j) for j in Z['top_feats']]
LIVE = np.array([int(j) for j in Z['live_feats']])          # (1011,) live feature ids
NLIVE = len(LIVE)
print(f"loaded 10x-data dictionary NFEAT={NFEAT} k={K}; {NLIVE} live features", flush=True)

DEC_ORIG = (W_dec.T / SCALE).contiguous()                   # (NFEAT,D) decoder dir in mo space
DEC_LIVE = DEC_ORIG[LIVE]                                    # (NLIVE,D)
dec_norm_live = DEC_LIVE.norm(dim=1)                         # (NLIVE,)

def encode(mo):
    pre = F.relu(((mo - MU) * SCALE - b_dec) @ W_enc.T + b_enc)
    vals, idx = pre.topk(K, dim=-1)
    f = torch.zeros_like(pre).scatter_(-1, idx, vals)
    return f

# ---------------- forward (VERBATIM) collecting mo + f ----------------
@torch.no_grad()
def forward_collect(idx):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    fcollect = None; mo_at = None
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
        if li == LI:
            fcollect = encode(mo); mo_at = mo
        x = x + mo
    return fcollect, mo_at

# ---------------- grad-enabled forward: dCE/dmo at LI ----------------
# Everything up to mo is computed under no_grad; mo is detached into a leaf, re-added,
# and grad flows only through the tail (layers LI+1.. and lm_head) -> memory cheap.
def grad_forward(idx, tgt):
    B, T = idx.shape
    # ---- pre-mo (up to and including LI's mo) under no_grad ----
    with torch.no_grad():
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
        mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        mo_leaf = None; x_before = None
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
            if li == LI:
                mo_leaf = mo.detach().clone(); x_before = x.detach(); v1c = v1.detach(); x0c = x0.detach()
                break
            x = x + mo
    # ---- tail WITH grad: only through mo_leaf ----
    mo_leaf.requires_grad_(True)
    x = x_before + mo_leaf
    for lj in range(LI+1, NL):
        blk2 = m.transformer.h[lj]; xg = blk2.lambdas[0]*x + blk2.lambdas[1]*x0c; a2 = blk2.attn
        hc = F.rms_norm(xg, (D,))
        def qk2(lin): z = F.rms_norm(lin(hc).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v_ = a2.c_v(hc).view(B, T, NH, HD)
        v_ = (1-a2.lamb)*v_ + a2.lamb*v1c.view_as(v_)
        q_, k_, q2_, k2_ = qk2(a2.c_q), qk2(a2.c_k), qk2(a2.c_q2), qk2(a2.c_k2)
        s1_ = torch.einsum('bqhd,bkhd->bhqk', q_, k_)/HD; s2_ = torch.einsum('bqhd,bkhd->bhqk', q2_, k2_)/HD
        pat_ = (s1_*s2_).masked_fill(~mask, 0.0)
        yh4_ = torch.einsum('bhqk,bkhd->bqhd', pat_, v_)
        xg = xg + a2.c_proj(yh4_.reshape(B, T, -1))
        xg = xg + blk2.mlp(F.rms_norm(xg, (D,)))
        x = xg
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    logp = F.log_softmax(logits[:, :SEQL-1], dim=-1)
    ce = -logp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1)     # (B,T-1)
    ce.sum().backward()
    return mo_leaf.grad.detach()                                    # (B,T,D) = dCE/dmo

# =====================================================================================
# PASS A: activations over live features + per-position feature means (all NFEAT) + mo
# covariance (for SVD subspace).  ONE pass.
# =====================================================================================
print("PASS A: activations + per-position means + mo covariance ...", flush=True)
FSUM = torch.zeros(SEQL, NFEAT, device=DEV)                  # per-position mean over ALL feats
act_live = np.zeros((NHELD, SEQL, NLIVE), np.float32)        # live activations cache
MO_SUM = torch.zeros(D, device=DEV); MO_GRAM = torch.zeros(D, D, device=DEV); MO_N = 0
for i in range(0, NHELD, BATCH):
    f, mo = forward_collect(HELD[i:i+BATCH])
    FSUM += f.sum(0)
    b = f.shape[0]
    act_live[i:i+b] = f[:, :, LIVE].cpu().numpy()
    mof = mo.reshape(-1, D)
    MO_SUM += mof.sum(0); MO_GRAM += mof.T @ mof; MO_N += mof.shape[0]
FMEAN = (FSUM / NHELD)                                       # (SEQL,NFEAT)
FMEAN_live = FMEAN[:, LIVE].cpu().numpy()                    # (SEQL,NLIVE)
del FSUM
print("PASS A done.", flush=True)

# ---- mo covariance -> top-28 SVD subspace (§73) ----
mo_mean = (MO_SUM / MO_N)
COV = MO_GRAM / MO_N - torch.outer(mo_mean, mo_mean)
COV = 0.5*(COV + COV.T)
evals, evecs = torch.linalg.eigh(COV)                       # ascending
U28 = evecs[:, -28:].contiguous()                          # (D,28) top-28 subspace
# alignment of each live decoder dir with the 28-dim subspace
proj = DEC_LIVE @ U28                                       # (NLIVE,28)
align_frac = (proj.norm(dim=1) / (dec_norm_live + 1e-9)).cpu().numpy()   # 0..1 fraction of dec norm in subspace
print(f"top-28 SVD subspace captures {(evals[-28:].sum()/evals.clamp(min=0).sum()).item()*100:.1f}% of mo variance", flush=True)

# =====================================================================================
# deviation energy + first-order gradient attribution
# =====================================================================================
# deviation cache: dev_live[b,t,l] = act_live - FMEAN_live
dev_live = act_live - FMEAN_live[None, :, :]               # (NHELD,SEQL,NLIVE)
# valid-next mask (exclude last position)
pos_t = np.tile(np.arange(SEQL), NHELD).reshape(NHELD, SEQL)
valid_next = pos_t < (SEQL - 1)
vmask_np = valid_next[:, :]                                 # positions 0..SEQL-2 predict next
# energy per live feature (over valid positions), weighted by decoder norm^2
vm3 = vmask_np[:, :, None]
energy = ((dev_live**2) * vm3).sum(axis=(0, 1)) / max(1, vm3.sum()) * (dec_norm_live.cpu().numpy()**2)

print("gradient pass: first-order attribution -sum dev_i <dCE/dmo, dec_i> ...", flush=True)
attr = np.zeros(NLIVE, np.float64)                          # signed first-order dCE for ablating each feat
tgt_all = HELD
for i in range(0, NHELD, BATCH):
    sb = slice(i, min(i+BATCH, NHELD))
    g = grad_forward(HELD[sb], HELD[sb])                    # (b,T,D) dCE/dmo (over T-1 valid; last col ~0)
    b = g.shape[0]
    gdot = g @ DEC_LIVE.T                                   # (b,T,NLIVE) = <grad, dec_i>
    devb = torch.from_numpy(dev_live[sb]).to(DEV)           # (b,T,NLIVE)
    vm = torch.from_numpy(vmask_np[sb]).to(DEV).unsqueeze(-1)
    contrib = -(devb * gdot * vm).sum(dim=(0, 1))          # (NLIVE,) first-order dCE contribution
    attr += contrib.double().cpu().numpy()
    del g, gdot, devb
# per-position average (match dCE-per-valid-position currency): divide by n valid positions
n_valid = int(vmask_np.sum())
attr_per_pos = attr / n_valid
print(f"first-order attribution computed; sum(positive)/full ~ {attr_per_pos[attr_per_pos>0].sum():.4f} nats (proxy)", flush=True)

# =====================================================================================
# BUILD CANDIDATE COALITIONS  (feature ids are ORIGINAL NFEAT ids)
# =====================================================================================
SIZES = [8, 32, 128, 512]
rng = np.random.default_rng(0)
cands = {}   # name -> list of original feature ids

def live_ids(local_idx):
    return [int(LIVE[j]) for j in local_idx]

# (a) ATTRIBUTION-top: features with largest POSITIVE first-order attribution (ablation raises CE)
order_attr = np.argsort(-attr_per_pos)
for S in SIZES:
    cands[f'attr_pos_{S}'] = live_ids(order_attr[:S])
# also attribution by |value|
order_absattr = np.argsort(-np.abs(attr_per_pos))
for S in SIZES:
    cands[f'attr_abs_{S}'] = live_ids(order_absattr[:S])

# (b) ENERGY-top
order_energy = np.argsort(-energy)
for S in SIZES:
    cands[f'energy_{S}'] = live_ids(order_energy[:S])

# (c) CO-ACTIVATION family: seed = top-energy; grow by correlation of activation pattern
#     build correlation on the fly against seed's pattern, then greedily by mean corr to set.
act_flat = act_live.reshape(NHELD*SEQL, NLIVE)             # (Npos,NLIVE)
act_bin = (act_flat > 0).astype(np.float32)
# normalise columns for cosine-of-activation-pattern
col_norm = np.linalg.norm(act_flat, axis=0) + 1e-9
act_unit = act_flat / col_norm
seed_c = int(order_energy[0])
def coact_family(seed, S):
    # similarity to seed by activation cosine
    sim = act_unit.T @ act_unit[:, seed]                    # (NLIVE,)
    sim[seed] = 2.0
    idx = np.argsort(-sim)[:S]
    return idx
for S in SIZES:
    cands[f'coact_{S}'] = live_ids(coact_family(seed_c, S))

# (d) DECODER-DIRECTION family: seed = top-energy; grow by |cosine| of decoder dirs
dec_unit = (DEC_LIVE / (dec_norm_live[:, None] + 1e-9))
def decdir_family(seed, S):
    cos = (dec_unit @ dec_unit[seed]).abs().cpu().numpy()
    cos[seed] = 2.0
    idx = np.argsort(-cos)[:S]
    return idx
for S in SIZES:
    cands[f'decdir_{S}'] = live_ids(decdir_family(seed_c, S))

# (e) SVD-SUBSPACE aligned: most aligned with top-28 mo subspace
order_align = np.argsort(-align_frac)
for S in [64, 128]:
    cands[f'svd28_align_{S}'] = live_ids(order_align[:S])
# also absolute projection energy in subspace
proj_energy = (proj.norm(dim=1).cpu().numpy() ** 2) * energy / (energy + 1e-12)  # keep it aligned+energetic
order_proje = np.argsort(-(proj.norm(dim=1).cpu().numpy() * dec_norm_live.cpu().numpy()))
for S in [64, 128]:
    cands[f'svd28_projE_{S}'] = live_ids(order_proje[:S])

# (f) RANDOM same-size controls (3 seeds each)
for S in SIZES:
    for s in range(3):
        r = np.random.default_rng(100+s)
        idx = r.choice(NLIVE, size=S, replace=False)
        cands[f'random_{S}_s{s}'] = live_ids(idx)

# also the §80 top-32 (nameability) set as a coalition for direct comparison
cands['top32_nameability'] = [int(j) for j in TOP_FEATS]

print(f"built {len(cands)} candidate coalitions", flush=True)
for nm, ids in cands.items():
    print(f"  {nm:22s} n={len(ids)}", flush=True)

# =====================================================================================
# SAVE for SCRIPT 2
# =====================================================================================
np.savez(f'{QK}/qk_coalition_cands.npz',
         LIVE=LIVE, attr_per_pos=attr_per_pos, energy=energy, align_frac=align_frac,
         order_attr=order_attr, order_energy=order_energy, order_align=order_align,
         **{f'cand__{k}': np.array(v, np.int64) for k, v in cands.items()})
meta = {
    'held_slice': 'FW[448:600,:128]', 'NLIVE': int(NLIVE), 'NFEAT': int(NFEAT), 'K': int(K),
    'full_MLP1_ref_nats': 5.574, 'all_live_frac_of_full_sec80': 0.0201,
    'top28_svd_var_frac': float((evals[-28:].sum()/evals.clamp(min=0).sum()).item()),
    'cand_names': list(cands.keys()),
    'proxy_positive_attr_sum_nats': float(attr_per_pos[attr_per_pos>0].sum()),
    'n_positive_attr_features': int((attr_per_pos > 0).sum()),
}
json.dump(meta, open(f'{QK}/qk_coalition_cands_meta.json', 'w'), indent=2)
print("\nSaved qk_coalition_cands.npz + meta. SCRIPT 1 DONE", flush=True)
