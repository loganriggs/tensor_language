"""COMPRESSED FUNCTIONAL MODEL of the mid-stack distributed region (blocks 5-11) -- script 2 of 2.

EXPERIMENT 2 -- THE CONDITIONING-VARIABLE HUNT (what is the mid-stack's "current token"?).
For the top 16 principal directions of the joint write W (script 1 basis), variance explained by
candidate conditioning variables on HELD data (group-R^2 method VERBATIM qk_arc_mlp0.py group_r2 +
shuffle_control):
  (a) current-token identity (min_count 5 + shuffled-label control) -- the block-0 contrast
  (b) positional state: position-in-sequence buckets (t//16; NOTE: exactly zero on the deviation by
      construction -- the per-position mean carries it; also reported on the RAW write coords) +
      distance-since-newline (capped 32, 33 = no newline yet)
  (c) topic proxy: k-means clusters (K=8,16, fit on TRAIN per-sequence mean layer-4 residual, held
      sequences assigned to nearest centroid; sequence-level shuffle control)
  (d) six-way next-token category (VERBATIM qk_category_engine.py CAT classes)
Concrete examples: for the 3 most-explained message dims, held snippets at extreme activations
(context + what the dim pushes through the readout = top lm_head tokens of the direction).

EXPERIMENT 3 -- THE BESPOKE SURROGATE (redundancy as a fundamental object).
message m(x) = top-k projection of the live joint write (best k from experiment 1 = smallest swept k
with >=90% recovery). Surrogate write = DECODER(m):
  PCA decoder (G=identity)  -> must reproduce experiment 1's svd_keep number exactly.
  REPETITION code, factor r in {1,4,16}: r copies of m/r, copy i carried in a random orthogonal
  k-frame R_i of the code space (code space = r*k coordinates); decoding = masked least squares,
  whose composite map is G = projection onto range(sum_i R_i^T M_i R_i). Deleting a random half of
  the code coordinates: for r=1 G becomes a rank-~k/2 projector (half the message lost); for r>=4
  the surviving copies still determine m exactly (G=I) -- the repetition-code signature.
  Real-region comparison: keep a random 576-of-1152 subspace of every block's deviation (the section
  83 half-deletion, fresh seeds) vs the surrogate's half-deletions.
Report: surrogate delta cross-entropy vs the joint floor at each r; half-deletion robustness of
surrogate vs real region.

Held FW[448:600,:128], paired standard errors, batch 6, footprint <4GB, GPU guard.
Extends qk_msg_bottleneck.json (exp2_* and exp3_* keys)."""
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

res = json.load(open(OUT))
ST = torch.load(STATE, weights_only=False)
U = ST['U']; EIG = ST['eigval']; M_HELD = ST['M_HELD']            # (S,T,144) deviation coords
WBAR_HELD = ST['WBAR_HELD']                                       # (T,D)
L4_TRAIN = ST['L4_TRAIN'].double().numpy(); L4_HELD = ST['L4_HELD'].double().numpy()
FW = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
HELD = FW[448:600, :128]; held_np = HELD.numpy()
S_, T_ = HELD.shape; N = S_*T_; B0 = 6
D = U.shape[0]

# =====================================================================================
# EXPERIMENT 2
# =====================================================================================
print("EXP 2: conditioning-variable hunt on the top-16 message dims ...", flush=True)
M16 = M_HELD[:, :, :16].reshape(N, 16).double()
M144 = M_HELD.reshape(N, 144).double()
RAW16 = (M_HELD[:, :, :16] + (WBAR_HELD @ U[:, :16]).unsqueeze(0)).reshape(N, 16).double()

def group_r2(X64, gids, elig, min_count):
    """VERBATIM qk_arc_mlp0.py: variance explained by group identity on positions elig &
    group-count>=min_count. X64 (N,d) float64 cpu torch; gids (N,) int64 numpy."""
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
    r2 = 1.0 - within/max(tot, 1e-12)
    return r2, G, cov

def shuffle_control(X64, gids, cov, seed=0):
    rng = np.random.default_rng(seed)
    g = gids.copy()
    idxc = np.where(cov)[0]
    g[idxc] = g[rng.permutation(idxc)]
    r2, _, _ = group_r2(X64, g, cov, 1)
    return r2

# ---- candidate variables ----
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('gpt2')
import string as _string
_P = set(_string.punctuation)
FUNC = {'the','of','and','to','a','in','is','that','it','for','was','as','with','on','be','at','by','this','are','from','or','an','but','not','which'}
V = 50257
CAT = np.full(max(50304, int(held_np.max())+1), 5, dtype=np.int64)
NL_IDS = set()
for i in range(V):
    s = tok.convert_ids_to_tokens(i)
    if s is None: continue
    core = s.replace('Ġ', ''); lead = s.startswith('Ġ')
    if 'Ċ' in s: NL_IDS.add(i)
    if len(core) and all(c in _P for c in core): CAT[i] = 1
    elif len(core) and all(c.isdigit() for c in core): CAT[i] = 3
    elif core.lower() in FUNC: CAT[i] = 4
    elif not lead and len(core) and core[0].isalpha() and core[0].islower(): CAT[i] = 0
    elif lead and len(core) and core[0].isupper(): CAT[i] = 2
CATNAMES = ['subword', 'punct', 'capital', 'digit', 'funcword', 'other']

flat_ids = held_np.reshape(-1)
pos_t = np.tile(np.arange(T_), S_).reshape(S_, T_)
pos_bucket = (pos_t // 16).reshape(-1)
dist_nl = np.full((S_, T_), 33, dtype=np.int64)
for s in range(S_):
    last = -1
    for t in range(T_):
        if int(held_np[s, t]) in NL_IDS: last = t
        dist_nl[s, t] = min(t - last, 32) if last >= 0 else 33
dist_nl = dist_nl.reshape(-1)
next_cat = np.full(N, -1, dtype=np.int64)
nc = CAT[held_np[:, 1:].reshape(-1)]
next_cat_2d = np.full((S_, T_), -1, dtype=np.int64); next_cat_2d[:, :-1] = nc.reshape(S_, T_-1)
next_cat = next_cat_2d.reshape(-1)

def kmeans(X, K, iters=100, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), K, replace=False)].copy()
    a = None
    for _ in range(iters):
        dist = ((X[:, None, :] - C[None])**2).sum(-1)
        a_new = dist.argmin(1)
        if a is not None and (a_new == a).all(): break
        a = a_new
        for kk in range(K):
            if (a == kk).any(): C[kk] = X[a == kk].mean(0)
    return C

topic = {}
for K in (8, 16):
    C = kmeans(L4_TRAIN, K)
    assign = ((L4_HELD[:, None, :] - C[None])**2).sum(-1).argmin(1)      # (S,)
    topic[K] = np.repeat(assign, T_)
    print(f"  topic K={K}: held cluster sizes {np.bincount(assign, minlength=K).tolist()}", flush=True)

all_elig = np.ones(N, bool)
CANDS = {
 'cur_token':    (flat_ids, all_elig, 5),
 'pos_bucket':   (pos_bucket, all_elig, 1),
 'dist_newline': (dist_nl, all_elig, 5),
 'topic8':       (topic[8], all_elig, 1),
 'topic16':      (topic[16], all_elig, 1),
 'next_cat6':    (next_cat, next_cat >= 0, 1),
}
exp2 = {'aggregate_top16': {}, 'aggregate_top144': {}, 'per_dim_top16': {}, 'controls': {},
        'raw_pos_bucket_top16': None, 'eig_share_top16': [round(float(EIG[j]/EIG.sum()), 4) for j in range(16)]}
r2raw, _, _ = group_r2(RAW16, pos_bucket, all_elig, 1)
exp2['raw_pos_bucket_top16'] = round(r2raw, 4)
print(f"  position bucket on RAW write coords (mean structure): R^2 {r2raw:.4f}", flush=True)
per_dim = {}
for name, (gids, elig, mc) in CANDS.items():
    r2a, Gn, cov = group_r2(M16, gids, elig, mc)
    r2b, _, _ = group_r2(M144, gids, elig, mc)
    exp2['aggregate_top16'][name] = {'R2': round(r2a, 4), 'n_groups': int(Gn),
                                     'covered_frac': round(float(cov.mean()), 4)}
    exp2['aggregate_top144'][name] = round(r2b, 4)
    if name in ('topic8', 'topic16'):
        # sequence-level shuffle control (per-sequence variable)
        rng = np.random.default_rng(0)
        seq_lab = gids.reshape(S_, T_)[:, 0]
        sh = np.repeat(seq_lab[rng.permutation(S_)], T_)
        ctl, _, _ = group_r2(M16, sh, elig, 1)
    else:
        ctl = shuffle_control(M16, gids, cov)
    exp2['controls'][name] = round(ctl, 4)
    pd = []
    for j in range(16):
        r2j, _, _ = group_r2(M16[:, j:j+1], gids, elig, mc)
        pd.append(round(r2j, 4))
    per_dim[name] = pd
    print(f"  {name:12s} R2(top16) {r2a:.4f} (ctl {ctl:.4f}, {Gn} groups, cov {cov.mean():.3f}) "
          f"R2(top144) {r2b:.4f}  per-dim {pd}", flush=True)
exp2['per_dim_top16'] = per_dim

# supplementary: is the current-token score carried by the <|endoftext|> boundary state?
EOT = 50256
no_eot = flat_ids != EOT
exp2['eot_fraction_of_positions'] = round(float((~no_eot).mean()), 4)
r2ne, Gne, covne = group_r2(M16, flat_ids, no_eot, 5)
ctlne = shuffle_control(M16, flat_ids, covne)
pdne = []
for j in range(16):
    r2j, _, _ = group_r2(M16[:, j:j+1], flat_ids, no_eot, 5)
    pdne.append(round(r2j, 4))
exp2['cur_token_no_eot'] = {'R2': round(r2ne, 4), 'control': round(ctlne, 4),
                            'n_groups': int(Gne), 'per_dim': pdne}
print(f"  cur_token EXCLUDING <|endoftext|> (eot = {(~no_eot).mean():.4f} of positions): "
      f"R2(top16) {r2ne:.4f} (ctl {ctlne:.4f})  per-dim {pdne}", flush=True)

# ---- concrete examples for the 3 most-explained message dims ----
SUBST = ['cur_token', 'dist_newline', 'topic8', 'topic16', 'next_cat6']
best_per_dim = [(max(per_dim[c][j] for c in SUBST), j,
                 max(SUBST, key=lambda c: per_dim[c][j])) for j in range(16)]
top3 = sorted(best_per_dim, reverse=True)[:3]
m2d = M_HELD.reshape(N, 144)
print("EXP 2 examples ...", flush=True)
m_model, _cfg = load_elriggs('bilin18')
LMW = m_model.lm_head.weight.detach().float().cpu()               # (V?, D)
examples = []
for r2v, j, cname in top3:
    col = m2d[:, j]
    logit = LMW @ U[:, j]
    topp = [tok.convert_ids_to_tokens(int(i)) for i in logit.topk(8).indices]
    topn = [tok.convert_ids_to_tokens(int(i)) for i in (-logit).topk(8).indices]
    ex = {'dim': j, 'best_candidate': cname, 'best_R2': r2v,
          'eig_share': round(float(EIG[j]/EIG.sum()), 4),
          'readout_positive': topp, 'readout_negative': topn, 'snippets': []}
    order = col.abs().argsort(descending=True)
    used_seq = set()
    for idx in order.tolist():
        s, t = idx // T_, idx % T_
        if t < 16 or s in used_seq: continue
        used_seq.add(s)
        ctx = tok.decode(held_np[s, max(0, t-28):t+1])
        ex['snippets'].append({'seq': s, 'pos': t, 'm_value': round(float(col[idx]), 2),
                               'cur_token': tok.convert_ids_to_tokens(int(held_np[s, t])),
                               'topic8': int(topic[8][idx]), 'dist_newline': int(dist_nl[idx]),
                               'context_tail': ctx[-160:]})
        if len(ex['snippets']) == 3: break
    # non-boundary extremes (excluding <|endoftext|> current tokens), labeled
    order_ne = col.clone(); order_ne[~torch.from_numpy(no_eot)] = 0.0
    used_seq = set()
    for idx in order_ne.abs().argsort(descending=True).tolist():
        s, t = idx // T_, idx % T_
        if t < 16 or s in used_seq: continue
        used_seq.add(s)
        ctx = tok.decode(held_np[s, max(0, t-28):t+1])
        ex.setdefault('snippets_no_eot', []).append(
            {'seq': s, 'pos': t, 'm_value': round(float(col[idx]), 2),
             'cur_token': tok.convert_ids_to_tokens(int(held_np[s, t])),
             'topic8': int(topic[8][idx]), 'dist_newline': int(dist_nl[idx]),
             'context_tail': ctx[-160:]})
        if len(ex['snippets_no_eot']) == 3: break
    examples.append(ex)
    print(f"  dim {j} (best {cname} R2 {r2v:.3f}) readout+ {topp[:5]} readout- {topn[:5]}", flush=True)
    for sn in ex['snippets']:
        print(f"    m={sn['m_value']:+.1f} pos {sn['pos']} cur '{sn['cur_token']}' "
              f"topic {sn['topic8']} dnl {sn['dist_newline']} | ...{sn['context_tail'][-90:]!r}", flush=True)
exp2['examples'] = examples
res['exp2_conditioning_hunt'] = exp2
json.dump(res, open(OUT, 'w'), indent=1)
print("EXP 2 saved.", flush=True)

# =====================================================================================
# EXPERIMENT 3
# =====================================================================================
BESTK = None
for kk, rec in sorted(((int(kk), rec) for kk, rec in res['exp1_joint_write_bottleneck']['svd_keep'].items())):
    if rec['recovered_fraction'] >= 0.90: BESTK = kk; break
if BESTK is None: BESTK = 576
print(f"EXP 3: bespoke surrogate at best k = {BESTK} ...", flush=True)
NH, HD = _cfg['n_head'], _cfg['n_embd'] // _cfg['n_head']
Vc = _cfg['vocab_size']; NLm = len(m_model.transformer.h)
m = m_model
BLK = list(range(5, 12)); BLKSET = set(BLK)
MEANS = {li: ST['MEANS'][li].to(DEV) for li in BLK}
HELD_DEV = HELD.to(DEV)
Uk = U[:, :BESTK].to(DEV).contiguous()

@torch.no_grad()
def fwd(idx, mode=None, P=None, G=None):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(NLm):
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
            if mode == 'floor':
                mo = MEANS[li].unsqueeze(0).to(mo.dtype).expand(B, -1, -1)
            elif mode == 'code':
                dev = mo - MEANS[li].unsqueeze(0)
                mm = dev @ P
                if G is not None: mm = mm @ G
                mo = MEANS[li].unsqueeze(0) + mm @ P.T
            elif mode == 'keep':
                dev = mo - MEANS[li].unsqueeze(0)
                mo = MEANS[li].unsqueeze(0) + (dev @ P) @ P.T
        x = x + mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, Vc).float(), idx[:, 1:].reshape(-1),
                         reduction='none').view(B, T-1)
    return ce

base = ST['base_ce']
def dstat(ce):
    d = (ce - base).flatten().double(); return float(d.mean()), float(d.std()/np.sqrt(d.numel()))
def run(mode, P=None, G=None):
    return dstat(torch.cat([fwd(HELD_DEV[i:i+B0], mode=mode, P=P, G=G).cpu()
                            for i in range(0, S_, B0)], 0))

# gates
f_mn, f_se = run('floor')
f1 = res['exp1_joint_write_bottleneck']['joint_floor']['dCE']
print(f"  floor gate: {f_mn:+.4f} (script 1: {f1:+.4f})", flush=True)
assert abs(f_mn - f1) < 0.02*abs(f1) + 0.002, "floor gate FAILED"
p_mn, p_se = run('code', P=Uk, G=None)
p1 = res['exp1_joint_write_bottleneck']['svd_keep'][str(BESTK)]['dCE']
print(f"  PCA-decoder gate (G=I): {p_mn:+.5f} (script 1 svd_{BESTK}: {p1:+.5f})", flush=True)
assert abs(p_mn - p1) < 0.02*abs(f1) + 0.002, "PCA-decoder gate FAILED"

def code_G(r, seed, delete_half):
    """Repetition code: r copies of m (scale m/r absorbed by least-squares decode), copy i in a
    random orthogonal k-frame R_i. Delete a random half of all r*k code coordinates; composite
    encode->delete->decode map G = projection onto range(sum_i R_i^T M_i R_i)."""
    k = BESTK
    gen = torch.Generator().manual_seed(7000 + 97*r + seed)
    H = torch.zeros(k, k, dtype=torch.float64)
    if delete_half:
        perm = torch.randperm(r*k, generator=gen)
        keepmask = torch.ones(r*k, dtype=torch.bool); keepmask[perm[:r*k//2]] = False
    else:
        keepmask = torch.ones(r*k, dtype=torch.bool)
    for i in range(r):
        R = torch.linalg.qr(torch.randn(k, k, generator=gen, dtype=torch.float64))[0]
        Mi = keepmask[i*k:(i+1)*k].double()
        H += R.T @ (Mi[:, None] * R)
    ev, evec = torch.linalg.eigh(H)
    keep = ev > 1e-8 * float(ev.max())
    Vk = evec[:, keep]
    G = (Vk @ Vk.T).float()
    lam = EIG[:k].double()
    retained = float((G.double()**2 * lam[None, :]).sum() / lam.sum())
    return G.to(DEV), int(keep.sum()), retained

exp3 = {'best_k': BESTK, 'joint_floor': {'dCE': round(f_mn, 4), 'SE': round(f_se, 5)},
        'pca_decoder_no_deletion': {'dCE': round(p_mn, 5), 'SE': round(p_se, 6),
            'recovered_fraction': round(1 - p_mn/f_mn, 4),
            'note': 'identical for every r when nothing is deleted: the repetition decode map is exactly the identity'},
        'half_deletion': {}, 'real_region_half_deletion': []}
for r in (1, 4, 16):
    Gc, rk, ret = code_G(r, 0, False)
    assert rk == BESTK and float((Gc.cpu() - torch.eye(BESTK)).abs().max()) < 1e-4, f"no-deletion G != I at r={r}"
    runs = []
    for seed in (0, 1, 2):
        Gc, rk, ret = code_G(r, seed, True)
        mn, se = run('code', P=Uk, G=Gc)
        runs.append({'seed': seed, 'dCE': round(mn, 5), 'SE': round(se, 6),
                     'decode_rank': rk, 'message_energy_retained': round(ret, 4)})
        print(f"  r={r:2d} half-del seed {seed}: dCE {mn:+.5f} +- {se:.6f} "
              f"(rank {rk}/{BESTK}, energy retained {ret:.4f})", flush=True)
    exp3['half_deletion'][f'r{r}'] = {
        'runs': runs, 'mean_dCE': round(float(np.mean([x['dCE'] for x in runs])), 5),
        'mean_excess_over_no_deletion': round(float(np.mean([x['dCE'] for x in runs])) - p_mn, 5)}
print("  real region: keep random 576-of-1152 subspace of every block's deviation ...", flush=True)
for seed in (0, 1):
    g = torch.Generator(device=DEV).manual_seed(200 + seed)
    Q = torch.linalg.qr(torch.randn(D, D, generator=g, device=DEV))[0][:, :D//2].contiguous()
    mn, se = run('keep', P=Q)
    exp3['real_region_half_deletion'].append({'seed': seed, 'dCE': round(mn, 5), 'SE': round(se, 6)})
    print(f"  real half-keep seed {seed}: dCE {mn:+.5f} +- {se:.6f}", flush=True)
exp3['exp1_rand576_reference'] = res['exp1_joint_write_bottleneck']['rand_keep']['576']
res['exp3_bespoke_surrogate'] = exp3
json.dump(res, open(OUT, 'w'), indent=1)
print(f"Saved {OUT}", flush=True)
print("QK MSG BOTTLENECK (script 2) DONE", flush=True)
