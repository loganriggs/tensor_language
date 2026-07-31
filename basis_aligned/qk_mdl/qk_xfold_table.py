"""CROSS-LAYER COMPOSITION experiment 2 -- TOKEN-TABLE FOLD: block 0 folded into block 1's bilinear.

Block 0's output is causally ~85.5% a token-conditional table T[token] (section 101). Substituting T
symbolically into block 1's bilinear feed-forward: with the token-determined part of block 1's input
  s(t) = lamE * rms_norm(wte(t)) + lam0 * T[t]          (embedding stream exact + block-0 table)
and the contextual rest
  r    = lam0 * attn0_out + attn1_out                    (the two attention streams)
the exact input is x_pre1 = s(t) + r + dM with dM = lam0*(mo0 - T[token]) the table residual.
Folding s(t) into one side of the bilinear numerator B(a,b) = ((L a) * (R b)) @ Down^T gives a
TOKEN-INDEXED FAMILY OF AFFINE MAPS on the rest of the stream:
  folded_t(r)  = [B(s,s) + B(s,r) + B(r,s)] / rho2 + bias         (linear in r, indexed by token)
versus the token-blind linear approximation (section-88-style linearization around the mean input):
  blind(x)     = [B(mu,x) + B(x,mu) - B(mu,mu)] / rho2 + bias     (mu = train global mean of x_pre1)
plus references: folded_quad = folded + B(r,r)/rho2 (exact except the table residual dM) and
const_t = B(s,s)/rho2 + bias (token table alone, no contextual linear part).
All predictions share the ACTUAL gauge rho2 from the true x_pre1 (the gauge is modulatory, carries
0.13% of function, section 86 red-team) -- stated, shared across all comparisons.

TESTS: (a) EXPLICITNESS -- fraction of block 1's held output variance captured (global and
per-position centering; also restricted to top-200 held-frequency tokens); (b) CAUSAL -- substitute
block 1's output with each prediction on held data, delta cross-entropy vs the mean-ablation floor
(5.5744); (c) CONDITIONALITY READOUT -- per-token effective linear maps M_t = Down @ (diag(L s_t) R
+ diag(R s_t) L) for the top-200 held-frequency tokens: pairwise matrix cosines, and for maximally
different tokens the top amplified output directions (input-covariance weighted) read through the
unembedding. (Mediation caveat: block 1's output is largely consumed by later squares, sections
96-99; the unembed readout is an interpretive summary, so noted.)

TABLE + mu + C_r fitted on TRAIN FW[0:256,:128]; ALL causal/explicitness numbers on held
FW[448:600,:128] with paired standard errors. Machinery verbatim from qk_hub_streampairs.py
(streams, lamE/lam0, forward) and qk_redteam_arcs.py (table build). Batch 4, <4GB."""
import json, sys, time, subprocess
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
from transformers import AutoTokenizer
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/qk_xfold_table.json'

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
TRAIN = FW[0:256, :128].to(DEV)
HELD = FW[448:600, :128].to(DEV)
B0 = 4; LI = 1
STR, TTR = TRAIN.shape; S_, T_ = HELD.shape
tokzr = AutoTokenizer.from_pretrained('gpt2')
b0, b1 = m.transformer.h[0], m.transformer.h[1]
L1 = b1.mlp.Left.weight.detach().float(); R1 = b1.mlp.Right.weight.detach().float()
D1w = b1.mlp.Down.weight.detach().float(); bias1 = b1.mlp.Down_bias.detach().float()
lamE = (b1.lambdas[0] * (b0.lambdas[0] + b0.lambdas[1]) + b1.lambdas[1]).item()
lam0 = b1.lambdas[0].item()
print(f"bilin18 D={D} hidden={L1.shape[0]} lamE={lamE:.4f} lam0={lam0:.4f}", flush=True)

def Bnum(a, b):
    """bilinear numerator ((L a)*(R b)) @ Down^T -- shapes (...,D)->(...,D)"""
    return ((a @ L1.T) * (b @ R1.T)) @ D1w.T

@torch.no_grad()
def fwd_to1(idx):
    """Truncated forward through layer 1's mlp (VERBATIM skeleton from qk_hub_streampairs).
    Returns dict with mo0, a0c (attn0 out), aout1, x_pre1, mo1."""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    out = {'x0': x0}
    for li in range(2):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(l): z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        aout = a.c_proj(yh.reshape(B, T, -1)); x = x + aout
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if li == 0: out['a0c'] = aout; out['mo0'] = mo
        if li == 1: out['aout1'] = aout; out['x_pre1'] = x; out['mo1'] = mo
        x = x + mo
    return out

# =====================================================================================
# PASS T (TRAIN): token table of mo0, global mean mu of x_pre1, rest covariance C_r
# =====================================================================================
print("PASS T: train table + mu + C_r ...", flush=True)
tok_sum = torch.zeros(V, D, device=DEV); tok_cnt = torch.zeros(V, device=DEV)
mu_sum = torch.zeros(D, device=DEV)
r_sum = torch.zeros(D, device=DEV); r_outer = torch.zeros(D, D, device=DEV)
npos = 0
for i in range(0, STR, B0):
    o = fwd_to1(TRAIN[i:i+B0])
    fl = TRAIN[i:i+B0].reshape(-1)
    tok_sum.index_add_(0, fl, o['mo0'].reshape(-1, D).float())
    tok_cnt.index_add_(0, fl, torch.ones_like(fl, dtype=torch.float))
    mu_sum += o['x_pre1'].reshape(-1, D).float().sum(0)
    rv = (lam0*o['a0c'] + o['aout1']).reshape(-1, D).float()
    r_sum += rv.sum(0); r_outer += rv.T @ rv
    npos += rv.shape[0]
GLOB = tok_sum.sum(0)/tok_cnt.sum()
seen = tok_cnt > 0
TT = torch.where(seen[:, None], tok_sum/tok_cnt.clamp_min(1)[:, None], GLOB[None])   # (V,D)
MU = mu_sum/npos
r_mean = r_sum/npos
C_r = r_outer/npos - torch.outer(r_mean, r_mean)
print(f"train tokens seen {int(seen.sum())}; npos {npos}", flush=True)
del tok_sum, r_outer

# s(t) for every token: lamE*rms(wte) + lam0*T[t]
WTE_N = F.rms_norm(m.transformer.wte.weight.detach().float(), (D,))
S_ALL = lamE*WTE_N + lam0*TT                                                        # (V,D)

# =====================================================================================
# PASS G (HELD, truncated): decomposition gate + explicitness (FVU) of the four predictors
# =====================================================================================
print("PASS G: held gates + explicitness ...", flush=True)
held_np = HELD.cpu().numpy()
cnts = np.bincount(held_np.reshape(-1), minlength=V)
TOP200 = np.argsort(-cnts)[:200]
top200_set = torch.zeros(V, dtype=torch.bool, device=DEV); top200_set[torch.from_numpy(TOP200.copy()).to(DEV)] = True
tok_seen_frac = float(seen[HELD.reshape(-1)].float().mean())
print(f"held token-seen-in-train fraction {tok_seen_frac:.4f}; "
      f"top200 coverage of held positions {float(top200_set[HELD.reshape(-1)].float().mean()):.4f}", flush=True)

# per-position held mean of mo1 (for centering + floor) -- accumulate here
mo1_sum = torch.zeros(T_, D, device=DEV)
acc = {k: 0.0 for k in ['err_folded', 'err_blind', 'err_quad', 'err_const', 'den_glob', 'den_pos',
                        'err_folded_200', 'err_blind_200', 'den_glob_200']}
gate_stream = 0.0; gate_recon = 0.0
preds_names = ['folded', 'blind', 'quad', 'const']
# first sub-pass: mo1 means
for i in range(0, S_, B0):
    o = fwd_to1(HELD[i:i+B0]); mo1_sum += o['mo1'].float().sum(0)
MO1_PMEAN = mo1_sum/S_                       # (T,D) per-position mean
MO1_GMEAN = MO1_PMEAN.mean(0)                # (D,)

for i in range(0, S_, B0):
    idx = HELD[i:i+B0]
    o = fwd_to1(idx)
    x = o['x_pre1'].float(); mo1 = o['mo1'].float()
    rho2 = x.pow(2).sum(-1, keepdim=True)/D
    s = S_ALL[idx]                                        # (B,T,D)
    r = (lam0*o['a0c'] + o['aout1']).float()
    dM = lam0*(o['mo0'].float() - TT[idx])
    gate_stream = max(gate_stream, float(((s + r + dM - x).norm(dim=-1)/x.norm(dim=-1).clamp_min(1e-8)).max()))
    recon = Bnum(x, x)/rho2 + bias1
    gate_recon = max(gate_recon, float(((recon - mo1).norm(dim=-1)/mo1.norm(dim=-1).clamp_min(1e-8)).max()))
    p = {}
    p['folded'] = (Bnum(s, s) + Bnum(s, r) + Bnum(r, s))/rho2 + bias1
    p['blind'] = (Bnum(MU[None, None].expand_as(x), x) + Bnum(x, MU[None, None].expand_as(x))
                  - Bnum(MU, MU)[None, None])/rho2 + bias1
    p['quad'] = p['folded'] + Bnum(r, r)/rho2
    p['const'] = Bnum(s, s)/rho2 + bias1
    for nm in preds_names:
        acc[f'err_{nm}'] = acc.get(f'err_{nm}', 0.0) + float((mo1 - p[nm]).pow(2).sum())
    acc['den_glob'] += float((mo1 - MO1_GMEAN[None, None]).pow(2).sum())
    acc['den_pos'] += float((mo1 - MO1_PMEAN[None]).pow(2).sum())
    m200 = top200_set[idx]                                # (B,T)
    if m200.any():
        acc['err_folded_200'] += float((mo1 - p['folded'])[m200].pow(2).sum())
        acc['err_blind_200'] += float((mo1 - p['blind'])[m200].pow(2).sum())
        acc['den_glob_200'] += float((mo1 - MO1_GMEAN[None, None])[m200].pow(2).sum())
print(f"GATE stream-sum rel err {gate_stream:.2e} | bilinear recon rel err {gate_recon:.2e}", flush=True)
assert gate_stream < 1e-4 and gate_recon < 1e-4, "fold gate FAILED"
explicit = {
    'R2_global_centering': {nm: round(1 - acc[f'err_{nm}']/acc['den_glob'], 4) for nm in preds_names},
    'R2_perpos_centering': {nm: round(1 - acc[f'err_{nm}']/acc['den_pos'], 4) for nm in preds_names},
    'R2_top200_positions': {'folded': round(1 - acc['err_folded_200']/acc['den_glob_200'], 4),
                            'blind': round(1 - acc['err_blind_200']/acc['den_glob_200'], 4)}}
print("EXPLICITNESS:", json.dumps(explicit), flush=True)

# =====================================================================================
# PASS C (HELD, full forward): causal substitution of mo1
# =====================================================================================
@torch.no_grad()
def fwd_full(idx, sub=None):
    """Full forward (VERBATIM qk_hub_streampairs skeleton); sub in
    {None,'floor','folded','blind','quad','const'} replaces mo1."""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    a0c = None
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(l): z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        aout = a.c_proj(yh.reshape(B, T, -1)); x = x + aout
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if li == 0: a0c, mo0 = aout, mo
        if li == LI and sub is not None:
            if sub == 'floor':
                mo = MO1_PMEAN.unsqueeze(0).expand(B, -1, -1).to(x.dtype)
            else:
                xf = x.float(); rho2 = xf.pow(2).sum(-1, keepdim=True)/D
                s = S_ALL[idx]; r = (lam0*a0c + aout).float()
                if sub == 'folded': num = Bnum(s, s) + Bnum(s, r) + Bnum(r, s)
                elif sub == 'quad': num = Bnum(s, s) + Bnum(s, r) + Bnum(r, s) + Bnum(r, r)
                elif sub == 'const': num = Bnum(s, s)
                elif sub == 'blind':
                    num = (Bnum(MU[None, None].expand_as(xf), xf) + Bnum(xf, MU[None, None].expand_as(xf))
                           - Bnum(MU, MU)[None, None])
                mo = (num/rho2 + bias1).to(x.dtype)
        x = x + mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1), reduction='none').view(B, T-1)
    return ce

print("PASS C: causal substitutions ...", flush=True)
base = torch.cat([fwd_full(HELD[i:i+B0]).cpu() for i in range(0, S_, B0)], 0)
print(f"base CE {float(base.mean()):.4f}", flush=True)

def dstat(ce):
    d = (ce - base).flatten().double(); return float(d.mean()), float(d.std()/np.sqrt(d.numel()))

causal = {}
for sub in ['floor', 'blind', 'const', 'folded', 'quad']:
    ce = torch.cat([fwd_full(HELD[i:i+B0], sub=sub).cpu() for i in range(0, S_, B0)], 0)
    mn, se = dstat(ce)
    causal[sub] = {'dCE': round(mn, 4), 'SE': round(se, 5)}
    print(f"  substitute mo1 <- {sub:8s} dCE {mn:+.4f} +- {se:.5f}", flush=True)
floor_d = causal['floor']['dCE']
causal_recovered = {nm: round(1 - causal[nm]['dCE']/floor_d, 4) for nm in ['blind', 'const', 'folded', 'quad']}
print("recovered fraction of floor:", json.dumps(causal_recovered), flush=True)

res = {'meta': {'model': 'bilin18', 'layer_folded_into': LI,
                'train': 'FW[0:256,:128]', 'held': 'FW[448:600,:128]', 'batch': B0,
                'lamE': lamE, 'lam0': lam0,
                'gauge_note': 'all predictors share the ACTUAL rho2 gauge from the true x_pre1 '
                              '(gauge carries 0.13% of function, section-86 red-team)',
                'held_token_seen_in_train': round(tok_seen_frac, 4),
                'base_ce': round(float(base.mean()), 4)},
       'gate': {'stream_sum_rel_err_max': gate_stream, 'bilinear_recon_rel_err_max': gate_recon,
                'floor_vs_census_5.5744': floor_d},
       'explicitness': explicit, 'causal': causal, 'causal_recovered_fraction': causal_recovered}
json.dump(res, open(OUT, 'w'), indent=1)

# =====================================================================================
# PASS M: per-token effective linear maps for the top-200 held-frequency tokens
# =====================================================================================
print("PASS M: per-token effective maps ...", flush=True)
NTOK = len(TOP200)
flat_store = torch.zeros(NTOK, D*D, dtype=torch.float16)
snorm = {}
for jj, t in enumerate(TOP200):
    st = S_ALL[int(t)]
    a = L1 @ st; b = R1 @ st
    Mt = D1w @ (a[:, None]*R1 + b[:, None]*L1)           # (D,D) acting on rest r
    flat_store[jj] = Mt.reshape(-1).half().cpu()
    snorm[int(t)] = float(st.norm())
    if jj % 50 == 0: print(f"  map {jj}/{NTOK}", flush=True)
# pairwise cosine via chunked Gram
G = torch.zeros(NTOK, NTOK)
CH = 25
for i0 in range(0, NTOK, CH):
    Xi = flat_store[i0:i0+CH].float().to(DEV)
    for j0 in range(i0, NTOK, CH):
        Xj = flat_store[j0:j0+CH].float().to(DEV)
        G[i0:i0+CH, j0:j0+CH] = (Xi @ Xj.T).cpu()
        del Xj
    del Xi
G = torch.maximum(G, G.T)
nrm = G.diagonal().clamp_min(1e-12).sqrt()
COS = G/(nrm[:, None]*nrm[None, :])
iu = torch.triu_indices(NTOK, NTOK, offset=1)
cos_off = COS[iu[0], iu[1]]
print(f"pairwise map cosine: median {float(cos_off.median()):.4f} min {float(cos_off.min()):.4f} "
      f"max {float(cos_off.max()):.4f}", flush=True)

def tname(t): return repr(tokzr.decode([int(t)]))

# most-different pairs among tokens with decent frequency
pairs_sorted = torch.argsort(cos_off)
diff_pairs = []
for pidx in pairs_sorted[:2000].tolist():
    i, j = int(iu[0, pidx]), int(iu[1, pidx])
    diff_pairs.append((float(cos_off[pidx]), int(TOP200[i]), int(TOP200[j])))
    if len(diff_pairs) >= 12: break

# input-covariance weighted top output directions, read through the unembedding
Wu = m.lm_head.weight.detach().float()
evals, evecs = torch.linalg.eigh(C_r.double())
Lc = (evecs * evals.clamp_min(1e-6).sqrt()[None, :]).float()     # C_r^(1/2)
def amplify(t, ktop=2, nvocab=8):
    st = S_ALL[int(t)]
    a = L1 @ st; b = R1 @ st
    Mt = D1w @ (a[:, None]*R1 + b[:, None]*L1)
    W = Mt @ Lc
    U, Sv, _ = torch.svd_lowrank(W, q=8)
    outdirs = []
    for k in range(ktop):
        u = U[:, k]
        lg = Wu @ u
        top = torch.argsort(lg, descending=True)[:nvocab]
        bot = torch.argsort(lg)[:nvocab]
        outdirs.append({'singular_value': float(Sv[k]),
                        'boosts': [tname(v) for v in top], 'suppresses': [tname(v) for v in bot]})
    return outdirs

examples = []
seen_toks = set()
for cosv, ta, tb in diff_pairs[:6]:
    ex = {'pair': [tname(ta), tname(tb)], 'map_cosine': round(cosv, 4),
          'held_counts': [int(cnts[ta]), int(cnts[tb])]}
    for lbl, t in [('A', ta), ('B', tb)]:
        ex[f'token_{lbl}_amplifies'] = amplify(t)
    examples.append(ex)
    seen_toks.update([ta, tb])
    print(json.dumps(ex, indent=1), flush=True)
# also a same-class high-cosine reference pair for contrast
best = torch.argsort(cos_off, descending=True)[0]
i, j = int(iu[0, best]), int(iu[1, best])
ref = {'pair': [tname(TOP200[i]), tname(TOP200[j])], 'map_cosine': round(float(cos_off[best]), 4),
       'note': 'highest-cosine pair for contrast'}
print(json.dumps(ref), flush=True)

res['effective_maps'] = {
    'n_tokens': NTOK,
    'pairwise_cosine': {'median': round(float(cos_off.median()), 4), 'min': round(float(cos_off.min()), 4),
                        'p05': round(float(cos_off.quantile(0.05)), 4), 'p95': round(float(cos_off.quantile(0.95)), 4)},
    'most_different_pairs': [{'cos': round(c, 4), 'a': tname(a), 'b': tname(b)} for c, a, b in diff_pairs],
    'highest_cosine_pair': ref,
    'examples': examples,
    'mediation_caveat': 'block 1 output is largely consumed by later squaring stages (sections 96-99); '
                        'the unembedding readout of amplified directions is an interpretive summary'}
json.dump(res, open(OUT, 'w'), indent=1)
print("QK XFOLD TABLE DONE", flush=True)
