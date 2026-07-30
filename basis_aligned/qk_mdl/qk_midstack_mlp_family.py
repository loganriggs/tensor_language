"""Function-ledger gap: bilin18 mid-stack feed-forward blocks 4-15 have NO family assignment.
The circuit atlas (§32) pins only MLP0-3 = category-prediction engine, MLP1 = shared match/induction
hub, MLP16/17 = lexical-near-output. Blocks 4-15 are uncharacterized. Three known families:
  (1) category engine  -- extends a decodable NEXT-TOKEN-CATEGORY code (probe with category trajectory)
  (2) induction/match fabric -- feeds the two-branch MATCH (MLP1's hub role; MLP1 = known positive)
  (3) layout/lexical
For each MLP block L (4..15, plus 0-3,16,17 as reference anchors) we MEAN-ABLATE that single MLP
(in-distribution zero point -- NOT zero-ablation, which inflates loads 10-60x) and measure three axes:
  (a) marginal cross-entropy cost on held-back FW[448:600], with a PAIRED per-token standard error;
  (b) CATEGORY-code drop: fixed linear category probe fit on the INTACT final residual; the drop in
      held-back next-token-category decode accuracy when MLP L is ablated -> "builds category code?" axis;
  (c) MATCH-fabric change: the two-branch induction match rate (argmax key = correct copy key) with
      MLP L mean-ablated vs intact -> "serves the match fabric?" axis (MLP1 is the known positive).
Prints [L | mean-abl dCE +/- SE | category-decode drop | match-rate change] and saves JSON. Parent
reads it to assign each mid-stack MLP to a family (category / match-fabric / layout / diffuse).

PROVENANCE (code copied verbatim, changes noted):
  - block-loop forward + single-MLP mean-ablation hook + mlp_means(): from qk_category_engine.py
    (collect() / mlp_means()). Changed: forward also returns the final residual; MLP-means computed
    over a train slice FINEWEB[0:256] disjoint from the held-back slice.
  - CAT (6-way next-token category) construction, CATNAMES, ridge one-hot probe + argmax decode:
    from qk_category_engine.py, verbatim. Changed: the ridge solve + argmax are factored into
    fit_probe()/eval_probe() so ONE probe (fit on the intact residual) is reused across ablations
    (the arithmetic -- 50*I ridge, one_hot(.,6), argmax -- is unchanged).
  - induction repeated-prefix match-rate forward (match_mass argmax measurement) + make(): from
    qk_mlp1_role.py, verbatim, with ablate_mlp={L}. Changed: repeated-prefix eval taken from the
    held-back slice FINEWEB[448:448+48] instead of FINEWEB[100:...], so the match measurement is
    also on held-back data.
"""
import json, sys
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
tok = AutoTokenizer.from_pretrained('gpt2')
import string as _string
_P = set(_string.punctuation)
FUNC = {'the','of','and','to','a','in','is','that','it','for','was','as','with','on','be','at','by','this','are','from','or','an','but','not','which'}

# ---- 6-way next-token category (verbatim from qk_category_engine.py) ----
CAT = torch.full((V,), 5, dtype=torch.long)
for i in range(50257):
    s = tok.convert_ids_to_tokens(i)
    if s is None: continue
    core = s.replace('Ġ', ''); lead = s.startswith('Ġ')
    if len(core) and all(c in _P for c in core): CAT[i] = 1
    elif len(core) and all(c.isdigit() for c in core): CAT[i] = 3
    elif core.lower() in FUNC: CAT[i] = 4
    elif not lead and len(core) and core[0].isalpha() and core[0].islower(): CAT[i] = 0
    elif lead and len(core) and core[0].isupper(): CAT[i] = 2
CAT = CAT.to(DEV)
CATNAMES = ['subword', 'punct', 'capital', 'digit', 'funcword', 'other']

# ======================================================================
# Block-loop forward with single-MLP mean-ablation (verbatim block loop from
# qk_category_engine.collect(); modified to also return the final residual).
# ======================================================================
@torch.no_grad()
def final_resid(idx, ablate=None):
    """Return the final residual x (B,T,D) after all blocks (pre final rms_norm).
    ablate: set of MLP layer indices to mean-ablate with MLPMEAN."""
    B, T = idx.shape; x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
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
        x = x + a.c_proj(yh4.reshape(B, T, -1)); mo = blk.mlp(F.rms_norm(x, (D,)))
        if ablate is not None and li in ablate: mo = MLPMEAN[li].expand_as(mo)
        x = x + mo
    return x

# MLP means for ablation (verbatim mlp_means() from qk_category_engine.py).
# Changed: computed over a train slice FINEWEB[0:256] disjoint from held-back FW[448:600].
@torch.no_grad()
def mlp_means(idx):
    B, T = idx.shape; x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool)); mm = {}
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1)); mo = blk.mlp(F.rms_norm(x, (D,))); mm[li] = mo.mean((0, 1)); x = x + mo
    return mm
MLPMEAN = mlp_means(FINEWEB[0:256, :128].to(DEV))

# ---- chunked lm_head -> per-token CE (memory-safe) ----
@torch.no_grad()
def per_token_ce(x, tgt, chunk=2048):
    """x: (B,T,D) final residual; tgt: (B,T) targets. Returns per-token CE (N,)."""
    xf = F.rms_norm(x.reshape(-1, D), (D,)); tf = tgt.reshape(-1); ces = []
    for i in range(0, xf.shape[0], chunk):
        lg = 30*torch.tanh(m.lm_head(xf[i:i+chunk])/30).float()   # soft-cap, verbatim
        ces.append(F.cross_entropy(lg, tf[i:i+chunk], reduction='none'))
    return torch.cat(ces)

# ---- category probe: ridge one-hot fit + argmax decode (verbatim math from
# qk_category_engine.probe_acc(), factored into fit/eval so ONE probe is reused) ----
def fit_probe(Xtr, Ytr):
    Xtr = torch.cat([Xtr, torch.ones(Xtr.shape[0], 1, device=DEV)], 1).double()
    Yoh = F.one_hot(Ytr, 6).double()
    W = torch.linalg.solve(Xtr.T @ Xtr + 50*torch.eye(Xtr.shape[1], device=DEV, dtype=torch.double), Xtr.T @ Yoh)
    return W
def eval_probe(W, Xte, Yte):
    Xte = torch.cat([Xte, torch.ones(Xte.shape[0], 1, device=DEV)], 1).double()
    pred = (Xte @ W).argmax(1)
    return float((pred == Yte).float().mean())

# ======================================================================
# Induction repeated-prefix match-rate forward (verbatim from qk_mlp1_role.py:
# make() + forward()). Changed: eval slice taken from held-back FINEWEB[448:448+48].
# ======================================================================
P = 64; NSEQ = 48
pref = FINEWEB[448:448+NSEQ, 1:1+P]; EV = torch.cat([pref, pref], 1).to(DEV)

def make(short):
    m, cfg = load_elriggs(short); NH, D = cfg['n_head'], cfg['n_embd']; HD = D//NH; V = cfg['vocab_size']; NL = len(m.transformer.h)
    return dict(m=m, NH=NH, HD=HD, D=D, V=V, NL=NL, two=bool(cfg.get('bilinear_attn')) and bool(cfg.get('squared_attn')), sq=bool(cfg.get('squared_attn')))

@torch.no_grad()
def match_forward(M, idx, ablate_mlp=frozenset(), MEAN=None, collect_mean=False):
    m = M['m']; NH, HD, D, NL, V = M['NH'], M['HD'], M['D'], M['NL'], M['V']
    B, T = idx.shape; x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool)); means = {}
    # correct induction key per query: for i in [P,2P), key = i-P+1
    qidx = torch.arange(P, 2*P-1, device=DEV); kidx = qidx - P + 1
    match_mass = 0.0; nlay = 0
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        if M['two']:
            q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
            pat = (s1*s2).masked_fill(~mask, 0.0)
            patn = pat / pat.sum(-1, keepdim=True).clamp_min(1e-9)   # normalize just to read match fraction
        elif M['sq']:
            q, k = qk(a.c_q), qk(a.c_k); s = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD
            pat = s.square().masked_fill(~mask, 0.0); patn = pat/pat.sum(-1, keepdim=True).clamp_min(1e-9)
        else:
            q, k = qk(a.c_q), qk(a.c_k); s = torch.einsum('bqhd,bkhd->bhqk', q, k)/(HD**0.5)
            s = s.masked_fill(~mask, float('-inf')); patn = F.softmax(s, -1)
        # match mass: prob at correct induction key, averaged over query positions/heads/batch
        # induction-match: does query i (2nd copy) attend MOST to the correct copy key i-P+1?
        rawpat = (pat if M['two'] else patn).clone()
        rawpat = rawpat.masked_fill(~mask, float('-inf'))   # only causal positions compete for argmax
        am = rawpat[:, :, qidx, :].argmax(-1)               # (B,NH,len(qidx)) argmax key per query
        match_mass += (am == kidx.view(1, 1, -1)).float().mean().item(); nlay += 1
        realpat = pat if M['two'] else patn   # true head pattern (unnormalized for two-branch)
        yh4 = torch.einsum('bhqk,bkhd->bqhd', realpat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1)); mo = blk.mlp(F.rms_norm(x, (D,)))
        if collect_mean: means[('m', li)] = mo.mean((0, 1))
        if li in ablate_mlp: mo = MEAN[('m', li)].expand_as(mo)
        x = x + mo
    lg = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30).float()
    tgt = EV[:, 1:]; ce = F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction='none').view(B, -1)
    FIR = torch.arange(1, P-1, device=DEV); SEC = torch.arange(P, 2*P-1, device=DEV)
    adv = ce[:, FIR].mean().item() - ce[:, SEC].mean().item()
    return match_mass/nlay, adv, means

# ======================================================================
# Data slices
# ======================================================================
TRAIN = range(0, 400, 4)        # probe-fit slice (intact residual), disjoint-ish from held-back
HELD  = range(448, 600, 4)      # held-back FW[448:600]
TLEN  = 128

@torch.no_grad()
def gather_feats_ce(rng, ablate=None, want_ce=False):
    """Concatenate final-residual features (N,D) + next-token CAT labels over rng.
    If want_ce, also returns per-token CE (N,) via chunked lm_head."""
    feats, labs, ces = [], [], []
    for i in rng:
        idx = FINEWEB[i:i+4, :TLEN].to(DEV)
        x = final_resid(idx[:, :-1], ablate)
        feats.append(x.reshape(-1, D))
        labs.append(CAT[idx[:, 1:].reshape(-1)])
        if want_ce: ces.append(per_token_ce(x, idx[:, 1:]))
    F_ = torch.cat(feats); L_ = torch.cat(labs)
    C_ = torch.cat(ces) if want_ce else None
    return F_, L_, C_

# ---- fit category probe on INTACT train residual ----
Xtr, Ytr, _ = gather_feats_ce(TRAIN, ablate=None, want_ce=False)
Wprobe = fit_probe(Xtr, Ytr)
del Xtr; torch.cuda.empty_cache()

# ---- intact baselines on held-back ----
Xhe0, Yhe, Che0 = gather_feats_ce(HELD, ablate=None, want_ce=True)
cat_acc_intact = eval_probe(Wprobe, Xhe0, Yhe)
maj = torch.bincount(Yhe, minlength=6).max().item() / Yhe.numel()
ce_intact = Che0.mean().item()
del Xhe0; torch.cuda.empty_cache()

# ---- intact match rate (held-back repeated-prefix eval) ----
M18 = make('bilin18')
_, _, MMEAN = match_forward(M18, EV[:, :-1], collect_mean=True)
mm_intact, adv_intact, _ = match_forward(M18, EV[:, :-1])

print(f"held-back FW[448:600] tokens {Yhe.numel()} | class dist {[round(c/Yhe.numel(),3) for c in torch.bincount(Yhe,minlength=6).tolist()]} | majority {maj:.3f}", flush=True)
print(f"INTACT baselines: CE {ce_intact:.4f} | category-decode acc {cat_acc_intact:.4f} | match-rate {mm_intact:.4f} (adv {adv_intact:.3f})", flush=True)
print(f"\n{'L':>3} | {'mean-abl dCE':>13} | {'+/- SE':>8} | {'cat-decode drop':>15} | {'match-rate chg':>14}", flush=True)
print("-"*70, flush=True)

LAYERS = list(range(NL))  # 0..17: 4-15 targets; 0-3,16,17 reference anchors
rows = []
for L in LAYERS:
    Xhe, YheL, CheL = gather_feats_ce(HELD, ablate={L}, want_ce=True)
    # (a) paired per-token dCE (ablated - intact), same tokens/order
    dce = (CheL - Che0)
    dce_mean = dce.mean().item(); dce_se = (dce.std(unbiased=True) / (dce.numel()**0.5)).item()
    # (b) category-decode drop (intact - ablated), fixed probe
    cat_acc_abl = eval_probe(Wprobe, Xhe, YheL)
    cat_drop = cat_acc_intact - cat_acc_abl
    # (c) match-rate change (intact - ablated)
    mm_abl, adv_abl, _ = match_forward(M18, EV[:, :-1], ablate_mlp={L}, MEAN=MMEAN)
    mm_chg = mm_intact - mm_abl
    anchor = '' if 4 <= L <= 15 else ' [anchor]'
    rows.append({'L': L, 'dce_mean': round(dce_mean, 4), 'dce_se': round(dce_se, 4),
                 'cat_acc_ablated': round(cat_acc_abl, 4), 'cat_decode_drop': round(cat_drop, 4),
                 'match_rate_ablated': round(mm_abl, 4), 'match_rate_change': round(mm_chg, 4),
                 'induction_adv_ablated': round(adv_abl, 4)})
    print(f"{L:>3} | {dce_mean:>+13.4f} | {dce_se:>8.4f} | {cat_drop:>+15.4f} | {mm_chg:>+14.4f}{anchor}", flush=True)
    del Xhe; torch.cuda.empty_cache()

out = {'model': 'bilin18', 'held_slice': 'FW[448:600]', 'seq_len': TLEN,
       'ablation': 'mean (in-distribution zero point; MLP means over FINEWEB[0:256])',
       'n_tokens_held': int(Yhe.numel()), 'majority': round(maj, 4),
       'baselines': {'ce': round(ce_intact, 4), 'category_decode_acc': round(cat_acc_intact, 4),
                     'match_rate': round(mm_intact, 4), 'induction_adv': round(adv_intact, 4)},
       'category_names': CATNAMES, 'per_layer': rows,
       'notes': 'dce_mean/se = paired per-token (ablated-intact) CE on held-back; '
                'cat_decode_drop = intact-ablated next-token-category acc (fixed probe on intact train residual); '
                'match_rate_change = intact-ablated two-branch induction argmax match rate (MLP1 = known positive). '
                'Anchors L in {0,1,2,3,16,17}; targets L in 4..15.'}
json.dump(out, open(f'{QK}/qk_midstack_mlp_family.json', 'w'), indent=2)
print("\nQK MIDSTACK MLP FAMILY DONE", flush=True)
