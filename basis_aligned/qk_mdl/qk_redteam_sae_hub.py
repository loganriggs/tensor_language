"""BOUNDED ADVERSARIAL RED-TEAM of §74's headline claim that MLP layer-1 (the hub) is
IRREDUCIBLY DISTRIBUTED and that naming its features "would require sparse-dictionary /
sparse-autoencoder methods."

§74 established: MLP1's top-32 SVD OUTPUT directions are 0-of-32 single-direction
nameable (max causal z 1.6) and 76% of the top-32 effect appears only under JOINT
removal (superposition signature). BOUNDARY CLAIM: the distributed early-layer bulk
"needs sparse-dictionary / SAE methods."

This script red-teams that specific assertion by asking whether a SMALL sparse
overcomplete dictionary can actually recover interpretable, causally-load-bearing
features from MLP1 that single-direction SVD could not -- i.e. does a dictionary CROSS
the §74 boundary, or does the hub resist even the tool §74 named? FEASIBILITY PROBE,
NOT a pivot to SAE interpretability; honest about the under-trained-SAE caveat.

SCRIPT 1 (this file): collect MLP1's feed-forward OUTPUT (the residual write `mo`, the
output-side quantity §74/§73 characterised; dim = model width 1152) on the TRAIN slice,
fit a small L1 sparse autoencoder (~3.5x overcomplete, 4096 features), report
reconstruction (fraction of variance explained) + sparsity (average active features),
and run the NAMEABILITY head-to-head on the HELD slice (per-feature trigger token-class
purity, the §68 way). Saves the trained dictionary to qk_redteam_sae_hub.npz for the
causal script (qk_redteam_sae_hub_2.py).

FORWARD (MLP1 activation extraction) + mean/collection convention COPIED VERBATIM from
qk_mlp1_tail.py / qk_mlp_superposition.py. Class library (lex1 / VOCAB_CLASS) +
trigger_class_sig COPIED VERBATIM from qk_mlp1_tail.py. TRAIN = FW[0:256,:128] (dictionary
fit only), EVALUATE = FW[448:600,:128] (held-back). tier2_model.load_elriggs('bilin18').
"""
import json, sys, math, time, subprocess
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter
from transformers import AutoTokenizer
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'

# ---------------- GPU GUARD (verbatim from qk_mlp1_tail.py) ----------------
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
def dec(t): return repr(tok.decode([int(t)]))
LI = 1              # MLP layer 1 -- the hub
NFEAT = 4096        # ~3.56x overcomplete dictionary (small, feasibility probe)
print(f"bilin18 NL={NL} NH={NH} HD={HD} D={D} V={V}; MLP L{LI} output-side dictionary NFEAT={NFEAT} ({NFEAT/D:.2f}x)", flush=True)

FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
SEQL = 128
TRAIN = FINEWEB[0:256, :SEQL].to(DEV)        # dictionary-fit slice
HELD = FINEWEB[448:600, :SEQL].to(DEV)       # held-back evaluation slice
NTRAIN = TRAIN.shape[0]; NHELD = HELD.shape[0]
BATCH = 8
KCAUSAL = 200                                # top-K activation positions per feature (matches §74 census)

# special/degenerate tokens excluded from trigger selection (matches §74)
_special = {tok.eos_token_id}
for _t in range(V):
    if '�' in tok.decode([_t]): _special.add(_t)
SPECIAL = np.array(sorted(_special))
print(f"{len(SPECIAL)} special token ids masked from trigger selection", flush=True)

# =====================================================================================
# LEXICAL CLASS LIBRARY -- VERBATIM from qk_mlp1_tail.py (lex1 / VOCAB_CLASS).
# =====================================================================================
BRACKETS_OPEN  = set("([{<")
BRACKETS_CLOSE = set(")]}>")
QUOTE_OPEN     = set("“‘`")
QUOTE_CLOSE    = set("”’")
QUOTE_STRAIGHT = set("\"'")
PUNCT  = set(".,;:!?—–-…*|/\\~@#%^&+=_")
COORDINATORS = {"and","or","but","nor","yet","so"}
DETERMINERS  = {"the","a","an","this","that","these","those","some","any","each",
                "every","no","another","such"}
PRONOUNS     = {"i","we","you","he","she","it","they","them","us","me","him","her","which","who"}

def lex1(s):
    if s == "": return 'other'
    if ('�' in s) or (s == tok.eos_token or '<|endoftext|>' in s): return 'special'
    if '\n' in s: return 'newline'
    body = s.strip(); low = body.lower()
    if body == "": return 'other'
    if all(ch in QUOTE_OPEN for ch in body): return 'quote_open'
    if all(ch in QUOTE_CLOSE for ch in body): return 'quote_close'
    if all(ch in QUOTE_STRAIGHT for ch in body): return 'quote'
    if all(ch in BRACKETS_OPEN for ch in body): return 'bracket_open'
    if all(ch in BRACKETS_CLOSE for ch in body): return 'bracket_close'
    if any(ch.isdigit() for ch in body): return 'digit'
    if all((ch in PUNCT or ch in QUOTE_STRAIGHT or ch in QUOTE_OPEN or ch in QUOTE_CLOSE
            or ch in BRACKETS_OPEN or ch in BRACKETS_CLOSE) for ch in body): return 'punct'
    if low in DETERMINERS: return 'determiner'
    if low in COORDINATORS: return 'coordinator'
    if low in PRONOUNS: return 'pronoun'
    if body[0].isupper(): return 'capital'
    lead_space = s.startswith(' ')
    if lead_space and body.isalpha() and len(body) > 1: return 'word'
    if (not lead_space) and body.isalpha() and body[0].islower(): return 'subword'
    return 'other'

VOCAB_CLASS = np.array([lex1(tok.decode([t])) for t in range(V)], dtype=object)
CLASS_LIST = sorted(set(VOCAB_CLASS.tolist()))
CIDX = {c: i for i, c in enumerate(CLASS_LIST)}
print(f"lexical classes ({len(CLASS_LIST)}): {CLASS_LIST}", flush=True)

# =====================================================================================
# MLP1 OUTPUT (residual write `mo`) collection -- forward truncated to layer LI, VERBATIM.
# =====================================================================================
@torch.no_grad()
def collect_mo(idx):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(LI + 1):
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
            return mo                                   # (B,T,D) -- the residual write §74 characterised
        x = x + mo

print("Collecting MLP1 output activations on TRAIN ...", flush=True)
tr_list = []
for i in range(0, NTRAIN, BATCH):
    mo = collect_mo(TRAIN[i:i+BATCH]).reshape(-1, D)
    tr_list.append(mo)
Xtr = torch.cat(tr_list, 0).contiguous()               # (NTRAIN*SEQL, D)
del tr_list
print(f"TRAIN activation matrix {tuple(Xtr.shape)}", flush=True)

# normalisation: centre + scale so mean L2 norm == sqrt(D) (Anthropic convention)
MU = Xtr.mean(0)                                        # (D,)
resid_norm = (Xtr - MU).norm(dim=1).mean()
SCALE = float(math.sqrt(D) / resid_norm)
print(f"activation normalisation: mean centred-norm {resid_norm:.3f}, scale {SCALE:.5f}", flush=True)
Xn = (Xtr - MU) * SCALE                                 # normalised training activations
tot_var = float(Xn.var(0).sum())                        # for FVU

# =====================================================================================
# SPARSE AUTOENCODER (untied encoder, unit-norm decoder columns, L1 penalty).
#   f    = relu((xn - b_dec) @ W_enc.T + b_enc)
#   xhat = f @ W_dec.T + b_dec           (W_dec columns unit-norm)
#   loss = mse + L1 * mean_i sum(f)
# Feasibility probe: short training, budget reported honestly.
# =====================================================================================
def build_sae(seed=0):
    g = torch.Generator(device=DEV); g.manual_seed(seed)
    Wd = torch.randn(D, NFEAT, device=DEV, generator=g)
    Wd = Wd / Wd.norm(dim=0, keepdim=True)             # unit-norm decoder columns
    We = Wd.T.clone()                                   # tied init (transpose)
    be = torch.zeros(NFEAT, device=DEV)
    bd = Xn.mean(0).clone()                             # geometric-median-ish init
    for t in (Wd, We, be, bd): t.requires_grad_(True)
    return Wd, We, be, bd

def sae_stats(Wd, We, be, bd, Xeval):
    with torch.no_grad():
        f = F.relu((Xeval - bd) @ We.T + be)
        xhat = f @ Wd.T + bd
        fvu = float(((Xeval - xhat)**2).sum(1).mean() / (Xeval.var(0).sum()))
        l0 = float((f > 0).float().sum(1).mean())
    return fvu, l0

def train_sae(L1, steps, lr=4e-4, bs=4096, seed=0, log_every=0):
    Wd, We, be, bd = build_sae(seed)
    opt = torch.optim.Adam([Wd, We, be, bd], lr=lr)
    N = Xn.shape[0]; g = torch.Generator(device=DEV); g.manual_seed(seed+1)
    for step in range(steps):
        idx = torch.randint(0, N, (bs,), device=DEV, generator=g)
        xb = Xn[idx]
        f = F.relu((xb - bd) @ We.T + be)
        xhat = f @ Wd.T + bd
        mse = ((xb - xhat)**2).sum(1).mean()
        l1 = f.sum(1).mean()
        loss = mse + L1 * l1
        opt.zero_grad(); loss.backward()
        # zero the decoder-parallel gradient component would need re-projection; instead
        # renormalise decoder columns after the step (standard, keeps L1 meaningful)
        opt.step()
        with torch.no_grad():
            Wd.data = Wd.data / Wd.data.norm(dim=0, keepdim=True).clamp_min(1e-8)
        if log_every and (step % log_every == 0 or step == steps-1):
            with torch.no_grad():
                l0 = float((f > 0).float().sum(1).mean())
                msef = float(mse.detach()); lossf = float(loss.detach())
            print(f"    step {step:5d}  mse {msef:.4f}  L0 {l0:.1f}  loss {lossf:.4f}", flush=True)
    return Wd.detach(), We.detach(), be.detach(), bd.detach()

# ---- lambda probe to land the average-active-features (L0) in a genuinely sparse range,
# then full train. A near-dense code (L0 ~ NFEAT) is NOT a sparse dictionary and its
# apparent "nameability" would just recover corpus base rates -- so we require real sparsity.
print("Lambda probe (1500-step runs) ...", flush=True)
probe = {}
for L1 in [0.05, 0.15, 0.4, 1.0, 2.5]:
    Wd, We, be, bd = train_sae(L1, steps=1500, seed=0)
    fvu, l0 = sae_stats(Wd, We, be, bd, Xn)
    probe[L1] = (fvu, l0)
    print(f"  L1={L1:.2f}: FVU {fvu:.4f}  L0 {l0:.1f}", flush=True)
# choose lambda: prefer L0 in [5,60] (genuinely sparse) with lowest FVU; else closest L0 to 20
cand = [(L1, fvu, l0) for L1, (fvu, l0) in probe.items() if 5 <= l0 <= 60]
if cand:
    CHOSEN_L1 = min(cand, key=lambda z: z[1])[0]
else:
    CHOSEN_L1 = min(probe.items(), key=lambda kv: abs(kv[1][1]-20))[0]
print(f"Chosen L1 = {CHOSEN_L1:.2f}", flush=True)

TRAIN_STEPS = 6000
print(f"Full dictionary training: {TRAIN_STEPS} steps at L1={CHOSEN_L1:.2f} ...", flush=True)
t0 = time.time()
Wd, We, be, bd = train_sae(CHOSEN_L1, steps=TRAIN_STEPS, seed=0, log_every=1000)
train_secs = time.time() - t0
fvu_tr, l0_tr = sae_stats(Wd, We, be, bd, Xn)
FVE_tr = 1.0 - fvu_tr
print(f"TRAIN: FVE {FVE_tr:.4f}  L0 {l0_tr:.1f}  ({train_secs:.0f}s)", flush=True)

# =====================================================================================
# HELD-BACK reconstruction + sparsity, and per-feature activations for nameability.
# =====================================================================================
print("Collecting MLP1 output activations on HELD ...", flush=True)
feat_held = np.zeros((NHELD, SEQL, NFEAT), np.float32)   # per-position feature activations (held)
held_mo_sumsq = 0.0; held_err_sumsq = 0.0; held_var_accum = []
Wd_c, We_c, be_c, bd_c = Wd.contiguous(), We.contiguous(), be.contiguous(), bd.contiguous()
held_mo_list_mean = torch.zeros(D, device=DEV); nhpos = 0
# two-pass-free: accumulate for FVU using global held mean of normalised acts
# first get held mean
tmp = []
for i in range(0, NHELD, BATCH):
    mo = collect_mo(HELD[i:i+BATCH]).reshape(-1, D)
    tmp.append(mo)
Xh = torch.cat(tmp, 0).contiguous(); del tmp
Xhn = (Xh - MU) * SCALE
with torch.no_grad():
    fh = F.relu((Xhn - bd_c) @ We_c.T + be_c)
    xhat_h = fh @ Wd_c.T + bd_c
    fvu_h = float(((Xhn - xhat_h)**2).sum(1).mean() / (Xhn.var(0).sum()))
    l0_h = float((fh > 0).float().sum(1).mean())
    FVE_h = 1.0 - fvu_h
    # per-feature activation frequency + total mass (held)
    feat_active = (fh > 0).float().mean(0).cpu().numpy()             # (NFEAT,) fraction active
    feat_mass = fh.sum(0).cpu().numpy()                              # (NFEAT,) total activation
    feat_meanact = (fh.sum(0) / (fh > 0).float().sum(0).clamp_min(1)).cpu().numpy()  # mean when active
    feat_held = fh.reshape(NHELD, SEQL, NFEAT).cpu().numpy()
print(f"HELD: FVE {FVE_h:.4f}  L0 {l0_h:.1f}  (dead features: {(feat_active==0).sum()}/{NFEAT})", flush=True)

# =====================================================================================
# NAMEABILITY head-to-head (the §68 way): rank features by activation frequency x norm,
# compute trigger token-class purity for the top features. VERBATIM trigger machinery.
# =====================================================================================
held_np = HELD.cpu().numpy()
pos_t = np.tile(np.arange(SEQL), NHELD).reshape(NHELD, SEQL)
is_special = np.isin(held_np, SPECIAL)
valid_next = pos_t < (SEQL - 1)
bad_trigger = (pos_t == 0) | is_special | ~valid_next

def conc(ids):
    ids = np.asarray(ids)
    if len(ids) <= 1: return 0.0
    _, c = np.unique(ids, return_counts=True); p = c/c.sum()
    return float(1 - (-(p*np.log(p)).sum())/math.log(len(ids)))

def trigger_class_sig(act2d):
    """top-KCAUSAL firing positions -> current-token class distribution + purity (VERBATIM §74)."""
    a = act2d.copy().reshape(-1)
    a[bad_trigger.reshape(-1)] = -1e30
    if (a > -1e29).sum() < KCAUSAL:
        return [], 0.0, np.array([], int)
    tk = np.argpartition(a, -KCAUSAL)[-KCAUSAL:]
    fs, fp = tk // SEQL, tk % SEQL
    cur = held_np[fs, fp]
    classes = VOCAB_CLASS[cur]
    cnt = Counter(classes.tolist())
    top = cnt.most_common(6)
    purity = conc([CIDX[c] for c in classes])
    return [(c, int(n)) for c, n in top], round(purity, 4), cur

# CORPUS BASE RATE of each class over valid trigger positions -- so a common-but-diffuse
# feature (high purity only because a class is common) cannot masquerade as monosemantic.
valid_cur = held_np[~bad_trigger]
_bc = Counter(VOCAB_CLASS[valid_cur].tolist())
_bt = sum(_bc.values())
BASE_RATE = {c: _bc.get(c, 0)/_bt for c in CLASS_LIST}

# rank by activation frequency x mean-active-norm (task: "activation frequency x norm")
# decoder columns are unit-norm, so norm proxy = mean activation when active
rank_score = feat_active * feat_meanact
order = np.argsort(-rank_score)
# restrict to live features with enough firing positions
live = [int(j) for j in order if feat_active[j] > 0
        and (feat_held[:, :, j].reshape(-1)[~bad_trigger.reshape(-1)] > 0).sum() >= KCAUSAL]
TOPN = 32                                       # head-to-head with SVD's top-32 (0-of-32)
top_feats = live[:TOPN]

feat_records = []
PURITY_BAR = 0.5                                # dominant class holds the majority
ENRICH_BAR = 2.0                                # dominant class >= 2x its corpus base rate
for j in top_feats:
    tclass_top, tpurity, cur = trigger_class_sig(feat_held[:, :, j])
    # top trigger tokens (for the human-readable name)
    vv, cc = np.unique(cur, return_counts=True); oo = np.argsort(-cc)[:6]
    trig_tok_top = [(dec(int(vv[k])), int(cc[k])) for k in oo]
    dom_class = tclass_top[0][0] if tclass_top else None
    dom_frac = (tclass_top[0][1]/KCAUSAL) if tclass_top else 0.0
    enrich = (dom_frac / BASE_RATE[dom_class]) if (dom_class and BASE_RATE[dom_class] > 0) else 0.0
    # monosemantic = dominant class holds a majority AND is strongly enriched over base rate
    nameable = bool(tpurity >= PURITY_BAR and enrich >= ENRICH_BAR)
    feat_records.append({
        'feature': int(j), 'act_freq': round(float(feat_active[j]), 5),
        'mean_active': round(float(feat_meanact[j]), 4), 'rank_score': round(float(rank_score[j]), 5),
        'trigger_class_top': tclass_top, 'trigger_purity': tpurity,
        'dominant_class': dom_class, 'dominant_class_frac': round(float(dom_frac), 3),
        'dominant_base_rate': round(float(BASE_RATE[dom_class]) if dom_class else 0.0, 4),
        'dominant_enrichment': round(float(enrich), 2),
        'trigger_token_top': trig_tok_top,
        'nameable_by_purity': bool(tpurity >= PURITY_BAR),
        'nameable': nameable,
    })

n_nameable_purity = int(sum(r['nameable_by_purity'] for r in feat_records))
n_nameable = int(sum(r['nameable'] for r in feat_records))
print(f"\nNAMEABILITY head-to-head (vs SVD 0/32):", flush=True)
print(f"  {n_nameable_purity}/{TOPN} top features have trigger class purity >= {PURITY_BAR}", flush=True)
print(f"  {n_nameable}/{TOPN} top features monosemantic (purity>={PURITY_BAR} AND >= {ENRICH_BAR}x base rate)", flush=True)
for r in feat_records[:20]:
    print(f"  feat {r['feature']:4d} freq {r['act_freq']:.4f} purity {r['trigger_purity']:.3f} "
          f"dom {r['dominant_class']}({r['dominant_class_frac']:.2f}, {r['dominant_enrichment']:.1f}x) "
          f"{'NAMEABLE' if r['nameable'] else '-'} top_tok {r['trigger_token_top'][:3]}", flush=True)

# =====================================================================================
# SAVE dictionary + top-feature list for the causal script.
# =====================================================================================
np.savez(f'{QK}/qk_redteam_sae_hub.npz',
         W_dec=Wd.cpu().numpy(), W_enc=We.cpu().numpy(),
         b_enc=be.cpu().numpy(), b_dec=bd.cpu().numpy(),
         MU=MU.cpu().numpy(), SCALE=np.float32(SCALE),
         top_feats=np.array(top_feats, np.int64),
         live_feats=np.array(live, np.int64),
         feat_active=feat_active, feat_meanact=feat_meanact,
         L1=np.float32(CHOSEN_L1), NFEAT=np.int64(NFEAT))
print(f"Saved dictionary to qk_redteam_sae_hub.npz", flush=True)

out = {
    'meta': {
        'model': 'bilin18', 'layer': LI, 'quantity': 'MLP1 feed-forward OUTPUT (residual write mo), dim=D=1152',
        'dictionary': 'L1 sparse autoencoder, untied encoder, unit-norm decoder columns',
        'n_features': NFEAT, 'overcompleteness': round(NFEAT/D, 3),
        'train_slice': 'FW[0:256,:128]', 'held_slice': 'FW[448:600,:128]',
        'chosen_L1': CHOSEN_L1, 'train_steps': TRAIN_STEPS, 'train_secs': round(train_secs, 1),
        'lr': 4e-4, 'batch_positions': 4096,
        'forward': 'VERBATIM qk_mlp1_tail.py / qk_mlp_superposition.py (MLP1 output residual write)',
        'nameability_method': ('trigger token-class purity (§68 way), top-KCAUSAL firing positions, VERBATIM §74 class '
                               'library; monosemantic requires purity>=bar AND dominant class >= ENRICH_BAR x its corpus '
                               'base rate (guards against a common-but-diffuse class inflating purity)'),
        'KCAUSAL': KCAUSAL, 'purity_bar': PURITY_BAR, 'enrich_bar': ENRICH_BAR, 'TOPN': TOPN,
        'FEASIBILITY_CAVEAT': ('SMALL, deliberately under-trained SAE (feasibility probe). A dictionary FAILING '
                               'here is weaker evidence than a converged one succeeding; a dictionary SUCCEEDING '
                               'here is strong evidence the §74 boundary is crossable.'),
    },
    'lambda_probe': {f'{L1:.2f}': {'FVU': round(fvu, 4), 'L0': round(l0, 1)} for L1, (fvu, l0) in probe.items()},
    'base_rate': {c: round(BASE_RATE[c], 4) for c in CLASS_LIST},
    'reconstruction': {
        'train_FVE': round(FVE_tr, 4), 'train_L0': round(l0_tr, 1),
        'held_FVE': round(FVE_h, 4), 'held_L0': round(l0_h, 1),
        'dead_features': int((feat_active == 0).sum()),
    },
    'nameability': {
        'TOPN': TOPN, 'purity_bar': PURITY_BAR, 'enrich_bar': ENRICH_BAR,
        'n_nameable_by_purity': n_nameable_purity,
        'n_nameable': n_nameable,
        'svd_reference': '§74: 0 of 32 SVD directions single-direction nameable/load-bearing',
        'top_feature_records': feat_records,
    },
}
json.dump(out, open(f'{QK}/qk_redteam_sae_hub.json', 'w'), indent=2)
print("\nSaved qk_redteam_sae_hub.json", flush=True)
print("QK REDTEAM SAE HUB (script 1: train + nameability) DONE", flush=True)
