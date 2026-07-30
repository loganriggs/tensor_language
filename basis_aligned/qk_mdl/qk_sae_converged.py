"""DEFINITIVE follow-up to §78: a CONVERGED, high-fidelity TOP-K sparse autoencoder on
the MLP layer-1 hub (the residual write `mo`) of bilin18 -- closing §78's one honest
caveat (its L1 SAE was under-trained, held-back fraction-of-variance-explained only 0.69).

§78 found a small L1 dictionary CROSSES the nameability boundary (23/32 top features
monosemantic vs SVD's 0/32) but NOT the causal boundary (0/32 load-bearing, all features
together only 2.15% of the 5.57-nat MLP1 effect; positive control: removing reconstruction
OR residual each preserves ~98% of loss = collective encoding). The question this run
settles: does HIGHER reconstruction fidelity change the CAUSAL verdict?

SCRIPT 1 (this file): collect MLP1 output on TRAIN, train a converged TOP-K SAE (k active
features per token, dead-feature AuxK revival, LR schedule, long budget, larger shuffled
activation cache). TARGET held FVE >= 0.90. Then run the §78 NAMEABILITY head-to-head on
the HELD slice. Saves the trained dictionary to qk_sae_converged.npz for the causal script.

FORWARD (MLP1 activation extraction), normalisation, class library (lex1 / VOCAB_CLASS),
trigger machinery, and nameability head-to-head all COPIED VERBATIM from
qk_redteam_sae_hub.py (which copies qk_mlp1_tail.py). Only the SAE (L1 -> TopK+AuxK,
convergence budget, feature count) changed. TRAIN = FW[0:256,:128], EVAL = FW[448:600,:128].
tier2_model.load_elriggs('bilin18').
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

# ---------------- GPU GUARD (verbatim from qk_redteam_sae_hub.py) ----------------
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
# LEXICAL CLASS LIBRARY -- VERBATIM from qk_redteam_sae_hub.py / qk_mlp1_tail.py.
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
# MLP1 OUTPUT (residual write `mo`) collection -- VERBATIM from qk_redteam_sae_hub.py.
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
Xtr = torch.cat(tr_list, 0).contiguous()               # (NTRAIN*SEQL, D) = larger shuffled cache
del tr_list
print(f"TRAIN activation matrix {tuple(Xtr.shape)}", flush=True)

# normalisation: centre + scale so mean L2 norm == sqrt(D) (Anthropic convention) -- VERBATIM
MU = Xtr.mean(0)
resid_norm = (Xtr - MU).norm(dim=1).mean()
SCALE = float(math.sqrt(D) / resid_norm)
print(f"activation normalisation: mean centred-norm {resid_norm:.3f}, scale {SCALE:.5f}", flush=True)
Xn = ((Xtr - MU) * SCALE).contiguous()                 # normalised training activations
del Xtr

# collect HELD activations once (reused for sweep + final eval), normalised
tmp = []
for i in range(0, NHELD, BATCH):
    tmp.append(collect_mo(HELD[i:i+BATCH]).reshape(-1, D))
Xh = torch.cat(tmp, 0).contiguous(); del tmp
Xhn = ((Xh - MU) * SCALE).contiguous(); del Xh
print(f"HELD activation matrix {tuple(Xhn.shape)}", flush=True)

# =====================================================================================
# TOP-K SPARSE AUTOENCODER (k active features / token; cleaner than L1). Untied encoder,
# unit-norm decoder columns. Dead-feature AuxK revival (OpenAI): dead latents reconstruct
# the residual so they get gradient and revive. Loss = MSE + aux_coef * AuxK_MSE.
#   pre  = relu((xn - b_dec) @ W_enc.T + b_enc)
#   f    = TopK_k(pre)                              (keep k largest positive per token)
#   xhat = f @ W_dec.T + b_dec                      (W_dec columns unit-norm)
# =====================================================================================
def build_sae(nfeat, seed=0):
    g = torch.Generator(device=DEV); g.manual_seed(seed)
    Wd = torch.randn(D, nfeat, device=DEV, generator=g)
    Wd = Wd / Wd.norm(dim=0, keepdim=True)             # unit-norm decoder columns
    We = Wd.T.clone()                                   # tied init (transpose)
    be = torch.zeros(nfeat, device=DEV)
    bd = Xn.mean(0).clone()                             # geometric-median-ish init
    for t in (Wd, We, be, bd): t.requires_grad_(True)
    return Wd, We, be, bd

def topk_encode(x, We, be, bd, k):
    pre = F.relu((x - bd) @ We.T + be)                  # (N,nfeat) nonneg pre-activations
    vals, idx = pre.topk(k, dim=-1)
    f = torch.zeros_like(pre).scatter_(-1, idx, vals)
    return f, pre

@torch.no_grad()
def sae_stats(Wd, We, be, bd, Xeval, k):
    f, _ = topk_encode(Xeval, We, be, bd, k)
    xhat = f @ Wd.T + bd
    fvu = float(((Xeval - xhat)**2).sum(1).mean() / (Xeval.var(0).sum()))
    l0 = float((f > 0).float().sum(1).mean())
    return fvu, l0

def train_topk_sae(nfeat, k, steps, lr=4e-4, bs=4096, seed=0, aux_coef=1/32,
                   k_aux=256, dead_steps=2500, log_every=0, warmup=200):
    Wd, We, be, bd = build_sae(nfeat, seed)
    opt = torch.optim.Adam([Wd, We, be, bd], lr=lr)
    def lr_at(step):
        if step < warmup: return lr * (step + 1) / warmup
        prog = (step - warmup) / max(1, steps - warmup)          # cosine decay to 0.05*lr
        return lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * prog)))
    N = Xn.shape[0]; g = torch.Generator(device=DEV); g.manual_seed(seed+1)
    last_fired = torch.zeros(nfeat, device=DEV)                   # step index each feat last fired
    for step in range(steps):
        for pg in opt.param_groups: pg['lr'] = lr_at(step)
        idx = torch.randint(0, N, (bs,), device=DEV, generator=g)
        xb = Xn[idx]
        f, pre = topk_encode(xb, We, be, bd, k)
        xhat = f @ Wd.T + bd
        resid = xb - xhat
        mse = (resid**2).sum(1).mean()
        # ---- AuxK dead-feature revival ----
        fired = (f > 0).any(0)
        last_fired[fired] = step
        dead = (step - last_fired) > dead_steps
        aux = torch.tensor(0.0, device=DEV)
        ndead = int(dead.sum())
        if ndead > k_aux:
            pre_dead = pre.clone()
            pre_dead[:, ~dead] = 0.0
            ka = min(k_aux, ndead)
            vals, ai = pre_dead.topk(ka, dim=-1)
            faux = torch.zeros_like(pre_dead).scatter_(-1, ai, vals)
            ehat = faux @ Wd.T                                    # reconstruct residual (no b_dec)
            aux = (resid.detach() - ehat).pow(2).sum(1).mean()
            aux = aux / (resid.detach().pow(2).sum(1).mean() + 1e-8)   # normalised
        loss = mse + aux_coef * aux
        opt.zero_grad(); loss.backward()
        opt.step()
        with torch.no_grad():
            Wd.data = Wd.data / Wd.data.norm(dim=0, keepdim=True).clamp_min(1e-8)
        if log_every and (step % log_every == 0 or step == steps-1):
            with torch.no_grad():
                l0 = float((f > 0).float().sum(1).mean())
                curdead = int(((step - last_fired) > dead_steps).sum())
            print(f"    step {step:6d}  mse {float(mse):.4f}  aux {float(aux):.3f}  "
                  f"L0 {l0:.1f}  dead {curdead}  lr {lr_at(step):.2e}", flush=True)
    return Wd.detach(), We.detach(), be.detach(), bd.detach()

# ---- config sweep: k in {32,64} x NFEAT in {8192,16384}, moderate steps, pick best held FVE
print("Config sweep (k x NFEAT) at 5000 steps ...", flush=True)
SWEEP_STEPS = 5000
sweep = {}
for nfeat in [8192, 16384]:
    for k in [32, 64]:
        Wd, We, be, bd = train_topk_sae(nfeat, k, steps=SWEEP_STEPS, seed=0)
        fvu_h, l0_h = sae_stats(Wd, We, be, bd, Xhn, k)
        fve_h = 1.0 - fvu_h
        sweep[(nfeat, k)] = fve_h
        print(f"  NFEAT={nfeat} k={k}: held FVE {fve_h:.4f}  L0 {l0_h:.1f}", flush=True)
        del Wd, We, be, bd
        torch.cuda.empty_cache()
BEST = max(sweep, key=sweep.get)
NFEAT, K = BEST
print(f"Chosen config: NFEAT={NFEAT} k={K} (sweep held FVE {sweep[BEST]:.4f})", flush=True)

# ---- full convergence training on the chosen config ----
FULL_STEPS = 50000
print(f"Full convergence training: NFEAT={NFEAT} k={K} for {FULL_STEPS} steps ...", flush=True)
t0 = time.time()
Wd, We, be, bd = train_topk_sae(NFEAT, K, steps=FULL_STEPS, seed=0, log_every=5000)
train_secs = time.time() - t0
fvu_tr, l0_tr = sae_stats(Wd, We, be, bd, Xn, K); FVE_tr = 1.0 - fvu_tr
print(f"TRAIN: FVE {FVE_tr:.4f}  L0 {l0_tr:.1f}  ({train_secs:.0f}s)", flush=True)

# =====================================================================================
# HELD-BACK reconstruction + per-feature activations for nameability (VERBATIM structure).
# =====================================================================================
Wd_c, We_c, be_c, bd_c = Wd.contiguous(), We.contiguous(), be.contiguous(), bd.contiguous()
with torch.no_grad():
    fh, _ = topk_encode(Xhn, We_c, be_c, bd_c, K)
    xhat_h = fh @ Wd_c.T + bd_c
    fvu_h = float(((Xhn - xhat_h)**2).sum(1).mean() / (Xhn.var(0).sum()))
    l0_h = float((fh > 0).float().sum(1).mean())
    FVE_h = 1.0 - fvu_h
    feat_active = (fh > 0).float().mean(0).cpu().numpy()
    feat_meanact = (fh.sum(0) / (fh > 0).float().sum(0).clamp_min(1)).cpu().numpy()
    feat_held = fh.reshape(NHELD, SEQL, NFEAT).cpu().numpy()
    del fh, xhat_h
print(f"HELD: FVE {FVE_h:.4f}  L0 {l0_h:.1f}  (dead features: {(feat_active==0).sum()}/{NFEAT})", flush=True)

# =====================================================================================
# NAMEABILITY head-to-head (the §68 way) -- VERBATIM from qk_redteam_sae_hub.py.
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

valid_cur = held_np[~bad_trigger]
_bc = Counter(VOCAB_CLASS[valid_cur].tolist())
_bt = sum(_bc.values())
BASE_RATE = {c: _bc.get(c, 0)/_bt for c in CLASS_LIST}

rank_score = feat_active * feat_meanact
order = np.argsort(-rank_score)
live = [int(j) for j in order if feat_active[j] > 0
        and (feat_held[:, :, j].reshape(-1)[~bad_trigger.reshape(-1)] > 0).sum() >= KCAUSAL]
TOPN = 32
top_feats = live[:TOPN]

feat_records = []
PURITY_BAR = 0.5
ENRICH_BAR = 2.0
for j in top_feats:
    tclass_top, tpurity, cur = trigger_class_sig(feat_held[:, :, j])
    vv, cc = np.unique(cur, return_counts=True); oo = np.argsort(-cc)[:6]
    trig_tok_top = [(dec(int(vv[k])), int(cc[k])) for k in oo]
    dom_class = tclass_top[0][0] if tclass_top else None
    dom_frac = (tclass_top[0][1]/KCAUSAL) if tclass_top else 0.0
    enrich = (dom_frac / BASE_RATE[dom_class]) if (dom_class and BASE_RATE[dom_class] > 0) else 0.0
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
print(f"\nNAMEABILITY head-to-head (vs SVD 0/32, §78 L1-SAE 23/32):", flush=True)
print(f"  {n_nameable_purity}/{TOPN} top features have trigger class purity >= {PURITY_BAR}", flush=True)
print(f"  {n_nameable}/{TOPN} top features monosemantic (purity>={PURITY_BAR} AND >= {ENRICH_BAR}x base rate)", flush=True)
for r in feat_records[:20]:
    print(f"  feat {r['feature']:5d} freq {r['act_freq']:.4f} purity {r['trigger_purity']:.3f} "
          f"dom {r['dominant_class']}({r['dominant_class_frac']:.2f}, {r['dominant_enrichment']:.1f}x) "
          f"{'NAMEABLE' if r['nameable'] else '-'} top_tok {r['trigger_token_top'][:3]}", flush=True)

# =====================================================================================
# SAVE dictionary + top-feature list for the causal script.
# =====================================================================================
np.savez(f'{QK}/qk_sae_converged.npz',
         W_dec=Wd.cpu().numpy(), W_enc=We.cpu().numpy(),
         b_enc=be.cpu().numpy(), b_dec=bd.cpu().numpy(),
         MU=MU.cpu().numpy(), SCALE=np.float32(SCALE),
         top_feats=np.array(top_feats, np.int64),
         live_feats=np.array(live, np.int64),
         feat_active=feat_active, feat_meanact=feat_meanact,
         K=np.int64(K), NFEAT=np.int64(NFEAT))
print(f"Saved dictionary to qk_sae_converged.npz", flush=True)

out = {
    'meta': {
        'model': 'bilin18', 'layer': LI, 'quantity': 'MLP1 feed-forward OUTPUT (residual write mo), dim=D=1152',
        'dictionary': 'TOP-K sparse autoencoder, untied encoder, unit-norm decoder columns, AuxK dead-feature revival',
        'n_features': NFEAT, 'k_active': K, 'overcompleteness': round(NFEAT/D, 3),
        'train_slice': 'FW[0:256,:128]', 'held_slice': 'FW[448:600,:128]',
        'sweep_steps': SWEEP_STEPS, 'full_steps': FULL_STEPS, 'train_secs': round(train_secs, 1),
        'lr': 4e-4, 'lr_schedule': 'linear warmup 200 + cosine decay to 0.05x', 'batch_positions': 4096,
        'aux': 'AuxK dead-feature revival: dead latents (unfired > 2500 steps) reconstruct residual, k_aux=256, coef 1/32',
        'train_cache_positions': int(Xn.shape[0]),
        'forward': 'VERBATIM qk_redteam_sae_hub.py / qk_mlp1_tail.py (MLP1 output residual write)',
        'nameability_method': ('trigger token-class purity (§68 way), top-KCAUSAL firing positions, VERBATIM §74/§78 class '
                               'library; monosemantic requires purity>=bar AND dominant class >= ENRICH_BAR x its corpus base rate'),
        'KCAUSAL': KCAUSAL, 'purity_bar': PURITY_BAR, 'enrich_bar': ENRICH_BAR, 'TOPN': TOPN,
        'PURPOSE': ('CONVERGED high-fidelity SAE closing §78 undertrained caveat (held FVE 0.69). If converged features '
                    'STILL do not clear the causal bar, §78 causal negative is airtight; if they DO, §78 is overturned.'),
    },
    'config_sweep': {f'NFEAT{nf}_k{k}': round(fve, 4) for (nf, k), fve in sweep.items()},
    'base_rate': {c: round(BASE_RATE[c], 4) for c in CLASS_LIST},
    'reconstruction': {
        'train_FVE': round(FVE_tr, 4), 'train_L0': round(l0_tr, 1),
        'held_FVE': round(FVE_h, 4), 'held_L0': round(l0_h, 1),
        'dead_features': int((feat_active == 0).sum()),
        'target_FVE': 0.90, 'reached_target': bool(FVE_h >= 0.90),
    },
    'nameability': {
        'TOPN': TOPN, 'purity_bar': PURITY_BAR, 'enrich_bar': ENRICH_BAR,
        'n_nameable_by_purity': n_nameable_purity, 'n_nameable': n_nameable,
        'svd_reference': '§74: 0 of 32 SVD directions nameable',
        'l1_sae_reference': '§78: 23 of 32 L1-SAE features nameable (held FVE 0.69)',
        'top_feature_records': feat_records,
    },
}
json.dump(out, open(f'{QK}/qk_sae_converged.json', 'w'), indent=2)
print("\nSaved qk_sae_converged.json", flush=True)
print("QK SAE CONVERGED (script 1: train + nameability) DONE", flush=True)
