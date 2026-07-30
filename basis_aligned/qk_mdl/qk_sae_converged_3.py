"""DEFINITIVE follow-up to §78 -- SCRIPT 3: BEST-HELD-FIDELITY TopK SAE.

The 50000-step converged run (qk_sae_converged.py) revealed that convergence on the
256-sequence TRAIN slice OVERFITS: train FVE rose to 0.945 but held FVE FELL to 0.637
(below §78's 0.69), and held FVE peaked ~0.70 at ~5000 steps then degraded. Held-domain
fidelity is GENERALISATION-bounded on 256 training sequences, not training-budget-bounded.

To give the causal test the FAIREST high-held-fidelity SAE (rather than an overfit one),
this script trains TopK SAEs with VALIDATION-CHECKPOINTED early stopping: validation slice
FW[256:448] (a THIRD, non-overlapping slice) selects the best checkpoint; FW[448:600] stays
pristine for held eval + the causal test. Sweeps NFEAT in {2048,4096,8192} (smaller dicts
generalise better here) at k=64, keeps the global best-validation-FVE checkpoint, then
reports held FVE / L0 / dead + the §78 NAMEABILITY head-to-head, and OVERWRITES
qk_sae_converged.npz with this best-held SAE so the causal script tests it.

All model-forward / normalisation / nameability machinery COPIED VERBATIM from
qk_sae_converged.py (which copies qk_redteam_sae_hub.py / qk_mlp1_tail.py).
"""
import json, sys, math, time, subprocess, copy
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter
from transformers import AutoTokenizer
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
tok = AutoTokenizer.from_pretrained('gpt2')
def dec(t): return repr(tok.decode([int(t)]))
LI = 1

FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
SEQL = 128
TRAIN = FINEWEB[0:256, :SEQL].to(DEV)
VAL   = FINEWEB[256:448, :SEQL].to(DEV)   # checkpoint-selection slice (clean, separate)
HELD  = FINEWEB[448:600, :SEQL].to(DEV)   # pristine held eval + causal slice
NTRAIN = TRAIN.shape[0]; NHELD = HELD.shape[0]
BATCH = 8
KCAUSAL = 200

_special = {tok.eos_token_id}
for _t in range(V):
    if '�' in tok.decode([_t]): _special.add(_t)
SPECIAL = np.array(sorted(_special))
print(f"{len(SPECIAL)} special token ids masked", flush=True)

# ------- class library (VERBATIM) -------
BRACKETS_OPEN=set("([{<"); BRACKETS_CLOSE=set(")]}>"); QUOTE_OPEN=set("“‘`"); QUOTE_CLOSE=set("”’")
QUOTE_STRAIGHT=set("\"'"); PUNCT=set(".,;:!?—–-…*|/\\~@#%^&+=_")
COORDINATORS={"and","or","but","nor","yet","so"}
DETERMINERS={"the","a","an","this","that","these","those","some","any","each","every","no","another","such"}
PRONOUNS={"i","we","you","he","she","it","they","them","us","me","him","her","which","who"}
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
CLASS_LIST = sorted(set(VOCAB_CLASS.tolist())); CIDX = {c: i for i, c in enumerate(CLASS_LIST)}

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
        if li == LI: return mo
        x = x + mo

def cache(slc):
    out = []
    for i in range(0, slc.shape[0], BATCH):
        out.append(collect_mo(slc[i:i+BATCH]).reshape(-1, D))
    return torch.cat(out, 0).contiguous()

print("Caching activations (train/val/held) ...", flush=True)
Xtr = cache(TRAIN)
MU = Xtr.mean(0)
resid_norm = (Xtr - MU).norm(dim=1).mean()
SCALE = float(math.sqrt(D) / resid_norm)
Xn  = ((Xtr - MU) * SCALE).contiguous(); del Xtr
Xvn = ((cache(VAL)  - MU) * SCALE).contiguous()
Xhn = ((cache(HELD) - MU) * SCALE).contiguous()
print(f"caches: train {tuple(Xn.shape)} val {tuple(Xvn.shape)} held {tuple(Xhn.shape)} scale {SCALE:.5f}", flush=True)

def topk_encode(x, We, be, bd, k):
    pre = F.relu((x - bd) @ We.T + be)
    vals, idx = pre.topk(k, dim=-1)
    f = torch.zeros_like(pre).scatter_(-1, idx, vals)
    return f, pre

@torch.no_grad()
def fve(Wd, We, be, bd, X, k):
    f, _ = topk_encode(X, We, be, bd, k)
    xhat = f @ Wd.T + bd
    fvu = float(((X - xhat)**2).sum(1).mean() / (X.var(0).sum()))
    l0 = float((f > 0).float().sum(1).mean())
    return 1.0 - fvu, l0

def train_bestval(nfeat, k, steps=15000, lr=4e-4, bs=4096, seed=0, aux_coef=1/32,
                  k_aux=256, dead_steps=2500, warmup=200, val_every=1000):
    g = torch.Generator(device=DEV); g.manual_seed(seed)
    Wd = torch.randn(D, nfeat, device=DEV, generator=g); Wd = Wd / Wd.norm(dim=0, keepdim=True)
    We = Wd.T.clone(); be = torch.zeros(nfeat, device=DEV); bd = Xn.mean(0).clone()
    for t in (Wd, We, be, bd): t.requires_grad_(True)
    opt = torch.optim.Adam([Wd, We, be, bd], lr=lr)
    def lr_at(step):
        if step < warmup: return lr*(step+1)/warmup
        prog = (step-warmup)/max(1, steps-warmup)
        return lr*(0.05+0.95*0.5*(1+math.cos(math.pi*prog)))
    N = Xn.shape[0]; gg = torch.Generator(device=DEV); gg.manual_seed(seed+1)
    last_fired = torch.zeros(nfeat, device=DEV)
    best_val = -1e9; best_ckpt = None; best_step = -1
    for step in range(steps):
        for pg in opt.param_groups: pg['lr'] = lr_at(step)
        idx = torch.randint(0, N, (bs,), device=DEV, generator=gg); xb = Xn[idx]
        f, pre = topk_encode(xb, We, be, bd, k); xhat = f @ Wd.T + bd; resid = xb - xhat
        mse = (resid**2).sum(1).mean()
        fired = (f > 0).any(0); last_fired[fired] = step
        dead = (step - last_fired) > dead_steps; ndead = int(dead.sum())
        aux = torch.tensor(0.0, device=DEV)
        if ndead > k_aux:
            pd = pre.clone(); pd[:, ~dead] = 0.0
            vals, ai = pd.topk(min(k_aux, ndead), dim=-1)
            faux = torch.zeros_like(pd).scatter_(-1, ai, vals)
            aux = (resid.detach() - faux @ Wd.T).pow(2).sum(1).mean() / (resid.detach().pow(2).sum(1).mean()+1e-8)
        loss = mse + aux_coef*aux
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            Wd.data = Wd.data / Wd.data.norm(dim=0, keepdim=True).clamp_min(1e-8)
        if step % val_every == 0 or step == steps-1:
            vfve, _ = fve(Wd.detach(), We.detach(), be.detach(), bd.detach(), Xvn, k)
            if vfve > best_val:
                best_val = vfve; best_step = step
                best_ckpt = tuple(t.detach().clone() for t in (Wd, We, be, bd))
    return best_ckpt, best_val, best_step

print("Best-validation-FVE sweep (k=64, val slice FW[256:448]) ...", flush=True)
CONFIGS = [(2048, 64), (4096, 64), (8192, 64)]
best = None
runs = {}
for nfeat, k in CONFIGS:
    ckpt, vbest, vstep = train_bestval(nfeat, k, steps=15000, seed=0)
    hfve, hl0 = fve(*ckpt, Xhn, k)
    runs[(nfeat, k)] = {'val_FVE': round(vbest, 4), 'val_step': vstep, 'held_FVE': round(hfve, 4), 'held_L0': round(hl0, 1)}
    print(f"  NFEAT={nfeat} k={k}: best val FVE {vbest:.4f} @step {vstep}; held FVE {hfve:.4f} L0 {hl0:.1f}", flush=True)
    if best is None or vbest > best[1]:
        best = ((nfeat, k), vbest, ckpt)
    torch.cuda.empty_cache()

(NFEAT, K), _, CK = best
Wd, We, be, bd = CK
print(f"CHOSEN best-held config: NFEAT={NFEAT} k={K}", flush=True)

# ---- held eval + per-feature acts for nameability ----
Wd_c, We_c, be_c, bd_c = Wd.contiguous(), We.contiguous(), be.contiguous(), bd.contiguous()
with torch.no_grad():
    fh, _ = topk_encode(Xhn, We_c, be_c, bd_c, K)
    xhat_h = fh @ Wd_c.T + bd_c
    FVE_h = 1.0 - float(((Xhn - xhat_h)**2).sum(1).mean() / (Xhn.var(0).sum()))
    l0_h = float((fh > 0).float().sum(1).mean())
    feat_active = (fh > 0).float().mean(0).cpu().numpy()
    feat_meanact = (fh.sum(0) / (fh > 0).float().sum(0).clamp_min(1)).cpu().numpy()
    feat_held = fh.reshape(NHELD, SEQL, NFEAT).cpu().numpy()
    del fh, xhat_h
FVE_tr, l0_tr = fve(Wd, We, be, bd, Xn, K)
print(f"BEST-HELD SAE: held FVE {FVE_h:.4f} L0 {l0_h:.1f} (train FVE {FVE_tr:.4f}); dead {(feat_active==0).sum()}/{NFEAT}", flush=True)

# ---- nameability (VERBATIM) ----
held_np = HELD.cpu().numpy()
pos_t = np.tile(np.arange(SEQL), NHELD).reshape(NHELD, SEQL)
is_special = np.isin(held_np, SPECIAL); valid_next = pos_t < (SEQL - 1)
bad_trigger = (pos_t == 0) | is_special | ~valid_next
def conc(ids):
    ids = np.asarray(ids)
    if len(ids) <= 1: return 0.0
    _, c = np.unique(ids, return_counts=True); p = c/c.sum()
    return float(1 - (-(p*np.log(p)).sum())/math.log(len(ids)))
def trigger_class_sig(act2d):
    a = act2d.copy().reshape(-1); a[bad_trigger.reshape(-1)] = -1e30
    if (a > -1e29).sum() < KCAUSAL: return [], 0.0, np.array([], int)
    tk = np.argpartition(a, -KCAUSAL)[-KCAUSAL:]; fs, fp = tk // SEQL, tk % SEQL
    cur = held_np[fs, fp]; classes = VOCAB_CLASS[cur]
    cnt = Counter(classes.tolist()); top = cnt.most_common(6)
    return [(c, int(n)) for c, n in top], round(conc([CIDX[c] for c in classes]), 4), cur
valid_cur = held_np[~bad_trigger]; _bc = Counter(VOCAB_CLASS[valid_cur].tolist()); _bt = sum(_bc.values())
BASE_RATE = {c: _bc.get(c, 0)/_bt for c in CLASS_LIST}
rank_score = feat_active * feat_meanact; order = np.argsort(-rank_score)
live = [int(j) for j in order if feat_active[j] > 0
        and (feat_held[:, :, j].reshape(-1)[~bad_trigger.reshape(-1)] > 0).sum() >= KCAUSAL]
TOPN = 32; top_feats = live[:TOPN]
PURITY_BAR = 0.5; ENRICH_BAR = 2.0; feat_records = []
for j in top_feats:
    tclass_top, tpurity, cur = trigger_class_sig(feat_held[:, :, j])
    vv, cc = np.unique(cur, return_counts=True); oo = np.argsort(-cc)[:6]
    trig_tok_top = [(dec(int(vv[k])), int(cc[k])) for k in oo]
    dom_class = tclass_top[0][0] if tclass_top else None
    dom_frac = (tclass_top[0][1]/KCAUSAL) if tclass_top else 0.0
    enrich = (dom_frac / BASE_RATE[dom_class]) if (dom_class and BASE_RATE[dom_class] > 0) else 0.0
    nameable = bool(tpurity >= PURITY_BAR and enrich >= ENRICH_BAR)
    feat_records.append({'feature': int(j), 'act_freq': round(float(feat_active[j]), 5),
        'mean_active': round(float(feat_meanact[j]), 4), 'trigger_purity': tpurity,
        'trigger_class_top': tclass_top, 'dominant_class': dom_class,
        'dominant_class_frac': round(float(dom_frac), 3), 'dominant_enrichment': round(float(enrich), 2),
        'trigger_token_top': trig_tok_top, 'nameable_by_purity': bool(tpurity >= PURITY_BAR), 'nameable': nameable})
n_nameable_purity = int(sum(r['nameable_by_purity'] for r in feat_records))
n_nameable = int(sum(r['nameable'] for r in feat_records))
print(f"NAMEABILITY (best-held): {n_nameable_purity}/{TOPN} purity>=0.5; {n_nameable}/{TOPN} monosemantic "
      f"(vs SVD 0/32, §78 L1-SAE 23/32, converged-overfit 25/32)", flush=True)

# ---- overwrite npz so the causal script tests THIS best-held SAE ----
np.savez(f'{QK}/qk_sae_converged.npz',
         W_dec=Wd.cpu().numpy(), W_enc=We.cpu().numpy(),
         b_enc=be.cpu().numpy(), b_dec=bd.cpu().numpy(),
         MU=MU.cpu().numpy(), SCALE=np.float32(SCALE),
         top_feats=np.array(top_feats, np.int64), live_feats=np.array(live, np.int64),
         feat_active=feat_active, feat_meanact=feat_meanact,
         K=np.int64(K), NFEAT=np.int64(NFEAT))
print("Overwrote qk_sae_converged.npz with BEST-HELD SAE (for causal script)", flush=True)

out = {
    'meta': {'model': 'bilin18', 'layer': LI, 'selection': 'validation-checkpointed early stopping (val=FW[256:448])',
             'held_slice': 'FW[448:600,:128]', 'chosen_NFEAT': NFEAT, 'chosen_k': K,
             'note': ('convergence overfits: 50000-step run reached train FVE 0.945 but held FVE 0.637 (<§78 0.69). '
                      'Held fidelity is generalisation-bounded on 256 training sequences. This best-held SAE is the '
                      'fairest high-held-fidelity dictionary for the causal test.')},
    'per_config': {f'NFEAT{nf}_k{k}': v for (nf, k), v in runs.items()},
    'reconstruction': {'held_FVE': round(FVE_h, 4), 'held_L0': round(l0_h, 1), 'train_FVE': round(FVE_tr, 4),
                       'dead_features': int((feat_active == 0).sum()),
                       'converged_run_held_FVE': 0.6367, 'sec78_L1_held_FVE': 0.69, 'target_FVE': 0.90,
                       'reached_target': bool(FVE_h >= 0.90)},
    'nameability': {'TOPN': TOPN, 'n_nameable_by_purity': n_nameable_purity, 'n_nameable': n_nameable,
                    'svd_reference': '0/32 (§74)', 'l1_sae_reference': '23/32 (§78)',
                    'converged_overfit_reference': '25/32 (this program, 50k-step run)',
                    'top_feature_records': feat_records},
}
json.dump(out, open(f'{QK}/qk_sae_converged3.json', 'w'), indent=2)
print("Saved qk_sae_converged3.json", flush=True)
print("QK SAE CONVERGED (script 3: best-held) DONE", flush=True)
