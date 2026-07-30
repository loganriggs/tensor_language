"""§79 CAVEAT-CLOSER -- SCRIPT 1: re-run the converged TopK SAE on the MLP-layer-1 hub of
bilin18 with ~10x more TRAINING data, to settle whether §79's held-back fraction-of-variance-
explained CEILING of ~0.72 (which DEGRADED with more training on only 256 sequences) is a
DATA bound or a genuine high-rank bound.

§79 (qk_sae_converged_3.py) trained on FW[0:256] (256 sequences); its best held FVE was
0.7162 (NFEAT=8192,k=64) reached at only ~1000 steps, after which held fidelity DEGRADED --
classic small-corpus overfitting. §79 flagged: "a much larger activation corpus might raise
achievable held fidelity."

A 10x larger FineWeb token file (data_fineweb_cooc_tokens.npy, 6000x513) is now available.
This script:
  TRAIN = cooc[600:6000]  (5400 sequences ~= 21x §79's 256)   -- 10x+ activation corpus
  VAL   = cooc[0:300]     (checkpoint-selection slice, disjoint)
  HELD  = canonical FW[448:600]  (the SAME slice §74-§79 evaluated -- exact comparability)
All three slices are pairwise DISJOINT (verified: cooc has ZERO row overlap with canonical).

Validation-checkpointed early stopping keeps the best-VAL-FVE checkpoint and logs the full
VAL trajectory (does 10x data remove the early-peak-then-degrade overfitting §79 saw?).
Sweeps NFEAT in {4096,8192,16384} at k=64 with AuxK dead-feature revival, then reports
train / val / held FVE, best held FVE vs §79's 0.72, L0, dead count, train/held gap, and the
§79 NAMEABILITY head-to-head. Saves the best-held dictionary to qk_sae_moredata.npz.

MEMORY: bilin18 is 545M params (2.18 GB fp32 on GPU), so the 5400-sequence activation cache
(~1.6 GB fp16) lives on CPU and streams minibatches to GPU; FVE / held-encode are chunked.
Footprint stays < 4 GB. GPU guard polls memory.free.

ALL model-forward / MLP1 extraction / normalisation / class library / topk_encode /
nameability machinery COPIED VERBATIM from qk_sae_converged_3.py (which copies
qk_sae_converged.py / qk_redteam_sae_hub.py / qk_mlp1_tail.py). Only data plumbing (10x
corpus, CPU cache, chunked FVE) changed. tier2_model.load_elriggs('bilin18').
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

# ---------------- GPU GUARD (verbatim) ----------------
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

# ---------------- DATA: 10x training corpus (cooc), canonical held eval ----------------
COOC  = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
CANON = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
SEQL = 128
TRAIN = COOC[600:6000, :SEQL].to(DEV)     # 5400 sequences ~= 21x §79's 256; the 10x corpus
VAL   = COOC[0:300,    :SEQL].to(DEV)     # checkpoint-selection slice (disjoint from train + held)
HELD  = CANON[448:600, :SEQL].to(DEV)     # the SAME canonical held slice §74-§79 used
NTRAIN = TRAIN.shape[0]; NHELD = HELD.shape[0]; NVAL = VAL.shape[0]
BATCH = 8
KCAUSAL = 200
print(f"TRAIN {NTRAIN} seq (cooc[600:6000]) | VAL {NVAL} seq (cooc[0:300]) | HELD {NHELD} seq (canon[448:600])", flush=True)

_special = {tok.eos_token_id}
for _t in range(V):
    if '�' in tok.decode([_t]): _special.add(_t)
SPECIAL = np.array(sorted(_special))
print(f"{len(SPECIAL)} special token ids masked", flush=True)

# ------- class library (VERBATIM from qk_sae_converged_3.py) -------
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

# ------- MLP1 output collection (VERBATIM) -------
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

# ------- cache TRAIN to CPU (fp16, memory-bounded); VAL/HELD to GPU (fp32, small) -------
def cache_cpu_fp16(slc):
    N = slc.shape[0] * SEQL
    out = torch.empty(N, D, dtype=torch.float16)   # CPU
    w = 0
    for i in range(0, slc.shape[0], BATCH):
        mo = collect_mo(slc[i:i+BATCH]).reshape(-1, D).float()
        out[w:w+mo.shape[0]] = mo.half().cpu(); w += mo.shape[0]
    return out
def cache_gpu_fp32(slc):
    out = []
    for i in range(0, slc.shape[0], BATCH):
        out.append(collect_mo(slc[i:i+BATCH]).reshape(-1, D).float())
    return torch.cat(out, 0).contiguous()

print("Caching activations: TRAIN->CPU fp16 (10x corpus) ...", flush=True)
t0 = time.time()
Xtr_cpu = cache_cpu_fp16(TRAIN)                     # (NTRAIN*SEQL, D) fp16 CPU
print(f"  TRAIN cache {tuple(Xtr_cpu.shape)} fp16 CPU ({Xtr_cpu.numel()*2/1e9:.2f} GB) in {time.time()-t0:.0f}s", flush=True)

# normalisation (VERBATIM convention) computed in fp32 chunks over the CPU cache
Ntr = Xtr_cpu.shape[0]
CH = 32768
ssum = torch.zeros(D, device=DEV)
for i in range(0, Ntr, CH):
    ssum += Xtr_cpu[i:i+CH].to(DEV).float().sum(0)
MU = (ssum / Ntr)
nrm = 0.0
for i in range(0, Ntr, CH):
    nrm += (Xtr_cpu[i:i+CH].to(DEV).float() - MU).norm(dim=1).sum().item()
resid_norm = nrm / Ntr
SCALE = float(math.sqrt(D) / resid_norm)
print(f"normalisation: mean centred-norm {resid_norm:.3f}, scale {SCALE:.5f}", flush=True)

# val/held: cache then normalise on GPU (fp32, small)
Xvn = ((cache_gpu_fp32(VAL)  - MU) * SCALE).contiguous()
Xhn = ((cache_gpu_fp32(HELD) - MU) * SCALE).contiguous()
print(f"caches: train(CPU) {tuple(Xtr_cpu.shape)}  val {tuple(Xvn.shape)}  held {tuple(Xhn.shape)}", flush=True)

# =====================================================================================
# TOP-K SAE (VERBATIM topk_encode) + chunked FVE + validation-checkpointed early stop.
# minibatches stream from the CPU fp16 cache, normalised on the fly.
# =====================================================================================
def topk_encode(x, We, be, bd, k):
    pre = F.relu((x - bd) @ We.T + be)
    vals, idx = pre.topk(k, dim=-1)
    f = torch.zeros_like(pre).scatter_(-1, idx, vals)
    return f, pre

@torch.no_grad()
def fve_chunked(Wd, We, be, bd, X, k, chunk=16384):
    """X already normalised, on GPU. FVE = 1 - sum((X-xhat)^2)/N / X.var(0).sum() (matches VERBATIM)."""
    N = X.shape[0]; sse = 0.0; act = 0.0
    denom = float(X.var(0).sum())
    for i in range(0, N, chunk):
        xb = X[i:i+chunk]
        f, _ = topk_encode(xb, We, be, bd, k)
        xhat = f @ Wd.T + bd
        sse += float(((xb - xhat)**2).sum())
        act += float((f > 0).float().sum())
    return 1.0 - (sse / N) / denom, act / N

def sample_minibatch(bs, g):
    idx = torch.randint(0, Ntr, (bs,), device='cpu', generator=g)
    xb = Xtr_cpu[idx].to(DEV).float()
    return (xb - MU) * SCALE

def train_bestval(nfeat, k, steps=25000, lr=4e-4, bs=4096, seed=0, aux_coef=1/32,
                  k_aux=256, dead_steps=2500, warmup=200, val_every=1000):
    g = torch.Generator(device=DEV); g.manual_seed(seed)
    Wd = torch.randn(D, nfeat, device=DEV, generator=g); Wd = Wd / Wd.norm(dim=0, keepdim=True)
    We = Wd.T.clone(); be = torch.zeros(nfeat, device=DEV); bd = torch.zeros(D, device=DEV)
    for t in (Wd, We, be, bd): t.requires_grad_(True)
    opt = torch.optim.Adam([Wd, We, be, bd], lr=lr)
    def lr_at(step):
        if step < warmup: return lr*(step+1)/warmup
        prog = (step-warmup)/max(1, steps-warmup)
        return lr*(0.05+0.95*0.5*(1+math.cos(math.pi*prog)))
    gg = torch.Generator(device='cpu'); gg.manual_seed(seed+1)
    last_fired = torch.zeros(nfeat, device=DEV)
    best_val = -1e9; best_ckpt = None; best_step = -1; traj = []
    for step in range(steps):
        for pg in opt.param_groups: pg['lr'] = lr_at(step)
        xb = sample_minibatch(bs, gg)
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
            vfve, _ = fve_chunked(Wd.detach(), We.detach(), be.detach(), bd.detach(), Xvn, k)
            traj.append((step, round(vfve, 4)))
            if vfve > best_val:
                best_val = vfve; best_step = step
                best_ckpt = tuple(t.detach().clone() for t in (Wd, We, be, bd))
            if step % (val_every*5) == 0 or step == steps-1:
                print(f"    NFEAT={nfeat} step {step:6d}  val FVE {vfve:.4f}  (best {best_val:.4f}@{best_step})  lr {lr_at(step):.2e}", flush=True)
    return best_ckpt, best_val, best_step, traj

print("Best-validation-FVE sweep (k=64) on 10x corpus ...", flush=True)
CONFIGS = [(4096, 64), (8192, 64), (16384, 64)]
STEPS = 25000
best = None; runs = {}
for nfeat, k in CONFIGS:
    tcfg = time.time()
    ckpt, vbest, vstep, traj = train_bestval(nfeat, k, steps=STEPS, seed=0)
    hfve, hl0 = fve_chunked(*ckpt, Xhn, k)
    # train FVE on a fixed 32768-position subset of the CPU cache (normalised)
    with torch.no_grad():
        sub = Xtr_cpu[:32768].to(DEV).float(); sub = ((sub - MU) * SCALE)
        trfve, _ = fve_chunked(*ckpt, sub, k); del sub
    runs[(nfeat, k)] = {'val_FVE': round(vbest, 4), 'val_step': vstep,
                        'held_FVE': round(hfve, 4), 'held_L0': round(hl0, 1),
                        'train_FVE': round(trfve, 4), 'val_trajectory': traj}
    print(f"  NFEAT={nfeat} k={k}: best val FVE {vbest:.4f}@{vstep}; held FVE {hfve:.4f} L0 {hl0:.1f}; "
          f"train FVE {trfve:.4f} ({time.time()-tcfg:.0f}s)", flush=True)
    if best is None or vbest > best[1]:
        best = ((nfeat, k), vbest, ckpt)
    torch.cuda.empty_cache()

(NFEAT, K), _, CK = best
Wd, We, be, bd = CK
print(f"CHOSEN best-VAL config: NFEAT={NFEAT} k={K}", flush=True)

# ------- held eval + per-feature acts for nameability (chunked held-encode -> CPU) -------
Wd_c, We_c, be_c, bd_c = Wd.contiguous(), We.contiguous(), be.contiguous(), bd.contiguous()
Nh = Xhn.shape[0]
feat_held = np.zeros((Nh, NFEAT), np.float32)
act_cnt = np.zeros(NFEAT, np.float64); act_sum = np.zeros(NFEAT, np.float64); sse_h = 0.0
denom_h = float(Xhn.var(0).sum())
with torch.no_grad():
    for i in range(0, Nh, 16384):
        xb = Xhn[i:i+16384]
        fh, _ = topk_encode(xb, We_c, be_c, bd_c, K)
        xhat = fh @ Wd_c.T + bd_c
        sse_h += float(((xb - xhat)**2).sum())
        fhn = fh.cpu().numpy()
        feat_held[i:i+fhn.shape[0]] = fhn
        act_cnt += (fhn > 0).sum(0); act_sum += fhn.sum(0)
FVE_h = 1.0 - (sse_h / Nh) / denom_h
l0_h = float((feat_held > 0).sum(1).mean())
feat_active = (act_cnt / Nh).astype(np.float32)
feat_meanact = (act_sum / np.clip(act_cnt, 1, None)).astype(np.float32)
feat_held = feat_held.reshape(NHELD, SEQL, NFEAT)
dead_features = int((feat_active == 0).sum())
with torch.no_grad():
    sub = ((Xtr_cpu[:32768].to(DEV).float() - MU) * SCALE)
    FVE_tr, l0_tr = fve_chunked(Wd, We, be, bd, sub, K); del sub
print(f"BEST-VAL SAE: held FVE {FVE_h:.4f} L0 {l0_h:.1f} (train FVE {FVE_tr:.4f}); dead {dead_features}/{NFEAT}", flush=True)
print(f"§79 ceiling was 0.7162 -- 10x data held FVE {FVE_h:.4f} => "
      f"{'EXCEEDS' if FVE_h > 0.7162 else 'does NOT exceed'} the ceiling", flush=True)

# =====================================================================================
# NAMEABILITY head-to-head (VERBATIM from qk_sae_converged_3.py).
# =====================================================================================
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
print(f"NAMEABILITY (10x data): {n_nameable_purity}/{TOPN} purity>=0.5; {n_nameable}/{TOPN} monosemantic "
      f"(vs SVD 0/32, §79 26/32)", flush=True)

# ------- save best-held dictionary for the causal script -------
np.savez(f'{QK}/qk_sae_moredata.npz',
         W_dec=Wd.cpu().numpy(), W_enc=We.cpu().numpy(),
         b_enc=be.cpu().numpy(), b_dec=bd.cpu().numpy(),
         MU=MU.cpu().numpy(), SCALE=np.float32(SCALE),
         top_feats=np.array(top_feats, np.int64), live_feats=np.array(live, np.int64),
         feat_active=feat_active, feat_meanact=feat_meanact,
         K=np.int64(K), NFEAT=np.int64(NFEAT))
print("Saved qk_sae_moredata.npz (best-VAL SAE, for causal script)", flush=True)

out = {
    'meta': {'model': 'bilin18', 'layer': LI, 'quantity': 'MLP1 feed-forward OUTPUT (residual write mo), dim=D=1152',
             'purpose': ('§79 caveat-closer: re-run converged TopK SAE with ~10x more TRAINING data to settle whether '
                         'the ~0.72 held-FVE ceiling (that DEGRADED with training on 256 seqs) is a DATA bound or a '
                         'genuine high-rank bound; and whether the causal negative survives at any fidelity reached.'),
             'train_slice': 'cooc[600:6000,:128] (5400 seq ~=21x §79 256)', 'val_slice': 'cooc[0:300,:128]',
             'held_slice': 'canonical FW[448:600,:128] (SAME as §74-§79)',
             'overlap_check': 'cooc has ZERO row overlap with canonical (incl. the held slice) -- verified',
             'selection': 'validation-checkpointed early stopping (best VAL FVE)',
             'sweep_configs': [f'NFEAT{nf}_k{k}' for nf, k in CONFIGS], 'steps': STEPS,
             'aux': 'AuxK dead-feature revival (dead>2500 steps reconstruct residual, k_aux=256, coef 1/32)',
             'train_cache': f'{Ntr} positions, fp16 on CPU, minibatches streamed to GPU (footprint<4GB)',
             'forward': 'VERBATIM qk_sae_converged_3.py / qk_redteam_sae_hub.py / qk_mlp1_tail.py',
             'chosen_NFEAT': int(NFEAT), 'chosen_k': int(K)},
    'per_config': {f'NFEAT{nf}_k{k}': v for (nf, k), v in runs.items()},
    'base_rate': {c: round(BASE_RATE[c], 4) for c in CLASS_LIST},
    'reconstruction': {'held_FVE': round(FVE_h, 4), 'held_L0': round(l0_h, 1), 'train_FVE': round(FVE_tr, 4),
                       'train_held_gap': round(FVE_tr - FVE_h, 4), 'dead_features': dead_features,
                       'sec79_ceiling_held_FVE': 0.7162, 'exceeds_sec79_ceiling': bool(FVE_h > 0.7162),
                       'target_FVE': 0.90, 'reached_target': bool(FVE_h >= 0.90)},
    'nameability': {'TOPN': TOPN, 'purity_bar': PURITY_BAR, 'enrich_bar': ENRICH_BAR,
                    'n_nameable_by_purity': n_nameable_purity, 'n_nameable': n_nameable,
                    'svd_reference': '0/32 (§74)', 'sec79_reference': '26/32 (§79, held FVE 0.7162)',
                    'top_feature_records': feat_records},
}
json.dump(out, open(f'{QK}/qk_sae_moredata.json', 'w'), indent=2)
print("Saved qk_sae_moredata.json", flush=True)
print("QK SAE MOREDATA (script 1: train + fidelity + nameability) DONE", flush=True)
