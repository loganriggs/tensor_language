"""COALITION red-team of the §80/§81 "unattributable in PARTS" claim -- SCRIPT 2:
JOINT ablation of every candidate coalition vs RANDOM same-size controls, plus an
attribution-guided greedy curve, all against the 5.574-nat full-MLP1 knockout.

For each coalition C: MEAN-ABLATE all members simultaneously
  mo -= sum_{i in C} (f_i - per-position-held-mean_i) * decoder_dir_i
and measure GLOBAL + coalition-TRIGGER delta cross-entropy on the held slice with PAIRED
standard errors, reported as a fraction of the full-MLP1 mean-ablation reference. The §61
control: compare each structured coalition to RANDOM same-size coalitions.

VERDICT: does ANY coalition of <=128 features carry >=25% (>=50%) of the hub effect above
its random control? If yes, collective computation is attributable at coalition
granularity (news vs §81). If even coalition search fails, §81's open problem hardens.

FORWARD + encode() + per-position mean-ablation + paired-SE convention COPIED VERBATIM
from qk_sae_moredata_2.py. Held-back canonical FW[448:600,:128].
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
KCAUSAL = 200

_special = {tok.eos_token_id}
for _t in range(V):
    if '�' in tok.decode([_t]): _special.add(_t)
SPECIAL = np.array(sorted(_special))

# ---------------- dictionary ----------------
Z = np.load(f'{QK}/qk_sae_moredata.npz')
W_dec = torch.from_numpy(Z['W_dec']).to(DEV)
W_enc = torch.from_numpy(Z['W_enc']).to(DEV)
b_enc = torch.from_numpy(Z['b_enc']).to(DEV)
b_dec = torch.from_numpy(Z['b_dec']).to(DEV)
MU = torch.from_numpy(Z['MU']).to(DEV)
SCALE = float(Z['SCALE'])
NFEAT = int(Z['NFEAT']); K = int(Z['K'])
LIVE = np.array([int(j) for j in Z['live_feats']]); NLIVE = len(LIVE)
DEC_ORIG = (W_dec.T / SCALE).contiguous()
dec_norm = DEC_ORIG.norm(dim=1).cpu().numpy()

# ---------------- candidate coalitions ----------------
C = np.load(f'{QK}/qk_coalition_cands.npz')
cand_names = [k[6:] for k in C.files if k.startswith('cand__')]
cands = {nm: [int(x) for x in C[f'cand__{nm}']] for nm in cand_names}
attr_per_pos = C['attr_per_pos']; order_attr = C['order_attr']
# attribution-guided greedy curve: cumulative top-N of POSITIVE-attribution features
pos_local = [int(j) for j in order_attr if attr_per_pos[int(j)] > 0]
def live_ids(local): return [int(LIVE[j]) for j in local]
for S in [8, 16, 32, 64, 128, 256, 512]:
    if S <= len(pos_local):
        cands[f'greedy_attr_{S}'] = live_ids(pos_local[:S])
cands['greedy_attr_ALLPOS'] = live_ids(pos_local)          # all positive-attribution features
print(f"{len(cands)} coalitions to evaluate; {len(pos_local)} positive-attribution features", flush=True)

def encode(mo):
    pre = F.relu(((mo - MU) * SCALE - b_dec) @ W_enc.T + b_enc)
    vals, idx = pre.topk(K, dim=-1)
    return torch.zeros_like(pre).scatter_(-1, idx, vals)

@torch.no_grad()
def forward(idx, collect=False, feat_abl=None):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    fcollect = None
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
            if collect: fcollect = encode(mo)
            if feat_abl is not None:
                ids, FMEAN = feat_abl
                f = encode(mo)
                dev = f[:, :, ids] - FMEAN[:, ids].unsqueeze(0)
                mo = mo - torch.einsum('btk,kd->btd', dev, DEC_ORIG[ids])
        x = x + mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    return (logits, fcollect) if collect else logits

@torch.no_grad()
def forward_fullabl(idx, MOMEAN=None, collect_mo=False):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool)); mo_at = None
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
            if collect_mo: mo_at = mo
            if MOMEAN is not None: mo = MOMEAN.unsqueeze(0).expand(B, -1, -1)
        x = x + mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    return (logits, mo_at) if collect_mo else logits

def stats(s, sq, n):
    if n <= 1: return 0.0, 0.0
    mean = s/n; var = max(sq/n - mean*mean, 0.0)*n/(n-1)
    return mean, math.sqrt(var/n)

# =====================================================================================
# PASS A: per-position feature means (all NFEAT) + live activation cache (for triggers)
# =====================================================================================
print("PASS A: per-position means + activation cache ...", flush=True)
FSUM = torch.zeros(SEQL, NFEAT, device=DEV)
act_live = np.zeros((NHELD, SEQL, NLIVE), np.float32)
for i in range(0, NHELD, BATCH):
    _, f = forward(HELD[i:i+BATCH], collect=True)
    FSUM += f.sum(0); b = f.shape[0]
    act_live[i:i+b] = f[:, :, LIVE].cpu().numpy()
FMEAN = (FSUM / NHELD).contiguous(); del FSUM
FMEAN_live = FMEAN[:, LIVE].cpu().numpy()
print("PASS A done.", flush=True)

held_np = HELD.cpu().numpy()
pos_t = np.tile(np.arange(SEQL), NHELD).reshape(NHELD, SEQL)
is_special = np.isin(held_np, SPECIAL)
valid_next = pos_t < (SEQL - 1)
bad_trigger = (pos_t == 0) | is_special | ~valid_next
tgt_all = torch.from_numpy(held_np).to(DEV)

# per-coalition trigger mask: top-KCAUSAL valid positions by coalition mo-space activation
# energy = sum_{i in C} (dev_i * dec_norm_i)^2
LIVE_pos = {int(f): k for k, f in enumerate(LIVE)}
dev_live = act_live - FMEAN_live[None, :, :]            # (NHELD,SEQL,NLIVE)
decnorm_live = dec_norm[LIVE]
def coalition_trigger(ids):
    loc = [LIVE_pos[j] for j in ids if j in LIVE_pos]
    if not loc: return None
    w = (dev_live[:, :, loc] * decnorm_live[loc][None, None, :]) ** 2
    e = w.sum(axis=2).reshape(-1)                      # (NHELD*SEQL,)
    e[bad_trigger.reshape(-1)] = -1e30
    kk = min(KCAUSAL, int((e > -1e29).sum()))
    tk = np.argpartition(e, -kk)[-kk:]
    mk = np.zeros(NHELD*SEQL, bool); mk[tk] = True
    return mk.reshape(NHELD, SEQL)
trig = {nm: coalition_trigger(ids) for nm, ids in cands.items()}

# =====================================================================================
# EVAL: joint ablation of each coalition (global + trigger dCE, paired SE)
# =====================================================================================
names = list(cands.keys())
g_sum = {n: 0.0 for n in names}; g_sq = {n: 0.0 for n in names}; g_n = {n: 0 for n in names}
t_sum = {n: 0.0 for n in names}; t_sq = {n: 0.0 for n in names}; t_n = {n: 0 for n in names}
ids_gpu = {n: cands[n] for n in names}
print(f"EVAL: joint-ablating {len(names)} coalitions ...", flush=True)
t0 = time.time()
for bi, i in enumerate(range(0, NHELD, BATCH)):
    sb = slice(i, min(i+BATCH, NHELD)); idx = HELD[sb]
    base = forward(idx).float(); tgt = tgt_all[sb]
    logp = F.log_softmax(base[:, :SEQL-1], dim=-1)
    base_ce = -logp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1); del logp
    vmask = torch.from_numpy(valid_next[sb, :SEQL-1]).to(DEV)
    for n in names:
        abl = forward(idx, feat_abl=(ids_gpu[n], FMEAN)).float()
        alogp = F.log_softmax(abl[:, :SEQL-1], dim=-1)
        abl_ce = -alogp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1); del alogp
        dce = (abl_ce - base_ce); dg = dce[vmask]
        g_sum[n] += float(dg.sum()); g_sq[n] += float((dg*dg).sum()); g_n[n] += int(dg.numel())
        if trig[n] is not None:
            tm = torch.from_numpy(trig[n][sb, :SEQL-1]).to(DEV)
            if tm.any():
                dt = dce[tm]
                t_sum[n] += float(dt.sum()); t_sq[n] += float((dt*dt).sum()); t_n[n] += int(dt.numel())
        del abl, dce
    if bi % 5 == 0:
        print(f"  batch {bi+1}/{(NHELD+BATCH-1)//BATCH}  elapsed {time.time()-t0:.0f}s", flush=True)
    del base, base_ce
print(f"EVAL done in {time.time()-t0:.0f}s", flush=True)

# =====================================================================================
# full-MLP1 mean-ablation reference (self-contained)
# =====================================================================================
print("full-MLP1 mean-ablation reference ...", flush=True)
MOSUM = torch.zeros(SEQL, D, device=DEV)
for i in range(0, NHELD, BATCH):
    _, mo = forward_fullabl(HELD[i:i+BATCH], collect_mo=True); MOSUM += mo.sum(0)
MOMEAN = MOSUM / NHELD
full_sum = full_sq = 0.0; full_n = 0
for i in range(0, NHELD, BATCH):
    sb = slice(i, min(i+BATCH, NHELD)); idx = HELD[sb]; tgt = tgt_all[sb]
    base = forward_fullabl(idx).float(); abl = forward_fullabl(idx, MOMEAN=MOMEAN).float()
    blp = F.log_softmax(base[:, :SEQL-1], -1); alp = F.log_softmax(abl[:, :SEQL-1], -1)
    bce = -blp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1)
    ace = -alp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1)
    vm = torch.from_numpy(valid_next[sb, :SEQL-1]).to(DEV)
    d = (ace - bce)[vm]
    full_sum += float(d.sum()); full_sq += float((d*d).sum()); full_n += int(d.numel())
    del base, abl
full_mean, full_se = stats(full_sum, full_sq, full_n)
print(f"full-MLP1 mean-ablation global dCE {full_mean:.4f} +- {full_se:.4f}", flush=True)

# =====================================================================================
# ASSEMBLE + random baselines per size + verdict
# =====================================================================================
recs = {}
for n in names:
    gm, gse = stats(g_sum[n], g_sq[n], g_n[n])
    tm_, tse = stats(t_sum[n], t_sq[n], t_n[n])
    recs[n] = {'size': len(cands[n]),
               'global_dCE': round(gm, 5), 'global_dCE_SE': round(gse, 5),
               'global_frac_of_full': round(gm/full_mean, 4) if full_mean > 0 else None,
               'trigger_dCE': round(tm_, 5), 'trigger_dCE_SE': round(tse, 5),
               'trigger_frac_of_full': round(tm_/full_mean, 4) if full_mean > 0 else None}

# random baseline per size = mean of random_{S}_s* coalitions
from collections import defaultdict
rand_by_size = defaultdict(list)
for n in names:
    if n.startswith('random_'):
        S = recs[n]['size']; rand_by_size[S].append(recs[n]['global_frac_of_full'])
rand_mean = {S: float(np.mean(v)) for S, v in rand_by_size.items()}

# best structured coalition (non-random) at each size <=128, and overall
structured = {n: r for n, r in recs.items() if not n.startswith('random_')}
best_le128 = max((r for n, r in structured.items() if r['size'] <= 128),
                 key=lambda r: r['global_frac_of_full'])
best_le128_name = [n for n, r in structured.items() if r is best_le128][0]
best_overall = max(structured.items(), key=lambda kv: kv[1]['global_frac_of_full'])

def rand_ref(size):
    # nearest random size
    if not rand_mean: return None
    S = min(rand_mean, key=lambda s: abs(s-size)); return rand_mean[S]

verdict = {
    'full_MLP1_ref_nats': round(full_mean, 5), 'full_MLP1_ref_SE': round(full_se, 5),
    'best_coalition_le128': {'name': best_le128_name, **best_le128,
                             'random_same_size_frac': rand_ref(best_le128['size'])},
    'best_coalition_overall': {'name': best_overall[0], **best_overall[1],
                               'random_same_size_frac': rand_ref(best_overall[1]['size'])},
    'any_le128_reaches_25pct': bool(best_le128['global_frac_of_full'] >= 0.25),
    'any_le128_reaches_50pct': bool(best_le128['global_frac_of_full'] >= 0.50),
    'random_baseline_frac_by_size': rand_mean,
    'all_live_frac_sec80': 0.0201,
}
out = {
    'meta': {'held_slice': 'FW[448:600,:128]', 'BATCH': BATCH, 'KCAUSAL': KCAUSAL,
             'NFEAT': NFEAT, 'NLIVE': int(NLIVE), 'k_active': K,
             'ablation': 'joint mean-ablation of coalition: mo -= sum_i (f_i - per-pos-held-mean_i)*dec_i',
             'currency': 'global mean-ablation delta cross-entropy per valid held position (nats), paired SE; attention intact',
             'reference': 'full-MLP1 mean-ablation knockout (§78/§79 = 5.57 nats)',
             'forward': 'VERBATIM qk_sae_moredata_2.py'},
    'verdict': verdict,
    'coalitions': recs,
}
json.dump(out, open(f'{QK}/qk_coalition_attr.json', 'w'), indent=2)

# ---- print report ----
print("\n===== COALITION JOINT-ABLATION (global delta cross-entropy, fraction of full-MLP1) =====", flush=True)
print(f"full-MLP1 reference {full_mean:.4f} nats; all-1011-live (§80) = 2.01%\n", flush=True)
order = sorted(structured.items(), key=lambda kv: -kv[1]['global_frac_of_full'])
print(f"{'coalition':22s} {'size':>5s} {'globalDCE':>10s} {'%full':>7s} {'rand%':>7s} {'trigDCE':>8s}", flush=True)
for n, r in order[:24]:
    rr = rand_ref(r['size'])
    print(f"{n:22s} {r['size']:5d} {r['global_dCE']:10.4f} {r['global_frac_of_full']*100:6.2f}% "
          f"{(rr*100 if rr is not None else 0):6.2f}% {r['trigger_dCE']:8.4f}", flush=True)
print("\nRANDOM same-size baselines (global %full):", flush=True)
for S in sorted(rand_mean): print(f"  size {S:4d}: {rand_mean[S]*100:.3f}%", flush=True)
print(f"\nBEST coalition <=128: {best_le128_name} size {best_le128['size']} "
      f"= {best_le128['global_frac_of_full']*100:.2f}% of full "
      f"(random same size {rand_ref(best_le128['size'])*100:.2f}%)", flush=True)
print(f"BEST coalition overall: {best_overall[0]} size {best_overall[1]['size']} "
      f"= {best_overall[1]['global_frac_of_full']*100:.2f}% of full", flush=True)
print(f"\nVERDICT: any <=128-feature coalition >=25% of full? {verdict['any_le128_reaches_25pct']}; "
      f">=50%? {verdict['any_le128_reaches_50pct']}", flush=True)
print("Saved qk_coalition_attr.json", flush=True)
print("QK COALITION ATTR (script 2) DONE", flush=True)
