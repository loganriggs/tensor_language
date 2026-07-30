"""§79 CAVEAT-CLOSER -- SCRIPT 2: the CAUSAL test at the 10x-data fidelity.

Loads the best-VAL TopK dictionary trained on the 10x corpus by qk_sae_moredata.py and asks
whether its features are CAUSALLY load-bearing on the canonical held slice FW[448:600] -- the
decisive head-to-head with §74/§78/§79, where 0 of 32 SVD directions and 0 of 32 SAE features
cleared the causal bar (trigger delta cross-entropy z>=3 AND >= 0.02 nats).

For each top-32 feature: MEAN-ABLATE its contribution to the MLP1 output residual write
(mo -= (f_i - per-position-held-mean_i) * decoder_direction_i) and measure global + trigger-
restricted delta cross-entropy with PAIRED standard errors. Cumulative sweep vs the full-MLP1
mean-ablation reference (5.57 nats in §78/§79). Positive controls: remove SAE reconstruction
vs remove SAE residual. Merges its results into qk_sae_moredata.json under 'causal'.

FORWARD + per-position mean-ablation + paired-SE convention + TopK encode() COPIED VERBATIM
from qk_sae_converged_2.py / qk_redteam_sae_hub_2.py / qk_mlp1_tail.py. Only the loaded npz
filename (qk_sae_moredata.npz) changed. Held-back canonical FW[448:600,:128].
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
HELD = FINEWEB[448:600, :SEQL].to(DEV)     # SAME canonical held slice as §74-§79
NHELD = HELD.shape[0]
BATCH = 6
KCAUSAL = 200

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
LIVE_FEATS = [int(j) for j in Z['live_feats']]
print(f"loaded 10x-data dictionary NFEAT={NFEAT} k={K}; testing top-{len(TOP_FEATS)} features causally", flush=True)

DEC_ORIG = (W_dec.T / SCALE).contiguous()

def encode(mo):
    pre = F.relu(((mo - MU) * SCALE - b_dec) @ W_enc.T + b_enc)
    vals, idx = pre.topk(K, dim=-1)
    f = torch.zeros_like(pre).scatter_(-1, idx, vals)
    return f

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
            if collect:
                fcollect = encode(mo)
            if feat_abl is not None:
                ids, FMEAN = feat_abl
                f = encode(mo)
                dev = f[:, :, ids] - FMEAN[:, ids].unsqueeze(0)
                mo = mo - torch.einsum('btk,kd->btd', dev, DEC_ORIG[ids])
        x = x + mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    return (logits, fcollect) if collect else logits

# PASS A
print("PASS A: per-position feature means + tested-feature activations ...", flush=True)
FSUM = torch.zeros(SEQL, NFEAT, device=DEV)
tested = TOP_FEATS
act_tested = np.zeros((len(tested), NHELD, SEQL), np.float32)
for i in range(0, NHELD, BATCH):
    _, f = forward(HELD[i:i+BATCH], collect=True)
    FSUM += f.sum(0)
    fnp = f[:, :, tested].cpu().numpy()
    b = fnp.shape[0]
    for k in range(len(tested)):
        act_tested[k, i:i+b] = fnp[:, :, k]
FMEAN = FSUM / NHELD
del FSUM
print("PASS A done.", flush=True)

held_np = HELD.cpu().numpy()
pos_t = np.tile(np.arange(SEQL), NHELD).reshape(NHELD, SEQL)
is_special = np.isin(held_np, SPECIAL)
valid_next = pos_t < (SEQL - 1)
bad_trigger = (pos_t == 0) | is_special | ~valid_next

trig_mask = {}
for k, j in enumerate(tested):
    a = act_tested[k].copy().reshape(-1)
    a[bad_trigger.reshape(-1)] = -1e30
    tk = np.argpartition(a, -KCAUSAL)[-KCAUSAL:]
    mk = np.zeros(NHELD*SEQL, bool); mk[tk] = True
    trig_mask[j] = mk.reshape(NHELD, SEQL)

def stats(s, sq, n):
    if n <= 1: return 0.0, 0.0
    mean = s/n; var = max(sq/n - mean*mean, 0.0)*n/(n-1)
    return mean, math.sqrt(var/n)

# PASS B
CUM_NS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, len(LIVE_FEATS)]
CUM_NS = sorted(set([n for n in CUM_NS if n <= len(LIVE_FEATS)]))
cum_sets = {n: LIVE_FEATS[:n] for n in CUM_NS}

g_sum = {j: 0.0 for j in tested}; g_sq = {j: 0.0 for j in tested}; g_n = {j: 0 for j in tested}
t_sum = {j: 0.0 for j in tested}; t_sq = {j: 0.0 for j in tested}; t_n = {j: 0 for j in tested}
t_pos = {j: 0 for j in tested}
cum_g_sum = {n: 0.0 for n in CUM_NS}; cum_g_sq = {n: 0.0 for n in CUM_NS}; cum_g_n = {n: 0 for n in CUM_NS}

FMEAN_all = FMEAN.contiguous()
tgt_all = torch.from_numpy(held_np).to(DEV)
print(f"PASS B: {len(tested)} single-feature ablations + {len(CUM_NS)} cumulative ...", flush=True)
t0 = time.time()
for bi, i in enumerate(range(0, NHELD, BATCH)):
    sb = slice(i, min(i+BATCH, NHELD)); idx = HELD[sb]; b = idx.shape[0]
    base = forward(idx).float()
    tgt = tgt_all[sb]
    logp = F.log_softmax(base[:, :SEQL-1], dim=-1)
    base_ce = -logp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1); del logp
    vmask = torch.from_numpy(valid_next[sb, :SEQL-1]).to(DEV)
    for j in tested:
        abl = forward(idx, feat_abl=([j], FMEAN_all)).float()
        alogp = F.log_softmax(abl[:, :SEQL-1], dim=-1)
        abl_ce = -alogp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1); del alogp
        dce = (abl_ce - base_ce); dg = dce[vmask]
        g_sum[j] += float(dg.sum()); g_sq[j] += float((dg*dg).sum()); g_n[j] += int(dg.numel())
        tm = torch.from_numpy(trig_mask[j][sb, :SEQL-1]).to(DEV)
        if tm.any():
            dt = dce[tm]
            t_sum[j] += float(dt.sum()); t_sq[j] += float((dt*dt).sum())
            t_n[j] += int(dt.numel()); t_pos[j] += int((dt > 0).sum())
        del abl, dce
    for n in CUM_NS:
        abl = forward(idx, feat_abl=(cum_sets[n], FMEAN_all)).float()
        alogp = F.log_softmax(abl[:, :SEQL-1], dim=-1)
        abl_ce = -alogp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1); del alogp
        dg = (abl_ce - base_ce)[vmask]
        cum_g_sum[n] += float(dg.sum()); cum_g_sq[n] += float((dg*dg).sum()); cum_g_n[n] += int(dg.numel())
        del abl
    if bi % 4 == 0:
        print(f"  batch {bi+1}/{(NHELD+BATCH-1)//BATCH}  elapsed {time.time()-t0:.0f}s", flush=True)
    del base, base_ce
print(f"PASS B done in {time.time()-t0:.0f}s", flush=True)

# PASS B2: full-MLP1 ref + positive controls
print("PASS B2: full-MLP1 mean-ablation reference + positive controls ...", flush=True)
@torch.no_grad()
def forward_fullabl(idx, MOMEAN=None, collect_mo=False, recon_only=False):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    mo_at = None
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
            if recon_only:
                f = encode(mo)
                xhat_n = f @ W_dec.T + b_dec
                mo = xhat_n / SCALE + MU
            if MOMEAN is not None: mo = MOMEAN.unsqueeze(0).expand(B, -1, -1)
        x = x + mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    return (logits, mo_at) if collect_mo else logits

MOSUM = torch.zeros(SEQL, D, device=DEV)
for i in range(0, NHELD, BATCH):
    _, mo = forward_fullabl(HELD[i:i+BATCH], collect_mo=True)
    MOSUM += mo.sum(0)
MOMEAN = MOSUM / NHELD
full_sum = full_sq = 0.0; full_n = 0
resid_sum = resid_sq = 0.0; resid_n = 0
for i in range(0, NHELD, BATCH):
    sb = slice(i, min(i+BATCH, NHELD)); idx = HELD[sb]; tgt = tgt_all[sb]
    base = forward_fullabl(idx).float()
    abl = forward_fullabl(idx, MOMEAN=MOMEAN).float()
    rec = forward_fullabl(idx, recon_only=True).float()
    blp = F.log_softmax(base[:, :SEQL-1], -1)
    alp = F.log_softmax(abl[:, :SEQL-1], -1)
    rlp = F.log_softmax(rec[:, :SEQL-1], -1)
    bce = -blp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1)
    ace = -alp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1)
    rce = -rlp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1)
    vm = torch.from_numpy(valid_next[sb, :SEQL-1]).to(DEV)
    d = (ace - bce)[vm]; dr = (rce - bce)[vm]
    full_sum += float(d.sum()); full_sq += float((d*d).sum()); full_n += int(d.numel())
    resid_sum += float(dr.sum()); resid_sq += float((dr*dr).sum()); resid_n += int(dr.numel())
    del base, abl, rec, blp, alp, rlp
full_mean, full_se = stats(full_sum, full_sq, full_n)
resid_mean, resid_se = stats(resid_sum, resid_sq, resid_n)
print(f"full-MLP1 mean-ablation global dCE {full_mean:.4f} +- {full_se:.4f}", flush=True)
print(f"REMOVE-SAE-RESIDUAL (keep reconstruction) global dCE {resid_mean:.4f} +- {resid_se:.4f} "
      f"= {resid_mean/full_mean*100:.1f}% of full  [positive control]", flush=True)

# ASSEMBLE
CLEAR_Z = 3.0; CLEAR_DCE = 0.02
records = []
n_clear = 0
for k, j in enumerate(tested):
    gm, gse = stats(g_sum[j], g_sq[j], g_n[j])
    tm_, tse = stats(t_sum[j], t_sq[j], t_n[j])
    z = tm_/tse if tse > 0 else 0.0
    clear = bool(z >= CLEAR_Z and tm_ >= CLEAR_DCE)
    n_clear += int(clear)
    records.append({
        'feature': j, 'global_dCE': round(gm, 6), 'global_dCE_SE': round(gse, 6),
        'trigger_dCE': round(tm_, 5), 'trigger_dCE_SE': round(tse, 5), 'trigger_dCE_z': round(z, 2),
        'trigger_dCE_frac_pos': round(t_pos[j]/max(1, t_n[j]), 3),
        'clears_74_causal_bar': clear,
    })
records_by_z = sorted(records, key=lambda r: -r['trigger_dCE_z'])

cum_curve = []
for n in CUM_NS:
    cm, cse = stats(cum_g_sum[n], cum_g_sq[n], cum_g_n[n])
    cum_curve.append({'n_features': n, 'captured_global_dCE': round(cm, 5), 'SE': round(cse, 5),
                      'frac_of_full_MLP1': round(cm/full_mean, 4) if full_mean > 0 else None})

def feats_for_frac(frac):
    for c in cum_curve:
        if c['frac_of_full_MLP1'] is not None and c['frac_of_full_MLP1'] >= frac:
            return c['n_features']
    return None
n80 = feats_for_frac(0.8); n50 = feats_for_frac(0.5)

causal = {
    'meta': {'held_slice': 'canonical FW[448:600,:128]', 'BATCH': BATCH, 'KCAUSAL': KCAUSAL,
             'n_features': NFEAT, 'k_active': K, 'tested_features': tested,
             'causal_bar': f'trigger_dCE_z >= {CLEAR_Z} AND trigger_dCE >= {CLEAR_DCE} nats (§74 bar)',
             'forward': 'VERBATIM qk_sae_converged_2.py / qk_redteam_sae_hub_2.py / qk_mlp1_tail.py'},
    'summary': {
        'full_MLP1_meanabl_global_dCE': round(full_mean, 5), 'full_MLP1_meanabl_global_dCE_SE': round(full_se, 5),
        'remove_SAE_residual_global_dCE': round(resid_mean, 5), 'remove_SAE_residual_global_dCE_SE': round(resid_se, 5),
        'remove_SAE_residual_frac_of_full': round(resid_mean/full_mean, 4) if full_mean > 0 else None,
        'all_dictionary_features_frac_of_full': round(max(c['frac_of_full_MLP1'] for c in cum_curve), 4),
        'n_features_tested_individually': len(tested),
        'n_single_features_clear_74_bar': n_clear,
        'svd_reference_single_dir_clear': '0 of 32 (§74)',
        'sec79_reference_single_clear': '0 of 32 (§79, held FVE 0.7162)',
        'max_single_feature_trigger_z': round(max(r['trigger_dCE_z'] for r in records), 2),
        'max_single_feature_trigger_dCE': round(max(r['trigger_dCE'] for r in records), 4),
        'cumulative_features_for_50pct_of_full': n50,
        'cumulative_features_for_80pct_of_full': n80,
    },
    'single_feature_records_by_z': records_by_z,
    'cumulative_curve': cum_curve,
}

# merge into qk_sae_moredata.json
outpath = f'{QK}/qk_sae_moredata.json'
try:
    full = json.load(open(outpath))
except Exception:
    full = {}
full['causal'] = causal
json.dump(full, open(outpath, 'w'), indent=2)

print("\n===== CAUSAL HEAD-TO-HEAD at 10x-data fidelity (delta cross-entropy, nats) =====", flush=True)
print(f"full-MLP1 mean-ablation {full_mean:.4f}", flush=True)
print(f"single features clearing §74 bar (z>=3 & trigger dCE>=0.02): {n_clear}/{len(tested)}  (SVD 0/32, §79 0/32)", flush=True)
print(f"max single-feature trigger z {causal['summary']['max_single_feature_trigger_z']}, "
      f"max trigger dCE {causal['summary']['max_single_feature_trigger_dCE']}", flush=True)
print("top features by trigger z:", flush=True)
for r in records_by_z[:12]:
    print(f"  feat {r['feature']:5d}  trigger dCE {r['trigger_dCE']:.4f} (z {r['trigger_dCE_z']:.1f})  "
          f"global dCE {r['global_dCE']:.5f}  {'CLEARS' if r['clears_74_causal_bar'] else '-'}", flush=True)
print(f"\nremove-SAE-residual {resid_mean:.4f} = {resid_mean/full_mean*100:.1f}% of full; "
      f"all dictionary features = {causal['summary']['all_dictionary_features_frac_of_full']*100:.2f}% of full", flush=True)
print(f"cumulative features for 50%/80% of full-MLP1 effect: {n50}/{n80}", flush=True)
print("\nMerged causal results into qk_sae_moredata.json", flush=True)
print("QK SAE MOREDATA (script 2: causal) DONE", flush=True)
