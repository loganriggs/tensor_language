"""CROSS-ARCHITECTURE HUB IRREDUCIBLY-DISTRIBUTED TEST (§74) on swiglu18 / bilin12.

Ports qk_mlp1_tail.py (bilin18's MLP-L1 hub characterization) to a DIFFERENT
architecture.  For the model's biggest early-layer feed-forward hub (the layer carrying
the most below-top-4 MLP tail, from qk_general_completeness.py), take its top-32 SVD
OUTPUT directions and per-direction compute: (a) solo mean-ablation global + trigger
delta cross-entropy (paired SE); (b) class-summed delta-logit signature + concentration;
(c) trigger class signature; (d) output entropy / token concentration.  Then CLASSIFY
each direction (class_pusher / class_suppressor / sharp_token / diffuse_distributed /
null) and report how much of the layer's causal effect the single-direction-nameable
ones account for, plus the joint-vs-sum-of-solos (irreducibility) test.

Usage: python qk_general_completeness_2.py <model> <hub_layer>

FORWARD + project-out mean-ablation + top-K SVD-from-TRAIN-gram + class library
(lex1/VOCAB_CLASS) + paired standard errors are copied VERBATIM from qk_mlp1_tail.py /
qk_unsup_classpush.py; ONLY the attention convention is ARCH-adapted (single QK branch;
softmax for swiglu18, squared-normalized for bilin12), matching qk_general_completeness.py.
Held-back FW[448:600,:128].
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
MODEL = sys.argv[1]; LI = int(sys.argv[2])
assert MODEL in ('swiglu18', 'bilin12')

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

m, cfg = load_elriggs(MODEL)
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
ARCH = 'softmax' if not cfg.get('squared_attn') else 'sqrd_norm'
tok = AutoTokenizer.from_pretrained('gpt2')
def dec(t): return repr(tok.decode([int(t)]))
K32 = 32
print(f"{MODEL} ({ARCH}) NL={NL} NH={NH} HD={HD} D={D} V={V}; characterizing MLP L{LI} top-{K32} SVD dirs", flush=True)

FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
SEQL = 128
TRAIN = FINEWEB[0:256, :SEQL].to(DEV)
HELD = FINEWEB[448:600, :SEQL].to(DEV)
NHELD = HELD.shape[0]
BATCH = 6
KCAUSAL = 200

_special = {tok.eos_token_id}
for _t in range(V):
    if '�' in tok.decode([_t]): _special.add(_t)
SPECIAL = np.array(sorted(_special))
print(f"{len(SPECIAL)} special token ids masked from trigger selection", flush=True)

def attn_pat(q, k, mask):
    if ARCH == 'softmax':
        sc = (torch.einsum('bqhd,bkhd->bhqk', q, k) / (HD**0.5)).masked_fill(~mask, float('-inf'))
        return F.softmax(sc, -1)
    else:
        s = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        pat = s.square().masked_fill(~mask, 0.0)
        return pat / pat.sum(-1, keepdim=True).clamp_min(1e-9)

# =====================================================================================
# LEXICAL CLASS LIBRARY -- VERBATIM from qk_unsup_classpush.py / qk_mlp1_tail.py.
# =====================================================================================
BRACKETS_OPEN = set("([{<"); BRACKETS_CLOSE = set(")]}>")
QUOTE_OPEN = set("“‘`"); QUOTE_CLOSE = set("”’"); QUOTE_STRAIGHT = set("\"'")
PUNCT = set(".,;:!?—–-…*|/\\~@#%^&+=_")
COORDINATORS = {"and","or","but","nor","yet","so"}
DETERMINERS = {"the","a","an","this","that","these","those","some","any","each","every","no","another","such"}
PRONOUNS = {"i","we","you","he","she","it","they","them","us","me","him","her","which","who"}

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
CMAT = torch.zeros(len(CLASS_LIST), V, device=DEV)
for t in range(V):
    CMAT[CIDX[VOCAB_CLASS[t]], t] = 1.0
PUSH_EXCLUDE = {'special'}
print(f"lexical classes ({len(CLASS_LIST)}): {CLASS_LIST}", flush=True)

# =====================================================================================
# MLP directions: top-K32 SVD dirs for layer LI from the TRAIN gram (VERBATIM).
# =====================================================================================
gram = torch.zeros(D, D, device=DEV)
@torch.no_grad()
def fwd_gram(idx):
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
        q, k = qk(a.c_q), qk(a.c_k)
        pat = attn_pat(q, k, mask); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1)); mo = blk.mlp(F.rms_norm(x, (D,)))
        if li == LI:
            gram.add_(torch.einsum('btd,bte->de', mo, mo))
        x = x + mo
print("Recomputing MLP SVD directions from TRAIN gram ...", flush=True)
for i in range(0, TRAIN.shape[0], BATCH): fwd_gram(TRAIN[i:i+BATCH])
_evals, _evecs = torch.linalg.eigh(gram)
DIRS = _evecs[:, -K32:].T.flip(0).contiguous()      # (K32,D) top descending
gram_eval = _evals.cpu().numpy()
del gram, _evecs
print("MLP directions ready.", flush=True)

# =====================================================================================
# Core forward (VERBATIM) with (a) collect, (b) project-out mean-ablation of layer-LI dirs.
# =====================================================================================
@torch.no_grad()
def forward(idx, collect=False, proj=None):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k = qk(a.c_q), qk(a.c_k)
        pat = attn_pat(q, k, mask); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if collect and li == LI:
            pr = torch.einsum('btd,kd->btk', mo, DIRS)
            PROJ_SUM[0] += pr.sum(0)
            out_proj[0] = pr.cpu().numpy()
        if proj is not None and li == LI:
            Dirs, PM = proj
            pr = torch.einsum('btd,kd->btk', mo, Dirs)
            coeff = pr - PM.unsqueeze(0)
            mo = mo - torch.einsum('btk,kd->btd', coeff, Dirs)
        x = x + mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    return logits

# =====================================================================================
# PASS A: per-position projection means + per-direction activation magnitudes.
# =====================================================================================
PROJ_SUM = [torch.zeros(SEQL, K32, device=DEV)]
out_proj = [None]
mlp_act = np.zeros((K32, NHELD, SEQL), np.float32)
print("PASS A: collect projection magnitudes + per-position means ...", flush=True)
for i in range(0, NHELD, BATCH):
    forward(HELD[i:i+BATCH], collect=True)
    b = out_proj[0].shape[0]
    pj = np.abs(out_proj[0])
    for kk in range(K32): mlp_act[kk, i:i+b] = pj[:, :, kk]
PROJMEAN = PROJ_SUM[0] / NHELD
del PROJ_SUM
print("PASS A done.", flush=True)

held_np = HELD.cpu().numpy()
pos_t = np.tile(np.arange(SEQL), NHELD).reshape(NHELD, SEQL)
is_special = np.isin(held_np, SPECIAL)
valid_next = pos_t < (SEQL - 1)
bad_trigger = (pos_t == 0) | is_special | ~valid_next

trig_mask = {}
for kk in range(K32):
    a = mlp_act[kk].copy().reshape(-1)
    a[bad_trigger.reshape(-1)] = -1e30
    tk = np.argpartition(a, -KCAUSAL)[-KCAUSAL:]
    mk = np.zeros(NHELD*SEQL, bool); mk[tk] = True
    trig_mask[kk] = mk.reshape(NHELD, SEQL)

def conc(ids):
    ids = np.asarray(ids)
    if len(ids) <= 1: return 0.0
    _, c = np.unique(ids, return_counts=True); p = c/c.sum()
    return float(1 - (-(p*np.log(p)).sum())/math.log(len(ids)))

def trigger_class_sig(kk):
    fs, fp = np.where(trig_mask[kk])
    cur = held_np[fs, fp]
    classes = VOCAB_CLASS[cur]
    cnt = Counter(classes.tolist())
    top = cnt.most_common(6)
    purity = conc([CIDX[c] for c in classes])
    return [(c, int(n)) for c, n in top], round(purity, 4)

def stats(s, sq, n):
    if n <= 1: return 0.0, 0.0
    mean = s/n; var = max(sq/n - mean*mean, 0.0)*n/(n-1)
    return mean, math.sqrt(var/n)

# =====================================================================================
# PASS B: per-direction solo mean-ablation -> global dCE + trigger dCE (paired) +
# class-summed delta-logit + mean delta-logit vector.
# =====================================================================================
g_sum = {k: 0.0 for k in range(K32)}; g_sq = {k: 0.0 for k in range(K32)}; g_n = {k: 0 for k in range(K32)}
t_sum = {k: 0.0 for k in range(K32)}; t_sq = {k: 0.0 for k in range(K32)}; t_n = {k: 0 for k in range(K32)}
t_pos = {k: 0 for k in range(K32)}
classsum = {k: torch.zeros(len(CLASS_LIST), device=DEV) for k in range(K32)}
dlogit_sum = {k: torch.zeros(V, device=DEV) for k in range(K32)}
nfire = {k: 0 for k in range(K32)}
tgt_all = torch.from_numpy(held_np).to(DEV)
print(f"PASS B: {K32} single-direction mean-ablations ...", flush=True)
t0 = time.time()
for bi, i in enumerate(range(0, NHELD, BATCH)):
    sb = slice(i, min(i+BATCH, NHELD)); idx = HELD[sb]; b = idx.shape[0]
    base = forward(idx).float()
    tgt = tgt_all[sb]
    logp = F.log_softmax(base[:, :SEQL-1], dim=-1)
    base_ce = -logp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1)
    del logp
    vmask = torch.from_numpy(valid_next[sb, :SEQL-1]).to(DEV)
    for kk in range(K32):
        proj = (DIRS[kk:kk+1].contiguous(), PROJMEAN[:, kk:kk+1].contiguous())
        abl = forward(idx, proj=proj).float()
        alogp = F.log_softmax(abl[:, :SEQL-1], dim=-1)
        abl_ce = -alogp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1)
        del alogp
        dce = (abl_ce - base_ce)
        dg = dce[vmask]
        g_sum[kk] += float(dg.sum()); g_sq[kk] += float((dg*dg).sum()); g_n[kk] += int(dg.numel())
        tm = torch.from_numpy(trig_mask[kk][sb, :SEQL-1]).to(DEV)
        if tm.any():
            dt = dce[tm]
            t_sum[kk] += float(dt.sum()); t_sq[kk] += float((dt*dt).sum())
            t_n[kk] += int(dt.numel()); t_pos[kk] += int((dt > 0).sum())
            dl = (base[:, :SEQL-1] - abl[:, :SEQL-1])[tm]
            classsum[kk] += CMAT @ dl.sum(0)
            dlogit_sum[kk] += dl.sum(0)
            nfire[kk] += int(dl.shape[0])
        del abl, dce
    if bi % 3 == 0:
        print(f"  batch {bi+1}/{(NHELD+BATCH-1)//BATCH}  elapsed {time.time()-t0:.0f}s", flush=True)
    del base, base_ce
print(f"PASS B done in {time.time()-t0:.0f}s", flush=True)

# =====================================================================================
# PASS B2: JOINT full-layer (all top-32 dirs together) mean-ablation reference.
# This is the IRREDUCIBILITY test: joint-vs-sum-of-solos.
# =====================================================================================
print("PASS B2: full-MLP-L{} (all top-32 dirs) mean-ablation reference ...".format(LI), flush=True)
full_sum = 0.0; full_sq = 0.0; full_n = 0
proj_all = (DIRS.contiguous(), PROJMEAN.contiguous())
for i in range(0, NHELD, BATCH):
    sb = slice(i, min(i+BATCH, NHELD)); idx = HELD[sb]; tgt = tgt_all[sb]
    base = forward(idx).float(); abl = forward(idx, proj=proj_all).float()
    blp = F.log_softmax(base[:, :SEQL-1], -1); alp = F.log_softmax(abl[:, :SEQL-1], -1)
    bce = -blp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1)
    ace = -alp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1)
    vm = torch.from_numpy(valid_next[sb, :SEQL-1]).to(DEV)
    d = (ace - bce)[vm]
    full_sum += float(d.sum()); full_sq += float((d*d).sum()); full_n += int(d.numel())
    del base, abl, blp, alp
full_mean, full_se = stats(full_sum, full_sq, full_n)
print(f"full-MLP-L{LI}(top32) joint global dCE {full_mean:.4f} +- {full_se:.4f}", flush=True)

# =====================================================================================
# ASSEMBLE per-direction records + classify.
# =====================================================================================
def entropy_of(vec):
    p = vec / vec.sum(); p = p[p > 0]
    return float(-(p*np.log(p)).sum())

records = []
for kk in range(K32):
    gm, gse = stats(g_sum[kk], g_sq[kk], g_n[kk])
    tm_, tse = stats(t_sum[kk], t_sq[kk], t_n[kk])
    z = tm_/tse if tse > 0 else 0.0
    cs = classsum[kk].cpu().numpy() / max(1, nfire[kk])
    abs_total = float(np.abs(cs).sum())
    order = np.argsort(-np.abs(cs))
    pushed = next((j for j in order if CLASS_LIST[j] not in PUSH_EXCLUDE), int(order[0]))
    push_class = CLASS_LIST[pushed]; push_val = float(cs[pushed])
    concentration = float(abs(cs[pushed]) / abs_total) if abs_total > 0 else 0.0
    class_top = {CLASS_LIST[j]: round(float(cs[j]), 4) for j in order[:6]}
    dmean = (dlogit_sum[kk] / max(1, nfire[kk]))
    dabs = dmean.abs().cpu().numpy()
    dsort = np.sort(dabs)[::-1]
    tot = float(dabs.sum()) + 1e-12
    top1_share = float(dsort[0]/tot); top10_share = float(dsort[:10].sum()/tot)
    out_ent = round(entropy_of(dabs + 1e-30), 3)
    dm = dmean.cpu().numpy()
    boost_ids = np.argsort(-dm)[:6]; supp_ids = np.argsort(dm)[:6]
    tclass_top, tpurity = trigger_class_sig(kk)
    clear = (z >= 3.0) and (tm_ >= 0.02)
    if not clear: bucket = 'null_subthreshold'
    elif concentration >= 0.45 and push_val >= 0: bucket = 'class_pusher'
    elif concentration >= 0.45 and push_val < 0: bucket = 'class_suppressor'
    elif top10_share >= 0.5: bucket = 'sharp_token'
    else: bucket = 'diffuse_distributed'
    records.append({
        'rank': kk, 'gram_eval_frac': round(float(gram_eval[-(kk+1)]/gram_eval.sum()), 5),
        'global_dCE': round(gm, 6), 'global_dCE_SE': round(gse, 6),
        'trigger_dCE': round(tm_, 5), 'trigger_dCE_SE': round(tse, 5), 'trigger_dCE_z': round(z, 2),
        'trigger_dCE_frac_pos': round(t_pos[kk]/max(1, t_n[kk]), 3),
        'pushed_class': push_class, 'pushed_class_summed_dlogit': round(push_val, 4),
        'pushed_class_sign': '+' if push_val >= 0 else '-',
        'class_concentration': round(concentration, 4), 'class_summed_top6': class_top,
        'trigger_class_top': tclass_top, 'trigger_purity': tpurity,
        'out_entropy_nats': out_ent, 'top1_token_share': round(top1_share, 4),
        'top10_token_share': round(top10_share, 4),
        'top_boosted_tokens': [(dec(t), round(float(dm[t]), 3)) for t in boost_ids],
        'top_suppressed_tokens': [(dec(t), round(float(dm[t]), 3)) for t in supp_ids],
        'n_firing': nfire[kk], 'bucket': bucket,
    })

INTERP = {'class_pusher', 'class_suppressor', 'sharp_token'}
bucket_counts = Counter(r['bucket'] for r in records)
sum_global_all = sum(max(0.0, r['global_dCE']) for r in records)
sum_global_interp = sum(max(0.0, r['global_dCE']) for r in records if r['bucket'] in INTERP)
sum_trig_all = sum(max(0.0, r['trigger_dCE']) for r in records)
sum_trig_interp = sum(max(0.0, r['trigger_dCE']) for r in records if r['bucket'] in INTERP)

summary = {
    'hub_layer': LI,
    'full_layer_top32_joint_global_dCE': round(full_mean, 5), 'full_layer_top32_joint_global_dCE_SE': round(full_se, 5),
    'sum_per_dir_solo_global_dCE': round(sum_global_all, 5),
    'joint_over_sum_ratio': round(full_mean / sum_global_all, 4) if sum_global_all > 0 else None,
    'sum_over_joint_ratio': round(sum_global_all / full_mean, 4) if full_mean > 0 else None,
    'irreducibility_note': 'joint (all 32 dirs together) vs sum-of-solos. sum >> joint => redundant/reducible; '
                           'joint >> sum => super-additive (irreducibly distributed: the layer computes in combinations '
                           'no single direction captures).',
    'bucket_counts': dict(bucket_counts),
    'n_causally_clear': int(sum(1 for r in records if r['bucket'] != 'null_subthreshold')),
    'n_single_direction_nameable': int(sum(1 for r in records if r['bucket'] in INTERP)),
    'frac_nameable_of_summed_solo_global_dCE': round(sum_global_interp / sum_global_all, 4) if sum_global_all > 0 else None,
    'frac_nameable_of_summed_trigger_dCE': round(sum_trig_interp / sum_trig_all, 4) if sum_trig_all > 0 else None,
    'frac_nameable_solo_global_over_joint_layer': round(sum_global_interp / full_mean, 4) if full_mean > 0 else None,
    'mean_output_entropy_nats': round(float(np.mean([r['out_entropy_nats'] for r in records])), 3),
    'log_vocab_nats': round(math.log(V), 3),
}

out = {
    'meta': {
        'model': MODEL, 'arch': ARCH, 'layer': LI, 'K_dirs': K32, 'held_slice': 'FW[448:600,:128]',
        'KCAUSAL': KCAUSAL, 'BATCH': BATCH, 'n_classes': len(CLASS_LIST), 'classes': CLASS_LIST,
        'currency': 'mean-ablation (per-position held mean) delta cross-entropy per valid held position (nats), paired SE',
        'forward': f'ARCH={ARCH} single-QK-branch; project-out mean-ablation + class library copied VERBATIM from '
                   'qk_mlp1_tail.py / qk_unsup_classpush.py',
        'classify_rule': 'clear = trigger_dCE_z>=3 & trigger_dCE>=0.02. class_pusher/suppressor: class_concentration>=0.45, '
                         'sign of pushed class. sharp_token: top10 token share>=0.5. diffuse_distributed: clear but neither. '
                         'null_subthreshold: not clear. SINGLE-DIRECTION-NAMEABLE = {class_pusher, class_suppressor, sharp_token}.',
    },
    'summary': summary,
    'records': records,
}
json.dump(out, open(f'{QK}/qk_general_completeness_{MODEL}_hub.json', 'w'), indent=2)

print(f"\n===== {MODEL} MLP L{LI} TOP-32 SVD DIRECTION CHARACTERIZATION (§74 hub test) =====", flush=True)
print(f"joint full-layer(top32) global dCE {full_mean:.4f}   sum-of-solos global dCE {sum_global_all:.4f}   "
      f"joint/sum {summary['joint_over_sum_ratio']}x (sum/joint {summary['sum_over_joint_ratio']}x)", flush=True)
print(f"bucket counts: {dict(bucket_counts)}", flush=True)
print(f"single-direction-nameable dirs: {summary['n_single_direction_nameable']}/{K32}; "
      f"they hold {summary['frac_nameable_of_summed_solo_global_dCE']} of summed-solo global dCE, "
      f"{summary['frac_nameable_of_summed_trigger_dCE']} of summed trigger dCE", flush=True)
print(f"mean output entropy {summary['mean_output_entropy_nats']} nats (uniform log V = {summary['log_vocab_nats']})", flush=True)
print("\nrank | bucket              | trigCE(z)      | conc  push        | top10tok | trigclass", flush=True)
for r in records:
    tc = r['trigger_class_top'][0] if r['trigger_class_top'] else ('', 0)
    print(f"  {r['rank']:2d} | {r['bucket']:19s} | {r['trigger_dCE']:.3f}(z{r['trigger_dCE_z']:.1f}) | "
          f"{r['class_concentration']:.2f} {r['pushed_class_sign']}{r['pushed_class']:11s} | "
          f"{r['top10_token_share']:.2f}    | {tc[0]}({tc[1]})", flush=True)
print(f"\nSaved qk_general_completeness_{MODEL}_hub.json", flush=True)
print("QK GENERAL COMPLETENESS HUB DONE", flush=True)
