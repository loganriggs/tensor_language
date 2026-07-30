"""TENTH unsupervised-circuit-discovery detector: CAUSAL CLASS-LEVEL EFFECT detector.

Fixes the §67 blind spot. The existing discovery loop ranks paths by CLEANLINESS
(trigger-purity x top-64 effect-purity), which the §67 census showed is UNCORRELATED
with causal importance (Pearson 0.006) and STRUCTURALLY misses the model's largest
single-path effects: circuits whose OUTPUT is a DISTRIBUTED push over an entire token
CLASS (near-uniform over thousands of tokens, top-token share ~= 0) so the top-64
concentration proxy scores them ~= 0 no matter how load-bearing they are.

THE NEW DETECTOR (per §67(c)):
  1. CAUSAL importance for all 234 paths = mean-ablation delta cross-entropy at each
     path's activation-selected top firing positions (recomputed IDENTICALLY to
     qk_census_difficulty.py), paired standard errors.
  2. CLASS-LEVEL OUTPUT characterization: mean-ablate the path, take the mean
     delta-logit vector (base - ablated) over its firing positions, and for each
     coarse token CLASS SUM that movement over all tokens in the class. The pushed
     class = class with the largest |class-summed movement|; CLASS-LEVEL output
     concentration = its share of the total absolute class-summed movement. This
     replaces top-64 token concentration -- a distributed class push scores HIGH.
  3. RANK by causal importance x class-level concentration (the causal class-push
     score), with a §61 FAMILY-JOINT REDUNDANCY pre-pass over the census-flagged
     redundant families (mlp.L17.d1/d2/d3; positional heads h.L0.8/h.L4.1) reporting
     joint-over-sum-of-solos so redundant members are not each dismissed as null.
  4. CAUSALLY VERIFY the top class-push candidates: does mean-ablation SPECIFICALLY
     suppress the pushed class's summed logit at the path's FIRING positions versus a
     matched control (positions where the path is INACTIVE)? Report class-summed drop
     +- standard error and specificity vs control.
  5. Sanity: recover the census's known missed-hard paths, surface any NEW ones, and
     confirm the class-push score CORRELATES with causal importance (vs 0.006).

FORWARD + mean-ablation copied VERBATIM from qk_census_difficulty.py /
qk_census_difficulty_2.py (single- and multi-component). Class library (lex1 /
VOCAB_CLASS) copied VERBATIM from qk_unsup_decouple.py. Held-back FW[448:600,:128].
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

# ---------------- GPU GUARD (verbatim from census) ----------------
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
V = cfg['vocab_size']; NL = len(m.transformer.h); N_SVD = 4
tok = AutoTokenizer.from_pretrained('gpt2')
def dec(t): return repr(tok.decode([int(t)]))
print(f"bilin18 NL={NL} NH={NH} HD={HD} D={D} V={V}; {NL*NH} head-paths + {NL*N_SVD} mlp-paths", flush=True)

FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
SEQL = 128
TRAIN = FINEWEB[0:256, :SEQL].to(DEV)        # discovery slice -- ONLY to recompute MLP dirs (path defn)
HELD = FINEWEB[448:600, :SEQL].to(DEV)       # held-back verification slice
NHELD = HELD.shape[0]
BATCH = 6
KCAUSAL = 200                                # top-K activation positions per path (matches census)

# special/degenerate tokens excluded from trigger selection (matches census/verify)
_special = {tok.eos_token_id}
for _t in range(V):
    if '�' in tok.decode([_t]): _special.add(_t)
SPECIAL = np.array(sorted(_special))
print(f"{len(SPECIAL)} special token ids masked from trigger selection", flush=True)

# =====================================================================================
# LEXICAL CLASS LIBRARY -- VERBATIM from qk_unsup_decouple.py (lex1 / VOCAB_CLASS).
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
print(f"lexical classes ({len(CLASS_LIST)}): {CLASS_LIST}", flush=True)
# class-indicator matrix (n_classes, V) on GPU for fast class-summed delta-logit
CIDX = {c: i for i, c in enumerate(CLASS_LIST)}
CMAT = torch.zeros(len(CLASS_LIST), V, device=DEV)
for t in range(V):
    CMAT[CIDX[VOCAB_CLASS[t]], t] = 1.0
CLASS_SIZE = {c: int((VOCAB_CLASS == c).sum()) for c in CLASS_LIST}
# classes NOT eligible to be the "pushed class" (artifact/degenerate buckets); still in denom
PUSH_EXCLUDE = {'special'}

# =====================================================================================
# MLP directions: recompute top-4 SVD dirs per block from the TRAIN gram (VERBATIM).
# =====================================================================================
gram = [torch.zeros(D, D, device=DEV) for _ in range(NL)]
@torch.no_grad()
def fwd_gram(idx):
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
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1)); mo = blk.mlp(F.rms_norm(x, (D,)))
        gram[li] += torch.einsum('btd,bte->de', mo, mo); x = x + mo
print("Recomputing MLP SVD directions from TRAIN gram ...", flush=True)
for i in range(0, TRAIN.shape[0], BATCH): fwd_gram(TRAIN[i:i+BATCH])
mlp_dirs = torch.zeros(NL, N_SVD, D, device=DEV)
for li in range(NL):
    _evals, _evecs = torch.linalg.eigh(gram[li])
    mlp_dirs[li] = _evecs[:, -N_SVD:].T.flip(0)
del gram
print("MLP directions ready.", flush=True)

# =====================================================================================
# Core forward (VERBATIM) with MULTI-component mean-ablation (from census_2) + collect.
# ablations: list of ('head',li,h) | ('mlp',li,kk).  collect=True: per-path activation mags.
# =====================================================================================
@torch.no_grad()
def forward(idx, ablations=(), yhmeans=None, projmeans=None, collect=False):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    ab_heads = {}; ab_mlps = {}
    for a_ in ablations:
        if a_[0] == 'head': ab_heads.setdefault(a_[1], []).append(a_[2])
        elif a_[0] == 'mlp': ab_mlps.setdefault(a_[1], []).append(a_[2])
    out = {} if collect else None
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0)                 # (B,NH,T,T) UNNORMALISED
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)         # (B,T,NH,HD)
        if collect:
            Wr = a.c_proj.weight.view(D, NH, HD)
            YH_SUM[li] += yh4.sum(0)
            comp = torch.einsum('bthc,ohc->btho', yh4, Wr).norm(dim=-1)  # (B,T,NH)
            out[('hnorm', li)] = comp.cpu().numpy()
        if li in ab_heads:
            yh4 = yh4.clone()
            for h in ab_heads[li]: yh4[:, :, h] = yhmeans[li][:, h].unsqueeze(0)
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if collect:
            pr = torch.einsum('btd,nd->btn', mo, mlp_dirs[li])   # (B,T,N_SVD)
            PROJ_SUM[li] += pr.sum(0)
            out[('mproj', li)] = pr.cpu().numpy()
        if li in ab_mlps:
            for kk in ab_mlps[li]:
                pr = torch.einsum('btd,d->bt', mo, mlp_dirs[li, kk])
                mo = mo - (pr - projmeans[li][:, kk].unsqueeze(0)).unsqueeze(-1) * mlp_dirs[li, kk]
        x = x + mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    return (logits, out) if collect else logits

# =====================================================================================
# PASS A: per-position means + per-path activation magnitudes (VERBATIM).
# =====================================================================================
YH_SUM = {li: torch.zeros(SEQL, NH, HD, device=DEV) for li in range(NL)}
PROJ_SUM = {li: torch.zeros(SEQL, N_SVD, device=DEV) for li in range(NL)}
head_act = np.zeros((NL*NH, NHELD, SEQL), np.float32)
mlp_act = np.zeros((NL*N_SVD, NHELD, SEQL), np.float32)
print("PASS A: collect activation magnitudes + per-position means ...", flush=True)
for i in range(0, NHELD, BATCH):
    _, out = forward(HELD[i:i+BATCH], collect=True)
    b = HELD[i:i+BATCH].shape[0]
    for li in range(NL):
        hn = out[('hnorm', li)]
        for h in range(NH): head_act[li*NH + h, i:i+b] = hn[:, :, h]
        pj = np.abs(out[('mproj', li)])
        for kk in range(N_SVD): mlp_act[li*N_SVD + kk, i:i+b] = pj[:, :, kk]
YHMEAN = {li: YH_SUM[li] / NHELD for li in range(NL)}
PROJMEAN = {li: PROJ_SUM[li] / NHELD for li in range(NL)}
del YH_SUM, PROJ_SUM
print("PASS A done.", flush=True)

held_np = HELD.cpu().numpy()
pos_t = np.tile(np.arange(SEQL), NHELD).reshape(NHELD, SEQL)
is_special = np.isin(held_np, SPECIAL)
valid_next = pos_t < (SEQL - 1)
bad_trigger = (pos_t == 0) | is_special | ~valid_next

PATHS = []
for li in range(NL):
    for h in range(NH): PATHS.append((f"h.L{li}.{h}", 'head', li, h))
for li in range(NL):
    for kk in range(N_SVD): PATHS.append((f"mlp.L{li}.d{kk}", 'mlp', li, kk))

def act_of(kind, li, ix):
    return head_act[li*NH + ix] if kind == 'head' else mlp_act[li*N_SVD + ix]

# top-KCAUSAL activation-selected trigger mask (VERBATIM) + matched INACTIVE control mask
trig_mask = {}; ctrl_mask = {}
for (comp, kind, li, ix) in PATHS:
    a = act_of(kind, li, ix).copy().reshape(-1)
    a[bad_trigger.reshape(-1)] = -1e30
    tk = np.argpartition(a, -KCAUSAL)[-KCAUSAL:]
    mk = np.zeros(NHELD*SEQL, bool); mk[tk] = True
    trig_mask[comp] = mk.reshape(NHELD, SEQL)
    # control = lowest-activation VALID positions (path inactive), matched count
    a2 = act_of(kind, li, ix).copy().reshape(-1)
    a2[bad_trigger.reshape(-1)] = 1e30
    bk = np.argpartition(a2, KCAUSAL)[:KCAUSAL]
    mc = np.zeros(NHELD*SEQL, bool); mc[bk] = True
    ctrl_mask[comp] = mc.reshape(NHELD, SEQL)
del head_act, mlp_act

def stats(s, sq, n):
    if n <= 1: return 0.0, 0.0
    mean = s/n; var = max(sq/n - mean*mean, 0.0)*n/(n-1)
    return mean, math.sqrt(var/n)

# =====================================================================================
# PASS B: for every path -- mean-ablate, accumulate TRIGGER-dCE (paired) + CLASS-SUMMED
# delta-logit (base-ablated) over its firing positions.
# =====================================================================================
g_sum = {c: 0.0 for c, *_ in PATHS}; g_sq = {c: 0.0 for c, *_ in PATHS}; g_n = {c: 0 for c, *_ in PATHS}
t_sum = {c: 0.0 for c, *_ in PATHS}; t_sq = {c: 0.0 for c, *_ in PATHS}; t_n = {c: 0 for c, *_ in PATHS}
t_pos = {c: 0 for c, *_ in PATHS}
classsum = {c: torch.zeros(len(CLASS_LIST), device=DEV) for c, *_ in PATHS}  # accum class-summed dlogit
nfire = {c: 0 for c, *_ in PATHS}

tgt_all = torch.from_numpy(held_np).to(DEV)
print(f"PASS B: {len(PATHS)} single-path mean-ablations (dCE + class-summed dlogit) ...", flush=True)
t0 = time.time()
for bi, i in enumerate(range(0, NHELD, BATCH)):
    sb = slice(i, min(i+BATCH, NHELD))
    idx = HELD[sb]; b = idx.shape[0]
    base = forward(idx).float()                            # (b,T,V)
    tgt = tgt_all[sb]
    logp = F.log_softmax(base[:, :SEQL-1], dim=-1)
    base_ce = -logp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1)   # (b,T-1)
    del logp
    vmask = torch.from_numpy(valid_next[sb, :SEQL-1]).to(DEV)
    for (comp, kind, li, ix) in PATHS:
        abl = forward(idx, ablations=[(kind, li, ix)], yhmeans=YHMEAN, projmeans=PROJMEAN).float()
        alogp = F.log_softmax(abl[:, :SEQL-1], dim=-1)
        abl_ce = -alogp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1)
        del alogp
        dce = (abl_ce - base_ce)                           # (b,T-1)  ablated-base
        dg = dce[vmask]
        g_sum[comp] += float(dg.sum()); g_sq[comp] += float((dg*dg).sum()); g_n[comp] += int(dg.numel())
        tm = torch.from_numpy(trig_mask[comp][sb, :SEQL-1]).to(DEV)
        if tm.any():
            dt = dce[tm]
            t_sum[comp] += float(dt.sum()); t_sq[comp] += float((dt*dt).sum())
            t_n[comp] += int(dt.numel()); t_pos[comp] += int((dt > 0).sum())
            # CLASS-SUMMED delta-logit (base - ablated) summed over firing positions in batch
            dl = (base[:, :SEQL-1] - abl[:, :SEQL-1])[tm]   # (n_fire_batch, V)
            classsum[comp] += CMAT @ dl.sum(0)             # (n_classes,)
            nfire[comp] += int(dl.shape[0])
        del abl, dce
    if bi % 4 == 0:
        print(f"  batch {bi+1}/{(NHELD+BATCH-1)//BATCH}  elapsed {time.time()-t0:.0f}s", flush=True)
    del base, base_ce
print(f"PASS B done in {time.time()-t0:.0f}s", flush=True)

# join cleanliness from census records
census = json.load(open(f'{QK}/qk_census_difficulty.json'))
cen_rec = {r['comp']: r for r in census['records']}

# assemble records with class-push score
records = []
for (comp, kind, li, ix) in PATHS:
    gm, gse = stats(g_sum[comp], g_sq[comp], g_n[comp])
    tm_, tse = stats(t_sum[comp], t_sq[comp], t_n[comp])
    cs = classsum[comp].cpu().numpy() / max(1, nfire[comp])   # MEAN class-summed dlogit
    abs_total = float(np.abs(cs).sum())
    # pushed class = max |class-summed movement|, excluding artifact buckets
    order = np.argsort(-np.abs(cs))
    pushed = None
    for j in order:
        if CLASS_LIST[j] not in PUSH_EXCLUDE:
            pushed = j; break
    push_class = CLASS_LIST[pushed]
    push_val = float(cs[pushed])
    concentration = float(abs(cs[pushed]) / abs_total) if abs_total > 0 else 0.0
    class_dict = {CLASS_LIST[j]: round(float(cs[j]), 4) for j in order[:8]}
    cr = cen_rec.get(comp, {})
    records.append({
        'comp': comp, 'kind': kind, 'li': li, 'idx': ix,
        'cleanliness': cr.get('cleanliness'), 'trigger_purity': cr.get('trigger_purity'),
        'effect_purity': cr.get('effect_purity'),
        'trigger_dCE': round(tm_, 5), 'trigger_dCE_SE': round(tse, 5),
        'trigger_dCE_z': round(tm_/tse, 3) if tse > 0 else 0.0,
        'trigger_dCE_frac_pos': round(t_pos[comp]/max(1, t_n[comp]), 3),
        'global_dCE': round(gm, 6),
        'census_trigger_dCE': cr.get('trigger_dCE'),   # cross-check we match census
        # class-level output characterization (the new move)
        'pushed_class': push_class, 'pushed_class_summed_dlogit': round(push_val, 4),
        'pushed_class_sign': '+' if push_val >= 0 else '-',
        'class_concentration': round(concentration, 4),
        'class_summed_top8': class_dict,
        'n_firing': nfire[comp],
        # the causal class-push score
        'classpush_score': round(max(tm_, 0.0) * concentration, 6),
    })

def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() == 0 or b.std() == 0: return 0.0
    return float(np.corrcoef(a, b)[0, 1])
def spearman(a, b):
    return pearson(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))

clean = np.array([r['cleanliness'] if r['cleanliness'] is not None else 0.0 for r in records])
tdce = np.array([r['trigger_dCE'] for r in records])
cps = np.array([r['classpush_score'] for r in records])
conc_arr = np.array([r['class_concentration'] for r in records])

correlations = {
    'pearson_classpush_vs_trigger_dCE': round(pearson(cps, tdce), 4),
    'spearman_classpush_vs_trigger_dCE': round(spearman(cps, tdce), 4),
    'pearson_concentration_alone_vs_trigger_dCE': round(pearson(conc_arr, tdce), 4),
    'pearson_cleanliness_vs_trigger_dCE_ourrun': round(pearson(clean, tdce), 4),
    'census_pearson_cleanliness_vs_trigger_dCE': census['summary']['correlations']['pearson_clean_vs_trigger_dCE'],
    'census_spearman_cleanliness_vs_trigger_dCE': census['summary']['correlations']['spearman_clean_vs_trigger_dCE'],
}
# cross-check that our recomputed trigger_dCE matches the census (forward-convention sanity)
matchdiff = [abs(r['trigger_dCE'] - r['census_trigger_dCE']) for r in records if r['census_trigger_dCE'] is not None]
correlations['recompute_vs_census_trigger_dCE_maxabsdiff'] = round(float(max(matchdiff)), 5) if matchdiff else None
correlations['recompute_vs_census_trigger_dCE_meanabsdiff'] = round(float(np.mean(matchdiff)), 6) if matchdiff else None

ranked = sorted(records, key=lambda r: -r['classpush_score'])
print("\n===== TOP 20 by CLASS-PUSH SCORE (causal importance x class concentration) =====", flush=True)
for r in ranked[:20]:
    print(f"  {r['comp']:12s} score={r['classpush_score']:.4f} tdCE={r['trigger_dCE']:.3f}"
          f"+-{r['trigger_dCE_SE']:.3f} z={r['trigger_dCE_z']:.1f} conc={r['class_concentration']:.2f}"
          f" push={r['pushed_class_sign']}{r['pushed_class']} clean={r['cleanliness']}", flush=True)

# =====================================================================================
# §61 FAMILY-JOINT REDUNDANCY PRE-PASS (census-flagged families).
# joint-over-sum-of-solos at the UNION of members' firing positions.
# =====================================================================================
FAMILIES = {
    'late_ff_mlpL17': [('mlp', 17, 1), ('mlp', 17, 2), ('mlp', 17, 3)],
    'positional_heads_L0.8_L4.1': [('head', 0, 8), ('head', 4, 1)],
}
def comp_name(k, l, i): return (f"h.L{l}.{i}" if k == 'head' else f"mlp.L{l}.d{i}")

@torch.no_grad()
def dce_paired_at_mask(mask, ablations):
    """paired ablated-base dCE at (NHELD,SEQL) query mask -> mean, SE."""
    vals = []
    seqs = np.where(mask[:, :SEQL-1].any(axis=1))[0]
    for i in range(0, len(seqs), BATCH):
        sb = seqs[i:i+BATCH]; idx = HELD[sb]
        base = forward(idx).float(); abl = forward(idx, ablations=ablations, yhmeans=YHMEAN, projmeans=PROJMEAN).float()
        tgt = tgt_all[sb]
        blp = F.log_softmax(base[:, :SEQL-1], -1); alp = F.log_softmax(abl[:, :SEQL-1], -1)
        bce = -blp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1)
        ace = -alp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1)
        mk = torch.from_numpy(mask[sb, :SEQL-1]).to(DEV)
        vals.append((ace - bce)[mk].cpu().numpy())
        del base, abl, blp, alp
    v = np.concatenate(vals) if vals else np.zeros(0)
    n = len(v)
    return (float(v.mean()), float(v.std(ddof=1)/math.sqrt(n)) if n > 1 else 0.0, n)

print("\n===== §61 FAMILY-JOINT REDUNDANCY PRE-PASS =====", flush=True)
family_report = {}
for fam, members in FAMILIES.items():
    umask = np.zeros((NHELD, SEQL), bool)
    for (k, l, i) in members: umask |= trig_mask[comp_name(k, l, i)]
    solos = {}
    for (k, l, i) in members:
        mn, se, n = dce_paired_at_mask(umask, [(k, l, i)])
        solos[comp_name(k, l, i)] = {'solo_dCE': round(mn, 4), 'solo_dCE_SE': round(se, 4)}
    jm, jse, jn = dce_paired_at_mask(umask, list(members))
    sum_solo = sum(v['solo_dCE'] for v in solos.values())
    ratio = float(jm / (sum_solo + 1e-9))
    family_report[fam] = {
        'members': [comp_name(*x) for x in members], 'n_union_positions': int(umask.sum()),
        'solos': solos, 'sum_solo_dCE': round(sum_solo, 4),
        'joint_dCE': round(jm, 4), 'joint_dCE_SE': round(jse, 4),
        'redundancy_ratio_joint_over_sumsolo': round(ratio, 3),
    }
    print(f"  {fam}: joint {jm:.3f} Σsolo {sum_solo:.3f} ratio {ratio:.2f} "
          f"solos={ {c: v['solo_dCE'] for c, v in solos.items()} }", flush=True)

# =====================================================================================
# CAUSAL VERIFICATION of top class-push candidates: class-summed drop at FIRING vs
# matched INACTIVE control (specificity). Per-position class-summed dlogit for the
# pushed class, paired SE over positions.
# =====================================================================================
# candidates: top by class-push score, restricted to statistically-clear causal paths
CAND = [r for r in ranked if r['trigger_dCE_z'] >= 3.0 and r['trigger_dCE'] >= 0.02][:8]
print(f"\n===== CAUSAL VERIFICATION of top {len(CAND)} class-push candidates =====", flush=True)

@torch.no_grad()
def classsummed_per_position(mask, kind, li, ix, class_j):
    """per-position class-summed delta-logit (base-ablated) for one class over a mask."""
    row = CMAT[class_j]                                    # (V,)
    vals = []
    seqs = np.where(mask.any(axis=1))[0]
    for i in range(0, len(seqs), BATCH):
        sb = seqs[i:i+BATCH]; idx = HELD[sb]
        base = forward(idx).float()
        abl = forward(idx, ablations=[(kind, li, ix)], yhmeans=YHMEAN, projmeans=PROJMEAN).float()
        dl = (base - abl)                                 # (b,T,V)  positive => path boosts (ablation drops)
        cs = torch.einsum('btv,v->bt', dl, row)           # (b,T) class-summed
        mk = torch.from_numpy(mask[sb]).to(DEV)
        vals.append(cs[mk].cpu().numpy())
        del base, abl, dl, cs
    return np.concatenate(vals) if vals else np.zeros(0)

verify = []
for r in CAND:
    comp = r['comp']; kind = r['kind']; li = r['li']; ix = r['idx']
    cj = CIDX[r['pushed_class']]
    fire = classsummed_per_position(trig_mask[comp], kind, li, ix, cj)
    ctrl = classsummed_per_position(ctrl_mask[comp], kind, li, ix, cj)
    def msd(v):
        n = len(v); return (float(v.mean()), float(v.std(ddof=1)/math.sqrt(n)) if n > 1 else 0.0, n)
    fmean, fse, fn = msd(fire); cmean, cse, cn = msd(ctrl)
    spec = fmean - cmean
    spec_se = math.sqrt(fse*fse + cse*cse)
    verdict = ('SPECIFIC-CLASS-PUSH' if (fmean > 0 and fmean/(fse+1e-9) >= 3 and spec/(spec_se+1e-9) >= 3)
               else 'PUSH-BUT-NOT-SPECIFIC' if (fmean > 0 and fmean/(fse+1e-9) >= 3)
               else 'FAILS-CLASS-PUSH')
    rec = {
        'comp': comp, 'kind': kind, 'li': li, 'pushed_class': r['pushed_class'],
        'trigger_dCE': r['trigger_dCE'], 'trigger_dCE_z': r['trigger_dCE_z'],
        'class_concentration': r['class_concentration'], 'classpush_score': r['classpush_score'],
        'cleanliness': r['cleanliness'],
        'classsummed_drop_at_firing': round(fmean, 4), 'classsummed_drop_at_firing_SE': round(fse, 4),
        'classsummed_at_control': round(cmean, 4), 'classsummed_at_control_SE': round(cse, 4),
        'specificity_firing_minus_control': round(spec, 4), 'specificity_SE': round(spec_se, 4),
        'specificity_z': round(spec/(spec_se+1e-9), 2),
        'firing_z': round(fmean/(fse+1e-9), 2),
        'class_summed_top8': r['class_summed_top8'],
        'verdict': verdict,
    }
    verify.append(rec)
    print(f"  {comp:12s} push={r['pushed_class']:9s} firing {fmean:.3f}+-{fse:.3f} (z{rec['firing_z']}) "
          f"ctrl {cmean:.3f}+-{cse:.3f}  spec {spec:.3f}+-{spec_se:.3f} (z{rec['specificity_z']}) -> {verdict}", flush=True)

# =====================================================================================
# SANITY: recover census missed-hard paths; new ones; correlation contrast.
# =====================================================================================
CENSUS_MISSED = ['h.L0.3', 'h.L11.2', 'mlp.L17.d1', 'mlp.L17.d2', 'mlp.L17.d3',
                 'h.L8.7', 'h.L8.3', 'h.L13.8', 'h.L0.8', 'h.L4.1']
rank_of = {r['comp']: i for i, r in enumerate(ranked)}
recovered = []
for c in CENSUS_MISSED:
    r = next((x for x in records if x['comp'] == c), None)
    if r: recovered.append({'comp': c, 'classpush_rank': rank_of[c]+1, 'classpush_score': r['classpush_score'],
                            'trigger_dCE': r['trigger_dCE'], 'pushed_class': r['pushed_class'],
                            'class_concentration': r['class_concentration']})
# NEW: high class-push score paths NOT in census top-10 missed and NOT census clean-winners
census_known = set(CENSUS_MISSED) | {r['comp'] for r in census['summary'].get('clean_winners_HH', [])}
new_surfaced = [{'comp': r['comp'], 'classpush_rank': i+1, 'classpush_score': r['classpush_score'],
                 'trigger_dCE': r['trigger_dCE'], 'trigger_dCE_z': r['trigger_dCE_z'],
                 'pushed_class': r['pushed_class'], 'class_concentration': r['class_concentration'],
                 'cleanliness': r['cleanliness']}
                for i, r in enumerate(ranked[:20]) if r['comp'] not in census_known
                and r['trigger_dCE_z'] >= 3.0 and r['trigger_dCE'] >= 0.02][:10]

summary = {
    'correlations': correlations,
    'family_joint_redundancy': family_report,
    'census_missed_hard_recovery': recovered,
    'new_surfaced_high_classpush': new_surfaced,
    'top15_by_classpush': [{'comp': r['comp'], 'classpush_score': r['classpush_score'],
                            'trigger_dCE': r['trigger_dCE'], 'trigger_dCE_z': r['trigger_dCE_z'],
                            'class_concentration': r['class_concentration'],
                            'pushed_class': r['pushed_class'], 'pushed_class_sign': r['pushed_class_sign'],
                            'cleanliness': r['cleanliness']} for r in ranked[:15]],
    'causal_verification': verify,
}
out = {
    'meta': {
        'model': 'bilin18', 'held_slice': 'FW[448:600,:128]', 'n_paths': len(PATHS),
        'KCAUSAL': KCAUSAL, 'BATCH': BATCH, 'n_classes': len(CLASS_LIST), 'classes': CLASS_LIST,
        'class_sizes': CLASS_SIZE, 'push_excluded_classes': sorted(PUSH_EXCLUDE),
        'detector': 'CAUSAL CLASS-LEVEL EFFECT (class-push). Rank = trigger-position mean-ablation '
                    'delta cross-entropy x CLASS-LEVEL output concentration (share of total absolute '
                    'class-summed delta-logit captured by the single most-moved token CLASS). Replaces '
                    'the top-64 token-concentration effect-purity proxy that §67 showed is blind to '
                    'distributed class-pushes. Forward + mean-ablation VERBATIM from qk_census_difficulty; '
                    'class library VERBATIM from qk_unsup_decouple (lex1/VOCAB_CLASS).',
        'verification': 'class-summed delta-logit for the pushed class at FIRING positions vs matched '
                        'INACTIVE control (lowest-activation valid positions), paired SE over positions.',
    },
    'summary': summary,
    'records': ranked,
}
json.dump(out, open(f'{QK}/qk_unsup_classpush.json', 'w'), indent=2)

print("\n===== CORRELATION CONTRAST (does the detector fix the blind spot?) =====", flush=True)
for k, v in correlations.items(): print(f"  {k}: {v}", flush=True)
print("\n===== CENSUS MISSED-HARD RECOVERY (class-push rank / 234) =====", flush=True)
for r in recovered: print(f"  {r['comp']:12s} rank {r['classpush_rank']:>3}/234 score={r['classpush_score']:.4f}"
                          f" push={r['pushed_class']} conc={r['class_concentration']:.2f}", flush=True)
print("\n===== NEW high class-push paths not in census top-10 =====", flush=True)
for r in new_surfaced: print(f"  {r['comp']:12s} rank {r['classpush_rank']} score={r['classpush_score']:.4f}"
                             f" tdCE={r['trigger_dCE']:.3f} push={r['pushed_class']} clean={r['cleanliness']}", flush=True)
print("\nSaved qk_unsup_classpush.json", flush=True)
print("QK UNSUP CLASSPUSH DONE", flush=True)
