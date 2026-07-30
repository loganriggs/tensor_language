"""GENERALITY TEST (§68): does the DISTRIBUTED CLASS-PUSHER circuit type replicate on
a DIFFERENT architecture -- swiglu18 (softmax attention + SwiGLU MLP)?

bilin18 (qk_unsup_classpush.py) showed the model's largest single-path effects are
DISTRIBUTED CLASS MOVERS: feed-forward directions / head pathways whose OUTPUT is a
near-uniform push over a whole token CLASS (capital / word), not a sharp token set.
This script asks whether the SAME phenomenon appears on swiglu18, a conventional
softmax-attention SwiGLU transformer, which would show it is architecture-general and
not a bilinear quirk.

METHOD (identical detector to qk_unsup_classpush.py; forward ADAPTED to swiglu18):
  1. CAUSAL importance for all paths = mean-ablation delta cross-entropy at each path's
     activation-selected top firing positions (paired SE).
  2. CLASS-LEVEL OUTPUT characterization: mean-ablate the path, take the mean delta-logit
     vector (base - ablated) over firing positions, SUM that movement per coarse token
     CLASS. Pushed class = max |class-summed movement|; class concentration = its share
     of total absolute class-summed movement. A distributed class push scores HIGH.
  3. RANK by classpush_score = trigger-position mean-ablation dCE x class concentration.
  4. CAUSALLY VERIFY top candidates: class-summed drop at FIRING vs matched INACTIVE
     control (specificity), paired SE. Plus output-entropy / top-token-share of the push.
  5. Report the classpush-score vs causal-importance correlation (the §68 blind-spot-fix
     check) on this softmax model, and compare capital/word integrator structure to bilin18.

ADAPTATION FROM qk_unsup_classpush.py (bilin18):
  (a) load_elriggs('bilin18') -> load_elriggs('swiglu18').
  (b) attention: bilin18's TWO-BRANCH squared UNNORMALIZED attention
        pat = (s1*s2).masked_fill(~mask, 0.0)
      replaced EVERYWHERE by swiglu18's SOFTMAX convention (verbatim from
      qk_content_gate_swiglu18.py / qk_swiglu18_commandeer.py):
        sc  = (einsum('bqhd,bkhd->bhqk', q, k)/(HD**0.5)).masked_fill(~mask, -inf)
        pat = F.softmax(sc, -1)
      swiglu18 has no c_q2/c_k2, so the second QK branch is dropped.
  (c) value path (attn.lamb v1-lerp), MLP (blk.mlp black box), head mean-ablation, MLP
      SVD-direction mean-ablation, class library (lex1/VOCAB_CLASS): UNCHANGED / VERBATIM.
  (d) swiglu18 has NO census -> cleanliness / census cross-check / census-missed recovery /
      the bilin18-specific §61 family pre-pass are DROPPED (not defined for this model).
      The §68 blind-spot correlation (classpush_score vs trigger_dCE) is kept.
Held-back FW[448:600,:128]. Output qk_general_classpush_swiglu.json.
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

m, cfg = load_elriggs('swiglu18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h); N_SVD = 4
tok = AutoTokenizer.from_pretrained('gpt2')
def dec(t): return repr(tok.decode([int(t)]))
print(f"swiglu18 NL={NL} NH={NH} HD={HD} D={D} V={V}; {NL*NH} head-paths + {NL*N_SVD} mlp-paths", flush=True)

FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
SEQL = 128
TRAIN = FINEWEB[0:256, :SEQL].to(DEV)        # discovery slice -- ONLY to recompute MLP dirs (path defn)
HELD = FINEWEB[448:600, :SEQL].to(DEV)       # held-back verification slice
NHELD = HELD.shape[0]
BATCH = 6
KCAUSAL = 200                                # top-K activation positions per path

# special/degenerate tokens excluded from trigger selection
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
CIDX = {c: i for i, c in enumerate(CLASS_LIST)}
CMAT = torch.zeros(len(CLASS_LIST), V, device=DEV)
for t in range(V):
    CMAT[CIDX[VOCAB_CLASS[t]], t] = 1.0
CLASS_SIZE = {c: int((VOCAB_CLASS == c).sum()) for c in CLASS_LIST}
PUSH_EXCLUDE = {'special'}

# =====================================================================================
# MLP directions: recompute top-4 SVD dirs per block from the TRAIN gram of MLP OUTPUT.
# Architecture-agnostic (operates on the MLP output mo). swiglu18 SOFTMAX attention.
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
        q, k = qk(a.c_q), qk(a.c_k)
        sc = (torch.einsum('bqhd,bkhd->bhqk', q, k)/(HD**0.5)).masked_fill(~mask, float('-inf'))
        pat = F.softmax(sc, -1)
        yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh.reshape(B, T, -1)); mo = blk.mlp(F.rms_norm(x, (D,)))
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
# Core forward (swiglu18 SOFTMAX) with MULTI-component mean-ablation + collect.
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
        q, k = qk(a.c_q), qk(a.c_k)
        sc = (torch.einsum('bqhd,bkhd->bhqk', q, k)/(HD**0.5)).masked_fill(~mask, float('-inf'))
        pat = F.softmax(sc, -1)                                # (B,NH,T,T) SOFTMAX-normalized
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)          # (B,T,NH,HD)
        if collect:
            YH_SUM[li] += yh4.sum(0)
            comp = torch.einsum('bthc,ohc->btho', yh4, a.c_proj.weight.view(D, NH, HD)).norm(dim=-1)  # (B,T,NH)
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
# PASS A: per-position means + per-path activation magnitudes.
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

trig_mask = {}; ctrl_mask = {}
for (comp, kind, li, ix) in PATHS:
    a = act_of(kind, li, ix).copy().reshape(-1)
    a[bad_trigger.reshape(-1)] = -1e30
    tk = np.argpartition(a, -KCAUSAL)[-KCAUSAL:]
    mk = np.zeros(NHELD*SEQL, bool); mk[tk] = True
    trig_mask[comp] = mk.reshape(NHELD, SEQL)
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
# PASS B: mean-ablate each path, accumulate TRIGGER-dCE (paired) + CLASS-SUMMED
# delta-logit (base-ablated) over firing positions + per-class-push entropy inputs.
# =====================================================================================
g_sum = {c: 0.0 for c, *_ in PATHS}; g_sq = {c: 0.0 for c, *_ in PATHS}; g_n = {c: 0 for c, *_ in PATHS}
t_sum = {c: 0.0 for c, *_ in PATHS}; t_sq = {c: 0.0 for c, *_ in PATHS}; t_n = {c: 0 for c, *_ in PATHS}
t_pos = {c: 0 for c, *_ in PATHS}
classsum = {c: torch.zeros(len(CLASS_LIST), device=DEV) for c, *_ in PATHS}
nfire = {c: 0 for c, *_ in PATHS}
# for the pushed-class OUTPUT distribution (entropy / top-token concentration):
# accumulate summed POSITIVE delta-logit over firing positions per vocab token, but only
# store lazily via the winning class after the fact -> keep full running per-token sum is
# 234*V floats (~47M) too big; instead accumulate a running |dlogit| sum per token in the
# top pushed direction during verification. Here store per-path per-CLASS sum only.

tgt_all = torch.from_numpy(held_np).to(DEV)
print(f"PASS B: {len(PATHS)} single-path mean-ablations (dCE + class-summed dlogit) ...", flush=True)
t0 = time.time()
for bi, i in enumerate(range(0, NHELD, BATCH)):
    sb = slice(i, min(i+BATCH, NHELD))
    idx = HELD[sb]; b = idx.shape[0]
    base = forward(idx).float()
    tgt = tgt_all[sb]
    logp = F.log_softmax(base[:, :SEQL-1], dim=-1)
    base_ce = -logp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1)
    del logp
    vmask = torch.from_numpy(valid_next[sb, :SEQL-1]).to(DEV)
    for (comp, kind, li, ix) in PATHS:
        abl = forward(idx, ablations=[(kind, li, ix)], yhmeans=YHMEAN, projmeans=PROJMEAN).float()
        alogp = F.log_softmax(abl[:, :SEQL-1], dim=-1)
        abl_ce = -alogp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1)
        del alogp
        dce = (abl_ce - base_ce)
        dg = dce[vmask]
        g_sum[comp] += float(dg.sum()); g_sq[comp] += float((dg*dg).sum()); g_n[comp] += int(dg.numel())
        tm = torch.from_numpy(trig_mask[comp][sb, :SEQL-1]).to(DEV)
        if tm.any():
            dt = dce[tm]
            t_sum[comp] += float(dt.sum()); t_sq[comp] += float((dt*dt).sum())
            t_n[comp] += int(dt.numel()); t_pos[comp] += int((dt > 0).sum())
            dl = (base[:, :SEQL-1] - abl[:, :SEQL-1])[tm]     # (n_fire_batch, V)
            classsum[comp] += CMAT @ dl.sum(0)
            nfire[comp] += int(dl.shape[0])
        del abl, dce
    if bi % 4 == 0:
        print(f"  batch {bi+1}/{(NHELD+BATCH-1)//BATCH}  elapsed {time.time()-t0:.0f}s", flush=True)
    del base, base_ce
print(f"PASS B done in {time.time()-t0:.0f}s", flush=True)

# assemble records with class-push score (NO census cleanliness on swiglu18)
records = []
for (comp, kind, li, ix) in PATHS:
    gm, gse = stats(g_sum[comp], g_sq[comp], g_n[comp])
    tm_, tse = stats(t_sum[comp], t_sq[comp], t_n[comp])
    cs = classsum[comp].cpu().numpy() / max(1, nfire[comp])
    abs_total = float(np.abs(cs).sum())
    order = np.argsort(-np.abs(cs))
    pushed = None
    for j in order:
        if CLASS_LIST[j] not in PUSH_EXCLUDE:
            pushed = j; break
    push_class = CLASS_LIST[pushed]
    push_val = float(cs[pushed])
    concentration = float(abs(cs[pushed]) / abs_total) if abs_total > 0 else 0.0
    class_dict = {CLASS_LIST[j]: round(float(cs[j]), 4) for j in order[:8]}
    records.append({
        'comp': comp, 'kind': kind, 'li': li, 'idx': ix,
        'cleanliness': None,
        'trigger_dCE': round(tm_, 5), 'trigger_dCE_SE': round(tse, 5),
        'trigger_dCE_z': round(tm_/tse, 3) if tse > 0 else 0.0,
        'trigger_dCE_frac_pos': round(t_pos[comp]/max(1, t_n[comp]), 3),
        'global_dCE': round(gm, 6),
        'pushed_class': push_class, 'pushed_class_summed_dlogit': round(push_val, 4),
        'pushed_class_sign': '+' if push_val >= 0 else '-',
        'class_concentration': round(concentration, 4),
        'class_summed_top8': class_dict,
        'n_firing': nfire[comp],
        'classpush_score': round(max(tm_, 0.0) * concentration, 6),
    })

def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() == 0 or b.std() == 0: return 0.0
    return float(np.corrcoef(a, b)[0, 1])
def spearman(a, b):
    return pearson(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))

tdce = np.array([r['trigger_dCE'] for r in records])
cps = np.array([r['classpush_score'] for r in records])
conc_arr = np.array([r['class_concentration'] for r in records])

correlations = {
    'pearson_classpush_vs_trigger_dCE': round(pearson(cps, tdce), 4),
    'spearman_classpush_vs_trigger_dCE': round(spearman(cps, tdce), 4),
    'pearson_concentration_alone_vs_trigger_dCE': round(pearson(conc_arr, tdce), 4),
    'note': 'swiglu18 has no census -> no cleanliness baseline; §68 check is classpush_score vs trigger_dCE.',
}

ranked = sorted(records, key=lambda r: -r['classpush_score'])
print("\n===== TOP 20 by CLASS-PUSH SCORE (causal importance x class concentration) =====", flush=True)
for r in ranked[:20]:
    print(f"  {r['comp']:12s} score={r['classpush_score']:.4f} tdCE={r['trigger_dCE']:.3f}"
          f"+-{r['trigger_dCE_SE']:.3f} z={r['trigger_dCE_z']:.1f} conc={r['class_concentration']:.2f}"
          f" push={r['pushed_class_sign']}{r['pushed_class']}", flush=True)

# =====================================================================================
# CAUSAL VERIFICATION of top class-push candidates: class-summed drop at FIRING vs
# matched INACTIVE control (specificity). Also OUTPUT ENTROPY / top-token share of the
# pushed-class delta to confirm the push is DISTRIBUTED (near-uniform) not a sharp token.
# =====================================================================================
CAND = [r for r in ranked if r['trigger_dCE_z'] >= 3.0 and r['trigger_dCE'] >= 0.02][:8]
print(f"\n===== CAUSAL VERIFICATION of top {len(CAND)} class-push candidates =====", flush=True)

@torch.no_grad()
def classsummed_per_position(mask, kind, li, ix, class_j):
    """per-position class-summed delta-logit (base-ablated) for one class over a mask."""
    row = CMAT[class_j]
    vals = []
    seqs = np.where(mask.any(axis=1))[0]
    for i in range(0, len(seqs), BATCH):
        sb = seqs[i:i+BATCH]; idx = HELD[sb]
        base = forward(idx).float()
        abl = forward(idx, ablations=[(kind, li, ix)], yhmeans=YHMEAN, projmeans=PROJMEAN).float()
        dl = (base - abl)
        cs = torch.einsum('btv,v->bt', dl, row)
        mk = torch.from_numpy(mask[sb]).to(DEV)
        vals.append(cs[mk].cpu().numpy())
        del base, abl, dl, cs
    return np.concatenate(vals) if vals else np.zeros(0)

@torch.no_grad()
def push_distribution(mask, kind, li, ix, class_j):
    """Within the pushed class, characterize HOW distributed the push is:
    accumulate mean POSITIVE delta-logit (base-ablated) per in-class token over firing
    positions; report normalized entropy over class tokens and top-token share.
    A near-uniform (distributed) push -> entropy ~1, top-token share small."""
    in_class = (CMAT[class_j] > 0)
    idx_in = torch.nonzero(in_class, as_tuple=False).squeeze(-1)   # class token ids
    acc = torch.zeros(idx_in.numel(), device=DEV); nfire = 0
    seqs = np.where(mask.any(axis=1))[0]
    for i in range(0, len(seqs), BATCH):
        sb = seqs[i:i+BATCH]; idx = HELD[sb]
        base = forward(idx).float(); abl = forward(idx, ablations=[(kind, li, ix)], yhmeans=YHMEAN, projmeans=PROJMEAN).float()
        dl = (base - abl)[:, :, idx_in]                            # (b,T,|class|) positive => path boosts
        mk = torch.from_numpy(mask[sb]).to(DEV)
        sel = dl[mk]                                               # (n_fire, |class|)
        acc += sel.sum(0); nfire += sel.shape[0]
        del base, abl, dl, sel
    mean_push = (acc / max(1, nfire))
    pos = mean_push.clamp_min(0.0)                                 # push mass on boosted tokens
    tot = float(pos.sum())
    if tot <= 0:
        return {'push_direction': 'suppress', 'entropy_norm': None, 'top1_share': None,
                'top8_share': None, 'n_class_tokens': int(idx_in.numel()),
                'frac_tokens_moved_same_sign': round(float((mean_push < 0).float().mean()), 3)}
    prob = pos / tot
    nz = prob[prob > 0]
    ent = float(-(nz * nz.log()).sum())
    ent_norm = ent / math.log(idx_in.numel())
    sortv = torch.sort(pos, descending=True).values
    top1 = float(sortv[0] / tot); top8 = float(sortv[:8].sum() / tot)
    return {'push_direction': 'boost', 'entropy_norm': round(ent_norm, 4),
            'top1_share': round(top1, 4), 'top8_share': round(top8, 4),
            'n_class_tokens': int(idx_in.numel()),
            'frac_tokens_moved_same_sign': round(float((mean_push > 0).float().mean()), 3)}

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
    dist = push_distribution(trig_mask[comp], kind, li, ix, cj)
    verdict = ('SPECIFIC-CLASS-PUSH' if (fmean > 0 and fmean/(fse+1e-9) >= 3 and spec/(spec_se+1e-9) >= 3)
               else 'PUSH-BUT-NOT-SPECIFIC' if (fmean > 0 and fmean/(fse+1e-9) >= 3)
               else 'FAILS-CLASS-PUSH')
    rec = {
        'comp': comp, 'kind': kind, 'li': li, 'pushed_class': r['pushed_class'],
        'trigger_dCE': r['trigger_dCE'], 'trigger_dCE_z': r['trigger_dCE_z'],
        'class_concentration': r['class_concentration'], 'classpush_score': r['classpush_score'],
        'classsummed_drop_at_firing': round(fmean, 4), 'classsummed_drop_at_firing_SE': round(fse, 4),
        'classsummed_at_control': round(cmean, 4), 'classsummed_at_control_SE': round(cse, 4),
        'specificity_firing_minus_control': round(spec, 4), 'specificity_SE': round(spec_se, 4),
        'specificity_z': round(spec/(spec_se+1e-9), 2),
        'firing_z': round(fmean/(fse+1e-9), 2),
        'push_distribution': dist,
        'class_summed_top8': r['class_summed_top8'],
        'verdict': verdict,
    }
    verify.append(rec)
    print(f"  {comp:12s} push={r['pushed_class']:9s} firing {fmean:.2f}+-{fse:.2f} (z{rec['firing_z']}) "
          f"ctrl {cmean:.2f}+-{cse:.2f}  spec {spec:.2f}+-{spec_se:.2f} (z{rec['specificity_z']}) "
          f"ent={dist['entropy_norm']} top1={dist['top1_share']} -> {verdict}", flush=True)

# =====================================================================================
# CROSS-ARCHITECTURE COMPARISON to bilin18 (qk_unsup_classpush.json), if present.
# =====================================================================================
cross = {}
try:
    bilin = json.load(open(f'{QK}/qk_unsup_classpush.json'))
    b_top = bilin['summary']['top15_by_classpush']
    b_verify = bilin['summary']['causal_verification']
    def push_by_class(recs, key='pushed_class'):
        c = Counter(x[key] for x in recs)
        return dict(c.most_common())
    cross = {
        'bilin18_top15_pushed_classes': push_by_class(b_top),
        'swiglu18_top15_pushed_classes': push_by_class(ranked[:15]),
        'bilin18_top15': [{'comp': x['comp'], 'pushed_class': x['pushed_class'],
                           'sign': x.get('pushed_class_sign'), 'classpush_score': x['classpush_score'],
                           'trigger_dCE': x['trigger_dCE']} for x in b_top],
        'bilin18_pearson_classpush_vs_trigger_dCE':
            bilin['summary']['correlations']['pearson_classpush_vs_trigger_dCE'],
        'bilin18_pearson_cleanliness_vs_trigger_dCE':
            bilin['summary']['correlations'].get('pearson_cleanliness_vs_trigger_dCE_ourrun'),
        'bilin18_verified_specific':
            [{'comp': x['comp'], 'pushed_class': x['pushed_class'], 'verdict': x['verdict']}
             for x in b_verify if x['verdict'] == 'SPECIFIC-CLASS-PUSH'],
    }
except Exception as e:
    cross = {'error': f'could not load bilin18 comparison: {e}'}

# late-layer feed-forward summary (where bilin18's integrators live)
late_start = NL - 3
late_ff = [r for r in ranked if r['kind'] == 'mlp' and r['li'] >= late_start]
late_ff_summary = [{'comp': r['comp'], 'li': r['li'], 'pushed_class': r['pushed_class'],
                    'pushed_class_sign': r['pushed_class_sign'], 'trigger_dCE': r['trigger_dCE'],
                    'trigger_dCE_z': r['trigger_dCE_z'], 'class_concentration': r['class_concentration'],
                    'classpush_score': r['classpush_score']}
                   for r in sorted(late_ff, key=lambda r: -r['classpush_score'])]

summary = {
    'correlations': correlations,
    'top15_by_classpush': [{'comp': r['comp'], 'classpush_score': r['classpush_score'],
                            'trigger_dCE': r['trigger_dCE'], 'trigger_dCE_z': r['trigger_dCE_z'],
                            'class_concentration': r['class_concentration'],
                            'pushed_class': r['pushed_class'], 'pushed_class_sign': r['pushed_class_sign']}
                           for r in ranked[:15]],
    'late_feedforward_class_pushers': late_ff_summary,
    'causal_verification': verify,
    'cross_architecture_vs_bilin18': cross,
}
out = {
    'meta': {
        'model': 'swiglu18', 'arch': 'softmax attention + SwiGLU MLP',
        'held_slice': 'FW[448:600,:128]', 'n_paths': len(PATHS),
        'NL': NL, 'NH': NH, 'N_SVD': N_SVD,
        'KCAUSAL': KCAUSAL, 'BATCH': BATCH, 'n_classes': len(CLASS_LIST), 'classes': CLASS_LIST,
        'class_sizes': CLASS_SIZE, 'push_excluded_classes': sorted(PUSH_EXCLUDE),
        'detector': 'CAUSAL CLASS-LEVEL EFFECT (class-push), ADAPTED to swiglu18 SOFTMAX attention + '
                    'SwiGLU MLP. Rank = trigger-position mean-ablation delta cross-entropy x CLASS-LEVEL '
                    'output concentration. Forward ADAPTED from qk_unsup_classpush.py (bilin18): two-branch '
                    'squared unnormalized attention replaced by softmax (verbatim from '
                    'qk_content_gate_swiglu18.py), second QK branch dropped; MLP is blk.mlp black box; '
                    'MLP SVD dirs / head + MLP mean-ablation / class library unchanged.',
        'verification': 'class-summed delta-logit for the pushed class at FIRING vs matched INACTIVE '
                        'control, paired SE; plus pushed-class output entropy / top-token share to '
                        'confirm the push is DISTRIBUTED (near-uniform) not a sharp token set.',
    },
    'summary': summary,
    'records': ranked,
}
json.dump(out, open(f'{QK}/qk_general_classpush_swiglu.json', 'w'), indent=2)

print("\n===== CORRELATION (does the §68 blind-spot story hold on softmax swiglu18?) =====", flush=True)
for k, v in correlations.items(): print(f"  {k}: {v}", flush=True)
print("\n===== LATE FEED-FORWARD CLASS-PUSHERS (li >= %d) =====" % late_start, flush=True)
for r in late_ff_summary[:12]:
    print(f"  {r['comp']:12s} push={r['pushed_class_sign']}{r['pushed_class']:9s} tdCE={r['trigger_dCE']:.3f}"
          f" z={r['trigger_dCE_z']:.1f} conc={r['class_concentration']:.2f} score={r['classpush_score']:.4f}", flush=True)
print("\n===== CROSS-ARCH vs bilin18 =====", flush=True)
if 'error' not in cross:
    print(f"  bilin18 top15 pushed classes: {cross['bilin18_top15_pushed_classes']}", flush=True)
    print(f"  swiglu18 top15 pushed classes: {cross['swiglu18_top15_pushed_classes']}", flush=True)
    print(f"  bilin18 pearson classpush-vs-dCE: {cross['bilin18_pearson_classpush_vs_trigger_dCE']}", flush=True)
print("\nSaved qk_general_classpush_swiglu.json", flush=True)
print("QK GENERAL CLASSPUSH SWIGLU DONE", flush=True)
