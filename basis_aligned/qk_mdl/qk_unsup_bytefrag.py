"""BYTE-FRAGMENT / ORTHOGRAPHIC-TRIGGER detector (§58 tool-gap: orthographic circuit TYPE).

The §56 class-based trigger fingerprints characterise a path's trigger by grammar/POS/
frequency-class concentration. That MISSES heads / MLP output directions whose trigger is a
sub-word BYTE or CHARACTER pattern -- a shared suffix ("-ing", "-ed", "-tion"), prefix ("un-",
"re-"), capitalisation, a digit-run, punctuation, whitespace-leading, byte-length, or a shared
internal character n-gram -- rather than a semantic token class.

Method (all forward passes copy the bilin18 convention VERBATIM from qk_unsup_discover.py /
qk_unsup_verify.py -- two-branch (q1k1/HD)(q2k2/HD) masked UNNORMALISED pattern, per-head QK
rms_norm THEN RoPE, v-lerp via a.lamb block-0 cache, 30*tanh logits):

  1. DISCOVERY (FW[0:256,:128] TRAIN, same as §56): for each of the 162 attention-head paths and
     72 MLP-SVD output directions, take the top-K firing positions. Decode their CURRENT tokens
     (and for heads their ATTENDED-SOURCE tokens) to strings. Score each path against an
     orthographic-predicate library: shared suffix (2/3 char), shared prefix (2/3 char),
     capitalised, all-caps, contains-digit, punctuation, whitespace-leading, byte-length bucket,
     shared internal 3-gram. PURITY = activation-weighted fraction of the path's trigger mass whose
     tokens satisfy the predicate; LIFT = purity / corpus base-rate of that predicate. Rank paths
     by best-predicate purity*lift, gated on magnitude and on >=3 DISTINCT satisfying tokens (so a
     single repeated byte-fallback token cannot masquerade as a "suffix").

  2. OUT-OF-SAMPLE (FW[256:448,:128], disjoint from both TRAIN and the causal slice -- the spec's
     FW[600:700] does not exist, the array has only 600 sequences): recompute each winner's chosen
     predicate purity to confirm the fingerprint is stable, not overfit.

  3. CAUSAL (held-back FW[448:600,:128], untouched by discovery): mean-ablate the path (per-position
     in-distribution mean, not zero) and measure the cross-entropy increase (ablated - base) over
     the full held grid with paired per-token STANDARD ERROR. Then CONDITION on the orthographic
     predicate: compare the causal cost on positions whose current (or attended-source, or next)
     token satisfies the pattern vs positions that do not. A genuine byte-fragment circuit
     concentrates its causal damage on pattern-matching positions.

  4. Honest negatives: if a top "suffix" winner is really a POS / semantic class in disguise, say so.
"""
import json, sys, math, os
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from transformers import AutoTokenizer
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
print(f"bilin18: NL={NL} NH={NH} HD={HD} D={D} V={V}  -> {NL*NH} head-paths", flush=True)
tok = AutoTokenizer.from_pretrained('gpt2')   # model exposes none via load_elriggs (matches §56) -- FLAGGED

FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
SEQL = 128
TRAIN = FINEWEB[0:256, :SEQL].to(DEV)         # discovery slice (matches §56)
OOS   = FINEWEB[256:448, :SEQL].to(DEV)       # out-of-sample purity slice (disjoint from TRAIN & HELD)
HELD  = FINEWEB[448:600, :SEQL].to(DEV)       # held-back causal slice (untouched by discovery)
BATCH = 8
K = 50                                        # top-K firing positions per path
N_SVD = 4                                     # MLP right-singular-vecs per block

# special / degenerate token ids: doc-separator + UTF-8 replacement (byte-fallback) tokens.
_special = {tok.eos_token_id}
for _t in range(V):
    if '�' in tok.decode([_t]): _special.add(_t)
SPECIAL = np.array(sorted(_special))
print(f"{len(SPECIAL)} special token ids masked from trigger selection (eos={tok.eos_token_id})", flush=True)

# =====================================================================================
# VERBATIM forward: PASS A collects head-output norm, argmax|attn| source, resid norm, MLP gram,
# MLP-dir projection. Returns per-path activation matrices for the given data slice.
# =====================================================================================
def alloc(nseq):
    return (np.zeros((NL*NH, nseq*SEQL), np.float32),   # head_act
            np.zeros((NL*NH, nseq*SEQL), np.int32),     # head_src (token at argmax|attn| key)
            np.zeros((NL*N_SVD, nseq*SEQL), np.float32),# mlp_proj
            np.zeros((NL, nseq*SEQL), np.float32))      # resid_nrm

@torch.no_grad()
def forward_collect(data, mlp_dirs=None, gram=None):
    """If gram given: accumulate uncentred MLP-output gram (to derive dirs). If mlp_dirs given:
    also record mlp projections. Always record head_act/head_src/resid_nrm."""
    nseq = data.shape[0]
    head_act, head_src, mlp_proj, resid_nrm = alloc(nseq)
    for i in range(0, nseq, BATCH):
        idx = data[i:i+BATCH]; B, T = idx.shape; p0 = i*SEQL; f0, f1 = p0, p0+B*T
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
            pat = (s1*s2).masked_fill(~mask, 0.0)
            yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
            Wr = a.c_proj.weight.view(D, NH, HD)
            comp = torch.einsum('bthc,ohc->btho', yh4, Wr)          # (B,T,NH,D)
            hn = comp.norm(dim=-1)                                  # (B,T,NH)
            srcpos = pat.abs().argmax(-1)                           # (B,NH,T)
            for h in range(NH):
                head_act[li*NH + h, f0:f1] = hn[:, :, h].reshape(-1).float().cpu().numpy()
                head_src[li*NH + h, f0:f1] = torch.gather(idx, 1, srcpos[:, h, :]).reshape(-1).cpu().numpy()
            x = x + a.c_proj(yh4.reshape(B, T, -1))
            mo = blk.mlp(F.rms_norm(x, (D,)))
            if gram is not None:
                gram[li] += torch.einsum('btd,bte->de', mo, mo)
            if mlp_dirs is not None:
                pr = torch.einsum('btd,nd->btn', mo, mlp_dirs[li])  # (B,T,N_SVD)
                for kk in range(N_SVD):
                    mlp_proj[li*N_SVD + kk, f0:f1] = pr[:, :, kk].reshape(-1).float().cpu().numpy()
            x = x + mo
            resid_nrm[li, f0:f1] = x.norm(dim=-1).reshape(-1).float().cpu().numpy()
    return head_act, head_src, mlp_proj, resid_nrm

print("PASS A (TRAIN): head norms + attn-source + resid + MLP gram ...", flush=True)
gram = [torch.zeros(D, D, device=DEV) for _ in range(NL)]
ha_tr, hs_tr, _, rn_tr = forward_collect(TRAIN, gram=gram)
mlp_dirs = torch.zeros(NL, N_SVD, D, device=DEV)
for li in range(NL):
    evals, evecs = torch.linalg.eigh(gram[li])
    mlp_dirs[li] = evecs[:, -N_SVD:].T.flip(0)                     # top-N_SVD descending
print("MLP directions computed. Re-pass for MLP projections ...", flush=True)
_, _, mp_tr, _ = forward_collect(TRAIN, mlp_dirs=mlp_dirs)
del gram

# =====================================================================================
# Orthographic predicate library over DECODED trigger strings
# =====================================================================================
_decode_cache = {}
def props(t):
    t = int(t)
    if t in _decode_cache: return _decode_cache[t]
    s = tok.decode([t])
    core = s.strip()
    p = {
        'decoded': s, 'core': core,
        'ws_lead': bool(s[:1].isspace()),
        'is_cap': bool(len(core) >= 1 and core[0].isalpha() and core[0].isupper()),
        'is_allcaps': bool(len(core) >= 2 and core.isupper() and any(c.isalpha() for c in core)),
        'has_digit': bool(any(c.isdigit() for c in core)),
        'is_punct': bool(len(core) >= 1 and all(not c.isalnum() for c in core)),
        'suffix2': core[-2:] if len(core) >= 2 else None,
        'suffix3': core[-3:] if len(core) >= 3 else None,
        'prefix2': core[:2] if len(core) >= 2 else None,
        'prefix3': core[:3] if len(core) >= 3 else None,
        'lenbucket': str(len(core)) if len(core) <= 5 else '6+',
        'ngrams3': [core[j:j+3] for j in range(len(core)-2)] if len(core) >= 3 else [],
    }
    _decode_cache[t] = p
    return p

FIXED = ['ws_lead', 'is_cap', 'is_allcaps', 'has_digit', 'is_punct']
PARAM = ['suffix2', 'suffix3', 'prefix2', 'prefix3', 'lenbucket']   # + ngram3 handled specially

# ---- corpus base rates (weighted by token frequency over the TRAIN slice) ----
corpus_ids, corpus_cnt = np.unique(TRAIN.cpu().numpy().reshape(-1), return_counts=True)
tot_corpus = float(corpus_cnt.sum())
base_fixed = {f: 0.0 for f in FIXED}
base_param = {p: defaultdict(float) for p in PARAM}
base_ngram = defaultdict(float)
for t, c in zip(corpus_ids, corpus_cnt):
    if t in SPECIAL: continue
    P = props(t); c = float(c)
    for f in FIXED:
        if P[f]: base_fixed[f] += c
    for p in PARAM:
        if P[p] is not None: base_param[p][P[p]] += c
    for g in set(P['ngrams3']): base_ngram[g] += c
for f in FIXED: base_fixed[f] /= tot_corpus
for p in PARAM:
    for kk in base_param[p]: base_param[p][kk] /= tot_corpus
for g in base_ngram: base_ngram[g] /= tot_corpus
EPS = 0.5 / tot_corpus     # base-rate floor for a value never seen in corpus

def score_axis(ids, weights):
    """ids/weights: trigger tokens (current OR source) with activation-magnitude weights.
    Returns the single best orthographic predicate for this axis as a dict, or None."""
    tw = defaultdict(float)
    for t, w in zip(ids, weights):
        if int(t) in SPECIAL: continue
        tw[int(t)] += float(w)
    if not tw: return None
    total = sum(tw.values())
    cands = []
    # fixed predicates
    for f in FIXED:
        mass = sum(w for t, w in tw.items() if props(t)[f])
        distinct = sum(1 for t in tw if props(t)[f])
        if distinct >= 3 and mass > 0:
            pur = mass/total; base = max(base_fixed[f], EPS)
            cands.append(dict(pred=f, value=None, purity=pur, base=base, lift=pur/base,
                              distinct=distinct, examples=_examples(tw, lambda t: props(t)[f])))
    # parametric predicates: choose the value with max weighted mass
    for p in PARAM:
        val_mass = defaultdict(float); val_ids = defaultdict(set)
        for t, w in tw.items():
            vv = props(t)[p]
            if vv is not None: val_mass[vv] += w; val_ids[vv].add(t)
        for vv, mass in val_mass.items():
            distinct = len(val_ids[vv])
            if distinct >= 3:
                pur = mass/total; base = max(base_param[p].get(vv, 0.0), EPS)
                cands.append(dict(pred=p, value=vv, purity=pur, base=base, lift=pur/base,
                                  distinct=distinct, examples=_examples(tw, lambda t: props(t)[p] == vv)))
    # shared internal 3-gram
    g_mass = defaultdict(float); g_ids = defaultdict(set)
    for t, w in tw.items():
        for g in set(props(t)['ngrams3']): g_mass[g] += w; g_ids[g].add(t)
    for g, mass in g_mass.items():
        distinct = len(g_ids[g])
        if distinct >= 3:
            pur = mass/total; base = max(base_ngram.get(g, 0.0), EPS)
            cands.append(dict(pred='ngram3', value=g, purity=pur, base=base, lift=pur/base,
                              distinct=distinct, examples=_examples(tw, lambda t: g in props(t)['ngrams3'])))
    if not cands: return None
    for c in cands: c['score'] = c['purity'] * c['lift']
    return max(cands, key=lambda c: c['score'])

def _examples(tw, pred_fn, n=6):
    ex = sorted([t for t in tw if pred_fn(t)], key=lambda t: -tw[t])[:n]
    return [props(t)['decoded'] for t in ex]

# ---- top-K trigger positions per path (exclude t=0, specials) ----
def topk_positions(act, nseq, exclude_ids_flat):
    pos_t = np.tile(np.arange(SEQL), nseq)
    a = act.copy(); a[(pos_t == 0) | np.isin(exclude_ids_flat, SPECIAL)] = -1.0
    tk = np.argpartition(a, -K)[-K:]
    return tk, act[tk]

train_ids_flat = TRAIN.cpu().numpy().reshape(-1)

def score_path(kind, act, src=None, nseq=256, ids_flat=None):
    tk, wt = topk_positions(np.abs(act), nseq, ids_flat)
    cur = ids_flat[tk]
    best_cur = score_axis(cur, wt)
    best_src = None
    if src is not None:
        best_src = score_axis(src[tk], wt)
    # pick the better axis
    opts = [(ax, b) for ax, b in [('cur', best_cur), ('src', best_src)] if b is not None]
    if not opts: return None
    axis, best = max(opts, key=lambda ab: ab[1]['score'])
    best = dict(best); best['axis'] = axis
    return best, tk

# ---- rank all 234 paths ----
records = []
for li in range(NL):
    for h in range(NH):
        p = li*NH + h
        r = score_path('head', ha_tr[p], src=hs_tr[p], nseq=256, ids_flat=train_ids_flat)
        if r is None: continue
        best, tk = r
        mag = float(np.mean(ha_tr[p, tk] / (rn_tr[li, tk] + 1e-9)))
        records.append(dict(comp=f"h.L{li}.{h}", kind='head', li=li, h=h, mag=mag, **best))
for li in range(NL):
    for kk in range(N_SVD):
        p = li*N_SVD + kk
        r = score_path('mlp', mp_tr[p], src=None, nseq=256, ids_flat=train_ids_flat)
        if r is None: continue
        best, tk = r
        mag = float(np.mean(np.abs(mp_tr[p, tk]) / (rn_tr[li, tk] + 1e-9)))
        records.append(dict(comp=f"mlp.L{li}.d{kk}", kind='mlp', li=li, k=kk, mag=mag, **best))

mags = np.array([r['mag'] for r in records])
gate = float(np.percentile(mags, 25))
for r in records: r['passes_mag_gate'] = bool(r['mag'] >= gate)
ranked = sorted([r for r in records if r['passes_mag_gate']], key=lambda r: -r['score']) + \
         sorted([r for r in records if not r['passes_mag_gate']], key=lambda r: -r['score'])

print("\n===== TOP 20 ORTHOGRAPHIC-TRIGGER PATHS (TRAIN, ranked purity*lift, mag-gated) =====", flush=True)
for r in ranked[:20]:
    print(f"{r['comp']:12s} [{r['axis']:3s}] {r['pred']:9s} val={str(r['value'])!r:>8} "
          f"pur={r['purity']:.3f} lift={r['lift']:6.1f} score={r['score']:7.2f} distinct={r['distinct']} "
          f"mag={r['mag']:.3f}  ex={r['examples']}", flush=True)

# =====================================================================================
# OUT-OF-SAMPLE purity on FW[256:448] for the top candidates
# =====================================================================================
def predicate_hits(pred, value):
    """returns a function id->bool for the chosen predicate."""
    if pred in FIXED:
        return lambda t: props(int(t))[pred]
    if pred == 'ngram3':
        return lambda t: value in props(int(t))['ngrams3']
    return lambda t: props(int(t))[pred] == value

def axis_purity_oos(kind, act, src, pred, value, axis, nseq, ids_flat):
    tk, wt = topk_positions(np.abs(act), nseq, ids_flat)
    ids = ids_flat[tk] if axis == 'cur' else src[tk]
    fn = predicate_hits(pred, value)
    tot = 0.0; hit = 0.0; distinct = set()
    for t, w in zip(ids, wt):
        if int(t) in SPECIAL: continue
        tot += w
        if fn(t): hit += w; distinct.add(int(t))
    return (hit/tot if tot > 0 else 0.0), len(distinct)

TOPN = 14
cand = ranked[:TOPN]
print(f"\nOUT-OF-SAMPLE: recompute top-{TOPN} winners' purity on FW[256:448] ...", flush=True)
ha_o, hs_o, _, _ = forward_collect(OOS)
_, _, mp_o, _ = forward_collect(OOS, mlp_dirs=mlp_dirs)
oos_ids_flat = OOS.cpu().numpy().reshape(-1)
for r in cand:
    if r['kind'] == 'head':
        p = r['li']*NH + r['h']
        pur, dist = axis_purity_oos('head', ha_o[p], hs_o[p], r['pred'], r['value'], r['axis'], 192, oos_ids_flat)
    else:
        p = r['li']*N_SVD + r['k']
        pur, dist = axis_purity_oos('mlp', mp_o[p], None, r['pred'], r['value'], 'cur', 192, oos_ids_flat)
    r['oos_purity'] = round(float(pur), 4); r['oos_distinct'] = dist
    print(f"  {r['comp']:12s} {r['pred']:9s} val={str(r['value'])!r:>8}  train_pur={r['purity']:.3f}  "
          f"oos_pur={r['oos_purity']:.3f}  oos_distinct={dist}", flush=True)

# =====================================================================================
# CAUSAL verification on HELD FW[448:600] for the top ~6 genuine orthographic candidates
# =====================================================================================
# select: stable (oos_purity>=0.5), genuine orthographic (distinct>=4), best score.
genuine = [r for r in cand if r['oos_purity'] >= 0.5 and r['distinct'] >= 4]
CAUSAL_PATHS = genuine[:6] if len(genuine) >= 3 else cand[:6]
print(f"\nCAUSAL targets ({len(CAUSAL_PATHS)}): {[r['comp'] for r in CAUSAL_PATHS]}", flush=True)

NHELD = HELD.shape[0]
held_np = HELD.cpu().numpy()
pos_t = np.tile(np.arange(SEQL), NHELD).reshape(NHELD, SEQL)
is_special_h = np.isin(held_np, SPECIAL)

# recompute MLP dirs used for ablation from TRAIN gram already have mlp_dirs (global). Good.
# collect per-position means (YHMEAN all layers) + per-candidate head/mlp activation + src on HELD
YH_SUM = {li: torch.zeros(SEQL, NH, HD, device=DEV) for li in range(NL)}
MLP_CAND = [(r['li'], r['k']) for r in CAUSAL_PATHS if r['kind'] == 'mlp']
PROJ_SUM = {lk: torch.zeros(SEQL, device=DEV) for lk in MLP_CAND}
h_act_held = {}; h_src_held = {}; m_proj_held = {}

@torch.no_grad()
def collect_held():
    for i in range(0, NHELD, BATCH):
        idx = HELD[i:i+BATCH]; B, T = idx.shape
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
            pat = (s1*s2).masked_fill(~mask, 0.0)
            yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
            Wr = a.c_proj.weight.view(D, NH, HD)
            YH_SUM[li] += yh4.sum(0)
            srcpos = pat.abs().argmax(-1)
            for r in CAUSAL_PATHS:
                if r['kind'] == 'head' and r['li'] == li:
                    th = r['h']
                    comp = torch.einsum('btc,oc->bto', yh4[:, :, th], Wr[:, th])
                    h_act_held.setdefault(r['comp'], np.zeros((NHELD, SEQL), np.float32))[i:i+B] = comp.norm(dim=-1).cpu().numpy()
                    h_src_held.setdefault(r['comp'], np.zeros((NHELD, SEQL), np.int64))[i:i+B] = torch.gather(idx, 1, srcpos[:, th, :]).cpu().numpy()
            x = x + a.c_proj(yh4.reshape(B, T, -1))
            mo = blk.mlp(F.rms_norm(x, (D,)))
            for (lk_li, lk_k) in MLP_CAND:
                if lk_li == li:
                    pr = torch.einsum('btd,d->bt', mo, mlp_dirs[li, lk_k])
                    PROJ_SUM[(lk_li, lk_k)] += pr.sum(0)
                    key = f"mlp.L{lk_li}.d{lk_k}"
                    m_proj_held.setdefault(key, np.zeros((NHELD, SEQL), np.float32))[i:i+B] = pr.cpu().numpy()
            x = x + mo

print("PASS (HELD): collect means + activations ...", flush=True)
collect_held()
YHMEAN = {li: YH_SUM[li] / NHELD for li in range(NL)}
PROJMEAN = {lk: PROJ_SUM[lk] / NHELD for lk in MLP_CAND}

@torch.no_grad()
def forward_full(idx, ablate=None):
    """VERBATIM forward returning logits (B,T,V). ablate: None|('head',li,h)|('mlp',li,k)."""
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
        pat = (s1*s2).masked_fill(~mask, 0.0)
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        if ablate is not None and ablate[0] == 'head' and ablate[1] == li:
            yh4 = yh4.clone(); yh4[:, :, ablate[2]] = YHMEAN[li][:, ablate[2]].unsqueeze(0)
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if ablate is not None and ablate[0] == 'mlp' and ablate[1] == li:
            dvec = mlp_dirs[li, ablate[2]]
            pr = torch.einsum('btd,d->bt', mo, dvec)
            mo = mo - (pr - PROJMEAN[(li, ablate[2])].unsqueeze(0)).unsqueeze(-1) * dvec
        x = x + mo
    return 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)

# per-position base CE over the whole held grid (predict t -> t+1)
@torch.no_grad()
def ce_grid(ablate=None):
    ce = np.full((NHELD, SEQL), np.nan, np.float32)
    for i in range(0, NHELD, BATCH):
        idx = HELD[i:i+BATCH]; B = idx.shape[0]
        lg = forward_full(idx, ablate=ablate).float()
        lp = F.log_softmax(lg[:, :-1], dim=-1)                 # (B,T-1,V)
        tgt = idx[:, 1:]
        nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)     # (B,T-1)
        ce[i:i+B, :-1] = nll.cpu().numpy()
    return ce

print("CAUSAL: base CE grid ...", flush=True)
base_ce = ce_grid(None)
# valid positions: not t=0, current & next not special, has a next token
valid = (pos_t > 0) & (pos_t < SEQL-1) & ~is_special_h
next_special = np.zeros_like(is_special_h); next_special[:, :-1] = is_special_h[:, 1:]
valid = valid & ~next_special

def paired(dce_flat):
    n = len(dce_flat)
    if n == 0: return dict(n=0, mean=None, se=None, frac_pos=None)
    return dict(n=int(n), mean=round(float(dce_flat.mean()), 4),
                se=round(float(dce_flat.std(ddof=1)/math.sqrt(n)) if n > 1 else float('nan'), 4),
                frac_pos=round(float((dce_flat > 0).mean()), 3))

CAUSAL_RESULTS = {}
for r in CAUSAL_PATHS:
    comp = r['comp']; print(f"CAUSAL ablate {comp} ...", flush=True)
    if r['kind'] == 'head':
        abl_ce = ce_grid(('head', r['li'], r['h']))
        act = h_act_held[comp]; src = h_src_held[comp]
    else:
        abl_ce = ce_grid(('mlp', r['li'], r['k']))
        act = m_proj_held[comp]; src = None
    dce = abl_ce - base_ce                                       # (NHELD, SEQL) paired per position
    vmask = valid & np.isfinite(dce)
    uncond = paired(dce[vmask])

    # trigger top-K positions (discovery signature) on HELD
    a = np.abs(act).copy(); a[~vmask] = -1e30
    flat = a.reshape(-1); tk = np.argpartition(flat, -K)[-K:]
    trig_mask = np.zeros_like(vmask); trig_mask.reshape(-1)[tk] = True; trig_mask &= vmask
    trig = paired(dce[trig_mask])

    # conditional contrast on the orthographic predicate
    fn = predicate_hits(r['pred'], r['value'])
    cur_match = np.vectorize(lambda t: fn(t))(held_np) if r['axis'] != 'src' else None
    conds = {}
    def bucket(match_mask, label):
        mm = vmask & match_mask; nm = vmask & ~match_mask & np.isfinite(dce)
        conds[label] = dict(match=paired(dce[mm]), nomatch=paired(dce[nm & np.isfinite(dce)]))
    if r['kind'] == 'head' and r['axis'] == 'src':
        src_match = np.vectorize(lambda t: fn(t))(src)
        bucket(src_match & np.isfinite(dce), 'source_token')
    else:
        cur_match = np.vectorize(lambda t: fn(t))(held_np)
        bucket(cur_match & np.isfinite(dce), 'current_token')
    # next-token conditioning (does ablation break prediction OF pattern tokens?)
    nxt = np.zeros_like(held_np); nxt[:, :-1] = held_np[:, 1:]
    next_match = np.vectorize(lambda t: fn(t))(nxt)
    bucket(next_match & np.isfinite(dce), 'next_token')

    CAUSAL_RESULTS[comp] = dict(
        kind=r['kind'], predicate=r['pred'], value=r['value'], axis=r['axis'],
        train_purity=r['purity'], oos_purity=r['oos_purity'], lift=round(r['lift'], 2),
        distinct=r['distinct'], examples=r['examples'], mag=round(r['mag'], 4),
        unconditional_dCE=uncond, trigger_topK_dCE=trig, conditional=conds)
    m0 = conds.get('current_token', conds.get('source_token'))
    print(f"   uncond dCE={uncond['mean']}±{uncond['se']} (n={uncond['n']})  "
          f"match={m0['match']['mean']}±{m0['match']['se']} (n={m0['match']['n']})  "
          f"nomatch={m0['nomatch']['mean']}±{m0['nomatch']['se']}", flush=True)

# =====================================================================================
OUT = dict(
    meta=dict(model='bilin18', tokenizer='gpt2 (model exposes none) -- FLAGGED',
              discovery_data='FineWeb FW[0:256,:128] TRAIN',
              oos_data='FW[256:448,:128] (disjoint; spec FW[600:700] does not exist -- array is 600 seqs)',
              causal_data='FW[448:600,:128] held-back',
              n_head_paths=NL*NH, n_mlp_paths=NL*N_SVD, K_topk=K,
              predicate_library=FIXED + PARAM + ['ngram3'],
              purity='activation-weighted trigger-mass fraction satisfying predicate',
              lift='purity / corpus token-frequency base-rate; genuineness requires >=3 (rank) / >=4 (causal) DISTINCT satisfying tokens',
              causal='mean-ablate path (per-position in-distribution mean); paired per-token standard error; conditional buckets by predicate on current/source/next token'),
    ranking=[{k: v for k, v in r.items() if k != 'examples' or True} for r in ranked],
    oos_candidates=cand,
    causal=CAUSAL_RESULTS,
)
json.dump(OUT, open(f'{QK}/qk_unsup_bytefrag.json', 'w'), indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else int(o) if isinstance(o, (np.integer,)) else str(o))
print("\nSaved qk_unsup_bytefrag.json", flush=True)
print("QK UNSUP BYTEFRAG DONE", flush=True)
