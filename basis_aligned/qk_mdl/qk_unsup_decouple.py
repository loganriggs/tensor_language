"""TRIGGER-vs-OUTPUT DECOUPLING detector for bilin18 (unsupervised circuit discovery).

Motivation (§56 discovery lesson): the linear direct-to-logits EFFECT proxy is
unreliable precisely because for many paths the TRIGGER class (what fires the path)
and the OUTPUT class (what it boosts) are DIFFERENT -- a "remap"/translation circuit
(fires on a determiner, boosts a noun/word; fires on an open-quote, boosts a
close-quote; fires on a digit, boosts a unit-word). Existing tools assume trigger
approx output (copy/induction) or measure them tangled together (a single
`cleanliness = trigger_purity * effect_purity`). This tool measures the TRIGGER
token-class distribution and the OUTPUT (boosted) token-class distribution SEPARATELY
per path, scores DECOUPLING (how different they are), ranks paths by strong-but-CLEAN
decoupling (trigger concentrated on class A, output concentrated on a DIFFERENT class
B), then CAUSALLY verifies the top candidates -- because the OUTPUT proxy is
known-unreliable, the linear ranking is only a CANDIDATE GENERATOR.

Method:
 1. Per path build TWO class histograms over the same lexical class library the other
    unsup tools use (token_classes / sets from qk_unsup_cluster.py), resolved to a
    single label per token for clean, peaked histograms:
      (a) TRIGGER-class hist = classes of the CURRENT tokens at the path's top-K firing
          positions (count-weighted). Current token is the class the causal test
          conditions on (position t, token[t] = class A, predicting token[t+1]).
      (b) OUTPUT-class hist = classes of the tokens the path boosts, from the
          direct-to-logits boost distribution (softmax-excess weighted over top-M).
    DECOUPLING = 1 - overlap(trigger_hist, output_hist)  [histogram intersection].
    Require BOTH histograms peaked (top-class fraction >= PURITY_MIN) and top trigger
    class A != top output class B, so we get clean A->B remaps not diffuse noise.
    Rank by decoupling * trigger_purity * output_purity (mag-gated, load-bearing).
    Name each "fires on {A} -> boosts {B}".
 2. CAUSAL verify OUTPUT side (top N candidates): mean-ablate the path and check whether
    P(class B at next) actually DROPS at class-A trigger positions on the HELD-BACK
    slice FW[448:600], paired standard errors. Specificity control: same measurement at
    class-A positions where the path is INACTIVE (should show ~0 drop). A genuine
    decoupled remap: ablation reduces class-B output specifically at path-active class-A
    positions. If ablation does nothing there, the "remap" was a proxy artifact.
 3. Report overall delta-cross-entropy +/- SE of ablating each candidate on FW[448:600].

FORWARD CONVENTIONS COPIED VERBATIM from qk_unsup_discover.py (PASS A/B) and
qk_unsup_verify.py (mean-ablation causal harness): bilin18 two-branch pattern
(q1k1/HD)(q2k2/HD) masked UNNORMALISED, per-head QK rms_norm THEN RoPE, v-lerp via
a.lamb (block-0 v cache), 30*tanh logits. A from-scratch forward rewrite previously
gave a spurious 0.001 -- do NOT rewrite.
"""
import json, sys, math
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter
from transformers import AutoTokenizer
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0); np.random.seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
tok = AutoTokenizer.from_pretrained('gpt2')
print(f"bilin18: NL={NL} NH={NH} HD={HD} D={D} V={V}  -> {NL*NH} head-paths", flush=True)

FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
NSEQ, SEQL = 256, 128
IDX = FINEWEB[0:NSEQ, :SEQL].to(DEV)           # discovery TRAIN slice
HELD = FINEWEB[448:600, :SEQL].to(DEV)         # held-back causal slice (untouched by discovery)
NHELD = HELD.shape[0]
BATCH = 8
K = 50                                          # top-K firing positions per path (trigger)
M_EFF = 40                                       # top-M boosted tokens for output histogram
N_SVD = 4                                        # MLP right-singular-vecs per block
KCAUSAL = 200                                    # top-K positions for causal statistics
N_CAUSAL = 6                                      # number of top decoupled candidates to verify
PURITY_MIN = 0.40                                 # both hists must be at least this peaked
Npos = NSEQ * SEQL
idx_flat = IDX.reshape(-1).cpu().numpy()

# special/degenerate tokens (document-sep + UTF-8 replacement) -- excluded from trigger selection
_special = {tok.eos_token_id}
for _t in range(V):
    if '�' in tok.decode([_t]): _special.add(_t)
SPECIAL = np.array(sorted(_special))
is_special = np.isin(idx_flat, SPECIAL)
print(f"{len(SPECIAL)} special token ids masked (eos={tok.eos_token_id})", flush=True)

LMW = m.lm_head.weight                            # (V, D)

# =====================================================================================
# LEXICAL CLASS LIBRARY -- reuses the exact sets from qk_unsup_cluster.py (token_classes),
# resolved to ONE label per token (priority order) so histograms are clean distributions.
# Additions in-spirit of that library (needed for the detector to see content words and
# to distinguish open/close quotes): 'word' (leading-space lowercase alphabetic content
# word) and quote_open/quote_close split.
# =====================================================================================
BRACKETS_OPEN  = set("([{<")
BRACKETS_CLOSE = set(")]}>")
QUOTE_OPEN     = set("“‘`")             # curly-left / left-single / backtick
QUOTE_CLOSE    = set("”’")              # curly-right / right-single
QUOTE_STRAIGHT = set("\"'")
PUNCT  = set(".,;:!?—–-…*|/\\~@#%^&+=_")
COORDINATORS = {"and","or","but","nor","yet","so"}
DETERMINERS  = {"the","a","an","this","that","these","those","some","any","each",
                "every","no","another","such"}
PRONOUNS     = {"i","we","you","he","she","it","they","them","us","me","him","her","which","who"}

def lex1(s):
    """Single-label lexical class of a decoded token string (priority order)."""
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
    if body[0].isupper(): return 'capital'                      # proper-noun-ish / capitalized
    lead_space = s.startswith(' ')
    if lead_space and body.isalpha() and len(body) > 1: return 'word'   # content word
    if (not lead_space) and body.isalpha() and body[0].islower(): return 'subword'
    return 'other'

# vocab -> single class (cache)
VOCAB_CLASS = np.array([lex1(tok.decode([t])) for t in range(V)], dtype=object)
CLASS_LIST = sorted(set(VOCAB_CLASS.tolist()))
print(f"lexical classes: {CLASS_LIST}", flush=True)

def hist_from_tokens(token_ids, weights=None):
    """Single-label class histogram (normalised to sum 1) over token ids."""
    cls = VOCAB_CLASS[np.asarray(token_ids)]
    h = Counter()
    if weights is None: weights = np.ones(len(cls))
    for c, w in zip(cls, weights): h[c] += float(w)
    tot = sum(h.values())
    return {k: v/tot for k, v in h.items()} if tot > 0 else {}

def overlap(h1, h2):
    return sum(min(h1.get(c, 0.0), h2.get(c, 0.0)) for c in set(h1) | set(h2))

def top_class(h):
    return max(h.items(), key=lambda kv: kv[1]) if h else ('other', 0.0)

# =====================================================================================
# PASS A / PASS B  -- copied VERBATIM from qk_unsup_discover.py
# =====================================================================================
head_act = np.zeros((NL * NH, Npos), dtype=np.float32)
head_src = np.zeros((NL * NH, Npos), dtype=np.int32)
resid_nrm = np.zeros((NL, Npos), dtype=np.float32)
gram = [torch.zeros(D, D, device=DEV) for _ in range(NL)]

@torch.no_grad()
def forward_passA(idx, p0):
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
        Wr = a.c_proj.weight.view(D, NH, HD)
        comp = torch.einsum('bthc,ohc->btho', yh4, Wr)
        hn = comp.norm(dim=-1)
        srcpos = pat.abs().argmax(-1)
        f0, f1 = p0, p0 + B*T
        for h in range(NH):
            head_act[li*NH + h, f0:f1] = hn[:, :, h].reshape(-1).float().cpu().numpy()
            head_src[li*NH + h, f0:f1] = srcpos[:, h, :].reshape(-1).cpu().numpy()
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        mo = blk.mlp(F.rms_norm(x, (D,)))
        gram[li] += torch.einsum('btd,bte->de', mo, mo)
        x = x + mo
        resid_nrm[li, f0:f1] = x.norm(dim=-1).reshape(-1).float().cpu().numpy()

print("PASS A: head norms + attn-source + resid + MLP gram ...", flush=True)
for i in range(0, NSEQ, BATCH):
    forward_passA(IDX[i:i+BATCH], i*SEQL)
    if i % 64 == 0: print(f"  seq {i}/{NSEQ}", flush=True)

mlp_dirs = torch.zeros(NL, N_SVD, D, device=DEV)
for li in range(NL):
    evals, evecs = torch.linalg.eigh(gram[li])
    mlp_dirs[li] = evecs[:, -N_SVD:].T.flip(0)
print("MLP directions computed.", flush=True)

pos_t = np.tile(np.arange(SEQL), NSEQ)
head_topk = {}; head_mask = np.zeros((NL*NH, Npos), dtype=bool)
for p in range(NL*NH):
    a = head_act[p].copy(); a[(pos_t == 0) | is_special] = -1.0
    tk = np.argpartition(a, -K)[-K:]
    head_topk[p] = tk; head_mask[p, tk] = True

mlp_proj = np.zeros((NL * N_SVD, Npos), dtype=np.float32)
head_eff_sum = torch.zeros(NL*NH, D, device=DEV)

@torch.no_grad()
def forward_passB(idx, p0):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    f0, f1 = p0, p0 + B*T
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
        comp = torch.einsum('bthc,ohc->btho', yh4, Wr).reshape(B*T, NH, D)
        for h in range(NH):
            sel = head_mask[li*NH + h, f0:f1]
            if sel.any():
                head_eff_sum[li*NH + h] += comp[torch.from_numpy(sel).to(DEV), h, :].sum(0)
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        mo = blk.mlp(F.rms_norm(x, (D,)))
        pr = torch.einsum('btd,nd->btn', mo, mlp_dirs[li])
        for kk in range(N_SVD):
            mlp_proj[li*N_SVD + kk, f0:f1] = pr[:, :, kk].reshape(-1).float().cpu().numpy()
        x = x + mo

print("PASS B: MLP projections + head effect directions ...", flush=True)
for i in range(0, NSEQ, BATCH):
    forward_passB(IDX[i:i+BATCH], i*SEQL)
    if i % 64 == 0: print(f"  seq {i}/{NSEQ}", flush=True)

# =====================================================================================
# OUTPUT histogram from the direct-to-logits boost proxy (softmax-excess weighted)
# =====================================================================================
@torch.no_grad()
def output_hist(dir_vec):
    d = dir_vec / (dir_vec.norm() + 1e-9)
    logits = (LMW @ d).float(); logits = logits - logits.mean()
    top = torch.topk(logits, M_EFF)
    v = top.values; w = (v - v.min())
    ids = top.indices.cpu().numpy()
    if w.sum() <= 1e-9:
        return {}, ids[:8]
    wnp = (w / w.sum()).cpu().numpy()
    return hist_from_tokens(ids, wnp), ids[:8]

def dec(t): return repr(tok.decode([int(t)]))

records = []
# --- head paths ---
for li in range(NL):
    for h in range(NH):
        p = li*NH + h; tk = head_topk[p]
        cur = idx_flat[tk]                                  # current tokens at firing positions
        trig_hist = hist_from_tokens(cur)
        eff_dir = head_eff_sum[p] / max(1, head_mask[p].sum())
        out_h, boosted = output_hist(eff_dir)
        A, Ap = top_class(trig_hist); Bc, Bp = top_class(out_h)
        dec_score = 1.0 - overlap(trig_hist, out_h)
        mag = float(np.mean(head_act[p, tk] / (resid_nrm[li, tk] + 1e-9)))
        records.append({
            'comp': f"h.L{li}.{h}", 'kind': 'head', 'li': li, 'h': h,
            'trigger_class': A, 'trigger_purity': round(Ap, 4),
            'output_class': Bc, 'output_purity': round(Bp, 4),
            'decoupling': round(dec_score, 4),
            'score': round(dec_score * Ap * Bp, 4),
            'mag_resid_frac': round(mag, 4),
            'trigger_hist': {k: round(v, 3) for k, v in sorted(trig_hist.items(), key=lambda x: -x[1])},
            'output_hist': {k: round(v, 3) for k, v in sorted(out_h.items(), key=lambda x: -x[1])},
            'top8_boost': [dec(t) for t in boosted],
            'name': f"fires on {A} -> boosts {Bc}",
        })
# --- MLP direction paths ---
for li in range(NL):
    for kk in range(N_SVD):
        p = li*N_SVD + kk; pr = mlp_proj[p].copy()
        a = np.abs(pr); a[(pos_t == 0) | is_special] = -1.0
        tk = np.argpartition(a, -K)[-K:]
        cur = idx_flat[tk]
        trig_hist = hist_from_tokens(cur)
        sgn = np.sign(pr[tk].mean()) or 1.0
        out_h, boosted = output_hist(mlp_dirs[li, kk] * float(sgn))
        A, Ap = top_class(trig_hist); Bc, Bp = top_class(out_h)
        dec_score = 1.0 - overlap(trig_hist, out_h)
        mag = float(np.mean(np.abs(pr[tk]) / (resid_nrm[li, tk] + 1e-9)))
        records.append({
            'comp': f"mlp.L{li}.d{kk}", 'kind': 'mlp', 'li': li, 'k': kk,
            'trigger_class': A, 'trigger_purity': round(Ap, 4),
            'output_class': Bc, 'output_purity': round(Bp, 4),
            'decoupling': round(dec_score, 4),
            'score': round(dec_score * Ap * Bp, 4),
            'mag_resid_frac': round(mag, 4),
            'trigger_hist': {k: round(v, 3) for k, v in sorted(trig_hist.items(), key=lambda x: -x[1])},
            'output_hist': {k: round(v, 3) for k, v in sorted(out_h.items(), key=lambda x: -x[1])},
            'top8_boost': [dec(t) for t in boosted],
            'name': f"fires on {A} -> boosts {Bc}",
        })

# magnitude gate (load-bearing); decoupling gate: A != B and both peaked
mags = np.array([r['mag_resid_frac'] for r in records])
gate = float(np.percentile(mags, 25))
for r in records:
    r['passes_mag_gate'] = bool(r['mag_resid_frac'] >= gate)
    r['clean_remap'] = bool(r['passes_mag_gate']
                            and r['trigger_class'] != r['output_class']
                            and r['trigger_purity'] >= PURITY_MIN
                            and r['output_purity'] >= PURITY_MIN
                            and r['trigger_class'] not in ('special', 'other')
                            and r['output_class'] not in ('special', 'other'))

candidates = sorted([r for r in records if r['clean_remap']], key=lambda r: -r['score'])
others = sorted([r for r in records if not r['clean_remap']], key=lambda r: -r['score'])
ranked = candidates + others
print(f"\n{len(candidates)} clean-remap candidates (mag-gated, A!=B, both purity>={PURITY_MIN})", flush=True)
print("\n===== TOP DECOUPLED CANDIDATES =====", flush=True)
for r in candidates[:15]:
    print(f"{r['comp']:12s} {r['name']:34s} decoup={r['decoupling']:.3f} "
          f"trigP={r['trigger_purity']:.2f} outP={r['output_purity']:.2f} "
          f"score={r['score']:.3f} mag={r['mag_resid_frac']:.3f}", flush=True)

# pick top-N distinct-path candidates for causal verification
CAUSAL_CANDS = candidates[:N_CAUSAL]

# =====================================================================================
# CAUSAL harness -- mean-ablation forward copied VERBATIM from qk_unsup_verify.py,
# extended to read P(class B at next) clean vs ablated at class-A trigger positions.
# =====================================================================================
# recompute MLP dirs on TRAIN gram already have (mlp_dirs); for causal MLP ablation we need
# the specific dir per candidate -> reuse mlp_dirs[li,kk] (defined on TRAIN, same as discovery).

@torch.no_grad()
def forward_causal(idx, collect_heads=None, collect_mlp=None,
                   ablate=None, yhmeans=None, mdir=None, projmean=None):
    """VERBATIM bilin18 forward. collect_heads: list of (li,h) -> return comp-norm (B,T).
    ablate: None | ('head',li,h) | ('mlp',li,mdir). Returns (logits, collected)."""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    coll = {}
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
        if collect_heads:
            Wr = a.c_proj.weight.view(D, NH, HD)
            for (tli, th) in collect_heads:
                if tli == li:
                    comp = torch.einsum('btc,oc->bto', yh4[:, :, th], Wr[:, th])
                    coll[('hnorm', tli, th)] = comp.norm(dim=-1).cpu().numpy()
            YH_SUM[li] += yh4.sum(0)
        if ablate is not None and ablate[0] == 'head' and ablate[1] == li:
            yh4 = yh4.clone(); yh4[:, :, ablate[2]] = yhmeans[li][:, ablate[2]].unsqueeze(0)
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if collect_mlp is not None and li == collect_mlp[0]:
            pr = torch.einsum('btd,d->bt', mo, mdir)
            coll[('mproj',)] = pr.cpu().numpy()
            PROJ_SUM[0] += pr.sum(0)
        if ablate is not None and ablate[0] == 'mlp' and ablate[1] == li:
            pr = torch.einsum('btd,d->bt', mo, mdir)
            mo = mo - (pr - projmean.unsqueeze(0)).unsqueeze(-1) * mdir
        x = x + mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    return logits, coll

held_np = HELD.cpu().numpy()
posmat = np.tile(np.arange(SEQL), NHELD).reshape(NHELD, SEQL)
held_special = np.isin(held_np, SPECIAL)
valid_next = posmat < (SEQL - 1)

def conc(ids):
    ids = np.asarray(ids)
    if len(ids) <= 1: return 0.0
    _, c = np.unique(ids, return_counts=True); p = c/c.sum()
    return float(1 - (-(p*np.log(p)).sum())/math.log(len(ids)))

def paired(x):
    x = np.asarray(x, dtype=float); n = len(x)
    if n == 0: return (float('nan'), float('nan'), 0)
    se = x.std(ddof=1)/math.sqrt(n) if n > 1 else float('nan')
    return (float(x.mean()), float(se), n)

RESULTS = {}
for rec in CAUSAL_CANDS:
    comp = rec['comp']; A = rec['trigger_class']; Bc = rec['output_class']
    print(f"\nCAUSAL {comp}  ({rec['name']}) ...", flush=True)
    classB_mask = torch.from_numpy((VOCAB_CLASS == Bc)).to(DEV)   # (V,) bool

    # --- PASS: collect per-position means + this path's activation over held slice ---
    YH_SUM = {li: torch.zeros(SEQL, NH, HD, device=DEV) for li in range(NL)}
    PROJ_SUM = [torch.zeros(SEQL, device=DEV)]
    act = np.zeros((NHELD, SEQL), np.float32)
    if rec['kind'] == 'head':
        li, h = rec['li'], rec['h']; ch = [(li, h)]; mdir = None; cml = None
    else:
        li, kk = rec['li'], rec['k']; ch = None; mdir = mlp_dirs[li, kk].contiguous(); cml = (li,)
    for i in range(0, NHELD, BATCH):
        _, coll = forward_causal(HELD[i:i+BATCH], collect_heads=ch, collect_mlp=cml, mdir=mdir)
        b = HELD[i:i+BATCH].shape[0]
        if rec['kind'] == 'head':
            act[i:i+b] = coll[('hnorm', li, h)]
        else:
            act[i:i+b] = np.abs(coll[('mproj',)])
    YHMEAN = {l: YH_SUM[l] / NHELD for l in range(NL)}
    PROJMEAN = PROJ_SUM[0] / NHELD

    # --- select positions ---
    a = act.copy(); a[(posmat == 0) | held_special | ~valid_next] = -1e30
    flat = a.reshape(-1)
    order = np.argsort(-flat)
    top_flat = order[:KCAUSAL]                                    # path-active trigger positions
    top_s, top_p = top_flat // SEQL, top_flat % SEQL
    cur_top = held_np[top_s, top_p]
    trig_hist_held = hist_from_tokens(cur_top)
    # class-A subset among path-active positions
    is_A_top = (VOCAB_CLASS[cur_top] == A)
    # control: class-A positions where the path is INACTIVE (bottom half of activation)
    classA_all = (VOCAB_CLASS[held_np.reshape(-1)] == A) & (flat > -1e29)
    med = np.median(flat[flat > -1e29])
    ctrl_pool = np.where(classA_all & (flat < med))[0]
    n_ctrl = int(is_A_top.sum())
    rng = np.random.default_rng(0)
    ctrl_flat = rng.choice(ctrl_pool, size=min(n_ctrl, len(ctrl_pool)), replace=False) if len(ctrl_pool) else np.array([], int)
    ctrl_s, ctrl_p = ctrl_flat // SEQL, ctrl_flat % SEQL

    # --- gather clean/ablated at a set of (seq,pos), return dCE, dP_B per position ---
    def measure(seqs, poss):
        dce, dpb = [], []
        uniq = np.unique(seqs)
        for i in range(0, len(uniq), BATCH):
            sb = uniq[i:i+BATCH]; idx = HELD[sb]
            base, _ = forward_causal(idx)
            if rec['kind'] == 'head':
                abl, _ = forward_causal(idx, ablate=('head', li, h), yhmeans=YHMEAN)
            else:
                abl, _ = forward_causal(idx, ablate=('mlp', li, mdir), mdir=mdir, projmean=PROJMEAN)
            base = base.float(); abl = abl.float()
            s2l = {int(s): j for j, s in enumerate(sb)}
            pB_base = torch.softmax(base, -1)[..., classB_mask].sum(-1)   # (b,T)
            pB_abl  = torch.softmax(abl,  -1)[..., classB_mask].sum(-1)
            for s, pp in zip(seqs, poss):
                if int(s) not in s2l: continue
                j = s2l[int(s)]; tgt = int(held_np[s, pp+1])
                bl = base[j, pp]; al = abl[j, pp]
                dce.append(F.cross_entropy(al.unsqueeze(0), torch.tensor([tgt], device=DEV)).item()
                           - F.cross_entropy(bl.unsqueeze(0), torch.tensor([tgt], device=DEV)).item())
                dpb.append(float(pB_base[j, pp] - pB_abl[j, pp]))     # clean - ablated (drop>0 = suppressed)
            del base, abl, pB_base, pB_abl
        return np.array(dce), np.array(dpb)

    # overall dCE at all KCAUSAL trigger positions (standard load-bearing metric)
    dce_all, dpb_all = measure(top_s, top_p)
    # class-A trigger positions
    A_s, A_p = top_s[is_A_top], top_p[is_A_top]
    dce_A, dpb_A = measure(A_s, A_p)
    # control (inactive class-A)
    dce_c, dpb_c = (measure(ctrl_s, ctrl_p) if len(ctrl_s) else (np.array([]), np.array([])))

    dce_all_m = paired(dce_all); dpbA_m = paired(dpb_A); dpbc_m = paired(dpb_c); dpball_m = paired(dpb_all)
    dceA_m = paired(dce_A)
    # verdict on the OUTPUT side: does ablation suppress class B specifically at class-A positions?
    zA = dpbA_m[0] / dpbA_m[1] if dpbA_m[1] and not math.isnan(dpbA_m[1]) and dpbA_m[1] > 0 else 0.0
    specific = (dpbA_m[0] > 0 and zA > 2.0
                and (math.isnan(dpbc_m[0]) or dpbA_m[0] > 2*max(dpbc_m[0], 0)))
    verdict = ("GENUINE-REMAP (ablation suppresses class-B at class-A positions)" if specific
               else "PROXY-ARTIFACT (ablation does not specifically suppress class-B at class-A positions)")

    RESULTS[comp] = {
        'kind': rec['kind'], 'name': rec['name'],
        'trigger_class': A, 'output_class': Bc,
        'decoupling': rec['decoupling'], 'trigger_purity': rec['trigger_purity'],
        'output_purity': rec['output_purity'], 'score': rec['score'],
        'held_trigger_purity_cur': round(conc(cur_top), 4),
        'held_trigger_top_class': top_class(trig_hist_held)[0],
        'held_trigger_hist': {k: round(v, 3) for k, v in sorted(trig_hist_held.items(), key=lambda x: -x[1])[:6]},
        'n_classA_trigger_pos': int(is_A_top.sum()),
        # OUTPUT-side causal test: P(class B at next) drop (clean - ablated)
        'dP_classB_at_classA_mean': round(dpbA_m[0], 5), 'dP_classB_at_classA_SE': round(dpbA_m[1], 5), 'n_A': dpbA_m[2],
        'dP_classB_control_mean': (round(dpbc_m[0], 5) if not math.isnan(dpbc_m[0]) else None),
        'dP_classB_control_SE': (round(dpbc_m[1], 5) if not math.isnan(dpbc_m[1]) else None), 'n_ctrl': dpbc_m[2],
        'dP_classB_all_trigger_mean': round(dpball_m[0], 5), 'dP_classB_all_trigger_SE': round(dpball_m[1], 5),
        # load-bearing
        'dCE_all_trigger_mean': round(dce_all_m[0], 4), 'dCE_all_trigger_SE': round(dce_all_m[1], 4), 'n_all': dce_all_m[2],
        'dCE_at_classA_mean': round(dceA_m[0], 4), 'dCE_at_classA_SE': round(dceA_m[1], 4),
        'z_dP_classB_at_classA': round(zA, 2),
        'verdict': verdict,
    }
    print(json.dumps(RESULTS[comp], indent=2), flush=True)

# =====================================================================================
out = {
    'meta': {
        'model': 'bilin18', 'tokenizer': 'gpt2 (model exposes none) -- FLAGGED',
        'discovery_data': 'FineWeb FW[0:256,:128] TRAIN', 'causal_data': 'FW[448:600,:128] HELD',
        'lexical_class_library': 'single-label; reuses qk_unsup_cluster.py sets + word/quote_open/quote_close',
        'K_trigger': K, 'M_output': M_EFF, 'KCAUSAL': KCAUSAL, 'N_svd_per_block': N_SVD,
        'purity_min': PURITY_MIN, 'mag_gate_pctile25': round(gate, 4),
        'decoupling': '1 - histogram-intersection(trigger_hist, output_hist)',
        'score': 'decoupling * trigger_purity * output_purity',
        'causal_output_test': 'mean-ablate path; P(class B at next) clean-minus-ablated at path-active class-A '
                              'trigger positions vs inactive class-A control; paired SE. OUTPUT proxy is '
                              'known-unreliable (§56) so linear ranking is CANDIDATE GENERATOR only.',
        'n_clean_candidates': len(candidates),
    },
    'ranking': ranked,
    'causal': RESULTS,
}
json.dump(out, open(f'{QK}/qk_unsup_decouple.json', 'w'), indent=2)
print("\n===== CAUSAL SUMMARY =====", flush=True)
for comp, r in RESULTS.items():
    print(f"{comp:12s} {r['name']:34s} | dP(B)@A={r['dP_classB_at_classA_mean']:+.4f}"
          f"+/-{r['dP_classB_at_classA_SE']:.4f} (ctrl={r['dP_classB_control_mean']}) "
          f"dCE={r['dCE_all_trigger_mean']:+.4f}+/-{r['dCE_all_trigger_SE']:.4f} -> {r['verdict'].split(' ')[0]}", flush=True)
print("\nSaved qk_unsup_decouple.json", flush=True)
print("QK UNSUP DECOUPLE DONE", flush=True)
