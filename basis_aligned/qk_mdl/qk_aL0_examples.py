"""CONCRETE HELD-DATA EXAMPLES for the attention-layer-0 dossier (Logan request).

For each layer-0 feature, ~2 in-distribution examples from the held slice FW[448:600,:128]
showing what it ACTIVATES ON and what it PREDICTS/pushes:
  1. h.L0.3 previous-token channel: top-2 output-norm positions with attention argmax at i-1;
     snippet, top-5 direct-readout tokens of the write (first-order unembed), actual next
     token, local mean-ablation delta cross-entropy (paired vs intact).
  2. h.L0.3 capital-class push: 2 capital-due firing positions + 1 non-capital contrast;
     capital-class summed delta-logit (first-order from write, and causal from ablation).
  3. h.L0.8 distributed capital/punct support: top-2 per-position ablation-delta positions.
  4. h.L0.2 verified null: top-2 output-norm positions; tiny deltas + mean +- SE.
  5. Archetypes: head 8 {the},{a/an},{of},{and}; head 5 comma + newline units. Top-8 loading
     tokens with weights (CP recomputed verbatim from qk_stage23.py, gated against the saved
     JSON), plus 2 held snippets each where the archetype's source tokens are attended
     (top |S[i,j]| with token j in the archetype set); what the head writes into i.

Forward conventions VERBATIM from qk_hub_streampairs.py / qk_unsup_classpush.py (bilinear
NO-softmax attention, pat = s1*s2 unnormalised, per-position mean-ablation of yh4).
GPU guard strict (shared with running red-team): free < 4500 MiB -> sleep 20 retry; batch 4.
Output: qk_aL0_examples.json. DO NOT COMMIT.
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import json, sys, time, subprocess
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'

# ---------------- GPU GUARD (verbatim; strict, shared GPU) ----------------
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
print(f"bilin18 NL={NL} NH={NH} HD={HD} D={D} V={V}", flush=True)

FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
SEQL = 128
HELD = FINEWEB[448:600, :SEQL].to(DEV)
S_, T_ = HELD.shape
B0 = 4

def dec(t): return tok.decode([int(t)])
def esc(s): return s.replace('\n', '\\n')

_special = {tok.eos_token_id}
for _t in range(V):
    if '�' in tok.decode([_t]): _special.add(_t)
SPECIAL = np.array(sorted(_special))
print(f"{len(SPECIAL)} special token ids masked from selection", flush=True)

# ---------- LEXICAL CLASS LIBRARY (verbatim from qk_unsup_decouple.py / classpush) ----------
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
CMAT = torch.zeros(len(CLASS_LIST), V)
for t in range(V):
    CMAT[CIDX[VOCAB_CLASS[t]], t] = 1.0
CMAT = CMAT.to(DEV)
CAP_ROW = CIDX['capital']
print(f"classes: {CLASS_LIST}", flush=True)

W_U = m.lm_head.weight.detach().float()            # (V, D) first-order readout
Wr = m.transformer.h[0].attn.c_proj.weight.detach().float().view(D, NH, HD)

# =====================================================================================
# PART 1: ARCHETYPES -- CP recompute VERBATIM from qk_stage23.py (R=32 block), with the
# saved qk_stage23.json top-token lists as a reproduction gate.
# =====================================================================================
print("PART 1: archetype CP factors (heads 8 and 5) ...", flush=True)
blob = torch.load(f'{QK}/qk_stage1_triple.pt', map_location=DEV, weights_only=False)
QP = (torch.bincount(FINEWEB.flatten(), minlength=V).float() + 0.5).to(DEV)
QP = QP / QP.sum()

def build_core(idx, coeff, mm):
    k = idx.shape[1]
    core = torch.zeros(mm * mm * mm, device=DEV)
    w = QP[:, None] * coeff
    for i in range(k):
        for j in range(k):
            keys = (idx[:, i].long() * mm + idx[:, j].long()) * mm
            vals = w[:, i] * coeff[:, j]
            for l in range(k):
                core.scatter_add_(0, keys + idx[:, l].long(), vals * coeff[:, l])
    return core.view(mm, mm, mm)

def cp_fit(core_raw, R, seed, n_starts=8, iters=60):
    mm = core_raw.shape[0]
    gg = torch.Generator().manual_seed(seed)
    scale = core_raw.norm().clamp_min(1e-30)
    res = (core_raw / scale).clone()
    nrm2 = float((res ** 2).sum())
    Us, lams = [], []
    for r in range(R):
        M1 = res.reshape(mm, mm * mm)
        best_u, best_lam = None, -1.0
        for s in range(n_starts):
            u = torch.rand(mm, generator=gg).to(DEV)
            u = u / u.norm()
            for _ in range(iters):
                u = (M1 @ (u[:, None] * u[None, :]).reshape(-1)).clamp_min(0)
                n = float(u.norm())
                if n < 1e-20: break
                u = u / n
            lam = float(torch.einsum('abc,a,b,c->', res, u, u, u))
            if lam > best_lam:
                best_lam, best_u = lam, u
        if best_lam <= 0: break
        Us.append(best_u); lams.append(best_lam)
        res = res - best_lam * torch.einsum('a,b,c->abc', best_u, best_u, best_u)
    U = torch.stack(Us, 1)
    rel = float(res.norm()) / max(nrm2, 1e-30) ** 0.5
    return U, torch.tensor(lams, device=DEV), rel

saved23 = json.load(open(f'{QK}/qk_stage23.json'))
ARCH = {}          # (head, rank_in_top5) -> {'tokens':[(str,id,weight)x8]}
for h in (8, 5):
    key = f'h{h}_unigram_nonneg'
    idx_h = blob[f'{key}_idx'].long().to(DEV)
    coeff_h = blob[f'{key}_coeff'].to(DEV)
    mm = 512
    core = build_core(idx_h, coeff_h, mm)
    fits = []
    for seed in range(3):
        _, _, rel = cp_fit(core, 32, seed)
        fits.append(rel)
    best = int(torch.tensor(fits).argmin())
    U, lam, _ = cp_fit(core, 32, best)
    S_dense = torch.zeros(V, mm, device=DEV)
    S_dense.scatter_(1, idx_h, coeff_h)
    saved_lists = saved23[f'h{h}']['top_archetype_tokens']
    for slot, r in enumerate(lam.argsort(descending=True)[:5].tolist()):
        load = S_dense @ U[:, r]
        top = load.argsort(descending=True)[:8]
        toks = [(esc(dec(t)), int(t), round(float(load[t]), 4)) for t in top.tolist()]
        got = [t[0] for t in toks]
        want = saved_lists[slot] if slot < len(saved_lists) else None
        gate_ok = (want is not None and got == [w for w in want])
        ARCH[(h, slot)] = {'tokens': toks, 'gate_matches_saved_json': bool(gate_ok)}
        print(f"  h{h} archetype slot {slot}: {got}  gate={'PASS' if gate_ok else 'MISMATCH'}", flush=True)
    del core, S_dense, U, idx_h, coeff_h
    torch.cuda.empty_cache()
del blob
torch.cuda.empty_cache()

# name the archetype slots we need (match on the top token)
def find_slot(h, first_tok):
    for slot in range(5):
        if ARCH[(h, slot)]['tokens'][0][0] == first_tok: return slot
    raise KeyError((h, first_tok))
ARCH_SPEC = [
    ('h8_the',   8, find_slot(8, ' the')),
    ('h8_a_an',  8, find_slot(8, ' a')),
    ('h8_of',    8, find_slot(8, ' of')),
    ('h8_and',   8, find_slot(8, ' and')),
    ('h5_comma', 5, find_slot(5, ',')),
    ('h5_newline', 5, find_slot(5, '\\n')),
]
ARCH_IDS = {name: set(t[1] for t in ARCH[(h, slot)]['tokens']) for name, h, slot in ARCH_SPEC}

# =====================================================================================
# PART 2: PASS A -- intact forward over held slice; collect CE, layer-0 head write norms,
# per-position yh4 means (for mean-ablation), head-3 prev-token pattern stats, and
# archetype (i,j) attention candidates.  Forward VERBATIM conventions.
# =====================================================================================
@torch.no_grad()
def forward(idx, abl_head=None, yhmean0=None, collect0=False, want_logits=False):
    """abl_head: layer-0 head index to mean-ablate (per-position mean of yh4)."""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    out = {}
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
        if li == 0 and collect0:
            out['yh4'] = yh4.float().cpu()
            out['pat'] = pat.float().cpu()
            out['v0'] = v.float().cpu()
        if li == 0 and abl_head is not None:
            yh4 = yh4.clone()
            yh4[:, :, abl_head] = yhmean0[:, abl_head].unsqueeze(0)
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        mo = blk.mlp(F.rms_norm(x, (D,)))
        x = x + mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1),
                         reduction='none').view(B, T-1)
    out['ce'] = ce.float().cpu()
    if want_logits: out['logits'] = logits.float()
    return out

print("PASS A: intact collect ...", flush=True)
held_np = HELD.cpu().numpy()
CE_INT = np.zeros((S_, T_-1), np.float32)
HNORM = np.zeros((S_, T_, NH), np.float32)
PREV_SHARE = np.zeros((S_, T_), np.float32)      # |pat[i,i-1]| / sum_j |pat[i,j]|  (head 3)
PREV_ARGMAX = np.zeros((S_, T_), np.int64)       # argmax_j |pat[i,j]|              (head 3)
YH0_SUM = torch.zeros(T_, NH, HD, device=DEV)
arch_cand = {name: [] for name, _, _ in ARCH_SPEC}   # (score, seq, i, j, patval)

for i0 in range(0, S_, B0):
    ib = HELD[i0:i0+B0]
    out = forward(ib, collect0=True)
    b = ib.shape[0]
    CE_INT[i0:i0+b] = out['ce'].numpy()
    yh4 = out['yh4']; pat = out['pat']
    YH0_SUM += yh4.sum(0).to(DEV)
    comp = torch.einsum('bthc,ohc->btho', yh4, torch.from_numpy(Wr.cpu().numpy())).norm(dim=-1)
    HNORM[i0:i0+b] = comp.numpy()
    p3 = pat[:, 3].abs()                          # (b,T,T)
    denom = p3.sum(-1).clamp_min(1e-12)
    for bb in range(b):
        for t in range(1, T_):
            PREV_SHARE[i0+bb, t] = float(p3[bb, t, t-1] / denom[bb, t])
    PREV_ARGMAX[i0:i0+b] = p3.argmax(-1).numpy()
    # archetype candidates
    toks_b = held_np[i0:i0+b]
    for name, h, slot in ARCH_SPEC:
        ids = ARCH_IDS[name]
        colmask = np.isin(toks_b, np.array(sorted(ids)))       # (b,T) token j in set
        ph = pat[:, h].abs().numpy()                            # (b,T,T)
        for bb in range(b):
            js = np.where(colmask[bb])[0]
            if len(js) == 0: continue
            sub = ph[bb][:, js]                                 # (T, nj)
            flat = np.argsort(sub, axis=None)[::-1][:4]
            for f in flat:
                qi, jj = np.unravel_index(f, sub.shape)
                j = int(js[jj])
                if qi < j or qi < 1 or qi > T_-2: continue
                if held_np[i0+bb, qi] in SPECIAL: continue
                arch_cand[name].append((float(sub[qi, jj]), i0+bb, int(qi), j,
                                        float(pat[bb, h, qi, j])))
    del out, yh4, pat
YHMEAN0 = YH0_SUM / S_
print("PASS A done.", flush=True)

pos_valid = np.ones((S_, T_), bool)
pos_valid[:, 0] = False; pos_valid[:, T_-1] = False
pos_valid &= ~np.isin(held_np, SPECIAL)
next_special = np.zeros((S_, T_), bool)
next_special[:, :T_-1] = np.isin(held_np[:, 1:], SPECIAL)
pos_valid &= ~next_special
NEXT_CLASS = np.empty((S_, T_), object)
NEXT_CLASS[:, :T_-1] = VOCAB_CLASS[held_np[:, 1:]]
NEXT_CLASS[:, T_-1] = 'invalid'

# =====================================================================================
# PART 3: position selection
# =====================================================================================
def topk_positions(score, mask, k, distinct_seq=True):
    sc = np.where(mask, score, -np.inf).reshape(-1)
    order = np.argsort(sc)[::-1]
    picks, seqs = [], set()
    for f in order:
        if not np.isfinite(sc[f]): break
        s, t = divmod(int(f), T_)
        if distinct_seq and s in seqs: continue
        picks.append((s, t)); seqs.add(s)
        if len(picks) == k: break
    return picks

# 1) h.L0.3 prev-token: largest write norm with attention argmax at i-1
m1 = pos_valid & (PREV_ARGMAX == (np.arange(T_)[None, :] - 1))
sel_prev = topk_positions(HNORM[:, :, 3], m1, 2)

# 2) h.L0.3 capital push: firing (top-decile norm) and next token IS capital
h3n = HNORM[:, :, 3]
fire_thr = float(np.quantile(h3n[pos_valid], 0.90))
m2 = pos_valid & (NEXT_CLASS == 'capital') & (h3n >= fire_thr)
sel_cap = topk_positions(h3n, m2, 2)
m2c = pos_valid & (NEXT_CLASS == 'word') & (h3n >= fire_thr)
for (s_, t_) in sel_prev + sel_cap:   # do not reuse item-1/2 positions for the contrast
    m2c[s_, t_] = False
sel_capctr = topk_positions(h3n, m2c, 1)

# 4) h.L0.2 null: largest write norm positions
sel_h2 = topk_positions(HNORM[:, :, 2], pos_valid, 2)
# top-200 firing positions of h.L0.2 (census KCAUSAL convention)
h2flat = np.where(pos_valid, HNORM[:, :, 2], -np.inf).reshape(-1)
h2top200 = np.argsort(h2flat)[::-1][:200]

# 5) archetype pairs: top-2 per archetype, distinct sequences
sel_arch = {}
for name, h, slot in ARCH_SPEC:
    cands = sorted(arch_cand[name], key=lambda x: -x[0])
    top1_id = ARCH[(h, slot)]['tokens'][0][1]      # archetype's top-loading token id
    picks, seqs = [], set()
    # first pick: top |S| pair overall
    for sc, s, qi, j, pv in cands:
        if s in seqs: continue
        picks.append((s, qi, j, pv)); seqs.add(s); break
    # second pick: top |S| pair whose SOURCE is the archetype's top-1 token (canonical member)
    for sc, s, qi, j, pv in cands:
        if s in seqs: continue
        if int(held_np[s, j]) != top1_id: continue
        picks.append((s, qi, j, pv)); seqs.add(s); break
    if len(picks) < 2:                             # fall back to next-best overall
        for sc, s, qi, j, pv in cands:
            if s in seqs: continue
            picks.append((s, qi, j, pv)); seqs.add(s)
            if len(picks) == 2: break
    sel_arch[name] = picks

# =====================================================================================
# PART 4: ablation passes (heads 3, 8, 2) -- CE everywhere, paired vs intact
# =====================================================================================
CE_ABL = {}
for hh in (3, 8, 2):
    print(f"ablation pass h.L0.{hh} ...", flush=True)
    ce = np.zeros((S_, T_-1), np.float32)
    for i0 in range(0, S_, B0):
        out = forward(HELD[i0:i0+B0], abl_head=hh, yhmean0=YHMEAN0)
        ce[i0:i0+out['ce'].shape[0]] = out['ce'].numpy()
    CE_ABL[hh] = ce
DCE = {hh: CE_ABL[hh] - CE_INT for hh in CE_ABL}

# 3) h.L0.8 top-effect positions from its per-position ablation delta
d8 = np.zeros((S_, T_), np.float32); d8[:, :T_-1] = DCE[8]
sel_h8 = topk_positions(d8, pos_valid, 2)

# =====================================================================================
# PART 5: gather logits (intact + ablated) and layer-0 internals at selected positions
# =====================================================================================
NEED = {}          # seq -> set of positions with (head needed for ablated logits)
def need(s, t, hh):
    NEED.setdefault((s, hh), set()).add(t)
for (s, t) in sel_prev + sel_cap + sel_capctr: need(s, t, 3)
for (s, t) in sel_h8: need(s, t, 8)
for (s, t) in sel_h2: need(s, t, 2)

def next_stats(s, t, hh):
    """causal effect of ablating head hh on the ACTUAL next token at (s,t)."""
    li, la = LOG_INT[(s, t)], LOG_ABL[(s, t, hh)]
    nt = int(held_np[s, t+1])
    p_int = float(F.softmax(li, -1)[nt]); p_abl = float(F.softmax(la, -1)[nt])
    return {'next_logit_delta_intact_minus_ablated': round(float(li[nt] - la[nt]), 3),
            'p_next_intact': round(p_int, 4), 'p_next_ablated': round(p_abl, 4)}

all_pos = set()
for (s, t) in sel_prev + sel_cap + sel_capctr + sel_h8 + sel_h2: all_pos.add((s, t))
for name in sel_arch:
    for (s, qi, j, pv) in sel_arch[name]: all_pos.add((s, qi))
seqs_needed = sorted(set(s for (s, t) in all_pos))

LOG_INT = {}; L0 = {}
for s in seqs_needed:
    out = forward(HELD[s:s+1], collect0=True, want_logits=True)
    for (ss, t) in all_pos:
        if ss == s: LOG_INT[(s, t)] = out['logits'][0, t].cpu()
    L0[s] = {'yh4': out['yh4'][0], 'pat': out['pat'][0], 'v0': out['v0'][0]}
    del out; torch.cuda.empty_cache()
LOG_ABL = {}
for (s, hh), ts in NEED.items():
    out = forward(HELD[s:s+1], abl_head=hh, yhmean0=YHMEAN0, want_logits=True)
    for t in ts: LOG_ABL[(s, t, hh)] = out['logits'][0, t].cpu()
    del out; torch.cuda.empty_cache()

# =====================================================================================
# PART 6: assemble examples
# =====================================================================================
W_U_c = W_U.cpu(); Wr_c = Wr.cpu(); CMAT_c = CMAT.cpu()
SPECIAL_SET = set(int(x) for x in SPECIAL)

def snippet(s, i, marks):
    """decode tokens up to i (last <=12), marking positions: marks = {pos: (l,r)}"""
    lo = max(0, i - 11)
    parts = []
    for t in range(lo, i+1):
        w = esc(dec(held_np[s, t]))
        if t in marks: w = marks[t][0] + w + marks[t][1]
        parts.append(w)
    out = ''.join(parts)
    while len(out) > 90 and lo <= i:
        lo += 1
        parts = parts[1:]
        out = ''.join(parts)
    return out

def readout(vec, topn=5):
    """first-order unembed of a residual write; returns top tokens + class sums"""
    r = W_U_c @ vec
    order = torch.argsort(r, descending=True)
    tops = []
    for t in order.tolist():
        if t in SPECIAL_SET: continue
        tops.append([esc(dec(t)), round(float(r[t]), 3)])
        if len(tops) == topn: break
    cs = CMAT_c @ r
    csd = {CLASS_LIST[i]: round(float(cs[i]), 2) for i in range(len(CLASS_LIST))}
    csd = dict(sorted(csd.items(), key=lambda kv: -abs(kv[1]))[:5])
    return tops, csd

def head_write(s, i, h):
    return Wr_c[:, h, :] @ L0[s]['yh4'][i, h]

def causal_capsum(s, t, hh):
    dl = LOG_INT[(s, t)] - LOG_ABL[(s, t, hh)]
    return float(CMAT_c[CAP_ROW] @ dl)

def class_sums_causal(s, t, hh):
    dl = LOG_INT[(s, t)] - LOG_ABL[(s, t, hh)]
    cs = CMAT_c @ dl
    d = {CLASS_LIST[i]: round(float(cs[i]), 2) for i in range(len(CLASS_LIST))}
    return dict(sorted(d.items(), key=lambda kv: -abs(kv[1]))[:5])

RES = {'meta': {
    'model': 'bilin18', 'held_slice': 'FW[448:600,:128]',
    'conventions': 'forward verbatim from qk_hub_streampairs.py/qk_unsup_classpush.py; '
                   'mean-ablation = per-position mean of layer-0 yh4 over the 152 held seqs; '
                   'readout = first-order unembed W_U @ write (ignores final rms/tanh); '
                   'pattern is bilinear unnormalised s1*s2 (no softmax), shares use |pat|.',
    'h3_fire_threshold_top_decile_norm': round(fire_thr, 3),
}}

# ---- 1) h.L0.3 prev-token channel ----
ex1 = []
for (s, t) in sel_prev:
    w = head_write(s, t, 3)
    tops, csd = readout(w)
    ex1.append({
        'seq': s + 448, 'pos': t,
        'snippet': snippet(s, t, {t: ('>>', '<<'), t-1: ('[', ']')}),
        'attended_prev_token': esc(dec(held_np[s, t-1])),
        'prev_attention_share_abs': round(float(PREV_SHARE[s, t]), 3),
        'write_norm': round(float(HNORM[s, t, 3]), 3),
        'top5_readout': tops, 'readout_class_sums_top5': csd,
        'actual_next_token': esc(dec(held_np[s, t+1])),
        'local_ablation_dCE': round(float(DCE[3][s, t]), 4),
        **next_stats(s, t, 3),
    })
RES['1_hL0_3_prev_token'] = ex1

# ---- 2) h.L0.3 capital push ----
def cap_item(s, t):
    w = head_write(s, t, 3)
    r = W_U_c @ w
    tops, csd = readout(w)
    return {
        'seq': s + 448, 'pos': t,
        'snippet': snippet(s, t, {t: ('>>', '<<')}),
        'write_norm': round(float(HNORM[s, t, 3]), 3),
        'capital_class_sum_firstorder_from_write': round(float(CMAT_c[CAP_ROW] @ r), 2),
        'capital_class_sum_causal_ablation_deltalogit': round(causal_capsum(s, t, 3), 2),
        'top5_readout': tops, 'readout_class_sums_top5': csd,
        'actual_next_token': esc(dec(held_np[s, t+1])),
        'next_class': str(NEXT_CLASS[s, t]),
        'local_ablation_dCE': round(float(DCE[3][s, t]), 4),
        **next_stats(s, t, 3),
    }
RES['2_hL0_3_capital_push'] = {
    'capital_due': [cap_item(s, t) for (s, t) in sel_cap],
    'contrast_not_due': [cap_item(s, t) for (s, t) in sel_capctr],
}

# ---- 3) h.L0.8 distributed support ----
ex3 = []
for (s, t) in sel_h8:
    w = head_write(s, t, 8)
    tops, csd = readout(w)
    ex3.append({
        'seq': s + 448, 'pos': t,
        'snippet': snippet(s, t, {t: ('>>', '<<')}),
        'write_norm': round(float(HNORM[s, t, 8]), 3),
        'top5_readout': tops, 'readout_class_sums_top5': csd,
        'causal_class_sums_top5': class_sums_causal(s, t, 8),
        'actual_next_token': esc(dec(held_np[s, t+1])),
        'local_ablation_dCE': round(float(DCE[8][s, t]), 4),
        **next_stats(s, t, 8),
    })
RES['3_hL0_8_distributed_support'] = ex3

# ---- 4) h.L0.2 verified null ----
d2 = np.zeros((S_, T_), np.float32); d2[:, :T_-1] = DCE[2]
d2top = d2.reshape(-1)[h2top200]
d2all = DCE[2].reshape(-1)
ex4 = []
for (s, t) in sel_h2:
    ex4.append({
        'seq': s + 448, 'pos': t,
        'snippet': snippet(s, t, {t: ('>>', '<<')}),
        'write_norm': round(float(HNORM[s, t, 2]), 3),
        'actual_next_token': esc(dec(held_np[s, t+1])),
        'local_ablation_dCE': round(float(DCE[2][s, t]), 4),
        **next_stats(s, t, 2),
    })
RES['4_hL0_2_null'] = {
    'examples': ex4,
    'mean_dCE_top200_firing': [round(float(d2top.mean()), 5),
                               round(float(d2top.std(ddof=1)/np.sqrt(len(d2top))), 5)],
    'mean_dCE_global': [round(float(d2all.mean()), 6),
                        round(float(d2all.std(ddof=1)/np.sqrt(len(d2all))), 6)],
}

# ---- 5) archetypes ----
ex5 = {}
for name, h, slot in ARCH_SPEC:
    entry = {'head': h, 'cp_slot': slot,
             'gate_matches_saved_json': ARCH[(h, slot)]['gate_matches_saved_json'],
             'top8_tokens_weights': [[t[0], t[2]] for t in ARCH[(h, slot)]['tokens']],
             'examples': []}
    for (s, qi, j, pv) in sel_arch[name]:
        w_full = head_write(s, qi, h)
        w_j = float(L0[s]['pat'][h, qi, j]) * (Wr_c[:, h, :] @ L0[s]['v0'][j, h])
        topsF, csdF = readout(w_full)
        topsJ, csdJ = readout(w_j)
        marks = {qi: ('>>', '<<')} if qi == j else {qi: ('>>', '<<'), j: ('[', ']')}
        entry['examples'].append({
            'seq': s + 448, 'dest_pos_i': qi, 'source_pos_j': j,
            'source_token': esc(dec(held_np[s, j])),
            'S_ij_bilinear': round(pv, 4),
            'snippet': snippet(s, qi, marks),
            'full_head_write_top5_readout': topsF,
            'full_head_write_class_sums_top5': csdF,
            'source_contribution_top5_readout': topsJ,
            'source_contribution_class_sums_top5': csdJ,
            'actual_next_token': esc(dec(held_np[s, qi+1])),
            'caveat': 'direct-unembed effect of layer-0 value writes is ~0 (5e); these '
                      'deposit class-occurrence signals consumed by later layers -- '
                      'class sums are the meaningful readout',
        })
    ex5[name] = entry
RES['5_archetypes'] = ex5

json.dump(RES, open(f'{QK}/qk_aL0_examples.json', 'w'), indent=1)
print("WROTE qk_aL0_examples.json", flush=True)
print(json.dumps(RES, indent=1)[:4000], flush=True)
print("QK AL0 EXAMPLES DONE", flush=True)
