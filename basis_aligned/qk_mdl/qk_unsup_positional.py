#!/usr/bin/env python
"""
qk_unsup_positional.py -- POSITION-vs-CONTENT decomposition of attention heads (bilin18).

NEW TOOL for a circuit TYPE the content-class pipeline (qk_unsup_discover/cluster)
CANNOT express: POSITIONAL / STRUCTURAL heads that route by RELATIVE POSITION or by
LINE / SEGMENT structure rather than by a token-content class.  §58's auto-cluster
flagged causally-real heads (h.L1.2 bullet/newline, h.L2.1 table-delimiter, and the
diffuse attention layers 4/9/17) as "no clean content output" -- because they route by
POSITION, which the token-content-class featurization cannot express.

TECHNIQUE (documented in the trailing block at bottom of this file):
  For every head (18L x 9H) decompose its raw bilinear attention pattern pat[b,q,k]
  (the UNNORMALISED s1*s2 that flows to v) into
    (A) a POSITIONAL component -- variance explained by query->key OFFSET (o = q-k),
        the head's mean pattern-by-offset (the "offset template"), plus argmax-based
        structural targets: fixed previous-token (offset 1), fixed-back-k (modal
        offset), absolute-position-0 sink, and attend-to-last-newline (segment start);
    (B) a CONTENT component -- variance explained by the KEY-token CLASS, and crucially
        the CONTENT RESIDUAL: how much class-variance remains AFTER the offset template
        is removed (R2_content_resid).  Symmetric R2_offset_resid = offset-variance
        beyond the content template.
  HONESTY (§56 / greater-of-two): a positional template fits the mean pattern of ALL
  heads (RoPE + causal mask give every head a positional envelope).  The decisive
  metric is not "has an offset template" but CONTENT-RESIDUAL ~= 0 (attention is
  ~entirely positional) AND causal load at STRUCTURAL positions.  A clean
  "most heads are content-driven; N are genuinely positional" is the expected result.

Discovery slice: TRAIN FW[0:256,:128].  Causal verification: HELD FW[448:600,:128].

Causal move (top positional heads, HELD, mean-ablation, per-position PAIRED dCE):
  mean-ablate the head (in-distribution per-position mean) and test whether the damage
  concentrates at STRUCTURAL positions -- bucket per-position dCE by DISTANCE-SINCE-
  NEWLINE (structural coordinate) and by current-token content CLASS.  A genuine
  distance-to-newline head's damage SCALES with distance-since-newline; a fixed-offset
  head's damage is spread across structure (not one content class).  Paired SEs.

FORWARD copied VERBATIM from qk_bracket_patch.py (tier2_model.reference_forward):
bilin18 two-branch pattern (q1k1/HD)(q2k2/HD) masked UNNORMALISED, per-head QK rms_norm
THEN RoPE, v-lerp via a.lamb (block-0 v cache), 30*tanh logits.  The ONLY addition is a
read-out hook `on_pat(li, pat, idx)` called right after `pat` is formed -- it changes no
computation line (same pattern verify.py used to add its collect hooks).
"""
import json, sys, math, os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
tok = AutoTokenizer.from_pretrained('gpt2')

FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
SEQL = 128
_SM = int(os.environ.get('SMOKE', '0'))      # >0 = tiny smoke test
TRAIN = FINEWEB[0:(8 if _SM else 256), :SEQL].to(DEV)        # discovery slice
HELD  = FINEWEB[448:(456 if _SM else 600), :SEQL].to(DEV)    # held-back causal slice
NTRAIN, NHELD = TRAIN.shape[0], HELD.shape[0]
BATCH = 4
print(f"bilin18 NL={NL} NH={NH} HD={HD} D={D} V={V}; train={tuple(TRAIN.shape)} held={tuple(HELD.shape)}", flush=True)

# ------------------------------------------------------------------ token classes
# special / degenerate tokens (document-sep + UTF-8 replacement) -> class 'special'
_special = {tok.eos_token_id}
for _t in range(V):
    try:
        if '�' in tok.decode([_t]): _special.add(_t)
    except Exception:
        pass
SPECIAL_SET = set(_special)
NEWLINE_SET = set(t for t in range(V) if '\n' in (tok.decode([t]) if True else ''))
# (rebuild newline robustly)
NEWLINE_SET = set()
for _t in range(V):
    try:
        if '\n' in tok.decode([_t]): NEWLINE_SET.add(_t)
    except Exception:
        pass

CLS_NAMES = ["newline", "delim", "quote", "punct", "digit",
             "cap_sp", "cap", "word_sp", "subword", "other", "special"]
NCLS = len(CLS_NAMES)
_BR = set("([{<)]}>"); _QU = set("\"'`“”‘’")

def token_class(tid_):
    if tid_ in SPECIAL_SET: return 10
    s = tok.decode([tid_])
    if s == "": return 9
    if '\n' in s: return 0
    body = s.strip(); low = body.lower(); lead = s.startswith(' ')
    if body == "": return 9
    if all(c in _BR for c in body): return 1
    if all(c in _QU for c in body): return 2
    if all((not c.isalnum()) and (not c.isspace()) for c in body): return 3
    if any(c.isdigit() for c in body): return 4
    if body[:1].isupper(): return 5 if lead else 6
    if body[:1].isalpha(): return 7 if lead else 8
    return 9

# build vocab lookup tables (class + special + newline) once
present = torch.unique(torch.cat([TRAIN.reshape(-1), HELD.reshape(-1)])).cpu().numpy()
CLS_LOOKUP = torch.full((V,), 9, dtype=torch.long, device=DEV)
for t in present.tolist():
    CLS_LOOKUP[t] = token_class(int(t))
SPECIAL_BOOL = torch.zeros(V, dtype=torch.bool, device=DEV)
for t in SPECIAL_SET: SPECIAL_BOOL[t] = True
NL_BOOL = torch.zeros(V, dtype=torch.bool, device=DEV)
for t in NEWLINE_SET: NL_BOOL[t] = True
print(f"n_special={len(SPECIAL_SET)} n_newline_tok={len(NEWLINE_SET)} n_present={len(present)}", flush=True)

QPOS = torch.arange(SEQL, device=DEV)
OFFSET = (QPOS[:, None] - QPOS[None, :]).clamp(min=0)      # (T,T) q-k, 0 where k>q (masked anyway)
TRIL = torch.tril(torch.ones(SEQL, SEQL, dtype=torch.bool, device=DEV))

def last_newline_matrix(tokens):
    """per (seq,pos) -> position of most recent newline STRICTLY BEFORE pos, else -1;
       and distance-since-newline (pos - lastnl, or pos if none)."""
    nl = NL_BOOL[tokens]                                   # (N,T) bool
    N, T = tokens.shape
    lastnl = torch.full((N, T), -1, dtype=torch.long, device=DEV)
    cur = torch.full((N,), -1, dtype=torch.long, device=DEV)
    for t in range(T):
        lastnl[:, t] = cur                                 # newline strictly before t
        cur = torch.where(nl[:, t], torch.full_like(cur, t), cur)
    seg = torch.where(lastnl >= 0, lastnl, torch.zeros_like(lastnl))
    dsn = QPOS[None, :] - seg                              # distance since last boundary (>=0)
    return lastnl, dsn

TR_LASTNL, TR_DSN = last_newline_matrix(TRAIN)
HE_LASTNL, HE_DSN = last_newline_matrix(HELD)

# =====================================================================================
# FORWARD -- copied VERBATIM from qk_bracket_patch.py; only added: on_pat read-out hook.
# =====================================================================================
@torch.no_grad()
def forward(idx, ablate=None, means=None, no_v1=False, v1_patch=None, want_yh=False, on_pat=None):
    """ablate: None | ('head',li,h) | ('attn',li) | ('mlp',li) | ('allattn',)
    means: dict with per-layer 'yh'(T,NH,HD) 'attn'(T,D) 'mlp'(T,D) per-position means.
    no_v1: force lamb->0 (no layer-0 value routing).
    v1_patch: (pos, donor_v1) donor_v1 shape (NH,HD) replaces v1[:,pos].
    want_yh: also accumulate per-layer yh/attn/mlp sums (for mean collection).
    on_pat: optional callback(li, pat, idx) read-out hook (added; no compute change)."""
    B, Tt = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(Tt, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(Tt, Tt, device=DEV, dtype=torch.bool))
    acc = {'yh': [], 'attn': [], 'mlp': []} if want_yh else None
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn; hc = F.rms_norm(x, (D,))
        def qk(l): z = F.rms_norm(l(hc).view(B, Tt, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hc).view(B, Tt, NH, HD)
        if v1 is None:
            v1 = v
            if v1_patch is not None:
                pos, donor = v1_patch; v1 = v1.clone(); v1[:, pos] = donor
        lamb = 0.0 if no_v1 else a.lamb
        v = (1 - lamb) * v + lamb * v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        if on_pat is not None: on_pat(li, pat, idx)          # <-- read-out hook (added)
        yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)          # (B,Tt,NH,HD)
        if ablate is not None and ablate[0] == 'head' and ablate[1] == li:
            yh = yh.clone(); yh[:, :, ablate[2]] = means['yh'][li][:, ablate[2]].unsqueeze(0)
        if want_yh: acc['yh'].append(yh.sum(0))
        attn = a.c_proj(yh.reshape(B, Tt, -1))
        if ablate is not None and ablate[0] == 'attn' and ablate[1] == li:
            attn = means['attn'][li].unsqueeze(0).expand(B, -1, -1)
        if ablate is not None and ablate[0] == 'allattn':
            attn = means['attn'][li].unsqueeze(0).expand(B, -1, -1)
        if want_yh: acc['attn'].append(attn.sum(0))
        x = x + attn
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if ablate is not None and ablate[0] == 'mlp' and ablate[1] == li:
            mo = means['mlp'][li].unsqueeze(0).expand(B, -1, -1)
        if want_yh: acc['mlp'].append(mo.sum(0))
        x = x + mo
    logits = 30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)
    return (logits, acc) if want_yh else logits

# =====================================================================================
# ACCUMULATORS for the position-vs-content decomposition (over TRAIN)
# =====================================================================================
z3o = lambda: torch.zeros(NL, NH, SEQL, device=DEV)
z2  = lambda: torch.zeros(NL, NH, device=DEV)
z3c = lambda: torch.zeros(NL, NH, NCLS, device=DEV)
# pass1: offset template, content-class template, totals, argmax structural
OFF_S, OFF_N = z3o(), z3o()
CLS_S, CLS_N = z3c(), z3c()
SUMY, SUMY2, NENT = z2(), z2(), z2()
OFFARG_HIST = z3o(); KEYCLS_ARG = z3c()
NQ, NQ_NL, BOS_CNT, LASTNL_CNT, SPEC_ARG = z2(), z2(), z2(), z2(), z2()
# pass2: residual-after-offset grouped by class; residual-after-class grouped by offset
CLS_RES_S = z3c(); RESoff_SUM, RESoff_SUM2 = z2(), z2()
OFF_RES_S = z3o(); REScls_SUM, REScls_SUM2 = z2(), z2()

HIDX = torch.arange(NH, device=DEV)

def _batch_masks(idx):
    """valid-entry (B,T,T), valid-query (B,T), key-class (B,T), specials, lastnl."""
    ksp = SPECIAL_BOOL[idx]                                 # (B,T)
    keycls = CLS_LOOKUP[idx]                                # (B,T)
    vq = (~ksp) & (QPOS[None, :] >= 1)                      # (B,T) valid query
    ve = TRIL[None] & vq[:, :, None] & (~ksp)[:, None, :]   # (B,T,T)
    return ve, vq, keycls, ksp

def accum_pass1(li, pat, idx):
    B = idx.shape[0]
    ve, vq, keycls, ksp = _batch_masks(idx)
    lastnl = _cur_lastnl                                     # (B,T) set by driver loop
    veh = ve[:, None].expand(B, NH, SEQL, SEQL)
    hgrid = HIDX[None, :, None, None].expand(B, NH, SEQL, SEQL)
    val = pat[veh]
    # offset template
    offb = OFFSET[None, None].expand(B, NH, SEQL, SEQL)[veh]
    OFF_S[li].view(-1).index_add_(0, hgrid[veh] * SEQL + offb, val)
    OFF_N[li].view(-1).index_add_(0, hgrid[veh] * SEQL + offb, torch.ones_like(val))
    # content-class template (class of the KEY token)
    clsb = keycls[:, None, None, :].expand(B, NH, SEQL, SEQL)[veh]
    CLS_S[li].view(-1).index_add_(0, hgrid[veh] * NCLS + clsb, val)
    CLS_N[li].view(-1).index_add_(0, hgrid[veh] * NCLS + clsb, torch.ones_like(val))
    # totals
    hh = hgrid[veh]
    SUMY[li].index_add_(0, hh, val); SUMY2[li].index_add_(0, hh, val * val)
    NENT[li].index_add_(0, hh, torch.ones_like(val))
    # argmax structural (where the head looks): argmax|pat|
    am = pat.abs().argmax(-1)                               # (B,NH,T)
    vqh = vq[:, None, :].expand(B, NH, SEQL)
    hq = HIDX[None, :, None].expand(B, NH, SEQL)
    offarg = (QPOS[None, None, :] - am)
    sel = vqh
    onesq = torch.ones(int(sel.sum()), device=DEV)
    OFFARG_HIST[li].view(-1).index_add_(0, (hq * SEQL + offarg)[sel], onesq)
    NQ[li].index_add_(0, hq[sel], onesq)
    bos = (am == 0) & sel
    BOS_CNT[li].index_add_(0, hq[bos], torch.ones(int(bos.sum()), device=DEV))
    lnl = lastnl[:, None, :].expand(B, NH, SEQL)
    has_nl = (lnl >= 0) & sel
    NQ_NL[li].index_add_(0, hq[has_nl], torch.ones(int(has_nl.sum()), device=DEV))
    amatch = (am == lnl) & has_nl
    LASTNL_CNT[li].index_add_(0, hq[amatch], torch.ones(int(amatch.sum()), device=DEV))
    kca = torch.gather(keycls[:, None, :].expand(B, NH, SEQL), 2, am)
    KEYCLS_ARG[li].view(-1).index_add_(0, (hq * NCLS + kca)[sel], onesq)
    kspa = torch.gather(ksp[:, None, :].expand(B, NH, SEQL), 2, am) & sel
    SPEC_ARG[li].index_add_(0, hq[kspa], torch.ones(int(kspa.sum()), device=DEV))

def accum_pass2(li, pat, idx):
    B = idx.shape[0]
    ve, vq, keycls, ksp = _batch_masks(idx)
    veh = ve[:, None].expand(B, NH, SEQL, SEQL)
    hgrid = HIDX[None, :, None, None].expand(B, NH, SEQL, SEQL)
    val = pat[veh]
    hh = hgrid[veh]
    offb = OFFSET[None, None].expand(B, NH, SEQL, SEQL)[veh]
    clsb = keycls[:, None, None, :].expand(B, NH, SEQL, SEQL)[veh]
    # residual after OFFSET template, grouped by CONTENT class
    off_mean = OFFMEAN[li].view(-1)[hh * SEQL + offb]
    r_off = val - off_mean
    CLS_RES_S[li].view(-1).index_add_(0, hh * NCLS + clsb, r_off)
    RESoff_SUM[li].index_add_(0, hh, r_off); RESoff_SUM2[li].index_add_(0, hh, r_off * r_off)
    # residual after CONTENT template, grouped by OFFSET
    cls_mean = CLSMEAN[li].view(-1)[hh * NCLS + clsb]
    r_cls = val - cls_mean
    OFF_RES_S[li].view(-1).index_add_(0, hh * SEQL + offb, r_cls)
    REScls_SUM[li].index_add_(0, hh, r_cls); REScls_SUM2[li].index_add_(0, hh, r_cls * r_cls)

# ---- run pass 1 over TRAIN ----
print("PASS 1: offset + content templates + argmax structural (TRAIN) ...", flush=True)
for i in range(0, NTRAIN, BATCH):
    _cur_lastnl = TR_LASTNL[i:i + BATCH]
    forward(TRAIN[i:i + BATCH], on_pat=accum_pass1)

OFFMEAN = OFF_S / OFF_N.clamp(min=1)
CLSMEAN = CLS_S / CLS_N.clamp(min=1)

print("PASS 2: residual R2 (content-beyond-offset, offset-beyond-content) ...", flush=True)
for i in range(0, NTRAIN, BATCH):
    forward(TRAIN[i:i + BATCH], on_pat=accum_pass2)

# =====================================================================================
# per-head metrics
# =====================================================================================
def between_ss(S, N, grand_sum, Ntot):
    """categorical between-group SS = sum_g S_g^2/N_g - grand_sum^2/Ntot."""
    g = (S ** 2 / N.clamp(min=1)).sum(-1)
    return g - grand_sum ** 2 / Ntot.clamp(min=1)

SS_tot = SUMY2 - SUMY ** 2 / NENT.clamp(min=1)               # (NL,NH)
SSb_off = between_ss(OFF_S, OFF_N, SUMY, NENT)
SSb_cls = between_ss(CLS_S, CLS_N, SUMY, NENT)
R2_off = (SSb_off / SS_tot.clamp(min=1e-12)).clamp(0, 1)
R2_cls = (SSb_cls / SS_tot.clamp(min=1e-12)).clamp(0, 1)
# content-residual (beyond offset)
SStot_roff = RESoff_SUM2 - RESoff_SUM ** 2 / NENT.clamp(min=1)
SSb_cls_res = between_ss(CLS_RES_S, CLS_N, RESoff_SUM, NENT)
R2_cls_resid = (SSb_cls_res / SStot_roff.clamp(min=1e-12)).clamp(0, 1)
# offset-residual (beyond content)
SStot_rcls = REScls_SUM2 - REScls_SUM ** 2 / NENT.clamp(min=1)
SSb_off_res = between_ss(OFF_RES_S, OFF_N, REScls_SUM, NENT)
R2_off_resid = (SSb_off_res / SStot_rcls.clamp(min=1e-12)).clamp(0, 1)

def conc_from_hist(h):
    """concentration = 1 - normalized entropy of a count histogram (last dim)."""
    p = h / h.sum(-1, keepdim=True).clamp(min=1)
    ent = -(p * (p.clamp(min=1e-12)).log()).sum(-1)
    k = (h > 0).sum(-1).clamp(min=2).float()
    return (1 - ent / k.log()).clamp(0, 1)

offarg_purity = conc_from_hist(OFFARG_HIST)                  # (NL,NH) fixed-offset concentration
keyclass_arg_purity = conc_from_hist(KEYCLS_ARG)            # content concentration of argmax key
modal_off = OFFARG_HIST.argmax(-1)                          # (NL,NH)
modal_off_frac = OFFARG_HIST.max(-1).values / NQ.clamp(min=1)
prev1_frac = OFFARG_HIST[:, :, 1] / NQ.clamp(min=1)
bos_frac = BOS_CNT / NQ.clamp(min=1)
lastnl_frac = LASTNL_CNT / NQ_NL.clamp(min=1)              # among queries WITH a prior newline
spec_frac = SPEC_ARG / NQ.clamp(min=1)

R2_off = R2_off.cpu().numpy(); R2_cls = R2_cls.cpu().numpy()
R2_cls_resid = R2_cls_resid.cpu().numpy(); R2_off_resid = R2_off_resid.cpu().numpy()
offarg_purity = offarg_purity.cpu().numpy(); keyclass_arg_purity = keyclass_arg_purity.cpu().numpy()
modal_off = modal_off.cpu().numpy(); modal_off_frac = modal_off_frac.cpu().numpy()
prev1_frac = prev1_frac.cpu().numpy(); bos_frac = bos_frac.cpu().numpy()
lastnl_frac = lastnl_frac.cpu().numpy(); spec_frac = spec_frac.cpu().numpy()

def classify(r):
    if r['lastnl_frac'] >= 0.35 and r['R2_cls_resid'] < 0.20:
        return "LINE-STRUCTURE (attend-last-newline / segment-start)"
    if r['bos_frac'] >= 0.60 and r['R2_cls_resid'] < 0.12:
        return "ABS-POS-0 SINK"
    if r['offarg_purity'] >= 0.55 and r['R2_cls_resid'] < 0.12:
        return f"FIXED-OFFSET (back-{r['modal_off']})"
    if r['R2_off'] >= 0.30 and r['R2_cls_resid'] < 0.10:
        return "POSITIONAL (offset-envelope)"
    if r['R2_cls_resid'] >= 0.20 or r['keyclass_arg_purity'] >= 0.55:
        return "CONTENT"
    return "MIXED / DIFFUSE"

heads = []
for li in range(NL):
    for h in range(NH):
        struct_score = float(max(prev1_frac[li, h], lastnl_frac[li, h], bos_frac[li, h],
                                 modal_off_frac[li, h]))
        r = dict(comp=f"h.L{li}.{h}", li=li, h=h,
                 R2_off=round(float(R2_off[li, h]), 3),
                 R2_cls=round(float(R2_cls[li, h]), 3),
                 R2_cls_resid=round(float(R2_cls_resid[li, h]), 3),
                 R2_off_resid=round(float(R2_off_resid[li, h]), 3),
                 offarg_purity=round(float(offarg_purity[li, h]), 3),
                 keyclass_arg_purity=round(float(keyclass_arg_purity[li, h]), 3),
                 modal_off=int(modal_off[li, h]),
                 modal_off_frac=round(float(modal_off_frac[li, h]), 3),
                 prev1_frac=round(float(prev1_frac[li, h]), 3),
                 bos_frac=round(float(bos_frac[li, h]), 3),
                 lastnl_frac=round(float(lastnl_frac[li, h]), 3),
                 spec_frac=round(float(spec_frac[li, h]), 3),
                 struct_score=round(struct_score, 3))
        r['label'] = classify(r)
        # positional dominance: strong structural argmax AND negligible content residual
        r['positional_purity'] = round(struct_score * (1.0 - min(1.0, r['R2_cls_resid'] / 0.20)), 3)
        heads.append(r)

heads.sort(key=lambda z: -z['positional_purity'])
for rank, r in enumerate(heads):
    r['rank'] = rank

# label buckets
def is_positional(r):
    return r['label'].startswith(("LINE-STRUCTURE", "ABS-POS-0", "FIXED-OFFSET", "POSITIONAL"))
positional = [r for r in heads if is_positional(r)]
content = [r for r in heads if r['label'] == "CONTENT"]
mixed = [r for r in heads if r['label'].startswith("MIXED")]

# =====================================================================================
# CAUSAL VERIFICATION of top positional heads on HELD (mean-ablation, paired dCE)
# =====================================================================================
print("collecting HELD per-position yh means ...", flush=True)
YH_SUM = [torch.zeros(SEQL, NH, HD, device=DEV) for _ in range(NL)]
ASUM = [torch.zeros(SEQL, D, device=DEV) for _ in range(NL)]
PSUM = [torch.zeros(SEQL, D, device=DEV) for _ in range(NL)]
for i in range(0, NHELD, BATCH):
    _, acc = forward(HELD[i:i + BATCH], want_yh=True)
    for li in range(NL):
        YH_SUM[li] += acc['yh'][li]; ASUM[li] += acc['attn'][li]; PSUM[li] += acc['mlp'][li]
YHMEAN = {'yh': [t / NHELD for t in YH_SUM], 'attn': [t / NHELD for t in ASUM],
          'mlp': [t / NHELD for t in PSUM]}

held_np = HELD.cpu().numpy()
he_dsn = HE_DSN.cpu().numpy()
he_cls = CLS_LOOKUP[HELD].cpu().numpy()
he_spec = SPECIAL_BOOL[HELD].cpu().numpy()
DSN_BINS = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 127)]

@torch.no_grad()
def causal_head(li, h):
    """mean-ablate h.Lli.h on HELD; per-position paired dCE; bucket by DSN & content class."""
    dce_all, dsn_all, cls_all = [], [], []
    for i in range(0, NHELD, BATCH):
        idx = HELD[i:i + BATCH]
        base = forward(idx).float()
        abl = forward(idx, ablate=('head', li, h), means=YHMEAN).float()
        B = idx.shape[0]
        for b in range(B):
            for p in range(SEQL - 1):
                if he_spec[i + b, p] or he_spec[i + b, p + 1] or p == 0:
                    continue
                tgt = int(held_np[i + b, p + 1])
                bce = F.cross_entropy(base[b, p:p + 1], torch.tensor([tgt], device=DEV)).item()
                ace = F.cross_entropy(abl[b, p:p + 1], torch.tensor([tgt], device=DEV)).item()
                dce_all.append(ace - bce)
                dsn_all.append(int(he_dsn[i + b, p]))
                cls_all.append(int(he_cls[i + b, p]))
    dce = np.array(dce_all); dsn = np.array(dsn_all); cls = np.array(cls_all)
    n = len(dce)
    def stat(mask):
        d = dce[mask]; k = len(d)
        if k == 0: return {'n': 0, 'dCE': None, 'SE': None}
        return {'n': int(k), 'dCE': round(float(d.mean()), 4),
                'SE': round(float(d.std(ddof=1) / math.sqrt(k)) if k > 1 else 0.0, 4)}
    by_dsn = {f"{lo}-{hi}": stat((dsn >= lo) & (dsn <= hi)) for lo, hi in DSN_BINS}
    by_cls = {}
    for c in range(NCLS):
        s = stat(cls == c)
        if s['n'] >= 15: by_cls[CLS_NAMES[c]] = s
    # correlation of per-position dCE with distance-since-newline
    if dce.std() > 1e-9 and dsn.std() > 1e-9:
        corr = float(np.corrcoef(dce, dsn)[0, 1])
    else:
        corr = 0.0
    # structural concentration: variance of bin-means across DSN vs across content class
    dsn_means = [v['dCE'] for v in by_dsn.values() if v['dCE'] is not None]
    cls_means = [v['dCE'] for v in by_cls.values() if v['dCE'] is not None]
    return {'overall': stat(np.ones(n, bool)),
            'dCE_by_distance_since_newline': by_dsn,
            'dCE_by_current_class': by_cls,
            'corr_dCE_vs_distance_since_newline': round(corr, 3),
            'spread_across_distance_bins': round(float(np.std(dsn_means)), 4) if dsn_means else None,
            'spread_across_content_classes': round(float(np.std(cls_means)), 4) if cls_means else None}

# select heads to causally test: top positional by positional_purity, UNION the §58-flagged
# plus every LINE-STRUCTURE head (the segment/newline routers the task targets).
FLAGGED = ["h.L1.2", "h.L2.1"] + [r['comp'] for r in heads
                                  if r['label'].startswith("LINE-STRUCTURE")]
sel_comps, seen = [], set()
for r in positional:
    if len(sel_comps) >= 6: break
    sel_comps.append((r['li'], r['h'])); seen.add(r['comp'])
for fc in FLAGGED:
    lookup = {x['comp']: x for x in heads}
    if fc in lookup and fc not in seen:
        r = lookup[fc]; sel_comps.append((r['li'], r['h'])); seen.add(fc)

print(f"CAUSAL: testing {len(sel_comps)} heads on HELD ...", flush=True)
causal = {}
lookup = {x['comp']: x for x in heads}
for (li, h) in sel_comps:
    comp = f"h.L{li}.{h}"
    print(f"  causal {comp} ({lookup[comp]['label']}) ...", flush=True)
    res = causal_head(li, h)
    res['label'] = lookup[comp]['label']
    res['metrics'] = {k: lookup[comp][k] for k in
                      ('R2_off', 'R2_cls', 'R2_cls_resid', 'offarg_purity', 'prev1_frac',
                       'bos_frac', 'lastnl_frac', 'modal_off', 'struct_score')}
    causal[comp] = res

# =====================================================================================
# SAVE + REPORT
# =====================================================================================
label_counts = {}
for r in heads:
    key = r['label'].split(" (")[0]
    label_counts[key] = label_counts.get(key, 0) + 1

out = {
    'meta': {
        'model': 'bilin18', 'n_heads': len(heads), 'NL': NL, 'NH': NH,
        'train_slice': 'FW[0:256,:128]', 'held_slice': 'FW[448:600,:128]',
        'technique': 'position-vs-content decomposition of bilinear attention patterns',
        'classes': CLS_NAMES,
        'metric_defs': {
            'R2_off': 'variance of raw pat explained by query-key OFFSET template',
            'R2_cls': 'variance of raw pat explained by KEY-token-class template',
            'R2_cls_resid': 'DECISIVE: content-class variance remaining AFTER offset removed',
            'R2_off_resid': 'offset variance remaining after content removed (symmetric)',
            'offarg_purity': 'concentration of argmax|pat| OFFSET distribution (fixed-offset)',
            'keyclass_arg_purity': 'concentration of argmax|pat| KEY-CLASS (content)',
            'lastnl_frac': 'frac of queries (w/ prior newline) whose argmax==last-newline pos',
            'bos_frac': 'frac of queries whose argmax==position 0 (abs-pos sink)',
            'prev1_frac': 'frac of queries whose argmax==previous token (offset 1)',
            'positional_purity': 'struct_score * (1 - min(1, R2_cls_resid/0.20))',
        },
        'honesty_note': 'A positional offset template fits the mean pattern of EVERY head '
                        '(RoPE + causal mask). The decisive signal is R2_cls_resid ~= 0 '
                        '(content adds nothing beyond position) together with causal load '
                        'at structural positions. Expected result: most heads content-driven, '
                        'a few genuinely positional.',
    },
    'label_counts': label_counts,
    'ranking_all_heads': heads,
    'positional_heads': positional,
    'content_heads_top': sorted(content, key=lambda z: -z['R2_cls_resid'])[:15],
    'causal_verification': causal,
}
def _np(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)
json.dump(out, open(f"{QK}/qk_unsup_positional.json", 'w'), indent=1, default=_np)
print(f"[saved] {QK}/qk_unsup_positional.json", flush=True)

print("\n==== LABEL COUNTS (of %d heads) ====" % len(heads))
for k, v in sorted(label_counts.items(), key=lambda z: -z[1]):
    print(f"  {k:<42} {v}")
print("\n==== TOP POSITIONAL / STRUCTURAL HEADS (by positional_purity) ====")
print(f"{'comp':<9}{'label':<48}{'pp':>6}{'R2off':>7}{'R2crsd':>8}{'offpur':>8}{'prev1':>7}{'bos':>6}{'lastnl':>7}{'modoff':>7}")
for r in positional[:20]:
    print(f"{r['comp']:<9}{r['label']:<48}{r['positional_purity']:>6}{r['R2_off']:>7}"
          f"{r['R2_cls_resid']:>8}{r['offarg_purity']:>8}{r['prev1_frac']:>7}{r['bos_frac']:>6}"
          f"{r['lastnl_frac']:>7}{r['modal_off']:>7}")
print("\n==== CAUSAL (HELD, mean-ablation, paired dCE) ====")
for comp, c in causal.items():
    o = c['overall']
    print(f"\n{comp}  [{c['label']}]  overall dCE={o['dCE']}±{o['SE']} (n={o['n']})  "
          f"corr(dCE,dist-since-nl)={c['corr_dCE_vs_distance_since_newline']}")
    print("   by distance-since-newline:")
    for b, s in c['dCE_by_distance_since_newline'].items():
        if s['dCE'] is not None:
            print(f"      dsn {b:<7} dCE={s['dCE']:>8}±{s['SE']:<7} n={s['n']}")
    print(f"   spread across distance bins={c['spread_across_distance_bins']}  "
          f"across content classes={c['spread_across_content_classes']}")
print("\nQK UNSUP POSITIONAL DONE", flush=True)
