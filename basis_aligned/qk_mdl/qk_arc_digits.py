"""OPTION-2 ALGORITHMIC ARC on the DIGIT-ATTENDING heads h.L8.7 and h.L8.3 (§63).

Both heads were discovered unsupervised to ATTEND-TO digit-containing tokens, with causal
damage concentrating on digit positions even after a position-matched control (§63 red-team:
~4x for h.L8.7, ~7.6x for h.L8.3). "Fires on digits" is a TRIGGER. This arc asks WHAT digit
ALGORITHM they implement, distinguishing:
  (H1) NUMBER/DIGIT CONTINUATION   -- predict the next digit in a numeric run.
  (H2) DIGIT COPYING / value-router -- attend back to a referenced number and COPY its identity.
  (H3) mere DIGIT DETECTOR feeding a diffuse downstream -- trigger-genuine / output-diffuse.

Discipline: VERIFY behavior -> PATCH to minimal causal circuit -> RED-TEAM -> report.

FORWARD copied VERBATIM from qk_unsup_copy.py / qk_unsup_verify.py (tier2_model.reference):
bilin18 two-branch pattern (s1*s2), per-head QK rms_norm THEN RoPE, v-lerp via a.lamb (block-0
v cache = v1-router payload), UNNORMALISED pattern, 30*tanh logits. NO softmax.
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
Wu = m.lm_head.weight.detach().float()               # (V,D) unembedding
print(f"bilin18 NL={NL} NH={NH} HD={HD} D={D} V={V}", flush=True)

FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
SEQL = 128
HELD = FINEWEB[448:600, :SEQL].to(DEV)               # held-back causal slice (untouched by discovery)
NHELD = HELD.shape[0]
BATCH = 6
held_np = HELD.cpu().numpy()

# special/degenerate token ids (doc-sep + UTF-8 replacement) excluded from analysis
_special = {tok.eos_token_id}
for _t in range(V):
    if '�' in tok.decode([_t]): _special.add(_t)
SPECIAL = np.array(sorted(_special))

# ---- decoded-token digit predicates (matches the §63 byte-level definition) ----
def dec(t): return repr(tok.decode([int(t)]))
_has_digit = np.zeros(V, bool); _is_num = np.zeros(V, bool)
for t in range(V):
    core = tok.decode([t]).strip()
    if core and any(c.isdigit() for c in core):
        _has_digit[t] = True
        if all(c.isdigit() for c in core): _is_num[t] = True   # pure numeric token e.g. '17','3'
print(f"{_has_digit.sum()} has-digit tokens, {_is_num.sum()} pure-numeric tokens; "
      f"{len(SPECIAL)} special ids", flush=True)

HEADS = [(8, 3), (8, 7)]                              # the two digit-attending heads
LAYER = 8

# =====================================================================================
# Core forward -- VERBATIM from qk_unsup_copy.py. Extensions: (a) head-LIST mean-ablation
# (joint), (b) 'allattn' static-prior floor, (c) per-target-head src collection.
# =====================================================================================
@torch.no_grad()
def forward(idx, ablate=None, means=None, want_yh=False, collect=None):
    """ablate: None | ('heads',li,[h,...]) | ('allattn',).
    means: dict 'yh'(per-layer T,NH,HD) and 'attn'(per-layer T,D) per-position means.
    want_yh: accumulate per-layer yh/attn sums (for mean collection).
    collect: dict-out per target head (hnorm,srcpos,srcid)."""
    B, Tt = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(Tt, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(Tt, Tt, device=DEV, dtype=torch.bool))
    acc = {'yh': [], 'attn': []} if want_yh else None
    out = {} if collect is not None else None
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn; hc = F.rms_norm(x, (D,))
        def qk(l): z = F.rms_norm(l(hc).view(B, Tt, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hc).view(B, Tt, NH, HD)
        if v1 is None: v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)          # (B,Tt,NH,HD)
        if collect is not None:
            Wp = a.c_proj.weight.view(D, NH, HD)
            srcpos = pat.abs().argmax(-1)                     # (B,NH,Tt)
            for (tli, th) in HEADS:
                if tli == li:
                    comp = torch.einsum('btc,oc->bto', yh[:, :, th], Wp[:, th])  # (B,Tt,D)
                    sp = srcpos[:, th, :]                                        # (B,Tt)
                    out[(tli, th)] = {'hnorm': comp.norm(dim=-1).cpu().numpy(),
                                      'srcpos': sp.cpu().numpy().astype(np.int16),
                                      'srcid': torch.gather(idx, 1, sp).cpu().numpy().astype(np.int32)}
        if ablate is not None and ablate[0] == 'heads' and ablate[1] == li:
            yh = yh.clone()
            for h in ablate[2]:
                yh[:, :, h] = means['yh'][li][:, h].unsqueeze(0)
        if want_yh: acc['yh'].append(yh.sum(0))
        attn = a.c_proj(yh.reshape(B, Tt, -1))
        if ablate is not None and ablate[0] == 'allattn':
            attn = means['attn'][li].unsqueeze(0).expand(B, -1, -1)
        if want_yh: acc['attn'].append(attn.sum(0))
        x = x + attn
        mo = blk.mlp(F.rms_norm(x, (D,)))
        x = x + mo
    logits = 30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)
    if want_yh: return logits, acc
    if collect is not None: return logits, out
    return logits

# =====================================================================================
# PASS A: per-position means (for ablation) + per-head src collection over HELD
# =====================================================================================
print("PASS A: means + per-head source collection over HELD ...", flush=True)
YH_SUM = {li: torch.zeros(SEQL, NH, HD, device=DEV) for li in range(NL)}
A_SUM = {li: torch.zeros(SEQL, D, device=DEV) for li in range(NL)}
hnorm = {p: np.zeros((NHELD, SEQL), np.float32) for p in HEADS}
srcpos = {p: np.zeros((NHELD, SEQL), np.int16) for p in HEADS}
srcid = {p: np.zeros((NHELD, SEQL), np.int32) for p in HEADS}
for i in range(0, NHELD, BATCH):
    _, acc = forward(HELD[i:i+BATCH], want_yh=True)
    b = HELD[i:i+BATCH].shape[0]
    for li in range(NL):
        YH_SUM[li] += acc['yh'][li]; A_SUM[li] += acc['attn'][li]
    _, out = forward(HELD[i:i+BATCH], collect={})
    for p in HEADS:
        hnorm[p][i:i+b] = out[p]['hnorm']
        srcpos[p][i:i+b] = out[p]['srcpos']
        srcid[p][i:i+b] = out[p]['srcid']
MEANS = {'yh': {li: YH_SUM[li] / NHELD for li in range(NL)},
         'attn': {li: A_SUM[li] / NHELD for li in range(NL)}}

# =====================================================================================
# masks and structure over the held grid
# =====================================================================================
pos_t = np.tile(np.arange(SEQL), NHELD).reshape(NHELD, SEQL)
is_special = np.isin(held_np, SPECIAL)
next_tok = np.zeros_like(held_np); next_tok[:, :-1] = held_np[:, 1:]
next_special = np.zeros_like(is_special); next_special[:, :-1] = is_special[:, 1:]
valid = (pos_t > 0) & (pos_t < SEQL - 1) & ~is_special & ~next_special

cur_digit = _has_digit[held_np]
next_digit = _has_digit[next_tok]
next_num = _is_num[next_tok]

# distance since last newline (for the position-matched red-team control)
nl_ids = set(t for t in range(V) if '\n' in tok.decode([t]))
is_nl = np.isin(held_np, np.array(sorted(nl_ids))) if nl_ids else np.zeros_like(held_np, bool)
dist_nl = np.zeros((NHELD, SEQL), np.int32)
for s in range(NHELD):
    d = 0
    for p in range(SEQL):
        dist_nl[s, p] = d
        d = 0 if is_nl[s, p] else d + 1

# "matching earlier number to copy": does the correct next token appear earlier in context?
next_copyable = np.zeros((NHELD, SEQL), bool)
next_src_hit = {p: np.zeros((NHELD, SEQL), bool) for p in HEADS}  # head attends the token it predicts
for s in range(NHELD):
    seen = set()
    for p in range(SEQL - 1):
        tn = int(next_tok[s, p])
        # token appears at some earlier position <= p (a referenceable number)
        if tn in seen or tn == int(held_np[s, p]):
            next_copyable[s, p] = True
        for hp in HEADS:
            if int(srcid[hp][s, p]) == tn:
                next_src_hit[hp][s, p] = True
        seen.add(int(held_np[s, p]))

# =====================================================================================
# STEP 1 -- VERIFY behavior on next-is-digit positions
# =====================================================================================
print("STEP 1: baseline next-digit behavior ...", flush=True)
@torch.no_grad()
def logit_grids(ablate=None):
    """return per-position: ce (nll of true next), p_correct (prob true next),
    and full logits are NOT stored -- only per-position gathered scalars."""
    ce = np.full((NHELD, SEQL), np.nan, np.float32)
    pc = np.full((NHELD, SEQL), np.nan, np.float32)
    top1 = np.full((NHELD, SEQL), -1, np.int64)
    for i in range(0, NHELD, BATCH):
        idx = HELD[i:i+BATCH]; b = idx.shape[0]
        lg = forward(idx, ablate=ablate, means=MEANS).float()
        lp = F.log_softmax(lg[:, :-1], -1)
        tgt = idx[:, 1:]
        nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        ce[i:i+b, :-1] = nll.cpu().numpy()
        pc[i:i+b, :-1] = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).exp().cpu().numpy()
        top1[i:i+b, :-1] = lg[:, :-1].argmax(-1).cpu().numpy()
    return ce, pc, top1

base_ce, base_pc, base_top1 = logit_grids(None)
nd = valid & next_digit                              # next-token-is-digit positions
n_nd = int(nd.sum())
base_acc_nd = float((base_top1[nd] == next_tok[nd]).mean())
base_pc_nd = float(base_pc[nd].mean())
# local structure at next-digit positions
frac_cur_digit = float(cur_digit[nd].mean())          # is previous(current) token a digit (a run)
frac_copyable = float(next_copyable[nd].mean())        # matching earlier number exists
VERIFY = dict(
    n_next_digit_pos=n_nd, n_valid=int(valid.sum()),
    base_acc_next_digit=round(base_acc_nd, 4),
    base_p_correct_next_digit=round(base_pc_nd, 4),
    base_ce_next_digit=round(float(np.nanmean(base_ce[nd])), 4),
    base_ce_all_valid=round(float(np.nanmean(base_ce[valid])), 4),
    frac_prev_token_is_digit=round(frac_cur_digit, 4),
    frac_has_matching_earlier_number=round(frac_copyable, 4),
    frac_next_is_pure_numeric=round(float(next_num[nd].mean()), 4),
)
for hp in HEADS:
    VERIFY[f'frac_srcid_eq_next_h{hp[0]}_{hp[1]}'] = round(float(next_src_hit[hp][nd].mean()), 4)
print("VERIFY:", json.dumps(VERIFY, indent=1), flush=True)

# =====================================================================================
# STEP 2 -- PATCH: mean-ablate heads (alone + joint) at digit-relevant positions.
# Measure dCE and drop in P(correct next digit); locate where damage concentrates; and
# for H2 do a source/value-router logit-drop analysis (dl at attended-source token).
# =====================================================================================
def paired(x):
    x = x[np.isfinite(x)]; n = len(x)
    if n == 0: return dict(n=0, mean=None, se=None)
    return dict(n=int(n), mean=round(float(x.mean()), 4),
                se=round(float(x.std(ddof=1)/math.sqrt(n)) if n > 1 else float('nan'), 4),
                frac_pos=round(float((x > 0).mean()), 3))

@torch.no_grad()
def ablate_analysis(ablate, src_head=None):
    """Full-grid paired dCE and dP(correct). Also, per position, the head's DLA
    contribution dl=base_logit-abl_logit at (correct-next token, attended-source token,
    a fixed random token) -- computed by running base+ablated together per batch."""
    dce = np.full((NHELD, SEQL), np.nan, np.float32)     # abl_ce - base_ce
    dpc = np.full((NHELD, SEQL), np.nan, np.float32)     # base_pc - abl_pc (drop in P(correct))
    dl_next = np.full((NHELD, SEQL), np.nan, np.float32)  # head DLA to correct-next token
    dl_src = np.full((NHELD, SEQL), np.nan, np.float32)   # head DLA to attended-source token
    dl_rnd = np.full((NHELD, SEQL), np.nan, np.float32)   # head DLA to a fixed random token
    rng = np.random.RandomState(0)
    rnd_tok = rng.randint(0, V, size=(NHELD, SEQL))
    for i in range(0, NHELD, BATCH):
        idx = HELD[i:i+BATCH]; b = idx.shape[0]
        lb = forward(idx, means=MEANS).float()
        la = forward(idx, ablate=ablate, means=MEANS).float()
        tgt = idx[:, 1:]
        lpb = F.log_softmax(lb[:, :-1], -1); lpa = F.log_softmax(la[:, :-1], -1)
        ceb = -lpb.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        cea = -lpa.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        dce[i:i+b, :-1] = (cea - ceb).cpu().numpy()
        pcb = lpb.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).exp()
        pca = lpa.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).exp()
        dpc[i:i+b, :-1] = (pcb - pca).cpu().numpy()
        dl = (lb - la)                                    # (b,T,V) head's logit contribution
        # gather at correct-next / random over valid positions
        nt = torch.from_numpy(next_tok[i:i+b]).to(DEV).long()
        rt = torch.from_numpy(rnd_tok[i:i+b]).to(DEV).long()
        dl_next[i:i+b] = dl.gather(-1, nt.unsqueeze(-1)).squeeze(-1).cpu().numpy()
        dl_rnd[i:i+b] = dl.gather(-1, rt.unsqueeze(-1)).squeeze(-1).cpu().numpy()
        if src_head is not None:
            st = torch.from_numpy(srcid[src_head][i:i+b].astype(np.int64)).to(DEV)
            dl_src[i:i+b] = dl.gather(-1, st.unsqueeze(-1)).squeeze(-1).cpu().numpy()
        del dl, lb, la
    return dce, dpc, dl_next, dl_src, dl_rnd

def concentration_report(dce, dpc, dl_next, dl_src, dl_rnd, src_head):
    R = {}
    R['dCE_all_valid'] = paired(dce[valid])
    R['dCE_next_digit'] = paired(dce[valid & next_digit])
    R['dCE_next_NONdigit'] = paired(dce[valid & ~next_digit])
    R['dP_correct_next_digit'] = paired(dpc[valid & next_digit])   # drop in P(correct next digit)
    R['dP_correct_next_NONdigit'] = paired(dpc[valid & ~next_digit])
    R['head_DLA_to_correct_next__next_digit'] = paired(dl_next[valid & next_digit])
    R['head_DLA_to_random__next_digit'] = paired(dl_rnd[valid & next_digit])
    # concentration multiplier (next-digit vs non-digit, dCE)
    a = R['dCE_next_digit']['mean']; bb = R['dCE_next_NONdigit']['mean']
    R['dCE_digit_over_nondigit_ratio'] = round(a/bb, 2) if (a and bb and bb > 0) else None
    if src_head is not None:
        sd = valid & _has_digit[srcid[src_head]]                   # head attends a digit source
        R['dCE_srcDIGIT'] = paired(dce[sd])
        R['dCE_srcNONdigit'] = paired(dce[valid & ~_has_digit[srcid[src_head]]])
        R['head_DLA_to_attended_source__all_valid'] = paired(dl_src[valid])
        R['head_DLA_to_attended_source__next_digit'] = paired(dl_src[valid & next_digit])
        # H2 value-router: copyable vs non-copyable next-digit positions
        R['dCE_next_digit_COPYABLE'] = paired(dce[valid & next_digit & next_copyable])
        R['dCE_next_digit_NONcopyable'] = paired(dce[valid & next_digit & ~next_copyable])
        R['dP_correct_next_digit_COPYABLE'] = paired(dpc[valid & next_digit & next_copyable])
        R['dP_correct_next_digit_NONcopyable'] = paired(dpc[valid & next_digit & ~next_copyable])
        # source-hit positions: head attends the very token it must predict
        sh = valid & next_digit & next_src_hit[src_head]
        R['dCE_next_digit_SRC_EQ_NEXT'] = paired(dce[sh])
        R['dP_correct__SRC_EQ_NEXT'] = paired(dpc[sh])
        R['head_DLA_to_source__SRC_EQ_NEXT'] = paired(dl_src[sh])
    return R

PATCH = {}
CONFIGS = [('h.L8.3', ('heads', LAYER, [3]), (8, 3)),
           ('h.L8.7', ('heads', LAYER, [7]), (8, 7)),
           ('h.L8.3+h.L8.7 JOINT', ('heads', LAYER, [3, 7]), (8, 3))]
for name, ab, src_head in CONFIGS:
    print(f"STEP 2: ablate {name} ...", flush=True)
    dce, dpc, dln, dls, dlr = ablate_analysis(ab, src_head=src_head)
    PATCH[name] = concentration_report(dce, dpc, dln, dls, dlr, src_head)
    r = PATCH[name]
    print(f"  dCE all={r['dCE_all_valid']['mean']}  next_digit={r['dCE_next_digit']['mean']}"
          f"  nondigit={r['dCE_next_NONdigit']['mean']}  ratio={r['dCE_digit_over_nondigit_ratio']}"
          f"  dP(correct|next_digit)={r['dP_correct_next_digit']['mean']}", flush=True)

# =====================================================================================
# STEP 3 -- RED-TEAM the frequency-prior confound.
# (a) static-prior floor: ablate ALL attention (like the §40 greater-of-two control) and
#     measure the residual next-digit CE / accuracy. The two heads only compute something
#     if the joint-head damage is a real fraction of the full attention contribution AND
#     next-digit prediction is not mostly a context-free prior.
# (b) position-matched next-digit control (extend §63): compare digit-next dCE vs
#     non-digit-next dCE within matched distance-since-newline bins.
# =====================================================================================
print("STEP 3: static-prior floor (all-attention ablated) ...", flush=True)
allattn_ce, allattn_pc, allattn_top1 = logit_grids(('allattn',))
REDTEAM = dict(
    static_prior_floor=dict(
        base_ce_next_digit=round(float(np.nanmean(base_ce[nd])), 4),
        allattn_ablated_ce_next_digit=round(float(np.nanmean(allattn_ce[nd])), 4),
        base_acc_next_digit=round(base_acc_nd, 4),
        allattn_ablated_acc_next_digit=round(float((allattn_top1[nd] == next_tok[nd]).mean()), 4),
        base_p_correct_next_digit=round(base_pc_nd, 4),
        allattn_ablated_p_correct_next_digit=round(float(np.nanmean(allattn_pc[nd])), 4),
        note='full attention contribution to next-digit prediction = base minus all-attn-ablated; '
             'compare the two heads joint dCE against this to gauge what fraction of the real '
             'attention computation they carry vs a context-free static prior floor.',
    ))
# full attention contribution (paired) and joint-heads share
allattn_dce = allattn_ce - base_ce
REDTEAM['static_prior_floor']['full_attention_dCE_next_digit'] = paired(allattn_dce[nd])
joint_r = PATCH['h.L8.3+h.L8.7 JOINT']['dCE_next_digit']
REDTEAM['static_prior_floor']['joint_two_head_dCE_next_digit'] = joint_r
if joint_r['mean'] and REDTEAM['static_prior_floor']['full_attention_dCE_next_digit']['mean']:
    REDTEAM['static_prior_floor']['two_head_share_of_full_attention'] = round(
        joint_r['mean'] / REDTEAM['static_prior_floor']['full_attention_dCE_next_digit']['mean'], 3)

# (b) position-matched next-digit control within distance-since-newline bins
print("STEP 3b: position-matched (distance-since-newline) next-digit control ...", flush=True)
def matched_bins(dce, label):
    bins = [(1, 3), (4, 7), (8, 15), (16, 31), (32, 127)]
    rows = []
    for lo, hi in bins:
        b = valid & (dist_nl >= lo) & (dist_nl <= hi)
        dgt = paired(dce[b & next_digit]); non = paired(dce[b & ~next_digit])
        ratio = (round(dgt['mean']/non['mean'], 2)
                 if (dgt['mean'] and non['mean'] and non['mean'] > 0) else None)
        rows.append(dict(bin=f"{lo}-{hi}", n_digit=dgt['n'], n_nondigit=non['n'],
                         digit=dgt, nondigit=non, ratio=ratio))
    return rows
# recompute joint dce grid for the matched control
dce_joint, _, _, _, _ = ablate_analysis(('heads', LAYER, [3, 7]), src_head=None)
REDTEAM['position_matched_next_digit'] = dict(
    circuit='h.L8.3+h.L8.7 joint', bins=matched_bins(dce_joint, 'joint'))

# =====================================================================================
OUT = dict(
    meta=dict(model='bilin18', heads=['h.L8.3', 'h.L8.7'], layer=LAYER,
              held_slice='FW[448:600,:128]', batch=BATCH,
              digit_predicate='decoded token core contains a digit character (has_digit); '
                              'is_num = core all digits',
              ablation='mean-ablate head yh to per-position HELD-slice mean (in-distribution, not zero)',
              hypotheses=dict(H1='number/digit continuation', H2='digit copying / value-router',
                              H3='mere digit detector, diffuse output')),
    step1_verify=VERIFY,
    step2_patch=PATCH,
    step3_redteam=REDTEAM,
)
json.dump(OUT, open(f'{QK}/qk_arc_digits.json', 'w'), indent=2,
          default=lambda o: float(o) if isinstance(o, np.floating) else int(o) if isinstance(o, np.integer) else str(o))
print("\nSaved qk_arc_digits.json", flush=True)
print("QK ARC DIGITS DONE", flush=True)
