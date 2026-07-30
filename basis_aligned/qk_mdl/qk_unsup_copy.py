"""UNSUPERVISED COPY / VALUE-ROUTING circuit discovery in bilin18.

WHY A NEW TOOL (different circuit types need different detectors).
The §56 class-boost pipeline (qk_unsup_discover/verify) ranks heads by whether
ablation drops a FIXED token CLASS -- a source-INDEPENDENT signature. It explicitly
concluded "NONE of the verified heads is an identity copy". COPY circuits (induction,
successor, the L13H8 v1-router delimiter payload) have the OPPOSITE signature: the
head's output at position t carries the identity of the token it ATTENDS TO (or the
token AFTER it), i.e. a source-DEPENDENT payload. A class-boost detector is blind to
this by construction, because the boosted token changes with the attended source.

THE COPY-DETECTION TECHNIQUE (what this file adds):
  For every head, at every position t:
    1. find the argmax-attended SOURCE position sp (unnormalised bilinear pattern).
    2. form the head's own residual contribution r_h = c_proj_h(yh_h) (D-vector).
    3. read r_h through the unembedding (final rms_norm + lm_head) -> which tokens
       this head BOOSTS at t.
    4. COPY-ALIGNMENT = does r_h boost (a) the attended-SOURCE token itself
       [verbatim copy] or (b) the token AFTER the source [induction/successor copy],
       vs a random-token baseline.
  COPY-PURITY = fraction of the head's top-50 activating positions where the
  attended-source token (a) or its successor (b) lands in the head's top-10 boosted
  tokens. Rank all 18x9 heads by this. This RE-DISCOVERS known copy heads unsupervised
  (positive controls) and surfaces any new ones.

  Positional/artifact heads are separated out: sink-attention fraction (sp==0),
  self-attention (sp==t), fixed-offset "previous-token" copy (sp==t-1) are all flagged,
  and the source-position offset distribution is reported so positional-copy is
  distinguished from content-copy.

CAUSAL VERIFICATION (honesty; §56 / greater-of-two lessons):
  A head that LOOKS copy-like by the proxy but whose mean-ablation does NOT
  specifically drop the source-token (or successor) logit is NOT a copy circuit -- we
  report it. Top heads + known controls are mean-ablated on the HELD-BACK slice
  FW[448:600] (untouched by discovery). We report dCE (paired SE), the drop in the
  attended-SOURCE token logit and the SUCCESSOR token logit vs a paired RANDOM-token
  drop, and classify verbatim-copy vs successor-copy vs positional-copy.

FORWARD copied VERBATIM from qk_bracket_patch.py (tier2_model.reference_forward):
bilin18 two-branch pattern (s1*s2), per-head QK rms_norm THEN RoPE, v-lerp via a.lamb
(block-0 v cache = v1-router payload), 30*tanh logits, UNNORMALISED pattern.
"""
import json, sys, math
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

FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
SEQL = 128
TRAIN = FINEWEB[0:256, :SEQL].to(DEV)                 # discovery slice
HELD = FINEWEB[448:600, :SEQL].to(DEV)                # held-back verification slice
NTR, NHELD = TRAIN.shape[0], HELD.shape[0]
BATCH = 4
KTOP = 60                                             # top activating positions kept per head
KUSE = 50                                             # top-50 positions used for purity (spec)
KBOOST = 10                                           # "top-boosted tokens" membership window
KCAUSAL = 200                                         # positions for causal statistics
print(f"bilin18 NL={NL} NH={NH} HD={HD} D={D} V={V}; train={NTR} held={NHELD}", flush=True)

def dec(t): return repr(tok.decode([int(t)]))

# special/degenerate tokens excluded from trigger selection
_special = {tok.eos_token_id}
for _t in range(V):
    if '�' in tok.decode([_t]): _special.add(_t)
SPECIAL = np.array(sorted(_special))


# =====================================================================================
# Core forward -- VERBATIM from qk_bracket_patch.py, with (a) per-head copy collection,
# (b) single-head mean-ablation. The load-bearing computation lines are IDENTICAL.
# =====================================================================================
Wr = m.transformer.h[0].attn.c_proj.weight.new_empty(0)   # placeholder; set per layer below

@torch.no_grad()
def forward(idx, ablate=None, means=None, no_v1=False, v1_patch=None, want_yh=False,
            collect=None, gspec=None, b0=None, r_top=None):
    """ablate: None | ('head',li,h) | ('attn',li) | ('mlp',li) | ('allattn',)
    means: dict with per-layer 'yh'(T,NH,HD) 'attn'(T,D) 'mlp'(T,D) per-position means.
    no_v1: force lamb->0 (no layer-0 value routing).
    v1_patch: (pos, donor_v1) donor_v1 shape (NH,HD) replaces v1[:,pos].
    want_yh: also accumulate per-layer yh/attn/mlp sums (for mean collection).
    collect: dict-out per-head copy stats (hnorm,srcpos,srcid,succid,src_sc,suc_sc).
    gspec/b0/r_top: PASS-2 gather of per-head r_h at pre-selected top positions."""
    B, Tt = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(Tt, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(Tt, Tt, device=DEV, dtype=torch.bool))
    acc = {'yh': [], 'attn': [], 'mlp': []} if want_yh else None
    out = {} if collect is not None else None
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
        yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)          # (B,Tt,NH,HD)
        # ---- per-head COPY collection (added; does not alter yh) ----
        if collect is not None or (gspec is not None and li in gspec):
            Wp = a.c_proj.weight.view(D, NH, HD)              # (D,NH,HD)
            comp = torch.einsum('bthc,dhc->bthd', yh, Wp)     # (B,Tt,NH,D) per-head residual add
            if collect is not None:
                srcpos = pat.abs().argmax(-1)                 # (B,NH,Tt) argmax key
                sp = srcpos.permute(0, 2, 1)                  # (B,Tt,NH)
                srcid = torch.gather(idx.unsqueeze(-1).expand(B, Tt, NH), 1,
                                     sp.clamp(max=Tt - 1))     # (B,Tt,NH) token at src
                sucpos = (sp + 1).clamp(max=Tt - 1)
                sucid = torch.gather(idx.unsqueeze(-1).expand(B, Tt, NH), 1, sucpos)
                Wsrc = Wu[srcid.reshape(-1)].view(B, Tt, NH, D)
                Wsuc = Wu[sucid.reshape(-1)].view(B, Tt, NH, D)
                src_sc = (comp * Wsrc).sum(-1)                # (B,Tt,NH) DLA to src token
                suc_sc = (comp * Wsuc).sum(-1)
                out[li] = {'hnorm': comp.norm(dim=-1).cpu().numpy(),   # (B,Tt,NH)
                           'srcpos': sp.cpu().numpy().astype(np.int16),
                           'srcid': srcid.cpu().numpy().astype(np.int32),
                           'sucid': sucid.cpu().numpy().astype(np.int32),
                           'src_sc': src_sc.cpu().numpy(), 'suc_sc': suc_sc.cpu().numpy()}
                del Wsrc, Wsuc, srcid, sucid, src_sc, suc_sc
            if gspec is not None and li in gspec:
                for (h, seqs_lh, poss_lh, slots_lh) in gspec[li]:
                    msk = (seqs_lh >= b0) & (seqs_lh < b0 + B)
                    if not msk.any():
                        continue
                    ls = seqs_lh[msk] - b0; pp = poss_lh[msk]; sl = slots_lh[msk]
                    r_top[(li, h)][sl] = comp[ls, pp, h].cpu().numpy()
            del comp
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
    if want_yh: return logits, acc
    if collect is not None: return logits, out
    return logits


# =====================================================================================
# PASS A: per-head means for ablation + per-head copy stats over TRAIN
# =====================================================================================
print("PASS A: collect copy stats + means over TRAIN ...", flush=True)
# per-head stat tensors (NL,NH,NTR,SEQL)
hnorm = np.zeros((NL, NH, NTR, SEQL), np.float32)
srcpos = np.zeros((NL, NH, NTR, SEQL), np.int16)
srcid = np.zeros((NL, NH, NTR, SEQL), np.int32)
sucid = np.zeros((NL, NH, NTR, SEQL), np.int32)
src_sc = np.zeros((NL, NH, NTR, SEQL), np.float32)
suc_sc = np.zeros((NL, NH, NTR, SEQL), np.float32)
YH_SUM = {li: torch.zeros(SEQL, NH, HD, device=DEV) for li in range(NL)}

for i in range(0, NTR, BATCH):
    _, out = forward(TRAIN[i:i + BATCH], collect={}, want_yh=False)
    b = out[0]['hnorm'].shape[0]
    for li in range(NL):
        o = out[li]
        # (B,Tt,NH) -> (NH,B,Tt)
        hnorm[li, :, i:i + b] = o['hnorm'].transpose(2, 0, 1)
        srcpos[li, :, i:i + b] = o['srcpos'].transpose(2, 0, 1)
        srcid[li, :, i:i + b] = o['srcid'].transpose(2, 0, 1)
        sucid[li, :, i:i + b] = o['sucid'].transpose(2, 0, 1)
        src_sc[li, :, i:i + b] = o['src_sc'].transpose(2, 0, 1)
        suc_sc[li, :, i:i + b] = o['suc_sc'].transpose(2, 0, 1)

# means (separate lightweight pass reusing want_yh)
print("PASS A2: per-position means for ablation ...", flush=True)
ASUM = {li: torch.zeros(SEQL, D, device=DEV) for li in range(NL)}
MSUM = {li: torch.zeros(SEQL, D, device=DEV) for li in range(NL)}
for i in range(0, NHELD, BATCH):
    _, acc = forward(HELD[i:i + BATCH], want_yh=True)
    for li in range(NL):
        YH_SUM[li] += acc['yh'][li]; ASUM[li] += acc['attn'][li]; MSUM[li] += acc['mlp'][li]
YHMEAN = {li: YH_SUM[li] / NHELD for li in range(NL)}   # HELD-slice per-position mean (for causal)

train_np = TRAIN.cpu().numpy()
pos_t = np.tile(np.arange(SEQL), NTR).reshape(NTR, SEQL)
is_special = np.isin(train_np, SPECIAL)
valid = (pos_t > 0) & (pos_t < SEQL - 1) & ~is_special      # exclude bos/last/special


# =====================================================================================
# select top-KTOP activating positions per head, then PASS B gathers r_h there for
# full-vocab DLA (top-boosted-token membership).
# =====================================================================================
def top_positions(li, h):
    a = hnorm[li, h].reshape(-1).copy()
    a[~valid.reshape(-1)] = -1e30
    tk = np.argpartition(a, -KTOP)[-KTOP:]
    tk = tk[np.argsort(-a[tk])]
    return tk // SEQL, tk % SEQL                             # seq, pos

gspec = {}
r_top = {}
topsel = {}
for li in range(NL):
    lst = []
    for h in range(NH):
        s, p = top_positions(li, h)
        topsel[(li, h)] = (s, p)
        r_top[(li, h)] = np.zeros((KTOP, D), np.float32)
        lst.append((h, s.astype(np.int64), p.astype(np.int64), np.arange(KTOP)))
    gspec[li] = lst

print("PASS B: gather per-head r_h at top positions for full-vocab DLA ...", flush=True)
for i in range(0, NTR, BATCH):
    forward(TRAIN[i:i + BATCH], gspec=gspec, b0=i, r_top=r_top)


# =====================================================================================
# COPY-PURITY per head from full-vocab direct-logit-attribution at top positions.
# top-boosted tokens = top-KBOOST by W_u . r_h (rms scale is a positive per-position
# scalar -> irrelevant to the top-k / ranking, so raw DLA is used).
# =====================================================================================
def conc(ids):
    ids = np.asarray(ids)
    if len(ids) <= 1: return 0.0
    _, c = np.unique(ids, return_counts=True); p = c / c.sum()
    return float(1 - (-(p * np.log(p)).sum()) / math.log(len(ids)))

Wu_dev = Wu.to(DEV)
records = []
for li in range(NL):
    for h in range(NH):
        s, p = topsel[(li, h)]
        r = torch.from_numpy(r_top[(li, h)]).to(DEV)          # (KTOP,D)
        dla = r @ Wu_dev.T                                    # (KTOP,V)
        topb = torch.topk(dla, KBOOST, dim=-1).indices.cpu().numpy()  # (KTOP,KBOOST)
        use = slice(0, KUSE)
        s50, p50 = s[use], p[use]
        sid = srcid[li, h, s50, p50]; uid = sucid[li, h, s50, p50]
        spo = srcpos[li, h, s50, p50].astype(np.int64)
        tb = topb[use]
        verb = np.array([sid[j] in tb[j] for j in range(KUSE)])
        succ = np.array([uid[j] in tb[j] for j in range(KUSE)])
        # rank of src token among V (0 = top)
        srank = np.array([int((dla[use][j] > dla[use][j, int(sid[j])]).sum().item()) for j in range(KUSE)])
        urank = np.array([int((dla[use][j] > dla[use][j, int(uid[j])]).sum().item()) for j in range(KUSE)])
        # z-score of src / succ logit vs vocab distribution at each position
        mu = dla[use].mean(-1, keepdim=True); sd = dla[use].std(-1, keepdim=True)
        z = (dla[use] - mu) / sd
        srcz = float(z[torch.arange(KUSE), torch.from_numpy(sid.astype(np.int64)).to(DEV)].mean())
        sucz = float(z[torch.arange(KUSE), torch.from_numpy(uid.astype(np.int64)).to(DEV)].mean())
        offset = (p50 - spo)                                   # t - src_pos
        rec = {
            'head': f"L{li}H{h}", 'li': li, 'h': h,
            'copy_purity': round(float((verb | succ).mean()), 3),
            'verbatim_purity': round(float(verb.mean()), 3),
            'successor_purity': round(float(succ.mean()), 3),
            'src_logit_z': round(srcz, 2), 'succ_logit_z': round(sucz, 2),
            'src_rank_median': int(np.median(srank)), 'succ_rank_median': int(np.median(urank)),
            'sink_frac': round(float((spo == 0).mean()), 3),
            'self_frac': round(float((spo == p50).mean()), 3),
            'prev_frac': round(float((offset == 1).mean()), 3),
            'offset_median': int(np.median(offset)), 'offset_conc': round(conc(offset), 3),
            'src_tok_conc': round(conc(sid), 3),
            'top_src_toks': [(dec(t), int(c)) for t, c in
                             sorted([(t, int((sid == t).sum())) for t in np.unique(sid)],
                                    key=lambda kv: -kv[1])[:4]],
        }
        records.append(rec)
        del r, dla

records.sort(key=lambda r: -r['copy_purity'])
for rank, r in enumerate(records):
    r['purity_rank'] = rank
print("\n===== COPY-PURITY RANKING (top 25) =====", flush=True)
for r in records[:25]:
    print(f"{r['purity_rank']:3d} {r['head']:6s} cp={r['copy_purity']:.2f} "
          f"vb={r['verbatim_purity']:.2f} sc={r['successor_purity']:.2f} "
          f"srcz={r['src_logit_z']:+.1f} sink={r['sink_frac']:.2f} self={r['self_frac']:.2f} "
          f"prev={r['prev_frac']:.2f} off_med={r['offset_median']} off_conc={r['offset_conc']:.2f}", flush=True)


# =====================================================================================
# CAUSAL VERIFICATION on HELD-BACK slice. Mean-ablate head; measure specific drop in
# attended-source / successor logit vs a paired random-token drop.
# =====================================================================================
KNOWN = {'L5H5': 'induction(atlas)', 'L8H3': 'successor/induction(L8 v1-router)',
         'L8H7': 'increment/successor(L8 v1-router)', 'L13H8': 'delimiter/closure(v1-router)'}
top5 = [r['head'] for r in records[:5]] + ['L6H7']   # +top NEW successor-proxy head
verify_heads = []
for hn in top5 + list(KNOWN):
    if hn not in verify_heads:
        verify_heads.append(hn)

# collect held-slice copy stats for the heads we verify (need src/succ ids out-of-sample)
vset = {(int(hn[1:hn.index('H')]), int(hn[hn.index('H') + 1:])) for hn in verify_heads}
held_np = HELD.cpu().numpy()
pos_th = np.tile(np.arange(SEQL), NHELD).reshape(NHELD, SEQL)
is_special_h = np.isin(held_np, SPECIAL)
valid_h = (pos_th > 0) & (pos_th < SEQL - 1) & ~is_special_h
h_hnorm = {p: np.zeros((NHELD, SEQL), np.float32) for p in vset}
h_srcpos = {p: np.zeros((NHELD, SEQL), np.int16) for p in vset}
h_srcid = {p: np.zeros((NHELD, SEQL), np.int32) for p in vset}
h_sucid = {p: np.zeros((NHELD, SEQL), np.int32) for p in vset}
print("\nPASS C: held-slice copy stats for verify heads ...", flush=True)
for i in range(0, NHELD, BATCH):
    _, out = forward(HELD[i:i + BATCH], collect={})
    b = out[0]['hnorm'].shape[0]
    for (li, h) in vset:
        o = out[li]
        h_hnorm[(li, h)][i:i + b] = o['hnorm'][:, :, h]
        h_srcpos[(li, h)][i:i + b] = o['srcpos'][:, :, h]
        h_srcid[(li, h)][i:i + b] = o['srcid'][:, :, h]
        h_sucid[(li, h)][i:i + b] = o['sucid'][:, :, h]

rng = np.random.RandomState(0)

@torch.no_grad()
def causal(li, h):
    a = h_hnorm[(li, h)].reshape(-1).copy(); a[~valid_h.reshape(-1)] = -1e30
    tk = np.argpartition(a, -KCAUSAL)[-KCAUSAL:]; tk = tk[np.argsort(-a[tk])]
    seqs, poss = tk // SEQL, tk % SEQL
    base_ce, abl_ce = [], []
    src_drop, suc_drop, rnd_drop = [], [], []
    src_in_top10, offsets = [], []
    uniq = np.unique(seqs)
    for i in range(0, len(uniq), BATCH):
        sb = uniq[i:i + BATCH]; idx = HELD[sb]
        base = forward(idx).float()
        abl = forward(idx, ablate=('head', li, h), means={'yh': YHMEAN}).float()
        loc = {int(s): j for j, s in enumerate(sb)}
        for s, p in zip(seqs, poss):
            if int(s) not in loc: continue
            j = loc[int(s)]; tgt = int(held_np[s, p + 1])
            bl, al = base[j, p], abl[j, p]
            base_ce.append(F.cross_entropy(bl.unsqueeze(0), torch.tensor([tgt], device=DEV)).item())
            abl_ce.append(F.cross_entropy(al.unsqueeze(0), torch.tensor([tgt], device=DEV)).item())
            dl = bl - al                                       # base - ablated (what head boosts)
            st = int(h_srcid[(li, h)][s, p]); ut = int(h_sucid[(li, h)][s, p])
            rt = int(rng.randint(V))
            src_drop.append(float(dl[st])); suc_drop.append(float(dl[ut])); rnd_drop.append(float(dl[rt]))
            src_in_top10.append(int(st in torch.topk(dl, 10).indices.cpu().numpy()))
            offsets.append(int(p - int(h_srcpos[(li, h)][s, p])))
    base_ce = np.array(base_ce); abl_ce = np.array(abl_ce); dce = abl_ce - base_ce
    n = len(dce); sd_, ud_, rd_ = map(np.array, (src_drop, suc_drop, rnd_drop))
    def se(x): return round(float(x.std(ddof=1) / math.sqrt(len(x))), 4)
    off = np.array(offsets)
    return {
        'n': n, 'dCE_mean': round(float(dce.mean()), 4), 'dCE_SE': se(dce),
        'dCE_frac_pos': round(float((dce > 0).mean()), 3),
        'src_logit_drop_mean': round(float(sd_.mean()), 4), 'src_logit_drop_SE': se(sd_),
        'succ_logit_drop_mean': round(float(ud_.mean()), 4), 'succ_logit_drop_SE': se(ud_),
        'rand_logit_drop_mean': round(float(rd_.mean()), 4), 'rand_logit_drop_SE': se(rd_),
        'src_minus_rand': round(float((sd_ - rd_).mean()), 4), 'src_minus_rand_SE': se(sd_ - rd_),
        'succ_minus_rand': round(float((ud_ - rd_).mean()), 4), 'succ_minus_rand_SE': se(ud_ - rd_),
        'src_in_top10_frac': round(float(np.mean(src_in_top10)), 3),
        'offset_median': int(np.median(off)), 'offset_conc': round(conc(off), 3),
    }

def classify(c):
    """The causal signature of a COPY is a SOURCE-specific (or successor-specific)
    logit drop under ablation -- NOT dCE. dCE only says whether that copy is
    LOAD-BEARING (pivotal for the true next token) at the head's own top positions;
    a copy can be real but redundant (dCE~0). Positional-copy = fixed source offset
    with no token-specific drop. A paired/transformed copy (e.g. bracket-closure)
    SUPPRESSES the source token (src drop < 0) while promoting a different paired
    token -- the verbatim/successor detector correctly declines it."""
    sr, srse = c['src_minus_rand'], c['src_minus_rand_SE']
    ur, urse = c['succ_minus_rand'], c['succ_minus_rand_SE']
    dsig = c['dCE_mean'] > 3 * c['dCE_SE']
    lb = ' [load-bearing]' if dsig else ' [redundant: dCE~0]'
    src_sig = sr > 3 * srse; suc_sig = ur > 3 * urse
    if src_sig and sr >= ur:
        return 'VERBATIM-COPY' + lb
    if suc_sig and ur > sr:
        return 'SUCCESSOR-COPY' + lb
    if not dsig:
        return 'NOT-COPY (no source/successor drop, dCE~0)'
    if sr < -3 * srse:
        return 'PAIRED/TRANSFORMED-COPY (suppresses source, promotes paired token) [causal]'
    if c['offset_conc'] > 0.5:
        return 'POSITIONAL-COPY (fixed offset, no token-specific drop) [causal]'
    return 'NOT-COPY (causal but no source/successor specificity) [causal]'

print("\n===== CAUSAL VERIFICATION =====", flush=True)
CAUSAL = {}
for hn in verify_heads:
    li = int(hn[1:hn.index('H')]); h = int(hn[hn.index('H') + 1:])
    c = causal(li, h)
    c['label'] = classify(c)
    c['load_bearing'] = bool(c['dCE_mean'] > 3 * c['dCE_SE'])
    c['known'] = KNOWN.get(hn, '')
    c['purity_rank'] = next(r['purity_rank'] for r in records if r['head'] == hn)
    CAUSAL[hn] = c
    print(f"{hn:6s} rank={c['purity_rank']:3d} dCE={c['dCE_mean']:.3f}±{c['dCE_SE']:.3f} "
          f"src-rnd={c['src_minus_rand']:+.3f}±{c['src_minus_rand_SE']:.3f} "
          f"succ-rnd={c['succ_minus_rand']:+.3f}±{c['succ_minus_rand_SE']:.3f} "
          f"off={c['offset_median']}({c['offset_conc']:.2f}) -> {c['label']} {c['known']}", flush=True)

OUT = {
    'technique': 'unsupervised COPY / value-routing detection: per-head, per-position '
                 'argmax-attended-source alignment of the head DLA with the SOURCE token '
                 '(verbatim) or its SUCCESSOR (induction) vs random baseline; ranks all '
                 '18x9 heads by copy-purity over top-50 activating positions. Complements '
                 'the source-INDEPENDENT class-boost detector of the section-56 pipeline.',
    'params': {'train_slice': 'FW[0:256,:128]', 'held_slice': 'FW[448:600,:128]',
               'KTOP': KTOP, 'KUSE': KUSE, 'KBOOST': KBOOST, 'KCAUSAL': KCAUSAL},
    'ranking': records,
    'causal': CAUSAL,
    'verify_heads': verify_heads,
}
json.dump(OUT, open(f'{QK}/qk_unsup_copy.json', 'w'), indent=2)
print("\nQK UNSUP COPY DONE", flush=True)
