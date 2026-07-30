"""CAUSAL verification + NAMING of top unsupervised-discovered paths in bilin18.

Discovery (qk_unsup_discover.py) ranked 234 paths by trigger/effect cleanliness on
TRAIN slice FW[0:256,:128]. The direct-to-logits effect proxy is WEAK (a head's real
effect is a downstream copy), so here every path is verified CAUSALLY on the HELD-BACK
slice FW[448:600,:128] (untouched by discovery).

Per path:
  1. TRIGGER confirmation out-of-sample: top-activating held-out positions' current-token
     and attended-source distributions.
  2. CAUSAL EFFECT: mean-ablate the path (in-distribution PER-POSITION mean, NOT zero)
     and measure CE increase (ablated - base) at trigger positions, paired SE, plus the
     mean Delta-logit vector (what the path boosts) and the true-next tokens whose
     prediction breaks most.
  3. COPY vs CLASS-BOOST: does ablation drop the ATTENDED-SOURCE token's logit (copy,
     source-dependent) or a FIXED token set (class boost)?
  4. NAME: "[held-out trigger] -> [verified causal output]".

FORWARD copied VERBATIM from qk_conditional_redirect.py / qk_bracket_patch.py
(tier2_model.reference_forward): bilin18 two-branch pattern (q1k1/HD)(q2k2/HD) masked
UNNORMALISED, per-head QK rms_norm THEN RoPE, v-lerp via a.lamb (block-0 v cache),
30*tanh logits. MLP is Bilinear (Left*Right -> Down).
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

FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
SEQL = 128
TRAIN = FINEWEB[0:256, :SEQL].to(DEV)        # discovery slice (only for MLP direction recompute)
HELD = FINEWEB[448:600, :SEQL].to(DEV)       # held-back verification slice
NHELD = HELD.shape[0]
BATCH = 8
KSIG = 50                                    # top-K for signature (matches discovery)
KCAUSAL = 200                                # top-K positions for causal statistics
print(f"bilin18 NL={NL} NH={NH} HD={HD} D={D} V={V}; held={tuple(HELD.shape)}", flush=True)

# special/degenerate tokens (document-sep + UTF-8 replacement) -- excluded from trigger selection
_special = {tok.eos_token_id}
for _t in range(V):
    if '�' in tok.decode([_t]): _special.add(_t)
SPECIAL = np.array(sorted(_special))

# --- target paths ---
HEAD_PATHS = [(8, 2), (9, 6), (10, 7), (13, 8), (13, 3),      # (li,h) original targets
              (7, 3), (13, 1), (10, 6), (3, 3),                # extra novel candidates from ranking scan
              (8, 4), (12, 2), (6, 0), (2, 1),                 # 2nd batch: year-copy / attribution / connective / list
              (1, 2), (4, 0), (6, 7)]                          # 3rd batch: bullet-item / number-paren-closer / citation
MLP_PATH = (1, 2)                                             # (li, svd-dir index)
TARGET_LAYERS = sorted({li for li, _ in HEAD_PATHS} | {MLP_PATH[0]})

def dec(t): return repr(tok.decode([int(t)]))

# =====================================================================================
# Recompute the MLP SVD direction on the TRAIN slice (path is DEFINED by discovery dir)
# =====================================================================================
gram1 = torch.zeros(D, D, device=DEV)

@torch.no_grad()
def fwd_gram(idx):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(MLP_PATH[0] + 1):
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
        if li == MLP_PATH[0]:
            gram1.add_(torch.einsum('btd,bte->de', mo, mo))
        x = x + mo

for i in range(0, TRAIN.shape[0], BATCH):
    fwd_gram(TRAIN[i:i+BATCH])
_evals, _evecs = torch.linalg.eigh(gram1)
mlp_dir = _evecs[:, -4:].T.flip(0)[MLP_PATH[1]].contiguous()  # top-4 desc, take index 2
mlp_dir = mlp_dir / mlp_dir.norm()
print("MLP direction recomputed from TRAIN gram.", flush=True)
del gram1, _evecs

# =====================================================================================
# Core forward (VERBATIM) with optional (a) collection, (b) single-path mean-ablation
# =====================================================================================
@torch.no_grad()
def forward(idx, collect=False, ablate=None, yhmeans=None, projmean=None):
    """ablate: None | ('head',li,h) | ('mlp',li,dir)  -- mean-ablated with per-position means.
    collect: return dict of per-target-head comp-norm (B,T) + src-token (B,T) + mlp proj (B,T),
             and accumulate per-position yh sums (for means) + mlp proj sum."""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
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
            YH_SUM[li] += yh4.sum(0)                          # (T,NH,HD) per-position sum
            srcpos = pat.abs().argmax(-1)                     # (B,NH,T) argmax|attn| key
            for (tli, th) in HEAD_PATHS:
                if tli == li:
                    comp = torch.einsum('btc,oc->bto', yh4[:, :, th], Wr[:, th])  # (B,T,D)
                    out[('hnorm', tli, th)] = comp.norm(dim=-1).cpu().numpy()        # (B,T)
                    sp = srcpos[:, th, :]                                            # (B,T) key idx
                    out[('src', tli, th)] = torch.gather(idx, 1, sp).cpu().numpy()   # token at key
        if ablate is not None and ablate[0] == 'head' and ablate[1] == li:
            yh4 = yh4.clone(); yh4[:, :, ablate[2]] = yhmeans[li][:, ablate[2]].unsqueeze(0)
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if collect and li == MLP_PATH[0]:
            pr = torch.einsum('btd,d->bt', mo, mlp_dir)
            out[('mproj',)] = pr.cpu().numpy()
            PROJ_SUM[0] += pr.sum(0)                          # (T,)
        if ablate is not None and ablate[0] == 'mlp' and ablate[1] == li:
            pr = torch.einsum('btd,d->bt', mo, mlp_dir)
            mo = mo - (pr - projmean.unsqueeze(0)).unsqueeze(-1) * mlp_dir
        x = x + mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    return (logits, out) if collect else logits

# =====================================================================================
# PASS A: collect per-position means + per-path activations/sources over held-out
# =====================================================================================
YH_SUM = {li: torch.zeros(SEQL, NH, HD, device=DEV) for li in range(NL)}
PROJ_SUM = [torch.zeros(SEQL, device=DEV)]
hnorm = {p: np.zeros((NHELD, SEQL), np.float32) for p in HEAD_PATHS}
hsrc = {p: np.zeros((NHELD, SEQL), np.int64) for p in HEAD_PATHS}
mproj = np.zeros((NHELD, SEQL), np.float32)

print("PASS A: collect ...", flush=True)
for i in range(0, NHELD, BATCH):
    _, out = forward(HELD[i:i+BATCH], collect=True)
    b = out[('mproj',)].shape[0]
    mproj[i:i+b] = out[('mproj',)]
    for p in HEAD_PATHS:
        hnorm[p][i:i+b] = out[('hnorm',)+p]
        hsrc[p][i:i+b] = out[('src',)+p]
YHMEAN = {li: YH_SUM[li] / NHELD for li in range(NL)}
PROJMEAN = PROJ_SUM[0] / NHELD

held_np = HELD.cpu().numpy()
pos_t = np.tile(np.arange(SEQL), NHELD).reshape(NHELD, SEQL)
is_special = np.isin(held_np, SPECIAL)
# next-token target (position t predicts token at t+1); last position has no target
valid_next = pos_t < (SEQL - 1)

def signature(act, K):
    """top-K firing flat positions (exclude t=0, specials, last-pos); return (seq,pos) arrays."""
    a = act.copy().reshape(-1)
    bad = (pos_t == 0) | is_special | ~valid_next
    a[bad.reshape(-1)] = -1e30
    tk = np.argpartition(a, -K)[-K:]
    tk = tk[np.argsort(-a[tk])]
    return tk // SEQL, tk % SEQL, a[tk]

def topcount(ids, n=6):
    v, c = np.unique(np.asarray(ids), return_counts=True)
    o = np.argsort(-c)[:n]
    return [(dec(v[i]), int(c[i])) for i in o]

def conc(ids):
    ids = np.asarray(ids)
    if len(ids) <= 1: return 0.0
    _, c = np.unique(ids, return_counts=True); p = c/c.sum()
    return float(1 - (-(p*np.log(p)).sum())/math.log(len(ids)))

# =====================================================================================
# CAUSAL: mean-ablate one path, restricted to its top-KCAUSAL held-out positions
# =====================================================================================
@torch.no_grad()
def causal(path_kind, path, act, srcmat=None):
    seqs, poss, avals = signature(act, KCAUSAL)
    sig_s, sig_p, _ = signature(act, KSIG)
    cur_sig = held_np[sig_s, sig_p]
    result = {
        'held_trigger_cur_top': topcount(cur_sig),
        'held_trigger_cur_purity': round(conc(cur_sig), 4),
        'n_causal_pos': int(len(seqs)),
    }
    if srcmat is not None:
        src_sig = srcmat[sig_s, sig_p]
        result['held_trigger_src_top'] = topcount(src_sig)
        result['held_trigger_src_purity'] = round(conc(src_sig), 4)

    # gather base + ablated logits at the causal trigger positions (query t -> target t+1)
    base_ce, abl_ce = [], []
    dlogit_sum = torch.zeros(V, device=DEV)      # base - ablated (what the path boosts)
    src_dlogit = []                              # Delta-logit at the attended-source token
    src_in_top = []                              # is source token in top-10 Delta-logit gainers?
    true_next_ids, per_pos_dce = [], []
    boost_ids = None
    # process by grouping trigger positions per sequence-batch
    uniq_seqs = np.unique(seqs)
    for i in range(0, len(uniq_seqs), BATCH):
        sb = uniq_seqs[i:i+BATCH]
        idx = HELD[sb]
        base = forward(idx).float()
        if path_kind == 'head':
            abl = forward(idx, ablate=('head', path[0], path[1]), yhmeans=YHMEAN).float()
        else:
            abl = forward(idx, ablate=('mlp', path[0], path[1]), projmean=PROJMEAN).float()
        s2local = {int(s): j for j, s in enumerate(sb)}
        for s, p in zip(seqs, poss):
            if s not in s2local: continue
            j = s2local[int(s)]
            tgt = int(held_np[s, p+1])
            bl = base[j, p]; al = abl[j, p]
            base_ce.append(F.cross_entropy(bl.unsqueeze(0), torch.tensor([tgt], device=DEV)).item())
            abl_ce.append(F.cross_entropy(al.unsqueeze(0), torch.tensor([tgt], device=DEV)).item())
            dl = bl - al
            dlogit_sum += dl
            true_next_ids.append(tgt)
            per_pos_dce.append(abl_ce[-1] - base_ce[-1])
            if srcmat is not None:
                st = int(srcmat[s, p])
                src_dlogit.append(float(dl[st]))
                top10 = torch.topk(dl, 10).indices.cpu().numpy()
                src_in_top.append(int(st in top10))
    base_ce = np.array(base_ce); abl_ce = np.array(abl_ce); dce = abl_ce - base_ce
    n = len(dce)
    result['base_CE_mean'] = round(float(base_ce.mean()), 4)
    result['ablated_CE_mean'] = round(float(abl_ce.mean()), 4)
    result['dCE_mean'] = round(float(dce.mean()), 4)
    result['dCE_paired_SE'] = round(float(dce.std(ddof=1)/math.sqrt(n)), 4)
    result['dCE_frac_positive'] = round(float((dce > 0).mean()), 3)
    # what the path BOOSTS (mean Delta-logit over trigger positions)
    dmean = (dlogit_sum / n)
    top_boost = torch.topk(dmean, 10)
    result['causal_boost_top'] = [(dec(int(t)), round(float(vv), 3))
                                  for t, vv in zip(top_boost.indices.cpu().numpy(), top_boost.values.cpu().numpy())]
    # which TRUE-NEXT tokens break most (summed dCE by true token)
    tn = np.array(true_next_ids); pd = np.array(per_pos_dce)
    order = {}
    for t, d in zip(tn, pd): order[t] = order.get(t, 0.0) + d
    tt = sorted(order.items(), key=lambda kv: -kv[1])[:8]
    result['true_next_break_top'] = [(dec(t), round(float(s), 3)) for t, s in tt]
    # COPY vs BOOST
    if srcmat is not None and src_dlogit:
        result['copy_src_dlogit_mean'] = round(float(np.mean(src_dlogit)), 3)
        result['copy_src_in_top10_frac'] = round(float(np.mean(src_in_top)), 3)
        # source diversity among trigger positions
        src_sig_full = srcmat[seqs, poss]
        result['copy_src_purity'] = round(conc(src_sig_full), 4)
    return result

# =====================================================================================
RESULTS = {}
for (li, h) in HEAD_PATHS:
    name = f"h.L{li}.{h}"
    print(f"CAUSAL {name} ...", flush=True)
    RESULTS[name] = causal('head', (li, h), hnorm[(li, h)], srcmat=hsrc[(li, h)])
    RESULTS[name]['kind'] = 'head'

print("CAUSAL mlp.L1.d2 ...", flush=True)
RESULTS['mlp.L1.d2'] = causal('mlp', MLP_PATH, np.abs(mproj), srcmat=None)
RESULTS['mlp.L1.d2']['kind'] = 'mlp'

json.dump(RESULTS, open(f'{QK}/qk_unsup_verify.json', 'w'), indent=2)
print("\n===== VERIFICATION RESULTS =====", flush=True)
print(json.dumps(RESULTS, indent=2), flush=True)
print("QK UNSUP VERIFY DONE", flush=True)
