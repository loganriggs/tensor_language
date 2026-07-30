"""MULTI-STEP composition discovery in bilin18 (§56 follow-up).

Single-path §56 (qk_unsup_discover/verify) named clean SINGLE heads. Here we ask:
which of those paths COMPOSE into 2-step algorithms -- an upstream path A whose
OUTPUT causally drives a downstream path B's computation (QK-composition = A moves
WHERE B attends; OV/V-composition = A changes WHAT B reads/writes)?

Pipeline:
 (1) STRENGTH (unsupervised, TRAIN FW[0:256,:128]): take the top-24 cleanest HEADS
     from qk_unsup_paths.json (+ whole-MLP layers as feed-forward upstream). For every
     ordered pair (A upstream layer < B downstream layer) mean-ablate A (in-distribution
     per-position mean) and measure the relative change in B's residual contribution
     comp_B = c_proj(yh_B). Rank pairs by this TOTAL dependency. This is the real
     causal-activation signal (no weight-based proxy).
 (2) CHARACTERISE the top pairs: A's held-out trigger, B's held-out trigger, and the
     chain input->output.
 (3) CAUSAL VERIFICATION on held-back FW[448:600,:128], at B's trigger positions:
     (a) DIRECT-EDGE patch A->B (freeze everything except A's contribution into B's
         input) -> dCE with paired SE. Isolates the direct 2-step edge.
     (b) QK-vs-OV: split the edge into q/k-side only vs v-side only; which drives B.
     (c) DEPENDENCE vs CO-OCCURRENCE (the key confound): B's damage when A is present
         minus B's damage when A is ablated. >0 & significant => B genuinely NEEDS A;
         ~0 => A and B merely co-occur on correlated triggers, NOT a real chain.
     (d) SPECIFICITY control: edge-patch a matched CONTROL upstream head into B (~0).

FORWARD copied VERBATIM from qk_bracket_patch.py / qk_unsup_verify.py
(tier2_model.reference_forward): bilin18 two-branch UNNORMALISED pattern (s1*s2),
per-head QK rms_norm THEN RoPE, v-lerp via a.lamb (block-0 v cache), 30*tanh logits.
Extended only with collect / mean-ablate / direct-edge-patch hooks.
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
TRAIN = FINEWEB[0:256, :SEQL].to(DEV)     # discovery / strength slice
HELD = FINEWEB[448:600, :SEQL].to(DEV)    # held-back verification slice
NTRAIN, NHELD = TRAIN.shape[0], HELD.shape[0]
BATCH = 8
KCAUSAL = 200                              # top-K B-trigger positions for causal stats
print(f"bilin18 NL={NL} NH={NH} HD={HD} D={D}; train={NTRAIN} held={NHELD}", flush=True)

_special = {tok.eos_token_id}
for _t in range(V):
    if '�' in tok.decode([_t]): _special.add(_t)
SPECIAL = np.array(sorted(_special))
def dec(t): return repr(tok.decode([int(t)]))

# ---- top-24 cleanest HEADS from the §56 ranking (heads only, file order = desc clean)
RANK = json.load(open(f'{QK}/qk_unsup_paths.json'))['ranking']
TOP_HEADS = [(x['li'], x['h']) for x in RANK if x['kind'] == 'head'][:24]
HEAD_CLEAN = {(x['li'], x['h']): x['cleanliness'] for x in RANK if x['kind'] == 'head'}
maxHL = max(li for li, _ in TOP_HEADS)
# feed-forward upstream candidates: whole-MLP layers strictly below the deepest head
MLP_UP = list(range(0, maxHL))
print("TOP24 heads:", TOP_HEADS, flush=True)
print("MLP upstream layers:", MLP_UP, flush=True)

Wr_all = [m.transformer.h[li].attn.c_proj.weight.view(D, NH, HD) for li in range(NL)]

# =====================================================================================
# FORWARD (VERBATIM core) + hooks: collect head-comp / mean-ablate head|mlp / edge-patch
# =====================================================================================
@torch.no_grad()
def forward(idx, collect_heads=None, ablate=None, yhmeans=None, mlpmeans=None,
            edge=None, collect_patB=None):
    """collect_heads: list (li,h) -> out[('comp',li,h)]=(B,T,D) contribution.
    ablate: ('head',li,h) | ('mlp',li) | list thereof  (mean-ablated).
    edge: dict(la, aA(('head',h)|('mlp',)), lb, hB, mode in {'edge','qk','ov'}) direct patch.
    collect_patB: (lb,hB) -> out[('patB',)] = pattern (B,T,T) for that head.
    """
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    abls = ablate if isinstance(ablate, list) else ([ablate] if ablate is not None else [])
    out = {} if (collect_heads or collect_patB) else None
    delta_A = None                                   # A's above-mean contribution for edge patch
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn
        hcur = F.rms_norm(x, (D,))
        # optional modified input view for the direct edge (B reads x - delta_A)
        emod = (edge is not None and edge['lb'] == li and delta_A is not None)
        if emod:
            hmod = F.rms_norm(x - delta_A, (D,))
        def qkf(lin, h): z = F.rms_norm(lin(h).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        q, k, q2, k2 = qkf(a.c_q, hcur), qkf(a.c_k, hcur), qkf(a.c_q2, hcur), qkf(a.c_k2, hcur)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        if emod:                                      # splice head hB's clean tensors with modded ones
            hB = edge['hB']; mode = edge['mode']
            if mode in ('edge', 'qk'):
                qm, km, q2m, k2m = qkf(a.c_q, hmod), qkf(a.c_k, hmod), qkf(a.c_q2, hmod), qkf(a.c_k2, hmod)
                q = q.clone(); k = k.clone(); q2 = q2.clone(); k2 = k2.clone()
                q[:, :, hB] = qm[:, :, hB]; k[:, :, hB] = km[:, :, hB]
                q2[:, :, hB] = q2m[:, :, hB]; k2[:, :, hB] = k2m[:, :, hB]
            if mode in ('edge', 'ov'):
                vm = a.c_v(hmod).view(B, T, NH, HD); vm = (1-a.lamb)*vm + a.lamb*v1.view_as(vm)
                v = v.clone(); v[:, :, hB] = vm[:, :, hB]
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0)
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        if collect_patB is not None and collect_patB[0] == li:
            out[('patB',)] = pat[:, collect_patB[1]].contiguous()
        if collect_heads:
            for (cli, ch) in collect_heads:
                if cli == li:
                    out[('comp', cli, ch)] = torch.einsum('btc,oc->bto', yh4[:, :, ch], Wr_all[li][:, ch])
        # cache A's above-mean contribution for the edge patch (A is a HEAD here)
        if edge is not None and edge['la'] == li and edge['aA'][0] == 'head':
            Ah = edge['aA'][1]
            compA = torch.einsum('btc,oc->bto', yh4[:, :, Ah], Wr_all[li][:, Ah])
            meanA = torch.einsum('tc,oc->to', yhmeans[li][:, Ah], Wr_all[li][:, Ah])
            delta_A = compA - meanA.unsqueeze(0)
        # mean-ablate heads
        for ab in abls:
            if ab[0] == 'head' and ab[1] == li:
                yh4 = yh4.clone(); yh4[:, :, ab[2]] = yhmeans[li][:, ab[2]].unsqueeze(0)
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if edge is not None and edge['la'] == li and edge['aA'][0] == 'mlp':
            delta_A = mo - mlpmeans[li].unsqueeze(0)
        for ab in abls:
            if ab[0] == 'mlp' and ab[1] == li:
                mo = mlpmeans[li].unsqueeze(0).expand(B, -1, -1)
        x = x + mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    return (logits, out) if out is not None else logits

# =====================================================================================
# per-position means (yh all heads, mo all layers) over a slice
# =====================================================================================
@torch.no_grad()
def collect_means(DATA):
    n = DATA.shape[0]
    YH = {li: torch.zeros(SEQL, NH, HD, device=DEV) for li in range(NL)}
    MO = {li: torch.zeros(SEQL, D, device=DEV) for li in range(NL)}
    for i in range(0, n, BATCH):
        idx = DATA[i:i+BATCH]; B, T = idx.shape
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
            YH[li] += yh4.sum(0)
            x = x + a.c_proj(yh4.reshape(B, T, -1))
            mo = blk.mlp(F.rms_norm(x, (D,))); MO[li] += mo.sum(0); x = x + mo
    return {li: YH[li]/n for li in range(NL)}, {li: MO[li]/n for li in range(NL)}

print("means: TRAIN ...", flush=True)
YHM_TR, MOM_TR = collect_means(TRAIN)

# =====================================================================================
# (1) STRENGTH matrix: mean-ablate each upstream A, measure rel change in comp_B (TRAIN)
# =====================================================================================
UP = [('head', li, h) for (li, h) in TOP_HEADS] + [('mlp', li, None) for li in MLP_UP]
def up_layer(a): return a[1]
def up_name(a): return f"h.L{a[1]}.{a[2]}" if a[0] == 'head' else f"mlp.L{a[1]}"

# accumulators: dep_num[A][B]=sum||dcomp_B||, dep_den[B]=sum||comp_B_base||
dep_num = {up_name(a): {(li, h): 0.0 for (li, h) in TOP_HEADS if li > up_layer(a)} for a in UP}
comp_base_norm = {(li, h): 0.0 for (li, h) in TOP_HEADS}

print("STRENGTH pass (TRAIN) ...", flush=True)
for i in range(0, NTRAIN, BATCH):
    idx = TRAIN[i:i+BATCH]
    _, base = forward(idx, collect_heads=TOP_HEADS)
    cb = {p: base[('comp',)+p] for p in TOP_HEADS}       # (B,T,D)
    for p in TOP_HEADS:
        comp_base_norm[p] += float(cb[p].norm(dim=-1).sum())
    for a in UP:
        downs = [p for p in TOP_HEADS if p[0] > up_layer(a)]
        if not downs: continue
        abl = ('head', a[1], a[2]) if a[0] == 'head' else ('mlp', a[1])
        _, ao = forward(idx, collect_heads=downs, ablate=abl, yhmeans=YHM_TR, mlpmeans=MOM_TR)
        for p in downs:
            d = (ao[('comp',)+p] - cb[p]).norm(dim=-1).sum()
            dep_num[up_name(a)][p] += float(d)

# relative dependency
pairs = []
for a in UP:
    an = up_name(a)
    for p in TOP_HEADS:
        if p[0] <= up_layer(a): continue
        rel = dep_num[an][p] / (comp_base_norm[p] + 1e-9)
        pairs.append({'A': an, 'B': f"h.L{p[0]}.{p[1]}", 'A_layer': up_layer(a), 'B_layer': p[0],
                      'rel_dB': round(rel, 4), 'A_kind': a[0]})
pairs.sort(key=lambda r: -r['rel_dB'])
print("STRENGTH done. top pairs:", flush=True)
for r in pairs[:15]:
    print(f"  {r['A']:10s} -> {r['B']:8s}  rel_dB={r['rel_dB']:.3f}", flush=True)

# =====================================================================================
# HELD-OUT means + B-trigger collection (comp norm, current token, attended source)
# =====================================================================================
print("means: HELD ...", flush=True)
YHM_H, MOM_H = collect_means(HELD)

held_np = HELD.cpu().numpy()
pos_t = np.tile(np.arange(SEQL), NHELD).reshape(NHELD, SEQL)
is_special = np.isin(held_np, SPECIAL)
valid_next = pos_t < (SEQL - 1)

# collect comp-norm + attended-source for every top head over held-out
HELD_HEADS = TOP_HEADS
hnorm = {p: np.zeros((NHELD, SEQL), np.float32) for p in HELD_HEADS}
hsrc = {p: np.zeros((NHELD, SEQL), np.int64) for p in HELD_HEADS}

@torch.no_grad()
def held_collect():
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
            srcpos = pat.abs().argmax(-1)                # (B,NH,T)
            for (tli, th) in HELD_HEADS:
                if tli == li:
                    comp = torch.einsum('btc,oc->bto', yh4[:, :, th], Wr_all[li][:, th])
                    hnorm[(tli, th)][i:i+B] = comp.norm(dim=-1).cpu().numpy()
                    hsrc[(tli, th)][i:i+B] = torch.gather(idx, 1, srcpos[:, th, :]).cpu().numpy()
            x = x + a.c_proj(yh4.reshape(B, T, -1)); x = x + blk.mlp(F.rms_norm(x, (D,)))
print("HELD collect ...", flush=True)
held_collect()

def signature(act, K):
    a = act.copy().reshape(-1)
    bad = (pos_t == 0) | is_special | ~valid_next
    a[bad.reshape(-1)] = -1e30
    tk = np.argpartition(a, -K)[-K:]; tk = tk[np.argsort(-a[tk])]
    return tk // SEQL, tk % SEQL
def topcount(ids, n=6):
    v, c = np.unique(np.asarray(ids), return_counts=True); o = np.argsort(-c)[:n]
    return [(dec(v[i]), int(c[i])) for i in o]
def conc(ids):
    ids = np.asarray(ids)
    if len(ids) <= 1: return 0.0
    _, c = np.unique(ids, return_counts=True); p = c/c.sum()
    return float(1 - (-(p*np.log(p)).sum())/math.log(len(ids)))
def pse(v): return float(np.std(v, ddof=1)/math.sqrt(len(v))) if len(v) > 1 else float('nan')

def head_trigger(p):
    s, q = signature(hnorm[p], 50)
    cur = held_np[s, q]; src = hsrc[p][s, q]
    return {'cur_top': topcount(cur), 'cur_purity': round(conc(cur), 3),
            'src_top': topcount(src), 'src_purity': round(conc(src), 3)}

# =====================================================================================
# CAUSAL VERIFICATION of a pair at B's trigger positions
# =====================================================================================
def ce_at(logits, rows, ss, ps):
    ces = []
    for r, s, p in zip(rows, ss, ps):
        tgt = int(held_np[s, p+1])
        ces.append(F.cross_entropy(logits[r, p].unsqueeze(0), torch.tensor([tgt], device=DEV)).item())
    return np.array(ces)

@torch.no_grad()
def verify_pair(A, B, ctrlA=None):
    """A,B are ('head',li,h) or A can be ('mlp',li,None). B is always ('head',li,h)."""
    la, Ah = A[1], A[2]; lb, Bh = B[1], B[2]
    aA = ('head', Ah) if A[0] == 'head' else ('mlp',)
    Bkey = (lb, Bh)
    seqs, poss = signature(hnorm[Bkey], KCAUSAL)          # B trigger positions
    uniq = np.unique(seqs)
    # edge configs
    def edge(mode): return {'la': la, 'aA': aA, 'lb': lb, 'hB': Bh, 'mode': mode}
    ablA = ('head', la, Ah) if A[0] == 'head' else ('mlp', la)
    ablB = ('head', lb, Bh)
    # accumulate per-position CE across batches
    C = {k: [] for k in ['clean', 'edge', 'qk', 'ov', 'ablB', 'ablA', 'ablAB', 'ctrl']}
    dl_edge = torch.zeros(V, device=DEV); ndl = 0
    pat_num = 0.0; pat_den = 0.0; yh_change_num = 0.0; yh_den = 0.0
    tn_ids, pd_edge = [], []
    Bcfg = None
    for i in range(0, len(uniq), BATCH):
        sb = uniq[i:i+BATCH]; idx = HELD[sb]
        s2l = {int(s): j for j, s in enumerate(sb)}
        sel = [(s, p) for s, p in zip(seqs, poss) if int(s) in s2l]
        if not sel: continue
        js = [s2l[int(s)] for s, _ in sel]; ps = [p for _, p in sel]
        def gather(L): return L[js, ps]                   # rows aligned to sel
        Lc = forward(idx).float()
        Le = forward(idx, edge=edge('edge'), yhmeans=YHM_H, mlpmeans=MOM_H).float()
        Lqk = forward(idx, edge=edge('qk'), yhmeans=YHM_H, mlpmeans=MOM_H).float()
        Lov = forward(idx, edge=edge('ov'), yhmeans=YHM_H, mlpmeans=MOM_H).float()
        LaB = forward(idx, ablate=ablB, yhmeans=YHM_H).float()
        LaA = forward(idx, ablate=ablA, yhmeans=YHM_H, mlpmeans=MOM_H).float()
        LaAB = forward(idx, ablate=[ablA, ablB], yhmeans=YHM_H, mlpmeans=MOM_H).float()
        Lct = None
        if ctrlA is not None:
            cA = ('head', ctrlA[2]) if ctrlA[0] == 'head' else ('mlp',)
            Lct = forward(idx, edge={'la': ctrlA[1], 'aA': cA, 'lb': lb, 'hB': Bh, 'mode': 'edge'},
                          yhmeans=YHM_H, mlpmeans=MOM_H).float()
        for name, L in [('clean', Lc), ('edge', Le), ('qk', Lqk), ('ov', Lov),
                        ('ablB', LaB), ('ablA', LaA), ('ablAB', LaAB)] + ([('ctrl', Lct)] if Lct is not None else []):
            C[name].append(ce_at(L, js, [s for s, _ in sel], ps))
        # delta-logit (clean - edge) = what the A->B edge supports
        for jj, (s, p) in enumerate(sel):
            dl_edge += (Lc[js[jj], ps[jj]] - Le[js[jj], ps[jj]]); ndl += 1
            tn_ids.append(int(held_np[s, p+1]))
            pd_edge.append(float(ce_at(Le[js[jj]:js[jj]+1], [0], [s], [ps[jj]])[0] - ce_at(Lc[js[jj]:js[jj]+1], [0], [s], [ps[jj]])[0]))
        # pattern change of B under edge (QK signal) vs value-only yh change
        _, oc = forward(idx, collect_patB=(lb, Bh)); patc = oc[('patB',)]
        _, oe = forward(idx, edge=edge('edge'), yhmeans=YHM_H, mlpmeans=MOM_H, collect_patB=(lb, Bh)); pate = oe[('patB',)]
        for jj, p in zip(js, ps):
            pc = patc[jj, p]; pe = pate[jj, p]
            pat_num += float((pe - pc).norm()); pat_den += float(pc.norm() + 1e-9)
    def arr(k): return np.concatenate(C[k]) if C[k] else np.array([])
    clean, edge_ce, qk_ce, ov_ce = arr('clean'), arr('edge'), arr('qk'), arr('ov')
    ablB_ce, ablA_ce, ablAB_ce = arr('ablB'), arr('ablA'), arr('ablAB')
    dEdge = edge_ce - clean; dQK = qk_ce - clean; dOV = ov_ce - clean
    # dependence: damage-from-B when A present vs when A ablated (paired interaction)
    dmgB_present = ablB_ce - clean
    dmgB_Aabl = ablAB_ce - ablA_ce
    depend = dmgB_present - dmgB_Aabl
    out = {
        'A': up_name(A), 'B': f"h.L{lb}.{Bh}", 'n_pos': int(len(clean)),
        'A_trigger': head_trigger((la, Ah)) if A[0] == 'head' else {'note': 'whole-MLP layer'},
        'B_trigger': head_trigger(Bkey),
        # (a) direct edge
        'edge_dCE_mean': round(float(dEdge.mean()), 4), 'edge_dCE_SE': round(pse(dEdge), 4),
        'edge_dCE_frac_pos': round(float((dEdge > 0).mean()), 3),
        # (b) QK vs OV
        'qk_dCE_mean': round(float(dQK.mean()), 4), 'ov_dCE_mean': round(float(dOV.mean()), 4),
        'B_pattern_rel_change_under_edge': round(pat_num/(pat_den+1e-9), 4),
        # (c) dependence vs co-occurrence
        'dmgB_when_A_present_mean': round(float(dmgB_present.mean()), 4),
        'dmgB_when_A_ablated_mean': round(float(dmgB_Aabl.mean()), 4),
        'dependence_mean': round(float(depend.mean()), 4), 'dependence_SE': round(pse(depend), 4),
        'dependence_z': round(float(depend.mean()/(pse(depend)+1e-12)), 2),
    }
    # composition type verdict
    if out['B_pattern_rel_change_under_edge'] > 0.15 and abs(out['qk_dCE_mean']) >= abs(out['ov_dCE_mean']):
        out['composition_type'] = 'QK (A moves WHERE B attends)'
    elif abs(out['ov_dCE_mean']) > abs(out['qk_dCE_mean']):
        out['composition_type'] = 'OV/V (A changes WHAT B reads/writes)'
    else:
        out['composition_type'] = 'mixed/weak'
    # what edge supports
    dmean = dl_edge / max(ndl, 1)
    tb = torch.topk(dmean, 8)
    out['edge_supports_top'] = [(dec(int(t)), round(float(vv), 3)) for t, vv in zip(tb.indices.cpu().numpy(), tb.values.cpu().numpy())]
    tn = np.array(tn_ids); pdv = np.array(pd_edge); order = {}
    for t, d in zip(tn, pdv): order[t] = order.get(t, 0.0) + d
    out['edge_true_next_break_top'] = [(dec(t), round(float(s), 3)) for t, s in sorted(order.items(), key=lambda kv: -kv[1])[:6]]
    out['edge_z'] = round(float(out['edge_dCE_mean']/(out['edge_dCE_SE']+1e-12)), 2)
    if ctrlA is not None:
        ctrl = arr('ctrl'); dCtrl = ctrl - clean
        out['ctrl_upstream'] = up_name(ctrlA)
        out['ctrl_edge_dCE_mean'] = round(float(dCtrl.mean()), 4); out['ctrl_edge_dCE_SE'] = round(pse(dCtrl), 4)
        out['edge_vs_control_ratio'] = round(float(out['edge_dCE_mean']/(abs(out['ctrl_edge_dCE_mean'])+1e-9)), 2)
    # VERDICT: composition <=> cutting the DIRECT A->B edge causally hurts (edge_z>=2.5,
    # edge_dCE>0.03) AND is specific to A (>2x its matched-control edge, when available).
    # co-occurrence <=> cutting the edge does ~nothing (B fires alongside A but doesn't read it).
    spec_ok = ('edge_vs_control_ratio' not in out) or (out['edge_vs_control_ratio'] >= 2.0)
    genuine = (out['edge_z'] >= 2.5 and out['edge_dCE_mean'] > 0.03 and spec_ok)
    if genuine:
        # steering (A redirects WHERE B attends; dependence not required) vs enabling (B needs A)
        steer = out['B_pattern_rel_change_under_edge'] > 0.3
        out['verdict'] = 'GENUINE 2-step composition' + (' (steering: A redirects B)' if steer else '')
    elif out['edge_dCE_mean'] < 0.02 or out['edge_z'] < 2.0:
        out['verdict'] = 'co-occurrence / not a chain (cutting A->B edge is inert)'
    else:
        out['verdict'] = 'weak/ambiguous'
    return out

# =====================================================================================
# pick pairs to verify: top head->head pairs (dedup by exact pair), + best ff->head
# =====================================================================================
UPMAP = {up_name(a): a for a in UP}
def to_tuple(name):
    if name.startswith('mlp.L'): return ('mlp', int(name.split('L')[1]), None)
    _, l, h = name.split('.'); return ('head', int(l[1:]), int(h))

head_pairs = [r for r in pairs if r['A_kind'] == 'head']
ff_pairs = [r for r in pairs if r['A_kind'] == 'mlp']
VERIFY = head_pairs[:5] + ff_pairs[:2]     # ~top head->head + strongest ff->head

def matched_control(A, B):
    """a top-24 head at A's layer (else nearest lower layer) with LOW rel_dB into B."""
    la, lb, Bh = A[1], B[1], B[2]
    cands = [(li, h) for (li, h) in TOP_HEADS if li < lb and (li, h) != (la, A[2])]
    if not cands: return None
    # prefer same layer as A
    same = [c for c in cands if c[0] == la]
    pool = same if same else cands
    # lowest dependency into B
    def rel(c):
        an = f"h.L{c[0]}.{c[1]}"; row = next((r for r in pairs if r['A'] == an and r['B'] == f"h.L{lb}.{Bh}"), None)
        return row['rel_dB'] if row else 0.0
    pool.sort(key=rel)
    c = pool[0]; return ('head', c[0], c[1])

RESULTS = {'meta': {'top24_heads': [f"h.L{l}.{h}" for l, h in TOP_HEADS],
                    'mlp_upstream_layers': MLP_UP,
                    'train_slice': 'FW[0:256,:128]', 'held_slice': 'FW[448:600,:128]',
                    'strength_metric': 'rel change in comp_B when A mean-ablated (TOTAL, all paths)',
                    'edge_metric': 'direct A->B edge patch (B input = residual - A above-mean contribution)'},
           'strength_ranking_top25': pairs[:25],
           'verified': []}

seen = set()
for r in VERIFY:
    A = to_tuple(r['A']); B = to_tuple(r['B'])
    key = (r['A'], r['B'])
    if key in seen: continue
    seen.add(key)
    ctrl = matched_control(A, B) if A[0] == 'head' else None
    print(f"VERIFY {r['A']} -> {r['B']} (rel_dB={r['rel_dB']}) ctrl={ctrl} ...", flush=True)
    res = verify_pair(A, B, ctrlA=ctrl)
    res['strength_rel_dB'] = r['rel_dB']
    RESULTS['verified'].append(res)

json.dump(RESULTS, open(f'{QK}/qk_unsup_compose.json', 'w'), indent=2)
print("\n===== COMPOSITION RESULTS =====", flush=True)
print(json.dumps(RESULTS, indent=2), flush=True)
print("QK UNSUP COMPOSE DONE", flush=True)
