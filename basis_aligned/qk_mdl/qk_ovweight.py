"""WHY DOES FVU DECOUPLE FROM delta-CE? (Logan 2026-07-22) — weight-only metric ladder.

Logan's hypothesis: the output-value circuit reads the attention pattern, so score errors should be
weighted by what the output-value side cares about. All quantities here are WEIGHT-ONLY (no data):

  ladder of structural metrics per arm, from naive to OV-composed —
    m_fac    : plain factor-table FVU (what we already report)
    m_score  : FVU at the score level, per branch (pre-RoPE q_hat k_hat^T on a 4096-token sample)
    m_pat    : FVU of the PATTERN (s1*s2 product of both branches) — the bilinear product weighting
    m_pat_ov : pattern FVU with key/value columns weighted by w_j = ||W_o^h W_v^h e_hat_j||, the
               output-value importance of attending to token j (Logan's "V Embedding composition")
    m_pat_rope    : pattern FVU WITH ROTARY POSITION applied — evaluated at relative offsets
                    delta in a log grid over [0, 511], each offset weighted by its pair count
                    under the causal mask at T=512 (Logan: "include rope/pos emb in the ladder")
    m_pat_rope_ov : the rotary rung with the OV column weighting on top
    m_pat_gram    : pattern error measured THROUGH the full OV map — ||dP @ U_h||^2/||P @ U_h||^2
                    with U_h[j] = W_o^h W_v^h e_hat_j in R^D. Unlike the norm weighting this
                    respects the OV NULL SPACE and cross-token cancellation (errors on tokens
                    whose OV outputs cancel or vanish cost nothing). Logan Q3, tick 156.
    m_pat_rope_gram : the OV-Gram rung with rotary position as well
    m_pat_freq    : pattern FVU with rows AND columns weighted by empirical unigram frequency
                    (FineWeb counts; the only data-informed rung, labeled as such) — tests whether
                    the uniform-vocabulary sample is what distorts the weighted rungs.

  DIAGNOSTICS per arm (Logan: metrics that say WHY a weighted rung disagrees, so a metric flip
  can be diagnosed rather than trusted blindly):
    diag_cancel_err : ||dP U||^2 / sum_j dP_j^2 ||u_j||^2 — how much of this arm's error mass
                      CANCELS through OV (1 = none, <1 = net cancellation).
    diag_cancel_sig : the same ratio for the TRUE pattern (the signal's own cancellation floor).
    diag_align      : Pearson correlation between per-column error mass and the OV weight w_j^2 —
                      does the arm put its error where OV cares (positive) or away from it?

  then Spearman rank-correlate each metric with the big-audit held-out delta-CE across arms.
  If m_pat_ov predicts delta-CE where m_fac fails, the decoupling is explained weight-only.

Requires qk_audit_big.json + qk_dict_l0_seed0.pt (run qk_audit_big.py first). Writes qk_ovweight.json.
"""
import json
import sys
import torch
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs
from tier2_folding import branch_factors
from qk_sae_lib import train_dict, encode_token, encode_omp, kmeans, arm_svd

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
BRANCHES = (('q1', 'k1'), ('q2', 'k2'))
M = 4096                                   # token sample for score/pattern metrics

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']

TAB = {}
for br, (qn, kn) in enumerate(BRANCHES, start=1):
    qh, kh = branch_factors(m, br)
    TAB[qn], TAB[kn] = qh.float().to(DEV), kh.float().to(DEV)
HB = [(h, qn, kn) for h in range(NH) for (qn, kn) in BRANCHES]


def rows(h, qn, kn):
    return torch.cat([TAB[qn][:, h], TAB[kn][:, h]], 1)


# --- output-value importance per (token, head): w = ||W_o^h (W_v^h e_hat)|| (weight-only) ---
with torch.no_grad():
    a = m.transformer.h[0].attn
    E = torch.nn.functional.rms_norm(m.transformer.wte.weight.detach().float(), (D,))
    Vv = a.c_v(E).view(V, NH, HD)                                   # value vectors per head
    Wo = a.c_proj.weight.detach().float().view(D, NH, HD)           # output projection per head
    W_IMP = torch.stack([(Vv[:, h] @ Wo[:, h].T).norm(dim=1) for h in range(NH)], 1)  # (V, NH)
print('OV importance: per-head mean', [round(float(W_IMP[:, h].mean()), 2) for h in range(NH)], flush=True)

g = torch.Generator().manual_seed(0)
SAMP = torch.randperm(V, generator=g)[:M].to(DEV)
WS = W_IMP[SAMP]                                                    # (M, NH)

# rotary tables for the positional rungs (exact fp32; T=512, the frozen regime)
T_POS = 512
M2 = 2048                                                           # smaller sample for rope rungs
SAMP2 = SAMP[:M2]
WS2 = W_IMP[SAMP2]
with torch.no_grad():                                               # OV output vectors on the sample
    US = [Vv[SAMP2, h] @ Wo[:, h].T for h in range(NH)]             # each (M2, D)

import numpy as np                                                  # unigram frequency weights
_cnt = torch.bincount(torch.from_numpy(
    np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64)).flatten(),
    minlength=V).float().to(DEV)
FRQ = (_cnt[SAMP2] + 0.5)
FRQ = FRQ / FRQ.mean()
FR2 = FRQ[:, None] * FRQ[None, :]                                   # (M2, M2) row*col weights
_inv = 1.0 / (10000 ** (torch.arange(0, HD, 2, dtype=torch.float32) / HD))
DELTAS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 511)
DW = torch.tensor([(T_POS - d) / T_POS for d in DELTAS], device=DEV)   # pair-count weight per offset
COSD = torch.stack([torch.cos(_inv * d) for d in DELTAS]).to(DEV)      # (n_delta, HD/2)
SIND = torch.stack([torch.sin(_inv * d) for d in DELTAS]).to(DEV)


def branch_scores(tabs, h, qn, kn):
    """(M, M) pre-RoPE score sample for one head-branch from factor tables."""
    return tabs[qn][SAMP, h] @ tabs[kn][SAMP, h].T / HD


def branch_scores_rope(tabs, h, qn, kn, di):
    """(M2, M2) score sample at relative offset DELTAS[di], rotary applied (fold convention)."""
    d = HD // 2
    Fq, Fk = tabs[qn][SAMP2, h], tabs[kn][SAMP2, h]
    qa, qb = Fq[:, :d], Fq[:, d:]
    ka, kb = Fk[:, :d], Fk[:, d:]
    c, s = COSD[di], SIND[di]
    return ((qa * c) @ ka.T + (qb * c) @ kb.T + (qb * s) @ ka.T - (qa * s) @ kb.T) / HD


def metrics(recs):
    """recs: list over HB of (V,256) reconstructions. Returns the metric ladder (energy-weighted)."""
    hat = {n: TAB[n].clone() for n in NAMES}
    for (h, qn, kn), rec in zip(HB, recs):
        hat[qn][:, h] = rec[:, :HD]
        hat[kn][:, h] = rec[:, HD:]
    KEYS = ('fac', 'score', 'pat', 'pat_ov', 'pat_rope', 'pat_rope_ov', 'pat_gram',
            'pat_rope_gram', 'pat_freq', 'pat_ctx')
    num = {k: 0.0 for k in KEYS}
    den = {k: 0.0 for k in KEYS}
    dg = {'gram_err': 0.0, 'diag_err': 0.0, 'gram_sig': 0.0, 'diag_sig': 0.0, 'align': []}
    for (h, qn, kn), rec in zip(HB, recs):
        X = rows(h, qn, kn)
        num['fac'] += float(((rec - X) ** 2).sum())
        den['fac'] += float(((X - X.mean(0)) ** 2).sum())
    for h in range(NH):
        S1 = branch_scores(TAB, h, 'q1', 'k1')
        S2 = branch_scores(TAB, h, 'q2', 'k2')
        S1h = branch_scores(hat, h, 'q1', 'k1')
        S2h = branch_scores(hat, h, 'q2', 'k2')
        for S, Sh in ((S1, S1h), (S2, S2h)):
            num['score'] += float(((Sh - S) ** 2).sum())
            den['score'] += float((S ** 2).sum())
        P, Ph = S1 * S2, S1h * S2h
        dP = Ph - P
        num['pat'] += float((dP ** 2).sum())
        den['pat'] += float((P ** 2).sum())
        w2 = (WS[:, h] ** 2)[None, :]                               # weight key/value columns
        num['pat_ov'] += float(((dP ** 2) * w2).sum())
        den['pat_ov'] += float(((P ** 2) * w2).sum())
        w2r = (WS2[:, h] ** 2)[None, :]
        P2s, dP2s = P[:M2, :M2], dP[:M2, :M2]                       # SAMP2 sub-block, pre-rotary
        g_err = float((dP2s @ US[h]).pow(2).sum())                  # error through the OV map
        g_sig = float((P2s @ US[h]).pow(2).sum())
        num['pat_gram'] += g_err
        den['pat_gram'] += g_sig
        num['pat_freq'] += float(((dP2s ** 2) * FR2).sum())
        den['pat_freq'] += float(((P2s ** 2) * FR2).sum())
        w2s = (WS2[:, h] ** 2)[None, :]
        dg['gram_err'] += g_err
        dg['diag_err'] += float(((dP2s ** 2) * w2s).sum())          # no-cross-term version
        dg['gram_sig'] += g_sig
        dg['diag_sig'] += float(((P2s ** 2) * w2s).sum())
        ej = dP2s.pow(2).sum(0)                                     # per-column error mass
        wj = w2s.squeeze(0)
        ejc, wjc = ej - ej.mean(), wj - wj.mean()
        dg['align'].append(float((ejc * wjc).sum() /
                                 (ejc.norm() * wjc.norm() + 1e-12)))
        # context-expected OV error (ov_metric_explainer.md eq. †): cancellation credited only
        # to the mean component (T^2 term); scatter charged diagonally (T term). i.i.d. unigram.
        qp = FRQ / FRQ.sum()
        T_CTX = 512.0
        for mat, side in ((dP2s, 'pat_ctx_n'), (P2s, 'pat_ctx_d')):
            mu = (mat * qp[None, :]) @ US[h]                        # (M2, D) mean error vector
            mu2 = mu.pow(2).sum(1)
            s_ = (mat.pow(2) * (qp * wj)[None, :]).sum(1)           # wj = ||u_j||^2 (per column)
            val = float((qp * (T_CTX * (s_ - mu2).clamp_min(0) + T_CTX * T_CTX * mu2)).sum())
            if side == 'pat_ctx_n':
                num['pat_ctx'] += val
            else:
                den['pat_ctx'] += val
        for di in range(len(DELTAS)):                               # rotary rungs (Logan)
            S1r = branch_scores_rope(TAB, h, 'q1', 'k1', di)
            S2r = branch_scores_rope(TAB, h, 'q2', 'k2', di)
            Pr = S1r * S2r
            dPr = branch_scores_rope(hat, h, 'q1', 'k1', di) * branch_scores_rope(hat, h, 'q2', 'k2', di) - Pr
            wd = float(DW[di])
            num['pat_rope'] += wd * float((dPr ** 2).sum())
            den['pat_rope'] += wd * float((Pr ** 2).sum())
            num['pat_rope_ov'] += wd * float(((dPr ** 2) * w2r).sum())
            den['pat_rope_ov'] += wd * float(((Pr ** 2) * w2r).sum())
            num['pat_rope_gram'] += wd * float((dPr @ US[h]).pow(2).sum())
            den['pat_rope_gram'] += wd * float((Pr @ US[h]).pow(2).sum())
    out = {k: round(num[k] / den[k], 5) for k in num}
    out['diag_cancel_err'] = round(dg['gram_err'] / (dg['diag_err'] + 1e-12), 4)
    out['diag_cancel_sig'] = round(dg['gram_sig'] / (dg['diag_sig'] + 1e-12), 4)
    out['diag_align'] = round(sum(dg['align']) / len(dg['align']), 4)
    return out


NAMES = ('q1', 'k1', 'q2', 'k2')

# --- arms (deterministic seed-0 refits, matching qk_audit_big names) ---
blob = torch.load(f'{QK}/qk_dict_l0_seed0.pt', map_location=DEV)
FITS = [(blob[f'Dn{i}'], blob[f'b{i}'], blob[f'We{i}']) for i in range(len(HB))]


def arm_recs(name):
    if name.startswith('svd rank'):
        r = int(name.split()[-1])
        return [arm_svd(rows(*hb), r) for hb in HB]
    if name == 'dict n=1024 k=8 token-linear':
        return [encode_token(rows(*hb), *f, 8) for f, hb in zip(FITS, HB)]
    if name == 'dict n=1024 k=8 token-OMP/LS':
        return [encode_omp(rows(*hb), f[0], f[1], 8) for f, hb in zip(FITS, HB)]
    if name == 'merge K=2048 per-head-branch':
        out = []
        for bi, hb in enumerate(HB):
            X = rows(*hb)
            assign, C = kmeans(X, 2048, seed=bi)
            out.append(C[assign])
        return out
    if name.startswith('two-stage'):
        out = []
        for bi, hb in enumerate(HB):
            X = rows(*hb)
            assign, C = kmeans(X, 2048, seed=bi)
            Dn, b, We = train_dict(C, 512, 8, seed=0, steps=2000)
            out.append(encode_omp(C, Dn, b, 8)[assign])
        return out
    raise ValueError(name)


big = json.load(open(f'{QK}/qk_audit_big.json'))['arms']
ARMS = [k for k in big if k != 'exact fold']
res = {'arms': {}}
for name in ARMS:
    mt = metrics(arm_recs(name))
    res['arms'][name] = {**mt, 'dce_pile': big[name]['dce_pile'], 'dce_fw': big[name]['dce_fw']}
    print(f'{name:46s} fac {mt["fac"]:.3f}  score {mt["score"]:.3f}  pat {mt["pat"]:.3f}  '
          f'pat_ov {mt["pat_ov"]:.3f}  rope {mt["pat_rope"]:.3f}  rope_ov {mt["pat_rope_ov"]:.3f}  '
          f'gram {mt["pat_gram"]:.3f}  rope_gram {mt["pat_rope_gram"]:.3f}  '
          f'freq {mt["pat_freq"]:.3f}  ctx {mt["pat_ctx"]:.3f}  cancel {mt["diag_cancel_err"]:.2f}'
          f'/{mt["diag_cancel_sig"]:.2f}  align {mt["diag_align"]:+.2f}  '
          f'| dCE fw {big[name]["dce_fw"]:+.4f}', flush=True)
    json.dump(res, open(f'{QK}/qk_ovweight.json', 'w'), indent=2)


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        for r, i in enumerate(order):
            rk[i] = float(r)
        return rk
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy + 1e-12)


for tgt in ('dce_pile', 'dce_fw'):
    corr = {k: round(spearman([res['arms'][a][k] for a in ARMS],
                              [res['arms'][a][tgt] for a in ARMS]), 3)
            for k in ('fac', 'score', 'pat', 'pat_ov', 'pat_rope', 'pat_rope_ov',
                      'pat_gram', 'pat_rope_gram', 'pat_freq', 'pat_ctx')}
    res[f'spearman_vs_{tgt}'] = corr
    print(f'Spearman vs {tgt}: {corr}', flush=True)

json.dump(res, open(f'{QK}/qk_ovweight.json', 'w'), indent=2)
print('wrote qk_ovweight.json', flush=True)
