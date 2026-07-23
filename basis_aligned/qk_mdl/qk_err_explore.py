"""ERROR EXPLORATION (tick 164, Logan: "what is the residual/error specifically? ...top 100
datapoints with highest error, any commonality? ...lots of analysis, then consider solutions").

Target: the MOST COMPRESSED dictionary — n=256 atoms, k=4, 183.4 Mbit (2.47% of raw),
OV-context-trained (frontier arm, expected dCE ~ +0.0073), with the plain-MSE dictionary at
the same budget as a contrast throughout.

Analyses (all written to qk_err_explore.md + arrays to qk_err_explore.pt):
  A. Per-prediction delta-CE over the 307k held-out FineWeb predictions:
     concentration (what share of the total loss comes from the top 0.1/1/10% positions),
     fraction of positions the dictionary IMPROVES, ctx-vs-MSE per-position correlation
     (intrinsic hard positions vs objective-driven), top-100 worst predictions DECODED with
     context, commonality stats on the top-1000 vs a random-1000 control: target frequency,
     position-in-sequence, repeat/induction structure (target seen earlier in context,
     bigram repeats, distance to last occurrence), document concentration.
  B. Per-head attribution: audit with ONLY head h compressed (others exact), 9 audits.
  C. Weight-space delivered error (eq. dagger, Delta=0) over the FULL vocabulary:
     which query tokens and which key tokens carry the error, decoded top-50 each,
     error contribution vs token-frequency decile.
  D. Factor-space residuals: per-token relative row error vs frequency decile; residual
     spectrum per head-branch (is what's left low-rank?); q-half vs k-half energy split.
"""
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, reference_forward
from tier2_folding import branch_factors, scores_from_factors
from qk_sae_lib import train_dict, encode_token

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
NAMES = ('q1', 'k1', 'q2', 'k2')
BRANCHES = (('q1', 'k1'), ('q2', 'k2'))
N_DICT, K_DICT = 256, 4
CTX_M, T_CTX, CTX_STEPS, CTX_LR = 1024, 512.0, 1500, 3e-4

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
NHB = NH * 2
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
NSEQ, SEQ = FINEWEB.shape[0], FINEWEB.shape[1] - 1          # 600, 512

TAB = {}
for br, (qn, kn) in enumerate(BRANCHES, start=1):
    qh, kh = branch_factors(m, br)
    TAB[qn], TAB[kn] = qh.float().to(DEV), kh.float().to(DEV)
HB = [(h, qn, kn) for h in range(NH) for (qn, kn) in BRANCHES]

with torch.no_grad():
    a0 = m.transformer.h[0].attn
    E = F.rms_norm(m.transformer.wte.weight.detach().float(), (D,))
    Vv = a0.c_v(E).view(V, NH, HD)
    Wo = a0.c_proj.weight.detach().float().view(D, NH, HD)
QFULL = (torch.bincount(FINEWEB.flatten(), minlength=V).float() + 0.5).to(DEV)
QP = QFULL / QFULL.sum()
FREQ_RANK = torch.zeros(V, dtype=torch.long, device=DEV)
FREQ_RANK[QFULL.argsort(descending=True)] = torch.arange(V, device=DEV)

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('gpt2')


def dec(ids):
    return tok.decode([i for i in ids]).replace('\n', '\\n')


def rows(h, qn, kn):
    return torch.cat([TAB[qn][:, h], TAB[kn][:, h]], 1)


def unit_rms(t):
    return t * (t.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt())


def tables_from(recs_by_hb):
    """recs_by_hb: dict hb_index -> (V, 256) reconstruction; missing head-branches stay exact."""
    out = {n: TAB[n].clone() for n in NAMES}
    for bi, rec in recs_by_hb.items():
        h, qn, kn = HB[bi]
        out[qn][:, h] = rec[:, :HD]
        out[kn][:, h] = rec[:, HD:]
    return {n: unit_rms(out[n]) for n in NAMES}


@torch.no_grad()
def audit_pos(tabs, batch=4):
    """Per-position CE, (NSEQ, SEQ)."""
    out = []
    for i in range(0, len(FINEWEB), batch):
        b = FINEWEB[i:i + batch].to(DEV)
        idx = b[:, :-1]

        def patch(li, s1, s2):
            if li != 0:
                return s1, s2
            n1 = scores_from_factors(tabs['q1'], tabs['k1'], idx, HD)
            n2 = scores_from_factors(tabs['q2'], tabs['k2'], idx, HD)
            return n1.to(s1.dtype), n2.to(s2.dtype)

        logits = reference_forward(m, idx, 'bf16', score_patch=None if tabs is None else patch).float()
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1),
                             reduction='none').view(idx.shape[0], -1)
        out.append(ce.cpu())
    return torch.cat(out)


def ctx_finetune_head(h, fits):
    """Plain OV-context recipe (tick 160), (Dn, b, We) ordering throughout."""
    g = torch.Generator(device='cpu').manual_seed(7 + 100 * h)
    parts, params = {}, []
    for br in (0, 1):
        Dn0, b0, We0 = fits[h * 2 + br]
        Dm = Dn0.clone().requires_grad_(True)
        b = b0.clone().requires_grad_(True)
        We = We0.clone().requires_grad_(True)
        parts[br] = (Dm, b, We)
        params += [Dm, b, We]
    opt = torch.optim.Adam(params, lr=CTX_LR)
    Uh = Vv[:, h] @ Wo[:, h].T
    W2 = Uh.pow(2).sum(1)
    for step in range(CTX_STEPS):
        sample = torch.randperm(V, generator=g)[:CTX_M].to(DEV)
        qs = QFULL[sample]
        qs = qs / qs.sum()
        Us = Uh[sample]
        qw = qs * W2[sample]
        Ph, P = None, None
        for br, (qn, kn) in enumerate(BRANCHES):
            X = torch.cat([TAB[qn][sample, h], TAB[kn][sample, h]], 1)
            Dm, b, We = parts[br]
            Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
            z = (X - b) @ We.T
            vals, idx = z.abs().topk(K_DICT, dim=1)
            coeff = torch.gather(z, 1, idx)
            rec = b + (coeff.unsqueeze(-1) * Dn[idx]).sum(1)
            Sh = unit_rms(rec[:, :HD]) @ unit_rms(rec[:, HD:]).T / HD
            So = X[:, :HD] @ X[:, HD:].T / HD
            Ph = Sh if Ph is None else Ph * Sh
            P = So if P is None else P * So
        dP = Ph - P
        mu = (dP * qs[None, :]) @ Us
        s_ = (dP.pow(2) * qw[None, :]).sum(1)
        num = (qs * (T_CTX * (s_ - mu.pow(2).sum(1)).clamp_min(0)
                     + T_CTX ** 2 * mu.pow(2).sum(1))).sum()
        with torch.no_grad():
            mu0 = (P * qs[None, :]) @ Us
            s0 = (P.pow(2) * qw[None, :]).sum(1)
            den = (qs * (T_CTX * (s0 - mu0.pow(2).sum(1)).clamp_min(0)
                         + T_CTX ** 2 * mu0.pow(2).sum(1))).sum().clamp_min(1e-12)
        loss = num / den
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    del Uh, W2
    out = []
    for br in (0, 1):
        Dm, b, We = parts[br]
        out.append(((Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)).detach(),
                    b.detach(), We.detach()))
    return out


# ---------------- fits + dictionaries ----------------
print('fitting MSE dictionaries (256, 4)...', flush=True)
fits_mse = [train_dict(rows(*hb), N_DICT, K_DICT, seed=0) for hb in HB]
print('ctx finetune...', flush=True)
fits_ctx = []
for h in range(NH):
    fits_ctx += ctx_finetune_head(h, fits_mse)

recs_mse = {bi: encode_token(rows(*hb), *f, K_DICT) for bi, (f, hb) in enumerate(zip(fits_mse, HB))}
recs_ctx = {bi: encode_token(rows(*hb), *f, K_DICT) for bi, (f, hb) in enumerate(zip(fits_ctx, HB))}
tabs_ctx = tables_from(recs_ctx)
tabs_mse = tables_from(recs_mse)

# ---------------- A: per-position delta-CE ----------------
print('per-position audits (baseline, ctx, mse)...', flush=True)
ce_base = audit_pos(None)
ce_ctx = audit_pos(tabs_ctx)
ce_mse = audit_pos(tabs_mse)
d_ctx = (ce_ctx - ce_base).flatten()                     # (307200,)
d_mse = (ce_mse - ce_base).flatten()
print(f'mean dCE ctx {d_ctx.mean():+.4f} | mse {d_mse.mean():+.4f}', flush=True)

flat_targets = FINEWEB[:, 1:].flatten()
order = d_ctx.argsort(descending=True)
tot = float(d_ctx.sum())
NPOS = d_ctx.numel()
conc = {f'top_{p}pct_share': round(float(d_ctx[order[:int(NPOS * p / 100)]].sum()) / tot, 3)
        for p in (0.1, 1, 5, 10)}
neg_share = round(float(d_ctx[d_ctx < 0].sum()) / tot, 3)
frac_neg = round(float((d_ctx < 0).float().mean()), 3)
corr = float(np.corrcoef(d_ctx.numpy(), d_mse.numpy())[0, 1])
rk_c = d_ctx.argsort().argsort().float()
rk_m = d_mse.argsort().argsort().float()
spear = float(np.corrcoef(rk_c.numpy(), rk_m.numpy())[0, 1])


def pos_stats(idx_list):
    """Commonality stats for a set of flattened prediction indices."""
    st = {'freq_rank_med': 0, 'seen_before': 0, 'bigram_repeat': 0, 'dist_med': [],
          'pos_med': [], 'docs': set(), 'prev_tok': {}, 'ranks': []}
    for fi in idx_list:
        doc, p = fi // SEQ, fi % SEQ
        row = FINEWEB[doc]
        tgt = int(row[p + 1])
        st['ranks'].append(int(FREQ_RANK[tgt]))
        st['pos_med'].append(p)
        st['docs'].add(int(doc))
        prev = int(row[p])
        st['prev_tok'][prev] = st['prev_tok'].get(prev, 0) + 1
        past = row[:p + 1]
        hits = (past == tgt).nonzero().flatten()
        if len(hits):
            st['seen_before'] += 1
            st['dist_med'].append(int(p + 1 - hits[-1]))
            if hits[-1] > 0 and p >= 0 and int(past[hits[-1] - 1]) == prev:
                st['bigram_repeat'] += 1
    n = len(idx_list)
    return {
        'n': n,
        'freq_rank_median': int(np.median(st['ranks'])),
        'freq_rank_p90': int(np.percentile(st['ranks'], 90)),
        'seen_before_frac': round(st['seen_before'] / n, 3),
        'bigram_repeat_frac': round(st['bigram_repeat'] / n, 3),
        'dist_to_prev_median': int(np.median(st['dist_med'])) if st['dist_med'] else None,
        'pos_median': int(np.median(st['pos_med'])),
        'n_docs': len(st['docs']),
        'top_prev_tokens': sorted(st['prev_tok'].items(), key=lambda kv: -kv[1])[:8],
    }


top1000 = order[:1000].tolist()
g = torch.Generator().manual_seed(0)
rand1000 = torch.randperm(NPOS, generator=g)[:1000].tolist()
stats_top = pos_stats(top1000)
stats_rand = pos_stats(rand1000)
stats_top['top_prev_tokens'] = [(dec([t]), c) for t, c in stats_top['top_prev_tokens']]
stats_rand['top_prev_tokens'] = [(dec([t]), c) for t, c in stats_rand['top_prev_tokens']]

# aggregate curves: dCE by target-frequency decile and by position decile
tgt_rank = FREQ_RANK[flat_targets.to(DEV)].cpu()
freq_curve, pos_curve = [], []
qtiles = torch.quantile(tgt_rank.float(), torch.linspace(0, 1, 11))
for i in range(10):
    mask = (tgt_rank.float() >= qtiles[i]) & (tgt_rank.float() <= qtiles[i + 1])
    freq_curve.append(round(float(d_ctx[mask].mean()), 5))
posidx = torch.arange(NPOS) % SEQ
for i in range(10):
    mask = (posidx >= i * SEQ // 10) & (posidx < (i + 1) * SEQ // 10)
    pos_curve.append(round(float(d_ctx[mask].mean()), 5))

# top-100 decoded examples
examples = []
for fi in order[:100].tolist():
    doc, p = fi // SEQ, fi % SEQ
    row = FINEWEB[doc]
    ctxt = dec(row[max(0, p - 40):p + 1].tolist())
    examples.append({
        'doc': int(doc), 'pos': int(p),
        'context_tail': ctxt[-160:],
        'target': dec([int(row[p + 1])]),
        'ce_base': round(float(ce_base.flatten()[fi]), 3),
        'ce_ctx': round(float(ce_ctx.flatten()[fi]), 3),
        'd_mse_here': round(float(d_mse[fi]), 3),
        'tgt_freq_rank': int(FREQ_RANK[int(row[p + 1])]),
    })

# ---------------- B: per-head attribution ----------------
print('per-head audits...', flush=True)
base_mean = float(ce_base.mean())
head_dce = []
for h in range(NH):
    sub = {bi: recs_ctx[bi] for bi in (2 * h, 2 * h + 1)}
    t_h = tables_from(sub)
    head_dce.append(round(float(audit_pos(t_h).mean()) - base_mean, 4))
    print(f'  head {h} alone: {head_dce[-1]:+.4f}', flush=True)

# ---------------- C: weight-space delivered error, full vocab ----------------
print('full-vocab delivered error (dagger, Delta=0)...', flush=True)
CH = 1024
q_err = torch.zeros(V, device=DEV)          # per-query-token contribution q_i * E||e_i||^2
k_scat = torch.zeros(V, device=DEV)         # per-key scatter attribution
k_coh = torch.zeros(V, device=DEV)          # per-key coherent (mean-term) attribution
for h in range(NH):
    Uh = Vv[:, h] @ Wo[:, h].T
    w2 = Uh.pow(2).sum(1)
    qh1o, kh1o = tabs_ctx['q1'][:, h], tabs_ctx['k1'][:, h]
    qh2o, kh2o = tabs_ctx['q2'][:, h], tabs_ctx['k2'][:, h]
    q1e, k1e = unit_rms(TAB['q1'][:, h]), unit_rms(TAB['k1'][:, h])
    q2e, k2e = unit_rms(TAB['q2'][:, h]), unit_rms(TAB['k2'][:, h])
    for c0 in range(0, V, CH):
        sl = slice(c0, min(c0 + CH, V))
        Ph = (qh1o[sl] @ kh1o.T / HD) * (qh2o[sl] @ kh2o.T / HD)
        P = (q1e[sl] @ k1e.T / HD) * (q2e[sl] @ k2e.T / HD)
        dP = Ph - P
        del Ph, P
        mu = (dP * QP[None, :]) @ Uh                        # (CH, D)
        s_ = (dP.pow(2) * (QP * w2)[None, :]).sum(1)
        mu2 = mu.pow(2).sum(1)
        e_i = T_CTX * (s_ - mu2).clamp_min(0) + T_CTX ** 2 * mu2
        q_err[sl] += QP[sl] * e_i
        k_scat += (dP.pow(2) * (QP[sl][:, None] * QP[None, :] * w2[None, :])).sum(0) * T_CTX
        A = Uh @ mu.T                                       # (V, CH): u_t . mu_i
        k_coh += (dP.T * A * (QP[sl][None, :] * QP[:, None])).sum(1) * T_CTX ** 2
        del dP, mu, A
    del Uh
    torch.cuda.empty_cache()
    print(f'  head {h} done', flush=True)

top_q = q_err.argsort(descending=True)[:50]
top_k = (k_scat + k_coh).argsort(descending=True)[:50]
q_list = [(dec([int(t)]), round(float(q_err[t] / q_err.sum()), 4), int(FREQ_RANK[t]))
          for t in top_q.tolist()]
k_list = [(dec([int(t)]), round(float((k_scat + k_coh)[t] / (k_scat + k_coh).sum()), 4),
           int(FREQ_RANK[t])) for t in top_k.tolist()]
qerr_top50_share = round(float(q_err[top_q].sum() / q_err.sum()), 3)

# error contribution by query-token frequency decile (weight-space)
qerr_freq = []
rank_sorted = QFULL.argsort(descending=True)
for i in range(10):
    sl = rank_sorted[i * V // 10:(i + 1) * V // 10]
    qerr_freq.append(round(float(q_err[sl].sum() / q_err.sum()), 4))

# ---------------- D: factor-space residuals ----------------
print('factor residual structure...', flush=True)
res_by_freq = torch.zeros(10)
spec = []
qk_split = []
tok_relerr = torch.zeros(V, device=DEV)
for bi, hb in enumerate(HB):
    X = rows(*hb)
    R = recs_ctx[bi] - X
    tok_relerr += R.pow(2).sum(1) / X.pow(2).sum(1).clamp_min(1e-12) / NHB
    sv = torch.linalg.svdvals(R)
    e = sv.pow(2)
    spec.append((round(float(e[:8].sum() / e.sum()), 3), round(float(e[:32].sum() / e.sum()), 3)))
    qk_split.append(round(float(R[:, :HD].pow(2).sum() / R.pow(2).sum()), 3))
for i in range(10):
    sl = rank_sorted[i * V // 10:(i + 1) * V // 10]
    res_by_freq[i] = tok_relerr[sl].mean()
worst_tokens = [(dec([int(t)]), round(float(tok_relerr[t]), 3), int(FREQ_RANK[t]))
                for t in tok_relerr.argsort(descending=True)[:40].tolist()]

# ---------------- save + report ----------------
torch.save({'d_ctx': d_ctx, 'd_mse': d_mse, 'q_err': q_err.cpu(),
            'k_scat': k_scat.cpu(), 'k_coh': k_coh.cpu(), 'tok_relerr': tok_relerr.cpu()},
           f'{QK}/qk_err_explore.pt')

R = {'mean_dce_ctx': round(float(d_ctx.mean()), 5), 'mean_dce_mse': round(float(d_mse.mean()), 5),
     'concentration': conc, 'improved_frac': frac_neg, 'improved_share': neg_share,
     'pearson_ctx_mse': round(corr, 3), 'spearman_ctx_mse': round(spear, 3),
     'stats_top1000': stats_top, 'stats_rand1000': stats_rand,
     'dce_by_target_freq_decile': freq_curve, 'dce_by_position_decile': pos_curve,
     'head_dce': head_dce, 'qerr_top50_share': qerr_top50_share,
     'qerr_by_freq_decile': qerr_freq, 'top_query_tokens': q_list, 'top_key_tokens': k_list,
     'residual_spectrum_e8_e32': spec, 'residual_qhalf_share': qk_split,
     'residual_by_freq_decile': [round(float(x), 4) for x in res_by_freq],
     'worst_factor_tokens': worst_tokens}
json.dump(R, open(f'{QK}/qk_err_explore.json', 'w'), indent=2)

with open(f'{QK}/qk_err_explore_examples.md', 'w') as f:
    f.write('# Top-100 highest-error predictions — ctx-trained dict (256, 4), FineWeb\n\n')
    f.write('| # | doc | pos | context tail (last tokens) | target | CE base→dict | mse Δ here | tgt freq rank |\n')
    f.write('|---|-----|-----|---------------------------|--------|--------------|-----------|---------------|\n')
    for i, ex in enumerate(examples):
        c = ex['context_tail'].replace('|', '\\|')
        t = ex['target'].replace('|', '\\|')
        f.write(f"| {i + 1} | {ex['doc']} | {ex['pos']} | …{c} | **{t}** | "
                f"{ex['ce_base']}→{ex['ce_ctx']} | {ex['d_mse_here']:+.2f} | {ex['tgt_freq_rank']} |\n")
print('\nEXPLORE COMPLETE', flush=True)
print(json.dumps({k: v for k, v in R.items() if k not in
                  ('top_query_tokens', 'top_key_tokens', 'worst_factor_tokens')}, indent=1))
