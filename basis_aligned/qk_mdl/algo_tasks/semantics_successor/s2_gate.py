"""s2: the CODE gate. Substitute CODED payloads (computed from the last
element's token embedding via fitted linear W) for the real ones; measure
successor accuracy per family, follow-rate under imposed-different-element
(placebo -- the strongest verification), zeroing, and real-swap comparison.

Two codes / sites:
  Code-A (spec site): W_A emb(e) -> L8 H3+H7 head outputs at pred pos (fitted in s1).
  Code-B (v1 pointer code): W_B emb(e) -> the token's v1 cache slice (all 9 heads),
          imposed at the last position at every layer 1-17 read (site established
          in s1b as carrying the full identity payload).
The real-swap wrong-content control uses the SAME imposed element e' as the
coded placebo (build_pairs seed=2), so follow-rates are directly comparable.
"""
import sys

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl/algo_tasks/semantics_successor')
from semlib import (HERE, DEV, L_PAY, HEADS, FAMILIES, get_model, build_pairs,
                    batched_run, v1_of_tokens, coded_payload, fit_ridge,
                    save_json, free_gb)

torch.manual_seed(0)
print(f'free GPU mem: {free_gb():.1f} GB', flush=True)
m, cfg = get_model()
NH, D = cfg['n_head'], cfg['n_embd']
HD = D // NH

Wd = torch.load(f'{HERE}/code_W.pt')
W_A = Wd['W_full']
HOLDOUT = set(Wd['holdout_elems'])

# ------------------------------------------------- fit Code-B (emb -> v1) ----
cal = torch.load(f'{HERE}/calibration.pt')
cal_toks = sorted({mm['last_tok'] for mm in cal['cal_meta']})
toks_t = torch.tensor(cal_toks, device=DEV)
v1_tgt = v1_of_tokens(m, cfg, toks_t).reshape(len(cal_toks), -1).double().cpu()
emb = m.transformer.wte.weight.detach().double().cpu()
Phi = emb[torch.tensor(cal_toks)]
W_B, lamB, r2B = fit_ridge(Phi, v1_tgt)
tok2elem = {}
for mm in cal['cal_meta']:
    tok2elem[mm['last_tok']] = mm['last_elem']
keep = [i for i, t in enumerate(cal_toks) if tok2elem[t] not in HOLDOUT]
W_B_hold, lamBh, r2Bh = fit_ridge(Phi[keep], v1_tgt[keep])
torch.save({'W_B': W_B, 'W_B_hold': W_B_hold, 'cal_toks': cal_toks}, f'{HERE}/code_WB.pt')
print(f'Code-B ridge: lam={lamB} split-R2={r2B:.4f} (holdout-fit R2={r2Bh:.4f})', flush=True)


def coded_v1(W, toks):
    X = torch.cat([emb[toks.cpu()], torch.ones(len(toks), 1, dtype=torch.float64)], 1)
    return (X @ W).float().to(DEV).view(len(toks), NH, HD)


# ------------------------------------------------------------- eval set ------
pairs = build_pairs(seed=2, length=4)
n = len(pairs)
ci = torch.tensor([p['tokens'] for p, q in pairs], device=DEV)
xi = torch.tensor([q['tokens'] for p, q in pairs], device=DEV)
pred_pos = torch.tensor([p['pred_pos'] for p, q in pairs], device=DEV)
last_pos = torch.tensor([p['last_pos'] for p, q in pairs], device=DEV)
true_ans = torch.tensor([p['ans_tok'] for p, q in pairs])
follow = torch.tensor([q['follow_tok'] for p, q in pairs])
true_toks = ci[torch.arange(n), last_pos]
imp_toks = xi[torch.arange(n), last_pos]          # imposed element e' tokens
fams = [p['family'] for p, q in pairs]
print(f'eval prompts: {n}', flush=True)


def stats(pr, name, ref):
    """ref: tensor of 'success' answer tokens (true_ans or follow)."""
    res = {'success_rate': (pr == ref).float().mean().item(),
           'stay_rate': (pr == true_ans).float().mean().item(), 'per_family': {}}
    for fam in list(FAMILIES) + ['numlist']:
        ii = torch.tensor([i for i, f in enumerate(fams) if f == fam])
        res['per_family'][fam] = {'n': len(ii),
                                  'success': (pr[ii] == ref[ii]).float().mean().item(),
                                  'stay': (pr[ii] == true_ans[ii]).float().mean().item()}
    print(f'{name}: success={res["success_rate"]:.3f} stay={res["stay_rate"]:.3f} '
          + ' '.join(f'{f}:{v["success"]:.2f}' for f, v in res['per_family'].items()), flush=True)
    return res


def cond(name, ref, **kw):
    lg, _ = batched_run(m, cfg, ci, bs=6, **kw)
    pr = lg[torch.arange(n), pred_pos.cpu()].argmax(-1)
    return stats(pr, name, ref)


out = {'code_B_fit': {'lam': lamB, 'split_r2': r2B, 'holdout_fit_r2': r2Bh}}

out['baseline'] = cond('baseline', true_ans)

# ---- Code-A (spec site: L8 H3+H7 @ pred) ----
pA_true = coded_payload(W_A, m, true_toks)
pA_imp = coded_payload(W_A, m, imp_toks)
out['A_code_self'] = cond('A_code_self (W_A(true))', true_ans,
                          head_sub={(L_PAY, h): (pA_true[h], pred_pos) for h in HEADS})
out['A_code_placebo'] = cond('A_code_placebo (W_A(e\'))', follow,
                             head_sub={(L_PAY, h): (pA_imp[h], pred_pos) for h in HEADS})
out['A_zero'] = cond('A_zero', true_ans,
                     head_sub={(L_PAY, h): (torch.zeros(n, HD, device=DEV), pred_pos)
                               for h in HEADS})
# real wrong-content at site A: donor run activations (same e')
lgx, cx = batched_run(m, cfg, xi, bs=6, want_head=(L_PAY,))
donor_pred = {h: [] for h in HEADS}
for bi, c in enumerate(cx):
    yh = c[('h', L_PAY)]
    for b in range(yh.shape[0]):
        i = bi * 6 + b
        for h in HEADS:
            donor_pred[h].append(yh[b, pairs[i][0]['pred_pos'], h, :])
donor_pred = {h: torch.stack(v).to(DEV) for h, v in donor_pred.items()}
out['A_real_wrongcontent'] = cond('A_real_wrongcontent (donor acts)', follow,
                                  head_sub={(L_PAY, h): (donor_pred[h], pred_pos)
                                            for h in HEADS})
prx = lgx[torch.arange(n), pred_pos.cpu()].argmax(-1)
out['ceiling_donor_run'] = stats(prx, 'ceiling (donor prompt run)', follow)

# ---- Code-B (v1 pointer code, global) ----
vB_true = coded_v1(W_B, true_toks)
vB_imp = coded_v1(W_B, imp_toks)
v1_real_imp = v1_of_tokens(m, cfg, imp_toks)
ALL_LAYERS = range(1, 18)
out['B_code_self'] = cond('B_code_self (W_B(true))', true_ans,
                          v1_sub={(li, h): (vB_true[:, h], last_pos)
                                  for li in ALL_LAYERS for h in range(NH)})
out['B_code_placebo'] = cond('B_code_placebo (W_B(e\'))', follow,
                             v1_sub={(li, h): (vB_imp[:, h], last_pos)
                                     for li in ALL_LAYERS for h in range(NH)})
out['B_zero'] = cond('B_zero (v1@last := 0)', true_ans,
                     v1_sub={(li, h): (torch.zeros(n, HD, device=DEV), last_pos)
                             for li in ALL_LAYERS for h in range(NH)})
out['B_real_wrongcontent'] = cond('B_real_wrongcontent (exact v1(e\'))', follow,
                                  v1_sub={(li, h): (v1_real_imp[:, h], last_pos)
                                          for li in ALL_LAYERS for h in range(NH)})

save_json('s2_gate.json', out)
print('done', flush=True)
