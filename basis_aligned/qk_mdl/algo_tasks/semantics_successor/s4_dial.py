"""s4: DIAL. Scale the coded payload s in {0, 0.5, 1, 1.5, 2} on the eval
prompts (self-identity payload): successor accuracy dose-response per family.
Code-A: s * W_A emb(true last) at L8 H3+H7 pred pos.
Code-B: s * W_B emb(true last) as the v1 slice at last pos (layers 1-17).
(The natural-CE side of the dial -- scaling the real channel content on
FineWeb rows 448:600 -- is in s2b: dCE < 0.007 for s in [0,2].)
"""
import sys

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl/algo_tasks/semantics_successor')
from semlib import (HERE, DEV, L_PAY, HEADS, FAMILIES, get_model, build_pairs,
                    batched_run, coded_payload, save_json, free_gb)

torch.manual_seed(0)
print(f'free GPU {free_gb():.1f} GB', flush=True)
m, cfg = get_model()
NH, D = cfg['n_head'], cfg['n_embd']
HD = D // NH

W_A = torch.load(f'{HERE}/code_W.pt')['W_full']
W_B = torch.load(f'{HERE}/code_WB.pt')['W_B']
emb = m.transformer.wte.weight.detach().double().cpu()

pairs = build_pairs(seed=2, length=4)
n = len(pairs)
ci = torch.tensor([p['tokens'] for p, q in pairs], device=DEV)
pred_pos = torch.tensor([p['pred_pos'] for p, q in pairs], device=DEV)
last_pos = torch.tensor([p['last_pos'] for p, q in pairs], device=DEV)
true_ans = torch.tensor([p['ans_tok'] for p, q in pairs])
true_toks = ci[torch.arange(n), last_pos]
fams = [p['family'] for p, q in pairs]

pA = coded_payload(W_A, m, true_toks)
X = torch.cat([emb[true_toks.cpu()], torch.ones(n, 1, dtype=torch.float64)], 1)
vB = (X @ W_B).float().to(DEV).view(n, NH, HD)

out = {}
for s in (0.0, 0.5, 1.0, 1.5, 2.0):
    for key in ('A', 'B'):
        if key == 'A':
            kw = {'head_sub': {(L_PAY, h): (s * pA[h], pred_pos) for h in HEADS}}
        else:
            kw = {'v1_sub': {(li, h): (s * vB[:, h], last_pos)
                             for li in range(1, 18) for h in range(NH)}}
        lg, _ = batched_run(m, cfg, ci, bs=6, **kw)
        pr = lg[torch.arange(n), pred_pos.cpu()].argmax(-1)
        res = {'acc': (pr == true_ans).float().mean().item(), 'per_family': {}}
        for fam in list(FAMILIES) + ['numlist']:
            ii = torch.tensor([i for i, f in enumerate(fams) if f == fam])
            res['per_family'][fam] = (pr[ii] == true_ans[ii]).float().mean().item()
        out[f'{key}_s{s}'] = res
        print(f'{key} s={s}: acc={res["acc"]:.3f} '
              + ' '.join(f'{f}:{v:.2f}' for f, v in res['per_family'].items()), flush=True)

save_json('s4_dial.json', out)
print('done', flush=True)
