"""s2c: prediction-level agreement between CODED imposition and REAL-activation
imposition of the same wrong element (build_pairs seed=2), per site."""
import sys, torch
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl/algo_tasks/semantics_successor')
from semlib import (HERE, DEV, L_PAY, HEADS, get_model, build_pairs, batched_run,
                    v1_of_tokens, coded_payload, save_json)
m, cfg = get_model()
NH, D = cfg['n_head'], cfg['n_embd']; HD = D // NH
W_A = torch.load(f'{HERE}/code_W.pt')['W_full']
W_B = torch.load(f'{HERE}/code_WB.pt')['W_B']
emb = m.transformer.wte.weight.detach().double().cpu()
pairs = build_pairs(seed=2, length=4); n = len(pairs)
ci = torch.tensor([p['tokens'] for p, q in pairs], device=DEV)
xi = torch.tensor([q['tokens'] for p, q in pairs], device=DEV)
pp = torch.tensor([p['pred_pos'] for p, q in pairs], device=DEV)
lp = torch.tensor([p['last_pos'] for p, q in pairs], device=DEV)
imp = xi[torch.arange(n), lp]
pA = coded_payload(W_A, m, imp)
X = torch.cat([emb[imp.cpu()], torch.ones(n, 1, dtype=torch.float64)], 1)
vB = (X @ W_B).float().to(DEV).view(n, NH, HD)
v1r = v1_of_tokens(m, cfg, imp)
def pred(**kw):
    lg, _ = batched_run(m, cfg, ci, bs=6, **kw)
    return lg[torch.arange(n), pp.cpu()].argmax(-1)
prAc = pred(head_sub={(L_PAY, h): (pA[h], pp) for h in HEADS})
lgx, cx = batched_run(m, cfg, xi, bs=6, want_head=(L_PAY,))
dh = {h: [] for h in HEADS}
for bi, c in enumerate(cx):
    yh = c[('h', L_PAY)]
    for b in range(yh.shape[0]):
        i = bi * 6 + b
        for h in HEADS:
            dh[h].append(yh[b, pairs[i][0]['pred_pos'], h, :])
dh = {h: torch.stack(v).to(DEV) for h, v in dh.items()}
prAr = pred(head_sub={(L_PAY, h): (dh[h], pp) for h in HEADS})
prBc = pred(v1_sub={(li, h): (vB[:, h], lp) for li in range(1, 18) for h in range(NH)})
prBr = pred(v1_sub={(li, h): (v1r[:, h], lp) for li in range(1, 18) for h in range(NH)})
res = {'A_coded_vs_real_agree': (prAc == prAr).float().mean().item(),
       'B_coded_vs_real_agree': (prBc == prBr).float().mean().item(), 'n': n}
print(res); save_json('s2c_agreement.json', res)
