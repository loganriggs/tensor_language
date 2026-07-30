"""s1b: site ladder with the true behavioral CEILING.

Ceiling = run the donor (corrupted) prompt directly: how often does the model
follow the replaced last element at all? (Prior report: 45-80% by family.)
Then real-activation swaps of increasing scope, donor -> clean:
  A_pred:      L8 H3+H7 head outputs at pred pos only
  A_pred_all9: all 9 L8 heads at pred pos
  A_allpos:    L8 H3+H7 at ALL positions
  A_allpos9:   all 9 L8 heads at all positions (full L8 attention write swap)
  V1_global:   v1 cache slice at last pos substituted at EVERY layer 1-17 read
               (all heads) -- 'identity swap in the value stream', QK/embedding
               routes keep the real token
  V1_global+A_pred: both
"""
import sys

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl/algo_tasks/semantics_successor')
from semlib import (DEV, L_PAY, HEADS, FAMILIES, get_model, build_pairs,
                    batched_run, v1_of_tokens, save_json, free_gb)

torch.manual_seed(0)
print(f'free GPU mem: {free_gb():.1f} GB', flush=True)
m, cfg = get_model()
NH = cfg['n_head']

pairs = build_pairs(seed=1, length=4)
print(f'pairs: {len(pairs)}', flush=True)
ci = torch.tensor([p['tokens'] for p, q in pairs], device=DEV)
xi = torch.tensor([q['tokens'] for p, q in pairs], device=DEV)
pred_pos = torch.tensor([p['pred_pos'] for p, q in pairs], device=DEV)
last_pos = torch.tensor([p['last_pos'] for p, q in pairs], device=DEV)
clean_ans = torch.tensor([p['ans_tok'] for p, q in pairs])
follow = torch.tensor([q['follow_tok'] for p, q in pairs])
fams = [p['family'] for p, q in pairs]


def stats(pr, name):
    res = {'follow_rate': (pr == follow).float().mean().item(),
           'stay_rate': (pr == clean_ans).float().mean().item(), 'per_family': {}}
    for fam in list(FAMILIES) + ['numlist']:
        ii = torch.tensor([i for i, f in enumerate(fams) if f == fam])
        res['per_family'][fam] = {'n': len(ii),
                                  'follow': (pr[ii] == follow[ii]).float().mean().item(),
                                  'stay': (pr[ii] == clean_ans[ii]).float().mean().item()}
    print(f'{name}: follow={res["follow_rate"]:.3f} stay={res["stay_rate"]:.3f} '
          + ' '.join(f'{f}:{v["follow"]:.2f}' for f, v in res['per_family'].items()), flush=True)
    return res


out = {}
# ceiling: donor prompt run directly
lgx, cx = batched_run(m, cfg, xi, bs=6, want_head=(L_PAY,))
prx = lgx[torch.arange(len(pairs)), pred_pos.cpu()].argmax(-1)
out['ceiling_donor_run'] = stats(prx, 'CEILING (donor prompt itself)')

# donor head outputs, full T
donor_full = {h: [] for h in range(NH)}
for c in cx:
    yh = c[('h', L_PAY)]
    for b in range(yh.shape[0]):
        for h in range(NH):
            donor_full[h].append(yh[b, :, h, :])
donor_full = {h: torch.stack(v).to(DEV) for h, v in donor_full.items()}
donor_pred = {h: donor_full[h][torch.arange(len(pairs)), pred_pos] for h in range(NH)}

v1_don = v1_of_tokens(m, cfg, xi[torch.arange(len(pairs)), last_pos]).to(DEV)


def cond(name, **kw):
    lg, _ = batched_run(m, cfg, ci, bs=6, **kw)
    pr = lg[torch.arange(len(pairs)), pred_pos.cpu()].argmax(-1)
    return stats(pr, name)


out['A_pred'] = cond('A_pred (H3+H7 @pred)',
                     head_sub={(L_PAY, h): (donor_pred[h], pred_pos) for h in HEADS})
out['A_pred_all9'] = cond('A_pred_all9',
                          head_sub={(L_PAY, h): (donor_pred[h], pred_pos) for h in range(NH)})
out['A_allpos'] = cond('A_allpos (H3+H7 @all)',
                       head_sub={(L_PAY, h): (donor_full[h], None) for h in HEADS})
out['A_allpos9'] = cond('A_allpos9 (full L8 write)',
                        head_sub={(L_PAY, h): (donor_full[h], None) for h in range(NH)})
out['V1_global'] = cond('V1_global (v1@last, layers 1-17, all heads)',
                        v1_sub={(li, h): (v1_don[:, h], last_pos)
                                for li in range(1, 18) for h in range(NH)})
out['V1_global_plus_A_pred'] = cond('V1_global + A_pred',
                                    v1_sub={(li, h): (v1_don[:, h], last_pos)
                                            for li in range(1, 18) for h in range(NH)},
                                    head_sub={(L_PAY, h): (donor_pred[h], pred_pos) for h in HEADS})

save_json('s1b_site_ladder.json', out)
print('done', flush=True)
