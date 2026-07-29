"""Step 1: build 40 clean/corrupted increment pairs, verify behavior, save stimuli."""
import json, sys
import torch
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl/algo_tasks/increment')
from common import get_model, forward, batched, build_stimuli, OUT

torch.manual_seed(0)
m, cfg = get_model()
S = build_stimuli(40, seed=0)
clean, corr = S['clean'].cuda(), S['corr'].cuda()
ca, xa = S['clean_ans'].cuda(), S['corr_ans'].cuda()

with torch.no_grad():
    lg_c = batched(lambda b: forward(m, b), clean)[:, -1].float()
    lg_x = batched(lambda b: forward(m, b), corr)[:, -1].float()

top1_c = (lg_c.argmax(-1) == ca).float().mean().item()
top1_x = (lg_x.argmax(-1) == xa).float().mean().item()
# margin: logit(clean answer) - logit(corrupted answer), final position
n = torch.arange(len(ca), device='cuda')
M_clean = (lg_c[n, ca] - lg_c[n, xa])
M_corr = (lg_x[n, ca] - lg_x[n, xa])
print(f"clean top-1 (predict k+2):     {top1_c:.3f}")
print(f"corrupted top-1 (predict k'+2): {top1_x:.3f}")
print(f"margin clean:  mean {M_clean.mean():.3f}  min {M_clean.min():.3f}")
print(f"margin corr:   mean {M_corr.mean():.3f}  max {M_corr.max():.3f}")

# per-pair sanity: margin must actually move (clean >> corr) for the metric to be usable
gap = (M_clean - M_corr)
print(f"margin gap (clean-corr): mean {gap.mean():.3f}  min {gap.min():.3f}")

res = {
    'n_pairs': 40, 'n_analysis': 30, 'n_heldout': 10,
    'corruption': 'constant shift: both list numbers k,k+1 -> k\',k\'+1 (k\'!=k, same words); '
                  'corrupted answer k\'+2 != clean answer k+2',
    'clean_top1': top1_c, 'corr_top1': top1_x,
    'margin_clean_mean': M_clean.mean().item(), 'margin_clean_min': M_clean.min().item(),
    'margin_corr_mean': M_corr.mean().item(), 'margin_corr_max': M_corr.max().item(),
    'margin_gap_mean': gap.mean().item(), 'margin_gap_min': gap.min().item(),
    'examples': S['meta'][:5],
}
json.dump(res, open(f'{OUT}/s1_stimuli.json', 'w'), indent=2)
torch.save({k: S[k] for k in ['clean', 'corr', 'clean_ans', 'corr_ans']}, f'{OUT}/stimuli.pt')
with open(f'{OUT}/stimuli_meta.json', 'w') as fh:
    json.dump(S['meta'], fh, indent=2)
print('saved s1_stimuli.json / stimuli.pt')
