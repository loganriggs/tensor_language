"""Control for the iter/menu audits: ONLY the four zero layers (8,14,15,17)
silenced, everything else live. This is the floor every composed menu number
sits on; marginal sum is +0.023 (+0.050-0.035+0.002+0.006)."""
import json
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, build_eval_tokens, reference_forward

torch.manual_seed(0)
DEV = 'cuda'
ZERO_L = {8, 14, 15, 17}
m, cfg = load_elriggs('bilin18')
AUDIT = build_eval_tokens(n_chunks=20, seq_len=513)[4:20]


def patch(li, s1, s2):
    if li in ZERO_L:
        return torch.zeros_like(s1), torch.zeros_like(s2)
    return s1, s2


@torch.no_grad()
def ce(p):
    tot, n = 0.0, 0
    for i in range(0, len(AUDIT), 4):
        b = AUDIT[i:i + 4].to(DEV)
        logits = reference_forward(m, b[:, :-1], 'bf16', score_patch=p).float()
        c = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1))
        tot += c.item() * b[:, 1:].numel(); n += b[:, 1:].numel()
    return tot / n


base = ce(None)
dz = ce(patch) - base
print(f'baseline {base:.4f}; zeros-only (8,14,15,17): dCE {dz:+.4f}', flush=True)
json.dump({'baseline': base, 'zeros_only': dz},
          open('/workspace/tensor_language/basis_aligned/qk_mdl/zeros_control.json', 'w'), indent=2)
print('zeros control done', flush=True)
