"""L5 mechanism probe: layer 5 is the only layer whose QK selection is
genuinely contextual (cond-mean tables leave +0.25 of a +2.51 zero gap).
Which heads? For each head h: (a) table h alone (rest live) — marginal cost;
(b) keep h live, table the rest — marginal recovery. Uses the saved all17
cond-mean tables; L5 patched only, dCE at T=512."""
import json
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, build_eval_tokens, reference_forward
from tier2_folding import scores_from_factors

torch.manual_seed(0)
DEV = 'cuda'
L = 5
OUT = '/workspace/tensor_language/basis_aligned/qk_mdl/l5_probe.json'
m, cfg = load_elriggs('bilin18')
NH, HD = cfg['n_head'], cfg['n_embd'] // cfg['n_head']
V = cfg['vocab_size']
AUDIT = build_eval_tokens(n_chunks=20, seq_len=513)[4:20]

raw = torch.load('/workspace/tensor_language/basis_aligned/qk_mdl/all17_tables.pt')
tabs = {n: raw[f'{L}_{n}'].float().to(DEV) for n in ('q1', 'k1', 'q2', 'k2')}
del raw


@torch.no_grad()
def audit_ce(head_mask=None):
    # head_mask: bool (NH,) — True = replace that head's scores with table scores
    tot, n = 0.0, 0
    for i in range(0, len(AUDIT), 4):
        b = AUDIT[i:i + 4].to(DEV)
        idx = b[:, :-1]

        def patch(li, s1, s2):
            if li != L or head_mask is None:
                return s1, s2
            n1 = scores_from_factors(tabs['q1'], tabs['k1'], idx, HD).to(s1.dtype)
            n2 = scores_from_factors(tabs['q2'], tabs['k2'], idx, HD).to(s2.dtype)
            mask = torch.tensor(head_mask, device=DEV)[None, :, None, None]
            return torch.where(mask, n1, s1), torch.where(mask, n2, s2)

        logits = reference_forward(m, idx, 'bf16', score_patch=patch).float()
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n


base = audit_ce()
res = {'baseline_ce': base, 'all_tabled': audit_ce([True] * NH) - base, 'heads': {}}
print(f"baseline {base:.4f}; all-tabled dCE {res['all_tabled']:+.4f}", flush=True)
for h in range(NH):
    only_h = [i == h for i in range(NH)]
    all_but_h = [i != h for i in range(NH)]
    ca = audit_ce(only_h) - base
    cb = audit_ce(all_but_h) - base
    res['heads'][h] = {'table_h_only': ca, 'live_h_only': cb}
    print(f'H{h}: table-alone {ca:+.4f} · live-alone {cb:+.4f} '
          f'(recovers {res["all_tabled"] - cb:+.4f})', flush=True)
    with open(OUT, 'w') as fh:
        json.dump(res, fh, indent=2)
print('l5 probe done', flush=True)
