"""Verify: quote-STYLE matching in bilin18. Known: an open double quote boosts the closing
double quote. NEW: does the model match the style — open ' -> close ', open " -> close "?
Argmax over the two closer tokens {", '}; chance 0.5. Both directions tested symmetrically.
"""
import json, sys
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, reference_forward
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
m, cfg = load_elriggs('bilin18')
tok = AutoTokenizer.from_pretrained('gpt2')
enc = lambda s: tok(s, return_tensors='pt')['input_ids'][0]

@torch.no_grad()
def next_logprobs(prompt):
    idx = enc(prompt).unsqueeze(0).to(DEV)
    lg = reference_forward(m, idx)
    return F.log_softmax(lg[0, -1].float(), -1)

DQ = tok('"')['input_ids'][0]   # closer "
SQ = tok("'")['input_ids'][0]   # closer '
rng = np.random.RandomState(0)
NOUNS = ['red', 'blue', 'green', 'dogs', 'cats', 'birds', 'cars', 'trees', 'books', 'stars']
res = {'per_style': {}}
hits, n, margins = 0, 0, []
for style, closer_tid in [('"', DQ), ("'", SQ)]:
    sh = 0
    for i in range(20):
        w = rng.choice(NOUNS, 2, replace=False)
        p = f'She said {style}the {w[0]} is near the {w[1]}'
        lp = next_logprobs(p)
        cd, cs = float(lp[DQ]), float(lp[SQ])
        pred = DQ if cd > cs else SQ
        ok = (pred == closer_tid)
        hits += ok; sh += ok; n += 1
        margins.append((cd - cs) if closer_tid == DQ else (cs - cd))
    res['per_style'][style] = round(sh/20, 3)
acc = hits/n
res.update({'acc': round(acc, 3), 'n': n, 'chance': 0.5,
            'mean_margin': round(float(np.mean(margins)), 3)})
verdict = 'YES' if acc >= 0.85 else ('WEAK' if acc > 0.65 else 'NO')
res['verdict'] = verdict
print(f"TASK quote_style acc={res['acc']} n={n} chance=0.5 per_style={res['per_style']} verdict={verdict}", flush=True)
json.dump(res, open(f'{QK}/qk_algoverify_quote_style.json', 'w'), indent=2)
print("DONE", flush=True)
