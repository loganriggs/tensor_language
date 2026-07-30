"""Verify: comparison (greater-of-two digits) in bilin18, few-shot.
Prompt: "7 3 -> 7\n2 8 -> 8\n{a} {b} ->" ; two scorings:
  acc2: argmax restricted to {a, b} (chance 0.5) — the comparison signal proper.
  acc9: argmax over all digits 1-9 (chance 1/9) — does it even stay on-task?
All ordered pairs a != b from 1-9 (72 items). Few-shot examples avoid the query digits' answer bias
by using two fixed examples with both orders.
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

def tid(s):
    t = tok(s)['input_ids']
    assert len(t) == 1, s
    return t[0]

D = {d: tid(' ' + str(d)) for d in range(1, 10)}
FS = "7 3 -> 7\n2 8 -> 8\n"
hits2, hits9, n = 0, 0, 0
margins = []
for a in range(1, 10):
    for b in range(1, 10):
        if a == b: continue
        gt = max(a, b)
        lp = next_logprobs(FS + f"{a} {b} ->")
        la, lb = float(lp[D[a]]), float(lp[D[b]])
        pred2 = a if la > lb else b
        cand9 = {d: float(lp[D[d]]) for d in range(1, 10)}
        pred9 = max(cand9, key=cand9.get)
        n += 1; hits2 += (pred2 == gt); hits9 += (pred9 == gt)
        margins.append((la - lb) if gt == a else (lb - la))
res = {'acc2': round(hits2/n, 3), 'acc9': round(hits9/n, 3), 'n': n,
       'chance2': 0.5, 'chance9': 0.111,
       'mean_margin2': round(float(np.mean(margins)), 3)}
verdict = 'YES' if res['acc2'] >= 0.85 else ('WEAK' if res['acc2'] > 0.65 else 'NO')
res['verdict'] = verdict
print(f"TASK greater_of_two acc2={res['acc2']} (chance 0.5) acc9={res['acc9']} (chance 0.111) n={n} verdict={verdict}", flush=True)
json.dump(res, open(f'{QK}/qk_algoverify_greater_of_two.json', 'w'), indent=2)
print("DONE", flush=True)
