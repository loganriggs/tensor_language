"""Verify: subject-verb agreement across an attractor PP in bilin18.
Prompt: "The {key|keys} to the {cabinet|cabinets}" -> " is" vs " are". Correct verb agrees
with the HEAD noun, not the attractor. 2-way argmax; chance 0.5. Reports accuracy split by
congruent (head/attractor same number) vs incongruent (attractor mismatches) — the
incongruent cells are the real test of structural agreement vs linear recency.
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

IS = tok(' is')['input_ids'][0]; ARE = tok(' are')['input_ids'][0]
# (singular, plural) noun pairs; all forms single-token with leading space in GPT-2
HEADS = [('key', 'keys'), ('door', 'doors'), ('book', 'books'), ('car', 'cars'), ('star', 'stars')]
ATTRS = [('cabinet', 'cabinets'), ('house', 'houses'), ('table', 'tables'), ('garden', 'gardens')]
cells = {'congruent': [0, 0], 'incongruent': [0, 0]}   # [hits, n]
margins = []
for hs, hp in HEADS:
    for as_, ap in ATTRS:
        for head_pl in (0, 1):
            for attr_pl in (0, 1):
                head = hp if head_pl else hs
                attr = ap if attr_pl else as_
                p = f"The {head} to the {attr}"
                lp = next_logprobs(p)
                li, la = float(lp[IS]), float(lp[ARE])
                pred_pl = int(la > li)
                ok = (pred_pl == head_pl)
                cell = 'congruent' if head_pl == attr_pl else 'incongruent'
                cells[cell][0] += ok; cells[cell][1] += 1
                margins.append((la - li) if head_pl else (li - la))
res = {}
for c, (h, n) in cells.items():
    res[f'acc_{c}'] = round(h/n, 3); res[f'n_{c}'] = n
tot_h = sum(v[0] for v in cells.values()); tot_n = sum(v[1] for v in cells.values())
res.update({'acc': round(tot_h/tot_n, 3), 'n': tot_n, 'chance': 0.5,
            'mean_margin': round(float(np.mean(margins)), 3)})
verdict = 'YES' if res['acc_incongruent'] >= 0.75 else ('WEAK' if res['acc_incongruent'] > 0.6 else 'NO')
res['verdict'] = verdict
print(f"TASK sv_agreement acc={res['acc']} (congruent {res['acc_congruent']}, incongruent {res['acc_incongruent']}) "
      f"n={tot_n} chance=0.5 verdict={verdict}", flush=True)
json.dump(res, open(f'{QK}/qk_algoverify_sv_agreement.json', 'w'), indent=2)
print("DONE", flush=True)
