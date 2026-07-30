"""Verify: reversal of a 3-digit sequence in bilin18, few-shot.
Prompt: "1 2 3 -> 3 2 1\n7 8 9 -> 9 8 7\n{a} {b} {c} ->" ; correct first output = c (the LAST
input digit). Argmax restricted to {a, b, c}; chance 1/3. Digits drawn without replacement,
avoiding monotone triples (which a successor/copy heuristic could fake).
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
FS = "1 2 3 -> 3 2 1\n7 8 9 -> 9 8 7\n"
rng = np.random.RandomState(0)
hits, n = 0, 0
pick_first = pick_mid = 0
while n < 40:
    trip = rng.choice(np.arange(1, 10), 3, replace=False)
    a, b, c = map(int, trip)
    if (a < b < c) or (a > b > c): continue   # skip monotone triples
    lp = next_logprobs(FS + f"{a} {b} {c} ->")
    cand = {d: float(lp[D[d]]) for d in (a, b, c)}
    pred = max(cand, key=cand.get)
    n += 1; hits += (pred == c)
    pick_first += (pred == a); pick_mid += (pred == b)
res = {'acc': round(hits/n, 3), 'n': n, 'chance': 0.333,
       'picked_first_frac': round(pick_first/n, 3), 'picked_middle_frac': round(pick_mid/n, 3)}
verdict = 'YES' if res['acc'] >= 0.667 else ('WEAK' if res['acc'] > 0.45 else 'NO')
res['verdict'] = verdict
print(f"TASK reverse3 acc={res['acc']} n={n} chance=0.333 picked_first={res['picked_first_frac']} verdict={verdict}", flush=True)
json.dump(res, open(f'{QK}/qk_algoverify_reverse3.json', 'w'), indent=2)
print("DONE", flush=True)
