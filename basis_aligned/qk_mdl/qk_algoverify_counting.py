"""Verify: counting repeated items in bilin18, few-shot.
Prompt: "x x x = 3\nx x = 2\nx x x x = 4\n<word repeated k times> =" -> " k".
Argmax over digits 1-9 (chance 1/9). k in 2..7, several filler words. Also reports
off-by-one rate (predicting k-1 or k+1), which distinguishes 'approximate count' from noise.
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
FS = "x x x = 3\nx x = 2\nx x x x = 4\n"
WORDS = ['dog', 'star', 'tree', 'book', 'car']
hits, near, n = 0, 0, 0
per_k = {}
for k in range(2, 8):
    kh = 0
    for w in WORDS:
        p = FS + " ".join([w] * k) + " ="
        lp = next_logprobs(p)
        cand = {d: float(lp[D[d]]) for d in range(1, 10)}
        pred = max(cand, key=cand.get)
        n += 1; hits += (pred == k); kh += (pred == k)
        near += (abs(pred - k) <= 1)
    per_k[k] = round(kh/len(WORDS), 3)
res = {'acc': round(hits/n, 3), 'n': n, 'chance': 0.111,
       'within_one_acc': round(near/n, 3), 'per_k': per_k}
verdict = 'YES' if res['acc'] >= 0.5 else ('WEAK' if res['acc'] > 0.25 else 'NO')
res['verdict'] = verdict
print(f"TASK counting acc={res['acc']} n={n} chance=0.111 within_one={res['within_one_acc']} per_k={per_k} verdict={verdict}", flush=True)
json.dump(res, open(f'{QK}/qk_algoverify_counting.json', 'w'), indent=2)
print("DONE", flush=True)
