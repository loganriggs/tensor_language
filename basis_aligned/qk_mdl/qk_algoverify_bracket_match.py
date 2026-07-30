"""Verify: bracket TYPE matching + depth tracking in bilin18.
Known (qk_algo_probe): "(" boosts ")" strongly. NEW questions here:
  (a) type matching: given an opener in {(,[,{}, does argmax over the three closers {),],}}
      pick the matching one? chance 1/3.
  (b) depth tracking: after "( ... ( ... )" (one still open) is logP(")") higher than after
      "( ... )" fully balanced with same surface words? report mean boost + frac positive.
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

rng = np.random.RandomState(0)
NOUNS = ['red', 'blue', 'green', 'dogs', 'cats', 'birds', 'cars', 'trees', 'books', 'stars']
PAIRS = [('(', ')'), ('[', ']'), ('{', '}')]
CLOSERS = {c: tid(c) for _, c in PAIRS}
res = {}

# (a) type matching: 3 openers x 20 word draws, per-opener breakdown + confusion counts
hits, n, margins = 0, 0, []
per_opener, confusion = {}, {}
for op, cl in PAIRS:
    oh = 0
    for i in range(20):
        w = rng.choice(NOUNS, 3, replace=False)
        p = f"The {w[0]} {op}which was near the {w[1]} and the {w[2]}"
        lp = next_logprobs(p)
        cand = {c: float(lp[t]) for c, t in CLOSERS.items()}
        pred = max(cand, key=cand.get)
        n += 1; hits += (pred == cl); oh += (pred == cl)
        confusion[f'{op}->{pred}'] = confusion.get(f'{op}->{pred}', 0) + 1
        margins.append(cand[cl] - max(v for k, v in cand.items() if k != cl))
    per_opener[op] = round(oh/20, 3)
res['type_match'] = {'acc': round(hits/n, 3), 'n': n, 'chance': 0.333,
                     'per_opener': per_opener, 'confusion': confusion,
                     'mean_margin': round(float(np.mean(margins)), 3)}

# (b) depth tracking: "The red ( dogs ( cats )" [depth1] vs "The red ( dogs cats )" ... balanced ctl
boosts = []
for i in range(20):
    w = rng.choice(NOUNS, 4, replace=False)
    deep = f"The {w[0]} (near the {w[1]} (not the {w[2]}) and the {w[3]}"   # one "(" still open
    flat = f"The {w[0]} (near the {w[1]}, not the {w[2]}) and the {w[3]}"    # balanced
    boosts.append(float(next_logprobs(deep)[CLOSERS[')']] - next_logprobs(flat)[CLOSERS[')']]))
res['depth_track'] = {'mean_logprob_boost': round(float(np.mean(boosts)), 3),
                      'frac_positive': round(float(np.mean([b > 0 for b in boosts])), 3), 'n': 20}

verdict = 'YES' if res['type_match']['acc'] >= 0.667 else ('WEAK' if res['type_match']['acc'] > 0.45 else 'NO')
res['verdict'] = verdict
print(f"TASK bracket_match acc={res['type_match']['acc']} n={res['type_match']['n']} chance=0.333 "
      f"per_opener={res['type_match']['per_opener']} confusion={res['type_match']['confusion']} "
      f"depth_boost={res['depth_track']['mean_logprob_boost']} frac_pos={res['depth_track']['frac_positive']} verdict={verdict}", flush=True)
json.dump(res, open(f'{QK}/qk_algoverify_bracket_match.json', 'w'), indent=2)
print("DONE", flush=True)
