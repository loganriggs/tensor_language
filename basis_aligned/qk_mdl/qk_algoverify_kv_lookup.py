"""Verify: in-context key-value lookup in bilin18, in two formats.
Format A (semantic): "John has a dog. Mary has a cat. Tom has a bird. ... <Name> has a"
  -> value token. Requires binding name->object across a sentence boundary.
Format B (literal-induction): "x = 4, y = 7, z = 2. y =" -> " 7". The query repeats the
  literal bigram "<key> =", so plain induction (match previous occurrence, copy successor)
  suffices. If B works and A fails, kv-lookup reduces to induction, not semantic binding.
Both: argmax restricted to the three context values; chance 1/3; query each key position
equally; per-position accuracy reported (recency check).
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

NAMES = ['John', 'Mary', 'Tom', 'Sarah', 'Peter', 'Anna']
ANIMALS = ['dog', 'cat', 'bird', 'fish', 'horse', 'mouse']
KEYS = ['x', 'y', 'z', 'a', 'b', 'c']
rng = np.random.RandomState(0)
res = {}

def run_format(fmt):
    hits, n = 0, 0
    per_pos = [0, 0, 0]; per_pos_n = [0, 0, 0]; margins = []
    for trial in range(30):
        qpos = trial % 3
        if fmt == 'A':
            names = rng.choice(NAMES, 3, replace=False)
            vals = rng.choice(ANIMALS, 3, replace=False)
            ctx = " ".join(f"{nm} has a {v}." for nm, v in zip(names, vals))
            p = f"{ctx} {names[qpos]} has a"
            cand_tok = {v: tid(' ' + v) for v in vals}
        else:
            keys = rng.choice(KEYS, 3, replace=False)
            vals = rng.choice(np.arange(1, 10), 3, replace=False)
            ctx = ", ".join(f"{k} = {v}" for k, v in zip(keys, vals))
            p = f"{ctx}. {keys[qpos]} ="
            cand_tok = {int(v): tid(' ' + str(v)) for v in vals}
        lp = next_logprobs(p)
        cand = {v: float(lp[t]) for v, t in cand_tok.items()}
        pred = max(cand, key=cand.get)
        tgt = list(cand_tok)[qpos]
        ok = (pred == tgt)
        hits += ok; n += 1; per_pos[qpos] += int(ok); per_pos_n[qpos] += 1
        margins.append(cand[tgt] - max(v for k, v in cand.items() if k != tgt))
    return {'acc': round(hits/n, 3), 'n': n, 'chance': 0.333,
            'per_position': [round(h/max(c, 1), 3) for h, c in zip(per_pos, per_pos_n)],
            'mean_margin': round(float(np.mean(margins)), 3)}

res['semantic_A'] = run_format('A')
res['literal_B'] = run_format('B')
accA, accB = res['semantic_A']['acc'], res['literal_B']['acc']
vd = lambda a: 'YES' if a >= 0.667 else ('WEAK' if a > 0.45 else 'NO')
res['verdict_semantic'] = vd(accA); res['verdict_literal'] = vd(accB)
verdict = res['verdict_literal']  # headline: can it do kv-lookup in ANY format
res['verdict'] = verdict
print(f"TASK kv_lookup semantic_acc={accA} ({res['verdict_semantic']}) literal_acc={accB} ({res['verdict_literal']}) "
      f"chance=0.333 per_pos_A={res['semantic_A']['per_position']} per_pos_B={res['literal_B']['per_position']}", flush=True)
json.dump(res, open(f'{QK}/qk_algoverify_kv_lookup.json', 'w'), indent=2)
print("DONE", flush=True)
