"""Verify: decade-crossing increment edge cases in bilin18 numbered lists.
Known (qk_algo_probe + algo_tasks/increment): within-decade increment (3.->4.) is 100%.
NEW: does the succession machine carry across decades? "8. x\n9. y\n" -> "10" ;
"18./19." -> "20" ; "28./29." -> "30" ; "98./99." -> "100". Line-start numbers are
single GPT-2 tokens without leading space. Argmax restricted to plausible number
candidates {correct, same-decade wrong ones, digit-echo}; chance ~1/6. Also reports
within-decade controls measured the same restricted way for comparability.
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

def probe(n1, n2, correct, distractors, reps=5):
    """list lines n1., n2. -> next line number; restricted argmax."""
    hits = 0
    for i in range(reps):
        w = rng.choice(NOUNS, 2, replace=False)
        p = f"{n1}. {w[0]}\n{n2}. {w[1]}\n"
        lp = next_logprobs(p)
        cand = {c: float(lp[tid(str(c))]) for c in [correct] + distractors}
        hits += (max(cand, key=cand.get) == correct)
    return hits / reps

res = {'carry': {}, 'control': {}}
# carry cases: 9->10, 19->20, 29->30, 99->100
CARRY = [(8, 9, 10, [9, 11, 1, 8, 20]), (18, 19, 20, [19, 21, 1, 18, 10]),
         (28, 29, 30, [29, 31, 2, 28, 20]), (98, 99, 100, [99, 9, 1, 98, 10])]
ch, cn = 0, 0
for n1, n2, cor, dis in CARRY:
    a = probe(n1, n2, cor, dis)
    res['carry'][f'{n2}->{cor}'] = round(a, 3); ch += a * 5; cn += 5
# within-decade controls at comparable magnitudes: 11->12..., 21->22, 95->96
CTRL = [(10, 11, 12, [11, 13, 1, 10, 20]), (20, 21, 22, [21, 23, 2, 20, 12]),
        (94, 95, 96, [95, 97, 9, 94, 86])]
th, tn = 0, 0
for n1, n2, cor, dis in CTRL:
    a = probe(n1, n2, cor, dis)
    res['control'][f'{n2}->{cor}'] = round(a, 3); th += a * 5; tn += 5
res['carry_acc'] = round(ch/cn, 3); res['control_acc'] = round(th/tn, 3)
res['n_carry'] = cn; res['n_control'] = tn; res['chance'] = round(1/6, 3)
verdict = 'YES' if res['carry_acc'] >= 0.667 else ('WEAK' if res['carry_acc'] > 0.4 else 'NO')
res['verdict'] = verdict
print(f"TASK increment_carry carry_acc={res['carry_acc']} control_acc={res['control_acc']} "
      f"chance=0.167 per_case={res['carry']} verdict={verdict}", flush=True)
json.dump(res, open(f'{QK}/qk_algoverify_increment_carry.json', 'w'), indent=2)
print("DONE", flush=True)
