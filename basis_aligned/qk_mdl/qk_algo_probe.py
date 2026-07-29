"""Step 1 of Logan's algorithmic-circuits plan: verify which algorithmic behaviors bilin18 actually
has. Tasks: single-digit addition, closing parenthesis, closing quote, numbered-list increment,
alphabet continuation, weekday continuation, month continuation, 3-element sorting (few-shot).
Metric per task: top-1 accuracy over generated stimuli + mean logprob margin of the correct token
over the strongest distractor class; chance level stated per task.
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
V = cfg['vocab_size']
tok = AutoTokenizer.from_pretrained('gpt2')
enc = lambda s: tok(s, return_tensors='pt')['input_ids'][0]

@torch.no_grad()
def next_logprobs(prompt):
    idx = enc(prompt).unsqueeze(0).to(DEV)
    lg = reference_forward(m, idx)
    return F.log_softmax(lg[0, -1].float(), -1)

def tid(s):
    t = tok(s)['input_ids']
    return t[0] if len(t) == 1 else None

rng = np.random.RandomState(0)
NOUNS = ['red', 'blue', 'green', 'dogs', 'cats', 'birds', 'cars', 'trees', 'books', 'stars']
res = {}

# 1. addition (single-token sums 2-9)
hits, margins, n = 0, [], 0
for a in range(1, 9):
    for b in range(1, 9):
        if a + b > 9: continue
        lp = next_logprobs(f"{a} + {b} = ")
        cand = {d: lp[tid(str(d))] for d in range(1, 10)}
        pred = max(cand, key=cand.get); n += 1; hits += (pred == a+b)
        margins.append(float(cand[a+b] - max(v for k, v in cand.items() if k != a+b)))
res['addition'] = {'acc': round(hits/n, 3), 'n': n, 'chance': round(1/9, 3), 'margin': round(float(np.mean(margins)), 3)}

# 2. closing parenthesis: differential P(")") with vs without "("
diffs = []
for i in range(20):
    w = rng.choice(NOUNS, 3, replace=False)
    base = f"The {w[0]} (which was near the {w[1]} and the {w[2]}"
    ctl = f"The {w[0]} which was near the {w[1]} and the {w[2]}"
    diffs.append(float(next_logprobs(base)[tid(")")] - next_logprobs(ctl)[tid(")")]))
res['paren_close'] = {'mean_logprob_boost': round(float(np.mean(diffs)), 3), 'frac_positive': round(float(np.mean([d > 0 for d in diffs])), 3)}

# 3. closing quote
diffs = []
for i in range(20):
    w = rng.choice(NOUNS, 2, replace=False)
    base = f'She said "the {w[0]} is near the {w[1]}'
    ctl = f'She said the {w[0]} is near the {w[1]}'
    diffs.append(float(next_logprobs(base)[tid('"')] - next_logprobs(ctl)[tid('"')]))
res['quote_close'] = {'mean_logprob_boost': round(float(np.mean(diffs)), 3), 'frac_positive': round(float(np.mean([d > 0 for d in diffs])), 3)}

# 4. numbered list increment: "1. x\n2. y\n" -> "3"
hits, n = 0, 0
for start in range(1, 7):
    for i in range(6):
        w = rng.choice(NOUNS, 2, replace=False)
        p = f"{start}. {w[0]}\n{start+1}. {w[1]}\n"
        lp = next_logprobs(p)
        cand = {d: lp[tid(str(d))] for d in range(1, 10)}
        n += 1; hits += (max(cand, key=cand.get) == start+2)
res['list_increment'] = {'acc': round(hits/n, 3), 'n': n, 'chance': round(1/9, 3)}

# 5. alphabet continuation "a, b, c," -> " d"
letters = 'abcdefghijklmnopqrstuvwxyz'
hits, n = 0, 0
for s in range(0, 20):
    p = f"{letters[s]}, {letters[s+1]}, {letters[s+2]},"
    lp = next_logprobs(p)
    cand = {c: lp[tid(' '+c)] for c in letters if tid(' '+c) is not None}
    n += 1; hits += (max(cand, key=cand.get) == letters[s+3])
res['alphabet'] = {'acc': round(hits/n, 3), 'n': n, 'chance': round(1/26, 3)}

# 6. weekday continuation
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
hits, n = 0, 0
for s in range(7):
    p = f"{days[s%7]}, {days[(s+1)%7]}, {days[(s+2)%7]},"
    lp = next_logprobs(p)
    cand = {d: lp[tid(' '+d)] for d in days}
    n += 1; hits += (max(cand, key=cand.get) == days[(s+3)%7])
res['weekday'] = {'acc': round(hits/n, 3), 'n': n, 'chance': round(1/7, 3)}

# 7. month continuation
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
hits, n = 0, 0
for s in range(12):
    p = f"{months[s%12]}, {months[(s+1)%12]}, {months[(s+2)%12]},"
    lp = next_logprobs(p)
    cand = {mo: lp[tid(' '+mo)] for mo in months if tid(' '+mo) is not None}
    n += 1; hits += (max(cand, key=cand.get) == months[(s+3)%12])
res['month'] = {'acc': round(hits/n, 3), 'n': n, 'chance': round(1/12, 3)}

# 8. 3-element sort, few-shot
hits, n = 0, 0
for i in range(24):
    perm = rng.permutation([1, 2, 3]) + rng.randint(0, 6)
    lo = min(perm)
    p = "5 3 4 -> 3 4 5\n8 6 7 -> 6 7 8\n" + " ".join(map(str, perm)) + " ->"
    lp = next_logprobs(p)
    cand = {d: lp[tid(' '+str(d))] for d in range(1, 10)}
    n += 1; hits += (max(cand, key=cand.get) == lo)
res['sort3_fewshot'] = {'acc': round(hits/n, 3), 'n': n, 'chance': 0.33}

for k, v in res.items(): print(k, v, flush=True)
json.dump(res, open(f'{QK}/qk_algo_probe.json', 'w'), indent=2)
print("QK ALGO PROBE DONE", flush=True)
