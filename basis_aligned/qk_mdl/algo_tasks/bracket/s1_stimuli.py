"""Step 1: build 40 clean/corrupted stimulus pairs for parens and 40 for quotes.
Design: clean uses a space-separated opener (" ( " / ' " ') so that the corrupted
token sequence is EXACTLY the clean sequence minus the single opener token
(verified: 'dogs ( which' -> [dogs, Ġ(, Ġwhich]; 'dogs which' -> [dogs, Ġwhich]).
This gives a clean position mapping for activation patching:
  corr pos t  <->  clean pos t          for t <  opener_pos
  corr pos t  <->  clean pos t+1        for t >= opener_pos
Verify behavior (closer-logprob boost clean vs corrupted) on all pairs; also
verify on the original attached-opener style ('(which') for comparison.
Split 30 analysis / 10 held-out per task. Saves stimuli.json.
"""
import json, sys
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, reference_forward
torch.manual_seed(0)
DEV = 'cuda'
OUT = '/workspace/tensor_language/basis_aligned/qk_mdl/algo_tasks/bracket'
tok = AutoTokenizer.from_pretrained('gpt2')
m, cfg = load_elriggs('bilin18')

rng = np.random.RandomState(7)
NOUNS = ['dogs', 'cats', 'birds', 'cars', 'trees', 'books', 'stars', 'rivers',
         'houses', 'chairs', 'apples', 'stones', 'clouds', 'roads', 'lamps',
         'boats', 'fields', 'doors', 'horses', 'flowers', 'windows', 'tables']

PAREN_PRE = ['The', 'Yesterday the', 'In the garden the', 'My friend saw the',
             'We think that the', 'Near the river the']
PAREN_CLAUSE = [
    'which was near the {w1} and the {w2}',
    'which sat beside the {w1}',
    'which the {w1} chased across the {w2} and past the {w3}',
    'not the {w1}',
    'the one near the {w1}, the {w2} and the {w3}',
    'formerly known as the {w1} of the {w2}',
]
QUOTE_PRE = ['She said', 'He whispered', 'The teacher announced',
             'Then John replied', 'My mother shouted', 'Someone wrote']
QUOTE_CONTENT = [
    'the {w0} is near the {w1}',
    'the {w0} will follow the {w1} and the {w2}',
    'bring me the {w0} and the {w1}',
    'the {w0} belongs to the {w1}, not the {w2}',
    'watch the {w0} by the {w1}',
    'the {w0} moved past the {w1} toward the {w2}',
]

def make_pairs(task, n=40):
    pairs = []
    for i in range(n):
        w = rng.choice(NOUNS, 4, replace=False)
        if task == 'paren':
            pre = PAREN_PRE[i % len(PAREN_PRE)]
            cl = PAREN_CLAUSE[(i // len(PAREN_PRE)) % len(PAREN_CLAUSE)]
            body = cl.format(w1=w[1], w2=w[2], w3=w[3])
            clean = f'{pre} {w[0]} ( {body}'
            corr = f'{pre} {w[0]} {body}'
            opener_id = 357  # 'Ġ('
        else:
            pre = QUOTE_PRE[i % len(QUOTE_PRE)]
            ct = QUOTE_CONTENT[(i // len(QUOTE_PRE)) % len(QUOTE_CONTENT)]
            body = ct.format(w0=w[0], w1=w[1], w2=w[2])
            clean = f'{pre} " {body}'
            corr = f'{pre} {body}'
            opener_id = 366  # 'Ġ"'
        ci = tok(clean)['input_ids']
        xi = tok(corr)['input_ids']
        op = ci.index(opener_id)
        assert ci[:op] + ci[op + 1:] == xi, f'misalign pair {i}: {clean}'
        pairs.append({'clean': clean, 'corr': corr, 'clean_ids': ci,
                      'corr_ids': xi, 'opener_pos': op})
    return pairs

@torch.no_grad()
def final_logprobs(ids_list, batch=8):
    """right-pad to common length; return logprobs at each true final position"""
    out = []
    for i in range(0, len(ids_list), batch):
        chunk = ids_list[i:i + batch]
        T = max(len(c) for c in chunk)
        idx = torch.full((len(chunk), T), 50256, dtype=torch.long)
        for j, c in enumerate(chunk):
            idx[j, :len(c)] = torch.tensor(c)
        lg = reference_forward(m, idx.to(DEV)).float()
        for j, c in enumerate(chunk):
            out.append(F.log_softmax(lg[j, len(c) - 1], -1).cpu())
    return out

res = {}
all_pairs = {}
for task, closers in [('paren', {')': 8, ' )': 1267}), ('quote', {'"': 1, ' "': 366})]:
    pairs = make_pairs(task)
    lp_clean = final_logprobs([p['clean_ids'] for p in pairs])
    lp_corr = final_logprobs([p['corr_ids'] for p in pairs])
    # pick the closer variant with the higher mean clean logprob
    stats = {name: float(np.mean([lp[cid].item() for lp in lp_clean]))
             for name, cid in closers.items()}
    closer_name = max(stats, key=stats.get)
    cid = closers[closer_name]
    boosts = [float(lp_clean[i][cid] - lp_corr[i][cid]) for i in range(len(pairs))]
    for i, p in enumerate(pairs):
        p['boost'] = round(boosts[i], 3)
    res[task] = {
        'closer_token': closer_name, 'closer_id': cid,
        'closer_clean_logprob_by_variant': {k: round(v, 3) for k, v in stats.items()},
        'mean_boost': round(float(np.mean(boosts)), 3),
        'min_boost': round(float(np.min(boosts)), 3),
        'frac_positive': round(float(np.mean([b > 0 for b in boosts])), 3),
        'mean_boost_analysis30': round(float(np.mean(boosts[:30])), 3),
        'mean_boost_holdout10': round(float(np.mean(boosts[30:])), 3),
    }
    all_pairs[task] = pairs
    print(task, res[task], flush=True)

# sanity: original attached-opener style, 10 samples each
att = {}
for task in ['paren', 'quote']:
    diffs = []
    for i in range(10):
        w = rng.choice(NOUNS, 3, replace=False)
        if task == 'paren':
            clean = f'The {w[0]} (which was near the {w[1]} and the {w[2]}'
            corr = f'The {w[0]} which was near the {w[1]} and the {w[2]}'
            cid = 8
        else:
            clean = f'She said "the {w[0]} is near the {w[1]}'
            corr = f'She said the {w[0]} is near the {w[1]}'
            cid = 1
        lpc = final_logprobs([tok(clean)['input_ids']])[0]
        lpx = final_logprobs([tok(corr)['input_ids']])[0]
        diffs.append(float(lpc[cid] - lpx[cid]))
    att[task] = round(float(np.mean(diffs)), 3)
res['attached_opener_style_boost'] = att
print('attached-opener style (original probe):', att, flush=True)

json.dump({'summary': res, 'pairs': all_pairs},
          open(f'{OUT}/stimuli.json', 'w'), indent=1)
print('S1 DONE', flush=True)
