"""Verify: induction / list copying over rare random tokens in bilin18.
Sequence of L unique random mid-frequency tokens, repeated; cut the second occurrence at
position j and ask for full-vocab argmax of the next token (the copy target).
Baseline/control: same first-occurrence prefix WITHOUT the repeat (chance of hitting the
'target' then is ~1/V; we measure it empirically as ctl_acc). Also reports mean logprob
of the target under both conditions.
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

@torch.no_grad()
def next_logprobs_ids(ids):
    idx = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(DEV)
    lg = reference_forward(m, idx)
    return F.log_softmax(lg[0, -1].float(), -1)

rng = np.random.RandomState(0)
L = 8
res = {}
hits, ctl_hits, n = 0, 0, 0
lp_ind, lp_ctl = [], []
for trial in range(30):
    seq = rng.choice(np.arange(2000, 30000), L, replace=False).tolist()  # mid-freq, unique
    j = int(rng.randint(1, L))          # cut point in second occurrence
    target = seq[j]
    ind_ids = seq + seq[:j]             # full first pass + partial second pass
    ctl_ids = seq[:j]                   # first-occurrence prefix only (no repeat)
    lp1 = next_logprobs_ids(ind_ids)
    lp0 = next_logprobs_ids(ctl_ids)
    n += 1
    hits += int(int(lp1.argmax()) == target)
    ctl_hits += int(int(lp0.argmax()) == target)
    lp_ind.append(float(lp1[target])); lp_ctl.append(float(lp0[target]))
acc, ctl_acc = hits/n, ctl_hits/n
res = {'acc': round(acc, 3), 'ctl_acc': round(ctl_acc, 3), 'n': n, 'seq_len': L,
       'chance': round(1/50257, 6),
       'mean_logprob_target_induction': round(float(np.mean(lp_ind)), 3),
       'mean_logprob_target_control': round(float(np.mean(lp_ctl)), 3)}
verdict = 'YES' if acc >= 0.5 and acc > ctl_acc + 0.3 else ('WEAK' if acc > ctl_acc + 0.15 else 'NO')
res['verdict'] = verdict
print(f"TASK induction_copy acc={res['acc']} ctl_acc={res['ctl_acc']} n={n} chance~=0 "
      f"lp_tgt={res['mean_logprob_target_induction']} vs ctl {res['mean_logprob_target_control']} verdict={verdict}", flush=True)
json.dump(res, open(f'{QK}/qk_algoverify_induction_copy.json', 'w'), indent=2)
print("DONE", flush=True)
