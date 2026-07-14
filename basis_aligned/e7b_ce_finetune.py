"""e7 stage B: CE-finetune the frontier dictionaries through the frozen model.

For each selected config: freeze each token's atom SUPPORT (which atoms it uses),
make the dictionary atoms, coefficients, and bias trainable, replace embed_in
with a module that assembles token embeddings on the fly, and train on pile-10k
CE (chunks disjoint from the e6/e7 eval set). This answers how much of the
compressed-embedding damage is metric mismatch (Frobenius-fit codebooks) vs
genuine incompressibility.

Configs: per-n best stage-A dictionary + the kmeans-25.6k corner (k=1).
"""

import json
import sys

import torch
import torch.nn as nn

sys.path.insert(0, '/workspace/tensor_language/basis_aligned')
import e6_embedding_objects as e6

DEV = 'cuda'
BASE = '/workspace/tensor_language/basis_aligned'
V, D_MODEL = e6.E.shape
torch.manual_seed(0)

res = json.load(open(f'{BASE}/e7_results.json'))
# drop rows from the diverged fp16 run (train CE rose; retracted)
res['rows'] = [r for r in res['rows'] if not r['method'].endswith('_ceft')]

print('converting model to bfloat16 (fp16 backward diverged without scaling)...')
e6.model.to(torch.bfloat16)
CE0 = None  # recomputed in bf16 below for a consistent baseline

print('building train tokens (disjoint from eval)...')
ALL = e6.build_eval_tokens(n_chunks=64 + 512)
TRAIN = ALL[64:].to(DEV)  # eval set is the first 64 chunks
print(f'train chunks: {tuple(TRAIN.shape)}')


class DictEmbed(nn.Module):
    def __init__(self, supports, coeffs, D, b):
        super().__init__()
        self.register_buffer('supports', supports)
        self.coeffs = nn.Parameter(coeffs.clone())
        self.D = nn.Parameter(D.clone())
        self.b = nn.Parameter(b.clone())

    def forward(self, ids):
        atoms = self.D[self.supports[ids]]                     # (..., k, d)
        out = self.b + (self.coeffs[ids].unsqueeze(-1) * atoms).sum(-2)
        return out.to(torch.bfloat16)


@torch.no_grad()
def eval_ce_current():
    tot, n = 0.0, 0
    for i in range(0, len(e6.TOKENS), 8):
        batch = e6.TOKENS[i:i + 8]
        logits = e6.model(batch[:, :-1]).logits.float()
        ce = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1))
        tot += ce.item() * batch.numel()
        n += batch.numel()
    return tot / n


def ce_finetune(supports, coeffs, D, b, steps=1500, lr=1e-4, log_every=100):
    orig_embed = e6.model.gpt_neox.embed_in
    for p in e6.model.parameters():
        p.requires_grad_(False)
    de = DictEmbed(supports.to(DEV), coeffs.float().to(DEV),
                   D.float().to(DEV), b.float().to(DEV)).to(DEV)
    e6.model.gpt_neox.embed_in = de
    ce_before = eval_ce_current()
    opt = torch.optim.Adam(de.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    g = torch.Generator(device='cpu'); g.manual_seed(0)
    run = None
    for step in range(steps):
        batch = TRAIN[torch.randint(0, len(TRAIN), (8,), generator=g)]
        logits = e6.model(batch[:, :-1]).logits.float()
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(de.parameters(), 1.0)
        opt.step(); sched.step()
        run = loss.item() if run is None else 0.95 * run + 0.05 * loss.item()
        if step % log_every == 0:
            print(f'  step {step:5d}  train CE (ema) {run:.4f}', flush=True)
    ce_after = eval_ce_current()
    e6.model.gpt_neox.embed_in = orig_embed
    torch.cuda.empty_cache()
    return ce_before, ce_after


CE0 = eval_ce_current()  # bf16 baseline with the ORIGINAL embedding
res['baseline_ce_bf16'] = CE0
print(f'bf16 baseline CE: {CE0:.4f}')

# ---- config selection: per-n best stage-A dCE + kmeans corner
by_n = {}
for r in res['rows']:
    if r['method'] == 'topk_dict':
        if r['n_atoms'] not in by_n or r['dce'] < by_n[r['n_atoms']]['dce']:
            by_n[r['n_atoms']] = r

jobs = []
for n, r in sorted(by_n.items()):
    st = torch.load(f"{BASE}/e7_dict_n{n}_k{r['k']}.pt")
    jobs.append((f"topk n={n} k={r['k']}",
                 dict(method='topk_dict_ceft', n_atoms=n, k=r['k']),
                 (st['supports'], st['coeffs'], st['D'], st['b'])))

print('building kmeans-25.6k corner...')
C, assign = e6.kmeans(e6.E, 25600)
jobs.append(('kmeans n=25.6k k=1',
             dict(method='kmeans_ceft', n_atoms=25600, k=1),
             (assign[:, None].cpu(), torch.ones(V, 1), C.cpu(),
              torch.zeros(D_MODEL))))

for label, meta, (supports, coeffs, D, b) in jobs:
    print(f'=== CE-finetune {label}')
    ce_before, ce_after = ce_finetune(supports, coeffs, D, b)
    row = {**meta, 'ce_before': ce_before, 'ce_after': ce_after,
           'dce_before': ce_before - CE0, 'dce_after': ce_after - CE0}
    res['rows'].append(row)
    print(f"{label}: dCE {row['dce_before']:+.4f} -> {row['dce_after']:+.4f}",
          flush=True)
    with open(f'{BASE}/e7_results.json', 'w') as fh:
        json.dump(res, fh, indent=2)
print('stage B done')
