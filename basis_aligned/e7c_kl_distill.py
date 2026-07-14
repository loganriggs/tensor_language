"""e7c: KL-distillation variant of stage B, answering "did you train on the
original model's distribution or on ground-truth data?"

e7b trained on ground-truth pile-10k CE, which allows the compressed embedding
to REPAIR/adapt rather than stay faithful. Here the loss is cross-entropy to the
ORIGINAL model's output distribution (teacher = same model with the original
embedding), so the objective is faithfulness: represent the same function with
fewer objects.

Both audits reported for MSE-fit (before) and KL-trained (after):
  dce_data : delta-CE vs ground truth on the held-out eval chunks
  kl_orig  : mean KL(original || compressed) per token on held-out chunks
"""

import json
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language/basis_aligned')
import e6_embedding_objects as e6

DEV = 'cuda'
BASE = '/workspace/tensor_language/basis_aligned'
V, D_MODEL = e6.E.shape

res = json.load(open(f'{BASE}/e7_results.json'))

print('converting model to bfloat16...')
e6.model.to(torch.bfloat16)
print('building train tokens (disjoint from eval)...')
TRAIN = e6.build_eval_tokens(n_chunks=64 + 512)[64:].to(DEV)
ORIG_EMBED = e6.model.gpt_neox.embed_in


class DictEmbed(nn.Module):
    def __init__(self, supports, coeffs, D, b):
        super().__init__()
        self.register_buffer('supports', supports)
        self.coeffs = nn.Parameter(coeffs.clone())
        self.D = nn.Parameter(D.clone())
        self.b = nn.Parameter(b.clone())

    def forward(self, ids):
        atoms = self.D[self.supports[ids]]
        out = self.b + (self.coeffs[ids].unsqueeze(-1) * atoms).sum(-2)
        return out.to(torch.bfloat16)


@torch.no_grad()
def eval_ce_current():
    tot, n = 0.0, 0
    for i in range(0, len(e6.TOKENS), 8):
        batch = e6.TOKENS[i:i + 8]
        logits = e6.model(batch[:, :-1]).logits.float()
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                             batch[:, 1:].reshape(-1))
        tot += ce.item() * batch.numel()
        n += batch.numel()
    return tot / n


CE0 = eval_ce_current()
res['baseline_ce_bf16'] = CE0
print(f'bf16 baseline CE: {CE0:.4f}')


def teacher_ce_chunked(student_logits, teacher_logits, chunk=2048):
    """-(p_teacher * log_softmax(student)).sum(-1).mean(), chunked in fp32."""
    s = student_logits.reshape(-1, student_logits.shape[-1])
    t = teacher_logits.reshape(-1, teacher_logits.shape[-1])
    total = 0.0
    n = s.shape[0]
    for i in range(0, n, chunk):
        p_t = F.softmax(t[i:i + chunk].float(), dim=-1)
        total = total + -(p_t * F.log_softmax(s[i:i + chunk].float(), dim=-1)
                          ).sum(-1).sum()
    return total / n


@torch.no_grad()
def eval_kl_current(de):
    """Mean KL(original || current DictEmbed model) per token, held-out."""
    tot, n = 0.0, 0
    for i in range(0, len(e6.TOKENS), 4):
        batch = e6.TOKENS[i:i + 4]
        e6.model.gpt_neox.embed_in = ORIG_EMBED
        t = e6.model(batch[:, :-1]).logits
        e6.model.gpt_neox.embed_in = de
        s = e6.model(batch[:, :-1]).logits
        tf = t.reshape(-1, t.shape[-1])
        sf = s.reshape(-1, s.shape[-1])
        for j in range(0, tf.shape[0], 2048):
            p_t = F.log_softmax(tf[j:j + 2048].float(), dim=-1)
            p_s = F.log_softmax(sf[j:j + 2048].float(), dim=-1)
            tot += (p_t.exp() * (p_t - p_s)).sum().item()
        n += tf.shape[0]
    return tot / n


def kl_finetune(supports, coeffs, D, b, steps=1500, lr=1e-4):
    for p in e6.model.parameters():
        p.requires_grad_(False)
    de = DictEmbed(supports.to(DEV), coeffs.float().to(DEV),
                   D.float().to(DEV), b.float().to(DEV)).to(DEV)
    e6.model.gpt_neox.embed_in = de
    before = {'dce_data': eval_ce_current() - CE0, 'kl_orig': eval_kl_current(de)}
    opt = torch.optim.Adam(de.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    g = torch.Generator(device='cpu'); g.manual_seed(0)
    run = None
    for step in range(steps):
        batch = TRAIN[torch.randint(0, len(TRAIN), (8,), generator=g)]
        with torch.no_grad():
            e6.model.gpt_neox.embed_in = ORIG_EMBED
            t_logits = e6.model(batch[:, :-1]).logits
        e6.model.gpt_neox.embed_in = de
        s_logits = e6.model(batch[:, :-1]).logits
        loss = teacher_ce_chunked(s_logits, t_logits)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(de.parameters(), 1.0)
        opt.step(); sched.step()
        run = loss.item() if run is None else 0.95 * run + 0.05 * loss.item()
        if step % 150 == 0:
            print(f'  step {step:5d}  teacher-CE (ema) {run:.4f}', flush=True)
    after = {'dce_data': eval_ce_current() - CE0, 'kl_orig': eval_kl_current(de)}
    e6.model.gpt_neox.embed_in = ORIG_EMBED
    torch.cuda.empty_cache()
    return before, after


for n, k in [(1024, 64), (4096, 64)]:
    print(f'=== KL-distill topk n={n} k={k}')
    st = torch.load(f'{BASE}/e7_dict_n{n}_k{k}.pt')
    before, after = kl_finetune(st['supports'], st['coeffs'], st['D'], st['b'])
    row = {'method': 'topk_dict_klft', 'n_atoms': n, 'k': k,
           'before': before, 'after': after}
    res['rows'].append(row)
    print(f"n={n}: dCE_data {before['dce_data']:+.4f} -> {after['dce_data']:+.4f}   "
          f"KL(orig||comp) {before['kl_orig']:.4f} -> {after['kl_orig']:.4f}",
          flush=True)
    with open(f'{BASE}/e7_results.json', 'w') as fh:
        json.dump(res, fh, indent=2)
print('e7c done')
