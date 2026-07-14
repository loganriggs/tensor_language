"""e9: BatchTopK variant of the e7 dictionaries (Logan's question — e7 used
per-token top-k with exactly k atoms/token; BatchTopK keeps the top k*batch
activations batch-wide, so L0 adapts per token, converted to a global
threshold for deployment).

Configs mirror the e7 frontier points; deployment encoding uses the estimated
threshold, giving variable per-token L0 (padded supports, zero-coeff padding).
Audits: FVU, swapped-in dCE, mean/max L0, and CE-finetune for n=1024.
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
E = e6.E
V, D_MODEL = E.shape
torch.manual_seed(0)


def train_batchtopk(n, k, steps=4000, batch=8192, lr=3e-3, seed=0):
    g = torch.Generator(device='cpu'); g.manual_seed(seed)
    idx0 = torch.randperm(V, generator=g)[:n]
    Dm = E[idx0].clone()
    Dm = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
    We = Dm.clone()
    b_pre = E.mean(0).clone()
    b = E.mean(0).clone()
    for t in [Dm, We, b_pre, b]:
        t.requires_grad_(True)
    opt = torch.optim.Adam([Dm, We, b_pre, b], lr=lr)
    usage = torch.zeros(n, device=DEV)
    thresh_ema = None

    for step in range(steps):
        x = E[torch.randint(0, V, (batch,), device=DEV)]
        Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
        z = (x - b_pre) @ We.T
        # batch-topk on |z|: keep top k*batch entries batch-wide
        flat = z.abs().flatten()
        kth = flat.kthvalue(flat.numel() - k * batch + 1).values
        mask = z.abs() >= kth
        zs = z * mask
        xhat = b + zs @ Dn
        loss = ((xhat - x) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            usage += mask.float().sum(0)
            thresh_ema = kth.item() if thresh_ema is None else \
                0.98 * thresh_ema + 0.02 * kth.item()
        if step % 500 == 499:
            with torch.no_grad():
                dead = (usage == 0).nonzero().flatten()
                if len(dead):
                    Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
                    sample = E[torch.randint(0, V, (4096,), device=DEV)]
                    zz = (sample - b_pre) @ We.T
                    m = zz.abs() >= thresh_ema
                    err = ((b + (zz * m) @ Dn - sample) ** 2).sum(1)
                    worst = sample[err.topk(min(len(dead), 4096)).indices]
                    w = worst[:len(dead)] - b
                    w = w / w.norm(dim=1, keepdim=True).clamp(min=1e-8)
                    Dm.data[dead[:len(w)]] = w
                    We.data[dead[:len(w)]] = w
                usage.zero_()

    with torch.no_grad():
        Dn = (Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)).detach()
        z = (E - b_pre) @ We.T
        mask = z.abs() >= thresh_ema
        L0 = mask.sum(1)
        Lmax = int(L0.max())
        vals = (z * mask)
        top = vals.abs().topk(Lmax, dim=1)
        supports = top.indices
        coeffs = torch.gather(vals, 1, supports)  # zeros beyond each token's L0
        Ehat = b + coeffs.unsqueeze(-1).mul(Dn[supports]).sum(1)
    state = {'D': Dn.cpu(), 'supports': supports.cpu(), 'coeffs': coeffs.detach().cpu(),
             'b': b.detach().cpu()}
    stats = {'mean_L0': float(L0.float().mean()), 'max_L0': Lmax,
             'frac_zero_L0': float((L0 == 0).float().mean()),
             'threshold': thresh_ema}
    return Ehat, state, stats


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


if __name__ == '__main__':
    results = {'rows': []}
    print('baseline CE (fp16 model, e6 convention)...')
    CE0 = e6.eval_ce()
    results['baseline_ce'] = CE0

    states = {}
    for n, k in [(1024, 64), (4096, 64), (16384, 16)]:
        Ehat, state, stats = train_batchtopk(n, k)
        row = {'method': 'batchtopk', 'n_atoms': n, 'k_train': k, **stats,
               'fvu': e6.fvu(Ehat), 'dce': e6.eval_ce(Ehat) - CE0}
        results['rows'].append(row)
        states[(n, k)] = state
        print(f"n={n:6d} k={k:3d}  meanL0 {stats['mean_L0']:6.1f} "
              f"maxL0 {stats['max_L0']:4d}  fvu {row['fvu']:.4f}  "
              f"dCE {row['dce']:+.4f}", flush=True)
        del Ehat
        torch.cuda.empty_cache()

    # CE-finetune the n=1024 variant (compare to fixed-topk's +0.26)
    print('=== CE-finetune batchtopk n=1024')
    e6.model.to(torch.bfloat16)
    CE0b = eval_ce_current()
    TRAIN = e6.build_eval_tokens(n_chunks=64 + 512)[64:].to(DEV)
    for p in e6.model.parameters():
        p.requires_grad_(False)
    st = states[(1024, 64)]
    de = DictEmbed(st['supports'].to(DEV), st['coeffs'].float().to(DEV),
                   st['D'].float().to(DEV), st['b'].float().to(DEV)).to(DEV)
    orig = e6.model.gpt_neox.embed_in
    e6.model.gpt_neox.embed_in = de
    print(f'  before: dCE {eval_ce_current() - CE0b:+.4f}')
    opt = torch.optim.Adam(de.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1500)
    g = torch.Generator(device='cpu'); g.manual_seed(0)
    for step in range(1500):
        batch = TRAIN[torch.randint(0, len(TRAIN), (8,), generator=g)]
        logits = e6.model(batch[:, :-1]).logits.float()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               batch[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(de.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 300 == 0:
            print(f'  step {step:5d}  train CE {loss.item():.4f}', flush=True)
    dce_after = eval_ce_current() - CE0b
    results['ceft_n1024'] = {'dce_after': dce_after}
    print(f'  after: dCE {dce_after:+.4f}')
    e6.model.gpt_neox.embed_in = orig

    with open(f'{BASE}/e9_results.json', 'w') as fh:
        json.dump(results, fh, indent=2)
    print('e9 done')
