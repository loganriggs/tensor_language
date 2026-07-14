"""e10: Logan's claim — the behavioral audit should be recoverable from WEIGHTS
ALONE via a tensor-sim-style metric.

The embedding's input measure is trivially known (one-hots), so a function-space
inner product between E and Ehat reduces to an OUTPUT-side metric: weight the
row errors by what downstream weights read from the residual stream.

  M = sum over readers of (W diag(ln_gain))^T (W diag(ln_gain)) / trace,
  readers = every matrix consuming the residual stream (per layer: attention
  QKV and MLP h->4h with their input-LN gains; plus the final unembedding with
  its LN gain).

  weighted FVU: wfvu(Ehat) = tr(dE M dE^T) / tr((E-rowmean) M (E-rowmean)^T)

This is the linear-reader / first-order version of tensor similarity (LayerNorm
normalization and depth effects ignored); the exact Isserlis machinery would
apply to bilinear models.

Part 1: recompute the e6/e7 perturbations whose dCE we already measured, and
        test whether wfvu predicts dCE where plain FVU catastrophically failed
        (noise-vs-deletion at matched FVU; k=1 vs k=64 dicts at matched FVU).
Part 2: refit the n=1024/k=64 dictionary under M-weighted Frobenius — pure
        weight-based optimization, no data — and CE-eval it. How much of the
        (+2.11 MSE-fit -> +0.26 CE-trained) gap does the weight metric close?
"""

import json
import sys

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned')
import e6_embedding_objects as e6

DEV = 'cuda'
BASE = '/workspace/tensor_language/basis_aligned'
E, V, D = e6.E, *e6.E.shape
torch.manual_seed(0)

# ---------------------------------------------------------------- the metric

def build_metric():
    M = torch.zeros(D, D, device=DEV)
    n_readers = 0

    def add(W, gain):
        nonlocal M, n_readers
        Wg = (W.float() * gain.float()[None, :])
        G = Wg.T @ Wg
        M += G / G.trace()
        n_readers += 1

    for layer in e6.model.gpt_neox.layers:
        add(layer.attention.query_key_value.weight.detach(),
            layer.input_layernorm.weight.detach())
        add(layer.mlp.dense_h_to_4h.weight.detach(),
            layer.post_attention_layernorm.weight.detach())
    add(e6.model.embed_out.weight.detach(),
        e6.model.gpt_neox.final_layer_norm.weight.detach())
    return M / n_readers


M = build_metric()
evals, evecs = torch.linalg.eigh(M.double())
MS = (evecs @ torch.diag(evals.clamp(min=0).sqrt()) @ evecs.T).float()  # M^{1/2}
MS = MS / (MS ** 2).mean().sqrt()  # scale-normalize for training stability
CENT = E - e6.ROWMEAN
WDENOM = ((CENT @ MS) ** 2).sum().item()


def wfvu(Ehat):
    return float((((E - Ehat) @ MS) ** 2).sum().item() / WDENOM)


# ---------------------------------------------------------------- part 1

r6 = json.load(open(f'{BASE}/e6_results.json'))
r7 = json.load(open(f'{BASE}/e7_results.json'))


def dce_of(method, label=None, n=None, k=None):
    for row in r6['rows']:
        if row['method'] == method and (label is None or row['label'] == label):
            return row['dce']
    for row in r7['rows']:
        if row['method'] == method and row.get('n_atoms') == n and row.get('k') == k:
            return row['dce']
    raise KeyError((method, label, n, k))


points = []

def add_point(name, Ehat, dce):
    points.append({'name': name, 'fvu': e6.fvu(Ehat), 'wfvu': wfvu(Ehat), 'dce': dce})
    p = points[-1]
    print(f"{name:24s} fvu {p['fvu']:.4f}  wfvu {p['wfvu']:.4f}  dCE {p['dce']:+.3f}",
          flush=True)

for r, label in [(19, 'r=19'), (100, 'r=100'), (512, 'r=512')]:
    add_point(f'svd {label}', e6.arm_svd(r)[0], dce_of('svd', label))
add_point('svd random r=100', e6.arm_svd(100, random_basis=True)[0],
          dce_of('svd_random', 'r=100'))
add_point('kmeans n=1k', e6.arm_kmeans(1024)[0], dce_of('kmeans', 'n=1k'))
add_point('kmeans n=25k', e6.arm_kmeans(25600)[0], dce_of('kmeans', 'n=25k'))
add_point('rq c=1k h=5', e6.arm_rq(1024, 5)[0], dce_of('rq', 'c=1k,h=5'))

# noise: replicate e6c's exact RNG sequence
rms = ((E - e6.ROWMEAN) ** 2).mean().sqrt()
for target in [0.1, 0.32, 0.75, 1.5, 3.0]:
    torch.manual_seed(1)
    Ehat = E + rms * target ** 0.5 * torch.randn_like(E)
    dce = next(r['dce'] for r in r6['rows']
               if r['method'] == 'noise' and abs(r['fvu'] - target) < 0.05)
    add_point(f'noise fvu~{target}', Ehat, dce)
    del Ehat
    torch.cuda.empty_cache()

# learned dicts from saved states (reconstruct Ehat)
for n, k in [(1024, 64), (16384, 64), (32768, 1), (16384, 1)]:
    st = torch.load(f'{BASE}/e7_dict_n{n}_k{k}.pt')
    Dm, sup, cf, b = [st[x].to(DEV) for x in ['D', 'supports', 'coeffs', 'b']]
    Ehat = torch.empty_like(E)
    for i in range(0, V, 4096):
        Ehat[i:i + 4096] = b + cf[i:i + 4096].unsqueeze(-1).mul(Dm[sup[i:i + 4096]]).sum(1)
    add_point(f'dict n={n} k={k}', Ehat, dce_of('topk_dict', n=n, k=k))
    del Ehat
    torch.cuda.empty_cache()

import math
def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        for pos, i in enumerate(order):
            rk[i] = pos
        return rk
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den

fvus = [p['fvu'] for p in points]
wfvus = [p['wfvu'] for p in points]
dces = [p['dce'] for p in points]
sp_f, sp_w = spearman(fvus, dces), spearman(wfvus, dces)
print(f'\nSpearman(fvu, dCE)  = {sp_f:+.3f}')
print(f'Spearman(wfvu, dCE) = {sp_w:+.3f}')

# ---------------------------------------------------------------- part 2

def train_topk_dict_weighted(n, k, steps=4000, batch=8192, lr=3e-3, seed=0):
    g = torch.Generator(device='cpu'); g.manual_seed(seed)
    idx0 = torch.randperm(V, generator=g)[:n]
    Dm = E[idx0].clone(); Dm = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
    We = Dm.clone()
    b_pre = E.mean(0).clone(); b = E.mean(0).clone()
    for t in [Dm, We, b_pre, b]:
        t.requires_grad_(True)
    opt = torch.optim.Adam([Dm, We, b_pre, b], lr=lr)
    usage = torch.zeros(n, device=DEV)
    for step in range(steps):
        x = E[torch.randint(0, V, (batch,), device=DEV)]
        Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
        z = (x - b_pre) @ We.T
        vals, idx = z.abs().topk(k, dim=1)
        coeff = torch.gather(z, 1, idx)
        xhat = b + (coeff.unsqueeze(-1) * Dn[idx]).sum(1)
        loss = (((xhat - x) @ MS) ** 2).mean()   # <-- the ONLY change vs e7
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            usage.index_add_(0, idx.flatten(), torch.ones(idx.numel(), device=DEV))
        if step % 500 == 499:
            with torch.no_grad():
                dead = (usage == 0).nonzero().flatten()
                if len(dead):
                    Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
                    sample = E[torch.randint(0, V, (4096,), device=DEV)]
                    z = (sample - b_pre) @ We.T
                    v_, i_ = z.abs().topk(k, dim=1)
                    xh = b + (torch.gather(z, 1, i_).unsqueeze(-1) * Dn[i_]).sum(1)
                    err = (((xh - sample) @ MS) ** 2).sum(1)
                    worst = sample[err.topk(min(len(dead), 4096)).indices]
                    w = worst[:len(dead)] - b
                    w = w / w.norm(dim=1, keepdim=True).clamp(min=1e-8)
                    Dm.data[dead[:len(w)]] = w
                    We.data[dead[:len(w)]] = w
                usage.zero_()
    with torch.no_grad():
        Dn = (Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)).detach()
        Ehat = torch.empty_like(E)
        for i in range(0, V, 8192):
            x = E[i:i + 8192]
            z = (x - b_pre) @ We.T
            _, idx = z.abs().topk(k, dim=1)
            coeff = torch.gather(z, 1, idx)
            Ehat[i:i + 8192] = b + (coeff.unsqueeze(-1) * Dn[idx]).sum(1)
    return Ehat


print('\n=== part 2: n=1024 k=64 dictionary fit under the WEIGHT metric (no data)')
Ehat_w = train_topk_dict_weighted(1024, 64)
dce_w = e6.eval_ce(Ehat_w) - r7['baseline_ce']
row = {'fvu': e6.fvu(Ehat_w), 'wfvu': wfvu(Ehat_w), 'dce': dce_w}
print(f"weighted-fit dict:  fvu {row['fvu']:.4f}  wfvu {row['wfvu']:.4f}  "
      f"dCE {dce_w:+.4f}   (plain-MSE fit was +2.11, CE-trained +0.26)")

out = {'spearman_fvu': sp_f, 'spearman_wfvu': sp_w, 'points': points,
       'weighted_fit_n1024_k64': row}
with open(f'{BASE}/e10_results.json', 'w') as fh:
    json.dump(out, fh, indent=2)
print('e10 done')
