"""e6 (thread 3, tick 1): how many "objects" does a real embedding matrix contain,
and does the answer depend on the metric?

E = pythia-410m embed_in (V=50304, d=1024). Compress under four representation
classes at matched float budgets, then audit each compression TWICE:

  weight audit:     FVU = ||E - Ehat||^2_F / ||E - rowmean||^2_F
  behavior audit:   delta-CE of pythia-410m on pile-10k with embed_in := Ehat

Classes (per Logan's 4-class map):
  svd      class 1, rank-r truncation (Eckart-Young optimal for the weight audit
           by construction). params = (V + d) * r
  kmeans   class 3 degenerate: n centroids, each token = its centroid (1 object
           per token). params = n * d  (+ V*log2(n) index bits, not counted)
  rq       class 3: residual VQ, h levels of shared codebooks of size c; token =
           sum of h centroids. params = h * c * d
  tree     class 3, the hierarchy prior: recursive k-means on residuals, token =
           sum of its root-to-leaf node vectors. params = (#nodes) * d

Controls: random-basis rank-r (same params as svd, random column space) and
shuffled-assignment kmeans (same centroids, random tokens->centroid map).

The thread-1/2 question, one level up: do the classes rank the same under both
audits, or does the behavioral metric prefer different structure than Frobenius?
"""

import json
import sys

import torch

torch.manual_seed(0)
DEV = 'cuda'
BASE = '/workspace/tensor_language/basis_aligned'

# ---------------------------------------------------------------- load model + E

from transformers import AutoModelForCausalLM, AutoTokenizer

print('loading pythia-410m...')
tok = AutoTokenizer.from_pretrained('EleutherAI/pythia-410m')
model = AutoModelForCausalLM.from_pretrained(
    'EleutherAI/pythia-410m', torch_dtype=torch.float16).to(DEV).eval()
E = model.gpt_neox.embed_in.weight.detach().float().clone()
V, D = E.shape
print(f'E: {V} x {D}  ({V * D / 1e6:.1f}M params)')
ROWMEAN = E.mean(0, keepdim=True)
DENOM = ((E - ROWMEAN) ** 2).sum().item()


def fvu(Ehat):
    return float(((E - Ehat) ** 2).sum().item() / DENOM)


# ---------------------------------------------------------------- CE evaluation

def build_eval_tokens(n_chunks=64, seq_len=513, seed=0):
    from datasets import load_dataset
    ds = load_dataset('NeelNanda/pile-10k', split='train')
    ids, chunks = [], []
    for doc in ds:
        ids.extend(tok(doc['text'])['input_ids'])
        while len(ids) >= seq_len:
            chunks.append(torch.tensor(ids[:seq_len]))
            ids = ids[seq_len:]
            if len(chunks) >= n_chunks:
                return torch.stack(chunks)
    return torch.stack(chunks)


print('building eval tokens...')
TOKENS = build_eval_tokens().to(DEV)
print(f'eval set: {tuple(TOKENS.shape)}')


@torch.no_grad()
def eval_ce(Ehat=None):
    w = model.gpt_neox.embed_in.weight
    orig = w.data.clone()
    if Ehat is not None:
        w.data.copy_(Ehat.half())
    tot, n = 0.0, 0
    for i in range(0, len(TOKENS), 8):
        batch = TOKENS[i:i + 8]
        logits = model(batch[:, :-1]).logits.float()
        ce = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1))
        tot += ce.item() * batch.numel()
        n += batch.numel()
    w.data.copy_(orig)
    return tot / n


# ---------------------------------------------------------------- methods

def kmeans(X, n, iters=25, seed=0):
    g = torch.Generator(device='cpu'); g.manual_seed(seed)
    C = X[torch.randperm(len(X), generator=g)[:n]].clone()
    for _ in range(iters):
        assign = torch.empty(len(X), dtype=torch.long, device=DEV)
        for i in range(0, len(X), 8192):  # chunked distances
            xc = X[i:i + 8192]
            d2 = (xc ** 2).sum(1, keepdim=True) - 2 * xc @ C.T + (C ** 2).sum(1)[None]
            assign[i:i + 8192] = d2.argmin(1)
        Cn = torch.zeros_like(C)
        cnt = torch.zeros(n, device=DEV)
        Cn.index_add_(0, assign, X)
        cnt.index_add_(0, assign, torch.ones(len(X), device=DEV))
        dead = cnt == 0
        C = torch.where(dead[:, None], C, Cn / cnt.clamp(min=1)[:, None])
    return C, assign


def arm_svd(r, random_basis=False):
    if random_basis:
        Q, _ = torch.linalg.qr(torch.randn(D, r, device=DEV))
    else:
        _, _, Vt = torch.linalg.svd(E - ROWMEAN, full_matrices=False)
        Q = Vt[:r].T
    Ehat = ROWMEAN + (E - ROWMEAN) @ Q @ Q.T
    return Ehat, (V + D) * r


def arm_kmeans(n, shuffle=False):
    C, assign = kmeans(E, n)
    if shuffle:
        assign = assign[torch.randperm(len(assign), device=DEV)]
    return C[assign], n * D


def arm_rq(c, h):
    resid = E.clone()
    Ehat = torch.zeros_like(E)
    for _ in range(h):
        C, assign = kmeans(resid, c)
        Ehat += C[assign]
        resid -= C[assign]
    return Ehat, h * c * D


def arm_tree(b, depth, min_size=None):
    """Recursive k-means on residuals; token = sum of path node vectors."""
    min_size = min_size or 4 * b
    Ehat = torch.zeros_like(E)
    n_nodes = 0
    frontier = [(torch.arange(V, device=DEV), E.clone(), 0)]
    while frontier:
        idx, X, level = frontier.pop()
        if level >= depth or len(idx) < min_size:
            continue
        C, assign = kmeans(X, b, iters=15, seed=level)
        n_nodes += b
        Ehat[idx] += C[assign]
        resid = X - C[assign]
        for k in range(b):
            m = assign == k
            if m.sum() >= min_size:
                frontier.append((idx[m], resid[m], level + 1))
    return Ehat, n_nodes * D


# ---------------------------------------------------------------- run (script only)

CONFIGS = [
    ('svd', 'r=19', lambda: arm_svd(19)),        # ~1.9% budget
    ('svd', 'r=50', lambda: arm_svd(50)),        # ~5%
    ('svd', 'r=100', lambda: arm_svd(100)),      # ~10%
    ('svd', 'r=250', lambda: arm_svd(250)),      # ~25%
    ('svd_random', 'r=100', lambda: arm_svd(100, random_basis=True)),
    ('kmeans', 'n=1k', lambda: arm_kmeans(1024)),
    ('kmeans', 'n=5k', lambda: arm_kmeans(5120)),
    ('kmeans', 'n=12k', lambda: arm_kmeans(12800)),
    ('kmeans_shuffled', 'n=5k', lambda: arm_kmeans(5120, shuffle=True)),
    ('rq', 'c=1k,h=1', lambda: arm_rq(1024, 1)),
    ('rq', 'c=1k,h=2', lambda: arm_rq(1024, 2)),
    ('rq', 'c=1k,h=5', lambda: arm_rq(1024, 5)),
    ('rq', 'c=1k,h=12', lambda: arm_rq(1024, 12)),
    ('tree', 'b=64,h=2', lambda: arm_tree(64, 2)),
    ('tree', 'b=32,h=3', lambda: arm_tree(32, 3)),
]

if __name__ == '__main__':
    print('baseline CE...')
    CE0 = eval_ce()
    print(f'baseline CE: {CE0:.4f}')

    results = {'model': 'pythia-410m', 'V': V, 'd': D, 'baseline_ce': CE0, 'rows': []}
    for method, label, fn in CONFIGS:
        Ehat, params = fn()
        row = {'method': method, 'label': label, 'params': params,
               'budget': params / (V * D), 'fvu': fvu(Ehat), 'ce': eval_ce(Ehat)}
        row['dce'] = row['ce'] - CE0
        results['rows'].append(row)
        print(f"{method:16s} {label:10s} budget {row['budget']:6.1%}  "
              f"fvu {row['fvu']:.4f}  CE {row['ce']:.4f}  dCE {row['dce']:+.4f}")
        del Ehat
        torch.cuda.empty_cache()

    # qualitative: what do tree/kmeans objects look like?
    C, assign = kmeans(E, 1024)
    samples = {}
    for k in [int(i) for i in torch.randperm(1024)[:12]]:
        toks = (assign == k).nonzero().flatten()[:12]
        samples[k] = [tok.decode([t]) for t in toks.tolist()]
    results['kmeans1k_cluster_samples'] = samples
    print('\nexample kmeans-1k clusters:')
    for k, ts in list(samples.items())[:8]:
        print(f'  {k}: {ts}')

    with open(f'{BASE}/e6_results.json', 'w') as fh:
        json.dump(results, fh, indent=2)
    print('saved e6_results.json')
