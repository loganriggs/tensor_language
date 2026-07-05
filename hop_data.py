"""Depth-ladder task (session 5): in-context k-hop associative retrieval.

Each document defines a random function f on E entity tokens (a single random E-cycle,
so f^k(e) is well-defined and never returns to e for k < E — no short-cycle cheating).
The document has two parts:

  bindings : for every entity e (random order), the pair  [e, f(e)]         (2E tokens)
  queries  : N_Q blocks  [Q, e, H_k, a]  with a = f^k(e), k ∈ 0..K_MAX      (4 tokens each)

We score the model's prediction of the answer token a (read at the H_k position, which
predicts a), bucketed by the hop count k. The depth ladder:
  k=0 : return e itself                — trivial copy (floor anchor)
  k=1 : f(e), the token after e         — one lookup ≈ INDUCTION (needs ~2 layers)
  k=2 : f(f(e))                         — compose two lookups (needs more depth)
  k=3 : f(f(f(e)))                      — three chained lookups

Prediction: a 2-layer attn-only model solves k≤1 and fails k≥2; adding depth (a middle
bilinear MLP, or a third attention layer) unlocks the higher-hop categories. Those
higher-hop answer tokens are the "next induction head" — a category that needs depth.
"""

import torch

E = 32                                   # entity tokens 0..31
Q = E                                    # query marker, id 32
K_MAX = 3
H0 = E + 1                               # hop markers H0..H3 -> ids 33..36
V = E + 1 + (K_MAX + 1)                  # vocab size 37
N_Q = 48                                 # 2*32 + 4*48 = 256
N_CTX = 2 * E + 4 * N_Q                  # 256
ANS_OFFSET = 2                           # answer read at H_k position within its block
ANS_POS = torch.tensor([2 * E + 4 * j + ANS_OFFSET for j in range(N_Q)])  # input positions


def _fpow(fmap, kmax):
    """fpow[:,k] = f applied k times (batch, kmax+1, E)."""
    batch, e = fmap.shape
    out = [torch.arange(e).expand(batch, e).clone()]
    for _ in range(kmax):
        out.append(fmap.gather(1, out[-1]))
    return torch.stack(out, 1)


def sample_docs(batch, gen):
    """Returns tokens (batch, N_CTX), and per-block answer targets + hop categories."""
    # random E-cycle per doc: cyc is a permutation; f: cyc[i] -> cyc[(i+1) % E]
    cyc = torch.rand(batch, E, generator=gen).argsort(1)
    nxt = cyc.roll(-1, dims=1)
    fmap = torch.empty(batch, E, dtype=torch.long)
    fmap.scatter_(1, cyc, nxt)                       # fmap[e] = f(e)
    fpow = _fpow(fmap, K_MAX)                         # (batch, K_MAX+1, E)

    # bindings: pairs [e, f(e)] in a random order
    order = torch.rand(batch, E, generator=gen).argsort(1)
    vals = fmap.gather(1, order)
    bindings = torch.empty(batch, 2 * E, dtype=torch.long)
    bindings[:, 0::2] = order
    bindings[:, 1::2] = vals

    # queries
    rows = torch.arange(batch)[:, None]
    qe = torch.randint(E, (batch, N_Q), generator=gen)
    qk = torch.randint(K_MAX + 1, (batch, N_Q), generator=gen)
    qa = fpow[rows, qk, qe]                           # (batch, N_Q) = f^k(e)
    blocks = torch.stack([torch.full_like(qe, Q), qe, H0 + qk, qa], -1)  # (batch,N_Q,4)

    tokens = torch.cat([bindings, blocks.reshape(batch, 4 * N_Q)], 1)
    return tokens, qa, qk


def score_by_hop(logits, qa, qk):
    """Top-1 accuracy at each answer position, bucketed by hop count. logits are the
    model outputs on tokens[:, :-1] (index p predicts token p+1)."""
    pred = logits[:, ANS_POS].argmax(-1)             # (batch, N_Q)
    correct = (pred == qa)
    acc = {}
    for k in range(K_MAX + 1):
        m = qk == k
        acc[k] = (correct & m).sum().item() / max(1, m.sum().item())
    return acc
