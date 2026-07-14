"""e8 (class 4): tensor-train / hierarchical-Tucker structure of the embedding.

Reshape the vocab index into digits (pad V to 65536 = 16^4), so E becomes an
order-5 tensor (16,16,16,16,1024), and TT-SVD it (TT = HT with a linear index
tree). The class-4 claim: hierarchical vocab structure manifests as lower
TT-ranks under an index ordering that matches the semantic tree. Measurement:
FVU at fixed rank caps under three orderings —

  bpe       the native vocab order (already semi-semantic: BPE merges cluster
            morphologically, frequency-sorted)
  random    random permutation of real tokens (destroys locality at all scales)
  semantic  balanced recursive 16-way k-means (digit hierarchy = cluster tree)

Padding rows equal the row-mean (zero after centering) and sit at the end for
every ordering. FVU uses e6's denominator, computed on real rows only.

Also: swap-in dCE for the rmax=256 decompositions, and a CE-finetuned TT-cores
arm (semantic ordering) for the MDL graph.
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
VPAD, DIGITS, B = 65536, 4, 16
torch.manual_seed(0)

ROWMEAN = e6.ROWMEAN
E_C = torch.cat([e6.E - ROWMEAN,
                 torch.zeros(VPAD - V, D_MODEL, device=DEV)])  # centered + pad


# ---------------------------------------------------------------- orderings

def balanced_groups(X, idx, b):
    """Split idx into b EQUAL groups by k-means with capacity constraints."""
    cap = len(idx) // b
    C, _ = e6.kmeans(X, b, iters=15)
    d2 = ((X ** 2).sum(1, keepdim=True) - 2 * X @ C.T + (C ** 2).sum(1)[None])
    conf = d2.min(1).values.argsort()  # most confident first
    counts = [0] * b
    groups = [[] for _ in range(b)]
    pref = d2.argsort(1)
    for i in conf.tolist():
        for c in pref[i].tolist():
            if counts[c] < cap:
                groups[c].append(idx[i])
                counts[c] += 1
                break
    return [torch.tensor(g, device=DEV) for g in groups]


def semantic_order():
    order = []
    lvl1 = balanced_groups(E_C, torch.arange(VPAD, device=DEV), B)
    for g1 in lvl1:
        lvl2 = balanced_groups(E_C[g1], g1, B)
        for g2 in lvl2:
            lvl3 = balanced_groups(E_C[g2], g2, B)
            for g3 in lvl3:
                order.append(g3)  # 16 tokens; last digit order arbitrary
    return torch.cat(order)


def make_orderings():
    pads = torch.arange(V, VPAD, device=DEV)
    g = torch.Generator(device='cpu'); g.manual_seed(0)
    rand = torch.cat([torch.randperm(V, generator=g).to(DEV), pads])
    bpe = torch.arange(VPAD, device=DEV)
    print('building semantic ordering (balanced recursive k-means)...')
    sem = semantic_order()
    return {'bpe': bpe, 'random': rand, 'semantic': sem}


# ---------------------------------------------------------------- TT-SVD

def tt_svd(T_mat, rmax):
    """T_mat: (VPAD, d) already permuted. Returns cores, params, recon."""
    cores = []
    W = T_mat.reshape(B, -1)
    r_prev = 1
    for _ in range(DIGITS):
        M = W.shape[0]
        G = W @ W.T
        evals, evecs = torch.linalg.eigh(G.double())
        order = evals.argsort(descending=True)
        r = min(rmax, M)
        U = evecs[:, order[:r]].float()
        cores.append(U.reshape(r_prev, M // r_prev, r))
        W = (U.T @ W).reshape(r * B, -1) if len(cores) < DIGITS else (U.T @ W)
        r_prev = r
    cores.append(W)  # (r4, d)
    params = sum(c.numel() for c in cores)
    # reconstruct
    R = cores[0].reshape(-1, cores[0].shape[-1])           # (16, r1)
    for c in cores[1:DIGITS]:
        R = torch.einsum('ab,bcd->acd', R, c).reshape(R.shape[0] * c.shape[1], -1)
    R = R @ cores[DIGITS]                                   # (VPAD, d)
    return cores, params, R


def fvu_real(Ehat_perm, perm):
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(VPAD, device=DEV)
    Ehat = Ehat_perm[inv][:V] + ROWMEAN
    return e6.fvu(Ehat), Ehat


# ---------------------------------------------------------------- TT embed (CE-ft)

class TTEmbed(nn.Module):
    def __init__(self, cores, digits):
        super().__init__()
        self.register_buffer('digits', digits)  # (V, 4)
        self.cores = nn.ParameterList([nn.Parameter(c.clone()) for c in cores])
        self.register_buffer('rowmean', ROWMEAN.squeeze(0).clone())

    def forward(self, ids):
        dg = self.digits[ids]                                # (..., 4)
        x = self.cores[0][0, dg[..., 0], :]                  # (..., r1)
        for i in range(1, DIGITS):
            c = self.cores[i][:, dg[..., i], :]              # (r_prev, ..., r)
            c = c.movedim(0, -2)                             # (..., r_prev, r)
            x = (x.unsqueeze(-2) @ c).squeeze(-2)
        out = x @ self.cores[DIGITS] + self.rowmean
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


# ---------------------------------------------------------------- run

if __name__ == '__main__':
    e6.model.to(torch.bfloat16)
    CE0 = eval_ce_current()
    print(f'bf16 baseline CE: {CE0:.4f}')
    orderings = make_orderings()
    results = {'baseline_ce_bf16': CE0, 'V': V, 'd': D_MODEL, 'rows': []}

    best = {}
    for name, perm in orderings.items():
        for rmax in [32, 64, 128, 256]:
            cores, params, R = tt_svd(E_C[perm], rmax)
            fvu, Ehat = fvu_real(R, perm)
            row = {'ordering': name, 'rmax': rmax, 'params': params, 'fvu': fvu}
            if rmax == 256:
                row['dce'] = e6.eval_ce(Ehat) - CE0
                best[name] = (cores, perm)
            results['rows'].append(row)
            print(f"{name:9s} rmax={rmax:4d}  params {params / 1e6:5.2f}M  "
                  f"fvu {fvu:.4f}" + (f"  dCE {row['dce']:+.4f}" if 'dce' in row else ''),
                  flush=True)
            del R, Ehat
            torch.cuda.empty_cache()

    # CE-finetune the semantic rmax=256 cores
    print('=== CE-finetune TT cores (semantic, rmax=256)')
    cores, perm = best['semantic']
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(VPAD, device=DEV)
    pos = inv[:V]  # position of each real token in permuted tensor
    digits = torch.stack([(pos // B ** (DIGITS - 1 - i)) % B
                          for i in range(DIGITS)], 1)
    TRAIN = e6.build_eval_tokens(n_chunks=64 + 512)[64:].to(DEV)
    for p in e6.model.parameters():
        p.requires_grad_(False)
    te = TTEmbed(cores, digits).to(DEV)
    orig = e6.model.gpt_neox.embed_in
    e6.model.gpt_neox.embed_in = te
    print(f'  before: dCE {eval_ce_current() - CE0:+.4f}')
    opt = torch.optim.Adam(te.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1500)
    g = torch.Generator(device='cpu'); g.manual_seed(0)
    for step in range(1500):
        batch = TRAIN[torch.randint(0, len(TRAIN), (8,), generator=g)]
        logits = e6.model(batch[:, :-1]).logits.float()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               batch[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(te.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 300 == 0:
            print(f'  step {step:5d}  train CE {loss.item():.4f}', flush=True)
    dce_after = eval_ce_current() - CE0
    results['tt_ce_finetune'] = {
        'ordering': 'semantic', 'rmax': 256,
        'params': sum(c.numel() for c in te.cores), 'dce_after': dce_after}
    print(f'  after: dCE {dce_after:+.4f}')
    e6.model.gpt_neox.embed_in = orig

    with open(f'{BASE}/e8_results.json', 'w') as fh:
        json.dump(results, fh, indent=2)
    print('e8 done')
