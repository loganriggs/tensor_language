"""Tier-3 first-order path codebook (the fix for tick 9's 0th-order negative):
keep the LIVE layer-0 attention pattern (context-dependent), quantize only the
transported CONTENT — layer-0 v from k-class tables — for the layer-1 key-side
input. 0th-order (conditional-mean lookup) destroyed the circuit (−0.62…−0.74);
prediction: first-order survives at modest k because the context-dependent
pattern weights were the missing component.

Model: attn2-s30k-mix50-rp-dense-seed0. Metrics: held-out P(copy) (seeds 30-34)
+ tiled CE. Arms: v classed at k for ALL of layer-1's input vs for the key-side
of the identity branches only.
"""
import json, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from folding import load_tiny

torch.manual_seed(0)
DEV = 'cuda'
RUN = 'attn2-s30k-mix50-rp-dense-seed0'
P, NSEQ = 96, 64
COPY = [(0, 1), (3, 2)]
model, cfg = load_tiny(RUN, dtype=torch.float32, device=DEV)
NH, DH, V, T = cfg['n_head'], cfg['d_model'] // cfg['n_head'], cfg['vocab'], cfg['n_ctx']
l0, l1 = model.layers
E = model.embed.weight.detach()
h_emb = l0.norm(E)
VT = (h_emb @ l0.v.weight.detach().T).view(V, NH, DH)   # layer-0 v tables (exact)

def kmeans(X, k, iters=15, seed=0):
    g = torch.Generator(device='cpu'); g.manual_seed(seed)
    C = X[torch.randperm(len(X), generator=g)[:k]].clone()
    for _ in range(iters):
        d2 = (X**2).sum(1, keepdim=True) - 2*X@C.T + (C**2).sum(1)[None]
        assign = d2.argmin(1)
        Cn = torch.zeros_like(C); cnt = torch.zeros(k, device=X.device)
        Cn.index_add_(0, assign, X); cnt.index_add_(0, assign, torch.ones(len(X), device=X.device))
        C = torch.where((cnt == 0)[:, None], C, Cn / cnt.clamp(min=1)[:, None])
    return C, assign

def tiled(seed):
    g = torch.Generator(); g.manual_seed(seed)
    w = torch.randint(V, (NSEQ, P), generator=g)
    return w.repeat(1, (T + P) // P)[:, :T + 1].to(DEV)

@torch.no_grad()
def forward(b, v_tab=None, key_only_branches=None):
    """v_tab: classed layer-0 v tables (V, NH, DH). If key_only_branches, the
    classed-v attention-out feeds ONLY those (head, branch) key inputs at layer 1;
    otherwise it replaces layer-0 v everywhere."""
    x = model.embed(b)
    hs = lambda t: t.reshape(*t.shape[:-1], NH, DH)
    h = l0.norm(x)
    q1 = l0.rotary(hs(l0.q1(h))); k1 = l0.rotary(hs(l0.k1(h)))
    q2 = l0.rotary(hs(l0.q2(h))); k2 = l0.rotary(hs(l0.k2(h)))
    s1 = torch.einsum('bihd,bjhd->bhij', q1, k1)
    s2 = torch.einsum('bihd,bjhd->bhij', q2, k2)
    mask = torch.tril(torch.ones(b.shape[1], b.shape[1], device=DEV))
    pat = s1 * s2 / DH**2 * mask                       # LIVE pattern (first-order)
    v_exact = hs(l0.v(l0.norm(x)))
    z_exact = torch.einsum('bhij,bjhd->bihd', pat, v_exact).reshape(*b.shape, -1)
    x1_exact = torch.lerp(x, l0.o(z_exact), l0.scale)
    if v_tab is not None:
        v_c = v_tab[b]
        z_c = torch.einsum('bhij,bjhd->bihd', pat, v_c).reshape(*b.shape, -1)
        x1_c = torch.lerp(x, l0.o(z_c), l0.scale)
    else:
        x1_c = x1_exact
    # layer 1
    h1 = l1.norm(x1_exact)
    h1c = l1.norm(x1_c)
    s = {}
    for br, (wq, wk) in {1: (l1.q1, l1.k1), 2: (l1.q2, l1.k2)}.items():
        q = l1.rotary(hs(wq(h1)))
        if v_tab is not None and key_only_branches is None:
            k = l1.rotary(hs(wk(h1c)))                 # classed content everywhere
            q = l1.rotary(hs(wq(h1c)))
        else:
            k = l1.rotary(hs(wk(h1)))
        s[br] = torch.einsum('bihd,bjhd->bhij', q, k)
        if v_tab is not None and key_only_branches:
            kc = l1.rotary(hs(wk(h1c)))
            sc = torch.einsum('bihd,bjhd->bhij', q, kc)
            sb = s[br].clone()
            for (hh, tbr) in key_only_branches:
                if tbr == br:
                    sb[:, hh] = sc[:, hh]
            s[br] = sb
    pat1 = s[1] * s[2] / DH**2 * mask
    v1v = hs(l1.v(l1.norm(x1_exact if key_only_branches else x1_c)))
    z1 = torch.einsum('bhij,bjhd->bihd', pat1, v1v).reshape(*b.shape, -1)
    x2 = torch.lerp(x1_exact if key_only_branches else x1_c, l1.o(z1), l1.scale)
    return model.head(x2)

def metrics(seeds, **kw):
    pc, cet = 0.0, 0.0
    for sd in seeds:
        bb = tiled(sd)
        b, y = bb[:, :-1], bb[:, 1:]
        qpos = torch.arange(P + 2, T - 2, device=DEV)
        logits = forward(b, **kw)
        p = torch.softmax(logits.float(), -1)
        pc += float(p[:, qpos, :].gather(2, y[:, qpos].unsqueeze(-1)).mean())
        cet += float(F.cross_entropy(logits[:, P:].reshape(-1, V), y[:, P:].reshape(-1)))
    return pc / len(seeds), cet / len(seeds)

HOLD = list(range(30, 35))
pc0, ce0 = metrics(HOLD)
print(f'base P(copy) {pc0:.4f}  tiled CE {ce0:.4f}')
results = {'base_pcopy': pc0, 'base_ce': ce0, 'arms': {}}
for k in [64, 256, 1024]:
    vt = torch.empty_like(VT)
    for hh in range(NH):
        C, a_ = kmeans(VT[:, hh].contiguous(), k)
        vt[:, hh] = C[a_]
    pc, cet = metrics(HOLD, v_tab=vt)
    results['arms'][f'all-content k={k}'] = {'dpcopy': pc - pc0, 'dce_tiled': cet - ce0}
    print(f'first-order all-content k={k:5d}: dP(copy) {pc - pc0:+.4f}  dCE {cet - ce0:+.4f}', flush=True)
    pck, cek = metrics(HOLD, v_tab=vt, key_only_branches=COPY)
    results['arms'][f'identity-keys k={k}'] = {'dpcopy': pck - pc0, 'dce_tiled': cek - ce0}
    print(f'first-order identity-keys k={k:5d}: dP(copy) {pck - pc0:+.4f}  dCE {cek - ce0:+.4f}', flush=True)
    json.dump(results, open('first_order_path.json', 'w'), indent=2)
print('first order path done')
