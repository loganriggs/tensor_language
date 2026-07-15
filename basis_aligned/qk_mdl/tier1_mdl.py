"""Tier 1.1: full layer-0 MDL analysis of the tiny bilinear models
(attn2-mix10-seed0: V=5120, d_model=128, 4 heads, d_head=32, 16 RoPE bands).

Per (head, branch): folded factors q=[qa|qb], k=[ka|kb] in (V, 32) (exact,
tier0 gate). Codebooks {svd-r, vq-k, band-m, zero}, FULL grid dCE-audited
(binding metric per Logan), eval = data_owt val.bin (the model's training
distribution), T=256. Joint frontier: all-(head,branch) vq-k.

Mini-gate inside: the reference forward (needed for score patching) must
reproduce the model's own forward exactly (fp64) and the recorded val CE
(history.jsonl: 4.637) approximately.
"""

import json
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from folding import load_tiny, branch_factors

torch.manual_seed(0)
DEV = 'cuda'
RUN = 'attn2-mix10-seed0'
OUT = f'/workspace/tensor_language/basis_aligned/qk_mdl/tier1_mdl_{RUN}.json'

model, cfg = load_tiny(RUN, dtype=torch.float32, device=DEV)
NH = cfg['n_head']
DH = cfg['d_model'] // NH
FB = DH // 2                      # bands per branch
V = cfg['vocab']
T = cfg['n_ctx']

val = np.memmap('/workspace/tensor_language/data_owt/val.bin', dtype=np.uint16, mode='r')
g = torch.Generator(); g.manual_seed(0)
starts = torch.randint(0, len(val) - (T + 1), (64,), generator=g)
EVAL = torch.stack([torch.tensor(val[int(s):int(s) + T + 1].astype(np.int64))
                    for s in starts]).to(DEV)


def head_scores(q, k, tokens, layer):
    """(B, T, T) scores for ONE head from (V, 32) factor stacks, model trig."""
    Tn = tokens.shape[-1]
    cs = layer.rotary.cos_cached[0, :Tn, 0, :FB].to(q.dtype)
    sn = layer.rotary.sin_cached[0, :Tn, 0, :FB].to(q.dtype)
    cosD = torch.einsum('if,jf->ijf', cs, cs) + torch.einsum('if,jf->ijf', sn, sn)
    sinD = torch.einsum('if,jf->ijf', sn, cs) - torch.einsum('if,jf->ijf', cs, sn)
    qa, qb = q[tokens][..., :FB], q[tokens][..., FB:]
    ka, kb = k[tokens][..., :FB], k[tokens][..., FB:]
    return (torch.einsum('bif,bjf,ijf->bij', qa, ka, cosD)
            + torch.einsum('bif,bjf,ijf->bij', qb, kb, cosD)
            + torch.einsum('bif,bjf,ijf->bij', qa, kb, sinD)
            - torch.einsum('bif,bjf,ijf->bij', qb, ka, sinD))


@torch.no_grad()
def reference_forward(tokens, patch=None):
    """patch: dict (head, branch) -> (q_stack, k_stack) replacing layer-0 scores."""
    x = model.embed(tokens)
    for li, layer in enumerate(model.layers):
        h = layer.norm(x)
        Tn = tokens.shape[-1]
        hs = lambda t: t.reshape(*t.shape[:-1], NH, DH)
        q1 = layer.rotary(hs(layer.q1(h)))
        k1 = layer.rotary(hs(layer.k1(h)))
        q2 = layer.rotary(hs(layer.q2(h)))
        k2 = layer.rotary(hs(layer.k2(h)))
        s1 = torch.einsum('bihd,bjhd->bhij', q1, k1)
        s2 = torch.einsum('bihd,bjhd->bhij', q2, k2)
        if li == 0 and patch:
            s1, s2 = s1.clone(), s2.clone()
            for (hh, br), (qs, ks) in patch.items():
                sc = head_scores(qs, ks, tokens, layer)
                (s1 if br == 1 else s2)[:, hh] = sc
        mask = torch.tril(torch.ones(Tn, Tn, device=tokens.device, dtype=x.dtype))
        pat = s1 * s2 / DH ** 2 * mask
        v = hs(layer.v(layer.norm(x)))
        z = torch.einsum('bhij,bjhd->bihd', pat, v).reshape(*tokens.shape, -1)
        x = torch.lerp(x, layer.o(z), layer.scale)
    return model.head(x)


@torch.no_grad()
def ce(patch=None, batch=16):
    tot, n = 0.0, 0
    for i in range(0, len(EVAL), batch):
        b = EVAL[i:i + batch]
        logits = reference_forward(b[:, :-1], patch).float()
        tot += F.cross_entropy(logits.reshape(-1, V),
                               b[:, 1:].reshape(-1)).item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n


# mini-gate: reference == model forward (fp64), and CE ~ recorded 4.637
m64, _ = load_tiny(RUN, dtype=torch.float64, device=DEV)
tok = EVAL[:2, :-1]
lo_model = m64(tok)
model64_backup = model
model = m64
lo_ref = reference_forward(tok)
model = model64_backup
gate_err = float((lo_model - lo_ref).abs().max())
del m64
CE0 = ce()
print(f'reference-forward gate: max logit diff {gate_err:.2e} '
      f'({"PASS" if gate_err < 1e-10 else "FAIL"})')
print(f'baseline CE {CE0:.4f} (recorded 4.637)')
assert gate_err < 1e-10

FACT = {}
for br in (1, 2):
    qa, qb, ka, kb = branch_factors(model, 0, br)
    for hh in range(NH):
        FACT[(hh, br)] = (torch.cat([qa[:, hh], qb[:, hh]], 1).contiguous(),
                          torch.cat([ka[:, hh], kb[:, hh]], 1).contiguous())


@torch.no_grad()
def kmeans(X, k, iters=20, seed=0):
    gg = torch.Generator(device='cpu'); gg.manual_seed(seed)
    C = X[torch.randperm(len(X), generator=gg)[:k]].clone()
    for _ in range(iters):
        d2 = (X ** 2).sum(1, keepdim=True) - 2 * X @ C.T + (C ** 2).sum(1)[None]
        assign = d2.argmin(1)
        Cn = torch.zeros_like(C)
        cnt = torch.zeros(k, device=X.device)
        Cn.index_add_(0, assign, X)
        cnt.index_add_(0, assign, torch.ones(len(X), device=X.device))
        C = torch.where((cnt == 0)[:, None], C, Cn / cnt.clamp(min=1)[:, None])
    return C, assign


def candidates(hh, br):
    q, k = FACT[(hh, br)]
    out = []
    for r in [1, 2, 4, 8, 16]:
        def tr(X, r=r):
            U, S, Vt = torch.linalg.svd(X, full_matrices=False)
            return U[:, :r] @ torch.diag(S[:r]) @ Vt[:r]
        out.append((f'svd{r}', 2 * 32 * r * (V + 2 * FB + 1), tr(q), tr(k)))
    for kk in [1, 4, 16, 64, 256, 1024]:
        if kk == 1:
            cq, ck = q.mean(0, keepdim=True).expand_as(q), k.mean(0, keepdim=True).expand_as(k)
            dl = 32 * 4 * FB
        else:
            C, assign = kmeans(torch.cat([q, k], 1), kk)
            cq, ck = C[assign][:, :2 * FB].contiguous(), C[assign][:, 2 * FB:].contiguous()
            dl = 32 * kk * 4 * FB + V * math.log2(kk)
        out.append((f'vq{kk}', dl, cq, ck))
    qa, qb, ka, kb = q[:, :FB], q[:, FB:], k[:, :FB], k[:, FB:]
    mass = ((qa ** 2).sum(0) + (qb ** 2).sum(0)) * ((ka ** 2).sum(0) + (kb ** 2).sum(0))
    order = mass.argsort(descending=True)
    for mm in [1, 2, 4, 8]:
        keep = torch.zeros(FB, dtype=torch.bool, device=DEV)
        keep[order[:mm]] = True
        msk = torch.cat([keep, keep]).float()
        out.append((f'band{mm}', 32 * 2 * V * 2 * mm + FB, q * msk, k * msk))
    out.append(('zero', 0, torch.zeros_like(q), torch.zeros_like(k)))
    return out


FULL_DL = 32 * 2 * V * 2 * FB
results = {'model': RUN, 'baseline_ce': CE0, 'gate_err': gate_err,
           'full_dl_bits_per_headbranch': FULL_DL, 'rows': [], 'joint': {}}
VQCACHE = {}
for hh in range(NH):
    for br in (1, 2):
        row = {'head': hh, 'branch': br, 'cands': []}
        for name, dl, qc, kc in candidates(hh, br):
            dce = ce({(hh, br): (qc, kc)}) - CE0
            row['cands'].append({'name': name, 'dl_bits': dl, 'dce': dce})
            if name.startswith('vq'):
                VQCACHE[(name, hh, br)] = (qc, kc)
        results['rows'].append(row)
        keep = [c for c in row['cands'] if c['dce'] <= 0.01]
        best = min(keep, key=lambda c: c['dl_bits']) if keep else None
        print(f"L0H{hh} b{br}: min-DL@0.01 = {best['name'] if best else 'NONE'}; "
              + '  '.join(f"{c['name']} {c['dce']:+.3f}" for c in row['cands'][:6]),
              flush=True)
        with open(OUT, 'w') as fh:
            json.dump(results, fh, indent=2)

for kk in [1, 4, 16, 64, 256, 1024]:
    patch = {(hh, br): VQCACHE[(f'vq{kk}', hh, br)]
             for hh in range(NH) for br in (1, 2)}
    d = ce(patch) - CE0
    dl = 8 * (32 * kk * 4 * FB + (V * math.log2(kk) if kk > 1 else 0))
    results['joint'][f'all vq{kk}'] = {'dce': d, 'dl_bits': dl,
                                       'ratio': dl / (FULL_DL * 8)}
    print(f'joint all vq{kk}: dCE {d:+.4f}  ratio {dl / (FULL_DL * 8):.2e}', flush=True)
patch = {(hh, br): (torch.zeros_like(FACT[(hh, br)][0]),
                    torch.zeros_like(FACT[(hh, br)][1]))
         for hh in range(NH) for br in (1, 2)}
results['joint']['all zero'] = {'dce': ce(patch) - CE0, 'dl_bits': 0, 'ratio': 0.0}
print(f"joint all zero: dCE {results['joint']['all zero']['dce']:+.4f}")
with open(OUT, 'w') as fh:
    json.dump(results, fh, indent=2)
print('tier1 done')
