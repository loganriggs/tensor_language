"""Tier 3 opener: PATH-FOLDED lookup codebooks for layer-1 QK of the rp model,
including the joint/shared key-side dictionary (spec Tier-3 'joint QK-OV with
shared key-side dictionary', in miniature).

Idea: tick 8 showed layer-1 key vectors (pre-rotary, per branch) are, on
induction data, well summarized by their conditional mean given the PREVIOUS
token. Tier-3 codebook: replace the live key/query computation of a branch by a
LOOKUP TABLE: k(position j) := kbar_b(token_{j-1}), q(position i) :=
qbar_b(token_i). This folds the entire L0→L1 path (OV transport + norms +
mixing) into V×128 tables — the path-folded vocab-space object, ΔCE/ΔP(copy)
audited on HELD-OUT seeds (tables fit on seeds 0..19, audit on 30..35).

Arms: identity branches k-side only; +q-side; all four branches both sides;
SHARED identity table across the two copy heads (sign-gauge-aligned average) —
if the two heads implement one conjunct, one table should serve both.

Logan's note (recorded for deeper tiers): from the first MLP layer on, each
layer has TWO input path families (attention-out + the residual token path);
deep paths will need heuristics or CE/KL-trained path selection. The 2-layer
attn-only rp model has exactly {direct, 4 OV paths} and is the clean base case.
"""

import json
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from folding import load_tiny

torch.manual_seed(0)
DEV = 'cuda'
RUN = 'attn2-s30k-mix50-rp-dense-seed0'
P, NSEQ = 96, 64
COPY = [(0, 1), (3, 2)]          # (head, identity branch) from ticks 6/8
OUT = '/workspace/tensor_language/basis_aligned/qk_mdl/tier3_pathfold.json'

model, cfg = load_tiny(RUN, dtype=torch.float32, device=DEV)
NH, DH, V, T = (cfg['n_head'], cfg['d_model'] // cfg['n_head'],
                cfg['vocab'], cfg['n_ctx'])
l0, l1 = model.layers


def tiled(seed):
    g = torch.Generator(); g.manual_seed(seed)
    w = torch.randint(V, (NSEQ, P), generator=g)
    return w.repeat(1, (T + P) // P)[:, :T + 1].to(DEV)


@torch.no_grad()
def layer1_inputs(b):
    x = model.embed(b)
    h = l0.norm(x)
    hs = lambda t: t.reshape(*t.shape[:-1], NH, DH)
    q1 = l0.rotary(hs(l0.q1(h))); k1 = l0.rotary(hs(l0.k1(h)))
    q2 = l0.rotary(hs(l0.q2(h))); k2 = l0.rotary(hs(l0.k2(h)))
    s1 = torch.einsum('bihd,bjhd->bhij', q1, k1)
    s2 = torch.einsum('bihd,bjhd->bhij', q2, k2)
    mask = torch.tril(torch.ones(b.shape[1], b.shape[1], device=DEV))
    pat = s1 * s2 / DH ** 2 * mask
    v = hs(l0.v(l0.norm(x)))
    z = torch.einsum('bhij,bjhd->bihd', pat, v).reshape(*b.shape, -1)
    return torch.lerp(x, l0.o(z), l0.scale)


# ---- fit conditional-mean q/k tables per (head, branch) on train seeds
q_sum = {(hh, br): torch.zeros(V, DH, device=DEV) for hh in range(NH) for br in (1, 2)}
k_sum = {(hh, br): torch.zeros(V, DH, device=DEV) for hh in range(NH) for br in (1, 2)}
qc = torch.zeros(V, device=DEV)
kc = torch.zeros(V, device=DEV)
hs = lambda t: t.reshape(*t.shape[:-1], NH, DH)
for seed in range(20):
    b = tiled(seed)[:, :-1]
    x1 = layer1_inputs(b)
    h1 = l1.norm(x1)
    qpos = torch.arange(P + 2, T, device=DEV)
    kpos = torch.arange(1, T, device=DEV)
    qtok = b[:, qpos].flatten()
    ktokprev = b[:, kpos - 1].flatten()
    for br, (wq, wk) in {1: (l1.q1, l1.k1), 2: (l1.q2, l1.k2)}.items():
        Qp = hs(wq(h1))
        Kp = hs(wk(h1))
        for hh in range(NH):
            q_sum[(hh, br)].index_add_(0, qtok, Qp[:, qpos, hh, :].reshape(-1, DH))
            k_sum[(hh, br)].index_add_(0, ktokprev, Kp[:, kpos, hh, :].reshape(-1, DH))
    qc.index_add_(0, qtok, torch.ones_like(qtok, dtype=torch.float))
    kc.index_add_(0, ktokprev, torch.ones_like(ktokprev, dtype=torch.float))
QBAR = {k2_: v / qc[:, None].clamp(min=1) for k2_, v in q_sum.items()}
KBAR = {k2_: v / kc[:, None].clamp(min=1) for k2_, v in k_sum.items()}

# shared identity table across the copy pair (sign-gauge aligned on H0.b1)
ref_q, ref_k = QBAR[COPY[0]], KBAR[COPY[0]]
signs = {}
for (hh, br) in COPY:
    s = torch.sign((KBAR[(hh, br)] * ref_k).sum())
    signs[(hh, br)] = float(s if s != 0 else 1.0)
SHARED_K = sum(signs[c] * KBAR[c] for c in COPY) / len(COPY)
SHARED_Q = sum(signs[c] * QBAR[c] for c in COPY) / len(COPY)


@torch.no_grad()
def forward_lookup(b, table_spec):
    """table_spec: dict (head, branch) -> {'k': tensorV×DH or None, 'q': ...}."""
    x1 = layer1_inputs(b)
    h1 = l1.norm(x1)
    Tn = b.shape[1]
    s = {}
    for br, (wq, wk) in {1: (l1.q1, l1.k1), 2: (l1.q2, l1.k2)}.items():
        Qp = hs(wq(h1))
        Kp = hs(wk(h1))
        for (hh, tbr), tabs in table_spec.items():
            if tbr != br:
                continue
            if tabs.get('q') is not None:
                Qp = Qp.clone()
                Qp[:, :, hh, :] = tabs['q'][b]
            if tabs.get('k') is not None:
                Kp = Kp.clone()
                kv = torch.zeros_like(Kp[:, :, hh, :])
                kv[:, 1:] = tabs['k'][b[:, :-1]]
                Kp[:, :, hh, :] = kv
        q = l1.rotary(Qp)
        k = l1.rotary(Kp)
        s[br] = torch.einsum('bihd,bjhd->bhij', q, k)
    mask = torch.tril(torch.ones(Tn, Tn, device=DEV))
    pat = s[1] * s[2] / DH ** 2 * mask
    v = hs(l1.v(l1.norm(x1)))
    z = torch.einsum('bhij,bjhd->bihd', pat, v).reshape(*b.shape, -1)
    x2 = torch.lerp(x1, l1.o(z), l1.scale)
    return model.head(x2)


def pcopy_on(seeds, spec=None):
    tot = 0.0
    for sd in seeds:
        bb = tiled(sd)
        b, y = bb[:, :-1], bb[:, 1:]
        qpos = torch.arange(P + 2, T - 2, device=DEV)
        logits = forward_lookup(b, spec or {})
        p = torch.softmax(logits.float(), -1)
        tot += float(p[:, qpos, :].gather(2, y[:, qpos].unsqueeze(-1)).mean())
    return tot / len(seeds)


HOLD = list(range(30, 35))
base = pcopy_on(HOLD)
print(f'held-out base P(copy) {base:.4f}')
arms = {
    'identity branches, k-side lookup':
        {c: {'k': KBAR[c]} for c in COPY},
    'identity branches, q+k lookup':
        {c: {'k': KBAR[c], 'q': QBAR[c]} for c in COPY},
    'ALL 8 (head,branch), q+k lookup':
        {(hh, br): {'k': KBAR[(hh, br)], 'q': QBAR[(hh, br)]}
         for hh in range(NH) for br in (1, 2)},
    'identity branches, SHARED k table (sign-aligned)':
        {c: {'k': signs[c] * SHARED_K} for c in COPY},
    'identity branches, SHARED q+k tables':
        {c: {'k': signs[c] * SHARED_K, 'q': signs[c] * SHARED_Q} for c in COPY},
}
results = {'base_pcopy_holdout': base, 'arms': {},
           'signs': {f'H{h}b{b}': v for (h, b), v in signs.items()}}
for name, spec in arms.items():
    d = pcopy_on(HOLD, spec) - base
    n_tables = sum(len([1 for t in tabs.values() if t is not None])
                   for tabs in spec.values())
    dl_bits = 32 * V * DH * (2 if 'SHARED' in name and 'q+k' in name
                             else 1 if 'SHARED' in name else n_tables)
    results['arms'][name] = {'dpcopy': d, 'dl_bits_tables': dl_bits}
    print(f'{name:48s} dP(copy) {d:+.4f}   tables DL {dl_bits / 8 / 1024:.0f} KiB',
          flush=True)

with open(OUT, 'w') as fh:
    json.dump(results, fh, indent=2)
print('tier3 done')
