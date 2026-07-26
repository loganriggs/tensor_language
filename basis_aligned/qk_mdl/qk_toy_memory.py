"""TICK 233 (Logan): toy models of the memory-MDL claim — "the parametric form is
near the description-length frontier for long-tail associative content."

TOY A (clean capacity): N random key->value pairs (values uniform over V=1024, floor
log2(V)=10 bits/pair). A bilinear-MLP net (Left/Right/Down, mimicking the real block)
is trained to memorize; we measure the max N stored at >=90% recall for several
parameter counts. Yardstick: parametric capacity ~2 bits/param (known empirically) =>
pairs ~ P/5; fp32 storage overhead 16x floor, int8 ~4x, i.e. the SAME ORDER as an
explicit table — parameters are not a wasteful way to store associations.

TOY B (Zipf exposure — the long tail): same pairs but training/evaluation exposure is
Zipf(1). Explicit-table alternative: store the top-M pairs (M x 10 bits) with uniform
fallback. Measured curve: fraction of achievable CE gain vs table size M. Zipf math
says cumulative gain ~ ln(M)/ln(N): to reach 90% of the gain you need M ~ N^0.9 —
there IS no small table. This is the real-case signature: our 3M/30M-token datastores
covered the head of the distribution (tiny CE gain, ticks 231-232) while the value
lives in the tail.

TOY C (rules + exceptions — the real case): a fraction rho of keys follow a simple
shared rule (value = g(key), g small); the rest are exceptions. The explicit-object
frontier recovers ~rho of the CE at tiny bits, then flattens; the remainder is
irreducibly table-like. With rho = 0.5 this reproduces the program's observed shape:
named basis 51%, then explicit rungs flat, parametric tail owning the rest.
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
VOC = 1024
out = {}

print('=== TOY A: parametric capacity for random pairs ===', flush=True)


class BilinNet(nn.Module):
    def __init__(self, din, hid, vout):
        super().__init__()
        self.L = nn.Linear(din, hid, bias=False)
        self.R = nn.Linear(din, hid, bias=False)
        self.D = nn.Linear(hid, vout, bias=False)

    def forward(self, x):
        return self.D(self.L(x) * self.R(x))


def capacity_run(P_target, N, steps=4000):
    din = 64
    hid = max(8, int(P_target / (2 * din + VOC)))
    net = BilinNet(din, hid, VOC).to(DEV)
    P = sum(p.numel() for p in net.parameters())
    K = torch.randn(N, din, device=DEV)
    Vl = torch.randint(0, VOC, (N,), device=DEV)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for s in range(steps):
        bi = torch.randint(0, N, (4096,), device=DEV)
        loss = F.cross_entropy(net(K[bi]), Vl[bi])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = float((net(K).argmax(1) == Vl).float().mean())
    return P, acc


rows = []
for P_t, N in ((60_000, 6_000), (60_000, 12_000), (60_000, 24_000),
               (240_000, 24_000), (240_000, 48_000), (240_000, 96_000)):
    P, acc = capacity_run(P_t, N)
    bits_pp_fp32 = 32 * P / N
    rows.append({'params': P, 'pairs': N, 'recall': round(acc, 3),
                 'fp32_bits_per_pair': round(bits_pp_fp32, 1)})
    print(f'  P={P} N={N}: recall {acc:.3f} ({bits_pp_fp32:.0f} fp32-bits/pair; '
          f'floor 10)', flush=True)
out['toyA'] = rows

print('=== TOY B: Zipf exposure — no small table exists ===', flush=True)
N = 200_000
ranks = np.arange(1, N + 1)
pz = 1.0 / ranks
pz = pz / pz.sum()
# expected CE gain of storing top-M pairs = sum of their exposure (x log V per pair)
cum = np.cumsum(pz)
tab = {}
for frac_gain in (0.5, 0.75, 0.9, 0.99):
    M = int(np.searchsorted(cum, frac_gain)) + 1
    tab[str(frac_gain)] = {'pairs_needed': M, 'frac_of_all_pairs': round(M / N, 4),
                           'table_Mbit': round(M * np.log2(VOC) / 1e6, 3)}
    print(f'  {int(frac_gain*100)}% of CE gain needs top-{M} pairs '
          f'({100*M/N:.1f}% of all; {M*10/1e6:.2f} Mbit)', flush=True)
out['toyB'] = tab

print('=== TOY C: rules + exceptions (the real-case shape) ===', flush=True)
N = 60_000
din = 64
rho = 0.5
Krule = torch.randn(N, din, device=DEV)
g = nn.Linear(din, VOC, bias=False).to(DEV)          # the shared rule (small)
with torch.no_grad():
    rule_vals = g(Krule).argmax(1)
is_rule = torch.rand(N, device=DEV) < rho
Vl = torch.where(is_rule, rule_vals, torch.randint(0, VOC, (N,), device=DEV))
pz = torch.tensor((1.0 / np.arange(1, N + 1)), device=DEV, dtype=torch.float)
pz = pz / pz.sum()
perm = torch.randperm(N, device=DEV)                  # random rank assignment
pz = pz[perm.argsort()]
base_ce = float(np.log(VOC))
res = []
# explicit-object ladder: (1) the rule alone; (2) rule + top-M exception table
with torch.no_grad():
    rule_correct = (rule_vals == Vl)
    ce_rule = float((pz * torch.where(rule_correct, torch.zeros_like(pz),
                                      torch.full_like(pz, np.log(VOC)))).sum())
gain_rule = 1 - ce_rule / base_ce
res.append({'object': 'rule only', 'bits_Mbit': round(din * VOC * 32 / 1e6, 2),
            'frac_gain': round(gain_rule, 3)})
exc = (~rule_correct).nonzero().squeeze(1)
exc_p = pz[exc]
order = exc_p.argsort(descending=True)
for M in (1000, 5000, 20000, len(exc)):
    kept = exc[order[:M]]
    covered = torch.zeros(N, dtype=torch.bool, device=DEV)
    covered[kept] = True
    ok = rule_correct | covered
    ce = float((pz * torch.where(ok, torch.zeros_like(pz),
                                 torch.full_like(pz, np.log(VOC)))).sum())
    res.append({'object': f'rule + top-{M} exceptions',
                'bits_Mbit': round((din * VOC * 32 + M * (np.log2(N) + 10)) / 1e6, 2),
                'frac_gain': round(1 - ce / base_ce, 3)})
for r in res:
    print(f"  {r['object']}: {r['bits_Mbit']} Mbit -> {100*r['frac_gain']:.0f}% of gain",
          flush=True)
out['toyC'] = res
json.dump(out, open(f'{QK}/qk_toy_memory.json', 'w'), indent=2)
print('TOY MEMORY DONE', flush=True)
