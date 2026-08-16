"""Diagnostic for A2: where does the mass of the grokked {Q_c} actually live,
and how far is the family from exactly block-diagonalisable?"""

import math
import sys

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
import a2_modular as m
from bq_common import interaction, support_basis, restrict, commutant_basis, sbd, block_mass

torch.set_default_dtype(torch.float64)
P = m.P
cache = torch.load('a2_cache.pt', weights_only=False)
p0 = cache[0]['p']
Q = interaction(p0)
print('test acc', cache[0]['hist'][-1]['test_acc'])

# --- 2-D Fourier power: a-side frequency x b-side frequency
t = torch.arange(P, device=Q.device, dtype=Q.dtype)
FB = [torch.ones(P, device=Q.device, dtype=Q.dtype)[None] / math.sqrt(P)]
for w in range(1, P // 2 + 1):
    FB.append(torch.stack([torch.cos(2 * math.pi * w * t / P),
                           torch.sin(2 * math.pi * w * t / P)]) * math.sqrt(2 / P))
nf = len(FB)
Ea = torch.zeros(nf, nf)          # cross a-b block
Es = torch.zeros(nf, nf)          # same-side (a-a and b-b) blocks
tot = float((Q ** 2).sum())
Qab, Qaa, Qbb = Q[:, :P, P:], Q[:, :P, :P], Q[:, P:, P:]
for i in range(nf):
    for j in range(nf):
        Ea[i, j] = float((torch.einsum('pi,mij,qj->mpq', FB[i], Qab, FB[j]) ** 2).sum())
        Es[i, j] = float((torch.einsum('pi,mij,qj->mpq', FB[i], Qaa, FB[j]) ** 2).sum() +
                         (torch.einsum('pi,mij,qj->mpq', FB[i], Qbb, FB[j]) ** 2).sum())
print(f'\nmass: cross a-b {2*Qab.pow(2).sum()/tot:.4f}  same-side {(Qaa.pow(2).sum()+Qbb.pow(2).sum())/tot:.4f}')
print(f'cross block: diagonal (w_a=w_b) {Ea.diag().sum()/Ea.sum():.4f} of cross mass')
print('top cross (w_a,w_b) shares:',
      [(int(i), int(j), round(float(Ea[i, j] / Ea.sum()), 4))
       for i, j in zip(*torch.where(Ea > 0.02 * Ea.sum()))])
print('same-side diag shares:',
      [(int(i), int(j), round(float(Es[i, j] / max(Es.sum(), 1e-30)), 4))
       for i, j in zip(*torch.where(Es > 0.05 * max(Es.sum(), 1e-30)))])
print('per-frequency total share:',
      [(w, round(float((Ea[w, w] + Es[w, w]) / tot), 4)) for w in range(nf)])

# --- support spectrum
U, sv = support_basis(Q, thresh=1e-3)
print('\nsupport singular values (of the d x md unfolding):')
print('  ', ' '.join(f'{float(v):.3g}' for v in sv))
cum = (sv ** 2).cumsum(0) / (sv ** 2).sum()
print('  dims to reach 99%/99.9% energy:', int((cum < 0.99).sum()) + 1, int((cum < 0.999).sum()) + 1)

# --- how close to exactly block diagonalisable? commutator Gram spectrum
for keep_energy in (0.99, 0.999, 1.0):
    k = int((cum < keep_energy).sum()) + 1 if keep_energy < 1 else U.shape[1]
    Ur = U[:, :k]
    Qr = restrict(Q, Ur)
    B, ev = commutant_basis(Qr)
    evn = ev / ev.max()
    print(f'\nrestricted to {k} dims (energy {keep_energy}):  commutator Gram spectrum (normalised, lowest 16)')
    print('  ', ' '.join(f'{float(v):.2e}' for v in evn[:16]))
