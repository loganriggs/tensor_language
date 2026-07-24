"""Toy examples for the two-ledgers explainer (tick 179). Three tiny, fully checkable
constructions whose numbers get embedded in qk_two_ledgers_explainer.md.

A. Function ledger: 8-token, 4-dim key table with 2 planted clusters. Raw bits vs
   2-atom dictionary bits; dCE of a toy bilinear LM (logits = score rows, targets =
   softmax of the true scores).
B. Mechanism ledger: 12 tokens in R^6, two planted classes (shared direction + noise).
   Third moment M = sum_t p_t y_t^(x3); rank-2 symmetric CP (power+deflation, same
   fitter as qk_stage23) recovers the class directions.
C. Decoupling: 2 tokens — 'the' frequent (p=.99, |y|=1) vs 'Kowalski' rare (p=.01,
   |y|=8). One-atom compression: fitting the frequent token wins the function metric,
   fitting the rare-big token wins the moment metric. Same story as heads 0/4.
"""
import torch
torch.manual_seed(0)
g = torch.Generator().manual_seed(1)

print('=== TOY A: function ledger ===')
V, d = 8, 4
c1 = torch.tensor([1., 0., 0., 0.])
c2 = torch.tensor([0., 1., 0., 0.])
K = torch.stack([c1] * 4 + [c2] * 4) + 0.05 * torch.randn(V, d, generator=g)
Q = torch.randn(V, d, generator=g)
S = Q @ K.T                                   # true scores; toy LM: logits row = S[i]
targets = torch.softmax(S, 1)
raw_bits = V * d * 32
atoms = torch.stack([K[:4].mean(0), K[4:].mean(0)])
code = torch.tensor([0] * 4 + [1] * 4)
Khat = atoms[code]
dict_bits = 2 * d * 32 + V * 1               # 2 atoms fp32 + 1-bit index per token
Shat = Q @ Khat.T
ce = lambda L: -(targets * torch.log_softmax(L, 1)).sum(1).mean()
print(f'raw {raw_bits} bits, dict {dict_bits} bits ({raw_bits / dict_bits:.1f}x)')
print(f'CE true {ce(S):.4f}  CE dict {ce(Shat):.4f}  dCE {ce(Shat) - ce(S):+.5f}')

print('=== TOY B: mechanism ledger ===')
V, d = 12, 6
u1 = torch.zeros(d); u1[0] = 1.
u2 = torch.zeros(d); u2[1] = 1.
Y = torch.stack([u1] * 6 + [u2] * 6) + 0.08 * torch.randn(V, d, generator=g)
p = torch.full((V,), 1. / V)
M = torch.einsum('t,ta,tb,tc->abc', p, Y, Y, Y)


def cp_power(core, R, iters=200, starts=8):
    res = core.clone(); Us, lams = [], []
    for _ in range(R):
        M1 = res.reshape(d, d * d); best_u, best_l = None, -1.
        for s in range(starts):
            u = torch.rand(d, generator=g); u /= u.norm()
            for _ in range(iters):
                u = (M1 @ (u[:, None] * u[None, :]).reshape(-1)).clamp_min(0)
                if u.norm() < 1e-20: break
                u = u / u.norm()
            l = float(torch.einsum('abc,a,b,c->', res, u, u, u))
            if l > best_l: best_l, best_u = l, u
        Us.append(best_u); lams.append(best_l)
        res -= best_l * torch.einsum('a,b,c->abc', best_u, best_u, best_u)
    return torch.stack(Us, 1), lams, float(res.norm() / core.norm())


U, lams, rel = cp_power(M, 2)
cos = (U.T @ torch.stack([u1, u2], 1)).abs()
print(f'lambdas {[round(l, 4) for l in lams]}, residual {rel:.3f}')
print(f'matched cosines to planted classes: {cos.max(1).values.tolist()}')
# permutation null: shuffle coordinates of each row independently -> class structure gone
Yn = torch.stack([y[torch.randperm(d, generator=g)] for y in Y])
Mn = torch.einsum('t,ta,tb,tc->abc', p, Yn, Yn, Yn)
_, _, rel_n = cp_power(Mn, 2)
print(f'rank-2 residual on permuted null: {rel_n:.3f} (vs {rel:.3f} real)')

print('=== TOY C: decoupling ===')
d = 4
y_the = torch.zeros(d); y_the[0] = 1.
y_rare = torch.zeros(d); y_rare[1] = 8.
Y = torch.stack([y_the, y_rare])
p = torch.tensor([0.99, 0.01])
M = torch.einsum('t,ta,tb,tc->abc', p, Y, Y, Y)
for name, atom in (('fit-frequent', y_the), ('fit-rare-big', y_rare)):
    A = atom / atom.norm()
    coeff = Y @ A                              # 1-atom nonneg code
    Yhat = coeff.clamp_min(0)[:, None] * A
    func = float((p * ((Yhat - Y) ** 2).sum(1)).sum())          # exposure-weighted (~dCE)
    Mh = torch.einsum('t,ta,tb,tc->abc', p, Yhat, Yhat, Yhat)
    mom = float((Mh - M).norm() / M.norm())
    print(f'{name}: function-damage {func:.4f}   moment-residual {mom:.4f}')
print(f'moment mass p|y|^3: the {0.99 * 1:.2f} vs rare {0.01 * 512:.2f}  (rare wins)')
print(f'function mass p|y|^2: the {0.99 * 1:.2f} vs rare {0.01 * 64:.2f}  (the wins)')
