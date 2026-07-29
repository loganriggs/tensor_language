"""Lesson 2 (the gauge trap): the neuron basis is a CHOICE, not a fact. Only the folded tensor is
real — and even part of that is invisible. Two demonstrations on a small self-interaction bilinear
MLP  out = Dn( (L h) ⊙ (R h) ),  folded  T[o,i,j] = Σ_p Dn[o,p] L[p,i] R[p,j].
(1) A GAUGE that changes every neuron's activation (rescale neuron p by α_p in L, compensate 1/α_p
    in Dn) leaves the function and T IDENTICAL. So per-neuron activations are arbitrary.
(2) The raw tensor's ANTISYMMETRIC part is pure gauge (both legs get the same h, so it cancels):
    computing the output from only the symmetric part is identical.
"""
import json
import numpy as np

rng = np.random.default_rng(0)
d, m, dout = 8, 16, 6
L = rng.standard_normal((m, d)); R = rng.standard_normal((m, d)); Dn = rng.standard_normal((dout, m))
H = rng.standard_normal((5, d))                       # sample inputs

def act(Lm, Rm): return (H @ Lm.T) * (H @ Rm.T)       # (n, m) neuron activations
def out(Lm, Rm, Dm): return act(Lm, Rm) @ Dm.T        # (n, dout)
def fold(Lm, Rm, Dm): return np.einsum('op,pi,pj->oij', Dm, Lm, Rm)

A0 = act(L, R); O0 = out(L, R, Dn); T0 = fold(L, R, Dn)
# folded reproduces the layer
Ofold = np.einsum('oij,ni,nj->no', T0, H, H)
fold_err = float(np.abs(Ofold - O0).max())

# (1) activation-rescaling gauge: neuron p -> alpha_p in L, 1/alpha_p in Dn
alpha = np.exp(rng.standard_normal(m) * 1.1)          # arbitrary per-neuron scales
Lg = L * alpha[:, None]; Dng = Dn / alpha[None, :]
Ag = act(Lg, R); Og = out(Lg, R, Dng); Tg = fold(Lg, R, Dng)
gauge_out_err = float(np.abs(Og - O0).max())
gauge_T_err = float(np.abs(Tg - T0).max())
# also permute neurons (labels arbitrary)
perm = rng.permutation(m)
Tp = fold(L[perm], R[perm], Dn[:, perm])
perm_T_err = float(np.abs(Tp - T0).max())

# (2) antisymmetric part is gauge
Tsym = 0.5 * (T0 + T0.transpose(0, 2, 1))
antisym_frac = float(np.linalg.norm(T0 - Tsym) / np.linalg.norm(T0))
Osym = np.einsum('oij,ni,nj->no', Tsym, H, H)
sym_out_err = float(np.abs(Osym - O0).max())          # identical -> antisym invisible

res = {'d': d, 'm': m, 'dout': dout, 'fold_err': fold_err,
       'sample_activations_original': [round(float(x), 2) for x in A0[0]],
       'sample_activations_regauged': [round(float(x), 2) for x in Ag[0]],
       'output_original': [round(float(x), 3) for x in O0[0]],
       'output_regauged': [round(float(x), 3) for x in Og[0]],
       'gauge_output_maxerr': gauge_out_err, 'gauge_tensor_maxerr': gauge_T_err,
       'permutation_tensor_maxerr': perm_T_err,
       'antisymmetric_fraction': round(antisym_frac, 3), 'symmetric_only_output_maxerr': sym_out_err}
json.dump(res, open('tn_gauge_demo.json', 'w'), indent=1)
print(f"fold err {fold_err:.1e} | REGAUGE: output err {gauge_out_err:.1e}, tensor err {gauge_T_err:.1e} "
      f"(function IDENTICAL, activations changed) | perm tensor err {perm_T_err:.1e}", flush=True)
print(f"antisymmetric fraction {antisym_frac:.3f}; output from symmetric part only, err {sym_out_err:.1e} (INVISIBLE)", flush=True)
print(f"orig act[0][:5] {res['sample_activations_original'][:5]} vs regauged {res['sample_activations_regauged'][:5]}", flush=True)
print('TN GAUGE DEMO DONE', flush=True)
