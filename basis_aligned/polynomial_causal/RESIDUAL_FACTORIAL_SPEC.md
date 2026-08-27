# Factorial residual cross-tab specification

## Why the current marginals are insufficient

The current ship is about `+0.9345` CE above clean. Two useful but non-joint
localizations exist:

- target cells: copy `0.258`, novel/frequent `0.270`, novel/rare `0.473` of damage;
- within the top-100 frequent targets only: attention `0.137`, MLP0-2 `0.499`,
  deep MLPs `0.364`.

These do not show that MLP0-2 causes the novel/rare error. Novel/rare targets are
outside the top-100 target set by construction. Multiplying `0.500 * 0.499` says
that the registered sequential MLP0-2 increment occupies `0.2495` of total ship
damage *inside the top-100 partition*; it says nothing about group attribution in
the other half. The current four arms are also order-dependent, while existing
whole-model evidence shows large replacement interactions.

## Registered factorial audit

Treat the replacement groups as Boolean variables

```
A = attention planks
M = MLP0-2 planks and correction
D = MLP3-17 planks
```

Run all eight subsets `{}, {A}, {M}, {D}, {A,M}, {A,D}, {M,D}, {A,M,D}` with the
same frozen fits. On discovery and held-out rows separately, retain loss sums and
counts for:

1. copy, novel/frequent, and novel/rare targets;
2. top-100 and non-top-100 targets as a denominator cross-check;
3. powered behavior/output slices already in the circuit registry;
4. each held-out intervention family used for admission of a compiler fragment.

For every cell, let `v(S)` be its mean CE or response error. Report its exact Boolean
Mobius coefficients

```
m(T) = sum_{U subset T} (-1)^(|T|-|U|) v(U)
```

and allocate each interaction equally among its members to obtain the Shapley value

```
phi_i = sum_{T containing i} m(T) / |T|.
```

Raw signed nats are primary. Shares are secondary and may be emitted only after the
cell counts close the global denominator. Negative effects and interactions are not
clipped. `factorial_causal_attribution.py` is the frozen CPU scorer.

## Preregistered decisions

1. **Early-content license.** Advance the MLP0-2 content compiler as a novel/rare
   ship correction only if held-out `phi_M >= 0.05` nats within novel/rare and its
   share of the cell's full damage is at least `0.20`.
2. **Split target.** If that gate fails while MLP0-2 remains dominant on top-100,
   split frequent construction from novel/rare content; do not advertise one
   correction as serving both.
3. **Joint compilation.** If the L1 norm of pair/triple Mobius terms exceeds `20%`
   of the full effect in a target cell, optimize that cell under the current joint
   ship. Marginal module replacements are not admissible evidence there.
4. **Stability.** The sign and dominant group of every powered cell must agree on
   discovery and held-out rows. Otherwise report the attribution as unresolved.
5. **Closure.** Empty/full arms must reproduce the frozen clean/ship anchors within
   `0.01` CE, Shapley allocations must close to floating-point tolerance, and the
   weighted cell totals must reproduce global damage within `0.005` CE.

This audit chooses the target of the next compiler; it does not itself increase the
fraction of the model reverse engineered.
