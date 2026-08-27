# Factorial residual cross-tab specification

## Why the current marginals are insufficient

The current ship is about `+0.9345` CE above clean. Two useful but non-joint
localizations exist:

- target cells: copy `0.258`, novel/frequent `0.270`, novel/rare `0.473` of damage;
- within the top-100 frequent targets only: attention `0.137`, MLP0-2 `0.499`,
  deep MLPs `0.364`.

These do not show that MLP0-2 causes the novel/rare error. Novel/rare targets are
mostly outside the top-100 **most-frequent-token** set. More importantly, the
published `0.500` and `0.499` use different sets: `0.500` is the share carried by
the 100 most-damaged token types, whereas `0.499` is the sequential MLP0-2 share
within the 100 most-frequent token types. Multiplying them is invalid. The current
four arms are also order-dependent, while existing whole-model evidence shows
large replacement interactions.

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

## Completed token-cell stage

`ship_error_factorial_results.json` completed the eight arms on 480 discovery and
480 untouched held-out rows. The output-slice and intervention-family extensions
remain pending.

The v1 token-cell labels have two post-run audit caveats. Its recurrence mask uses
context distances 2 through 65 rather than the intended 1 through 64, and it
recomputes the rare vocabulary independently on each split. Therefore the global
factorial and exact group allocations remain valid, but copy/novel cell values are
provisional engineering localizations rather than fixed-stratum replication. New
screens freeze the discovery vocabulary and use the corrected lag definition; a
future factorial v2 must do the same without overwriting this preserved result.

On held-out rows, the full ship adds `0.8727` nats. The cell damage shares are copy
`0.2486`, novel/frequent `0.2818`, and novel/rare `0.4697`. Exact weighted Shapley
effects are attention `-0.0670`, MLP0-2 `+0.7277`, and deep MLPs `+0.2120` nats.
MLP0-2 is the dominant group in every cell on both splits. In novel/rare its
held-out Shapley effect is `1.0776 / 1.1755` nats, so the early-content license
passes decisively.

The result does **not** license an independently optimized early module. The
held-out interaction L1 fractions are `0.429` (copy), `0.576` (novel/frequent),
and `0.636` (novel/rare), all beyond the registered joint-compilation gate. In
novel/rare the largest term is the attention x MLP0-2 interaction at `-0.6389`
nats. The next correction must therefore read live full-ship activations and be
trained and evaluated with all other replacements installed.

As a denominator cross-check, the 100 most-frequent token types carry `0.3187` of
held-out ship damage. This is not the earlier `0.500` result for the 100
most-damaged token types, and both labels must remain explicit in downstream
reports.
