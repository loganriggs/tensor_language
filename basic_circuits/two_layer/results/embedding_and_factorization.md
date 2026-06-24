# The 2-D convex embedding, and how to enforce a factorized solution

Two follow-ups to the 4-hot toy (`toy_2layer_4hot.py`), where 15 mutually-exclusive
4-AND outputs are computed by a tiny `h1=2, h2=2` net that turned out **not** to use
genuine conjunctions.

## What the "2-D convex embedding" is

The 4-hot task is **one-hot detection**: 15 inputs (each a distinct 4-subset),
output `t` should fire iff the input *is* pattern `t`. The net maps each input
through layer 1 then layer 2 to a point `g ∈ ℝ²` (because `h2 = 2`), and each output
is a **linear** readout `logit_t = Wo[t]·g + bo[t]`.

A linear readout can fire on exactly one of the 15 points (and not the other 14)
**iff that point is a vertex of the convex hull** — there's a hyperplane with it on
one side and the rest on the other. For *all* 15 outputs to work, *all* 15 points
must be hull vertices, i.e. the 15 patterns sit in **convex position** — a convex
15-gon in the 2-D `g`-space. That is exactly what we see (`fig_toy2L_4hot_embed`:
15/15 on the hull).

So the network solves the task **geometrically**: it scatters the 15 patterns as a
convex polygon and slices off each vertex with a hyperplane. It is *not* computing
"are features a,b,c,d all on?" — the underlying degree-4 polynomial is whatever
produces that embedding (dense, distributed; genuine 4-AND coeff ≈ 1%).

**Why `h2 = 2` is enough:** you can place arbitrarily many points in convex position
in 2-D (on a circle / moment curve), so a 2-D bottleneck can one-hot-separate *any*
number of classes. The geometric trick is cheap, which is precisely why the sweep's
minimum is `(2,2)` — and why the genuine, factorized solution (which needs many edge
detectors) never appears.

## How to enforce the factorized solution

The factorized solution computes each output as a genuine product of sparse edges,
`(x_a x_b)(x_c x_d)` — a *sparse* quartic that `sparse_pursuit.py` could read off as
clean edges. Pushing the net there is hard, because of a real tension:

- **Superposition (small h) ⇒ geometric.** The convex-embedding solution is low-norm
  *and* uses few units, so it is the optimiser's default and is favoured by ordinary
  weight penalties (L1/L2 on weights *reinforce* it rather than dislodge it).
- The factorized solution needs **enough width to hold the edge factors**: layer-1
  must supply the needed single-edge detectors (`x_a x_b`) and layer-2 the products.
  Clean factors therefore live in the **wide, non-superposed** regime — the opposite
  of superposition.

Levers, roughly in order of effectiveness:

1. **Penalise representation density, not weight density.** Put L1 on the *folded
   quartic* (or its square-free polynomial coefficients), so a few-monomial
   (sparse-conjunction) solution beats the dense geometric one. This is the 2-layer
   analogue of the 1-layer **Qf-L1** experiment (`../../sparsity/`), which *did* drive
   the pullback sparse. It is the most direct knob but the most expensive (you must
   fold the quartic each step).
2. **Give it width.** With `h1 ≥ #edges` and `h2 ≥ #products` the factorized solution
   becomes representable; combine with (1) so it is also preferred.
3. **Remove the geometric escape.** Make the task big enough (many inputs/features)
   that no low-dimensional convex embedding can separate everything, forcing genuine
   computation. Necessary but not sufficient — the 5-hot/21-input toy was already
   non-degenerate yet still distributed.

### What weight-L1 alone does (`enforce_factorized.py`)

L1 on the **layer-1 weights** (with extra width `h1=h2=8`) only goes part way:

| model | acc | layer-1 reads / neuron |
|---|---|---|
| geometric `h1=2,h2=2`, no L1 | 100% | — |
| wide, L1 = 0 | 100% | ~4.2 |
| wide, L1 = 0.02–0.05 | 100% | ~2.9 |

(2.0 reads/neuron would be a clean edge `x_i x_j`.) So weight-L1 sparsifies the
layer-1 reads from ~4.2 toward ~2.9 features but **does not reach edges** — the geometric
pull persists. Enforcing a genuinely factorized model needs the representation-level
penalty of lever (1), which is the open sparse-pursuit / bond-canonicalisation
problem (`../CONTEXT.md` threads #3/#4); `sparse_pursuit.py` is then the read-out tool
once a sparse structure actually exists.
