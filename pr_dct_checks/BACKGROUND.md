# Background: what the checks test and why

## The method in one paragraph

Perturb the residual at source layer s by θ = αl + βr, and let Δ be the change
at target layer t, averaged over the last few positions. The DCT term is
u⊤H[l, r] = ∂²(u·Δ)/∂α∂β at 0. It measures how much l and r *jointly* push the
output along u, beyond what each does alone.

AJ adds, for each intermediate MLP hidden unit j, the same mixed derivative of
that unit's activation, c_j. He then penalizes PR(c) = (Σc_j²)² / Σc_j⁴, the
effective number of units carrying the interaction. His objective is
½(u⊤H[l,r])² − λ·S·PR/D.

## Why PR alone can't establish a circuit

PR is scale-invariant. Multiplying c by 0.01 leaves it unchanged. So PR measures
how *concentrated* the MLP-mediated part of the interaction is, never how *much*
of the interaction is MLP-mediated. Separately, AJ's c only covers blocks
s+1 … t−1. The source block's MLP, which sees l and r directly, and every
attention layer are invisible to it.

The toy test `test_completeness_catches_pr_loophole` builds two networks where
the interaction has almost identical strength and PR is ≈ 1.00 in both:

- **"Loophole":** the interaction happens in the source-block MLP, with a 5%
  remainder in one intermediate unit. Freezing the top intermediate unit
  removes **4.8%** of u⊤H[l, r]; freezing the source MLP removes **95%**.
- **"Genuine":** the interaction happens in one intermediate unit. Freezing it
  removes **100%**.

PR cannot tell these apart. Completeness can.

## Freezing semantics

Freezing a unit replaces its activation with its clean value (the value at
θ = 0). For derivatives at θ = 0 this is equivalent to `.detach()`, and
`apply_freeze` uses detach when no clean cache is supplied. Freezing removes
two things: interactions the unit *creates*, and upstream interactions it
*transmits* linearly. That's the right notion of "the circuit passes through
this unit."

Completeness fractions for different groups need not sum to 1. Pathways can
cancel (giving signed or negative fractions), and downstream nonlinearities
create cross terms between groups.

E3 is the finite-size version: it patches to clean and uses the finite mixed
difference. On the toy, it agrees with E1.

## Why check against individual neurons (E2)

A bilinear hidden unit computes (a·n)(b·n), where n is the MLP's normalized
input. To first order in θ, n ≈ n₀ + Jθ, so the unit's mixed derivative along
(l, r) is (a·Jl)(b·Jr) + (a·Jr)(b·Jl), plus terms from upstream curvature. The
unit is therefore a ready-made DCT factor with effective inputs (J⊤a, J⊤b).

The cheapest way to get PR ≈ 1 is to set (l, r) ∝ (J⊤a, J⊤b) for one unit. E2
asks whether the penalized factors are doing that. Alignment is L/R-swap
symmetric, and should be compared to `E2_null`: with thousands of units, a
random pair already gets a nonzero best-match score.

Finding that factors *are* neuron-aligned isn't damning. In a bilinear MLP the
hidden basis is privileged: there's no rotation symmetry, only permutation,
rescaling, and the L/R swap. So "this interaction runs through neuron 1234" is
meaningful. But it would mean the method's contribution is finding *which*
neurons matter, not finding new structure.

## Why stability matters (E4)

The per-factor numbers ("keeps 70–100% of strength") only mean something if
the same factors come back from different seeds and different data. Matching
uses Hungarian assignment on |cos u| · max(|cos l||cos r|, swapped), so it's
invariant to factor order, sign, and L/R swap. The null is two random
dictionaries of the same shape.

## Weight-space identities (E5, toy only)

For a norm-free residual stack of bilinear blocks z ← z + D[(Az)⊙(Bz)]:

- **One block:** H[l, r] = D[(Al)⊙(Br) + (Ar)⊙(Bl)], independent of x. The DCT
  objective here is a pure weight quantity: the interaction tensor contracted
  with (u, l, r).
- **Two blocks:** H[l, r] is a degree-2 polynomial in x. Its average over any
  dataset equals the closed form evaluated at the dataset's mean and covariance.
  The test checks this to 1e-9. The "weight-space" DCT therefore needs only
  input moments, not per-datapoint passes.

The real TensorGPT applies RMSNorm before each MLP. That makes these identities
approximate at best there; they are exact only for norm-free variants.
