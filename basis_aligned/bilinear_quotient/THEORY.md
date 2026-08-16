# The mathematics behind each setting — what is already known, what is provable, and which "results" were theorems all along

Every proposition here is verified numerically by `theory_checks.py` (**47/47 pass**).
The point is not decoration. Three things came out of writing it:

- **two of the program's headline "findings" are algebraic identities**, so the
  experiments that produced them could not have returned anything else, and they are
  relabelled below and in `RESULTS.md`;
- one of them (T8) independently reproduces a bug Reviewer 2 found, and turns it from a
  bug report into a **rule for which statistics are admissible**;
- the corrected argument for bilin18 runs along a different axis than the one the
  connection document used (T6).

Notation throughout: a bilinear layer `y_i = xᵀQ_i x` with `Q_i` symmetric, `x ∈ ℝ^d`,
`i = 1..m`; `Sym²(V)` the space of symmetric `d×d` matrices, dimension `d(d+1)/2`.

---

## T1. The layer is a linear map on the Veronese lift, and that is the whole framework

`y_i = ⟨Q_i, xxᵀ⟩`, so the layer is the composition of the quadratic Veronese embedding
`x ↦ xxᵀ` with a **linear** map `W : Sym²(V) → ℝ^m`. Everything the program does is linear
algebra on `W`; the nonlinearity is entirely in a fixed, known, model-independent lift.

**Proposition 1.** A direction `δ` is invisible — `y(x + tδ) = y(x)` for all `x` and all
`t` — if and only if `Q_i δ = 0` for every `i`.

*Proof.* `y_i(x+tδ) − y_i(x) = 2t·xᵀQ_iδ + t²·δᵀQ_iδ`. If `Q_iδ = 0` both terms vanish.
Conversely, if it holds for all `x, t`, then the coefficient of `t` vanishes for all `x`,
so `Q_iδ = 0`. ∎

So the "blind subspace" is `⋂_i ker(Q_i)`, an exact algebraic object, and A1's causal
verification is a test of Proposition 1 rather than a discovery. Verified: a planted blind
direction gives `max|Δy| = 1.8e-15` at `t = 0.5` and `6.3e-15` at `t = 10`, against `4.9`
for a row-space direction.

Worth noting what this rules out: for a generic layer with `m·d ≥ d` there is **no** blind
direction at all. Blindness in A1 exists because the task planted it.

**Proposition 2 (the data quotient).** Fix a dataset `X`. Two form families agree as
functions on `X` iff they agree after orthogonal projection onto `span{xxᵀ : x ∈ X}` inside
`Sym²(V)`. The projection is therefore the minimum-norm representative of the function.

This is A2-3's canonicalisation, and it is a definition-level fact, not a finding. What is
empirical is *how big* the unidentifiable part is — and see the correction in `RESULTS.md`,
because the trained model is well *above* chance on that measure, not below.

---

## T2. The gauge group, and why neurons are never the parts

The function determines `Q` and nothing more. Two parameterisations `(L,R,D)` and
`(L',R',D')` are functionally identical iff `sym(Σ_k D_ik l_k r_kᵀ)` agrees for every `i`.
That fibre is enormous: the familiar neuron permutation and rescaling
`(l_k, r_k, D_{:,k}) → (a l_k, b r_k, D_{:,k}/ab)` is a tiny subgroup of it. `gauge_refactor`
exploits the full freedom by re-solving for `D` against a *fresh random* `L', R'`, which is
why it can move the hidden width from 128 to 300 while changing the function by 2e-13.

The consequence used throughout: any statistic that is not a function of `Q` alone is not a
statistic about the computation. This is the standing rule in `qk_mdl/METHODS.md:26-27`;
T8 below shows Part B violated it.

---

## T3. A4's "curvature, not gain" is a theorem, and the gain cancels identically

This is the biggest relabelling. Let a component contribute `f(x) = xᵀSx`, with inputs
`x = μ + δ`, `δ ~ N(0, Σ)`. Linearising at the mean gives the tangent
`2μᵀSx − μᵀSμ`, so the residual left by straightening is **exactly**

    f(x) − tangent(x) = δᵀSδ

with no approximation. Applying Isserlis' theorem to both surgeries:

    err_lin   = E[(δᵀSδ)²]  = 2 tr(SΣSΣ) + (tr SΣ)²
    err_prune = E[(xᵀSx)²]  = (μᵀSμ + tr SΣ)² + 4 μᵀSΣSμ + 2 tr(SΣSΣ)

**Proposition 3.** The ratio `err_lin / err_prune` is invariant under `S → γS`.

*Proof.* Every term in the numerator is homogeneous of degree 2 in `S`, and so is every
term in the denominator. The factor `γ²` cancels. ∎

That is A4-3. The measured correlation of `+0.000` between straightenability and the
planted gain is **exact and forced**, not an empirical near-zero — which is why it came out
to four significant figures. In A4's own design (`a,b` orthonormal, `Σ = I`, `a·μ = b·μ = ρ`)
the ratio collapses to

    err_lin / err_prune = 1 / (1 + ρ²)²

giving 1, 1/25 = 0.04, and 1/101² = 9.803e-5 at ρ = 0, 2, 10 — against measured
1.0000000, 0.039817, 9.691e-5. Verified to 12 decimal places in closed form and to
Monte-Carlo error against samples, for general non-orthonormal `S`, general `Σ`, general `μ`.

**What this means for A4.** The claim "curvature governs straightenability, gain does not"
is *true and useful* but it is a derivation. The experiment confirmed arithmetic. Two
things remain genuinely empirical: that the components exist and are recoverable at all
(which the nulls confirm is task-specific), and the Pareto frontier magnitudes. And
Proposition 3 explains A4-5 — the nulls found the linearization advantage in a *randomly
initialised* network because the ratio does not depend on the weights' provenance at all,
only on `S`, `μ` and `Σ`.

**The corollary that is actually actionable.** Proposition 3 says a truncation that ranks
components by size is answering a different question from "what must stay quadratic". But
note the ranking that matters for a *budget* is neither: it is `err_prune − err_lin` per
parameter saved, which does depend on the gain. Size and curvature each answer half.

---

## T4. A5's spectrum is exactly computable, and the R+1 identity is knife-edge

Let `Q_u = g·S₀ + h·S_u` for `u = 1..R` with `{S₀, S₁, …, S_R}` orthonormal in the metric
used to build the moment. In those coordinates the reader-weighted second moment is

    M = [[ R g² ,  g h 1ᵀ ],
         [ g h 1,  h² I   ]]

**Proposition 4.** `M` has eigenvalues

    λ± = ½ [ R g² + h² ± √((R g² − h²)² + 4 R g² h²) ],   and  h²  with multiplicity R−1.

*Proof.* The `R−1` dimensional space of private vectors summing to zero is annihilated by
the coupling and scaled by `h²`. On the remaining 2-dimensional space spanned by `S₀` and
the mean private direction the matrix is `[[Rg², √R gh],[√R gh, h²]]`, whose eigenvalues are
the stated roots. ∎

At `g = h = 1` the discriminant is `(R+1)²`, so `λ₊ = R+1`, `λ₋ = 0`, and the ratio to the
next eigenvalue is exactly `R+1` — verified 4.000, 6.000, 9.000 at R = 3, 5, 8. The top
eigenvector is `∝ (R, 1, …, 1)`, so its overlap with the pure shared form is

    √(R / (R+1))  =  0.866 at R = 3.

**Both corrections to the plan in A5-1 are theorems.** And Proposition 4 exposes a fragility
the experiment hid by using equal gains: the ratio is `R+1` *only* at `g = h`. At `g = 1.2,
h = 1` it is 5.32; at `g = 2` it is 13.0; at `h = 2` it is 1.75. So "the top eigenvalue is
about R times the private ones" is not a usable diagnostic for reader multiplicity unless
the gains are known to be balanced — which in a real model they are not.

---

## T5. A2's block structure is Wedderburn plus a ℤ₂, and the multiplicity is forced

The forms for `(a+b) mod p` commute with the exchange operator `S` swapping the two input
slots — verified `max |[Q_c, S]| = 0.0` exactly, which is just the commutativity of the
group operation. The algebra generated by `{Q_c}` is a real matrix ∗-algebra, so
Artin–Wedderburn decomposes it into isotypic components; the ℤ_p structure indexes those by
frequency, and the ℤ₂ exchange symmetry gives each frequency multiplicity 2.

**Proposition 5.** For an isotypic component of multiplicity `μ` carrying a real-type
irreducible, the *symmetric* commutant contributes `μ(μ+1)/2` dimensions and the component
splits into `μ` blocks whose individual identity is not canonical.

At `μ = 2` this gives 3 symmetric commutant dimensions per frequency, so for `p = 23` with
11 frequencies the commutant dimension is `3 × 11 = 33` — verified exactly, alongside 22
fine blocks and 11 isotypic components.

So A2-1's "22 blocks, not 11, and only the isotypic component is canonical" is
representation theory, and the experiment confirmed it. What the experiment genuinely
contributed is the instrument calibration (A2-2), which is not derivable.

---

## T6. A3's identifiability limit is Kruskal's, and the binding axis is the *output* dimension

Kruskal's theorem: a rank-`R` three-way tensor with factor matrices `A, B, C` has an
essentially unique CP decomposition when `k_A + k_B + k_C ≥ 2R + 2`, where `k` denotes
Kruskal rank. For an `(m, d, d)` partially-symmetric family with generic factors,
`k_A = k_B = min(R, d)` and `k_C = min(R, m)`, giving guaranteed uniqueness up to

    R ≤ (m + 2d − 2) / 2.

**This is Reviewer 2's finding, and it is right.** A3 fixed `m = 8, d = 16` and varied only
`R`, so its observed breakdown between R = 24 and R = 32 is equally consistent with a limit
in `R/d` and one in `R/m` — and the bound above says the binding mode is `k_C ≤ m = 8`.
Verified directly: the form family's effective rank is **8 at every K** (8, 24, 48), i.e.
exactly `m`, independent of the number of planted components.

Kruskal gives `R ≤ 19` for A3's setting, and A3 measured exact recovery to R = 24: Kruskal
is sufficient, not necessary, and generic identifiability results reach further. That is
consistent, not a contradiction.

**The corrected bilin18 argument.** `BILIN18_CONNECTION.md` argued from `K/d = 4608/1152 = 4`
being "twice" A3's `K/d ≈ 2`. That extrapolates along an axis A3 cannot resolve. The right
argument uses the same bound with bilin18's real shape, `m = d = 1152`:

    Kruskal guarantees uniqueness only to R ≤ (1152 + 2·1152 − 2)/2 = 1727,
    and bilin18's MLP has R = 4608.

So the conclusion survives — a 4608-component decomposition of a bilin18 MLP is far outside
any uniqueness guarantee — but for a reason that does not depend on A3's one-point
extrapolation. The connection document has been corrected.

---

## T7. PCA and a dictionary *must* tie on error; identification needs a constraint

**Proposition 6.** If the planted parts span a `k`-dimensional subspace of `Sym²(V)`, then
at budget `k` *any* basis of that subspace reconstructs the data exactly. Consequently PCA
and a sparse dictionary have identical (zero) error at the true budget, and error-versus-
budget cannot distinguish them.

*Proof.* Reconstruction depends only on the span (Eckart–Young gives the optimal span; any
basis of it spans it). Verified: an arbitrary orthogonal rotation of the planted atoms
reproduces the data to 6.3e-16. ∎

This is the classical rotation indeterminacy of factor analysis, and it is exactly A5-5.
Identifying *which* basis requires an extra constraint — sparsity, non-negativity,
independence — that is not implied by fit. And Reviewer 2's observation follows too: PCA
returns an *orthonormal* triple, so when the planted atoms have mutual cosine 0.64 no PCA
output can match all three, by definition. A5-5 is therefore also a theorem, and the
empirical content is only the rate at which the dictionary degrades (which does fail at
overlap 0.95).

**The consequence for the plan's model-selection protocol.** The plan says to choose among
candidate structures by plotting functional error against description length. Proposition 6
says that criterion is blind to the distinction the program cares about whenever the
candidates span the same subspace. Model selection by fit cannot substitute for calibration
against planted structure.

---

## T8. Part B has a scale gauge, so per-factor statistics must be scale-invariant

For any product placement — `(qᵀW₁k)(qᵀW₂k)` before a softmax, or unnormalised —

    (W₁, W₂) → (c W₁, W₂ / c)

leaves the function **exactly** invariant. Verified: function change 2.7e-16 (score-level)
and 8.0e-16 (unnormalised) under `c = 20`.

**Proposition 7.** A per-factor statistic is meaningful only if invariant under that
rescaling. The entropy of `softmax(qᵀW_i k)` is not; the participation ratio
`PR(w) = (Σw²)²/Σw⁴` is.

*Proof.* Softmax entropy depends on the scale of its argument (as temperature); PR is
homogeneous of degree 0. ∎

Measured under `c = 20`: softmax entropies move `[2.768, 2.769] → [1.799, 2.773]` while
participation ratios are unchanged to 9 significant figures.

**This retracts the entropy numbers in B2-2 for the score-level placement** — Reviewer 2
found the same thing empirically. It also *justifies* the statistic B0 switched to on
independent grounds: PR was chosen because bilin18's pattern is signed and unnormalised so
entropy is undefined, and Proposition 7 says PR is the right choice even where entropy *is*
defined. Two separate arguments landing on the same statistic.

Note the post-softmax placement has no such gauge — each factor is separately normalised —
so its entropies are meaningful, and B2-4's specialisation result (which is a post-softmax
result) is unaffected.

---

## What this exercise changed

| result | was presented as | actually is |
|---|---|---|
| A4-3 curvature not gain | an empirical correction to the plan | **Proposition 3**; the gain cancels identically |
| A5-1 the R+1 ratio | a measured correction | **Proposition 4**; and knife-edge at equal gains |
| A5-5 PCA smears overlapping parts | a measurement | **Proposition 6**; forced by orthonormality |
| A2-1 22 blocks, isotypic canonical | a measured correction | **Proposition 5**; Wedderburn + ℤ₂ |
| B2-2 entropy signature non-specific | measured on control tasks | true, and **Proposition 7** shows the statistic was gauge-dependent in the first place |
| A3 limit at K/d ≈ 1.5 | a calibration curve | a limit in `R/m` (**T6**); the axis was not resolved |

What survives as genuinely empirical, because no closed form predicts it: A2-2's instrument
calibration and the fact that a trained model's residual is 2–3× harder than matched noise;
A2-4's surgery result (whatever its correct interpretation — see the retraction in
`RESULTS.md`); A4's component-recovery being task-specific; B2-1's finding that additive
scores solve the conjunctive task; and every null that separated trained from untrained.

## Open, and worth a proof

1. **Why is a trained residual harder for joint block-diagonalisation than matched noise?**
   A2-5 measures 2–3× and A2-8 attributes about half to symmetry-breaking. A perturbation
   bound on the commutator spectrum in terms of the perturbation's symmetry content would
   settle it. This is the one place where a theorem would tell us something we do not know.
2. **A sharp condition for when JADE's coupling graph recovers the true partition.** The
   tolerance is currently chosen by oracle (Reviewer 2's finding 6); a Davis–Kahan style
   bound on the coupling matrix would give a blind rule.
3. **The identifiability of the shared/private split with `R` readers.** T4 gives the
   spectrum; the deflation that isolates `S₀` exactly is immediate from the eigenvector, but
   the noisy case wants a perturbation bound.
