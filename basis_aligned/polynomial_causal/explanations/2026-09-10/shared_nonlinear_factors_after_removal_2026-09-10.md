# Shared nonlinear computation after the removal test

September 10, 2026. Updated 01:09 UTC.

The current MLP1 task directions do not meet the requested circuit standard. A new
static weight-removal experiment is mechanically valid but fails selective removal.
The mathematical route worth retaining is **joint factorization of the actual
computations used by multiple consumers**, with normalization and downstream behavior
included. Reducing the input state and finding shared arithmetic are different problems.

The original proposal is the
[handoff](../bilinear_circuit_reconstruction_codex_handoff.md); the
[pilot report](../bilinear_reconstruction_pilot_report.md) records its initial evaluation.
The pilot explicitly proposed joint read–route–write operations after its local span
and duplicate-factor searches failed. This note develops a bounded mathematical tool
for that direction. It does not restart the is/was command-mode search.

## What the latest experiment establishes

Write the MLP as y=b+W phi(n), where phi(n)=(L n)⊙(R n). Here n is the normalized
input, phi is the vector of 4,608 scalar products, and W writes them into the residual
stream. C_A and C_B read four coordinates each. With C=[C_A;C_B], the dual writer
D=Cᵀ(CCᵀ)⁻¹ satisfies CD=I: it converts desired reader changes into output writes.

The fixed component for task t is D_t C_t W phi(n). Removing it changes the native
weight to W−D_t(C_t W), retains b, and recomputes the entire downstream model.
There is no donor-dependent offset. All weights and biases were restored afterward.

| Removal | Own cue contrast retained | Other task endpoint prediction changes | P control endpoint prediction changes |
|---|---:|---:|---:|
| Temporal component | 73.87% | 2 | 3 |
| Is/was component | 40.57% | 2 | 2 |
| Both components | 71.02% temporal; 40.21% is/was | Not applicable | 7 |

“Contrast” means the change in donor-answer-minus-foil logit margin between each
paired context. Retention is the projection of the edited contrast vector onto the
native vector, divided by native squared length; it is not accuracy. The registered
suppression limit was 10%, with a separate RMS retention limit of 25%. Both fail.
Control mean teacher KL is .04448/.01056/.09186 nats for the three arms, above .001.
KL measures change in the entire predicted distribution.

All instrument checks pass, including unchanged inputs, exact weight-edit identities,
deployed MLP replay, and native baseline replay. Runtime was 4.084 seconds for 16
forwards/512 sequence evaluations. Texts were already opened. No OOD claim follows.
See the [immutable result](../../BILIN18_MLP1_WEIGHT_DEFINED_REMOVAL_V1_RESULT.json).

The result rejects this component definition as independently removable. It does not
show that the task has no circuit: redundant routes, retained background and nonlinear
downstream dependence remain possible. No rank, site, dose or offset rescue is licensed.

## A shared nonlinear library can survive full input rank

For symmetric quadratic reader forms Q_k, the proposed shared-square grammar is

    z=A x;   h_j=z_j²;   output_k=Σ_j Λ[k,j] h_j.
    Equivalently: Q_k=Aᵀ diag(Λ[k,:]) A.

All consumers reuse the same computed h_j, with different linear combinations. A is
an invertible change of input coordinates here; this grammar does not discard inputs.

A two-dimensional exact example is

    A = [[1,1],[0,1]]
    Q0 = [[1,1],[1,2]]; Q1 = [[2,2],[2,5]].

Compute h0=(x0+x1)² and h1=x1² once. The outputs are h0+h1 and 2h0+3h1.
Both Q matrices have full rank. Their commutator Q0 Q1−Q1 Q0 is invertible, with
determinant 1. Thus even the stronger commutator obstruction to a smaller
linear-input-plus-norm state does **not** forbid a shared nonlinear program.

This is an exact counterexample to an overbroad inference, not a correction to the
properly scoped native certificate. It is also a planted example that a competent
algebraic compiler already factors; it supplies no trained-model interpretability win.

## A test that actually targets this grammar

If Q0 is nonsingular and the square bank exists, define B_k=Q0⁻¹ Q_k. Then

    B_k=A⁻¹ diag(Λ[0,:])⁻¹ diag(Λ[k,:]) A.

Therefore every B_i commutes with every B_j. One nonzero commutator rules out this
invertible d-feature square bank. Zero alone is inconclusive without simultaneous
diagonalizability over the reals and the appropriate congruence conditions. Adding
Q2=[[0,1],[1,0]] to the example produces a nonzero relative commutator and rejects
the two-square grammar. It does not reject a larger feature bank or mixed products.

With a known positive-definite reference, whitening reduces the question to orthogonal
joint diagonalization of symmetric matrices, where pairwise commutation is sufficient.
A generalized-eigenvalue approach using two random linear combinations is developed
by [He and Kressner](https://arxiv.org/abs/2402.16557). Their exact-recovery result
assumes the family is simultaneously diagonalizable by congruence. It does not say
arbitrary trained forms have that property or that the recovered features are semantic.

Dense numerical analysis of m forms costs roughly O(m d³) arithmetic and O(m d²)
storage; explicit native forms here have d=1152 and m=8. Exact symbolic elimination
has additional coefficient-growth costs. The implemented exact routine is for small
fixtures, not a proposed 1152-dimensional symbolic job. A native positive-definite
reference has not been established. No native factorization result is claimed.

Compared with the August 30 review of whitened consumer Grams and common invariant
blocks, this test concerns the actual quadratic functions and a nonorthogonal shared
square representation. It is a restricted candidate grammar, not a replacement theorem
for arbitrary bilinear circuits.

## Normalization and attention constrain what may be folded

Changing coordinates does not let us replace ||x||² with ||z||². Its correct expression is

    ||x||² = zᵀ(A⁻ᵀ A⁻¹)z.

In the example the norm metric is [[1,−1],[−1,2]], which contains a cross product.
One may instead keep x and compute its original norm. Either implementation must be
charged; a two-square numerator is not the whole normalized executable circuit.

For actual product attention, let x and y denote the already residual-normalized query
and source inputs, and R_t/R_s the rotary matrices. For score factor i define

    F_i(t,s)=W_Qiᵀ R_tᵀ R_s W_Ki
    a_i(x)=sqrt(xᵀ W_Qiᵀ W_Qi x/h + ε)
    b_i(y)=sqrt(yᵀ W_Kiᵀ W_Ki y/h + ε).

The causal routing coefficient is exactly, over real arithmetic,

    p(t,s) = [(xᵀ F_1(t,s)y)(xᵀ F_2(t,s)y)]
             / [h² a_1(x)b_1(y)a_2(x)b_2(y)],  s≤t.

Here h=128, and ε is the actual head-normalization epsilon. Output readers can fold
into O and the value maps, but this does not eliminate the routing or norm producers.
The mixed value retains both contextual and first-layer sources with their real coefficients.

A raw-dot factor gauge q→S q, k→S⁻ᵀ k preserves qᵀk. It generally changes normalized
attention. The exact fixture q=(1,1), k=(1,0), S=diag(2,1/2) changes the squared
normalized dot from 1/2 to 16/17, even at zero epsilon and zero rotary offset.
Nonzero epsilon and positional rotations require their own retained operators. Thus
matching folded numerator tensors alone cannot identify a shared native router.

## What this changes for circuit discovery

The useful object is a shared bank of explicit scalar operations plus its consumers,
not necessarily a smaller residual subspace. Sharing a computed feature can reduce
arithmetic while consumer-specific branches remain separately editable. Conversely,
deleting a genuinely shared producer should affect every dependent consumer; perfect
task selectivity is not a sensible requirement for every shared internal operation.

The next candidate should specify those producer and branch interventions separately,
retain normalization, and predict their downstream joint effects on held-out inputs.
Any joint factor bank must beat independently factored consumers and the existing
native DAG after charging A, Λ, normalization, and retained native modules. A dense
change of coordinates alone is not a structural explanation.

The concrete progress here is the executed mathematical discriminator, not another
native scan: [eight exact controls](../../SHARED_QUADRATIC_FACTOR_MATH_V1_CONTROLS.json)
pass in [the small reference](../../shared_quadratic_factor_math_v1.py), with zero model
forwards. It separates smaller state, reusable arithmetic, normalization fidelity and
causal reuse before another expensive candidate is proposed. None of the four final
circuit properties is newly passed by this fixture.

## Native follow-up: ordinary function equality loses operand identity

The next test has now run on the trained model. For captured normalized base and
donor inputs b and d, define the two independently edited MLP outputs

    y_L = W[(L d)⊙(R b)] + bias
    y_R = W[(L b)⊙(R d)] + bias.

These are actual native Left- or Right-projection swaps, followed by the complete
live suffix. A symmetric extension produces y_S=(y_L+y_R)/2 for either assignment.
The missing oriented component is (y_L−y_R)/2. In a scalar output reader it is
dᵀ skew(Lᵀ diag(CW) R)b. It is invisible when both inputs are the same.

The [registered native screen](../../BILIN18_MLP1_OPERAND_DOMAIN_V1_RESULT.json)
completes in 2.083 seconds with 16 forwards/512 sequence evaluations. All instrument
checks pass. Swapping both operands and directly injecting the donor MLP output
give identical final logits. The largest local native/FP64 oracle error is .000578,
within the combined registered absolute and relative limits. Hooks are restored.

The symmetric extension fails both registered behavioral prediction criteria:

| Panel | Full centered-logit causal error versus Left edit | Versus Right edit | Mean teacher KL, Left / Right |
|---|---:|---:|---:|
| Temporal | 33.39% | 23.42% | .004795 / .004197 |
| Is/was | 29.94% | 35.34% | .001174 / .001028 |
| P controls | 32.32% | 31.34% | .002234 / .001873 |

The limits were 1% causal-vector error and .001 mean KL, with additional p99 and
top1 requirements. These KL values compare the symmetric candidate with the actual
edited reference, not with native unedited behavior. One temporal Left-reference
prediction changes its top token. All original contexts are retained.

### Is the error merely a bad symmetric approximation?

A saved-data calculation tests a stronger, precise question. Suppose a predictor
returns the same margin change p_i for Left and Right edits of pair i. Let their
actual changes be l_i and r_i. Then

    Σ[(p_i−l_i)²+(p_i−r_i)²]
      = 2Σ[p_i−(l_i+r_i)/2]² + (1/2)Σ(l_i−r_i)².

Even allowing a different optimal prediction for every row cannot remove the last
term. Relative to combined squared native margin effects, the resulting RMS floors
are **13.97% temporal, 14.54% is/was, and 20.85% P**. The actual symmetric model's
corresponding errors are 14.07%, 14.62%, and 25.72%. Thus 98.6%/98.9% of its target
margin squared error is unavoidable if it forgets which operand was changed.
These margin floors are different measurements from the full-logit errors above.

The [zero-forward audit](../../OPERAND_IDENTITY_INFORMATION_V1_AUDIT.json) verifies
the squared-error identity on the saved floating-point values, with zero numerical
closure discrepancy. This is a finite-cohort bound for a predictor that identifies
the two edit labels. It is not a population theorem or a bound on decoders that
retain operand identity.

### Consequence for the decomposition

Keep the intervention interface explicit. A symmetric factorization remains valid
for ordinary tied-input execution and tied-input changes. A program claiming to
predict independent native operand edits must also preserve or translate their
orientation; ordinary function equality alone is insufficient. Retaining the
oriented term gives exact local reconstruction but does not simplify its weights.

These native operand edits are a diagnostic domain chosen for this screen, **not a
new requirement that every discovered circuit support every native-neuron edit**.
A circuit may instead specify semantically meaningful producer/consumer edits and
demonstrate their correspondence. This result prevents an unjustified extension of
a rewrite's causal claims; it does not close weight-based circuit decomposition.
