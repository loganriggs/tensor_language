# Three-hour mathematical review — 2026-09-03 10:20 UTC

## Circuit target and current decision

The target remains a smaller executable tensor program, not a lower-rank description.  A useful unit must specify
what information is read, the operation performed, what is written, and which later computations use it.  It may
join pieces of different native modules or split one module.  It must predict held-out and shifted inputs, compose
with other replacements, support selective finite interventions, and eventually reduce literal storage, compute,
edges, states, or program length.

Rungs 525 and 526 close token grouping at their tested grain.  The exact token-by-context operators did not group
beyond ordinary token-vector similarity, and grouping them by full-suffix circuit gradients was document-specific.
The remaining MLP0 context-only branch is worth one exact test because its removal costs a Shapley-average
`0.3506` nat on FIT and `0.4177` nat on SELECT.  Rung 527 will split that branch into interactions between named
attention-source relations and test those pieces by finite downstream effects.  Rank, reconstruction error, and
quantization are not objectives.

## Exact Theseus object

For layer 0:

- residual width `D = 1152`;
- MLP product width `P = 4608`;
- attention has `H = 9` heads of width `128`;
- the five fixed source relations are `SELF`, `PREVIOUS`, `NEAR` (lags 2–7), `DISTANT_SAME` (lag at least 8 and
  the same token), and `DISTANT_OTHER` (all remaining causal source positions).

Let `L,R in R^(4608 x 1152)` and `D0 in R^(1152 x 4608)` be MLP0's Left, Right, and Down matrices.  Define the
vector-valued bilinear map

`B(x,y) = D0[(Lx) elementwise-multiplied by (Ry)] in R^1152`.

The deployed quadratic numerator is `Q(z)=B(z,z)`.  Its effective bilinear form is the symmetrization
`Bs(x,y)=B(x,y)+B(y,x)`; the antisymmetric part of `B` cannot be observed in `Q`.  Let `e` be the token-only input,
`a` the attention0 write, `mu=E_FIT[e+a]`, `delta_a=a-E_FIT[a]`, and `gbar` the frozen mean squared RMS-normalization
gain.  The already measured context-only branch is

`C(a) = gbar * ( B(delta_a,mu) + B(mu,delta_a) + B(delta_a,delta_a)
                 - E_FIT[B(delta_a,delta_a)] )`.

The attention write is split additively into five centered relation writes
`delta_a = sum_s delta_s + epsilon`, where `epsilon` is the small finite-precision remainder needed to match the
deployed attention write.  For the five semantic sources, define

`linear_s = gbar * (B(delta_s,mu) + B(mu,delta_s))`,

`self_s = gbar * (B(delta_s,delta_s) - E_FIT[B(delta_s,delta_s)])`, and, for `s<t`,

`cross_st = gbar * (B(delta_s,delta_t) + B(delta_t,delta_s)
                    - E_FIT[B(delta_s,delta_t)+B(delta_t,delta_s)])`.

There are 5 linear, 5 self, and 10 cross-source terms.  Their sum is the centered context function of the semantic
source sum.  Rung 527 will retain `epsilon` through an explicit numerical remainder and require that remainder to be
small.  This is the correction to rung 517: its named terms omitted the constant
`-E_FIT[B(delta_a,delta_a)]`, so the reported context “closing” energy was 47–52%.  Assigning each expectation to
its own self or cross term removes that accounting omission without changing the model.

The contraction graph for one term is two `1152 -> 4608` projections, one elementwise product over 4,608 product
coordinates, and one `4608 -> 1152` projection.  The map is degree two in the five source writes, although each
source write is itself a nonlinear degree-four attention function of the token sequence before RMS normalization.
The outputs to preserve are MLP0's deployed write, final logits/CE, and the registered effects on the 62 existing
circuits.  The algebraic decomposition has no approximation norm; downstream tests use exact finite removals.  It
adds no deployed values and is diagnostic until a smaller executable replacement passes the adoption gates.

## Symmetries, gauge, and what is identifiable

MLP0's hidden product coordinates permit a shared permutation, reciprocal Left/Right rescaling per product
coordinate, and interchange of Left with Right accompanied by symmetrization.  Individual product coordinates are
therefore not semantic units.  The output-valued maps `linear_s`, `self_s`, and `cross_st` are unchanged by these
hidden-coordinate gauges.  They do depend on the chosen five-way attention-source partition; that partition is an
operational definition, not a theorem-selected basis.

The polarization identity recovers the symmetric bilinear form of a quadratic function exactly:
`Bs(x,y)=Q(x+y)-Q(x)-Q(y)` (with the corresponding factor-of-two convention).  This proves that, after fixing an
additive source split, the self and unordered cross-source functions above are algebraically determined rather than
an arbitrary Tucker/SAE rotation.  See the [Encyclopedia of Mathematics statement of polarization](https://encyclopediaofmath.org/wiki/Polarization_identity)
and its reference to Landsberg's *Tensors: Geometry and Applications*.

Classical functional ANOVA gives unique orthogonal main and interaction effects under a product distribution.
Our five attention-source writes are strongly dependent because they are computed from the same sequence and their
masks partition one attention pattern.  Generalized functional ANOVA can obtain uniqueness under dependent inputs
using a joint-distribution weak-annihilation condition, but solving that projection is a different statistical
object from the exact polynomial split above; see the assumptions summarized in the
[JMLR treatment of generalized functional ANOVA](https://www.jmlr.org/papers/volume25/23-0699/23-0699.pdf).
Therefore polarization licenses exact accounting, while neither theorem makes the five relations semantic circuits.
Held-out causal behavior must do that work.

## Neighboring exact-realization results

Weighted-automaton theory says that a rational sequence function has a finite-state linear realization exactly when
its prefix-by-suffix Hankel matrix has finite rank, and a complete Hankel block yields a minimal realization by rank
factorization.  The statement and constructive formulas are given in
[Arrivault et al., *A Toolbox for the Spectral Learning of Weighted Automata*](https://proceedings.mlr.press/v57/arrivault16.pdf).
Linear second-order RNNs can likewise be recovered from Hankel tensors under multilinearity and complete-measurement
assumptions; see
[Rabusseau, Li, and Precup (AISTATS 2019)](https://proceedings.mlr.press/v89/rabusseau19a.html).

The mapping would make a model prefix a row, a continuation plus causal readout a column, and the observed final
effect their Hankel entry.  Equal rows would define downstream-equivalent states.  This is attractive for the later
predictive-state route, but it does not exactly solve rung 527: Theseus has RMS normalization, input-dependent
attention, nonlinear composition across 18 layers, and only a sparse set of circuit readouts rather than a complete
Hankel basis.  Classical bilinear-system realization also assumes a controlled bilinear state recurrence and access
to its input-output series; the more general realization setting is described by
[Arbib and Manes, *Generalized Hankel Matrices and System Realization*](https://epubs.siam.org/doi/10.1137/0511038).
Our fixed-depth transformer does not satisfy those recurrence assumptions.  The theorem is therefore a design for a
future finite predictive-state quotient, not evidence that a low-rank state exists here.

## Executable consequence and falsifiers

The exact consequence is a 20-term centered source-interaction evaluator with a planted polynomial test:

1. verify the 20 terms plus retained numerical remainder reconstruct `C(a)` before any causal scoring;
2. verify that distributing the FIT expectation reduces rung 517's 47–52% omitted constant to the frozen numerical
   tolerance without altering the native endpoint;
3. remove each term from the real MLP0 output and measure final CE plus the same member-minus-control effects for the
   registered circuits on disjoint document halves;
4. open the held-out circuit family only if at least one source term has a stable, selective discovery effect; and
5. treat exact algebra with unstable or broad downstream effects as anatomy, not a circuit.

This dominates another token-family or rank screen because it fixes a known exact-accounting hole in a branch with
`0.4177` nat SELECT importance and directly tests within-module splitting, held-out prediction, and selective
manipulation.  It is killed as a circuit route if the exact centered pieces have document-unstable circuit
fingerprints, if every material term affects circuits broadly, or if the numerical remainder remains too large.

Ranked alternatives are: (1) this exact finite context-source test; (2) a finite prefix-by-continuation
predictive-state table using real suffix interventions rather than gradients; (3) return to attention and identify
shared Q/K or output functions across heads by downstream interchange; and (4) move to another MLP if MLP0's final
branch also lacks stable selective pieces.  Only (1) repairs an already measured causal branch and produces an exact
finite intervention at low cost, so it remains the highest-information next step.
