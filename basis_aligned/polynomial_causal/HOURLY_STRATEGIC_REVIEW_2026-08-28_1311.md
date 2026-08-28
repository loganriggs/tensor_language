# Hourly strategic review: make the physical-gate assay executable

Date: 2026-08-28 13:11 UTC

Status: evidence audit, ranked plan, preregistration repair, and tested CPU graph
interface. No model outcome, row selection, authority receipt, or GPU result was opened.

## What changed since the 13:00 mathematical review

Before this turn, nothing had changed after commit `2b44c38b`: no response collector,
authority, response tensor, selector outcome, or competing result existed. Both project
queues were empty, no model Python process was running, and the RTX 5090 was idle at
1 MiB and 0% utilization. The latest MLP1 evidence remained the negative split-frame
assay. There was no external blocker; the blockers were missing and inconsistent
experimental interfaces in our own prospective code.

Three independent audits agreed that the trajectory-complete physical-gate assay is
the next nonredundant experiment. They also identified two defects which made a GPU
launch a NO-GO:

1. `trajectory_complete_response` emitted `[probe, context, gate]`, while every
   selector interpreted it as `[context, probe, gate]`. Since both layouts are
   rank-three tensors, existing validation silently accepted the swap.
2. “Cross-half projection capture” rebuilt the selected support's span on evaluation
   data. With 16 contexts times 32 probes = 512 rows and $K=512$, a generic support
   can span the entire evaluation matrix and score one without transferring anything
   learned on the fit half.

The preregistered finite edit also moved selected gates to 0.9. That measures selected
gate sensitivity, not whether omitted gates can be removed. These are mathematical
interface failures, not small implementation details, and were repaired before any
model outcome was opened.

## Denominator-separated state of reverse engineering

There is still no single honest “percent reverse engineered.” The currencies remain:

| currency | current evidence | interpretation |
|---|---:|---|
| structural tensor formulas | 36 / 36 modules | exact forward structure, not semantics or minimality |
| standalone owned program | 545,904,054 / 545,904,054 values owned | zero native calls, not compression |
| admitted executable compression | 29,196,288 values removed = 5.348245% | rank640 attention simplification |
| admitted program size | 516,707,766 values | includes all exact dense MLPs |
| dense MLP banks | 286,675,200 values = 52.513843% of dense | 55.481109% of rank640; zero admitted gate removal |
| named semantic coverage | 32.1% +/- 6.4% | separate semantic-program currency |
| strict named causal coverage | 0.57968 / 5.30682 nat = 10.923% | leaves 4.72714 nat = 89.077% unnamed |

Rank640 remains predictively strong in CE (+0.005532/+0.004449 nat on its two roles)
and preserves 99.437/99.673% of live task accuracy, but its 95.782/96.077% top-token
agreement fails the frozen 98% identity bar. Its intervention-bank recovery 0.94442
(95% lower bound 0.92726) says the compressed attention shell preserves most measured
causal behavior; it does not name or explain that behavior.

## Largest gaps that determine the next move

1. **MLP executable interface:** more than half the model remains in dense MLPs, and
   no physical product has been admitted for removal.
2. **Unexplained causal/semantic residue:** 89.077% of the strict named causal ledger
   remains outside named mechanisms.
3. **Interaction and composition:** early-MLP joint effects are strongly non-additive;
   at MLP1, SVD32 joint removal cost 0.3832 nat versus 0.0345 from the sum of
   singletons. A support selected by first derivatives can therefore fail jointly.
4. **OOD/behavioral identity:** fresh FineWeb rows test replication, not OOD. Rank640's
   approximately 4% argmax disagreement and coverage-dependent semantic-program tail
   show why CE alone is insufficient.
5. **Intrinsic versus checkpoint-relative simplicity:** native gates are invariant to
   scale/permutation but not general polynomial refactorization. Gate sparsity would be
   useful and executable without proving that the retained gates are intrinsic atoms.

## Candidate pruning

- Another local PCA, response-frame rank sweep, or per-context basis search repeats the
  failed MLP1 frame hypothesis and is pruned.
- Generic sparse regression over native products already exists in compiler v2.1 and
  underperformed the affine family; repeating it with a new name is pruned.
- Token-prefix Hankel, context-free tables, and information-bottleneck scores lack the
  RMSNorm/residual/attention composition interface and are pruned now.
- An all-18-layer gate sweep, cross-layer shared dictionary, or exact CP search is too
  expensive and underidentified before one physical support survives finite tests.
- MDL or parameter count alone cannot validate prediction, extraction, removal, or
  editability. They remain downstream comparison currencies.
- Immediate full gate deletion or refitted Down is pruned until a small movement along
  the actual candidate path agrees with its tangent prediction.

## Ranked top five

### 1. Finish and run the 2-by-2 trajectory-complete MLP1 response assay

Freeze fit versus untouched documents crossed with independent probe halves. Measure
the exact shared-gate derivative through every token position. Fit support and
coefficients on the fit half only; evaluation may accept or reject that frozen bundle
but may not alter it. Compare ridge, response-energy, activation-times-Down, immediate
reader, factor-derangement, and matched-random controls in raw and context-balanced
currencies.

This has the highest information gain because one small assay determines whether a
stable executable physical support exists at all. It is causal, plugs directly into
the complete rank640 program, has crisp negative outcomes, and is approximately one
split-probe-scale GPU job. The remaining pre-launch work is a serialized plan with
exact rows, seeds, target-rank rule, inference margins, bootstrap unit, multiplicity,
source closure, locks, and a create-only no-outcome authority.

### 2. Compute the native quadratic-form Gram spectrum

For native input products, compute

$$
G_{nm}=\tfrac12[(\ell_n^\top\ell_m)(r_n^\top r_m)
+(\ell_n^\top r_m)(r_n^\top\ell_m)].
$$

A well-conditioned full-rank result would certify that exact deletion inside the
native 4,608-form dictionary is impossible, while leaving alternate polynomial
factorizations open. This is zero-GPU, gauge-checkable, cheap relative to model
evaluation, and prevents an approximate causal sparsifier from being mislabeled as an
exact algebraic simplification.

### 3. Calibrate the actual sparse-candidate path at epsilon 0.1

If action 1 promotes a fit-frozen support $S$ and coefficients $\beta$, use

$$
\alpha(\epsilon)=\mathbf1+epsilon(\widetilde\beta-\mathbf1),
\quad \widetilde\beta_n=\begin{cases}\beta_n&n\in S\\0&n\notin S.\end{cases}
$$

At $\epsilon=0.1$, omitted gates move to 0.9; at one, the sparse candidate is reached.
Require signed Fisher prediction, the $\tfrac12\epsilon^2$ KL normalization, CE,
top-one, rare/coverage strata, and worst-document bounds. This is the cheapest
falsifier of finite pruning and is much cheaper than deletion.

### 4. Identify finite package interactions before composing sites

For eight surviving gate packages, fit a quadratic/Volterra action law on balanced
signed masks at small scales and test it on unseen masks. It must beat the additive
law, exhibit the predicted small-scale remainder, and pass documentwise bounds. This
directly targets the observed joint-versus-singleton failure and supplies a predictive
composition rule.

### 5. Build a one-site executable frontier and test background transport

Only after actions 1/3/4 pass, compare native hard retention and selected-products plus
refitted Down at complete storage/multiplication prices. Replay the unchanged support
inside rank640 attention, native attention, and a compiled-upstream MLP0 background,
then test code or another genuine OOD corpus. Background reversal would show useful
shell-specific compensation rather than a modular MLP1 circuit; that distinction is
still actionable.

## Highest-priority action executed

The graph-level shared-gate interface now exists in
`tensor_bilin18_global_gate_intervention.py`:

- each context owns one independent `alpha[gate]` leaf;
- that vector is broadcast over every token position at MLP1 before Down;
- one backward pass per categorical-Fisher probe returns separate
  `[context, probe, gate]` responses;
- the complete 18-layer owned tensor program, RMSNorms, residual streams, attention,
  later MLPs, unembedding, and softcap remain inside the graph;
- the one-use transaction revokes model/graph aliases and returns no logits, targets,
  or residual-write VJPs.

The analysis contract was repaired at the same time:

- response layout is now unambiguously `[context, probe, gate]`;
- a deliberately unequal 3-context/5-probe integration test prevents recurrence;
- fit-frozen CSS interpolation is evaluated unchanged on the opposite half;
- in-half support span remains labeled non-promotive;
- sparse all-on coefficients have a frozen executable representation;
- the candidate-path scaling rule includes omitted gates;
- a known-answer categorical test verifies that small observed KL matches
  $\tfrac12\epsilon^2$ times the Fisher quadratic.

Focused CPU result: **14/14 tests pass in 4.39 s**. The preregistration was amended
prospectively and records remaining plan constants as launch blockers. Therefore the
GPU remains intentionally idle: the next safe action is source-closing the exact 2-by-2
plan and authority, not opening an outcome from underspecified bytes.
