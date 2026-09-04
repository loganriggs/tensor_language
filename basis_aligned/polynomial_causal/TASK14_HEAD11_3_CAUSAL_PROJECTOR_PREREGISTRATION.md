# Task 14 head-11.3 causal projector — preregistration

Frozen at 2026-09-04 16:14 UTC, before fitting any Task 14 projector or evaluating any projected intervention.

## Question

Does a small rotated linear subspace inside the 128 pre-output-projection values of attention head 11.3 reproduce the head's
subject–verb-agreement interchange effect across constructions while leaving answer-preserving noun and distractor changes small?

The object is the causal action

$$
o'_b=o_b+UU^{\mathsf T}(o_d-o_b),\qquad
U\in\mathbb{R}^{128\times k},\qquad U^{\mathsf T}U=I_k.
$$

Only $P=UU^{\mathsf T}$ is interpreted. A basis rotation $U\mapsto UQ$ does not change the intervention. Rank $k$ is the cost of the
candidate intervention, not its meaning and not evidence by itself.

## Why this is not a rank-reduction experiment

The frame is trained and selected only by finite counterfactual behavior. It receives no reconstruction, variance, activation-energy,
or parameter-count objective. Success requires:

1. reproducing the complete head's causal effect on answer-changing relations;
2. transferring across sentence constructions and lexical groups not used for fitting;
3. keeping active answer-preserving controls small;
4. beating dimension-matched random subspaces and label-permuted fits;
5. stable causal responses across restarts; and
6. a held-out complement test that distinguishes a within-head split from a merely redundant sufficient direction.

Even a pass remains a grammatical-number **candidate**, not a complete semantic identification, because two independently designed
unrelated `is`/`are` tasks failed native capability before intervention.

## Immutable prior evidence

- Task 14 native capability: 249/256 correct across the eight registered cells.
- Full-state screen: attention 11 recovers 61.26%; head 11.3 recovers 60.36%; noun-identity and distractor controls are 3.02% and
  3.94%.
- Literal PP ↔ relative-clause validation interchange: attention 11 recovers 60.21%, head 11.3 recovers 58.93%, and all 64 rows move
  toward the donor.
- Fixed block-11 factorial: incoming state plus attention recovers 87.42%; fixed-component interactions are small.

These results select the site and justify the subspace question. They do not choose a projector, rank, optimizer seed, or subspace
success threshold.

## Physical two-program separation

The experiment is implemented as two small programs, not a general stage compiler.

### Program A: DISCOVERY fit and inner selection

Program A may read only the existing Task 14 `DISCOVERY` donor records and immutable prior receipts. Its output is a create-only fit
receipt plus a frozen projector bundle. It must not load or derive any `VALIDATION` token sequence, logit, activation, or row-level
effect.

### Program B: one outer validation opening

Program B may run only if Program A's receipt hash, projector-bundle hash, source hashes, success flag, selected rank, and seed rule all
match the frozen contract. It opens all Task 14 `VALIDATION` donor records once, scores every predeclared target/control cell, publishes
row-level denominators and effects, and cannot update or select a projector.

This physical split is the leakage guard. It replaces the abandoned multi-thousand-line compiler and is testable by making Program A
fail if any validation record is presented.

## Inner lexical split

Rows are split by indivisible lexical mirror units, never individual donor relations. The split was selected from metadata only by
maximizing the smallest exact semantic-cell count across halves, then the smaller relation count, then minimizing imbalance. No model
value was used.

Projector FIT group numbers are

$$
\{0,9,10,11,16,25,26,27\}.
$$

Projector SELECT group numbers are

$$
\{1,4,6,15,17,20,22,31\}.
$$

A donor relation is retained only if both endpoints belong to the same half. This gives 153 FIT relations and 145 SELECT relations;
246 DISCOVERY relations crossing the two halves are permanently unused. The canonical retained-ordinal hashes are:

- FIT: `5c24f97e98de6ff351514e19586d6ec4e72b5d1af6a3d5971d3d0b7d1b2267db`;
- SELECT: `4b7de6802d6f6fd23c669cde5276e57987ba552dd2b1ac67068c21fea9c0f823`.

The endpoint sets are disjoint: 64 unique FIT endpoints and 64 unique SELECT endpoints.

## Target and control relations

The donor manifest already labels each relation.

- A target has `expected_relation = opposite_subject_toward_donor`. Program A has 116 FIT and 106 SELECT target relations.
- A control has `expected_relation = same_subject_zero_projected_effect`. Program A has 37 FIT and 39 SELECT control relations.

No failed cell may be hidden by pooling. The exact cell key is

$$
(\text{arm},\text{sentence type},\text{matching rule},\text{recipient subject state}).
$$

## Full-head-relative causal objective

For each target relation, define $s_b$, $s_h$, and $s_U$ as the donor-oriented `is`/`are` logit score under native recipient, complete
head interchange, and projected interchange. The complete-head effect and projected effect are

$$
E_h=s_h-s_b,\qquad E_U=s_U-s_b.
$$

The primary fit target is $E_U/E_h=1$, not the native donor answer itself. This distinction prevents the optimizer from receiving
credit for an arbitrary steering direction that overshoots or differs from the head's actual effect. All complete-head denominators
are frozen before training and saved row by row.

For a same-answer control, $s$ is the native correct-answer-minus-foil margin and the desired projected effect is zero. FIT loss is the
equal-cell average of a robust squared error on target $E_U/E_h-1$ plus the equal-cell average squared normalized control movement.
The target/control coefficient, robust-loss transition, and all normalizers must be literal constants in Program A and included in its
dry-run receipt.

## Downstream-informed analytic initializer

Before finite fitting, Program A constructs one discovery-only analytic frame. For target $i$, let

$$
d_i=o_{d,i}-o_{b,i},\qquad
g_i=\nabla_{o_b}s_i.
$$

With equal weighting across exact semantic cells, define

$$
S=\operatorname{mean}_{\text{cells}}\operatorname{mean}_{i\in\text{cell}}
\frac{d_i g_i^{\mathsf T}+g_i d_i^{\mathsf T}}{2E_{h,i}}.
$$

The largest algebraic eigenvectors initialize the candidate frame. This is an interaction-determined basis: it depends jointly on the
head-state changes and how the fixed downstream model reads them. It is only a local proposal. Finite SELECT interventions determine
whether it is useful.

## Fitting schedule and bounded rank ladder

Primary candidate ranks are $k\in\{1,2,4\}$. Each gets three deterministic starts and 100 finite-interchange updates with batch size
32. Rank 8 is allowed once only if at least one healthy rank-4 fit passes all target cells but fails solely a stability or control bar;
otherwise ranks above 4 are not opened.

Do not use raw Adam through differentiable QR, symmetric-polar differentiation, or the failed direct Grassmann optimizer from earlier
rungs. Use a planted-tested orthogonal parametrization with the analytic frame plus deterministic small tangent perturbations. Model
weights remain frozen.

The expected primary Program-A price is 1,047 forwards, 900 backwards, 33,504 example evaluations, and 10,752 stored frame bytes.
Rank 8, if legally opened, adds at most 339 forwards, 300 backwards, 10,848 evaluations, and 12,288 frame bytes.

## Fit-health gates

Every fit must satisfy all of the following before its SELECT result can count:

- all losses, activations, logits, and frame gradients are finite;
- every model parameter has no gradient and the checkpoint-weight hash is unchanged;
- the head hook fires exactly once and changes only head 11.3 at the declared final-token position;
- rank 0 exactly reproduces native logits and rank 128 exactly reproduces complete-head logits;
- $\max|U^{\mathsf T}U-I|\le10^{-5}$;
- normalized projector movement from initialization is at least 0.02;
- the final 20-update mean FIT objective is at least 0.05 below the first 20-update mean; and
- the deterministic update schedule is complete.

A failed fit-health gate is an invalid fit, not a scientific null. The experiment requires at least two healthy fits at a rank to score
that rank.

## Inner SELECT success bars

For every answer-changing cell, require all three:

$$
\frac{\operatorname{mean}E_U}{\operatorname{mean}E_h}\ge0.80,
$$

at least 75% of rows moving toward the donor, and absolute native-donor recovery at least 0.40. The coordinated-plural to ordinary-
singular cells use the preregistered 0.35 absolute-recovery floor.

For every answer-preserving P, C, and coordinated-plural cell, require all three:

- normalized absolute answer-margin movement at most 0.10;
- movement no more than the row-matched complete-head control effect plus 0.025; and
- normalized full-vocabulary logit RMS at most 0.10.

Sixteen deterministic dimension-matched Haar-random projectors are scored at every opened rank. A candidate must exceed the 95th
percentile random worst-cell full-head-relative recovery by at least 0.10 while its controls pass.

The smallest rank for which at least two of three healthy fits pass every SELECT cell is provisionally selected. Two confirmation
starts are then run only at that rank; at least four of the resulting five must pass. Two fits using permuted target labels must both
fail the full selection rule.

## Stability and interpretation

For projectors $P_i,P_j$ of rank $k$, report chance-corrected overlap

$$
o^*_{ij}=\frac{\operatorname{tr}(P_iP_j)/k-k/128}{1-k/128}.
$$

Also compare row-level causal effects between seeds after dividing by the complete-head effect. Median pairwise absolute difference
must be at most 0.10 and the 90th percentile at most 0.20. Stable effects with unstable geometry license only an operationally
equivalent set of projectors, not a unique internal subspace.

## Outer validation

After Program A is frozen and passes, Program B opens donor ordinals 544–1087 exactly once: 416 answer-changing targets and 128
answer-preserving controls. Its canonical ordinal-list hash is
`32ecc3a9ab78a031df9207bfc0f788d4d7040ed4fa2f5ddeb61321dfd23e7191`.

The same per-cell SELECT bars apply unchanged. Program B additionally evaluates:

- rank-0 and rank-128 replay;
- the selected projector $P$;
- its complement $I-P$;
- their exact joint full-head replacement; and
- native midpoint neutralization along $P$ as a necessity diagnostic.

If $P$ passes but the complement remains strong, the conclusion is “sufficient but redundant/non-exclusive.” Only if complement
recovery is at most 0.25 of the complete-head effect in every target cell and the two-factor Shapley share of $P$ is at least 0.70 may
we call it a within-head split.

## Registered outcomes

1. **Selective sufficient projector:** Program A passes and the frozen projector passes every outer target/control/random bar.
2. **Within-head split:** outcome 1 plus the complement and Shapley conditions.
3. **Operational response class only:** causal responses are stable but projector geometry is not.
4. **Strong small-linear-subspace null:** no legal rank at or below 4, or conditionally opened rank 8, has four of five healthy fits
   that pass every target/control/random cell. Do not extend the rank sweep.
5. **Instrument invalid:** source, replay, gradient, optimizer-health, schedule, or write-integrity gate fails. Repair the instrument
   without interpreting model behavior.

A positive outcome still requires later out-of-distribution data, exact output-projection/downstream-weight contraction, and causal
reader interventions before becoming a complete circuit.
