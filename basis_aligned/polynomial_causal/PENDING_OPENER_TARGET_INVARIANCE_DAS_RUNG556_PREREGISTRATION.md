# Rung 556 preregistration: pending-opener target-plus-invariance DAS at L13H8

**Frozen:** 2026-09-03 16:46 UTC, before any R556 model call or optimization

## Why this differs from R540

R540 optimized only answer-changing examples. It evaluated answer-preserving edits afterward, and those controls
failed. R556 puts the controls into the training objective itself and uses the fresh, duplicate-free three-value R545
dataset at the confirmed 128-dimensional L13H8 output site.

This is a causal representation test, not a compression test. The tested dimension is the capacity of an interchange
operation. A smaller dimension has no scientific value unless it transfers the intended pending-opener change and is
inert on all registered answer-preserving changes.

## Intervention

For target head output $h$, donor output $h'$, and an orthonormal
$Q\in\mathbb{R}^{128\times k}$, patch only the final-token L13H8 input to the layer-13 output projection:

$$
h_{\mathrm{patched}}=h+(h'-h)QQ^\top.
$$

All other heads and model computations remain native.

For an answer-changing row, let $E_{\mathrm{full}}>0$ be the saved donorward closer-logit change produced by the
complete L13H8 swap in R546, and $E_Q$ the projected-swap change. The target recovery is

$$
r_Q=E_Q/E_{\mathrm{full}}.
$$

For an answer-preserving row, let $L_Q$ and $L_0$ be the patched and native final vocabulary-logit vectors, and let
$s_{\mathrm{full}}$ be the saved full-vocabulary root-mean-square logit change from the complete swap. The normalized
control change is

$$
c_Q^2=\frac{\operatorname{mean}_v(L_Q(v)-L_0(v))^2}
{\max(s_{\mathrm{full}},0.01)^2}.
$$

The $0.01$ floor is frozen and only prevents division by numerical noise; R546 already established every control cell
as live above this scale.

## FIT optimization

Use ranks $k\in\{1,2,4,8,16\}$ and seeds $\{0,1,2\}$. Each of the 15 fits runs 240 Adam updates at learning rate
$0.005$. Every update samples eight FIT directions uniformly across the two answer-changing families and eight FIT
directions uniformly across the three answer-preserving families. QR orthonormalization is applied on every update.
The fixed loss is

$$
\mathcal{L}=\operatorname{mean}_{\mathrm{targets}}(r_Q-1)^2
+\operatorname{mean}_{\mathrm{controls}}c_Q^2.
$$

There is no penalty-weight sweep. No SELECT, FINAL_TEST, or OOD row enters optimization.

## SELECT bars

For every answer-changing family and swap direction separately:

- median $r_Q\geq0.50$;
- 95% group-bootstrap lower mean $r_Q>0$;
- at least 75% of rows have $r_Q>0$.

For every answer-preserving family and direction separately:

- mean absolute closer-margin change $\leq0.10$ logit;
- that mean is $\leq25\%$ of the row-matched complete-head mean absolute closer-margin change;
- mean full-vocabulary logit RMS is $\leq25\%$ of the row-matched complete-head mean RMS.

A rank is stable only when at least two of three seeds pass every target and control cell. Five dimension-matched
random subspaces per rank must have mean target recovery below 0.10. Select the smallest stable rank that also passes
the random control; within it, report every passing seed and use the lowest seed only for later frozen testing.

## Budget, inputs, and outcome meanings

Inputs are hash-bound R545 rows/receipt, R546 result, R548 audit, and this preregistration. The run evaluates FIT and
SELECT only; FINAL_TEST and OOD remain unopened. It saves raw row-level target recoveries and both control measurements
so an independent audit can recompute all summaries.

The maximum planned optimization is $15\times240=3{,}600$ gradient-bearing layer-13-through-17 suffix evaluations,
plus native state collection and no more than 700 no-gradient suffix evaluations for SELECT scoring and random
controls. Model weights are frozen.

- Held: a projector learned with explicit invariance constraints transfers both constructions and all three delimiter
  values while leaving three live nuisance changes alone.
- Null: no tested projector satisfies all cells. Record that as evidence against a single linear pending-opener
  subspace at L13H8; do not respond with a larger capacity sweep.
- Invalid: any hash, checkpoint, row alignment, forward/backward budget, or split-opening check fails.
