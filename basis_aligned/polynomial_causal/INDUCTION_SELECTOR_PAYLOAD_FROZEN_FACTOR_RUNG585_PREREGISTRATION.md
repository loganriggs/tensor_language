# Rung 585 preregistration: frozen attention selector × payload factors

**Frozen:** 2026-09-03 UTC, after the outcome-blind R585 red team and before any R585 model output exists

## Dependency and question

R585 may execute only after the clean R586 native-capability replication and its independently frozen R587 audit both
return held. A scientific or audit null is terminal for this run. R585 does not inspect FINAL_TEST or OOD.

The experiment asks whether four previously localized equality-related attention terms carry an operational
factorization into:

- a **selector**, the strength assigned to the earlier token that matches the final query; and
- a **payload**, the residual-stream vector copied from the position immediately after that matching token.

The fixed sites are

$$
H=\{\mathrm{L5H5},\mathrm{L7H3},\mathrm{L8H3},\mathrm{L8H4}\}.
$$

They are treated as four locations of one distributed computation, not as four independently meaningful heads. This
run uses the complete set in every condition and performs no subset, rank, dimension, or threshold search.

## Exact term and frozen factors

For site $h$, endpoint $x$, its final query position $q_x$, and semantic source role
$r\in\{A,C\}$, let $k_x(r)$ be the payload position immediately after source $r$. Cache from an unmodified forward
pass

$$
e_h^x(r)=p_h^x(q_x,k_x(r))E^x(q_x,k_x(r)),
\qquad
u_h^x(r)=W_h^O v_h^x(k_x(r)).
$$

Here $p$ is the model's continuous attention score, $E$ is the already registered exact token-equality support, and
$u$ is the value vector after that head's output projection. Thus $e_h^x(r)u_h^x(r)$ is one exact contribution to the
residual stream at the final query. This experiment identifies the operational product $e u$; it does not claim that
a unique query or key feature has been found.

For every directed recipient $r_0$ and donor $d$, cache all recipient and donor $e$ and $u$ factors before applying any
intervention. At each fixed site, subtract the live recipient equality term and insert exactly one of

$$
\begin{aligned}
\text{replay:}&\quad e^{r_0}u^{r_0},\\
\text{score-only:}&\quad e^d u^{r_0},\\
\text{payload-only:}&\quad e^{r_0}u^d,\\
\text{joint:}&\quad e^d u^d.
\end{aligned}
$$

At later layers the subtracted term is computed from the current live state, while the inserted hybrid always uses the
frozen pre-intervention factors. This prevents an earlier intervention from changing the supposedly fixed half of a
later factor. The four site contributions are summed as $\sum_h e_hu_h$; scores and values are never summed separately
and multiplied, because that would create cross-site products absent from the model.

All source, payload, and final-query coordinates come from R578 semantic metadata. Donor and recipient prompts may have
different lengths. Any right padding occurs strictly after the saved final query; no padded key can contribute.

## Frozen rows and directions

R585 uses the R578 FIT and SELECT groups bound through R586. Each stored row is evaluated once in each of its two
declared physical directions. It does not synthesize a second reversed row.

The scored families are:

1. selector swap;
2. target-payload swap;
3. joint selector-plus-payload diagonal with unchanged answer;
4. selected-match break, scored separately coherent-to-broken and broken-to-coherent;
5. neutral-source edit;
6. neutral-payload edit;
7. filler replacement; and
8. lag extension.

The contrast-target-source edit is retained as a reported diagnostic but is not an invariance control because it
changes the context immediately before the competing payload. FIT must pass completely before SELECT is opened. The
same arms, mappings, normalizers, thresholds, and complete four-site set are then evaluated once on SELECT.

## Required algebraic and replay checks

Before a scientific decision is allowed:

- unmodified manual replay must match the native final logits for every endpoint and length class within absolute
  tolerance $10^{-5}$ in float32;
- at every row and site, the saved removed and inserted terms must reproduce the actual hook delta within absolute
  tolerance $10^{-5}$;
- padded and unpadded final logits must match within absolute tolerance $10^{-5}$ for every observed length class;
- selector and lag payload-only arms must reproduce replay, and their joint arms must reproduce score-only, within
  absolute tolerance $10^{-5}$;
- broken-recipient payload-only must reproduce replay within the same tolerance; and
- all four sites, both directions, all rows, and all executed batches must have finite saved evidence.

A failure here is `invalid_instrument`, not a scientific factor null.

## Target metrics

For answer-changing rows, orient the logit margin toward the donor answer:

$$
m(z)=\operatorname{logit}_{a_d}(z)-\operatorname{logit}_{a_r}(z).
$$

For a complete cell of semantic groups $g$, define the whole-cell recovery

$$
R=\frac{\mathbb E_g[m(I)-m(r)]}{\mathbb E_g[m(d)-m(r)]}.
$$

Ratios of individual rows are forbidden. A cell is invalid rather than omitted when the natural denominator is
nonpositive or its group-bootstrap 95% lower confidence bound is not above zero. Save the numerator, denominator, and
ratio. To distinguish answer transfer from generic damage to the recipient, also save donor-answer cross-entropy and
require

$$
C=\mathbb E_g[\operatorname{CE}_r(a_d)-\operatorname{CE}_I(a_d)]
$$

to have a 95% lower confidence bound above zero for an intended transfer. Every confidence interval resamples complete
semantic groups using 2,000 SHA-defined bootstrap replicates and the frozen NumPy quantile convention.

Cells are split by dataset split, family variant, factorial condition, and physical direction wherever those fields
change the causal prediction. No direction may be hidden by pooling.

## Opposing predictions and FIT/SELECT gates

The following conditions are conjunctive on FIT and are repeated unchanged on SELECT:

### Selector swap

- score-only and joint each have mean and median recovery at least $0.30$;
- each has a positive 95% lower mean effect and positive intervention effect in at least 75% of groups;
- donor-answer $C$ has a positive 95% lower bound for each;
- payload-only has absolute recovery at most $0.25$; and
- joint and score-only differ by at most the numerical identity tolerance.

### Payload swap

- payload-only and joint each have mean and median recovery at least $0.30$;
- each has a positive 95% lower mean effect and positive intervention effect in at least 75% of groups;
- donor-answer $C$ has a positive 95% lower bound for each;
- score-only has absolute recovery at most $0.25$; and
- joint is not worse than payload-only by more than $0.10$ recovery.

### Answer-preserving joint diagonal

Let $c$ be correct-answer minus other-payload margin and use subscripts $r,s,p,sp$ for replay, score-only,
payload-only, and joint. Require:

- $c_r$ and $c_{sp}$ are positive in at least 75% of groups;
- both $c_r-c_s$ and $c_r-c_p$ have positive bootstrap lower means;
- the factorial interaction

$$
\frac{c_{sp}-c_s-c_p+c_r}{4}
$$

  has a positive bootstrap lower mean;
- joint correct-answer cross-entropy increases by at most $0.10$ nat; and
- joint full-vocabulary logit root-mean-square change is at most $0.25$ of the matched FIT target scale.

No prompt-swap recovery ratio is used here because recipient and donor have the same answer.

### Selected-match break

Score coherent-to-broken and broken-to-coherent separately. For score-only and joint, mean and median recovery must be
at least $0.30$, the 95% lower mean effect must be positive, and at least 70% of group effects must be positive.
Payload-only is an opposing control and has absolute recovery at most $0.25$; broken-recipient payload-only must also
meet the exact replay check. Joint and score-only may differ on broken-to-coherent rows because the restored value can
contain source context, but both must pass independently.

If target transfer holds but the opposing arms do not separate, the result is `factorization_not_identified`. If the
complete four-site set fails a target ceiling, the result is `factor_capacity_null`. No smaller site set may rescue it.

## Active selectivity controls

A control only tests selectivity if the intervention actually changed the tested term. For each arm, direction, and
factorial condition, freeze the matched FIT target scale

$$
T=\operatorname{median}_{g\in\text{matched FIT target}}
\|\Delta_{\text{insert},g}\|_2.
$$

For a control family, group $g$ is active when its median insertion norm across the fixed sites is at least $0.10T$.
The control family has adequate activity when at least 75% of its groups are active. The FIT value of $T$ is reused on
SELECT. Structural no-ops—selector payload-only, lag payload-only, and broken-recipient payload-only—are checked for
exactness and never counted as selectivity successes.

For every non-structural arm × direction × factorial condition, at least two of neutral-source, neutral-payload,
filler, and lag controls must have adequate activity. Otherwise the terminal result is
`insufficient_active_controls`. Every active control cell must satisfy:

- median absolute correct-minus-other margin change at most $0.25$ of the matched FIT target scale;
- median full-vocabulary final-logit root-mean-square change at most $0.25$ of that scale;
- correct-answer cross-entropy increase at most $0.10$ nat; and
- correct answer remains above the other target payload in at least 75% of rows.

Target transfer with any active-control failure is `broad_contextual_equality_write`, not selector/payload
identification.

## Evidence and independent audit contract

The result must save, for every row, direction, arm, and site:

- R578 row, group, split, family, variant, endpoint, and sequence identifiers;
- token IDs, answer IDs, final query position, semantic source and payload positions, and equality support;
- native, replay, score-only, payload-only, and joint logits for both target tokens, both target cross-entropies,
  log-normalizer, and full-vocabulary logit change from replay;
- frozen recipient and donor score scalars and projected value vectors, the live removed term, inserted term,
  per-site insertion norm, residual-stream delta norm, and every exactness error;
- all cell memberships, ordered group IDs, unnormalized numerators and denominators, bootstrap draw/statistic hashes,
  confidence intervals, activity decisions, gates, and failed clauses; and
- hashes of all inputs, code, tests, preregistration, checkpoint, result, and receipt; opened splits; exact forward and
  backward counts; and confirmation of zero weight updates.

The future CPU auditor must rebuild exact directed membership from R578, independently recompute every whole-cell
metric and all 2,000-replicate group bootstraps, verify algebraic identities and activity coverage, and bind the exact
result bytes. A complete scientific null is written normally. Malformed or incomplete evidence fails the audit.

## Execution price

The gated FIT/SELECT decision excludes the diagnostic contrast edit. At batch size 32 the conservative ceiling is:

- FIT: 1,872 rows, 3,744 directed pairs, 1,728 unique endpoints, at most 459 model forwards;
- conditional SELECT: 936 rows, 1,872 directed pairs, 864 unique endpoints, at most 231 model forwards;
- total: at most 690 model forwards, zero backwards, zero fitted vectors, and zero weight updates.

The runner may use a cheaper bound only if it freezes and tests reuse of native captures before execution. It may not
claim savings retrospectively. Any price excess aborts without a scientific result.

## Licensed conclusion

Only if every target, opposing-arm, algebraic, active-control, split, price, and evidence condition holds may R585 say:
the fixed four equality-gated terms causally implement an operational selector × payload factorization on R578
FIT/SELECT prompts. This does not yet identify unique Q/K features, establish OOD generalization, compile the factors
into a reusable weight-level circuit, prove any one site necessary, or prove selective removal. Those require
separately frozen follow-up experiments.
