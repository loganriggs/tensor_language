# Head8.2 inputs to the Q7/MLP8/head9 interaction

13 September, continuation after the requested 00:29 report. This targets the specific H-input contribution confirmed in [the parent-path note](MLP7_PHI_READERS_FOLD_V1_MATH.md), not all uses of head8.2. Existing first-layer-value regional producer work is prior evidence, not a substitute for testing this path.

## Native cache and failed sector simplification

Write the head's four readings as

$$
H=\Gamma(C+F),
$$

where $\Gamma\in\mathbb R^{T\times T}$ is the **joint product of both** native normalized, rotated QK scores, and $C,F\in\mathbb R^{T\times4}$ are its current-stream and shared first-layer value readings, with actual learned mixing and folded OV coefficients included. The native inputs of C and F are distinct contextual states.

The 96-prefix cache reproduces head8.2 readings to $2.32\times10^{-7}$ relative error and its conditional head9 donor write to $1.09\times10^{-7}$; original endpoint margins replay exactly. The registered first-layer generated-sector simplification fails: errors are 18.81%,22.27%,**26.82%**,12.97% for original/fronted/near-quote/distant, versus a 20% near-quote bar. This is not a failed instrument or absent first-layer contribution. Current values cannot simply be dropped at the declared tolerance. [Receipt](ATTENTION8_PHI_VALUE_ROUTING_V1_RESULT.json).

## Red-team: generated-sector donation differs from source-value donation

Using recipient quantities without a prime and donor changes denoted by delta, the exact expansion is

$$
\Delta H=
\Delta\Gamma(C+F)+\Gamma\Delta C+\Gamma\Delta F
+\Delta\Gamma\Delta C+\Delta\Gamma\Delta F.
$$

The first term changes routing alone; the next two change individual value inputs with recipient routing; the last two are routing/value interactions. A generated first-sector donation is instead

$$
\Delta(\Gamma F)=\Delta\Gamma F+\Gamma\Delta F+\Delta\Gamma\Delta F.
$$

We apply the same recipient Q7/norm and head9 readout to each term, retaining only the previously specified post-city source positions. The five write fields sum to the prior full field within $1.09\times10^{-7}$. Their near-quote aligned fractions are routing −0.19%, current +23.16%, first +86.45%, routing/current +1.54%, routing/first −10.96%. These are inner-product allocations against the full **all-position write field**, not causal fractions or probabilities.

Routing-only has 105.08% field error, first-value-only 30.88%, both-values-only 27.87%. Dropping both mixed terms has 58.07% error despite their modest net aligned fraction: alignment hides components orthogonal to the full field. This motivates checking the full native suffix before interpreting the size of each behavioral effect. [Five-term CPU receipt](ATTENTION8_PHI_ROUTING_VALUE_DELTA_V1_RESULT.json).

## Physical test registered

Nine arms over the same96 prefixes: native, full head8.2 H-input donation, generated current sector, generated first sector, routing-only, current-value-only, first-value-only, both values, and both mixed terms together. The specified conditional scalar write is injected at the same head9 edge; all subsequent native computation is rerun. Native prefix/routing/contextual interfaces remain external.

A: native/full replay <=1e-4. B: near-quote first-value-only has at least20/24 opposing effects and <=40% relative effect error to the full head8.2 path. C: separately measured generated current/first effects sum to full within10% relative error in every construction. Other routing/value compositions are reported descriptively. The earlier20% generated-first field criterion remains failed; this is a distinct physical input-port test, not a revised pass for that criterion.

Price:864 full forwards,180-second runtime cap, cached conditional inputs, no data fitting. Primary script: `../bilinear_quotient/ops/run_attention8_phi_value_ports_v1.py`. No new OOD or broad preservation claim.

## Physical value-port results

The test completed with A/B/C held: full/native replay $2.37\times10^{-7}$,864 forwards in8.57seconds. In the near-quote construction, full head8.2 transfer is−5.787%; generated current is−1.491% and generated first is−4.312%, each24/24 opposing. Their separately measured effects sum within0.097–0.361% relative error across constructions.

With recipient joint routing fixed, **first-value-only** transfer is−4.880%,24/24 opposing,31.06% relative effect error to the full path. Current-value-only is−1.379%,also24/24 opposing; routing-only is+0.079%,12/24 positive. Both values together give−6.241%. The two mixed routing/value terms together give+0.407% on average but mixed directions. Omitting mixed terms causes59.30% relative effect error across individual near-quote endpoints despite that small mean; the full three-group sum(routing,values,mixed) predicts the full effect within0.14–0.82% across constructions. This preserves the distinction between mean signed effect and the norm of a vector of per-prefix effects. [Physical receipt](ATTENTION8_PHI_VALUE_PORTS_V1_RESULT.json).

First-value-only is not a uniformly opposing path: it is mixed on old anchors,19/24 opposing when fronted,24/24 opposing near-quote, and23/24 positive when distant. Its context dependence still comes through native routing and Q7. No broad sign-generalization claim follows.

## Closing the first-value token generator

Unlike the full contextual phi8 value(which failed the earlier token-only hypothesis), the shared first-layer **source value** precedes attention0. Its fixedweight generator is

$$
F(t)=A_{0,8.2}\operatorname{RMS}\!\left[(\alpha_0+\beta_0)\operatorname{RMS}(E_t)\right].
$$

The implementation retains actual FP32 operations rather than simplifying the normalization numerically. Here E is the native embedding matrix; both learned block0 coefficients are6.09375. On all63 distinct tokens in the96-prefix panel, the folded generator reproduces native first-value readings to $1.40\times10^{-7}$ relative error. Donor/recipient first-value differences are **exactly zero away from the changed city token**. No token table is fitted. [Generator check](ATTENTION8_PHI_FIRST_TOKEN_GENERATOR_V1_RESULT.json).

The resulting [conditional executor](extracted_circuits/first_token_value_path_v1/README.md) computes the token differences from weights, then routes them through supplied recipient context. It reproduces the physically tested first-value-only fields within $2.91$–$4.67\times10^{-8}$ across constructions. It stores4,615 coefficients, but explicitly references the57,950,208-weight embedding matrix and native QK/Q7/norm/writer/suffix interfaces. Closing this input port is real extraction progress; calling the whole result a4.6k-parameter extracted model would be false.

The next high-information step is an untouched-context test of this fixed executable path, with native capability and selective controls, followed by decomposition of the remaining contextual routing/Q7 ports. Do not turn the first-generated-sector miss into an excuse to discard the structure, or the successful token generator into a claim that all contextual computation has been eliminated.

## 00:55 — Effective context reader and registered fresh test

The remaining contextual dependence can be gathered into an explicit four-vector for each recipient position t and changed city position c:

$$
K_{t,c}=2\alpha_9\sum_{j>c}
\frac{\gamma_9(t,j)\gamma_8(j,c)}{\rho_{9,j}s_{8,j}}\Lambda Q_j,
\qquad
\Delta w_t=K_{t,c}^{T}\Delta F_c.
$$

This is an exact contraction, not a fitted predictor. CPU replay of all-position first-value-only fields is $2.31\times10^{-16}$. On the existing panel, the directed final-position write is positive24/24 near-quote but negative24/24 distant, although the city-token value difference is context-independent. The physical regional effect is opposing24/24 near-quote and positive23/24 distant. Thus contextual sign changes already appear in this composed reader before the final suffix. Final-write/effect correlations range−0.80 to−0.91, but that descriptive relation does not establish final-position-only sufficiency. [Context contraction receipt](FIRST_TOKEN_EFFECTIVE_CONTEXT_V1_RESULT.json).

A fixed, score-free confirmation has been registered:24oldnear-quote anchors plus72untouched prefixes(two near-quote constructions andone distant). It tests native capability, predicted near-opposing/distant-positive signs, first-path fidelity to full head8.2, and half-strength scaling. The executor, rows and runner are frozen before scoring. This is context generalization, not corpus OOD; the cities and lexical endpoints are reused. At00:55 the job is queued behind the live shared v675 experiment; no result is claimed yet. See `FIRST_TOKEN_PATH_FRESH_V1_ROWS.json` and the prediction docstring of `../bilinear_quotient/ops/run_first_token_path_fresh_v1.py`.

## 01:03 — Fresh confirmation and a causal explanation of its remaining miss

The fresh test is completed. Native capability holds on36/36 new paired contrasts. Both new near-quote constructions have first-value-only opposing effects on24/24 prefixes each, with directed transfer−3.19% and−2.40%. The distant construction gives+0.864% on average but only18/24 positive directions, missing the registered20/24 bar. **A and C pass; B fails.** First-path effect errors relative to full head8.2 are21.89%,30.64%,38.00%; doubling half-strength effects predicts full first-path effects within0.45–0.59% relative error. These support approximation and local edit scaling, not the failed universal sign rule. [Fresh receipt](FIRST_TOKEN_PATH_FRESH_V1_RESULT.json).

All six distant exceptions are Baltimore recipients, across all six endpoints. The full head8.2 path agrees with the first-value path's sign on all24 distant prefixes. Thus the sign miss is not a first-value approximation artifact. [Executed sign audit](FIRST_TOKEN_PATH_FRESH_V1_SIGN_AUDIT.json).

### Which context input changes the sign?

For paired recipients, the **directed** token contrast is the same fixed British-minus-American four-vector. Their different writes therefore arise from the effective context reader K. We evaluated all16 combinations of recipient/donor settings for gamma8, gamma9, Q7 and the combined inverse RMS factor. Averaging marginal changes over the other settings gives an exact four-input Shapley allocation; the sum error is $2.09\times10^{-16}$.

For the distant Sheffield/Baltimore pairs, Q7 contributes130–138% of the aligned all-position context-write difference; routing contributions oppose it, and normalization contributes about1.3–1.4%. Fractions above100% reflect cancellation, not probabilities. This is an algebraic allocation on hybrid context inputs, so it motivated a physical discriminator rather than establishing causality by itself. [Context-input allocation](FIRST_TOKEN_CONTEXT_PORT_ALLOCATION_V1_RESULT.json).

### Physical input corners confirm Q7 modulation

Restrict the tested generated value to the Q7/first-value cross term, retaining recipient routing and normalizers. Its dependence on the two input ports is bilinear:

$$
G(Q,F)\propto Q^T\Lambda\Gamma_8F.
$$

We physically injected changes corresponding to F alone, Q alone, Q and F together, and the isolated mixed term. In particular,

$$
G(Q+\Delta Q,F+\Delta F)-G(Q+\Delta Q,F)
-G(Q,F+\Delta F)+G(Q,F)
\propto\Delta Q^T\Lambda\Gamma_8\Delta F.
$$

All fields are transported through the retained head9 path, and the native suffix is recomputed. The Q-only change acts on the specified all-source first-value cross term; the F difference is supported only at the changed city token. Neither is a whole-MLP7 intervention.

The six previously negative Baltimore first-value effects become positive6/6 after donating the paired Sheffield Q7 readings: the conditional marginal effects range+0.0305 to+0.0699 logit-margin units, versus−0.0096 to−0.0230 before that Q donation. The isolated mixed-term intervention predicts the physically measured difference-of-differences within1.29–2.09% across all four constructions. Native/F replay is $2.64\times10^{-7}$;480 forwards took4.91seconds. All registered A/B/C criteria pass. [Physical Q/token interaction](FIRST_TOKEN_Q_INTERACTION_V1_RESULT.json).

**Counter-review:** the six Sheffield recipients flip the other way when receiving Baltimore Q readings, while Nottingham/Detroit's twelve distant recipients remain positive. The manipulation transfers the asymmetry; it does not improve the count of positive distant effects or retroactively pass the fresh sign criterion. This is evidence for a contextual modulator, not a repaired universally positive regional circuit. [All-recipient counter-review](FIRST_TOKEN_Q_INTERACTION_V1_COUNTER_REVIEW.json).

This also clarifies why two earlier statements can coexist. Donating the entire Q-containing MLP8 path showed that much changing information entered through B/H partners. Within the now isolated first-token-value interaction, holding the token contrast fixed and changing the recipient context exposes an important Q7 modulation. These are different interventions on different arguments, not contradictory assignments of a single global causal percentage to MLP7.

## 01:16 — Donor-free removal and unrelated-behavior controls

Donation alone does not define an independently removable component. We therefore tested the absolute crossfirst value

$$
G_j=\frac{2Q_j^T\Lambda H^{\mathrm{first}}_{8.2,j}}{s_{8,j}},
\qquad
w_t=\alpha_9\sum_j\frac{\gamma_9(t,j)G_j}{\rho_{9,j}}.
$$

The **all-source** version sums every attention8 source contributing to H and every head9 source j; it needs no city annotation or donor prompt. The city-source version restricts attention8 to the city source c and head9 sources to j>c. Both retain contextual input generation and native background. Subtracting w at the specified head9 writer removes that generated interaction contribution; it does not zero a whole module.

The native city-source and all-source fields have roughly0.45–0.63times the norm of the earlier donor field across the four constructions. Replacing the native city value with its donor reconstructs the earlier donor field to $3.81\times10^{-8}$. This is a direct algebraic check that the absolute component and donation use the same defined path. [Definition/control](FIRST_TOKEN_ABSOLUTE_REMOVAL_V1_CONTROL.json).

Physical removals pass the registered regional A/B/C criteria:

| Construction | City-source removal: change in native regional contrast | All-source removal: change in native regional contrast |
|---|---:|---:|
| Old near-quote anchor | +5.51% | +4.42% |
| New near-message | +3.40% | +2.70% |
| New near-person | +2.24% | +1.52% |
| New distant-note | −0.85% | −1.33% |

Positive entries here mean **stronger** regional contrast after removal, unlike the damage-sign convention in the JSON. Every near pair increases contrast. Distant city-source removal decreases11/12 pairs and all-source removal decreases12/12. Doubling half-strength all-source effects predicts full removal within0.20–0.32% relative error. This confirms a small context-dependent inhibitory/supportive interaction, not a universally supportive regional unit. [Removal receipt](FIRST_TOKEN_ABSOLUTE_REMOVAL_V1_RESULT.json).

### FineWeb preservation: small observed damage, with a preserved control-design miss

The annotation-free all-source component was removed on32 existing ground-truth-next-newline FineWeb prefixes(two16-row families). All32 native newline-versus-comma margins arepositive. Native andwholehead8.2-zero reference outputs replayexactly; weight-derived token input readings replaywithin $2.49\times10^{-7}$.

| FineWeb family | Mean absolute CE change | Maximum absolute CE change | Whole-head8.2-zero mean absolute CE change |
|---|---:|---:|---:|
| 0 | 0.000187 nats | 0.000946 nats | 0.038229 nats |
| 1 | 0.000350 nats | 0.002932 nats | 0.037896 nats |

A(instrument replay) andB(candidate preservation)pass. **C fails:** I registered a requirement thatwholehead removal produce>0.1nats meanabsolute change, although the already published fixture strength is onlyabout0.038. That was an avoidable preregistration mistake, not a newly weak model capability. The registered result remainsA/Btrue,Cfalse. [FineWeb receipt](FIRST_TOKEN_REMOVAL_NEWLINE_V1_RESULT.json).

The executed counter-review reads the prior fixture directly: wholeheadzero effects are108–204times thecandidate's meanabsolute effects; prior meanreplacement effects are116–219times. This indicates comparative sensitivity, but doesnot retrospectivelypass the.1criterion. We will not rerun the same experiment or silently substitute another threshold to manufacture a pass. [Control-strength audit](FIRST_TOKEN_REMOVAL_NEWLINE_V1_CONTROL_AUDIT.json).

The hourly review marked the control-fixture novelty/lesson gate failed and repaired it with the small `newline_control_fixture_v1.py` helper: proposedminimumcontrolstrength is checked against thepublishedfixture before registration. Running it with.1 correctly rejects that design. The next control panel has been built without model-score selection:32 new-to-this-path FineWeb prefixes plus32oldanchors, excluding priorcontrol andmean-calibration cacheindices and known fullprefix duplicates. These share a cache; documentindependence and corpusOOD are notclaimed. The new panel is not yet scored. [Rows and selection scope](FIRST_TOKEN_REMOVAL_NEWLINE_FRESH_V1_ROWS.json).

## 01:23 — Fresh FineWeb preservation passes; explicit-state implementation begins

The frozen all-source removal passed allthree prospectively registered checks on the32newFineWebprefixes, with32oldanchors replaying exactly. New family meanabsoluteCEchanges are0.000278 and0.000214nats; maxima0.001415 and0.000686. Native newline margins arepositive31/32. Wholehead8.2 removal meanabsoluteCEchanges are0.03274 and0.05091, meeting the newprospective minimum.005 and10-times-candidate sensitivity bars. The earlier incorrectlyspecified.1controlcriterion remains failed; this is a distinct new-prefix test. [Fresh control receipt](FIRST_TOKEN_REMOVAL_NEWLINE_FRESH_V1_RESULT.json).

These are prefixes new to thispath's validation from disjoint cacheindices, not established document-independent or corpusOOD data. The result supports preservation of this known unrelated newline service; it doesnot establish broad language preservation.

The next implementation removes opaque computed Q/routing inputs from the executor interface. `crossfirst_state_executor_v1` now computes Q7, fulljointhead8.2 routing, fixedtokenfirstvalues, andselectedjointhead9.8 routing from declared frozenweights andthree normalized stateports. It still needsRMS8/RMS9 andthe nativeprefix/suffix. Import/loading andliteralpricing pass:69,970,952declared tensor scalars, including68,567,040externalembedding/MLP7inputweights. No smallstandalone-model orspeedupclaim follows.

**Native replay of this assembled implementation is pending.** The preregistered nextvalidation compares itsfields againstnativehook calculations anditsphysicalremoval against theexisting96regional/64FineWebreceipts(320forwards). It must pass before replacing thevalidated executor. [Implementation and status](extracted_circuits/crossfirst_state_executor_v1/README.md), [registered replay](CROSSFIRST_STATE_EXECUTOR_V1_PREREGISTRATION.md).

## 01:29 — Explicit-state replay passes and parentage is made explicit

The assembled executor passes all registered native replay criteria on96regional and64FineWeb prefixes. Native anchors replayexactly. Aggregate field errors against native-hook calculations are $7.93\times10^{-7}$ and $7.79\times10^{-7}$; physical removal outcome errors are $3.94\times10^{-7}$ and $1.39\times10^{-7}$. Individual Q7/token/gamma8 errors remain below $1.82\times10^{-6}$; the largest individual field error is $5.96\times10^{-6}$.320forwards took6.04seconds. [Native executor receipt](CROSSFIRST_STATE_EXECUTOR_V1_RESULT.json).

This is now a validated executable computation from three normalized stateports,two normports,tokens anddeclaredweights. It doesnot require cachedQ orattentionpatterns. Its nativeprefix andsuffix dependencies remain; the literal declared69.97Mtensor count isnot a smaller autonomous model.

The child path is contained in the already studied head9 current-value component. The downstream reader matches **exactly**. Writing

$$
M_8=U\Lambda U^T+M_{\mathrm{rest}},
$$

and expanding the four-mode term exposes the crossfirst child while retaining all other terms andthe fullmatrix remainder. A direct FP64control against the native MLP8 weights(including bias) reconstructs that scalar read within $3.49\times10^{-15}$. [Parentage control](CROSSFIRST_PARENTAGE_V1_RESULT.json).

At the shared head9 writer interface, define

$$
P=C+R,
$$

where P is the full selected current-value parent, C is the derived crossfirst child, and R retains everything else. This is an algebraic hierarchy, not a claim that R is an identified semantic circuit or that the two inputs are orthogonal. Removing both P and C independently would subtract C twice. Joint removal must use C+R once each.

The [hierarchy record](extracted_circuits/crossfirst_state_executor_v1/HIERARCHY.json) makes this containment explicit. Native parent/child/remainder physical composition remains the next test: exact equality of injected writes doesnot guarantee that separately measured logit effects add through the nonlinear suffix. That test should report composition error relative to the smaller child's effect as well as the parent, avoiding another easy pass from a large denominator.
