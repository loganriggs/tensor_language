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
