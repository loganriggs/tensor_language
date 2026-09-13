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
