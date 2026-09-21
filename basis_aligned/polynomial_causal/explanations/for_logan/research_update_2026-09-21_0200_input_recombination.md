# Reusing the folded program on recombined inputs

21 September 2026, 02:00 UTC.

The 512-product program passed a limited recombination screen: it reproduced source-only swap effects with 19.5–20.0% error. A follow-up control found meaningful context dependence, but the context-dependent part alone was less accurate, at 34.4–35.8% error. This is evidence of partial functional reuse, not identification of semantic circuits.

## What was tested

The exact folded bilinear function takes normalized midpoint input $n$ and previous-source input $m$. Earlier full-contribution swaps changed both inputs together. This test holds $n$ fixed and replaces only $m$ with another document's source vector at the same token position:

$$
\Delta F=F(n,m')-F(n,m)=F(n,m'-m).
$$

The source documents are cyclically shifted by 1, 7 or 16 among the 32 cached calibration documents. Inputs are reused, but their pairings are new. We directly evaluate the original trained weights for the reference; paired replay against the saved native outputs agrees to relative error $3.32\times10^{-8}$.

There is no refitting. The program means cancel in the differences. Both inputs retain their cached normalization scaling, so these are tests of the extracted bilinear interface, not recomputed model states or native final-logit interventions.

## Results

| Document shift | Original 1,024 products: effect error | Smaller 512 products: effect error |
|---|---:|---:|
| 1 | 18.71% | 19.50% |
| 7 | 18.99% | 19.77% |
| 16 | 19.19% | 20.01% |

The registered screen asked for errors below 30% and no more than 5% relative degradation from product removal. Both conditions pass for all three shifts. This does not establish generalization to new input marginals, external OOD text, selective semantic interventions or composition with other replacements.

## Does it capture an interaction, or just an average source effect?

A successor control separates the average-context response from the context-dependent interaction:

$$
F(n,\Delta m)
=F(\bar n,\Delta m)+F(n-\bar n,\Delta m),
$$

where $\bar n$ is the calibration mean midpoint input. This identity is exact by bilinearity; its terms need not be orthogonal under the observed joint distribution.

Using the exact native average-context response alone leaves 51.4–52.1% relative effect error. Thus recipient context makes a substantial contribution. The extracted program's context-dependent term has errors:

| Program | Error on context-dependent interaction alone |
|---|---:|
| Original 1,024 products | 32.92–34.20% |
| Smaller 512 products | 34.43–35.77% |

These are relative norm errors, not percentages of unexplained variance. No 30% threshold was preregistered for this successor diagnostic. It reveals a weakness hidden by the better aggregate effect error: the non-average interaction is less faithfully reconstructed.

The next useful target is this context-dependent operator, with average response scored separately. That would test whether improvements capture the interaction needed for reuse rather than mainly the easier average response. The existing graph remains an approximation of a specified interface; its intermediate nodes still lack stable semantic identities.

Evidence: `direct_tensor_match/MIDPOINT_RECOMBINATION_V1.json` records the initial screen; `MIDPOINT_RECOMBINATION_V2.json` adds the average-context control. `midpoint_recombination_screen.py` reproduces the latter. All runs used CPU and existing calibration inputs.
