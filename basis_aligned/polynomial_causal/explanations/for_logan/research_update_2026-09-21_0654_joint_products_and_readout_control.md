# Jointly learned products save storage, but the weakest component still fails

21 September 2026, 06:54 UTC. This follows the [shared-core and reuse tests](research_update_2026-09-21_0638_shared_cores_and_reuse.md).

**We fitted shared products directly to the six folded quadratic source reads.** The best 512-product program uses about33% fewer coefficients than three separate pair programs, but misses the registered fidelity limits. The third component's native scalar error is17.54%, versus11.94% for the baseline. An independent dense audit reproduces the result. Solving output coefficients by least squares does not repair it.

This is progress in testing the proposed arithmetic representation, not a newly identified circuit. The next experiment permits a product of two different learned reads, targeting384 products at approximately the baseline's storage.

## What changed from the previous experiment?

Previously, we projected independently discovered computations through one shared linear input bank. That lost too much fidelity. Here we learn the product directions and their output weights jointly:

$$
\widehat Q_o=\sum_{r=1}^{512}w_{or}p_rp_r^\top,
\qquad
\widehat q_o(z)=\sum_{r=1}^{512}w_{or}(p_r^\top z)^2.
$$

The six outputs are two quadratic reads for each of three native components. Every product can serve all six outputs. Centered affine terms and their trace corrections remain exact; the displayed equation describes the quadratic part only.

This is a symmetric CP representation of the joint order-three tensor. It supplies an arithmetic DAG with shared product nodes. It still does not search arbitrary graph topologies or identify the semantics of those nodes.

The coefficient objective uses the expanded input weighting and normalizes each pair's coefficient energy so that one pair cannot dominate solely by scale. Candidate selection uses only this weight-space objective. Native scalar errors are diagnostics on seven previously opened, distinct cached prefixes, not fresh confirmation.

## Four fits at fixed complexity

Each fit used float64 Adam,4,000 cosine-decayed steps, and512 products. The two starts were spectral and a1% perturbation of that spectral start. The best coefficient-loss iterate was retained, which matters when a higher learning rate temporarily worsens the objective.

| Learning rate / start | Best coefficient error | Component1 error | Component2 error | Component3 error |
|---|---:|---:|---:|---:|
| 0.01 spectral | 10.30% | 3.55% | 2.49% | 17.54% |
| 0.01 spectral 1pct perturbed | 10.60% | 3.77% | 2.61% | 17.99% |
| 0.05 spectral | 13.69% | 5.20% | 3.10% | 21.98% |
| 0.05 spectral 1pct perturbed | 11.34% | 3.98% | 2.92% | 19.82% |

The selected program has604,428 stored floating-point coefficients. The separate reference has900,108; deduplicating its identical writer gives the fairer897,804. All three programs' source products total768, versus512 in the joint fit. These counts exclude native input production, normalization and the final component products. They do not establish wall-clock speedup.

The selected fit's per-component errors are3.55%,2.49%,17.54%, versus3.06%,2.76%,11.94% for the separate reference. The registered gate required every component to be at most15% and at most1.10 times its baseline. Components1 and3 fail the relative limit; component3 also fails the absolute limit. No native intervention promotion followed.

## Is the failure an export bug or poor output-coefficient fitting?

An independent CPU audit rebuilt dense quadratic matrices in original coordinates. It reproduced coefficient scores within $3.1\times10^{-13}$ and native scalar errors within $1.2\times10^{-15}$. Exact centered affine terms and mean corrections were preserved. Recomputing the RMS denominator from saved native states differs by only $1.8\times10^{-8}$ relative norm.

We then froze the512 product directions and solved their output weights by least squares. For unit-normalized metric-space directions, the Gram matrix is

$$
K_{rs}=(p_r^\top p_s)^2.
$$

The right-hand side for output $o$ is $p_r^\top Q_op_r$. No Gram directions were discarded at the registered relative eigenvalue cutoff $10^{-12}$; normal-equation residuals were below $3.4\times10^{-14}$.

For the already-selected primary, coefficient error changes from10.3047% to10.2961%, while component3 error changes from17.5432% to17.5491%. Every refitted control still fails fidelity. Thus inaccurate output coefficients are not the main problem for these frozen directions. This does not prove the learned directions are globally optimal.

## Capacity and optimization are still distinct

Any512-direction shared-square model has input-mode unfolding rank at most512. The target's singular-value tail gives a coefficient-error lower bound of6.39%. The fitted10.30% is above that bound, so we cannot attribute the whole remaining error to capacity. This bound also says nothing direct about native-function error.

Five planted shared-square structures passed independent loss/gradient comparisons. Random-start Adam with1,200 constant-rate steps recovered three of five;4,000 cosine-decayed steps recovered all five at the $10^{-4}$ threshold. Both budget and schedule changed, so that control does not isolate which change repaired the misses. The trained fits likewise show strong learning-rate and initialization sensitivity.

## Next structural assumption: products of distinct reads

A square is restrictive. A more general product contributes

$$
\widehat Q_o=\frac12\sum_r w_{or}
\left(l_rr_r^\top+r_rl_r^\top\right).
$$

One product can now represent an indefinite rank-two quadratic form. In particular,

$$
a^2-b^2=(a+b)(a-b).
$$

Using384 such products needs768 learned input directions: about the separate baseline's input storage, but half its source products. The price gate allows at most1% more stored floats than the writer-deduplicated baseline, and requires at most384 products. Fidelity limits stay unchanged.

Initialization pairs positive and negative spectral terms before fitting. This also avoids initializing every left/right pair identically, which would keep them tied under symmetric gradient updates. Five planted mixed-product structures have passed dense loss/gradient checks and random-start recovery with4,000 cosine-decayed Adam steps. A full-size export-shape check precedes managed execution.

This experiment changes the kind of computation each node can express. It is not another attempt to force the same fixed dictionaries into a common projection.

## Reproduction and scope

- [Four fit results](../../direct_tensor_match/SHARED_PRODUCT_NATIVE_FIT_V1.json), [independent export audit](../../direct_tensor_match/SHARED_PRODUCT_NATIVE_AUDIT_V1.json), and [fixed-product readout refit](../../direct_tensor_match/SHARED_PRODUCT_READOUT_REFIT_V1.json).
- [Capacity and fair-price audit](../../direct_tensor_match/SHARED_PRODUCT_RANK_AND_PRICE_AUDIT_V1.json).
- [Original shared-square toys](../../direct_tensor_match/SHARED_QUADRATIC_PRODUCTS_TOYS_V1.json), [longer decayed-rate toys](../../direct_tensor_match/SHARED_QUADRATIC_PRODUCTS_TOYS_COSINE_4000_V1.json), and [mixed-product toys](../../direct_tensor_match/MIXED_QUADRATIC_PRODUCTS_TOYS_V1.json).
- [Next mixed-product preregistration](../../direct_tensor_match/MIXED_PRODUCT_NATIVE_PLAN_V1.json).

The completed four-fit managed job took211.6seconds and made no native-model forwards. The audits and readout refits were CPU-only. Cached input rows do not establish independent source-document identities; these remain opened diagnostics. Modes2/3 are not semantically identified, and native upstream inputs remain outside the extracted programs. The full circuit goal is unfinished.
