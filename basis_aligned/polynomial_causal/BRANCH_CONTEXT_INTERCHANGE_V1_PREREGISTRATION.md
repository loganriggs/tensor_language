# Frozen branch amplitude interchange

The two original weight-derived parent1 branches have token-facing suppression but failed family input-gating checks. The exact weight-defined radial constant also fails to approximate their amplitudes. This test asks whether their remaining context variation benefits prediction beyond the token family and corpus domain.

For frozen branch $j$, let $a_j(x)=(u^\top x)(p_j^\top x)$ and physical writer $w_j$. At the cached final residual $h$, replace its amplitude with a donor from another document:

$$
h' = h + [a_j(x_{\pi(i)})-a_j(x_i)]w_j.
$$

The joint arm replaces both amplitudes using the same donor. Donors stay within the same endpoint token family, and within the same domain for the corpus-shift panel. Four permutations use seeded random order followed by cyclic shifts 1, 7, 13, and 19; every stratum has at least 32 documents. Seed 5701. There are no self-donors. This preserves the empirical amplitude distributions within each stratum, including joint dependence between branches in the joint arm. Donors are not matched by individual target token, lexical context, or difficulty. This is a deliberately limited conditional randomization, not proof of exchangeability or an identified causal variable.

Reuse the separate 128-document FineWeb panel and 128-document four-domain panel, with 384 endpoints each. These are previously inspected validation panels, so this is a registered-before-execution **post-result diagnostic**, not fresh confirmation. No parameters, means, or factors are learned from data.

Evaluate the exact final RMS normalization, full unembedding, tanh cap and target cross entropy. Retain the entire native background. Native and original removal scores must reproduce their cached values. Separately compute the first-order loss change

$$
\Delta\ell_{\mathrm{linear}} = \nabla_h\ell(h)^\top(h'-h).
$$

A positive exact swap cost alone can reflect nonlinear sensitivity to a noisy replacement. The linear term provides a separate alignment check; neither outcome identifies a semantic task by itself. Negative swap cost means this donor intervention improves prediction on average and must not be described as helpful context alignment.

Predictions, scored on each panel independently:

- **A:** finite/source/pairing checks, all permutations valid, cached native and removal CE replay max error at most $10^{-5}$, identity swap error at most $10^{-5}$, analytic versus autograd FP64 tail gradient relative error at most $10^{-9}$.
- **B:** mean exact replacement CE cost at least 0.005 nats for each single branch, and both paired-document bootstrap 95% lower bounds above zero, on both panels.
- **C:** mean first-order replacement CE cost at least 0.001 nats for each single branch, and both paired-document bootstrap 95% lower bounds above zero, on both panels.

Average the four donor realizations and three endpoint families within each document before resampling 128 document units (2000 bootstrap samples, seed 5702). These intervals condition on the fixed donor assignments and do not account fully for dependence through donors; report them as descriptive. Show family/domain breakdowns and joint-minus-singles effects without adding post hoc success bars. Failure of B or C is a narrow miss, not evidence that all input context or other factorizations are irrelevant.

Price: zero transformer body forwards; 768 cached native endpoints; per endpoint one native, three original removals, and twelve donor replacements = 12,288 full-vocabulary tail rows, plus small derivative/identity controls. Load only the unembedding from the bound checkpoint. Execute on managed lane1, no direct GPU launch. No replacement adopted and no four-property promotion from this diagnostic.
