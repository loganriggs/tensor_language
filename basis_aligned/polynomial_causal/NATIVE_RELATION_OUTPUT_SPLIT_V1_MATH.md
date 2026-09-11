# Separating shared output effects from inflection choices

The frozen suffix component can be split using weights so that one branch retains much of its number-related effect while scarcely changing neighboring past/progressive grammatical choices. It still changes the probability assigned to those words relative to other vocabulary items. Therefore this is a useful distinction between computations, not a repaired full-preservation result or an identified four-property circuit.

This is a deterministic output decomposition of the existing program. It does not fit data, discover new input products, or implement general DAG search. The spelling relations supplying the output readers are human-specified annotations; this step is not wholly unsupervised semantic discovery.

## What is being split?

The existing component writes

$$
w(x)=Vf(x),\qquad V=A^\top(AA^\top)^{-1}.
$$

The three rows of $A$ are normalized mean unembedding contrasts for `s`, `es` and `y`→`ies`. The three scalar functions in $f$ are the frozen, trace-corrected bilinear approximations, including their original biases. Their execution uses three general products and 42 signed squares. Let $N$ contain the analogous two mean readers for `ed` and `ing`. No validation examples determine $A$, $N$ or the split.

We compare two meanings of a private output branch:

1. **Within the original writer span.** Find the part of $\operatorname{span}(V)$ annihilated by $N$. This is one-dimensional here. Project $w$ onto that line; the remaining two dimensions form the shared branch.
2. **Within the full residual space.** Project away the two-dimensional row space of $N$:

$$
P_N=N^\top(NN^\top)^{-1}N,
\qquad w_{\mathrm{private}}=(I-P_N)Vf(x),
\qquad w_{\mathrm{shared}}=P_NVf(x).
$$

In both constructions the branches are physically orthogonal and sum exactly to the old write. The private branch has zero mean `ed`/`ing` pre-normalization logit-contrast response. This does **not** guarantee zero response for each word pair, unchanged capped logits, or unchanged cross-entropy.

The ambient construction can leave the original three-dimensional writer span. Its private and shared writer ranks are three and two; the private writer matrix has 17.87% of its norm outside the original span. This freedom is explicit, not a claim that we found a new branch entirely inside the old span. Storing both writer matrices literally costs 6,912 floats versus the original 3,456. The scalar functions and their 45 variable products are reused, with no increase in their number. Native background computation remains necessary.

## Weight-only checks

Sum, orthogonality and mean-null identities pass within $2.31\times10^{-16}$. Applying each split to a write equal to each unit target reader gives:

| Private branch | Retained `s` mean response | Retained `es` | Retained `ies` |
|---|---:|---:|---:|
| Within original span | 2.87% | 11.99% | 26.13% |
| Full residual space | 85.71% | 91.40% | 93.19% |

The registered 50%-retention requirement fails for the first and passes for the second. These are responses to specified test writes, not behavioral recovery predictions. Over all matched vocabulary pairs, the ambient-private branch still has 46–61% of the original RMS `ed`/`ing` pair response despite its zero mean. Mean cancellation must not be confused with tokenwise invariance. [Compiler](native_relation_output_split_v1.py) · [Geometry and all-pair receipt](NATIVE_RELATION_OUTPUT_SPLIT_V1.json).

## Physical comparison on the existing panels

The managed run completed at **21:20:01 UTC** using 1,632 native tail rows and no model-body forwards. It evaluates both splits on the 32 original task pairs, 32 construction-holdout task pairs, and 32 neighboring-inflection pairs. These are known developmental panels; the construction panel is no longer fresh confirmation for this new split. Whole-program effects replay previous GPU measurements within $8.11\times10^{-6}$ nats. [Physical result](NATIVE_RELATION_OUTPUT_SPLIT_PHYSICAL_V1_RESULT.json).

The ambient-private swap retains about **73–77% of the whole verb effect** and **87–89% of the whole noun effect**. The registered 80% bar for every task/direction/panel therefore fails. Its removal still attenuates intended contrasts by about **12–18%**, but the requirement to retain 80% of the original attenuation also fails on verbs. The within-span private branch is weak, as separately predicted, and sometimes opposes the noun change.

| Neighboring-control outcome | Original whole component | Ambient-private branch |
|---|---:|---:|
| Past contrast attenuation | 4.17% | 0.23% |
| Progressive contrast attenuation | 3.21% | −0.41% |
| Past mean absolute CE change | 0.07078 nats | 0.06567 nats |
| Progressive mean absolute CE change | 0.05948 nats | 0.05104 nats |

The mean-null branch greatly reduces the change in these grammatical contrasts, while the 0.05-nat CE preservation criterion still fails on both families. **A/E pass; B/C/D fail.** The original component's failures and this split's failures remain authoritative. No fresh validation or adoption follows from this developmental screen.

## Why can the grammatical choice be preserved while CE changes?

For an answer $a$, its grammatical alternative $b$, and the full model probabilities $p$, the exact identity is

$$
-\log p(a)
=-\log\frac{p(a)}{p(a)+p(b)}
 -\log\bigl(p(a)+p(b)\bigr).
$$

The first term is binary-choice cross-entropy: how well the model chooses the correct form *conditional on choosing one of these two words*. The second measures how much probability the model assigns to that pair rather than the rest of the vocabulary. A write may preserve the relative grammatical choice while moving probability into or out of both alternatives together.

A separately registered CPU successor recomputed the full vocabulary and both terms for ambient-private removal. The identity holds within $6.44\times10^{-15}$ and original GPU CE effects replay within $5.84\times10^{-6}$ nats. Its three accounting criteria all pass:

| Mean absolute change | Past | Progressive |
|---|---:|---:|
| Full target CE | 0.06567 | 0.05104 |
| Binary-choice CE | 0.00118 | 0.00309 |
| Pair probability cost | 0.06529 | 0.04911 |

Pair-probability-cost changes have 99.52% and 98.97% of the full CE-change RMS. These ratios are not additive percentages of variance: the terms can correlate or oppose each other. The conclusion is that probability assigned to the lexical pair dominates the remaining CE effects. [Independent accounting](NATIVE_RELATION_OUTPUT_PAIRMASS_V1.json).

This explains the preservation failure without changing its verdict. The useful new distinction is between relative inflection choice and lexical probability allocation. A stronger circuit decomposition should make that distinction explicit and predict both effects on fresh examples. Repeatedly projecting out additional control directions until this particular panel passes would instead fit the test design; that is not the next discovery method.
