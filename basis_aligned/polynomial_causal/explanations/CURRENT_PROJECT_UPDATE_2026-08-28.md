# Current bilin18 reverse-engineering update

Date: 2026-08-28

This document separates four questions which are easy to conflate:

1. Can we write every model operation ourselves?
2. Can we make the resulting executable smaller without materially changing it?
3. Can we assign human-readable meanings to its internal variables?
4. Can we predict and selectively edit the model from that explanation?

We have a complete answer to the first question, the first prospectively certified
positive answer to part of the second, and much less complete answers to the third and
fourth.

## Executive update

The largest new positive result is a **complete rank-640 shared-QK tensor program**.
It stores 516,707,766 floating-point values instead of 545,904,054, a reduction of
29,196,288 values or 5.3481%. It owns the complete forward pass: there is no call back
to the checkpoint, no hidden lookup table, and no unsupported token fallback.

The same fixed executable passed two distinct prospective tests:

- On two cross-task FineWeb roles, its increase in cross-entropy was only
  +0.00553 and +0.00445 nat per target. On targets whose current token was unseen in
  the fit rows, the increases were +0.00766 and +0.00391.
- On a frozen bank of 16 prefix interventions, it recovered 94.44% of the squared
  downstream change on average, with a one-sided 95% bootstrap lower bound of 92.73%.
  Its change vector had mean cosine 0.9724 with the native model, lower bound 0.9637,
  and 14/16 fixtures passed both individual thresholds.

This is an important proof of concept for the proposed definition of simplicity:
fewer executable numbers bought retained prediction **and** retained causal response.
It is not a semantic explanation of the 640 coordinates. The program still keeps all
dense MLPs exactly, and those MLPs contain 286,675,200 values, 52.51% of the original
model.

The current high-priority mechanistic target is therefore the early MLP interface,
especially MLP1. The first tangent experiment found no small context-independent
linear interface: the rank needed for 95% response energy was 24/22 at the first cut,
27/26 at the second, and 31/29 at the third, and 94 of 96 context/site cases hit the
16-probe measurement ceiling. Equal-RMS interventions attributed 94.90--95.44% of the
cut-3 response energy to MLP1, less than 1% to MLP0, and about 4% to MLP2. Thus the
negative result was still useful: it localized the present wall to MLP1 and showed
that a universal state of dimension at most 16 is not supported by the current data.

A stricter MLP1 follow-up is frozen and CPU-audited, but **has not been measured yet**.
It uses the same context twice with two independent sets of downstream probes. This
will distinguish genuine context-dependent rotation from Monte-Carlo probe noise. The
GPU is available; the blocker is that the create-only collector enforcing the frozen
plan and result lifecycle has not yet been implemented.

## Honest completion balance

There is no defensible single percentage for “the model is understood.” Different
denominators answer different questions.

| question | current result | what it does not imply |
|---|---:|---|
| structural inventory | 36/36 attention and MLP sites | meaning or minimality |
| exact standalone ownership | 545,904,054 / 545,904,054 values | compression |
| certified whole-program storage removal | 29,196,288 values = 5.3481% | semantic names for retained state |
| named semantic behavior | 32.1% ± 6.4% | a complete executable explanation |
| named strict causal recovery | 10.923% | most causal behavior remains unnamed |
| unexplained strict causal currency | 4.72714 nat = 89.077% | not all of this must require separate circuitry |

The most meaningful global statement is: **the tensor network is fully reimplemented,
5.35% is now removable under prediction-and-causality tests, but roughly 89% of the
strict causal behavior is not yet assigned to named, composable mechanisms.**

## The computations in the model

### RMSNorm

For a residual vector $x\in\mathbb R^{1152}$, RMSNorm divides by its root-mean-square
magnitude and applies learned coordinate scales:

$$
\operatorname{RMSNorm}(x)
=g\odot\frac{x}{\sqrt{1152^{-1}\sum_j x_j^2+\epsilon}}.
$$

It preserves the direction of $x$ up to coordinate scaling but removes one overall
magnitude degree of freedom. It is nonlinear only through the scalar denominator.

### Bilinear MLP

If $z=\operatorname{RMSNorm}(x)$, each MLP computes

$$
h_n(z)=(\ell_n^\top z)(r_n^\top z),\qquad
M(z)=b+\sum_{n=1}^{4608}d_n h_n(z).
$$

So every hidden feature is a product of two linear measurements. Each output is a
quadratic polynomial in $z$. The matrices called `Left`, `Right`, and `Down` store
$\ell_n$, $r_n$, and $d_n$ respectively; `Down_bias` is the separate $b$ term. One
older exact-control experiment omitted this bias and was corrected: it had measured a
zero-bias ablation, not the native identity.

### Shared-QK rank 640

At each attention site there are four related routing projections, `q`, `k`, `q2`,
and `k2`. The compiler measures the covariance $A=\mathbb E[zz^\top]$ of the natural
input state and finds one rank-640 input encoder $E$ plus four typed decoders $D_j$:

$$
W_j\approx D_jE,
\qquad j\in\{q,k,q2,k2\}.
$$

It minimizes the activation-weighted error

$$
\sum_j\left\|A^{1/2}(W_j-D_jE)\right\|_F^2.
$$

Whitening by $A^{1/2}$, concatenating the four maps, taking the leading singular
subspace, and unwhitening gives the weighted Eckart--Young solution. “Rank 640” means
the four projections share 640 intermediate continuous coordinates. It does **not**
mean 640 token classes or 640 named concepts.

## What is currently understood about MLP0

MLP0 is not a hard clustering function. Its exact function is

$$
m_0(z)=D((Lz)\odot(Rz))+b.
$$

A useful descriptive decomposition of its observed writes is

$$
m_0(t,c)=\mu+C[g(t)]+\bigl(T[t]-C[g(t)]\bigr)+R(t,c)+\varepsilon(t,c).
$$

Here:

- $T[t]$ is the mean MLP0 write for token $t$ on fit documents;
- $C[g(t)]$ is the mean of token codes in a lexical group such as numbers or
  punctuation;
- $T[t]-C[g(t)]$ retains within-class token distinctions;
- $R(t,c)$ predicts continuous context-dependent residual structure;
- $\varepsilon$ is what remains unexplained.

This is how “numbers cluster” and “downstream layers distinguish every number” can
both be true. A shared component need not erase continuous token-specific differences.
The literal stack of immediate linear readers has rank 1152/1152, so almost every
difference is distinguishable at exactly zero tolerance. That makes exact equivalence
too strict to be a useful definition of simplicity. The operational question is which
differences can be removed at bounded downstream cost.

Several distinct compression results support a continuous, low-dimensional view, but
they concern different objects:

- A legacy 256-quadratic-feature surrogate retained about 97.8--97.9% under its older
  mean-floor-relative behavioral denominator.
- A token table plus inherited-context correction retained 84.33% at rank 0 and
  89.39% at correction rank 256 under a later held-out protocol.
- The C512 program leaves RMSNorm, `Left`, `Right`, and all 4,608 products exact, but
  replaces `Down` by a rank-512 map. It makes `Down` about 3.60 times smaller and had
  maximum ordinary output KL about 0.00533 and maximum CE harm about 0.00549 over 384
  unseen FineWeb documents.
- A rank-64 output subspace contains only about 37% of MLP0 residual energy but
  recovers about 79.9% of its measured held-out causal CE effect. This demonstrates
  why variance alone is the wrong importance metric.

None of these says that the semantic meaning of every retained coordinate is known.
The rank-64 axes can be rotated by any orthogonal matrix without changing the physical
subspace, a freedom called a **gauge**. Semantics should be assigned only after jointly
choosing a gauge that makes both the MLP0 producer and downstream consumers simple,
and after interventions confirm the proposed names.

MLP0 also cannot be compiled independently of its consumers. Exact restoration gives
rough singleton CE gains $+0.119,+0.167,-0.230$ for MLP0/1/2, whose sum is only
$+0.056$, while restoring all three together gives about $+0.514$. The interaction
surplus, about $0.458$ nat, is much larger than the additive prediction. This is direct
evidence that the early MLPs form a state-conditioned program rather than three
independent feature writers.

## Why the context-free lexical program is not the whole explanation

A separate position-wise program predicts from the current token and position-wise
tables but has exactly zero cross-position dependence. Its full-table arm agrees with
the live model's top choice on only 22.7--24.2% of positions. Its top-1 accuracy is
13.6--14.3%, versus 38.9--42.3% for the live model, and
$\mathrm{KL}(p_{\rm live}\|p_{\rm table})$ is 2.75--3.05 nats.

The error is highly structured. The program keeps about 62.9--63.5% of live accuracy
when the correct target occurred at least 125 times in the fit data, but only 2.7--6.2%
for unseen targets. It is therefore a useful frequent-lexical-target baseline, not a
model of contextual computation. These measurements do not evaluate the rank-640
whole program; an analogous top-1/KL audit of rank640 is still required.

## The MLP1 tangent computation

A **tangent response** is the first derivative of downstream behavior with respect to
a small internal edit. For context $c$, write-direction $d_i$, and downstream score
probe $s_a$, the measured matrix is

$$
(H_c)_{ai}=\left.\frac{\partial s_a}{\partial\alpha}
\right|_{\alpha=0,\;m_1\mapsto m_1+\alpha d_i}.
$$

The probes are categorical-Fisher score directions. Informally, they ask how the
model's predicted token distribution moves, weighting logit changes in the local
geometry of probability distributions rather than treating every raw logit equally.
This places RMSNorm, residual addition, attention, later MLPs, and the final softmax
inside the measured downstream environment.

The singular values of $H_c$ say how many independent input directions have observable
downstream effects. The statistic $r_{95}$ is the smallest rank whose squared singular
values contain at least 95% of $\|H_c\|_F^2$. SVD is relevant because the truncated SVD
is the best rank-$r$ linear approximation in squared error.

The first run used 32 write directions but only 16 probes per context. A measured rank
near 16 can therefore be a measurement ceiling. It also compared different documents,
so apparent context rotation could be probe noise. The frozen follow-up corrects both:
each of 16 contexts gets two disjoint sets of 32 probes, at the same MLP1 edit and the
same 128 downstream positions.

Because the 32 physical write directions are not orthogonal, their coefficient-space
SVD is coordinate-dependent. The corrected computation takes

$$
D^\top=QR,\qquad \widetilde H=HR^{-1},\qquad U_r=QV_r(\widetilde H).
$$

$Q$ is an orthonormal basis for the actually edited write subspace; $R^{-1}$ removes
the arbitrary skew and scaling of the registered direction coordinates; $U_r$ is the
resulting physical rank-$r$ frame. Tests verify that an invertible change of direction
coordinates leaves the physical projector unchanged.

The two probe halves are compared at the same fixed rank 16. If their same-context
projectors agree but projectors differ across contexts, that supports a
**context-varying response bundle**: a small local causal subspace whose orientation
depends on context. It still would not identify whether MLP1's encoder, the downstream
decoder, or both rotate, because $H_c=D_cE_c$ has an internal gauge.

The promotion cohort is the first 12 hash-frozen documents; the other four are
diagnostic only. This fixed cohort avoids choosing the documents that happened to look
stable and then bootstrapping the selected winners. The current analyzer and plan pass
62/62 tangent tests. Plan fingerprint:
`236d83c6779b064e266a51594edaab2bf4c961006c4ab7905f0e946aa48e16c6`.

## Statistical terms used

- **Cross-entropy (CE):** $-\log p(y)$ averaged over true next tokens. A harm of
  +0.005 nat means the simpler model assigns slightly less probability to the truth on
  average. CE is sensitive to the full predicted probability of the truth but does not
  show whether both models choose the same top token.
- **KL divergence:**
  $\mathrm{KL}(p\|q)=\sum_v p_v\log(p_v/q_v)$. Here it measures the whole live output
  distribution against the program, not only the true-token probability.
- **Top-1 agreement:** fraction of positions where both models' argmax token is the
  same. **Top-1 accuracy** instead compares each argmax with the actual next token.
- **Causal recovery:** for native and program downstream change vectors $\Delta_N$ and
  $\Delta_P$,
  $1-\|\Delta_P-\Delta_N\|_2^2/\|\Delta_N\|_2^2$. One is exact, zero is no better than
  predicting no change, and it may be negative.
- **Cosine:**
  $\langle\Delta_N,\Delta_P\rangle/(\|\Delta_N\|\|\Delta_P\|)$. It tests whether the
  direction of the response is correct independently of its magnitude.
- **Bootstrap lower confidence bound (LCB):** repeatedly resample whole intervention
  fixtures with replacement, recompute the mean, and take the fifth percentile. The
  one-sided 95% LCB asks whether even a conservative population estimate clears the
  preregistered threshold.
- **Preregistration:** freeze rows, hashes, metrics, thresholds, and branch decisions
  before opening outcomes. This prevents choosing a flattering metric or subset after
  seeing the answer.
- **Source closure / create-only result:** bind the exact code and parent artifacts and
  permit one atomic result publication. These are safeguards against silently changing
  the experiment after observing it.

## Current plan, in priority order

1. Implement and independently audit the create-only MLP1 paired-probe collector, then
   run the frozen assay. This directly resolves whether MLP1 has a stable local low-rank
   response state or a genuinely high-rank/context-rotating interface.
2. Branch on that result. If both independent halves require rank above 16, stop trying
   to force a tiny local state and build the cross-depth finite-horizon tangent
   realization. If a stable rank-16 local frame exists but rotates across documents,
   fit a small context-to-frame transport and test it on held-out documents.
3. Use downstream tangent consequences, rather than local activation MSE, to select
   actual MLP product gates or a shared factor dictionary. This directly attacks the
   52.51% dense-MLP storage block while keeping the factors executable.
4. Run top-1 agreement and full-distribution KL on the admitted rank-640 whole program,
   then add extraction, selective removal, collateral-effect, and finite-edit tests.
5. Require any proposed semantic coordinates to make a new prediction: better OOD
   transport, cheaper consumers, selective removal with bounded collateral, or a
   verifiable lower-dimensional executable. A prettier basis without one of those
   consequences does not count as increased understanding.

The central story is now coherent: tensor structure let us build and certify a smaller
complete attention shell; polynomial structure tells us exactly what the MLP atoms are;
causal tangent geometry tells us which combinations the downstream model can observe;
and joint gauge/description-length optimization is the route from compressibility to
human-readable, editable programs.
