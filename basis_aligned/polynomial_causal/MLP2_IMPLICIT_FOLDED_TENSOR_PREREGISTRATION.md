# Preregistration: implicit folded-tensor diagnostic for MLP2

Date: 2026-08-28

Status: no-outcome CPU contract. This document was frozen before loading any
checkpoint-derived MLP2 factor or spectrum. It authorizes no GPU work and no model,
row, activation, CE, KL, or intervention outcome.

## Frozen object

For MLP2, with `Down` columns $d_n$ and `Left`/`Right` rows $\ell_n,r_n$, the
quadratic part is the partially symmetric tensor

$$
T_{oij}=\frac12\sum_{n=1}^{K}d_{on}
(\ell_{ni}r_{nj}+r_{ni}\ell_{nj}),
\qquad K=4608,quad d_{in}=d_{out}=1152.
$$

`Down_bias` is not part of $T$. Every executable price and later candidate must retain
all 1,152 bias values exactly.

The diagnostic must not materialize the $1152^3$ tensor. It may materialize the two
$1152\times1152$ mode Grams and gate-pair workspaces. It computes:

1. the singular spectrum of scale-balanced `Down`, as the conservative decoder-only
   baseline;
2. the output-mode HOSVD spectrum from
   $G_o=T_{(o)}T_{(o)}^{\mathsf T}$;
3. one shared input-mode spectrum from
   $G_i=T_{(i)}T_{(i)}^{\mathsf T}$, after explicitly symmetrizing the two input
   modes.

The two input modes must be identical by construction. A result that treats them as
independent checkpoint axes is invalid.

## Gauge and numerical contract

For every nonzero native product, first minimize the sum of squared factor norms over
the exact gate gauge

$$
(\ell_n,r_n,d_n)\mapsto(a_n\ell_n,b_nr_n,d_n/(a_nb_n)).
$$

The balanced representative has
$\|\ell_n\|=\|r_n\|=\|d_n\|=(\|\ell_n\|\|r_n\|\|d_n\|)^{1/3}$.
Zero-contribution terms are canonicalized to zero and reported separately. Published
folded spectra and the balanced-Down spectrum must be invariant, within numerical
tolerance, to arbitrary nonzero gate rescalings, sign redistributions, `Left`/`Right`
swaps, and gate permutations.

All Gram matrices are accumulated in float64, explicitly symmetrized before
eigendecomposition, and rejected if non-finite or materially non-PSD. Roundoff-scale
negative eigenvalues are clipped to zero. Energy ranks are frozen at 90%, 95%, 99%,
and 99.9% of squared singular-value energy.

## Distinct hypotheses and claim boundaries

The following branches are frozen before seeing MLP2 spectra.

1. **Decoder-only branch.** If balanced `Down` has a small 95%-energy rank but the
   folded input mode does not, an SVD can simplify only the final mixing map. It does
   not reduce the 4,608 native bilinear products and earns no product-rank claim.
2. **Gate-reducing Tucker screen.** A dense symmetric Tucker core uses
   $r_i(r_i+1)/2$ input products. Therefore $r_i\leq95$ is necessary for that grammar
   to use fewer products than the native 4,608. Promotion additionally requires a
   mode-tail HOSVD error upper bound at most 5%, complete standalone storage below the
   native MLP2 price, and later finite causal/CE validation. Passing is only a Tucker
   feasibility screen, not a CP-rank or minimality certificate.
3. **Diffuse branch.** If neither the folded modes nor the price screen is compact,
   weight-only HOSVD is rejected as the next MLP2 compiler. It does not imply
   distributional or causal incompressibility.
4. **Intrinsic-refactor branch.** A folded spectrum materially more compact than
   balanced `Down` motivates a new factorization; it does not establish that the HOSVD
   axes are semantic, sparse, unique, or executable as few bilinear gates.

Down-only matrix error and folded-tensor Frobenius error are different currencies and
must never be plotted or subtracted as though they shared a denominator. HOSVD mode
tail sums are an upper bound for a Tucker truncation, not the measured error of a
constructed core. Description length, response rank, stored values, multiplication
count, and causal equivalence remain separate.

## Frozen executable prices

Bias is included exactly in every family.

- Native $K$-gate MLP:
  $K(2d_{in}+d_{out})+d_{out}$ stored values, $K$ bilinear products per token,
  and $K(2d_{in}+d_{out})$ dense linear-weight multiplications per token.
- Rank-$r$ Down-only SVD with native products:
  $2Kd_{in}+r(K+d_{out})+d_{out}$ stored values, $K$ bilinear products, and
  $2Kd_{in}+r(K+d_{out})$ linear-weight multiplications per token.
- Symmetric Tucker ranks $(r_o,r_i,r_i)$ with a dense triangular core:
  $d_{in}r_i+d_{out}r_o+r_o r_i(r_i+1)/2+d_{out}$ stored values,
  $r_i(r_i+1)/2$ bilinear products, and
  $d_{in}r_i+r_o r_i(r_i+1)/2+d_{out}r_o$ linear-weight multiplications per token.

Support indices, precision/quantization codes, fit artifacts, and source dependencies
are not silently free. The formulas above are fixed-grammar floating-value prices, not
literal MDL.

## Prospective MLP2 compensation tests

No cube result is opened here. Let `N` denote the native site and `P` an independently
frozen simplifier in the ordered MLP0/MLP1/MLP2 cube. Define the conditional MLP2 harms

$$
C_2^N=L(NNP)-L(NNN),\quad
C_2^{P0}=L(PNP)-L(PNN),\quad
C_2^{P1}=L(NPP)-L(NPN),\quad
C_2^{P01}=L(PPP)-L(PPN).
$$

The hypotheses for a later prospectively frozen cube are:

- **modular preservation:** a successful folded MLP2 has every conditional CE harm at
  most 0.02 nat and every $|C_2^u-C_2^N|$ at most 0.01 nat;
- **native compensation dependence:** if MLP2 specifically compensates for simplified
  upstream states and the folded candidate misses that role, then
  $C_2^{P01}-C_2^N>0.01$ nat, with the same sign in the relevant KL and causal-response
  interaction;
- **robust captured compensation:** `PPP` meets the later whole-program CE/KL gate,
  the frozen causal bank retains at least 90% response with its confidence bound, and
  the conditional interaction does not worsen on the second domain;
- **fragile cancellation:** in-domain `PPP` succeeds but the conditional effect changes
  sign or breaches its bound on OOD rows or interventions. This licenses behavioral
  compression only, not a composable causal abstraction.

The cube must use common rows and denominators, preserve all eight cells, and report
CE, KL, top-1, causal response, mixture/Mobius interactions, and target/off-target edit
collateral. The weight-only diagnostic earns none of those credits by itself.

## Synthetic admission tests required before checkpoint use

The implementation must pass known-answer tests that:

1. match implicit output/input Grams and spectra to an explicitly materialized small
   symmetric tensor;
2. preserve the tensor and equalize factor norms under balancing;
3. preserve all authoritative spectra under scale/sign gauges, gate permutations, and
   `Left`/`Right` swaps;
4. exhibit a constructed case where rank-one `Down` coexists with a rank-two folded
   input mode;
5. verify energy ranks, HOSVD tail bounds, PSD failure behavior, zero-gate handling,
   and every price formula including bias.

Checkpoint-derived MLP2 factors or numbers may be opened only after these tests pass.
