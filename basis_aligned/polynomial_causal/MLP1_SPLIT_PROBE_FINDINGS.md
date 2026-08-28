# MLP1 split-probe finding: concentrated energy, but no repeatable local subspace

Date: 2026-08-28

Status: preregistered gate failure, `no_admitted_local_bundle`.
Result SHA-256:
`0d4314f34c4bf55c3542bac5ac0a7ebec593841669d3b5a484f0a9f3a47cdcdd`.
Authority SHA-256:
`badbf2c69d76cca580155b3cfb274e96d8a3e0c4e8ee1f4bbe8cbdeae2c49e4d`.

## What was measured

For each of 16 fixed FineWeb contexts, insert an additive residual-stream variable
$\delta_c\in\mathbb R^{1152}$ immediately after MLP1 at token position 128. For a
sampled suffix target trajectory $y_a$, define the categorical score

$$
s_{c,a}(\delta_c)
=\sum_{t=128}^{255}\log p_\theta(y_{a,t}\mid x_c,\delta_c).
$$

Each $y_a$ is drawn deterministically from the model's own 50,304-way output
distribution using a frozen seed. Its gradient at zero edit,

$$
g_{c,a}=\left.\nabla_{\delta_c}s_{c,a}(\delta_c)\right|_{\delta_c=0},
$$

is a categorical-Fisher probe: repeated draws estimate which MLP1 write directions
can change the model's entire future output distribution, not merely the next token.
The probe response along each of 32 previously frozen natural MLP1 directions $d_n$
is

$$
H_{c,a,n}=g_{c,a}^{\mathsf T}d_n.
$$

Thus each context produces two independent $32\times32$ matrices,
$H_c^{(A)}$ and $H_c^{(B)}$, with identical contexts and directions but disjoint
Fisher draws. If a real low-dimensional local interface exists, both halves should
recover nearly the same physical subspace.

The $d_n$ are not orthogonal. To avoid treating a change of coefficient coordinates
as a physical discovery, the analysis factors

$$
D^{\mathsf T}=QR,\qquad \widetilde H_c=H_cR^{-1},
$$

and maps right-singular frames back into residual-write space as $QV_r$. All reported
subspace distances are therefore physical and invariant to invertible remixes of the
32 probe directions.

## What the terms mean

- **Support rank** counts singular values above a numerical tolerance. It was 32 in
  both halves for every context, so none of the matrices was numerically singular.
- **$r_{95}$** is the smallest number of singular directions containing 95% of squared
  response energy. It ranged from 10 to 17.
- **Selected rank** accepts $r_{95}$ only when the next singular value is at least a
  factor of two smaller. Zero means “no defensible spectral cutoff,” not rank zero.
- **Physical projector distance** at rank $r$ is

  $$
  d(U,V)=\sqrt{\frac{r-\lVert U^{\mathsf T}V\rVert_F^2}{r}}.
  $$

  It is zero for identical subspaces and one for orthogonal subspaces.
- **Same-context distance** compares the two independent Fisher halves for the same
  document. It estimates probe/sampling instability.
- **Cross-context distance** compares halves from different documents. A larger value
  can indicate context variation, but only after the same-context estimate is stable.

## Result

| physical rank | same-context mean | cross-context mean | cross minus same | 95% bootstrap lower bound |
|---:|---:|---:|---:|---:|
| 8 | 0.6167 | 0.8532 | 0.2365 | 0.2184 |
| 16 | 0.5621 | 0.6965 | 0.1343 | 0.1202 |
| 24 | 0.4525 | 0.4969 | 0.0444 | 0.0368 |

These table entries are diagnostics over all 16 contexts. On the fixed 12-context
promotion cohort, the registered rank-16 values are 0.5604 same-context, 0.6931
cross-context, 0.1327 difference, and 0.1180 bootstrap lower bound.

The predeclared rank-16 same-context ceiling was 0.15; the observed mean was 0.5621,
and every individual context exceeded the ceiling. Every energy-plus-gap selected
rank was zero. Consequently:

- stable local low-rank fraction: **0/16**;
- stable fraction in the fixed 12-context promotion cohort: **0/12**;
- probe-limited high-rank fraction: **0/16**;
- context-varying response-bundle gate: **failed**.

The run took 92.94 seconds: 67.41 seconds to rebuild and verify the complete rank-640
program, and 24.75 seconds for all 256 backward passes plus analysis.

## Interpretation

MLP1's downstream response energy is moderately concentrated: roughly 10--17
directions contain 95% in each finite probe sample. But there is no sharp spectral
knee, and two independent 32-probe estimates on the same context recover very
different subspaces. The cross-context contrast is real in this sample, especially at
ranks 8 and 16, but it sits on top of large same-context estimation variation.
Therefore it cannot authorize a context-conditioned low-rank compiler.

There is nevertheless some context signal. At rank 16, the normalized overlap implied
by the distance is $1-0.5621^2\approx0.684$ within context, versus
$1-0.6965^2\approx0.515$ across contexts; two random rank-16 subspaces in the measured
32-dimensional space have expected overlap 0.5. The likely object is therefore a
smooth context-dependent response covariance whose finite-probe eigenspaces are not
canonical, rather than either a clean rank-16 state or pure noise.

This resolves the earlier ambiguity. The prior 16-probe saturation was not evidence
for a stable rank-16 state, and simply doubling to 32 probes does not reveal one. It
also does not prove that MLP1 is intrinsically high-dimensional: a much larger probe
bank could estimate a smooth covariance spectrum more accurately. That route is
expensive and, without a sharp cutoff, unlikely to yield an executable decomposition.

The more useful next object is the model's exact physical product gates. For MLP1 gate
$n$ at token position $q$ in context $c$,

$$
h_n(z_{c,q})=(\ell_n^{\mathsf T}z_{c,q})(r_n^{\mathsf T}z_{c,q}).
$$

A gate is shared across all positions, so the exact first-order response to scaling
that gate globally is trajectory-complete:

$$
E_{(c,a),n}
=\sum_q h_n(z_{c,q})\,d_n^{\mathsf T}g_{c,a,q}.
$$

The expression without $\sum_q$ is exact only for a position-local edit and cannot
justify pruning a shared gate. $E$ directly attributes downstream Fisher response to
existing bilinear gates. Paired ridge-leverage or response-energy column selection can
be evaluated on independent probes and fresh documents without requiring a spectral
knee. Selected columns have an executable meaning—retain or remove specific products—
but response capture alone is not a compiler: finite CE, KL, coverage-stratified
accuracy, causal recovery, and collateral-damage tests must still pass.

## Claim boundary

This is conditional response geometry on 16 historical documents. It neither changes
the whole-model explanation denominators nor licenses a finite MLP replacement,
semantic axis, causal extraction, or removal. Raw responses, logits, Fisher targets,
frames, and projectors were not published. The result only rejects the preregistered
local-bundle alternatives at this probe budget and geometry.
