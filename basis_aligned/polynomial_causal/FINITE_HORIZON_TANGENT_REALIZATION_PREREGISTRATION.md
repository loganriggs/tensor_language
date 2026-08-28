# Preregistration: finite-horizon tangent minimal-realization discriminator

Date: 2026-08-28

Status: CPU mathematics, proof kernel, authoritative row/document plan, and sealed
response-bank lifecycle are frozen. This document opens no model and authorizes no GPU
measurement until a model-side Fisher-VJP collector is separately committed and independently
audited.

## Exact object

Use the admitted rank640 complete program. At early MLP write interfaces $i=0,1,2$,
inject deterministic natural-covariance-shaped residual directions $u_i$ at one frozen
position per row. For each of 16 probes, independently sample one categorical score at
every final-logit position from the injection position through 255 and sum those
scores. Thus $y$ includes every causally future output, not only a same-position logit.
The same-forward tangent responses form blocks

$$
H_i=\frac{\partial y}{\partial u_i}.
$$

For a depth cut $k$, vertically retain the row/probe contexts and horizontally
concatenate every upstream response:

$$
\mathcal H_k=[H_0\mid\cdots\mid H_{k-1}].
$$

This final-output-only pilot is the smallest behaviorally meaningful cut operator; it
does not pretend that an arbitrary intermediate logit lens is a model output. For one
fixed context $c$, every $r$-dimensional tangent state factors
$H_{c,k}=D_{c,k}E_{c,k}$, so $r\ge\operatorname{rank}(H_{c,k})$. Report the complete
distribution of these per-context ranks.

Vertical context stacking supports a narrower but directly compilable theorem. If one
**context-independent linear encoder** $E_k$ maps the registered edit coefficients
into an $r$-dimensional shared tangent code while the decoder may vary with context,
then $H_{c,k}=D_{c,k}E_k$ for every $c$, hence
$r\ge\operatorname{rank}(\mathcal H_k)$. If the encoder itself rotates with context,
the stacked rank is not a lower bound on an arbitrary nonlinear state. It is the
minimal shared linear interface/dictionary for the measured contexts and probes.
The SVD supplies the optimal shared rank-$r$ factorization in squared response error.
RMSNorm, residual addition, attention, and later MLPs are inside the measured Jacobian
rather than treated as separable Euclidean modules.

## Frozen pilot

- Program: immutable admitted shared-QK rank640 shell with exact MLPs.
- Role: the already authorized `fineweb_n96_skip80.pt` tensor, exact serialized SHA256
  `94bc1fb3...eda` and raw tensor SHA256 `a703cadb...1cc`.
- Sites: MLP0, MLP1, MLP2; no site selection after observation.
- Inputs: on all 96 registered rows and positions 64:256, compute the exact natural MLP
  write covariance $C_i$ separately at each site in float64. Numerical eigenvalues
  above `-1e-10 * max(1, lambda_max)` are projected to zero; measured support uses
  `1e-12 * max(1, lambda_max)`. Low support rank is admissible and reported. Construct
  32 directions as $g^T C_i^{1/2}$ with CPU Rademacher seed
  `2026082801 + 1000003*site + direction`, then normalize each to unit coordinate RMS.
  The finite direction bank is the measured intervention object; it is not claimed to
  have empirical covariance exactly $C_i$. Every
  direction is evaluated on every registered row. Sparse assignment over rows is
  forbidden because it would make the operator non-rectangular and turn missing
  Jacobian entries into implicit zeros.
- Tests: 16 fixed-seed categorical Fisher score probes of the final 50,304-logit
  distributions at every causally future position. Stateless uniforms derive from
  `SHA256(seed:row_id:absolute_output_position)`, so results are invariant to batch
  partitioning and cannot replay one uniform grid across batches. The inverse CDF is
  plateau-safe (`right=True`) and the CDF is clamped into `[0,1]`. Accumulate only
  detached CPU float64 responses and hashes; raw logits, graph-bearing tensors, and
  model aliases may not escape the collector.
- Seeds: directions `2026082801`, categorical probes `2026082802`, positions
  `2026082803`. Positions are frozen by the stateless SHA256 rule recorded in
  `finite_horizon_tangent_plan.json`, not by a version-dependent RNG stream.
- Replication: the authoritative provenance has 96 chunks but only 33 source documents,
  including documents contributing 16 and 18 chunks. Whole-document allocation is
  frozen at 48 rows/16 documents primary and 48 rows/17 documents replication. A naive
  48/48 row split is forbidden.
- Cut-3 operator shape: $768\times96$ independently on each split (48 rows times 16
  probes; three sites times 32 directions). Cuts 1 and 2 use the corresponding first
  32 and 64 columns.
- Cut rule: smallest rank retaining 95% squared singular energy, only when the singular
  gap is at least 2 and the rank is strictly below full measured support.

The exact plan fingerprint is
`062ad87d552112bd2064726848a5f3d1a1e1ee13118e01cf3a4b462c2c8e0141`.
`finite_horizon_tangent_response_bank.py` enforces complete source coverage, exact
shapes, detached CPU float64 responses, duplicate rejection, whole-document splits,
one-use sealing, content hashes, and fail-closed incomplete collection. It does not
claim to implement or audit the model-side VJP.

## Gates and consequence beyond reconstruction

1. Primary/replication response traces differ by at most 5%; selected shared-linear
   ranks differ by at most two; trace-normalized squared spectra have $L^1$ distance at
   most 0.10; normalized right-projector chordal distance in the common intervention
   coordinates is at most 0.15. Per-context rank distributions are reported separately
   and never called the stacked shared-interface rank.
2. Eight typed orthogonal gauge replays preserve every cut spectrum to relative
   $10^{-8}$ and physical projected responses to $10^{-6}$.
3. The selected cut factor predicts heldout single-direction Fisher responses with
   pooled $R^2\ge0.5$ and median relative error at most 25%.
4. Sixteen heldout two-site mixtures have median relative prediction error at most 30%.
5. Replacing discarded directions in the complete rank640 program changes natural-row
   CE by at most 0.005 nat and preserves at least 90% context recovery; retained matched-
   RMS directions must cause at least five times the discarded median response.

Failure of rank stability or nonlinear mixtures falsifies a shared linear state, even
if local reconstruction is good. A pass licenses only a three-site tangent-state
compiler; it does not certify finite edits or the remaining fifteen MLPs until repeated
without rank selection.

## Remaining pre-execution closure

`tensor_bilin18_tangent_collector.py` now implements the exact covariance/direction
rule, batch-invariant Fisher targets, additive write leaves, and a one-use projected-VJP
transaction on the admitted rank640 program. Zero leaves are added to complete writes;
the writes themselves are never detached, so an MLP0 effect still changes MLP1/2 and
all later tensor computation. Synthetic graph tests prove exact zero-edit identity and
nonzero late-logit gradients through all three interfaces. The source must be committed
and independently lifecycle-audited before GPU execution. Until then status is
**NO-GO**; this is a software/authority boundary, not missing FineWeb data, checkpoint,
or compute.
