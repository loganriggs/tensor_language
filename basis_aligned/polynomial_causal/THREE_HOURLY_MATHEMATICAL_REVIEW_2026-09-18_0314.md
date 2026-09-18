# Mathematical review: conditional reader folds versus causal abstraction

Actual review UTC: 2026-09-18T03:14:10.379528+00:00. Next mathematical deadline: 2026-09-18T06:14:10.379528+00:00. Previous review00:05; the02:58scheduled invocation skipped at173minutes because its prompt required175minutes. This review catches up at an actual safe boundary; no offline review is fabricated. ACTIVE_TRACK remains WEIGHT_FOLDING from the02:57hourly review; do not restart the hourly clock.

The packed city operator predicts native effects on new FineWeb documents but fails directional removal and interchange. Its new six-piece MLP8→head9.8 value fold reproduces native responses, and opened interventions show that the MLP8 and quadratic value terms matter. Neither fact establishes a universally directional regional circuit, independent composition, or a token-only executable.

## Exact object and price

For each source position s, z_s,d_s∈R^1152 are the native unnormalized MLP8 input and upstream attention8 edit. L,R∈R^(4608×1152), D∈R^(1152×4608), b∈R^1152. Define S(z)=||z||²/1152+epsilon, epsilon=FP32 machine epsilon. Native M(z)=D[(Lz)⊙(Rz)]/S(z)+b. The hidden numerator is degree2; normalized M is rational, not globally polynomial. RMS9 introduces square roots again, and full downstream attention includes normalized products and softcap. No finite-degree full-network polynomial is asserted.

For head9.8 current-value reader W∈R^(128×1152), fold C=WD∈R^(128×4608). Exactly in real arithmetic:

W[M(z+d)−M(z)] = C[(Ld⊙Rz + Lz⊙Rd + Ld⊙Rd)/S(z+d) + (Lz⊙Rz)(1/S(z+d)−1/S(z))].

Bias cancels here, but stays in the supplied native background. With m0,m1 the native/edited pre-attention9 mixed residuals, rho_i=RMS(m_i), lambda the block9 residual multiplier and mu the inherited-value mixture, the value difference is:

(1−mu) lambda/rho1 [Wd + WΔM] + (1−mu) Wm0(1/rho1−1/rho0).

This gives six explicit pieces: direct; two ordered crosses; quadratic; MLP8 norm change; block9 norm change. All share the same native factors and background. L↔R exchange and reciprocal hidden-channel rescaling preserve the function; hidden permutations also do. Named left/right terms are gauge-dependent native-coordinate attributions, not individually identified semantic variables.

Current helper takes z,d,m0,m1 and fixed factors; it does **not** derive the mixed9 norms from a smaller closed state. Four [1,T,1152] tensor inputs remain (d is an intervention specification, not another unedited native state). Program11,354,114FP32 values/45,416,452bytes, including L,R,C,W and two scalars. Computing C once costs O(128·1152·4608); each position then needs four L/R projections plus128-dimensional folded contractions. The full native suffix remains external, and its compute cost is not saved by this local attribution instrument.

Native-context40-sequence replay passes: max value-relative error2.64e-5, below1e-4. FP32 factor rounding/native execution explains why symbolic exactness is not bit equality. Dropping MLP8 gives response error2.01reversed/.418other versus.35gate; quadratic norm is.367/.349 versus.10gate. The five-arm installed factor intervention passes its declared magnitude screens, including88%reversed-group effect change when subtracting MLP8 response and26%when subtracting quadratic. These are opened response/edit results; no random-value-boundary null or fresh mediation confirmation yet.

## Literature mapping and assumptions

| Candidate | Exact mapping / guarantee | Assumptions not established here; executable consequence |
|---|---|---|
| Constructive soft causal abstraction | Low-level deterministic graph has token, residual, Q/K/V, MLP and readout variables. Proposed high-level variable is the128-dimensional value response with explicit paired-context inputs. The theory imposes intervention consistency and conditions under which the low-to-high intervention map is uniquely induced. | We have no surjective high-level map satisfying consistency for all permitted settings/interventions; cached per-row corrections do not supply one. No discovery/runtime guarantee follows. Require one uniform value-response function with declared source/base inputs, then test new interventions and contexts; never equate a response-conditioned scalar reader with a predictive abstraction. [Massidda et al.](https://proceedings.mlr.press/v213/massidda23a/massidda23a.pdf) |
| Path patching | Fix the upstream city swap, alter only its induced head9.8 value response, preserve both keys and other heads, recompute suffix. This is an explicit path-localization hypothesis at an internal factor boundary. | A large response contribution alone is not the intervention result. The existing five-arm mediator test supplies that distinction; matched controls and fresh replication still follow. Runtime is the number of recomputed model forwards; no guarantee of unique circuit or minimal representation. [Goldowsky-Dill et al.](https://arxiv.org/html/2304.05969) |
| Distributed alignment | Native W identifies a specific value port; a fitted rotated subspace would instead require interchange validation. | Alignment training can propose representations; our exact native fold requires no such fit. Rotational freedom is not removed merely by good reconstruction. No reason to introduce a probe basis after native response replay succeeds. [Geiger et al.](https://proceedings.mlr.press/v236/geiger24a) |
| Tensor-train decomposition | A sampled multi-index response tensor could be represented via ranks of matrix unfoldings and sequential low-rank factorizations. | This would factor a chosen grid, not the normalized causal program for unseen contexts. Rank/approximation bounds do not establish selective edits or independent pieces; sampling storage grows with grid size and factorization cost depends on unfolding sizes. No closed causal-domain approximation certificate is available, so demote a new TT sweep. [Oseledets](https://epubs.siam.org/doi/10.1137/090752286) |
| Hankel / weighted-automaton realization | String→effect map would define H(u,v)=f(uv); finite-rank linear recurrence could admit spectral realization. | Exact equivalence applies to weighted automata/linear second-order recurrence. Our position-dependent normalized transformer and paired intervention contexts have no demonstrated finite-rank closed transition. SVD of a finite sampled block cannot prove that assumption. No unique finite-state circuit or intervention mapping follows. [Rabusseau et al.](https://proceedings.mlr.press/v89/rabusseau19a.html) |

Other neighboring routes retain the previous review's constraints: arithmetic-circuit/bilinear complexity applies to the unnormalized contraction, not automatically across RMS; polynomial identities certify equality rather than causal ownership; invariant/gauge methods require operational reader equivalence; graph-width contraction can price evaluation but not identify directionality. Do not repeat prior reader-kernel lower-bound or rational-strength theorem work without a changed domain assumption.

## Executable consequence and next choice

The algebra above is already implemented uniformly in [mlp8_current_value_response_v1.py](mlp8_current_value_response_v1.py), checked with actual weights, validated on native contexts, and used for the factor-specific intervention. Its separate block9-normalizer input prevents a false port-closure claim. The response-conditioned final secant reader is used only for attribution; installed interventions use the128-dimensional value correction and the native full suffix, not observed target logits.

Highest-information next step: registered per-position norm-matched random corrections at the **same head9.8 value boundary and same upstream-swap background**. Opposing predictions: a specifically aligned MLP8 response should exceed matched random target movement while preserving unrelated readers; mere high gain at the value port will also amplify random directions. A pass requires fresh confirmation before naming a reusable mediator. The mathematical abstraction conditions favor this consistency test over another low-rank representation. A failure blocks selective mediation promotion even though algebraic replay remains true.

CIRCUIT handoff: opened mediation specificity, then fresh context/structural endpoint tests without dropping reversed documents. WEIGHT_FOLDING handoff: keep the six-term rational response and supplied normalization context; pursue a smaller closed dependency only after its causal role survives controls. Preserve hourly alternation.

## Organization and efficiency

Registries and Logan03:04report preserve FineWeb failures and distinguish response, fold and edit. New native-fold/mediation receipts will be linked in the next report and canonical registries. Package manifests retain Pile selective evidence separately from failed FineWeb direction;392-token extension and external state are explicit. No promotion follows merely from the48-package inventory or two historical four-trait entries.

Scientific execution: native six-piece replay3.183s/80equivalents; mediation7.189s/200equivalents. Recent source-backed pipelines still duplicate short batching/scoring blocks, but immutable bound experiments should not be retroactively refactored. Prefer reusing one established template for the next test; no new framework or GPU job is warranted. Sparse phase timestamps do not establish a complete labor allocation, and no time-saving estimate is invented.

Executed bounded process repair: [review_due.py](../../session_recovery/review_due.py) makes the cron wrapper wait until the previous receipt's interval has elapsed, **before taking the shared review lock**. Filename timestamps round down; adding one minute prevents premature launch. Waits use≤30-second sleeps and a55-minute total guard. Boundary tests reproduce the exact failure (00:05receipt/02:58cron→8-minute wait), due-time immediate start, missing-receipt immediate start and early hourly slot. Bash syntax check passes. This removes the early-trigger/175-minute skip gap without changing cron times, models, queues or active experiments. The already-skipped00:05receipt is not relabeled a new review.

Four properties: fresh effect prediction supported at native boundary; extraction conditional; FineWeb directional removal/interchange failed; composition unresolved/failed for tested splits. Simplicity priced separately, matched-effect null unresolved. No program-level completion.
