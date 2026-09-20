---
name: logan-decomposition-pullback-direction
description: "Logan's 19 Sep 22:12 UTC direction — decompose the unembedding pulled back through the last MLP and last attention block with unsupervised methods, compare decompositions by simplicity; plus DCT/fixed-position/mixed-partials/distributional folding, all approved"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3a5c1d2d-901e-4c52-8edc-21792b8663e5
  modified: 2026-09-19T22:12:46.258Z
---

Logan (19 Sep 2026, ~22:12 UTC), replying to my accounting of his earlier folding-families message ([[logan-template-contraction-direction]]):

**Approved from the earlier menu:** DCT feature folding, fixed-position/fixed-token forward folding, mixed partials / eigenfilters, distributional
folding (empirical/Gaussian moments). **Declined:** probe/steering-vector targets, transcoder latents, checkpoint-diff tensor-sim.

**Priority ("work hard on this"): the subspace fold, taken further.** His words, paraphrased tightly:
- Don't just decompose W_U (the unembedding) with SVD — decompose it with an unsupervised method (he was unsure whether we'd discussed top-k SAE,
  k-clustering, or something else; check for prior art first, "surely there's work there").
- Better: decompose the COMPOSED path unembedding → Down projection of the last MLP (W_U @ Down_17), so the decomposition lives in the last MLP's
  hidden-unit basis (4608-wide), not the residual basis. "Can you pull back farther?"
- Ideal: pull the full unembedding back through the last MLP AND the last attention block (unembedding → last MLP → last attn) and find structure
  in that whole composed path; fold that structure back. "That structure is whatever decomposition method you use."
- Run multiple decomposition methods (hierarchical, DAG, sparse, SVD, etc.) on the same composed object and determine which is SIMPLER than the
  others for the same fidelity — need an actual comparison metric (e.g. reconstruction error vs. parameter/bit budget; Pareto frontier), not a vibe.
- Any trained decomposition (SAE, dictionary learning, clustering) must be trained to convergence — track the loss curve and state the convergence
  criterion, don't truncate for compute budget.
- SVD is the mandatory baseline every other method is compared against.
- Explicit instruction: **run by default, set hourly reminders** (his cadence rule continues), and prioritize this over the smaller approved items.

**Why:** he wants the unembedding's composition along the residual-stream path treated as a first-class object with real structure-discovery
(not just a hand-picked token-class subspace as in v390–v400), and wants competing decompositions judged on a real complexity/fidelity trade rather
than picked by feel.

**How to apply:** build the pullback objects at increasing depth (W_U; W_U @ Down_17; the last-MLP input directions L_17/R_17 pulled back through
attention 17's per-head OV paths) as weights-only matrices, no forward pass needed for the objects themselves. Run SVD baseline first, then at
least one sparse method (dictionary learning / NMF / small SAE trained to convergence) and one clustering method (k-means flat, agglomerative for
a DAG/hierarchy), same object, same reconstruction-fidelity target, and report a parameter-or-bits-vs-error curve for each. See
[[claude-circuit-lane-2026-09-17]] for the lane conventions (GPU only via ops/enqueue.sh, board, scorecard) this still runs under where it fits,
but this is an analysis/tooling thread, not a circuit-DoD experiment — don't force it into the BQGATE pass/fail prediction format where it doesn't
belong (no fits vs preregistered predictions tension: fits here are the deliverable, not a nominating step, as long as they're run to convergence
and compared honestly against SVD).

**Progress (19 Sep, same evening, v600-v602).** Extracted M0=W_U [50304,1152] and M1=W_U@Down_17 [50304,4608] weights-only (v600); attention-17's own
OV paths explain only ~2% median of what MLP-17 reads (RESID ~0.98) -- most of MLP-17's read direction is already present before attention 17 acts, or
comes from the deeper token/v1 branch. Compared SVD (baseline)/k-means/sparse-dictionary (ISTA, trained to convergence)/hierarchical (reused
polynomial_causal/unembedding_backward_views_v1.hierarchy(), already validated in the repo at 0.94-0.98 live-forward error) on both M0 and M1 at
K=4..128 (v601): SVD wins as it must (Eckart-Young); sparse dictionary nearly MATCHES SVD's fidelity with only ~3% nonzero codes -- the real "simpler
for the same fidelity" result; k-means/hierarchical both clearly worse. Pulling back through Down_17 did NOT change decomposability at these K (curves
track within 0.005-0.02). v602 extends SVD to K=1024+ to find the knee. Found existing prior art mid-stream: `polynomial_causal/unembedding_backward_views_v1.py`
(hierarchy + compile_maps/folded_state, live-forward validated) and several SVD/pullback scripts at layer 17 already exist from a different campaign
(naming convention "_v1" without BQGATE tags in most cases) -- read-only reuse, did not modify or re-run their gated files.
