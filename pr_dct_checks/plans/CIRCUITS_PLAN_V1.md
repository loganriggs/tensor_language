# Circuits lane, rung C1a: per-direction attribution of transport heads and curvature units for 100 output directions

23 September 2026, 06:50 UTC. Claude (Fable). Registered before running. Logan (23 Sep, 06:30): distributed circuits are fine; find ~100 of them, look for reusable clusters, add unembedding directions, run the finite-scale check, and make the description non-context-bound using the weight tensor network. This rung is the attribution step; C1b fits the weight-space dictionary; C2 is the finite check; C3 the four properties.

**Output directions (100).** The 16 root readers, plus the unembedding rows of the 84 most frequent next tokens in the held-out FineWeb rows (rows 96–191 of `fineweb_n192_skip7000`, positions 1–32), each unit-normalised and applied to the final rms-normalised residual (the tanh softcap is ignored; monotone, and irrelevant at second order up to a scalar).

**Contexts.** 16 held-out FineWeb contexts (rows 96–111, first 32 tokens), as in the symmetric-DCT lane.

**Per (context, direction).** Probe direction v = top-|eigenvalue| eigenvector of the exact interaction form B_{c,u}, by 30 power iterations on Hessian-vector products (forward-over-reverse; agrees with the exact eigenvector to cos 0.9999 on the tiny model with a separated spectrum; on the real forms, whose top eigenvalues are close, the probe is a high-curvature direction rather than the exact top one); the Rayleigh quotient is the direction's interaction energy. Then, all as fractions of u·H[v, v] removed, computed in batches with vmap over freeze masks (batched = sequential to 1e-8): values of each of the 90 heads alone (blocks 8–17), values of all heads of each block, the MLP of each block, all MLPs, the top-20 / top-100 MLP units by hidden mixed derivative along v, and everything (control, on the span without the final norm). Recorded per unit: membership in the top-100 set.

**Aggregation.** Per direction: mean head shares over contexts, the set of heads with mean share ≥ 0.1, the set of units in the top-100 in ≥ 8 of 16 contexts ("stable units"), block shares. Across directions: Jaccard clustering of stable-unit sets and cosine clustering of head-share vectors (hierarchical, threshold reported), counts of units and heads shared by ≥ 25% and ≥ 50% of directions; readers vs token directions compared.

**Controls.** (a) Everything-frozen = 1.000 on the no-final-norm span (≤ 1e-6). (b) Batched freezes equal sequential freezes for 6 random masks per context 0 (≤ 1e-5). (c) Power iteration's Rayleigh quotient within 20% of the top |eigenvalue| from the exact Hessian on 2 (context, direction) pairs (amended from 5% before the run after the smoke test: near-degenerate spectra converge slowly; the probe need not be the exact top direction). (d) CPU smoke test.

**Predictions (scored as written; failures preserved).**
1. pred_a_instrument: (a)–(c) pass.
2. pred_b_transport_reuse: heads 9.8, 9.7, 8.2 are each among the top-3 heads (by mean share) for ≥ 50% of the 100 directions. Prior: unsure (they led for the readers).
3. pred_c_curvature_blocks: for ≥ 70% of directions, MLP blocks 8 and 17 are the two largest single-block removals. Prior: unsure — token directions need not go through block 17's quartic branch.
4. pred_d_unit_reuse: ≥ 100 units are stable for ≥ 25% of the directions. Prior: unsure.
5. pred_e_tokens_like_readers: the median all-MLP removal for token directions ≥ 0.9 (readers: 0.98). Prior: likely.

**Price.** 1,600 (context, direction) pairs × (12 Hessian-vector products + hidden responses + ~115 batched freeze evaluations) ≈ 4–6 s each ≈ 2–3 h on the GPU.
