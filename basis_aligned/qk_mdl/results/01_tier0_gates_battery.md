# Tier 0: exactness gates + ground-truth MDL battery

## What "folding" means here

For a layer-0 attention head, the query/key inputs are deterministic functions of the
token alone (embedding → norms → projections), so the head's ENTIRE QK computation can be
rewritten in vocab space: per branch, per RoPE frequency band f, two rank-≤2 V×V matrices

    score(t_q@i, t_k@j) = Σ_f cos(ω_f(i−j))·C_f[t_q,t_k] + sin(ω_f(i−j))·S_f[t_q,t_k]

equivalently a pair of per-token factor matrices (q̂, k̂). This is EXACT — the gate
requires reconstructing the live model's patterns from the folded objects at fp64:

| model | pattern err | branch errs | status |
|---|---|---|---|
| attn2-mix10-seed0 | 1.6e-15 | ≤5.7e-14 | PASS |
| attn2-dense-seed0 | 3.6e-15 | ≤1.4e-13 | PASS |
| attn1-seed0 | 2.7e-15 | ≤1.1e-13 | PASS |
| bilin18 (546M) layer-0 factors | ≤1.3e-15 | both branches | PASS |

(One real subtlety found by the gate: the tiny models' `Rotary` builds its trig tables in
fp32, so exact-fp64 trig differs from the deployed models by ~1e-4 — the gate uses the
models' own tables via difference identities; the deviation is documented, not hidden.)

## The ground-truth battery (planted structure, known true DL)

Each codebook must WIN at matched distortion on the plant that matches its prior and LOSE
elsewhere. Selectivity: **PASS 4/4**.

| plant | svd | bicluster | toeplitz | conjunction | true DL | winner |
|---|---|---|---|---|---|---|
| low-rank(8) | **262.4k (=true)** | fail | fail | fail | 262.4k | svd ✓ |
| bicluster(8²) | 229.6k | **12.3k** | fail | 12.4k | 5.1k | bicluster ✓ |
| Toeplitz(6 modes) | 393.6k | fail | **0.4k (=true)** | fail | 0.4k | toeplitz ✓ |
| bicluster ⊙ positive gate | 1246.4k | fail | fail | **38.3k** | 5.6k | conjunction ✓ |

The battery caught the same solver bug twice (random-init biclustering losing its own
plant); spectral initialization fixed it both times — this is what the ground-truth
component is for. Known gaps (logged): bicluster meets ε at 2.4× true DL; conjunction at
7×; the blind conjunction fit assumes a positive positional gate (the real pipeline
decomposes branches separately, so blindness never arises).

SVD pays 33–240× the owner's DL on structured plants — the "computational MDL ≪ spectral
MDL" claim, quantified where truth is known.
