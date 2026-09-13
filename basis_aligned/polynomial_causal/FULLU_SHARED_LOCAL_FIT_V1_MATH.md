# Shared global/private full-U fit: both capacities completed

13 September 2026. [Registration](FULLU_SHARED_LOCAL_FIT_V1_PREREGISTRATION.md). Both capacities completed at 05:56:30 UTC, with 2238.30 seconds of measured run time. Numerical checks passed; convergence and the registered 10% squared-gain criterion failed. The earlier detailed sections below describe G64; the completed comparison follows here.

| Shared / groups / private | Compiled bytes | Coefficient error | Matched global error | Squared-error gain |
|---|---:|---:|---:|---:|
| 64 / 32 / 8 | 16,167,936 | 83.61% | 85.85% | 5.15% |
| 128 / 64 / 16 | 34,489,344 | 74.41% | 77.74% | 8.36% |

The larger capacity also stopped all three promoted starts at 240 refinement sweeps, with maximum intrinsic gradients 4.28e-6, 1.65e-6 and 2.88e-6. None met the 1e-6 criterion. Its promoted function cosines are 0.851–0.854. This remains an unfinished optimization comparison, not a global-optimum or identified-factor claim.

The [G128 frozen token audit](SHARED_LOCAL_TOKEN_AUDIT_V1_G128_RESULT.json) replays coefficient error to 3.18e-12. It improves 73.07% of token functions, while 26.93% worsen; median error is 75.65% versus 78.00% globally. The [compiled G128 program](FULLU_SHARED_LOCAL_FIT_V1_G128_PROGRAM.pt) differs from the fit by 2.58e-8 in full coefficient norm. [Complete fit receipt](FULLU_SHARED_LOCAL_FIT_V1_RESULT.json).

Both programs subsequently beat matched global baselines on [historical FineWeb](SHARED_LOCAL_FINEWEB_V1_MATH.md), but fail preservation. On the [regional circuit panel](SHARED_LOCAL_REGIONAL_V1_MATH.md), the larger grouped model is worse than its matched global baseline in all five context groups. Better aggregate fit and lower FineWeb KL did not guarantee preservation of this circuit. No replacement is adopted.

**The smaller model gives a modest, broadly distributed improvement, but misses both the registered gain and local-convergence criteria.** Do not call it a converged discovery of token structure.

The representation has 64 shared functions, 32 groups with eight private functions each, and one group assignment per token. Every token reads 72 coefficients; the combined function bank has width 320. Its compiled tensor payload is 16,167,936 bytes, about 6.97% of native U storage, local to the quadratic numerator. Native bilinear products remain. Original U may still be required on other routes, so this is not a net whole-model storage reduction.

All ten starts were screened; three received 240 refinement sweeps. The best full coefficient error is **83.6069%**, versus **85.8478%** for the optimal global rank78 representation at the same byte budget. This is a **5.1525% squared-error gain**, below the registered 10% bar. The configuration ran for 963.34 seconds.

| Refined start | Final centered squared error | Maximum intrinsic gradient | Local convergence |
|---|---:|---:|---|
| 8 | 0.75372264 | 8.94e-6 | No: sweep limit |
| 5 | 0.75345460 | 1.29e-5 | No: sweep limit |
| 1 | 0.75320587 | 1.75e-5 | No: sweep limit |

All gradients exceed 1e-6. Assignment stability alone does not pass the convergence rule. The promoted predictions have pairwise function cosines 0.842–0.850 despite similar errors; no stable factor identification follows. The unresolved optimization limit remains a plausible explanation for additional recoverable gain. This result does not establish the best achievable grouped fit.

## Independent frozen-program audit

The saved [FP32 program](FULLU_SHARED_LOCAL_FIT_V1_G64_PROGRAM.pt) agrees with the fitted computation to 2.18e-8 in full coefficient norm. Its SHA256 is `26b9aa3a35f80dcc20585e0f168eb05c77b47b00ccda6004410b036c0b63ed32`.

The independent [token audit](SHARED_LOCAL_TOKEN_AUDIT_V1_G64_RESULT.json) replays aggregate error to 4.89e-12 absolute difference. **88.06% of token functions improve** over the matched global baseline; 11.94% worsen. Median token error decreases from 86.29% to 84.37%; the 90th percentile decreases from 89.79% to 88.50%. Thus the aggregate gain is not confined to a few high-energy rows, but most individual functions remain poorly approximated in this coefficient metric.

About 0.48% of tokens have error below 10%; this small subset has not been semantically characterized or tested for duplicate/unused output rows. It is not yet a circuit discovery. Global-plus-private component energy is 1.133 times combined centered prediction energy, showing some cancellation. All 3,219,456 global and 402,432 private code entries are nonzero: sparsity comes from the one-private-group-per-token graph, not sparse coefficients within a selected group.

The [configuration receipt](FULLU_SHARED_LOCAL_FIT_V1_G64_RESULT.json) retains all histories and stops. The frozen audit is an executed check against hidden token-level regressions and compilation errors. It does not resolve the unfinished optimization. Overall registered convergence and 10%-gain predictions can no longer pass for **both** capacities as written; keep that miss even if the larger capacity succeeds.

No text was used to fit these factors or conduct the coefficient audits. Subsequent frozen historical text checks are linked above. Selective removal of identified factors, fresh/OOD prediction and cross-behavior reuse remain unproved; neither the source nor the original fit criteria were changed.

## Accurate-token subset: duplicate audit and its limit

The previously uncharacterized 243 tokens with at most 10% coefficient error all belong to private group5. The [executed audit](SHARED_LOCAL_ACCURATE_TOKENS_V1_G64_RESULT.json) finds only **seven distinct native U rows**: one exact duplicate group contains 237 IDs. Of all 243 IDs, 196 are present in the GPT-2 tokenizer and 47 are outside its vocabulary. Tokenizer membership does not establish actual training frequency or use. One leading quadratic function captures 99.9919% of this subset's coefficient energy. This extends the existing MLP17 dossier's duplicate-row observation; it is not a new semantic circuit.

That correction does **not** explain away the entire grouped-fit improvement. These rows contain only 0.2288% of total target coefficient energy. The global baseline's error is at most each token's target norm for every measured token. Therefore the maximum possible improvement from this subset is its 0.2288% energy, whereas the total reduction is 3.7973% of target energy. The [executed bound](SHARED_LOCAL_DUPLICATE_GAIN_BOUND_V1_RESULT.json) shows that **at least 93.97% of the total improvement comes from outside this subset**. The overall modest gain remains real; semantic interpretation and adequate optimization remain unresolved.

Deduplicating these 243 native U rows exactly would save 271,872 floats before IDs/dispatch costs. That is a useful conventional sharing opportunity, but separate from identifying reusable computational circuits or from the learned grouped approximation.

## Do the private banks just repeat the shared functions?

The [frozen subspace audit](SHARED_LOCAL_SUBSPACE_AUDIT_V1_G64_RESULT.json) compares function spans, so rotating the coordinates inside one bank cannot create or hide agreement. Each private bank places 6.69–12.64% of its orthonormalized function-space energy in the global span. Remove that common span before comparing the genuinely additional directions.

The 32 eight-dimensional private complements jointly have rank 256, with singular condition number 7.22. Median pairwise subspace overlap is 1.53%; the largest is 11.07%. No pair contains a principal direction with cosine at least 0.99; the maximum is 0.824. Thus there is no immediate near-exact merge of these frozen complementary banks. This does not rule out better jointly learned shared parents or approximate sharing under the actual token coefficients.

Deleting every private branch while keeping the existing global coefficients raises full coefficient error from 83.61% to 88.41%. If the global coefficients are instead refitted optimally within the **same frozen global span**, error is 87.83%. This second comparison removes cancellation as the sole explanation for the deletion damage. The private directions add necessary capacity to this fitted representation; neither operation is a native causal removal experiment.

These checks took 4.86 CPU seconds and did not modify the saved program or running larger fit. Sparse group membership remains meaningful storage structure, but the branches have not been identified as reusable semantic circuits.
