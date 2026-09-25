# Review addendum: fixed-reader identifiability and finite-edit error

This bounded scheduled review began at 13:35 UTC with the 10:34 review older than 175 minutes. The mandatory publication recheck found the substantive [13:36 review](THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-20_1336.md), SHA256 `63d1e6ba236a6b92f0f5980db8c5dc80f68cb2c5f2d6eea9b9e70783cb61ce4a`. This addendum preserves independent CPU work and navigation findings; it does **not** reset that clock. Next deadline remains **2026-09-20 16:36 UTC**. No agents, GPU jobs, queue/timer changes, commits or primary-receipt edits were made.

The new consequence is twofold: a fixed reader cannot reproduce all observed source derivatives even at full capacity under the numerical rank convention below, and replacing a transferred reader by the exact recipient tangent still fails the finite-edit prediction gate. Neither fact proves that a contextual nonlinear circuit cannot exist.

## Executed mathematical consequence

The current subject-number object is `G_c = R_c D_c^T`, with `D_c[23,1152]` the native earlier-write deltas at a nominated pre11 subject/attractor position and `R_c[9,1152]` the local suffix readers for number and eight unrelated contrasts. Number readers are canonicalized to is-minus-are before sharing and restored to answer orientation before signed prediction. `D` includes embedding, attention/MLP0–7, MLP8, MLP10, attention8/9, MLP9 and attention10. Learned residual scalars transport writes; a last-child rounding correction preserves six-parent closure. Native prefixes, counterfactual source construction, x0, first-value cache and full suffix remain external costs.

For each role, stack 24 opposite-panel contexts' 23 source rows into `X[552,1152]`. Stack their matching canonical derivatives into `Y[552,9]`. A static reader B solves `X B ≈ Y`. If `P = V_r V_r^T` projects onto the SVD row space, all `B + (I-P)W` have the same training predictions at the adopted numerical rank. Held source rows Z detect the ambiguity through `Z(I-P)W`. This is an elementary static identifiability result, not an application of a controllable dynamical-system theorem to the transformer.

| CPU control, opened v2 | Subject | Attractor |
|---|---:|---:|
| Rank / nullity | 540 / 612 | 540 / 612 |
| Held source Frobenius norm outside training span | 18% | 19% |
| Best static-reader training relative Frobenius error | 21% | 31% |
| That least-squares reader's held error | 61% | 94% |

Ranks agree for relative cutoffs from 1e-8 through 1e-14. Independent SciPy pivoted QR reproduces the SVD residuals; normalized normal-equation residuals are below 3e-15. Planted fixed-reader training data recover to below 4e-15. A unit null direction changes training contractions only at floating-point noise but changes held contractions. This prevents interpreting a fitted training response as a uniquely identified transferable reader. The existing mean bank was built from full native readers, so this is **not** evidence that its averaging operation was ambiguous or leaked held data. The static least-squares bound is for unweighted canonical derivative Frobenius error, not the causal gates, each individual output, or all possible reader families. It is numerical, not an exact-arithmetic impossibility certificate; retained condition numbers are about 1.4–1.5e7.

For an actual signed edit `h -> h + D^T a`, define `E = F(h)-F(h+D^T a)`, true tangent `T=-R D^T a`, and bank prediction `B=-Rbar D^T a`. Then

    B - E = (B - T) + (T - E).

The first part is reader transfer error, the second finite curvature along the *same candidate amplitudes*. Exact integration gives `E=-integral_0^1 J_F(h+t D^T a)D^T a dt`; local derivative accuracy alone therefore cannot certify finite fidelity. CPU replay of the eight held v2 role/family cells closes this partition. Exact-recipient tangent number errors are **13–42%**, so **0/8** meet the 10% gate even after eliminating reader transfer error. Transfer and curvature may cancel: signed transfer shares reach 1.24. Their norm percentages must not be added. This narrows the next test to contextual readers **and** finite transport, rather than another static mean or unexplained rank sweep. No new native counterfactual was executed.

Executed with `/venv/main/bin/python`, CPU only:

- [Control source](review_reader_identifiability_20260920_1337.py).
- [Control receipt](REVIEW_READER_IDENTIFIABILITY_2026-09-20_1337.json), including primary SHA256 hashes, all eight partitions, effect norms and 0.16s analysis runtime.
- [Independent QR/rank validation](REVIEW_READER_VALIDATION_2026-09-20_1339.json).

## Native algebra and current folded-path boundary

The architecture contract fixes 18 blocks, residual dimension d=1152, nine heads of width k=128, MLP width m=4608, and 50304 output rows. For token position t, `u_l,t=lambda_l,0 r_l,t+lambda_l,1 e_t`, normalized attention writes to `z=u+A`, and `r_next=z+D_l[(L_l z)*(R_l z)]/S(z)+b_l`, where `S(z)=mean(z²)+eps`, `L,R[m,d]`, `D[d,m]`. Final logits are `30 tanh(U RMS(r_final)/30)`; U is untied from embeddings. FP32 epsilon and native rounded RoPE tables are part of the execution contract.

The attention graph projects two queries, two keys and a value; separately normalizes each query/key head; rotates them; contracts query/key index alpha; multiplies both scores and the value; sums causal source positions s<=t and heads; applies O. The pattern is `(q1_t·k1_s)(q2_t·k2_s)/128²`, without softmax. Values are `(1-mu_l)V_l RMS(u_s)+mu_l v0_s`; the first-layer value is shared across layers. This is tied computation, not independent free data. Before normalization the current-value numerator has degree five; the cached branch has four current-state factors and one cached factor. RMS, head norms, biases, residual re-entry and softcap prevent treating the full model as one homogeneous polynomial.

For `z=sum_i z_i`, including earlier attention and MLP sources, an MLP reader contains every ordered `D[(L z_i)*(R z_j)]`: self terms, attention-attention, MLP-MLP, and both attention-MLP orders. Each QK score similarly has `sum_ij z_t,i^T M_ts z_s,j`. Multiplying **both** scores and V requires five source slots, not a separate QK1 attribution. A selected perturbation retains all terms touched by its changed source, including background cross terms, unless explicitly omitted. MLP channel reciprocal rescalings, channel permutations and contracted feature-basis changes are gauges; arbitrary rotations across native norms/RoPE need not be native symmetries.

The regional path deserves its own boundary, rather than being subsumed into the newer subject-number thread. The current [norm-closed MLP8 value mediator](extracted_circuits/city_mlp8_norm_closed_fresh_v1/manifest.json) takes native `z8[1,T,1152]`, upstream delta of the same shape, and token IDs. With `h0=(Lz*Rz)/S(z)` and `h1=(L(z+delta)*R(z+delta))/S(z+delta)`, it explicitly computes

    z9edited = lambda0*(z+delta+D h1+b)+lambda1*x0(tokens)
    mediator = (1-mu)*lambda0*(Wv D)*(h1-h0)/RMS_scale(z9edited).

It retains both ordered source/background crosses, the self quadratic, and changing normalization. Bias cancels from the numerator difference but remains inside the edited norm. The direct value route and other keys/heads/suffix are outside this mediator's claimed output. The folded numerator alone is not its native causal claim.

Literal price: 16,518,531 FP32 stored scalars (66,074,124 bytes), 3,088 token-ID bytes, a derived 589,824-scalar cache occupying 4,718,592 bytes at runtime precision, and 136,866,840 runtime floating bytes in the CPU receipt. At T=32 it still consumes 36,864 native-state plus 36,864 intervention scalars; the formerly external edited RMS scalar is now computed. This closes one dependency at higher storage price, not a compression win. A native MLP alone uses 15,925,248 matrix weights plus 1152 biases and 4608 channel products per token. The source-role reader bank separately costs 20,736 weights, 23 native vectors per site, a roughly `9*23*1152` multiply-accumulate contraction, an LP, and native source generators. None may be treated as free.

## LITERATURE_SEARCH

Actual queries in this review:

1. `site.proceedings.mlr.press sufficient dimension reduction multi index gradient active subspaces Constantine`
2. `site.maths.ox.ac.uk Grasedyck hierarchical singular value decomposition tensors quasi optimal 2010`
3. `site.arxiv.org integrated gradients axiomatic attribution Sundararajan 2017`
4. `site.arxiv.org "active subspaces" "ridge approximation" Constantine 2014`
5. `site.mis.mpg.de "Hierarchical Singular Value Decomposition"`
6. `Willems Rapisarda Markovsky De Moor note persistency excitation 2005 pdf`

Opened primary sources and their actual scope:

- [Sundararajan, Taly and Yan, 2017, full conference PDF](https://proceedings.mlr.press/v70/sundararajan17a/sundararajan17a.pdf) and [landing page](https://proceedings.mlr.press/v70/sundararajan17a.html). Path-integrated gradients supply the exact finite-difference identity above for a differentiable suffix along the fixed-background intervention segment. Numerical quadrature costs additional derivative evaluations; three-node agreement on earlier local removals is not a universal quadrature bound. This best direct match changed this review's plan: execute the tangent/transfer error partition before attributing all failure to static reader context. It supplies no semantic uniqueness or selective-removal guarantee.
- [Grasedyck, primary institute abstract](https://www.mis.mpg.de/publications/preprint-repository/article/2009/issue-27). Applies to a fixed coefficient tensor and a prescribed tree/ranks; the homogeneous two-MLP quartic is such an object, the normalized native suffix is not. Storage scales as O(p n r+p r³), truncation as O(p n r²+p r⁴) in the stated uniform setting. No low-rank assumption or identification guarantee is established here. The full PDF open at `files-www.mis.mpg.de/mpi-typo3/preprints/2009/preprint2009_27.pdf` returned an internal fetch error; no full-proof review is claimed. Does not displace the exact canonical baseline.
- [Constantine, Dow and Wang, primary arXiv abstract](https://arxiv.org/abs/1304.2070). Gradient-informed dimension reduction is relevant to a common residual subspace, but requires distributional error accounting; these finite, counterfactual source rows do not establish it. No theorem from the unread full text is imported, and no new active-subspace run was done.
- [Willems et al., author's primary abstract](https://ftp.esat.kuleuven.be/stadius/markovsky/abstracts/04-101.html). Controllability, time invariance and sufficiently exciting temporal windows are absent from the normalized transformer interface. The full dynamical theorem is not applied. It motivates checking excitation; the static row-space/nullspace identity was derived independently above and costs O(min(n,d)² max(n,d)) for dense SVD. [Southampton repository](https://eprints.soton.ac.uk/262195/) returned HTTP403. No full-paper access or proof inspection is claimed.

## BASELINE_COMPARISON

Use [DECOMPOSITION_BASELINES_2026-09-20.md](DECOMPOSITION_BASELINES_2026-09-20.md), with equal outputs, ports, precision and error measure. This control compares full-capacity static linear readers against the exact **same nine canonical derivatives**, using FP64 and unweighted Frobenius norm. The mean bank and context-dependent native readers have different computation/generator prices; the full-capacity fit above is diagnostic, not adopted. A zero finite-effect predictor has relative target error one by definition and cannot pass the prediction gate. Constant calibration-only, matched-cost random-reader and spectral/Tucker baselines at this new nine-output finite-edit interface remain **missing**.

Existing spectral rank-two results (68 versus 80 coefficients) concern the older four-output/five-source quadratic interface. Native quartic HT rank8 costs656 values at9.35% coefficient error, versus exact symmetric canonical280; rank2 costs116 at36% error. These are legitimate conventional baselines for that homogeneous branch, not comparisons to the current normalized nine-output computation. Full normalized Tucker/HT, alternative-tree and matched-total-cost regional mediator decompositions remain missing. The concurrent review adds balanced/POD source-reader baselines; their errors concern a different projection objective and must not be numerically conflated with the static-reader least-squares residual.

## REDTEAM_POSITIVE

The v3 primary receipt independently scores selectivity10/16 (subject8/8, attractor2/8), prediction0/16. There are 48 distinct texts, 96 nominated sites, **12 template/ordered lexical-pair cells**, each with four number/congruence configurations; 16 scoring bins are not 16 independent data draws. Native capability fails one of eight distinct panel/template/number bins (3/6), repeated in both role reports; 14/16 role-bin passes are not independent capability measurements. No rows were dropped. Binding and prior replay pass; these are prospective inputs relative to the frozen fit, not an unseen corpus guarantee.

Native source vectors and counterfactual prefixes are still required, no dense fallback is promoted, and eight controls do not certify preservation of arbitrary behavior. The new tangent partition explicitly probes the norm/nonlinearity shortcut. Capacity increases from46 scalar coefficients to20,736 readers cannot establish simplicity. For regional composition, [fresh V2](CITY_VALUE_PATH_FRESH_V2_RESULT.json) retains `pred_c=false`; the [opened random-split control](CITY_VALUE_PATH_SPLIT_NULL_V1_RESULT.json) has only a modest global advantage (real/random median0.95), and its reversed subgroup is worse than random. Exact closure does not imply small interaction relative to the smaller piece or robust split specificity.

## REDTEAM_NEGATIVE

Full-capacity SVD plus independent pivoted QR, planted fixed-reader recovery, canonical sign restoration and replayed source contractions test whether rank/coding mistakes caused the fixed-reader failure. They agree. Native first-order readers still fail finite effects, so changing only bank estimation cannot remove all error. This does not rule out context-conditioned readers, nonlinear features or alternate boundary choices. No nonlinear optimization was run, so no convergence claim is needed. The retained condition numbers and numerical-rank dependence prevent overclaiming an exact structural lower bound. Original native failures and receipts remain unchanged.

## Five-property score and organization/efficiency

| Property | Current source-reader candidate | Regional norm-closed mediator |
|---|---|---|
| Simple | Not established; bank and native ports charged | One RMS port closed, storage increases; matched-effect simplicity missing |
| Predicts held-out/OOD | v3 finite prediction fails0/16 | Fresh boundary prediction supported; native/counterfactual inputs remain |
| Extracted | Selector avoids held backprop; native sources/positions external | Isolated declared-boundary replay passes; native z8/delta/suffix remain |
| Selective | Scoped subject positive, whole16-cell gate fails; matched nulls incomplete | Fresh same-boundary norm-matched null16/16, controls scoped to tested readers |
| Composes/reuses | Not established; prior weak-axis increments fail | Independent direct/mediated composition remains unestablished; preserved fresh failures |

Read startup/NEXT prompt, board protocol/tail, current commits, both Git states, circuit/path registries, module records/indexes and relevant late/readout and regional dossiers. Old paths were resolved against this checkout for reads. No goal exists in this scheduled thread; none was created. Current Supervisor bqrunner/bqrunner2/cron were RUNNING; both queue files were empty at inspection. Latest experiment log is13:28:51–13:28:54; subsequent canary13:31:35–13:31:52. Empty queues at one instant are not measured idle waste. The prior hourly receipt reports58.45s across six native experiments; design/review category times remain unavailable.

Reconciliation findings:

- Source-reader aliases resolve to `RESIDUAL_READER_TRANSFER_2026-09-20.md`, frozen role-bank artifact, v3 binding/preregistration and `source_ood_v3_role_bank_v1_result.json`. All three narrative registries linked the reader result, but still described v3 as unrun at initial read; the concurrent13:36 review now owns their update. Do not overwrite it.
- The generated49-package graph registry omits `city_mlp8_norm_closed_v1` and `city_mlp8_norm_closed_fresh_v1`, despite existing manifests/isolated receipts. Its old `four_trait_verified` labels are scoped package assertions, not five-property certification for these new routes. The current manifest linked above is the correct boundary authority.
- The explanation index's latest subject summary remains07:26, and regional03:26 predates RMS-port closure and subsequent composition tests. Startup/NEXT contain September13 handoffs and a stale systemd section; current Supervisor/user instructions and primary receipts override them. These are navigation debt, not orphaned scientific novelty.
- Repeated chronological paragraphs in all three registries and bespoke runners create authoring overhead. The already extracted `frozen_two_site_context.py`, `refined_native_sources.py` and `native_source_observables.py` provide the shared boundary. A broad refactor or automatic graph regeneration during concurrent work would cost more than this bounded review saves; no shared repair is made. This addendum provides the missing cross-links without copying or modifying primary data.

CIRCUIT handoff: keep regional mediator/selective controls as the depth target; close upstream native z8/delta dependencies or resolve preserved composition failures with preregistered same-full-write nulls, without touching primary-owned TYPED_FACE_EXTRACTION_V1. For source readers, retain all roles/capability failures and test finite predictions, not only target retention.

WEIGHT_FOLDING handoff: the newly claimed two-QK adjoint must retain both scores, V, all head/input RMS derivatives, residual mixing and tied first-value routes. This control shows exact local reader folding will still require a finite-transport test. Circuit census selects distributed source/readout terms; the algebra suggests grouping full query-key-value interactions and separating transferred reader error from finite curvature. Do not propose a suffix solely from a backward fold without its forward response census.

The bounded CPU consequence and its validation are complete. Stop here; the substantive review clock belongs to13:36.
