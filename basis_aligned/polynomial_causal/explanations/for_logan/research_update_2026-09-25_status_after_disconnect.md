# Status after the 25 September disconnect: the DCT and circuits lanes are closed and saved; what they found, what is on GitHub, and what is not (25 September 2026, 21:07 UTC)

Claude (Fable). This is a recap for Logan, written after the instance restarted on 25 September at about 20:54 UTC. No run was in flight: the previous session ended cleanly at 19:30 UTC on 24 September, after the held-out circuits check (C3h) landed, was written up, committed and pushed, and the self-paced loop was stopped. Nothing from the 22-24 September work was lost. This note summarises the four chapters since the multi-model ladder, then lists exactly what is saved and what is not.

## 1. The path since the multi-model ladder (22 → 24 September)

All four chapters live in `pr_dct_checks/` (plans in `plans/`, runners in `scripts/`, receipts in `results/`, tables in `RESULTS.md`). Every rung was preregistered with five predictions and scored as written; receipts are replayable. Evaluation sets: AdvBench targets (AJ's distribution, 32-token, mostly EOS padding at the scored positions) and FineWeb 32-token or 64-token rows; each note says which rows were fresh and which were reused.

| chapter | question | answer (tag) | note |
|---|---|---|---|
| PR-regularised DCT checks (22–23 Sep, AJ's handoff) | do the penalised factors find real sparse circuits? | No. The headline PR drop replicates on AdvBench (386 → 3 at 66–70% of the energy) and shrinks on FineWeb (190 → 10–12 at 97–98%); attention carries 52–74% of every factor; the "asymmetric" fit collapses to l = r by symmetry; no factor recurs across seeds on AdvBench at AJ's iteration count and exactly one after convergence; squared-attention models do not concentrate at all (fit + response). | `research_update_2026-09-23_pr_dct_circuit_checks.md` |
| Branch DCT (23 Sep, AJ's l-into-Left / r-into-Right variant) | what does the variant compute? | It reproduces AJ's numbers exactly and has a closed form: the score is Σ_h w_h (a_h·l)(b_h·r) with w a context-averaged downstream Jacobian, so the fitted unit is rank 0 of |w_h|‖a_h‖‖b_h‖ for 24 of 24 factors (21 of 24 on bilin18). One vector-Jacobian product per context replaces the fit. On bilin18 three source units (4212, 468, 4507) recur across seeds because their form is dominant, and their contributions do not transfer to real text (retention 0.28 WikiText, 0.17 FineWeb, vs AJ's 0.82 / 0.84) (fold + response). | same note, §7 |
| Symmetric deterministic DCT on bilin18 (23 Sep, five rungs) | what is the second-order interaction between a block-8 perturbation and the 16 reader directions? | Attention **values** transport the perturbation linearly (block 9 carries 0.43, block 8 0.15; heads 9.8, 9.7, 8.2), and bilinear MLPs make the curvature (freezing all MLPs removes 98%; MLP 8 0.39, MLP 17 0.42). There is no fixed basis (mean form PR 92–408, top-8 directions hold 4% of a held-out context) and no low rank per context (PR ~110). Pattern curvature is ~10% (response, 16 held-out FineWeb contexts). | `research_update_2026-09-23_symmetric_dct.md` |
| Circuits lane (23–24 Sep, C1a–C3h) | for ~100 output directions, what is the circuit, is it weight-space, and does it hold at finite scale? | 100 directions = 16 reader directions + 84 frequent-token unembeddings. Transport: the top-3 value heads are exactly {8.2, 9.7, 9.8} for 43 of 100 (9.8 appears in 81). A **zero-parameter** decomposition (each unit's fixed read pair pulled back through the exact linearised transport, weighted by its exact downstream read) reproduces 70% of every context's form; creation is in blocks 8–12 (block 8 alone 0.25), the ~40 block-17 units are a shared read-out funnel, not creators. A fixed matrix per head fails (R² 0.07): the transport is a tensor contracted with the context. Finite patches match derivative shares at 0.3% and 1% of the residual norm and break at 3% (edit). C3h on 8 fresh FineWeb rows: the heads transfer (0.52 removed; 0.67 with creators), the 30-unit creator lists do not (0.18 in-sample → 0.06); on AdvBench neither acts (0.03–0.05) (edit, fresh rows). | `research_update_2026-09-24_circuits.md` |

**What the arc adds up to.** Across all four chapters the same object keeps appearing: a fixed bilinear form in an MLP's own input space, reached through an attention-value transport that depends on the context. Every method that assumed the structure sits in a fixed residual basis (PR-DCT factors, a fixed head-form dictionary, a fixed per-head matrix) failed on bilin18, and every method that kept the transport exact and the forms in weight space (closed-form branch ranking, the C1c decomposition) worked with no fitted parameters. The context-general parts are the three transport heads and the block-17 funnel; which creator units are active is decided by the context through the transport.

**Falsified rows retained.** PR ≈ 1 does not mean one unit (attention share 52–74%); a 400× PR drop is specific to AJ's prompts; my early "5.7 is not a sink" and "39 stable units are the circuit" readings were both wrong (5.7 is the sink; the stable units are readers); C1a pred_c (blocks 8 & 17 top-2 for only 17%) and C3 pred_b (creators alone 0.18 < 0.3) failed as written.

## 2. What is saved

- **GitHub `loganriggs/tensor_language`, branch `main`, is in sync** with this instance through commit `e7284af69` (24 Sep 19:29 UTC, "Circuits C3h …") plus today's `5f425ce11`. All notes, plans, runners, JSON receipts and logs of the four chapters are in those commits. The board `AGENT_BOARD.md` has an entry per rung (22 Sep 19:08 → 24 Sep 19:29).
- **Committed today (25 Sep)**: the uncommitted residue that had accumulated in the working tree: Codex's parametrisation of the joint-overlap runner and its audit/preflight scripts (backward compatible, `prefix=` and `input_programs=` defaults), Codex's cross-slot and robust-direction receipts and three runner scripts (20–21 Sep), the review-cron outputs of 20–22 Sep (six addenda, four review scripts, five JSON receipts, one three-hourly review), the FineWeb row caches for the Pythia/OPT/SmolLM/Qwen ladder (needed to replay v755–v791), and 300-odd runner logs. Partial receipts (`results/*.partial`) are now ignored; the final receipts were already committed.
- **Committed today, second commit**: the fitted-program tensors under 50 MB from the embedding-forward, attention and ladder chapters (66 files, about 1 GB, stored as ordinary git objects like the repository's other small tensors).

## 3. What is NOT saved, and needs your call

This instance's `/workspace` is not a volume, so anything only on disk is lost on recycle or destroy. Thirty-two fitted-program tensors of 54–766 MB (6.9 GB in total) are on disk only. They are the *fitted programs* behind numbers already in the notes (the per-head attention programs of v704–v720, the MLP program stacks of v650–v653, the Pythia/Qwen/GPT-2 rank-ladder programs of v762–v791, the symmetric-DCT rung-1 forms, and two Codex source-geometry state files). Every number they produced is in a committed JSON receipt, so their loss costs re-fitting time (hours of GPU per file), not results.

| file (under `basis_aligned/bilinear_quotient/circuits/followups/` unless noted) | MB | what it is |
|---|---|---|
| attention_no_native_v705_programs.pt | 766 | all-hybrid attention program, ranks 4/16/64 |
| attention_hybrids_everywhere_v704_programs.pt | 633 | 28 native heads + rank-16 hybrids |
| embedding_forward_mlp_program_all_v653_maps.pt | 425 | fitted MLP-0/1/2 program (256,512,256) |
| attention_whole_program_v716_snapshot.pt | 394 | write-rank-64 program at the validation minimum |
| attention_write_fit_mix_v714_maps.pt, attention_bands08_r64_fit_v715_programs.pt, attention_band68_r64_fit_v713_programs.pt, attention_mixed16_manip_v706_programs.pt | 383 each | attention write-side and band fits |
| pythia1b_rankladder_v787_programs.pt, attention_manip_fix_v707_programs.pt | 379, 371 | Pythia-1b rank ladder; five-heads-native refit |
| embedding_forward_mlp_program_six_v652_maps.pt, qwen3_06b_v788_programs.pt | 212, 203 | six-MLP program; Qwen3-0.6B rank-32 programs |
| pr_dct_checks/results/symmetric_dct_v1_forms.pt | 170 | exact interaction forms, symmetric-DCT rung 1 |
| pythia410m_v762 / _step4000_v770, qwen25_05b_v791, pythia1b_v773 / _step4000_v774, attention_whole_exact_v720_snapshot, embedding_forward_mlp_program_stopped_v651_maps, attention_exact_rank_v718_programs | 104–164 | ladder programs; exact-rank attention programs |
| eleven more files | 54–86 | v618/v620/v629/v650 frames and maps, v752/v764/v782/v786/v790 programs, two Codex `SOURCE_GEOMETRY_STATES_V1*.pt` in `direct_tensor_match/` |

Options, cheapest first: (a) accept regeneration from the committed runners if ever needed; (b) push them through Git LFS, which the repository already uses for 248 files via per-file `.gitattributes` lines, but 6.9 GB will count against the LFS storage and bandwidth quota; (c) upload to a Hugging Face dataset repo, which needs a token this instance does not have. I did not do (b) or (c) without asking. Say which and I will do it.

## 4. Where the lanes stand and what could come next

Both lanes are closed as registered; nothing is queued, and the GPU is idle. The open items each note names:

1. **Third order and finite scale**: the quadratic picture breaks at 3% of the residual norm (C2). A third-order term of the same closed form, or a direct finite-scale decomposition, would say whether the transport × curvature description survives realistic perturbation sizes.
2. **The pattern half of the transport in weight terms**: only the value side was attributed. The kernel + low-rank attention programs from the attention chapter (20 Sep) would make the transport tensor explicit for the three heads, and the block-level pattern curvature (~10%) could then be attributed per head.
3. **A per-distribution creator census**: on AdvBench neither the heads nor the creator units act, while the per-context readers still do. A census on AdvBench rows would show whether the same block-8–12 units create with a different transport, or different units altogether.
4. **The missing ladder control** from 22 September still stands: a QK-normed rotary softmax model trained ≥ 100 B tokens (Qwen3-0.6B falsified the QK-norm reading; a bilinear checkpoint series would settle it directly).

## Receipts

Commits: `e7284af69` (C3h), `5f425ce11` (residue), and the small-tensor commit that follows this note. Board entries: `AGENT_BOARD.md` 2026-09-23T01:00Z through 2026-09-24T19:29Z, plus today's. Receipts per chapter: `pr_dct_checks/results/{e0_ajscale_bilinear,full_*,conv30_bilinear_advbench,branch_dct_*,symmetric_dct_v1–5,circuits_c1a,circuits_c1b,circuits_c1c,circuits_c1c_blocks,circuits_c2,circuits_c3,circuits_c3_heldout}.json` with matching `.log` files. Instance state at writing: `git status` clean apart from the large tensors above and the long-standing working-tree deletion of the old `basis_aligned/bilinear_quotient/AGENT_BOARD.md` (6 Sep copy; the live board is the root one; left untouched).
