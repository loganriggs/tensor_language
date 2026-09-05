# Hourly circuit and systems review — 2026-09-05 08:08 UTC

## Circuit interpretation targets

The controlling goal is a reusable codebase and organized evidence for hundreds of high-quality, nonduplicated causal circuits. A useful circuit must eventually state what is read, what operation is performed, what is written, and what later computation uses it; split native heads or MLPs when their parts serve different computations and group pieces across modules when downstream behavior treats them as one variable; predict held-out and OOD cases; support executable extraction or sufficiency; allow selective removal, interchange, or editing; compose with other circuits; and remain stable across data splits and gauge choices. The program-level target remains a smaller transparent tensor program that is predictive, composable, manipulable, and literally simpler. Rank, quantization, activation variance, and reconstruction are not substitutes for these circuit properties.

## What changed in the preceding hour

- Eight scientific terminals landed in the rolling 60-minute window: the shared/type opener decomposition, its interaction test, quote-parity control repair, MLP15 mediation, three downstream-head mediation, joint-head mediation, complete downstream-module mediation, and the residual-versus-write-bank factorial.
- The L13H8 semantic-opener write was split exactly into a triplet mean $\mu$ and zero-sum construction-specific difference $\delta$. Both are causal, approximately additive, and not cleanly separated by the hand-chosen common/type closer axes.
- R549 response-localized MLP15 and attention heads failed exact causal restoration, separately and jointly. A complete-module scan also found no substantial single mediator. This prevented another response-cosine or activation-reconstruction loop.
- The grouped $2\times2$ path factorial found that the three-closer semantic effect is carried mainly by the persistent residual route: median residual projection 0.974–1.224, later-write projection -0.221 to +0.037, and interaction magnitude below 0.019.
- CE revealed a distinct compensatory role. Holding later writes native after factor removal caused 0.73–1.35 nat median damage; their natural response repaired roughly 0.35–0.55 nat. Lesson 118 now records that small projection onto a semantic logit axis does not imply CE irrelevance.
- The throughput parser was repaired for midnight rollover and two same-minute completions; focused regression tests now make the hourly count trustworthy.
- The next collision-checked claim is a three-forward exact readout fold of $\mu/\delta$ through final RMS normalization, the fixed unembedding, and the output softcap. Implementation is active and no prior runner is being edited.

## Throughput and time allocation

The authoritative throughput tool reports eight terminals in the preceding hour against the target of six, with median serial time 6.7 minutes. The newest factorial used eight forwards and about six seconds of GPU time. Focused tests took under one second. Scientific design, interpretation, and small one-off dispatch code still dominate; safeguards did not. The last-hour rerun tax is one nonzero execution out of 60 and four minutes. The historical 209-minute numbered-list delay remains the largest old outlier but is not a current bottleneck.

`CIRCUIT_FOCUS: PASS` — every scientific terminal advanced causal factorization, mediation, path composition, a control repair, or circuit throughput. No new rank, quantization, variance, or reconstruction experiment was opened.

`CEREMONY_BUDGET: PASS` — basic screens used three or four focused tests, one dry run, and the shared managed gate. The active successor is deliberately limited to three forwards rather than acquiring OOD, crash, or independent audit suites before it has a signal.

`NOVELTY_LESSON_GATE: PASS` — R549/R551, the pending-opener dossier, the previous direct-read numbered-list experiment, active claims, and canonical lessons were searched. The new fold differs from the old R540 closer-logit shortcut because it fits no direction: it contracts an already causal exact tensor through fixed weights and checks it against the isolated causal path.

## Confounds and direction choice

- Native capability, exact replay, factor liveness, and both algebraic factorial identities passed.
- The weak same-answer control threshold is not treated as selectivity evidence. Current bracket rewrite families test stability only.
- CE is nonlinear in all vocabulary logits, so compensation cannot be inferred from the three closer logits alone; both outputs are reported.
- A direct unembedding lens by itself would be descriptive. The active experiment is only informative because the grouped factorial already isolated the residual causal corner and the weight contraction must reproduce that actual intervention.
- The exact factor accumulates through residual skip multipliers between layer 13 and the final readout. Omitting those multipliers would be a false weight translation; the active implementation carries them explicitly and verifies the final residual identity.

The current route remains higher-information than the alternatives. Splitting the later-write bank is demoted because it does not relay the semantic direction; held-out/OOD promotion is premature until the computation is explicit; another site, rank, or response sweep repeats known failures. The direct readout fold changes computational specification and extraction. It dies if the exact fixed-weight terms do not reconstruct the causal closer-logit change, or if RMS normalization/softcap corrections dominate rather than the folded opener vector.

## Immediate continuation receipt

Claim `bracket.pending_opener.l13h8_mu_delta_direct_readout_fold` is active against prior-art SHA `194f9df0…`. Candidate, three-forward runner, and focused tests are implemented in an isolated successor; validation is running. After review, the next action is to inspect those exact bytes, run the CPU tests/dry-run/gate, commit and enqueue through the managed runner, then preserve either the explicit readout equation or the null before opening another circuit.
