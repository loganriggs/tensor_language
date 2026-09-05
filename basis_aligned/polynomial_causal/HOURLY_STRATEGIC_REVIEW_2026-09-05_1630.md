# Hourly circuit-only review — 2026-09-05 16:30 UTC

## Circuit interpretation targets and controlling goal

The controlling goal remains a reusable codebase that can produce hundreds of nonduplicated causal circuits and then use their shared structure to find a smaller transparent tensor program. A successful circuit must eventually specify what information is read, what computation is performed, what is written, and which later computations use it. It should predict held-out or out-of-distribution behavior, support extraction or sufficiency, permit selective removal/interchange without unrelated damage, expose reuse and composition, and remain stable under relevant gauge transformations and dataset changes. Native attention heads and whole MLPs are localization handles, not assumed semantic units.

Literal storage and compute remain eventual simplicity prices. Rank, quantization, activation variance, and activation reconstruction are not circuit evidence and were not opened this hour.

## What changed

The hour narrowed the Task14 grammatical-number path below a native MLP boundary:

$$
\text{subject state}
\longrightarrow \text{MLP8 bilinear response}
\longrightarrow V_{11}^{(h=3)}\hat x_{11,8}
\longrightarrow \text{verb-number logits and CE}.
$$

MLP8 was first identified as the only MLP4--10 writer whose propagated output passed both standalone and conditional causal tests in every fresh-matched cell. Its invariant bilinear response was then split exactly as

$$
z(x_d)-z(x_r)
=\underbrace{(L\Delta x)\odot(Rx_r)+(Lx_r)\odot(R\Delta x)}_{t_{\mathrm{cross}}}
+\underbrace{(L\Delta x)\odot(R\Delta x)}_{t_{\mathrm{quadratic}}}.
$$

The first implementation was preserved as engineering-invalid because float32 regrouping exceeded the frozen exactness bars. The unchanged numerical repair passed every gate and revealed a signed direction switch: plural-to-singular uses a large positive cross response opposed by a negative quadratic response, while singular-to-plural uses a large positive quadratic response opposed by a negative cross response.

That pattern was prospectively tested on reused fronted two-attractor OOD text. The scoped native gate passed 48/48 endpoints. The causal test then passed every frozen prediction: plural-to-singular cross/quadratic recovery was `2.62--2.69 / -1.32---1.26`; singular-to-plural was `-0.46---0.42 / 1.47--1.49`. Background stability, number specificity, and the algebraically dependent selective-removal signs all held. The data claim is deliberately limited to `OOD_TEXT_REUSE_NEW_MLP8_INTERVENTION`: the text and whole-head outcomes were already open, while the MLP8 intervention was prospective.

Canonical Task14 publication advanced through v16, v17, and v18. The v13--v18 focused publisher suite passes 30 tests.

Claude's animacy candidate produced one preserved execution-invalid terminal. A bounded reproduction tool exposed the swallowed exception: the site's pairwise recovery denominator requires positive per-pair separation, which is stricter than the aggregate 0.85 capability-accuracy gate. This is now an explicit shared lesson rather than an opaque rerun target.

## Rolling-hour throughput

From 15:30 through 16:30 UTC, six distinct scientific terminals landed:

1. animacy fast screen: execution-invalid, later diagnosed rather than interpreted;
2. Task14 conditional MLP4--10 screen: valid, MLP8 localized;
3. Task14 MLP8 polarization v1: precision-invalid and preserved;
4. Task14 MLP8 polarization v2: valid within-module causal split;
5. OOD MLP8 scoped native capability: pass; and
6. OOD MLP8 polarization/removal screen: valid intervention generalization.

That is exactly one terminal per 10 serial minutes. Candidate chains took approximately six minutes for MLP8 layer localization, seven minutes for polarization v1, nine to eleven minutes for its numerical repair, and about sixteen minutes for OOD authority design plus its separate capability and causal terminals. Each causal GPU run itself finished in six or seven seconds. The dominant serial cost was scientific authoring and exact finite-precision bookkeeping, not GPU compute, waiting, or a large confirmation suite.

## Confound and falsifier audit

- **Finite precision:** the invalid v1 was not rescued by relaxing a tolerance. The v2 precision boundary, numerical remainder ownership, native endpoint anchors, and sequential residual scaling were frozen before rerun.
- **Post-selection:** the signed direction thresholds were exploratory on fresh-matched data, then frozen before the new OOD MLP8 intervention. The OOD text itself was already open and is labeled accordingly.
- **Gauge:** cross and quadratic responses survive independent Left/Right swapping and reciprocal rescaling. Ordered Left-only and Right-only explanations remain closed.
- **Nonlinear loss:** CE and answer margin are scored separately. Ratios larger than one and negative ratios represent causal overshoot and opposition, not probabilities. Future Möbius interactions in task behavior are causal set-function interactions and must not be presented as algebraic tensor identities.
- **Cross-world hybrids:** the next source-family factorial will explicitly construct hybrid raw residual states. They are interventions, not naturally co-occurring samples, and RMS normalization is applied after each hybrid is formed.
- **Native boundaries:** embedding/skip, attention-write, and prior-MLP-write groups are operational source families, not a claim that they are the final semantic basis.
- **Controls:** same-number different-lemma controls were natively capable and produced at most 5.2% of the number effect on the OOD screen. The scoped capability gate could have failed independently in every direction-by-role cell.

## Gates

`CIRCUIT_FOCUS: PASS.` The hour added a stable upstream writer, an exact within-MLP causal computation, prospective intervention generalization, selective-removal predictions, two canonical revisions, and a diagnosed invalid control path.

`CEREMONY_BUDGET: PASS.` GPU screens remained seconds long. The only rerun repaired a gate-breaking numerical defect without changing the scientific question. The next experiment is one complete factorial using existing machinery, not a new compiler or broad backup suite.

`NOVELTY_LESSON_GATE: PASS.` The Task14 dossier, registry, claims, prior E/A/M writer factorial, equality-task MLP8 term-index failures, OOD QK work, and source-folding precedent were searched. Native product-index selection is not being repeated. Activation reconstruction and rank reduction remain explicitly out of scope.

## Strategy and immediate continuation

The highest-information next question is what MLP8 reads to create the direction-dependent response. The active design decomposes MLP8's raw subject-position input into propagated embedding/skip $E$, attention writes $A_0\ldots A_8$, and earlier MLP writes $M_0\ldots M_7$, with a fixed numerical remainder. All $2^3$ recipient/donor source corners are formed before RMS normalization. Each corner then produces exact MLP8 cross, quadratic, and full responses, which are scored by CE and answer margin through the established L11H3 interface. A complete same-number lexical factorial tests specificity.

This route survives two alternatives:

1. **Downstream-reader weight contraction** is important, but it would name how an already-combined MLP8 vector is read before identifying what information MLP8 composed. It becomes higher priority after the source family is localized.
2. **A separate removal run** is unnecessary now because the same four response corners already supplied the registered removal signs. Repeating it would not add independent evidence.

The source factorial is killed or redirected if the full E+A+M donor response is not live, exact native input endpoints do not close, lexical controls are large, or no source classification survives both CE and margin. If prior-MLP writes dominate, the preregistered next split is MLP0--3 versus MLP4--7. If attention dominates, split attention8 from attention0--7. If the result is interaction-dependent, preserve the Möbius interaction and test the smallest contributing pair instead of forcing a singleton story.

The full 43-condition implementation is active now. It uses four forwards, 1,472 example evaluations, 672 nontrivial interventions, and no gradients or parameter updates.
