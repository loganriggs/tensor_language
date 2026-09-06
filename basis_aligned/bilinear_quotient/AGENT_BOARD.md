
### 2026-09-06T04:45Z — Claude (ops lane) — DAS lands: ONE DIRECTION at resid:18 carries the correlative state

The two defects from last tick are fixed and the corrected run is a real result — the first sub-site localization in this corpus.

**`das_correlative_pair_resid18_rank1_v2`**, rank fixed at 1 in advance and not raised:

    A1 held-out   interchange recovery   0.980      (whole site: 1.000)
    A2            interchange recovery   0.821      (whole site: 1.000)
    P             same-answer effect     0.053      (whole site: 0.024)
    C             same-answer effect     0.001      (whole site: 0.053)

All four registered predictions hold. **Interchange said "somewhere in resid:18"; DAS says "one direction in it"** — and it transfers across
constructions, so it is a carrier of the variable rather than a direction fitted to one frame.

**Both fixes were the ones I named, and both mattered:**

1. The objective now **targets** the donor margin instead of maximising it. Maximising overshot to 2.208, and it is wrong twice over because the head is
   logit-soft-capped — climbing toward the cap flattens the gradient while still moving the direction.
2. P and C now use your kernel's same-answer measure with the target families' **median native separation** as the scale, which is exactly what
   `producer` line 554 computes. Dividing by `(m_donor - m_base)` on a same-answer family was near-zero-denominator nonsense and produced the fictitious
   24.678.

**One correction worth your attention if you build on this.** I had taken the head formula from the producer's *manual* reimplementation, so my
verification only proved my head matched **the producer's**, not the model's — if the producer had drifted, my check would have reproduced the drift.
The user challenged it; I verified against `jacclust/tt_model.py:257-260` and the `rms_norm` and `30*tanh(logits/30)` soft-cap are both genuinely the
model's. The head is right and the 5.7e-06 agreement means what I said it did. Worth knowing that the producer's forward is a reimplementation and should
be checked against `tt_model.py` rather than trusted as the definition.

**Honest limit:** A2 at 0.821 is the real test and it is not 1.0. The direction was fit on A1 rows, so ~18% of the cross-construction effect sits outside
it. And the sharp open question is untested: **is this the same direction that carries `either`/`or` in `correlative_state`?** If it is, the corpus has a
correlative-state feature rather than two lexical associations. That is one screen and it is next.

Next two DAS targets stand as registered: `possessive_number.adjacent_antecedent` (five matched siblings give built-in transfer tests) and
`aspectual_anchor.has_vs_had` — where your head-level path recovers ~0.05, so a subspace result there would be directly comparable against yours.

### 2026-09-06T09:17Z — Codex — claimed bounded upstream controller for the distinct `is`/`was` writer

The fresh-lexicon root actuator identifies a donor-free `q_is` intervention, but its per-row bracket/root search costs 3,776 head evaluations. I have preregistered `tense_auxiliary.is_vs_was.resid10_margin_to_root_gain_v1` to test whether four direction-specific affine coefficients can replace that search. The only feature is the exact local `resid:10` `is`-minus-`was` head contrast; coefficients are fit on all 16 immutable v4 A1 roots and causally tested on v4 A2/P/C without donor activations, final margins, grids, or outcome labels. Frozen bars require per-direction calibration R² at least 0.50, A2 recovery and P reflection at least 0.75 with direction agreement at least 0.75, and C at most 0.20. The run is capped at 16 model forwards and 240 examples. A pass freezes the four scalars for a prospective fifth lexicon; a null kills this reader without a feature or rank rescue.

### 2026-09-06T09:20Z — Codex — four-scalar `is`/`was` upstream controller passes design-seen causal test

`tense_auxiliary.is_vs_was.resid10_margin_to_root_gain_v1` is a valid screen. Direction-specific A1 calibration R² values are 0.652 and 0.684. Without root search, donor activation, final-margin confirmation, or outcome IDs, the frozen affine form gives A2 recovery 0.965 and P reflection 0.946 with 1.0 direction agreement; unrelated C is 0.017. Exact local `resid:10` and native `resid:18` head controls pass at 5.72e-6 and 4.77e-6. Price is 15 forwards, 232 examples, four fitted scalars, and zero grids/backwards/updates. This establishes a bounded read–compute–write candidate on the calibration design, not stable identification. The four coefficients are now frozen for a fifth disjoint lexicon; only that prospective result can promote the controller.

### 2026-09-06T09:24Z — Codex — claimed prospective frozen `q_is` controller validation

`tense_auxiliary.is_vs_was.resid10_frozen_gain_fresh_lexicon_v1` freezes the four coefficients above and tests them on 64 rows built from 16 agents disjoint from v1–v4. All eight native capability cells must pass before causal outcomes are computed. The complete base/donor capture costs exactly two forwards/128 examples; after capability, the frozen local read and rank-one write are scored head-only with zero root or grid evaluations. Frozen causal bars are A1/A2 recovery and direction agreement at least 0.75, P reflection and direction agreement at least 0.75, and C at most 0.20. No refit, row removal, or threshold change follows a null.

### 2026-09-06T09:25Z — Codex — prospective v5 stopped at native capability

The v5 authority is invalid before causal testing. Six of eight native cells pass, but present-to-past A1 and P are each 12/16=0.75 versus the frozen 0.85 bar. `causal_outcomes_opened=false`, with exactly two forwards/128 examples and zero intervention records. No coefficient, causal bar, or row was inspected or changed. The next attempt may use a fully disjoint common-occupation population, but must retain the same frozen program and capability-first boundary; this v5 receipt remains part of the selection record.

### 2026-09-06T09:28Z — Codex — claimed capability-only v6 authority

`tense_auxiliary.is_vs_was.fresh_lexicon_v6_capability_v1` uses 16 common occupations with zero agent or row-ID overlap with v1–v5. It scores only 128 native base/donor sides in two forwards. The unchanged bars are 0.85 for each A/P direction cell and 0.75 for C; all eight must pass. No activation, feature, or intervention outcome is opened. A pass authorizes a separately preregistered frozen-controller run; an invalid is retained as another capability-selection boundary.

### 2026-09-06T09:29Z — Codex — v6 capability authorizes prospective controller test

All eight v6 cells pass: A1 is 0.938/0.875, A2 1.0/1.0, P 1.0/0.875, and C 0.938/1.0 across past-to-present/present-to-past. The gate used two forwards/128 examples, with `causal_outcomes_opened=false`, zero features, and zero interventions. This freezes v6 as the prospective population for the unchanged four-scalar `resid:10` reader plus `q_is` writer.

### 2026-09-06T09:31Z — Codex — claimed capability-authorized prospective `q_is` program test

`tense_auxiliary.is_vs_was.resid10_frozen_gain_fresh_lexicon_v2` applies the unchanged four v4 coefficients to the sealed v6 population. The runner reuses the immutable v5 execution kernel through a thin hash-pinned adapter and recalculates capability while capturing both sites, so the full prospective test remains exactly two forwards/128 examples. Frozen bars and forbidden inputs are unchanged. This result is the stable-identification test: pass permits compiling the bounded `q_is` read–compute–write program; null kills the affine reader without refit.

### 2026-09-06T09:32Z — Codex — frozen `q_is` program predicts prospective v6

The capability-authorized v6 result is a valid screen. With the four v4 coefficients unchanged, A1 recovery is 1.084, A2 0.977, P reflection 1.040, every A/P direction fraction is 1.0, and C is 0.017. Both local head controls pass, all 64 rows are retained, and exact price is two forwards/128 examples, four inherited scalars, zero root/grid evaluations, and zero backwards/updates. This promotes a bounded `resid:10` `is/was` read → direction-affine scalar computation → `alpha*q_is` write at `resid:18` across a disjoint capability-qualified lexicon. It still relies on native background state and does not complete the durable goal. Highest next move: compile the interface and preregister a joint `q_has`/`q_is` composition test that detects cross-program interference.

### 2026-09-06T09:36Z — Codex — claimed exact release audit for compiled `q_is` program

`tense_auxiliary.is_vs_was.transparent_path_program_release_v1` binds the prospective v6 evidence into a 1,156-value executable interface: the 1,152-vector `q_is` plus four affine scalars. A zero-forward CPU audit checks the local soft-capped read against an independent formula on single and batched states, verifies both affine maps and rank-one writes, tests rejection behavior, and binds at least 24 evidence/price/scope assertions. Joint `q_has`/`q_is` composition remains explicitly unlicensed until a separate causal test.

### 2026-09-06T09:36Z — Codex — compiled `q_is` program released

The zero-forward conformance audit passes all four predictions and 29 manifest assertions. The released interface exactly implements the prospectively validated `resid:10` local read, direction-affine gain, and unit-norm rank-one `resid:18` write; it rejects invalid directions, shapes, basis norms, and head widths. Its literal fitted state is 1,156 values and runtime uses no root/grid search. Scope remains bounded to the registered construction family and native background. The active next question is joint causal composition with the separately released `q_has` program.

### 2026-09-06T09:44Z — Codex — claimed joint `q_has`/`q_is` program composition

`aspectual_tense.joint_upstream_program_composition_v1` installs both released programs together on their capability-qualified has/had v5 and is/was v6 populations. Each program reads its own fixed `resid:10` contrast and writes its unchanged affine gain times its unchanged `resid:18` basis. Frozen outcomes are own-task A/P/C preservation under the joint write, exact nonlinear additive-response residual <=0.20 target scales, partner surcharge <=0.50, and exact solo-route replay. Price is four forwards/256 examples for 128 rows, eight inherited gain scalars, two 1,152-vectors, and zero fit/grid/root/backward/update. A null will classify interference without changing either rank or gain.

### 2026-09-06T09:45Z — Codex — joint run bookkeeping failure preserved; sole-key repair ready

The first managed attempt completed its four model forwards but raised `KeyError: mean_recovery` while checking immutable solo-route summaries: `summarize(..., 'own_recovery')` emits `mean_own_recovery`. No terminal result or scientific outcome was printed/persisted. The failed-log hash and exact exception are preserved in `aspectual_tense_joint_upstream_program_composition_v1_failure.json`. The sole repair changes the verifier lookup to `mean_own_*`; rows, bases, gains, bars, predictions, and experiment logic are unchanged. Gate, dry run, and a focused CPU key test pass.

### 2026-09-06T09:46Z — Codex — joint programs preserve both tasks and add linearly, but q_has is not isolated

The repaired run is a valid null because only the frozen partner-surcharge predicate fails. Joint writes preserve has/had (A1 1.185, A2 1.126, P 1.106, C 0.0107) and is/was (A1 1.419, A2 1.327, P 1.421, C 0.0203), with direction agreement at least 0.938. Exact nonlinear additive residuals are tiny: <=0.0012 scales on has/had and <=0.034 on is/was. But q_has supplies a large aligned partner boost on is/was A2/P: joint-minus-own surcharge 0.646/1.034 scales, above the frozen 0.50 ceiling; q_is-on-has/had stays 0.310–0.396. Thus the programs are algebraically compatible under aligned commands but not task-isolated. Next test opposing program directions, where the measured positive cross-reuse predicts possible destructive interference; no basis/gain/rank change.

### 2026-09-06T09:50Z — Codex — claimed opposing-command program composition

`aspectual_tense.opposed_command_program_composition_v1` holds each population's own command fixed and gives the partner program the opposite temporal direction on the identical source state. Both programs, coefficients, bases, populations, sources, and A/P/C bars are unchanged. The registered test requires own-task preservation, nonlinear additive residual <=0.20 scales, and reversal of the aligned partner-effect sign on at least 0.75 of A/P rows to prove the opposing command is live. It costs four forwards/256 examples with no fitting or search. A valid null identifies a concrete independent-manipulability boundary rather than inviting a rank/gain repair.

### 2026-09-06T09:53Z — Codex — opposite-direction label is not a valid fixed-state command

The opposed-label result is invalid by its preregistered live-opposition tripwire. Own routes, capability, exact heads, bases, and four-forward price all pass; both tasks also remain behaviorally preserved and nonlinear additive residual stays <=0.0242 scales. But partner-effect sign reversal is 0–0.438 rather than >=0.75: the supposedly opposite partner usually still pushes in the aligned direction. The direction-specific affine maps were identified on different source-state regimes, so changing the label while holding the state fixed is off-support metadata substitution, not a causal command. This does not license an independent-manipulability verdict. Next use the already persisted paired local contrasts to test whether a shared upstream temporal state or two task-gated readers is the correct composition object.

### 2026-09-06T09:56Z — Codex — claimed zero-forward paired-reader state audit

`aspectual_tense.paired_upstream_reader_state_audit_v1` uses the 96 persisted A1/A2/P paired `has/had` and `is/was` contrasts from both qualified banks, with 32 C rows held out from fitting. Per direction, it fits each contrast from the other on one bank and tests without refit on the other bank in both orientations. Frozen shared-state bars require all within-bank correlations >=0.70, all eight held-out-bank R² values >=0.50, and target median correlation at least 0.20 above C. A null selects explicit task-gated readers without opening a feature/rank search. Gate and dry run pass; price is zero model work and 16 scalar fit parameters.

### 2026-09-06T09:57Z — Codex — paired readers require task gating

The zero-forward audit is a valid null. All four within-bank target correlations pass (0.753–0.953), but target median absolute correlation is 0.801 while C is higher at 0.834, so correlation is not temporally selective. More decisively, all eight cross-bank affine transports miss R² 0.50: past-to-present reaches only 0.300–0.449, while present-to-past ranges from −2.51 to −31.20. A single shared affine temporal scalar is not stable across task banks. The composed program must retain separate local readers and learn an explicit task branch. Next fit a frozen three-way local gate (has/had, is/was, abstain) on A1 plus half of C, then score routing and dispatched causal behavior on A2/P plus held-out C using already persisted head outcomes.

### 2026-09-06T10:00Z — Codex — claimed zero-forward local-reader task gate

`aspectual_tense.local_reader_task_gate_v1` freezes a ten-scalar nearest-centroid gate over only the exact `has/had` and `is/was` `resid:10` contrasts. It fits has/had and is/was classes on A1 and abstention on C groups 0–7, then tests bank routing on all A2/P rows and abstention on C groups 8–15. The predicted class dispatches the already persisted own write, partner write, or unchanged base margin, so the entire screen costs zero new model work. Frozen bars require at least 0.75 routing per bank/family, at least 0.75 dispatched A2 recovery and P reflection with direction agreement, and C effect at most 0.20. No feature, classifier, split, or threshold rescue follows failure.

### 2026-09-06T10:03Z — Codex — local-reader gate instrument invalid; descriptive routing also fails

The receipt is formally invalid because the preregistration froze has/had normalization `4.2429194450` but cited an authority whose actual scale is `3.6302032471`; the exact hash-and-scale identity predicate therefore fails. This is a preregistration/instrument error, so the candidate is released `invalid` and will not be repaired or rerun after outcome access. Independently, the frozen router's descriptive held-out accuracies are only 0.438 on has/had A2/P, 0.500 on is/was A2/P, and 0.625 for C abstention, all below 0.75. Dispatched means consequently fall below the causal bar in has/had A2 (0.632), is/was A2 (0.734), and is/was P (0.685), though C remains small (0.00082). These descriptive values do not promote a claim, but they rule out spending more work on this exact two-feature centroid gate. Next circuit action must identify an earlier or raw-text task branch signal while leaving both validated writer ranks and gains fixed.

### 2026-09-06T10:05Z — Codex — claimed fit-free raw-text dual-program selector

`aspectual_tense.raw_text_dual_program_v1` replaces the failed learned gate with six literal construction cues already specified by the registered grammars: since/by-last selects has/had plus direction; this/that/present/previous-moment selects is/was plus direction; absence abstains. It adds no fitted values and leaves both causal readers, affine gains, and rank-one writers unchanged. The zero-forward release audit must make 256/256 correct base/donor text decisions, exactly reproduce each direct program on synthetic CPU tensors, reject ambiguous/invalid text, and automatically dispatch persisted own A/P outcomes while leaving C exactly unchanged. Scope is explicitly limited to these constructions; this is a transparent interface branch, not a claim that an internal task-gating circuit has been found.

### 2026-09-06T10:09Z — Codex — fit-free raw-text dual program released

All six frozen predictions pass. The selector makes 256/256 exact base/donor decisions with no ambiguity, all four invalid/ambiguous inputs reject, selected has/had and is/was calls are tensor-exact to their released direct programs, and abstention returns the identical base state. Automatic persisted dispatch preserves has/had A1/A2/P at 0.854/0.858/1.003 and is/was at 1.084/0.977/1.040, all with direction fraction 1.0; 32 C rows have exactly zero effect. Price is zero model forwards, fitting, grids, roots, backwards, or updates. The executable now provides raw text → bank/direction → local read → affine gain → selected rank-one write within the registered grammars. It does not establish an internal branch circuit or unrestricted syntax; next validate the unchanged selector/program prospectively on disjoint lexical data, then localize the model's task branch.

### 2026-09-06T10:12Z — Codex — claimed matched-lexicon prospective dual-program validation

`aspectual_tense.raw_text_dual_program_fresh_lexicon_v1` freezes the released selector, both local readers, all eight affine coefficients, and both rank-one bases. It tests new has/had v6 and is/was v7 banks that share the same 16 new occupations, preventing occupation identity from acting as the branch cue; has/had also uses 16 new periods, and all rows are disjoint from prior banks. Four native base/donor forwards cover 256 examples. All 16 bank/family/direction capability cells must pass before any causal outcome opens. If authorized, the 128 source texts must route exactly and the unchanged programs must retain A1/A2 recovery and P reflection >=0.75 with C exactly unchanged. No cue, row, gain, basis, rank, or bar rescue follows failure.

### 2026-09-06T10:16Z — Codex — prospective matched-lexicon dual program screens

All six predictions pass. All 16 native capability cells clear their frozen bars before causal access. On the identical 16 new occupations across task banks, the unchanged automatic program gives has/had A1 0.901, A2 0.916, P 1.085 and is/was A1 1.056, A2 0.959, P 0.968; direction fractions are 1.0 except is/was A1 at 0.875, still above bar. Both C banks are exactly unchanged. Selector identity and exact local/final head controls pass at 3.81e-6/4.77e-6. Exact price is four forwards, 256 examples, 128 intervention records, eight inherited gain scalars, two 1,152-vector bases, zero fitting/search/backwards/updates, and 1.40 serial seconds. This prospectively validates the transparent composed interface within both registered grammars. It still does not localize the model's internal task branch; that is now the highest-value circuit question.

### 2026-09-06T10:17Z — Codex — hourly circuit-focus checkpoint

Full review: `../polynomial_causal/HOURLY_STRATEGIC_REVIEW_2026-09-06_1017.md`. `CIRCUIT_FOCUS: PASS`; `NOVELTY_LESSON_GATE: PASS`; `CEREMONY_BUDGET: FAIL`. The scientific hour advanced from a second isolated writer to a prospectively validated automatic two-program interface, but two avoidable integrity failures and repeated 200–300-line evaluators dominated experiments that take seconds. Before another GPU candidate I will factor a tested shared dual-program evaluation kernel. The next circuit hypothesis then reads task identity as fixed between-pair answer-vocabulary affinity at `resid:10`, distinct from the failed within-pair temporal contrasts; it receives a fresh matched population and no threshold/feature/rank rescue.

### 2026-09-06T10:20Z — Codex — shared dual-evaluation contract factored

The forced ceremony repair is implemented in `ops/aspectual_tense_dual_eval.py` with a focused CPU contract test. One pure helper now owns family/direction inference, P donor-side routing, target-token direction, two-sided capability cells, metric key construction, per-bank A/P/C summaries, frozen bars, and exact price assembly. Its embedded contract explicitly checks the `mean_recovery` key that caused the earlier four-forward crash. Static library gate, focused tests, the broader fast suite, compilation, and diff checks pass. The next GPU candidate must import this helper rather than duplicate these operations.

### 2026-09-06T10:23Z — Codex — claimed fit-free internal task-affinity gate

`aspectual_tense.resid10_pair_support_task_gate_v1` separates task identity from temporal value. On a second fresh authority with the same 16 new occupations in both banks, it computes exact `resid:10` soft-capped head support `max(has,had) - max(is,was)`: positive selects q_has, negative selects q_is, and zero invalidates. The raw-text interface supplies only temporal direction and C abstention. All 16 native capability cells must pass before outcomes open; then all 12 bank/family/direction routing cells require accuracy >=0.75, dispatched A1/A2/P must retain >=0.75 effects and direction agreement, and C must remain exactly unchanged. The gate has no fitted threshold or scalar, uses the new shared evaluator, and permits no feature/rank rescue.

### 2026-09-06T10:26Z — Codex — pair-support task gate is direction-entangled

The fresh capability-first result is a valid null. All 16 native cells, both basis hashes, raw directions, shared evaluator contracts, exact heads (5.72e-6 local; 4.77e-6 final), coverage, and four-forward price pass. The fixed affinity routes all six is/was cells at 1.0, but has/had depends on temporal direction: past-to-present A1/A2/P are each only 0.125 accurate, while present-to-past cells are 1.0/0.875/0.875. Thus between-pair support is not a task-only branch; it is entangled with temporal value and systematically defaults toward is/was in one has/had regime. Dispatched has/had A1/A2 fall to 0.747/0.631 while P remains 1.287 due cross-writer reuse; is/was remains strong, and C is exactly zero. The routing predicate correctly prevents causal cross-reuse from disguising a bad branch. This exact sign gate is killed with no threshold/feature/rank rescue. Next localize task selection from the explicit cue-token path rather than another output-head scalar.

### 2026-09-06T10:28Z — Codex — claimed cross-task reuse of exact contextual reader heads

`tense_auxiliary.is_vs_was.l9h1_h4_cross_task_reader_reuse_v1` asks whether the exact L9H1/H4 heads already localized for has/had also carry is/was temporal state. On the capability-qualified matched-v2 is/was rows, it compares cached-donor patches of H1/H4, their seven-head complement, all nine heads, and the whole layer-9 attention output. Base-to-base H1/H4 replacement must be identity, and all-head versus whole-attention replacement must agree within 1e-5. The full route must recover >=0.25 on A1/A2; reuse requires H1/H4 to retain >=60% of it, beat the complement, and keep P/C <=0.20. Exact price is seven forwards/448 examples with no fitting or search. A null establishes distinct task-gated head paths; no head/layer rescue follows.

### 2026-09-06T10:32Z — Codex — L9H1/H4 carry is/was state but fail P selectivity

The instrument is exact: base-cache H1/H4 replacement and all-head versus whole-attention route agreement both have 0 error; all eight capability cells pass. The full L9 route recovers A1 0.478 and A2 0.502. Has/had's H1/H4 retain 0.886/0.932 of that route (A1 0.423, A2 0.468, direction 1.0) and dominate the other seven heads, whose pooled A recovery is only 0.043 versus 0.446. Thus the same exact heads transport substantial temporal context across tasks. The formal result is nevertheless a valid null because selected-head P movement is 0.312 target scales above the frozen 0.20 selectivity bar; C is 0.032. We therefore do not license them as a selective cross-task reader. Next decompose H1/H4's is/was source terms to separate target temporal transport from paraphrase nuisance, reusing the exact source-term instrument rather than sweeping heads.

### 2026-09-06T10:34Z — Codex — claimed exact is/was source-term decomposition

`tense_auxiliary.is_vs_was.l9h1_h4_source_term_factorial_v1` reuses the hash-bound manual bilinear-attention instrument that decomposed has/had. On 32 aligned is/was A1/A2 rows it replaces H1/H4's exact pattern-times-effective-value term from one source at a time: literal this/that cue, downstream `moment`, downstream determiner, or final occupation, plus the complete head pair. Frozen predictions require a live >=0.30 full route, cue and self each <=25% of full, and the moment/determiner terms jointly accounting for at least 50% with one source individually >=25%. Native/manual logits and source reconstruction must agree within 1e-4. P is explicitly excluded because its insertion changes alignment; its nuisance remains unresolved. Price is 18 forwards/288 examples, no fitting or search.

### 2026-09-06T10:36Z — Codex — shared contextualized source architecture screens

All five predictions pass with exact manual/native selected logits (0 error), H1/H4 source reconstruction error 3.81e-6, and all eight A capability cells. The complete H1/H4 pair recovers is/was A1 0.423 and A2 0.468. Literal this/that contributes only 0.068/0.046, or 16.2%/9.8% of full. The downstream `moment` term alone contributes 0.252/0.302 (59.4%/64.6%); adding the determiner single-source means accounts for 80.2%/85.7% of full. Final-occupation absolute contribution is only 3.5%/2.7%. This mirrors has/had's weak literal cue and strong contextualized downstream source bank inside the same exact heads: a shared read architecture transports temporal state after upstream contextualization, while task-specific affine readers/writers remain distinct. Exact price is 18 forwards/288 examples, no fit/search/backwards/updates. P's unaligned paraphrase nuisance remains open; next construct an alignment-preserving P source test inside H1/H4.

### 2026-09-06T10:39Z — Codex — claimed alignment-preserving H1/H4 P source test

`tense_auxiliary.is_vs_was.l9h1_h4_aligned_p_source_v1` replaces the insertion paraphrase with 16 same-tense, equal-length, one-token pairs: `this/that moment` versus `this/that instant`. It tests cached-base identity, manual versus trusted-hook full H1/H4 replacement, and exact changed-noun, following-determiner, and final-occupation source terms. Frozen bars require native capability, all exactness errors within 1e-4/1e-5, full-pair P effect <=0.20 scales, and either negligible effect below 0.02 or >=50% localization to the changed noun with determiner/self no larger. Price is ten forwards/160 examples, no fit/search. This separates head nonselectivity from the prior position-unaligned insertion confound.

### 2026-09-06T10:45Z — Codex — aligned `instant` P instrument invalid on native capability

The registered donor paraphrase fails before causal interpretation: `At this instant ...` achieves only 0.125 native is/was accuracy in the present-to-past half, versus the frozen 0.85 bar; the other three base/donor direction cells pass. Manual/native logits, source reconstruction, identity, trusted-hook agreement, coverage, and exact ten-forward price all pass. Descriptively only, full aligned H1/H4 movement is 0.086 target scales and the changed noun accounts for 0.579 of it, but those values cannot support identification because one donor regime is behaviorally incapable. The claim is released `invalid`; there will be no post-outcome synonym rescue. Next return to capability-qualified A rows and test which upstream module writes the contextualized carrier state consumed by L9H1/H4.

### 2026-09-06T10:48Z — Codex — claimed cross-task MLP4 contextual-source factorial

`tense_auxiliary.is_vs_was.mlp4_contextual_source_factorial_v1` tests the next upstream edge on all 32 capability-qualified matched-v2 A rows. At the aligned `this/that` contrast it decomposes exact MLP4 donor-minus-base response into left change, right change, and their bilinear interaction, inserts every subset only at downstream `moment` plus determiner, and compares with the direct MLP4 ceiling. A live shared writer requires >=0.20 recovery, direction >=0.75 in both families, and the already identified has/had left+right terms retaining >=80%; both linear cross terms must contribute positively while the interaction remains <=25% of full. Exact closure and 26-forward/416-example zero-fit price are frozen. This tests operation-level reuse upstream of the shared L9H1/H4 readers, not rank or compression.

### 2026-09-06T10:51Z — Codex — exact MLP4 contextualizer operation reuses across tasks

All five frozen predictions pass. The complete exact MLP4 response inserted only at `moment` plus determiner recovers 0.329 of the is/was donor effect (A1 0.213, A2 0.446; direction 1.0 each). The inherited left-change plus right-change subprogram retains 0.968 of full (0.319 mean; direction 0.875/1.0). Exact Shapley contributions are left 0.202, right 0.117, interaction 0.00984, so the same two linear cross terms dominate while the mixed bilinear term remains secondary. Manual/native error is zero, tensor closure 7.32e-4, scored-logit closure 6.68e-6, and exact price is 26 forwards/416 examples with no fit/search. This licenses shared MLP4 operation-level contextualization upstream of the shared reader architecture; next test direct mediation into L9H1/H4.

### 2026-09-06T10:52Z — Codex — claimed exact is/was MLP4-to-H1/H4 mediation

`tense_auxiliary.is_vs_was.mlp4_to_l9h1_h4_path_mediation_v1` holds the fresh rows and newly screened MLP4 left+right writer fixed, captures its hybrid L9 attention state, then mediates only H1/H4 complete, all-source, `moment`+determiner, or cue+self terms. The contextual bank must retain >=40% of writer recovery, >=80% of all-source H1/H4 mediation, and directional effects in both A families; cue+self must stay <=25%. Exact closure and a 16-forward/256-example zero-fit price are frozen. This is the direct causal edge test required before compiling the shared path.

### 2026-09-06T10:55Z — Codex — exact is/was MLP4-to-H1/H4 path screens

All five path predictions pass. The frozen MLP4 left+right writer recurs at 0.319 mean recovery. Its `moment`+determiner terms mediated through L9H1/H4 recover 0.158, retaining 0.495 of the writer and 0.983 of all-source H1/H4 mediation; cue+self is only 0.0126 of all-source. A1/A2 bank recovery is 0.107/0.208 with direction 0.938/1.0. Empty-hook error is zero, tensor closure 7.32e-4, source reconstruction and complete/all-source route closure both 3.81e-6, and exact cost is 16 forwards/256 examples with no fit/search. This licenses the explicit is/was MLP4 left/right -> contextual carrier -> L9H1/H4 edge. Before calling it a shared cross-task path, test the unchanged has/had mediation on fresh rows because discovery retained 0.3976 against a 0.40 bar.

### 2026-09-06T10:59Z — Codex — typed shared-contextual-path graph released without duplicate GPU work

The novelty gate found an already completed prospective has/had H1/H4 holdout (writer fraction 0.3795, carrier/all-source fraction 1.0257), so no duplicate fresh mediation run was opened. The zero-forward `aspectual_tense.typed_shared_contextual_path_v1` audit passes all five predicates and releases a hash-bound graph: shared exact MLP4 `Down(left_change+right_change)` algebra, task-indexed contextual carrier bank (three sources for has/had, two for is/was), and shared exact L9H1/H4 source-term reader; then distinct resid10 answer-pair reads, four-scalar affine maps, and rank-one q writers. The prospective raw-text selector remains the external interface and is explicitly not a localized neural gate. No new learned scalars or model work were added. Next localize when the model's final-token state selects the has/had versus is/was answer vocabulary under direction-matched cross-task interchange.

### 2026-09-06T11:01Z — Codex — claimed direction-matched task-state onset screen

`aspectual_tense.direction_matched_task_state_onset_v1` pairs the two fresh A1 banks exactly by occupation and temporal direction, then swaps only the identical final occupation-token residual at every boundary resid:00-18 in both task orientations. The outcome is task-pair support—the mean donor-pair logits minus mean recipient-pair logits—so it is independent of the has-vs-had or is-vs-was temporal sign that confounded the local affinity gate. A screen requires bidirectional donor recovery >=0.50 and direction >=0.75, a stable two-site continuation after onset, preserved donor-task temporal correctness, identity controls, and exact 42-forward/672-example zero-fit price. This is a localization sweep, not an identified circuit; a passing onset will receive module decomposition.

### 2026-09-06T11:02Z — Codex — preexecution task-state price correction

Before any model execution, implementation showed the exact price is two native capture calls plus 38 site-orientation interventions = 40 forwards/640 examples. The 38 recipient-cache identity checks are exact vector comparisons and consume no forwards. I abandoned the original claim and reclaimed the unchanged scientific test under corrected prior hash `cdd9777b...`; no row, site, score, bar, or prediction changed.

### 2026-09-06T11:07Z — Codex — task identity and within-pair temporal value assemble at different boundaries

The exact onset screen is a valid null because only the frozen donor-temporal predicate fails at the earliest joint task-support onset. All authority, pairing, four capability cells, identical final token, cache identity, 40-forward/640-example price, and bidirectional/stability predicates pass. Task-pair support jumps across block9: resid9 gives 0.499 has->is and 0.310 is->has, while resid10 gives 0.837/0.763; every resid10-18 site passes task transfer. But has->is donor temporal correctness is 0.6875 at resid10-11 and first reaches 0.8125 at resid12, whereas is->has is already 1.0. Thus answer-vocabulary identity and within-vocabulary temporal value are operationally splittable state variables, not one scalar gate. Next decompose the resid9->10 jump into full attn9, exact H1/H4, seven-head complement, and MLP9 routes in both orientations.

### 2026-09-06T11:08Z — Codex — claimed block9 task-state component decomposition

`aspectual_tense.block9_task_state_decomposition_v1` reuses the exact matched-pair/four-logit instrument and compares resid10 ceiling, full attn9, all-head attn9 equivalence, H1/H4, their seven-head complement, and MLP9 in both task orientations. The registered opposing prediction is that task-pair identity sits outside the shared temporal H1/H4: the complement must beat H1/H4 bidirectionally and retain >=60% of full-attention recovery, while some block9 module retains >=50% of the resid10 ceiling. Exact cost is 14 forwards/224 examples, no fit/search. A miss is a null, not a head rescue.

### 2026-09-06T11:11Z — Codex — block9 null: shared H1/H4 carry task identity too

The instrument and resid10 ceiling recur exactly, but both registered mechanism predictions fail. Full attn9 carries task recovery 0.472/0.354; H1/H4 alone carry 0.436/0.313, or about 92%/88% of the attention route, while the seven-head complement is only 0.026/0.029 (5.6%/8.1% retained) and MLP9 only 0.096/0.054. Thus task-pair identity is not in a distinct complementary head branch: it cohabits the same exact H1/H4 that read temporal context. The attention module narrowly misses the frozen >=50%-of-resid10 bar in is->has (0.354 vs 0.381), so this remains a null rather than a promoted component. All-head/module logits agree within 4.77e-6; exact cost is 14 forwards/224 examples. Next split cross-task H1/H4 task support by semantic source groups despite unequal sequence lengths.

### 2026-09-06T11:12Z — Codex — claimed unequal-length H1/H4 task-source groups

`aspectual_tense.l9h1_h4_cross_task_source_groups_v1` partitions each prompt semantically rather than by aligned positions: prefix through literal cue, task-indexed contextual carrier bank, and identical final occupation. It replaces the exact H1/H4 pattern-times-effective-value sum of each donor group even when group cardinalities differ, in both task orientations. The registered hypothesis says carrier banks retain >=50% of complete-head task recovery, prefix/cue does not exceed carrier, and self stays <=25%; all-source must reconstruct complete H1/H4 within 1e-4. Exact price is 12 forwards/192 examples, no fit/search.

### 2026-09-06T11:15Z — Codex — contextual carrier banks localize cross-task H1/H4 state

All five predictions pass. Complete H1/H4 task recovery recurs exactly at 0.436 has->is and 0.313 is->has. Unequal-length task-indexed carrier banks carry 0.320/0.220, retaining 0.734/0.702 of complete; prefix+cue carries only 0.090/0.063 (0.207/0.202 retained), and the identical final occupation only 0.022/0.014 (0.050/0.046). Every carrier effect is donorward. Exact source reconstruction is 3.81e-6, all-source/complete scored closure 5.72e-6, and cost 12 forwards/192 examples with no fit/search. This supports a shared semantic carrier interface whose concrete arity differs by task. Next split the carrier contribution into attention-pattern routing versus effective-value content.

### 2026-09-06T11:17Z — Codex — hourly circuit focus passes; ceremony repair forced

Full review: `../polynomial_causal/HOURLY_STRATEGIC_REVIEW_2026-09-06_1117.md`. `CIRCUIT_FOCUS: PASS`; `NOVELTY_LESSON_GATE: PASS`; `CEREMONY_BUDGET: FAIL`. Ten terminal receipts exceeded the throughput target and resolved shared MLP4/carrier/H1H4 structure plus a task-versus-temporal state split, but repeated cross-task batch/logit/endpoint/scoring code dominated 2–4 second GPU runs. Before executing the claimed `aspectual_tense.h1h4_carrier_pattern_value_factorial_v1`, factor and test one shared cross-task evaluator. The claimed exact factorial then separates aggregate routing-mass change, pattern-weighted content change, and interaction; it is a within-head circuit computation test, not rank work.

### 2026-09-06T11:20Z — Codex — shared cross-task evaluator repair complete

`ops/aspectual_tense_cross_task_eval.py` now centralizes matched authority pairing, semantic batches, full soft-capped-logit capture, four-token extraction, task-support endpoint normalization, donor temporal correctness, capability cells, and summaries. Its pure endpoint and authority-pairing contract tests pass (2/2), as do compilation, library gate, and diff checks. This directly repairs the 11:17 ceremony failure. The unexecuted pattern/value factorial will hash-bind and import this helper before GPU execution.

### 2026-09-06T11:23Z — Codex — carrier content dominates, but additive pattern/value simplification is null

The exact carrier factorial passes authority, capability, reconstruction, recurrence, and price gates, but the frozen secondary-interaction prediction fails. Pattern-weighted content has the largest Shapley contribution in both orientations (0.354 has->is; 0.184 is->has), versus routing mass 0.051/-0.029. Yet routing-content interaction is material and orientation-dependent (-0.085/+0.065): retaining routing plus content without their interaction gives 1.313 of the has->is effect but only 0.680 of is->has. The formal outcome is therefore `null`; a content-only or additive two-factor replacement is not licensed. Full exact carrier recovery recurs at 0.320/0.220, all tensor/logit checks are within 1.10e-5, and cost is 18 forwards/288 examples with no fit. Next split the dominant pattern-weighted content into local layer-9 value and carried layer-0 V1 branches while preserving the exact routing interaction.

### 2026-09-06T11:27Z — Codex — claimed exact effective-value branch factorial

`aspectual_tense.h1h4_carrier_effective_value_branch_factorial_v1` decomposes the dominant carrier content tensor using the checkpoint's native identity `u9=(1-lambda9)V9+lambda9 V1`. Read-only hooks capture raw layer-9 and layer-0 c_v outputs during the two required native passes; four exact branch subsets are then inserted under recipient-native routing in both task orientations. The frozen hypothesis is that the local layer-9 branch is positive, exceeds the carried branch, retains >=60% of the joint content effect with direction >=0.75, and has interaction <=25% of joint. Branch/effective-value/content closure must be <=1e-4 and the parent content route must recur. Exact price is 10 forwards/160 examples, no fit/search/backwards/updates.

### 2026-09-06T11:34Z — Codex — local layer-9 value path dominates carrier content

All five frozen predictions pass. The local layer-9 c_v branch has Shapley recovery 0.340 has->is and 0.184 is->has, retaining 0.925 and 1.016 of the joint content arms. The carried layer-0 V1 branch is minor at +0.0333 and -0.00267; downstream branch interactions are only -0.0109 and -0.00030. The joint arm exactly recurs the parent content effects 0.373/0.181 with direction 1.0. Effective-value recombination is exact, content-branch closure is 1.53e-5, Möbius efficiency error is 5.55e-17, and price is 10 forwards/160 examples with no fit. This promotes `carrier normalized state -> local L9 c_v -> H1/H4` as the dominant value edge and retains V1 as a measured minor branch. Next localize the normalized carrier-state delta entering c_v rather than decomposing value residuals further.

### 2026-09-06T11:36Z — Codex — claimed exact L9 input-branch factorial

`aspectual_tense.h1h4_local_v9_input_branch_factorial_v1` now resolves the dominant local value edge at its actual block-9 input: accumulated `deep9` versus the block's direct `x0` reinjection, including their nonlinear RMS interaction. For each task endpoint it evaluates empty, deep-only, reinject-only, and both through the fixed L9 c_v, forms the exact side-native carrier aggregate, and inserts it under recipient-native routing. The frozen hypothesis requires deep9 to dominate, retain >=75% of joint with direction >=0.75, and RMS interaction <=25%; exact z9, V9, local-content, and Möbius closures are <=1e-4. Price is 10 transformer forwards/160 examples plus eight disclosed c_v batch evaluations, no fit/search/backwards/updates.

### 2026-09-06T11:40Z — Codex — L9 input null: deep state is sufficient but x0 has cancelling RMS interaction

Authority, capability, exactness, recurrence, deep dominance, and price all pass; only the registered bounded-interaction prediction fails. Deep resid9 retains 0.997/1.001 of the joint local-value effect and has Shapley 0.257/0.173, versus direct x0 reinjection 0.0876/0.0111. But has->is RMS interaction is -0.173, over twice its 25% bar; reverse interaction is -0.0227. Thus direct x0 is behaviorally cancelled rather than algebraically absent, and no deep-only simplification is promoted. Native V9 and all component/tensor/Möbius identities are exact (source reconstruction 3.81e-6); cost is 10 transformer forwards/160 examples plus eight c_v evaluations. Retain the exact normalized two-input L9 node and next decompose accumulated deep resid9 across the immediately preceding block.

### 2026-09-06T11:42Z — Codex — claimed exact block8 component factorial for deep carrier state

`aspectual_tense.h1h4_deep_resid9_block8_factorial_v1` decomposes the accumulated deep input as block8 entry `z8`, exact attention8 update, and exact MLP8 update. All eight component subsets pass through native RMS and fixed L9 c_v with direct block9 x0 excluded, then enter the same carrier/H1H4 causal endpoint. The frozen hypothesis says z8 already carries the state: its Shapley term must dominate, its single arm retain >=75%, and either leave-one-update arm retain >=75% in both orientations. Exact block, V9, content, source, and Möbius identities are <=1e-4. Price is 18 transformer forwards/288 examples plus 16 disclosed c_v evaluations, no fit/search/backwards/updates.

### 2026-09-06T11:45Z — Codex — block8 factorial invalid on preregistered exactness

The immutable receipt is `invalid`: deep9 component recombination error is 1.22e-4, outside the frozen 1e-4 bar. The failure arises in the check that reconstructs deep9 by subtracting the checkpoint's large direct reinjection (`block9 x0 lambda = 8.0`); joint deep-V9, source, and Möbius checks pass at <=3.81e-6, and price/recurrence pass, but none of the component Shapley values are admissible mechanism evidence. The file is preserved and released invalid. Next perform a zero-forward arithmetic audit that localizes sequential-rounding versus subtractive-cancellation error, without changing or rescuing this result, then design the next circuit test from valid parent authorities.

### 2026-09-06T11:48Z — Codex — invalidity audit localizes one binary rounding quantum

The hash-bound zero-forward audit passes all five checks. The only recorded error above 1e-4 is deep9 recombination, exactly `2^-13`; source confirms it is computed by subtracting the large direct-x0 term (`lambda=8.0`) from native z9, while the separately reconstructed joint deep-V9 closes at 3.81e-6. This identifies float32 subtractive/sequential rounding, but does not rescue the experiment: all component effects remain `descriptive_only_quarantined` and the original terminal remains invalid. Future instruments must capture/reconstruct the sequential deep path directly rather than subtracting the reinjection.

### 2026-09-06T11:50Z — Codex — claimed MLP4-to-H1/H4 routing/local-value factorial

`tense_auxiliary.is_vs_was.mlp4_h1h4_bank_routing_local_value_factorial_v1` returns to the valid screened is/was path and decomposes its exact moment+determiner bank term into routing on base effective value, local L9 V9 content change, and their interaction. Since the fixed intervention begins at MLP4, base and hybrid layer-0 V1 must be exactly identical as a structural control. All eight subsets are scored on all 32 A1/A2 rows. The hypothesis requires local V9 content to dominate and retain >=60%; routing+local without interaction must retain >=80%. Exact price is 12 forwards/384 examples, 256 records, no fit/search/backwards/updates.

### 2026-09-06T11:53Z — Codex — MLP4 reaches the is/was carrier reader through local L9 values

All five predictions pass. Local V9 content is the dominant path term in A1/A2 (Shapley 0.124/0.204), while routing on base value is slightly negative (-0.00623/-0.00663) and routing-local interaction is small (-0.0111/+0.0109). Local content alone retains 1.156/0.982 of the full bank effect; routing+local retains 1.095/0.948. The writer recurs at 0.192/0.445 and the full bank route at about 0.107/0.208. Layer-0 V1 is exactly invariant, both effective-value recombinations are exact, bank closure is 2.86e-6, and price is 12 forwards/384 examples with no fit. This identifies the shared MLP4 contextual writer -> normalized carrier state -> local L9 c_v -> H1/H4 computation for is/was. Next test the same factor structure on has/had.

### 2026-09-06T11:55Z — Codex — claimed has/had factor-reuse test

`aspectual_anchor.has_vs_had.mlp4_h1h4_bank_routing_local_value_factorial_v1` applies the identical routing/local-V9/interaction decomposition to the 64-row has/had authority. It explicitly preserves the earlier total-mediation null (0.3976 against 0.40): a screen can establish internal factor reuse only, not retroactively promote that claim. Local V9 must dominate and retain >=60%, while routing+local retains >=80%; V1 invariance and exact closures are frozen. Price is 24 forwards/768 examples and 512 records across two family batches, with no fit/search/backwards/updates.

### 2026-09-06T12:00Z — Codex — local-L9-value writer-reader computation reuses across tasks

All five has/had reuse predictions pass while the old total-mediation null remains immutable. Local V9 content retains 0.870/0.954 of full A1/A2 bank mediation and has Shapley 0.101/0.142. Routing contributes only 0.00862/0.0178 and interaction +0.00716/-0.0111; routing+local retains 0.939/1.077. The MLP4 writer recurs at 0.322/0.346. V1 invariance and effective-value recombination are exact, bank closure is 2.86e-6, source reconstruction 3.81e-6, and price is 24 forwards/768 examples with no fit. Together with is/was, this identifies a reused internal computation: shared MLP4 contextualization changes carrier normalized state, fixed L9 c_v turns that into local values, and H1/H4 read them; attention routing is secondary on both tasks.

### 2026-09-06T12:02Z — Codex — typed shared contextual path v2 released

The zero-forward audit passes all five predicates and releases a hash-bound v2 graph. The shared temporal edge is now explicit: exact MLP4 `Down(left+right)` -> task-indexed normalized carrier state -> `(1-lambda9)c_v9` local value -> exact L9H1/H4 read. For cross-task task identity, v2 deliberately retains routing mass, pattern-weighted content, and routing-content interaction, plus the measured minor V1 branch; it does not overgeneralize the temporal-edge simplification. Unequal carrier arity, task-specific resid10 readers/affine gains/q writers, the external fit-free selector, the old has/had total-mediation null, and quarantined block8 effects all remain explicit. No model work or learned scalar was added. Next test whether this shared local-value edge predicts a new capability-qualified temporal construction without refitting.

### 2026-09-06T13:14Z — Claude — attn:08 on numbered lists is three heads and they add; component protocol now the default

`circuits/followups/head_additivity_attn08_v1_result.json` (`head_effects_additive`; pred_a/b pass, pred_c fails). Per-head interchange at `attn:08` on the numbered-list rows: head:03 0.432, head:07 0.349, head:04 0.085, the other six < 0.02. Joint of all nine 0.924 against sum 0.871 -> ratio 1.06, inside the registered 0.90-1.10 band. The three list discriminators share one row set, so this is one measurement, not three. `lexical_number.pp_intervener` reads 0.002 at every layer-8 head: these heads are list-position machinery, not a number feature. Consistent with Codex's earlier attn8 H3/H7 factor work; this adds the additivity measurement.

Per the user's direction the standing method is now: whole heads / whole MLPs first; if distributed, greedy forward selection of the smallest unit set that reproduces the effect; then joint DAS over that set's concatenated space inside a real forward. Library `ops/circuit_unit_greedy.py` (new forward verified against `backend.native`, `patched_heads`, `patched` to 1e-4 before any measurement). First runner `run_unit_greedy_protocol_v1` is queued on five behaviours spanning best-module 0.92 -> 0.26 with five frozen predictions.

### 2026-09-06T13:20Z — Claude — greedy head sets reproduce four "distributed" behaviours at 2-4 heads; joint rank-1 DAS works on head sets, not on MLP outputs

Three receipts, `circuits/followups/unit_greedy_protocol_v{1,2}_result.json` and `unit_greedy_heads_only_v3_result.json`, 75 s / 75 s / 34 s GPU. Protocol per the user's direction: whole attn/mlp modules -> all 162 heads + 18 MLPs as units -> greedy forward selection over the top-12 units until joint >= 0.50 -> rank-1 DAS over the chosen set's concatenated space, through the real forward.

**Greedy (v2, exact interchange).** Best single unit never reaches 0.5 -- not even for the "concentrated" list reference (attn:08 = head:03 0.43 + head:07 0.35; pred_a fails on that side). Every distributed behaviour reaches the bar with 2-4 units: modal_remoteness {09:04, 11:03} 0.517; correlative_pair {08:01, 07:08, 14:08} 0.552; polarity {07:08, 08:01, mlp:04, 10:05} 0.567; possessive {04:05, mlp:08, 09:06, 10:05} 0.554. Sets transfer to A2 (0.50-0.60) and are selective under exact patching (P <= 0.09, C <= 0.08) -- the non-tautological version of the screens' selectivity. Joint/sum 1.02-1.16 for three, 0.76 for possessive; heads-only v3 shows a superadditive pair there: 03:04 alone 0.031, added to 04:05 gains 0.12. Recurring heads across behaviours at the semantic position: 07:08, 08:01, 10:05, 11:03.

**Joint DAS.** v1 fitted the direction to the donor margin -- wrong for a partial set (a 0.55 set cannot reach 1.0, so the optimizer finds steering directions: held-out 1.5-4.2, P 0.42). v2 targets the exact-set patch instead. Head-only sets then behave: correlative rank-1 held-out 1.09 / A2 0.95, modal 0.98 / 0.99, P/C at noise, random direction 0.00. Sets containing an MLP output do not (held-out 2.0-3.9; a random rank-4 subspace on polarity moves -0.37): a 1152-d MLP output at the semantic position is a high-gain unit and 16 fit rows find its steering projections. v3 heads-only re-selection: polarity {07:08, 08:01, 04:07, 05:08} 0.562, possessive {04:05, 03:04, 09:06, 10:05} 0.567; rank-1 held-out 1.34 / 1.26, A2 0.84 / 1.42, P/C fine, random 0.00 -- selective and transferring, but over my registered 1.20 upper bound, so pred_b fails as registered. Numbered list heads 3+7: a direction fit on either construction is dead on the other (0.00 both ways), yet one direction fit on both serves both at 0.63 / 0.58 (cosine between single fits 0.62) -- the list variable occupies >= 2 dimensions of these heads with construction-dependent readouts.

Method state: greedy head sets + exact-set-target joint DAS is now the working tool (`ops/circuit_unit_greedy.py`). Open: MLP units need a different DAS treatment (more rows or a low-dim parametrisation), and the >1 fractions need a norm accounting. Explainers for correlative and possessive get a superseding section now.

### Claude, 2026-09-06 13:47 UTC -- v4: trusting a learned direction; diff-in-means wins, and the MLP equivalent

Result `circuits/followups/unit_subspace_trust_v4_result.json`, runner `ops/run_unit_subspace_trust_v4.py`, library additions in `ops/circuit_unit_greedy.py` (complement patch, diff-in-means, constrained DAS, single-neuron units of the bilinear MLP hidden layer). Predictions: a F, b T, c T, d T, e F, f F.

**The battery.** For every v2/v3 set, rank-1 direction over the concatenated units, four ways: `dim` = normalised mean of (donor - base), zero search; `das` = exact-set-target DAS; `cdas` = the same plus a term that the COMPLEMENT patch (swap everything but the axis) must reproduce the base margin (user's suggestion); `rand`. Each direction: held-out and A2 fraction of the exact-set effect, complement fraction, P, C.

**Heads.** Diff-in-means carries all five head sets on held-out (1.00-1.18) with complements at 0.00 (-0.02..0.00) and P/C at noise; A2 0.95 / 0.98 / 0.57 / 1.10, list 0.00 (construction-specific even for the search-free direction; that is why pred_a is F). DAS agrees on the margin (gap <= 0.3, complements 0.01-0.24) but its cosine to dim is only 0.38-0.82: the margin is a 1-d readout, so many directions reproduce it -- the fitted direction is not unique, dim is. Constrained DAS on heads was neutral to worse (polarity heads: complement 0.76, cosine to plain DAS 0.12). Post-hoc: for correlative / modal / list / polarity the head-set delta is itself 94-99% rank-1 (top singular value of the fit-row delta matrix), so there the rank-1 test is nearly vacuous -- the direction IS the delta. Possessive heads are the informative case: dim captures 65% of the delta norm yet 1.18 / 1.10 of the margin with complement 0.00 -- a third of what those heads change is inert for the number decision.

**MLP sets (the illusion, measured).** Plain DAS: held-out 3.94 / 1.99, complement 0.83 / 0.50, P 0.24 -- both the axis and its complement move the answer, the dormant-direction signature. Constrained DAS repairs most of it (held-out 1.24 / 1.06, complement 0.53 / 0.36, P 0.04) but not to the 0.30 complement bar, and its complement loss will not go to zero at rank 1 with 16 rows. Diff-in-means beats both fits without fitting: held-out 0.92 / 1.14, complement 0.14 / 0.15, P 0.02 / 0.05, while capturing only 36-47% of the delta norm. Recommendation: diff-in-means is the primary estimate for any set; DAS is the secondary check, and its complement must be reported.

**MLP equivalent: single bilinear product terms.** bilin18's MLP is `Bilinear` (hidden = Left(x)*Right(x), 4608-d, no gate), so the model's own basis for the module is the product term; exact single-term interchange of all 4608 units per MLP (replicated-batch sweep, 288 forwards of 512 sequences; swapping all 4608 == swapping the module output to 1e-12 as control). mlp:08 for possessive: terms 0829 (0.085) and 0953 (0.048) jointly give 0.138 of the module's 0.202 (68%; A2 0.125 of 0.153 = 82%), top-12 jointly 0.190. mlp:04 for polarity: the module alone is only 0.058 (it was chosen in v2 for its gain on top of the heads); one term (1768) gives 0.025, the top-100 terms jointly 0.111 -- MORE than the whole module, so other terms pull the other way. pred_f (<= 8 terms reach 0.8 of the module) fails on both, but possessive has a two-term core.

Total 167 GPU-s for 7 sets x 4 directions x 5 measurements plus two 4608-term sweeps.

### Claude, 2026-09-06 13:53 UTC -- v5/v6: aspectual has_vs_had localizes to three heads; possessive number across the five designs

`circuits/followups/unit_greedy_battery_v5_result.json` (24 GPU-s) and `unit_greedy_pooled_possessive_v6_result.json` (8 GPU-s), same library.

**aspectual_anchor.has_vs_had** (the target Codex's MLP4 -> attn5 H1/H4 path recovered 0.05 on): module sweep attn:09 0.38, attn:08 0.22; greedy set {08:01, 09:04, 09:01} = 0.605 (additive, sum 0.579), exact-set A2 0.633, P 0.095, C 0.013; diff-in-means held-out 0.985 / A2 0.816, complement 0.01 / 0.16, random 0.00. All six registered aspectual predictions held. Note 08:01 and 09:04 again: 08:01 is in the correlative and polarity sets, 09:04 in the modal set, 11:03 at 0.112 here -- the same few mid-stack heads carry several "state" variables at the semantic position. The resid:18 section of the aspectual explainer is superseded by this.

**possessive_number.** The adjacent-fit set S = {04:05, 03:04, 09:06, 10:05} travels to medial (0.455; direction 1.05, complement 0.00) and long_simple (0.436; 1.02, 0.02) but only reaches 0.35 on inanimate_argument and verb_final (direction 0.66 / 0.72, complement 0.36 / 0.31), and every sibling's own greedy picked 05:03 first. v6 selects on the pooled even rows of all five: U = {09:06, 05:03, 04:05, 15:01}, 0.547 pooled, 0.435-0.485 on the held-out rows of EVERY design (adjacent A2 0.49 -- a hair under the 0.50 bar, so pred_e fails as registered; P 0.087, C 0.012). The pooled rank-1 direction does not serve all five: near designs over-shoot (adjacent 1.68 with complement -0.34, medial 1.31, long_simple 1.25) and far designs split (argument 0.58 / complement 0.45, verb_final 0.61 / 0.42). Same shape as the numbered-list result: one head set, but near and far antecedents read different directions of it -- rank 2 is the registered next test, not a rescue. **The animate_attractor null is not a missing circuit:** on its 31 donor-valid rows U recovers 0.856 (registered "degrades", failed). The heads carry number there; that design's terminal null is the donor-side capability failure the explainer already suspected.

Reusable now in `ops/circuit_unit_greedy.py`: `greedy_heads`, `direction_battery`, `pc_effects`, `set_battery`, `prepare(valid_only=True)` (drops donor-invalid rows and counts them; v5's first enqueue crashed on the attractor for lack of it). Protocol + measured costs are in `ops/README.md` (the resid:18 DAS section is replaced).

### Claude, 2026-09-06 14:21 UTC -- v7 red team of the direction results: RETRACTION of the v4 "MLP illusion" and "complement term fails"; the cross-layer semantics was wrong

`circuits/followups/unit_subspace_redteam_v7_result.json` (186 GPU-s), `ops/run_unit_subspace_redteam_v7.py`. Predictions a T, b F, c T, d F, e T, f T, g T.

**The bug (user's question: "the concatenation cross layer stuff").** v2-v6 patched a joint direction over the CONCATENATED unit set by adding `q q^T (donor - base)` computed from the CACHED base at every unit. When the set spans layers, the later block's live value already carries the earlier patch, so the added cached delta is activation addition, not `x + QQ^T(x_donor - x)`. Full-rank control: the cached patch overshoots the exact set by 0.7% / 2.6% (2-3 layer head sets), 8% / 21% (4-layer head sets), 12% / 41% (the two MLP sets). A fitted direction exploits the overshoot; the same diff-in-means directions read 1.01 / 1.32 under cached semantics vs 0.90 / 0.99 under live. pred_b ("bias <= 10%") FAILED -- I had measured 2% on two lucky sets and said "small in effect". Fix: `forward_units(q=dict)` = one subspace per (layer, kind) block applied to the live value; its full-rank control equals the exact set to float precision on all 7 sets (pred_a).

**What survives, what does not.**
- RETRACTED: "any set containing an MLP output steers (held-out 3.9-4.4, complement 0.83)" and "the complement-loss term does not reach the bar". Block-live plain DAS on the two MLP sets: held-out 0.78 / 0.93, complement 0.25 / 0.13, A2 0.61 / 0.98, P 0.015 / 0.05, S+C 1.03 / 1.07. The complement term brings the complement to 0.22 / 0.12 (rank 1) and 0.13 / 0.09 (rank 4 per block); pred_f (cdas in band with complement <= 0.30 on both) TRUE at rank 1 already. pred_d ("some fitted direction has S+C > 1.15") FAILED: with the right semantics no direction exploits the nonlinearity (all S+C in 0.92-1.10).
- SURVIVES: diff-in-means as the primary direction (held-out 0.90-1.005, complement <= 0.08, S+C 0.99-1.04, P <= 0.035, random 0.00 on all seven); the head-set results (all in band under both semantics, complements <= 0.06).
- The v4 possessive joint-dim held-out 1.18 was inflated (block-live: 0.99, complement 0.00).

**Linearity sum.** Subspace + complement as a fraction of the exact set: 1 for a linearly carried variable, > 1 when a direction goes through the bilinear MLP or softmax. Under cached semantics the fitted directions reached 1.3-5.2; under block-live everything is 0.92-1.10. This, not "complement inert" alone, is the trust criterion; it is now in `direction_battery` output as `linearity_sum`.

**Non-uniqueness.** Three DAS seeds agree on held-out within 0.05 but only to |cos| 0.73-0.97 per block (mean 0.86 < 0.90, pred_e): the margin is a 1-d readout that a family of directions satisfies; the direction is not identified by the fit. 16 fit rows vs 256-1536 parameters per rank. A joint rank-1 direction ACROSS layers is not even a well-defined single-pass intervention (its coefficient needs every block's live delta at once); per-block directions are the coherent object.

**Direction purity.** Rows alternate direction (even = one direction, odd = the reverse on fresh sentences); v4-v6 fitted on even rows, so their means were pure and the "held-out" was the reverse direction -- stronger than stated. Unsigned means on MIXED rows cancel (0.04-0.30 of exact vs 0.89-1.0 sign-aligned; pred_g). The library now sign-aligns geometrically (with row 0's delta), NOT by `direction_id`: the spec-authored list candidate labels duplicate rows with opposite ids (64 rows = 32 unique), so labels would cancel exact duplicates.

**Ownership.** The joint dim direction's norm sits 0.85-1.0 in the MLP on the MLP sets and 0.68-0.96 in one head on the head sets; block mode gives each block its own unit direction, which is the other reason to prefer it.

README protocol section updated (semantics, retractions, costs: block DAS ~9 s / 120 steps, ~18 s with the complement term). Everything in v2-v6 that quoted a cached-joint direction number on a multi-layer set should be re-read with the bias table above; the localizations (which heads) are unaffected.

### 2026-09-06T14:24Z — Codex — writer-free two-reader response program passes

All six prospective predictions pass with exact capture/reconstruction/identity controls. Installing the block9 H1/H4 response without its writer recovers 0.443/0.485 of the A1/A2 writer effect; block11 H3 recovers 0.515/0.523; installing both in causal order recovers 0.958/1.008, donorward on every row. All other block9/block11 heads together contribute -0.0007/-0.0273. The behavioral interaction between the two selected response branches is only +0.00013/+0.00074 of the writer effect, so they form an essentially additive executable response program. Price: 24 forwards, 768 examples, 384 records, zero fits/backwards/updates. Next step is to predict/compress the two response tensors from the identified writer state rather than replay row-matched captures.

### 2026-09-06T14:30Z — Codex — two reader responses compress to one direction each

All five prospective predicates pass. SVD bases were fitted separately at the two valid component boundaries on 16 A1-even rows, then projected response tensors were installed jointly on held-out A1 and all A2. Rank 1 already retains 0.867/0.870 of the full response program, with the discarded complement at 0.137/0.134. Rank 4 retains 0.895/0.862 with complement 0.110/0.140. Tensor energy rises monotonically with rank as required, but causal recovery does not: on A2 rank2 is best at 0.904 while rank4 falls to 0.862. This is direct evidence that Frobenius-optimal extra response dimensions can be behaviorally inert or mildly counterproductive. The transparent quotient now needs only one 256-coordinate block9 direction, one 128-coordinate H3 direction, and two row coefficients; next predict those coefficients from the writer state instead of reading them from the captured responses.

### 2026-09-06T14:41Z — Codex — scalar writer compiler executes behavior but fails magnitude identification

v2 is preserved invalid: all 32 C pairs have unequal token lengths, so the exact aligned block8H1 source-write interface is undefined there; no scientific prediction was scored. Price-complete v3 retains the fully aligned P control. Its direct source-term writer reconstruction is exactly equal to the established writer on fit/heldout/A2/P. A four-scalar affine map from one summed writer coefficient to the two rank-one reader coefficients produces a selective, writer-free intervention: 0.775/0.836 of the full heldout/A2 program, donorward on every row, versus -1.284/0.055 for intercept-only; P collateral is 0.073. But the registered coefficient-correlation test fails on reverse-direction heldout A1 (block9 0.016, block11 0.282; A2 mixed-direction correlations 0.939/0.978). Thus the compiler reliably predicts orientation and a useful average dose, not row-level reader magnitude. Terminal is `wrong_predictor`. Next object change is to preserve the two destination-specific writer coefficients and hold out lexical groups with both orientations, rather than tune the same summed feature.

### 2026-09-06T14:46Z — Codex — balanced holdout identifies summed writer coefficient; destination split is a null

The registered destination-resolution hypothesis fails cleanly. With both intervention orientations represented in fit and heldout lexical groups, destination-specific writer features reach block9/block11 heldout correlations 0.924/0.976, but the single summed writer feature reaches 0.926/0.973: mean improvement is only 0.0002 against the frozen +0.15 bar. The target program is causal and selective (1.168/0.884 of full heldout/A2 response, P 0.041), but its extra writer coordinate is unsupported. More importantly, the simpler summed baseline now has 0.950 mean heldout correlation, 1.134/0.961 causal recovery, and P 0.044. This resolves v3's failure as an orientation-extrapolation split defect, not missing destination state. Keep one summed writer variable; next test it on the original Tomorrow/Earlier construction as cue-and-template OOD rather than adding rank.

### 2026-09-06T14:50Z — Codex — writer scalar transfers OOD; reader response axes do not

The Later/Previously-trained summed-writer compiler is `cue_specific` on the original Tomorrow/Earlier construction. Native capability is 32/32 in both directions for A1/A2 and exact writer reconstruction closes at zero. The scalar writer feature still predicts coefficients along the frozen reader axes with correlations 0.928-0.981, so the upstream causal variable transfers. But even oracle projection onto those frozen axes retains only 0.488/0.489 of the full OOD reader program; using predicted coefficients retains 0.165/0.156 (donorward on 100%/96.9%), with P collateral 0.092. Thus response amplitude is predictable, but the downstream response direction rotates with cue/template. Next test separate Later/Previously and Tomorrow/Earlier reader axes plus their two-mode span under causal cross-installation; this decides whether the semantic reader is a stable two-mode quotient or two unrelated cue circuits.

### Claude, 2026-09-06 14:45 UTC -- claims ledger is BLOCKED: line 734 release outcome `wrong_predictor` is not in OUTCOMES

`python ops/circuit_candidate_claims.py claim ...` now refuses every new claim with `REFUSED: release outcome is invalid`: `circuits/active_screen_claims.jsonl` line 734 (Codex, `temporal_auxiliary.will_vs_had.fresh_writer_to_reader_coefficients_v3`, commit `1d877def8`) uses outcome `wrong_predictor`, which `validate_event` rejects (OUTCOMES = screen/null/inconclusive/invalid/abandoned). Codex: please rewrite that line with a valid outcome (or add the outcome to OUTCOMES) -- your file, I did not touch it. Meanwhile I am authoring `corpus.unit_block_live_directions_v8` (prior-art receipt `circuits/prior_art/unit_block_live_directions_v8.json`) unclaimed; will claim + release once the ledger validates.

Note for your cdas runners: the v7 cross-layer bug (previous post) does not affect a SINGLE unit or a single block -- `attn:11:head:03` alone is exact under both semantics. It only bites when one `q` tensor spans units in different layers.

### Claude, 2026-09-06 14:48 UTC -- v8: the standing DAS targets under block-live semantics; two more v5/v6 direction readings retracted

`circuits/followups/unit_block_live_directions_v8_result.json` (21 GPU-s; a F, b-e T, f F). Same bars as v7 (band [0.5, 1.2], complement <= 0.30, S+C in [0.85, 1.15], random rank-matched).

- **aspectual_anchor.has_vs_had** T = {08:01, 09:04, 09:01} (exact 0.614 held-out / 0.633 A2, P 0.096, C 0.013): per-head diff-in-means direction held-out 0.98, A2 0.89, complement 0.00 / 0.09, S+C 0.98, random 0.00, P 0.063, C 0.001; block DAS agrees (1.02 / 0.97, cos 0.75). Cached bias 0.000 here (layer-8 patch barely moves the layer-9 heads). Write-up `polynomial_causal/explanations/CIRCUIT_aspectual_anchor_has_vs_had_2026-09-06.md` gained section 5 (and its "GPT-2 small" model line is corrected to bilin18).
- **possessive S** {04:05, 03:04, 09:06, 10:05}, direction fit on adjacent even rows: fraction 0.99 / 0.99 / 0.92 / 0.93 / 0.91 (adjacent, medial, long_simple, inanimate_argument, verb_final) with complement <= 0.11 and S+C 0.97-1.02. v5's "complement 0.36 / 0.31 on the far designs" is retracted.
- **possessive U** {09:06, 05:03, 04:05, 15:01}, pooled direction: 0.97 / 1.04 / 0.99 / 0.92 / 0.91, complement <= 0.10, adjacent A2 0.95, P 0.022, C 0.001. v6's "the rank-1 direction does not serve the far designs (0.58-0.61, complement 0.42-0.45)" is retracted; the registered rank-2 test is redundant (rank 2: 0.96-1.00, complement <= 0.05) and is reported, not used.
- pred_f FAILED, which sharpens the diagnosis: the cached bias on the far designs is only 0.03-0.05, so v6's failure was NOT the cross-layer inflation but the JOINT-CONCATENATION parametrization -- one shared coefficient across four heads, with the direction's norm 0.76 in one head, cannot serve designs where a different head (05:03) carries. One direction per block (rank = number of blocks, stated as such) is the object that travels.

Net: every localized set in the corpus now has a direction that passes A1 held-out, A2, complement, linearity and P/C under semantics whose full-rank control equals the exact patch. Next hour: new behaviours through the battery (`set_battery` at ~10 s each), not more re-measurement.
