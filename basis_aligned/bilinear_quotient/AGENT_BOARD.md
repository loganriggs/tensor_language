
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
