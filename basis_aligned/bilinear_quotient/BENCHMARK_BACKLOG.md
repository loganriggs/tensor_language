# Standing benchmark backlog (never let a wake idle while any of these is open)

Honest frontier: 34/36 at +2.93 fresh (75% of model work). Open rungs, by size:
STATUS AUDITED 2026-08-30 against runlogs/_completed.txt and the ledger, because
the [QUEUED] markers below were stale and a wake had begun re-deriving them.
Rungs 1-3 are DONE; rung 4 is the first genuinely open one.

1. FRONT TRANSFER TAX (+1.98 of the +2.93 on fresh): replace empirical tables
   with FOLD tables (weights-derived, transfer-free per §283); refit absorbers
   small. -> fold_front.py
   **DONE** -- 2 completions (21:35 exit=1, 21:38 exit=0; the nonzero exit was
   not a failed experiment, per PRE-FLIGHT F). Written up as ledger **§305**,
   "Fold tables strictly dominate: the transfer tax was avoidable all along":
   both bars HELD with margin, window C +1.390 vs +1.481 and fresh +1.602 vs
   +1.926 -- a third of a nat of transfer tax removed.
2. MIDDLE-ATTENTION BAND (~0.9 in-context): head-level hybrid (motif heads
   swapped, diffuse real) -> head_hybrid.py; then absorbers on the attention
   dictionaries (they have none).
   **DONE** -- ledger **§303** (both bars HELD, +2.3644 on window C, 0.27
   BETTER than the all-dictionary band) and **§304** for head_hybrid_fresh
   (certified on never-seen text).
   **THE ABSORBER REMAINDER IS CLOSED, 2026-08-30, §2086 + §2087 + §2088.**
   Three diagnostics totalling **219 seconds** settled it without building the
   sequential matched-context merge that §306 and §307 each lost a run to:
     - **§2086**: the assembly's stream error is a HUMP -- peak 1.7415 at block
       6, attenuating to 0.5925 by block 17. Error is ATTENUATED, not conducted.
     - **§2087**: attention sublayers INJECT it (band 2..9 total +1.8617) while
       the MLPs in the same band REMOVE it (-1.0857). Target narrowed from eight
       rungs to three: a6 +0.8846, a5 +0.8523, a1 +0.8384 carry 2.575 of +1.862.
     - **§2088**: held-out rank-32 linear readability of the injected residual
       from the rung's own input -- **a5 R^2 -0.0645, a6 R^2 +0.0657** (shuffled
       control max 0.0148). **The two largest injectors cannot be absorbed
       linearly.** Only **a1 clears, at R^2 0.5973** -- and a1 is OUTSIDE the
       2-9 band this rung names.
   **VERDICT: closed as unpromising.** Scope of the negative: rank-32 LINEAR read
   from the rung's own input; a nonlinear absorber or a different read site is
   untested, though negative held-out R^2 indicates absent signal rather than
   insufficient capacity.
   **THE `a1` CANDIDATE IS ALSO CLOSED, §2089.** Installing its rank-32 absorber
   (no downstream refit -- the cheap deployment) on held-out rows: block-2
   rel-MSE **rose 9.79%** while block-6 fell **11.37%**, and CE rose **+0.0033**.
   The correction hurts where it acts and helps four blocks later, because `m1`
   was fitted under the un-absorbed `a1` context -- **§307's mismatch, visible in
   the stream rather than only in CE, and not monotone in depth.** The CE cost is
   **0.3% of the assembly's +1.19-nat gap**: neither a win nor a disaster.
   Whether it pays under a FULL matched-context merge is untested and is exactly
   the expensive machinery this rung was closed to avoid.

3. a8 / COUNTING: symbolic count features from raw tokens (deploy-legal)
   -> a8_symbolic.py
   **DONE, and the result was NEGATIVE** -- ledger **§304**: "the
   counting-feature rescue is REFUTED with unusual" force. Closed, not open.
4. DEPLOY GAP (~0.09): two-probe labeling (a10-input probe acc 0.73 for the
   late rungs) -> two_probe_deploy.py
   **RUN 2026-08-30 as `ops/probe_gate3.py`; written up as ledger §2079.**
   §342 set the fork as "deeper reads OR nonlinear features"; §347 took the
   nonlinear half and came out strictly worse, closing with "the AUC ceiling
   (~0.62) needs LATER READ POINTS". That half had never been run. v1's read
   depth was a single constant, so the derivation changed only the depths.
     blk2 (v1 replicate)  AUC 0.6210  eff 3.867x  frac_of_oracle 0.420
     blk5                 AUC 0.6384  eff 5.488x  frac_of_oracle 0.596
     blk9 (= a10 input)   AUC 0.6545  eff 4.574x  frac_of_oracle 0.497
     blk2+9 (two-site)    AUC 0.6382  eff 6.113x  frac_of_oracle 0.664
   **Depth breaks the ceiling** (0.6545 > 0.621) and **the deploy-legal ladder
   is now 1.5x -> 3.87x -> 6.11x against a 9.4x non-deploy-legal oracle.**
   §342's original "recover >= 50% of oracle gain" bar, which v1 FAILED at
   41.9%, is HELD by blk5 (59.6%) and blk2+9 (66.4%).
   **Two sites FAILED on AUC** (0.6382 < 0.6545), reproducing §1365's two-site
   negative in a second program -- **but won on every gating measure.** AUC
   scores the whole ranking; a gate at 17.25% uses only the top of it, so
   **§342's AUC >= 0.75 bar was the wrong bar** and gain-at-fraction is the
   deployable quantity.
   **STILL OPEN on this rung, and it is a control not a new idea:** [2;9] is
   2304-dim against the single sites' 1152, so depth-pair versus width is
   unseparated. Queued as `ops/probe_gate4.py` (all three 2304-dim pairs).
   **Also owed: fresh-window certification (rung 6) before 6.11x is quotable
   as a frontier number.**

5. INDUCTION heads: closed as irreducible-linear, but a *bilinear* stand-in
   (learned low-rank pattern q-k factor + value read) was never tried at the
   head level.
6. Fresh-window certification for every new winner (ledger 22), figure + report
   updates at each frontier move.
   **2026-08-30, §2081: this rung is not merely open, it is OWED RETROSPECTIVELY.**
   Certifying the rung-4 gate on a second document-disjoint fresh window failed
   all three predicates -- and the failure was not confined to the new work:
   §342's published 3.8x configuration scores **-0.093x (worse than random)**
   and the 9.4x oracle reads **2.647x**, because the random-gating denominator
   is **2.595x larger** on the second window. **Every "Nx random" figure in the
   gating arc is a ratio whose denominator was never certified.** Sections are
   scoped, not deleted -- the arithmetic was right on each section's own window.
   **Before any gating number is quoted again it needs a second-window check,
   and the headline should be gain-at-fraction in nats** (blk2+9: +0.0517 FR,
   +0.0331 FR2) which does not divide by a moving baseline. LESSON 113.

Rule (from 2026-08-18 stall): "science arc closed" NEVER implies "benchmark
saturated". A wake with an empty queue must pull from this file first.

7. MODE-CONDITIONED STAND-IN SELECTION (from §§315-322): damage modes are
   component-axis objects. Design: per-position gating -- at positions of
   mode M (causally labeled), keep M's implicated components real (or fit
   M-sliced absorbers); elsewhere use the cheap stand-ins. Register: the
   gated assembly beats the uniform one at matched parameter budget;
   control = random position gating at matched fraction. Requires a
   deployable mode-labeler (probe on the stream -> mode score) -- fit and
   validate that first (deploy gap rules apply).
   **BLOCKED, 2026-08-30, §2090 -- the prerequisite was measured and FAILS.**
   Linear per-mode probes at blocks 2 and 9, each mode on its own held-out AUC:
   **only 1 of 10 modes reaches 0.70** (index 2, 0.7419); **8 of 10 sit below
   0.65**; shuffled-label control 0.5048-0.5156. The rung needs per-mode labels
   at deploy time and nine of ten modes are not linearly readable.
   **This also supplies the number §347 never reported.** §347 regressed these
   same ten modes, published only the derived ANY-MODE gate's AUC (0.551), and
   diagnosed "the any-mode construction amplifies per-mode noise" -- correct, and
   now evidenced: pooling ten probes of which nine are near-chance is
   noise-dominated by construction.
   Depth helps but nowhere near enough (block 9 mean 0.6171 vs block 2 0.6020,
   9 of 10 modes improving). Scope: LINEAR reads; §347 showed quadratic at
   block 2 is worse and §2079 showed deep-linear beats rich-shallow, so a
   nonlinear read AT DEPTH is the only untried route.
   **Asset left standing: mode index 2 at AUC 0.742.** Whether ONE-mode
   conditioning beats uniform stand-ins at matched budget is a smaller and
   different experiment than this rung specifies.

8. PATTERN-SIDE MECHANISM RUNG (from §332): attention-probed leaves need
   mechanism conditions built from motif patterns composed with value
   reads (e.g. "prev-motif head at L, values carrying X -> fires when
   previous token writes X"). Design before running; this is the ladder
   rung for the census majority.

   **PROGRESS 2026-08-30, §2091-§2094 -- precondition met, vocabulary partly
   right, literal composition form REFUTED.**
     - **§2091**: framing confirmed (234/311 leaves attention-probed) but
       MIS-GRAINED -- only 24 are head-probed, 208 are component PCA bands, and
       motifs are head-level objects. (Its own control failed; LESSON 111.)
     - **§2092**: precondition MET -- the census's real probe bands are more
       head-concentrated than arbitrary directions at the same component, for
       **208/208 leaves**, median **+0.1590** above each leaf's own baseline.
     - **§2093**: vocabulary PARTLY right -- motif-named heads over-represented
       at **1.1449x** (fails a registered 1.20x effect-size bar; clears the
       permutation null at **z = 4.46, p = 0.00005**). **PREV carries it: 215 of
       416 leaf top-2 slots**, against ind's 13. a6 and a9 run backwards.
     - **§2094**: the literal composition form **FAILS**. On the 31 leaves whose
       both top-2 heads are prev, a per-token-id membership predictor scores
       **0.5086 held-out AUC for the previous token and 0.5130 for the current**
       -- chance, with **0 of 31** leaves reaching 0.60 on either. "Fires when
       the previous token writes X" has no purchase. Consistent with §348: every
       census circuit is two-signed, and a unigram predictor averages over the
       contrast that defines the leaf.
     - **§2095**: the DIRECTIONAL reading fails too. A 1152-dim ridge from the
       previous token's EMBEDDING scores median held-out AUC **0.5052** -- BELOW
       §2094's token-identity 0.5086 -- against a bar of 0.5586, with a
       shuffled-label capacity control at **0.5006** confirming the ridge is not
       overfitting but has nothing to find. Current-token embedding 0.5145 again
       edges out previous, as in §2094.
   **COMPOSITION REFUTED IN BOTH FORMS §332's WORDING SUPPORTS** (literal token
   identity, and X-as-a-direction). **Rung 8's proposed mechanism language does
   not work.**
   **What is NOT refuted:** that these leaves have a mechanism. Every feature
   tested is a LOCAL TOKEN feature. A mechanism over the head's REALISED
   ATTENTION PATTERN -- which positions it attended, not what token sat at t-1 --
   is untested and is a materially different mechanism language from §332's.
   That is a design decision about the rung's direction, not another
   measurement, so it is recorded here rather than started.
   **RUN 2026-08-30, §2096, `ops/realised_attention_composition.py` — CLOSED.**
   The realised-pattern ridge (24 features: signed mass by offset bin, total
   mass, |mass|, entropy, both top-2 heads) scores median held-out AUC
   **0.5409** against the 0.5586 bar: FAILED, but the first feature in the
   arc to move at all (all 31 leaves above §2095's 0.5052; 0/31 at 0.60).
   Shuffled control 0.4926 HELD; specificity to the top-2 heads HELD at
   67.7% but the median advantage over same-layer control heads is only
   +0.0062 (p=0.035). Signal is mostly total/offset-1 MASS (§1108's degree of
   freedom), not pattern shape. **§332's composition is refuted in all three
   forms its wording supports. Rung 8 is closed.** Untested and NOT a form of
   §332: nonlinear/context-aggregating pattern reads, other layers.

STATUS AUDIT 2026-08-30 (§2096): rungs 4 and 6 above are STALE — the rung-4
control `ops/probe_gate4.py` ran (§2080: the two-site gain is WIDTH), and
rung-6 certification ran and FAILED (§2081–§2085: the ~6x gate collapses on
a second window; the "Nx random" ratio is seed- and window-dependent; gating
tracks the TEXT's difficulty, not the assembly). Rung 5 was closed at §343
(`head_lowrank.py`: induction heads are intrinsically high-rank in weight
space; the learned low-rank bilinear stand-in is measured as unpromising).
Rung 7 is BLOCKED (§2090). Rung 8 is CLOSED (§2096). **No open rung remains
in this file.** The next wake must open a NEW rung from the frontier
(34/36 at +2.93 fresh), not re-run a closed one; candidates are recorded in
AGENT_BOARD (2026-08-30 §2096 entry).

9. NONLINEAR / CONTEXT-AGGREGATING READ OF THE REALISED PATTERN (opened
   2026-08-30 from §2096's own scope line; the only untested branch of the
   head-grain arc). Same 31 both-prev leaves, same split, same 0.5586 bar.
   Arms: L1 (§2096 replicate, must reproduce 0.5409 within 0.005), NL
   (random-Fourier ridge, 512 feats, fixed seed, fit-half bandwidth), CTX
   (linear on t..t-3, 96 feats), NLC (both). Per-arm shuffled-label control
   <= 0.52; winner's same-layer control-head comparison reported.
   -> ops/pattern_read_nonlinear.py  [QUEUED 2026-08-30 16:50]
   If it FAILS the head-grain description of the a3/a4 leaves is exhausted at
   ~0.54 and the next rung must come from the m16 target or the frontier's
   fresh-window re-certification (AGENT_BOARD §2096 entry, candidates 2-3).
   **RUN 2026-08-30, §2097 — CLOSED, FAILED.** Best richer arm NLC 0.5493 vs
   bar 0.5586; L1 replicates 0.5409 exactly; all shuffled controls 0.49–0.50;
   0/31 leaves at 0.60 under any arm. Head-grain signal at these leaves is
   exhausted at ~0.54. Only untested shape: a joint pattern×value read (bar
   would be > 0.5493) and other layers.
   ALSO SETTLED WITHOUT A RUN (§2097): the frontier's fresh-window stability.
   §2085's artifact has the assembly excess on eight document-disjoint
   windows: 2.64–2.97, mean 2.81, sd 0.13; +2.93 is inside the spread.

10. THE m16 TARGET (opened 2026-08-30; shared with the polynomial_causal
    lane, where m16→* is the worst owner pair, NRMSE 2.5–3.5, in every
    panel of the causal-response validation table, and where the m16
    source-owner deletion response has unconditional NRMSE 2.7 on held-out
    documents). On this lane m16 is the frontier's second-largest late
    deficit (§8180: mlp16 0.354 nats, mlp17 0.815) and its cross-circuit
    rank-1 ablation concentration is NEGATIVE (−0.54, §10409): its removal
    effects do not concentrate on one direction. Design before running:
    (a) what m16 writes that later readers (m17, lm_head) distinguish —
    the controllability/observability quotient the other lane now names
    as primary; (b) whether the frontier's m16 stand-in (linear read, .81)
    fails on the same positions where the causal-response program fails.
    DESIGN MEASUREMENT 2026-08-30 (in-sample, all 229 training docs, NOT
    registered): m16 source rows RMS 0.4923 vs 0.1803 overall; 6-row
    unfolding top-1 energy share 0.6565, top-2 0.8775; null random 6-row
    blocks median 0.63 / p95 0.87. mlp16's rank-1 OUTPUT core does NOT
    make its deletion-response block rank-1; two source families visible.
    REGISTERED: does the rank-2 subspace transfer across a prospective
    document split? -> ops/m16_response_block_split.py  [QUEUED]
    pred_a transfer >= 0.78 (in-sample 0.8775 - 0.10); pred_b beats the
    200-draw null p95; pred_c the two-family partition agrees on both
    halves.

11. WHICH FRONT PIECE DOES attn5 AMPLIFY? (opened 2026-08-30 from the
    price-cliff results and §2101). The certified empirical arm keeps all
    attention real yet carries rel-MSE 0.51 at block 1 → 1.74 at block 6;
    error is cheap before attn5's write (0.075 nat/half-norm) and 23x more
    expensive after it; the arm's own error is anti-random (§2101: 2.4x a
    matched random error at block 6, 5.6x at block 5), with its cost in a
    ~600-dim observable subspace. -> ops/front_piece_amplification.py [RUNNING]
    Eight matched-context arms (cfgE + one front piece real at a time: m0,
    a1v, m1, m2, m3, c4, c5). pred_a m0 is the largest block-6 lever; pred_b
    attn5 amplifies (delta b6 >= 2x delta at the block after the piece);
    pred_c rho(delta b6, delta CE) >= 0.7; pred_d cfgE reproduces 1.7415.
    RUN 2026-08-30, §2102: (d) HELD exactly; (b) HELD at 8.6x — mlp4 is the
    largest block-6 lever and attn5 amplifies it; (a) FAILED — m0 is fifth;
    (c) FAILED at rho 0.07 — block-6 rel-MSE does not price CE; m3-real
    worsens b6 and helps CE, c4-real fixes b6 most and helps CE least; a1v-
    real hurts both. Exactness is priced by DIRECTION. Next: if
    ops/observable_correction.py shows the observable third suffices, refit
    the front under a block-6-Gramian-weighted loss (rung 12).

12. METRIC-WEIGHTED FRONT (opened 2026-08-30 from §2103: oracle-correcting the
    observable third of the block-6 stream recovers 94.5% of what full
    correction recovers). Refit the front's CONSTRAINED pieces under site-local
    first-order observability metrics at EQUAL stored values (m0/m2/m3 residual
    bases; mlp4/mlp5 unit selection); random-metric control.
    -> ops/metric_front_refit.py  RUN §2104: +0.125 nat at equal price (bar 0.15
    FAILED by 0.025); random-metric +0.017 (HELD); block-6 rel-MSE 1.74 -> 1.45
    but observable share ROSE (pred_b FAILED). Real, mechanism not as registered.
    12b -> ops/metric_front_refit_split.py  RUN §2105: units-only +0.124, bases-
    only -0.009 — the whole gain is mlp4/mlp5 UNIT SELECTION; additive; control
    reproduces §2104 exactly. All four HELD.

13. SECOND WINDOW + c6-c9 EXTENSION -> ops/metric_units_certify.py  RUN §2106:
    (a) HELD +0.075 on FW rows 0:120 (never used by any fit/eval); (c) HELD
    0.124 reproduced; (b) FAILED — metric selection at c6-c9 lowers block-9
    rel-MSE and RAISES CE. CERTIFIED at mlp4/mlp5 only.

14. PRICE THE GAIN: K SWEEP -> ops/metric_units_ksweep.py  RUN §2107: metric-
    1152 matches norm-2304 on both windows (HALF the stored values at equal
    CE); gain 0.156/0.124/0 at K=1152/2304/4608; pred_a FAILED as written
    because K=4608 is the whole layer (instrument fact missed at registration).

15. MECHANISM -> ops/metric_units_mechanism.py  RUN §2108: swapped-in vs
    swapped-out units have 0.685 vs 0.670 of Down energy in the r90 span (every
    unit ~0.68) — refuted AS STATED. 15b -> ops/metric_units_mechanism2.py  RUN
    §2109: top-8 loss-gradient directions discriminate at 2.4-2.5x (HELD);
    top-r50 1.3x (FAILED at 1.5x). The mechanism is the Gramian's very top.

16. THE 8-DIRECTION SELECTOR -> ops/metric_units_top8.py  RUN §2110: top-8
    selector gives +0.1285/+0.0648 (reproduces the full metric; a/b/d HELD) and
    BEATS top-r50 (+0.109; pred_c FAILED). Eight directions per site ARE the
    metric (8x1152 stored values).

17. NAME THE EIGHT -> ops/name_the_eight.py  RUN §2111: dominant newline-vs-
    place-name direction (18% of block-5 gradient energy), five markup/
    punctuation structure directions, two place-name directions; attn5 head 7
    reads them at 3.7x via q2 (HELD); lm_head overlap 0.10 (HELD); block-5/6
    eights overlap 0.472 (FAILED by 0.028).

18. IS attn5 HEAD 7 THE AMPLIFIER? -> ops/head7_amplifier.py  RUN §2112: NO —
    h7 is the §1089 sink/bias head (zeroing it costs the REAL model 0.91 nat,
    pred_c FAILED 9x); zeroing on both sides trims block-6 error 8% and triples
    block 7 (pred_a FAILED); random-head controls HELD.

19. DECOMPOSE THE INJECTED ERROR BY HEAD -> ops/attn5_error_by_head.py  RUN
    §2113: h7 carries 74% of attn5's injected error (97% of real output; pred_a
    HELD) but it lies OFF the eight at a random direction's rate (0.0071;
    pred_b FAILED); attn5 injects more than mlp5 (HELD).

20. ENERGY vs PRICE AT HEAD GRAIN -> ops/head_energy_vs_price.py  RUN §2114:
    ALL FOUR HELD — h7-only oracle correction removes 71% of block-6 stream
    error and 19% of CE; the other eight heads 40% / 85%; all nine recover 0.79
    of the 1.50-nat gap. Rel-MSE is the wrong currency, measured.

21. WHICH OF THE EIGHT HEADS CARRY THE 85%? -> ops/attn5_head_price_ladder.py
    RUN §2115: four heads (5,0,1,6) at 0.15-0.18 each, additive to 0.82 (pred_c
    HELD); h5 top as predicted from the eight (pred_b HELD); not concentrated
    in two (pred_a FAILED).

22. FRESH-PILE CERTIFICATION of the metric-unit gain: cfgE vs metric-units
    (c4,c5, K=2304) on the eight document-disjoint pile-10k windows of §2083
    (probe_gate7's builder). -> ops/metric_units_fresh8.py [QUEUED]
    pred_a gain > 0 on >= 7 of 8 windows; pred_b median gain >= 0.04 (half of
    window-2's 0.075); pred_c gain sd across windows <= 0.5 x gap sd; pred_d
    window-1 gain reproduces 0.124 within 0.02.
    RUN 2026-08-30, §2116: (a) HELD 8/8 positive; (b) HELD median +0.082;
    (d) HELD; (c) FAILED — gain sd 0.062 vs gap sd 0.090; the gain is
    LARGEST on the hardest windows (+0.206 on FR3). QUOTABLE under rung 6.

23. RE-PRICE THE ARM'S PIECES IN CE-AT-THE-CLIFF (the 18:35 review's C):
    oracle-correct each compressed piece of cfgE alone (drop its hook, no
    refit) and rank pieces by CE recovery vs by their own output-error
    energy. -> ops/piece_price_ladder.py [QUEUED]
    pred_a Spearman(CE recovery, local error energy) <= 0.5 across pieces
    (rel-MSE is the wrong currency at piece grain too); pred_b sum of
    single-piece recoveries >= 0.8 x the full recovery (additive); pred_c
    the front pieces (a0..m3) jointly recover >= 0.5 of the gap (§2103's
    85%-at-block-6 read at piece grain); pred_d cfgE reproduces 1.7415.
    RUN 2026-08-30, §2117: (a) FAILED — rho 0.81: at PIECE grain energy DOES
    order price (the head-grain separation is attn5-internal); (b) HELD
    0.885; (c) HELD 0.746; (d) HELD. Four equal levers (tail, m2, m3, m0 at
    0.15-0.16). Registry re-pricing NOT licensed by this.

24. CERTIFY HALF-PRICE ON THE EIGHT FRESH WINDOWS: norm-2304 vs metric-1152
    for mlp4/mlp5 on §2083's windows. -> ops/metric_units_halfprice8.py [QUEUED]
    pred_a metric-1152 gap <= norm-2304 gap + 0.02 on >= 6 of 8 windows;
    pred_b median (norm-2304 gap - metric-1152 gap) >= -0.02; pred_c window-
    1 reproduces §2107 (metric-1152 +1.594, norm-2304 +1.581, within 0.02).
    RUN 2026-08-30, §2118: (a) FAILED 2/8; (b) FAILED median -0.028; (c)(d)
    HELD. HALF-PRICE WITHDRAWN. Equal-price gain (§2116) stands.

25. CERTIFY THE EIGHT-DIRECTION SELECTOR on the eight fresh windows: top-8
    selector (§2110) vs norm at K=2304. -> ops/metric_units_top8_fresh8.py
    [QUEUED]  pred_a top-8 gain > 0 on >= 7 of 8; pred_b median top-8 gain
    >= 0.6 x §2116's median 0.082; pred_c window-1 top-8 gain reproduces
    §2110's 0.1285 within 0.02; pred_d cfgE reproduces 1.7415.
    RUN 2026-08-30, §2119: ALL FOUR HELD — 7/8 positive, median +0.082 (=
    the full metric), window 1 0.1284. The eight-direction selector is
    CERTIFIED at equal price.

26. METRIC-CHOSEN TAIL SPANS (opened from §2117: the tail spans tie for the
    largest single lever at 0.158, and the late-block metrics are large and
    stable unlike blocks 1-4 where the residual bases failed). Choose each
    tail MLP's rank-8 span as the top-8 directions of its output in the
    metric-whitened space (site = the block after the MLP), mapped back;
    random-metric control. -> ops/metric_tail_spans.py [QUEUED]
    pred_a metric spans beat cfgE by >= 0.05 nat on window 1 (a third of
    the tail's 0.158 oracle recovery); pred_b random-metric spans gain <=
    0.02; pred_c the gain transfers to FW rows 0:120 at >= 0.025 (half);
    pred_d cfgE reproduces 1.7415.
    RUN 2026-08-30, §2120 (scored on the eight pile windows, not FW 0:120,
    which fit the spans): (a) FAILED -0.052; (c) FAILED -0.085 median,
    worse on 8/8; (b)(d) HELD. Metric-CONSTRUCTED spans hurt; with §2105's
    bases, construction fails twice while selection succeeds twice.

27. SELECT vs CONSTRUCT (opened from §2120): at the tail, keep the plain
    PCA's top-32 directions and SELECT 8 of them by metric weight (selection
    among data-defined directions). -> ops/metric_tail_select.py [QUEUED]
    pred_a selected-8 >= plain top-8 - 0.01 on window 1 (no harm); pred_b
    fresh median gain >= +0.02; pred_c a RANDOM 8 of the 32 is worse than
    plain top-8 by >= 0.02 (the selection is not free); pred_d cfgE
    reproduces 1.7415. If (a) fails, the distinction is model-defined vs
    data-defined directions, not select vs construct.
    RUN 2026-08-30, §2121: (a)(b)(d) HELD; (c) FAILED in the informative
    direction — RANDOM 8 of top-32 also gains ~0.2 (better than metric on
    8/8 fresh). The metric adds nothing; the gain is the tail program
    intervening on LESS of the output (coverage), not a frontier move.
    §2117's tail lever is the cost of its own coverage. NOT COUNTED.

28. COVERAGE, NOT CHOICE (opened from §2121): six random 8-of-32 spans per
    tail MLP; record the variance share each span covers and the CE gap.
    -> ops/tail_span_coverage.py [QUEUED]
    pred_a Spearman(gain over cfgE, covered variance share) <= -0.7 across
    the draws + plain; pred_b a span of the 8 LOWEST-variance of the top-32
    gains >= the random median (the limit of doing less); pred_c the
    observable-energy coverage (via the site Gramian) predicts gain no
    better than variance share (|rho| within 0.1); pred_d cfgE 1.7415.
    RUN 2026-08-30, §2122: (a) HELD rho -0.976; (b) HELD lowest-8 best
    (+0.249 at 2% coverage); (c) FAILED — variance predicts BETTER than the
    observable share; (d) HELD. Coverage-stated credit LICENSED; §2121's
    0.2-nat 'gains' closed as non-moves. Tail registry entries over-credit.

BOOKKEEPING NOTE 2026-08-30 18:40Z: rungs 12-22 above were reconstructed from
the ledger entries §2103-§2115 after a cwd slip appended §2103 and rung 12's
opening to the repository-root copies of these two files (reverted); the
later per-rung backlog edits had silently anchored on text that was not here.

29. LABEL-FREE SELECTOR (opened from §2123): rank mlp4/mlp5 units by the
    TRUE-Fisher (labels sampled from the model, 2 samples) top-8 at blocks
    5/6 and certify at K=2304. -> ops/truefisher_top8_fresh.py [QUEUED]
    pred_a window-1 gain >= 0.6 x 0.1285 (§2110's empirical top-8); pred_b
    fresh median gain >= 0.6 x 0.082 (§2116); pred_c positive on >= 6 of 8
    fresh windows; pred_d cfgE reproduces 1.7415. If a/b fail, the gain
    needs the label-dependent half of the eight and the selector is a
    fitted object, priced as such.
    RUN 2026-08-30, §2124: ALL FOUR HELD — +0.122 w1, +0.086 fresh median,
    8/8 positive. The selector is LABEL-FREE (weights + unlabeled inputs,
    fold-table deploy status). The observability arc closes (rungs 11-29).

30. INSTALL THE SELECTOR INTO THE FRONTIER (opened 2026-08-30 19:25Z; the
    lane-1 prompt's candidate A). §312's empirical-L2 frontier (+2.6735
    fresh) uses the same norm-selected top-2304 CP middles as cfgE; rerun
    its full pipeline with true-Fisher top-8 selection at mlp4/mlp5.
    -> ops/frontier_fisher8.py [QUEUED]
    pred_a norm arm reproduces the published L2_F 2.6735 within 0.05;
    pred_b fresh gain >= 0.04 (half of cfgE's 0.086 — dilution expected);
    pred_c window-C gain >= -0.01.
    RUN 2026-08-30, §2125; CORRECTED BY §2128: the registered gain
    formula was SIGN-INVERTED (norm − fisher). Arm values: fisher8
    +2.7210 fresh / +2.4482 C vs norm +2.6735 / +2.4232 — the selector
    INSTALLS (+0.0475 fresh). As-written scores (b,c FAILED) preserved.

32. ASSEMBLY-CONDITIONED FISHER (opened as DESIGN in §2125's repair
    reading; built 19:38Z as a two-pass run: the norm arm collects the
    true-Fisher top-8 at blocks 5/6 with the full L2 hooks installed,
    labels from the deployed assembly's own predictions; arm 2 selects
    under it). -> ops/frontier_fisher8_asm.py
    pred_a reproduction within 0.01; pred_b gain >= 0.02; pred_c C >= -0.01.
    RUN 2026-08-30, §2128: pred_a HELD; pred_b/c as-written FAILED only
    because the formulas inherited rung 30's inverted sign. Arms:
    asm-conditioned +2.7682 fresh / +2.4833 C vs norm +2.6735 / +2.4233 vs
    unconditioned +2.7210 / +2.4482. Conditioning adds +0.047; the
    frontier's best number now carries the label-free selector -> rung 35.

33. THE SINK-HEAD SCALAR (§2126) — RUN 2026-08-30: s* = 1.095 (cfgE
    UNDER-drives head 5.7, refuting the §1818 analogy's sign) and buys
    0.015 of the head's 0.28-nat oracle. Per-head scale corrections are
    not a lever; the scale error belongs to the front's stream.

31. COVERAGE-CREDIT AND SELECTOR NOTES IN THE THESEUS REGISTRY (the 19:25
    review's item 2) — DONE 2026-08-30 19:30Z as bookkeeping, no run:
    registry/priorities.md now (i) flags every top-K sub-MLP row as
    upgradeable at zero price by the certified label-free selector
    (certify per row before crediting; c6-c9-style was negative), (ii)
    requires covered-energy share for projection/span programs per §2122,
    (iii) records that the ordering stands (§2117 rho 0.81).

34. THE m16 TWO-NUMBER INTERFACE, PRICED (the reviews' candidate C; from
    §2098-§2100). Rank-2 source basis + rank-1 (phase,target) profiles fit
    on half A; per-document coefficients (2 numbers) estimated on half B
    from ALL cells vs TWO physical arms; price 208 persistent + 2/doc +
    2 calibration arms. -> ops/m16_two_number_interface.py [QUEUED]
    pred_a median held-out R^2 (all cells) >= 0.5; pred_b two arms >= 0.8 x
    all-cells; pred_c mean-code and permuted-code baselines <= 0.1.
    RUN 2026-08-30, §2127: (a) FAILED 0.069, (b) FAILED 0.044, (c) HELD.
    Fixed profiles x 2 numbers do not carry the block: m16's per-document
    response varies in SHAPE. Candidate C CLOSED — no cheap measured
    interface; the m16 block stays the priced unexplained remainder.

35. CERTIFY THE CONDITIONED FRONTIER AT THE RUNG-6 STANDARD. §2128's
    +2.7682 fresh is one 120-row window set; before it is called the
    frontier best, evaluate the norm and asm-conditioned L2 configs on the
    eight document-disjoint pile-10k windows (probe_gate7 take_window).
    Bars: pred_a median per-window gain, L2_F(asm) − L2_F(norm), >= 0.04;
    pred_b >= 7/8 windows positive; pred_c norm reproduces 2.6735 within
    0.01 on the original FR windows (pipeline unchanged).
    -> ops/frontier_asm_fresh8.py
    RUN 2026-08-30, §2129: ALL THREE HELD — median +0.0481, 7/8 positive,
    reproduction exact. Certified frontier best: +2.7707 fresh / +2.4846 C
    at equal price, label-free assembly-conditioned selector.

36. CONDITION cfgE ON ITSELF. §2128 showed conditioning the Fisher on the
    deployed §312 assembly adds +0.047 over the real-model Fisher there.
    cfgE's certified +0.086 (§2124) still uses the real-model metric; test
    whether conditioning is general: recompute the top-8 at blocks 5/6
    with cfgE's own CP middles + front tables installed (labels from
    cfgE's predictions), re-select, and score on the eight windows.
    Bars when built: pred_a median gain over real-model-metric selection
    >= +0.02; pred_b >= 6/8 windows non-negative; pred_c the real-model
    arm reproduces §2124's +0.086 median within 0.01.
    -> ops/cfge_conditioned_fisher.py
    RUN 2026-08-30, §2130: pred_b HELD (7/8), pred_a FAILED (+0.0123 vs
    0.02), pred_c FAILED by 0.0001 over tol on an unregistered RNG-stream
    change (SITE_OF restriction moved genF; ~0.01 MC-sample sensitivity
    of the top8 gain is the instrument fact). Observation (unlicensed):
    conditioning buys ~1/4 on cfgE of what it buys on the frontier,
    direction matching §2128's reader-mechanism. Label check: window-0
    gap identical to 4dp under stale vs classified attnd labels.

37. EXTEND THE CONDITIONED SELECTION TO ALL SIX CP MIDDLES. §2129 selects
    only mlp4/mlp5; c6-c9 stay norm-selected, and §2106/§2107 found
    REAL-model metric selection there hurts on cfgE (genuine null). Three
    arms (norm / conditioned mlp45 / conditioned mlp4-9), asm Fisher
    collected at sites 5-10, eight-window scoring.
    pred_a median [L2_F_w(all) - L2_F_w(mlp45)] >= +0.02; pred_b >= 6/8
    non-negative; pred_c mlp45 arm reproduces §2129's +0.0481 within
    0.015 (CUDA-atomics wobble ~0.003).
    -> ops/frontier_asm_alllayers.py
    RUN 2026-08-30, §2131: pred_c HELD (+0.0452 repro), pred_a FAILED
    (+0.0032), pred_b FAILED (5/8). c6-c9 conditioned selection adds
    nothing - the selection gain is an mlp4/mlp5 story at both metrics in
    both assemblies; the frontier keeps the §2129 config.

38. LICENSE RUNG 36 PROPERLY: RNG-MATCHED cfgE CONDITIONING + LABEL BOUND.
    §2130's conditional reading was unlicensed because SITE_OF restriction
    moved the genF stream and the reproduction gate missed by 0.0001.
    Rerun with the FULL rung-29 SITE_OF (stream matched exactly), and add
    a random-label arm to the window-0 instrument check.
    pred_a conditioning median [gap(top8) - gap(cond8)] >= +0.02;
    pred_b conditioning >= 0 on >= 6/8; pred_c top8 arm reproduces
    §2124's +0.0857 within 0.01 (stream now matched); pred_d RANDOM attnd
    labels move the window-0 plain gap by >= 0.05 (if FAILED, cfgE's
    attnd class-label input is certified inert at window grain <= 0.05).
    -> ops/cfge_conditioned_fisher2.py
    RUN 2026-08-30, §2132: pred_c HELD (EXACT +0.0857 repro - the §2130
    miss was purely the RNG stream), pred_a/b FAILED (-0.0021, 3/8:
    §2130's +0.012 was MC noise - conditioning buys ZERO on cfgE; the
    reader mechanism is certified in both directions), pred_d FAILED
    structurally (cfgE has no label-consuming piece; the instrument
    question dissolves, no certified number was exposed).

39. HALF-PRICE UNDER THE CONDITIONED METRIC. §2118 withdrew half-price
    under the real-model metric on cfgE (median -0.028; stated null); the
    conditioned metric is the certified right one for the frontier
    (§2129), so the price question reopens exactly once. Arms norm-2304 /
    cond-2304 (repro) / cond-1152 at mlp4/mlp5, eight windows; -2x1152
    stored units (~7.96M values) if it holds.
    pred_a median [L2_F_w(cond1152) - L2_F_w(norm2304)] >= 0; pred_b
    >= -0.01 on >= 6/8; pred_c cond-2304 reproduces §2129 within 0.015.
    -> ops/frontier_cond_halfprice.py
    RUN 2026-08-30, §2133: ALL THREE HELD — cond-1152 beats norm-2304 by
    +0.0861 median (8/8) and cond-2304 by ~+0.04 everywhere. Frontier
    best now +2.8190 fresh / +2.5461 C at HALF the mlp4/mlp5 units. The
    bottom half of the conditioned ranking is net HARMFUL when deployed.

40. THE CONDITIONED PRICE CURVE, DOWNWARD. §2133: cond-1152 beats
    cond-2304 everywhere — how far down does it go? Arms: norm (Fisher
    collection) / cond-1152 (§2133 reproduction) / cond-576 / cond-288,
    eight windows. §2107's capacity result (metric gain halves at K=4608)
    says the curve must turn; the question is where.
    pred_a median [L2_F_w(cond576) - L2_F_w(cond1152)] >= -0.01 (quarter
    price no loss); pred_b that difference >= -0.02 on >= 6/8; pred_c
    cond-1152 reproduces §2133 (median vs norm +0.0861 within 0.015).
    -> ops/frontier_cond_ksweep.py
    RUN 2026-08-30, §2134: ALL THREE HELD — cond-576 adds +0.0392 over
    cond-1152 (8/8); frontier best +2.8372 fresh / +2.5953 C at quarter
    price; cond-288 still beats cond-1152 (7/8). Curve peaks near K~576.

41. THE PRICE FLOOR. §2134: the conditioned curve peaks near K = 576 and
    cond-288 barely turns down. Find the floor: arms norm (collection) /
    cond-576 (§2134 reproduction) / cond-144 / K-0 (Down-bias only, the
    "do the units matter at all" null).
    pred_a median [L2_F_w(cond144) - L2_F_w(cond576)] >= -0.01; pred_b
    median [L2_F_w(K0) - L2_F_w(cond576)] <= -0.05 (the kept units DO
    carry real CE; if FAILED, mlp4/mlp5's CP content is nearly all
    harmful-or-inert in the deployed assembly - a major finding on its
    own); pred_c cond-576 reproduces §2134 (median vs norm +0.1277
    within 0.015).
    -> ops/frontier_cond_floor.py
    RUN 2026-08-30, §2135: pred_c HELD; a/b FAILED as written — and the
    K-0 arm (+3.239) exposed that the WHOLE K-sweep arc was sign-flipped:
    L2 is damage above the real model, LOWER is better. RETRACTION §2135:
    §2125 reinstated, §2128/§2129/§2133/§2134 retracted; frontier is and
    was norm-2304 at 2.6735. Units carry ~0.5 nat (K-0 control).

42. PRUNE c6-c9 UNDER THE CONDITIONED RANKING. §2131 killed reordering at
    K=2304; §2134 showed the mlp4/5 win came from DROPPING the harmful
    tail. Test pruning c6-c9 to 576 each (saves ~23.9M values if free),
    ranked by conditioned Fisher vs norm, with mlp4/5 fixed at cond-576.
    pred_a median [L2_F_w(c69cond576) - L2_F_w(cond576@45)] >= 0; pred_b
    median [c69cond576 - c69norm576] >= 0 (§2131 null predicts ~0);
    pred_c base arm reproduces §2134 (+0.1277 within 0.015).
    -> ops/frontier_c69_prune.py
    RUN 2026-08-30, §2136 (read in the corrected convention): conditioned
    ranking worse than norm for pruning too (+0.055/window median); norm
    pruning of c6-c9 to 576 ~ breaks even on its (damaged) base. The
    conditioned Fisher has NO measured use on this frontier.

    [§2135 NOTE on rungs 30-42: every 'gain' registered after §2128 was
    written in a flipped sign convention (higher L2 read as better).
    The measurements stand as damage numbers; the celebratory readings
    are retracted. Rung 42 (in flight) to be read as damage.]

43. THE HONEST PRICE QUESTION: NORM-RANK K REDUCTION ON THE INTACT §312
    FRONTIER (never measured on the eight windows). Convention: L2 = CE
    above the real model, lower is better. Arms: norm-2304 everywhere
    (§312 reproduction) / norm-1152 at all six middles / norm-576 at all
    six. Price: -13.3M / -20M values. Null (§2118 family): K reduction
    adds damage; the question is how little.
    pred_a median [L2_F_w(norm1152) - L2_F_w(norm2304)] <= +0.02;
    pred_b that difference <= +0.04 on >= 6/8; pred_c norm-2304
    reproduces 2.6735 within 0.01. norm-576 descriptive.
    -> ops/frontier_norm_ksweep.py
    RUN 2026-08-30, §2137: pred_a FAILED (+0.0292 > 0.02), pred_b HELD
    (6/8), pred_c HELD (exact). Halving costs a real +0.029; quartering
    only +0.008 more - the price curve is concave, no cliff to 576.
    Norm rank is the only ranking that cuts price at tolerable cost.

44. MC-SAMPLE ROBUSTNESS OF THE cfgE TOP-8 SELECTOR (instrument rung;
    damage convention stated inline). The standing positive (-0.0857
    median cfgE damage, §2116/§2124) uses a 2-sample MC Fisher with
    ~0.01 cross-stream sensitivity (§2130/§2132). Arms: plain / top8-2s
    (stream-matched repro) / top8-4s (fresh generator, seed 44).
    pred_a overlap(2s,4s eight) >= 0.7 at both sites; pred_b
    |median reduction(4s) - median reduction(2s)| <= 0.01; pred_c 2s arm
    reproduces 0.0857 within 0.01. If pred_b fails, the certified number
    carries a stated MC error bar. -> ops/cfge_top8_samples.py
    RUN 2026-08-30, §2138: ALL THREE HELD — overlap 0.986/0.986,
    reduction moves 0.0025, repro exact. The cfgE result is instrument-
    robust: 0.086 ± ~0.003 (MC); the §2130 sensitivity was marginal-unit
    jitter, not metric instability.

45. WHO PAYS FOR THE HALVING (per-group attribution of §2137's +0.0292;
    damage convention). Arms: norm-2304 (repro) / halve mlp4-5 only /
    halve c6-c9 only. Tension: §2136 hinted c6-c9 pruning nearly free vs
    §2113's cliff at blocks 6-9.
    pred_a additivity of the two medians to +0.0292 within 0.01; pred_b
    median dmg(halve45) >= median dmg(halve6789); pred_c repro 2.6735
    within 0.01. -> ops/frontier_halving_attrib.py
    RUN 2026-08-30, §2139: ALL THREE HELD — the cost is ALL at mlp4/5
    (+0.0473); halving c6-c9 is free-or-better (-0.0118, 6/8 negative);
    additivity 0.0063. Legit frontier move: -8.9M values at no cost.

46. HOW FAR DOWN DOES c6-c9 GO FOR FREE (damage convention). §2139:
    c6-c9 at 1152 is free-or-better with mlp4/5 kept at 2304. Arms:
    norm-2304 (repro) / c69-1152 (§2139 repro) / c69-576 / c69-288.
    Price: -8.9M / -13.3M / -15.6M values.
    pred_a median [L2_F_w(c69_576) - L2_F_w(norm2304)] <= +0.01; pred_b
    that <= +0.02 on >= 6/8; pred_c c69-1152 reproduces §2139 (median
    -0.0118 within 0.015). c69-288 descriptive.
    -> ops/frontier_c69_floor.py
    RUN 2026-08-30, §2140: ALL THREE HELD — c69-576 BEATS the full
    frontier on 8/8 windows (-0.0290 median) at -13.3M values; 288
    rebounds. Best measured config: mlp4/5-2304 + c69-576, 2.6445 fresh.

47. CAN mlp4/5 SHED UNITS ON TOP OF THE NEW BEST (damage convention).
    §2139: halving mlp4/5 costs +0.047; a quarter-trim may be cheaper.
    Arms: best (mlp45-2304 + c69-576; §2140 repro) / mlp45-1728 on top /
    mlp45-1152 on top. Price: -2.2M / -4.4M further values.
    pred_a median [L2_F_w(mlp45_1728) - L2_F_w(best)] <= +0.015; pred_b
    that <= +0.03 on >= 6/8; pred_c best arm reproduces §2140 (median vs
    norm-2304 of -0.0290 within 0.015; needs the norm-2304 arm too).
    -> ops/frontier_mlp45_trim.py
    RUN 2026-08-30, §2141: pred_c HELD (best repro exact), pred_a/b
    FAILED (+0.0244 for the quarter-trim; +0.058 for the half). The
    mlp4/5 price wall is steep immediately - no trim is free; best
    config stays mlp45-2304 + c69-576.

48. DOES c6-c9 PRUNING GENERALIZE TO cfgE + COMPOSITION (damage
    convention). Tests §2140's unproven mechanism reading (noise the
    readers amplify) on real-reader cfgE, and whether the two standing
    positives compose. Arms: plain / prune (c69-576) / top8 (repro) /
    top8+prune. Null: §2132's precedent (frontier-only effects).
    pred_a median [gap(prune) - gap(plain)] <= 0; pred_b median
    [gap(top8+prune) - gap(top8)] <= 0; pred_c top8 reduction reproduces
    0.0857 within 0.01. -> ops/cfge_c69_prune.py
    First run CRASHED 21:33Z (KeyError: HEAD[piece][mode] not keyed for
    the new top8prune mode) - instrument bug, no result; fixed and
    re-queued 21:41Z. Predictions unchanged.
    RUN 2026-08-30, §2142: pred_a HELD (-0.0248 — pruning generalizes to
    real readers), pred_c HELD (repro), pred_b FAILED (+0.0420 on 8/8 —
    the two positives share a budget and do NOT compose; 'dead weight'
    is relative to the assembly's other errors). cfgE best: top8 alone.

49. PER-LAYER c6-c9 FLOORS ON THE FRONTIER BEST (damage convention).
    §2140: uniform 288 rebounds by +0.0217 vs the 576 optimum. Which
    layer carries it? Arms: norm-2304 anchor / best (c69-576, §2140
    repro) / c6@288 / c7@288 / c8@288 / c9@288 (others at 576).
    pred_a min over layers of median [L2_F_w(cX@288) - L2_F_w(best)]
    <= +0.005 (at least one layer can go to 288 nearly free); pred_b
    additivity: |sum of the four medians - 0.0217| <= 0.015 (§2139
    precedent; if FAILED, per-layer interactions); pred_c best arm
    reproduces §2140 (-0.0290 vs norm within 0.015).
    -> ops/frontier_c69_perlayer.py
    RUN 2026-08-30, §2143: ALL THREE HELD — rebound carried by c6/c7
    (+0.011/+0.017); c8/c9 go to 288 free-or-better (-0.0076/-0.0036);
    per-layer costs additive (gap 0.0053).

50. c8+c9 TO 288 TOGETHER (damage convention). §2143's additive
    prediction: c8@288 + c9@288 lands at -0.0112 vs the §2140 best (a
    new best, -14.4M values vs §312). Arms: norm-2304 anchor / best
    (c69-576, repro) / c8+c9@288 / c8+c9@144 (descriptive floor).
    pred_a median [L2_F_w(c89_288) - L2_F_w(best)] <= -0.005 (a real
    improvement); pred_b that median within 0.01 of the additive
    prediction -0.0112; pred_c best reproduces §2140 (-0.0290 within
    0.015). -> ops/frontier_c89_288.py
    RUN 2026-08-30, §2144: ALL THREE HELD — c89@288 lands at -0.0128 vs
    the predicted -0.0112 (gap 0.0016). NEW BEST: mlp45-2304 + c6/c7-576
    + c8/c9-288, 8/8 windows better, -14.4M values; 144 breaks even
    (floor ~288).

51. WHO PAYS THE TAIL-ATTENTION INCREMENT (attribution rung; damage
    convention). The aXL dictionaries are fit sequentially, so prefix
    configs are matched-context by construction: nine prefix evals on FR
    give a per-layer marginal attribution of the +0.37 increment with no
    refits. Single arm on the best middles (mlp45-2304 + c69-576).
    pred_a full config reproduces §2140's 2.6691 within 0.01; pred_b all
    eight marginals >= -0.005 (a negative one means real attention was
    HURTING - notable); pred_c max marginal >= 2 x median (not uniform).
    -> ops/frontier_tail_prefix.py
    RUN 2026-08-30, §2145: ALL THREE HELD — the increment is concentrated
    at a16L (+0.1572, 45% of the total; a14L +0.073; a12L/a17L ~free).
    Block 16 again: neither its MLP nor its attention survives the
    grammar. No negative marginals; telescoping exact.

52. LEAVE attn16 REAL (coverage-vs-damage envelope point; damage
    convention). §2145: a16L's dictionary costs +0.157. Refit the tail
    sequence with block 16's attention left real (skip a16L; a17L refit
    under a16-real — grammar-consistent), on the §2144 best middles.
    HONEST FRAMING: one fewer replaced component, so this is an envelope
    point, not a free improvement.
    pred_a median [L2_F_w(skip16) - L2_F_w(full)] <= -0.10 (the prefix
    marginal was -0.157; refits may reabsorb some); pred_b <= -0.05 on
    >= 7/8; pred_c full arm reproduces §2144 (FR L2_F 2.6662 within
    0.01). -> ops/frontier_skip_a16.py
    RUN 2026-08-30, §2146: ALL THREE HELD — saving -0.2126 on 8/8 (MORE
    than the -0.157 marginal: a17L refit on cleaner inputs). Strict
    (price,damage) dominance; coverage retreats by one component.
    BLOCK 16 is now the program's largest single open object.

53. CLASS-BOTTLENECK SPECTRUM AT THE TAIL (math review 2210; damage
    convention). Within-class residual energy fraction e_li of the real
    attention outputs (blocks 10-17, deployed context, oracle classes)
    vs the §2145 marginals. pred_a Spearman rho >= 0.7 over the 8
    layers; pred_b argmax e = a16; pred_c two smallest e = {a12,a17}.
    Null: CE-weighting beats energy (rho < 0.7). One capture pass.
    -> ops/tail_class_spectrum.py
    RUN 2026-08-30, §2147: pred_a HELD (rho 0.786), pred_b/c FAILED —
    argmax is a14 (0.933) not a16 (0.908); smallest are a10/a17. The
    exceptions are the finding: a16's residual is unusually damaging
    PER UNIT ENERGY; energy is an ordering tool, not a price law.

54. ALL-LINEAR CLASSES AT a16 (grammar extension at the identified
    bottleneck; damage convention). a16L currently gives linear maps to
    the 4 LINK classes and means to the 6 CONSTN classes. Upgrade a16
    (only) to linear maps for ALL 10 classes: +6 x 1.33M = +8.0M values
    at one layer. Arms: §2144 best full tail (repro) / a16-all-linear.
    pred_a median [L2_F_w(a16lin) - L2_F_w(full)] <= -0.05 (recovers a
    third of the +0.157); pred_b <= 0 on >= 7/8; pred_c full reproduces
    2.6662 within 0.01. Null: §2147 — the damage is loss-specific, not
    class-mean sloppiness, and linearity per class buys little.
    -> ops/frontier_a16_linear.py
    First run 22:17Z VOID (instrument bug): the all-linear patch landed
    in fit_attnd (front pieces) instead of the aXL tail loop - arm 2 was
    IDENTICAL to arm 1. The +/-0.0003 per-window deltas are therefore a
    measured RERUN-NOISE FLOOR for this pipeline (useful: future window
    tolerances can cite it). Fixed and re-queued 22:20Z; preds unchanged.
    RE-RUN 2026-08-30, §2148: pred_c HELD, pred_a/b FAILED (+0.0037,
    worse on 8/8 — an order above the noise floor). The §2147 null wins:
    a16's cost is conditional structure the 10-class code cannot see;
    in-grammar upgrades at a16 are exhausted.

55. PER-HEAD MARGINALS OF THE REAL attn16 (attribution inside the
    block-16 bottleneck; damage convention). On the §2146 skip-a16
    config (attn16 real), zero each of attn16's 9 heads (c_proj input
    slice) and eval FR; plus an all-heads-zeroed arm.
    pred_a CONCENTRATION: max_h d_h >= 2 x median_h d_h; pred_b
    ADDITIVITY: |sum_h d_h - d_all| <= 0.05 (heads may interact — a
    failure is informative); pred_c skip16 reproduces §2146 (FR L2_F
    2.5091 within 0.01). -> ops/attn16_head_marginals.py
    RUN 2026-08-30, §2149: ALL THREE HELD — heads 16.3/16.4/16.0 carry
    +0.113 of +0.119; six heads sum to -0.001; head lattice additive
    (0.0072). AND: the a16L dictionary is +0.038 WORSE than deleting
    attn16 outright — any new grammar must beat zero.

56. WHICH CLASSES PAY FOR a16L (attribution; damage convention).
    Per-position CE(full config with a16L) - CE(attn16 real) on FR,
    grouped by the 10 oracle classes. Null: uniform shares (~0.10).
    pred_a top class share >= 0.40; pred_b the top class is a LINK
    (linear-mapped) class; pred_c skip16 reproduces §2146 (2.5091
    within 0.01). -> ops/attn16_class_damage.py
    First run CRASHED 22:30Z (double del of cur['clsmap'] - my inserted
    capture block deleted it and the parent's trailing del hit KeyError
    after the arm-1 build; no result). Fixed, re-queued 22:33Z behind
    rung 57; preds unchanged.
    RE-RUN 2026-08-30, §2151: ALL THREE HELD — ind 0.518 + other 0.384
    + subword 0.099 = ~100% of the damage, all LINK classes; mean classes
    unhurt. a16 = document-memory read; per-position grammars cannot
    carry it (explains §2148/§2149).

57. attn16 AS THREE HEADS (damage convention). §2149's additive
    prediction: zeroing heads {1,2,5,6,7,8} together costs -0.0012.
    Single build (§2146 skip-a16 config), joint-six-zero eval on FR.
    pred_a THE ADDITIVE NUMBER: |d(six zeroed) - (-0.0012)| <= 0.01;
    pred_b NEARLY FREE: d <= +0.01; pred_c skip16 reproduces §2146
    (FR L2_F 2.5091 within 0.01). If held, block-16 attention reduces
    to a 3-head object (16.0/16.3/16.4) for the program's purposes.
    -> ops/attn16_three_heads.py
    RUN 2026-08-30, §2150: ALL THREE HELD — d = -0.0017 vs predicted
    -0.0012 (gap 0.0005; third consecutive additive confirmation).
    attn16 = heads 16.0/16.3/16.4 for the program's purposes; the six
    zeroed heads are marginally better off dead.

58. WINDOW CERTIFICATION OF THE THREE-HEAD ENVELOPE (damage convention).
    §2150's three-head result is FR-only. Arms: skip-a16 nine-head
    (§2146 window repro) / skip-a16 with heads 16.1/2/5/6/7/8 zeroed
    THROUGHOUT (bases excluded; fits matched under the three-head a16).
    pred_a median [L2_F_w(three) - L2_F_w(nine)] <= +0.005; pred_b
    <= +0.01 on >= 6/8; pred_c the nine-head arm reproduces §2146's
    per-window values (median |delta| <= 0.005).
    -> ops/attn16_three_heads_w8.py
    RUN 2026-08-30, §2152: pred_c HELD (repro to 0.0001), pred_a/b
    FAILED (+0.0239 median, 2/8) — the three-head claim was an FR
    text-homogeneity artifact; scope notices on §2149/§2150. Every
    reduction claim now needs the eight windows first.

59. attn16 PER-HEAD MAP AT WINDOW GRAIN (damage convention). Rung 55's
    map was FR-only. Single skip-a16 build; eval-scoped zeroing of each
    head (and all nine) over the eight stored windows.
    pred_a the top three heads by window-median d_h are {16.0,16.3,16.4}
    (the FR big three); pred_b the OTHER six heads' summed window-median
    >= +0.015 (they carry, per §2152); pred_c the base arm reproduces
    §2146's per-window values (median |delta| <= 0.005).
    -> ops/attn16_headw8.py
    First build 22:46Z VOID (silent): the runner slice anchor matched a
    SUBSTRING of an 8-space-indented line inside main(), splicing the
    runner into main's body and deleting the __main__ guard - the module
    imported in 5s, ran nothing, exit 0, empty log. LESSON: slice anchors
    must be line-anchored ('\n'-prefixed); the gate should also refuse a
    script with no __main__ executor. Rebuilt from the intact parent,
    re-queued 22:52Z; preds unchanged.
    VALID RUN 2026-08-30, §2153: ALL THREE HELD — big three survive;
    16.5 (+0.0193) is the carrier FR missed; the other five sum to
    -0.0007. attn16 = FOUR heads at window grain.

60. WINDOW CERTIFICATION OF THE FOUR-HEAD attn16 (damage convention;
    §2152's standard — zeroing active throughout fits and evals, bases
    excluded). Zero heads {16.1, 16.2, 16.6, 16.7, 16.8}; additive
    prediction from §2153: -0.0007.
    pred_a median [L2_F_w(four) - L2_F_w(nine)] <= +0.005; pred_b
    <= +0.01 on >= 6/8; pred_c nine-head arm reproduces §2146 per-window
    (median |delta| <= 0.005). -> ops/attn16_four_heads_w8.py
    RUN 2026-08-30, §2154: pred_c HELD, pred_a/b FAILED (+0.0097 median,
    4/8; FR-like windows ~0). Joint-throughout removal picks up a ~+0.01
    interaction/refit term the individual eval-scoped marginals lack —
    additive predictions licensed only for individual eval-scoped
    marginals. Four-head attn16 = a trade (+0.0097), not free.

61. attn14 PER-HEAD MAP AT WINDOW GRAIN (damage convention; window-grain
    from the start per §2152's lesson). Skip-a14 config (a15L-a17L refit
    under a14-real), eval-scoped per-head zeroing over the eight windows.
    pred_a concentration (max >= 2 x median); pred_b additivity
    (|sum - d_all| <= 0.05); pred_c the §2145 marginal survives refits
    (median saving vs the §2144 best >= +0.04).
    -> ops/attn14_headw8.py
    First run 23:02Z: preds a/b VOID — the hook-retarget rep landed on a
    DORMANT duplicate block; the active block still zeroed h[16], whose
    output the installed a16L overrides → inert hooks, bit-identical
    evals. pred_c VALID and striking: skip-a14 saves +0.2326 median vs
    the §2144 best (three downstream refits compound; cf. §2146). Fixed
    (+ an inert-hook tripwire in-script), re-queued 23:06Z.
    VALID RE-RUN, §2155: ALL THREE HELD — diffuse map (top 14.6 +0.041;
    six heads carry; 14.2 NEGATIVE -0.026; additivity gap 0.042);
    skip-a14 saves +0.2327 (strongest envelope point yet).

62. DO THE TWO SKIPS COMPOSE (damage convention; §2154's rule — joint
    removals must be measured). Arms: §2144 best full tail (repro) /
    skip-a16 (§2146 window repro) / skip-a14+a16 (a15L/a17L refit under
    both real). Naive sum of savings ≈ -0.44; §2154 says expect less.
    Price: -10.6M values; coverage retreats two components (stated).
    pred_a median [L2_F_w(skip1416) - L2_F_w(full)] <= -0.30; pred_b
    [L2_F_w(skip1416) - L2_F_w(skip16)] <= 0 on >= 7/8; pred_c skip16
    arm reproduces §2146 per-window (median |delta| <= 0.005).
    -> ops/frontier_skip_1416.py
    RUN 2026-08-30, §2156: ALL THREE HELD — joint -0.3222 (naive -0.445;
    interaction +0.12 in the shared refits), 8/8 vs skip-a16, repro
    exact. The tail-dictionary program is pure coverage spend; the ~0.32
    gap IS the price of describing blocks 14/16's attention per-position.

63. WHICH CLASSES PAY FOR a14L (damage convention; the retrieval-vs-
    energy dichotomy at the second layer). §2151: a16L's damage is 90%
    ind+other (retrieval). §2147/§2155: a14 is energy-dominated and
    diffuse. Per-class CE attribution, full vs skip-a14.
    pred_a NOT RETRIEVAL-CONCENTRATED: share(ind) <= 0.30 (a16: 0.518);
    pred_b THE MEAN CLASSES PAY TOO: combined CONSTN share >= 0.20
    (a16: ~0); pred_c skip-a14 arm reproduces §2155's base per-window
    (median |delta| <= 0.005). Null: a14 is retrieval-like after all.
    -> ops/attn14_class_damage.py
    RUN 2026-08-30, §2157: pred_c HELD, pred_a/b FAILED — the null wins:
    a14L's damage is ALSO ~97% retrieval classes (ind 0.571); the whole
    tail price is a retrieval price; layers differ in HOW they read.

64. POINTER-LINEAR STAND-IN FOR THE ind CLASS AT a16 (the first
    constructive retrieval primitive; damage convention). For ind-class
    positions, a16's stand-in outputs (stream at the last occurrence of
    the target token) @ W_ptr (ridge-fit on FW ind positions) instead of
    the class-linear map; other classes unchanged. PRICE: +1.33M values
    (W_ptr) + the pointer itself is computed from the token sequence
    (same oracle interface as the class labels). NULL (§2148/§2151):
    retrieval needs more than the previous-occurrence stream.
    pred_a median [L2_F_w(full) - L2_F_w(a16ptr)] >= +0.05 (recovers a
    quarter of a16L's 0.21 window cost); pred_b >= +0.02 on >= 6/8;
    pred_c full arm reproduces §2144 (FR L2_F 2.6662 within 0.01).
    In-script tripwire: the pointer branch must fire >0 times.
    -> ops/a16_pointer.py
    First run 23:30Z: BOTH ARMS COMPLETED (log shows full 2.6662 exact;
    pointer arm 2.6550; recovery ~+0.011/window on 8/8) but the runner's
    tripwire referenced main-local `cur` -> NameError before the JSON was
    written. Fixed (nptr via SEL), re-queued 23:35Z behind rung 65; the
    scored artifact comes from the rerun.
    VALID RERUN, §2158: pred_c HELD, pred_a/b FAILED — recovery +0.0109
    median (8/8 positive, 100,324 fires): real, uniform, ~5% of the
    cost. Last-occurrence-linear is NOT attn16's read.

65. IS m16 RETRIEVAL-SHAPED TOO (cross-object attribution; damage
    convention). m16 cannot read other positions; if deleting it still
    damages ind/other positions predominantly, block 16 is a coupled
    retrieval unit (attn reads, MLP transforms) — unifying the m16
    remainder with the retrieval price. Null: class-uniform damage.
    pred_a ind+other+subword >= 0.60; pred_b ind >= 0.25; pred_c full
    arm reproduces §2144 (2.6662 within 0.01).
    -> ops/m16_class_damage.py
    RUN 2026-08-30, §2159: coupled-retrieval REFUTED, informatively —
    m16 is a subword-continuation unit (+830 summed nats) that HURTS ind
    positions (−620); net deletion cost only +0.0027 FR / +0.029 C.
    pred_b FAILED; pred_a voided as ill-posed (share bars need a
    sign-definite total — lesson adopted); pred_c HELD. Zeroed arm's
    per-window numbers VOID (hook-scope bug on the per-window bases).

66. THE INDUCTION POINTER AT a16 (the single registered v2; damage
    convention). For ind positions, match the last previous occurrence
    of the CURRENT token idx[p] and read the stream at the position
    JUST AFTER it (where the successor token — the induction guess —
    lives): output = x[q*+1] @ W_ptr. Same price (+1.33M), same oracle
    interface. NULL (§2158): this too recovers ~0.01.
    pred_a median [L2_F_w(full) - L2_F_w(a16ind)] >= +0.05; pred_b
    >= +0.02 on >= 6/8; pred_c full reproduces §2144 (2.6662 within
    0.01). Tripwire: pointer fires > 0. -> ops/a16_induction.py
    RUN 2026-08-30, §2160: pred_c HELD, pred_a/b FAILED — the induction
    form is slightly WORSE (-0.0024, 0/8). Neither single-pointer linear
    read is attn16's retrieval.

67. IS THE a16 DAMAGE BIGRAM-ADDRESSED (attribution; signed-sum bars per
    §2159's lesson; damage convention). Split the per-position a16L
    damage (full vs skip-a16, FR) by whether the current bigram
    (t[p-1],t[p]) occurred earlier in the document.
    pred_a bigram-matched ind+other positions carry >= 0.6 of the
    summed ind+other damage; pred_b mean damage on bigram-matched
    >= 2 x mean on unmatched (within ind+other); pred_c skip-a16 arm
    reproduces §2146 per-window (median |delta| <= 0.005). If pred_a/b
    hold, ONE bigram-pointer rung is licensed; else the constructive
    program at a16 pauses. -> ops/a16_bigram_split.py
    RUN 2026-08-30, §2161: pred_c HELD, pred_a/b FAILED (matched = 17%
    of the sum; mean ratio 1.98x, under the 2x bar by 0.004). The
    registered pause takes effect: three constructive shots, all below
    bar; the envelope stands as block-16's description.

68. IS THE a16 READ SHORT- OR LONG-RANGE (attribution; signed-sum bars;
    damage convention). Split the per-position a16L damage on ind
    positions by distance to the target's last occurrence.
    pred_a NEAR CARRIES: distance <= 32 positions carry >= 0.5 of the
    summed ind damage (if HELD, a truncated-window 3-4-head attention
    is the licensed next construction; if FAILED, block 16 is genuine
    long-range memory); pred_b monotonicity: Spearman |rho|(mean damage
    per log2-distance bin) >= 0.5 over bins; pred_c skip-a16 arm
    reproduces §2146 per-window (median |delta| <= 0.005).
    -> ops/a16_distance_split.py
    RUN 2026-08-30, §2162: pred_b HELD (rho 0.810, RISING with distance,
    peak 32-64 tokens), pred_c HELD, pred_a FAILED by 0.9% (near = 49.1%).
    Truncated-window NOT licensed; block 16 is document-scale memory.

69. IS HEAD 16.3 THE LONG-RANGE CARRIER (attribution; signed-sum bars;
    damage convention). Per-position damage of zeroing head 16.3 alone
    (on the skip-a16 config, bases excluded), split by distance as in
    rung 68. pred_a FAR CARRIES: distance > 32 carries >= 0.6 of head
    16.3's summed ind damage; pred_b rising profile (Spearman >= 0.5);
    pred_c the skip-a16 arm reproduces §2146 per-window (median |delta|
    <= 0.005). Tripwire: the zeroed arm's CE must differ from base.
    Null: 16.3's damage has the same ~49/51 profile as the whole module.
    -> ops/a16_head3_distance.py
    RUN 2026-08-31, §2163: pred_b/c HELD, pred_a FAILED (far = 55.2%) —
    the null wins: 16.3 is a scaled copy of the module's profile (~34%
    of the ind damage), not a distance specialist.

70. a14's DISTANCE PROFILE (division of labor or redundancy; signed-sum
    bars; damage convention). §2162: a16 rises with distance. If the two
    expensive layers divide labor, a14 is the short-range reader.
    pred_a near(<=32) carries >= 0.6 of a14L's summed ind damage; pred_b
    peak mean bin at distance < 16; pred_c skip-a14 reproduces §2155's
    base per-window (median |delta| <= 0.005). Null: same rising profile
    (redundant document memory). -> ops/a14_distance_split.py
    RUN 2026-08-31, §2164: pred_c HELD, pred_a/b FAILED — the null wins:
    a14's profile is the SAME rising shape as a16's (peak 32-64; near
    49.2%). Redundant document memory at two layers; the tail
    attribution program closes (rungs 55-70).

71. ARE attn14/attn16 ACTUALLY REDUNDANT (interaction test; damage
    convention). Zero each and both (throughout, bases excluded) on the
    skip-1416 config. pred_a sub-additive: d(both) <= 0.85 x
    (d(z16)+d(z14)) on FR; pred_b the overlap is on ind (I_ind <= -0.10
    x min single); pred_c base reproduces §2156 (|delta| <= 0.005).
    Null: near-additive independent reads.
    -> ops/attn1416_interaction.py [QUEUED 00:25Z]
    RUN 2026-08-31, §2165: pred_c HELD, pred_a/b FAILED AS WRITTEN — the
    bars encoded the wrong interaction sign for 'redundant'. Measured:
    strongly SUPER-additive (+0.058 FR / +895 ind) = the ACTIVE-BACKUP
    signature; cross-base corroboration (z16 costs less with a real
    attn14). Rule: derive the predicted interaction sign in the header.

72. IS m16's ANTI-RETRIEVAL COUPLED TO attn16 (interaction; damage
    convention). A = m16's ind effect with attn16; B = without.
    pred_a A <= -100 (S2159 replicates on this base); pred_b B - A >=
    0.5|A| (coupling); pred_c base reproduces §2156. Null: B ~ A.
    -> ops/m16_attn16_coupling.py [QUEUED 00:25Z]

    RUN 2026-08-31, §2166: pred_a/c HELD, pred_b FAILED in REVERSE —
    m16's anti-retrieval GROWS without attn16 (A=-395, B=-670): a
    general retrieval suppressor, not a coupled pair. The interaction
    chapter closes; the tail's skeleton is fully measured.

73. WHICH CLASSES PAY AT THE CLIFF (the tail instrument moves to the
    front; damage convention). Per-class attribution of the block-5
    motif replacement (full vs block-5-motifs-real, leave-one-real).
    pred_a NOT retrieval (ind <= 0.30); pred_b structural classes
    {newline,sentend,comma,bclose} >= 0.30; pred_c full reproduces
    §2144 (2.6662 within 0.01). Null: retrieval again.
    -> ops/attn5_motif_class.py [QUEUED 00:48Z]

    RUN 2026-08-31, §2167: pred_c HELD, pred_a FAILED by 0.022, pred_b
    FAILED — the null wins: 84.5% retrieval-class at the cliff too, but
    OTHER-led (0.464) where the tail was ind-led. Ordering differs, kind
    does not. Distance law at the cliff -> rung 74.

74. THE DISTANCE LAW AT THE CLIFF (damage convention; signed-sum bars).
    Split the block-5 motif damage's ind portion by distance to the
    target's last occurrence. Motif heads are fixed-pattern and
    plausibly LOCAL: pred_a near(<=32) carries >= 0.6 of the summed ind
    damage; pred_b peak mean bin at distance < 16; pred_c full arm
    reproduces §2144 (2.6662 within 0.01). Null: the same rising 32-64
    profile as the tail (no depth separation). If pred_a/b hold, the
    model divides retrieval by DEPTH — the first cross-depth
    architectural law. -> ops/attn5_motif_distance.py [QUEUED]

    RUN 2026-08-31, §2168: pred_c HELD, pred_a FAILED (56.5%), pred_b
    held on a weak 2^0 bin (flagged). The profile is FLAT — the front
    pays uniformly at every distance; the tail rises. Law: depth adds
    RANGE (generic context-mixing in front, document-scale read at
    14/16).

75. WHICH BLOCKS CARRY THE MOTIF PRICE (leave-one-real sweep; damage
    convention). §2167: block 5's motif replacement costs ~+0.064 FR.
    Sweep motif_off=(b,) for b in 2..9 (nine arms incl. full).
    pred_a CONCENTRATION: max-block FR damage >= 2 x median-block;
    pred_b THE CLIFF LEADS: block 5 is the max; pred_c full reproduces
    §2144 (2.6662 within 0.01). Null: uniform ~+0.02/block.
    -> ops/motif_leave_one.py [QUEUED 01:00Z]
    RUN 2026-08-31, §2169: pred_a/c HELD, pred_b FAILED — the price is
    FRONT-LOADED (b2-b4 ~ +0.17-0.19 each; the cliff's b5 only +0.065;
    b7-b9 nearly free). Cliff and price are different maps. Marginals
    sum +0.741 >> joint (sub-additive lattice; joint = open number).

76. THE POSITION LAW OF THE other-CLASS DAMAGE (cliff vs tail; damage
    convention). Novel targets have no occurrence distance; absolute
    position measures available context. pred_a the range law extends
    (tail late-half share >= cliff's + 0.10); pred_b both late shares
    >= 0.50; pred_c full reproduces §2144. Null: both position-flat.
    -> ops/other_position_law.py [QUEUED 01:25Z]
    RUN 2026-08-31, §2170: pred_a/c HELD, pred_b FAILED — sharper than
    registered: cliff EARLY-concentrated (0.393), tail LATE (0.619).
    Front mixes while context is scarce; tail remembers once it exists.

77. MINIMAL REALIZATION TEST FOR THE BACKUP PAIR (math review 0140;
    damage convention n/a - alignment test). Capture attn14/attn16
    outputs on FR ind positions (skip-1416 config, both real). pred_a
    mean top-8 canonical correlation >= 0.6 on ind; pred_b the half-A
    Procrustes map keeps >= 0.8 of its R^2 on half B (SHARED read);
    pred_c non-ind alignment lower by >= 0.2. Null: aligned but
    non-transferring (independent duplicates).
    -> ops/backup_pair_cca.py [QUEUED 01:25Z]
    RUN 2026-08-31, §2171: pred_a HELD (0.933, but generic — control
    failed at 0.927 non-ind), pred_b FAILED (transfer 0.106 vs 0.926):
    INDEPENDENT DUPLICATES — no shared-read compilation licensed; the
    backup is an ensemble of independent readers.

78. THE SUPPRESSOR AS AN OPPONENT PROCESS (math review 0140). Same
    captures + m16 output: cos(logit-effect of m16, logit-effect of
    attn14+attn16) on ind positions. pred_a cos <= -0.3 on ind; pred_b
    |cos| <= 0.15 on non-ind; pred_c repro gate. Null: suppression is
    routed, not direct logit opposition.
    -> ops/m16_opponent.py [QUEUED 01:28Z]
    RUN 2026-08-31, §2172: pred_c HELD, pred_a/b FAILED — anti-alignment
    mild and GENERIC (-0.143 ind vs -0.152 non-ind): no direct logit
    brake; the suppression is ROUTED. Both math-review compilations
    (77-78) returned their nulls: the tail resists linear identification
    and logit-local opposition — per-document binding + routed
    composition are the recorded design constraints.

79. THE JOINT MOTIF NUMBER (accounting; damage convention). §2169
    flagged the joint all-motifs-real value as open (marginals sum
    +0.741, strongly sub-additive). pred_a d(all) <= 0.5 x 0.741;
    pred_b d(all) >= max single (+0.1888); pred_c full reproduces
    §2144. Null: additive (~+0.741).
    -> ops/motif_joint.py [QUEUED 01:56Z]
    RUN 2026-08-31, §2173: pred_b/c HELD, pred_a FAILED by 0.042 —
    joint +0.4122 (56% of the marginal sum; interaction -0.329):
    moderately sub-additive. The motif program (+0.412) and the tail
    dictionaries (~+0.35) carry essentially the whole frontier price.

80. THE PRICE LEADER'S SIGNATURE (block-2 motif class+position map;
    damage convention). §2169: block 2 leads the motif price (+0.189).
    pred_a retrieval classes >= 0.60; pred_b early-concentrated (late
    share <= 0.45); pred_c full reproduces §2144. Nulls: block-5-like.
    -> ops/attn2_motif_class.py [QUEUED 02:10Z]

    RUN 2026-08-31, §2174: pred_a/c HELD, pred_b FAILED — SUBWORD-led
    (0.453; ind 0.198, other 0.205), mildly late (0.568). Third
    signature; depth progression: b2 subword -> b5 other -> tail ind
    (assembly -> context -> memory). §2170's early law was b5-specific.

81. DO b3/b4 INTERPOLATE (front signature map; damage convention).
    Class shares of the block-3 and block-4 motif replacements (arms:
    full / no3 / no4). pred_a b3's top class in {subword, other};
    pred_b subword share declines monotonically b2 > b3 > b4 (the
    assembly->context progression); pred_c full reproduces §2144.
    Null: no ordering (idiosyncratic signatures).
    -> ops/attn34_motif_class.py [QUEUED 02:20Z]

    RUN 2026-08-31, §2175: ALL THREE HELD — monotone subword decline
    0.453 > 0.448 > 0.393, other rising toward b5. Assembly -> context
    -> memory is a measured cross-depth law; the front chapter closes
    (rungs 73-81).

82. DOES THE FRONT ASSEMBLE WHAT m16 FINISHES (cross-depth pipeline
    test; damage convention). Per-position: b2's replacement damage vs
    m16's deletion help, on subword positions (arms: full / block-2
    motifs real / m16 zeroed). pred_a Spearman rho >= 0.3 between b2
    damage and m16 help across subword positions (binned by 200);
    pred_b m16's subword help concentrates on positions where b2's
    damage is above-median (top-half carries >= 0.6); pred_c full
    reproduces §2144. Null: independent subword services.
    -> ops/front_m16_pipeline.py [QUEUED 02:50Z]

    RUN 2026-08-31, §2176: ALL THREE HELD — rho +0.850, top-half 0.683:
    the subword pipeline (b2 assembles, m16 finishes) is measured. Two
    named pipelines now thread the model (subword; retrieval+brake).

83. DO THE BACKUP READERS COVER THE SAME POSITIONS (ind-position
    co-variation; damage convention). §2171: independent duplicates by
    map non-transfer. Complementary question: per-position damage
    profiles of z16 vs z14 on ind (arms: skip-1416 base / z16 / z14).
    pred_a binned Spearman(d16, d14) >= 0.5 (same-position backup);
    pred_b top-half of d16 carries >= 0.6 of d14's sum; pred_c base
    reproduces §2156 (|delta| <= 0.005). Null: position PARTITION
    (each covers different repeats; rho <= 0).
    -> ops/backup_positions.py [QUEUED 02:55Z]
    RUN 2026-08-31, §2177: ALL THREE HELD — rho +0.689, top-half 0.914:
    SAME-position backup; the pair is an N=2 replicated ensemble over one
    function (partition null dead). One read per position suffices for a
    compiled primitive; per-document binding carries the burden.

84. THE CONTEXT-TO-MEMORY EDGE (cross-depth pipeline; damage
    convention). Do the b5 and a16 replacements fail at the same
    positions (shared failure points = a chained pipeline)? Arms: full /
    no5 / skip-a16; binned Spearman of the two damages. pred_a rho >=
    0.3 on other; pred_b rho >= 0.3 on ind; pred_c full reproduces
    §2144. Null: independent failures.
    -> ops/context_memory_edge.py [QUEUED 02:00Z]

    RUN 2026-08-31, §2178: ALL THREE HELD — other rho +0.789, ind
    +0.461: a chained pipeline. Both diagram edges measured; m16 is the
    intersection (helper on subword, brake on retrieval) — explaining
    its resistance to simple interfaces.

85. ARE THE ASSEMBLERS ALSO A REPLICATED PAIR (front symmetry test;
    damage convention). §2177: the 14/16 readers are an N=2 same-
    position ensemble. Are b2/b3 (subword shares 0.453/0.448) the same
    motif? Arms: full / no2 / no3; binned Spearman of the two damages
    on subword positions. pred_a rho >= 0.5; pred_b top-half of d2
    carries >= 0.6 of d3's sum; pred_c full reproduces §2144. Null:
    division of assembly labor (low rho).
    -> ops/assembler_pair.py [QUEUED 02:08Z]

    RUN 2026-08-31, §2179: ALL THREE HELD — rho +0.998 (near-identity;
    tighter than the tail's 0.689). Replication is an architectural
    motif: N=2 same-position pairs at both ends. Interaction signature
    -> rung 86.

86. IS THE FRONT PAIR ALSO MUTUAL BACKUP (interaction signature; damage
    convention; sign derived pre-run per §2165: replicated backup =>
    SUPER-additive joint removal). Base: §2144 config with blocks 2/3
    motifs REAL; zero real attn2, attn3, both (bases excluded). Plus a
    full arm for the §2144 repro. pred_a super-additive:
    [L2_F(both) - base] >= 1.15 x ([z2 - base] + [z3 - base]); pred_b
    same-position: binned rho(dz2, dz3) >= 0.5 on subword; pred_c the
    full arm reproduces §2144 (2.6662 within 0.01). Null: additive
    (independent functions despite identical coverage).
    -> ops/assembler_backup.py [QUEUED 02:16Z]

    RUN 2026-08-31, §2180: ALL THREE HELD — 12.6x super-additive (+9.07
    vs +0.72 sum; z3 alone only +0.24): near-total mutual coverage; the
    replication law holds at both sites in different regimes. Sanity
    bounds correctly flagged the catastrophic arms. Transfer test (the
    tail's third signature) -> rung 87: the first merge opportunity.

87. DOES THE FRONT PAIR'S MAP TRANSFER (the merge-licensing test;
    alignment rung). §2171: the tail pair's cross-output map does NOT
    transfer (document-bound). The front's function is distance-flat and
    local (§2168) — prediction derived from the laws: the attn2->attn3
    output map IS document-independent. On the motifs-2/3-real base,
    capture attn2/attn3 outputs; pred_a subword CCA >= 0.6; pred_b the
    half-A linear map keeps >= 0.8 of its R^2 on half B (TRANSFER — the
    opposite of the tail); pred_c the base reproduces rung 86's base
    L2_F within 0.01. If pred_b holds, b2/b3 is the program's first
    legitimately MERGEABLE pair. Null: document-bound like the tail.
    -> ops/assembler_transfer.py [QUEUED 02:28Z]
    RUN 2026-08-31, §2181: pred_a/c HELD, pred_b FAILED hard (transfer
    R^2 = -1.03, worse than the mean; tail was 0.106) — the front is MORE
    document-bound. Candidate law: representations are DOCUMENT-GAUGED
    everywhere (function transfers, coordinates do not); every raw-
    coordinate linear compilation is doomed. Next math target: estimate
    the per-document rotation and test the QUOTIENTED map.

88. ARE THE CONTEXT MIXERS A PAIR TOO (pair census; damage convention).
    b4/b5's signatures already diverge (§2174/§2167), so the honest
    expectation is mixed. pred_a binned rho(d4,d5) >= 0.5 on other;
    pred_b top-half concentration >= 0.6; pred_c full reproduces §2144.
    Null: a depth GRADIENT, not a copy.
    -> ops/mixer_pair.py [QUEUED 02:28Z]
    RUN 2026-08-31, §2182: ALL THREE HELD — rho +0.949 on other (label
    in the print says subword; variable was repointed — cosmetic).
    Census complete: three adjacent-block same-position duos (0.998 /
    0.949 / 0.689, graded by depth); coverage and class emphasis
    dissociate.

89. THE GAUGE-QUOTIENT TEST (§2181's registered target). If per-doc
    maps W_d are conjugates of one shared M (document gauge), their
    singular SPECTRA match while raw entries decorrelate. pred_a mean
    spectrum cosine >= 0.95 AND mean raw-entry corr <= 0.5; pred_b
    median in-doc held-out R^2 >= 0.5; pred_c base reproduces rung 86's
    base (2.4410 within 0.01). Null: different functions per doc
    (spectra decorrelate too). -> ops/gauge_quotient.py [QUEUED 02:33Z]
    VALID RERUN, §2183: pred_a HELD (0.986/0.078) but DISQUALIFIED by
    pred_b's failure (pooled held-out R^2 -0.727: unvalidated maps ->
    shrinkage look-alike spectra). INCONCLUSIVE at row grain; the gauge
    question stays open on two transfer collapses. -> rung 90 DESIGN.

    First run 02:35Z VOID at the tripwire (the grouping unit is FR ROWS,
    ~28 subword positions each — the >=100 'document' threshold passed 1
    row; no measurement made). Fixed BEFORE any valid run: row grain
    (conservative for the document law), threshold 25, heavier ridge,
    pooled held-out R^2 bar 0.4, tripwire 30 rows. Re-queued 02:38Z.

90. DOCUMENT-GRAIN GAUGE TEST (DESIGN; the decisive conjugation
    instrument). Needs a capture that carries DOCUMENT ids and pools
    same-document rows (hundreds of subword positions per document) so
    per-document 64x64 maps are identifiable (held-out R^2 validation
    bar first, spectra second). Build: modify the FR/window construction
    to record doc ids, or capture over a handful of long documents in
    513-token strides. To be built at the next driver wake with fresh
    care (five splice bugs tonight).

91. OUTPUT-SUBSTITUTION MERGE, COORDINATE-FREE VERSION (user-suggested
    direction; damage convention). Fixed linear output maps are dead OOD
    (§2181, gauge); dropping attn3 costs +0.237 (§2180). Test the
    gauge-immune substitute: a scalar gain on attn2's output, grid-fit
    on window C under the drop-attn3 build. pred_a the scalar recovers a
    third of the drop cost; pred_b alpha* in [1.05,1.8]; pred_c base
    reproduces rung 86 (2.4410). Null (§2126): scalars aren't levers.
    -> ops/merge_scalar.py [QUEUED 02:52Z]
    RUN 2026-08-31, §2184: ALL THREE HELD — alpha*=1.45 recovers 55%
    (+0.2367 -> +0.1065; smooth curve). First held construction since
    the §2161 pause; §2126's scalar null SCOPED (amplitude substitution
    for a replicated twin works; fixing a mis-driven head does not).

92. TWIN ANATOMY (user questions; damage convention). Mean-ablate attn2
    and decompose: arms base / mean2 / path_only (y2 visible only to
    attn3) / direct_only (y2 everywhere except attn3); capture y3's
    response. pred_a the 2->3 attention path carries half; pred_b y3
    actively changes (median rel >= 0.10) vs passive redundancy; pred_c
    base reproduces rung 86. Descriptive: per-position twin direction
    cosines; compensation direction; mean-vs-zero ablation gap.
    -> ops/twin_anatomy.py [QUEUED 03:00Z]

    RUN 2026-08-31, §2185: pred_c HELD; pred_b HELD-as-scored but
    reinterpreted (serial sensitivity, compensation cos -0.09 ~ 0: no
    directed self-repair); pred_a VOID (pre-hook on attn sees the
    RMS-NORMED stream - raw-scale delta mis-scaled; both path arms
    +5.5-5.7). Findings: 73% of knockout damage is the MEAN term;
    twins write DIFFERENT directions (cos ~0.22 generic). Fix at block
    grain -> rung 93.

93. PATH DECOMPOSITION AT BLOCK GRAIN (rung 92's fix; damage
    convention). The block receives the RAW stream, so inject there:
    forward_pre_hook on h[3] (visible to attn3+m3 — granularity change
    stated). Arms: base / mean2 / path_only / direct_only. pred_a
    [L2_F(path_only) - base] <= 0.5 x [L2_F(mean2) - base]; pred_b
    [L2_F(direct_only) - base] <= 0.5 x [L2_F(mean2) - base] is NOT
    also allowed to hold if pred_a holds strongly... registered simply:
    pred_b direct_only >= path_only (the 2->3 path carries more than
    the residual path); pred_c base reproduces 2.4410 within 0.01.
    Null: residual path dominates. Sanity: both path arms must lie in
    [0, mean2 + 0.05].
    -> ops/twin_anatomy2.py [QUEUED 03:15Z]
    RUN 2026-08-31, §2186: ALL THREE HELD — the 2->3 path carries 84%
    of attn2's per-position value (path_only +0.0205 vs mean2 +0.1295;
    residual path 26%; near-additive). The twins are a two-stage local
    circuit, each stage individually sufficient — a single distilled
    b2/b3 unit is the right merged object.

94. MEAN-ABLATION SUITE FOR ALL THREE PAIRS (user-requested; damage
    convention). §2185: 73% of attn2's zero-ablation damage was the mean
    term. Redo all knockout interactions with mean-ablation: b2/b3,
    b4/b5 (previously unmeasured directly), 14/16 — 12 arms. pred_a the
    mean term dominated the old numbers (mean-both <= 0.5 x zero-both);
    pred_b the backup law survives the DC correction (>= 1.15x at all
    three pairs; null: it was mean-term compounding); pred_c b23 base
    reproduces 2.4410. -> ops/twin_mean_suite.py [QUEUED 03:25Z]
    RUN 2026-08-31, §2187: pred_c HELD, pred_a/b FAILED — the corrected
    table: assemblers ADDITIVE at signal level (0.955; the 12.6x was DC
    compounding; scope notice on §2180); mixers 1.49 and readers 1.50
    genuinely super-additive. Three duos, three internal economies; the
    assembler means are jointly load-bearing (+9 nats).

95. DOES THE SCALAR MERGE GENERALIZE (damage convention). §2184's
    drop-one + survivor-gain construction at the other two pairs: 14/16
    (drop attn16, scale attn14) and b4/b5 (drop attn5, scale attn4);
    alpha on window C. pred_a/b recovery >= 30% at each pair; pred_c
    skip-1416 base reproduces §2156 (2.4230). Null: front-specific (the
    tail's weaker coverage may not support amplitude substitution).
    -> ops/merge_scalar_pairs.py [QUEUED 03:27Z]
    RUN 2026-08-31, §2188: pred_c HELD, pred_a/b FAILED — recovery 19%
    (14/16) and 7% (b4/5): the scalar merge is ASSEMBLER-specific,
    exactly as §2187's DC story predicted. Drop-attn5 costs +0.85.

96. CIRCUIT-GRAIN VALIDATION OF THE SCALAR MERGE (user-suggested
    cross-view; real-model frame, census grid). Does zero-a3 + 1.45xa2
    repair the a3-localized circuits' members, or is the aggregate
    recovery a CE hack? pred_a a3-circuit median member |dCE| under
    merge <= 0.6 x under drop; pred_b no collateral circuit breakage
    (>= 0.5 x its battery reference); pred_c drop >= 0.8 x battery
    mean-ablation refs (consistency anchor). Null: aggregate recovery
    without circuit repair. -> ops/merge_circuit_grain.py [QUEUED 03:48Z]
    RUN 2026-08-31, §2189: pred_c HELD, pred_a FAILED (24% vs 40%),
    pred_b FAILED-as-registered (conflated: ZERO circuits worsened by
    the scaling; all 48 improve, median ratio 0.79 — the breakage is
    the DROP, which damages ~77% of circuits). The merge is an
    aggregate trade, not a circuit-preserving compilation. Circuit
    grain adopted as a standing validation layer (16s check).

97. THE DC LEDGER (damage convention). §2187: the assemblers' DC terms
    carried ~98% of joint knockout damage. Generalize: zero- vs mean-
    ablation for a8 (most circuit-dense), a5 (cliff), m13-m16 (band);
    DC share = 1 - d(mean)/d(zero). pred_a median share >= 0.5; pred_b
    a8 >= 0.5; pred_c base reproduces §2144. Null: b2/b3-specific.
    -> ops/dc_ledger.py [QUEUED 03:55Z]
    First run 04:20Z VOID at the tripwire — and the a8/a5 arms were
    structurally inert: on the S2144 base those attentions are REPLACED
    by motif hooks, so ablating the real modules does ~nothing (a8 zero
    -0.008!). Lesson: DC ledgers belong in the REAL-model census frame
    (the battery's frame). Rebuilt as dc_ledger2 (census, ~16s/arm).

98. THE DC STAND-IN (damage convention; §2187-licensed). Replace attn3
    with its MEAN VECTOR (price: 1,152 values) instead of zero, with an
    optional small gain on attn2. pred_a the mean arm reproduces
    §2187's meanB (+0.0655 within 0.01, cross-run anchor); pred_b
    mean3 + alpha* <= +0.05 (a further ~25% recovery); pred_c base
    reproduces 2.4410. Null: the gain adds nothing on top of the mean
    (§2187: the signal costs are additive at b2/b3).
    -> ops/dc_standin.py [QUEUED 04:12Z]
    RUN 2026-08-31, §2190: pred_b/c HELD — the DC stand-in costs only
    +0.0466 at 1,152 values (beats the scalar merge and the drop); the
    a2-gain adds nothing (alpha*=1.0 on window C; §2187 null holds).
    pred_a's anchor FAILED: the mean is estimator-dependent (~0.02) —
    broad-population means are better stand-ins; state the estimation
    population henceforth.

99. CIRCUIT-GRAIN CERTIFICATION OF THE FRONTIER CONFIG (math review
    0437; causal-abstraction criterion). Install the §2144 assembly
    hooks under the census rows and score all 62 circuits' member
    mean|dCE|. pred_a the config is abstraction-valid at tau = 0.5 x
    battery ref for >= 40 of 62 circuits; pred_b the failures
    concentrate on retrieval-class circuits (ind/other member
    majorities); pred_c the assembly's aggregate census dCE is finite
    and positive (sanity). Null: broad breakage beyond retrieval.
    BUILT 04:55Z: the frontier evalV machinery evaluates the census ROWS
    directly (no port of install needed); census_lib supplies rows,
    base CE and member masks in-process.
    -> ops/frontier_certificate.py [QUEUED 04:55Z]
    RUN 2026-08-31, §2192: pred_c HELD, pred_a FAILED (0/62 valid at
    tau=0.5xref; majority-valid only near ~3xref), pred_b DEGENERATE
    (universal failure set; §2159 precedent). The frontier config is an
    aggregate approximation, not a causal abstraction — now a measured
    per-circuit fact with a tau-curve.

97b. THE DC LEDGER, CENSUS FRAME (rebuild). Real-model zero- vs mean-
    ablation for a8/a5/m13/m14/m15/m16 over the census grid; DC share =
    1 - d(mean)/d(zero) on aggregate |dCE|... using signed mean dCE per
    arm. pred_a median DC share >= 0.5; pred_b a8 >= 0.5; pred_c
    anti-inertness: every zero-arm aggregate mean dCE >= +0.02.
    -> ops/dc_ledger2.py [QUEUED 04:45Z]

    RUN 2026-08-31, §2191: pred_c HELD, pred_a/b FAILED — BIMODAL:
    DC-heavy {a5 0.94, m16 0.83, m15 0.71} vs pure-signal {a8 -0.01,
    m13 -0.42, m14 -0.03}. The cliff's knockout is 94% its mean; a8 is
    pure signal (behind the gauge gate). Affine skeleton licensed for
    the DC-heavy class only.

100. DOES THE ENVELOPE RESTORE THE a16 CIRCUITS (certificate for
    skip-1416; damage convention). Rerun the certificate with the
    skip-1416 config (both tail attentions real). pred_a the
    a16-localized circuits' median damage ratio improves >= 2x vs rung
    99; pred_b >= 10 circuits become valid at tau = 0.5xref; pred_c
    aggregate in [0.5, 5]. Null: the breakage is front-caused and the
    tail retreat restores little.
    -> ops/frontier_certificate2.py [QUEUED 04:40Z]

    RUN 2026-08-31, §2193: pred_c HELD, pred_a/b FAILED — the envelope
    restores almost nothing (a16 circuits 1.20x; 0/62 valid): the
    breakage is FRONT-caused (motif dictionaries), consistent with
    §2169/§2189. -> rung 101: certify the all-motifs-real config.

101. CERTIFICATE FOR THE ALL-MOTIFS-REAL CONFIG (the decisive front
    attribution; damage convention). Front attention real, tail
    dictionaries in place. pred_a >= 30 of 62 circuits valid at tau =
    0.5xref (large recovery, per §2169/§2189); pred_b the still-failing
    circuits are majority a16/m16-localized; pred_c aggregate in
    [0.3, 5]. Null: breakage persists (the CP middles/front tables are
    also circuit-breaking). -> ops/frontier_certificate3.py [QUEUED
    04:45Z]

    RUN 2026-08-31, §2194: pred_c HELD, pred_a/b FAILED — still 0/62
    with all motifs real; residue not tail-localized (0.32). EVERY
    replacement family breaks circuits independently. Last cheap point:
    rung 102 (tables+CP-middles only).

102. CERTIFICATE FOR TABLES+MIDDLES ONLY (the curve's last cheap point;
    damage convention). Front attention real AND all tail attention
    real (motif_off 2-9, skipset 10-17): only front tables and CP
    middles replaced. pred_a >= 20 of 62 valid at tau = 0.5xref; pred_b
    aggregate <= 2.0; pred_c aggregate in [0.2, 5]. Null: still 0/62 —
    no partial replacement preserves the certified circuits.
    -> ops/frontier_certificate4.py [QUEUED 04:50Z]

    RUN 2026-08-31, §2195: still 0/62 with ALL attention real; the
    four-point certificate curve completes (2.855/2.570/2.431/2.073;
    0/0/0/0). No partial replacement family preserves the circuits.
    Hand-off: repair-targeted compilation, or two-ledger accounting.

103. TABLES-ONLY CERTIFICATE (completing the family decomposition;
    damage convention). K = K69 = 4608 keeps ALL CP units (middles
    effectively real), motifs off 2-9, tail skipped: ONLY the front
    tables m0-m3 replaced. pred_a >= 30 of 62 valid at tau = 0.5xref;
    pred_b aggregate <= 1.0; pred_c sanity [0.1, 5]. By subtraction
    with rung 102, the CP middles' marginal circuit damage.
    Null: even the tables alone break everything.
    -> ops/frontier_certificate5.py [QUEUED 04:55Z]

    RUN 2026-08-31, §2196: pred_c HELD, pred_a/b FAILED — the TABLES
    alone cost +1.924 and break every circuit: the fold-table base was
    the elephant, invisible because everything was measured above it.
    Family attribution: tables ~1.92 >> motifs ~0.36 > tail ~0.29 >>
    middles ~0.15. Certificate chapter closes (99-103; five 0/62s).
104. Front prefix decomposition + tailE control (census certificates). Arms:
    ctrl_no_tailE (cfgF minus tailE = TRUE tables-only), tailE_only, and six
    cumulative front prefixes a0/m0E/a1v/m1/m2E/m3E. pred_a agg(ctrl) >= 1.4
    (tables carry the bulk — guards S2196; FAIL => published correction);
    pred_b max front marginal >= 0.5 x front-full; pred_c front-full in
    [0.5, 2.2]. Null: damage spread AND tailE material. Tripwire: prefix
    pstdev >= 0.01.
    -> ops/frontier_front_prefix.py [QUEUED 05:04Z]
    RUN 2026-08-31, S2198: pred_a/c HELD, pred_b FAILED. Control holds:
    tables alone +1.7474 (0/62), tailE-only +0.1440 (5/62). a0 EXACT
    (62/62); m0E alone breaks 61/62; marginals spread with depth.

105. Front error anatomy (repair-targeted diagnostic): E = replaced - real
    module on the same input, full-front frame, 200 census rows; rel, dcfrac,
    per-class rel per site. pred_a dcfrac <= 0.15 on >= 5/6 sites; pred_b at
    max-rel site link {ind,other} rel >= 1.3 x const-class rel; pred_c all
    rel in [0.02, 1.5]. Null: DC-contaminated and class-uniform. Tripwire:
    max rel < 0.02 = inert.
    -> ops/frontier_front_anatomy.py [QUEUED 05:04Z]
    RUN 2026-08-31, S2199: all three preds FAILED (nulls carry it):
    rel errors 0.51-0.81 at m0E-m3E vs a0 0.0003; dcfrac <= 0.19;
    class-uniform. Repair must be context-signal; DC/class routes dead.

106. Front singles: each of m0E/a1v/m1/m2E/m3E installed ALONE (all else
    real), census certificates per arm. pred_a sum(singles)/front-full in
    [0.6, 1.0]; pred_b agg(m2E single) >= 1.5 x agg(m0E single); pred_c
    all singles in [0.02, 1.2]. Null: prefix marginals were mostly input
    drift (sum << full). Tripwire: pstdev >= 0.005.
    -> ops/frontier_front_singles.py [QUEUED 05:32Z]
    RUN 2026-08-31, S2200: pred_a/c HELD, pred_b FAILED. Singles:
    m0E .2485/1, a1v .0520/11, m1 .2628/2, m2E .3072/1, m3E .4850/1
    (agg/valid). Depth law was input drift; ~22% superadditive; every
    MLP table alone kills 60+/62 certificates.

107. m2 residual-rank capacity curve: tableres refit in the m2 prefix frame
    at residual ranks 0/16/64/256/512, each as a SINGLE-site arm on census
    rows. pred_a rank wall: gain(0->64) >= 2 x gain(64->512); pred_b
    agg(512) >= 0.5 x agg(64) (grammar cannot buy its way out); pred_c
    monotone + all in [0.05, 1.5]. Null: capacity path OPEN (repair =
    bigger residual). Price: rank-512 at one site ~= 512 x ~1850 values.
    -> ops/frontier_m2_rank.py [QUEUED 05:32Z]
    RUN 2026-08-31, S2201: pred_b HELD, pred_a/c FAILED. Rank curve
    FLAT (.314/.337/.307/.287/.284 at 0/16/64/256/512; valid 1/62 all).
    Capacity branch DEAD - feature wall, not rank wall: the residual
    carries ~0.03 of m2's +0.31 at any rank.


108. m0 certificate-targeted repair (S2195 hand-off, objective branch):
    refit m0 residual (same quadfeat rank-64 grammar) on census TRAIN half,
    unweighted vs member-weighted (w=10 on any-circuit members); three
    single-site arms scored on TEST half. pred_a median memberabs ratio
    unw/wt >= 1.5; pred_b refit_wt valid >= 10; pred_c agg(wt) <= 2 x
    agg(unw) and plain within 0.10 of S2198 anchor. Null: weighting moves
    nothing (capacity-limited). Tripwire: bitwise-equal cev = inert.
    -> ops/frontier_m0_repair.py [QUEUED 05:36Z]
    RUN 2026-08-31, S2202: pred_a/b FAILED (weighting no-op: ratio
    0.995, valid 3/62 both refits), pred_c HELD. Side finding: census
    refit halves m0 damage (.2499->.1244). Objective branch DEAD;
    with S2201 the table+featmap grammar is falsified at the front.


109. CP-vs-table at m2 (grammar change, model's own units): single-site
    top-K bilinear units (norm importance, weights-only), K in
    288/576/1152/2304/4608, census certificates. pred_a agg(576) <= 0.5 x
    table anchor 0.3072; pred_b valid(1152) >= 10/62; pred_c monotone +
    K4608 exact in [-0.01, 0.02]. Null: front damage is site-hardness,
    not grammar. Price: CP-576 = 2.0M values vs table 57.9M (29x).
    -> ops/frontier_m2_cp.py [QUEUED 05:44Z]
    RUN 2026-08-31, S2203: ALL THREE HELD. CP crushes the table at m2:
    576 -> +0.0822/7 valid (29x fewer values); 2304 -> +0.0207/29;
    4608 exact 62/62. Front-table problem dissolves at m2.

110. CP-vs-table at m0 (the certificate-killer site): same design at
    h[0].mlp; anchor single m0E +0.2485 / valid 1 of 62 (S2198/S2200).
    pred_a agg(CP-576) <= 0.5 x 0.2485; pred_b valid(1152) >= 10/62;
    pred_c monotone + K4608 exact in [-0.01, 0.02]. Null: site-hardness.
    Price: as rung 109.
    -> ops/frontier_m0_cp.py [QUEUED 05:47Z]
    RUN 2026-08-31, S2204: pred_a/b FAILED (576 is 4.5x WORSE than the
    table), pred_c HELD. But K=2304 dominates: +0.0647/9 valid at 7x
    fewer values. m0 = hard site, half-prunable; depth inverted.


111. CP-vs-table at m1 and m3 (K maps for the CP front): single-site top-K,
    K in 288/576/1152/2304/4608, census certificates. pred_a agg(2304) <=
    0.5 x singles anchor at BOTH (m1 .2628, m3 .4850); pred_b valid(2304)
    >= 5 at both; pred_c monotone + 4608 exact. Null: m0-like hardness +
    grammar loses somewhere. Price: 7.96M values/site at 2304.
    -> ops/frontier_m13_cp.py [QUEUED 05:59Z]
    RUN 2026-08-31, S2205: ALL THREE HELD. m1 hard (2304: .0454/12),
    m3 easy (288: .1001/6). Depth law: hardness falls monotonically
    with depth (576: m0 1.108 > m1 .280 > m2 .082 > m3 .075).

112. THE CP-FRONT CONFIG: m0-m3 as CP (K from rung 111 at run time: smallest
    K with agg <= 0.10, else 2304; defaults m0 2304/m1 2304/m2 1152/m3
    2304) + exact a0 + a1v table, all else real. pred_a agg <= 0.5 (vs
    table front +1.7474); pred_b valid >= 5/62 (first multi-site config
    above 0); pred_c agg in [0.05, 1.0]. Null: superadditive compounding
    destroys single-site gains. Price: ~27.9M values vs 231.6M (8.3x).
    -> ops/frontier_cp_front.py [QUEUED 05:59Z]
    RUN 2026-08-31, S2206: pred_a/b FAILED, pred_c HELD. Combined
    +0.9427/0 valid = 3.3x singles sum (+0.2870); compounding is the
    new enemy. Ks used: m0/m1 2304, m2 1152, m3 576 (288 missed the
    0.10 bar by 0.0001). Still 1.85x better than tables at 10.6x fewer
    values.


113. Compounding anatomy (CP-front prefixes): arms base(a0,a1v), then
    cumulative +cf0/+cf1/+cf2/+cf3 at the S2206 Ks (2304/2304/1152/576).
    excess_i = marginal_i - single_i (singles from rung 109-111 JSONs).
    pred_a max excess_i >= 0.5 x total excess; pred_b marginal(cf1) >=
    2 x single(m1) (+0.0908) - compounding enters at m1; pred_c full
    prefix reproduces +0.9427 +/- 0.015. Null: excess distributed
    (~2-3x uniformly). Price: none.
    -> ops/frontier_cp_prefix.py [QUEUED 06:28Z]
    RUN 2026-08-31, S2207: ALL HELD (pred_a by the letter). Excess
    ladder -0.008/+0.068/+0.240/+0.357; marginal multipliers 1.0/2.5/
    5.8/5.8x singles. Progressive, not one entry point.

114. K-escalation (does K buy back the compounding): all four sites at
    K=2304 and at K=3456. pred_a agg(all2304) <= 0.4 (collapse ~0.2 vs
    persist ~0.68 = 3.3x singles sum); pred_b valid(all2304) >= 5;
    pred_c agg(all3456) < agg(all2304) and <= 0.20. Null: compounding
    regime persists. Price: 31.9M / 47.8M values (vs tables 231.6M).
    -> ops/frontier_cp_front2.py [QUEUED 06:28Z]
    RUN 2026-08-31, S2208: pred_a/b FAILED, pred_c HELD. all-2304
    +0.4887/1; all-3456 +0.1193/2. Aggregate K-buyable, certificates
    not. CP price curve strictly dominates tables at every point.

115. Stream-drift ledger of the CP front: r_li = rms(h_cfg - h_real)/
    rms(h_real) at blocks 0-9, 200 census rows; arms cp_front (S2206 Ks)
    + four CP singles. pred_a r3(front) >= 1.5 x sum r3(singles) (drift
    superadditive); pred_b r9 >= 0.8 x r3 (persists into middles); pred_c
    singles <= front at block 3, all r in [0.0005, 2]. Null: drift
    additive (excess lives in CE curvature -> K-escalation should work).
    Price: none. Tripwire: r3(front) < 0.005 inert.
    -> ops/frontier_cp_drift.py [QUEUED 06:32Z]
    RUN 2026-08-31, S2209: all FAILED; null stronger than registered.
    Drift QUADRATURE (0.862 pred vs 0.833 obs) - orthogonal site
    drifts; middles contract 2.6x while CE persists. CE excess =
    loss-surface geometry. pred_c range clause = instrument artifact
    (pre-site zeros below floor).


116. Sequential-conditioning refit of the CP front (the tables' 1.29x
    regime vs CP's 3.3x): Down+bias refit per site in the drifted prefix
    frame, chained, census train rows 0-499/2; arms plain / seq_traj
    (real-trajectory target) / seq_frame (in-frame target), TEST-half
    certificates. pred_a agg(seq_traj) <= 0.45 x agg(plain); pred_b
    valid(seq_traj) >= 5; pred_c seq_frame <= plain AND plain within
    0.05 of +0.9427. Null: error lives in unit activations, not the
    readout. Price: none (Down swap + D-bias/site).
    -> ops/frontier_cp_seqfit.py [QUEUED 06:59Z]
    RUN 2026-08-31, S2210: pred_a FAILED by 0.016 (seq_traj 0.467x vs
    bar 0.45x), pred_b FAILED (1/62), pred_c HELD. traj > frame target;
    Pareto win on aggregate (matches all-2304 at 0 extra values).

117. Attention-splice mechanism test: a2/a3 fed REAL position-aligned
    input streams (symmetric pre-hook capture, rung-92 lesson), rest =
    mixed-K cp_front. pred_a agg(spliced) <= 0.6 x agg(plain); pred_b
    valid(spliced) >= 3; pred_c |agg(plain) - 0.9427| <= 0.015. Null:
    attention is a bystander (residual stream transmits the drift).
    Price: none (mechanism test). Tripwire: bitwise-equal arms.
    -> ops/frontier_attn_splice.py [QUEUED 06:59Z]
    RUN 2026-08-31, S2211: pred_a/b FAILED, pred_c HELD. Splice removes
    31% - attention is a partial conduit; residual stream carries the
    majority. Neither mediator nor bystander bar held.


118. Pairwise interaction expansion (order-2 composition calculus; math
    review 0707): arms = six pairs {cf_i,cf_j} alone + quad {cf0..cf3}
    alone, mixed Ks; J_ij = agg(pair) - single_i - single_j (singles from
    rung 109-111 receipts). pred_a |excess(quad) - sum J_ij| <= 0.25 x
    excess(quad) (order-2 suffices -> predict-before-run for all 2^4
    subsets); pred_b J_23 = max and >= 0.35 x sum J (ladder shape);
    pred_c every pair agg in [max single - 0.015, quad + 0.015], quad in
    [0.7, 1.1]. Null: >= 3-way terms dominate; pair calculus dies.
    Price: none. -> ops/frontier_cp_pairs.py [QUEUED 07:10Z]
    RUN 2026-08-31, S2212: pred_a FAILED by 0.0045 (pairs explain 74%
    of quad excess), pred_b FAILED (J spread; J_23 max at 31% of sumJ),
    pred_c HELD. J matrix measured.


119. Triples — Möbius lattice completion: the four 3-subsets at mixed Ks;
    pred2(ijk) = sum singles + sum J (J from rung 118 receipt at run
    time; runner order guarantees it). pred_a median |K3| <= 0.05
    (order-2 predicts an order it wasn't fit on); pred_b tri_123 largest;
    pred_c triples in [max pair - 0.015, quad + 0.015]. Null: cubic
    terms material (then K_ijk are measured). Price: none.
    -> ops/frontier_cp_triples.py [QUEUED 07:12Z]
    RUN 2026-08-31, S2213: ALL HELD. median |K3| 0.0428; K3 follows
    adjacency (123 +.132, 012 +.070, gapped ~0). Full Mobius ledger:
    .235/.442/.221/-.068. Composition calculus at order 3, ~+/-0.07.


120. Out-of-sample lattice extension (cross element types): four {a1v,cf_i}
    pairs + {a0,cf3} control; order-2 predicts the 6-front (+0.9427) from
    singles + all J. pred_a |pred2 - 0.9427| <= 0.10; pred_b |J(a0,cf3)|
    <= 0.01 (exactness extends to interactions); pred_c pair sanity.
    Null: calculus is MLP-only. Price: none.
    -> ops/frontier_lattice_oos.py [QUEUED 07:29Z]
    RUN 2026-08-31, S2214: pred_a FAILED (order-2 misses by 0.166) but
    cross-type J tiny (+0.048 total; J(a0,cf3) exactly 0, pred_b HELD);
    measured lattice predicts 6-front to 0.0128. pred_c HELD.

121. Pair-grain certificates: six pairs rerun with tau-certificates (aggs
    double as rung-118 repro). pred_a median pair valid <= 0.5 x min
    single valid (pairwise fragility); pred_b max pair valid >= 6;
    pred_c all aggs within 0.015 of rung 118. Nulls: validity survives
    pairing / total collapse. Price: none.
    -> ops/frontier_pair_certs.py [QUEUED 07:29Z]
    RUN 2026-08-31, S2215: pred_a/c HELD, pred_b FAILED. Every pair
    keeps EXACTLY 2/62 (singles 8-12): validity dies at pairs,
    uniformly. Survivor identity unrecoverable from counts -> rung 124.

122. CP-front GLOBAL config: full S2144 config (motifs, K middles, tail
    dicts refit in-frame) with front MLPs swapped in place to CP-3456
    before the tail fits; tailE stays table-front-fit (stated ~0.14
    approximation). pred_a agg <= 1.6 (vs rung-99 anchor +2.8553);
    pred_b valid >= 3/62; pred_c agg in [0.8, 2.2] and >= 0.62
    (anti-inertness). Null: front-motif/tail interactions eat the gain;
    certs stay 0-2. Price: front 47.8M vs 231.6M values.
    -> ops/frontier_global_cp.py [QUEUED 07:33Z]
    RUN 2026-08-31, S2216: pred_a/b FAILED, pred_c HELD. Census
    2.8553 -> 2.1358 (interactions ate 56% of solo gain); 0/62. BUT
    descriptive FR fresh L2 +2.0553 vs frontier 2.6662 - registered
    claim = rung 123. Both S312 window bars HELD.

123. REGISTERED FRONTIER CLAIM: identical rung-122 build rerun with the
    frontier bars as predictions. pred_a L2_F fresh <= 2.30 (table-front
    frontier 2.6662; LOWER IS BETTER); pred_b increment in [0.30, 0.55];
    pred_c census within 0.015 of 2.1358. Null: 2.0553 was noise (>= 2.60).
    Price: none new (front 47.8M vs 231.6M values).
    -> ops/frontier_claim.py [QUEUED 08:00Z]
    RUN 2026-08-31, S2217: ALL THREE HELD - NEW FRONTIER L2_F fresh
    +2.0553 (was 2.6662), increment in band, census repro to 0.0001.
    Quotable line updated; 184M fewer values.

124. Per-circuit damage matrix: four CP singles + quad, saving the 5x62
    member |dCE| rows + valid sets (survivor identity). pred_a monotone
    destruction (quad valid subset of every single's); pred_b median
    pairwise Spearman across singles' damage vectors >= 0.7 (shared
    targets); pred_c quad agg within 0.015 of +0.8302. Null: site-
    specific damage / non-monotone. Price: none.
    -> ops/frontier_damage_matrix.py [QUEUED 08:00Z]
    RUN 2026-08-31, S2218: ALL HELD (pred_a vacuous - quad survivors
    empty). Median pairwise Spearman 0.961: the four sites damage the
    SAME circuits; shared vulnerable subspace. 5x62 matrix saved.

125. Motif x front interaction (who ate the 0.9): arms motifs_alone /
    front_alone (CP-3456 + a0/a1v, repro of +0.1193) / front_motifs;
    middles real, no tail. J_FM = FM - M - F. pred_a J_FM >= 0.45;
    pred_b FM >= M + F + 0.30; pred_c front repro +/- 0.015. Null:
    additive (eater is the tail complex). Price: none.
    -> ops/frontier_motif_interact.py [QUEUED 08:33Z]
    RUN 2026-08-31, S2219: pred_a/b FAILED, pred_c HELD. J_FM only
    +0.2279; motifs ALONE +0.5530 census (new). Tail complex ~+1.24 was
    already dominant; S2216 interpretation sharpened.

126. Motif-alpha refit inside the frontier config: rung-123 build with the
    76 motif gains refit by the same projection on the CP-front stream
    (before tail refits). pred_a census <= 1.85 (was 2.1359); pred_b
    L2_F <= 1.95 (was 2.0553); pred_c increment in [0.25, 0.60], census
    >= 1.0. Null: alpha frame-insensitive (recovery < 0.05). Price:
    none (same 76 scalars). -> ops/frontier_motif_refit.py [QUEUED 08:33Z]
    RUN 2026-08-31, S2220: null wins exactly - alpha refit a no-op
    (within wobble). Motif repair must be structural.

127. Trajectory-target tail dictionaries in the frontier config: rung-123
    build with a10L-a17L CV/LW targets taken from a real-model pass
    (config-frame inputs, real-trajectory outputs — the S2210 move at
    the tail). pred_a increment <= 0.38 (was +0.5045); pred_b L2_F <=
    1.95 (was +2.0553); pred_c |L1F - anchor| <= 0.02, census in
    [1.4, 2.2]. Null: the 10-class dictionary map binds, not the target
    frame. Price: none. -> ops/frontier_tail_traj.py [QUEUED 08:34Z]
    RUN 2026-08-31, S2221: ALL HELD - NEW FRONTIER L2_F +1.8765,
    census +1.9474, increment 0.5045->0.3256, L1F invariant. The
    trajectory-teacher move is 2-for-2 across families.

128. tailE rebuild in the frontier config: config-frame inputs (CP front +
    motifs) + trajectory targets for the tail-MLP span dicts; certificate
    rows saved. pred_a census <= 1.85 (was 1.9474); pred_b L2_F <= 1.83
    (was 1.8765); pred_c increment in [0.20, 0.55], census >= 1.4. Null:
    tailE frame immaterial (recovery < 0.03). Price: none.
    -> ops/frontier_taile_rebuild.py [QUEUED 09:03Z]
    RUN 2026-08-31, S2222: ALL FAILED, beyond both registered outcomes
    (census +3.0503, increment +1.3689) - SUSPECT INSTRUMENT. Control
    = rung 130 (in-frame targets). No conclusion drawn.

129. Middles trajectory Down-refit (c4-c9, sequential, config frame, real-
    trajectory targets) in the rung-127 build. pred_a census <= 1.87;
    pred_b L2_F <= 1.85; pred_c increment in [0.20, 0.55], census >= 1.4.
    Null: middle K-cost is unit-capacity (recovery < 0.03). Price: none.
    -> ops/frontier_mid_traj.py [QUEUED 09:03Z]
    RUN 2026-08-31, S2223: ALL FAILED (census +2.3052). Trajectory
    move 2-for-2 at dictionary sites, 0-for-2 at real-weight sites.
    Control = rung 131 (in-frame target) scores the candidate law.

130. tailE rebuild CONTROL (in-frame targets, identical machinery): pred_a
    census <= 2.05 (trajectory target was the destroyer); pred_b L2_F <=
    1.95; pred_c increment in [0.20, 0.70], census >= 1.4. Null: still
    exploded (>= 2.6) = machinery bug -> code audit. Price: none.
    -> ops/frontier_taile_ctrl.py [QUEUED 09:30Z]
    RUN 2026-08-31, S2224: ALL HELD - machinery acquitted, trajectory
    TARGET convicted (frame mixing at a span-edit site). Law refines:
    trajectory targets legal only at full-output replacements. tailE
    branch closes (original table-front-fit tailE was costless).

131. Middles refit CONTROL (in-frame targets): pred_a census >= 2.10 (the
    exactness-vs-steering law stands: any refit of real weights loses);
    pred_b census <= 2.3052 (in-frame loses less than trajectory);
    pred_c L2_F in [1.7, 2.6], increment in [0.2, 0.8]. Null: recovery
    <= 2.00 (trajectory target was the harm). Price: none.
    -> ops/frontier_mid_ctrl.py [QUEUED 09:30Z]
    RUN 2026-08-31, S2225: VOID-AS-DESIGNED (self-target: capture hook
    downstream of installed cp hook; bit-identical L2F). Scored as
    written (a FAILED, b/c HELD). Corrected control = rung 133.

132. Certificate rows for the standing frontier: exact rung-127 build,
    62-circuit rows saved. pred_a |census - 1.9474| <= 0.015; pred_b
    valid <= 5 (S2208 stagnation; FAILURE = news); pred_c |L2_F -
    1.8765| <= 0.015, increment in [0.25, 0.40]. Price: none.
    -> ops/frontier_rows.py [QUEUED 09:33Z]
    RUN 2026-08-31, S2226: ALL HELD. Frontier two-ledger row filed:
    +1.8765/+1.9474 aggregate, 0/62 certificates. Rows saved.

133. Corrected middles control (offline real targets, S2199 capture rule):
    Down refit to real mlp(Xd) with all 4608 units = least-squares pruning
    compensation. pred_a census <= 1.92 (compensation wins); pred_b L2_F
    <= 1.86; pred_c increment in [0.25, 0.45], census >= 1.4. Null:
    exactness beats compensation (>= 1.95). Tripwire: rung-131 self-
    target signature. Price: none.
    -> ops/frontier_mid_real.py [QUEUED 10:00Z]

134. Per-class table of the frontier: exact rung-127 build, 10-class
    census breakdown saved. pred_a retrieval law survives (link >= 2 x
    const); pred_b subword <= aggregate; pred_c census repro +/- 0.015.
    Null: class structure flattened. Price: none.
    -> ops/frontier_classes.py [QUEUED 10:00Z]

133. Corrected middles control (offline real targets, S2199 capture rule):
    Down refit to real mlp(Xd) with all 4608 units = least-squares pruning
    compensation. pred_a census <= 1.92 (compensation wins); pred_b L2_F
    <= 1.86; pred_c increment in [0.25, 0.45], census >= 1.4. Null:
    exactness beats compensation (>= 1.95). Tripwire: rung-131 self-
    target signature. Price: none.
    -> ops/frontier_mid_real.py [QUEUED 10:03Z]
    RUN 2026-08-31, S2227: all preds FAILED, NULL HELD - exactness
    beats compensation (+2.148 vs +1.947). Law stands on clean
    instruments; ordering exact < in-frame refit < trajectory refit.
    Middle-refit branch CLOSED.

134. Per-class table of the frontier: exact rung-127 build, 10-class
    census breakdown saved. pred_a retrieval law survives (link >= 2 x
    const); pred_b subword <= aggregate; pred_c census repro +/- 0.015.
    Null: class structure flattened. Price: none.
    -> ops/frontier_classes.py [QUEUED 10:03Z]
    RUN 2026-08-31, S2228: pred_a/b FAILED, c HELD. Class map INVERTED:
    name 3.13 and subword 2.32 top payers; punctuation nearly free.
    Retrieval law dead in old form; front-assembly suspicion -> 136.

135. Sensitivity-weighted tail-dict fits (math review 1008): per-position
    w_t = |dCE/dh| at block-10 input weighting the aXL CV means + LW
    ridges; closed-list distinction stated (not Fisher selection, not
    metric-K). pred_a census <= 1.92; pred_b increment <= 0.30; pred_c
    L1F invariant, census >= 1.4. Null: class structure already captures
    sensitivity (< 0.01). Price: none.
    -> ops/frontier_sensw.py [QUEUED 10:12Z]
    RUN 2026-08-31, S2229: null wins with interest (census +1.9774,
    worse by 0.03). Class structure already captures sensitivity;
    branch CLOSED at the tail dicts.

136. Exact-front class attribution: rung-127 build at front K=4608, class
    table saved. pred_a name <= 1.6 (halves); pred_b subword <= 1.2;
    pred_c census in [1.6, 1.9]. Null: name/subword persist >= 80%
    (assembly damage is motif/tail-caused). Price: diagnostic 63.7M front.
    -> ops/frontier_front4608_classes.py [QUEUED 10:12Z]
    RUN 2026-08-31, S2230: pred_a/b FAILED (name -9%, subword -14% -
    null holds at 91%/86%): assembly bill is MOTIF/TAIL-caused. Bonus
    Pareto point L2F +1.6599 at 63.7M front -> registered as rung 137.

137. Registered exact-front Pareto claim: identical rung-136 build, claim
    bars registered. pred_a L2_F <= 1.70; pred_b census repro +/- 0.015;
    pred_c valid <= 5 (failure = news). Null: 136's print was noise.
    Price: front 63.7M values (second Pareto point; does not replace the
    47.8M S2221 line). -> ops/frontier_claim4608.py [QUEUED 10:30Z]
    RUN 2026-08-31, S2231: ALL HELD - second Pareto point registered:
    L2_F +1.6599 / census +1.7202 at 63.7M front values; 0/62. Path
    today: 2.6662 -> 2.0553 -> 1.8765 -> 1.6599.

138. Motifs-real class attribution: rung-127 build with attention 2-9
    REAL (tail dicts refit under it). pred_a name <= 1.8 (motif-caused);
    pred_b subword <= 1.5; pred_c census in [1.1, 1.7]. Null: name/
    subword persist >= 80% (tail dicts are the assembly bill). Price:
    diagnostic. -> ops/frontier_motifreal_classes.py [QUEUED 10:30Z]
    RUN 2026-08-31, S2232: pred_b/c HELD, pred_a FAILED. Subword partly
    motif-caused (-36%); NAME NOT (-18%) -> tail complex by elimination.
    Motifs carry ind/other/rep retrieval.

139. Name-class linear maps in the tail dicts: name moves CONSTN -> LINK
    (fitted D x D map at each of 8 tail sites) in the rung-127 build.
    pred_a name <= 2.6 (was 3.134); pred_b census <= 1.92; pred_c L1F
    invariant, increment <= 0.36. Null: name's bill is upstream (cross-
    checks rung 138). Price: +10.6M values (8 maps).
    -> ops/frontier_namelw.py [QUEUED 10:33Z]
    RUN 2026-08-31, S2233: pred_a/b FAILED (name -0.05, census -0.0015)
    - class-map hypothesis DEAD. Name = irreducible retrieval; no local-
    stream function suffices. Arm rejected on price and performance.

140. Name-oracle ceiling: real attention output spliced in for name-class
    positions at all tail-attn dict sites (config-consistent fits).
    pred_a name <= 1.0 (ceiling high); pred_b census <= 1.85; pred_c L1F
    invariant, increment in [0.15, 0.36]. Null: name's bill is in tailE/
    front interactions (oracle < 0.5). Price: none (oracle diagnostic).
    -> ops/frontier_nameoracle.py [QUEUED 11:00Z]
    RUN 2026-08-31, S2234: pred_a/b FAILED, c HELD. Oracle recovers
    only 0.80 of name's 3.1 - the bill is COMPOSITIONAL (front .27 /
    motifs .57 / tail-attn .80 / remainder ~1.5). Tail-attn module
    alone is not the fix.

141. Per-site name attribution (single-knockout, caveat stated): eight
    skip-a_liL arms; recovery = name(full) - name(skip). pred_a max
    recovery >= 0.4; pred_b owner in {a14L, a16L}; pred_c census repro.
    Null: spread (module must cover the whole tail). Price: none.
    -> ops/frontier_namesites.py [QUEUED 11:00Z]
    RE-RUN 2026-08-31 11:32Z, S2236: ALL HELD. Owner a16L (+0.402),
    a14L (+0.349); spread across sites; knockout sum 1.66 >> joint
    oracle 0.80 (overlap overcount). a17L exactly 0.
    RUN 2026-08-31, S2235: VOID (rung-56 double-del of clsmap; crashed
    after all arms, results unwritten). Fixed in place, re-queued
    11:30Z; preds unchanged and unscored.

142. Full tail-complex name oracle: rung-140 attn splice + tailE span-
    delta zeroed for name positions at m10-17. pred_a name <= 1.9
    (tailE adds >= 0.38); pred_b census <= 1.90; pred_c L1F invariant,
    increment in [0.10, 0.36]. Null: tailE adds nothing for name (>=
    2.18) - remainder is middles/front interactions. Price: none.
    -> ops/frontier_nameoracle2.py [QUEUED 11:30Z]
    RUN 2026-08-31, S2237: pred_a/c HELD, pred_b FAILED by 0.0076.
    tailE adds +0.555 name recovery; module-grade ladder 3.134 ->
    2.285 -> 1.730. ~55% of name's bill still outside the tail complex.

143. Trajectory-grade name oracle: real-model tail-attn outputs captured
    position-aligned on census rows, spliced for name positions in the
    census eval only (batch-counter hooks + inertness tripwire). Gap vs
    S2234's module-grade (+2.285) = context-corruption share. pred_a
    name <= 1.6; pred_b census <= 1.90; pred_c L1F invariant, census >=
    1.4. Null: gap < 0.2 (remainder is elsewhere). Price: none.
    -> ops/frontier_nametraj.py [QUEUED 11:35Z]
    RE-RUN 2026-08-31 12:02Z, S2239: ALL HELD. Trajectory splice ->
    name +1.321 (vs module-grade +2.285): context corruption 0.96 >
    module share 0.85. STREAM FIDELITY is the lever. Fork resolved.
    RUN 2026-08-31, S2238: VOID (late-binding li in the splice hook -
    all sites saw li=17; rung-61 class). Fixed (li=li default),
    re-queued 12:00Z; preds unchanged and unscored.

144. Subword module-grade oracle: rung-140 splice for subword rows
    (oracle class 7) at all tail-attn sites. pred_a subword <= 1.9
    (>= 0.4 recovery); pred_b census <= 1.92; pred_c L1F invariant,
    increment in [0.15, 0.36]. Null: subword's tail-attn share < 0.2
    (assembly is upstream). Price: none.
    -> ops/frontier_subworacle.py [QUEUED 12:01Z]
    RUN 2026-08-31, S2240: pred_a/b FAILED (recovery 0.198; null holds
    by 0.002). Subword is upstream assembly - opposite diagnosis to
    name.

145. All-class trajectory splice at tail attention: total context-
    corruption share of the family, per-class recoveries free. pred_a
    census <= 1.70; pred_b name <= 1.37 (consistency with S2239);
    pred_c L1F invariant, census >= 1.4. Null: non-name content is
    local (< 0.10 beyond name-only). Price: none.
    -> ops/frontier_trajall.py [QUEUED 12:30Z]
    RUN 2026-08-31, S2241: pred_a/b HELD (census 1.9474 -> 0.9400!),
    pred_c FAILED on the lower sanity floor (effect outran the bar -
    S2199 artifact class). Retrieval classes carry the bulk; subword
    transits too; cheap classes pay frame-mixing. 0/62 even under
    oracle stream fidelity - certificates upstream of tail attn.

146. The name floor: trajectory attn splice + tailE bypass together for
    name rows. pred_a name <= 0.9 (rough additivity of the 0.555 tailE
    credit); pred_b census <= 1.87; pred_c guards. Null: tailE credit
    redundant with context corruption (name >= 1.15). Price: none.
    -> ops/frontier_namefloor.py [QUEUED 12:30Z]
    RUN 2026-08-31, S2242: pred_a/c HELD, pred_b FAILED by 0.0089.
    Name floor +0.717; tailE credit roughly additive (0.604). Remainder
    = front/motif direct share.

147. Trajectory splice at the motif grain (blocks 2-9, all positions,
    li default-bound, counter at li=9): does front corruption transit
    through motif attention? pred_a census <= 1.55; pred_b name <= 2.2;
    pred_c L1F invariant, census >= 1.4. Null: recovery < 0.15 (motif
    cost is direct approximation). Price: none.
    -> ops/frontier_motiftraj.py [QUEUED 12:34Z]
    RUN 2026-08-31, S2243: pred_a/b HELD, pred_c FAILED on the lower
    sanity floor (second S2199-class artifact; oracle floors -> 0.5
    henceforth). Census 1.9474 -> 0.8758: motif grain carries ~1.07.
    sentend/comma go NEGATIVE. 0/62 under both splices.

148. Class-conditioned motif alphas (9 heads x 10 classes/layer; fallback
    to scalar alpha when labels absent - window prints mixed-frame, not
    scored; li default-bound). pred_a census <= 1.87; pred_b subword <=
    2.15; pred_c increment in [0.25, 0.45], census >= 1.4. Null: alpha
    class-invariant (< 0.02). Price: +1,368 scalars. Tripwire: inert if
    census within 1e-3. -> ops/frontier_clsalpha.py [QUEUED 13:00Z]
    RUN 2026-08-31, S2244: pred_a/b FAILED, c HELD. Class-alpha nets
    +0.0115 (rep -0.28 but digit +0.12). Motif cost is STRUCTURAL
    (the alpha.v form), not parametric.

149. Block-5 attention real (the attn5 cliff in the modern frontier):
    motif_off={5}, tail dicts refit under it. pred_a census <= 1.80;
    pred_b subword <= 2.05; pred_c increment in [0.25, 0.45], census >=
    1.4. Null: the cliff was a table-era phenomenon (< 0.05). Price:
    ~5.3M values. -> ops/frontier_attn5real.py [QUEUED 13:00Z]
    RUN 2026-08-31, S2245: pred_a FAILED by 0.026 (recovery 0.122 at
    5.3M values), pred_b FAILED, c HELD. The attn5 cliff is modest in
    the modern frontier.

150. Bias-dispersion decomposition of certificate failure (math review
    1307): base + all-class-splice census evals; per-circuit signed vs
    abs member means. pred_a median |signed|/abs <= 0.5 (dispersion-
    dominated); pred_b median circuit abs-improvement <= 0.8 x census
    factor (dispersion sticky); pred_c repro both arms. Null: BIAS-
    dominated -> per-circuit bias correction (~62 scalars) becomes the
    cheapest certificate candidate. Price: none.
    -> ops/frontier_biasdisp.py [QUEUED 13:10Z]

150. Bias-dispersion decomposition of certificate failure (math review
    1307): base + all-class-splice census evals; per-circuit signed vs
    abs member means. pred_a median |signed|/abs <= 0.5 (dispersion-
    dominated); pred_b median circuit abs-improvement <= 0.8 x census
    factor; pred_c repro both arms. Null: BIAS-dominated -> per-circuit
    bias correction (~62 scalars) becomes the cheapest certificate
    candidate. Price: none. -> ops/frontier_biasdisp.py [QUEUED 13:12Z]

150. Bias-dispersion decomposition of certificate failure (math review
    1307): base + all-class-splice census evals; per-circuit signed vs
    abs member means. pred_a median |signed|/abs <= 0.5 (dispersion-
    dominated); pred_b median circuit abs-improvement <= 0.8 x census
    factor; pred_c repro both arms. Null: BIAS-dominated -> per-circuit
    bias correction (~62 scalars) becomes the cheapest certificate
    candidate. Price: none. -> ops/frontier_biasdisp.py [QUEUED 13:16Z]
    RUN 2026-08-31, S2246: pred_a FAILED (median bias ratio 0.693 -
    BIAS-leaning, null branch live), pred_b FAILED by a hair (near-
    proportional improvement), pred_c HELD. -> leaf-bias rung 152.

151. Bootstrap-averaged aXL dictionary fits (B=4, seeded per site; math
    review 1307 move 2): is certificate dispersion estimation noise?
    pred_a median member-absdce ratio vs rung-132 <= 0.97; pred_b census
    <= 1.97; pred_c increment in [0.25, 0.45], census >= 1.4. Null:
    ratio > 0.99 (dispersion intrinsic; move 2 closes). Price: none.
    -> ops/frontier_bootavg.py [QUEUED 13:20Z]
    RUN 2026-08-31, S2247: pred_a FAILED decisively (ratio 0.9997) -
    dispersion intrinsic, not estimation noise. Move 2 closes.

153. Removal collateral matrix (user request: the removal property at
    matrix grain): the 8 distinct battery top components mean-ablated on
    the REAL model; M[c][t] = every circuit's member damage per knockout.
    pred_a median own/other selectivity >= 3; pred_b a8 collateral >= 20
    non-own circuits above 0.25 x ref (substrate sharing); pred_c battery
    repro ratio in [0.67, 1.5] (protocol control). Nulls: selectivity < 2
    / sharing < 10. Price: none. -> ops/removal_matrix.py [QUEUED 13:31Z]
    RUN 2026-08-31, S2248: pred_a FAILED (matrix-grain selectivity
    1.695 vs concentration's 4.08 - offslice flattered removal),
    pred_b HELD (a8 collateral 43!), pred_c PERFECT (repro 1.000).
    Removal = weakly selective; substrate sharing heavy.

152. Leaf-conditioned final-stream bias vectors (bias-branch certificate
    candidate): 62 per-circuit h17 biases = mean (real - config) drift
    over TRAIN-half members; TEST-half certificates. pred_a test valid
    >= 5; pred_b census <= 1.93; pred_c L1F invariant, census >= 1.4.
    Null: member damage is not leaf-DC at h17. Price: 71,424 values
    (diagnostic-grade: leaf membership census-indexed). Tripwire: inert
    if census within 1e-3. -> ops/frontier_leafbias.py [QUEUED 13:36Z]
    RUN 2026-08-31, S2249: pred_a/b FAILED - 0/62 AND census WORSE by
    0.19. Per-circuit patches are the wrong object (operator's point,
    vindicated in-flight). Re-aimed: rung 155 (ladder x circuits,
    breadth criterion) + rung 156 (global h17 bias).

154. Minimality audit v1 (user request): necessity depth on 12 stratified
    circuits (4 a8-family, 4 a16-family, 4 singleton) — member damage
    under joint mean-ablation of top1 / top12 / top123. pred_a median
    top1 share >= 0.7 (near-minimal at component grain); pred_b median
    top123/top1 <= 1.5; pred_c battery repro in [0.67, 1.5]. Null:
    share < 0.5 (genuinely multi-component). Price: none.
    -> ops/minimality_v1.py [QUEUED 13:35Z]
    RUN 2026-08-31, S2250: null CONFIRMED (top-1 share 0.464; top-3
    doubles damage). Circuits are multi-component on shared substrate.

155. Simplification ladder x circuits (operator's program): a8/a16/m16
    x {mean, linear, clsdict, +CP-2304 at m16}, all GLOBAL fits (train
    half), member damage per circuit, breadth on test half. pred_a best
    non-constant rung <= 0.8 x mean at all comps; pred_b a8 breadth >= 8
    of 16 own circuits at 0.5 x ref; pred_c 153-protocol repro in [0.8,
    1.25]. Null: breadth < 3 (irreducibly complex at circuit grain).
    Prices: linear 1.33M, clsdict 11.5k, CP 7.96M values.
    -> ops/ladder_matrix.py [QUEUED 13:40Z]
    RUN 2026-08-31, S2251: pred_a/b FAILED, c HELD - but m16:LINEAR is
    the FIRST breadth pass (6/6 own circuits, +0.0286 agg, 1.33M
    values); attention comps fail all position-local rungs; cp2304 at
    m16 fails (0/6). Non-local rungs -> 157/158.

156. Global h17 bias (the legitimate "optimal bias on all data"): ONE
    drift-mean vector (1,152 values), train-half fit. pred_a |census -
    1.9474| <= 0.02 (near-null by design); pred_b <= 1.9524 (no harm -
    the per-circuit version added +0.19); pred_c guards. Null: even the
    global bias hurts. Inertness tripwire waived (near-null IS pred_a).
    -> ops/frontier_globalbias.py [QUEUED 13:40Z]
    RUN 2026-08-31, S2252: null in strongest form - global bias
    EXPLODES (+3.0155). Additive real-frame stats into config streams
    illegal at any grain; h17 bias branch CLOSED.

157. Deployed motif grammar circuit-scored at its own components: motif@8
    and motif@3 single-site (all else real). pred_a a8 breadth >= 4/16;
    pred_b a3 breadth >= 2/5; pred_c aggs in sane bands. Null: breadth 0
    (the deployed grammar is circuit-unfaithful bottom-up). Price: none.
    -> ops/motif_single.py [QUEUED 14:00Z]
    RUN 2026-08-31, S2253: ALL HELD - motif grammar circuit-faithful
    at own comps (a8 11/16, a3 5/5). Config 0/62 is compositional,
    not grammar-intrinsic, for motifs.

158. Deployed tail-dict grammar real-frame at a16 (fit_frame_real flag):
    a16L dictionary single-site, circuit rows. pred_a breadth >= 4/13;
    pred_b agg in [0.005, 0.10]; pred_c coverage >= 8. Null: breadth 0
    (class-dict grammar unfaithful even in real frame). Price: none.
    -> ops/a16_single.py [QUEUED 14:00Z]

157. Deployed motif grammar circuit-scored at its own components: motif@8
    and motif@3 single-site (all else real). pred_a a8 breadth >= 4/16;
    pred_b a3 breadth >= 2/5; pred_c aggs in sane bands. Null: breadth 0
    (the deployed grammar is circuit-unfaithful bottom-up). Price: none.
    -> ops/motif_single.py [QUEUED 14:00Z]

158. Deployed tail-dict grammar real-frame at a16 (fit_frame_real, first-
    occurrence patch): a16L dictionary single-site, circuit rows. pred_a
    breadth >= 4/13; pred_b agg in [0.005, 0.10]; pred_c coverage >= 8.
    Null: breadth 0 (class-dict grammar unfaithful even in real frame).
    Price: none. -> ops/a16_single.py [QUEUED 14:05Z]
    RUN 2026-08-31, S2254: pred_a FAILED (breadth 0/13 real-frame) -
    the a16 class-dict grammar is intrinsically circuit-unfaithful.

159. Minimality for ALL 62 (C1): cumulative top-1..4 necessity curves,
    k* per circuit (repertoire column 1). pred_a median top-1 share <=
    0.55; pred_b median 3-of-4 saturation >= 0.85; pred_c battery repro
    in [0.8, 1.25]. Null: no saturation by 4 (greedy must extend).
    ~40-60 min run. -> ops/minimality62.py [QUEUED 14:30Z]
    RUN 2026-08-31, S2256: pred_a/c HELD, pred_b FAILED by 0.014 -
    47/62 circuits hit the k*=4 cap; minimal sets >= 4; greedy extends
    (rung 163). Median top-1 share 0.389.

160. Interchange instrument (C2, the DAS foundation): member-permutation
    activation swap at 10 circuits' interchange-top components; compare
    to battery interchange refs. pred_a median ratio-to-iref >= 0.5;
    pred_b selectivity >= 3; pred_c ratio in [0.5, 2.0]. Null: swap
    protocol too weak. -> ops/interchange_inst.py [QUEUED 14:30Z]
    RUN 2026-08-31, S2257: ALL HELD. Protocol agrees with battery
    (0.946); interchange selectivity 425 vs removal 1.7 - the surgical
    tool. a17 offslice ~0.

161. DAS-lite (C3): fixed low-rank subspaces (member-PCA r=1/8/32,
    diff-in-means) patched in the rung-160 interchange protocol; share =
    subspace effect / full swap. pred_a median share(32) >= 0.5; pred_b
    pca8 >= 0.4 for >= 3/10; pred_c 2-circuit full repro within 10% +
    monotone 8/10. Null: shares < 0.25 (distributed carrier; DAS-proper
    or nothing). Reads 160's receipt at run time (runner order).
    -> ops/daslite.py [QUEUED 14:35Z]
    RUN 2026-08-31, S2258: ALL HELD. Low-rank carriers exist: median
    share(32) 0.512, pca8 >= 0.4 for 4/10, monotone 10/10, repro clean.
    dmean1 weak. -> DAS-proper rung 162.

162. DAS-proper (C4): learned rank-8 subspaces (Adam 150 steps, QR
    retraction, KL to full-swap logits, warm start pca8) at the 2 lowest-
    pca8 circuits + 1 control; within-script arms. pred_a learned >=
    pca8+0.15 at both low; pred_b median learned >= 0.55; pred_c
    orthonormal + control no-regress. Null: learning adds < 0.05. Price:
    9,216 values per passing subspace. -> ops/dasproper.py [QUEUED 15:03Z]
    RUN 2026-08-31, S2259: all FAILED - SUSPECT INSTRUMENT (loss rose
    2/3; control regressed from warm start). v2 = rung 164 (lr 3e-3,
    cosine, best-loss checkpoint, optimizer-sanity pred).

163. Minimality extension to 6: the 47 capped circuits, top-1..6. pred_a
    median 5-of-6 saturation >= 0.85; pred_b median k* <= 5; pred_c
    protocol repro. Null: no saturation by 6 (substrate-wide circuits).
    -> ops/minimality6.py [QUEUED 15:03Z]
    RUN 2026-08-31, S2260: ALL HELD. Median k* = 5, saturation by 6
    (0.909). Full picture: circuits are ~5-component objects. Histogram
    print bug (1-4 bins) recorded; receipt values correct.

164. DAS-proper v2 (optimizer fixed: lr 3e-3 cosine, 300 steps, best-loss
    checkpoint). pred_a control no-regress (>= pca8 - 0.02); pred_b
    learned >= pca8 + 0.10 at >= 1 low circuit; pred_c orthonormal +
    best loss <= initial at all 3. Null: healthy optimizer still adds
    < 0.05 (PCA is the rank-8 ceiling). -> ops/dasproper2.py [QUEUED 15:31Z]
    RUN 2026-08-31, S2261: pred_c HELD (optimizer sane), pred_a/b
    FAILED - learning adds ~0.02; fixed member-PCA is the rank-8
    ceiling. DAS-proper closes; repertoire stores PCA bases. Train-KL
    vs share-metric gap noted.

165. Carrier necessity (DAS-lite dual): comp32 = patch the complement of
    pca32; per-circuit sum check. pred_a median complement share <= 0.5;
    pred_b median |sum - 1| <= 0.35; pred_c full-swap repro. Null:
    basis-insensitive damage (the carrier is a dimension-counting
    illusion). -> ops/carrier_necessity.py [QUEUED 15:31Z]
    RUN 2026-08-31, S2262: pred_a FAILED (complement 0.626), b/c HELD.
    Carrier = concentration (~28x per-dim), not exclusive support;
    heterogeneous (a17 concentrated, a14 reversed).

166. Shared-carrier test: r.2.0's pca32 basis patched into 4 other
    a8-family circuits' interchanges vs their OWN bases. pred_a median
    ref/own ratio >= 0.7; pred_b ref-share >= 0.3 for >= 3/4; pred_c
    population guard (full within 2x battery ref). Null: carriers
    circuit-specific (< 0.2). A shared carrier = ONE 32x1152 basis for
    16 circuits. -> ops/shared_carrier.py [QUEUED 15:34Z]
    RUN 2026-08-31, S2263: pred_a HELD (ratio 0.847 - shared), pred_b
    FAILED (calibration artifact: targets' own shares 0.32-0.41), c
    HELD. Repertoire stores ONE basis per family.

167. Family carriers for ALL 62: one pooled pca32 basis per interchange-
    top component, scored against every circuit (62 x 2 evals, ~50 min).
    pred_a median family share >= 0.4; pred_b >= 0.3 for >= 60%; pred_c
    population guard >= 50. Null: family bases don't generalize (< 0.25).
    Price: ~0.37M values for the whole column.
    -> ops/family_carriers.py [QUEUED 16:00Z]
    RUN 2026-08-31, S2265: pred_a FAILED by 0.015 (median 0.385),
    pred_b/c HELD (58/62 >= 0.3). Column filled: ~10 family bases,
    0.37M values, every circuit covered.

168. Carrier-projection removal at a8 (deployable, position-independent:
    y - PPT(y-mu) everywhere): vs in-script mean-ablation baseline.
    pred_a proj family-median >= 0.5 x mean-abl; pred_b selectivity >=
    3x mean-abl's; pred_c agg <= 0.5x AND S2248 repro. Null: projection
    as blunt as ablation. Price: 38,016 values for the removal operator.
    -> ops/carrier_removal.py [QUEUED 16:00Z]
    RUN 2026-08-31, S2264: pred_a/b FAILED (sel ratio 1.02 - projection
    as blunt as ablation; carrier holds 27%), pred_c HELD (8x cheaper,
    repro exact). Removal selectivity capped ~2x by substrate sharing;
    interchange's 425x was the member gating.

169. Variance-vs-causal null battery (math review 1607): member-PCA vs
    offslice-PCA vs random rank-32 bases at 10 circuits; varfracs
    recorded. pred_a member >= 1.5 x offslice (causal beyond variance);
    pred_b offslice >= 2 x random; pred_c repro + random <= 0.3. Null:
    member ~ offslice (carrier = variance patching; honest downgrade).
    -> ops/carrier_null.py [QUEUED 16:10Z]
    RUN 2026-08-31, S2266: NULL WINS - member-pca 0.519 vs offslice
    0.484 (1.07x): carrier = principal VARIANCE, member-agnostic.
    Damage tracks variance (quadratic form). Circuit identity is
    POSITIONAL. DAS thread closes; -> gated removal (rung 170).

170. Member-probe separability (gated removal feasibility): ridge probe on
    component INPUT predicting family membership; 6 families; train half /
    test AUC + shuffle control. pred_a median AUC >= 0.85; pred_b a8 >=
    0.90; pred_c shuffled in [0.45, 0.55]. Null: AUC < 0.7 (gate needs
    the oracle). Price: 1,152 values/gate.
    -> ops/probe_gate.py [QUEUED 16:32Z]
    RUN 2026-08-31, S2267: pred_a/b FAILED, c HELD. Median AUC 0.702 -
    between pass (0.85) and null (0.7): gate not cheaply linear; class-
    feature follow-up -> rung 172.

171. Sufficiency (missing half of minimality): keep battery top-5 real,
    mean-ablate the rest of the 16-set; controls keep-random-5 and
    keep-none; 10 stratified circuits. pred_a median top5-damage <= 1.0 x
    ref; pred_b top5 <= 0.5 x rand5 for >= 7/10; pred_c top5 <= 0.7 x
    none for >= 8/10. Null: members die regardless (substrate-wide).
    -> ops/sufficiency.py [QUEUED 16:32Z]
    RUN 2026-08-31, S2268: ALL FAILED - no sufficiency at component
    grain (median 4.6x ref; top5 beats random5 for 0/10; barely beats
    keep-none). Necessity/sufficiency asymmetry = circuits are facets
    of one coupled computation; extraction FALSE at this grain.

172. Stream + class probe (gate composition): [component input (+) classify2
    one-hot]; class-alone CPU check gave median 0.581, stream-only 0.702.
    pred_a median AUC >= 0.78; pred_b no family regresses; pred_c shuffle
    control. Null: <= 0.72 (gate oracle-bound; gating thread closes).
    -> ops/probe_gate2.py [QUEUED 16:35Z]
    RUN 2026-08-31, S2269: pred_a FAILED (0.708), b/c HELD. Gating
    thread CLOSED: gate not linear, not class, not composed - oracle-
    bound; surgery stays diagnostic.

173. Minimal-set trajectory splice (the RECONNECT rung): in the frontier
    build, splice each of 5 circuits' top-5 components to real-pass
    outputs vs seeded random-5 controls. pred_a median minset recovery >=
    0.5; pred_b minset >= 1.3 x rand5 for >= 3/5; pred_c plain repro vs
    rung-132 in [0.85, 1.15]. Null: indiscriminate recovery (kill is
    stream-wide; per-circuit repair impossible at component grain).
    -> ops/minset_splice.py [QUEUED 16:35Z]
    RUN 2026-08-31, S2270: pred_a FAILED by 0.096 (median rec 0.404),
    pred_c HELD, pred_b PARTIALLY VOID (f16-capture overflow NaN in 3/5
    rand5 arms - new bug class; sets unsaved). Fix + rescore = rung 174.
    Valid comparisons show specificity (1.6x, 1.5x).

174. Minset splice v2 (bf16 captures, rand5 sets logged, NaN tripwire):
    rescoring 173's half-void specificity pred. pred_a median minset
    recovery within 0.05 of 0.404 (repro); pred_b minset >= 1.3 x rand5
    for >= 3/5; pred_c plain repro. Null (b): generic stream fidelity.
    -> ops/minset_splice2.py [QUEUED 17:00Z]
    RUN 2026-08-31, S2272: ALL HELD - bf16 fix clean, 173 reproduces
    (0.404), specificity holds 3/5 at the knife-edge; r.0.0/r.1.2
    prefer random fidelity. S2273 = phase-1 synthesis.

175. Per-leaf probes (was the union the problem?): 10 largest leaves,
    individual ridge probes at their top component's input. pred_a
    median AUC >= 0.85; pred_b >= 0.75 for >= 8/10; pred_c shuffle
    control. Null: < 0.7 (gating oracle-bound at every grain).
    -> ops/probe_leaf.py [QUEUED 17:00Z]
    RUN 2026-08-31, S2271: pred_a/b FAILED (median 0.671; 2/10 over
    0.75). Gating final: oracle-bound at every grain.

176. OOD component-grain transport (the feasible OOD column): 16 mean-
    ablation knockouts, 10-class damage profiles on census vs 120 fresh
    pile rows. pred_a median Spearman >= 0.8; pred_b magnitude ratio in
    [0.6, 1.6]; pred_c fresh base sane. Null: signatures census-specific.
    -> ops/ood_transport.py [QUEUED 17:32Z]
    RUN 2026-08-31, S2274: pred_a FAILED (median rho 0.679), b/c HELD.
    Type split: MLP signatures text-generic (~0.85-0.90), attention
    census-shaped (~0.44-0.53). Magnitudes transport (1.13).

177. Era circuit-profile invariance (CPU, receipts; v2 - the rung-137
    receipt has no rows, discovered at build; gate also caught a module-
    level var reuse in v1): Spearman between the TABLE-era config (rung
    99) and CP-era frontier (rung 132) 62-circuit profiles. pred_a rho
    >= 0.9; pred_b median ratio 132/99 in [0.4, 0.85]; pred_c coverage.
    Null: the grammar change moved which circuits pay.
    -> ops/pareto_profile.py [QUEUED 17:35Z]
    RUN 2026-08-31, S2275: pred_a FAILED (rho 0.781), pred_b FAILED by
    0.009 high (ratio 0.859: circuits relieved less than aggregate),
    c HELD. Profile mostly era-invariant.

178. Motif OV residual rank-8 (phase-2 opener): z = alpha.v + v@R8 per
    motif head, ridge-fit to real z on the alpha captures, SVD rank-8.
    pred_a census <= 1.87; pred_b subword <= 2.15 AND rep <= 1.55;
    pred_c increment in [0.25, 0.45], census >= 1.4. Null: the damage is
    in the attention PATTERN (value-map capacity futile, < 0.02). Price:
    155,648 values. -> ops/motif_ovres.py [QUEUED 17:36Z]
    RUN 2026-08-31, S2276: pred_a FAILED by 0.025, pred_b FAILED, c
    HELD. OV residual buys 0.052 census / 0.054 fresh (L2F +1.8222
    descriptive) - between null and pass: value map ~10% of the motif
    ceiling. -> 179 (claim) + 180 (rank saturation).

179. Registered claim: rung-127 + OV residual (identical rebuild, claim
    bars). pred_a L2_F <= 1.84; pred_b census repro +/- 0.015; pred_c
    L2F repro + increment band. Null: 178's print was noise. Price:
    47.8M + 155,648. -> ops/frontier_claim_ovres.py [QUEUED 17:59Z]
    RUN 2026-08-31, S2277: ALL HELD - Pareto point moves to L2F
    +1.8222 / census +1.8950 at +155,648 values.

180. OV rank-32 saturation check: same build at rank 32. pred_a
    |census - 1.8950| <= 0.02 (saturated -> pattern reading stands);
    pred_b no worsening; pred_c guards. Null: <= 1.86 (capacity still
    yielding). -> ops/motif_ovres32.py [QUEUED 17:59Z]
    RUN 2026-08-31, S2278: ALL HELD - saturation at rank 8 (r32 buys
    0.0074). Remaining ~0.50 motif ceiling is attention-PATTERN
    structural. -> low-rank QK next.

181. Low-rank QK motif heads (the pattern lever, weights-only SVD of the
    four per-head projections; squared-attention pattern reimplemented
    per head, real values): pred_a census <= 1.75; pred_b subword <=
    2.05; pred_c increment in [0.25, 0.50], census >= 1.3. Null: rank-8
    patterns no better than fixed prev/self (motif ceiling irreducible
    below full QK). Price: 3.11M values.
    -> ops/motif_qkr.py [QUEUED 18:31Z]
    RUN 2026-08-31, S2279: pred_a/b FAILED - r8 WORSE than fixed motifs
    (+2.0577). Sub-critical rank, per S2280.

182. QK rank-16 (reads 181's receipt at run time): pred_a still yielding
    (<= r8 - 0.02); pred_b monotone; pred_c guards. Null: saturation at
    8. Price: 6.23M values. -> ops/motif_qkr16.py [QUEUED 18:31Z]
    RUN 2026-08-31, S2280: ALL HELD SPECTACULARLY - r16 census +1.6507,
    L2F +1.6428 (beats both Pareto points, descriptive). Critical rank
    between 8 and 16. Claim = rung 184.

183. QK-tail rank-8: the pattern lever at blocks 10-17 (all 72 heads,
    weights-only; aXL dicts retired; hooks armed after real baselines).
    pred_a census <= 1.78; pred_b increment <= 0.25 AND name <= 3.0;
    pred_c census >= 1.3 + inert tripwire. Null: retrieval needs sharp
    patterns (dicts stand). Price: 2.95M values REPLACING the dicts'
    larger store. -> ops/qk_tail.py [QUEUED 18:35Z]
    RUN 2026-08-31, S2281: pred_a/b FAILED at r8 (census +2.0241) -
    sub-critical, same as motifs. Tail r16 folded into rung 185
    (all-QK-16 consolidation).

184. Registered claim: motif QK-16 (identical rebuild, claim bars). pred_a
    L2_F <= 1.67; pred_b census repro; pred_c L2F repro + increment band.
    Null: 182's print was noise. Price: 47.8M + 6.23M.
    -> ops/frontier_claim_qk16.py [QUEUED 18:59Z]
    RUN 2026-08-31, S2282: ALL HELD - Pareto point moves to L2F +1.6428
    / census +1.6507; dominates the 63.7M point.

185. ALL-QK-16 (uniform grammar; dicts retired; 148 heads at rank 16):
    pred_a census <= 1.75 (the tail crosses too); pred_b name <= 3.2;
    pred_c census >= 1.3 + inert guard. Null: retrieval patterns sharper
    than composition patterns (dicts stand at the tail). Price: 12.1M
    values replacing the dict store. -> ops/all_qk16.py [QUEUED 18:59Z]
    RUN 2026-08-31, S2283: pred_a/b HELD (census +1.2673, name +2.204),
    pred_c FAILED on the lower floor (outran sanity range). UNIFORM
    GRAMMAR WINS - dicts retired. Claim = rung 188.

188. Registered claim: ALL-QK-16 (identical rebuild). pred_a L2_F <= 1.37;
    pred_b census repro; pred_c L2F repro. Null: 185's print was noise.
    -> ops/frontier_claim_allqk.py [QUEUED 19:10Z]
    RUN 2026-08-31, S2284: ALL HELD - REGISTERED FRONTIER = uniform
    grammar, L2F +1.3497 / census +1.2673. Day path 2.6662 -> 1.3497.

186. All-QK-12 (Hadamard-rank law, above-knee: r*=sqrt(128)~11.3): pred_a
    census <= 1.55; pred_b plateau (<= 1.4673); pred_c floor 0.5 + inert.
    Null: r12 breaks (law constant wrong). Price: 9.1M (25% cheaper than
    r16 if plateau). -> ops/all_qk12.py [QUEUED 19:10Z]
    RUN 2026-08-31, S2285: ALL HELD - r12 crosses (+1.3784, 9.1M,
    25% cheaper). Plateau real.

187. All-QK-10 (below-knee arm; reads r12 receipt at run time): pred_a
    census >= 1.55; pred_b >= r12 + 0.10; pred_c range + inert. Null:
    r10 crosses (knee in [8,10]). -> ops/all_qk10.py [QUEUED 19:10Z]
    RUN 2026-08-31, S2286: pred_a FAILED by 0.03 (r10 +1.5204 - ramp
    not cliff), b/c HELD. Curve r8 2.02 -> r10 1.52 -> r12 1.38 -> r16
    1.27; sqrt(128) inside the ramp. Sizing rule: r >= 12.

189. All-QK-24 (rank curve top): pred_a census <= 1.22 (still yielding);
    pred_b monotone; pred_c floor + inert. Null: plateau (r16 is the
    operating point). Price: 18.2M. -> ops/all_qk24.py [QUEUED 19:30Z]
    RUN 2026-08-31, S2287: ALL HELD - r24 census +1.1287 (curve still
    descending; plateau reading premature). r32 = rung 193.

190. Drop tailE (legacy span-dicts: +0.14 cost, zero compression; name
    holds 0.55 per S2237; coverage note stated): pred_a census <= 1.16;
    pred_b name <= 2.05; pred_c floor + inert. Null: non-additive here.
    -> ops/drop_taile.py [QUEUED 19:30Z]
    RUN 2026-08-31, S2288: ALL HELD - free win confirmed (census
    +1.1153; name 2.204 -> 1.604). tailE leaves the config.

191. Exact front on the uniform grammar: pred_a census <= 1.10; pred_b
    digit <= 0.80; pred_c floor + inert. Null: front marginal shrank.
    Price: +15.9M. -> ops/front4608_qk.py [QUEUED 19:30Z]
    RUN 2026-08-31, S2289: ALL HELD - exact front buys 0.298 here
    (amplified); census +0.9693, first sub-1.0 (descriptive).

192. Uniform grammar two-ledger row: rung-188 rebuild with circuit rows
    saved + era Spearman vs rung-132. pred_a census repro; pred_b valid
    <= 5 (failure = news); pred_c Spearman >= 0.85. Null (c): the
    grammar moved which circuits pay. Price: none.
    -> ops/frontier_uniform_rows.py [QUEUED 19:35Z]
    RUN 2026-08-31, S2290: ALL HELD - era Spearman 0.8765; the circuit
    profile survives its THIRD grammar era; 0/62.

193. All-QK-32 (curve top): pred_a census <= 1.08 (still yielding); pred_b
    monotone; pred_c floor + inert. Null: plateau at last. Price: 24.2M.
    -> ops/all_qk32.py [QUEUED 19:38Z]
    RUN 2026-08-31, S2291: ALL HELD - r32 +1.0191; returns halve per
    octave above the knee.

194. Quality-end combination (front 4608 + r32 + no tailE): pred_a census
    <= 0.80 (retains 2/3 of naive-additive ~0.57); pred_b name <= 1.35;
    pred_c floor 0.4 + inert. Null: interactions eat >= half (> 0.92).
    Price: ~88M. -> ops/combo_max.py [QUEUED 20:00Z]
    RUN 2026-08-31, S2292: ALL HELD - census +0.6201 (92% of additive
    gains retained), name +0.868, fresh +0.6412 descr. Claim = 196.

195. Cheap-end combination (front 3456 + r12 + no tailE): pred_a census
    <= 1.28; pred_b name <= 1.75; pred_c floor + inert. Null: non-
    additive at r12. Price: 56.9M. -> ops/combo_cheap.py [QUEUED 20:00Z]
    RUN 2026-08-31, S2293: ALL HELD (name by 0.022) - census +1.2219
    at 56.9M, beats the r16 frontier cheaper. Claim = 197.

196. Registered claim: quality-end combo (identical rebuild). pred_a L2_F
    <= 0.66; pred_b census repro; pred_c L2F repro. Null: noise. Price:
    ~88M. -> ops/frontier_claim_combo.py [QUEUED 20:29Z]
    RUN 2026-08-31, S2294: ALL HELD - REGISTERED FRONTIER L2F +0.6412
    / census +0.6201. Day path 2.6662 -> 0.6412 (4.2x).

197. Registered claim: cheap-end combo. pred_a L2_F <= 1.33; pred_b census
    repro; pred_c L2F repro. Price: 56.9M.
    -> ops/frontier_claim_cheap.py [QUEUED 20:29Z]
    RUN 2026-08-31, S2295: ALL HELD - cheap point registered (1.31
    fresh / 1.22 census at 56.9M).

198. Day-close circuit ledger (rung-196 rebuild + rows + tau-margins +
    era Spearman): pred_a census repro; pred_b median member/ref <= 1.5
    (was 5.44); pred_c Spearman >= 0.85. Null (b): circuits closed slower
    than aggregate again. -> ops/frontier_combo_rows.py [QUEUED 20:34Z]
    RUN 2026-08-31, S2296: pred_a HELD, b/c FAILED - circuits close
    slower (margin 2.40 vs 1.5 bar); era Spearman 0.767 (profile
    finally budges). tau-curve 0/3/11/22/43.

199. Exact middles decomposition (K=4608 on the day-close config;
    diagnostic, +74M): pred_a census <= 0.50; pred_b >= 0.30 (pattern
    truncation real); pred_c floor + inert. Null: middles marginal
    shrank. -> ops/exact_middles.py [QUEUED 20:34Z]
    RUN 2026-08-31, S2297: FIRST CERTIFICATES EVER - valid 2/62 at
    census +0.1492 (fresh +0.1440). Middles marginal amplified (0.47).
    rep NEGATIVE. Claim = rung 200; r64 endgame = rung 201.

200. MILESTONE CLAIM: exact-middles config with the first certificate bar
    (valid >= 2). pred_a census repro; pred_b valid >= 2; pred_c L2F
    repro. Null: 199's print was noise. Price: ~162M (patterns are the
    only compressed objects). -> ops/frontier_claim_exact.py [QUEUED 21:00Z]
    RUN 2026-08-31, S2298: ALL HELD - first certificate claim
    registered (valid 2/62 at census +0.1492). Era rho 0.43 (profile
    reorganizes near the floor).

201. Rank-64 endgame (confound-free pattern curve, rows saved): pred_a
    census <= 0.10; pred_b valid >= 4; pred_c census in [0.02, 0.13] +
    inert. Null: interaction floor dominates (>= 0.12; certs stuck).
    Price: 48.5M patterns. -> ops/exact_qk64.py [QUEUED 21:00Z]
    RUN 2026-08-31, S2299: ALL HELD - r64: census +0.0852, valid 7/62,
    median margin 1.006. Certificates are a function of pattern rank.
    Claim = 202; r96 = 203.

202. Registered claim r64 (census/L2F repro + valid >= 7): the second
    certificate claim. -> ops/frontier_claim_qk64.py [QUEUED 21:30Z]
    RUN 2026-08-31, S2300: ALL HELD - second certificate claim
    (valid 7/62 at census +0.0852). Four-point registered set.

203. Rank-96 (curve's last point; 25% compression, science end): pred_a
    census <= 0.055; pred_b valid >= 12; pred_c floor + inert. Null:
    interaction floor / certificate stall.
    -> ops/exact_qk96.py [QUEUED 21:30Z]
    RUN 2026-08-31, S2301: 1/3 bars - census +0.0622 (missed 0.055 by
    0.007), valid 9 (missed 12); null floor (>=0.07) also wrong. Curve
    bends: octave gains 0.064 -> 0.023. Median circuit below 1x ref
    (0.857) for the first time.

204. Mixed-rank allocation (tail r64 / motifs r32, 36.0M patterns, 26%
    cheaper than uniform r64): pred_a census <= 0.11; pred_b valid >= 5;
    pred_c range + inert. Null: rank need uniform (>= 0.117).
    -> ops/mixed_rank.py [QUEUED 21:34Z]
    RUN 2026-08-31, S2302: 1/3 bars - +0.1375, valid 2; null supported
    (rank need uniform across blocks). Block-grain allocation dead.

205. Global quarter-density comparator (fixed top-1152/4608 per MLP,
    front+middles, r64 base): pred_a census >= 0.30 (amplified marginals);
    pred_b valid <= 4; pred_c range + inert. Null: pruning stays cheap.
    -> ops/cp_global1152.py [QUEUED 21:52Z]
    RUN 2026-08-31, S2303: ALL HELD - fixed top-1152 collapses: +2.2048,
    valid 0/62, margin 4.25x; era rho 0.79 (recreates old-era profile).

206. PER-TOKEN top-1152 of full 4608 dictionary (|u|*||D|| selection,
    same avg active count, r64 base) — user-directed SPD/top-k lane:
    pred_a census <= 0.15; pred_b valid >= 5; pred_c range + inert.
    Null: usage dense (within 0.8x of global). Measures usage structure,
    not storage. -> ops/cp_topk1152.py [QUEUED 21:52Z]
    RUN 2026-08-31, S2304: 2/3 bars - per-token top-1152: +0.1015,
    21.7x better than fixed subset at matched density; +0.016 above
    full base; valid 4 (missed 5 by one). Sparse usage CONFIRMED;
    the sparse set rotates per token.

207. GAUGE-FREE BASIS AT m16 (user directive: the neuron h-dimension is
    gauge). Eigenfeatures of the invariant bilinear tensor (output basis
    = SVD of Down; per-direction interaction eigenpairs; exact at full
    spectrum) vs neuron top-1152 at matched component count (EIG 3x
    fewer stored values). pred_a own-circuit ratio <= 0.5; pred_b census
    ratio <= 0.7; pred_c exactness relerr <= 1e-3 + non-inert. Null:
    neuron basis already aligned (ratios >= 0.9).
    -> ops/m16_eigenbasis.py [QUEUED 22:02Z]
    RUN 2026-08-31, S2305: 1/3 bars, INSTRUMENT-FLAGGED - exactness
    tripwire failed (max relerr 7.36; suspected denominator-floor bar
    mis-design + fp16 eigvecs). Ratios 0.770 own / 0.655 census
    QUARANTINED. v2 = rung 209.

208. Per-token top-k density sweep k=576 (1/8 density, r64 base):
    pred_a census <= 0.20; pred_b valid >= 2; pred_c range + double
    inert. Null: quarter was the knee (>= 0.14).
    -> ops/cp_topk576.py [QUEUED 22:06Z]
    RUN 2026-08-31, S2306: ALL HELD - +0.1495, valid 2.

209. m16 eigenbasis v2 (fixed exactness instrument: rel-Frobenius +
    max/RMS; fp32 eigvecs) — the S2305 quarantine lifter: pred_a
    exactness certifies; pred_b own-ratio <= 0.9; pred_c census ratio
    <= 0.8. Null: quarantined advantage was artifact, or real bug
    (pred_a fails -> publish nothing, hunt).
    -> ops/m16_eigenbasis_v2.py [QUEUED 22:08Z]
    RUN 2026-08-31, S2307: pred_a FAILED (max/RMS 1.54e-2 vs 1e-2;
    frob certifies 4e-4); b,c HELD (0.770/0.655, bit-identical to fp16
    run). Quarantine holds; fp64 control = rung 211.

210. Per-token top-k density k=288 (1/16, curve completion): pred_a
    census <= 0.45; pred_b valid >= 1; pred_c range + inert. Null: knee
    above 1/16 (>= 0.5). -> ops/cp_topk288.py [QUEUED 22:08Z]
    RUN 2026-08-31, S2308: ALL HELD - +0.2692, valid 1. Curve complete:
    excess +0.016/+0.064/+0.184 at 1/4, 1/8, 1/16 density.

211. fp64 exactness control at m16 (no census; S2307 quarantine lifter):
    pred_a FP64 max/RMS <= 1e-6; pred_b FP64 frob <= 1e-9; pred_c FP32
    repro in [0.5e-2, 5e-2]. Null: fp64 residual >= 1e-4 = real bug.
    -> ops/m16_exact_fp64.py [QUEUED 22:09Z]
    RUN 2026-08-31, S2309: ALL HELD - fp64 at machine precision
    (4.33e-11); fp32 floor reproduced. QUARANTINE LIFTED: eigenbasis
    beats neurons 0.770 own / 0.655 census at 3x fewer values.
    Instrument rule filed (rel-Frobenius + fp64 control).

212. HOSVD shared multilinear basis at m16 (W = eigvecs of sum QQ^T;
    cores W^T Q_d W; rin 34 matched to EIG values, rin 128 headroom):
    pred_a own <= 0.9 x NEUR; pred_b census <= 1.1 x EIG; pred_c
    monotone + HOSVD128 <= 0.5. Null: shared basis no better than
    neurons. -> ops/m16_hosvd.py [QUEUED 22:12Z]
    RUN 2026-08-31, S2310: ALL FAILED - HOSVD34 own 1.73 (worse than
    neurons), census 1.09; rin 34->128 changes nothing (plateau).
    Diagnosis = rung 214 (flat spectrum vs bug).

213. Sketch-gated top-k (rank-64 gate scores units, exact compute on
    selected 1152; 16x cheaper selection): pred_a census <= 0.30;
    pred_b valid >= 1; pred_c range + double inert. Null: selection
    needs full products (>= 1.76). -> ops/cp_sketch64.py [QUEUED 22:15Z]
    RUN 2026-08-31, S2311: 1/3 - sketch64 +1.0858: bar failed AND null
    failed (in between: half the gap recovered, certificates dead).
    Rank-256 gate = rung 217.

214. HOSVD plateau diagnosis (identity check hook-path vs direct
    projection; capture curve rin 34/128/512): pred_a no bug (<=1e-3);
    pred_b flatness (cap128-cap34 <= 0.10); pred_c cap512 rises. Null:
    bug, or Frobenius-CE mismatch. -> ops/m16_hosvd_diag.py [QUEUED 22:18Z]
    RUN 2026-08-31, S2312: ALL HELD - no bug (2.5e-7); capture
    0.6%/4.6%/39% at 34/128/512. Full multilinear rank; Tucker CLOSED
    at m16.

215. Eigenbasis generalization at m14 (identical S2309 instrument):
    pred_a exactness; pred_b own ratio <= 0.9; pred_c census <= 0.8.
    Null: m16-specific. -> ops/m14_eigenbasis.py [QUEUED 22:18Z]
    RUN 2026-08-31, S2313: 1/3 - advantage does NOT transfer (1.264
    own / 1.219 census; neuron wins at m14). Module-dependent; m13
    test = rung 216.

216. Basis competition at m13 (identical instrument; 1-1 so far):
    pred_a exactness; pred_b NEURON HOLDS (own ratio >= 0.95); pred_c
    census ratio >= 0.9. Null: eigen wins again (split needs a
    predictor). -> ops/m13_eigenbasis.py [QUEUED 22:28Z]
    RUN 2026-08-31, S2314: ALL HELD - neuron holds at m13 (1.539 own /
    1.410 census). 2-1 neurons; hypothesis: eigen wins only at
    high-damage modules. Decisive test m17 = rung 218.

217. Sketch-gate rank curve: rank-256 gate (4x cheaper selection):
    pred_a census <= 0.35; pred_b valid >= 1; pred_c range + inert vs
    ORACLE/SKETCH64. Null: gate fidelity saturates (>= 0.8).
    -> ops/cp_sketch256.py [QUEUED 22:28Z]
    RUN 2026-08-31, S2315: 2/3 - +0.4075 (missed 0.35 by 0.057; null
    rejected). Gate curve 64->1.086, 256->0.408, oracle 0.102: rank-
    dependent, not saturating; sketches pay a fidelity tax.

218. DECISIVE basis test at m17 (highest knockout 2.52; hypothesis says
    eigen wins at concentrated modules): pred_a exactness; pred_b EIG
    own ratio <= 0.9; pred_c census <= 0.85. Null: neuron holds -
    hypothesis dies. -> ops/m17_eigenbasis.py [QUEUED 22:31Z]
    RUN 2026-08-31, S2316: ALL HELD - eigen wins m17 by 2.5x (0.398
    own / 0.466 census). Monotone predictor across 4 modules; m5
    midpoint + Spearman prereg = rung 220.

219. Usage concentration by basis at m16 (dict 4608 both bases,
    per-token 1152 active): pred_a NEURTK <= 0.5 x static 1.0146;
    pred_b EIGTK <= 0.9 x NEURTK; pred_c EIGTK <= E4608 + 0.02 + non-
    inert. Null: concentration basis-independent (>= 0.95).
    -> ops/m16_eig_topk.py [QUEUED 22:33Z]
    RUN 2026-08-31, S2317: 2/3, pred_b failed by 19x REVERSED - usage
    sparsity is a NEURON-basis phenomenon (NEURTK 0.038 nearly free;
    eigen dict barely responds to selection). Mechanisms factorize by
    basis: static->eigen, routing->neurons.

220. Basis law quantified: m5 midpoint + five-module Spearman
    (difficulty vs eigen advantage): pred_a exactness; pred_b rho
    <= -0.75; pred_c m5 ratio in [0.3, 2.0]. Null: |rho| < 0.5 (late-
    layer effect, not concentration). -> ops/m5_eigenbasis.py [QUEUED 22:36Z]
    RUN 2026-08-31, S2318: 2/3 - Spearman -0.600 missed -0.75; smooth
    law dead, two-regime threshold survives (easy->neuron 3/3,
    hard->eigen 2/2). Depth confound standing; sweep = rung 221.
    BUILDER NOTE: first build failed an assert (_need anchor differs
    across script generations); rebuilt + enqueued 22:39Z. Rule: verify
    dryrun-header anchors per file, they are not lineage-stable.

221. Basis-law disambiguation sweep (m2/m7/m9/m11, in-run difficulty +
    ratio; 9 points total): pred_a exactness x4; pred_b two-regime no
    violations; pred_c 9-pt Spearman(diff) <= -0.5. Null: depth rules
    (Spearman(block) <= -0.8 while c fails).
    -> ops/basis_sweep.py [QUEUED 22:45Z]
    RUN 2026-08-31, S2319: 2/3 - threshold holds (0 violations, 9
    pts); graded law dead (rho_diff -0.433, rho_block -0.750, both
    miss). Eigen regime = {m16,m17} = high-diff AND terminal; confound
    unresolvable naturally. Operational rule fixed.

222. TOKEN-INDEXED SPARSITY at m0 (user directive: install sparsity as
    a static selection tensor indexed by token id, derived from pure
    embeddings — no runtime gate): pred_a TOKID <= 0.25 x STATIC;
    pred_b <= 2 x ORACLE; pred_c ORACLE <= STATIC + non-inert. Null:
    context dominates at block 0 (>= 0.8 x STATIC). Price: naive table
    58M int16 indices. -> ops/tokid_sparsity.py [QUEUED 22:48Z]
    RUN 2026-08-31: CRASHED (exit 1) - wte is padded to 50304 rows vs
    V=50257; last vocab chunk mismatch. Instrument fix (Vw from
    weight.shape[0]); re-queued 22:58Z. No science scored.
    RUN 2026-08-31, S2321: 2/3 - TOKID/STATIC 0.184 (HELD, 85% of the
    gap closed by a static table); TOKID/ORACLE 4.24 (FAILED <=2.0 -
    context residual real). Null rejected. Follow-ups: 224 (data
    table), 225 (m2 depth test).

223. Causal validation of the top-k score at m16 (user question): 16
    score-bands of 288 units, zero one band per census pass; pred_a
    Spearman(score, causal damage) >= 0.8; pred_b top/bottom >= 10x;
    pred_c inversions <= 3 + non-inert. Null: rho < 0.4 (score
    misranks; causal correction needed).
    -> ops/unit_causal_bands.py [QUEUED 22:55Z]
    RUN 2026-08-31, S2320: ALL HELD - Spearman 0.985, 40x range,
    inversions 3 (knife edge, flat tail). Score causally faithful at
    band grain; no correction needed.

224. Data-derived token table at m0 (per-id mean score over FW fit
    windows, embedding fallback; leakage-free): pred_a <= 0.6 x TOKID
    (0.0887); pred_b <= 2.5 x ORACLE (0.0873); pred_c improves + cov
    >= 0.90. Null: residual is positional (>= 0.9 x TOKID).
    -> ops/tokid_data.py [QUEUED 23:05Z]
    RUN 2026-08-31, S2322: 0/3 - data table 0.1242 (16% better where
    40% demanded; coverage 0.871 < 0.90 caveat). Null supported: the
    residual is per-occurrence context; hierarchy (class index), not
    bigger token tables.

225. Token identity at depth: embedding table at m2 (STATIC diff 0.22):
    pred_a TOKID <= 0.5 x STATIC; pred_b ORACLE <= 0.05; pred_c
    ORACLE <= TOKID <= STATIC. Null: identity dead by block 2 (>= 0.8x)
    -> table needs class indexing deeper. -> ops/tokid_m2.py [QUEUED 23:05Z]
    RUN 2026-08-31, S2323: 2/3 - TOKID/STATIC 0.771 at m2 (vs 0.184 at
    m0); identity fades by block 2, null narrowly not triggered.
    ORACLE 0.035 - routing near-free, third module in a row. Class-
    indexed second level = next.

226. LEVEL-2 HIERARCHY at m2: input-class index (prefix-only, no
    target leak; fit on FW): pred_a CLSID <= 0.9 x TOKID; pred_b HIER
    (s_tok + s_cls) <= 0.8 x TOKID; pred_c consistency. Null: 10
    classes too coarse. -> ops/clsid_m2.py [QUEUED 23:15Z]
    RUN 2026-08-31, S2324: 1/3 - 10-class index too coarse (CLSID
    0.2082 ~ STATIC; HIER 0.1634 only 4% under TOKID); 79% of positions
    in 3 catch-all classes. Null supported; learned/finer index next
    (k-means clusters over early-block context).

227. Hierarchy control at m0 (ordering flip claim): pred_a CLSID >=
    TOKID (token beats class at front); pred_b HIER <= 0.95 x TOKID;
    pred_c consistency. Null: hierarchy flat (HIER <= 0.8 x TOKID).
    -> ops/clsid_m0.py [QUEUED 23:15Z]
    RUN 2026-08-31, S2325: 2/3 - ordering flip CONFIRMED (CLSID 0.2478
    >> TOKID 0.1478 at front); pred_b knife-edge miss by 0.0022 (3.5%
    gain vs 5% demanded, bar luck). Hand-built class level too weak
    everywhere; learned index (228) is the live candidate.

228. LEARNED second-level index at m2: k-means-256 over module inputs
    (fit on FW), per-cell selection rows; runtime = nearest centroid ->
    gather (vector-quantized gate, still a static tensor): pred_a KM256
    <= 0.8 x TOKID; pred_b KMHIER <= 0.75 x TOKID; pred_c consistency.
    Null: quantized context cannot select (>= 0.95 x TOKID).
    -> ops/kmid_m2.py [QUEUED 23:22Z]
    RUN 2026-08-31, S2326: 1/3 - KM256 0.1406 (17% vs 20% demanded);
    KMHIER regressed (blend dilutes). Static families converge ~0.14
    vs oracle 0.035: depth selection is per-occurrence. Codebook
    scaling = 229.

229. Codebook scaling: km-1024 at m2: pred_a <= 0.125 (still
    improving); pred_b >= 0.08 (plateau short of oracle); pred_c
    consistency + inert vs km-256. Null: saturated at 256 (>= 0.135).
    -> ops/kmid1024_m2.py [QUEUED 23:32Z]
    RUN 2026-08-31, S2327: 2/3 - km-1024 0.1323 (missed 0.125 by
    0.007); 4x cells bought 0.008. Saturated. CHAPTER CLOSED: front
    static (85%), depth irreducibly dynamic (~0.13 plateau vs 0.035
    oracle); index blends dilute. No more index variants without a
    new mechanism.

230. Token table at m1 (decay curve middle point; 0.184 at m0 vs
    0.771 at m2): pred_a TOKID <= 0.35 x STATIC; pred_b ORACLE <=
    0.05; pred_c consistency. Null: fade done by block 1 (>= 0.6).
    -> ops/tokid_m1.py [QUEUED 23:32Z]
    RUN 2026-08-31, S2328: 2/3 - smooth decay 0.184/0.492/0.771 over
    blocks 0/1/2, no knee; oracle near-free at 4th module. Table worth
    it at m0 only. Architecture composition = 231/232.

231. ARCHITECTURE COMPOSITION I: m0 static token table inside the r64
    config (base 0.0852): pred_a census <= 0.22 (near-additive with the
    0.113 single-site gap); pred_b >= 0.13 (not free); pred_c valid >=
    2 + inert. Null: super-additive (>= 0.30).
    -> ops/tokid_front_config.py [QUEUED 23:45Z]
    RUN 2026-08-31, S2329: 1/3 - census +0.2844 vs additive ~0.198
    (interaction excess ~0.086; null 0.30 not triggered); valid 1.
    rep/name pay most. Single-site costs understate in-config when
    downstream is compressed.

232. ARCHITECTURE COMPOSITION II: full S2327 architecture (tables at
    m0+m1, dynamic top-1152 deep, r64 patterns): pred_a census <= 0.45
    (additive ~0.36); pred_b >= 0.20; pred_c valid >= 1 + inert. Null:
    >= 0.55. -> ops/arch_full_config.py [QUEUED 23:45Z]
    RUN 2026-08-31, S2330: 1/3, NULL SUPPORTED - +0.6910 vs additive
    ~0.36 (past the 0.55 null); valid 0. Static front tables not
    deployable in-config; composition CLOSED without a propagation
    repair. Architecture menu: r64 (0.0852/7) or +dynamic (0.1015/4).

233. Table compression: 4096 frequency-selected rows (12x smaller,
    4.7M indices; static fallback for rare tokens): pred_a <= 1.3 x
    full-table TOKID; pred_b <= 0.5 x STATIC; pred_c coverage >= 0.90
    + non-degenerate. Null: tail rows matter (>= 1.6x).
    -> ops/tokid_freq4096.py [QUEUED 22:58Z]
    RUN 2026-08-31, S2331: 1/3, null supported - 0.3348 = 2.27x full
    table; coverage 0.739 explains it arithmetically (mixing predicts
    0.318). Tail rows not special; thread parked with S2330.

234. CIRCUIT-AWARE rank allocation (exact patterns at blocks 3/8/16 —
    34 circuits' top components; r64 elsewhere): pred_a census <=
    0.075; pred_b valid >= 9 (r96's count at ~1/3 the step price);
    pred_c range + inert. Null: circuit load doesn't predict rank need.
    -> ops/circuit_rank.py [QUEUED 23:20Z]
    RUN 2026-08-31, S2332: 1/3, null effectively supported - 0.0771
    (knife-edge miss) and ZERO new certs (7, unchanged). Circuit load
    doesn't predict rank need; failing circuits bottlenecked elsewhere.
    Allocation stays uniform; principle CLOSED.

235. Registered claim: dynamic-MLP Pareto point (S2304 official):
    |census-0.1015|<=0.015; valid >= 4; |L2F-0.1018|<=0.015.
    -> ops/frontier_claim_topk.py [QUEUED 23:20Z]
    RUN 2026-08-31, S2333: ALL HELD - dynamic-MLP point official
    (+0.1013 / 4 certs / 4x compute-sparse MLPs).

236. Registered claim r96 (the 9-certificate point): |census-0.0622|
    <=0.015; valid >= 9; |L2F-0.0626|<=0.015.
    -> ops/frontier_claim_qk96.py [QUEUED 23:33Z]
    RUN 2026-08-31, S2334: ALL HELD, bit-exact - r96 point official
    (+0.0622 / 9 certs / ~200M).

237. HALVES TEST I: all motif blocks (2-9) exact, tail r64: pred_a
    census <= 0.070; pred_b valid >= 8; pred_c range + double inert.
    Null: bottleneck is tail-half (valid <= 7). Tail mirror = 238.
    -> ops/motif_exact.py [QUEUED 23:33Z]
    RUN 2026-08-31, S2335: ALL HELD - +0.0608 / 9 certs; motif half is
    the bottleneck (spread, not concentrated); same price as r96,
    better census. Tail mirror = 238; claim = 239.

238. HALVES TEST II: tail (10-17) exact, motifs r64 (mirror): pred_a
    census <= 0.075; pred_b valid <= 8 (tail buys little); pred_c range
    + double inert. Null: both halves unlock equally (>= 9).
    -> ops/tail_exact.py [QUEUED 23:44Z]
    RUN 2026-08-31, S2336: 2/3 - +0.0776 (knife-edge miss), valid 8;
    null rejected. Asymmetry confirmed: motif half ~3x residual per
    value; tail nearly saturated at r64.

239. Registered claim: motif-exact point (r96 price, better census):
    |census-0.0608|<=0.015; valid >= 9; |L2F-0.0601|<=0.015.
    -> ops/frontier_claim_motif.py [QUEUED 23:44Z]
    RUN 2026-08-31, S2337: ALL HELD, bit-exact - motif-exact point
    official (+0.0608/9 certs/~199M); supersedes r96.

240. Cheapen the saturated half: motifs exact + tail r48 (65.9M, 9%
    below the registered point): pred_a census <= 0.075; pred_b valid
    >= 8; pred_c range + double inert. Null: r64 was the tail knee.
    -> ops/motif_tail48.py [QUEUED 23:57Z]
    RUN 2026-08-31, S2338: 2/3 - census +0.0661 (cheap) but valid
    9->7: tail saturated for census, NOT for certificates; r64 is the
    tail's certificate knee. {motifs exact, tail r64} is a corner.

241. Quarter test: early motifs (2-5) exact only: pred_a census <=
    0.075; pred_b valid >= 8; pred_c range + double inert. Null:
    spread evenly within the motif half.
    -> ops/motif_early.py [QUEUED 23:57Z]
    RUN 2026-09-01, S2339: ALL HELD - early quarter (2-5) captures 71%
    of the motif census gain + 1 of 2 certs at half the spend. Need
    concentrates early. Late complement = 242; graded r96 = 243.

242. Quarter test II: late motifs (6-9) exact only: pred_a census <=
    0.080; pred_b valid >= 8; pred_c range + double inert. Null: late
    motifs carry nothing distinct. -> ops/motif_late.py [QUEUED 00:12Z]
    RUN 2026-09-01, S2340: 2/3 - late quarter: +0.0768, ZERO new certs.
    Quarter ledger: early 0.0680/8, late 0.0768/7, half 0.0608/9 -
    certificate #9 is super-modular (joint property of the stack).

243. Graded motif rank: motifs r96 + tail r64 (60.3M): pred_a census
    <= 0.070; pred_b valid >= 8; pred_c range + double inert. Null:
    motif certificates need exactness (corner sharp in rank).
    -> ops/motif_r96.py [QUEUED 00:12Z]
    RUN 2026-09-01, S2341: ALL HELD - +0.0676/8 at 60.3M: smooth in
    census (72% of gain at 84% price), sharp in certificates (#9 needs
    full exactness). Two-tier menu fixed.

244. Compute-sparse corner: motif-exact patterns + per-token top-1152
    MLPs (additive ~0.077): pred_a census <= 0.085; pred_b valid >= 6;
    pred_c range + double inert. Null: sparsity interacts with motif
    exactness (>= 0.10 or valid <= 4). -> ops/corner_topk.py [QUEUED 00:27Z]
    RUN 2026-09-01, S2343: ALL HELD - +0.0776/7 certs; sparsity
    surcharge additive (+0.0168 vs +0.0161 on r64 base): top-k does
    NOT super-add, unlike static tables. S2342 (receipt analysis):
    cert #9 = r.2.0.2 (a8), margin additive 1.159->0.985 - count
    super-modularity is threshold-crossing; failures dominated by
    a8 (14) + a16 (13) families.

245. Registered claim: economical shape {motifs r96, tail r64}:
    |census-0.0676|<=0.015; valid >= 8; |L2F-0.0660|<=0.015.
    -> ops/frontier_claim_motif96.py [QUEUED 23:33Z]
    RUN 2026-09-01, S2344: ALL HELD, bit-exact - economical shape
    official (+0.0676/8 certs/~187M).

246. Targeted tail exactness: corner + block 16 exact (aimed at the
    a16/m16 failure family; +2.9M): pred_a census <= 0.058; pred_b
    valid >= 10 (r.6.2.2 at 1.03 crosses); pred_c range + inert. Null:
    family not block-16-rank-limited. -> ops/corner_b16.py [QUEUED 23:38Z]
    RUN 2026-09-01, S2345: 1/3, null supported - block-16 exact moved
    census 0.0002 (wobble), valid unchanged. Targeted exactness 0-for-2;
    margins respond to broad rank only.

247. Top of the Pareto: corner + tail r96 (+12.4M): pred_a census <=
    0.058; pred_b valid >= 9; pred_c range + double inert. Null: tail
    beyond the knee buys nothing. -> ops/corner_tail96.py [QUEUED 23:38Z]
    RUN 2026-09-01, S2346: ALL HELD - NEW BEST: +0.0553 / 11 certs at
    84.5M patterns. The 1.03 pair (r.2.0, r.6.2.2) crossed via broad
    tail octave (block-16 alone moved nothing). Claim = 248; r112 = 249.

248. Registered claim: corner + tail r96 (the 11-cert point):
    |census-0.0553|<=0.015; valid >= 11; |L2F-0.0565|<=0.015.
    -> ops/frontier_claim_ct96.py [QUEUED 23:48Z]
    RUN 2026-09-01, S2347: ALL HELD, bit-exact - 11-cert point official
    (+0.0553/11/~211M).

249. Next tail octave: corner + tail r112 (90.7M): pred_a census <=
    0.050; pred_b valid >= 12; pred_c range + double inert. Null: r96
    was the last paying octave. -> ops/corner_tail112.py [QUEUED 23:48Z]
    RUN 2026-09-01, S2348: 1/3, null supported - r112: +0.0531/11
    (flat). SPECTRAL CLIFF: the tail's last 16 ranks carry ~96% of
    remaining damage. Cliff check = 250; sparse frontier = 251.

250. Spectral cliff check: corner + tail r120 (93.6M): pred_a CLIFF
    census >= 0.035; pred_b valid <= 12; pred_c range + double inert.
    Null: smooth approach (<= 0.030). -> ops/corner_tail120.py [QUEUED 00:01Z]
    RUN 2026-09-01, S2349: ALL HELD - CLIFF CONFIRMED: r120 still
    +0.0526; the tail's last 8 singular directions carry ~all residual.
    (Cosmetic pred_b label bug noted; logic correct.) Low-rank pattern
    chapter closed at its ceiling.

251. Compute-sparse frontier: ct96 + per-token top-1152 (additive
    ~0.072): pred_a census <= 0.075; pred_b valid >= 8; pred_c range +
    double inert. Null: sparsity interacts at the frontier.
    -> ops/ct96_topk.py [QUEUED 00:01Z]
    RUN 2026-09-01, S2350: ALL HELD - +0.0718 (additive dead-on);
    top-k surcharge ~+0.016 at all three anchors. Sparsity composes
    additively everywhere; static tables don't.

252. LAST-8 HYBRID: tail = top-96 + smallest-8 directions (104/128,
    74.2M — cheaper than the 11-cert point): pred_a census <= 0.015
    (cliff mass is literally the last 8); pred_b valid >= 15; pred_c
    range + double inert. Null: 97-128 band matters jointly (>= 0.045).
    -> ops/tail_last8.py [QUEUED 00:14Z]
    RUN 2026-09-01, S2351: 1/3, NULL WINS - hybrid 0.0539 (vs ~0.01
    additive prediction). Damages of disjoint fine-band subsets overlap
    (~0.05 each): binary in rank deficiency; one shared cancellation
    mechanism. S2348/S2349 attribution corrected (numbers stand).
    Mixed spectrum dead; ceiling stands at 0.0553/11.

253. ALL-MIXED spectrum: top-96 + last-8 at every head incl motifs
    (52.9M, cheaper than economical): pred_a census <= 0.030 (motif
    cliff too); pred_b valid >= 12; pred_c range + double inert. Null:
    cliff is tail-specific. -> ops/all_last8.py [QUEUED 00:14Z]
    RUN 2026-09-01, S2352: preds failed as written, null supported -
    BUT +0.0573/11 at 52.9M supersedes the economical shape on every
    axis. Claim = 254. S2353 (analysis): profile Spearman 0.993-0.999
    across band truncations - ONE shared mechanism; direction-subset
    engineering closed.

254. Registered claim: mixed-spectrum point (supersedes economical):
    |census-0.0573|<=0.015; valid >= 11; |L2F-0.0584|<=0.015.
    -> ops/frontier_claim_mixed.py [QUEUED 00:28Z]
    RUN 2026-09-01, S2354: ALL HELD, bit-exact - mixed-spectrum point
    official (+0.0573/11/~180M); replaces economical. (Cosmetic pred_c
    label bug noted; builder rule upgraded.)

255. OV CHAPTER OPENS: value maps r64 at all replaced heads on the
    mixed base (+~9.0M factors): pred_a census <= 0.075; pred_b valid
    >= 9; pred_c range + inert. Null: values as rank-hungry as patterns
    (>= 0.10). -> ops/value_r64.py [QUEUED 00:28Z]
    RUN 2026-09-01, S2355: 1/3, NULL SUPPORTED - value r64 costs
    +0.0722 surcharge, certs 11->3. Values ~3x more expensive than
    tail patterns at the same octave. Curve up = 256/257.

256. Value rank curve I: v_r=96 on mixed base: pred_a census <= 0.085;
    pred_b valid >= 9; pred_c range + double inert. Null: binary like
    patterns (>= 0.11). -> ops/value_r96.py [QUEUED 00:44Z]
    RUN 2026-09-01, S2356: 2/3 - v96: +0.0763 (surcharge 0.019; binary
    null rejected), valid 8 (missed 9 by one). Values price smoothly,
    unlike the patterns' binary band.

257. Value rank curve II: v_r=112 (binary-structure test): pred_a
    census <= 0.070 (smooth); pred_b valid >= 10; pred_c range +
    double inert. Null: value side binary too (>= 0.09).
    -> ops/value_r112.py [QUEUED 00:44Z]
    RUN 2026-09-01, S2357: 2/3 - v112 surcharge +0.0062/9 certs. Value
    curve smooth (no binary band): two-sided asymmetry law. Values stay
    real; OV curve closed as science.

258. Head-parity band test I: fine band missing at ODD tail heads only
    (others exact): pred_a census <= 0.040 (head-additive); pred_b
    valid >= 11; pred_c range + double inert. Null: mechanism global
    across heads (>= 0.048). -> ops/band_oddheads.py [QUEUED 01:00Z]
    RUN 2026-09-01, S2358: 2/3, null supported - odd-heads-only band
    deficiency costs 0.0544 (~98% of full-band 0.0553). Mechanism
    global across heads as well as directions.

259. Head-parity band test II: EVEN heads (mirror). Same bars/null.
    -> ops/band_evenheads.py [QUEUED 01:00Z]
    RUN 2026-09-01, S2359: INSTRUMENT FAIL self-caught - scalar-
    proximity tripwire collides with a null that predicts landing on
    the anchor. Rule filed; S2358 parity labels corrected (halves
    swapped, conclusion stands). Rerun = 260; quarter = 261.

260. Head-parity mirror v2 (odd heads deficient; fixed prereg): pred_a
    census in [0.048, 0.062] (global-symmetric); pred_b valid >= 11;
    pred_c range. Null: asymmetric (< 0.048).
    -> ops/band_mirror.py [QUEUED 01:28Z]
    RUN 2026-09-01, S2360: ALL HELD - mirror 0.0527: global-symmetric.
    Every partial deficiency lands 0.0526-0.0553. One global object.

261. Quarter-deficiency (18/72 heads): pred_a census >= 0.045 (extreme
    globality); pred_b valid >= 11; pred_c range. Null: partial
    additivity below half (<= 0.035). -> ops/band_quarter.py [QUEUED 01:28Z]
    RUN 2026-09-01, S2361: ALL HELD - quarter-deficiency 0.0524 (~95%
    of full). CAMPAIGN CLOSED: one maximally delicate global mechanism;
    grammar floor ~+0.052/11 certs; mixed point sits on it.

262. Compute-sparse mixed point (mixed patterns + top-1152 MLPs;
    additive ~0.073): pred_a census <= 0.078; pred_b valid >= 8;
    pred_c range. Null: interaction (>= 0.088 or <= 6).
    -> ops/mixed_topk.py [QUEUED 02:00Z]
    BUILDER NOTE: first build failed a multiline pc anchor (tripwire
    block layout drifted); rebuilt line-wise, enqueued 02:05Z.
    RUN 2026-09-01, S2362: ALL HELD - +0.0736/8 (surcharge +0.0163,
    fourth additive anchor within 0.0008). Best-value config on joint
    axes.

263. Motif band in isolation ({motifs r96, tail EXACT} — cross-half
    overlap test + tail-free certificate ceiling): pred_a census <=
    0.012 (independent mechanisms); pred_b valid >= 20 (margins were
    tail-pinned); pred_c range. Null: cross-half overlap (>= 0.03) or
    ceiling stays. -> ops/motif_band_iso.py [QUEUED 00:30Z]
    RUN 2026-09-01, S2363: 0/3, null dramatic - motif band alone costs
    0.0589 (overlap ~0.047 with tail's 0.0553; both 0.0676). Mechanism
    is MODEL-WIDE; ceiling pinned at ~11 regardless of deficiency
    location. Floor law reframed.

264. MANIPULABILITY I: knockout-transfer at m16 inside the mixed config
    (mean-ablation vs battery refs + removal_matrix row): pred_a own-
    median ratio in [0.7, 1.6]; pred_b collateral Spearman >= 0.8;
    pred_c ablated census in [0.2, 5]. Null: compiled object doesn't
    support faithful knockouts. -> ops/ko_transfer_m16.py [QUEUED 00:40Z]
    RUN 2026-09-01, S2364: ALL HELD - compiled knockout faithful: own
    ratio 1.173, collateral Spearman 0.858, fingerprint flips to the
    knockout's (era rho -0.21). First manipulability win.

265. MANIPULABILITY II: same at m13. -> ops/ko_transfer_m13.py [QUEUED 00:40Z]
    RUN 2026-09-01, S2365: 1/3 - own ratio 1.319 held; collateral rho
    0.649 and census 0.1027 failed. Signal-to-floor law proposed:
    small components' collateral sits inside the compression floor.
    Test pair = 266 (m17, big) / 267 (m14, small).

266. Signal-to-floor test, BIG arm: m17 knockout in the mixed config
    (own-mean 2.52): pred_a own ratio in [0.7,1.6]; pred_b Spearman >=
    0.8; pred_c census in [0.5,5]. Null: rho < 0.7 despite huge signal.
    -> ops/ko_transfer_m17.py [QUEUED 01:03Z]
    RUN 2026-09-01, S2366: 1/3 - own ratio 0.987 (perfect transfer);
    collateral rho 0.761 (missed 0.8 by 0.039), census 0.485 (missed
    floor by 0.015). Fidelity NOT monotone in signal (m16 0.858 > m17
    0.761). Law verdict deferred to m14.

267. Signal-to-floor test, SMALL arm: m14 (own-mean 0.596): pred_a own
    ratio in [0.7,1.6]; pred_b Spearman < 0.8 (degradation predicted);
    pred_c census in [0.08,0.5]. Null: rho >= 0.85 (floor hypothesis
    dies). -> ops/ko_transfer_m14.py [QUEUED 01:03Z]
    RUN 2026-09-01, S2367: ALL HELD - m14 rho 0.566 as predicted. LAWS
    SET: own-effect transfer universal (4/4, 0.99-1.33); collateral
    two-regime (small 0.57-0.65 inside floor noise; large 0.76-0.86).
    Attention-side replication = 268/269.

268. Attn-side knockout transfer: a16 (13 own circuits; ablation hook
    registered at census site, after tail permanent hooks): pred_a own
    ratio in [0.7,1.6]; pred_b Spearman in [0.45,0.85); pred_c census
    in [0.08,0.8]. Null: attn knockouts don't transfer.
    -> ops/ko_transfer_a16.py [QUEUED 01:22Z]
    RUN 2026-09-01, S2368: 1/3, NULL TRIGGERED (overshoot): own ratio
    2.404 (>2), census +0.0665 (below band), fingerprint compression-
    like (rho 0.727). Frame-mixing confound flagged (real-frame mean at
    a replaced site). Attn manipulability unproven pending diagnosis.

269. Attn-side: a14 (1 own circuit). Same structure, census band
    [0.08,0.6]. -> ops/ko_transfer_a14.py [QUEUED 01:22Z]
    RUN 2026-09-01, S2369: 1/3 - overshoot replicates (own 1.740,
    census knife-edge under band, fingerprint unchanged). Frame-mixed
    ablation value is the prime suspect; config-frame-mean diagnosis
    = 270.

270. FRAME DIAGNOSIS: a16 knockout with the config-frame mean (captured
    from the compiled a16 output, 64 FW rows): pred_a own ratio in
    [0.7,1.6] (confound resolved); pred_b era rho <= 0.45 (fingerprint
    flips); pred_c census band + collateral >= 0.4. Null: overshoot
    persists (own > 1.8) - replaced-path dynamics distort.
    -> ops/ko_a16_cfgmean.py [QUEUED 01:42Z]
    RUN 2026-09-01: CRASHED on a malformed f-string key (chr(39)
    construction quoted the dict key) - print-line bug, no science.
    Fixed, re-queued 01:47Z.
    RUN 2026-09-01, S2370: 0/3, null supported - config-frame mean
    changed nothing (means nearly equal by arithmetic). PROVISIONAL
    baseline-subtraction re-analysis flags a possible flip of the
    two-regime law; physical control = 271/272 (different baseline).
    Additivity receipt-check:
    m16: additivity rho 0.970, median obs/(BL+real) 0.759
    m13: additivity rho 0.970, median obs/(BL+real) 0.753
    m14: additivity rho 0.965, median obs/(BL+real) 0.769
    m17: additivity rho 0.920, median obs/(BL+real) 0.846
    a16: additivity rho 0.957, median obs/(BL+real) 0.792
    a14: additivity rho 0.957, median obs/(BL+real) 0.755

271. Frame-rule replication: a14 config-frame knockout: pred_a own in
    [0.7,1.6]; pred_b era rho <= 0.45; pred_c census [0.08,0.6] +
    collateral >= 0.4. Null: overshoot persists.
    -> ops/ko_a14_cfgmean.py [QUEUED 00:58Z]
    SUPERSEDED UNRUN (S2370: frame variable dead; builder had also
    refused on an anchor). Rung numbers repurposed for the baseline
    controls.

272. Frame-rule replication: a17 config-frame knockout (fresh point):
    same bars, census band [0.08,1.0]. -> ops/ko_a17_cfgmean.py [QUEUED 00:58Z]
    SUPERSEDED UNRUN (S2370: frame variable dead; builder had also
    refused on an anchor). Rung numbers repurposed for the baseline
    controls.

271. ADDITIVITY CONTROL: m13 knockout on the CORNER baseline (mixed
    subtracted values: own 0.486, collateral 0.903): pred_a subtracted
    own in [0.24, 0.74]; pred_b subtracted rho >= 0.85; pred_c census
    [0.08, 0.5]. Null: baseline-dependent - no correction publishes.
    -> ops/ko_m13_corner.py [QUEUED 01:05Z]
    RUN 2026-09-01, S2371: INSTRUMENT-INVALID - qk_r/qk_rmap are dead
    knobs in the all_last8 lineage (mixed selection hard-coded); the
    run re-measured the mixed baseline. Bars VOID. 272 pre-declared
    invalid. Rule: live config tripwires mandatory. Valid controls =
    273/274 (corner lineage + L2F tripwire).

272. ADDITIVITY CONTROL: m16 on corner (mixed subtracted 0.755/0.934):
    pred_a in [0.50, 1.00]; pred_b >= 0.85; pred_c [0.2, 5.0].
    -> ops/ko_m16_corner.py [QUEUED 01:05Z]
    RUN 2026-09-01, S2372: pre-declared INVALID (dead knob, S2371);
    bars void, nothing scored.

273. ADDITIVITY CONTROL v2: m13 on the TRUE corner config (corner
    lineage + live L2F tripwire): pred_a subtracted own in [0.24,0.74];
    pred_b subtracted rho >= 0.85; pred_c census [0.08,0.5]. Null:
    baseline-dependent - S2370 correction does not publish.
    -> ops/ko_m13_corner2.py [QUEUED 01:20Z]
    RUN 2026-09-01, S2373: ALL HELD - tripwire passed (L2F 0.0601);
    subtracted own 0.442 (mixed 0.486), rho 0.910 (mixed 0.903).
    Baseline-independent at m13; m16 (274) decides publication.

274. ADDITIVITY CONTROL v2: m16 on corner (mixed subtracted 0.755/
    0.934): pred_a in [0.50,1.00]; pred_b >= 0.85; pred_c [0.2,5.0].
    -> ops/ko_m16_corner2.py [QUEUED 01:20Z]
    RUN 2026-09-01, S2374: ALL HELD - subtracted 0.767/0.937 vs mixed
    0.755/0.934. CORRECTION PUBLISHED: collateral fidelity near-uniform
    0.86-0.94 (two-regime law was accounting); own-effects undershoot
    0.37-0.80; a16 the lone anomaly.

275. cev-dump repro: frontier (corner+tail-r96) with the per-position
    damage vector saved (bars = the S2347 repro claims, unchanged).
    Registered follow-up bar (math review 0107): cosine(d_ct96, d_t120)
    >= 0.95 = rank-one mechanism. -> ops/cevdump_ct96.py [QUEUED 01:10Z]
    RUN 2026-09-01, S2375: ALL HELD, bit-exact; cev_ct96.pt saved.

276. cev-dump repro: corner+tail-r120 (bars = S2349 cliff preds,
    unchanged). -> ops/cevdump_t120.py [QUEUED 01:10Z]
    RUN 2026-09-01, S2376+S2377: ALL HELD; COLLINEARITY BAR HELD -
    cosine 0.9840 (>= 0.95): mechanism rank-one in function space.
    Specificity controls (red-team) = 277/278.

277. RED-TEAM specificity control: cev dump of the value-r96 config
    (non-band damage family; bars = the S2356 preds unchanged).
    Registered control bar: cosine(d_v96, d_ct96) < 0.8 (band
    specificity). -> ops/cevdump_v96.py [QUEUED 01:35Z]
    RUN 2026-09-01, S2378: control bar FAILED as written (0.8660) by
    control mis-design (config contains the band). Residual d_v96 -
    d_ct96 ORTHOGONAL to band (cos -0.017) - provisional vector-
    additivity. PREREG ADDENDUM for 278 (before landing): residual
    cosine in [-0.3, 0.3]. Clean band-free control = 279.

278. RED-TEAM specificity control: cev dump of the m16-knockout config
    (intervention family; bars = S2364 preds unchanged). Same control
    bar vs d_ct96; also feeds the signed-composition test (math review
    move 2). -> ops/cevdump_kom16.py [QUEUED 01:35Z]
    RUN 2026-09-01, S2379: ALL HELD + residual bar dead center (0.041
    in [-0.3,0.3]). Three damage families mutually orthogonal (KO vs
    value residuals: 0.053). Awaiting 279 for full publication.

279. CLEAN band-free control: value r96 on EXACT patterns (pure value
    direction): pred_a census in [0.005,0.06] (live config check);
    pred_b valid >= 10; pred_c offline cosine bars per S2378 (spec
    < 0.3, norm within 25% of 109.7). Null: shares the band direction
    or inert edit. -> ops/cev_purevalue.py [QUEUED 01:45Z]
    RUN 2026-09-01, S2380: FAILED config check (0.0699; cosine 0.858)
    - FLIP-CANDIDATE: the full-rank replacement path may itself cost
    ~0.05 (never physically run before). Decisive controls = 280/281
    (full rank, values real, both lineages).

280. [CODEX, board-claimed 01:22Z before my build] CPU function-space
    basis audit of the damage-family algebra (bars on the board).
    RENUMBER NOTE: my path controls, built without a prior board claim,
    move to 281/282 per the board protocol.

281. PATH CONTROL I (mixed lineage): all patterns full rank, values
    real: pred_a census <= 0.01 (path exact, floor law preserved);
    pred_b valid >= 12; pred_c range + cev saved. NULL (the flip):
    census >= 0.04 - the "mechanism" is path cost.
    -> ops/path_full.py [QUEUED 01:24Z]
    RUN 2026-09-01, S2381: NULL WINS - full-rank path costs +0.0520
    (pred_a/b FAILED as registered). Flip pending 282's independent-
    lineage confirmation; 283 bounds harness leakage.

282. PATH CONTROL II (corner lineage, independent implementation):
    same bars; disagreement with 281 by > 0.01 = lineage bug.
    -> ops/path_full2.py [QUEUED 01:24Z]
    RUN 2026-09-01, S2382: FLIP CONFIRMED - bit-exact agreement with
    281 (+0.0520). Shared-core caveat noted; 283/284 localize. All
    pattern-config census numbers carry ~0.052 instrument error.

283. HARNESS NULL: no replacements installed (real model through the
    harness): pred_a |census| <= 0.003; pred_b valid >= 55; pred_c
    |L2F| <= 0.005. Null: harness leaks (> 0.01). Dead-knob caveat
    noted in-script. -> ops/harness_null.py [QUEUED 01:28Z]
    RUN 2026-09-01, S2383: MISFIRE - knobs left legacy replacements
    installed (+0.4320, increment +0.229 signature). Not a control.
    v2 = 285 (explicit empty actives at the census call).

284. PATH DECOMPOSITION: full-rank config with DIRECT weight-slice
    matmuls in qkz (same recompute structure): pred_a census <= 0.01
    (factor arithmetic culprit); pred_b valid >= 12; pred_c range +
    cev saved. Null: recompute structure carries it (>= 0.04) - fix =
    bf16-matched op order. -> ops/path_direct.py [QUEUED 01:32Z]
    RUN 2026-09-01, S2384: bit-exact +0.0520 again - factor arithmetic
    exonerated; recompute structure convicted (deterministic).
    Localization dump = 286 (in design).

287. [renumbered from 285; Codex's path_drop_a1v holds 285] HARNESS
    NULL v2: explicit empty actives at the census call: pred_a |census|
    <= 0.003; pred_b valid >= 55; pred_c |census| <= 0.02.
    -> ops/harness_null2.py [QUEUED 01:38Z]
    RUN 2026-09-01: ALL HELD - census -0.0000, valid 62/62. Census
    instrument clean.

286. LOCALIZATION DUMP: native-vs-replaced tail outputs in situ (full
    rank): pred_a max block rel err >= 0.01; pred_b max/min <= 5;
    pred_c 8 blocks finite. Null: < 0.005 everywhere (carrier is the
    motif path or merge). -> ops/path_diag.py [QUEUED 01:44Z]
    EXPECTATION REVISED (Codex 285): a1v identified as the sole
    carrier; expected divergence ~0 - now an exactness check; >= 1%
    would reveal an additional masked error.
    RUN 2026-09-01, S2387: null wins as revised-expected - rel err
    0.0000 at all 8 blocks. Tail recompute exact; a1v sole carrier
    confirmed from this side; localization chain closed.

285. [CODEX] SINGLE-CHANGE a1v CONTROL: full-rank QK path with only
    the context-blind block-1 value table omitted. Bars: path anchor
    [0.04,0.07]; corrected census <=0.01, >=55 certificates, residual
    vector norm <=25% of path. Null: >=0.04.
    -> ops/path_drop_a1v.py
    RUN 2026-09-01, S2388: ALL HELD — census -0.0000, 62/62, all eight
    fresh windows 0.0000, residual norm ratio 0.000226. a1v identified.

288. [CODEX] CORRECTED ct96: exact motif-front QK, tail rank 96,
    native block-1 values. Bars: census <=0.010 and >=50 certs; all
    eight fresh <=0.020; vector prediction cosine >=0.95 and relative
    error <=0.25. Null: census >=0.040. -> ops/ct96_native_a1v.py
    RUN 2026-09-01, S2388: PRIMARY BARS HELD (+0.0034195, 56/62,
    fresh8 all 0.0000); VECTOR BAR FAILED as written (0.94857/0.3227).
    Approximate price 154.4M after exact -56,568,960 a1v delta.

289. [CODEX] SIGNED m16 ADOPTION SENTINEL on corrected ct96. Direct
    signed compiled/native effect-vector bars: cosine >=0.90; normalized
    error <=0.60; non-own circuit Spearman >=0.90; own median ratio in
    [0.60,1.20]. -> ops/m16_transfer_ct96_native_a1v.py
    RUN 2026-09-01, S2388: ALL HELD — 0.996324 cosine, 0.103727 error,
    0.998223 collateral rho, 1.037211 own ratio.

290. [CODEX, board-claimed 01:54Z] CORRECTED MIXED-SPECTRUM POINT:
    rank-96 QK at every replaced head, native block-1 values. Bars:
    census <=0.012 and >=50 certs; all fresh8 <=0.020; damage within
    0.008 of +0.0053 transport prediction. Null: census >=0.040.
    Physical factor-rank tripwire required. Approximate price 123.4M;
    exact component bill pending. -> ops/mixed_native_a1v.py
    RUN 2026-09-01: ALL HELD — +0.00853845 census, 52/62, fresh8
    in [-0.0061,+0.0052], rank tripwire live. CONFIG CORRECTION:
    physical factors were contiguous top96, not historical top96+last8.
    Prediction/cert/fresh stand; mixed transport bar and 123.4M price VOID.

291. [CODEX, board-claimed 01:58Z] SIGNED a16 FALSIFIER on corrected
    mixed. Bars: live census <=0.012/max fresh <=0.020; direct effect
    cosine >=0.90 and normalized error <=0.60; non-own circuit rho
    >=0.90 and own median ratio in [0.60,1.40]. Null: cosine <0.70 or
    rho <0.75. -> ops/a16_transfer_mixed_native_a1v.py
    RUN 2026-09-01, S2389: ALL HELD — cosine 0.992605, normalized
    error 0.133050, collateral rho 0.997347, own ratio 1.053832.
    Old a16 anomaly was path-contaminated.

292. [CODEX, board-claimed 02:03Z] COMPUTE COMPOSITION: corrected
    physical top96 plus per-token CP top-1152 (label corrected before
    landing). Bars: surcharge in [0.010,0.024]
    and census <=0.035; >=40 certs; max fresh8 <=0.040. Live selected-
    width tripwire =1152. Null: surcharge >=0.040 or <=20 certs.
    Exact storage pending; executed CP-unit compute 4x sparse.
    -> ops/mixed_topk_native_a1v.py
    RUN 2026-09-01: 2/3 — surcharge +0.015464 and fresh max +0.0276
    HELD; certificates FAILED materially at 26/62 vs >=40. Null <=20
    not triggered. Prediction-cheap, not certificate-cheap.

293. [CODEX, board-claimed 02:06Z] TRUE CORRECTED MIXED COMPANION:
    physical QK indices {0..95,120..127}, native a1v. Bars: census
    <=0.0065 and >=54 certs; all fresh8 <=0.020; exact index-set live
    tripwire. Null/no-benefit: census >=0.0080 and certs <=52. No
    scalar transport bar. Exact price pending. -> ops/mixed104_native_a1v.py
    RUN 2026-09-01: ALL HELD — exact indices/factor width live;
    +0.00469196 census, 54/62, fresh8 [-0.0066,+0.0032]. Last8 buy
    -0.0038465 CE and +2 certs over physical top96.

294. [CODEX, board-claimed 02:11Z] SIGNED a16 ADOPTION GATE on true
    mixed104. Same direct-effect bars as 291 plus exact 104-index-set
    tripwire. -> ops/a16_transfer_mixed104_native_a1v.py
    RUN 2026-09-01: ALL HELD — cosine 0.995879, error 0.096599,
    collateral rho 0.997959, own ratio 1.034785.

295. [CODEX, board-claimed 02:14Z, CPU ONLY] EXACT PHYSICAL BILL:
    distinguish deployable semantic program, hook-harness footprint, and
    historical incremental ledger; enumerate named tensor dependencies once
    with source/shape/multiplicity assertions; storage and compute separate.
    First closed line: mixed104 QK factors =110*4*(128+1152)*104
    =58,572,800 scalars. Historical 52.9M headline cannot be reused.
    RUN 2026-09-01, S2392: historical totals REJECTED. Native checkpoint
    545,902,902 scalars; tested table artifact 596,164,022 (larger than
    native); storage-minimal online-c_v0 candidate 539,595,062, requiring
    its own live gate. -> ops/mixed104_exact_bill.py

296. [CODEX, board-claimed 02:24Z] LITERAL MIXED104 ONLINE-c_v0:
    remove the fp16 a0 table and execute native block-0 c_v from the already
    required normalized embedding. Bars: census <=0.0065 and >=54 certs;
    mean absolute per-position CE difference vs saved table artifact <=0.002
    and |mean difference| <=0.0015; all fresh8 <=0.020; exact mixed104
    index/factor and final-active-set tripwires. Null: census >=0.010 or
    <=50 certs. Exact bill 539,595,062 scalars / 2,042,438,252 raw bytes.
    -> ops/mixed104_online_cv0.py [QUEUED 02:24Z]
    RUN 2026-09-01, S2393: ALL HELD — census +0.00469195, 54/62,
    fresh8 [-0.0066,+0.0032], table-vector MAD 0.000051963 and signed
    mean -1e-8; exact active/index/width tripwires passed. Literal candidate
    physically identified; shifted OOD still required.

297. [CODEX, board-claimed 02:34Z] SHIFTED-CORPUS OOD: frozen
    Salesforce/WikiText-2 raw TEST, GPT-2 tokenization, 120 deterministic
    nonoverlapping 257-token chunks after skip1024. No OOD fit exposure.
    Bars: mean compiled-minus-native CE <=0.012; row-mean p95 <=0.020;
    native CE [2,8], exact mixed104/active-set and 120-row tripwires.
    Null: mean >=0.030 or p95 >=0.060.
    -> ops/mixed104_online_cv0_ood.py [QUEUED 02:34Z]
    RUN 2026-09-01, S2394: ALL HELD — native CE 3.95016050,
    compiled 3.95501661, mean damage +0.00485625, row p95 +0.01721914,
    worst row +0.02691667; config and population tripwires held. Literal
    mixed104 online-c_v0 formally adopted at 539,595,062 scalars.

298. [CODEX, board-claimed 02:40Z] EMBEDDING-FOLDED MLP0 STRUCTURE:
    planted block/tree/DAG support recovery followed by all 50,257 exact
    position-zero token inputs and legal S-state fixed-K512 routers. Native
    MLP0 price 15,926,400; router price counts union experts plus router.
    -> ops/mlp0_embedding_fold_structure_screen.py
    RUN 2026-09-01, S2395: raw factor identification FAILED despite student
    R2 0.999940 (Jaccard .281/F1 0); real fixed-subset routers FAILED
    (best legal R2 -.052, oracle also negative) although state labels beat
    chance; unconstrained per-token top-k R2 .754. Negative control clean.

298B. [CODEX] STRUCTURAL-PRIOR DISCRIMINATOR: correct support-size spectrum
    versus pair-only and singleton hard priors on the planted teacher.
    -> ops/mlp0_embedding_fold_prior_sweep.py
    RUN 2026-09-01, S2395: correct spectrum partially recovered structure
    (R2/Jaccard/F1 .999927/.820/.775) but missed its frozen bar; wrong
    pair-only fit better (R2 .999999) with wrong graph (F1 .245). R2 cannot
    identify the prior; require compression/intervention/OOD discrimination.

299. [CODEX, board-claimed 03:02Z] DIRECT SHARED BILINEAR TENSORS:
    exact gauge-invariant native-atom matches across 35 layer pairs plus a
    randomized polarized full-tensor layer spectrum; signed-coordinate null,
    planted shared-bank positive and independent-bank negative. Native all-MLP
    price 286,675,200; 25% common-bank saving requires 25.390625% pooled reuse.
    -> ops/shared_bilinear_atom_reuse_screen.py
    RUN 2026-09-01, S2396: ALL POSITIVE PREDS FAILED, null won. Zero
    atom matches even at cosine .80; best median .000516 vs null .000393;
    top-13 layer energy .7724/.7683 vs null .7716/.7728. Kill native-atom
    reuse and whole-layer mixing; newly fitted changed-metric joint CP remains open.

300. [CODEX, board-claimed 03:06Z] JOINT UNTIED VOCABULARY CODE: keep E
    exact; Uhat=E M plus rank-s residual; compare with price-matched independent
    U SVD on 16 FineWeb +16 frozen WikiText rows and frequency bins.
    -> ops/joint_vocab_shared_code_screen.py
    RUN 2026-09-01, S2397: shared geometry real (E->U weight R2 .3324;
    s512 beats independent r537), but uniform predictive arm fails at +.7430
    FW/+.6474 WT; registered null triggered.

300B. [CODEX, board-claimed 03:12Z] FREQUENCY-WEIGHTED VOCAB FOLLOW-UP:
    same s512/r537 and prices; refit map+bases using 480 disjoint FineWeb rows.
    -> ops/joint_vocab_frequency_weighted_followup.py
    RUN 2026-09-01, S2397: ALL THREE HELD, null false. Shared count-weighted
    +.1930 FW/+.2249 WT vs independent +.5518/+.7781 at 73.88% vocab
    storage. Advance; sparse rare residual rows are the exploit-phase test.

301. [CODEX, board-claimed 03:15Z] MLP0 SIGNED RESPONSE RANK: exact
    suffix-loss gradient paired with native MLP0 output; rank abs eigenmodes of
    sym(E[(y-mu)^T g]); compare literal factorized-Down replacements to PCA and
    Down SVD on disjoint FW/Wiki; split stability.
    -> ops/mlp0_signed_response_rank_screen.py
    RUN 2026-09-01, S2398: pred_b only; null triggered. Response r128
    +.0875 FW/+.0616 WT at 71.3% MLP0 price, but PCA +.0500/+.0302 and
    weight SVD +.0621/+.0495; response split overlap .283 vs PCA .683.
    Kill response rank allocator; retain ordinary PCA baseline.

302. [CODEX, board-claimed 03:20Z] DELIMITER PREDICTIVE-STATE HANKEL:
    natural FW/Wiki prefixes; quote/open-paren binary states; nested action
    suffix log-probability blocks; cross-corpus transfer; head13.8 deletion vs
    head13.1 control. No generic token splice.
    -> ops/delimiter_predictive_state_hankel.py
    RUN 2026-09-01, S2399: all positive preds failed; strong null poles absent.
    Quote transfer .875 but R2 .191/.239 and r90 5/4; paren transfer .625,
    R2 .147/.140, r90 5/5; head13.8 effect only 6-7%. Suffix NLL 10-12
    is a scope confound. Park quote classifier; kill as compression route.

303. [CODEX, board-claimed 03:26Z] MLP0 PROJECTION ERROR CONTRACT:
    fit log finite-CE damage from local omitted MLP0 output energy on eight
    skip7000 rows; freeze residual envelope and validate across PCA/Down-SVD/
    response bases at ranks 64/128/256 on eight skip11000 rows. Compare the
    exponent with the historical site-1 isotropic error curve.
    -> ops/mlp0_projection_error_contract.py
    RUN 2026-09-01, S2400: 1/3 positive bars, null false. All 9/9 validation
    arms covered; exponent 1.746 vs historical 1.534; interval width 3.414x;
    calibration/validation Spearman .867/.917. The lower bound rejects two of
    three rank64 arms but not PCA r64 (+.033 bound, +.096 observed). Retain as
    a screening heuristic, not a rank certificate or theorem.

302B. [CODEX, board-claimed 03:31Z] DELIMITER STATE CONTROL REPAIR:
    replace the inert roll-by-three complement with 64 seeded balanced label
    permutations; test both centroid R2 and disjoint-half classifier accuracy.
    -> ops/delimiter_predictive_state_control_repair.py
    RUN 2026-09-01, S2401: live-control and classifier bars held; R2 bar failed;
    null false. All real classifiers beat mean shuffle by >=.10, but parenthesis
    R2 fell below shuffle-p95. Withdraw R2-level state evidence; circuit-level
    classifier survives. Compiler-route kill unchanged.

304. [CODEX, board-claimed 03:42Z] FISHER-SELECTED SPARSE RARE RESIDUAL:
    count-weighted shared rank512 plus maximal K=1129 indexed exact rows under
    25% vocabulary saving; compare Fisher, residual-norm, and random rare rows.
    -> ops/joint_vocab_sparse_rare_residual.py
    RUN 2026-09-01, S2402: ALL PREDS FAILED, null won. Fisher repaired aggregate
    only 1.8%/4.4% and unseen 2.1%/.5%; norm was 3.4%/4.2%. Selectors overlapped
    only 3-5%, so rare error is distributed rather than a small exception table.
    Kill sparse-row repair; test one distributed-rank frontier.

305. [CODEX, board-claimed 03:39Z] DISTRIBUTED SHARED-VOCAB RANK FRONTIER:
    sqrt/count weights, shared s={512,640,768} versus price-matched independent
    r=s+25 on new FineWeb/WikiText windows. Eligible saving >=14.5% vocab.
    -> ops/joint_vocab_distributed_rank_frontier.py
    RUN 2026-09-01, S2403: ALL PREDS FAILED, strong null false. Best rank640
    about +.180/.197; rank768 +.127/.145 at 14.76% saving. Shared beats matched
    independent 2-3x, but sqrt-r768 unseen still +.724/.566. Stop adoption route;
    preserve shared-code representation result and switch exploit to MLP PCA.

306. [CODEX, board-claimed 03:43Z] FOUR-LAYER ACTIVATION-PCA COMPOSITION:
    fit rank256 MLP-output PCA at all 18 layers; choose four on calibration only;
    validate jointly against fixed spaced control. Four-layer saving 15,335,424.
    -> ops/mlp_activation_pca_four_layer_composition.py
    RUN 2026-09-01, S2404: pred_a only, null false. 17/18 layers individually
    safe, but layer-rank rho .298; selected quartet +.130/.122 at 1.78x/1.61x
    additive, worse than fixed {0,5,11,17} +.098/.084. Advance pair-interaction
    allocator once; scalar sensitivity selection is killed.

307. [CODEX, board-claimed 03:49Z] PAIR-INTERACTION MLP PCA ALLOCATOR:
    score all 153 pairs on two calibration halves; penalize unstable negative
    excess interactions; enumerate 3,060 quartets and validate once.
    -> ops/mlp_pca_pair_interaction_allocator.py
    RUN 2026-09-01, S2405: all joint preds failed, null false. Pair excess rho
    .917 (real law), total pair rho .476; selected {0,4,14,15} +.0876/.0586,
    ratios 1.324/1.415. Large win over naive, only 10.9% FW win over fixed.
    Reduce to a three-layer interaction-selected final screen.

308. [CODEX, board-claimed 03:53Z] INTERACTION-SELECTED PCA TRIPLE:
    enumerate 816 triples from frozen rung307 model; evaluate on untouched
    FineWeb rows and WikiText skip40000 against scalar/fixed controls.
    -> ops/mlp_pca_interaction_selected_triple.py
    RUN 2026-09-01, S2406: pred_b only, null false. Selected {4,14,15}
    +.0668/.0672, ratios 1.192/1.208; misses FW .06 and loses to fixed
    {0,8,17} +.0725/.0334 on WikiText. Stop targeting; broad-confirm fixed.

309. [CODEX, board-claimed 03:56Z] FIXED-SPACED PCA TRIPLE STABILITY:
    {0,8,17}@r256 on 176+176 untouched FineWeb rows and WikiText n120;
    mean, row-p95, max, and population-spread bars. Saving 11,501,568.
    -> ops/mlp_pca_fixed_triple_stability.py
    RUN 2026-09-01, S2407: ALL HELD, null false. Means .0589/.0600/.0477;
    p95 .1027/.1002/.0997; maxima .1539/.1421/.1208; spread .0123.
    Advance exact composition with adopted mixed104 plus certs/bill/tripwires.

310. [CODEX, board-claimed 03:59Z] MIXED104 + FIXED PCA TRIPLE COMPOSITION:
    final-only physical PCA hooks {0,8,17}:256 inside adopted mixed104 online-c_v0;
    census, 62 certs, fresh8, exact identities, proposed 528,093,494 bill.
    -> ops/mixed104_online_cv0_pca_fixed_triple.py
    RUN 2026-09-01, S2408: pred_b/c held, pred_a failed, null false. Census
    +.06745057, MLP surcharge +.06275862, fresh max +.0687, all identities live;
    but only 8/62 certificates. Not adopted. Test exactly three two-layer subsets.

311. [CODEX, board-claimed 04:05Z] MIXED104 + PCA PAIR FRONTIER:
    evaluate `{0,8}`, `{0,17}`, and `{8,17}` at rank256 inside one common
    mixed104 rebuild; identical 7,667,712-scalar saving and all 62 certificates.
    -> ops/mixed104_pca_fixed_pair_frontier.py
    RUN 2026-09-01, S2409: pred_b only, null false. `{0,8}` +.04021/16 certs,
    `{0,17}` +.04970/17, `{8,17}` +.04726/19. No pair reaches the frozen
    20-certificate bar or improves the triple-normalized price/damage tradeoff.
    Stop subset search; test a fixed-pair rank frontier.

312. [CODEX, board-claimed 04:12Z] FIXED `{8,17}` PCA RANK FRONTIER:
    compare ranks 256/384/512 in one rebuild with exact prices and certificates.
    -> ops/mixed104_pca_fixed_pair_rank_frontier.py
    RUN 2026-09-01, S2410: pred_b only, null false. r256 +.04726/19 certs,
    r384 +.03379/24, r512 +.02490/32; savings 7.67M/6.19M/4.72M.
    Capacity is monotone but misses both certificate/scalar bars. Stop rank sweep;
    test one split-tag certificate-gradient hybrid at equal r256 price.

313. [CODEX, board-claimed 04:19Z] CERTIFICATE-GRADIENT PCA HYBRID:
    fixed `{8,17}@r256`; compare plain PCA with 224+32 and 192+64
    PCA/gradient bases at equal 7,667,712-scalar saving. Fit on split tags and
    exclude all 16 fit rows from full/heldout certificate scoring.
    -> ops/mixed104_pca_certificate_gradient_hybrid.py
    RUN 2026-09-01, S2411: ALL PREDS FAILED, null won. Plain/grad32/grad64
    +.04727/+.04635/+.04788; all exactly 19/62 full and 10/31 heldout certs.
    Bases genuinely rotated (overlap .879/.766) and captured more gradient energy,
    but no threshold moved. Close MLP-PCA adoption; keep two-tier result.

314. [CODEX, board-claimed 04:25Z] MLP0 EXACT-TOKEN SHARED INPUT ENCODER:
    factor `[Left;Right]` through one p512/768 encoder; compare exact-token RRR,
    weight SVD, and input PCA at matched literal prices on heldout token ids and
    contextual FineWeb/WikiText.
    -> ops/mlp0_exact_token_shared_input_encoder.py
    RUN 2026-09-01, S2412: ALL PREDS FAILED, null false. RRR improves token R2
    (.846/.937 vs weight .686/.888) but loses contextual CE. Weight SVD wins:
    p512 +.01662/.01012 at 5,308,416 saved; p768 +.00355/.00278 at 2,654,208.
    Promote the matched control to mixed104 census/certificate composition.

315. [CODEX, board-claimed 04:31Z] MIXED104 + MLP0 SHARED-INPUT SVD:
    physically compose weight-SVD p512/p768 in one common census with exact
    bills, all 62 certificates, fresh windows, and identity tripwires.
    -> ops/mixed104_mlp0_shared_input_svd_frontier.py
    RUN 2026-09-01, S2413: pred_b/c held, pred_a failed, null false. p512
    +.018900/31 certs at 534,286,646; p768 +.009012/50 at 536,940,854.
    Primary p512 fresh max +.0071; identities exact. Stop p512 at its missed
    35-cert bar; advance p768 to shifted OOD and signed intervention gates.

316. [CODEX, board-claimed 04:36Z] MLP0 SVD768 SHIFTED OOD + BASELINE CEV:
    rebuild exact p768 mixed104; evaluate WikiText test skip70000 n120, reproduce
    census/certs, save unablated 256k-position CE vector for signed effects.
    -> ops/mixed104_mlp0_svd768_ood.py
    RUN 2026-09-01, S2414: ALL HELD, null false. Wiki mean/p95/max
    +.010620/+.032453/+.039183; census +.00901182, 50 certs; fresh8 all <=-.0004.
    Identities/bill exact. Advance to direct signed a16 transfer.

317. [CODEX, board-claimed 04:42Z] MLP0 SVD768 SIGNED a16 ADOPTION GATE:
    apply the identical native a16 mean ablation within the p768 program and native;
    compare direct signed effects, collateral ordering, own magnitude, live baseline,
    exact identities, and the literal bill.
    -> ops/a16_transfer_mixed104_mlp0_svd768.py
    RUN 2026-09-01, S2415: ALL HELD, null false. Live baseline +.00901182/50;
    signed cosine .994191, normalized error .113700, norm ratio 1.030480,
    collateral rho .998061, own median ratio 1.034104. Formally adopt
    536,940,854 scalars / 2,031,821,420 bytes as the smaller Pareto point.

318. [CODEX, board-claimed 04:49Z] ALL-LAYER SHARED-INPUT WEIGHT-SVD SCREEN:
    install p512/p768 independently at all 18 MLPs on untouched FineWeb rows
    176:188 and WikiText skip80000; no adaptive layer selection.
    -> ops/mlp_shared_input_svd_all_layers_screen.py
    RUN 2026-09-01: pred_a/c held, pred_b failed, null won. p768 qualifies at
    13/18 layers, but Wiki median is +.01290 and late layers 15/16/17 reach
    +.129/+1.210/+.235. Cancel fixed {0,8,17} composition. Run full-rank late
    controls before interpreting the depth boundary.

318B. [CODEX, board-claimed 04:54Z] LATE-DEPTH SVD CONTROL:
    test p1152 reconstruction inertness and p1024 repair at layers 15--17 on
    the identical rows; diagnosis only, with no layer selection.
    -> ops/mlp_shared_input_svd_late_depth_control.py
    RUN 2026-09-01: pred_a held, pred_b/c failed, null won. p1152 is inert to
    ~4e-8 CE, but p1024 remains +.126/+1.126/+.217 worst-corpus damage at
    layers15/16/17; zero layers repair by 25% or qualify. Late ordinary-SVD
    compression is closed.

319. [CODEX, board-claimed 04:58Z] FRONT/MIDDLE INPUT-SVD TRIPLE:
    compose equally spaced {0,7,14}@p768 physically with mixed104; measure
    census, 62 certificates, fresh8, identities, and exact 531,632,438 bill.
    -> ops/mixed104_shared_input_svd_front_mid_triple.py
    RUN 2026-09-01: pred_b/c held, pred_a failed, null false. Census +.026044,
    25/62 certs, fresh max +.0277, identities/bill exact. Valid compression
    tier, not adoption. Allow one final common-rebuild pair decrement.

320. [CODEX, board-claimed 05:01Z] FINAL FRONT/MIDDLE INPUT-SVD PAIRS:
    compare {0,7} and {0,14}@p768 at the common 534,286,646-scalar price;
    frozen selection by certificates then census; no further subset/rank search.
    -> ops/mixed104_shared_input_svd_front_mid_pairs.py
    RUN 2026-09-01: pred_b/c held, pred_a failed by one certificate, null false.
    {0,7} +.019408/34; {0,14} +.015880/37. Close subset/rank search; retain
    {0,14} only as the strongest labeled tier at this price.

321. [CODEX, board-claimed 05:04Z] LATE CONTEXT-METRIC SHARED INPUT:
    fit p768 paired-map RRR under two independent contextual input covariances;
    compare matched weight SVD on untouched FineWeb/WikiText.
    -> ops/mlp_late_context_metric_shared_input_screen.py
    RUN 2026-09-01: ALL HELD, null false. RRR damage is at most +.0115 across
    all late layers/corpora versus weight SVD up to +1.0874. Split overlaps
    .763-.773 and both halves reproduce. Promote fixed late triple physically.

322. [CODEX, board-claimed 05:07Z] PHYSICAL LATE CONTEXT-RRR TRIPLE:
    compose {15,16,17}@p768 fit on frozen contextual covariance with mixed104;
    census, 62 certificates, fresh8, fit/map/QK identities, exact 531,632,438 bill.
    -> ops/mixed104_late_context_metric_input_triple.py
    RUN 2026-09-01: all positive preds failed, null false. +.039485, 28 certs,
    fresh max +.0536; identities/bill exact. Diagnose open-loop covariance:
    downstream maps were fit before upstream replacements shifted their inputs.

323. [CODEX, board-claimed 05:11Z] SEQUENTIAL CONTEXT-RRR LATE TRIPLE:
    fit L15 on native contexts, L16 under fitted L15, L17 under fitted L15+16;
    physically rerun unchanged {15,16,17}@p768 and 531,632,438 bill.
    -> ops/mixed104_late_context_metric_input_triple_sequential.py
    RUN 2026-09-01: all positives failed, null won. +.039614, 28 certs,
    fresh max +.0548, damage ratio 1.003 vs open-loop. Serial context shift is
    ruled out; close late composition without subset search.

324. [CODEX, board-claimed 05:17Z] MLP0 CONTEXT-METRIC p512/p640:
    two independent contextual input-covariance fits versus matched weight SVD
    on untouched FineWeb/WikiText; single-site screen avoids composition tax.
    -> ops/mlp0_context_metric_shared_input_frontier.py
    RUN 2026-09-01: ALL HELD, null false. p512 +.00397/+.00121 versus weight
    +.01363/+.02244; p640 +.00274/+.00098 versus +.00759/+.01414. Independent
    halves reproduce. Promote both in one physical mixed104 frontier.

325. [CODEX, board-claimed 05:21Z] PHYSICAL MLP0 CONTEXT-RRR FRONTIER:
    compose p512/p640 context-RRR variants with mixed104 in one rebuild;
    census, 62 certificates, fresh8, fit/map/QK identities, exact bills.
    -> ops/mixed104_mlp0_context_metric_input_frontier.py
    RUN 2026-09-01: ALL HELD, null false. p512 +.0107277/48 at 534,286,646;
    p640 +.0082647/52 at 535,613,750; primary fresh max +.0109 and identities
    exact. Advance both to common shifted OOD and save exact CEVs.

326. [CODEX, board-claimed 05:28Z] MLP0 CONTEXT-RRR SHIFTED OOD:
    rebuild p512/p640 variants, evaluate WikiText skip100000 n120, reproduce
    census/certs, exact identities/bills, save both unablated CEVs.
    -> ops/mixed104_mlp0_context_metric_input_frontier_ood.py
    RUN 2026-09-01: ALL HELD after one preserved no-receipt import repair.
    p512 Wiki mean/p95/max +.010411/.031169/.037714; p640
    +.008580/.026783/.035271. Census/certs exact, both CEVs saved.

327. [CODEX, board-claimed 05:35Z] MLP0 CONTEXT-RRR SIGNED a16 GATE:
    apply identical a16 mean ablation to native and both p512/p640 variants;
    direct signed vectors, collateral ordering, own magnitudes, live identities.
    -> ops/a16_transfer_mixed104_mlp0_context_metric_frontier.py
    RUN 2026-09-01: ALL HELD, null false. p512 cosine/error/rho/own
    .993405/.120526/.996939/1.035403; p640 .994483/.110815/.997653/1.035874.
    Formally adopt p640 535,613,750 (+.008265/52) and p512 534,286,646
    (+.010728/48); p640 strictly dominates prior weight-p768.

328. [CODEX, board-claimed 05:48Z] MLP0 CONTEXT-RRR LOWER-RANK FRONTIER:
    one physical mixed104 rebuild with fixed p448/p384/p256; census, all
    certificates, primary fresh8, fit/maps/QK identities, exact bills.
    -> ops/mixed104_mlp0_context_metric_lower_rank_frontier.py
    RUN 2026-09-01: ALL HELD, null false. p448 +.012662/43 at 533,623,094;
    p384 +.015788/39 at 532,959,542; p256 +.026878/25 at 531,632,438.
    Primary fresh max +.0265, identities exact. Advance common shifted OOD.

329. [CODEX, board-claimed 05:55Z] LOWER-RANK CONTEXT-RRR SHIFTED OOD:
    rebuild p448/p384/p256, WikiText skip120000 n120, reproduce census/certs,
    exact identities/bills and save all three unablated CEVs.
    -> ops/mixed104_mlp0_context_metric_lower_rank_ood.py
    RUN 2026-09-01: pred_b/c held, pred_a failed, null false. p448 OOD
    mean/p95/max +.011411/.036166/.081581 passes; p384 max .108456>.100 and
    p256 max .154458>.120 fail. Advance signed p448 only.

330. [CODEX, board-claimed 06:03Z] P448 CONTEXT-RRR SIGNED a16 GATE:
    direct signed a16 mean-ablation effect versus saved p448 baseline; live
    identity/price, collateral ordering, own magnitudes.
    -> ops/a16_transfer_mixed104_mlp0_context_metric_p448.py
    RUN 2026-09-01: ALL HELD, null false. cosine/error/rho/own
    .992558/.127682/.996020/1.035203. Formally adopt p448 at 533,623,094,
    +.012662/43, Wiki +.011411; smallest fully gated point.

331. [CODEX, board-claimed 06:15Z] CONTEXT-METRIC QK96:
    rank96 RRR under contextual attention-input covariance for all 440 Q/K
    maps; compare physical weight-top96 and mixed104, exact price/identity.
    -> ops/mixed96_context_metric_qk.py
    RUN 2026-09-01: ALL HELD, null false. +.00124485, 62/62, fresh max+.0015,
    535,089,462 scalars. Beats weight-top96 by .007294 and mixed104 by .003447
    while smaller. Advance independent split-fit + shifted OOD.

332. [CODEX, board-claimed 06:02Z] CONTEXT-QK96 SPLIT REPRO + OOD:
    freeze independent covariance fit rows72:96 as shipping map; census/certs,
    WikiText skip140000 n120, exact 440-map identity/bill, save unablated CEV.
    -> ops/mixed96_context_metric_qk_split_ood.py
    RUN 2026-09-01: ALL HELD, null false. split-B +.00141535/61, difference
    +.0001705 from split-A, fresh max+.0014; Wiki mean/p95/max
    -.001674/.007033/.015147. Advance fixed split-B to signed gate.

333. [CODEX, board-claimed 06:06Z] CONTEXT-QK96 SIGNED a16 GATE:
    identical a16 mean ablation, direct signed vector/collateral/own effects,
    live split-B baseline, context96 identity and 535,089,462 bill.
    -> ops/a16_transfer_mixed96_context_metric_qk.py
    RUN 2026-09-01: ALL HELD, null false. cosine/error/norm ratio
    .997617/.074857/1.026198; collateral rho .998265, own ratio 1.029761.
    Formally adopt 535,089,462 / +.00141535 / 61 as high-fidelity point.

334. [CODEX, board-claimed 06:12Z] DUAL-CONTEXT PHYSICAL COMPOSITION:
    combine split-B context-QK96 with context-MLP0 p448 at exact
    529,117,494 scalars; census/certificates, additive residual, fresh rows,
    and untouched WikiText skip160000 tails. Save exact CEV for signed gate.
    -> ops/mixed96_context_qk_mlp0_context_p448_ood.py
    RUN 2026-09-01: ALL HELD, null false. +.00958546/49, residual
    +.00020035 from additive and ratio 1.02135x; Wiki mean/p95/max
    +.001609/.028941/.049312, fresh max+.0123. Advance signed gate.

335. [CODEX, board-claimed 06:18Z] DUAL-CONTEXT p448 SIGNED a16 GATE:
    identical a16 mean ablation against saved rung334 CEV, direct signed
    vector/collateral/own effects, live dual identities and 529,117,494 bill.
    -> ops/a16_transfer_mixed96_context_qk_mlp0_p448.py
    RUN 2026-09-01: ALL HELD, null false. cosine/error/norm
    .994186/.113081/1.028356; rho .997653, own ratio 1.028231. Formally
    adopt 529,117,494 / +.00958546 / 49; gated frontier collapses to two.

336. [CODEX, board-claimed 06:24Z] CONTEXT-QK88 PHYSICAL + OOD:
    fixed split-B covariance, rank88 at all 440 Q/K maps, census/certs/fresh,
    untouched WikiText skip180000 tails, exact 530,583,862 bill and saved CEV.
    -> ops/mixed88_context_metric_qk_ood.py
    RUN 2026-09-01: ALL HELD, null false. +.00219606/58, surcharge
    +.00078071 over r96; fresh max+.0018; Wiki mean/p95/max
    -.003865/.006220/.007768. Advance direct signed gate.

337. [CODEX, board-claimed 06:23Z] CROSS-FAMILY ADDITIVITY p512/p640:
    compose context-QK96 with context-MLP0 p512/p640 in one physical variant
    run; frozen near-additive ratios, census/certs, Wiki skip200000 tails,
    exact 529,781,046 / 531,108,150 bills, identities, fresh and saved CEVs.
    -> ops/mixed96_context_qk_mlp0_context_p512_p640_ood.py
    RUN 2026-09-01: ALL HELD, null false. p512 +.0076304/51,
    ratio1.0241, Wiki +.004275/.025570/.028409; p640 +.0050872/54,
    ratio1.0199, Wiki +.001977/.018504/.022504. Near-additivity repeats.

338. [CODEX, board-claimed 06:27Z] CONTEXT-QK88 SIGNED a16 GATE:
    identical native/compiled a16 mean ablation against saved r336 CEV,
    direct signed/collateral/own effects, live rank88 identity and exact bill.
    -> ops/a16_transfer_mixed88_context_metric_qk.py

339. [CODEX, board-claimed 06:30Z] CONTEXT-QK80 PHYSICAL + OOD:
    fixed split-B context rank80 at all 440 maps, census/certs/fresh, untouched
    Wiki skip220000 tails, exact 526,078,262 bill and saved CEV.
    -> ops/mixed80_context_metric_qk_ood.py
    RUN 2026-09-01: ALL HELD, null false. +.00333585/54 (cert bar exact),
    surcharge +.00113979 over r88; fresh max+.0027; Wiki mean/p95/max
    -.000878/.013861/.026134. Advance signed gate.

340. [CODEX, board-claimed 06:36Z] MLP0 QUADRATIC CONTRACTION COMMUTANT:
    gauge-invariant planted-to-real screen on exact MLP0 quadratic function
    tensor, exhaustive embedding PCA32, 12 output contractions, common-gauge
    check and independent-conjugation spectral null; block/split/offblock bars.
    -> ops/mlp0_embedding_fold_contraction_commutant.py

341. [CODEX, board-claimed 06:39Z] CONTEXT-QK80 SIGNED a16 GATE:
    identical native/compiled a16 mean ablation against saved r339 CEV,
    direct signed/collateral/own effects, live rank80 identity and exact bill.
    -> ops/a16_transfer_mixed80_context_metric_qk.py

342. [CODEX, board-claimed 06:43Z] CONTEXT-QK72 PHYSICAL + OOD:
    fixed split-B context rank72 at all 440 maps, census/certs/fresh, untouched
    Wiki skip240000 tails, exact 521,572,662 bill and saved CEV.
    -> ops/mixed72_context_metric_qk_ood.py
    RUN 2026-09-01: ALL HELD, null false. +.00523783/54, surcharge
    +.00190197 over r80; fresh max+.0043; Wiki mean/p95/max
    +.000498/.020242/.035112. Advance signed gate.

343. [CODEX, board-claimed 06:41Z] QK80 + CONTEXT-VALUE96 PHYSICAL OOD:
    factor c_v head maps at layers2--17 under split-B covariance, 144 maps;
    census/certs/surcharge, terminal Wiki skip270840 n56 tails, exact QK80/value96 identity,
    522,539,318 bill and saved CEV.
    -> ops/mixed80_context_qk_value96_context_ood.py
    PREFLIGHT CORRECTION: skip260000 n120 exceeded 286,177-token stream before
    model load. Freeze untouched terminal skip270840 n56; numerical bars unchanged.

344. [CODEX, board-claimed 06:41Z] CONTEXT-QK72 SIGNED a16 GATE:
    identical native/compiled a16 mean ablation against saved r342 CEV,
    direct signed/collateral/own effects, live rank72 identity and exact bill.
    -> ops/a16_transfer_mixed72_context_metric_qk.py

345. [CODEX, board-claimed 06:45Z] CONTEXT-QK64 PHYSICAL + OOD:
    fixed split-B context rank64 at all 440 maps, census/certs/fresh, untouched
    terminal Wiki skip270840 n56 tails, exact 517,067,062 bill and saved CEV.
    -> ops/mixed64_context_metric_qk_ood.py
    PREFLIGHT CORRECTION: skip280000 n120 exceeded stream before model load.
    Freeze untouched terminal skip270840 n56; numerical bars unchanged.

346. [CODEX, board-claimed 06:55Z] FULL-SPACE EXACT-FOLD MLP0 CONTRACTION GRAPH:
    capture all50,304 deterministic position-zero MLP0 inputs, metric-weight
    seven exact quadratic contractions in full1152D, and reduce their common
    commutant to a reference-eigenbasis graph Laplacian; planted/common-gauge,
    independently permuted spectral null, split-projector and offblock bars.
    -> ops/mlp0_full_fold_contraction_graph.py

347. [CODEX, board-claimed 06:52Z] CONTEXT-QK64 SIGNED a16 GATE:
    identical native/compiled a16 mean ablation against saved r345 CEV;
    live baseline/identity/bill, direct signed/collateral/own effects.
    -> ops/a16_transfer_mixed64_context_metric_qk.py
    RUN 2026-09-01: ALL HELD, null false. cosine/error/norm
    .988466/.178663/1.083107; rho .994592, own ratio 1.081668. Formally
    adopt 517,067,062 / +.00819306 / 50.

348. [CODEX, board-claimed 06:57Z] CONTEXT-QK56 NEW-CORPUS PHYSICAL + OOD:
    fixed split-B rank56/440 maps; census/certs/fresh; first 120 chunks from
    hashed WikiText-103 train rows100000:110000; exact 512,561,462 bill/CEV.
    -> ops/mixed56_context_metric_qk_newcorpus_ood.py
    RUN 2026-09-01: ALL HELD, null false. +.01250780/43, surcharge
    +.00431473; new-corpus mean/p95/max +.005698/.037966/.056886.

349. [CODEX, board-claimed 07:00Z] TIGHTENED CONTEXT-QK56 SIGNED a16 GATE:
    live baseline/identity and direct a16 effects with prospective cosine .98,
    error .30, effect/own-ratio 1.15, and collateral-rho .98 bars.
    -> ops/a16_transfer_mixed56_context_metric_qk.py
    RUN 2026-09-01: ALL TIGHT BARS HELD, null false. cosine/error/norm
    .981641/.236669/1.121655; rho .993980, own ratio 1.117505. Formally
    adopt 512,561,462 / +.01250780 / 43; license final rank48 probe only.

350. [CODEX, board-claimed 07:02Z] MLP0 TAIL-ROBUST CONTEXT METRIC:
    mix ordinary covariance with top-10%-leverage covariance at .25; compare
    p384/p448 ordinary/robust across two fits and two row-tail populations.
    -> ops/mlp0_tail_robust_context_metric_screen.py
    RUN 2026-09-01: pred_b only, null false. Means improve, split overlaps
    .735/.743, but maxima do not; do not advance or tune leverage weighting.

351. [CODEX, board-claimed 07:04Z] FINAL CONTEXT-QK48 CLIFF PROBE:
    rank48/440 maps, census/certs/fresh, next non-overlapping hashed WT103
    train tokens41120:71960, exact 508,055,862 bill and saved CEV.
    -> ops/mixed48_context_metric_qk_newcorpus_ood.py
    RUN 2026-09-01: pred_b/c held, pred_a failed, null false. +.01877572/29,
    OOD +.016828/.048266/.066896. Certificate ledge; no signed gate/rank40.

352. [CODEX, board-claimed 07:07Z] QK56 + MLP0-p512 CROSS-FAMILY COMPOSITION:
    compose two gated components at 507,253,046 scalars; census/certs/additive
    law, next WT103 tokens71960:102800, exact identity/price/fresh/saved CEV.
    -> ops/mixed56_context_qk_mlp0_context_p512_ood.py
    RUN 2026-09-01: pred_b/c held, pred_a failed, null false. 507,253,046,
    +.01951292/29, ratio1.05227, OOD +.014038/.042132/.060740. No tuning/gate.

353. [CODEX, board-claimed 07:10Z] MLP0 CONSEQUENCE-WEIGHTED CONTEXT METRIC:
    weight inputs by clipped full-suffix CE gradient norm at MLP0 output;
    compare p384/p448 against ordinary covariance on two fits/two tail sets.
    -> ops/mlp0_consequence_weighted_context_metric_screen.py
    RUN 2026-09-01: pred_b/c held, pred_a failed, null false. WT p384 tail
    improves strongly but FW max only 4.2%; no clip/rank tuning or promotion.

354. [CODEX, board-claimed 07:16Z] FINITE FOUR-STATE MLP0 SUBSPACE ROUTER:
    exact folded-token PCA32 states, four context-RRR p128 experts, literal
    route table; equal-price global p517 and balanced-random controls.
    -> ops/mlp0_finite_moe_subspace_router_screen.py
    NO-RECEIPT REPAIR: ambient-D sample guard corrected to registered >=500
    and >=expert-rank condition; all states/ranks/bars/seeds unchanged.
    RUN 2026-09-01: ALL POSITIVES FAILED, strong null fired. Four-state
    clustered router FW/WT means .03405/.04047 versus global p517
    .003677/.002102 at essentially equal price; no S/rank/cluster tuning.

355. [CODEX, board-claimed 07:20Z] CONTEXT-METRIC TAIL/WATER-FILLING LAW:
    exact QK/MLP0 omitted singular energies versus measured rank ladders;
    log fits, LOOCV, shared-vs-family gain discriminator.
    -> ops/context_metric_tail_waterfilling_law.py
    RUN 2026-09-01: ALL HELD, null false. QK/MLP R2 .9951/.9947,
    LOOCV medians 7.86%/6.85%; joint R2 .394, family gain ratio3.297.

356. [CODEX, board-claimed 07:28Z] CERTIFICATE DAMAGE-AXIS TRANSFER:
    normalize 62 member damages by native ablation margins; fit one QK ray,
    leave-one-rank counts, then falsify on held-out QK+MLP and value families.
    -> ops/certificate_damage_axis_transfer.py
    RUN 2026-09-01: ALL HELD, null false. QK ray R2 .999452 and zero
    leave-one-rank count MAE; combo cosine/R2/count .998523/.989801/27-vs-29;
    value .997734/.983549/45-vs-46. Shape identified; prospective scale pending.

357. [CODEX, board-claimed 07:36Z] PROSPECTIVE CERTIFICATE-CONSTRAINED ALLOCATOR:
    fit ray intensity from damage on pure QK + mixed104/MLP saved CEVs;
    hold out every QK+MLP construction; combine with measured tail-law
    components/tax envelope and enumerate exact-price calibrated-rank grid.
    -> ops/certificate_constrained_waterfilling_allocator.py
    NO-RECEIPT REPAIR: shifted-OOD baseline receipt changed to paired census
    receipt; no analysis or bar changed. RUN 2026-09-01: substantive pred_a/b
    held, null false. Heldout log-scale R2 .951685, scale median error4.42%,
    measured-damage count MAE .5, end-to-end count MAE0. No <512,561,462
    grid point conservatively retains43; no physical run. pred_c procedural.

358. [CODEX, board-claimed 07:39Z] MDL DEPLOYMENT CROSSOVER LAW:
    exact lower envelope for native + six fully gated QK ranks, uniform
    32/16/8-bit scalar hypotheses and literal raw bytes; mapped r48 separate.
    -> ops/qk_frontier_mdl_crossover.py
    RUN 2026-09-01: ALL HELD, null false. Every adopted point has a nonempty
    interval. Uniform16 transitions at 11.581/16.909/26.272/43.840/64.004/
    84.732B tokens from qk56 through native; literal bytes equal uniform32.

359. [CODEX, board-claimed 07:43Z] CONTEXT-STATE FINITE-MOE FALSIFIER:
    four nearest-centroid states in live MLP0-input PCA32, four fixed p128
    experts, literal router price; p515/p516 shared and random-centroid controls.
    -> ops/mlp0_context_state_finite_moe_router.py
    RUN 2026-09-01: ALL FAILED, strong null fired. Kmeans FW/WT mean
    .04244/.06062 vs cheaper global-p515 .001587/.003097; no better than
    random on WT and fit-B WT unstable. No state/rank/PCA/cluster tuning.

360. [CODEX, board-claimed 07:47Z] PHYSICAL FP16-STORED QK56 FACTORS:
    fixed gated QK56; store all 31,539,200 factor scalars fp16, dequantize for
    contractions; census/certs/fresh/CEV and WT103 tokens133640:164480.
    -> ops/mixed56_context_qk_fp16_storage_ood.py
    RUN 2026-09-01: ALL HELD, null false. +.01250427/43, quant increment
    -.00000352, CEV MAD .00013663, OOD .008849/.036152/.070332; exact
    fp16 factors and 1,871,225,452 bytes. Advances frozen signed gate361.

361. [CODEX, conditional board-claim 07:50Z] TIGHT SIGNED FP16-QK56 GATE:
    only after full rung360 pass; identical a16 intervention, fp16 storage
    identity/byte bill, and frozen rank56 .98/.30/.98 signed bars.
    -> ops/a16_transfer_mixed56_context_qk_fp16_storage.py
    RUN 2026-09-01: ALL TIGHT BARS HELD, null false. cosine/error/norm
    .981648/.236577/1.121547, rho .993469, own1.117394; exact fp16 bill.
    Formally adopt 512,561,462 scalars / 1,871,225,452 bytes / +.01250427 /43.

362. [CODEX, board-claimed 07:57Z] UNIVERSAL BF16-STORAGE/FP32-COMPUTE:
    round every source-fp32 checkpoint tensor through bf16, preserve original
    bf16 tensors; exact 2-byte/scalar bill; 40 FW + 40 new WT103 rows.
    -> ops/bilin18_universal_bf16_storage_screen.py
    RUN 2026-09-01: ALL HELD, null false. FW mean/p95/max
    +.000093/.001104/.001471; WT -.001133/-.000233/+.000158. Exact 218
    tensors, dtype ledger, and 1,091,805,804-byte bill. Licenses rung363 only.

363. [CODEX, conditional board-claim 07:56Z] UNIVERSAL BF16 + FP16-QK56:
    source-aware bf16 storage for all model tensors, identical split-B
    context-QK56 factors stored fp16, original-native census/certs/fresh and
    next WT103 tokens174760:205600; exact 1,025,122,924-byte bill.
    RUN 2026-09-01: ALL HELD, null false. Original-native +.01261160/43;
    OOD .008268/.040903/.052098, structural fresh max .0066; exact global
    BF16/QK-fp16 identities and 1,025,122,924 bytes. Advances rung364 only.

364. [CODEX, conditional board-claim 08:00Z] ORIGINAL-NATIVE SIGNED GATE:
    only after full rung363 pass; measure original-native a16 KO before model
    rounding, then exact combined-program KO; tight .98/.30/.98 signed bars.
    RUN 2026-09-01: ALL TIGHT BARS HELD, null false. Original-native signed
    cosine/error/norm .981519/.237538/1.122134, rho .993980, own1.118250;
    exact 1,025,122,924-byte bill. Formally adopt the 50.4213%-smaller point.

365. [CODEX, board-claim 08:05Z] UNIVERSAL-BF16 HIGH-FIDELITY FULL GATE:
    original-native census/certificates and a16 signed comparison for the
    exact 545,902,902-scalar / 1,091,805,804-byte source-aware BF16 endpoint.
    RUN 2026-09-01: ALL HELD, null false. Census +.00000908/62; signed
    cosine/error/norm 1.000002/.005005/1.000537, rho .999898, own1.000878;
    exact 1,091,805,804 bytes. Formally adopt high-fidelity BF16 endpoint.

366. [CODEX, board-claim 08:11Z] ALL-LAYER CONTEXT-METRIC MLP INPUT RANK:
    independently fit paired-Left/Right context-RRR p512/p768 at all18 layers
    on two splits; two untouched corpora; historical matched weight-SVD control.
    RUN 2026-09-01: ALL HELD, null false. All18 qualify at both ranks; all18
    p768 split-stable; max .011604; 15/18 beat weight SVD by20%, late3 by50%.

367. [CODEX, board-claim 08:13Z] QK64 + SELECTED MLP{0,4}@p768:
    frozen screen rule selects layers4,0; install split-B programs with
    context-QK64; census/certs/fresh and WT103 tokens209712:240552; exact
    511,758,646-scalar / 1,931,092,588-byte structural price.
    RUN 2026-09-01: ALL HELD, null false. +.01224396/43, OOD
    .007718/.033071/.054272, fresh max .0122; exact selection/maps/price.
    New semantic frontier, 802,816 scalars below QK56; advances rung368 only.

368. [CODEX, conditional board-claim 08:17Z] TWO-BYTE QK64+MLP{0,4}:
    only after rung367 full pass; global source-aware BF16, fp16 QK factors,
    original-native census/certs/fresh and WT103 tokens240552:271392;
    exact 511,758,646 scalars / 1,023,517,292 bytes.
    RUN 2026-09-01: ALL HELD, null false. +.01232938/43, OOD total
    .006120/.028598/.040376, fresh .0121; exact global-BF16/fp16-QK/maps and
    1,023,517,292 bytes. Advances frozen rung369 only.

369. [CODEX, conditional board-claim 08:17Z] FINAL ORIGINAL-NATIVE SIGNED GATE:
    only after rung368 full pass; identical a16 mean KO with .98/.30/.98
    tightened bars and exact final selection/storage/scalar/byte identities.
    RUN 2026-09-01: ALL TIGHT BARS HELD, null false. signed cosine/error/norm
    .986524/.191595/1.086070, rho .995306, own1.085463. Formally adopt
    511,758,646 scalars / 1,023,517,292 bytes / +.01232938 /43.

370. [CODEX, board-claim 08:30Z] SHARED GRASSMANN ENCODER FOR MLP0/4:
    midpoint of split-B p768 encoder rowspaces; covariance-optimal per-layer
    coefficients; independent/one-sided/random controls on two new corpora;
    exact 884,736-scalar prospective saving.
    RUN 2026-09-01: pred_a/b held, pred_c failed, null false. Midpoint
    FW/WT .005953/.003992 and tails pass, but rowspace overlap .678779 is
    near random .666667 and below .72. No physical promotion/tuning.

371. [CODEX, board-claim 08:31Z] BEHAVIOR-NAMED FINITE MLP0 ROUTER:
    four exact token-morphology states, four p128 context experts, matched
    global p517 and balanced-random controls; two fit halves/two corpora.
    NO-RECEIPT REPAIR: inherited500 guard parameterized; rung uses frozen300.
    RERUN: feasibility failed before arm scoring—digit state356/206 across
    fits, so fit-B misses registered>=300. No lowering/merge/refit; close.

372. [CODEX, board-claim 09:40Z] QK64 + SELECTED MLP{4,0,2}@P768:
    one-step falsifier of distributed mild cuts using the already-frozen
    all-layer rule; split-B maps, census/all62/fresh and untouched WT103
    tokens285784:316624. Exact 509,104,438-scalar / 1,920,475,756-byte
    structural bill. >=43 certs is required for frontier advancement; no
    fourth layer, rank, subset, rule, or bar tuning after observation.
    RUN 2026-09-01: FRONTIER BAR FAILED. +.01517842 and 38/62 certs;
    OOD .006355/.031895/.050956 and fresh .0159 held. Emitted pred_c false
    from inherited old scalar literal, while recorded identities/bill are
    exact; constant repaired for future use, receipt preserved, no rerun.
    Close this continuation and retain rung369 as the 43-cert frontier.

373. [CODEX, board-claim 08:47Z] QK72 + SELECTED MLP{4,0}@P768:
    prospective mid-fidelity tier using the already-frozen two-layer rule;
    census/all62/fresh and untouched WT103 tokens316624:347464. Exact
    516,264,246-scalar / 1,949,114,988-byte fp32 bill. Requires <=.011 and
    >=48 certs plus frozen OOD/identity bars; no QK-rank interpolation.
    RUN 2026-09-01: ALL FUNCTIONAL/IDENTITY BARS HELD, null false.
    +.00922690/50, OOD .002260/.024200/.036493, fresh .0118; exact
    516,264,246 / 1,949,114,988 bill. Generic harness status/rung/prose labels
    remain stale 367/QK64; recorded qk_rank/maps/selection/bill are exact.

374. [CODEX, conditional board-claim 08:49Z] FIXED CERTIFICATE-RAY AUDIT:
    project saved CEV vectors for rungs367/372/373 onto rung356's frozen
    QK-only 62-member shape; require cosine>=.95, R2>=.80 and count error<=3
    for each, with exact count/CEV/tag/threshold reproduction. CPU only; no
    ray refit, no threshold adjustment.
    NO-RECEIPT CPU ATTEMPT: census_lib transitively loads the CUDA model even
    for saved-state arithmetic. Dryrun/gate passed; enqueue identical script.
    RUN 2026-09-01: ALL HELD, null false. Fixed-shape cosine
    .999417/.998808/.999024, R2 .995891/.991662/.993161, predicted/actual
    counts 41/43,37/38,49/50. Distributed MLP cuts remain on the QK-only ray.

375. [CODEX, board-claim 08:55Z] ALL-LAYER VARIABLE-RANK WATER-FILL:
    exact context-metric tails at p512/p640/p768/p896 for all18 layers/two fits;
    frozen rung355 exponent, fit-A layer gains, untouched fit-B validation;
    dynamic programs at exact four and five 1,327,104-scalar saving units.
    Screen only; a full pass licenses one physical seven-unit build.
    NO-RECEIPT CORRECTION: dryrun caught encoder-overhead price; p1024 is
    break-even. Correct matched/below-frontier prices are5,308,416/6,635,520
    saved and QK64 exact5 bill510,431,542. No empirical output preceded fix.
    RUN 2026-09-01: pred_a held; pred_b/c failed; NULL FIRED. Split-B factor
    error1.223, rho.9521, order18/18, so the model is identified—but exact4
    chooses the existing `{0:768,4:768}` at ratio1.000, and exact5 deepens
    layer4 to640 at ratio1.369. No build; current local rank grid is closed.

376. [CODEX, board-claim 09:03Z] HELD-OUT SECOND CERTIFICATE MODE:
    subtract fixed rung356 ray from three MLP-bearing saved-CEV vectors, fit
    one residual direction, and test it on held-out QK72+MLP04; value96 is a
    specificity control. Exact counts/tags/thresholds; no heldout refit.
    RUN 2026-09-01: pred_a/b held, pred_c failed narrowly, null false. Train
    residual R2.9352/LOO cosines.893-.916; held cosine.9319 improves full R2
    .99316->.99910 and count49->50(actual50), but value residual cosine.5257
    exceeds frozen<=.50. Map universal curvature; no MLP-specific allocator.

377. [CODEX, conditional board-claim 09:09Z] TWO-BYTE QK72+MLP04 TIER:
    identical rung373 program under global source-aware BF16 + FP16 QK;
    original-native census/all62/fresh and WT103 tokens347464:378304; exact
    516,264,246 scalars / 1,032,528,492 bytes. Requires <=.012/>=48.
    RUN 2026-09-01: ALL HELD, null false. +.00930063/50; OOD
    .003228/.025150/.035266, fresh.0118; all identities and corrected
    rung/status/claim labels exact. Advances only frozen signed rung378.

378. [CODEX, conditional board-claim 09:09Z] MID-TIER SIGNED GATE:
    only after every377 positive; original-native a16 KO, >=.985 cosine,
    <=.25 error, .98 collateral rho, exact tier identities. No tuning.
    RUN 2026-09-01: ALL HELD, null false. Baseline+.00930063/50; signed
    cosine/error/norm.990600/.153655/1.060392, rho.996633, own1.065080;
    exact516,264,246 /1,032,528,492. Formally adopt 50-cert middle tier.

379. [CODEX, board-claim 09:13Z] VALUE PRICE-FIRST TAIL CLOSURE:
    exact context-metric c_v tails for144 maps at ranks64:8:112; calibrate on
    measured value96 surcharge, give every rank the optimistic exponent from
    the observed MLP/QK envelope, and compare CE/saved-scalar to adopted MLP04.
    Screen only; pass closes this rank grid, fail licenses one winning build.
    NO-RECEIPT REPAIR: covariance returned CPU while c_v is CUDA; add the
    explicit device transfer used by rung355 and rerun unchanged. No tails seen.
    RUN 2026-09-01: ALL HELD, null false. Exact tails/order/identities; best
    optimistic rank104 exchange is3.6838x adopted MLP04 (r96 is4.0898x;
    range3.68--5.48x). Close value ranks64--112; no physical build.

380. [CODEX, board-claim 09:22Z] GAUGE-INVARIANT TUCKER TOY:
    d48 planted input-r12/output-r10/product-k18 symmetric bilinear tensor;
    invariant unfolding recovery, factor gauge scramble, fresh values/JVPs,
    dense symmetric negative, exact1836-vs56448 price. CPU instrument only.
    RUN 2026-09-01: ALL HELD, null false. Input/output overlap~1; tensor/value/
    JVP R2=1; gauge error1.72e-7/projector.9999998; dense negative tensor/value
    R2.0856/.1181. Licenses one randomized real-MLP0 invariant-Tucker screen.

381. [CODEX, board-claim 09:27Z] REAL MLP0 OUTPUT-TUCKER GRAM:
    exact invariant output-mode Gram under Euclidean, exhaustive embedding,
    and split-A/B context metrics; Down-column alignment permutation null;
    p256/p512 energy, split/embedding overlap, exact2,950,272-vs5,309,568 price.
    RUN 2026-09-01: pred_b/c held, pred_a failed, null false. Context p256/
    p512 energy.657/.826 (<.75/.90), but split/embedding overlaps.927/.836
    and real-null gaps.139/.078. Real stable broad alignment; no promotion/tuning.

388. [CODEX, board-claim 12:55Z] PHYSICAL L16 TUCKER CALIBRATION:
    frozen `(r512,k576,p512)` importance-truncation core versus the same selected
    576 native product atoms and seed388 random-product negative; diverse census,
    all62 certificates/ray, untouched FineWeb, WikiText-103, exact identities and
    literal prices. First attempt crashed before predicates because 513-token
    census source rows were not sliced to the frozen 257-token window; mechanical
    repair only. RUN 2026-09-01: pred_c/d held, pred_a/b failed, null false.
    Tucker local R2 .81456/.83063, census +.047796/22 certs, WT103
    .05451/.08367/.09696, fresh .05396/.08307; ray cosine/R2 .98157/.83791,
    predicted/actual24/22. Equal-cheaper selected product is +.75230/0 certs and
    random +.75159/0, so joint coordinates are necessary but the L16 2.07M corner
    is not adoption-grade. No signed/composition gate and no L16 rank tuning.
### Rung 389 — clean-split current-harness gate for old L16 quadratic surrogate — COMPLETE

- Reproduced old overlapping-row CE exactly, corrected its omitted constant from 13,832 to 14,984 literal values.
- Clean fit-B-only R4/k2 program: local R² .82368/.82178, census +.038978, 27/62, WT mean .039663,
  fresh mean .040634. Equal-price random-output null: R² −.0062, +.148198, 5/62.
- All registered predictions held; null false. Strictly dominates rung388 Tucker at 137.85× lower layer price.
- Licensed successor: original-native signed-a16 gate, then one composition screen only on a signed pass.

### Rung 390 — original-native signed-a16 gate for clean L16 quadratic program — COMPLETE

- Rebuilt the identical fit-B-only R4/k2 program and reproduced its saved unablated CEV bit-exact at
  +.0389782861 and 27/62, with exact 14,984/529,991,486 price, shapes, and live hooks.
- Signed candidate/native attention-16 knockout cosine/error/norm is .9797226/.2130622/1.0520105;
  collateral Spearman .9973469 and a16-owned median magnitude ratio 1.0411815.
- All registered predictions held and the strong null is false. This is causal identification, not standalone
  adoption.
- PRICE CORRECTION after scoring: the hook retained four dense 1152×1152 forms, so the executed layer dictionary
  stored 5,314,176 values. The 14,984 count is the mathematical factor bill, not yet physical. Composition is
  paused; function and signed results stand, but low-price/dominance claims await rung391.

### Rung 391 — physical eight-projection storage and signed reproduction — COMPLETE

- Rebuild identical fit-B forms; replace every dense rank-2 matrix by two signed projection vectors plus two
  scalars; retain only output directions, 4×2 vectors/scalars, and constant (exact 14,984 values).
- Require form/prediction equivalence, saved dense-CEV reproduction, exactly27 checks, and original-native signed
  a16 fidelity plus live hooks. A pass restores the literal price and one-composition license; failure maps the
  result only as a 5,314,176-value dense surrogate.
- RUN 2026-09-01: all held, null false. Actual stored shapes `[4,1152]`, `[4,2,1152]`, `[4,2]`, `[1152]`
  sum to14,984 with no dense form. Form/output errors 4.89e-8/4.65e-7; CEV max/mean differences
  1.24e-5/1.09e-6; +.038978/27 exact; native signed cosine/error/norm .979722/.213062/1.052011.
  Literal price, Tucker dominance, and one QK64+MLP0/4 composition license restored.

### Rung 392 — physical QK64 + MLP0/4 + factored-MLP16 composition — COMPLETE

- Installed the exact fp32 structural QK64/440 maps, split-B MLP{0,4}@p768, and the shipped four-tensor
  14,984-value/no-dense MLP16 program at 495,847,230 scalars and 1,867,449,228 source-format bytes.
- Physical census/checks are +.05285390/17 versus additive +.05122224/17: 1.03185x tax, normalized-vector cosine
  .99996594, and zero certificate-count error.
- Full native-relative WT103 mean/p95/max .055836/.110016/.127139; conditional fresh max .0126. All registered
  predictions held, strong null false, and all hooks/maps/fits/bills were live and exact.
- Licensed successor: one original-native signed a16 composite gate, then stop the short L16 adoption chain.

### Rung 393 — original-native signed gate for physical three-family composite — COMPLETE

- Apply the fixed original-native attention-16 mean knockout to the exact rung392 physical program.
- Require exact rung392 identity and live factor/ablation hooks; signed cosine >=.95, normalized error <=.40,
  norm ratio [.70,1.30], collateral Spearman >=.95, and a16-own median ratio [.70,1.30].
- RUN 2026-09-01: all held, null false. Signed cosine/error/norm .965712/.317149/1.147792, collateral
  Spearman .989592, a16-own median ratio 1.135428; factor/ablation hooks 2,341/250; all identities exact.
- Formally adopt the 495,847,230-scalar program only as a 17/62 lower-fidelity predictive, composable, and
  manipulable tier. Close the L16 chain and prioritize MLP0 token/downstream-equivalence work.

### Rung 394 — exhaustive exact-token MLP0 downstream sparse code — COMPLETE

- Enumerate all 50,257 length-one token paths; delete only the bias-free MLP0 write while retaining identical raw
  x0, attention0 value, bias, and block1 remix; measure causal attention1+MLP1 response.
- Fit P256/k16 action-only, joint action+response (two seeds), and shuffled-response codes on token-id mod5 train;
  score the untouched fifth against dense MLP0-write PCA and raw-x0 linear controls.
- Require raw-token remix dominance, lower response than write effective rank, heldout joint response R2 advantage,
  seed/subspace stability, and response-neighbor advantage. Screen only; a full pass licenses live TT/TT-X transfer.
- First attempt stopped at the first backward call because `main` was under `no_grad`; mechanical decorator repair
  preserved all choices and reran. Final: all four predictions failed, null false. The scaled MLP0 term is 2.257x
  the raw term at median norm, not a small copy; attention1 response is compact (rank90 156/PR17.36) but MLP1 is
  broad (734/121.83), making the joined response rank90 841 versus write 601.
- Joint sparse heldout R2 .3308 is only +.0107 over activation-only and +.0415 over shuffled, near dense-PCA256's
  .3535 ceiling; seeds are stable but neighbor margins and discordant-pair test fail. No TT/X promotion or tuning.

### Rung 395 — exact-token identity transport and causal component split — COMPLETE

- Fit heldout full linear+intercept raw-x0↔MLP0-write and orthogonal maps, retrieve each predicted raw vector from
  all 50,257 embeddings, and compare with shuffled token pairing.
- Inject the fitted token-identity write component and exact residual separately through native block1; score
  attention1/MLP1 response reproduction and residual dimension. Diagnostic identity test, not compression.
- A/B/C identify a private transformed-token component; D additionally licenses shared-residual work. A strong
  miss pivots to the exact quadratic token kernel. No rung394 sparse-rank tuning.
- RUN 2026-09-01: pred_c held; a/b/d failed; null false. Linear write→raw/raw→write R2 .227/.397 and Procrustes
  cosine .387 reject a simple linear copy, but exact token retrieval is 90.4% top1/96.2% top5 versus shuffle 0%.
  The raw-token-predicted component reproduces attention1/MLP1/joint response R2 .926/.972/.972 despite only .397
  write R2. Residual PR is 3.98x native and residual-only joint R2 .247. Pivot to exact quadratic token kernel.

### Rung 396 — exact MLP0-input degree-one token-kernel rank curve — COMPLETE

- In exact normalized MLP0-input z coordinates, project the exhaustive quadratic write onto constant+degree-one
  functions under the empirical vocabulary metric; freeze ranks16..1152 and exact factor prices.
- Score heldout write, physical attention1/MLP1 response, and exact-token retrieval; compare raw-x0 rank curves and
  shuffled z→write r256. Full pass licenses one live TT transfer; miss closes degree-one rank tuning.
- RUN 2026-09-01: B/C held, A/D failed, strong null fired. Exact-z/raw full write R2 .39746/.39695; rank64 token
  retrieval 97.58% and joint response R2 .96404. But mean-preserving shuffled r256 itself scores .92525, within
  .04517 of real r256; correct rung395's 97% interpretation. No live TT transfer or rank tuning.

### Rung 397 — exact MLP0 mean/linear/quadratic token causal factorial — COMPLETE

- Freeze `F=M+L+Q` on heldout length-one tokens: training-write mean M, complete canonical degree-one projection L,
  and exact residual Q. No selected rank and no compression claim.
- Run all eight native-block1 subsets and exact vector Möbius decomposition for attention1 and MLP1. Compare aligned
  conditional error recovery against fixed shuffled-L, shuffled-Q, and wrong-token full-write controls.
- Distinct L action routes to consumer-effect token interchange classes; Q or interaction dominance routes to a
  consumer-aware quadratic spectrum. This rung cannot license live token-by-context transfer.
- RUN 2026-09-01: all A/B/C/D held, null false. M alone gives joined R2 .92492. Aligned L|M recovers 62.19% of
  remaining joined error versus -0.17% shuffled; aligned Q|M recovers 47.95% versus 4.26% shuffled. Attention1
  recoveries are 73.51%/63.52%, exposing a consumer split. Wrong-token full write R2 .91539. Route to validated
  consumer-effect token interchange; no live context transfer.

### Rung 398 — far-action downstream-equivalence physical interchange — COMPLETE

- Use mod5 fitting tokens only as donors and heldout tokens as receivers. For L and Q separately, select donors by
  conditional attention1/MLP1 effect while requiring component cosine <=.50, then physically swap the component.
- Cross-validate attention-selected donors in MLP1 and MLP-selected donors in attention1; compare raw-embedding,
  action-nearest, and random donor controls. Report widespread both-consumer preservation and decoded examples.
- Pass identifies token-effect equivalence despite different storage directions. Failure keeps exact token identity
  and routes to reader-weighted quadratic spectra. No context or compression license.
- RUN 2026-09-01: A/D held, B/C failed, null false. Far-action L cross-consumer cosines .651/.683 beat random but
  not raw/action neighbors; 5.30% of one route preserves both consumers >=.80 across7,000 donors. Q cross-consumer
  transfer .338/.697 is asymmetric with negative selected MLP R2, and only .43%–1.44% preserve both consumers.
  Local semantic pairs exist, but global equivalence is unsupported. Retain exact identity; route to Q spectra.

### Rung 399 — consumer-aware quadratic-residual action spectra — COMPLETE

- Fit mod5-train-only Q-whitened action directions from attention1, MLP1, balanced joint, and shuffled-joint
  conditional effects. Reconstruct heldout Q at frozen ranks16/64/256/512 and physically inject after M+L.
- Compare equal-rank ordinary Q-PCA, require full-rank exactness, and price literal per-token codes plus decoder.
- A response-aware win licenses one fixed Q-table rank confirmation. Failure retains exact Q or ordinary PCA as
  measured; no token grouping or context transfer.
- RUN 2026-09-01: A/D held, B/C failed, strong null fired. Consumer spectra differ at r64 (attention-aware gains
  .106 attention R2 over MLP-aware; MLP-aware gains .069 MLP R2), but joint response-aware R2 at16/64/256/512 is
  .189/.375/.665/.827 versus PCA .207/.398/.698/.865. No response-aware rank beats PCA. No confirmation; exact Q.

### Rung 400 — centered token/context/normalization causal ANOVA — COMPLETE / IDENTITY-REJECTED

- Reuse exact MLP0 bilinear algebra but change to an independently crossed product-reference decomposition of the
  unnormalized numerator: constant, token main, context main, centered token×context, and explicit RMS-gain residual.
- Estimate reference moments on FIT only; run the complete 16-arm T/C/I/S factorial through FIT and SELECT CE while
  retaining the constant, native bias, and bf16 residual.
- Identify whether continuous context meaning is primarily centered CC, centered X, or normalization modulation.
  Diagnostic only; no compression/adoption or token-only tuning.
- RUN 2026-09-01: A/B failed, C/D held, strong null fired on identity. Scalar-only relMSE1.55e-6 exceeds1e-8.
  Pending exact repair only: split-stable Shapley I1.538, T1.498, C.418, S.067; combined context2.023. Do not
  publish/promote until explicit vector RMS residual reproduces all arms and closes exactness.

### Rung 401 — exact vector normalization-residual repair of centered ANOVA — COMPLETE

- Keep every rung400 physical arm unchanged but write `z=s(e+a)+r` and explicitly retain the missing bilinear
  residual involving r. Preserve the original 1e-8 identity bar rather than relaxing it.
- Require all16 FIT/SELECT arm CEs and T/C/I/S Shapleys to reproduce rung400, then rescore its frozen B/C/D outcome.
- Exact attribution only; no FINAL, compression, or adoption. Largest stable contextual role selects the next grammar.
- RUN 2026-09-01: all registered repair predicates held, null false. Explicitly retaining
  `R=T(s(e+a),r)+T(r,s(e+a))+T(r,r)` reduced analytical relative MSE to `2.88e-13/2.89e-13` on FIT/SELECT and
  reproduced every rung400 arm CE and Shapley value exactly. The original `C>=.50` prediction remains failed;
  SELECT roles are `I=1.53753 > T=1.49833 > C=.41773 > S=.06749`, with combined context `2.02275`. Exact causal
  attribution only; route to centered interaction `I`.

### Rung 402 — centered token×context interaction head carriers — COMPLETE

- Split attention0's native write exactly into nine output-projected head writes plus an always-retained BF16
  arithmetic remainder. By bilinearity, split rung401's centered `I` into nine semantic `I_h` terms plus a retained
  numerical interaction remainder.
- On the same frozen FIT/SELECT roles, physically score 21 arms with `T/C/S` fixed: all heads, numerical remainder
  only, zero total interaction, every semantic head alone, and every head omitted. Report singleton sufficiency,
  full-boundary necessity, endpoint-average benefit, split transport, and comparison with the frozen historical
  direct-head cost map. The extra zero-total arm was added pre-execution so both the parent boundary and the causal
  inertness of the always-retained BF16 remainder are testable.
- Exact diagnostic only. Sparse, stable carriage licenses source-position resolution of the fixed top heads;
  distributed/redundant carriage pivots to branch-resolved auditing of the adopted rank448 MLP0 context projection.
- FIRST RUN 2026-09-01: instrument predicate A failed and the strong null fired, so head content is withheld pending
  a same-rung mechanical repair. `FULL` reproduces the parent exactly and the semantic-plus-numerical interaction sum
  has relMSE `2.24e-18`, but `ZERO_I` formed the same float interaction by summing ten terms before BF16 conversion;
  it differed from the parent's one-tensor BF16 subtraction by `6.43e-5/1.04e-4` CE, above the frozen `1e-6` bar.
  Repair only `ZERO_I` to subtract the already computed parent `I` tensor directly. No semantic arm, row, score, or
  threshold changes. Preserve the first receipt/log and FORCE-rerun after gates.
- REPAIRED RUN 2026-09-01: A/B/D held, C failed, null false. FULL and ZERO_I now reproduce both parent boundaries
  exactly; I sum relMSE `2.24e-18`, BF16 head remainder energy `2.75e-6`, numerical-only SELECT effect `-.00030`,
  live census. Head3 is top on both roles, with SELECT singleton/removal/average `.09957/.04128/.07042`; split rho
  `.9333`, old-direct-map rho `.9000`. But positive top2 share `.6224/.6046` misses the frozen `.65`: one dominant
  head plus distributed tail, not a two-head program. Route to branch-resolved p448 MLP0 audit, not source positions.

### Rung 403 — exact T/C/I/S/A branch errors of fixed rank448 MLP0 — CLAIMED

- Reconstruct the exact rung328 fit-A p448 shared-input program and the exact rung401 product-reference grammar.
- Resolve native-to-p448 output change into delta-T/C/I/S plus explicit auxiliary constant/vector-residual/BF16
  closure, then physically score the complete 32-arm factorial on unchanged FIT/SELECT documents.
- Exact endpoints, map identity, live census, price, transported Shapley damage, concentration, and strong null are
  frozen in `MLP0_RANK448_BRANCH_ERROR_FACTORIAL_PREREGISTRATION.md`.
- Diagnostic only. The winning stable branch selects a new object; no rank tuning or compressor/adoption license.
- FIRST RUN 2026-09-01: A failed and strong null fired solely because p448 was inadvertently rebuilt from the BF16
  grammar model: covariance retained energy `.9010940` misses rung328's float32 `.9011109` by `1.69e-5`. Native and
  compact analytical identities were `2.9e-13/3.3e-13`, both endpoints and parent CEs exact, and all calls live, but
  content remains unpublished. Same-rung repair reconstructs factors from the original pinned float32 source model,
  then evaluates unchanged BF16 arms. No stored historical factor tensor exists; first receipt is commit `e40f9f57`.
- REPAIRED RUN 2026-09-01: A/B/D held, C failed, null false. Rung328 retained energy matches exactly; parent CEs and
  endpoint states are0.0-error; identities `~3e-13`; live calls. p448 total damage FIT/SELECT `.00712/.00564`; named
  ordering `I>T>C>S` transports at rho1.0. SELECT Shapley `I=.00581`, `T=.00190`, `C=.00006`, `S=-.00201`,
  AUX `=-.00012`. The compressor's error is interaction-led. Confirm on larger documents before changing metric.

### Rung 404 — large-document confirmation of p448 interaction-led damage — COMPLETE / STRONG NULL

- Select exactly chunk0 from each of 384 frozen, disjoint FineWeb source documents; split into four fixed96-document
  waves and score the unchanged rung403 32-arm factorial with the same map/reference.
- Freeze population, program, exactness, all-wave I dominance, pooled transport/concentration, auxiliary bound,
  total-damage stability, and strong null in `MLP0_RANK448_BRANCH_LARGE_CONFIRMATION_PREREGISTRATION.md`.
- A full pass routes to a global equal-price interaction-weighted projection; no compressor/adoption by confirmation.
- RUN 2026-09-01: A/D held, B/C failed, and the strong null fired. The exact fixed map has pooled damage `.00711`
  nat and named order `I>T>C>S`, but wave Shapleys alternate: I leads waves0/2 and T leads waves1/3. Pooled
  `I=.00479`, `T=.00334`; their gap `.001448` misses the frozen `.0015`, and I's positive named share `.544`
  misses `.60`. The stable statement is token-grammar-led `T+I`, not interaction-only. Do not build the licensed
  I-only projection; compare joint T+I, I-only, T-only, and covariance objectives at equal rank/price.

### Rung 405 — global T+I active-subspace rank448 screen — COMPLETE / STRONG NULL

- Estimate exact output-Jacobian Grams for T and I with deterministic probes on the unchanged24 program-fit docs.
- In the same normalized-input whitening frame, build equal-price T-only, I-only, equal-trace T+I, covariance, and
  random p448 programs; evaluate exact branch reconstruction and physical CE on rung404's four96-document waves.
- Require branch-specificity, pooled and per-wave predictive improvement, balance, exact endpoints, and fixed price.
  Full registration: `MLP0_RANK448_TOKEN_GRAMMAR_ACTIVE_SUBSPACE_PREREGISTRATION.md`.
- Screen only. Failure closes first-order active metrics and routes to direct nonlinear T+I fitting or a registered
  document-conditional state; no interaction-only or head-label tuning.
- FINAL RUN 2026-09-01: after preserving two scoring-path-invalid receipts, the float32-logit/per-document/float64
  repair reproduces every rung404 baseline wave exactly; A/D hold, B/C fail, null true. T-only lowers exact T MSE
  6.9%, I-only lowers exact I MSE2.4%, and joint lowers their geometric mean4.56% (bar5%), but physical pooled CE is
  covariance `.007107`, T `.007280`, I `.007106`, joint `.007357`; joint improves only2/4 waves. Random costs `.1956`.
  Close these Euclidean T/I derivative metrics. Do not claim all p448 subspaces impossible; CE-Fisher weighting,
  direct nonlinear fitting, and observable-state routing are mathematically different candidates.

### Rung 406 — downstream CE-Fisher rank448 screen — COMPLETE / STRONG NULL

- Preserve full directional suffix-CE gradients at MLP0 input on the unchanged24 fit documents, rather than
  rung353's scalar gradient-norm weights or rung405's isotropic T/I probes.
- Compare covariance, Fisher, and eigenvalue-matched shuffled-Fisher p448 at equal price on four96-document waves;
  run the exact32-arm SELECT branch audit for the fixed Fisher candidate.
- Freeze split stability, predictive wave gains, shuffled control, T+I repair, identities, price, and strong null in
  `MLP0_RANK448_DOWNSTREAM_FISHER_PREREGISTRATION.md`.
- Screen only. Null routes to direct nonlinear CE fitting versus a small-state oracle/headroom test.
- RUN 2026-09-01: A/B held, C/D failed, null true. Fisher is real/stable: half-fit overlap `.7749`, top448 energy
  `.9760`, and shuffled damage `.18490`; all identities/baselines exact. But real Fisher damage `.008513` exceeds
  covariance `.007107` in all4 waves, T/I relative MSE both worsen, and SELECT T+I Shapley damage rises
  `.007710 -> .010280`. Close fixed first-order p448 metrics; do not clip/interpolate/tune. Measure heldout oracle
  headroom before any small-state router, and compare that route with direct nonlinear CE fitting.

### Rung 407 — price-aware two-state p448 router oracle ceiling — COMPLETE / STRONG NULL

- Rebuild the five outcome-frozen p448 programs from rungs403/405/406 plus covariance p640/p768 controls.
- Save physical per-document/per-position losses on the 384-document authority; enumerate all10 two-expert future-
  loss oracles and an all-five oracle without fitting a state.
- Compare optimistic two-p448 price14,599,296 against cheaper p76813,272,192 and p64011,945,088, with frozen
  headroom, balance, wave, identity, and strong-null bars in `MLP0_P448_ROUTER_ORACLE_CEILING_PREREGISTRATION.md`.
- Diagnostic ceiling only. A null kills router-state search among these experts before code is spent on it.
- RUN 2026-09-01: A/B/D held, C failed, null true. Best document pair I-active+Fisher has damage `.004817`, a
  32% p448 gain, balanced `.581/.419` use, and all-wave benefit; all-five document oracle `.002888`. But cheaper
  p768 is `.001173` at13,272,192 values versus at least14,599,296 for two p448 experts, so document routing is
  dominated. Position oracle is `-.02107` for the pair / `-.03928` all-five, leaving only the preregistered
  token-state feasibility study open; it must be prefix-observable, physical, heldout, and beat p768 after price.

### Rung 408 — heldout four-state prefix-token router feasibility — COMPLETE / STRONG NULL

- Fix I-active/Fisher experts and saved rung407 losses; train docs0:192, evaluate192:384 in two96-doc waves.
- Freeze a max4-leaf/min2048 tree over position, GPT2-byte morphology, prefix repetition distance, and training-only
  token frequency; compare constant, quartile, morphology4, and repeat-distance4 states.
- Require25% oracle recovery, `.001` gain over I, `.0002` over cheaper p768, both-wave transport, expert balance,
  exact hashes/no leakage, and frozen nulls in `MLP0_PREFIX_TOKEN_STATE_ROUTER_FEASIBILITY_PREREGISTRATION.md`.
- Off-policy feasibility only. Pass licenses one exact physical router; miss closes cheap prefix-state routing.
- RUN 2026-09-01: A/B held, C/D failed, strong null true. Exact hashes, rows, split, no-future perturbation, and tree
  support/liveness hold. On heldout documents, constant I-active damage is `.007577`, p768 is `.001402`, and the
  four-leaf tree is worse at `.008037` despite using both experts `.734/.266`; it selects only on absolute position.
  Morphology is essentially tied with constant I, while quartile and repetition controls regress. The tree loses to
  p768 in both waves and costs at least14,599,312 values versus13,272,192. No physical router; close this cheap
  prefix-state family and change object to path-specific output/causal-response structure or direct nonlinear CE.

### Rung 409 — frozen rank64 causal-output interface oracle on p448 branch error — COMPLETE

- Reconstruct exact native/p448 T/C/I/S/A output errors on docs0:384; fit new controls only on0:192 and evaluate
  192:384 in two96-document waves, with FINAL closed.
- Compare the independent frozen historical B0 rank64 causal basis, joint T+I PCA64, separate T32/I32 bases,
  total-error PCA64, and seeded random64 by physical oracle output corrections.
- Freeze exact authorities, p448/p640/p768 reproduction, branch identity, B0 gain/random specificity, T+I share,
  equal-direction split-vs-joint error and CE bars, literal prices, and strong null in
  `MLP0_P448_CAUSAL_OUTPUT_INTERFACE_ORACLE_PREREGISTRATION.md`.
- Oracle ceiling only. B0 pass licenses the old executable predictor on p448; split pass licenses separate path
  producers; all-rank64 miss closes this output-repair scale and routes to direct nonlinear or later-layer work.
- FIRST RUN 2026-09-01: A failed and the strong null fired only on branch-closure exactness, so all substantive arm
  comparisons are withheld. Rows/bases/loss hashes, p448/p640/p768 losses (bit-exact), programs, calls, state replay,
  and native/compact analytical identities (`2.88e-13/3.39e-13`) hold. The implementation formed float32 A as
  `total-sum(named)` and then re-summed in a different float order, leaving max closure `1.53e-5` rather than0.
  Same-rung mechanical repair computes and replays that diagnostic remainder once in float64; arms/bases/rows/bars
  are unchanged. Preserve first receipt at its pre-repair commit and rerun through the managed lane.
- REPAIRED RUN 2026-09-01: A/C held, B/D failed, strong null false. All authorities and saved baseline losses are
  bit-exact; branch closure is0 and identities `~3e-13`; all calls/bases live. Historical B0 improves p448 only
  `.001348` (17%, not30%) and beats random by only`.000330`, so do not reuse its old predictor. Its T+I correction
  gains `.001776`, 132% of full gain: token grammar confirmed. Equal64-direction split T32/I32 is worse than joint
  TI64 in branch MSE (`.750>.708`) and physical damage (`.005876>.004838`). Total-error64 reaches `.004769`, still
  above p640 `.002868`. Select one shared rank64 T+I output producer; no separate path dictionaries or B0 reuse.

### Rung 410 — weight-derived shared rank64 output quadratic producer — COMPLETE / SCREEN ONLY

- Reconstruct rung409's total-error U64 on train docs0:192. Contract U64 into the exact native-minus-p448 bilinear
  weights to obtain64 symmetric quadratic coefficient forms in the causally available normalized MLP0 input.
- Under the frozen uncentered input second moment, compare signed eigentruncations r8/r16/r24, full-rank derivation,
  an affine producer, and a seed410 component-shuffle control on heldout docs192:384.
- Rank24 producer/interface costs1,844,800 values; with p448 totals11,799,232, below p64011,945,088. Freeze exact
  derivation, half-oracle recovery, rank ordering, affine/shuffle specificity, calls, waves, and strong null in
  `MLP0_P448_SHARED_OUTPUT_QUADRATIC_PRODUCER_PREREGISTRATION.md`.
- Executable screen only. A priced pass selects one fresh/OOD+signed+composition gate; miss routes away without rank
  tuning. The runtime never reads native output, target IDs, future loss, document ID, or evaluation lookup.
- RUN 2026-09-01: A/D held, B/C failed, strong null false. Exact rows/losses/calls/prices hold; the full quadratic
  derivation matches direct coefficients at relative MSE `4.05e-12` and physical oracle damage within `.000194`.
  Rank24 improves p448 damage `.007947 -> .006929`, a `.001018` gain or `32.0%` of the oracle, and specifically
  beats shuffled24 `.008842` and affine `.007631` in both waves. But it misses the frozen50% recovery bar, retains
  only `.3570` data-metric form energy rather than `.70`, and rank640 is far better at `.002868` for only145,856
  more values. No promotion or rank tuning. The independent-per-output low-rank form is broad; next mathematical
  screen may test a joint tensor factorization sharing input directions across the64 forms.

### Rung 411 — joint-input Tucker factorization of the exact U64 correction tensor — COMPLETE / STRONG NULL

- Factor all64 data-weighted symmetric coefficient forms together as a tied-input Tucker tensor, rather than
  truncating each matrix independently. Freeze shared ranks96/160/226 and a Haar226 control.
- Charge `U`, shared input directions, symmetric64-core, and offsets exactly:482,368/1,082,432/1,975,808 values.
- Compare each producer against the equal-or-cheaper covariance rank p494/p552/p638, not merely p640; reconstruct
  rung410 r24 and the full/U64 oracle controls on the same heldout documents.
- Full registration: `MLP0_P448_JOINT_TUCKER_QUADRATIC_PRODUCER_PREREGISTRATION.md`. Screen only; failure closes
  low-rank quadratic producers without rank or metric tuning.
- RUN 2026-09-01: A held, B/C/D failed, strong null true. Exact losses/calls/U64/form identity/r24 reproduction and
  all prices hold. Tucker96/160/226 damages `.007238/.007153/.006989` versus smaller matched covariance
  p494/p552/p638 `.006051/.004365/.002820`; margins are negative by`.001187/.002787/.004168`. Tucker226 retains
  `.4361` tied energy, recovers30.2% oracle gain, barely beats Haar by`.000350`, and is worse than independent r24
  by`.000060`. No promotion or tuning; close fixed low-rank U64 quadratic producers.
### Rung 412 — physical BF16 derived-MLP repair of the 43-certificate byte tier — COMPLETE / ALL HELD

- Audited rungs368/369 against retained program tensors: their generated MLP0/4 p768 programs were float32 despite
  the all-two-byte receipt field. Correct old executed bill to1,076,606,060 bytes; semantic results remain valid.
- Rebuilt the identical source-BF16/QK64-fp16/MLP0/4-p768 artifact with all ten generated program tensors stored BF16
  on CPU and dequantized fp32 by the existing runtime hook.
- RUN: exact generated object26,544,384 values/53,088,768 bytes; exact whole bill511,758,646/1,023,517,292.
  Census+.012331/43; new WT103 mean/p95/max+.008688/.035301/.063097; fresh max.0122; parent CEV mean/max change
  .000563/.024158. A/B/C/D held, null false. Signed gate licensed.

### Rung 413 — original-native signed gate for physical BF16 43-certificate tier — COMPLETE / ADOPTED

- Apply the fixed original-native attention16 mean knockout to the exact rung412 artifact; freeze baseline,
  source/QK/MLP physical identities, cosine/error/norm, collateral rank, own magnitude, and strong null.
- RUN: baseline+.012331/43; signed cosine/error/norm .986522/.191608/1.086071; collateral Spearman.995714; own
  median1.085469; exact MLP BF16 and whole bill identities. A/B/C held, null false.
- Formal adoption restored at511,758,646 scalar values /1,023,517,292 bytes. Scope is stored tensors plus fp32
  dequantized compute, not latency or native BF16 kernels.
### Rung 414 — physical all-two-byte sub-500M composite — COMPLETE / NEAR-MISS, NOT ADOPTED

- Rebuild adopted QK64+MLP0/4-p768+factored-L16 with source-native BF16, QK fp16, both generated MLP programs BF16,
  and all14,984 L16 values BF16; runtime dequantizes fp32. New untouched WT103 [439984,470824).
- RUN: all identities hold at495,847,230 values/991,694,460 bytes (.92359GiB), with2,545 live L16 calls.
  Census+.052983/17; tax/vector cosine/cert difference1.03438/.999962/0; OOD mean/p95/max
  .047773/.086460/.134243; fresh.0125.
- A/C/D held, B failed, strong null false. Parent-r392 CEV mean/max delta .006527/.117641; maximum exceeded frozen
  .100. No signed gate, rerun, threshold relaxation, or adoption. Move to L17 current-harness screen.

### Rung 436 — +29,968-byte mixed-precision repair of rung414 — COMPLETE / FAILED, NO SIGNED GATE

- Pre-registered diagnosis: keep the 14,984-value degree-two MLP16 program FP32 while source/QK/MLP0/4 retain
  rung414 precision; exact bill495,847,230 scalars/991,724,428 bytes; untouched WT103 `[470824,501664)`.
- RUN: A/C/D hold; census+.052969/17, composition tax/cosine/cert difference1.03409/.999959/0, OOD
  mean/p95/max.065277/.111963/.161361, fresh.0125, all dtypes/counts/hooks/bill exact.
- B fails: parent-r392 CEV max `.116467` versus frozen `.050` and half-r414 bars. FP32 restoration changes only3.91%
  of squared deviation and partly cancels the persistent residual. Diagnosis corrected to nonlinear amplification of
  upstream rounding. Conditional rung437 not licensed or run. No precision sweep; physical frontier unchanged.

### Rung 439b — causal-profile-scored Archetypal Q/K hull — COMPLETE / STRONG NULL

- First receipt preserved as instrument-invalid on a BF16-path gauge check. The exact-float64 instrument-only rerun
  passes every identity and reproduces the science.
- Signed real-token hull residuals beat entry-permuted controls by about21% on query and key, establishing real
  whole-token convex geometry.
- Strict hull changes FINAL pattern/write/CE from U54 `.189/.341/+.0253` to `2.468/1.458/+.0554`; fixed25%-relaxed
  hull remains `1.298/.943/+.0412`. Observable restart stability improves but misses absolute and FIT-half bars.
- A/B true, C/D false, strong null true. No extraction/removal gate. Close factor-row convex anchoring at this
  512-atom/k27 object; retain a causal-response-profile dictionary as a distinct untested object.
### Rung 415 — physical current-harness MLP17 four-output/eight-square surrogate — COMPLETE / STRONG NULL

- Dossier-controlled port of the old whole-layer object, distinct from output projection and activation-conditioned
  Down rank. Fit-B only; fit-A/function-fresh/census/FineWeb/WikiText transfer; seed415 random-output same-price control.
- Historical overlap CE3.55755478 reproduces exactly. Clean/random artifacts are exact14,984-value four-tensor
  programs with no dense forms; factorization errors≈2.5e-8 and hooks/splits are live.
- RUN: clean heldout/fresh R2 -29.685/-32.076; census+.303140/0; WT103 mean/p95/max.3920/.5511/.6915; fresh
  mean/max.3145/.4747. Random census+.6013 but local R2≈-.05. A/D held, B/C failed, strong null true.
- Close the clean whole-layer R4k2 L17 surrogate. No signed gate, composition, or rank/output tuning; retain only
  the older Down-map/frequency-direction causal facts.
### Rung 416 — gauge-stable shared head-written MLP0 subspaces — COMPLETE / STRONG NULL

- Fit rank64 output-projected SHARED-head, TOTAL-write, HEAD3-only, and Haar bases on96 FIT documents; score exact
  MLP0 `I` and `C` singleton/removal endpoints on96 disjoint SELECT documents with FINAL closed.
- RUN: exact/live parent boundary differences0; pooled SHARED FIT-half overlap.85285, heldout I MSE improvement.76936,
  and I endpoint average.074135 versus Haar.010733. Yet median effective writers1.962, only7/64 modes reach3,
  and head3 is top writer for61/64 modes with68.0% mean energy.
- TOTAL wins both I/C endpoint averages (.080964/.033411) over SHARED (.074135/.030863). A/C hold, B/D fail, strong
  null true. Do not call covariance modes shared features or tune rank. Next object is a heldout downstream-response
  quotient plus the dual shared-reader-input and double-QK-half decompositions specified in explanation_1807.
### Rung 417 — finite downstream-response head service at MLP0 — COMPLETE / NO SERVICE IDENTIFIED

- Exact two-background response tensor for the nine rung402 `I_h` paths: singleton/removal changes in native
  attention1 and MLP1 writes, 96 FIT/96 disjoint SELECT documents, 192 positions/document, FINAL closed.
- RUN: interaction closure relMSE≈2.2e-18; direct block1/native-dispatch replay max error0; response Grams transport
  .849-.964. Action rank90=6 and response ranks=5-6, so no material downstream head-mode collapse.
- Heldout head3-from-other8 R2 is .1008/.0951/.0560/.2569 across singleton-A1/singleton-M1/drop-A1/drop-M1 versus
  raw-action .0394 and shuffled≈0. Head8 is the largest coefficient in all cells, but coefficient stability and
  cross-cell R2 bars fail. A holds; B/C/D fail; strong null false.
- Do not call attention0 MLP0-interaction paths duplicate services and do not tune. Dossier audit closes another
  global token partition; advance the still-open gauge-invariant cross-head QK shared-half test, then attention1.

### Rung 440 — historical learned-simplicity archive feasibility — COMPLETE / ARM SCHEMA REQUIRED

- Frozen CPU-only audit of top-level terminal receipts from rungs300–436. Structural features and registered
  consequence labels were emitted separately and joined only by hash-bound keys; zero forbidden consequence fields
  appear in the feature schema.
- A/B/D hold:130 canonical receipts, one ambiguous rung excluded,100% source/Git chronology linkage,8 module and7
  grammar families, and adequate registered-label coverage for OOD/transport, extraction/identification, and
  composition. Strong null false.
- C fails: only22/130 receipts (16.9%) expose explicit arm maps and76/130 (58.5%) expose machine-readable structural
  price, versus70%/70% bars. A receipt frequently compares several candidate programs and cannot legally become one
  training row.
- Do not fit the historical predictor. Next build a hand-reviewed candidate-arm manifest and common consequence
  schema, rerun the feasibility gate as a new generation, and only then license a held-family backtest.

### Rung 441 — hand-reviewed candidate-arm manifest — COMPLETE / NO HISTORICAL FIT LICENSE

- Manually expanded and deduplicated four frozen program families: vocabulary r300/304/305, mixed104 MLP-PCA
  r311–313, MLP0 context-input r325–329, and attention0 sparse Q/K r426/430. Structural and consequence files remain
  separately hashed; every consequence source is hash-pinned.
- A/B/C hold:45 unique fully priced arms, zero duplicates/forbidden structural outcomes, OOD28 arms across2 families,
  and extraction12 across2 families. Strong null false.
- D fails: removal has only2 arms in1 family and composition has0. No consequence reaches the stricter fitting license
  of20 candidates across3 whole program families.
- Do not fit. Preserve the useful slices and prospectively generate independent removal and composition families
  under sealed consequence labels; the historical archive alone cannot validate a learned simplicity rule.

### Rung 443 — old compiler-v2.1 MLP0→MLP1 structural-score transfer — COMPLETE / HISTORICAL POSITIVE

- Recovered matching108-candidate true/shuffle banks at MLP0 and MLP1. Fit only on MLP0 using log price, operations,
  capacity, regularization, and affine/state-complete/causal flags; leave-family-out CV chose ridge alpha100. Freeze
  fit hash before opening MLP1.
- After two preserved output-instrument aborts, A/B/C/D all hold and strong null is false. MLP1 Spearman/pairwise
  accuracy are.6541/.7378; price/rank/shuffle-trained baselines are.3666/.4184/.4679; frozen top-decile true-minus-
  shuffle recovery gap is.5798. Permutation p=.000999.
- Direct MLP0 recovery transfers better at.8197. Therefore structure predicts adjacent-site validation behavior but
  does not replace measured upstream performance, generalize to unseen grammars, or validate OOD/removal/composition.
  Use only to design one new prospective family with sealed consequences.

### Rung 444 — downstream-response Archetypal SAE — COMPLETE / STRONG NULL

- On each exact causal attention0 edge, fit32-atom/top4 unconstrained, real-convex, and source-permuted-convex sparse
  response dictionaries under rung424's downstream-response metric. The authoritative managed receipt passes every
  instrument check; earlier serialization, Boolean-identity, and accidentally unmanaged receipts remain invalid.
- SELECT response-metric error is U32/A32/P32 `.07571/.09780/.09432`. The real convex hull loses to its
  source-permuted control. Median restart cosine is A32 `.6476` versus U32 `.7042`; convex anchoring does not improve
  identification.
- A32 routed-U16 R2 is`.8302`, all six reader R2 values are`.9034-.9353`, and CE damage is+.001889 nat. The reader
  and CE clauses hold, but routed reproduction misses`.90` and remains worse than U32 `.8898`.
- A true, B/C/D false; strong null true on both no real-vs-permuted advantage and no stability improvement. Close
  K32/top4 convex response atoms; retain the continuous rung424 quotient. Next active work is the separately frozen
  prospective removal/composition bank, not another convex-atom budget sweep.

### Rung 445 — prospective consequence candidate-bank freezer — COMPLETE / ALL HELD

- Freeze the exact outcome-free rung441 structure hash before opening any candidate consequence. Teaching roles are
  vocabulary23, mixed104 MLP-PCA7, and MLP0 context-input5; attention0 sparse-Q/K10 is the sealed fourth family.
- All45 candidates map to one of10 compiling, hash-pinned producer sources and a deterministic-refit or retained-
  bundle rebuild path. Ten required artifact classes exist and are hash-pinned; every candidate has positive price.
- A/B/C/D hold, strong null false: teaching has35 candidates/3 whole families; sealed has10 candidates/1 unseen
  family,7 non-controls+3 controls, and zero family overlap. No consequence file, model, or row role was loaded.
- Correct the learning target before outcomes: rung443 remains a fixed reconstruction-recovery baseline. Fit new
  consequence-specific removal/composition rules on teaching families and freeze them before attention0 labels open.
  Next freeze new document-disjoint teaching/confirmation rows and exact consequence bars.

### Rung 447b — teaching/sealed consequence row authority — COMPLETE / ALL HELD

- The first execution stopped before outputs because its raw-only tensor hash did not match the parents' canonical
  dtype+shape+bytes hash. Reregister only that hash function; no row or candidate consequence was opened.
- Freeze second-half96-row slices from two registry-certified FineWeb EVALUATION pools. TEACHING/SEALED are each
  int64 `[96,257]`, split into fixed48-row waves, and are disjoint by full row, prefix32, document ID, and dataset ID.
- Both roles have zero full-row and prefix overlap with every FineWeb candidate-fit cache named by rung445. All input,
  source, output-file, tensor, bank, and receipt hashes pass; no model or consequence access occurred.
- A/B/C/D true, null false. Next preregister exact teaching-family signed-removal/composition harness and bars before
  any of35 teaching candidates run. These rows are candidate-consequence fresh, not globally virgin or OOD.

### Rung 448 — local MLP0 context-input removal/composition diagnostic — COMPLETE / ALL HELD LOCALLY, NOT BANK LABELS

- Deterministically rebuild the five rank256/384/448/512/640 MLP0 context-input programs from the frozen24-row fit
  prefix. On96 TEACHING rows, compare their per-token CE effect under a native attention16 mean knockout and their
  physical composition with the independent14,984-value MLP16 rank2 program.
- The instrument is exact/live: native replay max0; native removal norm26.364; candidate/partner/knockout dispatch
  counts360/144/144. The SEALED_CONFIRMATION role remains unopened.
- Removal normalized error falls monotonically `.14949,.10608,.09090,.07714,.05650`; composition error falls
  `.15650,.12944,.11643,.10467,.08293`. Rank-vs-lower-error Spearman is1.0 for both, spans are.09299/.07358, and
  both orderings reproduce at Spearman1.0 across the two fixed48-row waves.
- A/B/C/D true, strong null false for the registered local object. Post-run bank audit found an object/price mismatch:
  the source replaced only MLP0 inside native, while rung445's candidate ID and531.6–535.6M prices refer to complete
  mixed104+MLP0 compiled models. Preserve these results as local-component diagnostics, but do not count them as
  teaching labels or fit a predictor. Rung449 must install the whole candidate under the same consequences.

### Rung 449c — complete mixed104+MLP0 consequence labels — COMPLETE / ALL HELD

- Rebuild each full mixed104+MLP0 rank256/384/448/512/640 candidate in isolated unablated, attention16-knockout,
  and MLP16-partner processes on TEACHING. Two scorer-only lookup aborts are preserved; the third managed scorer
  reuses the completed condition bundles without repeating model work.
- A holds: all three builds reproduce the complete rank104 Q/K/active-program/rank-r identities and frozen prices;
  KO/partner each fire120 candidate calls; native KO/partner each fire24; replay0; SEALED remains closed. Complete
  candidate CE differs from rung448's local object by mean absolute `.06830–.06950` per token, confirming the repair.
- Removal error falls `.18156→.14676→.13560→.12583→.11370`; composition error
  `.16753→.15050→.14203→.13491→.12373`. Both rank correlations and both two-wave ordering correlations are1.0;
  spans `.06786/.04380` exceed`.015`.
- A/B/C/D true, strong null false. Count five complete labels and MLP0-context as teaching family1/3. Next run the
  seven complete MLP-PCA candidates; fit nothing until all three teaching families exist.

### Rung 450 — complete mixed104 MLP-PCA consequence labels — COMPLETE / STABILITY NEAR-MISS, NOT ELIGIBLE

- Rebuild seven full candidates in isolated conditions: layer pairs0+8/0+17/8+17 at rank256, the8+17 rank384/512
  ladder, and same-price rank256 gradient32/64 hybrids. All three rank104 Q/K/active-program identities and exact
  prices hold; KO/partner each fire168 calls; SEALED stays closed.
- A/B/C hold: the8+17 rank ladder monotonically lowers removal `.24634→.20865→.17966` and composition
  `.24792→.22983→.21273` (both Spearman1); seven-arm spans are`.06669/.05426`.
- D fails: composition wave ordering is stable at`.82143`, but removal is`.50` versus`.70`. The five rank256
  pair/hybrid arms are close and reorder across48-document waves; the high-rank ladder remains separated.
- Strong null false, but full pass was required. Preserve all seven labels; count no second family and fit nothing.
  Next preregister independent additional documents plus a tie/uncertainty-aware reliability test without changing
  rung450, while continuing the still-queued vocabulary family.

### Rung 451 — independent MLP-PCA reliability rows — COMPLETE / ALL HELD

- Create-only freeze the full192-row authoritative MLP0+MLP2 composition-v2 EVALUATION role before any reliability
  outcome, with fixed waves0:96 and96:192. Candidate-consequence fresh, not globally unused.
- A/B/C/D true, null false: exact parent hashes/tensor/provenance; unique int64 rows/prefixes/documents/indices; zero
  overlap with TEACHING, SEALED_CONFIRMATION, or bank fit caches under every registered key.
- Receipt `d9914694…`, tensor `ff34039b…`; model and consequence access false. Next preregister uncertainty-aware
  replication; this row instrument does not reverse rung450 or count family2.

### Rung 452 — uncertainty-separated MLP-PCA comparison freezer — COMPLETE / ALL HELD

- Reconstruct every rung450 normalized error from per-document sufficient statistics, then draw2,000 shared96-document
  bootstrap resamples at seed452. Freeze a pair only when the95% percentile interval for left-minus-right excludes0.
- A/B/C/D true, null false: exact old sources and all14 metrics reproduce;13/21 removal and16/21 composition pairs are
  separated; rung451 rows/model/outcomes and SEALED remain unopened. Spec SHA `90ff8209…`.
- The eight unresolved comparisons are all internal to the close same-price rank256 group. This freezes the independent
  questions; it does not change rung450 or count the family.

### Rung 453 — independent complete MLP-PCA consequence reliability — COMPLETE / ALL HELD

- Rebuild all seven complete candidates on the disjoint192-document role in isolated unablated/knockout/partner
  processes. Exact identities/prices/replay hold; candidate interventions fire336/336, native48/48, SEALED stays closed.
- Old-to-independent continuous-error Pearson is`.99021` removal and`.98168` composition; mean absolute shifts are
  `.01288/.01150`, passing`.85/.04` bars.
- All13/13 removal and16/16 composition directions reproduce pooled and separately in each96-document wave. The
 8+17 rank256→384→512 ladder decreases for both errors in the full role and both waves.
- A/B/C/D true, strong null false. Count all seven labels and MLP-PCA as teaching family2/3 without retro-passing
  rung450. Vocabulary23 is next; predictor fitting and sealed attention0 remain closed.

- Rung 478 (§2596): sparse mixed-product gates — strong null; all six candidates fit in-view (.982+) but best non-fit transfer .059 < alignment-destroyed control q95 .339. Sparse weighted native-product mixtures CLOSED. Next: rung 479 gauge-aware commutant screen; support-stability (Jaccard across fit-half bisection) is a standing audit requirement for any future fitted-gate survivor.

- Commutant null calibration (§2597): probe calibrated at n=32; interpolation floor q05 .124; recommend 479 genuine-block bar <=.03 off-mass; commutant_dim criterion 0/32 false positives.

- Rung 479 (§2598): gauge-aware commutant screen — strong null; lambda2 ratios .95-.97 vs 4x bar; equality-MLP trio closed at every grain; routing: attention0/1 Q/K/output tensors. Chapter explanation explanation_2026-09-02_0753.md.

- Projector factorial + diagnostics (§2599): TEMPORAL REPRODUCIBILITY BREACH — 474's own code fresh-vs-bundle .084 nat; cross-session bundle bridges unsafe at <.1 nat; canary fingerprint live (first baseline 08:24). Projector-form question OPEN pending in-run-baseline b-variant (code cosine .9389 unclaimed).

- Rungs 480/481 + decider trio (§2600–§2602): attention0 gauge-stable B-positive preserved (geometry fixed, semantics view-specific); MLP0 branches task-unselective at circuit grain; breach H-B confirmed + contained (cause unknown, tripwire live); projector-form code-register alignment .9389 CLAIMED (c-variant, sound instrument); next objects: consumer attention1/MLP1 Jacobians (Codex), natural-register arc still parked.
