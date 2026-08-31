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
