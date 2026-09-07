# Hourly strategic review — 2026-09-07 07:14 UTC

## Circuit target and evidence levels

The controlling target remains a predictive, composable, manipulable, simpler tensor program that
specifies source, read, compute, write, and downstream consumption. This review keeps three evidence
levels separate: a validated downstream response program, a donor-patch upstream writer atlas, and a
recursively runnable source-driven graph. Only the first is currently strong enough to call a
high-quality circuit; the third remains incomplete.

The shared temporal/is-was downstream circuit has eight sites: `MLP1`, `MLP3`, `MLP4`, `MLP6`,
`L8H1`, `L9H1`, `L9H4`, and `L11H3`. Its two independently fit rank-8 response-projector families
remain cross-fit stable and exactly readable through `c_proj`/`Down` weights. The factor-pruned causal
program now shows that all four attention responses can retain only base-pattern-weighted value
change from cue/post-cue sources, while all four MLPs retain only left-change plus right-change. It
achieves worst target residual `.0256`, minimum signed projection `.8658`, control margin `.0270`,
median KL `.00122`, and zero flips. The algebraically visible MLP1/is-was bilinear term is not
causally required: removing it costs only about `.002` projection. This deletes an exception rather
than adding one.

## Weight-derived coefficient generators

Each retained MLP response coordinate was pulled one level upstream through frozen
`Left/Right/Down` tensors. Rank 32 was slightly too aggressive: mean cross-panel coefficient RSE was
`.083-.086`, but one cell reached `.141` and one fit's target projection was `.794` against the `.80`
bar. The already frozen rank-64 frontier then received a genuinely later-corpus confirmation on
disjoint temporal-v13/is-was-v12 text. All five gates pass: maximum per-site OOD coefficient RSE
`.0685`, worst target residual `.0341`, minimum projection `.8166`, control margin `.0304`, median KL
`.000894`, zero flips, and no material degradation under three 10% input-noise seeds.

This establishes low **effective** rank on the task activation manifold, not globally low-rank
weights. Rank 32 captures only about `.54-.65` of raw effective-covector energy despite predicting
task coefficients well. That distinction is essential for the DAS red-team: useful task directions
need not be dominant singular directions of the naked weight matrix.

The attention value term was then compiled exactly into local-current and layer-0 cached-value
`c_v` pullbacks. The first receipt was formally invalid because local RSE `1.10e-9` exceeded a
registered `1e-10` float bar, although independent complete-program centered-logit RSE was
`2.49e-11`. A hash-bound zero-forward BF16 tolerance audit passed all five gates at a preregistered
`2e-9` local bar. The local-current branch alone remains target-sufficient; the cache adds only
`.013-.016` projection, while cached-only is strongly insufficient. The simpler current-value-only
program is promoted. Aligned attention input-weight maps have minimum cross-fit cosine `.9415`.

## Full head/module patching and recursive closure

A complete 110-component atlas patched every head and MLP through layer 10 and scored each forward
against all eight compiled coordinates. Coordinate incidence predicts native causal magnitude at
Spearman `.885`; both projector fits select exactly the same top 20; six of seven expected response
frontier components occur in that pool. The joint pool has only `.0227` mean coordinate residual and
native target projection about `.963-.968`, but its MLP1 coordinate projection is only `.719-.720`.
Thus aggregate donor-patch ranking is highly informative but not coordinate-complete.

The missing MLP1 source is the aligned changed-cue embedding carried through the persistent `x0`
stream. Cue embedding alone gives MLP1 projection `1.0`; top20 plus cue embedding reconstructs all
eight coordinates and the native target effect exactly. Full embedding replacement is not selective,
however: P controls have median KL `.102` and `18.75%` top-1 flips. Treating the entire donor token as
the circuit would repeat the complement-DAS mistake at the input boundary.

The first recursively runnable graph therefore inserted the cue, allowed only the frozen top20 to
respond, base-clamped every other full component, read the resulting weight coordinates, and
actuated only the eight compiled responses in an independent base run. It is a valid, cross-fit
stable null: minimum generated-coordinate projection `.387`, final target projection `.325`, and
worst residual `.481`. Controls are small in margin/KL but have one flip among 16. The donor-ranked
pool is not closed under live source-driven recurrence. Next work must locate its dependencies by
layer-band add-back, not enlarge the global list blindly.

## DAS and regularization verdict

The user's diagnosis remains supported with a qualification. Full-vocabulary KL materially repairs
the constrained DAS objective (`.4485 -> .2445` held-out error, near DIM's `.2385`), whereas tangent
noise alone leaves it near `.4486`. Yet row-held-out KL/noise optimization still fails complete
construction families. Complement inertness cannot justify the learned subspace by itself because
the optimizer can select a task-specific coordinate system that satisfies that same test. The
current circuit line avoids this loophole by requiring complete-family/cross-panel transfer, OOD
corpora, full-vocabulary controls, noise tests, literal weight pullbacks, and source-driven execution.

## Circuit breadth and concurrent route work

The broad program still has 13 behaviors passing the complete row-2/3/4/5 Tier-3 screen; these are
behavioral circuit candidates, not 13 finished transparent programs. Concurrent Tier-5 work now
shows that early carriers feed mid-MLP near-token writes on the same reader axis (v133, 5/5), while
cue information can travel through opposed embedding and written-value routes (v136). Downstream
margin readout usually selects the written route for lexical/perfect number but the embedding route
for quantifier number (v137/v139). The most recent column-routing v142 receipt is instrument-invalid
and must not be used. These results reinforce typed routes rather than a single universal circuit.

## Next highest-information work

1. Run frozen layer-band add-back around the source-clamped top20 graph to identify which non-pool
   dependencies restore `L11H3`, `MLP6`, and the L8/L9 coordinates. Score all eight coordinates and
   controls in the same forwards.
2. Within the winning band, use coordinate-coverage greedy selection, not aggregate final-logit
   greediness. Require recursive source-driven sufficiency and control selectivity before promotion.
3. Preserve the frozen cross-fit response projectors and rank-64 MLP input readers; do not reopen DAS
   optimization unless a new complete-family objective can beat these simpler geometric baselines.
4. Continue counting broad circuits by evidence tier. A route mask, donor-patch atlas, and runnable
   transparent program are distinct deliverables.

`CIRCUIT_FOCUS: PASS` — every main-line unit changed causal sufficiency, weight compilation, upstream
writer localization, or recursive source closure.

`NOVELTY_LESSON_GATE: PASS` — the next test is dependency closure by layer add-back, not another rank,
penalty, or complement-loss sweep.

`CEREMONY_BUDGET: FAIL` — one avoidable GPU OOM came from retaining 110 patched input trajectories,
and setup/SVD reconstruction is duplicated across several long runners. Before the next GPU
experiment, factor a shared compiled-program context/helper so band add-back does not duplicate this
setup or recompute fixed effective maps per arm.
