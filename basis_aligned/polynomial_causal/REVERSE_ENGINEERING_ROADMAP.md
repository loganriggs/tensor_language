# Roadmap to an executable reverse engineering of bilin18

## Operational end state

“Fully reverse engineered” does not mean assigning an English phrase to every
parameter. The target is a smaller, typed tensor program `P` with an explicit map to
the original model `F` such that:

1. `P` reproduces natural and OOD behavior at a declared distortion budget;
2. its response to held-out interventions predicts `F`'s response;
3. independently described fragments compose without large interaction drift;
4. target fragments can be removed, transplanted, or edited with predicted collateral;
5. its complexity is reported conditionally on a frozen grammar and decoder, with
   gauge-equivalent programs priced identically within tolerance;
6. every claimed exact simplification has a certificate, and heuristic compiler
   outputs are labeled as upper bounds.

Provisional whole-program gates are global delta-CE at most `0.02` on held-out
FineWeb, at most `0.05` on a second corpus, no powered behavior class losing more
than 10% of its clean advantage, normalized held-out intervention-response error at
most 25%, and no more than 1% price drift under registered gauges and rewrites. These
are adoption targets, not claims about the present model.

## Progress accounting

Maintain a circuit balance sheet rather than counting discoveries. Every candidate
fragment records:

- typed input and output interfaces, including precision and norm semantics;
- standalone and amortized description bits, product/FLOP cost, and interface size;
- natural/OOD replacement distortion;
- intervention families used for fitting versus held out for testing;
- selective-removal effect and collateral;
- overlap/shared dependencies with other fragments;
- composition error when installed with the current best partial program;
- the residual effect not explained by the fragment.

Replacement recovery is scored relative to an explicit null at the same site. The
primary dashboard is cumulative whole-program recovery and the remaining CE/KL and
causal-response residual, not the sum of overlapping ablation effects.

## Ranked workstreams

### P0 — Build the whole-model coverage and composition ledger

Inventory the already certified heads, MLP slices, tables, and shared services. Put
each through one common replacement harness and assemble the strongest non-duplicated
composite. Measure where its residual CE lives by layer, token class, output slice,
and intervention family.

**Why first:** this converts thousands of local facts into a map of what is actually
missing. Without it, another named circuit may duplicate an existing service or
explain no additional model behavior.

**First synthesis (2026-08-27):** `whole_model_balance_sheet.py` now assembles the
frozen Theseus anchor, current ship registry, causal coverage ledger, named-variable
draws, and composition diagnostics without flattening their denominators. The
current fidelity ship replaces 36/36 top-level modules but is still about +0.93 CE
above the 2.9455 clean anchor. Separately, analytic-interface substitution recovers
0.9982 of its own mean-floor denominator, named token+topic+previous variables score
0.321, named causal paths cover 0.1092 of global ablation headroom, and the older
all-stand-in composition stress test recovers 0.124. These facts are complementary,
not candidates for one averaged “percent understood.” The next P0 increment is
residual localization and a manifest-complete composite record, not a new registry.

### P1 — Trace typed causal interfaces backward from outputs

For each behaviorally important output slice, identify the minimal residual channel,
its writers, its computational readers, and its final readout. Represent the result
as a small directed tensor program with shared nodes. Use both class-seeded slices
and behavior-agnostic output bases, but evaluate discovery on disjoint classes.

**Why second:** output-to-input tracing supplies stable boundary variables for
replacement and intervention transport. Internal coordinates without a boundary are
gauge-dependent and do not compose.

### P2 — Compile exact polynomial fragments between norm boundaries

Exploit the literal bilinear MLP and attention structure. Canonicalize linear maps,
real scalar quadratics by inertia, and registered CP/tensor gauges. Next, develop
joint vector-valued quadratic factorizations and shared projection dictionaries
across output slices. Treat RMSNorm and softcap as explicit analytic primitives, or
freeze/approximate them on a certified domain.

**Why third:** algebra can give exact reductions that activation-only compression
misses. The question form already demonstrates this: spectral interface rank 2 but
exact multiplicative complexity 1. The harder payoff is shared vector-valued forms,
where one factor library may serve many outputs and circuits.

**Scope correction (2026-08-27):** the full-vector output flattening has numerical
rank 1152/1152 at registered relative tolerances `1e-4` through `1e-6` for audited
MLPs 0,1,2,11,17, against 4608 native products. This rules out near-exact coefficient
degeneracy at those tolerances, but does not certify symbolic rank, locate the useful
approximation knee, or measure natural-activation fidelity. In contrast, the selected
question scalar has an exact one-product algebraic certificate. The compiler target
must be a jointly discovered causally sufficient output/content API, not the whole
1152-output tensor by default; every claimed saving must name whether its error is in
coefficient, activation, or causal-behavior currency.

**Matched-cost causal result (2026-08-27):** the question eigenpair's exact paired
gate is numerically exact and bf16-stable, but the best one-square gate failed the
registered held-out causal-separation test. It makes `35.4%` scalar error while
incurring only `6.87e-5` question KL, `0.39%` of the zero-slice KL. Thus exact
algebraic simplification does not by itself identify behaviorally necessary geometry.
Do not extend the scalar direction merely because its identity is elegant; move to
the joint content/ship-residual frontier where causal value can be earned.

**Early content frontier (2026-08-27):** 32 learned paired products consistently
beat selected native and random products, but lose to a parameter-matched linear map
at every MLP0-2 content slice. Held-out paired versus linear `R2` is `.542/.639`,
`.454/.589`, and `.295/.465`. The registered tensor-specific gates fail, so these
factors are not installed. The content API is locally linearizable; whether a linear
correction removes unique ship damage is now the live question.

### P3 — Synthesize replacements jointly, not one module at a time

Fit tensor-program fragments against natural outputs plus selected intervention
families, price shared factors once, and install them into the current composite.
Optimize the residual after every addition. Include explicit transport maps only when
intervening on latent variables rather than common module boundaries.

**Why fourth:** local low reconstruction error routinely fails under composition.
The desired object is a globally useful program, not a folder of individually good
approximations.

### P4 — Attack the distributed middle only after its residual is localized

The early/late and narrow output-channel mechanisms are relatively tractable; the
contextual middle MLPs remain the likely irreducible wall. Once P0 localizes their
unique residual, test data-supported lifted subspaces, shared bilinear dictionaries,
conditional/routed fragments, and low-degree interventional response models. Require
held-out compositional gain over rank, parameter-count, and compressed-byte baselines.

**Why fifth:** this is probably the hardest and most expensive work. It should be
conditioned on exactly what the existing composite fails to reproduce, rather than
another broad search over 64M parameters.

### P5 — Validate extraction, editing, and generalization

For every mature fragment, predeclare transplant/removal edits and predict both target
damage and collateral. Test fresh rows, a second corpus, disjoint token classes,
unseen intervention families, pairwise composition, and all-module composition.

**Why sixth:** these are the practical benefits of reverse engineering and the final
defense against a merely descriptive compression.

## Immediate queue after the matched-product result

1. Run the complete `attention x MLP0-2 x deep` replacement factorial and score exact
   Mobius interactions plus Shapley allocations by token cell, output slice, and
   intervention family. The existing novel/rare and early-MLP headlines are different
   marginals and must not be multiplied into a causal story.
2. If MLP0-2 owns at least `.05` novel/rare nats and 20% of that cell under the
   factorial, fit a *linear* current-ship residual correction first; otherwise split
   frequent construction from novel/rare content and target the licensed group.
3. Move paired-product compilation to a deep content boundary only if that boundary
   has unique factorial residual and it beats the same matched linear/native controls.
4. Use the behavior-agnostic output basis only as a locator. Its rank-8 basis retained
   exactly half of oracle recall but only 13.5% of oracle causal damage, so it is not
   presently a control interface.
5. Rebuild and rescore the best whole-model program after every admitted correction;
   the remaining ship residual, not local reconstruction, chooses the next target.

## Pruning rules

Stop or demote a direction when:

- it does not improve held-out prediction beyond simple rank/bits/FLOP baselines;
- its evaluation distribution is badly OOD for the claim being made;
- it explains a local reconstruction but adds no composite replacement recovery;
- it re-discovers a service already priced in the shared library;
- its score changes materially under gauge-equivalent rewrites;
- it needs the same intervention outcomes later claimed as predictions;
- it lacks a certificate but is being described as a minimum;
- its expected information gain per GPU-hour is lower than measuring the current
  composite's largest residual.

The prefix/continuation Hankel route is currently demoted by these rules: splice CE
is roughly 3.5 nats/token above natural, rank-95 is 23–24 of 48, and low-rank
completion improves only 4.5–10.1% against a registered 30% bar.

## Hourly strategic review

The local session receives an hourly prompt from
`hourly_strategic_review.sh`. Each tick inspects new evidence, brainstorms candidate
actions, prunes them by the rules above, ranks the top five, and executes the highest
priority safe unblocked action. The cron is session-local and must be recreated after
a container/session recycle; this file and the script are the durable policy.

## Strategic checkpoint — 2026-08-27 20:16 UTC

The balance sheet still has several deliberately non-combinable currencies. Exact
top-level replacement inventory is `36/36`; named-variable behavior is
`32.1% +/- 6.4%`; named causal paths cover `0.57968 / 5.30682 = 10.92%` of the
registered causal headroom; the legacy all-stand-in composition recovers `12.4%` on
its own mean-ablation denominator. Analytic interface substitution reaches
`99.8162%` only against its separate `18.4185`-nat joint-MLP mean floor. On the live
paired operational currency, clean CE `2.9455` becomes ship CE `3.8431`, leaving
`+0.8976` nat/token, and no candidate yet has an admitted common-denominator
executable numerator. Thus certified executable whole-program recovery remains zero.

The largest gaps are now: no end-to-end executable MLP0/1 numerical compiler; no
admitted MLP2 interface despite strong suffix attenuation of C512 mismatch; no
content-routed attention/value program; twelve middle MLPs whose singleton stakes
are too small/noisy for the current instrument; strong early/middle composition
interactions; and no whole-program second-corpus OOD or selective-edit certificate.
The new attention exempt-one result localizes the lag-1 shortfall broadly rather than
to an early-only subsystem: `attn5/6/7` each restore `22.5--26.5%` of the shortfall,
while the mean exemption gain is `5.12` points for `attn0--3` and `6.60` for
`attn4--17`. These shares are nonadditive. More fixed lags are therefore demoted;
the remaining target is content-routed value/routing structure.

After pruning by expected information gain, causal relevance, whole-model
composability, falsifiability, GPU cost, and overlap with completed work, the current
top five are:

1. Complete the source-closed compiler-v2.1 CUDA stages and exact common-support
   final scorer. This is the nearest path to an admitted executable early-MLP
   numerator and directly tests autoregressive MLP0-to-MLP1 composition. Its claim is
   narrow: because Q0 is selected greedily before Q1, failure falsifies the ordered
   search pipeline, not joint A--E expressivity.
2. Mint the same-row current-ship macro denominator and score admitted early-MLP,
   attention, and deep arms on one integer support ledger. This converts the decisive
   `+0.8976` gap into additive/Mobius allocations and prevents denominator mixing.
3. Promote the replicated rank-128 all-MLP program to authority grade with fresh
   rows, poison/call guards, document bootstrap, current-ship currency, and OOD. It
   retains about `54.0%` of the all-MLP constant stake at `5,308,416` reals, but has
   no present claim because its rows and denominator are exploratory.
4. Compile the full attention interface, targeting content-routed values/routing
   rather than further positional-lag widening. The live value-simplification job is
   the cheapest discriminator of whether fixed real routing plus simple values is a
   composable half-program.
5. Cross the first admitted MLP and attention packages in a same-row `2 x 2`
   composition, then test a second corpus and preregistered removal/edit collateral.
   This is required before either local success can be called a whole-model circuit.

The highest-priority safe CPU action during the attention GPU run was executed. The
v2.1 lifecycle was audited and found to be a nonnumerical scaffold, not a launchable
runner. Before any validation forward, it was hardened so every selector metric must
recompute from serialized float64 sums and integer counts; site1 requires a separate
last-written site0 authorization after hook restoration and outer model return; and
the whole pipeline requires a create-only owned run lock. Fifty-nine focused tests
pass. This creates no metric, winner, final access, or recovery credit. The remaining
implementation boundary is the actual CUDA capture/scoring orchestration and the
one-support, one-bootstrap final evaluator.

Pruned for now: widening MLP0's regressors (its `[a0; x0]` input is information
complete), another hard token clustering pass, more fixed attention lags, identical
compiler-v2 retry, importing the token-table `86%` number across denominators, and
adding a table/direct-rank arm post hoc to frozen v2.1. A token table with explicit
unseen-token fallback and a matched-price direct-rank comparator remain high-value
prospective controls for the next jointly priced compiler, not amendments to this
confirmatory run.

## Strategic checkpoint — 2026-08-27 20:40 UTC

No whole-model accounting currency changed: common-current-ship executable recovery
is still zero and the paired ship residual is still `+0.8976` nat/token. The material
new evidence is attention-side and changes the interpretation, not the global
fraction. With real routing left intact, the committed heuristic value-rank curve is
`-26.10%, 2.37%, 67.05%, 100.01%` at ranks `8,64,256,1152` on the separate
`3.55704`-nat attention-output-write constant stake. Therefore “simple values, hard
routing” is falsified for that fitted family at ranks 8--64. It does not prove an
intrinsic value rank: the fit truncates an ordinary SVD of the ridge map rather than
solving covariance-optimal reduced-rank regression, optimizes `c_v` MSE rather than
suffix/CE loss, includes the `v1` path, and is neither nested nor price-matched with
the fixed-lag programs. The prior phrase “the remaining 29.9% is content routing” is
withdrawn. The licensed statement is only “29.9% lies outside the tested fixed-lag
linear family.” A denser exploratory curve and routing-rank follow-up are running in
Claude's lane; they are not yet authority.

The pruned priority order is now:

1. Finish the complete compiler-v2.1 numerical source closure before any validation
   forward: site0, site1, bundle/strata freeze, and common-support final scorer. The
   current ship factorial gives MLP0--2 signed Shapley `+0.7277` nats versus attention
   `-0.0670` and deep `+0.2120`, so early executable compilation remains the strongest
   evidence-backed numerator target.
2. Refresh the clean/current-ship `attention x MLP0--2 x deep` restoration cube on
   one integer support ledger and one source-document bootstrap, then insert any
   admitted compiled-early arm. This decides incremental value on the actual
   `+0.8976` gap rather than on standalone constant stakes.
3. Only if attention owns positive conditional current-ship residual, run a
   middle-band routing-by-value `2 x 2` using exact versus frozen/simple routing and
   exact versus correctly covariance- or suffix-weighted reduced-rank values, with
   explicit `v1` semantics and a shared price ledger.
4. If greedy v2.1 fails while exact early restoration stays valuable, preregister a
   joint early-state/write compiler with a shared latent, table and matched-price
   direct-rank controls, and an optional MLP2 interface. If it succeeds, test its
   composition before expanding the grammar.
5. After the first admitted executable numerator, run second-corpus/code OOD,
   held-out interventions, and selective removal/edit collateral.

This tick executed the next safe implementation step rather than launching an
incomplete experiment. A site0 numerical runner now performs the registered OON
teacher/NON baseline/QON candidate routing; scores both 108-cell banks plus a scored
mean and full-native control from raw float64 sums and integer support counts; freezes
and reloads both ledgers before selection; and makes the site1 receipt the last
fallible write after outer return and hook/component restoration. A source-identical
`resume_after_site0` boundary was added. Mocked tests cover routing, common support,
complete-bank-before-selector ordering, failed full-native gates, no-final access,
and source-equal resume; the combined focused suite is `70 passed`.

The runner is deliberately **not launched**. Site0 would freeze its source commit and
hashes, while site1/program/final numerical sources do not exist in the closure yet.
Observing site0 and then writing those sources would make implementation
outcome-adaptive; changing them would also fail the source-identical resume gate. The
next safe CPU task is therefore site1 plus bundle/strata/final implementation and
tests. More fixed-lag sweeps, ordinary-SVD value-rank interpretation, standalone
rank-128 promotion, new token clustering, and another MLP2 alignment factorial stay
pruned.

## Strategic checkpoint — 2026-08-27 22:05 UTC

The global certified fractions have not moved. Inventory remains `36/36`, the
named-variable behavioral account remains `32.1% +/- 6.4%`, named causal headroom
remains `10.92%`, and the common-current-ship executable numerator remains zero
against the paired `+0.8976` nat/token residual. The compiler-v2.1 work in this
interval is source closure, not recovery evidence: no v2.1 validation/final role has
been loaded and every authoritative output namespace is still absent.

The strongest new structural evidence is a four-way matched rank comparison. At
rank 64/1152, attention routing retains `62.82%` of its separate stake, attention
values `2.37%`, MLP Down `-15.16%`, and MLP Left/Right `-52.88%`; every full-rank
identity check is exact and every curve is monotone. Thus low-rank routing is a real
compression opportunity, but low-rank feature formation is not. This supports a
typed program whose routing and nonlinear feature/value components have different
grammars, not one uniform low-rank factorization. A new interleaved
whole-model-program source has also been committed to test the known composition
problem, but it has not produced an artifact and carries no result yet.

The largest unresolved interfaces are: executable MLP0-to-MLP1 composition; MLP2's
conditional dependence on the upstream state; the attention routing-to-value
interface; cross-half composition on one current-ship denominator; and OOD/edit
collateral. Compiler-v2.1 now implements the complete ordered MLP0/1 experiment,
but adversarial review found that its one-shot final transaction and registered
description-length ledger had to be completed before launch. This interval added a
semantic RESULT-to-MANIFEST-to-OUTCOME validator, fail-closed sparse collateral,
parent-v2 protocol pinning, constant-rank diagnostic handling, physical/native
gauge canaries, exact original-call counters, and planned reports for standalone and
amortized bits, native index encoding, metadata, artifact bytes, operations,
runtime/memory, conditioning, precision replay, quantization, search budget, and the
full-original comparator.

After pruning by information gain, causal relevance, composability, falsifiability,
GPU cost, and overlap, the current priority order is:

1. Finish adversarial transaction/report tests, independently re-audit the exact
   source closure, commit/push it, and only then run the frozen compiler-v2.1 stages.
2. Refresh the current-ship attention/early-MLP/deep cube on one integer support and
   insert the admitted compiler arm, converting local success into recovery of the
   actual `+0.8976` residual and exposing interactions.
3. Complete the interleaved bottom-up whole-program composition test. Jointly
   installing independently fitted halves is a specific, falsifiable distribution
   shift hypothesis, not another generic scaling sweep.
4. If attention retains positive conditional residual, compile routing and values
   with separate typed budgets: routing can use the observed low-rank family;
   values require a covariance/suffix-weighted or programmatic grammar.
5. After the first admitted common-denominator executable numerator, run the second
   corpus, code OOD, held-out intervention, and selective removal/edit certificate.

Executed this tick: the compiler closure was extended and `97` focused tests passed
before the newest transaction cases. Synthetic sparse cells now yield an
authoritative-negative package instead of consuming the final namespace through an
exception, and synthetic semantic-result corruption preserves RESULT in a failure
manifest while forbidding outcome authority. Launch remains blocked until the
enlarged full suite and fresh independent audit pass on committed source.

Post-checkpoint closure: the enlarged focused suite passes `105/105`. Independent
mathematical review and artifact/lifecycle review both give GO-to-commit; launch is
GO only after this exact snapshot is committed, pushed, and accepted by the
source-identity guard. The untouched-row rank replication also passed all three
registered checks, with rank-64 routing `63.00%`, values `2.288%`, Down `-14.423%`,
and Left/Right `-53.390%`, preserving the full ordering within `0.75` percentage
points. The first interleaved whole-program attempt emitted zero/negative stakes
and NaN ratios because its baseline was accidentally evaluated through active
constant hooks. A new baseline known-answer assertion caught the instrument error;
the corrected rerun gives MLP-only `60.814%`, attention-only `56.263%`, and joint
`50.939%` on a larger `5.56837`-nat 36-site joint constant stake. The halves therefore
compound instead of reproducing the old independent-fit collapse, but this remains a
separate ceiling currency—not recovery of the paired `+0.8976` current-ship residual.

## Strategic checkpoint — 2026-08-27 22:11 UTC

The newest 36-site experiment separates the two explanations left open by the
`50.939% -> 55.038%` best-family upgrade. On the same `5.56837`-nat joint constant
stake, upgrading attention alone gives `53.603%` (`+2.664` points), upgrading the
MLP0--2 family alone gives `52.249%` (`+1.310` points), and upgrading both gives
`55.038%` (`+4.099` points). The singleton gains sum to `3.974` points, slightly
*less* than the joint gain. The registered strong-redundancy prediction therefore
fails; the data favor error transport/compounding across approximate halves over
the hypothesis that both upgrades recover the same information. This makes
conditional, interleaved interfaces the main simplification target.

No global accounting fraction is upgraded by that result. The valid currencies
remain: replacement inventory `36/36`; named-variable behavior `32.1% +/- 6.4%`;
named causal paths `10.92%` of their mean-ablation headroom; current-ship executable
recovery credit zero against the paired `+0.8976` nat/token residual. The 36-site
`55.038%` is useful evidence about a separate constant-ablation stake, not a fraction
of that current-ship residual. The largest gaps are now executable MLP0-to-MLP1 state
transport, conditional MLP2 behavior, attention routing-to-value semantics, a shared
current-ship composition denominator, and OOD/edit certification.

After pruning by information gain, causal relevance, composability, falsifiability,
GPU cost, and duplication, the top five are:

1. Run the source-frozen autoregressive MLP0-to-MLP1 compiler. It directly tests the
   interface implicated by the new near-additive upgrade result and is the nearest
   path to an admitted executable early-MLP numerator.
2. Insert the first admitted early program into a same-row current-ship
   `attention x early-MLP x deep` cube with one integer support ledger and document
   bootstrap. This converts local fidelity into recovery of the actual `+0.8976`
   residual and exposes cross-group interactions.
3. If greedy MLP0-to-MLP1 compilation fails, preregister a jointly selected latent
   and conditional MLP2 interface. The failure would localize missing transported
   state; more isolated MLP0 clusters or Euclidean regressors would be redundant.
4. Compile attention with typed grammars: low-rank/content-dependent routing and a
   richer covariance- or suffix-weighted value program. Uniform low-rank treatment
   is ruled out by the replicated routing/value/MLP path ordering.
5. Once one program earns common-denominator recovery, require second-corpus and
   code OOD, held-out interventions, selective removal/edit collateral, and a
   gauge-minimized conditional-description-length ledger before calling it reverse
   engineered.

The highest-priority safe CPU action was executed while the upgrade-attribution run
owned the GPU. The pre-selection compiler abort is preserved and hash-pinned. A
prospective execution-only amendment now distinguishes the scientific source
closure from unrelated global repository movement: launch still requires
`HEAD==origin/main` and committed-clean registered source, while later transaction
boundaries inherit the launch commit and require exact source hashes, row authority,
protected snapshots, and the owned run lock. Candidate families, rows, selectors,
gates, and final seals are unchanged. The focused retry suite passes `98/98`, with a
new regression proving unrelated HEAD movement is accepted but source-hash drift
fails closed. Independent lifecycle and mathematical re-audits both give GO to
commit, and conditional GO to launch only after the exact snapshot is pushed and
accepted by source-identity preflight.

Post-launch update: compiler site0 entered its numerical stage from commit
`bd9a5820` at 22:24 UTC; validation-only whiteners and attention maps are built,
while selection, MLP1 authority, and final access remain absent. The intervening
attention-`v1` closure gives identical serialized CE `5.79570` for real and
covered-token-table `v1`, so the best 36-site program has no functionally live
`v1` dependency on its registered covered-token estimand. This does not increase
the `55.038%` ceiling or any global ledger. Unseen tokens still use native fallback
and originals still execute. Rank 8 loses only `0.287` points, falsifying the
registered `>=1`-point manipulation prediction, but no rank ladder, noninferiority
margin, factorized price, or all-token/real-write-background test exists; it is a
descriptive conditional approximation, not a certified eight-dimensional program.

## Strategic checkpoint — 2026-08-27 22:35 UTC

Compiler-v2.1 site0 is healthy and has scored both complete `108`-candidate banks:
the true validation bank and its matched shuffled-control bank. Selection and the
receipt transaction are still pending, so there is no MLP1 or final authority yet.
The global fractions remain unchanged: `36/36` replacement inventory,
`32.1% +/- 6.4%` named-variable behavior, `10.92%` named causal headroom, and zero
executable credit against the current-ship `+0.8976` nat/token residual. The separate
36-site constant-stake account remains `55.038%`; it is not a whole-model explained
fraction.

The largest gaps are interfaces, not another isolated MLP0 fit. MLP0 already has a
strong continuous low-rank description, including a 3.60x smaller C512 `Down` and a
rank-64 causal output subspace recovering about 79.9% of its local effect. What is
missing is an executable program that transports the right MLP0 state through MLP1
and into conditional MLP2 behavior. Attention routing is compressible while values
are not under the same low-rank grammar, and joint improvements are attenuated by
cross-half error transport. All-token behavior, code OOD, selective edits, and a
gauge-minimized physical price remain uncertified.

After pruning by expected information gain, causal relevance, whole-model
composability, falsifiability, GPU cost, and duplication, the top five are:

1. Validate the compiler-v2.1 site0 transaction and run its frozen site1 only if its
   training receipt authorizes it. This directly tests the MLP0-to-MLP1 state edge.
2. Run the amended four-arm joint 36-site replication on skip11000 with paired
   source-document uncertainty. This tests whether the `55.038%` account and its
   small factorial interaction travel together.
3. Insert an admitted compiler program into the one-support current-ship
   `attention x early-MLP x deep` cube, converting local fidelity into an executable
   numerator against the actual `+0.8976` residual.
4. If compilation fails, localize missing state with a joint latent plus conditional
   MLP2 interface; isolated token clustering and Euclidean MLP0 regressors are already
   saturated and composition-blind.
5. Compile attention with typed grammars—low-rank/content-dependent routing and a
   richer value program—then require second-corpus, code-OOD, intervention, edit,
   and gauge/description-length certificates for the composed program.

Executed during the GPU interval: the queued whole-model replication was dequeued
before first execution after independent audits found it non-authoritative. A frozen
prospective amendment now binds the complete committed source/statistics/model-loader
closure, exact offline model revision and weights, all row/constant/provenance hashes,
and source-document disjointness. It adds a 36-site identity known answer, the known
discovery stake, common finite support, non-vacuous singleton/joint/conditional-gain
gates, exact raw row sufficient statistics, a paired 2,000-draw source-document
bootstrap, interval equivalence and factorial-interaction gates, and atomic
create-only result/failure publication. The CPU-only statistics suite passes `3/3`.
The run remains dequeued until the amended committed source receives final audit GO.

Post-transaction update: site0 completed and the lifecycle loader independently
reproduced both frozen selectors. The true-data winner is `B_l5_r64`, a rank-64
state-complete affine program using `153,920` float parameters (`0.966%` of the
registered original-MLP parameter count). On site0 validation it recovers `66.07%`
of the teacher KL denominator (`0.03862 / 0.11380` KL remaining), with global CE
`3.65882`; its copy CE improves by `0.04514` rather than worsening. The matched
shuffle selector chooses native `E_k256` but recovers `-16.99%`, while the mean
control recovers `-151.34%`. This is strong validation separation for a compact
continuous MLP0 state program, not a final or whole-model result. The outer-return
receipt authorizes training at site1 only and forbids final scoring. Site1 has been
launched from the frozen source. The amended joint held-out source received final
mathematical and lifecycle GO at pushed commit `9ba2cb13`; it remains behind site1.

## Strategic checkpoint — 2026-08-27 23:11 UTC

Compiler-v2.1 completed with intact authority and an authoritative scientific
negative. MLP0 recovers `62.67%` of its local projected-teacher KL, conditional MLP1
recovers `43.55%`, and the pair recovers `33.69%`; the latter two miss the registered
50% rung. In common CE currency the executable pair gains `0.05914` against the
projected-oracle pair's `0.22658`, or `26.10%`, and the half-oracle interval is wholly
negative. Every other gate passes, including both ordered increments, joint over
singletons, true over mean/shuffle, label alignment, alternate MLP2 background, copy,
and all token-frequency collateral. The result is a real but insufficient
parent-conditional interface.

No global currency moves: inventory `36/36`; named behavior `32.1% +/- 6.4%`; named
causal headroom `10.92%`; current-ship executable recovery zero against `+0.8976`
nats; separate 36-site constant-stake ceiling `55.038%`. The main gaps are the
MLP0-to-MLP1 transported interface, a common current-ship denominator, typed
attention routing/value composition, conditional MLP2 after adequate upstream
transport, and OOD/edit certification.

Pruned priority is: (1) same-basis matched local-versus-suffix refits plus an explicit
gauge-invariant `B0 A B1^T` cross-map and teacher-response perturbation assay; (2)
only after its failure, an oracle residual rank curve and joint suffix-Fisher basis;
(3) the one-support current-ship macro cube; (4) typed attention compilation; (5)
conditional MLP2 followed by OOD/intervention/edit certification. Ordinary
ridge/native-K sweeps, semantic rotation of gauge-equivalent axes, isolated MLP0
clustering, and another MLP2 alignment factorial are pruned.

Executed this interval: the full negative transaction and program bundle were
committed, with the two large numerical artifacts tracked through Git LFS. The next
protocol's pure physical-map/gauge/intervention contract passes `9/9` tests. Independent review
rejected its first draft before any new rows were loaded because objective/data/
optimizer and intervention semantics were confounded. The revised draft now adds a
same-new-rows local-loss comparator, executable-code semantics, teacher-response
differences, true document-derangement nulls, dense-only price, and a complete
lifecycle. Mathematical and lifecycle reviewers now give GO to freeze the exact
protocol and pure contract. Numerical execution remains NO-GO until the complete
runner, source closure, and create-only artifact transaction are implemented, tested,
committed, pushed, and re-audited. Concurrently,
the hardened 36-site prospective composition replication completed `4/4` registered
predictions on skip11000 at `53.69% [52.92,54.39]`, versus
`55.04% [54.18,56.00]` on skip7000; this is replication of the joint program, not
broad OOD or current-ship recovery. A concurrent value-path floor also shows that the
entire native-v1 path contributes only `0.35` ceiling points `[0.30,0.40]` inside this
36-site program, while rank 8 retains only about one sixth of that small effect. This
prunes interpreting the earlier small rank-8 decrement as evidence that v1 is itself
eight-dimensional; the path is real but currently low leverage.

Implementation checkpoint: the frozen suffix-transport protocol now has CPU-only lifecycle
and statistics boundaries. Thirty focused tests cover create-only artifacts, exact source
sets and current drift, phase/lock-enforced role loading, canonical program/final-attempt
bindings, one-shot final deserialization, source-document bootstrap, pooled nonlinear
response metrics, both registered response modalities, exact 20-null recomputation, and
adversarial type/schema failures. The modules cannot import the model and cannot publish an
outcome authority. The next implementation unit is the registry-complete row freezer, then
one shared differentiable runtime/trainer for L/R/S/T; no numerical role is authorized yet.

The next whole-model shortfall localization leaves the global ledger unchanged but gives a
better allocation target. Exempt-one recompilation of the frozen 36-site program assigns the
largest total gains to the 12-site middle MLP and attention bands (`+12.515` and `+12.136`
ceiling points); after normalizing by site count, the four front MLPs are worst. These are
interaction-sensitive conditional gains rather than an additive decomposition, and their
intervals are row-clustered rather than source-document-clustered. Consequently they prioritize
front-MLP site localization and middle-band compiler work without increasing certified model
coverage.

The suffix-transport row boundary now has a prospective exact registry allowlist and pure
collision adjudicator. It covers canonical FineWeb, v3-basis, compiler-v2/v2.1, code-OOD,
frozen-ship, and 36-site-heldout bindings; checks document, dataset-index, full-row, and
prefix-32 identity across all candidate roles and the prior union; and emits only counts and
hashes for rejected candidates. The focused contract suite passes 54/54 on synthetic fixtures.
The actual canonical census and candidate harvest remain embargoed until these source bytes are
committed, pushed, and independently re-audited.
