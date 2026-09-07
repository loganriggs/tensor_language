# Hourly strategic review — 2026-09-07 10:14 UTC

## Circuit target and present frontier

The target remains a smaller transparent tensor program that is jointly predictive, composable,
manipulable, literally priced, and stably identified. A complete circuit must state what is read,
computed, and written; connect upstream writers to downstream consumers; predict held-out and OOD
effects; survive selective removals, swaps, reversals, and relevant perturbations; and split or merge
native modules according to their actual computation. Localization, reconstruction, variance, rank,
or a low scalar loss alone is not a complete circuit.

The temporal/is-was source-to-response line now has one deterministic complete-family pooled
projector over eight response sites and a frozen upstream support. This replaces split-specific
projector choice and is the strongest current stable tensor interface. It is still an interface plus
restricted native background, not a zero-native-call compiled model, so the overall goal remains
open.

## What changed since 09:14

The response-site reversion atlas and a separate fresh-family gain curve both rejected calibrated
gain as the stable solution. The one fresh paraphrase-control flip exists already near gain one and no
single-site gain reversion uniquely clears it; gains low enough to clear it under-recover the target.
Global gain tuning is closed rather than repeatedly optimized against the opened control.

Exact attention `W_V/W_O/OV` usage confirmed the MLP result: the tasks use the same native response
sites but different local weight functions. Complete-OV principal cosines are `.176-.182` at L8H1,
`.308-.324` at L9H1, `.569-.626` at L9H4, and `.620-.625` at L11H3. No shared attention OV function
clears `.8`; L11H3 output-only overlap near `.802` disappears once the complete value-to-output map
is included. Combined with MLP quadratic principal cosines below `.176`, this licenses shared
infrastructure plus task-typed computation, not an identical cross-task local circuit.

The first pooled reverse test was invalid because its fit and evaluation rows overlapped. It was
preserved and replaced by a row-disjoint temporal-v12/iswas-v11 OOD test. The repaired pooled reverse
program passed 5/5: coordinate projection is at least `.9465`, behavior recovery is `.8126/.8240`,
worst target residual `.03661`, median control KL `.000248`, and no control flips. Exact projector
hashes then replayed in the forward direction, which also passed 5/5 with behavior `.8144/.8275`,
worst residual `.03594`, median KL `.000263`, and zero flips. One hash-identified interface is
therefore bidirectional on disjoint clean OOD panels.

The same pooled interface then passed the preregistered 10%-cue-source-noise test across three seeds
and both directions. Across all six executions, minimum coordinate projection is `.9445`, minimum
behavior recovery `.8103`, worst target residual `.03758`, maximum paired direction gap `.00805`,
maximum seed range `.00444`, median control KL at most `.000284`, and zero flips. This is materially
stronger than the earlier split-specific noise result and argues against clean-panel memorization.

With reverse and noise stability closed, pruning reopened. All three frozen single deletions from
rank48 passed bidirectionally under the pooled interface. The declared order selected L1H3, producing
a rank47 program with minimum coordinate projection `.9425`, target recovery `.8108`, direction gap
`.00348`, and zero flips. The next physical greedy combination tested removing either L3H7 or L10H5
from that rank47 support; both scientific arms pass, with the declared order selecting L3H7 and a
46-site support. Its original result is formally invalid only because the reusable runner retained a
stale `len(base_support)==48` assertion. A hash-bound zero-GPU correction verifies the actual 47-to-46
transition, all one-token source tripwires, finite values, projector hashes, exact 31-forward price,
and all five corrected gates. No opened outcome was rerun.

The independent number/quantifier lane also moved from one semantic route to a cross-task subspace
test. Seven channel-8 family deltas are largely disjoint; the only material shared component is the
polarity/`neither` relation. Follow-up removal maps show that this shared activation component is read
by each family's own downstream readers (shared-versus-full reader-profile correlations `.847-.967`
in the reported correlative cells), not by one universal polarity reader. This is a concrete example
of the user's proposed decomposition: shared subspace does not imply shared downstream circuitry,
and contracted weights plus causal removals must identify shared and family-specific readers.

## DAS regularization verdict

The memorization diagnosis remains partly confirmed. Full-vocabulary KL changes the optimization
target and reduces held-out error from about `.4485` to `.2445`, near difference-in-means `.2385`;
tangent noise alone remains near `.4486`. But complete-family model selection rejects every learned
rotation and retains the pooled step-zero/DIM-like axis. Regularization is therefore a useful
anti-specialization constraint, not evidence that the learned complement axis is the transferable
mechanism. Complement-only DAS remains closed until a multi-environment objective with frozen
complete-family selection prospectively beats DIM on a genuinely unopened family.

## Confounds, invalids, and efficiency

From 09:14 through 10:14 the managed circuit lane produced 17 terminal receipts: nine deep
temporal/weight-interface receipts and eight broad number/quantifier receipts, approximately one
every 3.5 serial minutes. Shared pooled-projector, OOD-context, and generic greedy-deletion code cut
the latest tests to 16, 39, 39, and 31 forwards instead of rebuilding atlases. Candidate reduction
was evidence-driven: once three rank47 arms all passed, only two physical rank46 combinations were
tested.

Two deep artifacts failed closed. One pooled projector experiment used overlapping target rows and
was discarded as invalid before a row-disjoint repair. The rank46 result used the correct scientific
support but a stale support-count assertion; it was preserved and hash-bound corrected without a GPU
rerun. The queue also rejected a thin wrapper before GPU access because prediction keys were not
statically visible; explicit registration fixed the preflight. These are real ceremony defects, but
none changed or erased an opened scientific miss.

`CIRCUIT_FOCUS: PASS` — the hour advanced bidirectional manipulation, OOD and source-noise stability,
task-typed weight realization, source-graph simplicity, and an independent shared-subspace reader
map.

`CEREMONY_BUDGET: PASS WITH TWO FAIL-CLOSED CORRECTIONS` — scientific throughput remained high and
helpers reduced repeated setup, but overlap and a stale count assertion each invalidated an artifact.
Both were preserved; only the count-only defect received a zero-GPU hash-bound correction.

`NOVELTY_LESSON_GATE: PASS` — gain tuning stopped after fresh-control failure, support was pruned only
after bidirectional stability, individual removability was not assumed additive, and activation
subspace overlap was not called shared computation without reader evidence.

## Ranked next moves

1. **Finish the bounded greedy branch at rank45.** Starting from the corrected 46-site support, test
   deletion of the sole remaining frozen candidate L10H5 in both OOD directions. A pass freezes
   rank45; a miss freezes rank46 and closes this branch. Do not launch another full atlas.
2. **Re-run 10% bidirectional source noise at the final greedy boundary.** This checks that simplicity
   did not consume the robustness margin. A target, coordinate, direction-stability, or control miss
   restores the last noise-stable support.
3. **Translate the polarity/`neither` shared component through exact weight tensors.** Rank upstream
   writers and each family's readers with gauge-invariant contractions, then physically remove the
   predicted shared and task-specific edges. Shared activation with family-specific readers is the
   current hypothesis.
4. **Promote one independent broad circuit with complete-family OOD/removal evidence.** The noun-key,
   value-side, direct-route, channel, and reader results should be assembled into one falsifiable
   program rather than extended as disconnected screens.
5. **Keep DAS subordinate.** Reopen only for a prospective multi-environment optimizer whose unopened
   complete-family selection can beat DIM while retaining full-vocabulary and causal-complement
   selectivity.

The highest-information immediate action is the one-arm rank45 test followed by final-boundary noise.
It directly improves simplicity without destabilizing the now-licensed pooled tensor interface.
