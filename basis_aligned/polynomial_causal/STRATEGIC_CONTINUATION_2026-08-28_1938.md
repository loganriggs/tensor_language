# Strategic continuation — 2026-08-28 19:38 UTC

## UPDATE PART

Two completed discovery runs refine the composition story.

First, depth really matters for a single substituted site. Starting from the healthy
B0 arm at 64.8% gap recovery, adding one attention replacement leaves 27.4% at layer
1, 21.9% at layer 2, 44.9% at layer 3, 47.9% at layer 5, and 64.1--65.3% at layers
9, 13, and 17. One MLP replacement gives the same qualitative depth dependence.
Therefore the layer-1/2 target survives its strongest falsification attempt.

Second, individually cheap deep sites are not jointly cheap. Compiling the 20 sites
in layers 9--17 retains only 22.3%, versus a registered 50% bar. Twelve sites in
layers 13--17 retain 40.6%. There is no sharp suffix frontier: recovery falls roughly
as a dose response as more deep sites are added. The useful contrast is:

- the two layer-1 sites are nearly redundant failure routes (loss ratio 1.956);
- deep sites are mildly damaging alone but super-additive in groups (approximately
  1.8 times the sampled single-site loss rate, with that multiplier still noisy).

This changes the simplicity lesson. Counting replaced sites, summing local errors, or
pricing each replacement independently is not a valid whole-model simplicity metric.
A candidate program must price interaction terms or demonstrate composition directly.
The concurrent GPU lane is testing count versus contiguity with count-matched 12-site
sets; it is not duplicating the response-backend work.

## Work executed in this continuation

The paired-response backend now has these additional real execution pieces:

1. Final MLP0 edits are derived from frozen rows, assignments, geometry, selected
   teacher-only amplitude, program payload, and B0. Callers cannot choose a position,
   direction, amplitude, or residual-stream write.
2. Float64 semantic deltas, canonical float32 `[4,256,64]` code edits, and matching
   `[4,256,1152]` physical writes have separate hashes and mutation/replay guards.
3. Baseline, positive, and negative student forwards have distinct response-bound
   nonces. The prospectively frozen mapping is trace trials 0, 1, and 2; the outer
   identity additionally binds the actual edit and materialization.
4. A private exact O/O/N teacher forward runs exact MLP0 with the physical edit,
   continues the edited trajectory, captures the raw exact MLP1 write projected
   through B1, uses deployed MLP2 N, and retains post-softcap logits only internally.
5. A one-use student response consumer and adapter-private student forward now execute
   the same code edit through `StudentCorrectionHook`, consume the captured MLP1 code
   and logits, close the broker, and return only to the future triplet reducer.

The frozen semantics and remaining NO-GO boundary are recorded in
`EARLY_MLP_SUFFIX_TRANSPORT_V1_RESPONSE_EXECUTION_AMENDMENT.md`, whose SHA-256 is bound
into every response execution identity. The combined affected response, capability,
and observed-adapter suite passes 77/77, including end-to-end fake-model teacher and
student paths.

## Current priority order

1. Assemble these teacher/student primitives into one fail-closed 69-forward batch,
   reduce the 22 arms, and bind reductions to actual observed receipt triplets.
2. Add the ordered 48-batch accumulator and mandatory 144/3,168 call ledger.
3. Wire the response-run receipt into terminal final closure; only then can the final
   role be opened.
4. Use the response outcome plus the count/contiguity result to choose the next joint
   program structure. Do not fit another local compressor first.
5. After a composed program succeeds, test OOD transport, selective removal,
   collateral effects, executable cost, and prequential/MDL price.

Production final execution remains NO-GO. The final 68-action ledger remains 0/68 and
no new whole-model explained fraction is claimed.
