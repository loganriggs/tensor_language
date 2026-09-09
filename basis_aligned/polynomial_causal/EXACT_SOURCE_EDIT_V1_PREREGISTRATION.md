# Exact source-edit executor v1

The trained source-support interaction law passes all four gates, including OOD
withheld triple prediction and overlapping-state/nonlinear-probability controls.
Implement its shared reader/source dependencies directly instead of storing a
lookup table of native intervention outputs.

Program contains the original prefix and final Q1/K1/Q2/K2/V projections, with
final O/head folded using the already-known generic identity. Physically delete
final O. Literal independent constants387968 versus400640 native; all12672 saving
is generic, not a newly discovered semantic reduction. Runtime cache contains
post-prefix states, normalized projected Q/K/V, base source aggregate and logits,
all computed from input tokens by the program itself. No external teacher traces
or response table. Count cache bytes and temporary source/query contractions.

For edited source positions S, recompute native RMS and projected features only
at S, retaining native absolute-position rotary tables. Update source messages to
all unchanged queries; for queries in S, recompute their complete source sum with
all edited sources installed. Apply the folded readout and residual correction.
Overlapping component edits are summed before normalization and processed exactly;
never truncate their higher-order interaction using the disjoint-support theorem.

CPU controls must cover nonzero edits at multiple noncontiguous positions,
empty edit, overlapping edits, exact absolute RoPE, native all-logit closure1e-9
andrelativeRMS1e-10 (floor1e-6), plus absence of final O weights. No new fit.

Trained validation to implement next: fresh three-chain IID24-cycle/three8-cycle
worlds21909/21910,16worlds each, same generator algorithm as SOURCE_SUPPORT_V1;
all8 native disjoint join removals plus all8 overlapping E/Y0/Y1 removals and
one joint half-dose edit. Candidate produces its own native writes from its
retained prefix weights. Compare all29 logits/allpositions to independent native
full forwards or native suffix interventions, using the exact bars above and
full-distribution1e-3mean/1e-2p99 as additional diagnostics. Original checkpoints
remain untouched. Export candidate state_dict/config and replay in a fresh CPU
process without original checkpoint access before claiming portable execution.

Price preparation and repeated edit separately. ForT51 andK<=6, report affected
projection rows, contraction dimensions, measured FP64 warmed timing with device
synchronization, cache/temporary bytes and exact constant count. Do not claim
latency wins solely from operation counts or compare only to an exponential
response table: include full native forward and cached native suffix baselines.
B4FP64,1800s,<256MiB/tensor,managedGPU. A fidelity failure repairs a concrete
implementation issue; this exact executor does not replace the rejected semantic
kernel or establish a smaller learned circuit program by itself.
