# Live equality-contribution clamp: necessary invariance screen

Scientific question: does replacing the live contribution, rather than adding
a frozen baseline difference, resolve the failed answer-preserving joint and
filler-control conditions? R594 correctly implements its own centered edits;
this is a different intervention semantics, not an instrument repair or rescore.

At each fixed native site L5H5/L7H3/L8H3/L8H4, let C_live be the equality-supported
output contribution computed from the current state after earlier interventions.
Let C_d be the frozen donor contribution from the R594 native endpoint capture.
Apply `write_new = write_live + (C_d - C_live)`, combining both L8 changes
before one FP32 addition. All non-equality contributions and downstream states
remain live. The old operation adds `C_d-C_native_recipient` instead. Their
local difference is `C_native_recipient-C_live`. Full-write feedback measured
in the saved audit motivates the test but is not assumed to equal that difference.

Use the same864 FIT directed rows selected by INDUCTION_CONTEXT_TRANSPORT_V1_PLAN:
288 answer-preserving joint rows and576 filler rows, each cell containing all72
groups. No changed-answer rows or held-out partitions are opened in this gate.
Physical batch32, width30,27 batches per arm. Three arms: native, self-donor
clamp, actual-donor clamp.81 forwards/2592 sequences, zero fitting. All native
weights, source features and context remain charged. Native donor vectors are
interchange inputs, not an independently extracted feature generator.

A, instrument: exact count81; finite outputs; native versus saved R594 replay
and self-donor versus native max absolute<=1e-3 and relative Frobenius<=1e-5.
The same live factorizer used by R594 must reproduce its cached recipient
contributions on the first selected layer to relative1e-5. Every physical
FP32 addition must equal the correctly rounded sum. Self-donor changes must
remain within numerical replay bars. Each later site records the CURRENT
contribution separately; measure its difference from the native recipient.
The live-vs-frozen correction must be nonzero above1e-5 relative on at least
one later layer in at least75% of joint rows, or label the semantics comparison
insufficiently active. No claimed observed per-head addition when two heads
share one layer transaction.

B, necessary joint invariance: all four answer-preserving cells must have
mean correct-answer CE damage<=.10 nat AND median uncentered vocabulary RMS
change<=.25 times the SAME frozen R594 per-cell vocabulary scale. This keeps
the prior necessary bars. This small screen does not replay every R585
diagonal predicate, so B cannot promote the full factorization by itself.

C, necessary filler control: all eight directional filler cells must satisfy
mean CE damage<=.10, median absolute correct-margin change<=.25 times the
frozen R594 margin scale, median vocabulary RMS<=.25 times its vocabulary
scale, and correct-versus-other fraction>=.75. Existing insufficient active
control-family coverage remains unresolved even if C passes.

Opposing predictions: if stale baseline additions explain the failed gate,
live replacement passes B and C with materially active corrections. If it
does not, preserve the null and stop this semantics candidate; do not scan
heads, gains or ranks. If A/B/C hold, run a separately registered complete
factor interchange and control-coverage test before any identification claim.

Execution: shared R594 factor capture and managed runner only. Bind this plan,
new runner, saved compact results/receipt and existing factor/runtime sources.
Recheck hashes of the five consumed raw files and all derived row identities.
Use a new RAM-backed namespace with >=1GiB free; save native/self/live endpoint
logits (~522MB total), their hashes and compact per-row results. The raw output
is volatile; compact result is written to the repository. No old data deleted.
