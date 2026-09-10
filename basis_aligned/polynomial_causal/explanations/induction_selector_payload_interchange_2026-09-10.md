# Selector–payload interchange: strong transfer, unresolved circuit boundaries

The latest test finds a substantial part of the copying behavior, but it does
not identify the reusable circuit we want. Changing the source-selection part
recovers 86–91% of the model's mean answer-margin change; changing the payload
part recovers 77–86%. Both pass the registered transfer tests. Changing both
while preserving the answer still changes the broader output too much.

The mathematical follow-up clarifies what that failure means. A shared
selector can serve a copying computation and another computation that uses
source context. Swapping the entire shared quantity can then affect both,
even when the copying answer stays the same. We need to identify these
consumers and their separate uses before calling the module a selective circuit.
This is directly aligned with the original handoff's proposal to discover
shared operations and split native modules by their uses.

The work proceeded through a numerical repair, a completed GPU experiment,
an audit of the saved observations, and an exact mathematical counterexample.
The preceding value-producer experiment is described separately in
[Induction value producers](induction_value_producers_2026-09-10.md).
Neither experiment returns us to the is/was circuit.

Induction here means finding an earlier occurrence of the query token and
copying the token that followed it. A **selector** says how strongly each
earlier source is read; a **payload** is the vector carried from that source.
For a head, its equality-supported contribution is

    B(E,U) = sum over source positions r of E_r * U_r.

E retains the native continuous attention coefficient on the registered
equality-supported sources. U is the native value after its output weights
are folded in. The underlying attention still multiplies two QK dot products;
query/key normalization, positional rotations and shared-first-layer value
mixing remain intact. Folding output weights into U exposes an exact
contraction; it does not independently explain how E or U are produced.

At four fixed heads, we add one of three differences to a recipient run:

- Selector: `B(E_donor,U_recipient) - B(E_recipient,U_recipient)`.
- Payload: `B(E_recipient,U_donor) - B(E_recipient,U_recipient)`.
- Joint: `B(E_donor,U_donor) - B(E_recipient,U_recipient)`.

All later layers recompute. These are partial attention-output edits; they
do not replace the complete internal query/key/value states. The experiment
uses the four zero-index heads L5H5, L7H3, L8H3 and L8H4. The two layer 8
contributions are added together as one physical layer transaction.

The GPU run completed 639 batches, containing 20,448 sequence evaluations,
in 97.05 seconds including wrapper work. It used 72 FIT groups—the initial
discovery/evaluation partition. The SELECT, FINAL and out-of-distribution
partitions stayed closed because FIT failed. No weights were trained.

| Measurement | Result | Meaning |
|---|---|---|
| Selector mean margin recovery | 85.85–91.07% | Intended transfer passes |
| Payload mean margin recovery | 77.14–86.24% | Intended transfer passes |
| Answer-preserving joint changes |All 4 cells fail | Combined edit is insufficiently invariant |
| Filler-change joint controls | 4 cells fail | Some unrelated context changes cause excess loss |
| Active control coverage |All 24 keys insufficient | Only one control family is adequately active per key; two are required |

An **answer margin** is the donor-answer logit minus the recipient-answer
logit. Recovery divides the mean intervention-induced change by the mean
native donor-versus-recipient change; it is not accuracy or percentage of
the full model explained. Separate median, signed-effect and loss tests also
had to pass. **Cross-entropy damage** is the increase in negative log
probability of the correct answer; positive values are worse, measured in nats.

The four joint cells have mean cross-entropy damage 0.134, 0.231, 0.101 and 0.344
nats, all above the unchanged 0.10 limit. Their median root-mean-square changes
across vocabulary logits are 0.197–0.255, also above their registered limits.
The filler controls fail their loss criterion even though their margin and
vocabulary-change criteria pass. Inadequately active controls cannot establish
selectivity simply by having small effects.

The earlier version could not validly answer this scientific question because
its numerical check mishandled floating-point addition. Adding a small vector
to a large FP32 value rounds the result; subtracting the original value need
not recover the requested addition exactly. The repaired version records the
actual before/after states and checks the correctly rounded sum. An audit
verified 44,928 physical transactions and all 35 raw-file hashes. All rounded
post-states match exactly. The former absolute check would reject 17,287 valid
transactions. No scientific threshold changed. A first wrapper attempt was
dry-only because an imported legacy module changed the run-mode environment;
capturing the requested mode before imports fixed that separate execution issue.

The mathematical counterexample is small enough to inspect. Let s select one
of two sources, and p swap their payload assignment. A perfectly factorized
module can write

    y = (1-s)*p + s*(1-p),    c = s.

y is the copied answer; c is selector context used by another consumer.
Flipping both s and p leaves y unchanged but changes c. Thus failure of a
full-output invariance test does not prove an absence of shared computation.
It does reject treating this entire write as an isolated answer-only circuit.
Moreover, y alone cannot reconstruct c: two inputs with the same y have
different c. An executable explanation must retain and explain that information.

We checked the example exactly across 48 factor interchanges and 64
consumer-port cases. A **consumer port** means the input supplied to one
particular downstream computation. Editing a shared selector producer changes
every consumer; editing its copying port can preserve the context consumer.
Any later computation using both outputs must still recompute. This
distinction is consistent with formal causal abstraction, which requires an
explicit correspondence between interventions in detailed and simplified
models. [Massidda et al.](https://arxiv.org/html/2211.12270v1)

This example is a logical counterexample, not evidence that the trained heads
have that particular structure. Our registered result remains
`factorization_not_identified`. The useful next hypothesis must explain a
second consumer and predict separate and joint intervention effects with one
shared, independently generated computation. Merely discarding output
directions until the original answer test passes would not establish this.

All 545,902,902 native parameters remain necessary to the current executor;
there is no independently extracted circuit, OOD success or structural saving
from this test. The raw 6.56 GB evidence is temporarily in
`/dev/shm/bilin18_induction_r594` and is lost on instance restart. Exact compact
results, receipts and audit summaries are retained in the repository.

Evidence: [registered result](../induction_centered_fixed_geometry_rung594_results.json),
[saved-observation audit](../INDUCTION_R594_SAVED_AUDIT_V1_RESULT.json),
[exact counterexample](../SHARED_SELECTOR_INTERVENTION_COUNTEREXAMPLE_V1_RESULT.json),
and [mathematical review](../THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-10_1049.md).
