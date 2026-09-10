# Live replacement improves the answer effect but leaves the circuit unresolved

We found and tested a meaningful distinction in how interventions compose.
Replacing each layer's **current** equality contribution improves the
answer-preserving swaps substantially compared with adding a fixed difference
measured in the original model. It still changes the wider output distribution
too much, and the same filler controls fail. This candidate remains a null.

Two follow-up explanations also fail: the extra output change does not reproduce
the model's natural response to the donor context, and it does not carry the
donor's source-token identity in the expected direction. These results narrow
the search; they do not identify a second consumer or satisfy the four circuit
properties. We should stop refining this fixed induction interface.

This follows [the selector–payload experiment](induction_selector_payload_interchange_2026-09-10.md).
We first analyzed its saved full outputs, then measured feedback between edited
layers, ran a new live-replacement experiment on the GPU, and analyzed the
specific source-identity hypothesis. The new GPU experiment took 81 forward
batches, or 2,592 sequence evaluations, and 9.81 seconds inside the executor.
It used all 72 groups in the same FIT partition. No held-out partition was opened.

The key computational distinction is straightforward. Let C denote the part of
an attention head's output supported by a source whose preceding token matches
the query. Let W denote the whole attention output. The old operation is

    new output = current W + donor C - original recipient C.

But earlier edits change the input to later layers. Their current C can therefore
differ from the original recipient C. The new operation is

    new output = current W + donor C - current C.

This replaces the selected contribution while retaining the current remainder.
Every later computation still runs normally. It is a different intervention
definition, not a numerical repair of the old experiment: the old experiment
correctly implemented its registered fixed additions.

Saved states showed that feedback was large enough to warrant this test. For
the answer-preserving joint swaps, the median change in the whole attention
write before the layer-7 edit was 1.19–1.35 times the planned addition; at layer 8
it was 0.56–0.64 times. These are whole-write measurements, not measurements of
the equality contribution alone. The new experiment then measured the current
equality contribution directly and found an active correction in every joint row.

| Answer-preserving case | Fixed-addition loss damage | Live-replacement loss damage | Live vocabulary RMS / allowed limit |
|---|---:|---:|---:|
| B, s0p0 → donor | 0.134 | -0.012 | 0.192 / 0.138 |
| B, s1p1 → donor | 0.231 | 0.065 | 0.228 / 0.130 |
| D, s0p1 → donor | 0.101 | -0.002 | 0.233 / 0.130 |
| D, s1p0 → donor | 0.344 | 0.142 | 0.256 / 0.135 |

Here B and D identify the correct payload; s and p index selector and payload
assignment. **Loss damage** is the increase in negative log probability of the
correct answer, measured in nats; negative values improve that answer's loss.
The unchanged loss limit is 0.10 nat. **Vocabulary RMS** is the root mean square
change across all vocabulary logits. Its limits come from the original frozen
scales. Three loss tests now pass, but all four vocabulary tests still fail.
The four base-to-donor filler controls retain loss damage 0.139–0.174 nats,
above the same 0.10 limit. Their reverse-direction counterparts pass.

Execution is well checked: the new native outputs match the saved original
outputs exactly, self-donor replacement reproduces native outputs exactly,
and the first edited layer's contribution matches its cached value exactly.
All 5,184 measured layer transactions satisfy correctly rounded FP32 addition.
The source and tests explicitly recompute later contributions. The first queue
attempt was rejected because the shared static checker required literal
prediction-key strings; that declaration was corrected before any GPU execution.

To understand the remaining loss, we used an exact probability identity.
For original logits z, change delta, original probabilities p=softmax(z), and
edited probabilities q=softmax(z+delta), let mu be the p-weighted average of delta:

    loss damage = -(delta_correct - mu) + KL(p || q).

The first term measures the correct token's movement relative to the original
probability-weighted average. The second is the Kullback–Leibler divergence,
a nonnegative measure of distributional change. This is an exact identity,
not a small-change approximation. A common shift of all logits cancels.

For live replacement, the four joint KL values remain 0.107–0.142 nats, close
to the original 0.111–0.144 range. Most of the improvement comes from the signed
correct-token term. Improving the answer's loss therefore does not establish
that the broader computation has been preserved.

We also tested the natural-context explanation using complete saved vectors.
After removing the irrelevant common logit offset, we compared the joint edit
with the native donor-minus-recipient output change at unit scale. Across all
12 joint/filler cells, median relative errors are 0.995–1.010 and signed cosines
are only 0.031–0.113. An error near one is roughly what predicting no native
context change would give; a cosine near zero means poor directional agreement.
Thus the whole donor-context response is not an adequate explanation. The
initial CPU script failed while serializing a NumPy Boolean; a second version
changed only that representation and produced the audited result.

Finally, the source-identity diagnostic used a fixed, meaningful readout:
the donor-query token's logit minus the recipient-query token's logit. These
are the two source identities, separate from their following payloads. Native
input changes move this margin by about 3.92–3.96 logits. The live intervention
instead moves it by -0.125 to -0.288 logits: the wrong direction for the proposed
source-identity carry. That hypothesis fails all four cells. This was an
exploratory test on FIT, not a search over vocabulary directions.

The mathematical tool we retain is live replacement of an explicitly defined
contribution, with its remaining consumers and context still running. Its
correct composition matters, but neither that tool nor the positive transfer
results discover an independently executable shared circuit. All 545,902,902
native parameters remain, with no structural saving or OOD promotion.

Evidence: [context transport](../INDUCTION_CONTEXT_TRANSPORT_V2_RESULT.json),
[saved feedback measurement](../INDUCTION_LIVE_BASELINE_DRIFT_V1_RESULT.json),
[live-replacement result](../INDUCTION_LIVE_CLAMP_V1_RESULT.json),
and [source-identity diagnostic](../INDUCTION_SOURCE_IDENTITY_READER_V1_RESULT.json).
New full-output evidence occupies 625 MB in the volatile RAM-backed directory
`/dev/shm/bilin18_induction_live_clamp_v1`; compact results, row summaries and
hashes are retained in the repository.
