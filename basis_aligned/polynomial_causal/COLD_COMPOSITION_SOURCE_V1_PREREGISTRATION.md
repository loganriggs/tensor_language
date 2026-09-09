# Cold-query composition source test — 2026-09-09 15:08 UTC

The existing attn4-rms-seed0 checkpoint passes the prospective capability gate:
cold first-query hop2/3 accuracy1.0, unique-query hop3 .962, short-cycle hop3 .920.
The weak attn/MLP/attn checkpoint fails those cold/unique gates. Use the stronger
existing checkpoint to investigate actual multi-step composition. No new training.
Prior results_hop.md retracts a single-document attention story; later linear probes
show decodability but do not establish causal pointer advance. This test uses signed
full-vocabulary writes and interventions, not attention mass or probe accuracy.

Generate64 independent IID24-cycle binding worlds seed10909 and64 short-cycle
permutation worlds seed10910. For each world/entity, fork four cold queries k=0..3.
Each input is only48 binding tokens plus Q/entity/hop: zero earlier answers.
Report128 independent worlds,512 correlated query forks, not512 independent worlds.
Query0 requires no binding. For k>=1, define T as both token positions of the
requested answer's binding (key f^(k-1)(entity)); P as binding pairs for earlier
steps, excluding T. Remove duplicated positions, including on short cycles. Q is
the3 query tokens. T/P/Q are disjoint; remaining sources form O. Masks alter only
the final query position. All other destination outputs remain native.

Fold final W_O into the vocabulary reader using contextual_history_reference.py,
retaining and charging the live3-attention prefix, all final Q/K/V and residual
readout. Execute from tokens, no teacher activation cache. The parser computes
explicit source sets; it is not evidence that native keys implement those sets.
Known generic matrix folding earns no semantic discovery credit.

Arms: full, removeT, removeP, removeT+P, keepT+Q (remove other sources at finalquery).
For each, compare extracted/native full29-way logits on every input position.

A instrument: independent parser cases including short-cycle overlap; no future
answer access; direct/folded output on all arms atol=rtol=1e-9; native accuracy>=.8
in each population/hop1..3, no capability filtering.
B final answer-binding sufficiency: keepT+Q distribution KL mean<=1e-3/p99<=1e-2
separately for every population/hop1..3, full29-way query outputs. This tests whether
the final layer simply reads the already correct answer binding, not the old
unsupported one-layer-per-hop claim.
C source necessity/selectivity: removeT loses>=.25 mean gold probability while
absolute removeP change<=.10 for every population/hop2..3. No posthoc path selection.
D extraction/joint removal: native and extracted centered intervention vectors
agree at1e-9 and removeT+P obeys exact disjoint-source logit additivity at1e-9.

If B/C fail, preserve the failure and inspect which earlier binding sources carry
the answer; causally query-independent binding states may already contain composed
function information. Do not repeat the old attention-mass or linear-probe story.
A passing source test still leaves the learned key/value computation to explain.
All opaque weights and parser costs count. Batch4, FP64,1800s guard,<256MiB per
tensor, GPU exclusively through managed enqueue. No head/seed/rank sweep.
