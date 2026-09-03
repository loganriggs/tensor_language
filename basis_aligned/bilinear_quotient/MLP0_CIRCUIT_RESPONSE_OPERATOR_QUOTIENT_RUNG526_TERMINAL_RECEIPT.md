# Rung 526 terminal receipt — circuit-gradient token grouping does not transfer

**Completed:** 2026-09-03 10:12 UTC  
**Audited:** 2026-09-03 10:15 UTC  
**Decision:** registered discovery strong null; validation circuits and finite swaps remain closed

## Computation

For each of 32 discovery circuits, the run computed the difference between mean negative log likelihood on the
circuit's member positions and its matched control positions. It differentiated that quantity through the full model
suffix back to every MLP0 output. Those gradients were contracted with each vocabulary token's exact token-by-context
operator from rung 525. The resulting 32-number vector described the first-order downstream circuit effects predicted
for each token operator.

Tokens were paired using documents `0:124`. The unchanged pairs were scored on documents `124:248`. Ordinary nearest
token vectors, 16 far-random tokens, token-specific circuit-coordinate scrambling, and rung 525's task-free pairs
were fixed controls. The separate 30-circuit family on documents `500:1000` could open only after discovery passed.

## Result

- Prediction A passes. Identity-leaf logits match exactly; every gradient is nonzero; member/control weights sum to
  `+1/-1` within `4.44e-16`; contraction error is at most `5.38e-13`; planted and differentiable toys pass.
- Prediction B fails. Held-out-document candidate distance is `1.6149`, versus `0.8672` for ordinary raw-token
  neighbors: **186.2%** of the raw baseline, where the pass bar was at most 75% and the strong-null boundary was 95%.
  Candidate distance is also 84.3% of far random and 87.9% of scrambled, missing the 35% and 75% bars.
- Selection does not preserve receiver difficulty: the candidate-distance Spearman correlation between the selecting
  and scoring document halves is `0.00997`, versus the registered `0.40` bar.
- Prediction D passes descriptively: 1,694 donors repeat and cover 3,937 receivers, and 98.3% of donors differ from
  rung 525. This shows that downstream gradients changed the proposed groups, not that the changed groups are valid.
- Prediction C is false by gate because the 30 held-out circuits remain unopened.
- The strong null fires; no physical token replacement is licensed.

The run used 62 forwards, 496 batched backward calls representing 1,984 circuit-gradient objectives, 31.61 seconds,
and 7.54 GB peak GPU memory. It ran no finite intervention and added or saved no deployed values.

## Audit and consequence

The independent auditor recomputes every discovery statistic from the pair artifact, verifies token splits, phase
instruments, exact execution counts, and the validation seal, and passes four mutation tests.

- Result SHA: `4c60406f67359eab77de991983a6ff9bc756e0a6cd7902201dc0f1b0d0b721ea`
- Pair-artifact SHA: `c4d66f8900b581fcf8338eee799b75a86204404ed7ca0ba12d5244f357c9ca96`
- Audit SHA: `b78b6ea2ce41299c40363660855e2d1d92b317f81bec6f8314e986628928e022`

Do not increase circuit count, relax thresholds, or tune the tangent metric. Together, rungs 525 and 526 close both
task-free and circuit-gradient token grouping at this operator grain. The next MLP0 object is the causally material
context-only branch: it contributes about `0.418` nat of SELECT Shapley benefit, and its old source split left a
47--52% centering term. The proposed successor assigns that centering expectation exactly across source pairs, then
tests the resulting finite source-pair outputs on the circuit battery. This changes from token grouping and from rank.
