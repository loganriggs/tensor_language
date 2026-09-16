# Subject-number L11H3 late-writer specificity audit V1

## Motivation

The registered six-term late-writer graph transferred to lexical, context, and
joint holdouts, and its target error beat the random median by more than `0.10`.
However, 8/16 random readouts had lower joint-OOD error; the median gate passed
because the null distribution was bimodal. Also, the discovery curve reached
`0.13812` after three terms and only `0.13469` after six, suggesting the last
three terms may be unnecessary.

## Frozen audit

Recompute the exact eight-port, 256-corner Möbius instrument on the identical
authority. Bind the V1 runner/result and reproduce its frozen target selection:

1. `mlp_8`
2. `upstream_0_7`
3. `mlp_10`
4. `attn_9`
5. `attn_9*mlp_9`
6. `mlp_9*mlp_10`

Report every prefix on all four frozen panels. The three-term graph is accepted
as a pruning of V1 only if its relative L2 is at most `0.25` with positive
aligned recovery on all panels, and its error exceeds the six-term graph by at
most `0.02` on every panel.

Draw 64 new unit readouts orthogonal to the target writer (seed `20260928`). For
each null, report joint-OOD target RMS and two errors:

- competitive: its own six main/pair terms selected on discovery;
- frozen structure: the target writer's six masks applied without reselection.

Specificity is accepted separately for each null only if at least 90% of random
errors exceed the target writer's six-term joint-OOD error. This empirical-rank
gate supersedes no prior result; it is a stricter prospective audit prompted by
the observed bimodality.

Instrumentation retains the prior exactness, gauge, replay, closure, synthetic,
price, and checkpoint gates. No behavioral logits or coefficients are fit.
