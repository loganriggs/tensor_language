# Saved-output diagnostic: native context transport and loss decomposition

Declared after the R594 failure summary, before inspecting these full-vector
diagnostics. FIT only, CPU only, no new model evaluation or original-gate changes.
This is exploratory mechanism selection, not preregistered confirmation or promotion.

Use all answer-preserving selector/payload joint rows and all filler-change rows,
both directions and all four core conditions. Group by family, variant,
recipient condition and direction, retaining all72 groups per cell. No row or
vocabulary selection. Compare the joint arm only; do not search heads, arms,
output subspaces or gains. Input evidence is the R594 receipt-bound full
endpoint logits, directed replay logits, logit differences and JSONL metadata.

Let z be replay logits, delta the saved joint change, and n native donor logits
minus native recipient logits. Center each vector by its vocabulary mean,
denoted C. Common additive logit offsets do not affect output probabilities.
The native-context-transport hypothesis predicts C delta approximately C n.
Report per-cell median ||C delta-C n||/||C n||, signed cosine, and norm ratio.
All native denominators must exceed1e-4 RMS logits; otherwise report inadequate
scale. The descriptive decision rule is median relative error<=.25 in EVERY
cell. A failure rejects this simple unit-scale whole-context explanation;
it does not prove there is no smaller contextual consumer. No best-fit scaling
or task-only projection. Report common-offset energy separately.

For p=softmax(z), q=softmax(z+delta), define mu=sum_v p_v delta_v. Then

    CE(q,y)-CE(p,y) = -(delta_y-mu) + KL(p||q),
    KL(p||q) = logsumexp(z+delta)-logsumexp(z)-mu.

This exact identity separates signed linear loss change from the nonnegative
curvature term. Recompute KL independently using log p and log q. Check the
identity, correct-answer logit and registered CE to5e-5 absolute (the saved
delta was subtracted inFP32; native logits are softcapped at30). Include
common-shift invariance, zero edit, nontrivial finite edit and sign-reversed
edit controls. Common-shift removal alone cannot rescue R594: CE also failed.

No new learned representation is fitted. All native weights and opaque
producer/context costs remain charged. Persist per-cell summaries and compact
per-row scalar diagnostics; raw full-vocabulary arrays remain unmodified.
