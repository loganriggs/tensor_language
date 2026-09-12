# Does changing the second producer explain joint removal nonadditivity?

2026-09-12, before evaluation. Recursive unmerged component removal reduces67–70%of regional cue contrast, but individual effects sum poorly (17.7/32.8%relative discrepancy), and one natural newline row violates preservation. Preserve these misses. The two producers occur at consecutive attention layers with an intervening MLP8 and residual/RMS operations. Do not label this a direct attention8->attention9-only edge.

Frozen48regional rows, same original source readers and physical writers. Recover pristine native scalar a9(t) from the saved merged contribution w9*a9(t), using its known fixed w9=alpha9*u; this removes only an output encoding and is not a new fit. All source fields are from the original native48-row cache. Keep token order and lengths.

Three native body executions per row: pristine native (verify recovered scalar against runtime head_scalar on actual layer9 inputs); ordinary dynamic physical8+physical9 removal (replay previous result); physical8 plus removal of the frozen pristine-native physical9 write. In the last arm all downstream states still recompute, but the subtracted component9 field is held at its original value. This is an explicit hybrid counterfactual and may over-remove a contribution after its producer changes; it is a diagnostic, not an adopted intervention or proof of redundancy.

Let e8,e9,eJ be baseline-subtracted regional token-margin effects from the completed recursive test. I=e8+e9-eJ is its interaction residual. New frozen joint effect is eF, and D=eF-eJ measures the consequence of freezing the second removed component. Analyze EACHtemplate separately, signed vectors on24rows.

A: recovered scalar/native replay<=1e-5relative, native and dynamicjoint final regional/control margin replay<=1e-4relative versus previous artifacts; no approximate factor change.
B: D has norm>=.5*norm(I) and cosine(D,I)>=.8 eachtemplate. Opposing prediction: serial change of removed component9 is a substantial aligned part of joint nonadditivity.
C: norm(I-D)/norm(I)<=.5 eachtemplate, i.e. freezing the second removed write removes at least half the interaction-vector norm. Report original I,D,remaining norm and cue-contrast effects; do not confuse this with fully attributing native model computation to one direct edge. If B/C fail, other nonlinear/background routes dominate this explanation, or the hybrid counterfactual is not aligned with the interaction.

Price144single-row forwards, max22tokens,120seconds, no optimization. Reused validation contexts, not newOOD. Managedlane1. Newline outlier remains a failed control and is not filtered or repaired by this regional-only diagnostic.
