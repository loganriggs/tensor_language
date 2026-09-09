# Hop lookup MLP token/context terms — 2026-09-09 15:00 UTC

The producer/reader atlas is valid but nominates zero selective paths. MLP input
to final K1/K2/V removes .859–.957 repeated higher-hop probability and .567–.671
novel-hop1 probability; direct embedding/attention inputs have much smaller effects.
Thus the shared lookup dependency crosses the MLP, with large reader interactions.

Prior art: MLP0_TOKEN_CONTEXT_TENSOR_FACTORIAL_FINDINGS.md and explanation_0428.md
already establish the TT/X/CC algebra and a full-model factorial. Do not repeat that
as a novel identity or infer the small checkpoint shares the large model's ordering.
The new question is the source-defined hop lookup mechanism: does its mixed
current-token/context term suffice for both fresh binding and answer-history use?

Let first attention output be e+a with e the scaled embedding, a its contextual
write. Apply the original live MLP RMS gain to both, then compute the existing
three terms TT=D(Le*Re), X=D(Le*Ra+La*Re), CC=D(La*Ra). Preserve original residual
coefficients. Execute all8 subsets from tokens and stored weights; never call native
MLP in candidate arms. Final normalization/QK/V/readout recompute from each changed
residual. Native reference arms subtract omitted analytical terms at MLP output,
giving an independent native-corresponding intervention check.

Fresh64-cycle documents seed8909 and64 short-cycle permutation docs seed8910,
no fitting/filtering. Group repeated higher-hop, novel higher-hop and novel hop1.
All239 input positions, all29 logits, all8 subsets, batch4 FP64,1800s managed guard.

A: all CPU reassembly/cut/causality controls and every native-corresponding arm agree
at atol=rtol=1e-9; repeated higher-hop native gold P>=.8 in each population.
B: X-only predicts native full29-way distribution with mean KL<=1e-3/p99<=1e-2 on
all positions and query positions separately, both populations. No adjusted bar.
C: removing X lowers gold P>=.25 in both repeated higher-hop and novel-hop1 groups;
removing TT or CC individually changes absolute gold P<=.10 in these groups,
both populations. This would identify X as a shared dominant lookup dependency.
D: all8 candidate/native-corresponding centered intervention vectors agree at1e-9,
including joint removal of TT+CC and all3. Full recomputation supplies the composition
prediction; probability effects are not assumed additive. Report all8 group outcomes.

If B/C fail, preserve a distributed three-term computation; no branch/rank/feature
sweep. This is mechanism localization, not compression: all400640 native parameters
are charged and the algebra itself was already known. Any passing X hypothesis
still needs its explicit token/context readers, compact implementation and independent
field counterfactual test before a reusable circuit claim. Each tensor<256MiB; GPU
exclusively through managed enqueue. Reuse existing small-model loader and scorer.
