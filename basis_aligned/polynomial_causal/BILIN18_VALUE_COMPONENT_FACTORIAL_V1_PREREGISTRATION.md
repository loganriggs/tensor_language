# Causal interaction of contextual values and the shared token stream

The shared-token-only replacement failed full-output fidelity. Retain that
null. This successor identifies the role of the retained shared input,
without nominating another replacement or choosing a head subset.

Use the same fixed four-head union, full128-sequence dual-command-v2 cohort,
both existing phases and all templates. Parent result SHA256
447e72dbe5b032976c6aaac902fd8aea587eb3a4c2cc9fa91ca382975401014c.
These texts and contextual-deletion outcomes are opened. No fresh/OOD claim.
The shared first-value stream is a token-only producer before any attention;
it is reused by all layers. Selected values are C+S, with
C=(1-lambda)*raw_v and S=lambda*v0. Observed lambda9=-.65625,
lambda11=-1.75, lambda15=.55859375. Negative coefficients alone do not imply
opposite behavioral effects: learned value directions and later computation
also matter. This is why the following causal test is needed.

Four arms at all positions of the fixed union: native C+S; without_C (S);
without_S (C); without_both (zero). Modify only the effective selected value
input to native product attention. Preserve v0 itself and its other consumers;
recompute all later states. This is a consumer-local deletion of a shared
input, not deletion of its producer everywhere in the model. Context and
shared components are chosen using native raw values and v0 in the current
arm's live forward, not cached base activations.

A instrument: nine parent primitive controls; new tiny FP64 controls for
without_C versus the parent's c_v-row zeroing, without_both versus direct
selected head-output zeroing, live effects, no-cut identity and hook/method
restoration including an exception. The real audit must replay parent native
and without_C pooled metrics at absolute1e-5, with no change in batch/order.
On the first fixed four-cell batch, additionally compare without_both full
logits against direct c_proj input-head zeroing: FP32 absolute<=1e-4 and
relative RMS<=1e-5. No rescaling/relaxed tolerance on failure.

For each command and phase, let d_arm be the signed paired command-margin
change across its bit flip within each four-cell row. Define
e_C=d_without_C-d_native; e_S=d_without_S-d_native;
e_CS=d_without_both-d_native. Native paired RMS is the fixed scale, floor1e-6.

B fixed opposing-effects hypothesis: in every command/phase panel,
cos(e_C,e_S)<=-.5 and RMS(e_S)>=.1*RMS(d_native). A zero effect is not
opposition. Report individual signed recovery relative to d_native,
vector norms and all per-template panels; do not select panels that pass.

C independent contribution hypothesis: RMS(e_CS-e_C-e_S)/max(RMS(e_CS),1e-6)
<=.01 in every command/phase panel. Independently record the same interaction
for all-token centered logit vectors and full-distribution KL for every arm,
using the parent's streaming scorer. Probability changes need not add.
Failure keeps the measured joint term in the circuit explanation; it does
not license a fitted cancellation coefficient or erase the parent null.

This is an attribution/intervention screen. It cannot by itself establish
token semantics, a reduced executable, fresh/OOD prediction, or structural
savings. The complete545902902 native parameters remain priced. Opposing
effects would motivate identifying which token information is removed;
non-opposing effects redirect that hypothesis without a new head/scale sweep.

Price: managedGPU only, batch4,length<=27;32 batches x4 arms plus one first
batch whole-head oracle =129 forwards/516 sequence evaluations. No training,
fit or gradients, alarm600s, analysis tensors<256MiB. Use the existing native
manual forward; implement only the missing consumer-component intervention.
