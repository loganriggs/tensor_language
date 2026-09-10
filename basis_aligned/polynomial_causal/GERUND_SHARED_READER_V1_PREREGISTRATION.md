# Shared -ing reader from unembedding contrasts

10 September 2026. Original handoff/pilot and user token/structured-unembedding
views. Prior whole-token function sharing failed; this tests a shared grammatical
contrast that can be one component of otherwise different token functions.

Prior-art audit: MLP17 dossier; readout-L15-17; earlier two-task input/function
sharing distinction; calibration and signed-reader-pair tests; fixed hierarchy
and quadratic-reader geometry. The algebra is known; the new question is whether
a weight-defined inflection contrast is a reusable causal scalar across verbs.
No activation-fit, hierarchy refit, layer/rank/gain search or best-verb filtering.

Direction e is the normalized mean of eight U_ing-U_bare rows for the fixed FIT
list in gerund_shared_reader_v1.py. Sixteen distinct TEST verbs are held out from
this mean. All forms must be single GPT-2 tokens; lexical checks occur before
model execution. No failed tokenization is silently replaced.

For normalized last-MLP input u and native output m,

    c=e^T D; Q=sym(L^T diag(c) R)
    s_m=e^T m=u^T Q u+e^T bias.
    h=r+m; s_h=e^T h=e^T r+s_m.

This explicitly folds the shared readout into the bilinear layer and separates
the carried residual contribution from the MLP contribution. It retains the
complete upstream producer, MLP remainder, RMS normalization and final softcap.
Q is a conditional scalar description, not an independent extracted producer.

A1: 'The workers often VERB. the workers can still' versus '... are still'.
A2: same with 'According to the report, ' before the last clause. All16 verbs
in both frames, no outcome filtering. Last token and token count match. P uses
can versus may, both selecting the bare form. C reuses the16 capable either/not
correlative rows. Their role is preservation, not a new OOD claim. A2 and verbs
are new authored tests relative to this weight-mean construction, not a claim
of absence from model pretraining.

Capture native base/donor endpoints, raw last-MLP input, normalized input and
MLP output. At this terminal boundary evaluate exact native final RMS/softcap
after the following changes to the recipient final residual:

* full MLP donor replacement: delta=m_d-m_b;
* scalar MLP swap: delta=e*(s_m,d-s_m,b);
* scalar carried-input swap: delta=e*[e^T(r_d-r_b)];
* scalar final-state swap: delta=e*(s_h,d-s_h,b), equal to combining both above;
* cross-verb final-state cue delta: use row (i+1) mod16's scalar cue delta in row i;
* zero removal: h-e*s_h, separately at both native endpoints.

Cross-verb reuse transfers a measured causal delta, not an absolute donor state
or a donor-free algorithm. No learned rescaling. Control panels get the same
specified interventions; only A1/A2 support a cross-verb semantic claim.
The linear scalar components compose exactly before the shared nonlinear
readout; no additive-logit or additive-CE assumption is made.

Twelve native body forwards /144 sequence instances: eight16-row captures for
the four panels, plus four4-row online bridges for full-MLP, scalar-MLP,
scalar-final and cross-verb edits on A1. Online full MLP uses the original g
donor cache; scalar edits use a native MLP17 output hook. No optimizer/backward.
Post-readout arms do not repeat the transformer body. At most900managedseconds.
All545902902nativeparameters remain; additional e1152,c4608,Q1327104 values.

Predicates:

* A instrument: counts12/144; finite; e norm error<=1e-5; native final readout
  and four online bridges maxabs<=1e-3 and relative L2<=1e-5; folded scalar
  elementwise error<=1e-3+1e-5*abs(native scalar); tiny CPU controls<=1e-10.
* B native capability: all64 pairs correct at both endpoints, A1/A2/C cue
  denominators positive; no example is removed if this fails.
* C reusable grammatical state: A/B and scalar-final plus cross-verb cue-delta
  mean raw recovery>=.80 on BOTH A1/A2. P scalar-final mean absolute correct-CE
  change<=.10; C scalar-final absolute mean raw recovery<=.10 and mean absolute
  correct-CE change<=.10. Mean raw recovery is (base_margin-patched_margin)/
  (base_margin+donor_margin), with margins oriented to native endpoint answers.
* D MLP producer sufficiency: A/B and scalar-MLP mean raw recovery>=.80 on
  BOTH A1/A2. Report full MLP swap and carried-input swap regardless of outcome.
* E selective zero removal: A/B and mean correct-CE damage across both native
  endpoints>=.10 on each of A1/A2; C mean absolute correct-CE change<=.10.
  Report P removal, per-endpoint effects and full-vocabulary changes too.

C failure closes this fixed reusable reader proposal; D failure does not imply
upstream absence. E failure preserves any swap result without calling the state
independently removable. Even all passes leave independent production, broader
OOD, joint operation with other circuits and structural cost improvement open.
CPU successor: paired uncertainty on recovery/removal and exact scalar
skip/MLP delta accounting; no threshold or mean rescue.
