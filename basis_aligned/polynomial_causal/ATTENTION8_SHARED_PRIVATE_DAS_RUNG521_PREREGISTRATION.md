# Rung 521 preregistration: power-gated shared/private attention8 DAS

**Frozen:** 2026-09-03 04:25 UTC, before donor-map construction, implementation smoke, optimization, or any
rung521 model outcome.

## Decision this experiment resolves

Can one causally learned attention8 output subspace carry a computation reused by three documented circuits and a
historically known fourth member, while orthogonal private subspaces add circuit-specific effects and all four parts
compose under real interventions?

This is **confirmatory operational extraction**, not new grouping discovery. Sections 2075--2078 already proposed and
tested the attention8 cluster

`{r.2.0.1, r.2.0.2, r.2.1.1, r.2.2.1}`.

Learned rank-1 directions supported that cluster in four of five seeds but had 55% relative seed variation. Rung521
fits `r.2.0.2`, `r.2.1.1`, and `r.2.2.1`; it reserves the historically known `r.2.0.1` as a frozen reuse test. This is
not an unbiased cluster-discovery p-value. Every other circuit whose best component is attention8 is a frozen
negative or evidence that the claimed shared unit is broader than this quartet.

This is not a rank-reduction or compression experiment. Ranks are fixed matched capacities. The claim is decided by
held-out causal-response prediction, cross-circuit reuse, selective private increments, finite composition, and a
second intervention type.

## Exact intervention

Let `y in R^1152` be the native post-output-projection attention8 write for a recipient token and `d` the write at a
natural donor token. For an orthonormal frame `Q in R^(1152 x r)`, define the projector `P=QQ^T` and

`I_P(y,d) = y + ((d-y)Q)Q^T`.

The learned objects are projectors, not columns. `Q` and `QR` for orthogonal `R` are the same object. The fixed design
uses one rank-4 shared projector `P_S` and three mutually orthogonal rank-4 private projectors `P_202,P_211,P_221`.
Each fitted circuit uses rank 8 (`P_S+P_i`); the full experimental union has rank 16. The 16 columns are constrained
to be globally orthonormal by a symmetric-polar retraction. Shared is fitted first and frozen; private projectors are
then fitted only in its orthogonal complement. A simultaneous unconstrained shared/private fit is forbidden because
the roles can rotate without changing their union.

Mean-centered projection removal is a distinct physical action:

`R_P(y) = y - (yQ - mu_Q)Q^T`,

where `mu_Q` is the FIT-only mean projected attention8 write. Same-donor joint projection must equal sequential
projection at the activation level; the full nonlinear suffix is nevertheless run for every joint condition.

## Frozen targets, splits, and overlap control

Authoritative masks come only from `census_state_diverse.pt`, not the stale per-circuit JSON metadata. Each of the four
masks has 864 members and a 5,760-position parent slice. Among the three fitted targets, pairwise intersections are
185, 193, and 208 positions and the triple intersection is 77. Primary gates use each target's member positions after
excluding the other three quartet masks. The complete `2^4` overlap lattice is reported separately and cannot enter a
primary shared/private verdict.

Rows are document-disjoint by

`fold(docid) = uint64_little_endian(SHA256("a8-shared-private-v1:" + decimal(docid))[0:8]) mod 10`.

- FIT folds 0--5: 568 rows. Full/exclusive counts for `r.2.0.1/.2/.1.1/.2.1` are
  `494/290, 480/254, 461/242, 502/284`.
- VALIDATION folds 6--7: 216 rows, counts `195/95, 187/95, 208/112, 189/95`.
- TEST folds 8--9: 216 rows, counts `175/103, 197/111, 195/114, 173/91`.
- The power screen divides FIT into folds 0--2 versus 3--5: 297 versus 271 rows. Exclusive fitted-target counts are
  `123/131, 119/123, 145/139`.

No rank, seed, threshold, donor rule, or selected projector may read TEST. VALIDATION chooses the FIT-only Grassmann
medoid when a deterministic representative is needed; it cannot change the registered gates.

For each circuit, matched controls are drawn without replacement from its parent slice after excluding the union of
the quartet. Matching uses the frozen relaxation order: exact next-token identity + position bin of width 32 + native
CE decile; token identity + CE decile; token class + position bin + CE decile; then token class + CE decile. Ties use
SHA-256 with seed `52100`. Counts at every relaxation level and control hashes must be frozen in a preflight addendum
before any CUDA outcome. A positive control count equal to the exclusive-member count is required in each split.

## Frozen donor construction

There are two donor ensembles, `D0` and `D1`, each with four deterministic derangements. Donors stay inside the same
data split, come from a different document, and are matched using the same token/position/CE relaxation order as the
controls. FIT, VALIDATION, and TEST donor maps are disjoint and separately hashed before the smoke. Training cycles
only through FIT `D0`; FIT `D1` is unseen by gradients. VALIDATION/TEST use their own `D0/D1`. Both swap directions
are scored. Self-donor maps are exact no-op controls.

## Stage A: exact-object power and instrument gate

Before any optimizer is created, run native replay and whole-attention8 interchange on FIT for both four-map donor
ensembles. Let `delta_D` be the mean signed per-token CE response over an ensemble. Let the 32-circuit fingerprint be

`v_D[j] = mean_member_j |delta_D| - mean_matched_control_j |delta_D|`.

Prediction A holds only if all of the following pass:

1. native replay logits are exact; attention8 is called once; self-donor changes logits and the attention8 write by
   exactly zero; every real donor edit exceeds a smoke-frozen RMS floor;
2. in both FIT halves and for each fitted target's exclusive cell,
   `mean_member |delta_D| >= 0.10 nat`, member/control concentration is at least 3, the stratified-bootstrap lower
   95% bound of the member-minus-control absolute effect is positive, and the half magnitude ratio is in `[0.5,2]`;
3. on the same exclusive member positions, `D0` versus `D1` signed response cosine is at least `.70`, optimally scaled
   relative residual is at most `.60`, and aligned recovery is positive, for every target and both FIT halves; and
4. the FIT-half 32-circuit fingerprints have Pearson correlation at least `.50` for both donor ensembles and each
   strictly exceeds the higher-interpolation 95th percentile of 200 overlap-lattice-preserving label permutations.

If A fails, the run stops before gradients and is an **instrument-power failure**, not a model null. If disagreement
is mainly between donor ensembles, increase frozen donor count; otherwise increase documents and rebuild masks. No
DAS result may be read.

## Optimization and health

Model weights are frozen. Five real seeds are fixed: `52100..52104`. Every fit must have finite loss and gradients,
attention8 called once per execution, no model-parameter gradient, `max|Q^TQ-I|<=1e-5`, projector distance from its
initialization above `.02`, mean loss over the final 20 updates below the first 20, and validation objective better
than initialization. Any unhealthy required fit invalidates its scientific predicate.

For full-attention8 response `f` and projected response `p` on an exclusive member cell, the normalized causal fit is

`L_member = mean((p-f)^2) / (mean(f^2)+eps)`.

Matched-control penalty is `mean(p_control^2)/(mean(f_member^2)+eps)`. Shared fitting minimizes the maximum of the
participating circuits' member loss plus the control penalty. Three leave-one-circuit-out shared fits are run: each
uses two target identities on FIT and never reads the third target's mask. Each fit uses 600 balanced updates per
seed. A separate rank-4 oracle is fitted to each target with 400 updates per seed. After B, a final shared projector
uses all three FIT targets for 900 updates per seed; the representative is the VALIDATION medoid among healthy seeds,
not the best TEST seed.

Sixteen complete shared fits on overlap-lattice-preserving, within-matching-stratum permuted FIT labels use fixed
seeds `52200..52215`. Real-vs-null comparisons use the higher-interpolation 95th percentile and require a strict
exceedance. Closed-form mean difference and Haar-random rank-4 projectors are reported matched controls.

## Prediction B: a shared causal subspace generalizes to an unseen target

For each leave-one-out fold, the held-out target is evaluated on exclusive VALIDATION and TEST cells with unseen donor
ensembles. At least four of five seeds must pass both data splits and both donor ensembles:

- signed response cosine to whole-attention8 response at least `.75`;
- optimally scaled relative residual at most `.55`;
- positive aligned recovery `rho=<p,f>/||f||^2`, at least 50% of that target's independently fitted rank-4 oracle;
- projected member RMS at least `.02 nat`; and
- matched-control RMS at most 25% of exclusive-member RMS.

Across folds and seeds, normalized projector overlap `tr(PP')/4` must strictly beat the retrained permutation q95.
Passing only direction similarity or only the overlapping mask positions is a B failure.

## Prediction C: frozen reuse extends to the fourth circuit and is not generic attention8 damage

The final shared projector, fitted without ever using `r.2.0.1`, must pass every B response/selectivity threshold on
that circuit's exclusive VALIDATION and TEST cells under both donor ensembles. Quartet-vs-other-attention8 separation
must strictly beat the q95 of 20,000 cheap max-statistic label permutations. Every other attention8 circuit is
reported. If several non-quartet circuits pass equally, C fails under the quartet name and the evidence is reported as
a broader attention8 variable rather than hidden.

## Prediction D: orthogonal private projectors split circuit-specific residual effects

With the final shared projector frozen, fit one rank-4 private projector per fitted circuit on FIT only, using five
seeds and 400 owner updates. Each private fit targets the conditional residual `f-p_S`, penalizes the other two target
cells, and remains orthogonal to shared and other private projectors. Independently fitted rank-8 per-circuit DAS is
the matched-capacity oracle.

In both VALIDATION and TEST, both unseen donor ensembles, and at least four of five seeds, adding `P_i` to `P_S` must:

- improve owner aligned recovery by at least `.05` absolute and by at least 25% of the remaining gap to its rank-8
  oracle;
- change each non-owner target's recovery by at most `.05`;
- add at most `.002 nat` matched-control mean CE; and
- beat the overlap-preserving max-statistic permutation q95 for owner versus largest non-owner conditional gain.

High cross-private projector overlap or off-diagonal conditional effects are failures, not evidence for private units.
If B/C hold and D fails, retain only the shared-unit result.

## Prediction E: physical action and finite composition

On TEST, with decisions frozen, run shared alone, each private alone, every shared-plus-private pair, full-attention8,
and all 15 nonempty subsets of `{S,P_202,P_211,P_221}`. Run a common-donor factorial and an independently-donored
shared/private union. Repeat with the opposite swap direction and then with mean-centered projection removal.

E requires:

1. all B--D recovery/selectivity statements hold in both swap directions without fitted physical amplitude;
2. mean removal has the same qualitative target ownership and sign; otherwise label the object a swap direction;
3. `S+P_i` retains every other target's shared-only recovery within `.05` while delivering D's owner gain;
4. the all-four installation satisfies all three owner gains simultaneously and adds at most `.002 nat` off-target CE
   per TEST half; and
5. exact Möbius interactions from all 16 endpoints are finite and reconstruct every subset endpoint to numerical
   tolerance. No joint CE is predicted by adding marginal CE effects.

Finally repeat shared and shared-plus-private conditions with attention6 native versus a frozen full-output
interchange and mean-removal background. If the shared effect changes sign, loses more than 50% aligned recovery, or
reorders private specificity, E fails for an autonomous attention8 unit and the object is named an
`attention8 x attention6` interaction-dependent unit.

## Registered outcomes

- **A false:** instrument-power failure; raise donor count or document count before any learned-subspace inference.
- **A true, B false:** valid strong null for a shared causal projector at this fixed capacity and optimizer family;
  do not increase rank as an interpretability claim.
- **A/B true, C false:** a three-target shared response without demonstrated fourth-circuit reuse, or a broader
  attention8 response if negatives also pass.
- **A--C true, D false:** reusable shared unit identified, no stable private splitting.
- **A--D true, E false:** shared/private response coordinates identified but not a composable manipulable circuit.
- **A--E true:** a reusable causal attention8 subspace plus circuit-specific conditional subspaces on frozen census
  masks. This is not yet semantic naming, fresh-document OOD identification, low-dimensional containment, or an
  adopted compressed model.

## Literal price ceiling

The implementation must print exact executed calls. Registered ceilings, including controls, are 45,000
forward+backward optimizer calls and 12,000 inference-only model calls before E; conditional E may add 5,000
inference calls. Peak stored learned basis is 18,432 floating values plus FIT means and metadata. No deployed
parameters are changed or saved. A preflight addendum must give the tighter implementation-derived price before any
scientific outcome.

## Frozen dependency identities

- `census_state_diverse.pt`: `c785f3d938091253535aa4f613ab2b4107bf297c8d615da4f7eab4f8282f5e0b`
- `curated_rows.pt`: `faaf89f38ddf1471234a1d30d978213367a566a9927bb3c73b274ab32afaa9dd`
- `circuits/BATTERY.json`: `86d7ac72eeb95f9ec80a3e92ef65e28c0df66a36b9291d2d1d2d01f7bb6c5030`
- `circuit_das.py` (historical implementation only):
  `b2e6670a223a01c7115487eb886e91fd0aa1ca9d746d60f5fa5a57c43ebeffe7`
- `circuits/A8_GROUPING.json`: `08fd57c286e4908323afe2568c53d0193e8e002c4603976340582b32ac98a755`
- `circuits/A8_GROUPING_LEARNED.json`: `3def72ab041683c3923fa487799af9a34b44d6cd3d8abf3bf96e1e1f709bf45f`
- `circuits/DAS.json`: `91bd4cb80f8077cd62af4b5a402c06845af447796c63c873a7ab677a3a4310de`
- fold vector: `305f944328a83406a873a41cb3982288dff5c6bd0c5a3282540c8cd86815aa60`
- FIT/VALIDATION/TEST row masks:
  `cf68c6efb50399b07b4de99c6777b00176dd4cabe730451b3bb69dd199dc3128 /`
  `0a0df35d1db9df41cc717c9da28737f0ac7f7dbbc4678c6056bce4f1afc35c62 /`
  `f141d3362442f4a74849446993c6cf4c271172f1e8b34447b8fdc341bc377dc4`.
- full/exclusive quartet mask pairs, in listed circuit order:
  `fad6c5613776c0d069e9326d0991d46bb2c2337c430dfb5addc748b6a9e62299 / ac6fea9504bf6cc464edc7b8686ca8ed8ab921f65b804423b41d7597b6441c19`,
  `174cb79448e2192771388b7a048e2bba4f71eb04120d7e905d398086ac2a3551 / 846f4e2fa2aa5b40409ddda756fbf7ac547d1e778bb0a4dbe7339ff0fa182fe2`,
  `ef0ebaff2022b1c9a1d0630de3229d3ffb0b8464e9a4a8eeaf908919dc7012e6 / ad63ffa0396271381f724a451bbb053c3704ec69a7cfb9c324d6f190f66dab4d`,
  `01004d10ffe47b4fe8fbccac2cef87b54aca3cb3d2bc45e482c11a8f80eaf0f0 / d7105378481ce95d2e8bb1ccc4d36c5adc36d100a77545447ad7c6ea14473eb3`.
