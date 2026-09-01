# Six-direction comparison — 2026-09-01

This is the common decision pass for rungs 298–303 and the 302B control repair. The fixed goal is a
substantially smaller, predictive, composable, manipulable, literally priced tensor program. A route is ranked
by what it has actually earned, not by its untested maximum compression.

| Rank | Route | Literal upside earned so far | External predictive evidence | Identification / controls | Blocking risk | Decision |
|---:|---|---|---|---|---|---|
| 1 | Shared input/output vocabulary code | 30.28M scalars at the tested point; 26.1% of vocab | +.193 FW / +.225 Wiki versus matched independent +.552 / +.778 | disjoint fit, shifted corpus, price match | unseen-target damage +1.57/.895; post-result metric; no certs | exploit fit-selected sparse rare-row residual |
| 2 | Activation-PCA MLP0 output rank | 4.57M at r128; 3.82M at r256 relative to native MLP0 | r256 +.0209 FW / +.0114 Wiki; dominates response/SVD | split overlap .683, matched rank/price | only MLP0, no composition/certs/full bill | retain as secondary seed after vocab exploit |
| 3 | Local omitted-energy law | no direct storage; screening instrument | frozen interval covered 9/9, validation rho .917 | prospective split; exponent agrees with old random curve | 3.41x width; cannot reject PCA r64 | use for triage only, never adoption gating |
| 4 | Changed-metric joint MLP refactor | none earned | no executable candidate yet | native sharing null is clean | requires new joint CP and external metric | open mathematically, not tonight's exploit |
| 5 | Embedding-folded structural MDL | none earned | wrong prior fits as well as right prior | excellent planted/student/negative controls | structure non-identifiable by R2 | park until intervention/OOD discriminator exists |
| 6 | Predictive finite state | no priced replacement | classifier beats mean shuffle; quote transfer .875 | 302B live null repairs inert control | R2 fails shuffle-p95; suffix NLL 10–12; tiny head effect | kill as compiler route; circuit classifier only |

Direct native-atom sharing and the signed-response eigenbasis are not ranked as live routes: their matched
controls decisively dominate them. They remain useful negative information.

## Exploit choice

Use the 1,302,528-scalar headroom between shared residual rank 512 and the exact 25%-saving vocabulary ceiling
for at most 1,129 indexed full residual rows. The prospective candidate selects only fit-count <=2 token rows by
the diagonal Fisher estimate

`score_t = sum_i p_i(t)(1-p_i(t)) [logit_native(i,t)-logit_shared(i,t)]^2`.

This approximates how much second-order CE the row's current logit error contributes on disjoint fit contexts.
At identical `K=1129` and price, compare it with residual-norm selection and a seeded random rare-row control.
Evaluate on fresh FineWeb skip7000 and WikiText tokens after skip10000. The candidate must repair both aggregate
damage and the unseen-target tail without worsening the common-token bin.

## Non-negotiable adoption gates after a screen pass

1. Freeze a full census and shifted-corpus bar before running them.
2. Run all 62 certificates, with explicit rare-token circuit reporting.
3. Add the new vocabulary object to the standalone dependency bill, including row indices.
4. Compose with the adopted 539,595,062-scalar mixed104 online-c_v0 program and measure non-additivity.
5. Only then run signed intervention gates appropriate to whichever certificates move.
