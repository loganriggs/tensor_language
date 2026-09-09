# Which other input participates in the mixed query computation?

The I-local/J-complement query assignment failed on every registered panel.
Mixed query terms are substantial in both routes. Change the question to the
inputs of that bilinear mixed operation; do not select another pure-query arm.

Same512 opened requests,32 worlds,two orders,two query origins,four hops.
Using frozen native prefix payload transport, define exact query roots:
L = entity-query token at49; H = delimiter and hop-instruction tokens at48/50;
D = all48 binding tokens. All input roots are included, with L+H+D=x_query.
These are payload lineages, not raw token interventions or context-independent
semantic codes; all context-dependent prefix gates remain charged.

At the original query gain and original source K/V/skip, every endpoint route
R (I or J) is quadratic in raw Q input. Therefore its already measured mixed
entity/complement term splits exactly:

    R_LC = R(L+H+D)-R(L)-R(H+D)
         = [R(L+H)-R(L)-R(H)] + [R(L+D)-R(L)-R(D)]
         = R_LH + R_LD.

Use existing lineage, Q-direction and key/payload interaction primitives.
Only four new Q evaluations H,D,L+H,L+D are needed per case, plus saved
local and mixed terms. Input rows SHA
f790aa9c396b12e57d207f821be9217413255d1ac170b553999426163687b048.
Native Q-hook/four-arm oracles verify new route outputs; live-sum comparisons
verify the mixed-term partition, abs<=1e-9/relative<=1e-10. Recompute L and
compare saved pure-local route to protect root/frame correspondence.

Two distinct hypotheses are fixed in advance: H_task says LH accounts for
the mixed operation, while H_document says LD does. For each hypothesis and
each route separately on forward hop3 IID/OOD, knockout of the selected term
from original native logits must recover [.8,1.2] of the signed gold effect
of removing R_LC; knockout of the complementary term must have absolute
ratio<=.2. Full mixed-effect denominator below .001 invalidates rather than
filters that panel. A hypothesis passes only if all four route/population
panels pass. Do not announce the best individual panel as a shared operation.

Report centered full-vector errors relative to R_LC with floor1e-6 for every
route/query/hop group, signed interactions, and every task effect. Task-level
dominance is not full-output sufficiency, independent circuit extraction or
structural description reduction. Preserve all previous failed gates. No
new root subsets, fitted coefficients, heads or effect thresholds after a null.

CPU threads2,batch8 FP64,alarm180s,tensors below256MiB,no GPU/training.
All387968 opaque export coefficients, native routing and remainder are priced.
This is a bounded source-of-computation diagnostic; no fresh cohort claim.
